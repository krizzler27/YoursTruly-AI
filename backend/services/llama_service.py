import asyncio
import gc
import os
from pathlib import Path
from queue import Empty, Queue
from threading import Thread, Lock
from typing import AsyncGenerator, Dict, List, Optional

from fastapi import Request

from config import config
from services.prompt_manager import PromptManager

# lazy psutil — optional, fallback to os.cpu_count
try:
    import psutil
except ImportError:
    psutil = None


def get_physical_cores() -> int:
    """Physical cores via psutil, fallback to logical//2."""
    if psutil is not None:
        try:
            cores = psutil.cpu_count(logical=False)
            if cores:
                return int(cores)
        except Exception:
            pass
    return max(1, (os.cpu_count() or 4) // 2)


def get_total_ram_gb() -> float:
    if psutil is not None:
        try:
            return psutil.virtual_memory().total / (1024**3)
        except Exception:
            pass
    return 8.0


def get_default_ctx() -> int:
    if config.LLAMA_N_CTX is not None:
        return int(config.LLAMA_N_CTX)
    return 4096 if get_total_ram_gb() >= 12.0 else 2048


class LlamaEngine:
    """In-process llama.cpp engine with Vulkan fallback and thread-safe streaming."""

    _instance: Optional["LlamaEngine"] = None
    _lock: Lock = Lock()

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or config.LLAMA_MODEL
        self.llm = None  # llama_cpp.Llama
        self._generating = False
        self._gen_lock = Lock()

    @classmethod
    def get_instance(cls) -> "LlamaEngine":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def is_loaded(self) -> bool:
        return self.llm is not None

    def is_generating(self) -> bool:
        with self._gen_lock:
            return self._generating

    def _set_generating(self, value: bool) -> None:
        with self._gen_lock:
            self._generating = value

    def load(self) -> None:
        """Load GGUF via universal Vulkan (try -1) else CPU fallback."""
        if self.is_loaded():
            return
        # validate model file exists — downstream handles auto-download
        mp = Path(self.model_path)
        if not mp.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                f"Place qwen2.5-3b-Q4_K_M.gguf in {Path(config.LLAMA_MODEL_PATH)} "
                f"or trigger auto-download."
            )
        # lazy import — allows spike without wheel installed
        try:
            import llama_cpp
            from llama_cpp import Llama
        except ImportError as e:
            raise RuntimeError(
                "llama-cpp-python not installed. Run: "
                "pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/wheels/avx2 "
                "or .../wheels/vulkan for GPU"
            ) from e

        cores = config.LLAMA_N_THREADS or get_physical_cores()
        ctx = get_default_ctx()
        # explicit 8GB tuning: q4_0 KV, flash_attn, small batches
        common_kwargs = dict(
            model_path=str(mp),
            n_ctx=ctx,
            n_threads=cores,
            n_threads_batch=cores,
            n_batch=512,
            n_ubatch=256,
            type_k=llama_cpp.GGML_TYPE_Q4_0,
            type_v=llama_cpp.GGML_TYPE_Q4_0,
            flash_attn=True,
            use_mmap=True,
            use_mlock=False,
            verbose=False,
        )
        # universal Vulkan: -1 if driver present, else CPU 0
        gpu_layers = config.LLAMA_N_GPU_LAYERS
        if gpu_layers is None:
            # try Vulkan first — single exe path
            try:
                self.llm = Llama(n_gpu_layers=-1, **common_kwargs)
            except Exception:
                self.llm = Llama(n_gpu_layers=0, **common_kwargs)
        else:
            self.llm = Llama(n_gpu_layers=int(gpu_layers), **common_kwargs)

        # warmup fault mmap pages — 1 token to hit <1s TTFT
        try:
            list(self.llm.create_chat_completion(
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                stream=False,
            ))
        except Exception:
            pass

    def unload(self) -> None:
        """Explicit release for model switch on 8GB."""
        if self.llm is not None:
            try:
                del self.llm
            except Exception:
                pass
            self.llm = None
            gc.collect()

    async def astream_chat(
        self, messages: List[Dict[str, str]], request: Optional[Request] = None
    ) -> AsyncGenerator[str, None]:
        """Thread-safe streaming — C++ iterator on worker thread."""
        if not self.is_loaded():
            # lazy load on first chat if lifespan not warmed
            try:
                self.load()
            except FileNotFoundError as e:
                raise RuntimeError(str(e)) from e

        if self.is_generating():
            raise RuntimeError("System Busy — model is generating. Try again.")

        self._set_generating(True)
        token_queue: Queue = Queue()
        stop_signal = object()

        def worker() -> None:
            try:
                # sleep-wake safety: catch OS interrupt
                stream = self.llm.create_chat_completion(messages=messages, stream=True, temperature=0.6)
                for chunk in stream:
                    try:
                        delta = chunk["choices"][0]["delta"].get("content", "")
                    except Exception:
                        delta = ""
                    if delta:
                        token_queue.put(delta)
            except Exception as e:
                token_queue.put(e)
            finally:
                token_queue.put(stop_signal)

        Thread(target=worker, daemon=True).start()

        try:
            while True:
                if request is not None:
                    try:
                        if await request.is_disconnected():
                            break
                    except Exception:
                        pass
                try:
                    item = token_queue.get_nowait()
                except Empty:
                    await asyncio.sleep(0.005)
                    continue
                if item is stop_signal:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            self._set_generating(False)

    def health(self) -> Dict[str, object]:
        mp = Path(self.model_path)
        return {
            "status": "ready" if self.is_loaded() else ("loading" if mp.exists() else "error"),
            "model": mp.name if mp.name else "qwen2.5-3b-Q4_K_M.gguf",
            "path": str(mp),
            "exists": mp.exists(),
            "loaded": self.is_loaded(),
            "generating": self.is_generating(),
        }

    @classmethod
    def build_chat_messages(cls, history: List[Dict[str, str]], current_query: str) -> List[Dict[str, str]]:
        if not history:
            history_block = "No prior conversation."
        else:
            lines = []
            for m in history:
                role = "User" if m.get("role") == "user" else "Assistant"
                lines.append(f"{role}: {m.get('content', '')}")
            history_block = "\n".join(lines)
        system_content = PromptManager.render(
            "chat_instruction.j2",
            history_block=history_block,
            current_query=current_query,
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": current_query},
        ]
