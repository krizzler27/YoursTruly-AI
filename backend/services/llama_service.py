from threading import Thread, Lock
from queue import Empty, Queue
from fastapi import Request
from pathlib import Path
import asyncio
import psutil
import gc
import os

from typing import AsyncGenerator, Dict, List, Optional
from services.prompt_manager import PromptManager
from config import config


def get_physical_cores() -> int:
    """Physical cores via psutil, fallback to logical//2."""
    cores = psutil.cpu_count(logical=False)
    return int(cores) if cores else max(1, (os.cpu_count() or 4) // 2)


def get_total_ram_gb() -> float:
    """Total RAM in GB via psutil, fallback 8.0 if undetermined."""
    total = psutil.virtual_memory().total
    return total / (1024**3) if total else 8.0


def get_default_ctx() -> int:
    if config.LLAMA_N_CTX is not None:
        return int(config.LLAMA_N_CTX)
    return 4096 if get_total_ram_gb() >= 12.0 else 2048


class LlamaEngine:
    """In-process llama.cpp engine with Vulkan fallback and thread-safe streaming."""

    _instance: Optional["LlamaEngine"] = None
    _lock: Lock = Lock()

    def __init__(self, model_path: Optional[str] = None):
        self.model_path: Optional[str] = model_path
        self.llm = None
        self._generating = False
        self._gen_lock = Lock()

    @staticmethod
    def _discover_models() -> List[Path]:
        """List installed GGUFs sorted by most recent mtime first."""
        models_dir = Path(config.LLAMA_MODEL_PATH)
        try:
            return sorted(
                [p for p in models_dir.glob("*.gguf") if p.is_file()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except Exception:
            return []

    def switch_model(self, full_path: Optional[str]) -> None:
        """Switch to GGUF at full path."""
        if not full_path or not full_path.strip():
            return
        if self.is_generating():
            raise RuntimeError("System Busy — model is generating. Try again.")
        p = Path(full_path.strip())
        if not p.exists() or not p.is_file() or p.suffix.lower() != ".gguf":
            raise FileNotFoundError(f"Model not found: {full_path}")
        if self.model_path and Path(self.model_path).resolve() == p.resolve() and self.is_loaded():
            return
        self.unload()
        self.model_path = str(p)
        self.load()

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
        """Load most recent GGUF."""
        if self.is_loaded():
            return
        disc = self._discover_models()
        mp = Path(self.model_path) if self.model_path and Path(self.model_path).exists() else (disc[0] if disc else None)
        if mp is None or not mp.exists():
            raise FileNotFoundError(f"No GGUF in {Path(config.LLAMA_MODEL_PATH)}. Download a model via Explore.")
        self.model_path = str(mp)
        try:
            import llama_cpp
            from llama_cpp import Llama
        except ImportError as e:
            raise RuntimeError("llama-cpp-python not installed.") from e

        cores = config.LLAMA_N_THREADS or get_physical_cores()
        ctx = get_default_ctx()
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
        gpu_layers = config.LLAMA_N_GPU_LAYERS
        if gpu_layers is None:
            try:
                self.llm = Llama(n_gpu_layers=-1, **common_kwargs)
            except Exception:
                self.llm = Llama(n_gpu_layers=0, **common_kwargs)
        else:
            self.llm = Llama(n_gpu_layers=int(gpu_layers), **common_kwargs)

        try:
            list(self.llm.create_chat_completion(
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                stream=False,
            ))
        except Exception:
            pass

    def unload(self) -> None:
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
        """Stream via worker thread + queue."""
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
        disc = self._discover_models()
        if disc:
            mp = Path(self.model_path) if self.model_path and Path(self.model_path).exists() else disc[0]
            status = "ready"
            exists = True
        else:
            mp = Path(config.LLAMA_MODEL_PATH) / "no-model.gguf"
            status = "error"
            exists = False
        return {
            "status": status,
            "model": mp.name if mp.name != "no-model.gguf" else None,
            "path": str(mp),
            "exists": exists,
            "loaded": self.is_loaded(),
            "generating": self.is_generating(),
            "available": [p.name for p in disc],
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
