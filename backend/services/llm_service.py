"""Chat orchestration over a LlamaEngine.

Streaming for user-facing chat, blocking invoke for agentic internals.
Structured calls return the Pydantic model instance itself.
"""

import asyncio
import json
from queue import Empty, Queue
from threading import Thread
from typing import Dict, List, Optional, Type, TypeVar, Union

from fastapi import Request
from pydantic import BaseModel

try:
    from llama_cpp import LlamaGrammar
except ImportError as e:
    raise RuntimeError("llama-cpp-python not installed.") from e

from services.llama_engine import LlamaEngine
from services.prompt_manager import PromptManager

T = TypeVar("T", bound=BaseModel)

_GRAMMARS: Dict[type, LlamaGrammar] = {}


class LLMService:
    """Chat orchestration over a LlamaEngine — streaming plus blocking calls."""

    def __init__(self, engine: Optional[LlamaEngine] = None):
        self.engine = engine or LlamaEngine.get_instance()

    @staticmethod
    def build_chat_messages(
        history: List[Dict[str, str]], current_query: str
    ) -> List[Dict[str, str]]:
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

    async def astream_chat(
        self, messages: List[Dict[str, str]], request: Optional[Request] = None
    ):
        """Stream via worker thread + queue."""
        self.engine.ensure_loaded()
        self.engine.acquire()

        token_queue: Queue = Queue()
        stop_signal = object()
        handle = self.engine.llm

        def worker() -> None:
            try:
                # sleep-wake safety: catch OS interrupt
                stream = handle.create_chat_completion(
                    messages=messages, stream=True, temperature=0.6
                )
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
            self.engine.release()

    def invoke(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        structured_output: Optional[Type[T]] = None,
    ) -> Union[str, T]:
        """Blocking single call — plain str, or validated model instance."""
        self.engine.ensure_loaded()
        self.engine.acquire()
        try:
            if structured_output is None:
                resp = self.engine.llm.create_chat_completion(
                    messages=list(messages),
                    stream=False,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop or [],
                )
                return _content_from_response(resp)

            schema = structured_output.model_json_schema()
            resp = self.engine.llm.create_chat_completion(
                messages=_with_schema_instruction(messages, schema),
                stream=False,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop or [],
                response_format={"type": "json_object", "schema": schema},
                grammar=_grammar_for(structured_output),
            )
            raw = _content_from_response(resp)
            try:
                return structured_output.model_validate_json(raw)
            except Exception as e:
                raise RuntimeError(
                    f"structured output parse failed: {e}; raw: {raw[:500]}"
                ) from e
        finally:
            self.engine.release()

    async def ainvoke(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        structured_output: Optional[Type[T]] = None,
    ) -> Union[str, T]:
        """Async wrapper over invoke for graph and API callers."""
        return await asyncio.to_thread(
            self.invoke, messages, max_tokens, temperature, stop, structured_output
        )


def _grammar_for(model: Type[T]) -> LlamaGrammar:
    """Compile once per model class; grammars are weight-independent."""
    gram = _GRAMMARS.get(model)
    if gram is None:
        gram = LlamaGrammar.from_json_schema(
            json.dumps(model.model_json_schema()), verbose=False
        )
        _GRAMMARS[model] = gram
    return gram


def _content_from_response(resp: object) -> str:
    """Pull assistant text from a non-streaming chat completion."""
    try:
        msg = resp["choices"][0]["message"]
        content = (
            msg.get("content", "")
            if isinstance(msg, dict)
            else getattr(msg, "content", "")
        )
        return (content or "").strip()
    except Exception:
        return ""


def _with_schema_instruction(
    messages: List[Dict[str, str]], schema: Dict[str, object]
) -> List[Dict[str, str]]:
    """JSON-only instruction carrying the model schema; copies inputs."""
    instruction = (
        "Respond with a single JSON object matching this JSON Schema, "
        f"and nothing else (no markdown, no explanation):\n{json.dumps(schema)}"
    )
    built = [dict(m) for m in messages]
    for m in built:
        if m.get("role") == "system":
            m["content"] = f"{m.get('content', '')}\n\n{instruction}".strip()
            return built
    return [{"role": "system", "content": instruction}] + built
