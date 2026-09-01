from litellm import acompletion
from typing import AsyncGenerator, List

from services.llmfit_services import LLMFitServices
from services.prompt_manager import PromptManager
from config import config


class LLMServices():

    def __init__(self, provider: str = config.LLM_PROVIDER, model: str = config.LLM_MODEL, temperature: float = 0.6):
        # allow fully-qualified model like "ollama/qwen2.5:3b"
        if model and "/" in model:
            provider, model = model.split("/", 1)
        self.provider = provider
        self.model = model
        self.temperature = temperature

    async def astream(self, messages: list[dict]) -> AsyncGenerator[str, None]:
        """Yield plain text deltas (str) - decouples router from litellm types. Token budgeting can count len(delta) here."""
        if not await LLMFitServices().is_ollama_running():
            raise Exception("Ollama is not running. Run `ollama serve` in terminal")

        try:
            response = await acompletion(
                model=f"{self.provider}/{self.model}",
                temperature=self.temperature,
                messages=messages,
                stream=True,
            )

            async for chunk in response:
                # litellm chunk -> choices[0].delta.content
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        except Exception as e:
            raise RuntimeError(f"LLM stream failed: {e}") from e

    @classmethod
    def load_prompt(cls, name: str, **params) -> str:
        """Load j2 prompt via cached PromptManager."""
        return PromptManager.render(f"{name}.j2", **params)

    @classmethod
    def build_chat_messages(cls, history: List[dict], current_query: str) -> List[dict]:
        """Resolve chat_instruction.j2 with history + current query."""
        if not history:
            history_block = "No prior conversation."
        else:
            lines = []
            for m in history:
                role = "User" if m["role"] == "user" else "Assistant"
                lines.append(f"{role}: {m['content']}")
            history_block = "\n".join(lines)
        system_content = cls.load_prompt(
            "chat_instruction",
            history_block=history_block,
            current_query=current_query,
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": current_query},
        ]