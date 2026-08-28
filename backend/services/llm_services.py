from litellm import acompletion
from typing import AsyncGenerator

from services.llmfit_services import LLMFitServices
from config import config

class LLMServices():

    def __init__(self, provider: str = config.DEFAULT_LLM_PROVIDER, model: str = config.DEFAULT_LLM_MODEL, temperature: float = 0.6):
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