from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template


class PromptManager:
    """Jinja2 prompt loader with LRU-cached Environment and templates."""

    _PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

    @classmethod
    @lru_cache(maxsize=1)
    def _get_env(cls) -> Environment:
        return Environment(
            loader=FileSystemLoader(str(cls._PROMPTS_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    @classmethod
    @lru_cache(maxsize=32)
    def get_template(cls, template_name: str) -> Template:
        """Loads and caches up to 32 different prompt templates in memory."""
        env = cls._get_env()
        return env.get_template(template_name)

    @classmethod
    def render(cls, template_name: str, **kwargs) -> str:
        template = cls.get_template(template_name)
        return template.render(**kwargs)
