from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env"
    )

    DEFAULT_LLM_PROVIDER : str
    DEFAULT_LLM_MODEL: str
    DEFAULT_OLLAMA_HOST: str

config = Config()