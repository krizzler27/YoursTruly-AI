from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env"
    )

    LLM_PROVIDER : str = "ollama"
    LLM_MODEL: str = "qwen2.5:3b"
    OLLAMA_HOST: str = "http://localhost:11434"
    DATABASE_URL: str = "sqlite+pysqlite:///yourstrulyai.db"

config = Config()