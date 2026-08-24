from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    SARVAM_API_KEY: str

    # Gemini LLM
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-3.5-flash-lite"
    FIRECRAWL_API_KEY: str

    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
