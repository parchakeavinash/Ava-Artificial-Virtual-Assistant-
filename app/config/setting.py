from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ASSEMBLYAI_API_KEY: str

    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_API_KEY: str
    SARVAM_API_KEY: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()