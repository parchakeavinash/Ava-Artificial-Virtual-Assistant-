from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # AssemblyAI (legacy / alternative STT)
    ASSEMBLYAI_API_KEY: str = ""

    # Gemini LLM
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-3.5-flash-lite"

    # Sarvam AI (STT + TTS)
    SARVAM_API_KEY: str

    # Firecrawl (Web Search)
    FIRECRAWL_API_KEY: str = ""

    # Email Settings (SMTP / IMAP)
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    IMAP_HOST: str = "imap.gmail.com"
    IMAP_PORT: int = 993
    ADMIN_EMAIL: str = Field(default="", validation_alias="ADMIN_MAIL")
    EMAIL_PASSWORD: str = ""

    # Security / Auth
    SECRET_KEY: str = ""
    ALGORITHM: str = "HS256"
    EXPIRE_MINUTES: int = 30

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
