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

    # Notion
    NOTION_API_KEY: str = ""

    # Groq (Primary LLM & Whisper STT)
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "qwen/qwen3.6-27b"
    GROQ_WHISPER_PRIMARY: str = "whisper-large-v3"
    GROQ_WHISPER_FALLBACK: str = "whisper-large-v3-turbo"



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

    # Database (Supabase PostgreSQL)
    DATABASE_URL: str
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""
    PROJECT_ID: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
