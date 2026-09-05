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
    GROQ_MODEL: str = "openai/gpt-oss-120b"
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

    # Agent Memory System (Short-Term, Episodic, Semantic)
    GEMINI_EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    MEMORY_WINDOW: int = 10
    MEMORY_USER_ID: str = "default_user"
    EPISODIC_TOP_K: int = 3
    EPISODIC_MIN_SIMILARITY: float = 0.50
    SEMANTIC_TOP_K: int = 5
    SEMANTIC_MIN_SIMILARITY: float = 0.45

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
