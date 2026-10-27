from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM Settings
    LLM_PROVIDER: str = Field(default="gemini")
    GEMINI_API_KEY: Optional[str] = Field(default=None)
    GEMINI_MODEL: str = Field(default="gemini-2.0-flash")

    GROQ_API_KEY: Optional[str] = Field(default=None)
    GROQ_MODEL: str = Field(default="llama-3.3-70b-versatile")

    OPENAI_API_KEY: Optional[str] = Field(default=None)
    OPENAI_MODEL: str = Field(default="gpt-4o-mini")

    # Stub mode for D6 load testing
    USE_LLM_STUB: bool = Field(default=False)
    LLM_STUB_DELAY_SEC: float = Field(default=0.8)

    # Retrieval and Embeddings
    EMBEDDING_MODEL_NAME: str = Field(default="all-MiniLM-L6-v2")
    VECTOR_DB_TYPE: str = Field(default="chroma")
    VECTOR_DB_PATH: str = Field(default="./data/index")
    CORPUS_DIR: str = Field(default="./data/corpus")
    QUARANTINE_DIR: str = Field(default="./data/quarantine")

    # Service & Worker
    APP_HOST: str = Field(default="0.0.0.0")
    APP_PORT: int = Field(default=8000)
    WORKER_CONCURRENCY: int = Field(default=4)
    REDIS_URL: str = Field(default="redis://localhost:6379/0")

    # Timeouts and Circuit Breakers (A1, A5, D7)
    RATE_LIMIT_PER_MINUTE: int = Field(default=60)
    CIRCUIT_BREAKER_FAIL_MAX: int = Field(default=3)
    CIRCUIT_BREAKER_RESET_TIMEOUT_SEC: int = Field(default=30)
    MODEL_TIMEOUT_SEC: float = Field(default=15.0)
    TOOL_TIMEOUT_SEC: float = Field(default=8.0)
    MAX_AGENT_STEPS: int = Field(default=6)

    # SLO (D6)
    SLO_P95_SECONDS: float = Field(default=15.0)

    # LLMOps & Guardrails (L2, L4, L5)
    PROMPT_VERSION: str = Field(default="v1.0.0")
    ENABLE_PROMPT_INJECTION_FILTER: bool = Field(default=True)
    ENABLE_OUTPUT_FILTER: bool = Field(default=True)
    ENABLE_TOXICITY_FILTER: bool = Field(default=True)

    # Observability (D5)
    SERVICE_NAME: str = Field(default="tariff-triage-service")
    OTEL_EXPORTER_OTLP_ENDPOINT: str = Field(default="http://localhost:4317")
    PROMETHEUS_METRICS_PATH: str = Field(default="/metrics")


settings = Settings()
