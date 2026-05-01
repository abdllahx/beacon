from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "beacon"
    postgres_password: str = "beacon"
    postgres_db: str = "beacon"

    newsapi_key: str | None = None
    nasa_firms_key: str | None = None
    hf_token: str | None = None

    # ACLED OAuth — myACLED login credentials. Username/password are used only at
    # initial login; tokens persist in .acled_tokens.json (gitignored).
    acled_username: str | None = None
    acled_password: str | None = None

    # LangSmith tracing. When LANGSMITH_TRACING=true, LangGraph auto-streams traces
    # to https://smith.langchain.com under the LANGSMITH_PROJECT.
    langsmith_api_key: str | None = None
    langsmith_tracing: bool = False
    langsmith_project: str = "beacon"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # Langfuse Cloud (free Hobby tier). Activates @observe wrapping in observability.py.
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # Optional override for managed Postgres (Neon, Supabase, etc). Takes priority over
    # the host/port/user fields — set this when deploying to Streamlit Cloud where the
    # local docker DB isn't reachable.
    database_url: str | None = None

    beacon_claude_call_budget: int = Field(default=50, ge=0)

    @property
    def postgres_dsn(self) -> str:
        if self.database_url:
            return self.database_url
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
