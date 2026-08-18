from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = "postgresql://recipes:recipes@postgres:5432/recipes"
    openai_api_key: str | None = None
    openai_model: str = "gpt-5.4-mini"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_revision: str = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    input_max_chars: int = 1000
    rrf_k: int = 30
    coverage_weight: float = 2.0
    missing_weight: float = 0.55
    retrieval_mode: str = "keyword"
    recommendation_prompt: str = "concise"
    input_cost_per_million: float = 0.40
    output_cost_per_million: float = 1.60


@lru_cache
def get_settings() -> Settings:
    return Settings()
