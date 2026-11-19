"""Settings, read from environment variables and a local .env file.

Any OpenAI-compatible endpoint works: switching provider is a matter of
changing OPENAI_BASE_URL / OPENAI_API_KEY / MODEL_* in .env.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

StructuredMode = Literal["auto", "json_schema", "json_object", "prompt"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_base_url: str = "http://localhost:11434/v1"
    openai_api_key: str = "not-needed"
    model_extract: str = "qwen3.8"
    model_write: str = "qwen3.8"
    model_verify: str | None = None
    """Model for the verification pass; defaults to model_extract. A different or
    stronger model here catches more of the extractor's mistakes."""
    structured_output: StructuredMode = "auto"
    temperature: float = 0.0
    max_tokens: int = 8192
    request_timeout: float = 600.0
    llm_extra_body: dict[str, Any] | None = None
    """Provider-specific request fields, as JSON. For Qwen3 on vLLM, disable thinking
    (which otherwise eats the max_tokens budget) with
    {"chat_template_kwargs": {"enable_thinking": false}}."""
    max_repair_attempts: int = 1
    """Times an invalid JSON answer is sent back to the model with the error."""
    concurrency: int = 1
    """Parallel LLM requests during claim extraction. Only helps if the server
    runs several requests at once."""
    chunk_lines: int = 150
    """Approximate number of notebook lines per extraction request."""
    verify_claims: bool = True
    """Second LLM pass checking each claim against the code it cites."""
    verify_batch_size: int = 8
    verify_context_lines: int = 3
    """Lines of context shown around cited code during verification."""
    cache_dir: Path = Path(".cache")
    cache_enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8000
    open_browser: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
