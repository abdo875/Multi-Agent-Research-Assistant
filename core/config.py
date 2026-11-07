"""Configuration management for AgentForge.

Validates environment variables, enforces safe defaults, and manages
execution modes (Mock, Hosted LLM, Local vLLM Inference).
"""

from enum import Enum
from functools import lru_cache
from typing import Any, Dict, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppMode(str, Enum):
    MOCK = "mock"
    HOSTED = "hosted"
    VLLM = "vllm"


class Settings(BaseSettings):
    """Application configuration and inference provider settings."""

    # Operational Mode
    app_mode: AppMode = Field(
        default=AppMode.MOCK,
        validation_alias="APP_MODE",
        description="Execution mode: 'mock' (no credentials), 'hosted' (OpenAI API), or 'vllm' (local inference)",
    )

    # LLM Inference Connection
    llm_provider: str = Field(
        default="openai_compatible",
        validation_alias="LLM_PROVIDER",
        description="Type of LLM provider (OpenAI-compatible protocol)",
    )
    llm_base_url: str = Field(
        default="http://localhost:8000/v1",
        validation_alias="LLM_BASE_URL",
        description="Base URL for OpenAI-compatible endpoint",
    )
    llm_api_key: Optional[str] = Field(
        default=None,
        validation_alias="LLM_API_KEY",
        description="API key for hosted provider (optional for local vLLM / mock)",
    )
    llm_model: str = Field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        validation_alias="LLM_MODEL",
        description="Model name served by vLLM or hosted provider",
    )
    llm_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        validation_alias="LLM_TEMPERATURE",
        description="Sampling temperature",
    )
    llm_timeout: int = Field(
        default=60,
        ge=1,
        le=300,
        validation_alias="LLM_TIMEOUT",
        description="Request timeout in seconds",
    )

    # Workflow Orchestration
    max_revision_attempts: int = Field(
        default=1,
        ge=0,
        le=3,
        validation_alias="MAX_REVISION_ATTEMPTS",
        description="Maximum automatic revision loops triggered by Critic",
    )

    # Observability (Optional LangSmith)
    langsmith_tracing: bool = Field(
        default=False,
        validation_alias="LANGSMITH_TRACING",
        description="Enable LangSmith distributed tracing",
    )
    langsmith_api_key: Optional[str] = Field(
        default=None,
        validation_alias="LANGSMITH_API_KEY",
        description="LangSmith API key",
    )
    langsmith_project: str = Field(
        default="agentforge",
        validation_alias="LANGSMITH_PROJECT",
        description="LangSmith project name",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def is_mock(self) -> bool:
        """Check if running in deterministic mock mode."""
        return self.app_mode == AppMode.MOCK

    def is_vllm(self) -> bool:
        """Check if connecting to local vLLM server."""
        return self.app_mode == AppMode.VLLM

    def is_hosted(self) -> bool:
        """Check if connecting to a hosted provider."""
        return self.app_mode == AppMode.HOSTED

    def get_safe_info(self) -> Dict[str, Any]:
        """Return safe configuration dictionary without exposing secrets."""
        has_key = bool(self.llm_api_key and self.llm_api_key.strip())
        masked_key = (
            f"{self.llm_api_key[:4]}...{self.llm_api_key[-3:]}"
            if has_key and len(self.llm_api_key or "") > 8
            else ("Set" if has_key else "None")
        )
        return {
            "app_mode": self.app_mode.value,
            "llm_provider": self.llm_provider,
            "llm_base_url": self.llm_base_url,
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "llm_api_key": masked_key,
            "max_revision_attempts": self.max_revision_attempts,
            "langsmith_tracing": self.langsmith_tracing,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retrieve cached global settings instance."""
    return Settings()
