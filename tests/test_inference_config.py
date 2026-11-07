"""Tests for configuration validation and inference server diagnostics."""

import pytest
from pydantic import ValidationError
from core.config import AppMode, Settings
from core.inference import check_inference_health, get_inference_summary


def test_default_settings_is_mock():
    """Verify default configuration runs safely in Mock Mode with zero credentials."""
    settings = Settings()
    assert settings.app_mode == AppMode.MOCK
    assert settings.is_mock() is True
    assert settings.llm_base_url == "http://localhost:8000/v1"
    assert settings.max_revision_attempts == 1


def test_settings_secret_masking():
    """Ensure sensitive API keys are never leaked in safe representation."""
    settings = Settings(
        app_mode=AppMode.HOSTED,
        llm_api_key="sk-proj-1234567890abcdefg",
    )
    safe_info = settings.get_safe_info()

    assert "1234567890abcdefg" not in safe_info["llm_api_key"]
    assert safe_info["llm_api_key"].startswith("sk-p...")
    assert safe_info["app_mode"] == "hosted"


def test_invalid_temperature_validation():
    """Verify temperature bounds enforcement."""
    with pytest.raises(ValidationError):
        Settings(llm_temperature=3.5)  # Max allowed is 2.0


def test_invalid_revision_limit_validation():
    """Verify max revision attempts validation."""
    with pytest.raises(ValidationError):
        Settings(max_revision_attempts=10)  # Max allowed is 3


def test_inference_health_check_graceful_offline():
    """Verify health check gracefully handles unreachable ports without crashing."""
    result = check_inference_health("http://127.0.0.1:59999/v1", timeout=0.5)

    assert result["online"] is False
    assert result["status_code"] is None
    assert "Connection refused" in result["message"] or "failed" in result["message"].lower()


def test_get_inference_summary_mock_mode():
    """Verify inference summary for Mock Mode describes architecture accurately."""
    settings = Settings(app_mode=AppMode.MOCK)
    summary = get_inference_summary(settings)

    assert summary["mode"] == "Mock Mode"
    assert summary["requires_gpu"] is False
    assert "Synthetic" in summary["provider"] or "Knowledge" in summary["model"]
