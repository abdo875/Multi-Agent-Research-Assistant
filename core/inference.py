"""Inference abstraction and model server diagnostics for AgentForge.

Encapsulates connections to local vLLM or remote OpenAI-compatible model servers.
Provides health diagnostics and architectural clarity regarding inference vs orchestration.
"""

import time
from typing import Any, Dict, List, Optional
import httpx
from core.config import AppMode, Settings


def check_inference_health(base_url: str, api_key: Optional[str] = None, timeout: float = 3.0) -> Dict[str, Any]:
    """Perform a lightweight health and model check against an OpenAI-compatible inference endpoint.

    Args:
        base_url: The root API endpoint (e.g., http://localhost:8000/v1).
        api_key: Optional authorization token.
        timeout: Request timeout in seconds.

    Returns:
        Dict containing online status, detected models, response latency, and diagnostics.
    """
    clean_url = base_url.rstrip("/")
    models_url = f"{clean_url}/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    start_time = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(models_url, headers=headers)
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

            if resp.status_code == 200:
                data = resp.json()
                models: List[str] = []
                if isinstance(data, dict) and "data" in data:
                    models = [m.get("id", "unknown") for m in data["data"] if isinstance(m, dict)]
                return {
                    "online": True,
                    "status_code": 200,
                    "latency_ms": elapsed_ms,
                    "available_models": models,
                    "message": f"Inference server responsive ({elapsed_ms}ms). Found {len(models)} model(s).",
                }
            else:
                return {
                    "online": False,
                    "status_code": resp.status_code,
                    "latency_ms": elapsed_ms,
                    "available_models": [],
                    "message": f"Server returned HTTP {resp.status_code}: {resp.text[:120]}",
                }
    except httpx.ConnectError:
        return {
            "online": False,
            "status_code": None,
            "latency_ms": None,
            "available_models": [],
            "message": f"Connection refused at {clean_url}. Is vLLM or the model server running?",
        }
    except Exception as exc:
        return {
            "online": False,
            "status_code": None,
            "latency_ms": None,
            "available_models": [],
            "message": f"Health check failed: {type(exc).__name__}: {str(exc)[:100]}",
        }


def get_inference_summary(settings: Settings) -> Dict[str, Any]:
    """Generate an overview of the inference architecture and current configuration."""
    mode = settings.app_mode

    if mode == AppMode.MOCK:
        return {
            "mode": "Mock Mode",
            "provider": "AgentForge Synthetic Engine",
            "model": "Deterministic Technical Knowledge Base",
            "status": "Ready (Offline / Zero-latency)",
            "details": (
                "Simulates real LLM responses with deterministic domain knowledge. "
                "Perfect for testing and fast, reproducible interview demonstrations."
            ),
            "endpoint": "In-memory",
            "requires_gpu": False,
        }

    health = check_inference_health(settings.llm_base_url, settings.llm_api_key)

    if mode == AppMode.VLLM:
        return {
            "mode": "Local vLLM Inference",
            "provider": "vLLM (PagedAttention & Continuous Batching)",
            "model": settings.llm_model,
            "status": "Online" if health["online"] else "Offline / Unreachable",
            "details": health["message"],
            "endpoint": settings.llm_base_url,
            "requires_gpu": True,
            "latency_ms": health.get("latency_ms"),
            "available_models": health.get("available_models", []),
        }

    # Hosted
    return {
        "mode": "Hosted LLM Endpoint",
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "status": "Online" if health["online"] else "Offline / Unreachable",
        "details": health["message"],
        "endpoint": settings.llm_base_url,
        "requires_gpu": False,
        "latency_ms": health.get("latency_ms"),
        "available_models": health.get("available_models", []),
    }
