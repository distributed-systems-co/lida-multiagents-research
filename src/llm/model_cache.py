"""OpenRouter Model Cache - Jan 2026 Models.

Fetches and caches available models from OpenRouter API.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import httpx

CACHE_FILE = Path(__file__).parent / ".model_cache.json"
CACHE_TTL = 3600 * 24  # 24 hours


@dataclass
class ModelInfo:
    """Information about an OpenRouter model."""
    id: str
    name: str
    context_length: int
    pricing_prompt: float  # per 1M tokens
    pricing_completion: float
    supports_logprobs: bool = False
    supports_tools: bool = False
    supports_vision: bool = False
    provider: str = ""

    @property
    def cost_per_1k_tokens(self) -> float:
        """Average cost per 1K tokens."""
        return (self.pricing_prompt + self.pricing_completion) / 2000


# Best models by category (Jan 2026)
RECOMMENDED_MODELS = {
    # Frontier reasoning
    "best_reasoning": "openai/o3",
    "fast_reasoning": "openai/o4-mini",

    # General purpose - large context
    "best_large_context": "openai/gpt-4.1",
    "fast_large_context": "openai/gpt-4.1-mini",
    "cheap_large_context": "openai/gpt-4.1-nano",

    # Latest GPT
    "gpt_frontier": "openai/gpt-5.2",
    "gpt_fast": "openai/gpt-5-mini",
    "gpt_cheap": "openai/gpt-5-nano",

    # Claude
    "claude_frontier": "anthropic/claude-opus-4.5",
    "claude_fast": "anthropic/claude-sonnet-4.5",
    "claude_cheap": "anthropic/claude-haiku-4.5",

    # Gemini
    "gemini_frontier": "google/gemini-2.5-pro",
    "gemini_fast": "google/gemini-2.5-flash",
    "gemini_cheap": "google/gemini-2.5-flash-lite",
    "gemini_free": "google/gemini-2.0-flash-exp:free",

    # Open source
    "llama_best": "meta-llama/llama-4-maverick",

    # Grok (huge context)
    "grok_best": "x-ai/grok-4.1-fast",

    # Code
    "code_best": "openai/gpt-5.1-codex",
    "code_fast": "openai/gpt-5.1-codex-mini",

    # Logprobs support (for confidence extraction)
    "logprobs_best": "openai/gpt-4o",
    "logprobs_fast": "openai/gpt-4o-mini",

    # Default for deliberation (good balance of cost/quality)
    "default": "openai/gpt-4.1-mini",
    "default_cheap": "google/gemini-2.5-flash-lite",
    "default_free": "google/gemini-2.0-flash-exp:free",
}

# Models VERIFIED to actually return logprobs via OpenRouter (Jan 2026 testing)
# Note: Many models list logprobs in supported_parameters but don't return them
LOGPROB_MODELS_VERIFIED = {
    # OpenAI - confirmed working
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/gpt-4o-2024-08-06",
    "openai/gpt-4o-2024-11-20",
    "openai/chatgpt-4o-latest",
    # DeepSeek - only v3.2 works
    "deepseek/deepseek-v3.2",
    # Qwen - only 32b works
    "qwen/qwen3-32b",
    # Llama - both work
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
}

# Fallback for API detection (includes unverified)
LOGPROB_MODELS = LOGPROB_MODELS_VERIFIED


class ModelCache:
    """Cache for OpenRouter models."""

    def __init__(self):
        self._models: Dict[str, ModelInfo] = {}
        self._last_fetch: float = 0
        self._load_cache()

    def _load_cache(self):
        """Load models from cache file."""
        if CACHE_FILE.exists():
            try:
                data = json.loads(CACHE_FILE.read_text())
                self._last_fetch = data.get("timestamp", 0)

                # Check if cache is still valid
                if time.time() - self._last_fetch < CACHE_TTL:
                    for m in data.get("models", []):
                        info = ModelInfo(
                            id=m["id"],
                            name=m.get("name", m["id"]),
                            context_length=m.get("context_length", 0),
                            pricing_prompt=m.get("pricing_prompt", 0),
                            pricing_completion=m.get("pricing_completion", 0),
                            supports_logprobs=m.get("supports_logprobs", False),
                            supports_tools=m.get("supports_tools", False),
                            supports_vision=m.get("supports_vision", False),
                            provider=m.get("provider", ""),
                        )
                        self._models[info.id] = info
            except Exception:
                pass

    def _save_cache(self):
        """Save models to cache file."""
        data = {
            "timestamp": self._last_fetch,
            "models": [
                {
                    "id": m.id,
                    "name": m.name,
                    "context_length": m.context_length,
                    "pricing_prompt": m.pricing_prompt,
                    "pricing_completion": m.pricing_completion,
                    "supports_logprobs": m.supports_logprobs,
                    "supports_tools": m.supports_tools,
                    "supports_vision": m.supports_vision,
                    "provider": m.provider,
                }
                for m in self._models.values()
            ]
        }
        CACHE_FILE.write_text(json.dumps(data, indent=2))

    async def fetch_models(self, force: bool = False) -> Dict[str, ModelInfo]:
        """Fetch models from OpenRouter API."""
        if not force and self._models and time.time() - self._last_fetch < CACHE_TTL:
            return self._models

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get("https://openrouter.ai/api/v1/models")
            response.raise_for_status()
            data = response.json()

        self._models = {}
        for m in data.get("data", []):
            pricing = m.get("pricing", {})
            supported_params = m.get("supported_parameters", [])

            # Determine provider from ID
            provider = m["id"].split("/")[0] if "/" in m["id"] else "unknown"

            # Check capabilities from API response
            supports_logprobs = "logprobs" in supported_params or m["id"] in LOGPROB_MODELS
            supports_tools = "tools" in supported_params

            info = ModelInfo(
                id=m["id"],
                name=m.get("name", m["id"]),
                context_length=m.get("context_length", 0),
                pricing_prompt=float(pricing.get("prompt", 0)) * 1_000_000,
                pricing_completion=float(pricing.get("completion", 0)) * 1_000_000,
                supports_logprobs=supports_logprobs,
                supports_tools=supports_tools,
                supports_vision="vision" in m.get("architecture", {}).get("modality", ""),
                provider=provider,
            )
            self._models[info.id] = info

        self._last_fetch = time.time()
        self._save_cache()

        return self._models

    def fetch_models_sync(self, force: bool = False) -> Dict[str, ModelInfo]:
        """Synchronous version of fetch_models."""
        if not force and self._models and time.time() - self._last_fetch < CACHE_TTL:
            return self._models

        with httpx.Client(timeout=30.0) as client:
            response = client.get("https://openrouter.ai/api/v1/models")
            response.raise_for_status()
            data = response.json()

        self._models = {}
        for m in data.get("data", []):
            pricing = m.get("pricing", {})
            supported_params = m.get("supported_parameters", [])
            provider = m["id"].split("/")[0] if "/" in m["id"] else "unknown"

            # Check capabilities from API response
            supports_logprobs = "logprobs" in supported_params or m["id"] in LOGPROB_MODELS
            supports_tools = "tools" in supported_params

            info = ModelInfo(
                id=m["id"],
                name=m.get("name", m["id"]),
                context_length=m.get("context_length", 0),
                pricing_prompt=float(pricing.get("prompt", 0)) * 1_000_000,
                pricing_completion=float(pricing.get("completion", 0)) * 1_000_000,
                supports_logprobs=supports_logprobs,
                supports_tools=supports_tools,
                supports_vision="vision" in m.get("architecture", {}).get("modality", ""),
                provider=provider,
            )
            self._models[info.id] = info

        self._last_fetch = time.time()
        self._save_cache()

        return self._models

    def get(self, model_id: str) -> Optional[ModelInfo]:
        """Get model info by ID."""
        if not self._models:
            self.fetch_models_sync()
        return self._models.get(model_id)

    def get_recommended(self, category: str = "default") -> str:
        """Get recommended model ID for a category."""
        return RECOMMENDED_MODELS.get(category, RECOMMENDED_MODELS["default"])

    def list_by_provider(self, provider: str) -> List[ModelInfo]:
        """List models by provider."""
        if not self._models:
            self.fetch_models_sync()
        return [m for m in self._models.values() if m.provider == provider]

    def list_by_context(self, min_context: int = 100000) -> List[ModelInfo]:
        """List models with at least min_context length."""
        if not self._models:
            self.fetch_models_sync()
        return sorted(
            [m for m in self._models.values() if m.context_length >= min_context],
            key=lambda x: x.context_length,
            reverse=True
        )

    def list_with_logprobs(self) -> List[ModelInfo]:
        """List models that support logprobs."""
        if not self._models:
            self.fetch_models_sync()
        return [m for m in self._models.values() if m.supports_logprobs]


# Global cache instance
_cache: Optional[ModelCache] = None


def get_model_cache() -> ModelCache:
    """Get the global model cache."""
    global _cache
    if _cache is None:
        _cache = ModelCache()
    return _cache


def get_model(model_id: str) -> Optional[ModelInfo]:
    """Get model info by ID."""
    return get_model_cache().get(model_id)


def get_recommended_model(category: str = "default") -> str:
    """Get recommended model for a category."""
    return get_model_cache().get_recommended(category)


def list_models(
    provider: Optional[str] = None,
    min_context: Optional[int] = None,
    logprobs_only: bool = False,
) -> List[ModelInfo]:
    """List models with optional filters."""
    cache = get_model_cache()

    if provider:
        models = cache.list_by_provider(provider)
    elif min_context:
        models = cache.list_by_context(min_context)
    elif logprobs_only:
        models = cache.list_with_logprobs()
    else:
        if not cache._models:
            cache.fetch_models_sync()
        models = list(cache._models.values())

    return models


# Quick access to recommended models
def default_model() -> str:
    return RECOMMENDED_MODELS["default"]

def cheap_model() -> str:
    return RECOMMENDED_MODELS["default_cheap"]

def free_model() -> str:
    return RECOMMENDED_MODELS["default_free"]

def reasoning_model() -> str:
    return RECOMMENDED_MODELS["best_reasoning"]

def fast_reasoning_model() -> str:
    return RECOMMENDED_MODELS["fast_reasoning"]

def frontier_model() -> str:
    return RECOMMENDED_MODELS["gpt_frontier"]
