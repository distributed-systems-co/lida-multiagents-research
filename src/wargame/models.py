"""
Dynamic Model Registry for Wargame Engine

Fetches latest models from OpenRouter, categorizes by tier/capability,
and assigns optimal models to personas based on their characteristics.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Cache for model data
_MODEL_CACHE: Optional[dict] = None
_CACHE_FILE = Path(__file__).parent / ".model_cache.json"
_CACHE_TTL = 3600 * 2  # 2 hours


@dataclass
class ModelInfo:
    """Information about an LLM model."""
    id: str
    name: str
    provider: str
    context_length: int
    input_cost: float   # per 1M tokens
    output_cost: float  # per 1M tokens
    created: int

    @property
    def total_cost(self) -> float:
        """Rough total cost estimate (input + output)."""
        return self.input_cost + self.output_cost

    @property
    def tier(self) -> str:
        """Classify model into cost tier."""
        if self.total_cost > 20:
            return "premium"
        elif self.total_cost > 5:
            return "standard"
        elif self.total_cost > 1:
            return "budget"
        else:
            return "free"


def fetch_models(force_refresh: bool = False) -> dict[str, list[ModelInfo]]:
    """
    Fetch all models from OpenRouter, grouped by provider.

    Returns: {provider: [ModelInfo, ...]} sorted by capability (newest first)
    """
    global _MODEL_CACHE

    # Check cache
    if not force_refresh and _MODEL_CACHE:
        return _MODEL_CACHE

    # Check file cache
    if not force_refresh and _CACHE_FILE.exists():
        try:
            with open(_CACHE_FILE) as f:
                cache = json.load(f)
            age = datetime.now(timezone.utc).timestamp() - cache.get("fetched_at", 0)
            if age < _CACHE_TTL:
                _MODEL_CACHE = _parse_cache(cache)
                return _MODEL_CACHE
        except Exception:
            pass

    # Fetch from API
    try:
        result = subprocess.run(
            ["curl", "-s", "--max-time", "30", "https://openrouter.ai/api/v1/models"],
            capture_output=True, text=True, timeout=35
        )
        if result.returncode != 0:
            raise Exception(f"curl failed: {result.stderr}")

        raw_data = json.loads(result.stdout).get("data", [])
    except Exception as e:
        print(f"[ModelRegistry] API fetch failed: {e}, using fallback")
        return _get_fallback_models()

    if not raw_data:
        return _get_fallback_models()

    # Parse and group by provider
    models_by_provider: dict[str, list[ModelInfo]] = {}

    for model in raw_data:
        model_id = model.get("id", "")
        if "/" not in model_id:
            continue

        provider = model_id.split("/")[0]
        pricing = model.get("pricing", {})

        info = ModelInfo(
            id=model_id,
            name=model.get("name", model_id),
            provider=provider,
            context_length=model.get("context_length", 0),
            input_cost=float(pricing.get("prompt", 0)) * 1_000_000,
            output_cost=float(pricing.get("completion", 0)) * 1_000_000,
            created=model.get("created", 0),
        )

        if provider not in models_by_provider:
            models_by_provider[provider] = []
        models_by_provider[provider].append(info)

    # Sort each provider's models by created date (newest first)
    for provider in models_by_provider:
        models_by_provider[provider].sort(key=lambda m: -m.created)

    # Save to cache
    _save_cache(models_by_provider)
    _MODEL_CACHE = models_by_provider

    return models_by_provider


def _parse_cache(cache: dict) -> dict[str, list[ModelInfo]]:
    """Parse cached model data."""
    result = {}
    for provider, models in cache.get("models", {}).items():
        result[provider] = [ModelInfo(**m) for m in models]
    return result


def _save_cache(models: dict[str, list[ModelInfo]]):
    """Save models to cache file."""
    try:
        cache = {
            "fetched_at": datetime.now(timezone.utc).timestamp(),
            "models": {
                provider: [
                    {
                        "id": m.id, "name": m.name, "provider": m.provider,
                        "context_length": m.context_length,
                        "input_cost": m.input_cost, "output_cost": m.output_cost,
                        "created": m.created
                    }
                    for m in models
                ]
                for provider, models in models.items()
            }
        }
        with open(_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


def _get_fallback_models() -> dict[str, list[ModelInfo]]:
    """Fallback models if API is unavailable."""
    return {
        "anthropic": [
            ModelInfo("anthropic/claude-opus-4.5", "Claude Opus 4.5", "anthropic", 200000, 5.0, 25.0, 0),
            ModelInfo("anthropic/claude-sonnet-4.5", "Claude Sonnet 4.5", "anthropic", 1000000, 3.0, 15.0, 0),
            ModelInfo("anthropic/claude-haiku-4.5", "Claude Haiku 4.5", "anthropic", 200000, 1.0, 5.0, 0),
        ],
        "openai": [
            ModelInfo("openai/gpt-5.2-codex", "GPT-5.2 Codex", "openai", 400000, 1.75, 14.0, 0),
            ModelInfo("openai/gpt-4.1", "GPT-4.1", "openai", 128000, 2.0, 8.0, 0),
            ModelInfo("openai/gpt-4.1-mini", "GPT-4.1 Mini", "openai", 128000, 0.4, 1.6, 0),
        ],
        "google": [
            ModelInfo("google/gemini-3-pro-preview", "Gemini 3 Pro", "google", 1048576, 2.0, 12.0, 0),
            ModelInfo("google/gemini-3-flash-preview", "Gemini 3 Flash", "google", 1048576, 0.5, 3.0, 0),
        ],
        "x-ai": [
            ModelInfo("x-ai/grok-4-fast", "Grok 4 Fast", "x-ai", 2000000, 0.2, 0.5, 0),
            ModelInfo("x-ai/grok-4.1-fast", "Grok 4.1 Fast", "x-ai", 2000000, 0.2, 0.5, 0),
        ],
        "deepseek": [
            ModelInfo("deepseek/deepseek-v3.2", "DeepSeek V3.2", "deepseek", 163840, 0.25, 0.38, 0),
            ModelInfo("deepseek/deepseek-v3.2-speciale", "DeepSeek V3.2 Speciale", "deepseek", 163840, 0.27, 0.41, 0),
        ],
        "meta-llama": [
            ModelInfo("meta-llama/llama-4-maverick", "Llama 4 Maverick", "meta-llama", 1048576, 0.15, 0.6, 0),
            ModelInfo("meta-llama/llama-4-scout", "Llama 4 Scout", "meta-llama", 327680, 0.08, 0.3, 0),
        ],
    }


def get_latest_models(providers: list[str] = None, top_n: int = 3) -> dict[str, list[ModelInfo]]:
    """Get the latest N models from specified providers."""
    all_models = fetch_models()

    if providers is None:
        providers = ["anthropic", "openai", "google", "x-ai", "deepseek", "meta-llama", "mistralai", "qwen"]

    result = {}
    for provider in providers:
        if provider in all_models:
            result[provider] = all_models[provider][:top_n]

    return result


# =============================================================================
# Persona-to-Model Assignment
# =============================================================================

# Model assignment strategies based on persona characteristics
PERSONA_MODEL_MAPPING = {
    # By stance - what kind of reasoning style fits
    "doomer": {
        "providers": ["anthropic", "openai"],  # Strong reasoning models
        "tier": "premium",
        "reason": "Complex safety arguments need sophisticated reasoning"
    },
    "pro_safety": {
        "providers": ["anthropic", "google"],  # Careful, nuanced
        "tier": "standard",
        "reason": "Balanced safety/capability arguments"
    },
    "moderate": {
        "providers": ["anthropic", "google", "openai"],  # Flexible
        "tier": "standard",
        "reason": "Diplomatic, sees multiple sides"
    },
    "pro_industry": {
        "providers": ["x-ai", "meta-llama", "openai"],  # Fast, confident
        "tier": "budget",
        "reason": "Bold, action-oriented arguments"
    },
    "accelerationist": {
        "providers": ["x-ai", "deepseek", "meta-llama"],  # Edgy, fast
        "tier": "budget",
        "reason": "Provocative, contrarian style"
    },

    # By category - expertise area
    "tech_leader": {
        "providers": ["anthropic", "openai", "x-ai"],
        "tier": "standard",
    },
    "politician": {
        "providers": ["anthropic", "google"],
        "tier": "standard",
    },
    "researcher": {
        "providers": ["anthropic", "deepseek"],
        "tier": "premium",
    },
    "regulator": {
        "providers": ["anthropic", "google"],
        "tier": "standard",
    },
}


def assign_model_to_persona(
    persona_id: str,
    stance: str,
    category: str = "other",
    prefer_cheap: bool = False,
) -> str:
    """
    Assign an optimal model to a persona based on their characteristics.

    Args:
        persona_id: The persona identifier
        stance: doomer, pro_safety, moderate, pro_industry, accelerationist
        category: tech_leader, politician, researcher, etc.
        prefer_cheap: If True, prefer budget models

    Returns:
        Model ID string (e.g., "anthropic/claude-sonnet-4.5")
    """
    models = fetch_models()

    # Get mapping for this stance
    mapping = PERSONA_MODEL_MAPPING.get(stance, {
        "providers": ["anthropic"],
        "tier": "standard"
    })

    preferred_providers = mapping.get("providers", ["anthropic"])
    preferred_tier = "budget" if prefer_cheap else mapping.get("tier", "standard")

    # Find best matching model
    for provider in preferred_providers:
        if provider not in models:
            continue

        provider_models = models[provider]

        # Filter by tier preference
        if preferred_tier == "premium":
            candidates = [m for m in provider_models if m.tier == "premium"]
        elif preferred_tier == "budget":
            candidates = [m for m in provider_models if m.tier in ["budget", "free"]]
        else:
            candidates = [m for m in provider_models if m.tier in ["standard", "premium"]]

        if not candidates:
            candidates = provider_models[:2]  # Just take newest

        if candidates:
            return candidates[0].id

    # Fallback
    return "anthropic/claude-sonnet-4.5"


def get_model_diversity_assignment(personas: list[tuple[str, str, str]]) -> dict[str, str]:
    """
    Assign models to a list of personas ensuring diversity.

    Args:
        personas: List of (persona_id, stance, category) tuples

    Returns:
        {persona_id: model_id} mapping
    """
    # Models that require special handling or don't work well for chat
    EXCLUDED_MODELS = {
        "gpt-5", "gpt-audio", "codex", "reasoning", "preview",
        "o1", "o3", "realtime", "tts", "whisper", "dall-e",
        "embedding", "moderation"
    }

    def is_usable(model_id: str) -> bool:
        """Check if model is usable for standard chat."""
        model_lower = model_id.lower()
        return not any(ex in model_lower for ex in EXCLUDED_MODELS)

    models = fetch_models()
    assignments = {}
    used_models = set()

    for persona_id, stance, category in personas:
        mapping = PERSONA_MODEL_MAPPING.get(stance, {"providers": ["anthropic"]})
        preferred_providers = mapping["providers"]

        # Try to pick a model we haven't used yet
        for provider in preferred_providers:
            if provider not in models:
                continue

            for model in models[provider][:5]:  # Check more models to find usable ones
                if model.id not in used_models and is_usable(model.id):
                    assignments[persona_id] = model.id
                    used_models.add(model.id)
                    break

            if persona_id in assignments:
                break

        # Fallback to safe defaults if all models used or excluded
        if persona_id not in assignments:
            fallback = "anthropic/claude-3.5-haiku"
            assignments[persona_id] = fallback

    return assignments


def print_model_summary():
    """Print summary of available models."""
    models = get_latest_models(top_n=3)

    print("=" * 70)
    print("AVAILABLE MODELS (Latest 3 per Provider)")
    print("=" * 70)

    for provider, provider_models in sorted(models.items()):
        print(f"\n{provider.upper()}:")
        for m in provider_models:
            tier_emoji = {"premium": "💎", "standard": "⭐", "budget": "💰", "free": "🆓"}.get(m.tier, "")
            print(f"  {tier_emoji} {m.id}")
            print(f"      Context: {m.context_length:,} | ${m.input_cost:.2f}/${m.output_cost:.2f} per 1M tokens")


if __name__ == "__main__":
    print_model_summary()
