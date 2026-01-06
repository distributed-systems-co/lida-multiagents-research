"""LLM Provider configuration and management.

Dynamically fetches latest models from OpenRouter and other providers.
"""

from __future__ import annotations

import os
import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, AsyncIterator

import httpx

logger = logging.getLogger(__name__)


class ProviderType(str, Enum):
    """Supported LLM provider types."""
    OPENROUTER = "openrouter"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"
    VLLM = "vllm"
    CUSTOM = "custom"


class ModelCapability(str, Enum):
    """Model capabilities."""
    CHAT = "chat"
    COMPLETION = "completion"
    VISION = "vision"
    FUNCTION_CALLING = "function_calling"
    STREAMING = "streaming"
    JSON_MODE = "json_mode"
    LONG_CONTEXT = "long_context"
    CODE = "code"
    REASONING = "reasoning"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    model_id: str
    name: str
    provider: ProviderType
    provider_model_id: str
    context_length: int = 8192
    max_output_tokens: int = 4096
    capabilities: List[ModelCapability] = field(default_factory=list)
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    description: str = ""
    created_at: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "provider": self.provider.value,
            "provider_model_id": self.provider_model_id,
            "context_length": self.context_length,
            "max_output_tokens": self.max_output_tokens,
            "capabilities": [c.value for c in self.capabilities],
            "cost_per_1k_input": self.cost_per_1k_input,
            "cost_per_1k_output": self.cost_per_1k_output,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metadata": self.metadata,
        }


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    provider_id: str
    provider_type: ProviderType
    name: str
    base_url: str
    api_key_env: str
    api_key: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    default_model: Optional[str] = None
    timeout: float = 120.0
    max_retries: int = 3
    enabled: bool = True
    metadata: dict = field(default_factory=dict)

    def get_api_key(self) -> Optional[str]:
        if self.api_key:
            return self.api_key
        return os.getenv(self.api_key_env)

    def to_dict(self) -> dict:
        return {
            "provider_id": self.provider_id,
            "provider_type": self.provider_type.value,
            "name": self.name,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "default_model": self.default_model,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "enabled": self.enabled,
            "has_api_key": bool(self.get_api_key()),
        }


@dataclass
class AgentModelConfig:
    """Model configuration for an agent."""
    primary_model: str
    fallback_models: List[str] = field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = field(default_factory=list)
    use_streaming: bool = True
    retry_on_failure: bool = True
    timeout: float = 120.0
    custom_params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "primary_model": self.primary_model,
            "fallback_models": self.fallback_models,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop_sequences": self.stop_sequences,
            "use_streaming": self.use_streaming,
            "retry_on_failure": self.retry_on_failure,
            "timeout": self.timeout,
        }


# Built-in providers
BUILTIN_PROVIDERS: Dict[str, ProviderConfig] = {
    "openrouter": ProviderConfig(
        provider_id="openrouter",
        provider_type=ProviderType.OPENROUTER,
        name="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        default_model="anthropic/claude-sonnet-4",
        headers={
            "HTTP-Referer": "https://github.com/distributed-systems-co/lida-multiagents-research",
            "X-Title": "LIDA Multi-Agent System",
        },
    ),
    "anthropic": ProviderConfig(
        provider_id="anthropic",
        provider_type=ProviderType.ANTHROPIC,
        name="Anthropic",
        base_url="https://api.anthropic.com/v1",
        api_key_env="ANTHROPIC_API_KEY",
        default_model="claude-sonnet-4-20250514",
        headers={"anthropic-version": "2023-06-01"},
    ),
    "openai": ProviderConfig(
        provider_id="openai",
        provider_type=ProviderType.OPENAI,
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-4o",
    ),
    "ollama": ProviderConfig(
        provider_id="ollama",
        provider_type=ProviderType.OLLAMA,
        name="Ollama (Local)",
        base_url="http://localhost:11434/api",
        api_key_env="",
        default_model="llama3.2",
        enabled=True,
    ),
}


class OpenRouterModelFetcher:
    """Fetches latest models from OpenRouter API."""

    OPENROUTER_API = "https://openrouter.ai/api/v1/models"

    def __init__(self):
        self._cache: Dict[str, List[ModelConfig]] = {}
        self._last_fetch: Optional[datetime] = None
        self._cache_ttl = 3600  # 1 hour

    async def fetch_all_models(self, force: bool = False) -> List[ModelConfig]:
        """Fetch all models from OpenRouter."""
        now = datetime.utcnow()
        if not force and self._last_fetch and self._cache.get("all"):
            elapsed = (now - self._last_fetch).total_seconds()
            if elapsed < self._cache_ttl:
                return self._cache["all"]

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.get(self.OPENROUTER_API)
                resp.raise_for_status()
                data = resp.json()

                models = []
                for m in data.get("data", []):
                    model = self._parse_model(m)
                    if model:
                        models.append(model)

                # Sort by created date descending (newest first)
                models.sort(
                    key=lambda x: x.created_at or datetime.min,
                    reverse=True,
                )

                self._cache["all"] = models
                self._last_fetch = now
                logger.info(f"Fetched {len(models)} models from OpenRouter")
                return models

            except Exception as e:
                logger.error(f"Failed to fetch OpenRouter models: {e}")
                return self._cache.get("all", [])

    def _parse_model(self, data: dict) -> Optional[ModelConfig]:
        """Parse OpenRouter model data."""
        try:
            model_id = data.get("id", "")
            if not model_id:
                return None

            # Parse created timestamp
            created = None
            if data.get("created"):
                try:
                    created = datetime.fromtimestamp(data["created"])
                except (ValueError, TypeError):
                    pass

            # Parse pricing
            pricing = data.get("pricing", {})
            cost_in = float(pricing.get("prompt", 0)) * 1000 if pricing.get("prompt") else 0
            cost_out = float(pricing.get("completion", 0)) * 1000 if pricing.get("completion") else 0

            # Parse capabilities
            caps = [ModelCapability.CHAT, ModelCapability.STREAMING]
            arch = data.get("architecture", {})
            if arch.get("modality") == "multimodal" or "vision" in model_id.lower():
                caps.append(ModelCapability.VISION)
            if data.get("context_length", 0) > 32000:
                caps.append(ModelCapability.LONG_CONTEXT)

            # Simple name from ID
            name = data.get("name") or model_id.split("/")[-1].replace("-", " ").title()

            return ModelConfig(
                model_id=model_id,
                name=name,
                provider=ProviderType.OPENROUTER,
                provider_model_id=model_id,
                context_length=data.get("context_length", 8192),
                max_output_tokens=data.get("top_provider", {}).get("max_completion_tokens", 4096),
                capabilities=caps,
                cost_per_1k_input=cost_in,
                cost_per_1k_output=cost_out,
                description=data.get("description", ""),
                created_at=created,
                metadata={
                    "top_provider": data.get("top_provider", {}),
                    "architecture": arch,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to parse model: {e}")
            return None

    async def get_latest_by_provider(
        self,
        provider_prefix: str,
        limit: int = 5,
        dedupe: bool = True,
    ) -> List[ModelConfig]:
        """Get latest N models from a specific provider.

        Args:
            provider_prefix: e.g. "anthropic", "openai", "google", "meta-llama"
            limit: Max models to return
            dedupe: Deduplicate by base model name
        """
        all_models = await self.fetch_all_models()

        # Filter by provider
        filtered = [m for m in all_models if m.model_id.startswith(f"{provider_prefix}/")]

        if dedupe:
            # Dedupe by base name (e.g., keep only newest claude-sonnet-4)
            seen_bases = {}
            deduped = []
            for m in filtered:
                # Extract base name without version/date suffixes
                base = self._get_base_name(m.model_id)
                if base not in seen_bases:
                    seen_bases[base] = m
                    deduped.append(m)
            filtered = deduped

        return filtered[:limit]

    def _get_base_name(self, model_id: str) -> str:
        """Extract base model name for deduplication."""
        # Remove provider prefix
        name = model_id.split("/")[-1] if "/" in model_id else model_id

        # Remove date suffixes like -20250514, :20250514
        import re
        name = re.sub(r"[-:]20\d{6}$", "", name)
        name = re.sub(r"[-:]latest$", "", name)

        # Remove version suffixes
        name = re.sub(r"-\d+\.\d+$", "", name)

        return name

    async def get_all_latest(
        self,
        models_per_provider: int = 5,
        dedupe: bool = True,
    ) -> Dict[str, List[ModelConfig]]:
        """Get latest models from all major providers."""
        providers = [
            "anthropic",
            "openai",
            "google",
            "meta-llama",
            "mistralai",
            "deepseek",
            "qwen",
            "cohere",
            "x-ai",
            "perplexity",
        ]

        results = {}
        for provider in providers:
            models = await self.get_latest_by_provider(
                provider,
                limit=models_per_provider,
                dedupe=dedupe,
            )
            if models:
                results[provider] = models

        return results


class ModelRegistry:
    """Registry for managing models and providers."""

    def __init__(self):
        self._providers: Dict[str, ProviderConfig] = {}
        self._models: Dict[str, ModelConfig] = {}
        self._clients: Dict[str, Any] = {}
        self._fetcher = OpenRouterModelFetcher()
        self._initialized = False

        # Load built-in providers
        for pid, config in BUILTIN_PROVIDERS.items():
            self._providers[pid] = config

    async def initialize(self, force_refresh: bool = False):
        """Initialize registry with latest models from OpenRouter."""
        if self._initialized and not force_refresh:
            return

        logger.info("Initializing model registry with latest models...")

        try:
            all_latest = await self._fetcher.get_all_latest(
                models_per_provider=10,
                dedupe=True,
            )

            for provider, models in all_latest.items():
                for model in models:
                    self._models[model.model_id] = model
                    # Also register with short name
                    short_name = model.model_id.split("/")[-1]
                    if short_name not in self._models:
                        self._models[short_name] = model

            self._initialized = True
            logger.info(f"Loaded {len(self._models)} models from OpenRouter")

        except Exception as e:
            logger.error(f"Failed to initialize model registry: {e}")
            # Fall back to hardcoded latest models
            self._load_fallback_models()

    def _load_fallback_models(self):
        """Load fallback models if API fetch fails."""
        fallbacks = [
            # Anthropic latest
            ModelConfig(
                model_id="anthropic/claude-opus-4",
                name="Claude Opus 4",
                provider=ProviderType.OPENROUTER,
                provider_model_id="anthropic/claude-opus-4",
                context_length=200000,
                max_output_tokens=32000,
                capabilities=[ModelCapability.CHAT, ModelCapability.VISION, ModelCapability.STREAMING, ModelCapability.REASONING],
            ),
            ModelConfig(
                model_id="anthropic/claude-sonnet-4",
                name="Claude Sonnet 4",
                provider=ProviderType.OPENROUTER,
                provider_model_id="anthropic/claude-sonnet-4",
                context_length=200000,
                max_output_tokens=64000,
                capabilities=[ModelCapability.CHAT, ModelCapability.VISION, ModelCapability.STREAMING, ModelCapability.REASONING, ModelCapability.CODE],
            ),
            ModelConfig(
                model_id="anthropic/claude-haiku-4",
                name="Claude Haiku 4",
                provider=ProviderType.OPENROUTER,
                provider_model_id="anthropic/claude-haiku-4",
                context_length=200000,
                max_output_tokens=8192,
                capabilities=[ModelCapability.CHAT, ModelCapability.VISION, ModelCapability.STREAMING],
            ),
            # OpenAI latest
            ModelConfig(
                model_id="openai/gpt-4.1",
                name="GPT-4.1",
                provider=ProviderType.OPENROUTER,
                provider_model_id="openai/gpt-4.1",
                context_length=1047576,
                max_output_tokens=32768,
                capabilities=[ModelCapability.CHAT, ModelCapability.VISION, ModelCapability.STREAMING, ModelCapability.FUNCTION_CALLING],
            ),
            ModelConfig(
                model_id="openai/gpt-4.1-mini",
                name="GPT-4.1 Mini",
                provider=ProviderType.OPENROUTER,
                provider_model_id="openai/gpt-4.1-mini",
                context_length=1047576,
                max_output_tokens=32768,
                capabilities=[ModelCapability.CHAT, ModelCapability.VISION, ModelCapability.STREAMING, ModelCapability.FUNCTION_CALLING],
            ),
            ModelConfig(
                model_id="openai/o3-mini",
                name="O3 Mini",
                provider=ProviderType.OPENROUTER,
                provider_model_id="openai/o3-mini",
                context_length=200000,
                max_output_tokens=100000,
                capabilities=[ModelCapability.CHAT, ModelCapability.STREAMING, ModelCapability.REASONING],
            ),
            # Google latest
            ModelConfig(
                model_id="google/gemini-2.5-pro-preview",
                name="Gemini 2.5 Pro Preview",
                provider=ProviderType.OPENROUTER,
                provider_model_id="google/gemini-2.5-pro-preview-06-05",
                context_length=1048576,
                max_output_tokens=65536,
                capabilities=[ModelCapability.CHAT, ModelCapability.VISION, ModelCapability.STREAMING, ModelCapability.LONG_CONTEXT],
            ),
            # DeepSeek latest
            ModelConfig(
                model_id="deepseek/deepseek-r1",
                name="DeepSeek R1",
                provider=ProviderType.OPENROUTER,
                provider_model_id="deepseek/deepseek-r1",
                context_length=163840,
                max_output_tokens=163840,
                capabilities=[ModelCapability.CHAT, ModelCapability.STREAMING, ModelCapability.REASONING, ModelCapability.CODE],
            ),
            ModelConfig(
                model_id="deepseek/deepseek-chat-v3",
                name="DeepSeek Chat V3",
                provider=ProviderType.OPENROUTER,
                provider_model_id="deepseek/deepseek-chat-v3-0324",
                context_length=131072,
                max_output_tokens=16384,
                capabilities=[ModelCapability.CHAT, ModelCapability.STREAMING, ModelCapability.CODE],
            ),
            # Meta latest
            ModelConfig(
                model_id="meta-llama/llama-4-maverick",
                name="Llama 4 Maverick",
                provider=ProviderType.OPENROUTER,
                provider_model_id="meta-llama/llama-4-maverick",
                context_length=1048576,
                max_output_tokens=131072,
                capabilities=[ModelCapability.CHAT, ModelCapability.VISION, ModelCapability.STREAMING],
            ),
            # xAI
            ModelConfig(
                model_id="x-ai/grok-3-beta",
                name="Grok 3 Beta",
                provider=ProviderType.OPENROUTER,
                provider_model_id="x-ai/grok-3-beta",
                context_length=131072,
                max_output_tokens=131072,
                capabilities=[ModelCapability.CHAT, ModelCapability.STREAMING, ModelCapability.REASONING],
            ),
        ]

        for model in fallbacks:
            self._models[model.model_id] = model
            short_name = model.model_id.split("/")[-1]
            if short_name not in self._models:
                self._models[short_name] = model

        self._initialized = True
        logger.info(f"Loaded {len(fallbacks)} fallback models")

    def register_provider(self, config: ProviderConfig):
        self._providers[config.provider_id] = config

    def register_model(self, config: ModelConfig):
        self._models[config.model_id] = config

    def get_provider(self, provider_id: str) -> Optional[ProviderConfig]:
        return self._providers.get(provider_id)

    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        return self._models.get(model_id)

    def list_providers(self, enabled_only: bool = True) -> List[ProviderConfig]:
        providers = list(self._providers.values())
        if enabled_only:
            providers = [p for p in providers if p.enabled]
        return providers

    def list_models(
        self,
        provider: Optional[ProviderType] = None,
        capability: Optional[ModelCapability] = None,
    ) -> List[ModelConfig]:
        models = list(self._models.values())
        if provider:
            models = [m for m in models if m.provider == provider]
        if capability:
            models = [m for m in models if capability in m.capabilities]
        return models

    def get_models_by_provider(self, provider_id: str) -> List[ModelConfig]:
        provider = self._providers.get(provider_id)
        if not provider:
            return []
        return [m for m in self._models.values() if m.provider == provider.provider_type]

    async def refresh_models(self):
        """Force refresh models from OpenRouter."""
        self._initialized = False
        await self.initialize(force_refresh=True)

    def get_client(self, provider_id: str) -> Any:
        if provider_id not in self._clients:
            provider = self._providers.get(provider_id)
            if not provider:
                raise ValueError(f"Unknown provider: {provider_id}")

            headers = {"Content-Type": "application/json", **provider.headers}
            api_key = provider.get_api_key()
            if api_key:
                if provider.provider_type == ProviderType.ANTHROPIC:
                    headers["x-api-key"] = api_key
                else:
                    headers["Authorization"] = f"Bearer {api_key}"

            self._clients[provider_id] = httpx.AsyncClient(
                base_url=provider.base_url,
                headers=headers,
                timeout=httpx.Timeout(provider.timeout),
            )

        return self._clients[provider_id]

    async def close(self):
        for client in self._clients.values():
            if not client.is_closed:
                await client.aclose()
        self._clients.clear()


class UnifiedLLMClient:
    """Unified client for all LLM providers."""

    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or ModelRegistry()

    async def complete(
        self,
        messages: List[Dict[str, str]],
        model_id: str = "anthropic/claude-sonnet-4",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
        **kwargs,
    ) -> Dict[str, Any] | AsyncIterator[str]:
        # Auto-initialize if needed
        if not self.registry._initialized:
            await self.registry.initialize()

        model = self.registry.get_model(model_id)
        if not model:
            # Try as direct provider model ID
            model = ModelConfig(
                model_id=model_id,
                name=model_id,
                provider=ProviderType.OPENROUTER,
                provider_model_id=model_id,
            )

        provider = self.registry.get_provider(model.provider.value)
        if not provider:
            provider = self.registry.get_provider("openrouter")

        client = self.registry.get_client(provider.provider_id)

        if provider.provider_type == ProviderType.OLLAMA:
            return await self._complete_ollama(client, model, messages, temperature, max_tokens, stream, **kwargs)
        elif provider.provider_type == ProviderType.ANTHROPIC:
            return await self._complete_anthropic(client, model, messages, temperature, max_tokens, stream, **kwargs)
        else:
            return await self._complete_openai(client, model, messages, temperature, max_tokens, stream, **kwargs)

    async def _complete_openai(self, client, model, messages, temperature, max_tokens, stream, **kwargs):
        payload = {
            "model": model.provider_model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **kwargs,
        }
        if stream:
            return self._stream_openai(client, payload)
        else:
            response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            return {
                "content": data["choices"][0]["message"]["content"],
                "model": data.get("model", model.model_id),
                "usage": data.get("usage", {}),
                "finish_reason": data["choices"][0].get("finish_reason"),
            }

    async def _stream_openai(self, client, payload):
        import json
        async with client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    if "choices" in chunk and chunk["choices"]:
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                except json.JSONDecodeError:
                    continue

    async def _complete_anthropic(self, client, model, messages, temperature, max_tokens, stream, **kwargs):
        system = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append(msg)

        payload = {
            "model": model.provider_model_id,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if system:
            payload["system"] = system

        if stream:
            return self._stream_anthropic(client, payload)
        else:
            response = await client.post("/messages", json=payload)
            response.raise_for_status()
            data = response.json()
            return {
                "content": data["content"][0]["text"],
                "model": data.get("model", model.model_id),
                "usage": data.get("usage", {}),
                "finish_reason": data.get("stop_reason"),
            }

    async def _stream_anthropic(self, client, payload):
        import json
        async with client.stream("POST", "/messages", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                try:
                    chunk = json.loads(data_str)
                    if chunk.get("type") == "content_block_delta":
                        delta = chunk.get("delta", {})
                        text = delta.get("text", "")
                        if text:
                            yield text
                except json.JSONDecodeError:
                    continue

    async def _complete_ollama(self, client, model, messages, temperature, max_tokens, stream, **kwargs):
        payload = {
            "model": model.provider_model_id,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if stream:
            return self._stream_ollama(client, payload)
        else:
            response = await client.post("/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            return {
                "content": data["message"]["content"],
                "model": model.model_id,
                "usage": {
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                },
            }

    async def _stream_ollama(self, client, payload):
        import json
        async with client.stream("POST", "/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    if "message" in chunk:
                        content = chunk["message"].get("content", "")
                        if content:
                            yield content
                except json.JSONDecodeError:
                    continue

    async def close(self):
        await self.registry.close()


# Global instances
_registry: Optional[ModelRegistry] = None
_client: Optional[UnifiedLLMClient] = None


def get_model_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def get_unified_client() -> UnifiedLLMClient:
    global _client
    if _client is None:
        _client = UnifiedLLMClient(get_model_registry())
    return _client


async def fetch_latest_models(
    models_per_provider: int = 10,
    dedupe: bool = True,
) -> Dict[str, List[ModelConfig]]:
    """Fetch latest models from OpenRouter for all providers."""
    fetcher = OpenRouterModelFetcher()
    return await fetcher.get_all_latest(
        models_per_provider=models_per_provider,
        dedupe=dedupe,
    )
