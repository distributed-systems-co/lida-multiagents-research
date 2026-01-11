"""OpenRouter client with streaming support."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional, Any, Callable

import httpx

logger = logging.getLogger(__name__)

# LLM response logging
_llm_logger = None


def _is_full_logs() -> bool:
    """Check if full logging is enabled (re-checked each call)."""
    return os.getenv("FULL_LOGS", "").lower() in ("true", "1", "yes")


def _get_llm_logger():
    """Get or create the LLM response logger (DatasetStore)."""
    global _llm_logger
    if _llm_logger is None:
        try:
            from src.api.datasets import DatasetStore
            _llm_logger = DatasetStore()
        except Exception as e:
            logger.warning(f"Could not initialize LLM logger: {e}")
    return _llm_logger


def _extract_prompt(messages: list[dict]) -> str:
    """Extract prompt text from messages for logging."""
    # Get the last user message as the primary prompt
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            # Handle multimodal content (list format)
            if isinstance(content, list):
                texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                return " ".join(texts)
    return ""

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Popular models on OpenRouter (Jan 2026)
MODELS = {
    # Anthropic - Latest (use OpenRouter's short IDs)
    "opus-4.5": "anthropic/claude-opus-4.5",
    "claude-opus-4.5": "anthropic/claude-opus-4.5",
    "opus-4.1": "anthropic/claude-opus-4.1",
    "opus-4": "anthropic/claude-opus-4",
    "sonnet-4.5": "anthropic/claude-sonnet-4.5",
    "sonnet-4": "anthropic/claude-sonnet-4",
    "claude-sonnet-4": "anthropic/claude-sonnet-4",
    "haiku-4.5": "anthropic/claude-haiku-4.5",
    "claude-3.7-sonnet": "anthropic/claude-3.7-sonnet",
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "claude-3.5-haiku": "anthropic/claude-3.5-haiku",
    # OpenAI - Latest
    "gpt-4.1": "openai/gpt-4.1",
    "gpt-4.1-mini": "openai/gpt-4.1-mini",
    "gpt-4.1-nano": "openai/gpt-4.1-nano",
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "o3": "openai/o3",
    "o3-mini": "openai/o3-mini",
    "o1": "openai/o1",
    "o1-mini": "openai/o1-mini",
    # Google - Latest
    "gemini-2.5-pro": "google/gemini-2.5-pro-preview-06-05",
    "gemini-2.5-flash": "google/gemini-2.5-flash-preview-05-20",
    "gemini-2.0-flash": "google/gemini-2.0-flash-001",
    "gemini-exp": "google/gemini-exp-1206",
    # Meta Llama - Latest
    "llama-4-maverick": "meta-llama/llama-4-maverick",
    "llama-4-scout": "meta-llama/llama-4-scout",
    "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct",
    "llama-3.1-405b": "meta-llama/llama-3.1-405b-instruct",
    # DeepSeek - Latest
    "deepseek-r1": "deepseek/deepseek-r1",
    "deepseek-v3": "deepseek/deepseek-chat-v3-0324",
    "deepseek-chat": "deepseek/deepseek-chat",
    # Qwen - Latest
    "qwen3-235b": "qwen/qwen3-235b-a22b",
    "qwen3-32b": "qwen/qwen3-32b",
    "qwen3-30b": "qwen/qwen3-30b-a3b",
    "qwen-max": "qwen/qwen-max",
    # Mistral - Latest
    "mistral-large": "mistralai/mistral-large-2411",
    "mistral-medium": "mistralai/mistral-medium-3",
    "codestral": "mistralai/codestral-2501",
    # xAI - Latest
    "grok-4": "x-ai/grok-4",
    "grok-4-mini": "x-ai/grok-4-mini",
    "grok-3": "x-ai/grok-3-beta",
    "grok-3-mini": "x-ai/grok-3-mini-beta",
    # Cohere
    "command-r-plus": "cohere/command-r-plus-08-2024",
    "command-r": "cohere/command-r-08-2024",
}


@dataclass
class StreamingResponse:
    """Streaming response from OpenRouter."""

    content: str = ""
    model: str = ""
    finish_reason: Optional[str] = None
    usage: dict = field(default_factory=dict)
    raw_chunks: list = field(default_factory=list)

    def __str__(self) -> str:
        return self.content


@dataclass
class Message:
    """Chat message."""
    role: str
    content: str
    name: Optional[str] = None

    def to_dict(self) -> dict:
        d = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d


class OpenRouterClient:
    """Async OpenRouter client with streaming support."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "anthropic/claude-sonnet-4.5",
        base_url: str = OPENROUTER_BASE_URL,
        timeout: float = 120.0,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
        agent_id: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            logger.warning("No OpenRouter API key provided. Set OPENROUTER_API_KEY env var.")

        self.default_model = default_model
        self.base_url = base_url
        self.timeout = timeout
        self.site_url = site_url or "https://github.com/distributed-systems-co/lida-multiagents-research"
        self.site_name = site_name or "LIDA Multi-Agent System"
        self.agent_id = agent_id  # For LLM response logging

        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers=self._headers(),
            )
        return self._client

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name,
            "Content-Type": "application/json",
        }

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def complete(
        self,
        messages: list[Message | dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        **kwargs,
    ) -> StreamingResponse | AsyncIterator[str]:
        """
        Complete a chat conversation.

        Args:
            messages: List of messages
            model: Model to use (defaults to default_model)
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            stream: Whether to stream the response
            agent_id: Optional agent ID for logging (overrides client's agent_id)
            agent_name: Optional agent name for logging (e.g., "Elon Musk")
            **kwargs: Additional parameters passed to the API

        Returns:
            StreamingResponse if not streaming, async iterator of chunks if streaming
        """
        model = model or self.default_model

        # Resolve model aliases
        if model in MODELS:
            model = MODELS[model]

        # Convert messages to dicts
        msgs = [m.to_dict() if isinstance(m, Message) else m for m in messages]

        payload = {
            "model": model,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **kwargs,
        }

        # Use provided agent_id or fall back to client's agent_id
        effective_agent_id = agent_id or self.agent_id

        if stream:
            return self._stream_complete(payload, effective_agent_id, agent_name)
        else:
            return await self._complete(payload, effective_agent_id, agent_name)

    async def _complete(self, payload: dict, agent_id: Optional[str] = None, agent_name: Optional[str] = None) -> StreamingResponse:
        """Non-streaming completion."""
        client = await self._get_client()

        logger.info(f"OpenRouter API call - Requesting model: {payload['model']}")
        logger.debug(f"OpenRouter payload: {json.dumps({k: v for k, v in payload.items() if k != 'messages'}, indent=2)}")

        start_time = time.time()
        response = await client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        duration_ms = int((time.time() - start_time) * 1000)

        data = response.json()
        actual_model = data.get("model", payload["model"])

        logger.info(f"OpenRouter API response - Actual model used: {actual_model}")
        if actual_model != payload["model"]:
            logger.warning(f"Model mismatch! Requested: {payload['model']}, Got: {actual_model}")

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        # Log the LLM response
        llm_logger = _get_llm_logger()
        if llm_logger:
            try:
                prompt = _extract_prompt(payload.get("messages", []))
                llm_logger.log_llm_response(
                    agent_id=agent_id,
                    model_requested=payload["model"],
                    model_actual=actual_model,
                    prompt=prompt,
                    response=content,
                    tokens_in=usage.get("prompt_tokens"),
                    tokens_out=usage.get("completion_tokens"),
                    duration_ms=duration_ms,
                    full_logs=_is_full_logs(),
                    agent_name=agent_name,
                )
            except Exception as e:
                logger.warning(f"Failed to log LLM response: {e}")

        return StreamingResponse(
            content=content,
            model=actual_model,
            finish_reason=data["choices"][0].get("finish_reason"),
            usage=usage,
        )

    async def _stream_complete(self, payload: dict, agent_id: Optional[str] = None, agent_name: Optional[str] = None) -> AsyncIterator[str]:
        """Streaming completion yielding content chunks."""
        client = await self._get_client()

        logger.info(f"OpenRouter API call (streaming) - Requesting model: {payload['model']}")

        async with client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]  # Remove "data: " prefix

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

    async def stream_to_response(
        self,
        messages: list[Message | dict],
        on_chunk: Optional[Callable[[str], None]] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        **kwargs,
    ) -> StreamingResponse:
        """
        Stream completion and collect into a response.

        Args:
            messages: Chat messages
            on_chunk: Optional callback for each chunk
            agent_id: Optional agent ID for logging (overrides client's agent_id)
            agent_name: Optional agent name for logging (e.g., "Elon Musk")
            **kwargs: Passed to complete()

        Returns:
            Complete StreamingResponse with full content
        """
        model = kwargs.get("model", self.default_model)
        # Resolve model aliases for logging
        model_requested = MODELS.get(model, model) if model in MODELS else model
        effective_agent_id = agent_id or self.agent_id

        response = StreamingResponse(model=model_requested)

        start_time = time.time()
        async for chunk in await self.complete(messages, stream=True, agent_id=effective_agent_id, agent_name=agent_name, **kwargs):
            response.content += chunk
            response.raw_chunks.append(chunk)
            if on_chunk:
                on_chunk(chunk)
        duration_ms = int((time.time() - start_time) * 1000)

        response.finish_reason = "stop"

        # Log the streamed LLM response
        llm_logger = _get_llm_logger()
        if llm_logger:
            try:
                msgs = [m.to_dict() if isinstance(m, Message) else m for m in messages]
                prompt = _extract_prompt(msgs)
                llm_logger.log_llm_response(
                    agent_id=effective_agent_id,
                    model_requested=model_requested,
                    model_actual=response.model,
                    prompt=prompt,
                    response=response.content,
                    tokens_in=None,  # Not available for streaming
                    tokens_out=None,
                    duration_ms=duration_ms,
                    full_logs=_is_full_logs(),
                    agent_name=agent_name,
                )
            except Exception as e:
                logger.warning(f"Failed to log streamed LLM response: {e}")

        return response

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        **kwargs,
    ) -> StreamingResponse:
        """
        Simple generation from a prompt.

        Args:
            prompt: User prompt
            system: Optional system prompt
            agent_id: Optional agent ID for logging (overrides client's agent_id)
            agent_name: Optional agent name for logging (e.g., "Elon Musk")
            **kwargs: Passed to complete()

        Returns:
            StreamingResponse
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        return await self.complete(messages, agent_id=agent_id, agent_name=agent_name, **kwargs)

    async def stream_generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Stream generation from a prompt.

        Args:
            prompt: User prompt
            system: Optional system prompt
            agent_id: Optional agent ID for logging (overrides client's agent_id)
            agent_name: Optional agent name for logging (e.g., "Elon Musk")
            **kwargs: Passed to complete()

        Yields:
            Content chunks
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async for chunk in await self.complete(messages, stream=True, agent_id=agent_id, agent_name=agent_name, **kwargs):
            yield chunk

    async def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Non-streaming chat completion (convenience method).

        Args:
            messages: Chat messages
            model: Model to use (defaults to default_model)
            agent_id: Optional agent ID for logging (overrides client's agent_id)
            agent_name: Optional agent name for logging (e.g., "Elon Musk")
            **kwargs: Passed to complete()

        Returns:
            Response content as string
        """
        response = await self.complete(messages, model=model, stream=False, agent_id=agent_id, agent_name=agent_name, **kwargs)
        return response.content


# Global client instance
_client: Optional[OpenRouterClient] = None


def get_client() -> OpenRouterClient:
    """Get or create the global OpenRouter client."""
    global _client
    if _client is None:
        _client = OpenRouterClient()
    return _client


async def stream_chat(
    messages: list[dict],
    model: str = "anthropic/claude-sonnet-4.5",
    agent_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    **kwargs,
) -> AsyncIterator[str]:
    """Convenience function for streaming chat."""
    client = get_client()
    async for chunk in await client.complete(messages, model=model, stream=True, agent_id=agent_id, agent_name=agent_name, **kwargs):
        yield chunk


async def chat(
    messages: list[dict],
    model: str = "anthropic/claude-sonnet-4.5",
    agent_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    **kwargs,
) -> str:
    """Convenience function for non-streaming chat."""
    client = get_client()
    response = await client.complete(messages, model=model, stream=False, agent_id=agent_id, agent_name=agent_name, **kwargs)
    return response.content
