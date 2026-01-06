"""OpenRouter client with streaming support."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional, Any, Callable

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Popular models on OpenRouter (Jan 2026)
MODELS = {
    # Anthropic
    "claude-opus-4.5": "anthropic/claude-opus-4.5",
    "claude-sonnet-4.5": "anthropic/claude-sonnet-4.5",
    "claude-3.5-haiku": "anthropic/claude-3.5-haiku",
    # OpenAI
    "gpt-5.2-pro": "openai/gpt-5.2-pro",
    "gpt-5.1": "openai/gpt-5.1",
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "o3": "openai/o3",
    "o1": "openai/o1",
    # Google
    "gemini-3-pro": "google/gemini-3-pro-preview",
    "gemini-2.5-pro": "google/gemini-2.5-pro",
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "gemini-2.0-flash": "google/gemini-2.0-flash-001",
    # Meta Llama
    "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct",
    "llama-3.1-405b": "meta-llama/llama-3.1-405b-instruct",
    # DeepSeek
    "deepseek-v3.2": "deepseek/deepseek-v3.2",
    "deepseek-r1": "deepseek/deepseek-r1",
    "deepseek-chat": "deepseek/deepseek-chat-v3",
    # Qwen
    "qwen3-235b": "qwen/qwen3-235b-a22b",
    "qwen3-32b": "qwen/qwen3-32b",
    "qwen-max": "qwen/qwen-max",
    # Mistral
    "mistral-large": "mistralai/mistral-large-2411",
    "mistral-nemo": "mistralai/mistral-nemo",
    "mixtral-8x22b": "mistralai/mixtral-8x22b-instruct",
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
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            logger.warning("No OpenRouter API key provided. Set OPENROUTER_API_KEY env var.")

        self.default_model = default_model
        self.base_url = base_url
        self.timeout = timeout
        self.site_url = site_url or "https://github.com/distributed-systems-co/lida-multiagents-research"
        self.site_name = site_name or "LIDA Multi-Agent System"

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

        if stream:
            return self._stream_complete(payload)
        else:
            return await self._complete(payload)

    async def _complete(self, payload: dict) -> StreamingResponse:
        """Non-streaming completion."""
        client = await self._get_client()

        response = await client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
        )
        response.raise_for_status()

        data = response.json()

        return StreamingResponse(
            content=data["choices"][0]["message"]["content"],
            model=data.get("model", payload["model"]),
            finish_reason=data["choices"][0].get("finish_reason"),
            usage=data.get("usage", {}),
        )

    async def _stream_complete(self, payload: dict) -> AsyncIterator[str]:
        """Streaming completion yielding content chunks."""
        client = await self._get_client()

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
        **kwargs,
    ) -> StreamingResponse:
        """
        Stream completion and collect into a response.

        Args:
            messages: Chat messages
            on_chunk: Optional callback for each chunk
            **kwargs: Passed to complete()

        Returns:
            Complete StreamingResponse with full content
        """
        response = StreamingResponse(model=kwargs.get("model", self.default_model))

        async for chunk in await self.complete(messages, stream=True, **kwargs):
            response.content += chunk
            response.raw_chunks.append(chunk)
            if on_chunk:
                on_chunk(chunk)

        response.finish_reason = "stop"
        return response

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> StreamingResponse:
        """
        Simple generation from a prompt.

        Args:
            prompt: User prompt
            system: Optional system prompt
            **kwargs: Passed to complete()

        Returns:
            StreamingResponse
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        return await self.complete(messages, **kwargs)

    async def stream_generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Stream generation from a prompt.

        Args:
            prompt: User prompt
            system: Optional system prompt
            **kwargs: Passed to complete()

        Yields:
            Content chunks
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async for chunk in await self.complete(messages, stream=True, **kwargs):
            yield chunk


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
    **kwargs,
) -> AsyncIterator[str]:
    """Convenience function for streaming chat."""
    client = get_client()
    async for chunk in await client.complete(messages, model=model, stream=True, **kwargs):
        yield chunk


async def chat(
    messages: list[dict],
    model: str = "anthropic/claude-sonnet-4.5",
    **kwargs,
) -> str:
    """Convenience function for non-streaming chat."""
    client = get_client()
    response = await client.complete(messages, model=model, stream=False, **kwargs)
    return response.content
