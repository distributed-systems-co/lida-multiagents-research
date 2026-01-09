"""DSPy model wrappers, identity wrappers, and parallel context search.

This module provides:
- DSPy-compatible LM wrappers for various backends
- Identity wrappers that maintain persona context across calls
- Parallel web search for establishing baseline contexts
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable, Union, AsyncIterator, Tuple
from functools import wraps

import httpx

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DSPY MODEL WRAPPERS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TokenLogprob:
    """Log probability for a token."""
    token: str
    logprob: float
    prob: float  # exp(logprob)
    top_alternatives: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class LMResponse:
    """Standardized LLM response."""
    text: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Logprobs data
    logprobs: Optional[List[TokenLogprob]] = None
    mean_logprob: Optional[float] = None
    confidence: Optional[float] = None  # Derived from logprobs

    def get_confidence(self) -> float:
        """Get confidence score derived from logprobs (0-1)."""
        if self.confidence is not None:
            return self.confidence
        if self.mean_logprob is not None:
            # Convert mean logprob to confidence
            # logprob of -0.1 ≈ 90% confidence, -1 ≈ 37%, -2 ≈ 14%
            return min(1.0, max(0.0, math.exp(self.mean_logprob)))
        return 0.5  # Default uncertainty

    def get_token_confidences(self) -> List[float]:
        """Get per-token confidence scores."""
        if not self.logprobs:
            return []
        return [lp.prob for lp in self.logprobs]


class BaseLM(ABC):
    """Base class for DSPy-compatible language models."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        self.history: List[Dict] = []

    @abstractmethod
    async def __call__(
        self,
        prompt: str,
        **kwargs
    ) -> LMResponse:
        """Generate a response for the given prompt."""
        pass

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LMResponse:
        """Generate from message list."""
        pass

    def inspect_history(self, n: int = 5) -> List[Dict]:
        """Return last n interactions."""
        return self.history[-n:]

    def clear_history(self):
        """Clear interaction history."""
        self.history = []


class OpenRouterLM(BaseLM):
    """OpenRouter-backed language model with logprobs support."""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    # Models known to support logprobs on OpenRouter (Jan 2026)
    LOGPROB_MODELS = {
        # OpenAI GPT-4.1 series (1M+ context)
        "openai/gpt-4.1",
        "openai/gpt-4.1-mini",
        "openai/gpt-4.1-nano",
        # OpenAI GPT-5 series
        "openai/gpt-5",
        "openai/gpt-5-mini",
        "openai/gpt-5-nano",
        "openai/gpt-5.1",
        "openai/gpt-5.1-codex",
        "openai/gpt-5.2",
        # OpenAI GPT-4o
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        # xAI Grok (2M context)
        "x-ai/grok-4.1-fast",
        "x-ai/grok-4-fast",
        # Meta Llama 4
        "meta-llama/llama-4-maverick",
        # Older models still supported
        "meta-llama/llama-3.3-70b-instruct",
        "qwen/qwen-2.5-72b-instruct",
        "deepseek/deepseek-chat",
        "deepseek/deepseek-r1",
    }

    def __init__(
        self,
        model: str = "openai/gpt-4.1-mini",  # Jan 2026 default - 1M context, fast
        api_key: Optional[str] = None,
        enable_logprobs: bool = True,
        top_logprobs: int = 5,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY required")

        self.enable_logprobs = enable_logprobs
        self.top_logprobs = min(20, max(0, top_logprobs))

        self._client = httpx.AsyncClient(
            timeout=kwargs.get("timeout", 120.0),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": kwargs.get("referer", "https://github.com/lida-multiagents"),
                "X-Title": kwargs.get("title", "LIDA Multi-Agent System"),
            }
        )

    # Models verified to actually return logprobs (Jan 2026)
    VERIFIED_LOGPROB_MODELS = {
        "openai/gpt-4o", "openai/gpt-4o-mini", "openai/chatgpt-4o-latest",
        "openai/gpt-4o-2024-08-06", "openai/gpt-4o-2024-11-20",
        "deepseek/deepseek-v3.2",
        "qwen/qwen3-32b",
        "meta-llama/llama-3.3-70b-instruct", "meta-llama/llama-3.1-8b-instruct",
    }

    def supports_logprobs(self, model: Optional[str] = None) -> bool:
        """Check if model actually returns logprobs (verified)."""
        m = model or self.model
        # Check exact match first
        if m in self.VERIFIED_LOGPROB_MODELS:
            return True
        # Check patterns for model families that work
        return any(pattern in m for pattern in [
            "gpt-4o", "llama-3.1", "llama-3.3",
        ])

    async def __call__(self, prompt: str, **kwargs) -> LMResponse:
        messages = [{"role": "user", "content": prompt}]
        return await self.generate(messages, **kwargs)

    async def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LMResponse:
        start = time.time()

        model = kwargs.get("model", self.model)

        # Ensure min tokens (some providers like Azure require >= 16)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens < 16:
            max_tokens = 16

        payload = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": max_tokens,
        }

        # Add logprobs if enabled and model supports it
        request_logprobs = kwargs.get("logprobs", self.enable_logprobs)
        if request_logprobs and self.supports_logprobs(model):
            payload["logprobs"] = True
            payload["top_logprobs"] = kwargs.get("top_logprobs", self.top_logprobs)

        try:
            response = await self._client.post(self.BASE_URL, json=payload)
            response.raise_for_status()
            data = response.json()

            choice = data["choices"][0]
            text = choice["message"]["content"]
            usage = data.get("usage", {})

            # Parse logprobs if present
            token_logprobs = None
            mean_logprob = None
            confidence = None

            if "logprobs" in choice and choice["logprobs"]:
                token_logprobs, mean_logprob = self._parse_logprobs(choice["logprobs"])
                if mean_logprob is not None:
                    confidence = min(1.0, max(0.0, math.exp(mean_logprob)))

            result = LMResponse(
                text=text,
                model=data.get("model", self.model),
                usage=usage,
                latency_ms=(time.time() - start) * 1000,
                metadata={"id": data.get("id")},
                logprobs=token_logprobs,
                mean_logprob=mean_logprob,
                confidence=confidence,
            )

            # Track history
            self.history.append({
                "messages": messages,
                "response": text,
                "model": result.model,
                "confidence": confidence,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return result

        except Exception as e:
            logger.error(f"OpenRouter request failed: {e}")
            raise

    def _parse_logprobs(
        self,
        logprobs_data: Dict
    ) -> Tuple[List[TokenLogprob], Optional[float]]:
        """Parse logprobs from OpenRouter response."""
        token_logprobs = []
        total_logprob = 0.0
        count = 0

        content = logprobs_data.get("content", [])
        for item in content:
            token = item.get("token", "")
            logprob = item.get("logprob", 0.0)
            prob = math.exp(logprob) if logprob > -100 else 0.0

            # Get top alternatives
            alternatives = []
            for alt in item.get("top_logprobs", [])[:5]:
                alt_token = alt.get("token", "")
                alt_logprob = alt.get("logprob", 0.0)
                alternatives.append((alt_token, alt_logprob))

            token_logprobs.append(TokenLogprob(
                token=token,
                logprob=logprob,
                prob=prob,
                top_alternatives=alternatives
            ))

            total_logprob += logprob
            count += 1

        mean_logprob = total_logprob / count if count > 0 else None
        return token_logprobs, mean_logprob


class AnthropicLM(BaseLM):
    """Direct Anthropic API language model."""

    BASE_URL = "https://api.anthropic.com/v1/messages"

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY required")

        self._client = httpx.AsyncClient(
            timeout=kwargs.get("timeout", 120.0),
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
        )

    async def __call__(self, prompt: str, **kwargs) -> LMResponse:
        messages = [{"role": "user", "content": prompt}]
        return await self.generate(messages, **kwargs)

    async def generate(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        **kwargs
    ) -> LMResponse:
        start = time.time()

        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }

        if system:
            payload["system"] = system

        try:
            response = await self._client.post(self.BASE_URL, json=payload)
            response.raise_for_status()
            data = response.json()

            text = data["content"][0]["text"]
            usage = data.get("usage", {})

            result = LMResponse(
                text=text,
                model=data.get("model", self.model),
                usage={
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                },
                latency_ms=(time.time() - start) * 1000,
                metadata={"id": data.get("id")}
            )

            self.history.append({
                "messages": messages,
                "system": system,
                "response": text,
                "model": result.model,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return result

        except Exception as e:
            logger.error(f"Anthropic request failed: {e}")
            raise


class OllamaLM(BaseLM):
    """Ollama local model."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        **kwargs
    ):
        super().__init__(model, **kwargs)
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=kwargs.get("timeout", 300.0))

    async def __call__(self, prompt: str, **kwargs) -> LMResponse:
        messages = [{"role": "user", "content": prompt}]
        return await self.generate(messages, **kwargs)

    async def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LMResponse:
        start = time.time()

        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            }
        }

        try:
            response = await self._client.post(
                f"{self.base_url}/api/chat",
                json=payload
            )
            response.raise_for_status()
            data = response.json()

            text = data["message"]["content"]

            result = LMResponse(
                text=text,
                model=data.get("model", self.model),
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                },
                latency_ms=(time.time() - start) * 1000,
            )

            self.history.append({
                "messages": messages,
                "response": text,
                "model": result.model,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return result

        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
            raise


# ═══════════════════════════════════════════════════════════════════════════════
# IDENTITY WRAPPERS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Identity:
    """Represents an agent identity/persona."""
    name: str
    system_prompt: str
    traits: Dict[str, Any] = field(default_factory=dict)
    knowledge_base: List[str] = field(default_factory=list)
    communication_style: str = "professional"
    expertise_areas: List[str] = field(default_factory=list)

    # Context memory
    conversation_summary: str = ""
    key_facts: List[str] = field(default_factory=list)
    interaction_count: int = 0
    last_interaction: Optional[datetime] = None

    def to_system_message(self) -> str:
        """Generate full system message for this identity."""
        parts = [self.system_prompt]

        if self.expertise_areas:
            parts.append(f"\nAreas of expertise: {', '.join(self.expertise_areas)}")

        if self.communication_style:
            parts.append(f"\nCommunication style: {self.communication_style}")

        if self.knowledge_base:
            parts.append("\n\nRelevant knowledge:")
            for kb in self.knowledge_base[:5]:  # Limit context
                parts.append(f"- {kb[:500]}")

        if self.key_facts:
            parts.append("\n\nKey facts from previous interactions:")
            for fact in self.key_facts[-10:]:  # Last 10 facts
                parts.append(f"- {fact}")

        return "\n".join(parts)

    def update_from_interaction(self, user_input: str, response: str):
        """Update identity state after interaction."""
        self.interaction_count += 1
        self.last_interaction = datetime.now(timezone.utc)


class IdentityWrapper:
    """Wraps an LM with persistent identity context."""

    def __init__(
        self,
        lm: BaseLM,
        identity: Identity,
        max_context_tokens: int = 8000,
    ):
        self.lm = lm
        self.identity = identity
        self.max_context_tokens = max_context_tokens
        self.conversation_history: List[Dict[str, str]] = []

    async def __call__(
        self,
        prompt: str,
        maintain_history: bool = True,
        **kwargs
    ) -> LMResponse:
        """Generate response while maintaining identity."""

        # Build messages with identity context
        messages = []

        # System message with identity
        system_msg = self.identity.to_system_message()

        # Add conversation history if maintaining
        if maintain_history and self.conversation_history:
            messages.extend(self.conversation_history[-10:])  # Last 10 exchanges

        # Add current prompt
        messages.append({"role": "user", "content": prompt})

        # Generate with system prompt
        if isinstance(self.lm, AnthropicLM):
            response = await self.lm.generate(messages, system=system_msg, **kwargs)
        else:
            # Prepend system as first message for other providers
            full_messages = [{"role": "system", "content": system_msg}] + messages
            response = await self.lm.generate(full_messages, **kwargs)

        # Update history
        if maintain_history:
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": response.text})

        # Update identity
        self.identity.update_from_interaction(prompt, response.text)

        return response

    def add_to_knowledge_base(self, knowledge: str):
        """Add knowledge to identity's context."""
        self.identity.knowledge_base.append(knowledge)

    def add_key_fact(self, fact: str):
        """Record a key fact from interactions."""
        self.identity.key_facts.append(fact)

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

    def get_identity_summary(self) -> Dict[str, Any]:
        """Get summary of identity state."""
        return {
            "name": self.identity.name,
            "interaction_count": self.identity.interaction_count,
            "last_interaction": self.identity.last_interaction.isoformat() if self.identity.last_interaction else None,
            "knowledge_items": len(self.identity.knowledge_base),
            "key_facts": len(self.identity.key_facts),
            "history_length": len(self.conversation_history),
        }


def create_identity_from_prompt(
    prompt_text: str,
    name: Optional[str] = None,
) -> Identity:
    """Create an Identity from a system prompt."""
    # Extract name from "You are a/an X" pattern
    if not name:
        import re
        match = re.search(r"You are (?:a|an) ([^.]+)", prompt_text)
        name = match.group(1)[:50] if match else "Agent"

    return Identity(
        name=name,
        system_prompt=prompt_text,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PARALLEL WEB SEARCH FOR BASELINE CONTEXTS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SearchResult:
    """A web search result."""
    title: str
    url: str
    snippet: str
    source: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class BaselineContext:
    """Aggregated context from parallel searches."""
    query: str
    results: List[SearchResult] = field(default_factory=list)
    summary: str = ""
    key_facts: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    search_time_ms: float = 0.0

    def to_context_string(self, max_length: int = 4000) -> str:
        """Convert to a context string for LLM consumption."""
        parts = [f"# Baseline Context for: {self.query}\n"]

        if self.summary:
            parts.append(f"## Summary\n{self.summary}\n")

        if self.key_facts:
            parts.append("## Key Facts")
            for fact in self.key_facts:
                parts.append(f"- {fact}")
            parts.append("")

        if self.results:
            parts.append("## Sources")
            for r in self.results[:5]:
                parts.append(f"- [{r.title}]({r.url}): {r.snippet[:200]}...")

        text = "\n".join(parts)
        return text[:max_length]


class WebSearchProvider(ABC):
    """Base class for web search providers."""

    @abstractmethod
    async def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        pass


class BraveSearchProvider(WebSearchProvider):
    """Brave Search API provider."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("BRAVE_API_KEY")
        self._client = httpx.AsyncClient(timeout=30.0)

    async def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        if not self.api_key:
            return []

        try:
            response = await self._client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": num_results},
                headers={"X-Subscription-Token": self.api_key}
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("web", {}).get("results", [])[:num_results]:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("description", ""),
                    source="brave"
                ))
            return results

        except Exception as e:
            logger.warning(f"Brave search failed: {e}")
            return []


class SerpAPIProvider(WebSearchProvider):
    """SerpAPI provider (Google results)."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SERPAPI_KEY")
        self._client = httpx.AsyncClient(timeout=30.0)

    async def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        if not self.api_key:
            return []

        try:
            response = await self._client.get(
                "https://serpapi.com/search",
                params={
                    "q": query,
                    "api_key": self.api_key,
                    "num": num_results,
                    "engine": "google"
                }
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("organic_results", [])[:num_results]:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    source="serpapi"
                ))
            return results

        except Exception as e:
            logger.warning(f"SerpAPI search failed: {e}")
            return []


class JinaReaderProvider(WebSearchProvider):
    """Jina Reader API for content extraction."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        self._client = httpx.AsyncClient(timeout=60.0)

    async def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Use Jina's search endpoint."""
        headers = {
            "Accept": "application/json",
            "X-Return-Format": "markdown",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            # Use the search endpoint
            import urllib.parse
            encoded_query = urllib.parse.quote(query)
            response = await self._client.get(
                f"https://s.jina.ai/{encoded_query}",
                headers=headers
            )
            response.raise_for_status()

            content = response.text

            # Parse the markdown response into results
            results = []

            # Split by result blocks (Jina returns markdown with ### headers)
            blocks = content.split("\n### ")
            for block in blocks[1:num_results+1]:  # Skip first empty block
                lines = block.strip().split("\n")
                if lines:
                    title = lines[0].strip()
                    # Find URL if present
                    url = ""
                    snippet_lines = []
                    for line in lines[1:]:
                        if line.startswith("http"):
                            url = line.strip()
                        elif line.startswith("URL:"):
                            url = line.replace("URL:", "").strip()
                        elif line.strip():
                            snippet_lines.append(line.strip())

                    snippet = " ".join(snippet_lines)[:300]

                    results.append(SearchResult(
                        title=title[:100],
                        url=url or f"https://s.jina.ai/{encoded_query}",
                        snippet=snippet,
                        source="jina"
                    ))

            # If parsing failed, return raw content as single result
            if not results and content:
                results.append(SearchResult(
                    title=f"Search: {query}",
                    url=f"https://s.jina.ai/{encoded_query}",
                    snippet=content[:500],
                    source="jina"
                ))

            return results

        except Exception as e:
            logger.warning(f"Jina search failed: {e}")
            return []

    async def read_url(self, url: str) -> str:
        """Extract content from a URL."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = await self._client.get(
                f"https://r.jina.ai/{url}",
                headers=headers
            )
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.warning(f"Jina read failed: {e}")
            return ""


class DuckDuckGoProvider(WebSearchProvider):
    """DuckDuckGo HTML search (no API key required)."""

    def __init__(self):
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            }
        )

    async def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        try:
            import urllib.parse
            encoded = urllib.parse.quote(query)

            # Use DuckDuckGo HTML version
            response = await self._client.get(
                f"https://html.duckduckgo.com/html/?q={encoded}"
            )
            response.raise_for_status()

            # Parse results from HTML
            import re
            results = []

            # Find result blocks
            html = response.text
            result_pattern = re.compile(
                r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>.*?'
                r'<a[^>]*class="result__snippet"[^>]*>([^<]+)</a>',
                re.DOTALL
            )

            # Simpler pattern fallback
            link_pattern = re.compile(r'<a[^>]*href="([^"]+)"[^>]*class="result__a"[^>]*>([^<]+)</a>')
            snippet_pattern = re.compile(r'class="result__snippet">([^<]+)</a>')

            links = link_pattern.findall(html)
            snippets = snippet_pattern.findall(html)

            for i, (url, title) in enumerate(links[:num_results]):
                snippet = snippets[i] if i < len(snippets) else ""
                results.append(SearchResult(
                    title=title.strip()[:100],
                    url=url,
                    snippet=snippet.strip()[:300],
                    source="duckduckgo"
                ))

            return results

        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
            return []


class ParallelContextSearch:
    """Execute parallel web searches to establish baseline context."""

    def __init__(
        self,
        providers: Optional[List[WebSearchProvider]] = None,
        summarizer_lm: Optional[BaseLM] = None,
    ):
        # Default providers
        if providers is None:
            providers = []
            if os.getenv("BRAVE_API_KEY"):
                providers.append(BraveSearchProvider())
            if os.getenv("SERPAPI_KEY"):
                providers.append(SerpAPIProvider())
            # Always add DuckDuckGo as fallback (no key needed)
            providers.append(DuckDuckGoProvider())

        self.providers = providers
        self.summarizer_lm = summarizer_lm
        self._cache: Dict[str, BaselineContext] = {}

    async def search(
        self,
        queries: List[str],
        num_results_per_query: int = 3,
        use_cache: bool = True,
    ) -> List[BaselineContext]:
        """Execute parallel searches for multiple queries."""
        start = time.time()

        # Check cache
        results = []
        queries_to_search = []

        for query in queries:
            cache_key = hashlib.md5(query.encode()).hexdigest()
            if use_cache and cache_key in self._cache:
                results.append(self._cache[cache_key])
            else:
                queries_to_search.append((query, cache_key))
                results.append(None)  # Placeholder

        if not queries_to_search:
            return results

        # Execute parallel searches
        async def search_query(query: str, cache_key: str) -> BaselineContext:
            all_results = []

            # Search all providers in parallel
            provider_tasks = [
                provider.search(query, num_results_per_query)
                for provider in self.providers
            ]

            provider_results = await asyncio.gather(*provider_tasks, return_exceptions=True)

            for pr in provider_results:
                if isinstance(pr, list):
                    all_results.extend(pr)

            # Deduplicate by URL
            seen_urls = set()
            unique_results = []
            for r in all_results:
                if r.url not in seen_urls:
                    seen_urls.add(r.url)
                    unique_results.append(r)

            context = BaselineContext(
                query=query,
                results=unique_results,
                sources=[r.url for r in unique_results],
                search_time_ms=(time.time() - start) * 1000,
            )

            # Cache it
            self._cache[cache_key] = context
            return context

        # Execute all searches in parallel
        search_tasks = [
            search_query(q, ck) for q, ck in queries_to_search
        ]
        new_results = await asyncio.gather(*search_tasks)

        # Merge results
        new_idx = 0
        for i, r in enumerate(results):
            if r is None:
                results[i] = new_results[new_idx]
                new_idx += 1

        return results

    async def search_and_summarize(
        self,
        queries: List[str],
        num_results_per_query: int = 3,
    ) -> BaselineContext:
        """Search and create a unified summary."""
        contexts = await self.search(queries, num_results_per_query)

        # Merge all contexts
        all_results = []
        all_sources = set()

        for ctx in contexts:
            all_results.extend(ctx.results)
            all_sources.update(ctx.sources)

        merged = BaselineContext(
            query=" | ".join(queries),
            results=all_results,
            sources=list(all_sources),
        )

        # Summarize if LM available
        if self.summarizer_lm and all_results:
            snippets = "\n".join([
                f"- {r.title}: {r.snippet}" for r in all_results[:10]
            ])

            prompt = f"""Summarize the key information from these search results about: {merged.query}

Results:
{snippets}

Provide:
1. A brief summary (2-3 sentences)
2. Key facts as a bullet list

Format as:
SUMMARY: <summary>
FACTS:
- <fact 1>
- <fact 2>
..."""

            try:
                response = await self.summarizer_lm(prompt)

                # Parse response
                if "SUMMARY:" in response.text:
                    parts = response.text.split("FACTS:")
                    merged.summary = parts[0].replace("SUMMARY:", "").strip()
                    if len(parts) > 1:
                        facts = [f.strip().lstrip("- ") for f in parts[1].strip().split("\n") if f.strip()]
                        merged.key_facts = facts
                else:
                    merged.summary = response.text[:500]

            except Exception as e:
                logger.warning(f"Summarization failed: {e}")

        return merged


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_lm(
    provider: str = "openrouter",
    model: Optional[str] = None,
    **kwargs
) -> BaseLM:
    """Get an LM instance by provider name."""
    providers = {
        "openrouter": (OpenRouterLM, "anthropic/claude-sonnet-4"),
        "anthropic": (AnthropicLM, "claude-sonnet-4-20250514"),
        "ollama": (OllamaLM, "llama3.2"),
    }

    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(providers.keys())}")

    lm_class, default_model = providers[provider]
    return lm_class(model=model or default_model, **kwargs)


async def establish_context(
    topic: str,
    related_queries: Optional[List[str]] = None,
    summarize: bool = True,
) -> BaselineContext:
    """Quick function to establish baseline context for a topic."""
    searcher = ParallelContextSearch()

    queries = [topic]
    if related_queries:
        queries.extend(related_queries)

    if summarize:
        lm = None
        try:
            lm = get_lm("openrouter")
        except:
            pass
        searcher.summarizer_lm = lm
        return await searcher.search_and_summarize(queries)
    else:
        contexts = await searcher.search(queries)
        # Merge
        return BaselineContext(
            query=topic,
            results=[r for c in contexts for r in c.results],
            sources=[s for c in contexts for s in c.sources],
        )


def wrap_with_identity(
    lm: BaseLM,
    prompt_text: str,
    name: Optional[str] = None,
) -> IdentityWrapper:
    """Wrap an LM with an identity from a prompt."""
    identity = create_identity_from_prompt(prompt_text, name)
    return IdentityWrapper(lm, identity)
