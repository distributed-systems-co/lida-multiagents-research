"""
Redis Caching Layer for Research Results

Caches research results with TTL to avoid redundant expensive lookups.
Supports both full context caching and incremental source caching.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import json
import hashlib
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class CachedResearch:
    """A cached research result."""
    persona_id: str
    topic: str
    context: Dict[str, Any]
    sources: List[Dict[str, Any]]
    cached_at: str  # ISO format
    expires_at: str  # ISO format
    query_count: int
    source_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CachedResearch":
        return cls(**data)

    @property
    def is_expired(self) -> bool:
        expires = datetime.fromisoformat(self.expires_at)
        return datetime.now() > expires

    @property
    def age_seconds(self) -> float:
        cached = datetime.fromisoformat(self.cached_at)
        return (datetime.now() - cached).total_seconds()


class ResearchCache:
    """
    Redis-backed cache for research results.

    Cache key format: research:{persona_id}:{topic_hash}
    TTL: Configurable, default 24 hours

    Also supports:
    - Source-level caching (individual URLs)
    - Query result caching (search results)
    - Partial result storage (for resume capability)
    """

    def __init__(
        self,
        redis_client=None,
        default_ttl_hours: int = 24,
        source_ttl_hours: int = 48,
    ):
        self.redis = redis_client
        self.default_ttl = timedelta(hours=default_ttl_hours)
        self.source_ttl = timedelta(hours=source_ttl_hours)
        self._local_cache: Dict[str, CachedResearch] = {}  # Fallback if no Redis

    def _make_key(self, persona_id: str, topic: str) -> str:
        """Generate cache key from persona and topic."""
        topic_hash = hashlib.sha256(topic.lower().encode()).hexdigest()[:12]
        return f"research:{persona_id}:{topic_hash}"

    def _make_source_key(self, url: str) -> str:
        """Generate cache key for a source URL."""
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        return f"source:{url_hash}"

    def _make_query_key(self, query: str) -> str:
        """Generate cache key for a search query."""
        query_hash = hashlib.sha256(query.lower().encode()).hexdigest()[:16]
        return f"query:{query_hash}"

    async def get(
        self,
        persona_id: str,
        topic: str,
        max_age_hours: Optional[int] = None,
    ) -> Optional[CachedResearch]:
        """
        Get cached research for a persona and topic.

        Args:
            persona_id: Persona identifier
            topic: Research topic
            max_age_hours: Optional max age override (None = use default TTL)

        Returns:
            CachedResearch if found and not expired, None otherwise
        """
        key = self._make_key(persona_id, topic)

        try:
            if self.redis:
                data = await self.redis.get(key)
                if data:
                    cached = CachedResearch.from_dict(json.loads(data))

                    # Check custom max age
                    if max_age_hours:
                        if cached.age_seconds > (max_age_hours * 3600):
                            return None

                    if not cached.is_expired:
                        logger.info(f"Cache hit for {persona_id}:{topic[:30]}")
                        return cached
            else:
                # Local cache fallback
                if key in self._local_cache:
                    cached = self._local_cache[key]
                    if not cached.is_expired:
                        return cached
                    else:
                        del self._local_cache[key]

        except Exception as e:
            logger.warning(f"Cache get error: {e}")

        return None

    async def set(
        self,
        persona_id: str,
        topic: str,
        context: Dict[str, Any],
        sources: List[Dict[str, Any]],
        query_count: int = 0,
        source_count: int = 0,
        ttl_hours: Optional[int] = None,
    ) -> bool:
        """
        Cache research results.

        Args:
            persona_id: Persona identifier
            topic: Research topic
            context: Synthesized context dictionary
            sources: List of source dictionaries
            query_count: Number of queries executed
            source_count: Number of sources fetched
            ttl_hours: Custom TTL in hours

        Returns:
            True if cached successfully
        """
        key = self._make_key(persona_id, topic)
        ttl = timedelta(hours=ttl_hours) if ttl_hours else self.default_ttl

        now = datetime.now()
        cached = CachedResearch(
            persona_id=persona_id,
            topic=topic,
            context=context,
            sources=sources,
            cached_at=now.isoformat(),
            expires_at=(now + ttl).isoformat(),
            query_count=query_count,
            source_count=source_count,
        )

        try:
            if self.redis:
                await self.redis.setex(
                    key,
                    int(ttl.total_seconds()),
                    json.dumps(cached.to_dict())
                )
            else:
                self._local_cache[key] = cached

            logger.info(f"Cached research for {persona_id}:{topic[:30]}")
            return True

        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False

    async def get_source(self, url: str) -> Optional[Dict[str, Any]]:
        """Get cached source content by URL."""
        key = self._make_source_key(url)

        try:
            if self.redis:
                data = await self.redis.get(key)
                if data:
                    return json.loads(data)
        except Exception as e:
            logger.debug(f"Source cache get error: {e}")

        return None

    async def set_source(
        self,
        url: str,
        content: Dict[str, Any],
        ttl_hours: Optional[int] = None,
    ) -> bool:
        """Cache source content by URL."""
        key = self._make_source_key(url)
        ttl = timedelta(hours=ttl_hours) if ttl_hours else self.source_ttl

        try:
            if self.redis:
                await self.redis.setex(
                    key,
                    int(ttl.total_seconds()),
                    json.dumps(content)
                )
                return True
        except Exception as e:
            logger.debug(f"Source cache set error: {e}")

        return False

    async def get_query_results(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached search results for a query."""
        key = self._make_query_key(query)

        try:
            if self.redis:
                data = await self.redis.get(key)
                if data:
                    return json.loads(data)
        except Exception as e:
            logger.debug(f"Query cache get error: {e}")

        return None

    async def set_query_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        ttl_hours: int = 6,  # Query results expire faster
    ) -> bool:
        """Cache search results for a query."""
        key = self._make_query_key(query)

        try:
            if self.redis:
                await self.redis.setex(
                    key,
                    int(ttl_hours * 3600),
                    json.dumps(results)
                )
                return True
        except Exception as e:
            logger.debug(f"Query cache set error: {e}")

        return False

    async def invalidate(self, persona_id: str, topic: str) -> bool:
        """Invalidate cached research for a persona and topic."""
        key = self._make_key(persona_id, topic)

        try:
            if self.redis:
                await self.redis.delete(key)
            elif key in self._local_cache:
                del self._local_cache[key]
            return True
        except Exception as e:
            logger.warning(f"Cache invalidate error: {e}")
            return False

    async def invalidate_persona(self, persona_id: str) -> int:
        """Invalidate all cached research for a persona."""
        pattern = f"research:{persona_id}:*"
        count = 0

        try:
            if self.redis:
                async for key in self.redis.scan_iter(match=pattern):
                    await self.redis.delete(key)
                    count += 1
            else:
                keys_to_delete = [
                    k for k in self._local_cache
                    if k.startswith(f"research:{persona_id}:")
                ]
                for k in keys_to_delete:
                    del self._local_cache[k]
                    count += 1

            logger.info(f"Invalidated {count} cache entries for {persona_id}")
        except Exception as e:
            logger.warning(f"Cache invalidate persona error: {e}")

        return count

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "backend": "redis" if self.redis else "local",
            "default_ttl_hours": self.default_ttl.total_seconds() / 3600,
            "source_ttl_hours": self.source_ttl.total_seconds() / 3600,
        }

        try:
            if self.redis:
                info = await self.redis.info("memory")
                stats["memory_used"] = info.get("used_memory_human", "unknown")

                # Count research entries
                research_count = 0
                async for _ in self.redis.scan_iter(match="research:*"):
                    research_count += 1
                stats["cached_research_count"] = research_count

                # Count source entries
                source_count = 0
                async for _ in self.redis.scan_iter(match="source:*"):
                    source_count += 1
                stats["cached_source_count"] = source_count

            else:
                stats["local_cache_size"] = len(self._local_cache)

        except Exception as e:
            logger.debug(f"Stats error: {e}")

        return stats
