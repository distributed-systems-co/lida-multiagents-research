"""
Source Fetching and Ranking for Persona Research

Fetches web sources, extracts content, and ranks by authority,
recency, and relevance to the persona and topic.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, AsyncIterator
from datetime import datetime
from enum import Enum
import asyncio
import aiohttp
import hashlib
import re
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


class SourceType(Enum):
    """Types of sources with authority rankings."""
    PRIMARY = "primary"          # Person's own blog, official statement
    INTERVIEW = "interview"      # Direct quotes from interviews
    SOCIAL = "social"            # Twitter/X, LinkedIn
    NEWS_MAJOR = "news_major"    # NYT, WSJ, Reuters, etc.
    NEWS_TECH = "news_tech"      # TechCrunch, Wired, Ars Technica
    NEWS_AI = "news_ai"          # AI-specific outlets
    LEGAL = "legal"              # Court filings, SEC documents
    ACADEMIC = "academic"        # arXiv, papers
    OPINION = "opinion"          # Op-eds, analysis pieces
    AGGREGATOR = "aggregator"    # Reddit, HN comments
    UNKNOWN = "unknown"


@dataclass
class Source:
    """A discovered source with metadata."""
    url: str
    title: str
    snippet: str
    source_type: SourceType = SourceType.UNKNOWN
    authority_score: float = 0.5
    recency_score: float = 0.5
    relevance_score: float = 0.5
    published_date: Optional[datetime] = None
    domain: str = ""

    def __post_init__(self):
        if not self.domain:
            self.domain = urlparse(self.url).netloc.lower()

    @property
    def composite_score(self) -> float:
        """Weighted composite score for ranking."""
        return (
            self.authority_score * 0.4 +
            self.recency_score * 0.35 +
            self.relevance_score * 0.25
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "source_type": self.source_type.value,
            "authority_score": self.authority_score,
            "recency_score": self.recency_score,
            "relevance_score": self.relevance_score,
            "composite_score": self.composite_score,
            "domain": self.domain,
            "published_date": self.published_date.isoformat() if self.published_date else None,
        }


@dataclass
class ExtractedContent:
    """Content extracted from a source."""
    source: Source
    full_text: str
    quotes: List[str] = field(default_factory=list)
    key_points: List[str] = field(default_factory=list)
    mentioned_entities: List[str] = field(default_factory=list)
    sentiment: Optional[str] = None  # "positive", "negative", "neutral"
    extraction_time: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "full_text_length": len(self.full_text),
            "quotes": self.quotes,
            "key_points": self.key_points,
            "mentioned_entities": self.mentioned_entities,
            "sentiment": self.sentiment,
        }


class SourceRanker:
    """
    Ranks sources by authority, recency, and relevance.
    """

    # Domain to source type mapping
    DOMAIN_TYPES = {
        # Primary sources
        "openai.com": SourceType.PRIMARY,
        "anthropic.com": SourceType.PRIMARY,
        "deepmind.com": SourceType.PRIMARY,
        "deepmind.google": SourceType.PRIMARY,
        "meta.com": SourceType.PRIMARY,
        "ai.meta.com": SourceType.PRIMARY,
        "blogs.microsoft.com": SourceType.PRIMARY,
        "blog.google": SourceType.PRIMARY,
        "nvidia.com": SourceType.PRIMARY,

        # Social
        "twitter.com": SourceType.SOCIAL,
        "x.com": SourceType.SOCIAL,
        "linkedin.com": SourceType.SOCIAL,
        "threads.net": SourceType.SOCIAL,

        # Major news
        "nytimes.com": SourceType.NEWS_MAJOR,
        "wsj.com": SourceType.NEWS_MAJOR,
        "washingtonpost.com": SourceType.NEWS_MAJOR,
        "reuters.com": SourceType.NEWS_MAJOR,
        "bloomberg.com": SourceType.NEWS_MAJOR,
        "ft.com": SourceType.NEWS_MAJOR,
        "economist.com": SourceType.NEWS_MAJOR,
        "bbc.com": SourceType.NEWS_MAJOR,
        "theguardian.com": SourceType.NEWS_MAJOR,
        "cnn.com": SourceType.NEWS_MAJOR,

        # Tech news
        "techcrunch.com": SourceType.NEWS_TECH,
        "wired.com": SourceType.NEWS_TECH,
        "arstechnica.com": SourceType.NEWS_TECH,
        "theverge.com": SourceType.NEWS_TECH,
        "engadget.com": SourceType.NEWS_TECH,
        "venturebeat.com": SourceType.NEWS_TECH,
        "semafor.com": SourceType.NEWS_TECH,
        "theinformation.com": SourceType.NEWS_TECH,
        "platformer.news": SourceType.NEWS_TECH,

        # AI-specific
        "theaibreak.com": SourceType.NEWS_AI,
        "aiweirdness.com": SourceType.NEWS_AI,
        "lastweekinai.com": SourceType.NEWS_AI,
        "importai.net": SourceType.NEWS_AI,

        # Legal/Government
        "sec.gov": SourceType.LEGAL,
        "courtlistener.com": SourceType.LEGAL,
        "pacer.gov": SourceType.LEGAL,
        "congress.gov": SourceType.LEGAL,
        "regulations.gov": SourceType.LEGAL,
        "ftc.gov": SourceType.LEGAL,
        "whitehouse.gov": SourceType.LEGAL,
        "europa.eu": SourceType.LEGAL,

        # Academic
        "arxiv.org": SourceType.ACADEMIC,
        "openreview.net": SourceType.ACADEMIC,
        "semanticscholar.org": SourceType.ACADEMIC,
        "scholar.google.com": SourceType.ACADEMIC,
        "nature.com": SourceType.ACADEMIC,
        "science.org": SourceType.ACADEMIC,

        # Aggregators
        "reddit.com": SourceType.AGGREGATOR,
        "news.ycombinator.com": SourceType.AGGREGATOR,
    }

    # Authority scores by source type
    AUTHORITY_SCORES = {
        SourceType.PRIMARY: 1.0,
        SourceType.LEGAL: 0.95,
        SourceType.INTERVIEW: 0.9,
        SourceType.ACADEMIC: 0.85,
        SourceType.NEWS_MAJOR: 0.8,
        SourceType.SOCIAL: 0.75,
        SourceType.NEWS_TECH: 0.7,
        SourceType.NEWS_AI: 0.65,
        SourceType.OPINION: 0.5,
        SourceType.AGGREGATOR: 0.3,
        SourceType.UNKNOWN: 0.4,
    }

    def __init__(self, persona_name: str, topic: str):
        self.persona_name = persona_name.lower()
        self.topic = topic.lower()
        self.topic_keywords = set(self.topic.split())

    def classify_source_type(self, url: str, title: str = "") -> SourceType:
        """Determine source type from URL and title."""
        domain = urlparse(url).netloc.lower()
        # Remove www prefix
        if domain.startswith("www."):
            domain = domain[4:]

        # Check direct mappings
        if domain in self.DOMAIN_TYPES:
            return self.DOMAIN_TYPES[domain]

        # Check subdomains
        for known_domain, source_type in self.DOMAIN_TYPES.items():
            if domain.endswith(known_domain):
                return source_type

        # Infer from title
        title_lower = title.lower()
        if "interview" in title_lower or "podcast" in title_lower:
            return SourceType.INTERVIEW
        if "opinion" in title_lower or "editorial" in title_lower:
            return SourceType.OPINION

        return SourceType.UNKNOWN

    def calculate_authority_score(self, source: Source) -> float:
        """Calculate authority score based on source type and domain."""
        base_score = self.AUTHORITY_SCORES.get(source.source_type, 0.4)

        # Boost for verified primary sources
        if source.source_type == SourceType.PRIMARY:
            # Check if it's the actual org's blog
            if any(org in source.domain for org in ["openai", "anthropic", "deepmind"]):
                base_score = 1.0

        # Penalty for aggregators with low engagement
        if source.source_type == SourceType.AGGREGATOR:
            base_score *= 0.8

        return min(base_score, 1.0)

    def calculate_recency_score(
        self,
        published_date: Optional[datetime],
        fallback_from_snippet: bool = True
    ) -> float:
        """Calculate recency score based on publication date."""
        if published_date:
            days_old = (datetime.now() - published_date).days
        elif fallback_from_snippet:
            # Default to 30 days if unknown
            days_old = 30
        else:
            return 0.5

        # Decay function: 1.0 for today, ~0.5 for 30 days, ~0.2 for 90 days
        if days_old <= 0:
            return 1.0
        elif days_old <= 7:
            return 0.95
        elif days_old <= 14:
            return 0.85
        elif days_old <= 30:
            return 0.7
        elif days_old <= 60:
            return 0.5
        elif days_old <= 90:
            return 0.35
        elif days_old <= 180:
            return 0.2
        else:
            return 0.1

    def calculate_relevance_score(
        self,
        title: str,
        snippet: str,
    ) -> float:
        """Calculate relevance to persona and topic."""
        text = f"{title} {snippet}".lower()

        score = 0.0

        # Persona name match (high weight)
        name_parts = self.persona_name.split()
        if all(part in text for part in name_parts):
            score += 0.5
        elif any(part in text for part in name_parts):
            score += 0.25

        # Topic keyword matches
        topic_matches = sum(1 for kw in self.topic_keywords if kw in text)
        if self.topic_keywords:
            score += 0.3 * min(topic_matches / len(self.topic_keywords), 1.0)

        # Quote indicators (direct quotes are valuable)
        if '"' in text or "said" in text or "stated" in text:
            score += 0.1

        # Recent time indicators
        if any(w in text for w in ["today", "yesterday", "this week", "just", "breaking"]):
            score += 0.1

        return min(score, 1.0)

    def rank_sources(self, sources: List[Source]) -> List[Source]:
        """Rank sources by composite score."""
        for source in sources:
            source.source_type = self.classify_source_type(source.url, source.title)
            source.authority_score = self.calculate_authority_score(source)
            source.recency_score = self.calculate_recency_score(source.published_date)
            source.relevance_score = self.calculate_relevance_score(
                source.title, source.snippet
            )

        return sorted(sources, key=lambda s: s.composite_score, reverse=True)


class SourceFetcher:
    """
    Fetches and extracts content from web sources.

    Uses multiple strategies:
    1. Direct HTTP fetch with content extraction
    2. Jina Reader API for JavaScript-heavy pages
    3. Fallback to snippet if fetch fails
    """

    def __init__(
        self,
        jina_api_key: Optional[str] = None,
        timeout: float = 10.0,
        max_content_length: int = 50000,
    ):
        self.jina_api_key = jina_api_key
        self.timeout = timeout
        self.max_content_length = max_content_length
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                                  "Chrome/120.0.0.0 Safari/537.36"
                }
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def fetch_source(self, source: Source) -> ExtractedContent:
        """Fetch and extract content from a source."""
        try:
            # Try Jina Reader first for better extraction
            if self.jina_api_key:
                content = await self._fetch_via_jina(source.url)
            else:
                content = await self._fetch_direct(source.url)

            if content:
                return self._extract_content(source, content)
            else:
                # Fallback to snippet
                return ExtractedContent(
                    source=source,
                    full_text=source.snippet,
                    quotes=[],
                    key_points=[source.snippet] if source.snippet else [],
                )

        except Exception as e:
            logger.warning(f"Failed to fetch {source.url}: {e}")
            return ExtractedContent(
                source=source,
                full_text=source.snippet,
                quotes=[],
                key_points=[],
            )

    async def fetch_sources_parallel(
        self,
        sources: List[Source],
        max_concurrent: int = 5,
    ) -> AsyncIterator[ExtractedContent]:
        """Fetch multiple sources in parallel, yielding as they complete."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_with_semaphore(source: Source) -> ExtractedContent:
            async with semaphore:
                return await self.fetch_source(source)

        tasks = [
            asyncio.create_task(fetch_with_semaphore(source))
            for source in sources
        ]

        for task in asyncio.as_completed(tasks):
            yield await task

    async def _fetch_direct(self, url: str) -> Optional[str]:
        """Fetch URL directly and extract text."""
        try:
            session = await self._get_session()
            async with session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    return self._extract_text_from_html(html)
        except Exception as e:
            logger.debug(f"Direct fetch failed for {url}: {e}")
        return None

    async def _fetch_via_jina(self, url: str) -> Optional[str]:
        """Fetch URL via Jina Reader API for better extraction."""
        try:
            session = await self._get_session()
            jina_url = f"https://r.jina.ai/{url}"
            headers = {"Authorization": f"Bearer {self.jina_api_key}"}

            async with session.get(jina_url, headers=headers) as response:
                if response.status == 200:
                    return await response.text()
        except Exception as e:
            logger.debug(f"Jina fetch failed for {url}: {e}")
        return None

    def _extract_text_from_html(self, html: str) -> str:
        """Extract readable text from HTML."""
        # Simple extraction - remove tags
        # In production, use BeautifulSoup or similar
        import re

        # Remove script and style elements
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)

        # Remove tags
        text = re.sub(r'<[^>]+>', ' ', html)

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)

        # Decode entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')

        return text[:self.max_content_length]

    def _extract_content(self, source: Source, raw_content: str) -> ExtractedContent:
        """Extract structured content from raw text."""
        # Extract quotes (text in quotation marks)
        quotes = re.findall(r'"([^"]{20,500})"', raw_content)

        # Extract sentences that might be key points
        sentences = re.split(r'(?<=[.!?])\s+', raw_content)
        key_points = [
            s.strip() for s in sentences
            if len(s) > 50 and len(s) < 300
            and any(word in s.lower() for word in ['said', 'stated', 'announced', 'believes', 'argues'])
        ][:5]

        # Extract mentioned entities (simple name detection)
        mentioned = list(set(re.findall(
            r'(?:CEO|founder|researcher|professor|Dr\.|Mr\.|Ms\.)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
            raw_content
        )))[:10]

        return ExtractedContent(
            source=source,
            full_text=raw_content[:self.max_content_length],
            quotes=quotes[:5],
            key_points=key_points,
            mentioned_entities=mentioned,
        )


class SearchExecutor:
    """
    Executes search queries using available search APIs.

    Supports:
    1. Jina Search API
    2. Brave Search API
    3. SerpAPI (Google)
    4. Fallback to DuckDuckGo
    """

    def __init__(
        self,
        jina_api_key: Optional[str] = None,
        brave_api_key: Optional[str] = None,
        serp_api_key: Optional[str] = None,
        max_results_per_query: int = 10,
    ):
        self.jina_api_key = jina_api_key
        self.brave_api_key = brave_api_key
        self.serp_api_key = serp_api_key
        self.max_results = max_results_per_query
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15.0)
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def search(self, query: str, time_filter: Optional[str] = None) -> List[Source]:
        """Execute search and return sources."""
        # Try APIs in order of preference
        if self.jina_api_key:
            results = await self._search_jina(query)
            if results:
                return results

        if self.brave_api_key:
            results = await self._search_brave(query, time_filter)
            if results:
                return results

        # Fallback - return empty (in production, add more fallbacks)
        logger.warning(f"No search API available for query: {query}")
        return []

    async def search_parallel(
        self,
        queries: List[str],
        max_concurrent: int = 3,
    ) -> Dict[str, List[Source]]:
        """Execute multiple searches in parallel."""
        semaphore = asyncio.Semaphore(max_concurrent)
        results = {}

        async def search_with_semaphore(query: str) -> tuple:
            async with semaphore:
                sources = await self.search(query)
                return query, sources

        tasks = [
            asyncio.create_task(search_with_semaphore(q))
            for q in queries
        ]

        for task in asyncio.as_completed(tasks):
            query, sources = await task
            results[query] = sources

        return results

    async def _search_jina(self, query: str) -> List[Source]:
        """Search using Jina Search API."""
        try:
            session = await self._get_session()
            url = f"https://s.jina.ai/{query}"
            headers = {
                "Authorization": f"Bearer {self.jina_api_key}",
                "Accept": "application/json",
            }

            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_jina_results(data)
        except Exception as e:
            logger.warning(f"Jina search failed: {e}")
        return []

    async def _search_brave(
        self,
        query: str,
        time_filter: Optional[str] = None
    ) -> List[Source]:
        """Search using Brave Search API."""
        try:
            session = await self._get_session()
            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {
                "X-Subscription-Token": self.brave_api_key,
                "Accept": "application/json",
            }
            params = {
                "q": query,
                "count": self.max_results,
            }
            if time_filter:
                # Map our time filters to Brave's
                time_map = {
                    "past_week": "pw",
                    "past_month": "pm",
                    "past_year": "py",
                }
                if time_filter in time_map:
                    params["freshness"] = time_map[time_filter]

            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_brave_results(data)
        except Exception as e:
            logger.warning(f"Brave search failed: {e}")
        return []

    def _parse_jina_results(self, data: Dict[str, Any]) -> List[Source]:
        """Parse Jina search results into Sources."""
        sources = []
        results = data.get("data", [])

        for result in results[:self.max_results]:
            sources.append(Source(
                url=result.get("url", ""),
                title=result.get("title", ""),
                snippet=result.get("description", result.get("content", ""))[:500],
            ))

        return sources

    def _parse_brave_results(self, data: Dict[str, Any]) -> List[Source]:
        """Parse Brave search results into Sources."""
        sources = []
        results = data.get("web", {}).get("results", [])

        for result in results[:self.max_results]:
            sources.append(Source(
                url=result.get("url", ""),
                title=result.get("title", ""),
                snippet=result.get("description", "")[:500],
            ))

        return sources
