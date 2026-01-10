"""
Persona Research Hydrator

Multi-turn streaming research pipeline that enriches persona agents
with up-to-date contextual information from web sources.

Pipeline:
    1. Query Generation - Create targeted search queries
    2. Parallel Search - Execute queries across search APIs
    3. Source Ranking - Rank by authority, recency, relevance
    4. Content Extraction - Fetch and parse top sources
    5. Context Synthesis - LLM synthesizes into coherent context

Each turn streams progress events for real-time visibility.
"""

from dataclasses import dataclass, field, asdict
from typing import (
    Optional, Dict, Any, List, AsyncIterator,
    Callable, Awaitable
)
from datetime import datetime
from enum import Enum
import asyncio
import json
import logging
import os

from .queries import QueryGenerator, QueryPlan, SearchQuery
from .sources import (
    Source, SourceRanker, SourceFetcher, SearchExecutor,
    ExtractedContent, SourceType
)
from .cache import ResearchCache, CachedResearch

logger = logging.getLogger(__name__)


class ResearchPhase(Enum):
    """Phases of the research pipeline."""
    INITIALIZING = "initializing"
    GENERATING_QUERIES = "generating_queries"
    SEARCHING = "searching"
    RANKING = "ranking"
    FETCHING = "fetching"
    EXTRACTING = "extracting"
    SYNTHESIZING = "synthesizing"
    COMPLETE = "complete"
    ERROR = "error"
    CACHED = "cached"


@dataclass
class ResearchProgress:
    """Progress update during research."""
    phase: ResearchPhase
    message: str
    progress: float  # 0.0 to 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.value,
            "message": self.message,
            "progress": self.progress,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_sse(self) -> str:
        """Format as Server-Sent Event."""
        return f"data: {json.dumps(self.to_dict())}\n\n"


@dataclass
class ResearchContext:
    """Synthesized research context for persona hydration."""
    persona_id: str
    persona_name: str
    topic: str

    # Research results
    recent_statements: List[str] = field(default_factory=list)
    position_summary: str = ""
    key_quotes: List[Dict[str, str]] = field(default_factory=list)  # {"quote": ..., "source": ...}
    recent_events: List[str] = field(default_factory=list)
    controversies: List[str] = field(default_factory=list)
    relationships: List[Dict[str, str]] = field(default_factory=list)
    financial_info: List[str] = field(default_factory=list)

    # Metadata
    sources_consulted: int = 0
    queries_executed: int = 0
    research_time_seconds: float = 0.0
    cached: bool = False
    cache_age_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_prompt_injection(self) -> str:
        """
        Generate a context block to inject into persona prompt.

        This is the key output - a structured block that augments
        the static persona profile with current information.
        """
        lines = [
            "## RECENT CONTEXT (as of {date})".format(
                date=datetime.now().strftime("%Y-%m-%d")
            ),
            ""
        ]

        if self.position_summary:
            lines.extend([
                "### Current Position on Topic",
                self.position_summary,
                ""
            ])

        if self.recent_statements:
            lines.extend([
                "### Recent Statements",
                *[f"- {s}" for s in self.recent_statements[:5]],
                ""
            ])

        if self.key_quotes:
            lines.extend([
                "### Direct Quotes",
                *[f'- "{q["quote"]}" (Source: {q.get("source", "unknown")})'
                  for q in self.key_quotes[:3]],
                ""
            ])

        if self.recent_events:
            lines.extend([
                "### Recent Events",
                *[f"- {e}" for e in self.recent_events[:5]],
                ""
            ])

        if self.controversies:
            lines.extend([
                "### Recent Controversies/Disputes",
                *[f"- {c}" for c in self.controversies[:3]],
                ""
            ])

        if self.financial_info:
            lines.extend([
                "### Financial/Business Updates",
                *[f"- {f}" for f in self.financial_info[:3]],
                ""
            ])

        if self.relationships:
            lines.extend([
                "### Notable Relationships/Interactions",
                *[f"- {r.get('person', 'Unknown')}: {r.get('nature', '')}"
                  for r in self.relationships[:3]],
                ""
            ])

        lines.append(f"[Based on {self.sources_consulted} sources, {self.queries_executed} queries]")

        return "\n".join(lines)


# Type for LLM call function
LLMCallFn = Callable[[str, str], Awaitable[str]]


class PersonaHydrator:
    """
    Main research hydration engine.

    Orchestrates the multi-turn research pipeline and provides
    streaming progress updates.
    """

    def __init__(
        self,
        # API keys
        jina_api_key: Optional[str] = None,
        brave_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,

        # Redis client for caching
        redis_client=None,

        # LLM function for synthesis
        llm_call_fn: Optional[LLMCallFn] = None,

        # Configuration
        max_queries: int = 12,
        max_sources_to_fetch: int = 8,
        max_concurrent_fetches: int = 4,
        cache_ttl_hours: int = 24,

        # Cost control
        max_search_cost: float = 0.10,  # Max $ for search APIs
        max_llm_cost: float = 0.05,     # Max $ for synthesis
    ):
        # API keys from params or env
        self.jina_api_key = jina_api_key or os.getenv("JINA_API_KEY")
        self.brave_api_key = brave_api_key or os.getenv("BRAVE_API_KEY")
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")

        # Components
        self.query_generator = QueryGenerator(max_queries=max_queries)
        self.search_executor = SearchExecutor(
            jina_api_key=self.jina_api_key,
            brave_api_key=self.brave_api_key,
            max_results_per_query=10,
        )
        self.source_fetcher = SourceFetcher(
            jina_api_key=self.jina_api_key,
            timeout=10.0,
        )
        self.cache = ResearchCache(
            redis_client=redis_client,
            default_ttl_hours=cache_ttl_hours,
        )

        # LLM for synthesis
        self.llm_call_fn = llm_call_fn
        self.openrouter_api_key = openrouter_api_key

        # Config
        self.max_sources_to_fetch = max_sources_to_fetch
        self.max_concurrent_fetches = max_concurrent_fetches

        # Cost tracking
        self.max_search_cost = max_search_cost
        self.max_llm_cost = max_llm_cost
        self._session_search_cost = 0.0
        self._session_llm_cost = 0.0

    async def close(self):
        """Clean up resources."""
        await self.search_executor.close()
        await self.source_fetcher.close()

    async def hydrate(
        self,
        persona_id: str,
        persona_name: str,
        topic: str,
        persona_context: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
        include_financial: bool = True,
        include_legal: bool = True,
    ) -> AsyncIterator[ResearchProgress | ResearchContext]:
        """
        Execute the full research hydration pipeline.

        Yields ResearchProgress updates during execution,
        and a final ResearchContext when complete.

        Args:
            persona_id: Unique identifier for persona
            persona_name: Human-readable name (e.g., "Sam Altman")
            topic: Topic to research
            persona_context: Optional additional context from persona YAML
            force_refresh: If True, bypass cache
            include_financial: Include financial queries
            include_legal: Include legal queries

        Yields:
            ResearchProgress during execution
            ResearchContext as final yield
        """
        start_time = datetime.now()

        # Phase 0: Check cache
        yield ResearchProgress(
            phase=ResearchPhase.INITIALIZING,
            message=f"Starting research for {persona_name} on '{topic}'",
            progress=0.0,
            details={"persona_id": persona_id, "topic": topic}
        )

        if not force_refresh:
            cached = await self.cache.get(persona_id, topic)
            if cached:
                yield ResearchProgress(
                    phase=ResearchPhase.CACHED,
                    message=f"Found cached research ({cached.age_seconds / 3600:.1f}h old)",
                    progress=1.0,
                    details={"cache_age_hours": cached.age_seconds / 3600}
                )

                context = self._cached_to_context(cached, persona_id, persona_name, topic)
                yield context
                return

        # Phase 1: Generate queries
        yield ResearchProgress(
            phase=ResearchPhase.GENERATING_QUERIES,
            message="Generating search queries...",
            progress=0.1,
        )

        query_plan = self.query_generator.generate(
            persona_id=persona_id,
            persona_name=persona_name,
            topic=topic,
            persona_context=persona_context,
            include_financial=include_financial,
            include_legal=include_legal,
        )

        yield ResearchProgress(
            phase=ResearchPhase.GENERATING_QUERIES,
            message=f"Generated {len(query_plan.queries)} targeted queries",
            progress=0.15,
            details={
                "query_count": len(query_plan.queries),
                "queries": [q.query for q in query_plan.queries[:5]],
            }
        )

        # Phase 2: Execute searches in parallel
        yield ResearchProgress(
            phase=ResearchPhase.SEARCHING,
            message="Executing parallel searches...",
            progress=0.2,
        )

        all_sources: List[Source] = []
        queries_executed = 0

        # Execute searches
        query_strings = [q.to_search_string() for q in query_plan.by_priority()[:8]]
        search_results = await self.search_executor.search_parallel(
            query_strings,
            max_concurrent=3,
        )

        for query, sources in search_results.items():
            queries_executed += 1
            all_sources.extend(sources)

            yield ResearchProgress(
                phase=ResearchPhase.SEARCHING,
                message=f"Query {queries_executed}/{len(query_strings)}: Found {len(sources)} results",
                progress=0.2 + (0.2 * queries_executed / len(query_strings)),
                details={
                    "query": query[:50],
                    "results_found": len(sources),
                    "total_sources": len(all_sources),
                }
            )

        # Phase 3: Rank sources
        yield ResearchProgress(
            phase=ResearchPhase.RANKING,
            message=f"Ranking {len(all_sources)} sources...",
            progress=0.45,
        )

        ranker = SourceRanker(persona_name, topic)
        ranked_sources = ranker.rank_sources(all_sources)

        # Deduplicate by URL
        seen_urls = set()
        unique_sources = []
        for source in ranked_sources:
            if source.url not in seen_urls:
                seen_urls.add(source.url)
                unique_sources.append(source)

        top_sources = unique_sources[:self.max_sources_to_fetch]

        yield ResearchProgress(
            phase=ResearchPhase.RANKING,
            message=f"Selected top {len(top_sources)} sources",
            progress=0.5,
            details={
                "top_sources": [
                    {"title": s.title[:50], "score": round(s.composite_score, 2)}
                    for s in top_sources[:5]
                ]
            }
        )

        # Phase 4: Fetch and extract content
        yield ResearchProgress(
            phase=ResearchPhase.FETCHING,
            message="Fetching source content...",
            progress=0.55,
        )

        extracted_contents: List[ExtractedContent] = []
        fetch_count = 0

        async for content in self.source_fetcher.fetch_sources_parallel(
            top_sources,
            max_concurrent=self.max_concurrent_fetches,
        ):
            fetch_count += 1
            extracted_contents.append(content)

            yield ResearchProgress(
                phase=ResearchPhase.FETCHING,
                message=f"Fetched {fetch_count}/{len(top_sources)}",
                progress=0.55 + (0.15 * fetch_count / len(top_sources)),
                details={
                    "source": content.source.title[:40],
                    "quotes_found": len(content.quotes),
                    "key_points_found": len(content.key_points),
                }
            )

        # Phase 5: Extract structured information
        yield ResearchProgress(
            phase=ResearchPhase.EXTRACTING,
            message="Extracting key information...",
            progress=0.75,
        )

        extracted_data = self._extract_structured_info(
            extracted_contents,
            persona_name,
            topic,
        )

        yield ResearchProgress(
            phase=ResearchPhase.EXTRACTING,
            message=f"Extracted {len(extracted_data['quotes'])} quotes, {len(extracted_data['key_points'])} key points",
            progress=0.8,
            details=extracted_data,
        )

        # Phase 6: Synthesize context (LLM call if available)
        yield ResearchProgress(
            phase=ResearchPhase.SYNTHESIZING,
            message="Synthesizing research context...",
            progress=0.85,
        )

        context = await self._synthesize_context(
            persona_id=persona_id,
            persona_name=persona_name,
            topic=topic,
            extracted_data=extracted_data,
            sources=top_sources,
        )

        # Update metadata
        context.sources_consulted = len(top_sources)
        context.queries_executed = queries_executed
        context.research_time_seconds = (datetime.now() - start_time).total_seconds()

        # Cache the results
        await self.cache.set(
            persona_id=persona_id,
            topic=topic,
            context=context.to_dict(),
            sources=[s.to_dict() for s in top_sources],
            query_count=queries_executed,
            source_count=len(top_sources),
        )

        yield ResearchProgress(
            phase=ResearchPhase.COMPLETE,
            message=f"Research complete in {context.research_time_seconds:.1f}s",
            progress=1.0,
            details={
                "sources_consulted": context.sources_consulted,
                "queries_executed": context.queries_executed,
                "cached": True,
            }
        )

        yield context

    def _extract_structured_info(
        self,
        contents: List[ExtractedContent],
        persona_name: str,
        topic: str,
    ) -> Dict[str, Any]:
        """Extract structured information from fetched content."""
        all_quotes = []
        all_key_points = []
        all_entities = []

        persona_name_lower = persona_name.lower()

        for content in contents:
            # Filter quotes that mention the persona
            for quote in content.quotes:
                if persona_name_lower in quote.lower() or len(quote) < 200:
                    all_quotes.append({
                        "quote": quote,
                        "source": content.source.title,
                        "url": content.source.url,
                    })

            # Filter key points
            for point in content.key_points:
                if persona_name_lower in point.lower():
                    all_key_points.append({
                        "point": point,
                        "source": content.source.title,
                    })

            all_entities.extend(content.mentioned_entities)

        return {
            "quotes": all_quotes[:10],
            "key_points": all_key_points[:15],
            "entities": list(set(all_entities))[:20],
        }

    async def _synthesize_context(
        self,
        persona_id: str,
        persona_name: str,
        topic: str,
        extracted_data: Dict[str, Any],
        sources: List[Source],
    ) -> ResearchContext:
        """Synthesize extracted data into a coherent context."""
        context = ResearchContext(
            persona_id=persona_id,
            persona_name=persona_name,
            topic=topic,
        )

        # Add quotes
        context.key_quotes = extracted_data.get("quotes", [])[:5]

        # Add key points as recent statements
        context.recent_statements = [
            p["point"] for p in extracted_data.get("key_points", [])[:5]
        ]

        # Categorize sources by type
        for source in sources:
            if source.source_type == SourceType.LEGAL:
                context.financial_info.append(
                    f"Legal/SEC filing: {source.title}"
                )
            elif source.source_type in [SourceType.CONTROVERSIES]:
                context.controversies.append(source.title)

        # If we have an LLM, use it to synthesize a position summary
        if self.llm_call_fn or self.openrouter_api_key:
            try:
                context.position_summary = await self._llm_synthesize_position(
                    persona_name, topic, extracted_data
                )
            except Exception as e:
                logger.warning(f"LLM synthesis failed: {e}")
                context.position_summary = self._fallback_position_summary(
                    persona_name, topic, extracted_data
                )
        else:
            context.position_summary = self._fallback_position_summary(
                persona_name, topic, extracted_data
            )

        return context

    async def _llm_synthesize_position(
        self,
        persona_name: str,
        topic: str,
        extracted_data: Dict[str, Any],
    ) -> str:
        """Use LLM to synthesize a position summary."""
        quotes = extracted_data.get("quotes", [])
        points = extracted_data.get("key_points", [])

        # Build evidence string
        evidence_parts = []
        for q in quotes[:3]:
            evidence_parts.append(f'Quote: "{q["quote"]}" (from {q["source"]})')
        for p in points[:3]:
            evidence_parts.append(f'Statement: {p["point"]}')

        evidence = "\n".join(evidence_parts)

        prompt = f"""Based on the following recent evidence, summarize {persona_name}'s current position on "{topic}" in 2-3 sentences. Be specific and cite any notable recent statements or events.

EVIDENCE:
{evidence}

SUMMARY:"""

        if self.llm_call_fn:
            return await self.llm_call_fn("synthesis", prompt)
        elif self.openrouter_api_key:
            return await self._call_openrouter(prompt)
        else:
            return self._fallback_position_summary(persona_name, topic, extracted_data)

    async def _call_openrouter(self, prompt: str) -> str:
        """Call OpenRouter API for synthesis."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "anthropic/claude-3-haiku",  # Fast, cheap
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                    "temperature": 0.3,
                },
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    raise Exception(f"OpenRouter API error: {response.status}")

    def _fallback_position_summary(
        self,
        persona_name: str,
        topic: str,
        extracted_data: Dict[str, Any],
    ) -> str:
        """Generate a fallback position summary without LLM."""
        quotes = extracted_data.get("quotes", [])
        points = extracted_data.get("key_points", [])

        if quotes:
            return f'{persona_name} recently stated: "{quotes[0]["quote"][:150]}..."'
        elif points:
            return f"Recent coverage indicates {persona_name} has been discussing: {points[0]['point'][:150]}"
        else:
            return f"Limited recent information found about {persona_name}'s position on {topic}."

    def _cached_to_context(
        self,
        cached: CachedResearch,
        persona_id: str,
        persona_name: str,
        topic: str,
    ) -> ResearchContext:
        """Convert cached research back to ResearchContext."""
        ctx_data = cached.context
        context = ResearchContext(
            persona_id=persona_id,
            persona_name=persona_name,
            topic=topic,
            recent_statements=ctx_data.get("recent_statements", []),
            position_summary=ctx_data.get("position_summary", ""),
            key_quotes=ctx_data.get("key_quotes", []),
            recent_events=ctx_data.get("recent_events", []),
            controversies=ctx_data.get("controversies", []),
            relationships=ctx_data.get("relationships", []),
            financial_info=ctx_data.get("financial_info", []),
            sources_consulted=cached.source_count,
            queries_executed=cached.query_count,
            cached=True,
            cache_age_seconds=cached.age_seconds,
        )
        return context

    async def get_cached_context(
        self,
        persona_id: str,
        persona_name: str,
        topic: str,
        max_age_hours: Optional[int] = None,
    ) -> Optional[ResearchContext]:
        """Get cached context without executing research."""
        cached = await self.cache.get(persona_id, topic, max_age_hours)
        if cached:
            return self._cached_to_context(cached, persona_id, persona_name, topic)
        return None

    async def invalidate_cache(self, persona_id: str, topic: Optional[str] = None) -> int:
        """Invalidate cached research."""
        if topic:
            success = await self.cache.invalidate(persona_id, topic)
            return 1 if success else 0
        else:
            return await self.cache.invalidate_persona(persona_id)
