"""
Deep Research Hydration Module

Multi-turn streaming research pipeline that enriches persona agents
with up-to-date contextual information from web sources.

Architecture:
    Query Generation → Parallel Search → Source Extraction → Context Synthesis

Each turn streams progress events, enabling real-time visibility into
the research process.
"""

from .hydrator import PersonaHydrator, ResearchContext, ResearchProgress
from .queries import QueryGenerator, SearchQuery
from .sources import SourceFetcher, SourceRanker, Source, ExtractedContent
from .cache import ResearchCache

__all__ = [
    "PersonaHydrator",
    "ResearchContext",
    "ResearchProgress",
    "QueryGenerator",
    "SearchQuery",
    "SourceFetcher",
    "SourceRanker",
    "Source",
    "ExtractedContent",
    "ResearchCache",
]
