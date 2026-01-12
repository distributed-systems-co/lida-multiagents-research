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
from .deep_search import (
    DeepSearchEngine,
    IntelligenceDossier,
    IntelligenceFragment,
    IntelligenceType,
    deep_research,
)
from .dreamspace import (
    DreamspaceEngine,
    DreamspaceSession,
    DreamspaceDialogue,
    DreamspaceAspect,
    PsychologicalAspect,
    TemporalSelf,
    dreamspace,
)
from .dreamspace_advanced import (
    AdvancedDreamspaceEngine,
    AdvancedDreamspaceSession,
    AdvancedDialogue,
    JungianArchetype,
    ShadowArchetype,
    IFSPartType,
    IFSPart,
    DefenseMechanism,
    SomaticZone,
    SomaticMarker,
    DialecticalTriad,
    DreamLogicMode,
    PsychologicalComplex,
    InterventionType,
    Intervention,
    advanced_dreamspace,
)
from .persona_profiles import (
    ALL_PERSONA_PROFILES,
    get_profile,
    get_all_profiles,
    get_profiles_by_category,
)

__all__ = [
    # Original hydrator
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
    # Deep search
    "DeepSearchEngine",
    "IntelligenceDossier",
    "IntelligenceFragment",
    "IntelligenceType",
    "deep_research",
    # Dreamspace
    "DreamspaceEngine",
    "DreamspaceSession",
    "DreamspaceDialogue",
    "DreamspaceAspect",
    "PsychologicalAspect",
    "TemporalSelf",
    "dreamspace",
    # Advanced Dreamspace
    "AdvancedDreamspaceEngine",
    "AdvancedDreamspaceSession",
    "AdvancedDialogue",
    "JungianArchetype",
    "ShadowArchetype",
    "IFSPartType",
    "IFSPart",
    "DefenseMechanism",
    "SomaticZone",
    "SomaticMarker",
    "DialecticalTriad",
    "DreamLogicMode",
    "PsychologicalComplex",
    "InterventionType",
    "Intervention",
    "advanced_dreamspace",
    # Persona Profiles
    "ALL_PERSONA_PROFILES",
    "get_profile",
    "get_all_profiles",
    "get_profiles_by_category",
]
