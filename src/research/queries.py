"""
Query Generation for Persona Research

Generates intelligent, parallel search queries to gather
up-to-date information about a persona on a specific topic.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
import re


class QueryType(Enum):
    """Types of search queries for different information needs."""
    RECENT_STATEMENTS = "recent_statements"      # What they've said recently
    POSITION_CHANGES = "position_changes"        # How views have evolved
    CONTROVERSIES = "controversies"              # Recent disputes/conflicts
    OFFICIAL_ANNOUNCEMENTS = "official"          # Company/org announcements
    INTERVIEWS = "interviews"                    # Recent interviews/podcasts
    SOCIAL_MEDIA = "social"                      # Twitter/X, LinkedIn posts
    LEGAL_REGULATORY = "legal"                   # Legal filings, regulatory
    FINANCIAL = "financial"                      # Financial disclosures, deals
    RELATIONSHIPS = "relationships"              # Alliances, conflicts with others


@dataclass
class SearchQuery:
    """A single search query with metadata."""
    query: str
    query_type: QueryType
    priority: int = 1  # 1=highest
    time_filter: Optional[str] = None  # "past_week", "past_month", "past_year"
    site_filter: Optional[str] = None  # e.g., "twitter.com", "nytimes.com"

    def to_search_string(self) -> str:
        """Convert to actual search string with filters."""
        parts = [self.query]
        if self.site_filter:
            parts.append(f"site:{self.site_filter}")
        return " ".join(parts)


@dataclass
class QueryPlan:
    """A plan of queries to execute for persona research."""
    persona_id: str
    persona_name: str
    topic: str
    queries: List[SearchQuery] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)

    def by_priority(self) -> List[SearchQuery]:
        """Return queries sorted by priority."""
        return sorted(self.queries, key=lambda q: q.priority)

    def by_type(self, query_type: QueryType) -> List[SearchQuery]:
        """Return queries of a specific type."""
        return [q for q in self.queries if q.query_type == query_type]


class QueryGenerator:
    """
    Generates intelligent search queries for persona research.

    Uses persona metadata and topic to create targeted queries
    that surface recent, relevant information.
    """

    # Known social handles for personas (expand as needed)
    SOCIAL_HANDLES = {
        "sam_altman": {"twitter": "sama", "linkedin": "samaltman"},
        "elon_musk": {"twitter": "elonmusk", "linkedin": "elonmusk"},
        "dario_amodei": {"twitter": "daboratory", "linkedin": "dario-amodei"},
        "satya_nadella": {"twitter": "satlonanadella", "linkedin": "satyanadella"},
        "sundar_pichai": {"twitter": "sundarpichai", "linkedin": "sundarpichai"},
        "jensen_huang": {"twitter": "nvidia", "linkedin": "jenhsunhuang"},
        "geoffrey_hinton": {"twitter": "geoffreyhinton"},
        "yann_lecun": {"twitter": "ylecun", "linkedin": "yann-lecun"},
        "yoshua_bengio": {"twitter": "yoshuabengio"},
        "ilya_sutskever": {"linkedin": "ilya-sutskever"},
        "eliezer_yudkowsky": {"twitter": "ESYudkowsky"},
        "paul_christiano": {"twitter": "paulfchristiano"},
        "timnit_gebru": {"twitter": "timnitGebru"},
        "marc_andreessen": {"twitter": "pmarca", "linkedin": "mandreessen"},
        "peter_thiel": {"linkedin": "peterthiel"},
        "ezra_klein": {"twitter": "ezraklein"},
        "kara_swisher": {"twitter": "karaswisher"},
        "tristan_harris": {"twitter": "tristanharris"},
        "chuck_schumer": {"twitter": "SenSchumer"},
        "gina_raimondo": {"twitter": "SecRaimondo"},
        "thierry_breton": {"twitter": "ThierryBreton"},
        "emmanuel_macron": {"twitter": "EmmanuelMacron"},
        "fei_fei_li": {"twitter": "drfeifei"},
        "stuart_russell": {"twitter": "stuartjrussell"},
    }

    # Organization affiliations for personas
    ORG_AFFILIATIONS = {
        "sam_altman": ["OpenAI"],
        "dario_amodei": ["Anthropic"],
        "elon_musk": ["xAI", "Tesla", "SpaceX", "X"],
        "satya_nadella": ["Microsoft"],
        "sundar_pichai": ["Google", "Alphabet", "DeepMind"],
        "jensen_huang": ["NVIDIA"],
        "geoffrey_hinton": ["Google", "University of Toronto"],
        "yann_lecun": ["Meta", "Meta AI", "FAIR", "NYU"],
        "yoshua_bengio": ["Mila", "University of Montreal"],
        "ilya_sutskever": ["SSI", "Safe Superintelligence", "OpenAI"],
        "eliezer_yudkowsky": ["MIRI", "Machine Intelligence Research Institute"],
        "paul_christiano": ["ARC", "Alignment Research Center"],
        "timnit_gebru": ["DAIR", "Distributed AI Research Institute"],
        "marc_andreessen": ["a16z", "Andreessen Horowitz"],
        "peter_thiel": ["Founders Fund", "Palantir"],
        "fei_fei_li": ["Stanford HAI", "Stanford"],
        "stuart_russell": ["UC Berkeley", "CHAI"],
        "tristan_harris": ["Center for Humane Technology"],
    }

    # Topic-specific query templates
    TOPIC_TEMPLATES = {
        "ai_safety": [
            "{name} AI safety position",
            "{name} existential risk AI",
            "{name} AI alignment views",
        ],
        "ai_regulation": [
            "{name} AI regulation testimony",
            "{name} government AI policy",
            "{name} AI Act EU views",
        ],
        "open_source": [
            "{name} open source AI",
            "{name} open weights models",
            "{name} AI democratization",
        ],
        "agi": [
            "{name} AGI timeline",
            "{name} artificial general intelligence prediction",
            "{name} superintelligence views",
        ],
        "ethics": [
            "{name} AI ethics",
            "{name} AI bias fairness",
            "{name} responsible AI",
        ],
    }

    def __init__(self, max_queries: int = 12):
        self.max_queries = max_queries

    def generate(
        self,
        persona_id: str,
        persona_name: str,
        topic: str,
        persona_context: Optional[Dict[str, Any]] = None,
        include_financial: bool = True,
        include_legal: bool = True,
        time_focus: str = "recent",  # "recent", "all_time"
    ) -> QueryPlan:
        """
        Generate a comprehensive query plan for researching a persona.

        Args:
            persona_id: Unique identifier for persona
            persona_name: Human-readable name
            topic: Topic to research (can be specific or general)
            persona_context: Optional dict with additional persona info
            include_financial: Include financial/business queries
            include_legal: Include legal/regulatory queries
            time_focus: "recent" for past few months, "all_time" for broader search

        Returns:
            QueryPlan with prioritized search queries
        """
        queries = []

        # Normalize name for search
        search_name = self._normalize_name(persona_name)

        # 1. Recent statements on topic (highest priority)
        queries.extend(self._generate_statement_queries(
            search_name, topic, priority=1
        ))

        # 2. Organization-related queries
        if persona_id in self.ORG_AFFILIATIONS:
            queries.extend(self._generate_org_queries(
                search_name,
                self.ORG_AFFILIATIONS[persona_id],
                topic,
                priority=2
            ))

        # 3. Interview and podcast queries
        queries.extend(self._generate_interview_queries(
            search_name, topic, priority=2
        ))

        # 4. Social media queries
        if persona_id in self.SOCIAL_HANDLES:
            queries.extend(self._generate_social_queries(
                search_name,
                self.SOCIAL_HANDLES[persona_id],
                topic,
                priority=3
            ))

        # 5. Controversy/conflict queries
        queries.extend(self._generate_controversy_queries(
            search_name, topic, priority=3
        ))

        # 6. Financial queries (if enabled)
        if include_financial:
            queries.extend(self._generate_financial_queries(
                search_name, priority=4
            ))

        # 7. Legal/regulatory queries (if enabled)
        if include_legal:
            queries.extend(self._generate_legal_queries(
                search_name, priority=4
            ))

        # 8. Topic-specific templates
        topic_key = self._detect_topic_category(topic)
        if topic_key and topic_key in self.TOPIC_TEMPLATES:
            for template in self.TOPIC_TEMPLATES[topic_key]:
                queries.append(SearchQuery(
                    query=template.format(name=search_name),
                    query_type=QueryType.RECENT_STATEMENTS,
                    priority=2,
                    time_filter="past_month"
                ))

        # Apply time filters based on focus
        if time_focus == "recent":
            for q in queries:
                if q.time_filter is None:
                    q.time_filter = "past_month"

        # Limit and deduplicate
        queries = self._deduplicate_queries(queries)[:self.max_queries]

        return QueryPlan(
            persona_id=persona_id,
            persona_name=persona_name,
            topic=topic,
            queries=queries
        )

    def _normalize_name(self, name: str) -> str:
        """Convert persona_id style to searchable name."""
        # sam_altman -> Sam Altman
        # Handle special cases
        special_cases = {
            "fei_fei_li": "Fei-Fei Li",
            "yann_lecun": "Yann LeCun",
        }
        if name.lower().replace(" ", "_") in special_cases:
            return special_cases[name.lower().replace(" ", "_")]

        return " ".join(word.capitalize() for word in name.replace("_", " ").split())

    def _generate_statement_queries(
        self, name: str, topic: str, priority: int
    ) -> List[SearchQuery]:
        """Generate queries for recent statements on topic."""
        return [
            SearchQuery(
                query=f'"{name}" {topic}',
                query_type=QueryType.RECENT_STATEMENTS,
                priority=priority,
                time_filter="past_week"
            ),
            SearchQuery(
                query=f'"{name}" says {topic}',
                query_type=QueryType.RECENT_STATEMENTS,
                priority=priority,
                time_filter="past_month"
            ),
            SearchQuery(
                query=f'"{name}" interview {topic}',
                query_type=QueryType.INTERVIEWS,
                priority=priority,
                time_filter="past_month"
            ),
        ]

    def _generate_org_queries(
        self, name: str, orgs: List[str], topic: str, priority: int
    ) -> List[SearchQuery]:
        """Generate queries related to person's organizations."""
        queries = []
        for org in orgs[:2]:  # Limit to top 2 orgs
            queries.append(SearchQuery(
                query=f"{org} {topic} announcement",
                query_type=QueryType.OFFICIAL_ANNOUNCEMENTS,
                priority=priority,
                time_filter="past_month"
            ))
            queries.append(SearchQuery(
                query=f"{name} {org} {topic}",
                query_type=QueryType.RECENT_STATEMENTS,
                priority=priority + 1,
                time_filter="past_month"
            ))
        return queries

    def _generate_interview_queries(
        self, name: str, topic: str, priority: int
    ) -> List[SearchQuery]:
        """Generate queries for interviews and podcasts."""
        return [
            SearchQuery(
                query=f'"{name}" podcast interview 2024',
                query_type=QueryType.INTERVIEWS,
                priority=priority,
                time_filter="past_month"
            ),
            SearchQuery(
                query=f'"{name}" testimony Congress Senate',
                query_type=QueryType.INTERVIEWS,
                priority=priority,
                time_filter="past_year"
            ),
        ]

    def _generate_social_queries(
        self, name: str, handles: Dict[str, str], topic: str, priority: int
    ) -> List[SearchQuery]:
        """Generate social media queries."""
        queries = []
        if "twitter" in handles:
            queries.append(SearchQuery(
                query=f"@{handles['twitter']} {topic}",
                query_type=QueryType.SOCIAL_MEDIA,
                priority=priority,
                site_filter="twitter.com"
            ))
            queries.append(SearchQuery(
                query=f"from:@{handles['twitter']}",
                query_type=QueryType.SOCIAL_MEDIA,
                priority=priority,
                site_filter="x.com"
            ))
        return queries

    def _generate_controversy_queries(
        self, name: str, topic: str, priority: int
    ) -> List[SearchQuery]:
        """Generate queries for controversies and conflicts."""
        return [
            SearchQuery(
                query=f'"{name}" controversy',
                query_type=QueryType.CONTROVERSIES,
                priority=priority,
                time_filter="past_month"
            ),
            SearchQuery(
                query=f'"{name}" criticism response',
                query_type=QueryType.CONTROVERSIES,
                priority=priority,
                time_filter="past_month"
            ),
        ]

    def _generate_financial_queries(
        self, name: str, priority: int
    ) -> List[SearchQuery]:
        """Generate financial/business queries."""
        return [
            SearchQuery(
                query=f'"{name}" valuation funding investment',
                query_type=QueryType.FINANCIAL,
                priority=priority,
                time_filter="past_month"
            ),
            SearchQuery(
                query=f'"{name}" stock equity compensation',
                query_type=QueryType.FINANCIAL,
                priority=priority,
                time_filter="past_year"
            ),
            SearchQuery(
                query=f'"{name}" SEC filing deposition',
                query_type=QueryType.LEGAL_REGULATORY,
                priority=priority,
            ),
        ]

    def _generate_legal_queries(
        self, name: str, priority: int
    ) -> List[SearchQuery]:
        """Generate legal/regulatory queries."""
        return [
            SearchQuery(
                query=f'"{name}" lawsuit legal',
                query_type=QueryType.LEGAL_REGULATORY,
                priority=priority,
                time_filter="past_year"
            ),
            SearchQuery(
                query=f'"{name}" court filing deposition testimony',
                query_type=QueryType.LEGAL_REGULATORY,
                priority=priority,
            ),
        ]

    def _detect_topic_category(self, topic: str) -> Optional[str]:
        """Detect which topic category the query falls into."""
        topic_lower = topic.lower()

        if any(w in topic_lower for w in ["safety", "risk", "alignment", "doom"]):
            return "ai_safety"
        if any(w in topic_lower for w in ["regulation", "law", "government", "policy"]):
            return "ai_regulation"
        if any(w in topic_lower for w in ["open source", "open-source", "weights"]):
            return "open_source"
        if any(w in topic_lower for w in ["agi", "superintelligence", "general intelligence"]):
            return "agi"
        if any(w in topic_lower for w in ["ethics", "bias", "fairness", "harm"]):
            return "ethics"

        return None

    def _deduplicate_queries(self, queries: List[SearchQuery]) -> List[SearchQuery]:
        """Remove duplicate queries, keeping highest priority."""
        seen = {}
        for q in queries:
            key = q.query.lower().strip()
            if key not in seen or seen[key].priority > q.priority:
                seen[key] = q
        return list(seen.values())
