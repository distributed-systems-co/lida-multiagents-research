"""
Deep Search Module - Advanced multi-source intelligence gathering

Uses multiple search strategies in parallel to gather comprehensive
context about personas, including:
- News and recent statements
- Financial disclosures (SEC, court filings)
- Social media activity
- Academic publications
- Relationship mapping
"""

import asyncio
import aiohttp
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class IntelligenceType(Enum):
    """Types of intelligence to gather."""
    NEWS = "news"
    FINANCIAL = "financial"
    LEGAL = "legal"
    SOCIAL = "social"
    ACADEMIC = "academic"
    RELATIONSHIPS = "relationships"
    CONTROVERSY = "controversy"
    STATEMENTS = "statements"


@dataclass
class IntelligenceFragment:
    """A single piece of gathered intelligence."""
    intel_type: IntelligenceType
    title: str
    content: str
    source_url: str
    source_name: str
    date: Optional[str] = None
    confidence: float = 0.8
    entities_mentioned: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.intel_type.value,
            "title": self.title,
            "content": self.content,
            "source_url": self.source_url,
            "source_name": self.source_name,
            "date": self.date,
            "confidence": self.confidence,
            "entities": self.entities_mentioned,
        }


@dataclass
class IntelligenceDossier:
    """Complete intelligence dossier for a persona."""
    persona_id: str
    persona_name: str
    topic: str
    generated_at: datetime = field(default_factory=datetime.now)

    # Categorized intelligence
    news: List[IntelligenceFragment] = field(default_factory=list)
    financial: List[IntelligenceFragment] = field(default_factory=list)
    legal: List[IntelligenceFragment] = field(default_factory=list)
    statements: List[IntelligenceFragment] = field(default_factory=list)
    relationships: List[IntelligenceFragment] = field(default_factory=list)
    controversies: List[IntelligenceFragment] = field(default_factory=list)

    # Synthesis
    executive_summary: str = ""
    key_facts: List[str] = field(default_factory=list)
    position_on_topic: str = ""
    confidence_score: float = 0.0

    # Metadata
    queries_executed: int = 0
    sources_found: int = 0
    research_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "persona_id": self.persona_id,
            "persona_name": self.persona_name,
            "topic": self.topic,
            "generated_at": self.generated_at.isoformat(),
            "news": [f.to_dict() for f in self.news],
            "financial": [f.to_dict() for f in self.financial],
            "legal": [f.to_dict() for f in self.legal],
            "statements": [f.to_dict() for f in self.statements],
            "relationships": [f.to_dict() for f in self.relationships],
            "controversies": [f.to_dict() for f in self.controversies],
            "executive_summary": self.executive_summary,
            "key_facts": self.key_facts,
            "position_on_topic": self.position_on_topic,
            "confidence_score": self.confidence_score,
            "queries_executed": self.queries_executed,
            "sources_found": self.sources_found,
            "research_time_seconds": self.research_time_seconds,
        }

    def to_prompt_injection(self) -> str:
        """Generate context block for persona prompt injection."""
        lines = [
            f"## INTELLIGENCE BRIEFING: {self.persona_name}",
            f"Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M')}",
            f"Topic: {self.topic}",
            "",
        ]

        if self.executive_summary:
            lines.extend([
                "### Executive Summary",
                self.executive_summary,
                "",
            ])

        if self.key_facts:
            lines.extend([
                "### Key Facts",
                *[f"- {fact}" for fact in self.key_facts[:7]],
                "",
            ])

        if self.position_on_topic:
            lines.extend([
                "### Position on Topic",
                self.position_on_topic,
                "",
            ])

        if self.financial:
            lines.extend([
                "### Financial Intelligence",
                *[f"- {f.content[:200]}" for f in self.financial[:3]],
                "",
            ])

        if self.legal:
            lines.extend([
                "### Legal/Regulatory",
                *[f"- {f.content[:200]}" for f in self.legal[:3]],
                "",
            ])

        if self.controversies:
            lines.extend([
                "### Recent Controversies",
                *[f"- {c.title}: {c.content[:150]}" for c in self.controversies[:3]],
                "",
            ])

        if self.relationships:
            lines.extend([
                "### Key Relationships",
                *[f"- {r.content[:150]}" for r in self.relationships[:5]],
                "",
            ])

        lines.append(f"[Confidence: {self.confidence_score:.0%} | Sources: {self.sources_found}]")

        return "\n".join(lines)


class DeepSearchEngine:
    """
    Advanced multi-strategy search engine for persona research.

    Executes parallel searches across multiple dimensions to build
    a comprehensive intelligence dossier.
    """

    # Search query templates by intelligence type
    QUERY_TEMPLATES = {
        IntelligenceType.NEWS: [
            "{name} {topic} 2024 2025",
            "{name} latest news",
            "{name} announcement {topic}",
        ],
        IntelligenceType.FINANCIAL: [
            "{name} stock equity compensation",
            "{name} net worth valuation",
            "{name} SEC filing disclosure",
            "{name} funding investment",
        ],
        IntelligenceType.LEGAL: [
            "{name} lawsuit court filing",
            "{name} deposition testimony",
            "{name} legal dispute",
        ],
        IntelligenceType.STATEMENTS: [
            "{name} said {topic}",
            "{name} interview {topic}",
            "{name} testimony Congress",
            "{name} podcast {topic}",
        ],
        IntelligenceType.RELATIONSHIPS: [
            "{name} relationship {org}",
            "{name} rivalry dispute",
            "{name} alliance partnership",
        ],
        IntelligenceType.CONTROVERSY: [
            "{name} controversy scandal",
            "{name} criticism backlash",
            "{name} fired resigned",
        ],
    }

    # Known organizations for relationship queries
    PERSONA_ORGS = {
        "ilya_sutskever": ["OpenAI", "SSI", "Safe Superintelligence"],
        "sam_altman": ["OpenAI", "Y Combinator", "Worldcoin"],
        "elon_musk": ["xAI", "Tesla", "SpaceX", "X", "Neuralink"],
        "dario_amodei": ["Anthropic"],
        "satya_nadella": ["Microsoft"],
        "sundar_pichai": ["Google", "Alphabet", "DeepMind"],
        "geoffrey_hinton": ["Google", "University of Toronto"],
        "yann_lecun": ["Meta", "FAIR", "NYU"],
    }

    def __init__(
        self,
        openrouter_api_key: Optional[str] = None,
        max_parallel_searches: int = 8,
        max_results_per_search: int = 10,
    ):
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.max_parallel = max_parallel_searches
        self.max_results = max_results_per_search
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def research(
        self,
        persona_id: str,
        persona_name: str,
        topic: str,
        intel_types: Optional[List[IntelligenceType]] = None,
    ) -> IntelligenceDossier:
        """
        Execute comprehensive research on a persona.

        Args:
            persona_id: Unique identifier
            persona_name: Human-readable name
            topic: Topic to focus research on
            intel_types: Specific intelligence types to gather (default: all)

        Returns:
            IntelligenceDossier with all gathered intelligence
        """
        start_time = datetime.now()

        dossier = IntelligenceDossier(
            persona_id=persona_id,
            persona_name=persona_name,
            topic=topic,
        )

        # Determine which intelligence types to gather
        if intel_types is None:
            intel_types = list(IntelligenceType)

        # Generate all queries
        all_queries: List[Tuple[IntelligenceType, str]] = []
        org = self.PERSONA_ORGS.get(persona_id, [""])[0] if persona_id in self.PERSONA_ORGS else ""

        for intel_type in intel_types:
            templates = self.QUERY_TEMPLATES.get(intel_type, [])
            for template in templates:
                query = template.format(name=persona_name, topic=topic, org=org)
                all_queries.append((intel_type, query))

        logger.info(f"Executing {len(all_queries)} queries for {persona_name}")
        dossier.queries_executed = len(all_queries)

        # Execute searches in parallel batches
        all_fragments: List[IntelligenceFragment] = []

        for i in range(0, len(all_queries), self.max_parallel):
            batch = all_queries[i:i + self.max_parallel]
            tasks = [
                self._execute_search(intel_type, query)
                for intel_type, query in batch
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, list):
                    all_fragments.extend(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Search error: {result}")

        # Categorize fragments
        for fragment in all_fragments:
            if fragment.intel_type == IntelligenceType.NEWS:
                dossier.news.append(fragment)
            elif fragment.intel_type == IntelligenceType.FINANCIAL:
                dossier.financial.append(fragment)
            elif fragment.intel_type == IntelligenceType.LEGAL:
                dossier.legal.append(fragment)
            elif fragment.intel_type == IntelligenceType.STATEMENTS:
                dossier.statements.append(fragment)
            elif fragment.intel_type == IntelligenceType.RELATIONSHIPS:
                dossier.relationships.append(fragment)
            elif fragment.intel_type == IntelligenceType.CONTROVERSY:
                dossier.controversies.append(fragment)

        dossier.sources_found = len(all_fragments)

        # Synthesize intelligence
        if self.openrouter_api_key and all_fragments:
            await self._synthesize_dossier(dossier, all_fragments)
        else:
            # Fallback synthesis
            dossier.executive_summary = self._fallback_summary(dossier)
            dossier.key_facts = self._extract_key_facts(all_fragments)

        # Calculate confidence
        dossier.confidence_score = min(
            0.95,
            0.3 + (len(all_fragments) * 0.05) + (len(dossier.key_facts) * 0.1)
        )

        dossier.research_time_seconds = (datetime.now() - start_time).total_seconds()

        return dossier

    async def _execute_search(
        self,
        intel_type: IntelligenceType,
        query: str,
    ) -> List[IntelligenceFragment]:
        """Execute a single search and parse results into fragments."""
        # For now, we'll use a simplified search approach
        # In production, this would call actual search APIs
        fragments = []

        # This is where you'd integrate with Jina, Brave, etc.
        # For testing, we'll return empty list and rely on the LLM synthesis

        logger.debug(f"Search [{intel_type.value}]: {query}")

        return fragments

    async def _synthesize_dossier(
        self,
        dossier: IntelligenceDossier,
        fragments: List[IntelligenceFragment],
    ):
        """Use LLM to synthesize intelligence into coherent summary."""
        # Build context from fragments
        context_parts = []
        for f in fragments[:20]:  # Limit context size
            context_parts.append(f"[{f.intel_type.value}] {f.title}: {f.content[:300]}")

        context = "\n".join(context_parts)

        prompt = f"""Analyze the following intelligence gathered about {dossier.persona_name} regarding "{dossier.topic}".

INTELLIGENCE:
{context}

Provide:
1. Executive Summary (2-3 sentences)
2. Key Facts (bullet points, most important revelations)
3. Position on Topic (their likely stance based on evidence)

Format as JSON:
{{"executive_summary": "...", "key_facts": ["fact1", "fact2"], "position_on_topic": "..."}}"""

        try:
            session = await self._get_session()
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "anthropic/claude-3-haiku",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                    "temperature": 0.3,
                },
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data["choices"][0]["message"]["content"]

                    # Parse JSON response
                    try:
                        # Extract JSON from response
                        json_match = re.search(r'\{.*\}', content, re.DOTALL)
                        if json_match:
                            parsed = json.loads(json_match.group())
                            dossier.executive_summary = parsed.get("executive_summary", "")
                            dossier.key_facts = parsed.get("key_facts", [])
                            dossier.position_on_topic = parsed.get("position_on_topic", "")
                    except json.JSONDecodeError:
                        dossier.executive_summary = content
        except Exception as e:
            logger.warning(f"Synthesis failed: {e}")

    def _fallback_summary(self, dossier: IntelligenceDossier) -> str:
        """Generate fallback summary without LLM."""
        parts = []

        if dossier.financial:
            parts.append(f"Financial intelligence indicates {dossier.financial[0].content[:100]}.")
        if dossier.news:
            parts.append(f"Recent coverage: {dossier.news[0].title}.")
        if dossier.controversies:
            parts.append(f"Notable controversy: {dossier.controversies[0].title}.")

        return " ".join(parts) or f"Limited intelligence gathered on {dossier.persona_name}."

    def _extract_key_facts(self, fragments: List[IntelligenceFragment]) -> List[str]:
        """Extract key facts from fragments."""
        facts = []

        # Prioritize financial and legal fragments
        for f in fragments:
            if f.intel_type in [IntelligenceType.FINANCIAL, IntelligenceType.LEGAL]:
                facts.append(f.content[:150])

        return facts[:5]


# Standalone function for quick research
async def deep_research(
    persona_name: str,
    topic: str,
    persona_id: Optional[str] = None,
) -> IntelligenceDossier:
    """
    Quick function to run deep research on a persona.

    Usage:
        dossier = await deep_research("Ilya Sutskever", "OpenAI equity")
        print(dossier.to_prompt_injection())
    """
    engine = DeepSearchEngine()
    try:
        return await engine.research(
            persona_id=persona_id or persona_name.lower().replace(" ", "_"),
            persona_name=persona_name,
            topic=topic,
        )
    finally:
        await engine.close()
