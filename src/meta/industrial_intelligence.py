"""Industrial Intelligence Module for Real-time Causal Modeling.

Integrates industrial sector data with timeline events and GDELT signals
to model manufacturing, automation, and heavy industry dynamics.

Provides:
- Industrial sector tracking (manufacturing, logistics, energy, defense, etc.)
- Supply chain event modeling
- Automation/robotics company intelligence
- M&A signal detection for industrial targets
- Emotional quorum system for agent deliberation on events
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# INDUSTRIAL SECTOR ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class IndustrialSector(str, Enum):
    """Major industrial sectors for tracking."""
    MANUFACTURING_AUTOMATION = "manufacturing_automation"
    WAREHOUSE_LOGISTICS = "warehouse_logistics"
    AUTONOMOUS_VEHICLES = "autonomous_vehicles"
    ENERGY_UTILITIES = "energy_utilities"
    CONSTRUCTION_INFRASTRUCTURE = "construction_infrastructure"
    AGRICULTURE_FOOD = "agriculture_food"
    MINING_HEAVY_INDUSTRY = "mining_heavy_industry"
    SEMICONDUCTOR_CHIPS = "semiconductor_chips"
    DEFENSE_AEROSPACE = "defense_aerospace"
    HEALTHCARE_BIOTECH = "healthcare_biotech"
    AI_FOUNDATION = "ai_foundation"
    AI_INFRASTRUCTURE = "ai_infrastructure"
    AI_ENTERPRISE = "ai_enterprise"
    AI_CODING = "ai_coding"
    AI_ROBOTICS = "ai_robotics"


class IndustrialEventType(str, Enum):
    """Types of industrial events to track."""
    # Corporate Events
    ACQUISITION_ANNOUNCED = "acquisition_announced"
    ACQUISITION_COMPLETED = "acquisition_completed"
    ACQUISITION_BLOCKED = "acquisition_blocked"
    MERGER_ANNOUNCED = "merger_announced"
    IPO_FILED = "ipo_filed"
    IPO_COMPLETED = "ipo_completed"
    IPO_WITHDRAWN = "ipo_withdrawn"
    FUNDING_ROUND = "funding_round"
    VALUATION_CHANGE = "valuation_change"
    BANKRUPTCY = "bankruptcy"
    SHUTDOWN = "shutdown"
    LAYOFFS = "layoffs"

    # Operational Events
    FACTORY_OPENED = "factory_opened"
    FACTORY_CLOSED = "factory_closed"
    PRODUCTION_STARTED = "production_started"
    PRODUCTION_HALTED = "production_halted"
    SUPPLY_CHAIN_DISRUPTION = "supply_chain_disruption"
    CAPACITY_EXPANSION = "capacity_expansion"

    # Technology Events
    PRODUCT_LAUNCH = "product_launch"
    TECHNOLOGY_BREAKTHROUGH = "technology_breakthrough"
    PATENT_FILED = "patent_filed"
    PARTNERSHIP_ANNOUNCED = "partnership_announced"
    REGULATORY_APPROVAL = "regulatory_approval"
    REGULATORY_BLOCK = "regulatory_block"

    # Market Events
    CONTRACT_WON = "contract_won"
    CONTRACT_LOST = "contract_lost"
    MARKET_SHARE_CHANGE = "market_share_change"
    PRICE_CHANGE = "price_change"

    # Leadership Events
    CEO_CHANGE = "ceo_change"
    KEY_HIRE = "key_hire"
    KEY_DEPARTURE = "key_departure"
    FOUNDER_EXIT = "founder_exit"


class AcquisitionSignal(str, Enum):
    """Signals indicating acquisition likelihood."""
    GROWTH_STALLING = "growth_stalling"
    COMPETITIVE_PRESSURE = "competitive_pressure"
    INBOUND_INTEREST = "inbound_interest"
    STRATEGIC_REVIEW = "strategic_review"
    FOUNDER_FATIGUE = "founder_fatigue"
    KEY_TALENT_LEAVING = "key_talent_leaving"
    CASH_RUNWAY_LOW = "cash_runway_low"
    MARKET_PEAK = "market_peak"
    REGULATORY_TAILWIND = "regulatory_tailwind"
    STRATEGIC_FIT = "strategic_fit"
    BIDDING_WAR = "bidding_war"


# ═══════════════════════════════════════════════════════════════════════════════
# INDUSTRIAL ENTITY DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IndustrialCompany:
    """Profile of an industrial company being tracked."""
    company_id: str
    name: str
    sector: IndustrialSector
    subsector: str = ""
    headquarters: str = ""
    founded: int = 0

    # Financial metrics
    valuation_billions: float = 0.0
    last_valuation_date: Optional[datetime] = None
    arr_millions: float = 0.0
    revenue_millions: float = 0.0
    funding_total_millions: float = 0.0
    last_funding_round: str = ""
    last_funding_date: Optional[datetime] = None

    # Status
    is_public: bool = False
    ticker: Optional[str] = None
    is_acquired: bool = False
    acquirer: Optional[str] = None
    acquisition_price_billions: float = 0.0
    is_shutdown: bool = False

    # Leadership
    ceo: str = ""
    founders: List[str] = field(default_factory=list)
    key_people: Dict[str, str] = field(default_factory=dict)  # name -> role

    # Investors
    lead_investors: List[str] = field(default_factory=list)
    all_investors: List[str] = field(default_factory=list)

    # Metrics
    employees: int = 0
    market_share_pct: float = 0.0
    growth_rate_pct: float = 0.0

    # Intelligence
    acquisition_signals: Set[AcquisitionSignal] = field(default_factory=set)
    acquisition_probability: float = 0.0
    likely_acquirers: List[str] = field(default_factory=list)
    strategic_value_score: float = 0.0

    # Relationships
    partners: List[str] = field(default_factory=list)
    competitors: List[str] = field(default_factory=list)
    customers: List[str] = field(default_factory=list)
    suppliers: List[str] = field(default_factory=list)

    # Metadata
    tags: Set[str] = field(default_factory=set)
    notes: str = ""
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "company_id": self.company_id,
            "name": self.name,
            "sector": self.sector.value,
            "subsector": self.subsector,
            "headquarters": self.headquarters,
            "founded": self.founded,
            "valuation_billions": self.valuation_billions,
            "last_valuation_date": self.last_valuation_date.isoformat() if self.last_valuation_date else None,
            "arr_millions": self.arr_millions,
            "revenue_millions": self.revenue_millions,
            "funding_total_millions": self.funding_total_millions,
            "last_funding_round": self.last_funding_round,
            "is_public": self.is_public,
            "ticker": self.ticker,
            "is_acquired": self.is_acquired,
            "acquirer": self.acquirer,
            "acquisition_price_billions": self.acquisition_price_billions,
            "is_shutdown": self.is_shutdown,
            "ceo": self.ceo,
            "founders": self.founders,
            "key_people": self.key_people,
            "lead_investors": self.lead_investors,
            "employees": self.employees,
            "market_share_pct": self.market_share_pct,
            "acquisition_signals": [s.value for s in self.acquisition_signals],
            "acquisition_probability": self.acquisition_probability,
            "likely_acquirers": self.likely_acquirers,
            "strategic_value_score": self.strategic_value_score,
            "tags": list(self.tags),
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class IndustrialEvent:
    """A significant event in the industrial landscape."""
    event_id: str
    event_type: IndustrialEventType
    timestamp: datetime

    # Entities involved
    primary_company: str
    secondary_company: Optional[str] = None
    sector: Optional[IndustrialSector] = None

    # Event details
    title: str = ""
    description: str = ""
    value_billions: float = 0.0

    # Source
    source: str = ""
    source_url: str = ""
    gdelt_event_id: Optional[str] = None

    # Impact assessment
    market_impact: str = ""  # low, medium, high, critical
    acquisition_signal: Optional[AcquisitionSignal] = None

    # Metadata
    metadata: dict = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "primary_company": self.primary_company,
            "secondary_company": self.secondary_company,
            "sector": self.sector.value if self.sector else None,
            "title": self.title,
            "description": self.description,
            "value_billions": self.value_billions,
            "source": self.source,
            "market_impact": self.market_impact,
            "acquisition_signal": self.acquisition_signal.value if self.acquisition_signal else None,
            "metadata": self.metadata,
            "tags": list(self.tags),
        }


@dataclass
class SupplyChainNode:
    """A node in the global supply chain graph."""
    node_id: str
    name: str
    node_type: str  # supplier, manufacturer, distributor, customer
    location: str = ""
    sector: Optional[IndustrialSector] = None

    # Connections
    suppliers: List[str] = field(default_factory=list)
    customers: List[str] = field(default_factory=list)

    # Metrics
    criticality_score: float = 0.0  # How critical to the supply chain
    redundancy_score: float = 0.0   # How replaceable
    concentration_risk: float = 0.0  # Single points of failure

    # Status
    operational_status: str = "normal"  # normal, disrupted, offline
    disruption_history: List[dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# EMOTIONAL QUORUM SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class EmotionalStance(str, Enum):
    """Emotional/analytical stances agents can take on events."""
    BULLISH = "bullish"           # Positive outlook, opportunity
    BEARISH = "bearish"           # Negative outlook, risk
    SKEPTICAL = "skeptical"       # Questioning, needs verification
    ALARMED = "alarmed"           # Urgent concern
    EXCITED = "excited"           # High opportunity signal
    CAUTIOUS = "cautious"         # Proceed carefully
    NEUTRAL = "neutral"           # No strong signal
    CONTRARIAN = "contrarian"     # Opposite of consensus
    OPPORTUNISTIC = "opportunistic"  # Looking for angles
    ANALYTICAL = "analytical"     # Pure data focus


@dataclass
class AgentOpinion:
    """An agent's opinion on an event."""
    agent_id: str
    agent_role: str  # analyst, strategist, risk_manager, etc.
    stance: EmotionalStance
    confidence: float  # 0.0 to 1.0
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Predictions
    predicted_outcome: str = ""
    time_horizon: str = ""  # immediate, short-term, long-term

    # Action recommendations
    recommended_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "agent_role": self.agent_role,
            "stance": self.stance.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
            "predicted_outcome": self.predicted_outcome,
            "time_horizon": self.time_horizon,
            "recommended_actions": self.recommended_actions,
        }


@dataclass
class QuorumDeliberation:
    """Result of agents deliberating on an event."""
    deliberation_id: str
    event: IndustrialEvent
    timestamp: datetime

    # Opinions collected
    opinions: List[AgentOpinion] = field(default_factory=list)

    # Consensus metrics
    consensus_stance: Optional[EmotionalStance] = None
    consensus_strength: float = 0.0  # 0.0 = no consensus, 1.0 = unanimous
    dissent_level: float = 0.0

    # Aggregated analysis
    key_insights: List[str] = field(default_factory=list)
    risk_assessment: str = ""
    opportunity_assessment: str = ""

    # Recommended response
    urgency_level: str = "normal"  # low, normal, elevated, high, critical
    recommended_actions: List[str] = field(default_factory=list)
    watch_items: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "deliberation_id": self.deliberation_id,
            "event": self.event.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "opinions": [o.to_dict() for o in self.opinions],
            "consensus_stance": self.consensus_stance.value if self.consensus_stance else None,
            "consensus_strength": self.consensus_strength,
            "dissent_level": self.dissent_level,
            "key_insights": self.key_insights,
            "risk_assessment": self.risk_assessment,
            "opportunity_assessment": self.opportunity_assessment,
            "urgency_level": self.urgency_level,
            "recommended_actions": self.recommended_actions,
            "watch_items": self.watch_items,
        }


class EmotionalQuorum:
    """System for agents to deliberate on events and reach consensus."""

    AGENT_ROLES = [
        ("market_analyst", "Analyzes market dynamics and competitive positioning"),
        ("risk_manager", "Identifies and assesses risks"),
        ("opportunity_scout", "Finds acquisition and investment opportunities"),
        ("sector_specialist", "Deep domain expertise in specific sectors"),
        ("macro_strategist", "Big picture economic and geopolitical view"),
        ("contrarian", "Challenges consensus, finds alternative views"),
        ("quantitative", "Data-driven numerical analysis"),
        ("behavioral", "Human factors, leadership, culture assessment"),
    ]

    def __init__(self):
        self._deliberations: List[QuorumDeliberation] = []
        self._lock = asyncio.Lock()

    async def deliberate(
        self,
        event: IndustrialEvent,
        context: Optional[dict] = None,
    ) -> QuorumDeliberation:
        """Have agents deliberate on an event and reach consensus."""
        deliberation = QuorumDeliberation(
            deliberation_id=str(uuid.uuid4()),
            event=event,
            timestamp=datetime.utcnow(),
        )

        # Simulate agent opinions based on event type and context
        opinions = await self._gather_opinions(event, context or {})
        deliberation.opinions = opinions

        # Calculate consensus
        if opinions:
            stance_counts = defaultdict(int)
            total_confidence = 0.0

            for opinion in opinions:
                stance_counts[opinion.stance] += 1
                total_confidence += opinion.confidence

            # Find most common stance
            max_stance = max(stance_counts.items(), key=lambda x: x[1])
            deliberation.consensus_stance = max_stance[0]
            deliberation.consensus_strength = max_stance[1] / len(opinions)

            # Calculate dissent
            dissenting = len(opinions) - max_stance[1]
            deliberation.dissent_level = dissenting / len(opinions) if opinions else 0

            # Aggregate insights
            deliberation.key_insights = self._aggregate_insights(opinions)
            deliberation.risk_assessment = self._assess_risks(opinions)
            deliberation.opportunity_assessment = self._assess_opportunities(opinions)

            # Determine urgency
            deliberation.urgency_level = self._determine_urgency(event, opinions)

            # Compile recommended actions
            all_actions = []
            for opinion in opinions:
                all_actions.extend(opinion.recommended_actions)
            # Dedupe while preserving order
            seen = set()
            deliberation.recommended_actions = [
                a for a in all_actions
                if not (a in seen or seen.add(a))
            ][:5]  # Top 5 actions

        async with self._lock:
            self._deliberations.append(deliberation)
            # Keep last 1000 deliberations
            if len(self._deliberations) > 1000:
                self._deliberations = self._deliberations[-1000:]

        return deliberation

    async def _gather_opinions(
        self,
        event: IndustrialEvent,
        context: dict,
    ) -> List[AgentOpinion]:
        """Simulate gathering opinions from different agent perspectives."""
        opinions = []

        for role, description in self.AGENT_ROLES:
            opinion = self._simulate_agent_opinion(role, description, event, context)
            opinions.append(opinion)

        return opinions

    def _simulate_agent_opinion(
        self,
        role: str,
        description: str,
        event: IndustrialEvent,
        context: dict,
    ) -> AgentOpinion:
        """Simulate an agent's opinion based on their role and the event."""
        # Different roles have different default stances
        role_tendencies = {
            "market_analyst": (EmotionalStance.ANALYTICAL, 0.75),
            "risk_manager": (EmotionalStance.CAUTIOUS, 0.80),
            "opportunity_scout": (EmotionalStance.OPPORTUNISTIC, 0.70),
            "sector_specialist": (EmotionalStance.ANALYTICAL, 0.85),
            "macro_strategist": (EmotionalStance.NEUTRAL, 0.65),
            "contrarian": (EmotionalStance.CONTRARIAN, 0.60),
            "quantitative": (EmotionalStance.ANALYTICAL, 0.90),
            "behavioral": (EmotionalStance.CAUTIOUS, 0.70),
        }

        default_stance, base_confidence = role_tendencies.get(
            role, (EmotionalStance.NEUTRAL, 0.5)
        )

        # Adjust based on event type
        stance = default_stance
        confidence = base_confidence
        reasoning = ""
        actions = []

        if event.event_type == IndustrialEventType.ACQUISITION_ANNOUNCED:
            if role == "opportunity_scout":
                stance = EmotionalStance.EXCITED
                reasoning = f"Major M&A activity in {event.sector.value if event.sector else 'sector'} - signals active acquirer interest"
                actions = ["Monitor for follow-on deals", "Identify similar targets"]
            elif role == "risk_manager":
                stance = EmotionalStance.CAUTIOUS
                reasoning = "Acquisition could trigger regulatory scrutiny or market consolidation"
                actions = ["Assess regulatory risk", "Review competitive impact"]
            elif role == "contrarian":
                stance = EmotionalStance.SKEPTICAL
                reasoning = "High valuations may indicate market peak - consider timing carefully"
                actions = ["Wait for deal completion", "Monitor integration progress"]

        elif event.event_type == IndustrialEventType.FUNDING_ROUND:
            if role == "market_analyst":
                stance = EmotionalStance.BULLISH if event.value_billions > 0.5 else EmotionalStance.NEUTRAL
                reasoning = f"${event.value_billions}B raise indicates strong investor confidence"
                actions = ["Track valuation trajectory", "Monitor burn rate"]
            elif role == "opportunity_scout":
                stance = EmotionalStance.OPPORTUNISTIC
                reasoning = "Fresh capital may accelerate growth or make company acquisition target"
                actions = ["Model exit scenarios", "Identify strategic buyers"]

        elif event.event_type in [IndustrialEventType.SHUTDOWN, IndustrialEventType.BANKRUPTCY]:
            if role == "risk_manager":
                stance = EmotionalStance.ALARMED
                reasoning = "Market failure signals - review exposure and similar companies"
                actions = ["Conduct portfolio review", "Assess sector contagion risk"]
            elif role == "opportunity_scout":
                stance = EmotionalStance.OPPORTUNISTIC
                reasoning = "Distressed assets may be available at discount"
                actions = ["Identify valuable IP/talent", "Monitor asset sales"]

        elif event.event_type == IndustrialEventType.LAYOFFS:
            if role == "behavioral":
                stance = EmotionalStance.CAUTIOUS
                reasoning = "Talent disruption may impact execution and morale"
                actions = ["Track key personnel moves", "Assess cultural impact"]
            elif role == "opportunity_scout":
                stance = EmotionalStance.OPPORTUNISTIC
                reasoning = "Talent pool now available for hiring"
                actions = ["Identify key talent", "Monitor for further restructuring"]

        elif event.event_type == IndustrialEventType.TECHNOLOGY_BREAKTHROUGH:
            if role == "sector_specialist":
                stance = EmotionalStance.EXCITED
                confidence = 0.90
                reasoning = "Technical advancement could reshape competitive dynamics"
                actions = ["Assess patent implications", "Model adoption curve"]
            elif role == "contrarian":
                stance = EmotionalStance.SKEPTICAL
                reasoning = "Lab results often don't translate to commercial success"
                actions = ["Wait for production validation", "Review prior claims"]

        # Default reasoning if not set
        if not reasoning:
            reasoning = f"{role} analysis of {event.event_type.value} event for {event.primary_company}"

        return AgentOpinion(
            agent_id=f"{role}_agent",
            agent_role=role,
            stance=stance,
            confidence=confidence,
            reasoning=reasoning,
            recommended_actions=actions,
            time_horizon="short-term" if event.market_impact == "high" else "medium-term",
        )

    def _aggregate_insights(self, opinions: List[AgentOpinion]) -> List[str]:
        """Aggregate key insights from all opinions."""
        insights = []

        # Group by stance
        bullish = [o for o in opinions if o.stance in [EmotionalStance.BULLISH, EmotionalStance.EXCITED]]
        bearish = [o for o in opinions if o.stance in [EmotionalStance.BEARISH, EmotionalStance.ALARMED]]
        cautious = [o for o in opinions if o.stance == EmotionalStance.CAUTIOUS]

        if len(bullish) > len(opinions) / 2:
            insights.append(f"Majority bullish ({len(bullish)}/{len(opinions)} agents)")
        elif len(bearish) > len(opinions) / 2:
            insights.append(f"Majority bearish ({len(bearish)}/{len(opinions)} agents)")
        elif len(cautious) >= len(opinions) / 3:
            insights.append("Significant caution among analysts")

        # High confidence signals
        high_conf = [o for o in opinions if o.confidence > 0.8]
        if high_conf:
            insights.append(f"{len(high_conf)} agents have high confidence assessments")

        return insights

    def _assess_risks(self, opinions: List[AgentOpinion]) -> str:
        """Aggregate risk assessment."""
        risk_opinions = [
            o for o in opinions
            if o.stance in [EmotionalStance.ALARMED, EmotionalStance.CAUTIOUS, EmotionalStance.BEARISH]
        ]

        if not risk_opinions:
            return "Low risk signals detected"

        risk_level = len(risk_opinions) / len(opinions)
        if risk_level > 0.6:
            return "High risk - majority of agents expressing concern"
        elif risk_level > 0.3:
            return "Moderate risk - some agents expressing caution"
        else:
            return "Low risk - minority concerns noted"

    def _assess_opportunities(self, opinions: List[AgentOpinion]) -> str:
        """Aggregate opportunity assessment."""
        opp_opinions = [
            o for o in opinions
            if o.stance in [EmotionalStance.BULLISH, EmotionalStance.EXCITED, EmotionalStance.OPPORTUNISTIC]
        ]

        if not opp_opinions:
            return "Limited opportunity signals"

        opp_level = len(opp_opinions) / len(opinions)
        if opp_level > 0.5:
            return "Strong opportunity - multiple agents see potential"
        elif opp_level > 0.25:
            return "Moderate opportunity - some agents optimistic"
        else:
            return "Limited opportunity signals"

    def _determine_urgency(
        self,
        event: IndustrialEvent,
        opinions: List[AgentOpinion],
    ) -> str:
        """Determine urgency level based on event and opinions."""
        # High urgency event types
        high_urgency_types = [
            IndustrialEventType.ACQUISITION_ANNOUNCED,
            IndustrialEventType.BANKRUPTCY,
            IndustrialEventType.SHUTDOWN,
            IndustrialEventType.REGULATORY_BLOCK,
        ]

        if event.event_type in high_urgency_types:
            return "high"

        # Check for alarmed agents
        alarmed = [o for o in opinions if o.stance == EmotionalStance.ALARMED]
        if len(alarmed) >= 2:
            return "elevated"

        # High value events
        if event.value_billions > 5.0:
            return "elevated"

        # Market impact
        if event.market_impact == "critical":
            return "critical"
        elif event.market_impact == "high":
            return "elevated"

        return "normal"

    async def get_recent_deliberations(
        self,
        limit: int = 10,
        sector: Optional[IndustrialSector] = None,
    ) -> List[QuorumDeliberation]:
        """Get recent deliberations, optionally filtered by sector."""
        async with self._lock:
            results = self._deliberations.copy()

        if sector:
            results = [d for d in results if d.event.sector == sector]

        return sorted(results, key=lambda d: d.timestamp, reverse=True)[:limit]


# ═══════════════════════════════════════════════════════════════════════════════
# INDUSTRIAL INTELLIGENCE REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

class IndustrialRegistry:
    """Central registry for industrial companies and events."""

    def __init__(self):
        self._companies: Dict[str, IndustrialCompany] = {}
        self._events: List[IndustrialEvent] = []
        self._supply_chain: Dict[str, SupplyChainNode] = {}
        self._quorum = EmotionalQuorum()
        self._lock = asyncio.Lock()

        # Initialize with known companies from research
        self._initialize_companies()

    def _initialize_companies(self):
        """Initialize with known industrial companies."""
        companies_data = [
            # AI Foundation Models
            ("openai", "OpenAI", IndustrialSector.AI_FOUNDATION, 300.0, "SoftBank", False, None, "Sam Altman"),
            ("anthropic", "Anthropic", IndustrialSector.AI_FOUNDATION, 183.0, "Nvidia", False, None, "Dario Amodei"),
            ("xai", "xAI", IndustrialSector.AI_FOUNDATION, 50.0, "Various", False, None, "Elon Musk"),
            ("safe_superintelligence", "Safe Superintelligence", IndustrialSector.AI_FOUNDATION, 32.0, "Various", False, None, "Ilya Sutskever"),
            ("mistral", "Mistral AI", IndustrialSector.AI_FOUNDATION, 6.0, "a16z", False, None, "Arthur Mensch"),
            ("cohere", "Cohere", IndustrialSector.AI_FOUNDATION, 7.0, "Oracle", False, None, "Aidan Gomez"),

            # AI Infrastructure
            ("coreweave", "CoreWeave", IndustrialSector.AI_INFRASTRUCTURE, 15.0, "Public", True, "CRWV", "Michael Intrator"),
            ("lambda_labs", "Lambda Labs", IndustrialSector.AI_INFRASTRUCTURE, 12.5, "Various", False, None, "Stephen Balaban"),
            ("together_ai", "Together AI", IndustrialSector.AI_INFRASTRUCTURE, 3.3, "Various", False, None, "Vipul Prakash"),

            # AI Chips (some acquired)
            ("cerebras", "Cerebras", IndustrialSector.SEMICONDUCTOR_CHIPS, 15.0, "Various", False, None, "Andrew Feldman"),
            ("groq", "Groq", IndustrialSector.SEMICONDUCTOR_CHIPS, 20.0, "Nvidia", True, None, "Jonathan Ross"),
            ("tenstorrent", "Tenstorrent", IndustrialSector.SEMICONDUCTOR_CHIPS, 3.2, "Various", False, None, "Jim Keller"),

            # AI Enterprise
            ("cursor", "Cursor", IndustrialSector.AI_CODING, 29.3, "a16z", False, None, "Michael Truell"),
            ("cognition", "Cognition (Devin)", IndustrialSector.AI_CODING, 10.2, "Founders Fund", False, None, "Scott Wu"),
            ("harvey", "Harvey", IndustrialSector.AI_ENTERPRISE, 8.0, "Sequoia", False, None, "Winston Weinberg"),
            ("glean", "Glean", IndustrialSector.AI_ENTERPRISE, 7.2, "Sequoia", False, None, "Arvind Jain"),

            # AI Robotics
            ("figure_ai", "Figure AI", IndustrialSector.AI_ROBOTICS, 39.0, "Various", False, None, "Brett Adcock"),
            ("physical_intelligence", "Physical Intelligence", IndustrialSector.AI_ROBOTICS, 5.6, "Various", False, None, "Karol Hausman"),
            ("agility_robotics", "Agility Robotics", IndustrialSector.AI_ROBOTICS, 2.1, "Amazon", False, None, "Damion Shelton"),

            # Autonomous Vehicles
            ("waymo", "Waymo", IndustrialSector.AUTONOMOUS_VEHICLES, 100.0, "Alphabet", False, None, "Tekedra Mawakana"),
            ("aurora", "Aurora Innovation", IndustrialSector.AUTONOMOUS_VEHICLES, 5.0, "Public", True, "AUR", "Chris Urmson"),

            # Manufacturing/Industrial
            ("symbotic", "Symbotic", IndustrialSector.WAREHOUSE_LOGISTICS, 8.0, "Public", True, "SYM", "Rick Cohen"),
            ("locus_robotics", "Locus Robotics", IndustrialSector.WAREHOUSE_LOGISTICS, 2.0, "Tiger Global", False, None, "Rick Faulk"),

            # Defense
            ("anduril", "Anduril Industries", IndustrialSector.DEFENSE_AEROSPACE, 30.5, "a16z", False, None, "Palmer Luckey"),
            ("shield_ai", "Shield AI", IndustrialSector.DEFENSE_AEROSPACE, 5.3, "Various", False, None, "Ryan Tseng"),

            # Healthcare/Biotech
            ("xaira", "Xaira Therapeutics", IndustrialSector.HEALTHCARE_BIOTECH, 1.0, "Various", False, None, "Marc Tessier-Lavigne"),

            # Energy
            ("x_energy", "X-energy", IndustrialSector.ENERGY_UTILITIES, 2.5, "Various", False, None, "J. Clay Sell"),
        ]

        for data in companies_data:
            company_id, name, sector, valuation, lead_investor, is_public, ticker, ceo = data
            company = IndustrialCompany(
                company_id=company_id,
                name=name,
                sector=sector,
                valuation_billions=valuation,
                lead_investors=[lead_investor] if lead_investor else [],
                is_public=is_public,
                ticker=ticker,
                ceo=ceo,
            )
            self._companies[company_id] = company

    async def add_company(self, company: IndustrialCompany):
        """Add or update a company."""
        async with self._lock:
            self._companies[company.company_id] = company

    async def get_company(self, company_id: str) -> Optional[IndustrialCompany]:
        """Get a company by ID."""
        return self._companies.get(company_id)

    async def search_companies(
        self,
        sector: Optional[IndustrialSector] = None,
        min_valuation: float = 0.0,
        max_valuation: float = float('inf'),
        acquired_only: bool = False,
        active_only: bool = False,
    ) -> List[IndustrialCompany]:
        """Search companies with filters."""
        results = list(self._companies.values())

        if sector:
            results = [c for c in results if c.sector == sector]

        results = [
            c for c in results
            if min_valuation <= c.valuation_billions <= max_valuation
        ]

        if acquired_only:
            results = [c for c in results if c.is_acquired]

        if active_only:
            results = [c for c in results if not c.is_acquired and not c.is_shutdown]

        return sorted(results, key=lambda c: c.valuation_billions, reverse=True)

    async def record_event(
        self,
        event_type: IndustrialEventType,
        primary_company: str,
        title: str = "",
        description: str = "",
        value_billions: float = 0.0,
        secondary_company: Optional[str] = None,
        sector: Optional[IndustrialSector] = None,
        source: str = "",
        market_impact: str = "medium",
        deliberate: bool = True,
    ) -> Tuple[IndustrialEvent, Optional[QuorumDeliberation]]:
        """Record an industrial event and optionally trigger deliberation."""
        event = IndustrialEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.utcnow(),
            primary_company=primary_company,
            secondary_company=secondary_company,
            sector=sector,
            title=title,
            description=description,
            value_billions=value_billions,
            source=source,
            market_impact=market_impact,
        )

        async with self._lock:
            self._events.append(event)
            # Keep last 10000 events
            if len(self._events) > 10000:
                self._events = self._events[-10000:]

        # Trigger emotional quorum deliberation
        deliberation = None
        if deliberate:
            deliberation = await self._quorum.deliberate(event)

        logger.info(f"Recorded industrial event: {event_type.value} - {title}")

        return event, deliberation

    async def get_recent_events(
        self,
        limit: int = 50,
        sector: Optional[IndustrialSector] = None,
        event_type: Optional[IndustrialEventType] = None,
    ) -> List[IndustrialEvent]:
        """Get recent events with optional filters."""
        events = self._events.copy()

        if sector:
            events = [e for e in events if e.sector == sector]

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return sorted(events, key=lambda e: e.timestamp, reverse=True)[:limit]

    async def get_acquisition_targets(
        self,
        sector: Optional[IndustrialSector] = None,
        min_probability: float = 0.3,
    ) -> List[IndustrialCompany]:
        """Get companies likely to be acquired."""
        companies = await self.search_companies(sector=sector, active_only=True)

        targets = [
            c for c in companies
            if c.acquisition_probability >= min_probability or len(c.acquisition_signals) >= 2
        ]

        return sorted(targets, key=lambda c: c.acquisition_probability, reverse=True)

    @property
    def quorum(self) -> EmotionalQuorum:
        """Access the emotional quorum system."""
        return self._quorum

    async def get_sector_summary(self, sector: IndustrialSector) -> dict:
        """Get a summary of a sector."""
        companies = await self.search_companies(sector=sector)
        events = await self.get_recent_events(sector=sector, limit=100)

        total_valuation = sum(c.valuation_billions for c in companies)
        public_count = sum(1 for c in companies if c.is_public)
        acquired_count = sum(1 for c in companies if c.is_acquired)

        return {
            "sector": sector.value,
            "company_count": len(companies),
            "total_valuation_billions": total_valuation,
            "public_companies": public_count,
            "acquired_companies": acquired_count,
            "recent_event_count": len(events),
            "top_companies": [
                {"name": c.name, "valuation": c.valuation_billions}
                for c in companies[:5]
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# GDELT INTEGRATION FOR INDUSTRIAL EVENTS
# ═══════════════════════════════════════════════════════════════════════════════

class GDELTIndustrialProcessor:
    """Process GDELT events for industrial intelligence."""

    # CAMEO codes relevant to industrial/business events
    INDUSTRIAL_CAMEO_CODES = {
        # Economic cooperation
        "061": IndustrialEventType.PARTNERSHIP_ANNOUNCED,
        "0612": IndustrialEventType.CONTRACT_WON,
        # Material cooperation
        "071": IndustrialEventType.PARTNERSHIP_ANNOUNCED,
        "0712": IndustrialEventType.SUPPLY_CHAIN_DISRUPTION,
        # Yield
        "081": IndustrialEventType.ACQUISITION_ANNOUNCED,
        # Investigate
        "091": IndustrialEventType.REGULATORY_BLOCK,
        # Demand
        "10": IndustrialEventType.REGULATORY_BLOCK,
        # Disapprove
        "11": IndustrialEventType.REGULATORY_BLOCK,
        # Reduce relations
        "15": IndustrialEventType.PARTNERSHIP_ANNOUNCED,  # ending partnership
        "16": IndustrialEventType.SUPPLY_CHAIN_DISRUPTION,
    }

    # Keywords for industrial event detection
    INDUSTRIAL_KEYWORDS = {
        "acquisition": IndustrialEventType.ACQUISITION_ANNOUNCED,
        "acquired": IndustrialEventType.ACQUISITION_COMPLETED,
        "merger": IndustrialEventType.MERGER_ANNOUNCED,
        "ipo": IndustrialEventType.IPO_FILED,
        "goes public": IndustrialEventType.IPO_COMPLETED,
        "funding": IndustrialEventType.FUNDING_ROUND,
        "raises": IndustrialEventType.FUNDING_ROUND,
        "valuation": IndustrialEventType.VALUATION_CHANGE,
        "layoffs": IndustrialEventType.LAYOFFS,
        "bankruptcy": IndustrialEventType.BANKRUPTCY,
        "shutdown": IndustrialEventType.SHUTDOWN,
        "factory": IndustrialEventType.FACTORY_OPENED,
        "production": IndustrialEventType.PRODUCTION_STARTED,
        "supply chain": IndustrialEventType.SUPPLY_CHAIN_DISRUPTION,
        "partnership": IndustrialEventType.PARTNERSHIP_ANNOUNCED,
        "patent": IndustrialEventType.PATENT_FILED,
        "fda approval": IndustrialEventType.REGULATORY_APPROVAL,
        "regulatory": IndustrialEventType.REGULATORY_BLOCK,
        "ceo": IndustrialEventType.CEO_CHANGE,
        "appointed": IndustrialEventType.KEY_HIRE,
        "departs": IndustrialEventType.KEY_DEPARTURE,
    }

    def __init__(self, registry: IndustrialRegistry):
        self.registry = registry
        self._processed_count = 0

    async def process_gdelt_event(
        self,
        gdelt_event: dict,
    ) -> Optional[IndustrialEvent]:
        """Process a GDELT event for industrial relevance."""
        # Check if event is business-related
        actor1_type = gdelt_event.get("Actor1Type1Code", "")
        actor2_type = gdelt_event.get("Actor2Type1Code", "")

        # Business actors (BUS, MNC, etc.)
        business_types = {"BUS", "MNC", "CVL", "IGO"}
        is_business = (
            actor1_type in business_types or
            actor2_type in business_types or
            "BUS" in str(gdelt_event.get("Actor1Type2Code", "")) or
            "BUS" in str(gdelt_event.get("Actor2Type2Code", ""))
        )

        if not is_business:
            return None

        # Extract event details
        event_code = gdelt_event.get("EventCode", "")
        source_url = gdelt_event.get("SOURCEURL", "")

        # Determine event type from CAMEO code or keywords
        event_type = None
        for code, etype in self.INDUSTRIAL_CAMEO_CODES.items():
            if event_code.startswith(code):
                event_type = etype
                break

        if not event_type:
            # Try keyword matching from source URL or actor names
            source_lower = source_url.lower()
            for keyword, etype in self.INDUSTRIAL_KEYWORDS.items():
                if keyword in source_lower:
                    event_type = etype
                    break

        if not event_type:
            return None

        # Create industrial event
        actor1 = gdelt_event.get("Actor1Name", "Unknown")
        actor2 = gdelt_event.get("Actor2Name", "")

        goldstein = gdelt_event.get("GoldsteinScale", 0)
        impact = "low"
        if abs(goldstein) > 5:
            impact = "high"
        elif abs(goldstein) > 2:
            impact = "medium"

        event, _ = await self.registry.record_event(
            event_type=event_type,
            primary_company=actor1,
            secondary_company=actor2 if actor2 else None,
            title=f"{event_type.value}: {actor1}" + (f" and {actor2}" if actor2 else ""),
            source=source_url,
            market_impact=impact,
            deliberate=abs(goldstein) > 3,  # Only deliberate on significant events
        )

        self._processed_count += 1
        return event

    async def process_batch(
        self,
        gdelt_events: List[dict],
    ) -> List[IndustrialEvent]:
        """Process a batch of GDELT events."""
        results = []

        for gdelt_event in gdelt_events:
            try:
                event = await self.process_gdelt_event(gdelt_event)
                if event:
                    results.append(event)
            except Exception as e:
                logger.error(f"Error processing GDELT event: {e}")

        logger.info(f"Processed {len(results)} industrial events from {len(gdelt_events)} GDELT events")
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_industrial_registry: Optional[IndustrialRegistry] = None


def get_industrial_registry() -> IndustrialRegistry:
    """Get or create the global industrial registry."""
    global _industrial_registry
    if _industrial_registry is None:
        _industrial_registry = IndustrialRegistry()
    return _industrial_registry


async def record_industrial_event(
    event_type: IndustrialEventType,
    primary_company: str,
    **kwargs,
) -> Tuple[IndustrialEvent, Optional[QuorumDeliberation]]:
    """Convenience function to record an industrial event."""
    registry = get_industrial_registry()
    return await registry.record_event(event_type, primary_company, **kwargs)
