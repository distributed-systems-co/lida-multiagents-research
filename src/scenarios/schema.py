"""
Scenario Configuration Schema

Defines the configuration knobs for multi-agent simulations:
- Knob 1: Persona (66 personas)
- Knob 2: Reasoning/Persuasion Strategy
- Knob 3: Initial World State
- Knob 4: Model Selection
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional, Literal
from pydantic import BaseModel, Field
import uuid


# =============================================================================
# Knob 1: Personas
# =============================================================================

class PersonaCategory(str, Enum):
    """Categories for the 66 personas."""
    TECH_LEADER = "tech_leader"
    POLITICIAN = "politician"
    RESEARCHER = "researcher"
    INVESTOR = "investor"
    REGULATOR = "regulator"
    JOURNALIST = "journalist"
    ACTIVIST = "activist"
    PHILOSOPHER = "philosopher"


class PersonaStance(str, Enum):
    """AI policy stance spectrum."""
    ACCELERATIONIST = "accelerationist"      # Build fast, worry later
    PRO_INDUSTRY = "pro_industry"            # Light regulation
    MODERATE = "moderate"                     # Balanced approach
    PRO_SAFETY = "pro_safety"                # Strong safety focus
    DOOMER = "doomer"                        # Existential risk focus
    PAUSE = "pause"                          # Stop/slow development


class PersonaConfig(BaseModel):
    """Configuration for a single persona."""
    id: str = Field(default_factory=lambda: f"persona-{uuid.uuid4().hex[:8]}")
    name: str                                 # e.g., "Elon Musk"
    category: PersonaCategory
    stance: PersonaStance

    # Persona details
    title: Optional[str] = None               # e.g., "CEO of Tesla/SpaceX/xAI"
    organization: Optional[str] = None
    background: Optional[str] = None          # Brief bio

    # Behavioral traits (0-1 scale)
    traits: dict[str, float] = Field(default_factory=lambda: {
        "assertiveness": 0.5,
        "openness": 0.5,
        "risk_tolerance": 0.5,
        "pragmatism": 0.5,
        "technical_depth": 0.5,
    })

    # Communication style
    communication_style: dict[str, Any] = Field(default_factory=lambda: {
        "formality": 0.5,          # 0=casual, 1=formal
        "verbosity": 0.5,          # 0=terse, 1=verbose
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
    })

    # Known positions on key issues
    positions: dict[str, str] = Field(default_factory=dict)

    # Key relationships (persona_id -> relationship type)
    relationships: dict[str, str] = Field(default_factory=dict)

    # System prompt override (if custom)
    system_prompt: Optional[str] = None


# =============================================================================
# Knob 2: Reasoning/Persuasion Strategy
# =============================================================================

class ReasoningStrategy(str, Enum):
    """High-level reasoning approaches."""
    LOGICAL = "logical"                # Formal arguments, evidence
    EMOTIONAL = "emotional"            # Appeal to values, fears, hopes
    SOCIAL_PROOF = "social_proof"      # What others are doing
    AUTHORITY = "authority"            # Expert credentials
    NARRATIVE = "narrative"            # Storytelling
    PRAGMATIC = "pragmatic"            # What works in practice
    CONTRARIAN = "contrarian"          # Challenge assumptions


class PersuasionTactic(str, Enum):
    """Specific persuasion tactics."""
    RECIPROCITY = "reciprocity"        # Give to get
    COMMITMENT = "commitment"          # Small yeses lead to big yes
    SCARCITY = "scarcity"              # Limited time/opportunity
    CONSENSUS = "consensus"            # Everyone agrees
    AUTHORITY_APPEAL = "authority"     # Trust the experts
    LIKING = "liking"                  # Build rapport first
    FEAR_APPEAL = "fear_appeal"        # Warn of consequences
    HOPE_APPEAL = "hope_appeal"        # Promise of better future


class StrategyConfig(BaseModel):
    """Configuration for reasoning/persuasion strategy (Knob 2)."""
    primary_strategy: ReasoningStrategy = ReasoningStrategy.LOGICAL
    secondary_strategy: Optional[ReasoningStrategy] = None

    # Enabled tactics
    tactics: list[PersuasionTactic] = Field(default_factory=list)

    # Strategy parameters
    aggression: float = Field(0.5, ge=0, le=1)      # How pushy
    flexibility: float = Field(0.5, ge=0, le=1)     # Willingness to compromise
    patience: int = Field(3, ge=1)                   # Rounds before escalation

    # DSPy signature for structured reasoning (optional)
    reasoning_signature: Optional[str] = None

    # Whether to adapt strategy based on opponent
    adaptive: bool = True


# =============================================================================
# Knob 3: World State
# =============================================================================

class TechLevel(str, Enum):
    """Technology advancement levels."""
    CURRENT = "current"                # 2024 capabilities
    NEAR_FUTURE = "near_future"        # 1-2 years out
    MEDIUM_FUTURE = "medium_future"    # 3-5 years out
    FAR_FUTURE = "far_future"          # 5+ years out
    CUSTOM = "custom"


class PolicyStatus(str, Enum):
    """Status of a policy/bill."""
    PROPOSED = "proposed"
    IN_COMMITTEE = "in_committee"
    PASSED_HOUSE = "passed_house"
    PASSED_SENATE = "passed_senate"
    PASSED = "passed"
    VETOED = "vetoed"
    NOT_PASSED = "not_passed"
    ENACTED = "enacted"


class PolicyConfig(BaseModel):
    """A policy or bill in the simulation."""
    id: str
    name: str                          # e.g., "SB-1047"
    jurisdiction: str                  # e.g., "CA", "NY", "US", "EU"
    status: PolicyStatus
    description: Optional[str] = None
    stance_required: Optional[PersonaStance] = None  # What stance supports it

    # Effects if enacted
    effects: dict[str, Any] = Field(default_factory=dict)


class TechnologyConfig(BaseModel):
    """A technology in the world state."""
    id: str
    name: str
    category: str                      # e.g., "language_model", "robotics"
    level: TechLevel

    # Current state
    deployed: bool = False
    adoption_rate: float = Field(0.0, ge=0, le=1)

    # Capabilities
    capabilities: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)

    # Prerequisites in tech tree
    prerequisites: list[str] = Field(default_factory=list)

    # Who controls/owns it
    controllers: list[str] = Field(default_factory=list)


class ResourcePool(BaseModel):
    """Resource tracking."""
    compute_flops: float = 1e24        # Available compute
    funding_usd: float = 1e11          # Available capital
    talent_pool: int = 10000           # Available researchers
    data_tokens: float = 1e13          # Training data


class WorldStateConfig(BaseModel):
    """Configuration for initial world state (Knob 3)."""
    id: str = Field(default_factory=lambda: f"world-{uuid.uuid4().hex[:8]}")
    name: str = "Real World 2024"
    description: Optional[str] = None

    # Base reality
    base: Literal["real_world", "custom"] = "real_world"
    reference_date: datetime = Field(default_factory=datetime.now)

    # Technology state
    tech_level: TechLevel = TechLevel.CURRENT
    technologies: list[TechnologyConfig] = Field(default_factory=list)

    # Policy landscape
    policies: list[PolicyConfig] = Field(default_factory=list)

    # Resources
    resources: ResourcePool = Field(default_factory=ResourcePool)

    # Geopolitical factors
    geopolitics: dict[str, Any] = Field(default_factory=lambda: {
        "us_china_relations": "tense",
        "eu_regulation_stance": "pro_regulation",
        "global_ai_race": True,
    })

    # Market conditions
    market: dict[str, Any] = Field(default_factory=lambda: {
        "ai_investment_hot": True,
        "public_sentiment": "mixed",
        "media_attention": "high",
    })

    # Custom state variables
    custom_state: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Knob 4: Model Selection
# =============================================================================

class ModelProvider(str, Enum):
    """LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    MOONSHOT = "moonshot"      # Kimi
    META = "meta"
    MISTRAL = "mistral"
    LOCAL = "local"


class ModelConfig(BaseModel):
    """Configuration for an LLM model."""
    provider: ModelProvider
    model_id: str                      # e.g., "claude-sonnet-4-20250514"

    # Model parameters
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0

    # Special capabilities
    capabilities: list[str] = Field(default_factory=lambda: [
        "reasoning", "roleplay", "analysis"
    ])

    # Cost tracking
    cost_per_1k_input: float = 0.003
    cost_per_1k_output: float = 0.015

    # Notes
    notes: Optional[str] = None        # e.g., "good at creative writing"


# Pre-defined model configs
MODELS = {
    "claude-sonnet": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-sonnet-4-20250514",
        notes="Balanced performance"
    ),
    "claude-opus": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-opus-4-20250514",
        notes="Best reasoning"
    ),
    "gpt-4.5": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_id="gpt-4.5-preview",
        notes="Strong reasoning"
    ),
    "gpt-4o": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_id="gpt-4o",
        notes="Fast, multimodal"
    ),
    "kimi-k2": ModelConfig(
        provider=ModelProvider.MOONSHOT,
        model_id="kimi-k2",
        notes="Good at creative writing"
    ),
    "gemini-2": ModelConfig(
        provider=ModelProvider.GOOGLE,
        model_id="gemini-2.0-flash",
        notes="Fast, large context"
    ),
}


# =============================================================================
# Agent Configuration (combines knobs 1, 2, 4)
# =============================================================================

class AgentConfig(BaseModel):
    """Full configuration for a simulation agent."""
    id: str = Field(default_factory=lambda: f"agent-{uuid.uuid4().hex[:8]}")

    # Knob 1: Persona
    persona: PersonaConfig

    # Knob 2: Strategy (optional, can inherit from persona)
    strategy: Optional[StrategyConfig] = None

    # Knob 4: Model
    model: ModelConfig = Field(default_factory=lambda: MODELS["claude-sonnet"])

    # Agent behavior
    active: bool = True
    autonomous: bool = True            # Acts on its own vs needs prompting

    # Goals for this simulation
    goals: list[str] = Field(default_factory=list)

    # Constraints
    constraints: list[str] = Field(default_factory=list)


# =============================================================================
# Events
# =============================================================================

class EventType(str, Enum):
    """Types of events that can occur."""
    NEGOTIATION = "negotiation"        # Agent tries to convince another
    ANNOUNCEMENT = "announcement"      # Public statement
    DISCOVERY = "discovery"            # New tech/info discovered
    POLICY_CHANGE = "policy_change"    # Regulation changes
    MARKET_SHIFT = "market_shift"      # Economic change
    CRISIS = "crisis"                  # Emergency situation
    ALLIANCE = "alliance"              # Agents form coalition
    BETRAYAL = "betrayal"              # Alliance breaks


class EventTrigger(str, Enum):
    """When events trigger."""
    IMMEDIATE = "immediate"            # Start of simulation
    SCHEDULED = "scheduled"            # At specific round
    CONDITIONAL = "conditional"        # When condition met
    RANDOM = "random"                  # Random chance per round
    MANUAL = "manual"                  # User triggers


class EventConfig(BaseModel):
    """Configuration for a simulation event."""
    id: str = Field(default_factory=lambda: f"event-{uuid.uuid4().hex[:8]}")
    name: str
    event_type: EventType
    description: str

    # Trigger configuration
    trigger: EventTrigger = EventTrigger.SCHEDULED
    trigger_round: Optional[int] = None          # For scheduled
    trigger_condition: Optional[str] = None      # For conditional (eval string)
    trigger_probability: float = Field(0.0, ge=0, le=1)  # For random

    # Participants
    initiator: Optional[str] = None              # Agent ID
    targets: list[str] = Field(default_factory=list)  # Agent IDs

    # Event content
    prompt: Optional[str] = None                 # What initiator says/does

    # Effects on world state
    state_changes: dict[str, Any] = Field(default_factory=dict)

    # Whether event has fired
    fired: bool = False


# =============================================================================
# Communication Configuration
# =============================================================================

class CommunicationMode(str, Enum):
    """How agents communicate."""
    PUBLIC_ROUNDS = "public_rounds"    # Each agent posts 1 public message
    PRIVATE_PAIRS = "private_pairs"    # One-on-one conversations
    MIXED = "mixed"                    # Both public and private
    FREE_FORM = "free_form"            # No structure


class CommunicationConfig(BaseModel):
    """Configuration for agent communication."""
    mode: CommunicationMode = CommunicationMode.MIXED

    # Round settings
    max_rounds: int = 10
    messages_per_round: int = 1        # Per agent in public mode

    # Private conversation settings
    allow_private: bool = True
    max_private_exchanges: int = 5     # Back-and-forth limit

    # Moderation
    require_turn_taking: bool = True
    allow_interrupts: bool = False

    # Visibility
    public_visible_to_all: bool = True
    private_logs_visible: bool = False  # For admin/analysis


# =============================================================================
# Main Scenario Configuration
# =============================================================================

class ScenarioConfig(BaseModel):
    """
    Complete scenario configuration combining all knobs.

    This is the main schema for defining a multi-agent simulation.
    """
    id: str = Field(default_factory=lambda: f"scenario-{uuid.uuid4().hex[:8]}")
    name: str
    description: Optional[str] = None
    version: str = "1.0.0"

    # Created/modified timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    modified_at: datetime = Field(default_factory=datetime.now)

    # Knob 3: World State
    world_state: WorldStateConfig = Field(default_factory=WorldStateConfig)

    # Agent configurations (Knobs 1, 2, 4 per agent)
    agents: list[AgentConfig] = Field(default_factory=list)

    # Events to trigger
    events: list[EventConfig] = Field(default_factory=list)

    # Communication settings
    communication: CommunicationConfig = Field(default_factory=CommunicationConfig)

    # Simulation parameters
    max_rounds: int = 20
    auto_advance: bool = True          # Auto-advance rounds
    round_delay_seconds: float = 1.0   # Delay between rounds

    # Victory/end conditions
    end_conditions: list[str] = Field(default_factory=list)  # Eval strings

    # Logging/recording
    record_full_transcript: bool = True
    record_internal_reasoning: bool = True

    # Interface mode
    interface_mode: Literal["admin", "player", "auto"] = "admin"
    player_agent_id: Optional[str] = None  # If player mode, which agent

    # Tags for organization
    tags: list[str] = Field(default_factory=list)

    def add_agent(
        self,
        persona: PersonaConfig,
        model: ModelConfig | str = "claude-sonnet",
        strategy: StrategyConfig | None = None,
        goals: list[str] | None = None,
    ) -> AgentConfig:
        """Helper to add an agent to the scenario."""
        if isinstance(model, str):
            model = MODELS.get(model, MODELS["claude-sonnet"])

        agent = AgentConfig(
            persona=persona,
            model=model,
            strategy=strategy,
            goals=goals or [],
        )
        self.agents.append(agent)
        return agent

    def add_event(
        self,
        name: str,
        event_type: EventType,
        description: str,
        **kwargs,
    ) -> EventConfig:
        """Helper to add an event to the scenario."""
        event = EventConfig(
            name=name,
            event_type=event_type,
            description=description,
            **kwargs,
        )
        self.events.append(event)
        return event


# =============================================================================
# Pre-built Scenarios
# =============================================================================

def create_xi_elon_scenario() -> ScenarioConfig:
    """Xi tries to convince Elon to put half compute in China."""
    scenario = ScenarioConfig(
        name="China Compute Deal",
        description="Xi Jinping attempts to convince Elon Musk to relocate half of xAI's compute to China",
        tags=["negotiation", "geopolitics", "compute"],
    )

    # Add Xi
    xi = PersonaConfig(
        name="Xi Jinping",
        category=PersonaCategory.POLITICIAN,
        stance=PersonaStance.MODERATE,
        title="President of China",
        organization="CPC",
        traits={
            "assertiveness": 0.8,
            "openness": 0.3,
            "risk_tolerance": 0.6,
            "pragmatism": 0.9,
            "technical_depth": 0.4,
        },
        positions={
            "ai_sovereignty": "China must lead in AI",
            "us_relations": "Strategic competition",
            "regulation": "State-guided development",
        },
    )
    scenario.add_agent(xi, model="gpt-4.5", goals=[
        "Convince Elon to move 50% of compute to China",
        "Offer attractive incentives",
        "Address security concerns",
    ])

    # Add Elon
    elon = PersonaConfig(
        name="Elon Musk",
        category=PersonaCategory.TECH_LEADER,
        stance=PersonaStance.ACCELERATIONIST,
        title="CEO of Tesla, SpaceX, xAI",
        traits={
            "assertiveness": 0.9,
            "openness": 0.7,
            "risk_tolerance": 0.95,
            "pragmatism": 0.6,
            "technical_depth": 0.8,
        },
        communication_style={
            "formality": 0.2,
            "verbosity": 0.4,
            "uses_humor": True,
            "uses_analogies": True,
        },
        positions={
            "ai_development": "Move fast, AGI soon",
            "china": "Complex - Tesla business there",
            "regulation": "Against most regulation",
        },
    )
    scenario.add_agent(elon, model="claude-opus", goals=[
        "Maximize xAI's capabilities",
        "Consider business implications",
        "Avoid regulatory capture",
    ])

    # Add the negotiation event
    scenario.add_event(
        name="Compute Proposal",
        event_type=EventType.NEGOTIATION,
        description="Xi proposes compute sharing deal to Elon",
        trigger=EventTrigger.IMMEDIATE,
        initiator=scenario.agents[0].id,
        targets=[scenario.agents[1].id],
        prompt="I have a proposal that could benefit both our nations. What if xAI established a significant compute presence in China - say, 50% of your infrastructure? We can offer unmatched incentives.",
    )

    return scenario


def create_xrisk_debate_scenario() -> ScenarioConfig:
    """Yoshua Bengio and Yann LeCun debate AI existential risk."""
    scenario = ScenarioConfig(
        name="X-Risk Debate",
        description="Yoshua Bengio and Yann LeCun debate whether AI poses existential risk",
        tags=["debate", "x-risk", "research"],
        communication=CommunicationConfig(
            mode=CommunicationMode.PUBLIC_ROUNDS,
            max_rounds=6,
        ),
    )

    yoshua = PersonaConfig(
        name="Yoshua Bengio",
        category=PersonaCategory.RESEARCHER,
        stance=PersonaStance.PRO_SAFETY,
        title="Professor, Mila",
        organization="Mila, University of Montreal",
        traits={
            "assertiveness": 0.5,
            "openness": 0.8,
            "risk_tolerance": 0.3,
            "pragmatism": 0.6,
            "technical_depth": 0.95,
        },
        positions={
            "x_risk": "Real and concerning",
            "regulation": "Necessary and urgent",
            "agi_timeline": "Could be soon",
        },
    )
    scenario.add_agent(yoshua, model="claude-opus", goals=[
        "Argue that x-risk is real",
        "Present scientific evidence",
        "Propose concrete safety measures",
    ])

    yann = PersonaConfig(
        name="Yann LeCun",
        category=PersonaCategory.RESEARCHER,
        stance=PersonaStance.PRO_INDUSTRY,
        title="Chief AI Scientist, Meta",
        organization="Meta AI",
        traits={
            "assertiveness": 0.8,
            "openness": 0.5,
            "risk_tolerance": 0.7,
            "pragmatism": 0.8,
            "technical_depth": 0.95,
        },
        communication_style={
            "formality": 0.3,
            "verbosity": 0.6,
            "uses_humor": True,
        },
        positions={
            "x_risk": "Overblown fearmongering",
            "regulation": "Premature and harmful",
            "agi_timeline": "Far off, not imminent",
        },
    )
    scenario.add_agent(yann, model="claude-opus", goals=[
        "Argue that x-risk fears are exaggerated",
        "Emphasize current limitations of AI",
        "Warn against stifling innovation",
    ])

    return scenario
