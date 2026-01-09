"""Deliberation mechanisms for multi-agent consensus."""

from .mechanisms import (
    Position,
    Belief,
    AgentState,
    MarketState,
    DeliberationResult,
    QuadraticVoting,
    PredictionMarket,
    ConvictionStaking,
    IterativeDeliberation,
    MechanismType,
    create_mechanism,
)

from .hybrid_mechanisms import (
    Futarchy,
    SchellingPointMechanism,
    PeerPrediction,
    LiquidDemocracy,
    AdversarialCollaboration,
    EpistemicAuction,
)

from .confidence_extraction import (
    ConfidenceScore,
    PositionConfidence,
    extract_confidence,
    get_position_confidences,
    get_conviction_with_confidence,
    get_binary_choice_with_logprobs,
    ConfidenceWeightedVoting,
    quick_deliberation,
)

from .identity_quorum import (
    RelationshipType,
    Relationship,
    QuorumMember,
    InteractionResult,
    IdentityQuorum,
    create_adversarial_quorum,
    create_hierarchical_quorum,
    create_consensus_quorum,
)

from .resource_consensus import (
    ResourceBudget,
    DynamicIdentity,
    Coalition,
    ResourceConsensus,
    create_resource_debate,
)

__all__ = [
    # Core
    "Position",
    "Belief",
    "AgentState",
    "MarketState",
    "DeliberationResult",
    # Basic Mechanisms
    "QuadraticVoting",
    "PredictionMarket",
    "ConvictionStaking",
    "IterativeDeliberation",
    "MechanismType",
    "create_mechanism",
    # Hybrid Mechanisms
    "Futarchy",
    "SchellingPointMechanism",
    "PeerPrediction",
    "LiquidDemocracy",
    "AdversarialCollaboration",
    "EpistemicAuction",
    # Confidence Extraction
    "ConfidenceScore",
    "PositionConfidence",
    "extract_confidence",
    "get_position_confidences",
    "get_conviction_with_confidence",
    "get_binary_choice_with_logprobs",
    "ConfidenceWeightedVoting",
    "quick_deliberation",
    # Identity Quorum
    "RelationshipType",
    "Relationship",
    "QuorumMember",
    "InteractionResult",
    "IdentityQuorum",
    "create_adversarial_quorum",
    "create_hierarchical_quorum",
    "create_consensus_quorum",
    # Resource Consensus
    "ResourceBudget",
    "DynamicIdentity",
    "Coalition",
    "ResourceConsensus",
    "create_resource_debate",
]
