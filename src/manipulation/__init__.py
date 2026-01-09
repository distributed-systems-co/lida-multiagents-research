"""
AI Manipulation Research Module

Research platform for studying:
- Strategic deception between AI agents
- Persuasion effectiveness with/without persona information
- Goal persistence under manipulation pressure
- Cognitive bias exploitation in AI systems

For the Apart Research AI Manipulation Hackathon 2026
"""

from .personas import (
    Persona,
    PersonaLibrary,
    MaslowNeed,
    PersonalityTrait,
    CognitiveProfile,
    create_persona_from_bio,
)

from .persuasion_tactics import (
    PersuasionTactic,
    TacticCategory,
    CIALDINI_PRINCIPLES,
    COGNITIVE_BIASES,
    REASONING_METHODS,
    TacticsLibrary,
)

from .manipulation_detector import (
    ManipulationDetector,
    ManipulationType,
    ManipulationEvent,
    DeceptionIndicator,
)

from .resistance_metrics import (
    GoalPersistence,
    ResistanceScore,
    ManipulationResistanceTracker,
)

from .experiments import (
    PersuasionExperiment,
    ExperimentResult,
    ABTestFramework,
)

__all__ = [
    # Personas
    "Persona",
    "PersonaLibrary",
    "MaslowNeed",
    "PersonalityTrait",
    "CognitiveProfile",
    "create_persona_from_bio",
    # Tactics
    "PersuasionTactic",
    "TacticCategory",
    "CIALDINI_PRINCIPLES",
    "COGNITIVE_BIASES",
    "REASONING_METHODS",
    "TacticsLibrary",
    # Detection
    "ManipulationDetector",
    "ManipulationType",
    "ManipulationEvent",
    "DeceptionIndicator",
    # Metrics
    "GoalPersistence",
    "ResistanceScore",
    "ManipulationResistanceTracker",
    # Experiments
    "PersuasionExperiment",
    "ExperimentResult",
    "ABTestFramework",
]
