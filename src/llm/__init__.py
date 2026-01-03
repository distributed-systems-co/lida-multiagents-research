"""LLM integration with OpenRouter and DSPy.

This module provides:
- OpenRouter client with streaming support
- DSPy-style dynamic signatures
- Cognitive behavior modules (subgoal setting, backward chaining, etc.)
- Full cognitive agent with reasoning traces
"""

from .openrouter import OpenRouterClient, StreamingResponse, get_client
from .signatures import SignatureBuilder, DynamicSignature, Field, get_signature
from .dspy_integration import (
    DSPyModule,
    ChainOfThought,
    Predict,
    MultiModule,
    ModuleResult,
    create_module,
    create_persona_module,
    create_analysis_module,
    create_debate_module,
)
from .reasoning import (
    ReasoningStep,
    ReasoningTrace,
    AgentContext,
    StepType,
    StepStatus,
)
from .behaviors import (
    BehaviorModule,
    SubgoalSettingModule,
    BackwardChainingModule,
    VerificationModule,
    BacktrackingModule,
    HypothesisGenerationModule,
    SynthesisModule,
    get_behavior,
    list_behaviors,
)
from .cognitive_agent import (
    CognitiveAgent,
    CartesianProductPlanner,
    ExecutionPlan,
    PlanStep,
    create_cognitive_agent,
    reason,
    quick_reason,
)

__all__ = [
    # OpenRouter client
    "OpenRouterClient",
    "StreamingResponse",
    "get_client",
    # Signatures
    "SignatureBuilder",
    "DynamicSignature",
    "Field",
    "get_signature",
    # DSPy integration
    "DSPyModule",
    "ChainOfThought",
    "Predict",
    "MultiModule",
    "ModuleResult",
    "create_module",
    "create_persona_module",
    "create_analysis_module",
    "create_debate_module",
    # Reasoning traces
    "ReasoningStep",
    "ReasoningTrace",
    "AgentContext",
    "StepType",
    "StepStatus",
    # Behavior modules
    "BehaviorModule",
    "SubgoalSettingModule",
    "BackwardChainingModule",
    "VerificationModule",
    "BacktrackingModule",
    "HypothesisGenerationModule",
    "SynthesisModule",
    "get_behavior",
    "list_behaviors",
    # Cognitive agent
    "CognitiveAgent",
    "CartesianProductPlanner",
    "ExecutionPlan",
    "PlanStep",
    "create_cognitive_agent",
    "reason",
    "quick_reason",
]
