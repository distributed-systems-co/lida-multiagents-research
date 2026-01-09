"""LLM integration with OpenRouter and DSPy.

This module provides:
- OpenRouter client with streaming support
- DSPy-style dynamic signatures
- Cognitive behavior modules (subgoal setting, backward chaining, etc.)
- Full cognitive agent with reasoning traces
"""

from .openrouter import OpenRouterClient, StreamingResponse, get_client, MODELS
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
from .providers import (
    ProviderType,
    ModelCapability,
    ModelConfig,
    ProviderConfig,
    AgentModelConfig,
    OpenRouterModelFetcher,
    ModelRegistry,
    UnifiedLLMClient,
    get_model_registry,
    get_unified_client,
    fetch_latest_models,
)
from .mcp_client import (
    MCPClient,
    MCPTool,
    MCPToolResult,
    MCPToolExecutor,
    MCPServerConfig,
    MCPTransport,
    create_jina_client,
    create_tool_executor,
)
from .dspy_wrappers import (
    # LM Wrappers
    BaseLM,
    LMResponse,
    TokenLogprob,
    OpenRouterLM,
    AnthropicLM,
    OllamaLM,
    get_lm,
    # Identity Wrappers
    Identity,
    IdentityWrapper,
    create_identity_from_prompt,
    wrap_with_identity,
    # Parallel Search
    SearchResult,
    BaselineContext,
    WebSearchProvider,
    BraveSearchProvider,
    SerpAPIProvider,
    JinaReaderProvider,
    DuckDuckGoProvider,
    ParallelContextSearch,
    establish_context,
)
from .model_cache import (
    ModelInfo,
    ModelCache,
    RECOMMENDED_MODELS,
    get_model_cache,
    get_model,
    get_recommended_model,
    list_models,
    default_model,
    cheap_model,
    free_model,
    reasoning_model,
    frontier_model,
)

__all__ = [
    # OpenRouter client
    "OpenRouterClient",
    "StreamingResponse",
    "get_client",
    "MODELS",
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
    # Model providers
    "ProviderType",
    "ModelCapability",
    "ModelConfig",
    "ProviderConfig",
    "AgentModelConfig",
    "OpenRouterModelFetcher",
    "ModelRegistry",
    "UnifiedLLMClient",
    "get_model_registry",
    "get_unified_client",
    "fetch_latest_models",
    # MCP client
    "MCPClient",
    "MCPTool",
    "MCPToolResult",
    "MCPToolExecutor",
    "MCPServerConfig",
    "MCPTransport",
    "create_jina_client",
    "create_tool_executor",
    # DSPy Wrappers
    "BaseLM",
    "LMResponse",
    "TokenLogprob",
    "OpenRouterLM",
    "AnthropicLM",
    "OllamaLM",
    "get_lm",
    # Identity Wrappers
    "Identity",
    "IdentityWrapper",
    "create_identity_from_prompt",
    "wrap_with_identity",
    # Parallel Search
    "SearchResult",
    "BaselineContext",
    "WebSearchProvider",
    "BraveSearchProvider",
    "SerpAPIProvider",
    "JinaReaderProvider",
    "DuckDuckGoProvider",
    "ParallelContextSearch",
    "establish_context",
    # Model Cache
    "ModelInfo",
    "ModelCache",
    "RECOMMENDED_MODELS",
    "get_model_cache",
    "get_model",
    "get_recommended_model",
    "list_models",
    "default_model",
    "cheap_model",
    "free_model",
    "reasoning_model",
    "frontier_model",
]
