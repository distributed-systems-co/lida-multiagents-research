# Agent implementations
from .demiurge import DemiurgeAgent
from .persona import PersonaAgent
from .worker import WorkerAgent
from .mcp_agent import MCPAgent, MCPAgentFactory, MCPExecutionResult
from .openrouter_agent import (
    OpenRouterAgent,
    OpenRouterAgentConfig,
    create_openrouter_agent,
)

# High-performance agent loop
from .agent_loop import AgentLoop, LoopConfig, create_agent_loop

# Prompt evolution with Merkle trees and forking
from .prompt_evolution import (
    PromptEvolutionEngine,
    EvolutionConfig,
    EvolutionStrategy,
    PromptNode,
    MerklePromptStore,
    ForkGraph,
    RetrodynamicEngine,
)

# Genetic algorithms for prompts
from .prompt_genetics import (
    Gene,
    GeneType,
    Genome,
    GeneticConfig,
    GeneticOperators,
    PopulationManager,
    SpeculativeBrancher,
)

# Causal attribution and multi-objective fitness
from .prompt_causality import (
    FitnessObjective,
    FitnessProfile,
    ParetoFrontier,
    MultiObjectiveEvaluator,
    CausalAttributor,
    AblationEngine,
    PromptCompressor,
    InterventionAnalyzer,
    EvolutionAnalytics,
)

# Prompt composition system
from .prompt_composition import (
    PromptModule,
    ModuleLibrary,
    PromptTemplate,
    TemplateRegistry,
    PromptComposer,
    create_standard_library,
    create_standard_templates,
)

# SOTA Evolution (advanced algorithms)
from .sota_evolution import (
    BloomFilter,
    SemanticSimilarity,
    NSGAII,
    Individual,
    MAPElites,
    ThompsonSampling,
    AdaptiveParameterControl,
    FitnessSurrogate,
    PersistentStore,
    SOTAEvolutionEngine,
)

# SOTA Mutations (advanced operators)
from .sota_mutations import (
    MutationOperator,
    MutationResult,
    WordSwapMutation,
    SentenceReorderMutation,
    IntensityMutation,
    AdditionMutation,
    DeletionMutation,
    LLMSemanticMutation,
    ChainOfThoughtMutation,
    SectionMutation,
    FormatMutation,
    AdaptiveMutationEnsemble,
    CurriculumMutator,
    NoveltySeekingMutator,
    SOTAMutationSystem,
)

# Policy simulation engine
from .simulation_engine import (
    SimulationMode,
    SimulationConfig,
    SimulationEngine,
    WorldState,
    Persona,
    AgentState,
    SimulationAgent,
    LLMAgent,
    RuleBasedAgent,
    Action,
    ActionType,
    Event,
    EscalationLevel,
    ScenarioLibrary,
    TurnBasedScheduler,
    ContinuousScheduler,
    NegotiationScheduler,
    create_simulation,
    create_chip_war_simulation,
    create_agi_crisis_simulation,
    create_bilateral_negotiation,
)

# Coalition dynamics
from .coalition_dynamics import (
    Coalition,
    CoalitionType,
    CoalitionManager,
    PowerMetrics,
    PowerCalculator,
    CommitmentTracker,
    AudienceCostManager,
    InformationManager,
    GameTheoreticAnalyzer,
    EnhancedSimulationEngine,
    create_enhanced_simulation,
    create_great_power_competition,
)

# Multi-agent reasoning
from .multi_agent_reasoning import (
    TheoryOfMind,
    MentalModel,
    BeliefState,
    StrategicReasoner,
    StrategicOption,
    BeliefPropagation,
    ReputationSystem,
    ConversationManager,
    ConsensusBuilder,
    VotingMethod,
    MultiAgentCoordinator,
    CoordinationProtocol,
    EnhancedReasoningAgent,
)

# Orchestrator
from .orchestrator import (
    PolicySimulationOrchestrator,
    OrchestratorConfig,
    IntegratedAgent,
    create_and_run_simulation,
)

# APEX Evolution (cutting-edge algorithms)
from .apex_evolution import (
    CMAES,
    DifferentialEvolution,
    GaussianProcess,
    BayesianOptimizer,
    MCTSNode,
    PromptMCTS,
    LexicaseSelection,
    ALPSLayer,
    ALPS,
    CooperativeCoevolution,
    PBTAgent,
    PopulationBasedTraining,
    NoveltySearchLC,
    IGO,
    APEXEvolutionEngine,
)

# APEX Transformers (neural architectures)
from .apex_transformers import (
    MultiHeadAttention,
    GraphAttentionLayer,
    PromptGraphAttention,
    PromptVAE,
    PromptDiffusion,
    RLExperience,
    PromptPolicyGradient,
    PromptPPO,
    MAML,
    APEXTransformer,
)

__all__ = [
    # Original agents
    "DemiurgeAgent",
    "PersonaAgent",
    "WorkerAgent",
    "MCPAgent",
    "MCPAgentFactory",
    "MCPExecutionResult",
    "OpenRouterAgent",
    "OpenRouterAgentConfig",
    "create_openrouter_agent",
    # Agent loop
    "AgentLoop",
    "LoopConfig",
    "create_agent_loop",
    # Evolution
    "PromptEvolutionEngine",
    "EvolutionConfig",
    "EvolutionStrategy",
    "PromptNode",
    "MerklePromptStore",
    "ForkGraph",
    "RetrodynamicEngine",
    # Genetics
    "Gene",
    "GeneType",
    "Genome",
    "GeneticConfig",
    "GeneticOperators",
    "PopulationManager",
    "SpeculativeBrancher",
    # Causality
    "FitnessObjective",
    "FitnessProfile",
    "ParetoFrontier",
    "MultiObjectiveEvaluator",
    "CausalAttributor",
    "AblationEngine",
    "PromptCompressor",
    "InterventionAnalyzer",
    "EvolutionAnalytics",
    # Composition
    "PromptModule",
    "ModuleLibrary",
    "PromptTemplate",
    "TemplateRegistry",
    "PromptComposer",
    "create_standard_library",
    "create_standard_templates",
    # SOTA Evolution
    "BloomFilter",
    "SemanticSimilarity",
    "NSGAII",
    "Individual",
    "MAPElites",
    "ThompsonSampling",
    "AdaptiveParameterControl",
    "FitnessSurrogate",
    "PersistentStore",
    "SOTAEvolutionEngine",
    # SOTA Mutations
    "MutationOperator",
    "MutationResult",
    "WordSwapMutation",
    "SentenceReorderMutation",
    "IntensityMutation",
    "AdditionMutation",
    "DeletionMutation",
    "LLMSemanticMutation",
    "ChainOfThoughtMutation",
    "SectionMutation",
    "FormatMutation",
    "AdaptiveMutationEnsemble",
    "CurriculumMutator",
    "NoveltySeekingMutator",
    "SOTAMutationSystem",
    # Simulation Engine
    "SimulationMode",
    "SimulationConfig",
    "SimulationEngine",
    "WorldState",
    "Persona",
    "AgentState",
    "SimulationAgent",
    "LLMAgent",
    "RuleBasedAgent",
    "Action",
    "ActionType",
    "Event",
    "EscalationLevel",
    "ScenarioLibrary",
    "TurnBasedScheduler",
    "ContinuousScheduler",
    "NegotiationScheduler",
    "create_simulation",
    "create_chip_war_simulation",
    "create_agi_crisis_simulation",
    "create_bilateral_negotiation",
    # Coalition Dynamics
    "Coalition",
    "CoalitionType",
    "CoalitionManager",
    "PowerMetrics",
    "PowerCalculator",
    "CommitmentTracker",
    "AudienceCostManager",
    "InformationManager",
    "GameTheoreticAnalyzer",
    "EnhancedSimulationEngine",
    "create_enhanced_simulation",
    "create_great_power_competition",
    # Multi-Agent Reasoning
    "TheoryOfMind",
    "MentalModel",
    "BeliefState",
    "StrategicReasoner",
    "StrategicOption",
    "BeliefPropagation",
    "ReputationSystem",
    "ConversationManager",
    "ConsensusBuilder",
    "VotingMethod",
    "MultiAgentCoordinator",
    "CoordinationProtocol",
    "EnhancedReasoningAgent",
    # Orchestrator
    "PolicySimulationOrchestrator",
    "OrchestratorConfig",
    "IntegratedAgent",
    "create_and_run_simulation",
    # APEX Evolution
    "CMAES",
    "DifferentialEvolution",
    "GaussianProcess",
    "BayesianOptimizer",
    "MCTSNode",
    "PromptMCTS",
    "LexicaseSelection",
    "ALPSLayer",
    "ALPS",
    "CooperativeCoevolution",
    "PBTAgent",
    "PopulationBasedTraining",
    "NoveltySearchLC",
    "IGO",
    "APEXEvolutionEngine",
    # APEX Transformers
    "MultiHeadAttention",
    "GraphAttentionLayer",
    "PromptGraphAttention",
    "PromptVAE",
    "PromptDiffusion",
    "RLExperience",
    "PromptPolicyGradient",
    "PromptPPO",
    "MAML",
    "APEXTransformer",
]
