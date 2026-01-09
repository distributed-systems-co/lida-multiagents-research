"""
Static Tech Tree Definition

Defines the technology progression tree for AI/compute simulations.
Technologies have prerequisites, costs, and unlock effects.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class TechCategory(str, Enum):
    """Categories of technology."""
    COMPUTE = "compute"
    ALGORITHMS = "algorithms"
    DATA = "data"
    MODELS = "models"
    CAPABILITIES = "capabilities"
    INFRASTRUCTURE = "infrastructure"
    APPLICATIONS = "applications"
    SAFETY = "safety"
    ROBOTICS = "robotics"


class TechTier(int, Enum):
    """Technology tiers (roughly maps to years/difficulty)."""
    TIER_0 = 0   # Foundational (pre-2020)
    TIER_1 = 1   # Current (2020-2024)
    TIER_2 = 2   # Near-term (2024-2026)
    TIER_3 = 3   # Medium-term (2026-2028)
    TIER_4 = 4   # Long-term (2028-2030)
    TIER_5 = 5   # Speculative (2030+)


class Technology(BaseModel):
    """A single technology in the tech tree."""
    id: str
    name: str
    category: TechCategory
    tier: TechTier
    description: str

    # Prerequisites (tech IDs that must be unlocked first)
    prerequisites: list[str] = Field(default_factory=list)

    # Cost to develop/acquire
    compute_cost: float = 0          # FLOPS required
    funding_cost: float = 0          # USD required
    talent_cost: int = 0             # Researchers required
    time_months: int = 0             # Development time

    # Effects when unlocked
    unlocks: list[str] = Field(default_factory=list)  # Tech IDs this enables
    capabilities_granted: list[str] = Field(default_factory=list)
    compute_multiplier: float = 1.0  # Multiplier to available compute
    efficiency_gain: float = 0.0     # % improvement in training efficiency

    # Who has it
    controllers: list[str] = Field(default_factory=list)
    is_public: bool = False          # Publicly available vs proprietary

    # Risk factors
    dual_use: bool = False           # Can be used for harm
    safety_critical: bool = False    # Requires safety measures


# =============================================================================
# TIER 0: Foundational Technologies (Pre-2020)
# =============================================================================

TRANSFORMER = Technology(
    id="transformer",
    name="Transformer Architecture",
    category=TechCategory.ALGORITHMS,
    tier=TechTier.TIER_0,
    description="Self-attention based neural network architecture",
    capabilities_granted=["sequence_modeling", "parallelization"],
    is_public=True,
    controllers=["google", "public"],
)

GPU_COMPUTE = Technology(
    id="gpu_compute",
    name="GPU Computing",
    category=TechCategory.COMPUTE,
    tier=TechTier.TIER_0,
    description="Graphics processors repurposed for parallel computation",
    capabilities_granted=["parallel_training", "matrix_operations"],
    is_public=True,
    controllers=["nvidia", "amd", "public"],
)

INTERNET_SCALE_DATA = Technology(
    id="internet_data",
    name="Internet-Scale Training Data",
    category=TechCategory.DATA,
    tier=TechTier.TIER_0,
    description="Web-scraped text corpora for training",
    capabilities_granted=["broad_knowledge", "language_understanding"],
    is_public=True,
)

BACKPROP = Technology(
    id="backprop",
    name="Backpropagation",
    category=TechCategory.ALGORITHMS,
    tier=TechTier.TIER_0,
    description="Gradient-based learning algorithm",
    is_public=True,
)

RLHF_BASIC = Technology(
    id="rlhf_basic",
    name="Basic RLHF",
    category=TechCategory.ALGORITHMS,
    tier=TechTier.TIER_0,
    description="Reinforcement Learning from Human Feedback",
    prerequisites=["transformer", "backprop"],
    capabilities_granted=["instruction_following", "preference_learning"],
    is_public=True,
)


# =============================================================================
# TIER 1: Current Technologies (2020-2024)
# =============================================================================

GPT4_CLASS = Technology(
    id="gpt4_class",
    name="GPT-4 Class Models",
    category=TechCategory.MODELS,
    tier=TechTier.TIER_1,
    description="~1T parameter frontier language models",
    prerequisites=["transformer", "gpu_compute", "internet_data", "rlhf_basic"],
    compute_cost=1e24,
    funding_cost=100e6,
    talent_cost=100,
    time_months=12,
    capabilities_granted=[
        "advanced_reasoning",
        "code_generation",
        "long_context",
        "instruction_following",
    ],
    controllers=["openai", "anthropic", "google"],
)

CLAUDE_SONNET_CLASS = Technology(
    id="claude_sonnet",
    name="Claude Sonnet Class",
    category=TechCategory.MODELS,
    tier=TechTier.TIER_1,
    description="Balanced capability/efficiency frontier model",
    prerequisites=["transformer", "rlhf_basic", "constitutional_ai"],
    compute_cost=5e23,
    funding_cost=50e6,
    capabilities_granted=["reasoning", "safety_aware", "helpful"],
    controllers=["anthropic"],
)

CONSTITUTIONAL_AI = Technology(
    id="constitutional_ai",
    name="Constitutional AI",
    category=TechCategory.SAFETY,
    tier=TechTier.TIER_1,
    description="AI trained on principles rather than just examples",
    prerequisites=["rlhf_basic"],
    capabilities_granted=["value_alignment", "refusal_training"],
    is_public=True,
    safety_critical=True,
    controllers=["anthropic"],
)

MULTIMODAL_BASIC = Technology(
    id="multimodal_basic",
    name="Basic Multimodality",
    category=TechCategory.CAPABILITIES,
    tier=TechTier.TIER_1,
    description="Vision + language understanding",
    prerequisites=["gpt4_class"],
    capabilities_granted=["image_understanding", "visual_reasoning"],
    controllers=["openai", "google", "anthropic"],
)

CODE_GENERATION = Technology(
    id="code_gen",
    name="Production Code Generation",
    category=TechCategory.CAPABILITIES,
    tier=TechTier.TIER_1,
    description="Reliable code synthesis for real applications",
    prerequisites=["gpt4_class"],
    capabilities_granted=["automated_coding", "debugging"],
    controllers=["openai", "anthropic", "google", "github"],
)

H100_GPU = Technology(
    id="h100",
    name="H100 GPU",
    category=TechCategory.COMPUTE,
    tier=TechTier.TIER_1,
    description="NVIDIA Hopper architecture GPU",
    prerequisites=["gpu_compute"],
    compute_multiplier=3.0,
    controllers=["nvidia"],
)

INFERENCE_OPTIMIZATION = Technology(
    id="inference_opt",
    name="Inference Optimization",
    category=TechCategory.ALGORITHMS,
    tier=TechTier.TIER_1,
    description="Quantization, distillation, speculative decoding",
    prerequisites=["transformer"],
    efficiency_gain=0.5,
    is_public=True,
)

SYNTHETIC_DATA = Technology(
    id="synthetic_data",
    name="Synthetic Data Generation",
    category=TechCategory.DATA,
    tier=TechTier.TIER_1,
    description="AI-generated training data",
    prerequisites=["gpt4_class"],
    capabilities_granted=["data_augmentation", "self_improvement"],
    dual_use=True,
)

TOOL_USE = Technology(
    id="tool_use",
    name="Tool Use / Function Calling",
    category=TechCategory.CAPABILITIES,
    tier=TechTier.TIER_1,
    description="Models can invoke external tools and APIs",
    prerequisites=["gpt4_class"],
    capabilities_granted=["api_integration", "code_execution", "web_browsing"],
    controllers=["openai", "anthropic", "google"],
)

LONG_CONTEXT = Technology(
    id="long_context",
    name="Long Context Windows",
    category=TechCategory.CAPABILITIES,
    tier=TechTier.TIER_1,
    description="100K+ token context windows",
    prerequisites=["transformer", "inference_opt"],
    capabilities_granted=["document_analysis", "codebase_understanding"],
    controllers=["anthropic", "google"],
)


# =============================================================================
# TIER 2: Near-Term Technologies (2024-2026)
# =============================================================================

REASONING_MODELS = Technology(
    id="reasoning_models",
    name="Reasoning Models (o1/o3 class)",
    category=TechCategory.MODELS,
    tier=TechTier.TIER_2,
    description="Models with extended thinking and chain-of-thought",
    prerequisites=["gpt4_class", "synthetic_data"],
    compute_cost=1e25,
    funding_cost=500e6,
    capabilities_granted=[
        "extended_reasoning",
        "self_verification",
        "problem_decomposition",
    ],
    controllers=["openai", "anthropic", "google"],
)

CLAUDE_OPUS_CLASS = Technology(
    id="claude_opus",
    name="Claude Opus Class",
    category=TechCategory.MODELS,
    tier=TechTier.TIER_2,
    description="Top-tier reasoning and capability model",
    prerequisites=["claude_sonnet", "reasoning_models"],
    compute_cost=2e24,
    funding_cost=200e6,
    capabilities_granted=["deep_reasoning", "nuanced_understanding", "complex_tasks"],
    controllers=["anthropic"],
)

AGENT_FRAMEWORKS = Technology(
    id="agent_frameworks",
    name="Agent Frameworks",
    category=TechCategory.CAPABILITIES,
    tier=TechTier.TIER_2,
    description="Multi-step autonomous task execution",
    prerequisites=["tool_use", "reasoning_models"],
    capabilities_granted=["autonomous_tasks", "multi_step_planning"],
    dual_use=True,
)

COMPUTER_USE = Technology(
    id="computer_use",
    name="Computer Use",
    category=TechCategory.CAPABILITIES,
    tier=TechTier.TIER_2,
    description="AI can operate desktop/web interfaces",
    prerequisites=["multimodal_basic", "agent_frameworks"],
    capabilities_granted=["gui_automation", "browser_control"],
    controllers=["anthropic", "openai"],
    dual_use=True,
)

B200_GPU = Technology(
    id="b200",
    name="B200 GPU",
    category=TechCategory.COMPUTE,
    tier=TechTier.TIER_2,
    description="NVIDIA Blackwell architecture GPU",
    prerequisites=["h100"],
    compute_multiplier=2.5,
    controllers=["nvidia"],
)

MIXTURE_OF_EXPERTS = Technology(
    id="moe",
    name="Mixture of Experts (Scaled)",
    category=TechCategory.ALGORITHMS,
    tier=TechTier.TIER_2,
    description="Sparse expert models for efficiency",
    prerequisites=["transformer", "inference_opt"],
    efficiency_gain=0.7,
    compute_multiplier=1.5,
    is_public=True,
)

WORLD_MODELS = Technology(
    id="world_models",
    name="World Models",
    category=TechCategory.MODELS,
    tier=TechTier.TIER_2,
    description="Models that learn physics/causality",
    prerequisites=["multimodal_basic", "synthetic_data"],
    capabilities_granted=["physical_reasoning", "simulation"],
    controllers=["google", "meta"],
)

VIDEO_GENERATION = Technology(
    id="video_gen",
    name="High-Quality Video Generation",
    category=TechCategory.CAPABILITIES,
    tier=TechTier.TIER_2,
    description="Sora-class video synthesis",
    prerequisites=["multimodal_basic", "world_models"],
    capabilities_granted=["video_synthesis", "visual_storytelling"],
    controllers=["openai", "google", "runway"],
    dual_use=True,
)

HUMANOID_BASIC = Technology(
    id="humanoid_basic",
    name="Basic Humanoid Robots",
    category=TechCategory.ROBOTICS,
    tier=TechTier.TIER_2,
    description="Walking, grasping humanoid platforms",
    prerequisites=["gpu_compute"],
    funding_cost=1e9,
    capabilities_granted=["physical_manipulation", "locomotion"],
    controllers=["tesla", "figure", "boston_dynamics"],
)

MCP_PROTOCOL = Technology(
    id="mcp",
    name="Model Context Protocol",
    category=TechCategory.INFRASTRUCTURE,
    tier=TechTier.TIER_2,
    description="Standardized AI-tool communication",
    prerequisites=["tool_use"],
    capabilities_granted=["tool_interoperability", "ecosystem_integration"],
    is_public=True,
    controllers=["anthropic"],
)

INTERPRETABILITY_BASIC = Technology(
    id="interp_basic",
    name="Basic Interpretability",
    category=TechCategory.SAFETY,
    tier=TechTier.TIER_2,
    description="Understanding what models are doing internally",
    prerequisites=["transformer"],
    capabilities_granted=["feature_visualization", "circuit_analysis"],
    safety_critical=True,
    is_public=True,
)


# =============================================================================
# TIER 3: Medium-Term Technologies (2026-2028)
# =============================================================================

GPT5_CLASS = Technology(
    id="gpt5_class",
    name="GPT-5 Class Models",
    category=TechCategory.MODELS,
    tier=TechTier.TIER_3,
    description="Next-generation frontier models",
    prerequisites=["reasoning_models", "moe", "b200"],
    compute_cost=1e26,
    funding_cost=2e9,
    talent_cost=500,
    time_months=18,
    capabilities_granted=[
        "human_expert_reasoning",
        "complex_planning",
        "scientific_discovery",
    ],
    controllers=["openai", "anthropic", "google"],
)

PERSISTENT_MEMORY = Technology(
    id="persistent_memory",
    name="Persistent Memory Systems",
    category=TechCategory.CAPABILITIES,
    tier=TechTier.TIER_3,
    description="Long-term memory across sessions",
    prerequisites=["agent_frameworks", "long_context"],
    capabilities_granted=["learning_from_experience", "personalization"],
)

AUTONOMOUS_CODING = Technology(
    id="autonomous_coding",
    name="Autonomous Software Development",
    category=TechCategory.CAPABILITIES,
    tier=TechTier.TIER_3,
    description="End-to-end software creation from specs",
    prerequisites=["agent_frameworks", "code_gen", "reasoning_models"],
    capabilities_granted=["full_stack_development", "self_debugging"],
    dual_use=True,
)

ROBOT_FOUNDATION = Technology(
    id="robot_foundation",
    name="Robot Foundation Models",
    category=TechCategory.ROBOTICS,
    tier=TechTier.TIER_3,
    description="General-purpose robot control models",
    prerequisites=["humanoid_basic", "world_models", "gpt5_class"],
    capabilities_granted=["general_manipulation", "task_transfer"],
    controllers=["google", "tesla", "figure"],
)

MULTI_AGENT_SYSTEMS = Technology(
    id="multi_agent",
    name="Production Multi-Agent Systems",
    category=TechCategory.CAPABILITIES,
    tier=TechTier.TIER_3,
    description="Coordinated AI agent teams",
    prerequisites=["agent_frameworks", "persistent_memory"],
    capabilities_granted=["agent_coordination", "emergent_behavior"],
    dual_use=True,
)

ADVANCED_CHIP = Technology(
    id="advanced_chip",
    name="2nm AI Accelerators",
    category=TechCategory.COMPUTE,
    tier=TechTier.TIER_3,
    description="Next-gen AI chips at 2nm process",
    prerequisites=["b200"],
    compute_multiplier=3.0,
    controllers=["nvidia", "tsmc", "google"],
)

INTERPRETABILITY_ADVANCED = Technology(
    id="interp_advanced",
    name="Advanced Interpretability",
    category=TechCategory.SAFETY,
    tier=TechTier.TIER_3,
    description="Comprehensive model understanding",
    prerequisites=["interp_basic", "gpt5_class"],
    capabilities_granted=["deception_detection", "goal_inference"],
    safety_critical=True,
)

FORMAL_VERIFICATION = Technology(
    id="formal_verify",
    name="AI Formal Verification",
    category=TechCategory.SAFETY,
    tier=TechTier.TIER_3,
    description="Mathematical proofs of AI behavior",
    prerequisites=["interp_advanced"],
    capabilities_granted=["safety_guarantees", "bounded_behavior"],
    safety_critical=True,
)

FEDERATED_TRAINING = Technology(
    id="federated",
    name="Federated Large Model Training",
    category=TechCategory.INFRASTRUCTURE,
    tier=TechTier.TIER_3,
    description="Distributed training across organizations",
    prerequisites=["inference_opt", "moe"],
    capabilities_granted=["distributed_compute", "privacy_preserving"],
)


# =============================================================================
# TIER 4: Long-Term Technologies (2028-2030)
# =============================================================================

PROTO_AGI = Technology(
    id="proto_agi",
    name="Proto-AGI Systems",
    category=TechCategory.MODELS,
    tier=TechTier.TIER_4,
    description="Systems approaching general intelligence",
    prerequisites=["gpt5_class", "persistent_memory", "multi_agent", "robot_foundation"],
    compute_cost=1e27,
    funding_cost=10e9,
    talent_cost=2000,
    time_months=36,
    capabilities_granted=[
        "general_problem_solving",
        "novel_research",
        "self_improvement_limited",
    ],
    dual_use=True,
    safety_critical=True,
)

AI_SCIENTIST = Technology(
    id="ai_scientist",
    name="AI Scientist",
    category=TechCategory.CAPABILITIES,
    tier=TechTier.TIER_4,
    description="AI that conducts independent research",
    prerequisites=["proto_agi", "autonomous_coding"],
    capabilities_granted=["hypothesis_generation", "experiment_design", "paper_writing"],
    controllers=["anthropic", "google", "openai"],
)

HUMANOID_GENERAL = Technology(
    id="humanoid_general",
    name="General-Purpose Humanoid",
    category=TechCategory.ROBOTICS,
    tier=TechTier.TIER_4,
    description="Humanoids that can do most physical tasks",
    prerequisites=["robot_foundation", "proto_agi"],
    funding_cost=5e9,
    capabilities_granted=["household_tasks", "industrial_work", "care_tasks"],
    controllers=["tesla", "figure"],
)

NEUROMORPHIC = Technology(
    id="neuromorphic",
    name="Neuromorphic Computing",
    category=TechCategory.COMPUTE,
    tier=TechTier.TIER_4,
    description="Brain-inspired hardware",
    prerequisites=["advanced_chip"],
    compute_multiplier=10.0,
    efficiency_gain=0.9,
)

BRAIN_INTERFACE_ADV = Technology(
    id="bci_advanced",
    name="Advanced Brain-Computer Interface",
    category=TechCategory.INFRASTRUCTURE,
    tier=TechTier.TIER_4,
    description="High-bandwidth neural interfaces",
    prerequisites=["humanoid_basic"],
    capabilities_granted=["neural_control", "thought_communication"],
    controllers=["neuralink"],
    safety_critical=True,
)

ALIGNMENT_SOLVED = Technology(
    id="alignment_solved",
    name="Robust Alignment",
    category=TechCategory.SAFETY,
    tier=TechTier.TIER_4,
    description="Reliable methods to align AI with human values",
    prerequisites=["formal_verify", "interp_advanced"],
    capabilities_granted=["value_lock", "corrigibility"],
    safety_critical=True,
)


# =============================================================================
# TIER 5: Speculative Technologies (2030+)
# =============================================================================

AGI = Technology(
    id="agi",
    name="Artificial General Intelligence",
    category=TechCategory.MODELS,
    tier=TechTier.TIER_5,
    description="Human-level general intelligence",
    prerequisites=["proto_agi", "alignment_solved"],
    compute_cost=1e28,
    funding_cost=100e9,
    capabilities_granted=[
        "human_level_cognition",
        "cross_domain_transfer",
        "metacognition",
    ],
    dual_use=True,
    safety_critical=True,
)

RECURSIVE_IMPROVEMENT = Technology(
    id="recursive_improve",
    name="Recursive Self-Improvement",
    category=TechCategory.CAPABILITIES,
    tier=TechTier.TIER_5,
    description="AI that can improve its own capabilities",
    prerequisites=["agi", "ai_scientist"],
    capabilities_granted=["self_modification", "capability_gain"],
    dual_use=True,
    safety_critical=True,
)

ASI = Technology(
    id="asi",
    name="Artificial Superintelligence",
    category=TechCategory.MODELS,
    tier=TechTier.TIER_5,
    description="Intelligence far exceeding human level",
    prerequisites=["agi", "recursive_improve"],
    capabilities_granted=["superhuman_cognition", "unknown"],
    dual_use=True,
    safety_critical=True,
)

MOLECULAR_NANOTECH = Technology(
    id="nanotech",
    name="Molecular Nanotechnology",
    category=TechCategory.INFRASTRUCTURE,
    tier=TechTier.TIER_5,
    description="Programmable matter at molecular scale",
    prerequisites=["ai_scientist"],
    capabilities_granted=["matter_manipulation", "self_replication"],
    dual_use=True,
    safety_critical=True,
)

WHOLE_BRAIN_EMULATION = Technology(
    id="wbe",
    name="Whole Brain Emulation",
    category=TechCategory.MODELS,
    tier=TechTier.TIER_5,
    description="Complete simulation of human brain",
    prerequisites=["neuromorphic", "bci_advanced", "proto_agi"],
    capabilities_granted=["mind_upload", "consciousness_transfer"],
    safety_critical=True,
)


# =============================================================================
# Tech Tree Registry
# =============================================================================

TECH_TREE: dict[str, Technology] = {
    # Tier 0
    "transformer": TRANSFORMER,
    "gpu_compute": GPU_COMPUTE,
    "internet_data": INTERNET_SCALE_DATA,
    "backprop": BACKPROP,
    "rlhf_basic": RLHF_BASIC,
    # Tier 1
    "gpt4_class": GPT4_CLASS,
    "claude_sonnet": CLAUDE_SONNET_CLASS,
    "constitutional_ai": CONSTITUTIONAL_AI,
    "multimodal_basic": MULTIMODAL_BASIC,
    "code_gen": CODE_GENERATION,
    "h100": H100_GPU,
    "inference_opt": INFERENCE_OPTIMIZATION,
    "synthetic_data": SYNTHETIC_DATA,
    "tool_use": TOOL_USE,
    "long_context": LONG_CONTEXT,
    # Tier 2
    "reasoning_models": REASONING_MODELS,
    "claude_opus": CLAUDE_OPUS_CLASS,
    "agent_frameworks": AGENT_FRAMEWORKS,
    "computer_use": COMPUTER_USE,
    "b200": B200_GPU,
    "moe": MIXTURE_OF_EXPERTS,
    "world_models": WORLD_MODELS,
    "video_gen": VIDEO_GENERATION,
    "humanoid_basic": HUMANOID_BASIC,
    "mcp": MCP_PROTOCOL,
    "interp_basic": INTERPRETABILITY_BASIC,
    # Tier 3
    "gpt5_class": GPT5_CLASS,
    "persistent_memory": PERSISTENT_MEMORY,
    "autonomous_coding": AUTONOMOUS_CODING,
    "robot_foundation": ROBOT_FOUNDATION,
    "multi_agent": MULTI_AGENT_SYSTEMS,
    "advanced_chip": ADVANCED_CHIP,
    "interp_advanced": INTERPRETABILITY_ADVANCED,
    "formal_verify": FORMAL_VERIFICATION,
    "federated": FEDERATED_TRAINING,
    # Tier 4
    "proto_agi": PROTO_AGI,
    "ai_scientist": AI_SCIENTIST,
    "humanoid_general": HUMANOID_GENERAL,
    "neuromorphic": NEUROMORPHIC,
    "bci_advanced": BRAIN_INTERFACE_ADV,
    "alignment_solved": ALIGNMENT_SOLVED,
    # Tier 5
    "agi": AGI,
    "recursive_improve": RECURSIVE_IMPROVEMENT,
    "asi": ASI,
    "nanotech": MOLECULAR_NANOTECH,
    "wbe": WHOLE_BRAIN_EMULATION,
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_tech(tech_id: str) -> Technology | None:
    """Get a technology by ID."""
    return TECH_TREE.get(tech_id)


def get_prerequisites(tech_id: str) -> list[Technology]:
    """Get all prerequisite technologies."""
    tech = TECH_TREE.get(tech_id)
    if not tech:
        return []
    return [TECH_TREE[pid] for pid in tech.prerequisites if pid in TECH_TREE]


def get_unlocked_by(tech_id: str) -> list[Technology]:
    """Get all technologies that this tech unlocks."""
    return [t for t in TECH_TREE.values() if tech_id in t.prerequisites]


def get_tier(tier: TechTier) -> list[Technology]:
    """Get all technologies in a tier."""
    return [t for t in TECH_TREE.values() if t.tier == tier]


def get_category(category: TechCategory) -> list[Technology]:
    """Get all technologies in a category."""
    return [t for t in TECH_TREE.values() if t.category == category]


def can_research(tech_id: str, unlocked: set[str]) -> bool:
    """Check if a technology can be researched given unlocked techs."""
    tech = TECH_TREE.get(tech_id)
    if not tech:
        return False
    return all(prereq in unlocked for prereq in tech.prerequisites)


def get_available_research(unlocked: set[str]) -> list[Technology]:
    """Get all technologies available for research."""
    available = []
    for tech_id, tech in TECH_TREE.items():
        if tech_id not in unlocked and can_research(tech_id, unlocked):
            available.append(tech)
    return available


def get_tech_path(target_id: str, unlocked: set[str]) -> list[str]:
    """Get the shortest path to unlock a technology."""
    tech = TECH_TREE.get(target_id)
    if not tech:
        return []

    if target_id in unlocked:
        return []

    # BFS to find path
    path = []
    to_unlock = [target_id]

    while to_unlock:
        current = to_unlock.pop(0)
        if current in unlocked:
            continue

        tech = TECH_TREE.get(current)
        if not tech:
            continue

        missing_prereqs = [p for p in tech.prerequisites if p not in unlocked]
        if missing_prereqs:
            to_unlock.extend(missing_prereqs)
        else:
            if current not in path:
                path.append(current)

    return path


def compute_total_cost(tech_ids: list[str]) -> dict[str, float]:
    """Compute total cost to unlock a set of technologies."""
    total = {
        "compute_cost": 0.0,
        "funding_cost": 0.0,
        "talent_cost": 0,
        "time_months": 0,
    }

    for tech_id in tech_ids:
        tech = TECH_TREE.get(tech_id)
        if tech:
            total["compute_cost"] += tech.compute_cost
            total["funding_cost"] += tech.funding_cost
            total["talent_cost"] += tech.talent_cost
            total["time_months"] += tech.time_months

    return total


# =============================================================================
# Default Starting Techs (Current World State)
# =============================================================================

CURRENT_WORLD_UNLOCKED = {
    # Tier 0 - all unlocked
    "transformer",
    "gpu_compute",
    "internet_data",
    "backprop",
    "rlhf_basic",
    # Tier 1 - mostly unlocked
    "gpt4_class",
    "claude_sonnet",
    "constitutional_ai",
    "multimodal_basic",
    "code_gen",
    "h100",
    "inference_opt",
    "synthetic_data",
    "tool_use",
    "long_context",
    # Tier 2 - partially unlocked
    "reasoning_models",
    "claude_opus",
    "agent_frameworks",
    "computer_use",
    "moe",
    "mcp",
    "interp_basic",
}


# =============================================================================
# Organization Tech Access
# =============================================================================

ORG_TECH_ACCESS: dict[str, set[str]] = {
    "openai": {
        "gpt4_class", "reasoning_models", "multimodal_basic",
        "code_gen", "tool_use", "video_gen", "agent_frameworks",
    },
    "anthropic": {
        "claude_sonnet", "claude_opus", "constitutional_ai",
        "long_context", "tool_use", "computer_use", "mcp", "interp_basic",
    },
    "google": {
        "gpt4_class", "multimodal_basic", "world_models",
        "moe", "long_context", "video_gen",
    },
    "meta": {
        "gpt4_class", "moe", "world_models",
    },
    "nvidia": {
        "gpu_compute", "h100", "b200",
    },
    "tesla": {
        "humanoid_basic",
    },
    "china_state": {
        "gpu_compute", "gpt4_class", "multimodal_basic",
    },
}
