"""
MLX Model Registry for Multi-Agent System

Manages multiple MLX models optimized for different tasks:
- Function calling
- Fast inference
- Quality reasoning
- Balanced performance

All models are 4-bit quantized and under 5GB each.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any
import os


class ModelProfile(Enum):
    """Different model optimization profiles."""
    FUNCTION_CALLING = "function_calling"  # Best for tool use
    FAST = "fast"  # Fastest inference
    QUALITY = "quality"  # Best quality
    BALANCED = "balanced"  # Good balance
    REASONING = "reasoning"  # Best for complex reasoning


@dataclass
class ModelSpec:
    """Specification for an MLX model."""
    name: str
    hf_repo: str
    profile: ModelProfile
    size_gb: float
    strengths: list[str]
    ideal_for: list[str]
    max_tokens_default: int = 2048
    temperature_default: float = 0.7


# Registry of tested MLX models
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    # Best for function calling and tool use
    "qwen3-8b": ModelSpec(
        name="Qwen3 8B Instruct",
        hf_repo="mlx-community/Qwen3-8B-Instruct-4bit",
        profile=ModelProfile.FUNCTION_CALLING,
        size_gb=5.0,
        strengths=[
            "Superior function calling",
            "Excellent tool use",
            "Strong reasoning",
            "Human preference aligned",
            "Multi-turn conversations"
        ],
        ideal_for=[
            "Complex agentic workflows",
            "Function calling",
            "Tool orchestration",
            "Multi-step reasoning"
        ],
        max_tokens_default=2048
    ),

    # Gold standard for quality/size ratio
    "mistral-7b": ModelSpec(
        name="Mistral 7B Instruct v0.3",
        hf_repo="mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        profile=ModelProfile.QUALITY,
        size_gb=4.5,
        strengths=[
            "Best quality/size ratio",
            "Strong instruction following",
            "Good reasoning",
            "Reliable performance"
        ],
        ideal_for=[
            "General tasks",
            "High-quality responses",
            "Complex analysis",
            "Balanced workflows"
        ],
        max_tokens_default=2048
    ),

    # Fastest inference
    "llama-3.2-3b": ModelSpec(
        name="Llama 3.2 3B Instruct",
        hf_repo="mlx-community/Llama-3.2-3B-Instruct-4bit",
        profile=ModelProfile.FAST,
        size_gb=2.0,
        strengths=[
            "Fastest inference (~50 tokens/s)",
            "Default MLX model",
            "Low memory usage",
            "Reliable"
        ],
        ideal_for=[
            "Quick responses",
            "Real-time streaming",
            "Simple tasks",
            "Multiple concurrent agents"
        ],
        max_tokens_default=1024
    ),

    # Balanced option
    "qwen3-4b": ModelSpec(
        name="Qwen3 4B Instruct",
        hf_repo="mlx-community/Qwen3-4B-Instruct-4bit",
        profile=ModelProfile.BALANCED,
        size_gb=2.5,
        strengths=[
            "Good balance",
            "Compact",
            "Fast",
            "Capable reasoning"
        ],
        ideal_for=[
            "Balanced workloads",
            "Multiple agents",
            "Moderate complexity",
            "Quick analysis"
        ],
        max_tokens_default=1536
    ),

    # Fast and capable
    "phi-4-mini": ModelSpec(
        name="Phi-4 Mini",
        hf_repo="mlx-community/Phi-4-Mini-4bit",
        profile=ModelProfile.FAST,
        size_gb=2.0,
        strengths=[
            "Extremely fast",
            "Surprisingly capable",
            "Compact",
            "Good for specific tasks"
        ],
        ideal_for=[
            "Speed-critical tasks",
            "Simple reasoning",
            "High throughput",
            "Resource-constrained"
        ],
        max_tokens_default=1024
    ),
}


class ModelManager:
    """Manages MLX model selection and configuration."""

    def __init__(self):
        self.current_model: Optional[str] = None
        self.loaded_models: Dict[str, Any] = {}

    @staticmethod
    def get_model_for_task(task_type: str) -> str:
        """Get the best model for a specific task type."""
        task_mappings = {
            "function_calling": "qwen3-8b",
            "tool_use": "qwen3-8b",
            "fast": "llama-3.2-3b",
            "quality": "mistral-7b",
            "balanced": "qwen3-4b",
            "reasoning": "qwen3-8b",
            "streaming": "llama-3.2-3b",
        }
        return task_mappings.get(task_type.lower(), "qwen3-8b")

    @staticmethod
    def get_model_spec(model_id: str) -> ModelSpec:
        """Get specifications for a model."""
        if model_id not in MODEL_REGISTRY:
            raise ValueError(f"Model {model_id} not found. Available: {list(MODEL_REGISTRY.keys())}")
        return MODEL_REGISTRY[model_id]

    @staticmethod
    def list_models(profile: Optional[ModelProfile] = None) -> Dict[str, ModelSpec]:
        """List all available models, optionally filtered by profile."""
        if profile is None:
            return MODEL_REGISTRY
        return {
            k: v for k, v in MODEL_REGISTRY.items()
            if v.profile == profile
        }

    @staticmethod
    def get_total_size(model_ids: list[str]) -> float:
        """Calculate total size of multiple models."""
        return sum(MODEL_REGISTRY[mid].size_gb for mid in model_ids if mid in MODEL_REGISTRY)

    @staticmethod
    def recommend_model_set(max_total_gb: float = 15.0) -> list[str]:
        """Recommend a diverse set of models within memory budget."""
        if max_total_gb >= 15:
            # Full diverse set
            return ["qwen3-8b", "mistral-7b", "llama-3.2-3b"]
        elif max_total_gb >= 10:
            # Good diversity
            return ["qwen3-8b", "llama-3.2-3b", "qwen3-4b"]
        elif max_total_gb >= 7:
            # Balanced pair
            return ["mistral-7b", "llama-3.2-3b"]
        else:
            # Single best
            return ["qwen3-8b"]

    @staticmethod
    def print_model_info(model_id: str):
        """Print detailed information about a model."""
        spec = ModelManager.get_model_spec(model_id)

        print(f"\n{'='*60}")
        print(f" {spec.name}")
        print(f"{'='*60}")
        print(f"  Repo: {spec.hf_repo}")
        print(f"  Profile: {spec.profile.value}")
        print(f"  Size: {spec.size_gb}GB (4-bit quantized)")
        print(f"  Default tokens: {spec.max_tokens_default}")
        print(f"  Temperature: {spec.temperature_default}")
        print(f"\n  Strengths:")
        for strength in spec.strengths:
            print(f"    • {strength}")
        print(f"\n  Ideal for:")
        for use_case in spec.ideal_for:
            print(f"    • {use_case}")
        print(f"{'='*60}\n")


def get_recommended_models(
    budget_gb: float = 15.0,
    prioritize_function_calling: bool = True
) -> list[str]:
    """
    Get recommended model configuration based on memory budget.

    Args:
        budget_gb: Total memory budget in GB
        prioritize_function_calling: Whether to prioritize tool use capability

    Returns:
        List of recommended model IDs
    """
    if prioritize_function_calling:
        if budget_gb >= 15:
            # Best setup: function calling + quality + fast
            return ["qwen3-8b", "mistral-7b", "llama-3.2-3b"]
        elif budget_gb >= 10:
            # Good setup: function calling + fast + balanced
            return ["qwen3-8b", "llama-3.2-3b", "qwen3-4b"]
        elif budget_gb >= 7:
            # Minimal: function calling + fast
            return ["qwen3-8b", "llama-3.2-3b"]
        else:
            # Single model: best for function calling
            return ["qwen3-8b"]
    else:
        # Prioritize diversity
        if budget_gb >= 15:
            return ["mistral-7b", "qwen3-8b", "llama-3.2-3b", "phi-4-mini"]
        elif budget_gb >= 10:
            return ["mistral-7b", "llama-3.2-3b", "qwen3-4b"]
        else:
            return ["mistral-7b", "llama-3.2-3b"]


# Convenience function
def create_model_config(model_id: str, **overrides) -> Dict[str, Any]:
    """Create MLXModelConfig dict for a model."""
    spec = ModelManager.get_model_spec(model_id)

    config = {
        "model_path": spec.hf_repo,
        "max_tokens": spec.max_tokens_default,
        "temperature": spec.temperature_default
    }

    # Apply overrides
    config.update(overrides)

    return config


if __name__ == "__main__":
    # Demo usage
    print("\n🎯 MLX Model Registry\n")

    # Show all models
    print("Available Models:")
    print("-" * 60)
    for model_id, spec in MODEL_REGISTRY.items():
        print(f"  {model_id:15} - {spec.name:25} ({spec.size_gb}GB)")

    # Show recommendations
    print(f"\n\n📊 Recommendations for 15GB budget:")
    recommended = get_recommended_models(15.0, prioritize_function_calling=True)
    total_size = ModelManager.get_total_size(recommended)
    print(f"  Total: {total_size:.1f}GB / 15GB")
    for model_id in recommended:
        spec = MODEL_REGISTRY[model_id]
        print(f"    • {spec.name} ({spec.size_gb}GB) - {spec.profile.value}")

    # Show detailed info for best model
    print("\n\n📖 Recommended Primary Model:")
    ModelManager.print_model_info("qwen3-8b")
