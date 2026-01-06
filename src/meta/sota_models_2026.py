"""
SOTA MLX Models for January 2026

DeepSeek-R1, Qwen 2.5, and Llama 3.2 - The best open models available.
"""
from dataclasses import dataclass
from typing import Dict


@dataclass
class SOTAModel:
    """SOTA model specification."""
    name: str
    repo: str
    size_gb: float
    released: str
    strengths: list[str]
    ideal_for: list[str]


# January 2026 SOTA Models (All MLX-ready, under 15GB total)
SOTA_2026 = {
    "deepseek-r1-7b": SOTAModel(
        name="DeepSeek-R1 Distill Qwen 7B",
        repo="mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
        size_gb=4.5,
        released="January 20, 2026",
        strengths=[
            "🏆 SOTA open reasoning (Jan 2026)",
            "Beats OpenAI o1-mini on benchmarks",
            "Step-by-step thinking chains",
            "Excellent function calling",
            "Built for $5.6M (shocking efficiency)"
        ],
        ideal_for=[
            "Complex reasoning tasks",
            "Function calling & tool use",
            "Multi-step analysis",
            "Agentic workflows"
        ]
    ),

    "qwen-2.5-14b": SOTAModel(
        name="Qwen 2.5 14B Instruct",
        repo="mlx-community/Qwen2.5-14B-Instruct-4bit",
        size_gb=8.0,
        released="Late 2025",
        strengths=[
            "🥇 Overtook Llama in popularity (2026)",
            "Matches/exceeds GPT-4o on benchmarks",
            "29 languages supported",
            "Superior instruction following",
            "Best for technical content"
        ],
        ideal_for=[
            "Quality responses",
            "Technical analysis",
            "Multilingual tasks",
            "General reasoning"
        ]
    ),

    "llama-3.2-3b": SOTAModel(
        name="Llama 3.2 3B Instruct",
        repo="mlx-community/Llama-3.2-3B-Instruct-4bit",
        size_gb=2.0,
        released="2025",
        strengths=[
            "⚡ Fastest inference (50 tokens/s on M3 Max)",
            "Default MLX model",
            "Reliable and tested",
            "Low memory footprint",
            "Great for streaming"
        ],
        ideal_for=[
            "Real-time streaming",
            "Quick responses",
            "Multiple concurrent agents",
            "Simple tasks"
        ]
    )
}


def get_sota_config(model_id: str) -> dict:
    """Get MLX config for SOTA model."""
    model = SOTA_2026[model_id]

    # Optimize tokens based on model size
    max_tokens = {
        "deepseek-r1-7b": 2048,  # Reasoning needs more tokens
        "qwen-2.5-14b": 1536,     # Balanced
        "llama-3.2-3b": 1024      # Fast, shorter
    }

    return {
        "model_path": model.repo,
        "max_tokens": max_tokens[model_id],
        "temperature": 0.7
    }


def print_sota_summary():
    """Print summary of SOTA 2026 models."""
    print("\n" + "="*70)
    print(" SOTA MLX MODELS - JANUARY 2026")
    print("="*70 + "\n")

    total_size = sum(m.size_gb for m in SOTA_2026.values())
    print(f"Total Size: {total_size:.1f}GB (fits in 15GB budget)\n")

    for model_id, model in SOTA_2026.items():
        print(f"{'─'*70}")
        print(f"{model.name}")
        print(f"{'─'*70}")
        print(f"  📦 Repo: {model.repo}")
        print(f"  💾 Size: {model.size_gb}GB")
        print(f"  📅 Released: {model.released}")
        print(f"\n  Strengths:")
        for s in model.strengths:
            print(f"    • {s}")
        print(f"\n  Best for:")
        for i in model.ideal_for:
            print(f"    • {i}")
        print()


if __name__ == "__main__":
    print_sota_summary()
