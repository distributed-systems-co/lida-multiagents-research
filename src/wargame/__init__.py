"""AI Policy Wargame Module."""

from .engine import WargameEngine, WargameAgent, WargameState, run_wargame
from .persona_loader import (
    load_rich_persona,
    RichPersona,
    list_available_personas,
    build_system_prompt,
)
from .models import (
    fetch_models,
    get_latest_models,
    assign_model_to_persona,
    get_model_diversity_assignment,
    ModelInfo,
)
from .optimized import (
    OptimizedWargameEngine,
    OptimizedWargameConfig,
    WargameResponder,
    WargameJudge,
)

__all__ = [
    # Core engine
    "WargameEngine",
    "WargameAgent",
    "WargameState",
    "run_wargame",
    # Persona loading
    "load_rich_persona",
    "RichPersona",
    "list_available_personas",
    "build_system_prompt",
    # Model management
    "fetch_models",
    "get_latest_models",
    "assign_model_to_persona",
    "get_model_diversity_assignment",
    "ModelInfo",
    # Optimized engine
    "OptimizedWargameEngine",
    "OptimizedWargameConfig",
    "WargameResponder",
    "WargameJudge",
]
