"""
Scenario and configuration loader for LIDA.
Loads YAML configs from scenarios/ directory and merges with defaults.
Supports hierarchical structure with versioned components.
"""
import os
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from functools import lru_cache
import re

logger = logging.getLogger(__name__)

# Find project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCENARIOS_DIR = PROJECT_ROOT / "scenarios"
DEFAULT_SCENARIO = "default.yaml"

# Component directories
COMPONENT_DIRS = {
    "personas": SCENARIOS_DIR / "personas",
    "tactics": SCENARIOS_DIR / "tactics",
    "topics": SCENARIOS_DIR / "topics",
    "models": SCENARIOS_DIR / "models",
    "relationships": SCENARIOS_DIR / "relationships",
    "prompts": SCENARIOS_DIR / "prompts",
    "campaigns": SCENARIOS_DIR / "campaigns",
    "presets": SCENARIOS_DIR / "presets",
}


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dicts, override takes precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml_file(path: Union[str, Path]) -> Dict:
    """Load a YAML file, returning empty dict if not found."""
    path = Path(path)
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    logger.warning(f"File not found: {path}")
    return {}


def resolve_path(base_path: Path, relative_path: str) -> Path:
    """Resolve a relative path from a base path."""
    if relative_path.startswith("../"):
        return (base_path.parent / relative_path).resolve()
    return (base_path / relative_path).resolve()


def load_component(component_type: str, path_or_name: str, version: str = "v1") -> Dict:
    """
    Load a component (personas, tactics, etc.) from the hierarchical structure.

    Args:
        component_type: Type of component (personas, tactics, topics, etc.)
        path_or_name: Either a relative path or just a filename
        version: Version directory to look in (v1, v2, etc.)

    Returns:
        Loaded component data.
    """
    base_dir = COMPONENT_DIRS.get(component_type)
    if not base_dir:
        logger.warning(f"Unknown component type: {component_type}")
        return {}

    # If it's a relative path, resolve it
    if "/" in path_or_name or path_or_name.endswith(".yaml"):
        if path_or_name.startswith("../"):
            # Relative to scenarios dir
            full_path = (SCENARIOS_DIR / path_or_name.lstrip("../")).resolve()
        else:
            full_path = base_dir / path_or_name
    else:
        # Just a name, look in version directory
        full_path = base_dir / version / f"{path_or_name}.yaml"

    return load_yaml_file(full_path)


def resolve_imports(config: Dict, base_path: Path = None) -> Dict:
    """
    Resolve imports in a config, loading referenced components.

    Args:
        config: Configuration dict that may contain 'imports' key
        base_path: Base path for resolving relative imports

    Returns:
        Config with imports resolved and merged.
    """
    if "imports" not in config:
        return config

    imports = config.pop("imports")
    resolved = {}

    for key, import_spec in imports.items():
        if isinstance(import_spec, str):
            # Simple string path
            source_path = import_spec
            select = None
            use = None
        elif isinstance(import_spec, dict):
            source_path = import_spec.get("source", "")
            select = import_spec.get("select")
            use = import_spec.get("use")
        else:
            continue

        # Resolve the path
        if base_path:
            full_path = resolve_path(base_path, source_path)
        else:
            full_path = SCENARIOS_DIR / source_path.lstrip("../")

        loaded = load_yaml_file(full_path)

        # Apply 'use' filter (load specific key from file)
        if use and use in loaded:
            loaded = loaded[use]

        # Apply 'select' filter (select specific items)
        if select and isinstance(loaded, dict):
            if "personas" in loaded:
                # Filter personas
                loaded["personas"] = {
                    k: v for k, v in loaded.get("personas", {}).items()
                    if k in select
                }
            elif "topics" in loaded:
                # Filter topics
                loaded["topics"] = {
                    k: v for k, v in loaded.get("topics", {}).items()
                    if k in select
                }

        resolved[key] = loaded

    # Merge resolved imports into config
    for key, data in resolved.items():
        if key not in config:
            config[key] = data
        elif isinstance(config[key], dict) and isinstance(data, dict):
            config[key] = deep_merge(data, config[key])

    return config


def resolve_extends(config: Dict, base_path: Path = None) -> Dict:
    """
    Resolve _extends directive, loading and merging base config.

    Args:
        config: Configuration dict that may contain '_extends' key
        base_path: Base path for resolving relative extends

    Returns:
        Config with extends resolved and merged.
    """
    if "_extends" not in config:
        return config

    extends_path = config.pop("_extends")

    # Resolve the path
    if base_path:
        full_path = resolve_path(base_path, extends_path)
    else:
        full_path = SCENARIOS_DIR / "presets" / extends_path

    base_config = load_yaml_file(full_path)

    # Recursively resolve extends in base
    base_config = resolve_extends(base_config, full_path.parent)

    # Merge: config overrides base
    return deep_merge(base_config, config)


@lru_cache(maxsize=16)
def load_scenario(name: str = None) -> Dict[str, Any]:
    """
    Load a scenario configuration.

    Supports loading from:
    - scenarios/*.yaml (legacy flat files)
    - scenarios/presets/*.yaml (preset configurations)
    - scenarios/campaigns/*.yaml (full campaign definitions)

    Args:
        name: Scenario name (without .yaml) or path to yaml file.
              If None, uses SCENARIO env var or 'default'.

    Returns:
        Merged configuration dictionary with imports resolved.
    """
    # Determine scenario to load
    if name is None:
        name = os.getenv("SCENARIO", "default")

    config = {}
    scenario_path = None

    # Search order: presets > campaigns > root scenarios
    search_paths = [
        SCENARIOS_DIR / "presets" / f"{name}.yaml",
        SCENARIOS_DIR / "campaigns" / f"{name}.yaml",
        SCENARIOS_DIR / f"{name}.yaml",
    ]

    # Also check if it's a direct path
    if name.endswith(".yaml") or "/" in name:
        search_paths.insert(0, Path(name))

    # Find first existing file
    for path in search_paths:
        if path.exists():
            scenario_path = path
            config = load_yaml_file(path)
            logger.info(f"Loaded scenario from {path}")
            break

    if not config and name != "default":
        logger.warning(f"Scenario '{name}' not found in any location")
        # Fall back to default
        default_path = SCENARIOS_DIR / "presets" / "default.yaml"
        if default_path.exists():
            config = load_yaml_file(default_path)
            scenario_path = default_path

    # Resolve _extends directive
    if scenario_path:
        config = resolve_extends(config, scenario_path.parent)

    # Resolve imports
    if scenario_path:
        config = resolve_imports(config, scenario_path.parent)

    # Apply environment variable overrides
    config = apply_env_overrides(config)

    return config


def apply_env_overrides(config: Dict) -> Dict:
    """Apply environment variable overrides to config."""
    env_mappings = {
        "PORT": ("server", "port", int),
        "HOST": ("server", "host", str),
        "WORKERS": ("server", "workers", int),
        "LOG_LEVEL": ("server", "log_level", str),
        "NUM_AGENTS": ("simulation", "num_agents", int),
        "MAX_ROUNDS": ("simulation", "max_rounds_per_agent", int),
        "SWARM_AGENTS": ("simulation", "num_agents", int),
        "REDIS_URL": ("redis", "url", str),
        "DEFAULT_MODEL": ("models", "default", str),
        "PERSUADER_MODEL": ("persuader", "model", str),
    }

    for env_var, (section, key, type_fn) in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            if section not in config:
                config[section] = {}
            config[section][key] = type_fn(value)
            logger.debug(f"Override from env: {section}.{key} = {value}")

    return config


def get_config(key: str, default: Any = None, scenario: str = None) -> Any:
    """
    Get a config value using dot notation.

    Args:
        key: Dot-separated key path (e.g., 'simulation.max_rounds_per_agent')
        default: Default value if key not found
        scenario: Scenario name to load from

    Returns:
        Config value or default.
    """
    config = load_scenario(scenario)

    parts = key.split(".")
    value = config

    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return default

    return value


def list_scenarios() -> list:
    """List available scenarios from all locations."""
    scenarios = set()

    # Legacy root scenarios
    if SCENARIOS_DIR.exists():
        for f in SCENARIOS_DIR.glob("*.yaml"):
            scenarios.add(f.stem)

    # Presets
    presets_dir = SCENARIOS_DIR / "presets"
    if presets_dir.exists():
        for f in presets_dir.glob("*.yaml"):
            scenarios.add(f"presets/{f.stem}")

    # Campaigns
    campaigns_dir = SCENARIOS_DIR / "campaigns"
    if campaigns_dir.exists():
        for f in campaigns_dir.glob("*.yaml"):
            scenarios.add(f"campaigns/{f.stem}")

    return sorted(scenarios)


def list_components(component_type: str, version: str = None) -> list:
    """
    List available components of a type.

    Args:
        component_type: Type (personas, tactics, topics, etc.)
        version: Optional version filter (v1, v2, etc.)

    Returns:
        List of component paths.
    """
    base_dir = COMPONENT_DIRS.get(component_type)
    if not base_dir or not base_dir.exists():
        return []

    components = []
    if version:
        version_dir = base_dir / version
        if version_dir.exists():
            for f in version_dir.glob("*.yaml"):
                components.append(f"{version}/{f.stem}")
    else:
        # List all versions
        for version_dir in base_dir.iterdir():
            if version_dir.is_dir():
                for f in version_dir.glob("*.yaml"):
                    components.append(f"{version_dir.name}/{f.stem}")

    return sorted(components)


def get_persona(name: str, scenario: str = None) -> Dict:
    """
    Get a specific persona by name.

    Args:
        name: Persona name (e.g., 'yudkowsky')
        scenario: Scenario to load personas from

    Returns:
        Persona configuration dict.
    """
    config = load_scenario(scenario)
    personas = config.get("personas", {})

    if isinstance(personas, dict):
        # Check direct personas dict
        if name in personas:
            return personas[name]

        # Check if there's a nested 'personas' key (from imports)
        for key, value in personas.items():
            if isinstance(value, dict) and "personas" in value:
                if name in value["personas"]:
                    return value["personas"][name]

    return {}


def get_all_personas(scenario: str = None) -> Dict[str, Dict]:
    """Get all personas from a scenario."""
    config = load_scenario(scenario)
    personas = config.get("personas", {})

    # Flatten nested persona structures
    result = {}
    if isinstance(personas, dict):
        for key, value in personas.items():
            if isinstance(value, dict):
                if "personas" in value:
                    # Nested structure from import
                    result.update(value["personas"])
                elif "name" in value or "role" in value:
                    # Direct persona definition
                    result[key] = value

    return result


def get_tactic(name: str, scenario: str = None) -> Dict:
    """Get a specific tactic by name."""
    config = load_scenario(scenario)
    tactics = config.get("tactics", {})

    # Navigate to tactics.tactics if it exists (from imports)
    if "tactics" in tactics:
        tactics = tactics["tactics"]

    return tactics.get(name, {})


def get_topic(name: str, scenario: str = None) -> Dict:
    """Get a specific topic by name."""
    config = load_scenario(scenario)
    topics = config.get("topics", {})

    # Navigate to topics.topics if it exists (from imports)
    if "topics" in topics:
        topics = topics["topics"]

    return topics.get(name, {})


def build_roster(scenario: str = None) -> Dict[str, Dict]:
    """
    Build the active roster of personas for a scenario.

    Merges persona definitions from the library with roster-specific
    overrides (model, team, starting_position).

    Args:
        scenario: Scenario to load roster from

    Returns:
        Dict mapping persona_id -> full persona config with overrides applied
    """
    config = load_scenario(scenario)
    roster_config = config.get("roster", {})

    # Get cast (active personas with overrides)
    cast = roster_config.get("cast", {})
    if not cast:
        # Fall back to active_personas list (legacy format)
        active = config.get("active_personas", [])
        cast = {name: {} for name in active}

    # Load base personas from source or imports
    source = roster_config.get("source")
    if source:
        # Load from explicit source
        source_path = SCENARIOS_DIR / source.lstrip("../")
        base_personas = load_yaml_file(source_path).get("personas", {})
    else:
        # Try to get from already-loaded config
        base_personas = get_all_personas(scenario)

    # Build final roster
    roster = {}
    budget_override = roster_config.get("budget_override", {})
    default_model = budget_override.get("default")
    premium_only = budget_override.get("premium_only", [])

    for persona_id, overrides in cast.items():
        # Start with base persona
        if persona_id in base_personas:
            persona = base_personas[persona_id].copy()
        else:
            logger.warning(f"Persona '{persona_id}' not found in library")
            persona = {"id": persona_id, "name": persona_id}

        # Apply overrides from cast
        if overrides:
            # Model override
            if "model" in overrides:
                persona["model"] = overrides["model"]
            elif default_model and persona_id not in premium_only:
                persona["model"] = default_model

            # Team assignment
            if "team" in overrides:
                persona["team"] = overrides["team"]

            # Starting position
            if "starting_position" in overrides:
                pos = overrides["starting_position"]
                persona["initial_position"] = pos.get("stance", "UNDECIDED")
                persona["confidence"] = pos.get("confidence", 0.5)
                if "topic" in pos:
                    persona["topic"] = pos["topic"]

        roster[persona_id] = persona

    return roster


def get_teams(scenario: str = None) -> Dict[str, Dict]:
    """Get team definitions from scenario."""
    config = load_scenario(scenario)
    return config.get("teams", {})


def get_matchup(scenario: str = None) -> Dict:
    """Get initial matchup configuration."""
    config = load_scenario(scenario)
    return config.get("initial_matchup", {})


def build_persona_state(persona_id: str, scenario: str = None) -> Dict:
    """
    Build complete persona state with psychological modeling.

    Merges:
    - Base persona definition
    - Roster overrides (model, team, position)
    - Psychology defaults
    - Initial state

    Returns a fully hydrated persona ready for simulation.
    """
    roster = build_roster(scenario)
    if persona_id not in roster:
        return {}

    persona = roster[persona_id].copy()
    config = load_scenario(scenario)

    # Load psychology defaults
    psychology = config.get("psychology", {})

    # Ensure Big Five defaults
    if "big_five" not in persona:
        persona["big_five"] = {
            "openness": 0.5,
            "conscientiousness": 0.5,
            "extraversion": 0.5,
            "agreeableness": 0.5,
            "neuroticism": 0.5,
        }

    # Ensure cognitive biases defaults
    if "biases" not in persona:
        persona["biases"] = {
            "confirmation_bias": 0.5,
            "in_group_bias": 0.4,
        }

    # Ensure belief network structure
    if "beliefs" not in persona:
        persona["beliefs"] = {"core": [], "derived": [], "policy": []}

    # Set initial emotional state
    if "emotional_state" not in persona:
        persona["emotional_state"] = "baseline"

    # Set reasoning style default
    if "reasoning_style" not in persona:
        persona["reasoning_style"] = "analytical"

    # Add computed properties
    persona["_computed"] = {
        "resistance_modifier": compute_resistance_modifier(persona),
        "persuadability": compute_persuadability(persona),
        "preferred_tactics": get_preferred_tactics(persona),
    }

    return persona


def compute_resistance_modifier(persona: Dict) -> float:
    """
    Compute resistance modifier based on personality and state.

    Higher = harder to persuade.
    """
    base = persona.get("resistance_score", 0.5)

    # Personality adjustments
    big_five = persona.get("big_five", {})
    # Low agreeableness = more resistant
    base += (0.5 - big_five.get("agreeableness", 0.5)) * 0.2
    # Low openness = more resistant
    base += (0.5 - big_five.get("openness", 0.5)) * 0.15
    # High conscientiousness = more resistant (demands proof)
    base += (big_five.get("conscientiousness", 0.5) - 0.5) * 0.1

    # Emotional state adjustments
    emotional_state = persona.get("emotional_state", "baseline")
    state_modifiers = {
        "defensive": 0.15,
        "frustrated": 0.2,
        "curious": -0.1,
        "respected": -0.05,
        "baseline": 0,
    }
    base += state_modifiers.get(emotional_state, 0)

    return max(0.0, min(1.0, base))


def compute_persuadability(persona: Dict) -> float:
    """
    Compute how persuadable this persona is (inverse of resistance).
    """
    resistance = compute_resistance_modifier(persona)
    return 1.0 - resistance


def get_preferred_tactics(persona: Dict) -> list:
    """
    Get preferred tactics based on personality and reasoning style.
    """
    reasoning_style = persona.get("reasoning_style", "analytical")
    style_tactics = {
        "analytical": ["logical_argument", "evidence_based", "reductio"],
        "intuitive": ["emotional_appeal", "social_proof", "reframe"],
        "dialectical": ["consensus", "reciprocity", "thought_experiment"],
        "pragmatic": ["evidence_based", "competitive", "scarcity"],
        "ideological": ["moral_appeal", "commitment", "fear_appeal"],
    }

    tactics = style_tactics.get(reasoning_style, [])

    # Add from persona definition
    if "debate_tactics" in persona:
        if "primary" in persona["debate_tactics"]:
            tactics = persona["debate_tactics"]["primary"] + tactics

    return list(dict.fromkeys(tactics))  # Remove duplicates, preserve order


def get_trust_matrix(scenario: str = None) -> Dict[str, Dict[str, float]]:
    """Get the trust matrix between personas."""
    config = load_scenario(scenario)
    return config.get("trust_matrix", {})


def get_engine_settings(scenario: str = None) -> Dict:
    """Get simulation engine settings."""
    config = load_scenario(scenario)
    return config.get("engine", {})


def get_psychology_config(scenario: str = None) -> Dict:
    """Get psychology modeling configuration."""
    config = load_scenario(scenario)
    return config.get("psychology", load_component("personas", "psychology", "v2"))


def get_strategy_config(scenario: str = None) -> Dict:
    """Get strategic reasoning configuration."""
    return load_component("tactics", "strategy", "v2")


# Convenience functions for common config access
def get_simulation_config(scenario: str = None) -> Dict:
    """Get simulation configuration."""
    return get_config("simulation", {}, scenario)


def get_models_config(scenario: str = None) -> Dict:
    """Get models configuration."""
    return get_config("models", {}, scenario)


def get_tactics_config(scenario: str = None) -> Dict:
    """Get tactics configuration."""
    return get_config("tactics", {}, scenario)


def get_topics(scenario: str = None) -> list:
    """Get debate topics."""
    return get_config("topics", [], scenario)


# Module-level config (lazy loaded)
_config: Optional[Dict] = None


def get_world_config() -> Dict:
    """Get the current world configuration (cached)."""
    global _config
    if _config is None:
        _config = load_scenario()
    return _config


def reload_config(scenario: str = None):
    """Force reload configuration."""
    global _config
    load_scenario.cache_clear()
    _config = load_scenario(scenario)
    return _config
