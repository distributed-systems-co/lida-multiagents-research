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


@lru_cache(maxsize=16)
def load_scenario(name: str = None) -> Dict[str, Any]:
    """
    Load a scenario configuration.

    Args:
        name: Scenario name (without .yaml) or path to yaml file.
              If None, uses SCENARIO env var or 'default'.

    Returns:
        Merged configuration dictionary.
    """
    # Determine scenario to load
    if name is None:
        name = os.getenv("SCENARIO", "default")

    # Load default first
    default_path = SCENARIOS_DIR / DEFAULT_SCENARIO
    config = {}

    if default_path.exists():
        with open(default_path) as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"Loaded default scenario from {default_path}")

    # If requesting non-default, merge on top
    if name != "default":
        # Check if it's a path or just a name
        if name.endswith(".yaml") or "/" in name:
            scenario_path = Path(name)
        else:
            scenario_path = SCENARIOS_DIR / f"{name}.yaml"

        if scenario_path.exists():
            with open(scenario_path) as f:
                override = yaml.safe_load(f) or {}
            config = deep_merge(config, override)
            logger.info(f"Merged scenario '{name}' from {scenario_path}")
        else:
            logger.warning(f"Scenario '{name}' not found at {scenario_path}")

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
    """List available scenarios."""
    scenarios = []
    if SCENARIOS_DIR.exists():
        for f in SCENARIOS_DIR.glob("*.yaml"):
            scenarios.append(f.stem)
    return sorted(scenarios)


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
