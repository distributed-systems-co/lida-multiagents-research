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
