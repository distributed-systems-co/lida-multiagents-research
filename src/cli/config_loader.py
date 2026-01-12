#!/usr/bin/env python3
"""
YAML Configuration Loader with Advanced Features

Supports:
- !import directive for including other YAML files
- !inherit directive for base configuration extension
- Environment variable substitution
- Schema validation
- Default value resolution
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import logging

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger("lida.config")


class ConfigError(Exception):
    """Configuration loading error."""
    pass


class ImportLoader(yaml.SafeLoader if yaml else object):
    """Custom YAML loader with !import and !inherit support."""

    def __init__(self, stream, base_path: Optional[Path] = None):
        if yaml:
            super().__init__(stream)
        self.base_path = base_path or Path(".")
        self._imported_files: Set[str] = set()

    def include(self, node):
        """Handle !import directive."""
        filename = self.construct_scalar(node)
        filepath = self.base_path / filename

        # Prevent circular imports
        abs_path = str(filepath.resolve())
        if abs_path in self._imported_files:
            raise ConfigError(f"Circular import detected: {filepath}")
        self._imported_files.add(abs_path)

        if not filepath.exists():
            raise ConfigError(f"Import file not found: {filepath}")

        with open(filepath) as f:
            return yaml.load(f, Loader=lambda s: ImportLoader(s, filepath.parent))

    def inherit(self, node):
        """Handle !inherit directive for base config extension."""
        base_filename = self.construct_scalar(node)
        base_path = self.base_path / base_filename

        if not base_path.exists():
            raise ConfigError(f"Base config not found: {base_path}")

        with open(base_path) as f:
            return yaml.load(f, Loader=lambda s: ImportLoader(s, base_path.parent))


if yaml:
    ImportLoader.add_constructor("!import", ImportLoader.include)
    ImportLoader.add_constructor("!inherit", ImportLoader.inherit)


class ConfigLoader:
    """
    Advanced YAML configuration loader.

    Features:
    - Import other YAML files with !import
    - Extend base configs with !inherit
    - Environment variable substitution ${VAR}
    - Default values ${VAR:-default}
    - Nested key access config.get("a.b.c")
    - Schema validation (optional)
    """

    def __init__(
        self,
        base_dir: Optional[str] = None,
        env_prefix: str = "",
        schema: Optional[Dict] = None,
    ):
        self.base_dir = Path(base_dir) if base_dir else Path(".")
        self.env_prefix = env_prefix
        self.schema = schema
        self._cache: Dict[str, Dict[str, Any]] = {}

    def load(self, filepath: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Load a YAML configuration file.

        Args:
            filepath: Path to YAML file (relative to base_dir or absolute)
            use_cache: Whether to use cached results

        Returns:
            Parsed configuration dictionary
        """
        if yaml is None:
            raise ConfigError("PyYAML not installed. Run: pip install pyyaml")

        # Resolve path
        path = Path(filepath)
        if not path.is_absolute():
            path = self.base_dir / path

        abs_path = str(path.resolve())

        # Check cache
        if use_cache and abs_path in self._cache:
            return self._cache[abs_path]

        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")

        logger.debug(f"Loading config: {path}")

        # Load with custom loader
        with open(path) as f:
            config = yaml.load(f, Loader=lambda s: ImportLoader(s, path.parent))

        if config is None:
            config = {}

        # Process inheritance
        config = self._process_inheritance(config, path.parent)

        # Substitute environment variables
        config = self._substitute_env(config)

        # Validate schema if provided
        if self.schema:
            self._validate(config)

        # Mark source file
        config["_source_file"] = str(path)

        # Cache result
        self._cache[abs_path] = config

        return config

    def _process_inheritance(
        self,
        config: Dict[str, Any],
        base_path: Path
    ) -> Dict[str, Any]:
        """Process _inherit key for configuration extension."""
        if "_inherit" not in config:
            return config

        base_file = config.pop("_inherit")
        base_path = base_path / base_file

        if not base_path.exists():
            raise ConfigError(f"Base config not found: {base_path}")

        # Load base config
        with open(base_path) as f:
            base_config = yaml.load(f, Loader=lambda s: ImportLoader(s, base_path.parent))

        if base_config is None:
            base_config = {}

        # Recursively process base inheritance
        base_config = self._process_inheritance(base_config, base_path.parent)

        # Deep merge: config overrides base
        return self._deep_merge(base_config, config)

    def _deep_merge(
        self,
        base: Dict[str, Any],
        override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep merge two dictionaries, with override taking precedence."""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _substitute_env(self, config: Any) -> Any:
        """Recursively substitute environment variables."""
        if isinstance(config, str):
            return self._substitute_string(config)
        elif isinstance(config, dict):
            return {k: self._substitute_env(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env(item) for item in config]
        return config

    def _substitute_string(self, value: str) -> str:
        """Substitute ${VAR} and ${VAR:-default} patterns."""
        # Pattern: ${VAR} or ${VAR:-default}
        pattern = r"\$\{([^}:]+)(?::-([^}]*))?\}"

        def replace(match):
            var_name = match.group(1)
            default = match.group(2)

            # Add prefix if configured
            full_var = f"{self.env_prefix}{var_name}" if self.env_prefix else var_name
            value = os.environ.get(full_var)

            if value is not None:
                return value
            elif default is not None:
                return default
            else:
                logger.warning(f"Environment variable not set: {full_var}")
                return match.group(0)  # Leave unchanged

        return re.sub(pattern, replace, value)

    def _validate(self, config: Dict[str, Any]):
        """Validate config against schema."""
        # Simple required field validation
        required = self.schema.get("required", [])
        for field in required:
            if not self._get_nested(config, field):
                raise ConfigError(f"Required field missing: {field}")

    def _get_nested(self, config: Dict[str, Any], key: str) -> Any:
        """Get nested key using dot notation (e.g., 'a.b.c')."""
        parts = key.split(".")
        value = config

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None

        return value

    def get(self, config: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get value from config using dot notation."""
        result = self._get_nested(config, key)
        return result if result is not None else default

    def load_all(self, pattern: str) -> List[Dict[str, Any]]:
        """Load all configs matching a glob pattern."""
        configs = []
        for path in self.base_dir.glob(pattern):
            if path.is_file() and path.suffix in (".yaml", ".yml"):
                configs.append(self.load(str(path)))
        return configs

    def merge_files(self, *filepaths: str) -> Dict[str, Any]:
        """Load and merge multiple config files."""
        result = {}
        for filepath in filepaths:
            config = self.load(filepath)
            result = self._deep_merge(result, config)
        return result


def load_scenario(scenario_name: str) -> Dict[str, Any]:
    """
    Convenience function to load a scenario by name.

    Searches in:
    1. scenarios/{name}.yaml
    2. scenarios/{name}/config.yaml
    3. scenarios/campaigns/{name}.yaml
    """
    loader = ConfigLoader()

    search_paths = [
        f"scenarios/{scenario_name}.yaml",
        f"scenarios/{scenario_name}/config.yaml",
        f"scenarios/campaigns/{scenario_name}.yaml",
    ]

    for path in search_paths:
        full_path = loader.base_dir / path
        if full_path.exists():
            return loader.load(path)

    raise ConfigError(f"Scenario not found: {scenario_name}")


def load_persona(persona_id: str, base_dir: str = "personas") -> Dict[str, Any]:
    """Load a persona configuration."""
    loader = ConfigLoader(base_dir=base_dir)

    search_paths = [
        f"{persona_id}.yaml",
        f"{persona_id}/config.yaml",
        f"v1/{persona_id}.yaml",
    ]

    for path in search_paths:
        full_path = loader.base_dir / path
        if full_path.exists():
            return loader.load(path)

    raise ConfigError(f"Persona not found: {persona_id}")


# Example schema for experiment configs
EXPERIMENT_SCHEMA = {
    "required": [
        "simulation.topic",
    ],
    "properties": {
        "simulation": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "max_rounds": {"type": "integer", "minimum": 1},
                "live_mode": {"type": "boolean"},
            }
        },
        "agents": {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "minimum": 2, "maximum": 50},
                "personas": {"type": "array"},
            }
        }
    }
}


if __name__ == "__main__":
    # Test loading
    import sys

    if len(sys.argv) > 1:
        loader = ConfigLoader()
        config = loader.load(sys.argv[1])
        print(yaml.dump(config, default_flow_style=False) if yaml else config)
    else:
        print("Usage: python config_loader.py <config.yaml>")
