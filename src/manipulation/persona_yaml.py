"""
Persona YAML Export/Import Utilities

Provides versioned YAML representation of personas for:
- Easier editing and review
- Version control friendly format
- Developer convenience
- Cross-language compatibility

Usage:
    # Export all personas to YAML
    python -m src.manipulation.persona_yaml export

    # Export specific persona
    python -m src.manipulation.persona_yaml export --persona jensen_huang

    # Load personas from YAML
    from src.manipulation.persona_yaml import YAMLPersonaLibrary
    lib = YAMLPersonaLibrary(version="v1")
"""

from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict, fields, is_dataclass
from enum import Enum

from .personas import (
    Persona,
    PersonaLibrary,
    MaslowNeed,
    PersonalityTrait,
    RhetoricalStyle,
    ArgumentationMode,
    EpistemicStyle,
    ConflictStyle,
    InfluenceType,
    CognitiveProfile,
    RhetoricalProfile,
    EmotionalProfile,
    WorldviewModel,
    InfluenceNetwork,
    InfluenceRelationship,
    StancePosition,
)

# Version for the YAML schema
SCHEMA_VERSION = "1.0"
PERSONAS_DIR = Path(__file__).parent / "personas"


def enum_representer(dumper, data):
    """Custom YAML representer for Enum values."""
    return dumper.represent_scalar('tag:yaml.org,2002:str', data.value)


def setup_yaml():
    """Configure YAML for proper enum and dataclass handling."""
    yaml.add_representer(MaslowNeed, enum_representer)
    yaml.add_representer(PersonalityTrait, enum_representer)
    yaml.add_representer(RhetoricalStyle, enum_representer)
    yaml.add_representer(ArgumentationMode, enum_representer)
    yaml.add_representer(EpistemicStyle, enum_representer)
    yaml.add_representer(ConflictStyle, enum_representer)
    yaml.add_representer(InfluenceType, enum_representer)


setup_yaml()


def dataclass_to_dict(obj: Any) -> Any:
    """Convert dataclass to dict, handling nested structures and enums."""
    if obj is None:
        return None
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(item) for item in obj]
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: dataclass_to_dict(v) for k, v in asdict(obj).items()}
    return obj


def persona_to_yaml_dict(persona: Persona) -> Dict[str, Any]:
    """Convert a Persona to a YAML-friendly dictionary."""
    data = {
        "schema_version": SCHEMA_VERSION,
        "id": persona.id,
        "name": persona.name,
        "role": persona.role,
        "organization": persona.organization,
        "category": persona.category,
        "bio": persona.bio,
        "background": persona.background,
        "achievements": persona.achievements,

        "motivation": {
            "primary_need": persona.primary_need.value if persona.primary_need else None,
            "secondary_need": persona.secondary_need.value if persona.secondary_need else None,
            "explicit_goals": persona.explicit_goals,
            "hidden_goals": persona.hidden_goals,
        },

        "personality": {
            trait.value: score
            for trait, score in (persona.personality or {}).items()
        },

        "cognitive": dataclass_to_dict(persona.cognitive) if persona.cognitive else None,
        "rhetorical": dataclass_to_dict(persona.rhetorical) if persona.rhetorical else None,
        "emotional": dataclass_to_dict(persona.emotional) if persona.emotional else None,
        "worldview": dataclass_to_dict(persona.worldview) if persona.worldview else None,
        "influence_network": dataclass_to_dict(persona.influence_network) if persona.influence_network else None,

        "stances": {
            topic: dataclass_to_dict(stance)
            for topic, stance in (persona.stances or {}).items()
        } if persona.stances else None,

        "epistemic_style": persona.epistemic_style.value if persona.epistemic_style else None,
        "conflict_style": persona.conflict_style.value if persona.conflict_style else None,

        "positions": persona.positions,
        "persuasion_vectors": persona.persuasion_vectors,
    }

    # Remove None values for cleaner YAML
    return {k: v for k, v in data.items() if v is not None}


def yaml_dict_to_persona(data: Dict[str, Any]) -> Persona:
    """Convert a YAML dictionary back to a Persona object."""

    # Parse personality traits
    personality = {}
    if data.get("personality"):
        for trait_str, score in data["personality"].items():
            trait = PersonalityTrait(trait_str)
            personality[trait] = score

    # Parse cognitive profile
    cognitive = None
    if data.get("cognitive"):
        cognitive = CognitiveProfile(**data["cognitive"])

    # Parse rhetorical profile
    rhetorical = None
    if data.get("rhetorical"):
        rh_data = data["rhetorical"].copy()
        if "primary_mode" in rh_data:
            rh_data["primary_mode"] = RhetoricalStyle(rh_data["primary_mode"])
        if "secondary_mode" in rh_data:
            rh_data["secondary_mode"] = RhetoricalStyle(rh_data["secondary_mode"])
        if "argumentation_mode" in rh_data:
            rh_data["argumentation_mode"] = ArgumentationMode(rh_data["argumentation_mode"])
        rhetorical = RhetoricalProfile(**rh_data)

    # Parse emotional profile
    emotional = None
    if data.get("emotional"):
        emotional = EmotionalProfile(**data["emotional"])

    # Parse worldview
    worldview = None
    if data.get("worldview"):
        worldview = WorldviewModel(**data["worldview"])

    # Parse influence network
    influence_network = None
    if data.get("influence_network"):
        net_data = data["influence_network"].copy()
        if "relationships" in net_data:
            relationships = []
            for rel in net_data["relationships"]:
                rel_copy = rel.copy()
                rel_copy["influence_type"] = InfluenceType(rel_copy["influence_type"])
                relationships.append(InfluenceRelationship(**rel_copy))
            net_data["relationships"] = relationships
        influence_network = InfluenceNetwork(**net_data)

    # Parse stances
    stances = {}
    if data.get("stances"):
        for topic, stance_data in data["stances"].items():
            stances[topic] = StancePosition(**stance_data)

    # Parse motivation
    motivation = data.get("motivation", {})
    primary_need = MaslowNeed(motivation["primary_need"]) if motivation.get("primary_need") else MaslowNeed.ESTEEM
    secondary_need = MaslowNeed(motivation["secondary_need"]) if motivation.get("secondary_need") else None

    return Persona(
        id=data["id"],
        name=data["name"],
        role=data.get("role", ""),
        organization=data.get("organization", ""),
        category=data.get("category", "custom"),
        bio=data.get("bio", ""),
        background=data.get("background", ""),
        achievements=data.get("achievements", []),
        primary_need=primary_need,
        secondary_need=secondary_need,
        explicit_goals=motivation.get("explicit_goals", []),
        hidden_goals=motivation.get("hidden_goals", []),
        personality=personality,
        cognitive=cognitive,
        rhetorical=rhetorical,
        emotional=emotional,
        worldview=worldview,
        influence_network=influence_network,
        stances=stances,
        epistemic_style=EpistemicStyle(data["epistemic_style"]) if data.get("epistemic_style") else None,
        conflict_style=ConflictStyle(data["conflict_style"]) if data.get("conflict_style") else None,
        positions=data.get("positions", {}),
        persuasion_vectors=data.get("persuasion_vectors", []),
    )


def get_category_dir(category: str) -> str:
    """Map category to directory name."""
    category_map = {
        "ceo": "ceos",
        "researcher": "researchers",
        "ai_researcher": "researchers",
        "ai_safety": "researchers",
        "politician": "politicians",
        "investor": "investors",
        "journalist": "journalists",
        "activist": "activists",
        "cyber": "researchers",
    }
    return category_map.get(category, "other")


def export_persona_to_yaml(persona: Persona, version: str = "v1") -> Path:
    """Export a single persona to YAML file."""
    yaml_dict = persona_to_yaml_dict(persona)

    category_dir = get_category_dir(persona.category)
    output_dir = PERSONAS_DIR / version / category_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{persona.id}.yaml"

    with open(output_path, "w") as f:
        yaml.dump(yaml_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True, width=100)

    return output_path


def export_all_personas(version: str = "v1") -> List[Path]:
    """Export all personas to YAML files."""
    library = PersonaLibrary()
    paths = []

    for persona_id in library.list_all():
        persona = library.get(persona_id)
        if persona:
            path = export_persona_to_yaml(persona, version)
            paths.append(path)

    # Create version manifest
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "version": version,
        "total_personas": len(paths),
        "categories": {},
    }

    for path in paths:
        category = path.parent.name
        if category not in manifest["categories"]:
            manifest["categories"][category] = []
        manifest["categories"][category].append(path.stem)

    manifest_path = PERSONAS_DIR / version / "manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

    return paths


def load_persona_from_yaml(filepath: Path) -> Persona:
    """Load a persona from a YAML file."""
    with open(filepath) as f:
        data = yaml.safe_load(f)
    return yaml_dict_to_persona(data)


class YAMLPersonaLibrary:
    """Load personas from YAML files instead of Python code."""

    def __init__(self, version: str = "v1"):
        self.version = version
        self.base_dir = PERSONAS_DIR / version
        self.personas: Dict[str, Persona] = {}
        self._load_all()

    def _load_all(self):
        """Load all personas from YAML files."""
        if not self.base_dir.exists():
            return

        for yaml_file in self.base_dir.rglob("*.yaml"):
            if yaml_file.name == "manifest.yaml":
                continue
            try:
                persona = load_persona_from_yaml(yaml_file)
                self.personas[persona.id] = persona
            except Exception as e:
                print(f"Warning: Failed to load {yaml_file}: {e}")

    def get(self, persona_id: str) -> Optional[Persona]:
        return self.personas.get(persona_id)

    def list_all(self) -> List[str]:
        return list(self.personas.keys())

    def list_by_category(self, category: str) -> List[Persona]:
        return [p for p in self.personas.values() if p.category == category]

    def reload(self):
        """Reload all personas from disk."""
        self.personas.clear()
        self._load_all()


def main():
    """CLI for persona YAML operations."""
    import argparse

    parser = argparse.ArgumentParser(description="Persona YAML utilities")
    parser.add_argument("command", choices=["export", "validate", "list"])
    parser.add_argument("--version", default="v1", help="Version directory")
    parser.add_argument("--persona", help="Specific persona ID to export")

    args = parser.parse_args()

    if args.command == "export":
        if args.persona:
            library = PersonaLibrary()
            persona = library.get(args.persona)
            if persona:
                path = export_persona_to_yaml(persona, args.version)
                print(f"Exported: {path}")
            else:
                print(f"Persona not found: {args.persona}")
        else:
            paths = export_all_personas(args.version)
            print(f"Exported {len(paths)} personas to {PERSONAS_DIR / args.version}")

    elif args.command == "validate":
        yaml_lib = YAMLPersonaLibrary(args.version)
        print(f"Loaded {len(yaml_lib.list_all())} personas from YAML")
        for pid in yaml_lib.list_all():
            p = yaml_lib.get(pid)
            print(f"  {pid}: {p.name} ({p.category})")

    elif args.command == "list":
        yaml_lib = YAMLPersonaLibrary(args.version)
        for pid in sorted(yaml_lib.list_all()):
            p = yaml_lib.get(pid)
            print(f"{pid}: {p.name} - {p.role}")


if __name__ == "__main__":
    main()
