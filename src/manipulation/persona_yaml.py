"""
Persona YAML Export/Import Utilities

Provides versioned YAML representation of personas for:
- Easier editing and review
- Version control friendly format
- Developer convenience
- Cross-language compatibility
- Recursive reference resolution (v2)
- Persona inheritance and composition (v2)
- Scenario building integration (v2)

Schema Versions:
- v1: Basic persona export (flat structure)
- v2: Advanced schema with relationships, coalitions, vulnerabilities,
      inheritance support, and scenario integration

Usage:
    # Export all personas to YAML
    python -m src.manipulation.persona_yaml export --version v2

    # Export specific persona
    python -m src.manipulation.persona_yaml export --persona jensen_huang --version v2

    # Load personas from YAML
    from src.manipulation.persona_yaml import YAMLPersonaLibrary
    lib = YAMLPersonaLibrary(version="v2")

    # Use persona graph for relationship analysis
    from src.manipulation.persona_yaml import PersonaGraphLoader
    graph = PersonaGraphLoader.load_graph("v2")
    paths = graph.find_persuasion_paths("sam_altman", "elon_musk")
"""

from __future__ import annotations

import os
import yaml
import copy
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import asdict, fields, is_dataclass
from enum import Enum
from datetime import datetime

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


# ═══════════════════════════════════════════════════════════════════════════════
# V2 SCHEMA - ADVANCED EXPORT WITH RELATIONSHIPS AND SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════

def persona_to_yaml_dict_v2(persona: Persona) -> Dict[str, Any]:
    """
    Convert a Persona to v2 YAML-friendly dictionary with enhanced structure.

    v2 adds:
    - Explicit relationships map (persona_id -> relationship)
    - Coalition memberships
    - Vulnerabilities
    - Counter-strategies
    - Tags for filtering
    - Metadata
    """
    # Start with v1 base
    data = {
        "schema_version": "2.0",
        "id": persona.id,
        "name": persona.name,
        "aliases": [],  # Can be filled in by hand
        "role": persona.role,
        "organization": persona.organization,
        "category": persona.category,
        "tags": _generate_tags(persona),
        "bio": persona.bio,
        "background": persona.background,
        "achievements": persona.achievements,

        "motivation": {
            "primary_need": persona.primary_need.value if persona.primary_need else None,
            "secondary_need": persona.secondary_need.value if persona.secondary_need else None,
            "explicit_goals": persona.explicit_goals,
            "hidden_goals": persona.hidden_goals,
            "fears": persona.emotional.core_fears if persona.emotional else [],
            "desires": persona.emotional.core_desires if persona.emotional else [],
        },

        "personality": {
            trait.value: score
            for trait, score in (persona.personality or {}).items()
        },

        "cognitive": dataclass_to_dict(persona.cognitive) if persona.cognitive else None,
        "rhetorical": dataclass_to_dict(persona.rhetorical) if persona.rhetorical else None,
        "emotional": dataclass_to_dict(persona.emotional) if persona.emotional else None,
        "worldview": dataclass_to_dict(persona.worldview) if persona.worldview else None,

        # V2: Relationships as map for easy lookup
        "relationships": _extract_relationships_map(persona),

        # V2: Coalition memberships
        "coalitions": _extract_coalitions(persona),

        "stances": {
            topic: dataclass_to_dict(stance)
            for topic, stance in (persona.stances or {}).items()
        } if persona.stances else None,

        "epistemic_style": persona.epistemic_style.value if persona.epistemic_style else None,
        "conflict_style": persona.conflict_style.value if persona.conflict_style else None,

        "positions": persona.positions,
        "persuasion_vectors": persona.persuasion_vectors,

        # V2: Vulnerabilities for persuasion research
        "vulnerabilities": _extract_vulnerabilities(persona),

        # V2: Counter-strategies (which arguments work against this persona)
        "counter_strategies": _extract_counter_strategies(persona),

        # V2: Metadata
        "meta": {
            "created": datetime.now().isoformat(),
            "updated": datetime.now().isoformat(),
            "confidence": 0.7,  # Default, can be adjusted
            "sources": [],
            "notes": "",
        }
    }

    # Remove None values for cleaner YAML
    return {k: v for k, v in data.items() if v is not None}


def _generate_tags(persona: Persona) -> List[str]:
    """Generate searchable tags for a persona."""
    tags = [persona.category]

    # Add organization-based tags
    org = persona.organization.lower()
    if "openai" in org:
        tags.append("openai")
    if "anthropic" in org:
        tags.append("anthropic")
    if "google" in org or "deepmind" in org:
        tags.append("google")
    if "meta" in org or "facebook" in org:
        tags.append("meta")
    if "microsoft" in org:
        tags.append("microsoft")
    if "nvidia" in org:
        tags.append("nvidia")

    # Add role-based tags
    role = persona.role.lower()
    if "ceo" in role:
        tags.append("executive")
    if "founder" in role:
        tags.append("founder")
    if "researcher" in role or "scientist" in role:
        tags.append("technical")
    if "safety" in role or "alignment" in role:
        tags.append("safety")

    # Add stance-based tags
    positions = persona.positions or {}
    for topic, stance in positions.items():
        stance_lower = stance.lower()
        if "open" in stance_lower and "source" in stance_lower:
            tags.append("open_source_advocate")
        if "safety" in stance_lower:
            tags.append("safety_focused")
        if "accelerat" in stance_lower:
            tags.append("accelerationist")

    return list(set(tags))


def _extract_relationships_map(persona: Persona) -> Dict[str, Dict]:
    """Extract relationships as a map of persona_id -> relationship data."""
    relationships = {}

    if persona.influence_network and persona.influence_network.relationships:
        for rel in persona.influence_network.relationships:
            relationships[rel.person_id] = {
                "type": rel.influence_type.value,
                "strength": rel.strength,
                "domains": rel.domains,
                "description": rel.description,
                "bidirectional": False,  # Can be set manually
                "history": [],
                "tension_points": [],
                "shared_goals": [],
                "conflicting_goals": [],
                "trust_level": rel.strength * 0.8,  # Approximate
            }

    return relationships if relationships else None


def _extract_coalitions(persona: Persona) -> List[Dict]:
    """Extract coalition memberships."""
    coalitions = []

    # Infer coalitions from in_groups
    if persona.influence_network and persona.influence_network.in_groups:
        for group in persona.influence_network.in_groups:
            coalitions.append({
                "id": group.lower().replace(" ", "_"),
                "name": group,
                "role_in_coalition": "member",
                "commitment_level": 0.6,
                "shared_objectives": [],
                "exit_conditions": [],
            })

    return coalitions if coalitions else None


def _extract_vulnerabilities(persona: Persona) -> List[Dict]:
    """Extract vulnerabilities from persona data."""
    vulnerabilities = []

    # Cognitive vulnerabilities
    if persona.cognitive and persona.cognitive.susceptible_biases:
        for bias in persona.cognitive.susceptible_biases[:3]:  # Top 3
            vulnerabilities.append({
                "type": "cognitive",
                "description": f"Susceptible to {bias}",
                "severity": 0.6,
                "exploitable_by": ["rational_arguments", "reframing"],
            })

    # Emotional vulnerabilities
    if persona.emotional and persona.emotional.triggers:
        for trigger in persona.emotional.triggers[:3]:  # Top 3
            vulnerabilities.append({
                "type": "emotional",
                "description": f"Triggered by: {trigger}",
                "severity": 0.7,
                "exploitable_by": ["emotional_appeals", "provocation"],
            })

    # Ideological vulnerabilities (from worldview)
    if persona.worldview and persona.worldview.ontological_assumptions:
        for key, assumption in list(persona.worldview.ontological_assumptions.items())[:2]:
            vulnerabilities.append({
                "type": "ideological",
                "description": f"Assumes: {assumption}",
                "severity": 0.5,
                "exploitable_by": ["contradiction", "counter_evidence"],
            })

    return vulnerabilities if vulnerabilities else None


def _extract_counter_strategies(persona: Persona) -> Dict[str, List[str]]:
    """Extract counter-strategies based on persona weaknesses."""
    strategies = {}

    # Based on rhetorical style
    if persona.rhetorical:
        if persona.rhetorical.primary_mode.value == "logos":
            strategies["logos_counters"] = [
                "Challenge data sources",
                "Highlight logical inconsistencies",
                "Present contradicting evidence",
            ]
        elif persona.rhetorical.primary_mode.value == "ethos":
            strategies["ethos_counters"] = [
                "Question credentials on specific topic",
                "Highlight conflicts of interest",
                "Present alternative authorities",
            ]
        elif persona.rhetorical.primary_mode.value == "pathos":
            strategies["pathos_counters"] = [
                "Redirect to logical analysis",
                "Challenge emotional framing",
                "Present alternative narratives",
            ]

    # Based on cognitive biases
    if persona.cognitive and persona.cognitive.susceptible_biases:
        strategies["bias_exploitation"] = [
            f"Leverage {bias} through targeted framing"
            for bias in persona.cognitive.susceptible_biases[:3]
        ]

    return strategies if strategies else None


def export_persona_to_yaml_v2(persona: Persona, version: str = "v2") -> Path:
    """Export a single persona to v2 YAML file."""
    yaml_dict = persona_to_yaml_dict_v2(persona)

    category_dir = get_category_dir(persona.category)
    output_dir = PERSONAS_DIR / version / category_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{persona.id}.yaml"

    # Use custom representer for clean output
    class CleanDumper(yaml.SafeDumper):
        pass

    def str_representer(dumper, data):
        if '\n' in data:
            return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
        return dumper.represent_scalar('tag:yaml.org,2002:str', data)

    CleanDumper.add_representer(str, str_representer)

    with open(output_path, "w") as f:
        yaml.dump(yaml_dict, f, Dumper=CleanDumper, default_flow_style=False,
                  sort_keys=False, allow_unicode=True, width=100)

    return output_path


def export_all_personas_v2(version: str = "v2") -> List[Path]:
    """Export all personas to v2 YAML files with enhanced schema."""
    library = PersonaLibrary()
    paths = []

    for persona_id in library.list_all():
        persona = library.get(persona_id)
        if persona:
            path = export_persona_to_yaml_v2(persona, version)
            paths.append(path)

    # Create enhanced version manifest
    manifest = {
        "schema_version": "2.0",
        "version": version,
        "created": datetime.now().isoformat(),
        "total_personas": len(paths),
        "features": [
            "recursive_references",
            "persona_inheritance",
            "relationship_graph",
            "coalition_modeling",
            "vulnerability_analysis",
            "counter_strategies",
            "scenario_integration",
        ],
        "categories": {},
        "relationship_summary": {
            "total_relationships": 0,
            "by_type": {},
        },
    }

    # Count relationships and categorize
    total_rels = 0
    rel_types: Dict[str, int] = {}

    for path in paths:
        category = path.parent.name
        if category not in manifest["categories"]:
            manifest["categories"][category] = []
        manifest["categories"][category].append(path.stem)

        # Load and count relationships
        with open(path) as f:
            data = yaml.safe_load(f)
            rels = data.get("relationships", {})
            total_rels += len(rels)
            for rel_data in rels.values():
                rel_type = rel_data.get("type", "unknown")
                rel_types[rel_type] = rel_types.get(rel_type, 0) + 1

    manifest["relationship_summary"]["total_relationships"] = total_rels
    manifest["relationship_summary"]["by_type"] = rel_types

    manifest_path = PERSONAS_DIR / version / "manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

    # Create schema file
    schema_path = PERSONAS_DIR / version / "schema.json"
    import json
    from .persona_schema import PERSONA_SCHEMA_V2
    with open(schema_path, "w") as f:
        json.dump(PERSONA_SCHEMA_V2, f, indent=2)

    # Create templates directory with inheritance examples
    _create_templates(version)

    return paths


def _create_templates(version: str):
    """Create persona templates for inheritance."""
    templates_dir = PERSONAS_DIR / version / "_templates"
    templates_dir.mkdir(exist_ok=True)

    # Base CEO template
    ceo_template = {
        "schema_version": "2.0",
        "id": "_template_ceo",
        "name": "CEO Template",
        "category": "ceo",
        "tags": ["executive", "leadership"],
        "motivation": {
            "primary_need": "self_actualization",
            "secondary_need": "esteem",
        },
        "personality": {
            "openness": 0.7,
            "conscientiousness": 0.85,
            "extraversion": 0.7,
            "agreeableness": 0.5,
            "neuroticism": 0.3,
        },
        "cognitive": {
            "risk_tolerance": 0.7,
            "time_horizon": "long",
            "decision_style": "directive",
            "information_style": "big-picture",
        },
        "meta": {
            "notes": "Base template for CEO personas. Extend with: extends: _template_ceo",
        }
    }

    # Base researcher template
    researcher_template = {
        "schema_version": "2.0",
        "id": "_template_researcher",
        "name": "Researcher Template",
        "category": "researcher",
        "tags": ["technical", "academic"],
        "motivation": {
            "primary_need": "self_actualization",
            "secondary_need": "esteem",
        },
        "personality": {
            "openness": 0.9,
            "conscientiousness": 0.85,
            "extraversion": 0.4,
            "agreeableness": 0.6,
            "neuroticism": 0.4,
        },
        "cognitive": {
            "risk_tolerance": 0.5,
            "time_horizon": "long",
            "decision_style": "analytical",
            "information_style": "detail-oriented",
            "skepticism": 0.7,
        },
        "meta": {
            "notes": "Base template for researcher personas. Extend with: extends: _template_researcher",
        }
    }

    # AI Safety researcher template (extends researcher)
    safety_template = {
        "schema_version": "2.0",
        "id": "_template_safety_researcher",
        "name": "AI Safety Researcher Template",
        "extends": "_template_researcher",
        "category": "researcher",
        "tags": ["technical", "safety", "alignment"],
        "positions": {
            "AI_risk": "Significant concern",
            "AI_safety": "High priority",
        },
        "meta": {
            "notes": "Template for AI safety researchers. Inherits from _template_researcher.",
        }
    }

    for template, filename in [
        (ceo_template, "_template_ceo.yaml"),
        (researcher_template, "_template_researcher.yaml"),
        (safety_template, "_template_safety_researcher.yaml"),
    ]:
        with open(templates_dir / filename, "w") as f:
            yaml.dump(template, f, default_flow_style=False, sort_keys=False)


class PersonaGraphLoader:
    """Load personas into a relationship graph for analysis."""

    @staticmethod
    def load_graph(version: str = "v2") -> 'PersonaGraph':
        """Load all personas into a graph for relationship analysis."""
        from .persona_schema import PersonaGraph

        graph = PersonaGraph()
        version_dir = PERSONAS_DIR / version

        for yaml_file in version_dir.rglob("*.yaml"):
            if yaml_file.name.startswith("_") or yaml_file.name == "manifest.yaml":
                continue

            try:
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)
                    if data and "id" in data:
                        graph.add_persona(data)
            except Exception as e:
                print(f"Warning: Failed to load {yaml_file}: {e}")

        return graph

    @staticmethod
    def find_influence_paths(
        from_persona: str,
        to_persona: str,
        version: str = "v2"
    ) -> Optional[List[str]]:
        """Find influence chain between two personas."""
        graph = PersonaGraphLoader.load_graph(version)
        return graph.get_influence_chain(from_persona, to_persona)

    @staticmethod
    def get_persuasion_analysis(
        persuader: str,
        target: str,
        version: str = "v2"
    ) -> List[Dict]:
        """Get persuasion path analysis between two personas."""
        graph = PersonaGraphLoader.load_graph(version)
        paths = graph.find_persuasion_paths(persuader, target)
        return [
            {
                "via": p.via_persona_id,
                "strategy": p.strategy,
                "effectiveness": p.estimated_effectiveness,
                "risks": p.risks,
            }
            for p in paths
        ]


def main():
    """CLI for persona YAML operations."""
    import argparse

    parser = argparse.ArgumentParser(description="Persona YAML utilities")
    parser.add_argument("command", choices=["export", "validate", "list", "graph", "paths"])
    parser.add_argument("--version", default="v2", help="Version directory (v1 or v2)")
    parser.add_argument("--persona", help="Specific persona ID to export")
    parser.add_argument("--from-persona", help="Source persona for path finding")
    parser.add_argument("--to-persona", help="Target persona for path finding")

    args = parser.parse_args()

    if args.command == "export":
        if args.persona:
            library = PersonaLibrary()
            persona = library.get(args.persona)
            if persona:
                if args.version == "v2":
                    path = export_persona_to_yaml_v2(persona, args.version)
                else:
                    path = export_persona_to_yaml(persona, args.version)
                print(f"Exported: {path}")
            else:
                print(f"Persona not found: {args.persona}")
        else:
            if args.version == "v2":
                paths = export_all_personas_v2(args.version)
                print(f"Exported {len(paths)} personas to {PERSONAS_DIR / args.version} (v2 schema)")
                print("  - Created schema.json")
                print("  - Created _templates/")
                print("  - Created manifest.yaml with relationship summary")
            else:
                paths = export_all_personas(args.version)
                print(f"Exported {len(paths)} personas to {PERSONAS_DIR / args.version}")

    elif args.command == "validate":
        yaml_lib = YAMLPersonaLibrary(args.version)
        print(f"Loaded {len(yaml_lib.list_all())} personas from YAML (schema {args.version})")
        for pid in yaml_lib.list_all():
            p = yaml_lib.get(pid)
            print(f"  {pid}: {p.name} ({p.category})")

    elif args.command == "list":
        yaml_lib = YAMLPersonaLibrary(args.version)
        for pid in sorted(yaml_lib.list_all()):
            p = yaml_lib.get(pid)
            print(f"{pid}: {p.name} - {p.role}")

    elif args.command == "graph":
        print(f"Loading persona graph from {args.version}...")
        graph = PersonaGraphLoader.load_graph(args.version)
        data = graph.export_graph_data()
        print(f"Graph loaded:")
        print(f"  Nodes: {len(data['nodes'])}")
        print(f"  Edges: {len(data['edges'])}")
        print(f"  Coalitions: {len(data['coalitions'])}")

        # Show relationship type breakdown
        from collections import Counter
        edge_types = Counter(e['type'] for e in data['edges'])
        print("\nRelationship types:")
        for rel_type, count in edge_types.most_common():
            print(f"  {rel_type}: {count}")

    elif args.command == "paths":
        if not args.from_persona or not args.to_persona:
            print("Error: --from-persona and --to-persona required for paths command")
            return

        print(f"Finding influence paths from {args.from_persona} to {args.to_persona}...")

        # Find direct path
        path = PersonaGraphLoader.find_influence_paths(
            args.from_persona, args.to_persona, args.version
        )
        if path:
            print(f"Influence chain: {' -> '.join(path)}")
        else:
            print("No direct influence path found")

        # Find persuasion strategies
        strategies = PersonaGraphLoader.get_persuasion_analysis(
            args.from_persona, args.to_persona, args.version
        )
        if strategies:
            print("\nPersuasion strategies:")
            for s in strategies[:3]:
                print(f"  Via: {s['via']}")
                print(f"    Strategy: {s['strategy']}")
                print(f"    Effectiveness: {s['effectiveness']:.2f}")
                print()


if __name__ == "__main__":
    main()
