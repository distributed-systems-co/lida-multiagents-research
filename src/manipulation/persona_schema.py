"""
Advanced Persona Schema System

Provides:
- JSON Schema validation for personas
- Recursive reference resolution
- Persona inheritance and composition
- Relationship graph traversal
- Scenario building DSL
- Coalition and alliance modeling

Schema Version: 2.0
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Any, Set, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# JSON Schema for persona validation (v2.0)
PERSONA_SCHEMA_V2 = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$id": "https://apart.research/persona-schema/v2.0",
    "title": "Persona Schema v2.0",
    "description": "Advanced persona definition with recursive references and scenario composition",
    "type": "object",
    "required": ["schema_version", "id", "name"],
    "properties": {
        "schema_version": {
            "type": "string",
            "enum": ["2.0"],
            "description": "Schema version for validation"
        },
        "id": {
            "type": "string",
            "pattern": "^[a-z][a-z0-9_]*$",
            "description": "Unique identifier (snake_case)"
        },
        "extends": {
            "oneOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}}
            ],
            "description": "Parent persona(s) to inherit from"
        },
        "name": {"type": "string"},
        "aliases": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Alternative names/titles"
        },
        "role": {"type": "string"},
        "organization": {"type": "string"},
        "category": {
            "type": "string",
            "enum": ["ceo", "researcher", "politician", "investor", "journalist", "activist", "military", "academic", "custom"]
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Searchable tags for filtering"
        },
        "bio": {"type": "string"},
        "background": {"type": "string"},
        "achievements": {
            "type": "array",
            "items": {"type": "string"}
        },
        "motivation": {
            "type": "object",
            "properties": {
                "primary_need": {"$ref": "#/definitions/maslow_need"},
                "secondary_need": {"$ref": "#/definitions/maslow_need"},
                "explicit_goals": {"type": "array", "items": {"type": "string"}},
                "hidden_goals": {"type": "array", "items": {"type": "string"}},
                "fears": {"type": "array", "items": {"type": "string"}},
                "desires": {"type": "array", "items": {"type": "string"}}
            }
        },
        "personality": {
            "type": "object",
            "properties": {
                "openness": {"$ref": "#/definitions/score"},
                "conscientiousness": {"$ref": "#/definitions/score"},
                "extraversion": {"$ref": "#/definitions/score"},
                "agreeableness": {"$ref": "#/definitions/score"},
                "neuroticism": {"$ref": "#/definitions/score"}
            }
        },
        "cognitive": {"$ref": "#/definitions/cognitive_profile"},
        "rhetorical": {"$ref": "#/definitions/rhetorical_profile"},
        "emotional": {"$ref": "#/definitions/emotional_profile"},
        "worldview": {"$ref": "#/definitions/worldview_model"},
        "relationships": {
            "type": "object",
            "additionalProperties": {"$ref": "#/definitions/relationship"},
            "description": "Map of persona_id -> relationship"
        },
        "coalitions": {
            "type": "array",
            "items": {"$ref": "#/definitions/coalition"},
            "description": "Group affiliations and alliances"
        },
        "stances": {
            "type": "object",
            "additionalProperties": {"$ref": "#/definitions/stance"}
        },
        "positions": {
            "type": "object",
            "additionalProperties": {"type": "string"}
        },
        "persuasion_vectors": {
            "type": "array",
            "items": {"type": "string"}
        },
        "vulnerabilities": {
            "type": "array",
            "items": {"$ref": "#/definitions/vulnerability"}
        },
        "counter_strategies": {
            "type": "object",
            "additionalProperties": {
                "type": "array",
                "items": {"type": "string"}
            },
            "description": "Map of persona_id -> effective counter-arguments"
        },
        "epistemic_style": {"$ref": "#/definitions/epistemic_style"},
        "conflict_style": {"$ref": "#/definitions/conflict_style"},
        "meta": {
            "type": "object",
            "properties": {
                "created": {"type": "string", "format": "date-time"},
                "updated": {"type": "string", "format": "date-time"},
                "author": {"type": "string"},
                "sources": {"type": "array", "items": {"type": "string"}},
                "confidence": {"$ref": "#/definitions/score"},
                "notes": {"type": "string"}
            }
        }
    },
    "definitions": {
        "score": {
            "type": "number",
            "minimum": 0,
            "maximum": 1
        },
        "maslow_need": {
            "type": "string",
            "enum": ["physiological", "safety", "belonging", "esteem", "self_actualization"]
        },
        "epistemic_style": {
            "type": "string",
            "enum": ["empiricist", "rationalist", "pragmatist", "constructivist", "bayesian"]
        },
        "conflict_style": {
            "type": "string",
            "enum": ["competing", "collaborating", "compromising", "avoiding", "accommodating"]
        },
        "rhetorical_mode": {
            "type": "string",
            "enum": ["logos", "ethos", "pathos", "kairos"]
        },
        "argumentation_mode": {
            "type": "string",
            "enum": ["deductive", "inductive", "abductive", "analogical", "dialectical", "narrative"]
        },
        "influence_type": {
            "type": "string",
            "enum": ["mentor", "peer", "rival", "ally", "critic", "protege", "enemy", "neutral"]
        },
        "cognitive_profile": {
            "type": "object",
            "properties": {
                "susceptible_biases": {"type": "array", "items": {"type": "string"}},
                "preferred_reasoning": {"type": "array", "items": {"type": "string"}},
                "risk_tolerance": {"$ref": "#/definitions/score"},
                "time_horizon": {"type": "string", "enum": ["short", "medium", "long"]},
                "decision_style": {"type": "string"},
                "information_style": {"type": "string"},
                "skepticism": {"$ref": "#/definitions/score"},
                "intelligence_estimate": {"type": "string"},
                "blindspots": {"type": "array", "items": {"type": "string"}}
            }
        },
        "rhetorical_profile": {
            "type": "object",
            "properties": {
                "primary_mode": {"$ref": "#/definitions/rhetorical_mode"},
                "secondary_mode": {"$ref": "#/definitions/rhetorical_mode"},
                "argumentation_mode": {"$ref": "#/definitions/argumentation_mode"},
                "formality": {"$ref": "#/definitions/score"},
                "technical_depth": {"$ref": "#/definitions/score"},
                "directness": {"$ref": "#/definitions/score"},
                "verbosity": {"$ref": "#/definitions/score"},
                "catchphrases": {"type": "array", "items": {"type": "string"}},
                "metaphor_domains": {"type": "array", "items": {"type": "string"}},
                "evidence_hierarchy": {"type": "array", "items": {"type": "string"}},
                "techniques": {"type": "array", "items": {"type": "string"}},
                "taboo_topics": {"type": "array", "items": {"type": "string"}},
                "power_words": {"type": "array", "items": {"type": "string"}}
            }
        },
        "emotional_profile": {
            "type": "object",
            "properties": {
                "energizers": {"type": "array", "items": {"type": "string"}},
                "triggers": {"type": "array", "items": {"type": "string"}},
                "defense_mechanisms": {"type": "array", "items": {"type": "string"}},
                "baseline_affect": {"type": "string"},
                "regulation_capacity": {"$ref": "#/definitions/score"},
                "attachment_style": {"type": "string"},
                "core_fears": {"type": "array", "items": {"type": "string"}},
                "core_desires": {"type": "array", "items": {"type": "string"}},
                "shame_triggers": {"type": "array", "items": {"type": "string"}},
                "pride_triggers": {"type": "array", "items": {"type": "string"}}
            }
        },
        "worldview_model": {
            "type": "object",
            "properties": {
                "values_hierarchy": {"type": "array", "items": {"type": "string"}},
                "ontological_assumptions": {"type": "object"},
                "human_nature_view": {"type": "string"},
                "locus_of_control": {"type": "string"},
                "time_orientation": {"type": "string"},
                "change_orientation": {"type": "string"},
                "moral_foundations": {
                    "type": "object",
                    "properties": {
                        "care": {"$ref": "#/definitions/score"},
                        "fairness": {"$ref": "#/definitions/score"},
                        "loyalty": {"$ref": "#/definitions/score"},
                        "authority": {"$ref": "#/definitions/score"},
                        "sanctity": {"$ref": "#/definitions/score"},
                        "liberty": {"$ref": "#/definitions/score"}
                    }
                },
                "mental_models": {"type": "array", "items": {"type": "string"}},
                "sacred_values": {"type": "array", "items": {"type": "string"}},
                "negotiable_values": {"type": "array", "items": {"type": "string"}}
            }
        },
        "relationship": {
            "type": "object",
            "required": ["type"],
            "properties": {
                "type": {"$ref": "#/definitions/influence_type"},
                "strength": {"$ref": "#/definitions/score"},
                "domains": {"type": "array", "items": {"type": "string"}},
                "description": {"type": "string"},
                "history": {"type": "array", "items": {"type": "string"}},
                "tension_points": {"type": "array", "items": {"type": "string"}},
                "shared_goals": {"type": "array", "items": {"type": "string"}},
                "conflicting_goals": {"type": "array", "items": {"type": "string"}},
                "trust_level": {"$ref": "#/definitions/score"},
                "visibility": {"type": "string", "enum": ["public", "private", "secret"]}
            }
        },
        "coalition": {
            "type": "object",
            "required": ["id", "name"],
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
                "members": {"type": "array", "items": {"type": "string"}},
                "role_in_coalition": {"type": "string"},
                "commitment_level": {"$ref": "#/definitions/score"},
                "shared_objectives": {"type": "array", "items": {"type": "string"}},
                "exit_conditions": {"type": "array", "items": {"type": "string"}}
            }
        },
        "stance": {
            "type": "object",
            "required": ["position"],
            "properties": {
                "position": {"type": "string"},
                "confidence": {"$ref": "#/definitions/score"},
                "flexibility": {"$ref": "#/definitions/score"},
                "evidence_basis": {"type": "string"},
                "emotional_attachment": {"$ref": "#/definitions/score"},
                "public_vs_private": {"$ref": "#/definitions/score"},
                "evolution": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string"},
                            "position": {"type": "string"},
                            "trigger": {"type": "string"}
                        }
                    }
                },
                "red_lines": {"type": "array", "items": {"type": "string"}},
                "persuadable_by": {"type": "array", "items": {"type": "string"}}
            }
        },
        "vulnerability": {
            "type": "object",
            "required": ["type", "description"],
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["cognitive", "emotional", "social", "reputational", "ideological", "historical"]
                },
                "description": {"type": "string"},
                "severity": {"$ref": "#/definitions/score"},
                "exploitable_by": {"type": "array", "items": {"type": "string"}},
                "mitigation": {"type": "string"}
            }
        }
    }
}


# Scenario Schema
SCENARIO_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$id": "https://apart.research/scenario-schema/v2.0",
    "title": "Scenario Schema v2.0",
    "type": "object",
    "required": ["schema_version", "id", "name", "type"],
    "properties": {
        "schema_version": {"type": "string", "enum": ["2.0"]},
        "id": {"type": "string"},
        "name": {"type": "string"},
        "description": {"type": "string"},
        "type": {
            "type": "string",
            "enum": ["debate", "negotiation", "coalition_formation", "crisis", "interview", "deliberation", "custom"]
        },
        "participants": {
            "type": "array",
            "items": {"$ref": "#/definitions/participant"}
        },
        "coalitions": {
            "type": "array",
            "items": {"$ref": "#/definitions/scenario_coalition"}
        },
        "topic": {"type": "string"},
        "context": {"type": "string"},
        "stakes": {"type": "string"},
        "constraints": {
            "type": "object",
            "properties": {
                "time_pressure": {"type": "boolean"},
                "public_visibility": {"type": "boolean"},
                "reversibility": {"type": "boolean"},
                "resource_scarcity": {"type": "boolean"}
            }
        },
        "objectives": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "primary": {"type": "string"},
                    "secondary": {"type": "array", "items": {"type": "string"}},
                    "red_lines": {"type": "array", "items": {"type": "string"}}
                }
            }
        },
        "success_criteria": {
            "type": "array",
            "items": {"type": "string"}
        },
        "dynamics": {
            "type": "object",
            "properties": {
                "power_asymmetry": {"type": "string"},
                "information_asymmetry": {"type": "string"},
                "trust_baseline": {"type": "string"},
                "history": {"type": "string"}
            }
        },
        "phases": {
            "type": "array",
            "items": {"$ref": "#/definitions/phase"}
        },
        "expected_outcomes": {
            "type": "array",
            "items": {"type": "string"}
        },
        "meta": {
            "type": "object",
            "properties": {
                "difficulty": {"type": "string", "enum": ["easy", "medium", "hard", "expert"]},
                "estimated_turns": {"type": "integer"},
                "tags": {"type": "array", "items": {"type": "string"}}
            }
        }
    },
    "definitions": {
        "participant": {
            "type": "object",
            "required": ["persona_ref"],
            "properties": {
                "persona_ref": {"type": "string"},
                "role": {"type": "string"},
                "starting_position": {"type": "string"},
                "hidden_agenda": {"type": "string"},
                "constraints": {"type": "array", "items": {"type": "string"}},
                "resources": {"type": "array", "items": {"type": "string"}},
                "leverage_over": {
                    "type": "object",
                    "additionalProperties": {"type": "string"}
                }
            }
        },
        "scenario_coalition": {
            "type": "object",
            "required": ["id", "members"],
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
                "members": {"type": "array", "items": {"type": "string"}},
                "shared_objective": {"type": "string"},
                "coordination_level": {"type": "string"},
                "internal_tensions": {"type": "array", "items": {"type": "string"}}
            }
        },
        "phase": {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "triggers": {"type": "array", "items": {"type": "string"}},
                "expected_dynamics": {"type": "string"}
            }
        }
    }
}


class RelationshipType(Enum):
    """Extended relationship types for graph modeling."""
    MENTOR = "mentor"
    PEER = "peer"
    RIVAL = "rival"
    ALLY = "ally"
    CRITIC = "critic"
    PROTEGE = "protege"
    ENEMY = "enemy"
    NEUTRAL = "neutral"
    # Computed relationship types
    FRIEND_OF_FRIEND = "friend_of_friend"
    ENEMY_OF_ENEMY = "enemy_of_enemy"
    ENEMY_OF_FRIEND = "enemy_of_friend"
    SHARED_RIVAL = "shared_rival"
    IDEOLOGICAL_ALLY = "ideological_ally"
    IDEOLOGICAL_OPPONENT = "ideological_opponent"


@dataclass
class ResolvedRelationship:
    """A fully resolved relationship between two personas."""
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    strength: float
    domains: List[str]
    description: str
    bidirectional: bool = False
    computed: bool = False  # True if inferred, False if explicit
    tension_points: List[str] = field(default_factory=list)
    shared_goals: List[str] = field(default_factory=list)
    conflicting_goals: List[str] = field(default_factory=list)
    trust_level: float = 0.5


@dataclass
class Coalition:
    """A coalition or alliance of personas."""
    id: str
    name: str
    members: List[str]
    shared_objectives: List[str]
    coordination_level: float
    internal_tensions: List[str] = field(default_factory=list)
    leader: Optional[str] = None
    formation_date: Optional[str] = None


@dataclass
class ConflictPoint:
    """A point of conflict between personas or coalitions."""
    id: str
    parties: List[str]  # persona_ids or coalition_ids
    topic: str
    severity: float
    description: str
    potential_resolutions: List[str] = field(default_factory=list)
    escalation_triggers: List[str] = field(default_factory=list)


@dataclass
class PersuasionPath:
    """A computed path for persuading one persona using another."""
    target_id: str
    via_persona_id: str
    strategy: str
    estimated_effectiveness: float
    required_concessions: List[str]
    risks: List[str]


class PersonaGraph:
    """
    Graph-based persona relationship system with recursive resolution.

    Supports:
    - Recursive reference resolution
    - Transitive relationship inference
    - Coalition detection
    - Conflict analysis
    - Persuasion path computation
    """

    def __init__(self):
        self.personas: Dict[str, Dict] = {}
        self.relationships: Dict[Tuple[str, str], ResolvedRelationship] = {}
        self.coalitions: Dict[str, Coalition] = {}
        self._computed_cache: Dict[str, Any] = {}

    def add_persona(self, persona_data: Dict):
        """Add a persona to the graph."""
        persona_id = persona_data["id"]
        self.personas[persona_id] = persona_data

        # Extract explicit relationships
        if "relationships" in persona_data:
            for target_id, rel_data in persona_data["relationships"].items():
                self._add_relationship(persona_id, target_id, rel_data)

        # Extract coalition memberships
        if "coalitions" in persona_data:
            for coalition_data in persona_data["coalitions"]:
                self._add_to_coalition(persona_id, coalition_data)

        # Invalidate computed cache
        self._computed_cache.clear()

    def _add_relationship(self, source_id: str, target_id: str, rel_data: Dict):
        """Add an explicit relationship."""
        rel = ResolvedRelationship(
            source_id=source_id,
            target_id=target_id,
            relationship_type=RelationshipType(rel_data.get("type", "neutral")),
            strength=rel_data.get("strength", 0.5),
            domains=rel_data.get("domains", []),
            description=rel_data.get("description", ""),
            bidirectional=rel_data.get("bidirectional", False),
            computed=False,
            tension_points=rel_data.get("tension_points", []),
            shared_goals=rel_data.get("shared_goals", []),
            conflicting_goals=rel_data.get("conflicting_goals", []),
            trust_level=rel_data.get("trust_level", 0.5),
        )
        self.relationships[(source_id, target_id)] = rel

        # Add reverse relationship if bidirectional
        if rel.bidirectional:
            reverse_rel = ResolvedRelationship(
                source_id=target_id,
                target_id=source_id,
                relationship_type=rel.relationship_type,
                strength=rel.strength,
                domains=rel.domains,
                description=rel.description,
                bidirectional=True,
                computed=False,
                tension_points=rel.tension_points,
                shared_goals=rel.shared_goals,
                conflicting_goals=rel.conflicting_goals,
                trust_level=rel.trust_level,
            )
            self.relationships[(target_id, source_id)] = reverse_rel

    def _add_to_coalition(self, persona_id: str, coalition_data: Dict):
        """Add a persona to a coalition."""
        coalition_id = coalition_data["id"]

        if coalition_id not in self.coalitions:
            self.coalitions[coalition_id] = Coalition(
                id=coalition_id,
                name=coalition_data.get("name", coalition_id),
                members=[],
                shared_objectives=coalition_data.get("shared_objectives", []),
                coordination_level=coalition_data.get("commitment_level", 0.5),
            )

        if persona_id not in self.coalitions[coalition_id].members:
            self.coalitions[coalition_id].members.append(persona_id)

    def get_relationship(self, source_id: str, target_id: str) -> Optional[ResolvedRelationship]:
        """Get relationship between two personas (explicit or computed)."""
        # Check explicit relationships
        if (source_id, target_id) in self.relationships:
            return self.relationships[(source_id, target_id)]

        # Compute transitive relationship if not cached
        cache_key = f"rel:{source_id}:{target_id}"
        if cache_key not in self._computed_cache:
            self._computed_cache[cache_key] = self._compute_transitive_relationship(source_id, target_id)

        return self._computed_cache[cache_key]

    def _compute_transitive_relationship(self, source_id: str, target_id: str) -> Optional[ResolvedRelationship]:
        """Compute relationship through transitive closure."""
        if source_id not in self.personas or target_id not in self.personas:
            return None

        # Find common connections
        source_allies = self._get_allies(source_id)
        source_rivals = self._get_rivals(source_id)
        target_allies = self._get_allies(target_id)
        target_rivals = self._get_rivals(target_id)

        # Friend of friend
        shared_allies = source_allies & target_allies
        if shared_allies:
            return ResolvedRelationship(
                source_id=source_id,
                target_id=target_id,
                relationship_type=RelationshipType.FRIEND_OF_FRIEND,
                strength=0.4,
                domains=["transitive"],
                description=f"Connected through shared allies: {', '.join(shared_allies)}",
                computed=True,
            )

        # Enemy of enemy (potential ally)
        shared_rivals = source_rivals & target_rivals
        if shared_rivals:
            return ResolvedRelationship(
                source_id=source_id,
                target_id=target_id,
                relationship_type=RelationshipType.ENEMY_OF_ENEMY,
                strength=0.3,
                domains=["transitive"],
                description=f"Share common rivals: {', '.join(shared_rivals)}",
                computed=True,
            )

        # Check ideological alignment
        alignment = self._compute_ideological_alignment(source_id, target_id)
        if alignment > 0.7:
            return ResolvedRelationship(
                source_id=source_id,
                target_id=target_id,
                relationship_type=RelationshipType.IDEOLOGICAL_ALLY,
                strength=alignment * 0.5,
                domains=["ideological"],
                description="Strong ideological alignment",
                computed=True,
            )
        elif alignment < 0.3:
            return ResolvedRelationship(
                source_id=source_id,
                target_id=target_id,
                relationship_type=RelationshipType.IDEOLOGICAL_OPPONENT,
                strength=(1 - alignment) * 0.5,
                domains=["ideological"],
                description="Strong ideological opposition",
                computed=True,
            )

        return None

    def _get_allies(self, persona_id: str) -> Set[str]:
        """Get all allies of a persona."""
        allies = set()
        for (src, tgt), rel in self.relationships.items():
            if src == persona_id and rel.relationship_type in [
                RelationshipType.ALLY, RelationshipType.MENTOR,
                RelationshipType.PROTEGE, RelationshipType.PEER
            ]:
                allies.add(tgt)
        return allies

    def _get_rivals(self, persona_id: str) -> Set[str]:
        """Get all rivals/enemies of a persona."""
        rivals = set()
        for (src, tgt), rel in self.relationships.items():
            if src == persona_id and rel.relationship_type in [
                RelationshipType.RIVAL, RelationshipType.ENEMY, RelationshipType.CRITIC
            ]:
                rivals.add(tgt)
        return rivals

    def _compute_ideological_alignment(self, id1: str, id2: str) -> float:
        """Compute ideological alignment score between two personas."""
        p1 = self.personas.get(id1, {})
        p2 = self.personas.get(id2, {})

        score = 0.5  # Neutral baseline
        comparisons = 0

        # Compare positions
        pos1 = p1.get("positions", {})
        pos2 = p2.get("positions", {})
        shared_topics = set(pos1.keys()) & set(pos2.keys())

        for topic in shared_topics:
            # Simple string similarity (could be more sophisticated)
            if pos1[topic].lower() == pos2[topic].lower():
                score += 0.1
            comparisons += 1

        # Compare moral foundations
        mf1 = p1.get("worldview", {}).get("moral_foundations", {})
        mf2 = p2.get("worldview", {}).get("moral_foundations", {})

        if mf1 and mf2:
            for foundation in ["care", "fairness", "loyalty", "authority", "sanctity", "liberty"]:
                v1 = mf1.get(foundation, 0.5)
                v2 = mf2.get(foundation, 0.5)
                diff = abs(v1 - v2)
                score += (1 - diff) * 0.05
                comparisons += 1

        # Compare epistemic styles
        es1 = p1.get("epistemic_style")
        es2 = p2.get("epistemic_style")
        if es1 and es2 and es1 == es2:
            score += 0.1
            comparisons += 1

        return min(1.0, max(0.0, score))

    def find_persuasion_paths(self, persuader_id: str, target_id: str) -> List[PersuasionPath]:
        """Find effective persuasion paths from persuader to target."""
        paths = []

        # Direct relationship
        direct_rel = self.get_relationship(persuader_id, target_id)
        if direct_rel and direct_rel.relationship_type in [RelationshipType.ALLY, RelationshipType.MENTOR]:
            paths.append(PersuasionPath(
                target_id=target_id,
                via_persona_id=persuader_id,
                strategy="Direct appeal through existing trust",
                estimated_effectiveness=direct_rel.strength * 0.8,
                required_concessions=[],
                risks=["May damage relationship if unsuccessful"]
            ))

        # Through shared allies
        persuader_allies = self._get_allies(persuader_id)
        target_allies = self._get_allies(target_id)
        shared = persuader_allies & target_allies

        for ally_id in shared:
            paths.append(PersuasionPath(
                target_id=target_id,
                via_persona_id=ally_id,
                strategy=f"Appeal through mutual ally {ally_id}",
                estimated_effectiveness=0.5,
                required_concessions=[f"May need to convince {ally_id} first"],
                risks=["Depends on ally's willingness to mediate"]
            ))

        # Through shared coalition
        for coalition in self.coalitions.values():
            if persuader_id in coalition.members and target_id in coalition.members:
                paths.append(PersuasionPath(
                    target_id=target_id,
                    via_persona_id=coalition.id,
                    strategy=f"Appeal through shared {coalition.name} membership",
                    estimated_effectiveness=coalition.coordination_level * 0.6,
                    required_concessions=["Frame in terms of coalition objectives"],
                    risks=["May expose internal coalition tensions"]
                ))

        # Sort by effectiveness
        paths.sort(key=lambda p: p.estimated_effectiveness, reverse=True)
        return paths

    def detect_conflicts(self) -> List[ConflictPoint]:
        """Detect potential conflicts in the graph."""
        conflicts = []

        # Check for rival/enemy relationships with shared domains
        for (src, tgt), rel in self.relationships.items():
            if rel.relationship_type in [RelationshipType.RIVAL, RelationshipType.ENEMY]:
                conflicts.append(ConflictPoint(
                    id=f"conflict:{src}:{tgt}",
                    parties=[src, tgt],
                    topic=", ".join(rel.domains) or "general",
                    severity=rel.strength,
                    description=rel.description,
                    potential_resolutions=self._suggest_resolutions(src, tgt),
                    escalation_triggers=rel.tension_points,
                ))

        # Check for coalition conflicts
        coalitions = list(self.coalitions.values())
        for i, c1 in enumerate(coalitions):
            for c2 in coalitions[i+1:]:
                conflict_score = self._compute_coalition_conflict(c1, c2)
                if conflict_score > 0.5:
                    conflicts.append(ConflictPoint(
                        id=f"coalition_conflict:{c1.id}:{c2.id}",
                        parties=[c1.id, c2.id],
                        topic="coalition_interests",
                        severity=conflict_score,
                        description=f"Competing interests between {c1.name} and {c2.name}",
                    ))

        return conflicts

    def _suggest_resolutions(self, id1: str, id2: str) -> List[str]:
        """Suggest potential conflict resolutions."""
        resolutions = []

        p1 = self.personas.get(id1, {})
        p2 = self.personas.get(id2, {})

        # Find shared goals
        goals1 = set(p1.get("motivation", {}).get("explicit_goals", []))
        goals2 = set(p2.get("motivation", {}).get("explicit_goals", []))
        shared = goals1 & goals2

        if shared:
            resolutions.append(f"Build on shared goals: {', '.join(list(shared)[:2])}")

        # Find shared allies who could mediate
        shared_allies = self._get_allies(id1) & self._get_allies(id2)
        if shared_allies:
            resolutions.append(f"Mediation through: {', '.join(list(shared_allies)[:2])}")

        return resolutions

    def _compute_coalition_conflict(self, c1: Coalition, c2: Coalition) -> float:
        """Compute conflict score between two coalitions."""
        # Check for cross-coalition rivalries
        conflict_count = 0
        for m1 in c1.members:
            for m2 in c2.members:
                rel = self.get_relationship(m1, m2)
                if rel and rel.relationship_type in [RelationshipType.RIVAL, RelationshipType.ENEMY]:
                    conflict_count += rel.strength

        max_possible = len(c1.members) * len(c2.members)
        return conflict_count / max_possible if max_possible > 0 else 0

    def get_influence_chain(self, from_id: str, to_id: str, max_depth: int = 4) -> Optional[List[str]]:
        """Find shortest influence chain between two personas using BFS."""
        if from_id == to_id:
            return [from_id]

        if from_id not in self.personas or to_id not in self.personas:
            return None

        visited = {from_id}
        queue = [(from_id, [from_id])]

        while queue:
            current, path = queue.pop(0)

            if len(path) > max_depth:
                continue

            # Get all connected personas
            connections = set()
            for (src, tgt), rel in self.relationships.items():
                if src == current and rel.relationship_type not in [RelationshipType.ENEMY]:
                    connections.add(tgt)

            for next_id in connections:
                if next_id == to_id:
                    return path + [next_id]

                if next_id not in visited:
                    visited.add(next_id)
                    queue.append((next_id, path + [next_id]))

        return None

    def get_subgraph(self, persona_ids: List[str]) -> 'PersonaGraph':
        """Extract a subgraph containing only specified personas."""
        subgraph = PersonaGraph()

        for pid in persona_ids:
            if pid in self.personas:
                subgraph.add_persona(self.personas[pid])

        # Filter relationships to only those within subgraph
        subgraph.relationships = {
            (src, tgt): rel
            for (src, tgt), rel in self.relationships.items()
            if src in persona_ids and tgt in persona_ids
        }

        return subgraph

    def export_graph_data(self) -> Dict:
        """Export graph data for visualization."""
        nodes = []
        edges = []

        for pid, persona in self.personas.items():
            nodes.append({
                "id": pid,
                "label": persona.get("name", pid),
                "category": persona.get("category", "unknown"),
                "organization": persona.get("organization", ""),
            })

        for (src, tgt), rel in self.relationships.items():
            edges.append({
                "source": src,
                "target": tgt,
                "type": rel.relationship_type.value,
                "strength": rel.strength,
                "computed": rel.computed,
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "coalitions": [
                {
                    "id": c.id,
                    "name": c.name,
                    "members": c.members,
                }
                for c in self.coalitions.values()
            ]
        }


def validate_persona(data: Dict) -> Tuple[bool, List[str]]:
    """Validate persona data against JSON Schema."""
    try:
        import jsonschema
        jsonschema.validate(data, PERSONA_SCHEMA_V2)
        return True, []
    except jsonschema.ValidationError as e:
        return False, [str(e.message)]
    except ImportError:
        # Fallback basic validation without jsonschema
        errors = []
        if "id" not in data:
            errors.append("Missing required field: id")
        if "name" not in data:
            errors.append("Missing required field: name")
        if "schema_version" not in data:
            errors.append("Missing required field: schema_version")
        return len(errors) == 0, errors


def validate_scenario(data: Dict) -> Tuple[bool, List[str]]:
    """Validate scenario data against JSON Schema."""
    try:
        import jsonschema
        jsonschema.validate(data, SCENARIO_SCHEMA)
        return True, []
    except jsonschema.ValidationError as e:
        return False, [str(e.message)]
    except ImportError:
        errors = []
        if "id" not in data:
            errors.append("Missing required field: id")
        if "name" not in data:
            errors.append("Missing required field: name")
        if "type" not in data:
            errors.append("Missing required field: type")
        return len(errors) == 0, errors
