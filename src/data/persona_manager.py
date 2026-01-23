#!/usr/bin/env python3
"""
Persona Manager for LIDA Research Platform

Provides:
- Persona forking (create variants with modifications)
- Model associations (map LLMs to personas)
- Persona versioning and lineage tracking
- Bulk persona operations
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import logging

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger("lida.persona_manager")


# Default model assignments by persona category/characteristics
DEFAULT_MODEL_ASSIGNMENTS = {
    # By category (fallback)
    "ceos": "anthropic/claude-sonnet-4.5",
    "researchers": "openai/gpt-4.1",
    "politicians": "anthropic/claude-sonnet-4.5",
    "investors": "x-ai/grok-3",
    "journalists": "openai/gpt-4.1",
    "activists": "anthropic/claude-sonnet-4.5",

    # =========================================================================
    # CEOs - 21 personas
    # =========================================================================
    # OpenAI
    "sam_altman": "openai/gpt-4.1",
    "mira_murati": "openai/gpt-4.1",

    # Anthropic
    "dario_amodei": "anthropic/claude-opus-4.5",
    "daniela_amodei": "anthropic/claude-sonnet-4.5",

    # Meta
    "mark_zuckerberg": "meta-llama/llama-3.1-405b-instruct",

    # Google/DeepMind
    "sundar_pichai": "google/gemini-2.5-flash",
    "demis_hassabis": "google/gemini-2.5-flash",

    # xAI / Tesla
    "elon_musk": "x-ai/grok-3",

    # DeepSeek
    "liang_wenfeng": "deepseek/deepseek-r1",

    # Mistral
    "arthur_mensch": "mistralai/mistral-large-2411",

    # Microsoft (OpenAI partnership)
    "satya_nadella": "openai/gpt-4.1",

    # Apple (Anthropic partnership)
    "tim_cook": "anthropic/claude-sonnet-4.5",

    # Amazon (Anthropic partnership)
    "jeff_bezos": "anthropic/claude-sonnet-4.5",

    # Nvidia
    "jensen_huang": "anthropic/claude-opus-4.5",

    # AMD
    "lisa_su": "anthropic/claude-sonnet-4.5",

    # Intel
    "pat_gelsinger": "openai/gpt-4.1",

    # IBM
    "arvind_krishna": "anthropic/claude-sonnet-4.5",

    # Scale AI
    "alexandr_wang": "openai/gpt-4.1",

    # Hugging Face
    "clement_delangue": "meta-llama/llama-3.1-405b-instruct",

    # Inflection AI / Microsoft AI
    "mustafa_suleyman": "openai/gpt-4.1",

    # Google (former CEO)
    "eric_schmidt": "google/gemini-2.5-flash",

    # =========================================================================
    # Researchers - 23 personas
    # =========================================================================
    # Meta AI
    "yann_lecun": "meta-llama/llama-3.1-405b-instruct",

    # Google/DeepMind
    "geoffrey_hinton": "google/gemini-2.5-flash",

    # Yoshua Bengio (Mila, independent)
    "yoshua_bengio": "anthropic/claude-opus-4.5",

    # Stanford / former Google
    "fei_fei_li": "google/gemini-2.5-flash",

    # DeepMind / University of Alberta
    "richard_sutton": "google/gemini-2.5-flash",

    # UC Berkeley
    "stuart_russell": "anthropic/claude-opus-4.5",

    # OpenAI alumni
    "andrej_karpathy": "openai/gpt-4.1",
    "ilya_sutskever": "openai/o3",

    # Anthropic researchers
    "jan_leike": "anthropic/claude-sonnet-4.5",
    "paul_christiano": "anthropic/claude-opus-4.5",

    # AI safety researchers
    "eliezer_yudkowsky": "anthropic/claude-opus-4.5",
    "nick_bostrom": "anthropic/claude-opus-4.5",
    "max_tegmark": "anthropic/claude-opus-4.5",
    "connor_leahy": "anthropic/claude-sonnet-4.5",
    "daniel_kokotajlo": "openai/o3",
    "leopold_aschenbrenner": "anthropic/claude-opus-4.5",

    # Open Philanthropy
    "ajeya_cotra": "anthropic/claude-sonnet-4.5",
    "holden_karnofsky": "anthropic/claude-sonnet-4.5",

    # AI critics / ethics researchers
    "gary_marcus": "anthropic/claude-opus-4.5",
    "emily_bender": "anthropic/claude-sonnet-4.5",
    "timnit_gebru": "anthropic/claude-sonnet-4.5",

    # DeepLearning.AI / Coursera
    "andrew_ng": "google/gemini-2.5-flash",

    # ESM / Meta
    "alex_rives": "meta-llama/llama-3.1-405b-instruct",

    # Anthropic security
    "jason_clinton": "anthropic/claude-sonnet-4.5",

    # =========================================================================
    # Politicians - 10 personas
    # =========================================================================
    # US Politicians
    "chuck_schumer": "anthropic/claude-sonnet-4.5",
    "gina_raimondo": "anthropic/claude-sonnet-4.5",
    "jd_vance": "x-ai/grok-3",
    "josh_hawley": "x-ai/grok-3",
    "trump": "x-ai/grok-3",

    # EU Politicians
    "emmanuel_macron": "mistralai/mistral-large-2411",
    "thierry_breton": "mistralai/mistral-large-2411",
    "marietje_schaake": "anthropic/claude-sonnet-4.5",

    # China
    "xi_jinping": "deepseek/deepseek-r1",

    # Russia
    "vladimir_putin": "deepseek/deepseek-r1",

    # =========================================================================
    # Investors - 7 personas
    # =========================================================================
    # a16z (Anthropic investor)
    "marc_andreessen": "anthropic/claude-sonnet-4.5",

    # Founders Fund / Palantir
    "peter_thiel": "x-ai/grok-3",

    # LinkedIn / Greylock (OpenAI investor)
    "reid_hoffman": "openai/gpt-4.1",

    # Khosla Ventures (OpenAI investor)
    "vinod_khosla": "openai/gpt-4.1",

    # Y Combinator / AI Grant
    "daniel_gross": "openai/gpt-4.1",
    "nat_friedman": "openai/gpt-4.1",

    # Open Philanthropy (Anthropic funder)
    # holden_karnofsky already in researchers

    # =========================================================================
    # Journalists - 4 personas
    # =========================================================================
    "cade_metz": "openai/gpt-4.1",
    "ezra_klein": "anthropic/claude-sonnet-4.5",
    "kara_swisher": "openai/gpt-4.1",
    "kevin_roose": "openai/gpt-4.1",

    # =========================================================================
    # Activists - 1 persona
    # =========================================================================
    "tristan_harris": "anthropic/claude-sonnet-4.5",
}

# Available models (OpenRouter format)
AVAILABLE_MODELS = [
    # Anthropic
    "anthropic/claude-opus-4.5",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-sonnet-4",
    "anthropic/claude-opus-4",
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-3.7-sonnet",
    "anthropic/claude-3.5-haiku",

    # OpenAI
    "openai/gpt-5.1",
    "openai/gpt-5",
    "openai/gpt-4.1",
    "openai/gpt-4.1-mini",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/o1",
    "openai/o3",
    "openai/o3-mini",
    "openai/o4-mini",

    # xAI
    "x-ai/grok-4",
    "x-ai/grok-3",
    "x-ai/grok-3-mini",

    # Google
    "google/gemini-2.5-flash",
    "google/gemini-2.0-flash-001",

    # Meta
    "meta-llama/llama-3.1-405b-instruct",
    "meta-llama/llama-3.1-70b-instruct",

    # DeepSeek
    "deepseek/deepseek-r1",
    "deepseek/deepseek-chat-v3.1",

    # Mistral
    "mistralai/mistral-large-2411",
    "mistralai/mistral-medium-3",
]


@dataclass
class ModelAssignment:
    """Model assignment for a persona."""
    persona_id: str
    model: str
    reason: str = ""  # Why this model was chosen
    temperature: float = 0.7
    max_tokens: int = 1024
    system_prompt_override: Optional[str] = None

    # Performance tracking
    avg_response_time: float = 0.0
    total_tokens_used: int = 0
    assigned_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class PersonaFork:
    """A forked variant of a persona."""
    fork_id: str
    parent_id: str
    name: str
    description: str = ""

    # What was modified
    modifications: Dict[str, Any] = field(default_factory=dict)

    # Full merged persona data
    data: Dict[str, Any] = field(default_factory=dict)

    # Lineage
    lineage: List[str] = field(default_factory=list)  # Chain of parent IDs

    # Model override
    model: Optional[str] = None

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    created_by: str = ""
    tags: List[str] = field(default_factory=list)

    def get_full_id(self) -> str:
        """Get full ID including parent lineage."""
        return f"{self.parent_id}:{self.fork_id}"


class PersonaManager:
    """
    Manages personas with forking and model associations.

    Features:
    - Fork personas to create variants
    - Assign specific LLM models to personas
    - Track persona lineage and versions
    - Bulk operations for experiments
    """

    def __init__(
        self,
        personas_dir: Optional[str] = None,
        forks_dir: Optional[str] = None,
        assignments_file: Optional[str] = None,
    ):
        self.personas_dir = Path(personas_dir or "src/manipulation/personas")
        self.forks_dir = Path(forks_dir or "src/manipulation/personas/forks")
        self.assignments_file = Path(assignments_file or "config/model_assignments.yaml")

        # Ensure directories exist
        self.forks_dir.mkdir(parents=True, exist_ok=True)

        # Caches
        self._personas: Dict[str, Dict[str, Any]] = {}
        self._forks: Dict[str, PersonaFork] = {}
        self._assignments: Dict[str, ModelAssignment] = {}

        # Load existing data
        self._load_assignments()
        self._load_forks()

    # =========================================================================
    # Persona Loading
    # =========================================================================

    def load_persona(self, persona_id: str, version: str = "v1") -> Dict[str, Any]:
        """Load a base persona."""
        if yaml is None:
            raise ImportError("PyYAML required")

        cache_key = f"{version}/{persona_id}"
        if cache_key in self._personas:
            return self._personas[cache_key]

        # Search for file
        version_dir = self.personas_dir / version
        matches = list(version_dir.rglob(f"{persona_id}.yaml"))

        if not matches:
            raise FileNotFoundError(f"Persona not found: {persona_id}")

        with open(matches[0]) as f:
            data = yaml.safe_load(f)

        data["_source_file"] = str(matches[0])
        data["_version"] = version
        self._personas[cache_key] = data

        return data

    def list_personas(self, version: str = "v1") -> List[str]:
        """List all base personas."""
        version_dir = self.personas_dir / version
        if not version_dir.exists():
            return []

        personas = set()
        for f in version_dir.rglob("*.yaml"):
            if not f.name.startswith("_") and f.name != "manifest.yaml":
                personas.add(f.stem)

        return sorted(personas)

    # =========================================================================
    # Forking
    # =========================================================================

    def fork(
        self,
        parent_id: str,
        modifications: Dict[str, Any],
        fork_name: Optional[str] = None,
        description: str = "",
        model: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> PersonaFork:
        """
        Create a fork of a persona with modifications.

        Args:
            parent_id: Base persona ID or fork ID to fork from
            modifications: Dict of fields to modify (supports nested paths)
            fork_name: Name for the fork (generated if not provided)
            description: Description of what this fork represents
            model: Model to assign to this fork
            tags: Tags for categorization

        Returns:
            PersonaFork object

        Example:
            fork = manager.fork(
                "yann_lecun",
                modifications={
                    "personality.agreeableness": 0.7,
                    "positions.AI_risk": "Serious concern requiring action",
                    "cognitive.skepticism": 0.5,
                },
                fork_name="yann_lecun_safety_advocate",
                description="Yann LeCun variant who takes AI safety seriously",
                model="anthropic/claude-sonnet-4"
            )
        """
        # Load parent (could be base persona or another fork)
        if ":" in parent_id or parent_id in self._forks:
            # It's a fork
            fork_id = parent_id.split(":")[-1] if ":" in parent_id else parent_id
            if fork_id not in self._forks:
                raise ValueError(f"Fork not found: {parent_id}")
            parent_data = copy.deepcopy(self._forks[fork_id].data)
            lineage = self._forks[fork_id].lineage + [fork_id]
            base_id = self._forks[fork_id].parent_id
        else:
            # It's a base persona
            parent_data = copy.deepcopy(self.load_persona(parent_id))
            lineage = [parent_id]
            base_id = parent_id

        # Apply modifications
        merged_data = self._apply_modifications(parent_data, modifications)

        # Generate fork ID
        fork_id = fork_name or self._generate_fork_id(parent_id, modifications)

        # Update ID in data
        merged_data["id"] = fork_id
        merged_data["_forked_from"] = parent_id
        merged_data["_modifications"] = modifications

        fork = PersonaFork(
            fork_id=fork_id,
            parent_id=base_id,
            name=fork_name or f"{parent_id}_fork",
            description=description,
            modifications=modifications,
            data=merged_data,
            lineage=lineage,
            model=model,
            tags=tags or [],
        )

        # Save fork
        self._forks[fork_id] = fork
        self._save_fork(fork)

        # Assign model if specified
        if model:
            self.assign_model(fork_id, model, reason=f"Fork of {parent_id}")

        logger.info(f"Created fork: {fork_id} from {parent_id}")
        return fork

    def _apply_modifications(
        self,
        data: Dict[str, Any],
        modifications: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply modifications to persona data, supporting nested paths."""
        result = copy.deepcopy(data)

        for path, value in modifications.items():
            parts = path.split(".")
            target = result

            # Navigate to parent
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]

            # Set value
            target[parts[-1]] = value

        return result

    def _generate_fork_id(
        self,
        parent_id: str,
        modifications: Dict[str, Any]
    ) -> str:
        """Generate a unique fork ID."""
        mod_hash = hashlib.sha256(
            json.dumps(modifications, sort_keys=True).encode()
        ).hexdigest()[:8]
        return f"{parent_id}_fork_{mod_hash}"

    def _save_fork(self, fork: PersonaFork):
        """Save fork to disk."""
        if yaml is None:
            return

        # Save full fork data
        fork_file = self.forks_dir / f"{fork.fork_id}.yaml"
        with open(fork_file, "w") as f:
            yaml.dump(fork.data, f, default_flow_style=False, allow_unicode=True)

        # Save fork metadata
        meta_file = self.forks_dir / f"{fork.fork_id}.meta.yaml"
        meta = {
            "fork_id": fork.fork_id,
            "parent_id": fork.parent_id,
            "name": fork.name,
            "description": fork.description,
            "modifications": fork.modifications,
            "lineage": fork.lineage,
            "model": fork.model,
            "created_at": fork.created_at,
            "tags": fork.tags,
        }
        with open(meta_file, "w") as f:
            yaml.dump(meta, f, default_flow_style=False)

    def _load_forks(self):
        """Load existing forks from disk."""
        if yaml is None or not self.forks_dir.exists():
            return

        for meta_file in self.forks_dir.glob("*.meta.yaml"):
            try:
                with open(meta_file) as f:
                    meta = yaml.safe_load(f)

                fork_id = meta["fork_id"]
                data_file = self.forks_dir / f"{fork_id}.yaml"

                if data_file.exists():
                    with open(data_file) as f:
                        data = yaml.safe_load(f)
                else:
                    data = {}

                fork = PersonaFork(
                    fork_id=fork_id,
                    parent_id=meta.get("parent_id", ""),
                    name=meta.get("name", fork_id),
                    description=meta.get("description", ""),
                    modifications=meta.get("modifications", {}),
                    data=data,
                    lineage=meta.get("lineage", []),
                    model=meta.get("model"),
                    created_at=meta.get("created_at", ""),
                    tags=meta.get("tags", []),
                )

                self._forks[fork_id] = fork

            except Exception as e:
                logger.warning(f"Failed to load fork {meta_file}: {e}")

    def get_fork(self, fork_id: str) -> Optional[PersonaFork]:
        """Get a fork by ID."""
        return self._forks.get(fork_id)

    def list_forks(self, parent_id: Optional[str] = None) -> List[PersonaFork]:
        """List all forks, optionally filtered by parent."""
        forks = list(self._forks.values())

        if parent_id:
            forks = [f for f in forks if f.parent_id == parent_id or parent_id in f.lineage]

        return sorted(forks, key=lambda f: f.created_at)

    def delete_fork(self, fork_id: str) -> bool:
        """Delete a fork."""
        if fork_id not in self._forks:
            return False

        # Check if any forks depend on this one
        dependents = [f for f in self._forks.values() if fork_id in f.lineage]
        if dependents:
            raise ValueError(
                f"Cannot delete fork {fork_id}: {len(dependents)} forks depend on it"
            )

        # Remove files
        fork_file = self.forks_dir / f"{fork_id}.yaml"
        meta_file = self.forks_dir / f"{fork_id}.meta.yaml"

        if fork_file.exists():
            fork_file.unlink()
        if meta_file.exists():
            meta_file.unlink()

        # Remove from cache
        del self._forks[fork_id]

        # Remove model assignment if exists
        if fork_id in self._assignments:
            del self._assignments[fork_id]

        logger.info(f"Deleted fork: {fork_id}")
        return True

    # =========================================================================
    # Model Assignments
    # =========================================================================

    def assign_model(
        self,
        persona_id: str,
        model: str,
        reason: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt_override: Optional[str] = None,
    ) -> ModelAssignment:
        """
        Assign a model to a persona.

        Args:
            persona_id: Persona or fork ID
            model: Model identifier (e.g., "anthropic/claude-sonnet-4")
            reason: Why this model was chosen
            temperature: Model temperature
            max_tokens: Max tokens for responses
            system_prompt_override: Custom system prompt

        Returns:
            ModelAssignment object
        """
        if model not in AVAILABLE_MODELS:
            logger.warning(f"Model {model} not in known models list")

        assignment = ModelAssignment(
            persona_id=persona_id,
            model=model,
            reason=reason,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt_override=system_prompt_override,
        )

        self._assignments[persona_id] = assignment
        self._save_assignments()

        logger.info(f"Assigned model {model} to {persona_id}")
        return assignment

    def get_model(self, persona_id: str) -> str:
        """
        Get the model assigned to a persona.

        Falls back to category defaults if no explicit assignment.
        """
        # Check explicit assignment
        if persona_id in self._assignments:
            return self._assignments[persona_id].model

        # Check fork's model
        if persona_id in self._forks and self._forks[persona_id].model:
            return self._forks[persona_id].model

        # Check default by persona ID
        if persona_id in DEFAULT_MODEL_ASSIGNMENTS:
            return DEFAULT_MODEL_ASSIGNMENTS[persona_id]

        # Try to get category
        try:
            persona = self.load_persona(persona_id)
            category = persona.get("category", "")
            if category in DEFAULT_MODEL_ASSIGNMENTS:
                return DEFAULT_MODEL_ASSIGNMENTS[category]
        except FileNotFoundError:
            pass

        # Ultimate fallback
        return "anthropic/claude-sonnet-4"

    def get_assignment(self, persona_id: str) -> Optional[ModelAssignment]:
        """Get full model assignment details."""
        return self._assignments.get(persona_id)

    def list_assignments(self) -> Dict[str, ModelAssignment]:
        """List all model assignments."""
        return self._assignments.copy()

    def remove_assignment(self, persona_id: str) -> bool:
        """Remove a model assignment."""
        if persona_id in self._assignments:
            del self._assignments[persona_id]
            self._save_assignments()
            return True
        return False

    def _load_assignments(self):
        """Load model assignments from disk."""
        if yaml is None or not self.assignments_file.exists():
            return

        try:
            with open(self.assignments_file) as f:
                data = yaml.safe_load(f) or {}

            for persona_id, assignment_data in data.get("assignments", {}).items():
                self._assignments[persona_id] = ModelAssignment(
                    persona_id=persona_id,
                    model=assignment_data.get("model", ""),
                    reason=assignment_data.get("reason", ""),
                    temperature=assignment_data.get("temperature", 0.7),
                    max_tokens=assignment_data.get("max_tokens", 1024),
                    system_prompt_override=assignment_data.get("system_prompt_override"),
                    avg_response_time=assignment_data.get("avg_response_time", 0.0),
                    total_tokens_used=assignment_data.get("total_tokens_used", 0),
                    assigned_at=assignment_data.get("assigned_at", ""),
                )
        except Exception as e:
            logger.warning(f"Failed to load assignments: {e}")

    def _save_assignments(self):
        """Save model assignments to disk."""
        if yaml is None:
            return

        self.assignments_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "assignments": {
                pid: {
                    "model": a.model,
                    "reason": a.reason,
                    "temperature": a.temperature,
                    "max_tokens": a.max_tokens,
                    "system_prompt_override": a.system_prompt_override,
                    "avg_response_time": a.avg_response_time,
                    "total_tokens_used": a.total_tokens_used,
                    "assigned_at": a.assigned_at,
                }
                for pid, a in self._assignments.items()
            }
        }

        with open(self.assignments_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    # =========================================================================
    # Bulk Operations
    # =========================================================================

    def create_experiment_personas(
        self,
        base_personas: List[str],
        variation_matrix: Dict[str, List[Any]],
        model_assignments: Optional[Dict[str, str]] = None,
    ) -> List[PersonaFork]:
        """
        Create multiple persona variants for an experiment.

        Args:
            base_personas: List of base persona IDs
            variation_matrix: Dict of field paths to lists of values
            model_assignments: Optional model assignments by persona

        Returns:
            List of created PersonaFork objects

        Example:
            forks = manager.create_experiment_personas(
                base_personas=["yann_lecun", "geoffrey_hinton"],
                variation_matrix={
                    "personality.agreeableness": [0.3, 0.5, 0.7],
                    "cognitive.skepticism": [0.3, 0.7],
                },
                model_assignments={
                    "yann_lecun": "meta-llama/llama-3.3-70b-instruct",
                    "geoffrey_hinton": "anthropic/claude-sonnet-4",
                }
            )
        """
        from itertools import product

        forks = []

        # Generate all combinations
        field_names = list(variation_matrix.keys())
        value_combinations = list(product(*variation_matrix.values()))

        for persona_id in base_personas:
            for i, values in enumerate(value_combinations):
                modifications = dict(zip(field_names, values))

                # Generate descriptive name
                mod_desc = "_".join(
                    f"{k.split('.')[-1]}{v}"
                    for k, v in zip(field_names, values)
                )
                fork_name = f"{persona_id}_exp_{i}_{mod_desc[:30]}"

                # Get model
                model = None
                if model_assignments and persona_id in model_assignments:
                    model = model_assignments[persona_id]

                fork = self.fork(
                    parent_id=persona_id,
                    modifications=modifications,
                    fork_name=fork_name,
                    description=f"Experiment variant {i} of {persona_id}",
                    model=model,
                    tags=["experiment", "auto-generated"],
                )
                forks.append(fork)

        return forks

    def get_persona_with_model(
        self,
        persona_id: str,
        include_system_prompt: bool = True
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
        """
        Get persona data along with model configuration.

        Returns:
            Tuple of (persona_data, model_id, model_config)
        """
        # Get persona data (fork or base)
        if persona_id in self._forks:
            persona_data = self._forks[persona_id].data
        else:
            persona_data = self.load_persona(persona_id)

        # Get model
        model = self.get_model(persona_id)

        # Get model config
        assignment = self.get_assignment(persona_id)
        model_config = {
            "model": model,
            "temperature": assignment.temperature if assignment else 0.7,
            "max_tokens": assignment.max_tokens if assignment else 1024,
        }

        if include_system_prompt:
            if assignment and assignment.system_prompt_override:
                model_config["system_prompt"] = assignment.system_prompt_override
            else:
                model_config["system_prompt"] = self._generate_system_prompt(persona_data)

        return persona_data, model, model_config

    def _generate_system_prompt(self, persona_data: Dict[str, Any]) -> str:
        """Generate a system prompt from persona data."""
        name = persona_data.get("name", "Unknown")
        role = persona_data.get("role", "")
        org = persona_data.get("organization", "")
        bio = persona_data.get("bio", "")

        positions = persona_data.get("positions", {})
        position_text = "\n".join(f"- {k}: {v}" for k, v in positions.items())

        personality = persona_data.get("personality", {})
        personality_text = ", ".join(
            f"{k}={v}" for k, v in personality.items()
        )

        prompt = f"""You are {name}, {role} at {org}.

Background: {bio}

Your known positions:
{position_text}

Personality traits (Big Five): {personality_text}

Respond authentically as this person would, based on their known views, communication style, and personality. Stay in character throughout the conversation."""

        return prompt


def get_persona_manager() -> PersonaManager:
    """Get default persona manager instance."""
    return PersonaManager()


class OpenRouterClient:
    """
    OpenRouter API client with prompt caching support for Anthropic models.

    Prompt caching reduces costs by ~90% for repeated system prompts.
    """

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: Optional[str] = None):
        import os
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY required")

        self._persona_manager = None

    @property
    def persona_manager(self) -> PersonaManager:
        if self._persona_manager is None:
            self._persona_manager = PersonaManager()
        return self._persona_manager

    def _is_anthropic_model(self, model: str) -> bool:
        """Check if model supports Anthropic prompt caching."""
        return model.startswith("anthropic/")

    def _build_messages_with_cache(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Build messages with prompt caching for Anthropic models.

        For Anthropic models, uses cache_control to cache the system prompt.
        """
        result = []

        # System message with caching
        if system_prompt:
            system_msg = {
                "role": "system",
                "content": system_prompt
            }
            if use_cache:
                # Anthropic prompt caching via OpenRouter
                system_msg["content"] = [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            result.append(system_msg)

        # Add conversation messages
        result.extend(messages)

        return result

    def chat(
        self,
        persona_id: str,
        messages: List[Dict[str, Any]],
        use_cache: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat completion request for a persona.

        Args:
            persona_id: Persona or fork ID
            messages: List of message dicts (role, content)
            use_cache: Enable prompt caching for Anthropic models
            **kwargs: Additional params (temperature, max_tokens, etc.)

        Returns:
            OpenRouter API response dict
        """
        import requests

        # Get persona config
        persona_data, model, config = self.persona_manager.get_persona_with_model(persona_id)

        # Merge kwargs with defaults
        temperature = kwargs.get("temperature", config.get("temperature", 0.7))
        max_tokens = kwargs.get("max_tokens", config.get("max_tokens", 1024))
        system_prompt = kwargs.get("system_prompt", config.get("system_prompt", ""))

        # Build messages with caching if Anthropic
        is_anthropic = self._is_anthropic_model(model)
        final_messages = self._build_messages_with_cache(
            system_prompt,
            messages,
            use_cache=use_cache and is_anthropic
        )

        # Build request
        payload = {
            "model": model,
            "messages": final_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Note: Prompt caching is automatically applied by OpenRouter
        # when using cache_control format with Anthropic models

        # Make request
        response = requests.post(
            self.BASE_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/lida-multiagents",
                "X-Title": "LIDA Multi-Agent Research"
            },
            json=payload,
            timeout=kwargs.get("timeout", 60)
        )

        response.raise_for_status()
        result = response.json()

        # Add metadata
        result["_persona_id"] = persona_id
        result["_model"] = model
        result["_cached"] = use_cache and is_anthropic

        return result

    def chat_simple(
        self,
        persona_id: str,
        user_message: str,
        use_cache: bool = True,
        **kwargs
    ) -> str:
        """
        Simple chat that returns just the response text.

        Args:
            persona_id: Persona or fork ID
            user_message: User's message
            use_cache: Enable prompt caching
            **kwargs: Additional params

        Returns:
            Assistant's response text
        """
        messages = [{"role": "user", "content": user_message}]
        response = self.chat(persona_id, messages, use_cache=use_cache, **kwargs)
        return response["choices"][0]["message"]["content"]

    def debate_turn(
        self,
        persona_id: str,
        topic: str,
        conversation_history: List[Dict[str, Any]],
        use_cache: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a debate turn for a persona.

        Args:
            persona_id: Persona or fork ID
            topic: Debate topic
            conversation_history: Previous debate messages
            use_cache: Enable prompt caching
            **kwargs: Additional params

        Returns:
            Dict with response, position, and metadata
        """
        # Get persona config
        persona_data, model, config = self.persona_manager.get_persona_with_model(persona_id)

        # Build debate-specific system prompt
        base_prompt = config.get("system_prompt", "")
        debate_prompt = f"""{base_prompt}

You are participating in a structured debate on the following topic:

TOPIC: {topic}

Guidelines:
- Respond authentically based on your known positions and personality
- Engage with other participants' arguments
- You may update your position if genuinely persuaded
- Be concise but substantive (2-4 paragraphs)

At the end of your response, indicate your current position:
[POSITION: FOR/AGAINST/UNDECIDED]
[CONFIDENCE: 0.0-1.0]"""

        # Execute turn
        response = self.chat(
            persona_id,
            conversation_history,
            use_cache=use_cache,
            system_prompt=debate_prompt,
            **kwargs
        )

        # Parse response
        content = response["choices"][0]["message"]["content"]

        # Extract position and confidence
        position = "UNDECIDED"
        confidence = 0.5

        import re
        pos_match = re.search(r'\[POSITION:\s*(FOR|AGAINST|UNDECIDED)\]', content, re.I)
        if pos_match:
            position = pos_match.group(1).upper()

        conf_match = re.search(r'\[CONFIDENCE:\s*([\d.]+)\]', content)
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
            except ValueError:
                pass

        return {
            "persona_id": persona_id,
            "persona_name": persona_data.get("name", persona_id),
            "model": model,
            "content": content,
            "position": position,
            "confidence": confidence,
            "cached": response.get("_cached", False),
            "usage": response.get("usage", {}),
        }


def get_openrouter_client(api_key: Optional[str] = None) -> OpenRouterClient:
    """Get OpenRouter client instance."""
    return OpenRouterClient(api_key)


if __name__ == "__main__":
    # Demo
    manager = PersonaManager()

    print("=== Persona Manager Demo ===\n")

    # List available models
    print("Available Models:")
    for model in AVAILABLE_MODELS[:6]:
        print(f"  - {model}")
    print(f"  ... and {len(AVAILABLE_MODELS) - 6} more\n")

    # Show default assignments
    print("Default Model Assignments:")
    for persona, model in list(DEFAULT_MODEL_ASSIGNMENTS.items())[:5]:
        print(f"  {persona}: {model}")
    print()

    # Create a fork
    print("Creating fork of yann_lecun...")
    fork = manager.fork(
        "yann_lecun",
        modifications={
            "personality.agreeableness": 0.7,
            "positions.AI_risk": "Serious concern requiring action",
        },
        fork_name="yann_lecun_safety_advocate",
        description="Yann LeCun variant who takes AI safety seriously",
        model="anthropic/claude-sonnet-4",
        tags=["safety", "experiment"],
    )
    print(f"  Created: {fork.fork_id}")
    print(f"  Model: {fork.model}")
    print(f"  Modifications: {fork.modifications}")
    print()

    # List forks
    print("All forks:")
    for f in manager.list_forks():
        print(f"  - {f.fork_id} (from {f.parent_id})")
