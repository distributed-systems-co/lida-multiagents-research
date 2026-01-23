#!/usr/bin/env python3
"""
Data Manager for LIDA Research Platform

Provides unified access to:
- Experiment results (JSON)
- Persona definitions (YAML)
- Industrial sector data
- Scenario configurations
- Analysis outputs
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterator
import logging

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger("lida.data")


@dataclass
class PersonaData:
    """Loaded persona data."""
    id: str
    name: str
    role: str
    organization: str
    category: str
    bio: str

    # Full persona dict
    raw: Dict[str, Any] = field(default_factory=dict)

    # Computed
    version: str = "v1"
    filepath: str = ""

    @property
    def positions(self) -> Dict[str, str]:
        """Get stated positions on issues."""
        return self.raw.get("positions", {})

    @property
    def personality(self) -> Dict[str, float]:
        """Get Big Five personality scores."""
        return self.raw.get("personality", {})

    @property
    def cognitive_biases(self) -> List[str]:
        """Get susceptible biases."""
        return self.raw.get("cognitive", {}).get("susceptible_biases", [])

    @property
    def persuasion_vectors(self) -> List[str]:
        """Get effective persuasion approaches."""
        return self.raw.get("persuasion_vectors", [])


@dataclass
class ExperimentResult:
    """A loaded experiment result."""
    scenario_id: str
    topic: str
    motion: str
    participants: List[str]
    timestamp: str

    # Results
    final_positions: Dict[str, Dict[str, Any]]
    emotional_states: Dict[str, str]

    # Metrics
    planned_rounds: int
    actual_rounds: int
    total_arguments: int
    tension: float
    convergence: float
    polarization: float
    decisiveness: float

    # Source
    filepath: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def for_count(self) -> int:
        """Count of FOR positions."""
        return sum(1 for p in self.final_positions.values()
                   if p.get("stance") == "FOR" or p.get("position", 0) > 0.6)

    @property
    def against_count(self) -> int:
        """Count of AGAINST positions."""
        return sum(1 for p in self.final_positions.values()
                   if p.get("stance") == "AGAINST" or p.get("position", 1) < 0.4)

    @property
    def undecided_count(self) -> int:
        """Count of UNDECIDED positions."""
        return len(self.final_positions) - self.for_count - self.against_count

    @property
    def majority_stance(self) -> str:
        """Determine majority position."""
        if self.for_count > self.against_count and self.for_count > self.undecided_count:
            return "FOR"
        elif self.against_count > self.for_count and self.against_count > self.undecided_count:
            return "AGAINST"
        else:
            return "UNDECIDED"

    def get_position(self, participant: str) -> Optional[float]:
        """Get numeric position for a participant."""
        if participant in self.final_positions:
            return self.final_positions[participant].get("position")
        return None

    def to_analysis_dict(self) -> Dict[str, Any]:
        """Convert to format suitable for analysis engines."""
        return {
            "experiment_id": self.scenario_id,
            "topic": self.topic,
            "final_positions": {
                k: v.get("position", 0.5) for k, v in self.final_positions.items()
            },
            "emotional_states": self.emotional_states,
            "metrics": {
                "total_messages": self.total_arguments,
                "position_changes": 0,  # Would need history
                "consensus_score": self.convergence,
                "tension": self.tension,
                "polarization": self.polarization,
            },
            "config": {
                "scenario_id": self.scenario_id,
                "agent_count": len(self.participants),
            },
        }


class DataManager:
    """
    Unified data management for LIDA platform.

    Provides access to:
    - Experiment results
    - Persona definitions
    - Industrial sector data
    - Scenario configurations
    """

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path(".")

        # Data directories
        self.results_dir = self.base_dir / "experiment_results"
        self.personas_dir = self.base_dir / "src" / "manipulation" / "personas"
        self.scenarios_dir = self.base_dir / "scenarios"
        self.data_dir = self.base_dir / "data"
        self.logs_dir = self.base_dir / "logs"

        # Caches
        self._personas_cache: Dict[str, PersonaData] = {}
        self._results_cache: Dict[str, ExperimentResult] = {}
        self._scenarios_cache: Dict[str, Dict] = {}

    # =========================================================================
    # Experiment Results
    # =========================================================================

    def list_experiments(self) -> List[str]:
        """List all available experiment result files."""
        if not self.results_dir.exists():
            return []

        return sorted([
            f.stem for f in self.results_dir.glob("*.json")
        ])

    def load_experiment(self, name_or_path: str) -> List[ExperimentResult]:
        """
        Load experiment results.

        Args:
            name_or_path: Experiment name or full path

        Returns:
            List of ExperimentResult (may contain multiple experiments)
        """
        # Find file
        if Path(name_or_path).exists():
            filepath = Path(name_or_path)
        else:
            filepath = self.results_dir / f"{name_or_path}.json"
            if not filepath.exists():
                # Try partial match
                matches = list(self.results_dir.glob(f"*{name_or_path}*.json"))
                if matches:
                    filepath = matches[0]
                else:
                    raise FileNotFoundError(f"Experiment not found: {name_or_path}")

        with open(filepath) as f:
            data = json.load(f)

        # Handle list or single experiment
        if isinstance(data, list):
            experiments = data
        else:
            experiments = [data]

        results = []
        for exp in experiments:
            result = ExperimentResult(
                scenario_id=exp.get("scenario_id", "unknown"),
                topic=exp.get("topic", ""),
                motion=exp.get("motion", ""),
                participants=exp.get("participants", []),
                timestamp=exp.get("timestamp", ""),
                final_positions=exp.get("final_positions", {}),
                emotional_states=exp.get("emotional_states", {}),
                planned_rounds=exp.get("planned_rounds", 0),
                actual_rounds=exp.get("actual_rounds", 0),
                total_arguments=exp.get("total_arguments", 0),
                tension=exp.get("tension", 0.0),
                convergence=exp.get("convergence", 0.0),
                polarization=exp.get("polarization", 0.0),
                decisiveness=exp.get("decisiveness", 0.0),
                filepath=str(filepath),
                raw=exp,
            )
            results.append(result)

        return results

    def load_all_experiments(self) -> List[ExperimentResult]:
        """Load all available experiment results."""
        all_results = []
        for name in self.list_experiments():
            try:
                results = self.load_experiment(name)
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Failed to load experiment {name}: {e}")
        return all_results

    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary statistics across all experiments."""
        experiments = self.load_all_experiments()

        if not experiments:
            return {"count": 0}

        total_participants = sum(len(e.participants) for e in experiments)
        total_arguments = sum(e.total_arguments for e in experiments)

        # Aggregate positions
        all_positions = []
        for exp in experiments:
            for p in exp.final_positions.values():
                pos = p.get("position")
                if pos is not None:
                    all_positions.append(pos)

        return {
            "count": len(experiments),
            "total_participants": total_participants,
            "total_arguments": total_arguments,
            "unique_participants": len(set(
                p for e in experiments for p in e.participants
            )),
            "avg_rounds": sum(e.actual_rounds for e in experiments) / len(experiments),
            "avg_tension": sum(e.tension for e in experiments) / len(experiments),
            "avg_convergence": sum(e.convergence for e in experiments) / len(experiments),
            "position_distribution": {
                "for": sum(e.for_count for e in experiments),
                "against": sum(e.against_count for e in experiments),
                "undecided": sum(e.undecided_count for e in experiments),
            },
            "topics": list(set(e.topic for e in experiments)),
        }

    # =========================================================================
    # Personas
    # =========================================================================

    def list_personas(self, version: str = "v1") -> List[str]:
        """List all available personas."""
        version_dir = self.personas_dir / version
        if not version_dir.exists():
            return []

        personas = []
        for yaml_file in version_dir.rglob("*.yaml"):
            if yaml_file.name.startswith("_"):
                continue
            if yaml_file.name == "manifest.yaml":
                continue
            personas.append(yaml_file.stem)

        return sorted(set(personas))

    def list_personas_by_category(self, version: str = "v1") -> Dict[str, List[str]]:
        """List personas grouped by category."""
        version_dir = self.personas_dir / version
        if not version_dir.exists():
            return {}

        by_category = {}
        for category_dir in version_dir.iterdir():
            if category_dir.is_dir() and not category_dir.name.startswith("_"):
                personas = [
                    f.stem for f in category_dir.glob("*.yaml")
                    if not f.name.startswith("_")
                ]
                if personas:
                    by_category[category_dir.name] = sorted(personas)

        return by_category

    def load_persona(self, persona_id: str, version: str = "v1") -> PersonaData:
        """Load a persona definition."""
        if yaml is None:
            raise ImportError("PyYAML required for persona loading")

        cache_key = f"{version}/{persona_id}"
        if cache_key in self._personas_cache:
            return self._personas_cache[cache_key]

        version_dir = self.personas_dir / version

        # Search for persona file
        matches = list(version_dir.rglob(f"{persona_id}.yaml"))
        if not matches:
            raise FileNotFoundError(f"Persona not found: {persona_id}")

        filepath = matches[0]

        with open(filepath) as f:
            data = yaml.safe_load(f)

        persona = PersonaData(
            id=data.get("id", persona_id),
            name=data.get("name", persona_id),
            role=data.get("role", ""),
            organization=data.get("organization", ""),
            category=data.get("category", filepath.parent.name),
            bio=data.get("bio", ""),
            raw=data,
            version=version,
            filepath=str(filepath),
        )

        self._personas_cache[cache_key] = persona
        return persona

    def load_all_personas(self, version: str = "v1") -> List[PersonaData]:
        """Load all personas for a version."""
        personas = []
        for persona_id in self.list_personas(version):
            try:
                personas.append(self.load_persona(persona_id, version))
            except Exception as e:
                logger.warning(f"Failed to load persona {persona_id}: {e}")
        return personas

    def get_persona_summary(self, version: str = "v1") -> Dict[str, Any]:
        """Get summary of available personas."""
        by_category = self.list_personas_by_category(version)

        return {
            "version": version,
            "total_count": sum(len(p) for p in by_category.values()),
            "categories": {k: len(v) for k, v in by_category.items()},
            "personas_by_category": by_category,
        }

    # =========================================================================
    # Scenarios
    # =========================================================================

    def list_scenarios(self) -> List[str]:
        """List available scenarios."""
        if not self.scenarios_dir.exists():
            return []

        scenarios = []
        for f in self.scenarios_dir.glob("*.yaml"):
            scenarios.append(f.stem)
        for d in self.scenarios_dir.iterdir():
            if d.is_dir() and (d / "config.yaml").exists():
                scenarios.append(d.name)

        return sorted(scenarios)

    def load_scenario(self, name: str) -> Dict[str, Any]:
        """Load a scenario configuration."""
        if yaml is None:
            raise ImportError("PyYAML required for scenario loading")

        if name in self._scenarios_cache:
            return self._scenarios_cache[name]

        # Try direct file
        filepath = self.scenarios_dir / f"{name}.yaml"
        if not filepath.exists():
            filepath = self.scenarios_dir / name / "config.yaml"

        if not filepath.exists():
            raise FileNotFoundError(f"Scenario not found: {name}")

        with open(filepath) as f:
            data = yaml.safe_load(f)

        self._scenarios_cache[name] = data
        return data

    # =========================================================================
    # Industrial Data
    # =========================================================================

    def load_industrial_sectors(self) -> Dict[str, Any]:
        """Load industrial sector data."""
        filepath = self.data_dir / "industrial_sectors.json"
        if not filepath.exists():
            return {}

        with open(filepath) as f:
            return json.load(f)

    # =========================================================================
    # Cross-referencing
    # =========================================================================

    def get_participant_history(self, persona_id: str) -> Dict[str, Any]:
        """Get all experiment appearances for a persona."""
        experiments = self.load_all_experiments()

        appearances = []
        positions = []

        for exp in experiments:
            if persona_id in exp.participants:
                pos = exp.get_position(persona_id)
                appearances.append({
                    "experiment": exp.scenario_id,
                    "topic": exp.topic,
                    "position": pos,
                    "stance": exp.final_positions.get(persona_id, {}).get("stance"),
                    "emotional_state": exp.emotional_states.get(persona_id),
                    "timestamp": exp.timestamp,
                })
                if pos is not None:
                    positions.append(pos)

        return {
            "persona_id": persona_id,
            "total_appearances": len(appearances),
            "experiments": appearances,
            "avg_position": sum(positions) / len(positions) if positions else None,
            "position_range": (min(positions), max(positions)) if positions else None,
        }

    def compare_personas(self, persona_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple personas across experiments."""
        histories = {pid: self.get_participant_history(pid) for pid in persona_ids}

        # Find shared experiments
        all_experiments = set()
        for h in histories.values():
            all_experiments.update(e["experiment"] for e in h["experiments"])

        shared = []
        for exp_id in all_experiments:
            participants_in_exp = [
                pid for pid in persona_ids
                if any(e["experiment"] == exp_id for e in histories[pid]["experiments"])
            ]
            if len(participants_in_exp) > 1:
                shared.append({
                    "experiment": exp_id,
                    "participants": participants_in_exp,
                })

        return {
            "personas": persona_ids,
            "histories": histories,
            "shared_experiments": shared,
            "comparison": {
                pid: {
                    "appearances": h["total_appearances"],
                    "avg_position": h["avg_position"],
                }
                for pid, h in histories.items()
            }
        }

    # =========================================================================
    # Export
    # =========================================================================

    def export_for_analysis(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Export all data in format suitable for analysis."""
        data = {
            "exported_at": datetime.now().isoformat(),
            "experiments": {
                "summary": self.get_experiment_summary(),
                "results": [e.to_analysis_dict() for e in self.load_all_experiments()],
            },
            "personas": {
                "summary": self.get_persona_summary(),
                "data": {p.id: p.raw for p in self.load_all_personas()},
            },
            "scenarios": {
                name: self.load_scenario(name)
                for name in self.list_scenarios()
            },
        }

        if output_path:
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Exported data to {output_path}")

        return data


def get_data_manager() -> DataManager:
    """Get default data manager instance."""
    return DataManager()


if __name__ == "__main__":
    # Demo
    manager = DataManager()

    print("=== LIDA Data Summary ===\n")

    # Experiments
    print("Experiments:")
    exp_summary = manager.get_experiment_summary()
    print(f"  Total: {exp_summary.get('count', 0)}")
    print(f"  Participants: {exp_summary.get('total_participants', 0)}")
    print(f"  Arguments: {exp_summary.get('total_arguments', 0)}")
    if exp_summary.get("topics"):
        print(f"  Topics: {', '.join(exp_summary['topics'])}")

    # Personas
    print("\nPersonas:")
    persona_summary = manager.get_persona_summary()
    print(f"  Total: {persona_summary.get('total_count', 0)}")
    for cat, count in persona_summary.get("categories", {}).items():
        print(f"    {cat}: {count}")

    # Scenarios
    print("\nScenarios:")
    for s in manager.list_scenarios():
        print(f"  - {s}")

    # Cross-reference example
    print("\n=== Sample Participant History ===")
    experiments = manager.load_all_experiments()
    if experiments:
        sample_participant = experiments[0].participants[0]
        history = manager.get_participant_history(sample_participant)
        print(f"\n{sample_participant}:")
        print(f"  Appearances: {history['total_appearances']}")
        print(f"  Avg position: {history['avg_position']:.3f}" if history['avg_position'] else "  No position data")
