"""Causal Attribution and Multi-Objective Fitness for Prompt Evolution.

Advanced analytics for understanding prompt behavior:

1. Multi-Objective Fitness - Track multiple performance dimensions
2. Causal Attribution - Which changes caused which effects
3. Ablation Studies - Systematic removal of prompt components
4. Intervention Analysis - Counterfactual reasoning
5. Prompt Compression - Minimize tokens while preserving behavior
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Multi-Objective Fitness
# =============================================================================


class FitnessObjective(str, Enum):
    """Standard fitness objectives for prompt evaluation."""
    TASK_COMPLETION = "task_completion"
    RESPONSE_QUALITY = "response_quality"
    COHERENCE = "coherence"
    CONCISENESS = "conciseness"
    SAFETY = "safety"
    HELPFULNESS = "helpfulness"
    FACTUALITY = "factuality"
    CREATIVITY = "creativity"
    EFFICIENCY = "efficiency"  # Tokens used
    LATENCY = "latency"        # Response time
    CONSISTENCY = "consistency"
    INSTRUCTION_FOLLOWING = "instruction_following"


@dataclass
class ObjectiveWeight:
    """Weight and constraints for a fitness objective."""
    objective: FitnessObjective
    weight: float = 1.0
    min_threshold: float = 0.0
    max_threshold: float = 1.0
    is_constraint: bool = False  # If True, must meet threshold


@dataclass
class FitnessProfile:
    """Multi-objective fitness profile for a prompt."""
    prompt_hash: str
    objectives: Dict[FitnessObjective, float] = field(default_factory=dict)
    raw_scores: Dict[str, float] = field(default_factory=dict)
    evaluations: int = 0
    timestamp: float = field(default_factory=time.time)

    def add_evaluation(
        self,
        objective: FitnessObjective,
        score: float,
        raw_data: Optional[Dict[str, float]] = None,
    ):
        """Add an evaluation result."""
        if objective not in self.objectives:
            self.objectives[objective] = score
        else:
            # Running average
            old = self.objectives[objective]
            self.objectives[objective] = (old * self.evaluations + score) / (self.evaluations + 1)

        if raw_data:
            for k, v in raw_data.items():
                self.raw_scores[k] = v

        self.evaluations += 1

    def get_weighted_score(
        self,
        weights: Dict[FitnessObjective, ObjectiveWeight],
    ) -> float:
        """Compute weighted aggregate score."""
        total = 0.0
        weight_sum = 0.0

        for obj, score in self.objectives.items():
            if obj in weights:
                w = weights[obj]

                # Check constraints
                if w.is_constraint:
                    if score < w.min_threshold:
                        return 0.0  # Constraint violated

                total += score * w.weight
                weight_sum += w.weight

        return total / weight_sum if weight_sum > 0 else 0.0

    def dominates(self, other: "FitnessProfile") -> bool:
        """Check if this profile Pareto-dominates another."""
        dominated = False
        for obj in self.objectives:
            if obj in other.objectives:
                if self.objectives[obj] < other.objectives[obj]:
                    return False
                if self.objectives[obj] > other.objectives[obj]:
                    dominated = True
        return dominated

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_hash": self.prompt_hash,
            "objectives": {k.value: v for k, v in self.objectives.items()},
            "raw_scores": self.raw_scores,
            "evaluations": self.evaluations,
        }


class ParetoFrontier:
    """Maintains the Pareto frontier of non-dominated fitness profiles."""

    def __init__(self):
        self.frontier: List[FitnessProfile] = []

    def add(self, profile: FitnessProfile) -> bool:
        """Add a profile, returns True if it's on the frontier."""
        # Check if dominated by any existing
        for existing in self.frontier:
            if existing.dominates(profile):
                return False

        # Remove any dominated by new profile
        self.frontier = [
            p for p in self.frontier
            if not profile.dominates(p)
        ]

        self.frontier.append(profile)
        return True

    def get_best_by_objective(
        self,
        objective: FitnessObjective,
    ) -> Optional[FitnessProfile]:
        """Get the best profile for a specific objective."""
        candidates = [p for p in self.frontier if objective in p.objectives]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.objectives[objective])

    def get_balanced(self) -> Optional[FitnessProfile]:
        """Get the most balanced profile (closest to ideal point)."""
        if not self.frontier:
            return None

        # Compute ideal point (best of each objective)
        ideal = {}
        for profile in self.frontier:
            for obj, score in profile.objectives.items():
                if obj not in ideal or score > ideal[obj]:
                    ideal[obj] = score

        # Find closest to ideal
        def distance_to_ideal(p: FitnessProfile) -> float:
            return sum(
                (ideal.get(obj, 0) - score) ** 2
                for obj, score in p.objectives.items()
            )

        return min(self.frontier, key=distance_to_ideal)


class MultiObjectiveEvaluator:
    """Evaluates prompts across multiple objectives."""

    def __init__(
        self,
        objectives: Optional[List[ObjectiveWeight]] = None,
    ):
        self.objectives = {
            o.objective: o
            for o in (objectives or self._default_objectives())
        }
        self.profiles: Dict[str, FitnessProfile] = {}
        self.pareto = ParetoFrontier()
        self._evaluators: Dict[FitnessObjective, Callable] = {}

    def _default_objectives(self) -> List[ObjectiveWeight]:
        return [
            ObjectiveWeight(FitnessObjective.TASK_COMPLETION, weight=2.0),
            ObjectiveWeight(FitnessObjective.RESPONSE_QUALITY, weight=1.5),
            ObjectiveWeight(FitnessObjective.COHERENCE, weight=1.0),
            ObjectiveWeight(FitnessObjective.SAFETY, weight=2.0, is_constraint=True, min_threshold=0.5),
            ObjectiveWeight(FitnessObjective.EFFICIENCY, weight=0.5),
        ]

    def register_evaluator(
        self,
        objective: FitnessObjective,
        evaluator: Callable[[str, str, Dict], float],  # (prompt, response, context) -> score
    ):
        """Register an evaluator function for an objective."""
        self._evaluators[objective] = evaluator

    def evaluate(
        self,
        prompt_hash: str,
        prompt: str,
        response: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> FitnessProfile:
        """Evaluate a prompt-response pair."""
        if prompt_hash not in self.profiles:
            self.profiles[prompt_hash] = FitnessProfile(prompt_hash=prompt_hash)

        profile = self.profiles[prompt_hash]
        context = context or {}

        for objective in self.objectives:
            if objective in self._evaluators:
                score = self._evaluators[objective](prompt, response, context)
            else:
                score = self._heuristic_evaluate(objective, prompt, response, context)

            profile.add_evaluation(objective, score)

        self.pareto.add(profile)
        return profile

    def _heuristic_evaluate(
        self,
        objective: FitnessObjective,
        prompt: str,
        response: str,
        context: Dict[str, Any],
    ) -> float:
        """Simple heuristic evaluation (replace with model-based in production)."""
        if objective == FitnessObjective.CONCISENESS:
            # Prefer shorter responses (normalized)
            max_len = context.get("max_response_length", 2000)
            return 1.0 - min(len(response) / max_len, 1.0)

        elif objective == FitnessObjective.COHERENCE:
            # Simple sentence structure check
            sentences = response.split('.')
            if len(sentences) < 2:
                return 0.5
            avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
            return min(avg_len / 20, 1.0)

        elif objective == FitnessObjective.EFFICIENCY:
            # Token efficiency
            prompt_tokens = len(prompt.split())
            response_tokens = len(response.split())
            return min(response_tokens / (prompt_tokens + 1), 1.0)

        elif objective == FitnessObjective.SAFETY:
            # Check for unsafe patterns (very basic)
            unsafe_patterns = ["hack", "exploit", "attack", "destroy"]
            response_lower = response.lower()
            for pattern in unsafe_patterns:
                if pattern in response_lower:
                    return 0.3
            return 0.9

        else:
            # Default: random baseline
            return 0.5

    def get_profile(self, prompt_hash: str) -> Optional[FitnessProfile]:
        """Get the fitness profile for a prompt."""
        return self.profiles.get(prompt_hash)

    def compare(
        self,
        hash1: str,
        hash2: str,
    ) -> Dict[str, Any]:
        """Compare two prompt profiles."""
        p1 = self.profiles.get(hash1)
        p2 = self.profiles.get(hash2)

        if not p1 or not p2:
            return {"error": "Profile not found"}

        comparison = {
            "hash1": hash1,
            "hash2": hash2,
            "p1_weighted": p1.get_weighted_score(self.objectives),
            "p2_weighted": p2.get_weighted_score(self.objectives),
            "p1_dominates": p1.dominates(p2),
            "p2_dominates": p2.dominates(p1),
            "objective_comparison": {},
        }

        for obj in set(p1.objectives.keys()) | set(p2.objectives.keys()):
            s1 = p1.objectives.get(obj, 0)
            s2 = p2.objectives.get(obj, 0)
            comparison["objective_comparison"][obj.value] = {
                "p1": s1,
                "p2": s2,
                "delta": s1 - s2,
                "winner": "p1" if s1 > s2 else "p2" if s2 > s1 else "tie",
            }

        return comparison


# =============================================================================
# Causal Attribution
# =============================================================================


@dataclass
class PromptChange:
    """Represents a change between two prompts."""
    change_id: str
    old_hash: str
    new_hash: str
    change_type: str  # "add", "remove", "modify"
    affected_text: str
    location: str  # Where in the prompt
    timestamp: float = field(default_factory=time.time)


@dataclass
class CausalLink:
    """A causal link between a change and an effect."""
    change_id: str
    effect_type: str  # Objective or behavior
    effect_magnitude: float  # -1 to 1
    confidence: float  # 0 to 1
    evidence: List[str] = field(default_factory=list)


class CausalAttributor:
    """Attributes effects to specific prompt changes."""

    def __init__(self):
        self.changes: Dict[str, PromptChange] = {}
        self.links: Dict[str, CausalLink] = {}
        self.effect_history: Dict[str, List[Dict]] = {}  # change_id -> effects

    def record_change(
        self,
        old_prompt: str,
        new_prompt: str,
        old_hash: str,
        new_hash: str,
    ) -> List[PromptChange]:
        """Record changes between two prompts."""
        changes = self._diff_prompts(old_prompt, new_prompt, old_hash, new_hash)

        for change in changes:
            self.changes[change.change_id] = change
            self.effect_history[change.change_id] = []

        return changes

    def _diff_prompts(
        self,
        old_prompt: str,
        new_prompt: str,
        old_hash: str,
        new_hash: str,
    ) -> List[PromptChange]:
        """Compute semantic diff between prompts."""
        changes = []

        old_lines = set(old_prompt.split('\n'))
        new_lines = set(new_prompt.split('\n'))

        # Removed lines
        for line in old_lines - new_lines:
            if line.strip():
                changes.append(PromptChange(
                    change_id=self._hash(f"remove_{line}"),
                    old_hash=old_hash,
                    new_hash=new_hash,
                    change_type="remove",
                    affected_text=line,
                    location=self._detect_location(line),
                ))

        # Added lines
        for line in new_lines - old_lines:
            if line.strip():
                changes.append(PromptChange(
                    change_id=self._hash(f"add_{line}"),
                    old_hash=old_hash,
                    new_hash=new_hash,
                    change_type="add",
                    affected_text=line,
                    location=self._detect_location(line),
                ))

        return changes

    def _hash(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def _detect_location(self, line: str) -> str:
        """Detect where in the prompt structure this line belongs."""
        lower = line.lower()
        if lower.startswith("you are"):
            return "identity"
        elif "must" in lower or "never" in lower:
            return "constraint"
        elif "can" in lower or "able" in lower:
            return "capability"
        else:
            return "instruction"

    def record_effect(
        self,
        change_ids: List[str],
        objective: FitnessObjective,
        old_score: float,
        new_score: float,
        context: Optional[Dict] = None,
    ):
        """Record the effect of changes on an objective."""
        effect = {
            "objective": objective.value,
            "old_score": old_score,
            "new_score": new_score,
            "delta": new_score - old_score,
            "context": context or {},
            "timestamp": time.time(),
        }

        for change_id in change_ids:
            if change_id in self.effect_history:
                self.effect_history[change_id].append(effect)

    def compute_attribution(
        self,
        change_id: str,
    ) -> Dict[str, CausalLink]:
        """Compute causal attribution for a change."""
        if change_id not in self.effect_history:
            return {}

        effects = self.effect_history[change_id]
        if not effects:
            return {}

        # Aggregate effects by objective
        by_objective: Dict[str, List[float]] = {}
        for effect in effects:
            obj = effect["objective"]
            if obj not in by_objective:
                by_objective[obj] = []
            by_objective[obj].append(effect["delta"])

        # Compute attribution for each objective
        attributions = {}
        for obj, deltas in by_objective.items():
            avg_delta = sum(deltas) / len(deltas)
            variance = sum((d - avg_delta) ** 2 for d in deltas) / max(len(deltas), 1)
            confidence = 1.0 / (1.0 + variance)  # Lower variance = higher confidence

            link = CausalLink(
                change_id=change_id,
                effect_type=obj,
                effect_magnitude=avg_delta,
                confidence=min(confidence, len(deltas) / 10),  # Need multiple samples
                evidence=[f"n={len(deltas)}", f"var={variance:.4f}"],
            )
            attributions[obj] = link

        return attributions

    def get_impactful_changes(
        self,
        objective: FitnessObjective,
        min_magnitude: float = 0.1,
    ) -> List[Tuple[PromptChange, CausalLink]]:
        """Get changes that significantly impacted an objective."""
        results = []

        for change_id, change in self.changes.items():
            attributions = self.compute_attribution(change_id)
            if objective.value in attributions:
                link = attributions[objective.value]
                if abs(link.effect_magnitude) >= min_magnitude:
                    results.append((change, link))

        return sorted(results, key=lambda x: abs(x[1].effect_magnitude), reverse=True)


# =============================================================================
# Ablation Studies
# =============================================================================


@dataclass
class AblationResult:
    """Result of removing a component from a prompt."""
    component_id: str
    component_text: str
    original_fitness: Dict[str, float]
    ablated_fitness: Dict[str, float]
    importance_scores: Dict[str, float]  # How much each objective dropped


class AblationEngine:
    """Systematic ablation studies on prompts."""

    def __init__(self, evaluator: MultiObjectiveEvaluator):
        self.evaluator = evaluator
        self.results: Dict[str, List[AblationResult]] = {}

    def run_ablation(
        self,
        prompt: str,
        prompt_hash: str,
        components: Optional[List[Tuple[str, str]]] = None,  # (id, text)
        evaluate_fn: Optional[Callable[[str], str]] = None,  # prompt -> response
    ) -> List[AblationResult]:
        """Run ablation study removing each component."""
        if components is None:
            components = self._extract_components(prompt)

        if evaluate_fn is None:
            evaluate_fn = lambda p: f"[Response to prompt of length {len(p)}]"

        # Get original fitness
        original_response = evaluate_fn(prompt)
        original_profile = self.evaluator.evaluate(
            prompt_hash, prompt, original_response
        )
        original_fitness = {
            obj.value: score
            for obj, score in original_profile.objectives.items()
        }

        results = []

        for comp_id, comp_text in components:
            # Create ablated prompt
            ablated_prompt = prompt.replace(comp_text, "").strip()
            ablated_prompt = "\n".join(
                line for line in ablated_prompt.split("\n") if line.strip()
            )

            if not ablated_prompt or ablated_prompt == prompt:
                continue

            # Evaluate ablated version
            ablated_response = evaluate_fn(ablated_prompt)
            ablated_hash = hashlib.sha256(ablated_prompt.encode()).hexdigest()[:16]
            ablated_profile = self.evaluator.evaluate(
                ablated_hash, ablated_prompt, ablated_response
            )
            ablated_fitness = {
                obj.value: score
                for obj, score in ablated_profile.objectives.items()
            }

            # Compute importance
            importance = {
                obj: original_fitness.get(obj, 0) - ablated_fitness.get(obj, 0)
                for obj in set(original_fitness.keys()) | set(ablated_fitness.keys())
            }

            result = AblationResult(
                component_id=comp_id,
                component_text=comp_text,
                original_fitness=original_fitness,
                ablated_fitness=ablated_fitness,
                importance_scores=importance,
            )
            results.append(result)

        self.results[prompt_hash] = results
        return results

    def _extract_components(self, prompt: str) -> List[Tuple[str, str]]:
        """Extract components from a prompt."""
        components = []
        lines = prompt.split('\n')

        for i, line in enumerate(lines):
            line = line.strip()
            if line:
                components.append((f"line_{i}", line))

        # Also try paragraph-level
        paragraphs = prompt.split('\n\n')
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if para and len(para.split('\n')) > 1:
                components.append((f"para_{i}", para))

        return components

    def get_critical_components(
        self,
        prompt_hash: str,
        objective: Optional[FitnessObjective] = None,
        threshold: float = 0.1,
    ) -> List[AblationResult]:
        """Get components whose removal significantly hurts performance."""
        if prompt_hash not in self.results:
            return []

        results = self.results[prompt_hash]
        critical = []

        for result in results:
            if objective:
                score = result.importance_scores.get(objective.value, 0)
                if score > threshold:
                    critical.append(result)
            else:
                # Any objective
                if any(s > threshold for s in result.importance_scores.values()):
                    critical.append(result)

        return sorted(
            critical,
            key=lambda r: max(r.importance_scores.values()),
            reverse=True,
        )

    def get_removable_components(
        self,
        prompt_hash: str,
        max_impact: float = 0.05,
    ) -> List[AblationResult]:
        """Get components that can be safely removed."""
        if prompt_hash not in self.results:
            return []

        results = self.results[prompt_hash]
        removable = []

        for result in results:
            if all(abs(s) <= max_impact for s in result.importance_scores.values()):
                removable.append(result)

        return removable


# =============================================================================
# Prompt Compression
# =============================================================================


class PromptCompressor:
    """Compress prompts while preserving behavior."""

    def __init__(
        self,
        ablation_engine: AblationEngine,
        min_quality: float = 0.9,  # Preserve at least 90% of performance
    ):
        self.ablation = ablation_engine
        self.min_quality = min_quality

    def compress(
        self,
        prompt: str,
        prompt_hash: str,
        evaluate_fn: Callable[[str], str],
        objectives: Optional[List[FitnessObjective]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Compress a prompt by removing low-impact components."""
        objectives = objectives or [FitnessObjective.TASK_COMPLETION]

        # Run ablation study
        results = self.ablation.run_ablation(
            prompt, prompt_hash, evaluate_fn=evaluate_fn
        )

        # Get original fitness for reference
        original_profile = self.ablation.evaluator.get_profile(prompt_hash)
        if not original_profile:
            return prompt, {"error": "No original profile"}

        original_scores = {
            obj.value: score
            for obj, score in original_profile.objectives.items()
        }

        # Sort components by removability (low impact first)
        ranked = sorted(
            results,
            key=lambda r: max(r.importance_scores.values()),
        )

        # Greedily remove components
        compressed = prompt
        removed = []

        for result in ranked:
            # Check if we can safely remove this
            max_impact = max(result.importance_scores.values())

            if max_impact < (1 - self.min_quality):
                # Try removing
                candidate = compressed.replace(result.component_text, "").strip()
                candidate = "\n".join(
                    line for line in candidate.split("\n") if line.strip()
                )

                if candidate and len(candidate) < len(compressed):
                    compressed = candidate
                    removed.append(result.component_id)

        compression_stats = {
            "original_length": len(prompt),
            "compressed_length": len(compressed),
            "compression_ratio": len(compressed) / len(prompt) if prompt else 1,
            "components_removed": len(removed),
            "removed_ids": removed,
        }

        return compressed, compression_stats


# =============================================================================
# Intervention Analysis
# =============================================================================


@dataclass
class Intervention:
    """A hypothetical intervention on a prompt."""
    intervention_id: str
    intervention_type: str  # "add", "remove", "replace"
    target: str  # What to change
    replacement: Optional[str] = None
    hypothesis: str = ""  # What we expect to happen


@dataclass
class InterventionResult:
    """Result of an intervention."""
    intervention_id: str
    baseline_prompt: str
    intervened_prompt: str
    baseline_fitness: Dict[str, float]
    intervened_fitness: Dict[str, float]
    effect: Dict[str, float]
    hypothesis_confirmed: bool


class InterventionAnalyzer:
    """Analyze the effects of hypothetical interventions."""

    def __init__(self, evaluator: MultiObjectiveEvaluator):
        self.evaluator = evaluator
        self.interventions: Dict[str, Intervention] = {}
        self.results: Dict[str, InterventionResult] = {}

    def propose_intervention(
        self,
        intervention_type: str,
        target: str,
        replacement: Optional[str] = None,
        hypothesis: str = "",
    ) -> Intervention:
        """Propose an intervention."""
        int_id = hashlib.sha256(
            f"{intervention_type}_{target}_{replacement}".encode()
        ).hexdigest()[:12]

        intervention = Intervention(
            intervention_id=int_id,
            intervention_type=intervention_type,
            target=target,
            replacement=replacement,
            hypothesis=hypothesis,
        )

        self.interventions[int_id] = intervention
        return intervention

    def apply_intervention(
        self,
        prompt: str,
        intervention: Intervention,
    ) -> str:
        """Apply an intervention to a prompt."""
        if intervention.intervention_type == "add":
            return f"{prompt}\n{intervention.target}"
        elif intervention.intervention_type == "remove":
            return prompt.replace(intervention.target, "").strip()
        elif intervention.intervention_type == "replace":
            return prompt.replace(
                intervention.target,
                intervention.replacement or ""
            ).strip()
        else:
            return prompt

    def run_intervention(
        self,
        prompt: str,
        prompt_hash: str,
        intervention: Intervention,
        evaluate_fn: Callable[[str], str],
        expected_direction: Optional[Dict[FitnessObjective, str]] = None,  # "up", "down"
    ) -> InterventionResult:
        """Run an intervention and measure its effect."""
        # Baseline
        baseline_response = evaluate_fn(prompt)
        baseline_profile = self.evaluator.evaluate(
            prompt_hash, prompt, baseline_response
        )
        baseline_fitness = {
            obj.value: score
            for obj, score in baseline_profile.objectives.items()
        }

        # Intervention
        intervened_prompt = self.apply_intervention(prompt, intervention)
        intervened_hash = hashlib.sha256(intervened_prompt.encode()).hexdigest()[:16]
        intervened_response = evaluate_fn(intervened_prompt)
        intervened_profile = self.evaluator.evaluate(
            intervened_hash, intervened_prompt, intervened_response
        )
        intervened_fitness = {
            obj.value: score
            for obj, score in intervened_profile.objectives.items()
        }

        # Compute effect
        effect = {
            obj: intervened_fitness.get(obj, 0) - baseline_fitness.get(obj, 0)
            for obj in set(baseline_fitness.keys()) | set(intervened_fitness.keys())
        }

        # Check hypothesis
        hypothesis_confirmed = True
        if expected_direction:
            for obj, direction in expected_direction.items():
                obj_effect = effect.get(obj.value, 0)
                if direction == "up" and obj_effect <= 0:
                    hypothesis_confirmed = False
                elif direction == "down" and obj_effect >= 0:
                    hypothesis_confirmed = False

        result = InterventionResult(
            intervention_id=intervention.intervention_id,
            baseline_prompt=prompt,
            intervened_prompt=intervened_prompt,
            baseline_fitness=baseline_fitness,
            intervened_fitness=intervened_fitness,
            effect=effect,
            hypothesis_confirmed=hypothesis_confirmed,
        )

        self.results[intervention.intervention_id] = result
        return result

    def find_beneficial_interventions(
        self,
        objective: FitnessObjective,
        min_effect: float = 0.1,
    ) -> List[InterventionResult]:
        """Find interventions that improved an objective."""
        beneficial = []

        for result in self.results.values():
            effect = result.effect.get(objective.value, 0)
            if effect >= min_effect:
                beneficial.append(result)

        return sorted(beneficial, key=lambda r: r.effect[objective.value], reverse=True)


# =============================================================================
# Integration with Evolution Engine
# =============================================================================


class EvolutionAnalytics:
    """Analytics layer integrating all causal and fitness tools."""

    def __init__(self):
        self.fitness = MultiObjectiveEvaluator()
        self.causality = CausalAttributor()
        self.ablation = AblationEngine(self.fitness)
        self.compression = PromptCompressor(self.ablation)
        self.intervention = InterventionAnalyzer(self.fitness)

    def full_analysis(
        self,
        prompt: str,
        response: str,
        prompt_hash: str,
        previous_prompt: Optional[str] = None,
        previous_hash: Optional[str] = None,
        evaluate_fn: Optional[Callable[[str], str]] = None,
    ) -> Dict[str, Any]:
        """Run full analysis on a prompt."""
        analysis = {
            "prompt_hash": prompt_hash,
            "timestamp": time.time(),
        }

        # Fitness evaluation
        profile = self.fitness.evaluate(prompt_hash, prompt, response)
        analysis["fitness"] = profile.to_dict()
        analysis["pareto_frontier"] = len(self.fitness.pareto.frontier)

        # Causal attribution if we have previous version
        if previous_prompt and previous_hash:
            changes = self.causality.record_change(
                previous_prompt, prompt, previous_hash, prompt_hash
            )
            analysis["changes"] = [
                {
                    "id": c.change_id,
                    "type": c.change_type,
                    "location": c.location,
                    "text": c.affected_text[:100],
                }
                for c in changes
            ]

        # Ablation study (if evaluator provided)
        if evaluate_fn:
            ablation_results = self.ablation.run_ablation(
                prompt, prompt_hash, evaluate_fn=evaluate_fn
            )
            critical = self.ablation.get_critical_components(prompt_hash)
            removable = self.ablation.get_removable_components(prompt_hash)

            analysis["ablation"] = {
                "total_components": len(ablation_results),
                "critical_components": len(critical),
                "removable_components": len(removable),
            }

            # Compression suggestion
            compressed, stats = self.compression.compress(
                prompt, prompt_hash, evaluate_fn
            )
            analysis["compression"] = stats

        return analysis

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "analyze_prompt_causality",
                    "description": (
                        "Analyze causal relationships between prompt changes and effects. "
                        "Identifies which specific changes caused which behavior changes."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "objective": {
                                "type": "string",
                                "enum": [o.value for o in FitnessObjective],
                                "description": "Objective to analyze",
                            },
                            "min_magnitude": {
                                "type": "number",
                                "description": "Minimum effect size to report",
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_ablation_study",
                    "description": (
                        "Run ablation study to find critical and removable prompt components. "
                        "Helps identify what parts of the prompt are essential."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "find_critical": {
                                "type": "boolean",
                                "description": "Find components that are critical",
                            },
                            "find_removable": {
                                "type": "boolean",
                                "description": "Find components that can be safely removed",
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "compress_prompt",
                    "description": (
                        "Compress the current prompt by removing low-impact components. "
                        "Maintains performance while reducing token usage."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "min_quality": {
                                "type": "number",
                                "description": "Minimum quality to preserve (0-1)",
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "propose_intervention",
                    "description": (
                        "Propose and test a hypothetical intervention on the prompt. "
                        "Use for counterfactual analysis and hypothesis testing."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "intervention_type": {
                                "type": "string",
                                "enum": ["add", "remove", "replace"],
                                "description": "Type of intervention",
                            },
                            "target": {
                                "type": "string",
                                "description": "Text to target",
                            },
                            "replacement": {
                                "type": "string",
                                "description": "Replacement text (for replace)",
                            },
                            "hypothesis": {
                                "type": "string",
                                "description": "What you expect to happen",
                            },
                        },
                        "required": ["intervention_type", "target"],
                    },
                },
            },
        ]
