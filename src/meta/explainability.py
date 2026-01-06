"""Explainability and interpretability framework for agent decisions.

Provides:
- Decision trace visualization
- Attention mechanism analysis
- Saliency mapping for inputs
- Counterfactual explanations
- Feature importance attribution
- Natural language explanation generation
- Causal pathway extraction
"""
from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable, Set
from enum import Enum
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class ExplanationType(Enum):
    """Types of explanations."""
    FEATURE_IMPORTANCE = "feature_importance"  # Which features mattered most
    COUNTERFACTUAL = "counterfactual"  # What would change the decision
    EXAMPLE_BASED = "example_based"  # Similar examples
    RULE_BASED = "rule_based"  # Decision rules
    CAUSAL = "causal"  # Causal pathways
    ATTENTION = "attention"  # Attention weights
    SALIENCY = "saliency"  # Input saliency maps


class ExplanationAudience(Enum):
    """Target audience for explanation."""
    EXPERT = "expert"  # Technical expert
    DEVELOPER = "developer"  # System developer
    DOMAIN_EXPERT = "domain_expert"  # Domain specialist
    END_USER = "end_user"  # Non-technical user
    REGULATOR = "regulator"  # Compliance/audit


@dataclass
class DecisionTrace:
    """Complete trace of a decision-making process."""

    decision_id: str
    agent_id: str
    timestamp: datetime

    # Input information
    inputs: Dict[str, Any]
    context: Dict[str, Any]

    # Decision output
    decision: Any
    confidence: float

    # Reasoning steps
    reasoning_steps: List[Dict[str, Any]] = field(default_factory=list)

    # Intermediate activations/states
    intermediate_states: List[Dict[str, Any]] = field(default_factory=list)

    # Attention weights (if applicable)
    attention_weights: Optional[Dict[str, np.ndarray]] = None

    # Feature importance scores
    feature_importance: Optional[Dict[str, float]] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "decision_id": self.decision_id,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "inputs": self.inputs,
            "context": self.context,
            "decision": self.decision,
            "confidence": self.confidence,
            "reasoning_steps": self.reasoning_steps,
            "intermediate_states": self.intermediate_states,
            "attention_weights": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in (self.attention_weights or {}).items()
            },
            "feature_importance": self.feature_importance,
            "metadata": self.metadata,
        }


@dataclass
class Explanation:
    """An explanation for a decision."""

    explanation_type: ExplanationType
    content: str  # Natural language explanation
    evidence: Dict[str, Any]  # Supporting evidence
    confidence: float  # Confidence in explanation
    audience: ExplanationAudience

    # Visual representation (if applicable)
    visualization: Optional[Any] = None

    # Alternative explanations
    alternatives: List["Explanation"] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "type": self.explanation_type.value,
            "content": self.content,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "audience": self.audience.value,
            "alternatives": [alt.to_dict() for alt in self.alternatives],
        }


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════

class FeatureImportanceAnalyzer:
    """Analyze feature importance using various methods."""

    def __init__(self):
        pass

    def shapley_values(
        self,
        model_fn: Callable,
        instance: Dict[str, Any],
        baseline: Optional[Dict[str, Any]] = None,
        n_samples: int = 100,
    ) -> Dict[str, float]:
        """Compute SHAP-like feature importance using Shapley values.

        Args:
            model_fn: Function that takes feature dict and returns prediction
            instance: Instance to explain
            baseline: Baseline instance (average values if None)
            n_samples: Number of samples for approximation

        Returns:
            Dict mapping feature names to importance scores
        """
        features = list(instance.keys())

        if baseline is None:
            # Use zero/neutral baseline
            baseline = {f: 0 for f in features}

        # Compute marginal contributions
        importance = {f: 0.0 for f in features}

        for _ in range(n_samples):
            # Random permutation of features
            perm = np.random.permutation(features)

            # Build feature subset incrementally
            subset = baseline.copy()

            prev_pred = model_fn(subset)

            for feature in perm:
                # Add feature
                subset[feature] = instance[feature]
                curr_pred = model_fn(subset)

                # Marginal contribution
                contribution = curr_pred - prev_pred
                importance[feature] += contribution

                prev_pred = curr_pred

        # Average over samples
        for feature in features:
            importance[feature] /= n_samples

        return importance

    def permutation_importance(
        self,
        model_fn: Callable,
        instances: List[Dict[str, Any]],
        metric_fn: Callable,
        n_repeats: int = 10,
    ) -> Dict[str, Tuple[float, float]]:
        """Compute permutation importance.

        Args:
            model_fn: Prediction function
            instances: Validation instances
            metric_fn: Metric to measure (e.g., accuracy)
            n_repeats: Number of permutation repeats

        Returns:
            Dict mapping features to (mean_importance, std_importance)
        """
        # Baseline performance
        predictions = [model_fn(inst) for inst in instances]
        baseline_score = metric_fn(predictions)

        features = list(instances[0].keys())
        importance_scores = {f: [] for f in features}

        for feature in features:
            for _ in range(n_repeats):
                # Permute feature values
                permuted_instances = []
                feature_values = [inst[feature] for inst in instances]
                np.random.shuffle(feature_values)

                for inst, permuted_val in zip(instances, feature_values):
                    permuted_inst = inst.copy()
                    permuted_inst[feature] = permuted_val
                    permuted_instances.append(permuted_inst)

                # Evaluate
                permuted_predictions = [model_fn(inst) for inst in permuted_instances]
                permuted_score = metric_fn(permuted_predictions)

                # Importance = drop in performance
                importance_scores[feature].append(baseline_score - permuted_score)

        # Compute mean and std
        result = {}
        for feature, scores in importance_scores.items():
            result[feature] = (np.mean(scores), np.std(scores))

        return result


# ═══════════════════════════════════════════════════════════════════════════
# COUNTERFACTUAL EXPLANATIONS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CounterfactualExample:
    """A counterfactual example: minimal change that would alter decision."""

    original: Dict[str, Any]
    counterfactual: Dict[str, Any]
    changes: Dict[str, Tuple[Any, Any]]  # feature -> (old_value, new_value)
    distance: float  # How different from original
    plausibility: float  # How plausible the counterfactual is


class CounterfactualGenerator:
    """Generate counterfactual explanations."""

    def __init__(
        self,
        model_fn: Callable,
        feature_ranges: Dict[str, Tuple[Any, Any]],
    ):
        self.model_fn = model_fn
        self.feature_ranges = feature_ranges

    def generate(
        self,
        instance: Dict[str, Any],
        desired_output: Any,
        max_changes: int = 3,
        n_candidates: int = 100,
    ) -> Optional[CounterfactualExample]:
        """Generate counterfactual by search.

        Args:
            instance: Original instance
            desired_output: Target output value
            max_changes: Maximum number of features to change
            n_candidates: Number of candidates to try

        Returns:
            Best counterfactual found, or None
        """
        best_counterfactual = None
        best_distance = float('inf')

        features = list(instance.keys())

        for _ in range(n_candidates):
            # Randomly select features to change
            n_changes = np.random.randint(1, max_changes + 1)
            features_to_change = np.random.choice(features, n_changes, replace=False)

            # Generate candidate
            candidate = instance.copy()
            changes = {}

            for feature in features_to_change:
                # Sample new value from feature range
                if feature in self.feature_ranges:
                    min_val, max_val = self.feature_ranges[feature]
                    if isinstance(min_val, (int, float)):
                        new_val = np.random.uniform(min_val, max_val)
                    else:
                        # Categorical - random choice
                        new_val = np.random.choice([min_val, max_val])

                    candidate[feature] = new_val
                    changes[feature] = (instance[feature], new_val)

            # Check if produces desired output
            output = self.model_fn(candidate)

            if output == desired_output:
                # Compute distance
                distance = self._distance(instance, candidate)

                if distance < best_distance:
                    best_distance = distance
                    best_counterfactual = CounterfactualExample(
                        original=instance,
                        counterfactual=candidate,
                        changes=changes,
                        distance=distance,
                        plausibility=self._plausibility(candidate),
                    )

        return best_counterfactual

    def _distance(self, instance1: Dict, instance2: Dict) -> float:
        """Compute distance between instances."""
        total = 0.0
        for key in instance1.keys():
            val1, val2 = instance1[key], instance2[key]

            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric - normalized difference
                if key in self.feature_ranges:
                    min_val, max_val = self.feature_ranges[key]
                    if max_val > min_val:
                        total += abs(val1 - val2) / (max_val - min_val)
                else:
                    total += abs(val1 - val2)
            else:
                # Categorical - 0 if same, 1 if different
                total += 0 if val1 == val2 else 1

        return total

    def _plausibility(self, instance: Dict) -> float:
        """Estimate plausibility of instance (simplified)."""
        # In practice, would use data distribution
        # For now, check if within feature ranges
        score = 1.0

        for key, value in instance.items():
            if key in self.feature_ranges:
                min_val, max_val = self.feature_ranges[key]
                if isinstance(value, (int, float)):
                    if value < min_val or value > max_val:
                        score *= 0.5

        return score


# ═══════════════════════════════════════════════════════════════════════════
# ATTENTION VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

class AttentionAnalyzer:
    """Analyze and visualize attention mechanisms."""

    def __init__(self):
        pass

    def analyze_attention(
        self,
        attention_weights: np.ndarray,
        input_tokens: List[str],
        output_tokens: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Analyze attention patterns.

        Args:
            attention_weights: Attention matrix [query, key] or [layer, head, query, key]
            input_tokens: Input token labels
            output_tokens: Output token labels (if different from input)

        Returns:
            Analysis results including peak attention, entropy, etc.
        """
        if output_tokens is None:
            output_tokens = input_tokens

        # Handle multi-head attention
        if attention_weights.ndim == 4:
            # Average over layers and heads
            avg_attention = attention_weights.mean(axis=(0, 1))
        elif attention_weights.ndim == 3:
            # Average over heads
            avg_attention = attention_weights.mean(axis=0)
        else:
            avg_attention = attention_weights

        # Compute statistics
        peak_attention = []
        for i, output_token in enumerate(output_tokens):
            if i < avg_attention.shape[0]:
                max_idx = avg_attention[i].argmax()
                peak_attention.append({
                    "output_token": output_token,
                    "attends_to": input_tokens[max_idx] if max_idx < len(input_tokens) else "?",
                    "weight": float(avg_attention[i, max_idx]),
                })

        # Compute entropy of attention distribution
        entropy = []
        for i in range(avg_attention.shape[0]):
            dist = avg_attention[i]
            # Add small epsilon to avoid log(0)
            dist = dist + 1e-10
            dist = dist / dist.sum()
            ent = -(dist * np.log(dist)).sum()
            entropy.append(float(ent))

        return {
            "peak_attention": peak_attention,
            "avg_entropy": float(np.mean(entropy)),
            "max_entropy": float(np.max(entropy)),
            "attention_matrix": avg_attention.tolist(),
        }

    def visualize_attention(
        self,
        attention_weights: np.ndarray,
        input_tokens: List[str],
        output_tokens: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ):
        """Create attention heatmap visualization."""
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available for visualization")
            return None

        if output_tokens is None:
            output_tokens = input_tokens

        # Average if multi-head
        if attention_weights.ndim > 2:
            avg_attention = attention_weights.mean(axis=tuple(range(attention_weights.ndim - 2)))
        else:
            avg_attention = attention_weights

        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            avg_attention,
            xticklabels=input_tokens,
            yticklabels=output_tokens,
            cmap="viridis",
            cbar_kws={"label": "Attention Weight"},
        )
        plt.xlabel("Input Tokens")
        plt.ylabel("Output Tokens")
        plt.title("Attention Heatmap")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved attention visualization to {save_path}")

        return plt.gcf()


# ═══════════════════════════════════════════════════════════════════════════
# EXPLANATION GENERATION
# ═══════════════════════════════════════════════════════════════════════════

class ExplanationGenerator:
    """Generate natural language explanations."""

    def __init__(self, llm_function: Optional[Callable] = None):
        self.llm_function = llm_function
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[ExplanationAudience, Dict[str, str]]:
        """Load explanation templates for different audiences."""
        return {
            ExplanationAudience.END_USER: {
                "feature_importance": "The decision was primarily based on {top_features}. "
                                      "These factors had the most influence on the outcome.",
                "counterfactual": "If {changed_features} had been different, "
                                  "the decision would have changed to {alternative}.",
                "confidence": "The system is {confidence_level} confident in this decision.",
            },
            ExplanationAudience.EXPERT: {
                "feature_importance": "Feature importance analysis (Shapley values): {feature_scores}. "
                                      "Top contributing features: {top_features}.",
                "counterfactual": "Counterfactual analysis: minimal changes {changes} "
                                  "would flip decision to {alternative} (distance: {distance:.3f}).",
                "confidence": "Confidence: {confidence:.3f} based on {confidence_metric}.",
            },
        }

    async def generate_explanation(
        self,
        decision_trace: DecisionTrace,
        explanation_type: ExplanationType,
        audience: ExplanationAudience = ExplanationAudience.END_USER,
        additional_context: Optional[Dict] = None,
    ) -> Explanation:
        """Generate explanation for a decision.

        Args:
            decision_trace: Complete decision trace
            explanation_type: Type of explanation to generate
            audience: Target audience
            additional_context: Additional context for explanation

        Returns:
            Generated explanation
        """
        if explanation_type == ExplanationType.FEATURE_IMPORTANCE:
            return await self._explain_feature_importance(decision_trace, audience)
        elif explanation_type == ExplanationType.COUNTERFACTUAL:
            return await self._explain_counterfactual(decision_trace, audience)
        elif explanation_type == ExplanationType.CAUSAL:
            return await self._explain_causal(decision_trace, audience)
        else:
            return Explanation(
                explanation_type=explanation_type,
                content=f"Explanation type {explanation_type.value} not yet implemented.",
                evidence={},
                confidence=0.5,
                audience=audience,
            )

    async def _explain_feature_importance(
        self,
        decision_trace: DecisionTrace,
        audience: ExplanationAudience,
    ) -> Explanation:
        """Generate feature importance explanation."""
        if not decision_trace.feature_importance:
            return Explanation(
                explanation_type=ExplanationType.FEATURE_IMPORTANCE,
                content="Feature importance not available for this decision.",
                evidence={},
                confidence=0.0,
                audience=audience,
            )

        # Get top features
        sorted_features = sorted(
            decision_trace.feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        top_features = sorted_features[:3]

        # Format based on audience
        if audience == ExplanationAudience.END_USER:
            feature_str = ", ".join([f"{name}" for name, _ in top_features])
            content = self.templates[audience]["feature_importance"].format(
                top_features=feature_str
            )
        else:
            feature_str = ", ".join([f"{name} ({score:.3f})" for name, score in sorted_features])
            top_str = ", ".join([f"{name} ({score:.3f})" for name, score in top_features])
            content = self.templates[audience]["feature_importance"].format(
                feature_scores=feature_str,
                top_features=top_str
            )

        return Explanation(
            explanation_type=ExplanationType.FEATURE_IMPORTANCE,
            content=content,
            evidence={"feature_importance": dict(sorted_features)},
            confidence=0.9,
            audience=audience,
        )

    async def _explain_counterfactual(
        self,
        decision_trace: DecisionTrace,
        audience: ExplanationAudience,
    ) -> Explanation:
        """Generate counterfactual explanation."""
        # This would require generating actual counterfactuals
        # For now, return placeholder
        return Explanation(
            explanation_type=ExplanationType.COUNTERFACTUAL,
            content="Counterfactual explanation requires generating alternative scenarios.",
            evidence={},
            confidence=0.5,
            audience=audience,
        )

    async def _explain_causal(
        self,
        decision_trace: DecisionTrace,
        audience: ExplanationAudience,
    ) -> Explanation:
        """Generate causal explanation."""
        # Extract causal chain from reasoning steps
        causal_chain = []

        for step in decision_trace.reasoning_steps:
            if step.get("type") in ["inference", "action", "observation"]:
                causal_chain.append(step.get("description", ""))

        if causal_chain:
            content = "Causal pathway: " + " → ".join(causal_chain)
        else:
            content = "Causal pathway not available."

        return Explanation(
            explanation_type=ExplanationType.CAUSAL,
            content=content,
            evidence={"causal_chain": causal_chain},
            confidence=0.7,
            audience=audience,
        )


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED EXPLAINABILITY SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

class ExplainabilitySystem:
    """Unified explainability system for agent decisions."""

    def __init__(self, llm_function: Optional[Callable] = None):
        self.feature_analyzer = FeatureImportanceAnalyzer()
        self.counterfactual_gen = None  # Created on demand with feature ranges
        self.attention_analyzer = AttentionAnalyzer()
        self.explanation_gen = ExplanationGenerator(llm_function)

        # Store decision traces
        self.traces: Dict[str, DecisionTrace] = {}

    def record_decision(self, trace: DecisionTrace):
        """Record a decision trace."""
        self.traces[trace.decision_id] = trace

    async def explain(
        self,
        decision_id: str,
        explanation_types: List[ExplanationType],
        audience: ExplanationAudience = ExplanationAudience.END_USER,
    ) -> List[Explanation]:
        """Generate explanations for a decision.

        Args:
            decision_id: ID of decision to explain
            explanation_types: Types of explanations to generate
            audience: Target audience

        Returns:
            List of explanations
        """
        trace = self.traces.get(decision_id)
        if not trace:
            logger.warning(f"Decision trace {decision_id} not found")
            return []

        explanations = []

        for exp_type in explanation_types:
            explanation = await self.explanation_gen.generate_explanation(
                trace, exp_type, audience
            )
            explanations.append(explanation)

        return explanations

    def analyze_feature_importance(
        self,
        model_fn: Callable,
        instance: Dict[str, Any],
        method: str = "shapley",
    ) -> Dict[str, float]:
        """Analyze feature importance."""
        if method == "shapley":
            return self.feature_analyzer.shapley_values(model_fn, instance)
        else:
            raise ValueError(f"Unknown method: {method}")

    def visualize_attention(
        self,
        decision_id: str,
        save_path: Optional[str] = None,
    ):
        """Visualize attention for a decision."""
        trace = self.traces.get(decision_id)
        if not trace or not trace.attention_weights:
            logger.warning(f"Attention weights not available for {decision_id}")
            return None

        # Assume first attention weights
        attention = list(trace.attention_weights.values())[0]

        # Try to extract tokens from inputs
        input_text = trace.inputs.get("text", "")
        tokens = input_text.split() if input_text else ["token"] * attention.shape[-1]

        return self.attention_analyzer.visualize_attention(
            attention, tokens, save_path=save_path
        )


# Global instance
_explainability_system: Optional[ExplainabilitySystem] = None


def get_explainability_system() -> ExplainabilitySystem:
    """Get global explainability system."""
    global _explainability_system
    if _explainability_system is None:
        _explainability_system = ExplainabilitySystem()
    return _explainability_system


def explain_decision(
    decision_id: str,
    explanation_types: Optional[List[ExplanationType]] = None,
    audience: ExplanationAudience = ExplanationAudience.END_USER,
) -> List[Explanation]:
    """Convenience function to explain a decision."""
    if explanation_types is None:
        explanation_types = [ExplanationType.FEATURE_IMPORTANCE, ExplanationType.CAUSAL]

    system = get_explainability_system()
    # Note: This is async, so in practice you'd need to await this
    import asyncio
    return asyncio.run(system.explain(decision_id, explanation_types, audience))
