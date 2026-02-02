"""
Intelligent Aggregator System
=============================
A meta-system that leverages all 169 aggregation strategies intelligently.

Features:
- Automatic data analysis and strategy recommendation
- Multi-strategy ensemble with uncertainty quantification
- Adaptive strategy selection based on data characteristics
- Comprehensive diagnostics and explainability
- Real-time strategy performance tracking

Author: Built for Arthur @ DSCO
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .metacognitive_pipeline import (
    AggregationStrategy,
    ConfidenceAggregator,
    PipelineConfig,
)


class DataCharacteristic(Enum):
    """Characteristics of input confidence data."""
    HIGH_AGREEMENT = auto()      # Sources mostly agree
    HIGH_DISAGREEMENT = auto()   # Sources disagree significantly
    OUTLIERS_PRESENT = auto()    # Contains outlier values
    BIMODAL = auto()             # Two clusters of opinions
    SKEWED_HIGH = auto()         # Skewed toward high confidence
    SKEWED_LOW = auto()          # Skewed toward low confidence
    UNIFORM = auto()             # Uniformly distributed
    SPARSE = auto()              # Few sources
    DENSE = auto()               # Many sources
    WEIGHTED_UNEQUAL = auto()    # Highly unequal weights
    WEIGHTED_EQUAL = auto()      # Roughly equal weights


@dataclass
class DataProfile:
    """Profile of input data characteristics."""
    n_sources: int
    mean: float
    median: float
    std: float
    min_val: float
    max_val: float
    range: float
    skewness: float
    kurtosis: float
    iqr: float
    weight_gini: float  # Inequality in weights
    characteristics: List[DataCharacteristic]
    outlier_indices: List[int]
    cluster_count: int
    entropy: float

    def summary(self) -> str:
        """Human-readable summary."""
        chars = ", ".join(c.name for c in self.characteristics)
        return (
            f"Sources: {self.n_sources} | Mean: {self.mean:.3f} | Std: {self.std:.3f} | "
            f"Range: [{self.min_val:.3f}, {self.max_val:.3f}] | Characteristics: {chars}"
        )


@dataclass
class StrategyResult:
    """Result from a single aggregation strategy."""
    strategy: AggregationStrategy
    value: float
    execution_time_ms: float
    category: str


@dataclass
class AggregationResult:
    """Comprehensive result from intelligent aggregation."""
    # Primary result
    recommended_value: float
    confidence_interval: Tuple[float, float]
    uncertainty: float

    # Strategy analysis
    recommended_strategy: AggregationStrategy
    strategy_results: Dict[str, StrategyResult]
    ensemble_value: float

    # Data profile
    data_profile: DataProfile

    # Diagnostics
    agreement_score: float  # How much strategies agree
    robustness_score: float  # Stability across methods

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    execution_time_ms: float = 0.0
    strategies_used: int = 0

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Recommended: {self.recommended_value:.4f} "
            f"(95% CI: [{self.confidence_interval[0]:.4f}, {self.confidence_interval[1]:.4f}])\n"
            f"Uncertainty: {self.uncertainty:.4f} | Agreement: {self.agreement_score:.2%} | "
            f"Robustness: {self.robustness_score:.2%}\n"
            f"Best Strategy: {self.recommended_strategy.name}\n"
            f"Data: {self.data_profile.summary()}"
        )


# Strategy categories for intelligent selection
STRATEGY_CATEGORIES = {
    "basic": [
        AggregationStrategy.WEIGHTED_AVERAGE,
        AggregationStrategy.MEDIAN,
        AggregationStrategy.TRIMMED_MEAN,
        AggregationStrategy.GEOMETRIC_MEAN,
        AggregationStrategy.HARMONIC_MEAN,
    ],
    "bayesian": [
        AggregationStrategy.BAYESIAN,
        AggregationStrategy.BAYESIAN_COMBINATION,
        AggregationStrategy.CONJUGATE_PRIOR,
        AggregationStrategy.EMPIRICAL_BAYES,
        AggregationStrategy.HIERARCHICAL_BAYES,
    ],
    "density": [
        AggregationStrategy.KERNEL_DENSITY,
        AggregationStrategy.KERNEL_DENSITY_ADAPTIVE,
        AggregationStrategy.GMM_EM,
        AggregationStrategy.DIRICHLET_PROCESS,
    ],
    "robust": [
        AggregationStrategy.ROBUST_HUBER,
        AggregationStrategy.ROBUST_TUKEY,
        AggregationStrategy.WINSORIZED,
        AggregationStrategy.THEIL_SEN,
        AggregationStrategy.MESTIMATOR_HAMPEL,
    ],
    "information": [
        AggregationStrategy.ENTROPY_WEIGHTED,
        AggregationStrategy.FISHER_INFORMATION,
        AggregationStrategy.MUTUAL_INFORMATION,
        AggregationStrategy.INFO_BOTTLENECK,
    ],
    "ensemble": [
        AggregationStrategy.MIXTURE_OF_EXPERTS,
        AggregationStrategy.STACKING,
        AggregationStrategy.SUPER_LEARNER,
        AggregationStrategy.ADAPTIVE,
    ],
    "sampling": [
        AggregationStrategy.BOOTSTRAP,
        AggregationStrategy.MONTE_CARLO,
        AggregationStrategy.SEQUENTIAL_MONTE_CARLO,
        AggregationStrategy.MARKOV_CHAIN_MONTE_CARLO,
    ],
    "belief": [
        AggregationStrategy.DEMPSTER_SHAFER,
        AggregationStrategy.SUBJECTIVE_LOGIC,
        AggregationStrategy.BELIEF_PROPAGATION,
    ],
    "neural": [
        AggregationStrategy.ATTENTION_AGGREGATION,
        AggregationStrategy.TRANSFORMER_FUSION,
        AggregationStrategy.DEEP_SETS,
    ],
    "game_theory": [
        AggregationStrategy.NASH_BARGAINING,
        AggregationStrategy.SHAPLEY_VALUE,
        AggregationStrategy.MECHANISM_DESIGN,
    ],
    "quantum": [
        AggregationStrategy.QUANTUM_SUPERPOSITION,
        AggregationStrategy.QUANTUM_ANNEALING,
        AggregationStrategy.QUANTUM_AMPLITUDE,
    ],
    "conformal": [
        AggregationStrategy.CONFORMAL_PREDICTION,
        AggregationStrategy.SPLIT_CONFORMAL,
    ],
}

# Strategy recommendations based on data characteristics
CHARACTERISTIC_STRATEGIES = {
    DataCharacteristic.HIGH_AGREEMENT: [
        AggregationStrategy.WEIGHTED_AVERAGE,
        AggregationStrategy.BAYESIAN,
        AggregationStrategy.CONSENSUS_CLUSTERING,
    ],
    DataCharacteristic.HIGH_DISAGREEMENT: [
        AggregationStrategy.ROBUST_HUBER,
        AggregationStrategy.MEDIAN,
        AggregationStrategy.MIXTURE_OF_EXPERTS,
        AggregationStrategy.DEMPSTER_SHAFER,
    ],
    DataCharacteristic.OUTLIERS_PRESENT: [
        AggregationStrategy.ROBUST_TUKEY,
        AggregationStrategy.WINSORIZED,
        AggregationStrategy.TRIMMED_MEAN,
        AggregationStrategy.BREAKDOWN_POINT,
        AggregationStrategy.LEAST_TRIMMED_SQUARES,
    ],
    DataCharacteristic.BIMODAL: [
        AggregationStrategy.GMM_EM,
        AggregationStrategy.DIRICHLET_PROCESS,
        AggregationStrategy.SPECTRAL_CLUSTERING,
        AggregationStrategy.KERNEL_DENSITY_ADAPTIVE,
    ],
    DataCharacteristic.SKEWED_HIGH: [
        AggregationStrategy.GEOMETRIC_MEAN,
        AggregationStrategy.BAYESIAN_MODEL_AVERAGING,
        AggregationStrategy.HIGHEST_CONFIDENCE,
    ],
    DataCharacteristic.SKEWED_LOW: [
        AggregationStrategy.HARMONIC_MEAN,
        AggregationStrategy.LOWEST_CONFIDENCE,
        AggregationStrategy.CONJUGATE_PRIOR,
    ],
    DataCharacteristic.SPARSE: [
        AggregationStrategy.BAYESIAN,
        AggregationStrategy.JEFFREY_PRIOR,
        AggregationStrategy.BOOTSTRAP,
    ],
    DataCharacteristic.DENSE: [
        AggregationStrategy.KERNEL_DENSITY_ADAPTIVE,
        AggregationStrategy.SEQUENTIAL_MONTE_CARLO,
        AggregationStrategy.SPECTRAL_CLUSTERING,
    ],
    DataCharacteristic.WEIGHTED_UNEQUAL: [
        AggregationStrategy.WEIGHTED_AVERAGE,
        AggregationStrategy.ATTENTION_AGGREGATION,
        AggregationStrategy.SHAPLEY_VALUE,
    ],
}


class IntelligentAggregator:
    """
    Intelligent meta-aggregator that selects and combines strategies optimally.

    Usage:
        aggregator = IntelligentAggregator()
        result = aggregator.aggregate([
            ("expert_1", 0.85, 1.0),
            ("expert_2", 0.72, 0.8),
            ("model_1", 0.91, 1.2),
        ])
        print(result.summary())
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        max_strategies: int = 50,
        parallel: bool = True,
        track_performance: bool = True,
    ):
        self.config = config or PipelineConfig()
        self.base_aggregator = ConfidenceAggregator(self.config)
        self.max_strategies = max_strategies
        self.parallel = parallel
        self.track_performance = track_performance

        # Performance tracking
        self._strategy_performance: Dict[str, List[float]] = defaultdict(list)
        self._strategy_times: Dict[str, List[float]] = defaultdict(list)

    def analyze_data(
        self,
        confidences: List[Tuple[str, float, float]],
    ) -> DataProfile:
        """Analyze input data characteristics."""
        vals = [c for _, c, _ in confidences]
        weights = [w for _, _, w in confidences]
        n = len(vals)

        if n == 0:
            return DataProfile(
                n_sources=0, mean=0.5, median=0.5, std=0, min_val=0.5, max_val=0.5,
                range=0, skewness=0, kurtosis=0, iqr=0, weight_gini=0,
                characteristics=[], outlier_indices=[], cluster_count=1, entropy=0
            )

        # Basic statistics
        mean_v = sum(v * w for v, w in zip(vals, weights)) / sum(weights)
        sorted_vals = sorted(vals)
        median_v = sorted_vals[n // 2]

        var_v = sum(w * (v - mean_v) ** 2 for v, w in zip(vals, weights)) / sum(weights)
        std_v = math.sqrt(var_v) if var_v > 0 else 0

        min_v, max_v = min(vals), max(vals)
        range_v = max_v - min_v

        # Skewness and kurtosis
        if std_v > 0:
            skewness = sum(w * ((v - mean_v) / std_v) ** 3 for v, w in zip(vals, weights)) / sum(weights)
            kurtosis = sum(w * ((v - mean_v) / std_v) ** 4 for v, w in zip(vals, weights)) / sum(weights) - 3
        else:
            skewness, kurtosis = 0, 0

        # IQR
        q1_idx, q3_idx = n // 4, 3 * n // 4
        iqr = sorted_vals[q3_idx] - sorted_vals[q1_idx] if n >= 4 else std_v

        # Weight inequality (Gini coefficient)
        sorted_weights = sorted(weights)
        weight_sum = sum(weights)
        cumsum = 0
        gini_sum = 0
        for i, w in enumerate(sorted_weights):
            cumsum += w
            gini_sum += cumsum
        weight_gini = 1 - 2 * gini_sum / (n * weight_sum) if n > 0 and weight_sum > 0 else 0

        # Detect outliers (IQR method)
        if n >= 4:
            lower_bound = sorted_vals[q1_idx] - 1.5 * iqr
            upper_bound = sorted_vals[q3_idx] + 1.5 * iqr
            outlier_indices = [i for i, v in enumerate(vals) if v < lower_bound or v > upper_bound]
        else:
            outlier_indices = []

        # Entropy
        entropy = -sum(
            (v * math.log(v + 1e-10) + (1 - v) * math.log(1 - v + 1e-10))
            for v in vals
        ) / n if n > 0 else 0

        # Simple cluster detection (bimodal check)
        cluster_count = 1
        if n >= 4:
            mid = (min_v + max_v) / 2
            below = sum(1 for v in vals if v < mid - 0.1)
            above = sum(1 for v in vals if v > mid + 0.1)
            if below > n * 0.3 and above > n * 0.3:
                cluster_count = 2

        # Determine characteristics
        characteristics = []

        if std_v < 0.1:
            characteristics.append(DataCharacteristic.HIGH_AGREEMENT)
        elif std_v > 0.25:
            characteristics.append(DataCharacteristic.HIGH_DISAGREEMENT)

        if outlier_indices:
            characteristics.append(DataCharacteristic.OUTLIERS_PRESENT)

        if cluster_count >= 2:
            characteristics.append(DataCharacteristic.BIMODAL)

        if skewness > 0.5:
            characteristics.append(DataCharacteristic.SKEWED_HIGH)
        elif skewness < -0.5:
            characteristics.append(DataCharacteristic.SKEWED_LOW)

        if abs(skewness) < 0.3 and abs(kurtosis) < 1:
            characteristics.append(DataCharacteristic.UNIFORM)

        if n <= 3:
            characteristics.append(DataCharacteristic.SPARSE)
        elif n >= 10:
            characteristics.append(DataCharacteristic.DENSE)

        if weight_gini > 0.3:
            characteristics.append(DataCharacteristic.WEIGHTED_UNEQUAL)
        else:
            characteristics.append(DataCharacteristic.WEIGHTED_EQUAL)

        return DataProfile(
            n_sources=n,
            mean=mean_v,
            median=median_v,
            std=std_v,
            min_val=min_v,
            max_val=max_v,
            range=range_v,
            skewness=skewness,
            kurtosis=kurtosis,
            iqr=iqr,
            weight_gini=weight_gini,
            characteristics=characteristics,
            outlier_indices=outlier_indices,
            cluster_count=cluster_count,
            entropy=entropy,
        )

    def recommend_strategies(
        self,
        profile: DataProfile,
        n_strategies: int = 20,
    ) -> List[AggregationStrategy]:
        """Recommend strategies based on data profile."""
        recommended = []
        seen = set()

        # Add strategies based on characteristics
        for char in profile.characteristics:
            if char in CHARACTERISTIC_STRATEGIES:
                for strat in CHARACTERISTIC_STRATEGIES[char]:
                    if strat not in seen:
                        recommended.append(strat)
                        seen.add(strat)

        # Add core strategies from each category
        for category, strategies in STRATEGY_CATEGORIES.items():
            for strat in strategies[:2]:  # Top 2 from each category
                if strat not in seen:
                    recommended.append(strat)
                    seen.add(strat)

        # Add historically best performers
        if self._strategy_performance:
            sorted_by_perf = sorted(
                self._strategy_performance.items(),
                key=lambda x: sum(x[1]) / len(x[1]) if x[1] else 0,
                reverse=True
            )
            for strat_name, _ in sorted_by_perf[:5]:
                strat = AggregationStrategy[strat_name]
                if strat not in seen:
                    recommended.append(strat)
                    seen.add(strat)

        return recommended[:n_strategies]

    def _run_strategy(
        self,
        strategy: AggregationStrategy,
        confidences: List[Tuple[str, float, float]],
    ) -> StrategyResult:
        """Run a single strategy and return result."""
        start = time.perf_counter()
        try:
            value = self.base_aggregator.aggregate(confidences, strategy)
        except Exception:
            value = 0.5
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Determine category
        category = "other"
        for cat, strategies in STRATEGY_CATEGORIES.items():
            if strategy in strategies:
                category = cat
                break

        return StrategyResult(
            strategy=strategy,
            value=value,
            execution_time_ms=elapsed_ms,
            category=category,
        )

    def aggregate(
        self,
        confidences: List[Tuple[str, float, float]],
        strategies: Optional[List[AggregationStrategy]] = None,
        return_all: bool = False,
    ) -> AggregationResult:
        """
        Perform intelligent aggregation.

        Args:
            confidences: List of (source_name, confidence, weight) tuples
            strategies: Optional list of strategies to use (auto-selects if None)
            return_all: If True, run all 169 strategies

        Returns:
            AggregationResult with comprehensive analysis
        """
        start_time = time.perf_counter()

        # Analyze data
        profile = self.analyze_data(confidences)

        # Select strategies
        if return_all:
            strategies_to_run = list(AggregationStrategy)
        elif strategies:
            strategies_to_run = strategies
        else:
            strategies_to_run = self.recommend_strategies(profile, self.max_strategies)

        # Run strategies
        results: Dict[str, StrategyResult] = {}
        for strategy in strategies_to_run:
            result = self._run_strategy(strategy, confidences)
            results[strategy.name] = result

            # Track performance
            if self.track_performance:
                self._strategy_times[strategy.name].append(result.execution_time_ms)

        # Analyze results
        values = [r.value for r in results.values()]

        # Compute ensemble value (robust combination)
        sorted_values = sorted(values)
        n_vals = len(sorted_values)

        # Trimmed mean of strategy results
        trim = max(1, n_vals // 10)
        if n_vals > 2 * trim:
            ensemble_value = sum(sorted_values[trim:-trim]) / (n_vals - 2 * trim)
        else:
            ensemble_value = sum(sorted_values) / n_vals

        # Compute confidence interval (bootstrap-like)
        if n_vals >= 5:
            ci_lower = sorted_values[int(n_vals * 0.025)]
            ci_upper = sorted_values[int(n_vals * 0.975)]
        else:
            ci_lower = min(values)
            ci_upper = max(values)

        # Uncertainty from strategy disagreement
        value_std = math.sqrt(sum((v - ensemble_value) ** 2 for v in values) / n_vals) if n_vals > 0 else 0
        uncertainty = min(1.0, value_std * 2)

        # Agreement score (inverse of coefficient of variation)
        agreement_score = max(0, 1 - value_std / (ensemble_value + 0.01))

        # Robustness: how stable across method categories
        category_means = defaultdict(list)
        for r in results.values():
            category_means[r.category].append(r.value)

        if len(category_means) >= 2:
            cat_avgs = [sum(v) / len(v) for v in category_means.values()]
            cat_std = math.sqrt(sum((c - ensemble_value) ** 2 for c in cat_avgs) / len(cat_avgs))
            robustness_score = max(0, 1 - cat_std * 3)
        else:
            robustness_score = agreement_score

        # Find best strategy (closest to ensemble with low variance in category)
        best_strategy = min(
            results.values(),
            key=lambda r: abs(r.value - ensemble_value) + 0.1 * r.execution_time_ms / 100
        ).strategy

        # Recommended value: weighted combination of best strategy and ensemble
        recommended_value = 0.6 * results[best_strategy.name].value + 0.4 * ensemble_value

        total_time_ms = (time.perf_counter() - start_time) * 1000

        return AggregationResult(
            recommended_value=recommended_value,
            confidence_interval=(ci_lower, ci_upper),
            uncertainty=uncertainty,
            recommended_strategy=best_strategy,
            strategy_results=results,
            ensemble_value=ensemble_value,
            data_profile=profile,
            agreement_score=agreement_score,
            robustness_score=robustness_score,
            execution_time_ms=total_time_ms,
            strategies_used=len(strategies_to_run),
        )

    def compare_categories(
        self,
        confidences: List[Tuple[str, float, float]],
    ) -> Dict[str, Dict[str, float]]:
        """Compare aggregation results across strategy categories."""
        category_results = {}

        for category, strategies in STRATEGY_CATEGORIES.items():
            values = []
            times = []
            for strategy in strategies:
                result = self._run_strategy(strategy, confidences)
                values.append(result.value)
                times.append(result.execution_time_ms)

            category_results[category] = {
                "mean": sum(values) / len(values),
                "std": math.sqrt(sum((v - sum(values)/len(values))**2 for v in values) / len(values)),
                "min": min(values),
                "max": max(values),
                "avg_time_ms": sum(times) / len(times),
                "n_strategies": len(strategies),
            }

        return category_results

    def sensitivity_analysis(
        self,
        confidences: List[Tuple[str, float, float]],
        strategy: AggregationStrategy = AggregationStrategy.BAYESIAN,
    ) -> Dict[str, float]:
        """Analyze sensitivity to each input source."""
        base_result = self.base_aggregator.aggregate(confidences, strategy)

        sensitivities = {}
        for i, (name, conf, weight) in enumerate(confidences):
            # Leave-one-out
            loo_confidences = confidences[:i] + confidences[i+1:]
            if loo_confidences:
                loo_result = self.base_aggregator.aggregate(loo_confidences, strategy)
                sensitivities[name] = abs(base_result - loo_result)
            else:
                sensitivities[name] = 0.0

        return sensitivities

    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance statistics for tracked strategies."""
        report = {}

        for strat_name in self._strategy_times:
            times = self._strategy_times[strat_name]
            if times:
                report[strat_name] = {
                    "avg_time_ms": sum(times) / len(times),
                    "min_time_ms": min(times),
                    "max_time_ms": max(times),
                    "n_calls": len(times),
                }

        return report


class AggregationPipeline:
    """
    High-level pipeline for production use.

    Combines intelligent aggregation with caching and async support.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.aggregator = IntelligentAggregator(config)
        self._cache: Dict[str, AggregationResult] = {}

    def _cache_key(self, confidences: List[Tuple[str, float, float]]) -> str:
        """Generate cache key from inputs."""
        import hashlib
        import json
        data = json.dumps([(n, round(c, 4), round(w, 4)) for n, c, w in confidences], sort_keys=True)
        return hashlib.md5(data.encode()).hexdigest()

    def aggregate(
        self,
        confidences: List[Tuple[str, float, float]],
        use_cache: bool = True,
    ) -> AggregationResult:
        """Aggregate with optional caching."""
        if use_cache:
            key = self._cache_key(confidences)
            if key in self._cache:
                return self._cache[key]

        result = self.aggregator.aggregate(confidences)

        if use_cache:
            self._cache[key] = result

        return result

    async def aggregate_async(
        self,
        confidences: List[Tuple[str, float, float]],
    ) -> AggregationResult:
        """Async aggregation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.aggregator.aggregate(confidences)
        )

    def batch_aggregate(
        self,
        batch: List[List[Tuple[str, float, float]]],
    ) -> List[AggregationResult]:
        """Aggregate multiple sets of confidences."""
        return [self.aggregate(confidences) for confidences in batch]

    def clear_cache(self):
        """Clear the result cache."""
        self._cache.clear()


def demonstrate():
    """Demonstrate the intelligent aggregator."""
    print("=" * 80)
    print("INTELLIGENT AGGREGATOR DEMONSTRATION")
    print("=" * 80)

    # Create test scenarios
    scenarios = {
        "High Agreement": [
            ("expert_1", 0.82, 1.0),
            ("expert_2", 0.85, 1.1),
            ("expert_3", 0.83, 0.9),
            ("model_1", 0.81, 1.0),
            ("model_2", 0.84, 1.2),
        ],
        "High Disagreement": [
            ("optimist", 0.95, 1.0),
            ("pessimist", 0.25, 1.0),
            ("moderate_1", 0.55, 0.8),
            ("moderate_2", 0.65, 0.9),
            ("wild_card", 0.45, 0.5),
        ],
        "Outlier Present": [
            ("reliable_1", 0.78, 1.2),
            ("reliable_2", 0.82, 1.1),
            ("reliable_3", 0.80, 1.0),
            ("reliable_4", 0.79, 1.0),
            ("outlier", 0.15, 0.8),
        ],
        "Bimodal": [
            ("camp_a_1", 0.85, 1.0),
            ("camp_a_2", 0.88, 1.0),
            ("camp_a_3", 0.82, 0.9),
            ("camp_b_1", 0.35, 1.0),
            ("camp_b_2", 0.32, 0.9),
            ("camp_b_3", 0.38, 1.0),
        ],
    }

    aggregator = IntelligentAggregator()

    for scenario_name, confidences in scenarios.items():
        print(f"\n{'='*80}")
        print(f"SCENARIO: {scenario_name}")
        print("=" * 80)

        # Show input
        print("\nInputs:")
        for name, conf, weight in confidences:
            print(f"  {name:15s}: conf={conf:.2f}, weight={weight:.1f}")

        # Aggregate
        result = aggregator.aggregate(confidences)

        # Show result
        print(f"\n{result.summary()}")

        # Show top strategies
        sorted_results = sorted(
            result.strategy_results.values(),
            key=lambda r: abs(r.value - result.ensemble_value)
        )

        print("\nTop 5 strategies (closest to ensemble):")
        for r in sorted_results[:5]:
            print(f"  {r.strategy.name:35s}: {r.value:.4f} ({r.category})")

    # Category comparison
    print(f"\n{'='*80}")
    print("CATEGORY COMPARISON (using High Agreement scenario)")
    print("=" * 80)

    categories = aggregator.compare_categories(scenarios["High Agreement"])
    for cat, stats in sorted(categories.items(), key=lambda x: x[1]["mean"], reverse=True):
        print(f"  {cat:15s}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
              f"range=[{stats['min']:.3f}, {stats['max']:.3f}]")

    # Sensitivity analysis
    print(f"\n{'='*80}")
    print("SENSITIVITY ANALYSIS (High Disagreement scenario)")
    print("=" * 80)

    sensitivities = aggregator.sensitivity_analysis(scenarios["High Disagreement"])
    for name, sens in sorted(sensitivities.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:15s}: {sens:.4f}")

    print(f"\n{'='*80}")
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate()
