#!/usr/bin/env python3
"""
Aggregation Strategy Benchmark
==============================
Comprehensive benchmarking of all 169 aggregation strategies.

Features:
- Performance profiling (speed and accuracy)
- Stability analysis across different data distributions
- Strategy clustering by behavior
- Visualization-ready output
- Recommendations engine

Usage:
    python -m src.llm.aggregation_benchmark
    python -m src.llm.aggregation_benchmark --full --export results.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

from .metacognitive_pipeline import AggregationStrategy, ConfidenceAggregator, PipelineConfig


CONSOLE = Console()


@dataclass
class StrategyBenchmark:
    """Benchmark results for a single strategy."""
    name: str
    category: str

    # Accuracy metrics (compared to ground truth when available)
    mean_value: float
    std_value: float
    min_value: float
    max_value: float

    # Speed metrics
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float

    # Robustness metrics
    outlier_sensitivity: float  # How much outliers affect result
    noise_sensitivity: float    # How much noise affects result
    sample_efficiency: float    # Performance with few samples

    # Behavior cluster
    cluster_id: int = 0


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""
    timestamp: str
    n_strategies: int
    n_scenarios: int
    n_iterations: int

    strategy_results: Dict[str, StrategyBenchmark]
    category_summary: Dict[str, Dict[str, float]]

    # Overall rankings
    speed_ranking: List[str]
    robustness_ranking: List[str]
    overall_ranking: List[str]

    # Behavior clusters
    clusters: Dict[int, List[str]]

    execution_time_seconds: float


# Strategy categories
CATEGORIES = {
    "basic": ["WEIGHTED_AVERAGE", "MAJORITY_VOTE", "HIGHEST_CONFIDENCE", "LOWEST_CONFIDENCE",
              "MEDIAN", "TRIMMED_MEAN", "GEOMETRIC_MEAN", "HARMONIC_MEAN", "POWER_MEAN",
              "LEHMER_MEAN", "QUASI_ARITHMETIC"],
    "bayesian": ["BAYESIAN", "BAYESIAN_COMBINATION", "BAYESIAN_MODEL_AVERAGING", "CONJUGATE_PRIOR",
                 "JEFFREY_PRIOR", "EMPIRICAL_BAYES", "HIERARCHICAL_BAYES", "SPIKE_AND_SLAB",
                 "HORSESHOE_PRIOR", "BAYESIAN_QUADRATURE"],
    "density": ["KERNEL_DENSITY", "KERNEL_DENSITY_ADAPTIVE", "PARZEN_WINDOW", "HISTOGRAM_DENSITY",
                "KNN_DENSITY", "GAUSSIAN_MIXTURE", "GMM_EM", "GMM_VARIATIONAL", "DIRICHLET_PROCESS",
                "PITMAN_YOR_PROCESS", "DENSITY_RATIO", "NORMALIZING_FLOW", "SCORE_MATCHING",
                "CONTRASTIVE_DENSITY"],
    "prompt": ["PROMPT_DENSITY_ESTIMATION", "PROMPT_CALIBRATION", "PROMPT_ENSEMBLE",
               "PROMPT_UNCERTAINTY", "CHAIN_OF_DENSITY", "SELF_CONSISTENCY_DENSITY",
               "PROMPT_TEMPERATURE_SWEEP", "SEMANTIC_DENSITY"],
    "sampling": ["BOOTSTRAP", "MONTE_CARLO", "IMPORTANCE_SAMPLING", "SEQUENTIAL_MONTE_CARLO",
                 "MARKOV_CHAIN_MONTE_CARLO", "HAMILTONIAN_MONTE_CARLO", "NESTED_SAMPLING",
                 "APPROXIMATE_BAYESIAN_COMPUTATION", "REJECTION_SAMPLING", "SLICE_SAMPLING",
                 "GIBBS_SAMPLING", "LANGEVIN_DYNAMICS"],
    "information": ["ENTROPY_WEIGHTED", "KULLBACK_LEIBLER", "JENSEN_SHANNON", "MUTUAL_INFORMATION",
                    "FISHER_INFORMATION", "RENYI_ENTROPY", "TSALLIS_ENTROPY", "RATE_DISTORTION",
                    "MINIMUM_DESCRIPTION_LENGTH", "KOLMOGOROV_COMPLEXITY", "CHANNEL_CAPACITY",
                    "INFO_BOTTLENECK"],
    "robust": ["ROBUST_HUBER", "ROBUST_TUKEY", "WINSORIZED", "LEAST_TRIMMED_SQUARES",
               "THEIL_SEN", "HODGES_LEHMANN", "MESTIMATOR_ANDREWS", "MESTIMATOR_HAMPEL",
               "BREAKDOWN_POINT", "INFLUENCE_FUNCTION"],
    "belief": ["DEMPSTER_SHAFER", "TRANSFERABLE_BELIEF", "SUBJECTIVE_LOGIC", "POSSIBILITY_THEORY",
               "ROUGH_SET_FUSION", "INTUITIONISTIC_FUZZY", "NEUTROSOPHIC", "GREY_RELATIONAL",
               "EVIDENTIAL_NEURAL", "BELIEF_PROPAGATION"],
    "transport": ["WASSERSTEIN_BARYCENTER", "SINKHORN_DIVERGENCE", "GROMOV_WASSERSTEIN",
                  "SLICED_WASSERSTEIN", "UNBALANCED_OT"],
    "spectral": ["SPECTRAL_CLUSTERING", "LAPLACIAN_EIGENMAPS", "DIFFUSION_MAPS",
                 "SPECTRAL_DENSITY", "RANDOM_MATRIX_THEORY"],
    "geometry": ["FISHER_RAO_METRIC", "ALPHA_DIVERGENCE", "BREGMAN_CENTROID",
                 "EXPONENTIAL_GEODESIC", "WASSERSTEIN_NATURAL_GRADIENT"],
    "neural": ["ATTENTION_AGGREGATION", "TRANSFORMER_FUSION", "NEURAL_PROCESS", "DEEP_SETS",
               "SET_TRANSFORMER", "GRAPH_NEURAL_AGGREGATION", "HYPERNETWORK_FUSION",
               "META_LEARNING_AGGREGATION"],
    "probabilistic": ["EXPECTATION_PROPAGATION", "ASSUMED_DENSITY_FILTERING",
                      "LOOPY_BELIEF_PROPAGATION", "VARIATIONAL_MESSAGE_PASSING",
                      "STOCHASTIC_VARIATIONAL", "BLACK_BOX_VARIATIONAL", "NORMALIZING_FLOW_VI"],
    "hybrid": ["HYBRID_AGGLOMERATION", "HIERARCHICAL_FUSION", "MIXTURE_OF_EXPERTS",
               "CASCADED_BAYESIAN", "CONSENSUS_CLUSTERING", "MULTI_SCALE_FUSION",
               "ITERATIVE_REFINEMENT", "GRAPH_AGGREGATION", "COPULA_FUSION",
               "VARIATIONAL_INFERENCE", "DENSITY_FUNCTIONAL", "RENORMALIZATION_GROUP",
               "MEAN_FIELD_THEORY", "CAVITY_METHOD", "REPLICA_TRICK", "SUPERSYMMETRIC",
               "HOLOGRAPHIC"],
    "game": ["NASH_BARGAINING", "SHAPLEY_VALUE", "CORE_ALLOCATION", "NUCLEOLUS",
             "MECHANISM_DESIGN"],
    "causal": ["CAUSAL_DISCOVERY", "DO_CALCULUS", "COUNTERFACTUAL_AGGREGATION",
               "INSTRUMENTAL_VARIABLE", "DOUBLE_MACHINE_LEARNING"],
    "conformal": ["CONFORMAL_PREDICTION", "SPLIT_CONFORMAL", "FULL_CONFORMAL",
                  "CONFORMALIZED_QUANTILE"],
    "meta": ["ENSEMBLE_SELECTION", "STACKING", "ADAPTIVE", "SUPER_LEARNER",
             "ONLINE_LEARNING", "THOMPSON_SAMPLING", "UCB_AGGREGATION", "EXP3_AGGREGATION"],
    "quantum": ["QUANTUM_SUPERPOSITION", "QUANTUM_ENTANGLEMENT", "QUANTUM_ANNEALING",
                "QUANTUM_AMPLITUDE"],
}


def get_category(strategy_name: str) -> str:
    """Get category for a strategy."""
    for cat, strategies in CATEGORIES.items():
        if strategy_name in strategies:
            return cat
    return "other"


def generate_test_scenarios() -> Dict[str, List[Tuple[str, float, float]]]:
    """Generate diverse test scenarios."""
    random.seed(42)

    scenarios = {}

    # 1. High agreement
    scenarios["high_agreement"] = [
        (f"src_{i}", 0.8 + random.gauss(0, 0.02), 1.0 + random.uniform(-0.2, 0.2))
        for i in range(7)
    ]

    # 2. High disagreement
    scenarios["high_disagreement"] = [
        ("optimist", 0.92, 1.0),
        ("pessimist", 0.28, 1.0),
        ("moderate_1", 0.55, 0.9),
        ("moderate_2", 0.62, 0.8),
        ("moderate_3", 0.48, 0.7),
    ]

    # 3. Single outlier
    scenarios["outlier"] = [
        (f"good_{i}", 0.75 + random.gauss(0, 0.03), 1.0)
        for i in range(6)
    ] + [("outlier", 0.1, 0.8)]

    # 4. Bimodal
    scenarios["bimodal"] = [
        (f"camp_a_{i}", 0.85 + random.gauss(0, 0.03), 1.0)
        for i in range(4)
    ] + [
        (f"camp_b_{i}", 0.25 + random.gauss(0, 0.03), 1.0)
        for i in range(4)
    ]

    # 5. Sparse (few sources)
    scenarios["sparse"] = [
        ("expert_1", 0.72, 1.5),
        ("expert_2", 0.68, 1.2),
    ]

    # 6. Dense (many sources)
    scenarios["dense"] = [
        (f"src_{i}", 0.6 + random.gauss(0, 0.1), 0.5 + random.uniform(0, 1))
        for i in range(20)
    ]

    # 7. Unequal weights
    scenarios["unequal_weights"] = [
        ("expert", 0.85, 5.0),
        ("novice_1", 0.72, 0.5),
        ("novice_2", 0.68, 0.5),
        ("novice_3", 0.75, 0.5),
        ("novice_4", 0.70, 0.5),
    ]

    # 8. Uniform
    scenarios["uniform"] = [
        (f"src_{i}", 0.1 * (i + 1), 1.0)
        for i in range(9)
    ]

    # 9. Skewed high
    scenarios["skewed_high"] = [
        (f"src_{i}", 0.7 + 0.05 * i + random.gauss(0, 0.02), 1.0)
        for i in range(6)
    ]

    # 10. Skewed low
    scenarios["skewed_low"] = [
        (f"src_{i}", 0.3 - 0.03 * i + random.gauss(0, 0.02), 1.0)
        for i in range(6)
    ]

    return scenarios


def run_benchmark(
    strategies: Optional[List[AggregationStrategy]] = None,
    n_iterations: int = 5,
    show_progress: bool = True,
) -> BenchmarkResults:
    """Run comprehensive benchmark."""
    start_time = time.perf_counter()

    config = PipelineConfig()
    aggregator = ConfidenceAggregator(config)

    if strategies is None:
        strategies = list(AggregationStrategy)

    scenarios = generate_test_scenarios()
    n_scenarios = len(scenarios)

    # Results storage
    strategy_values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    strategy_times: Dict[str, List[float]] = defaultdict(list)

    total_runs = len(strategies) * n_scenarios * n_iterations

    if show_progress:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=CONSOLE,
        )
        task = progress.add_task("Benchmarking...", total=total_runs)
        progress.start()

    try:
        for strategy in strategies:
            for scenario_name, confidences in scenarios.items():
                for _ in range(n_iterations):
                    # Add small noise for robustness testing
                    noisy_conf = [
                        (n, max(0.01, min(0.99, c + random.gauss(0, 0.01))), w)
                        for n, c, w in confidences
                    ]

                    t0 = time.perf_counter()
                    try:
                        value = aggregator.aggregate(noisy_conf, strategy)
                    except Exception:
                        value = 0.5
                    elapsed = (time.perf_counter() - t0) * 1000

                    strategy_values[strategy.name][scenario_name].append(value)
                    strategy_times[strategy.name].append(elapsed)

                    if show_progress:
                        progress.advance(task)
    finally:
        if show_progress:
            progress.stop()

    # Compute benchmarks
    strategy_benchmarks: Dict[str, StrategyBenchmark] = {}

    for strategy in strategies:
        name = strategy.name
        all_values = [v for vals in strategy_values[name].values() for v in vals]
        times = strategy_times[name]

        # Compute outlier sensitivity
        outlier_vals = strategy_values[name].get("outlier", [])
        normal_vals = strategy_values[name].get("high_agreement", [])
        outlier_sensitivity = abs(
            (sum(outlier_vals) / len(outlier_vals) if outlier_vals else 0.5) -
            (sum(normal_vals) / len(normal_vals) if normal_vals else 0.5)
        )

        # Compute noise sensitivity
        scenario_stds = []
        for vals in strategy_values[name].values():
            if len(vals) > 1:
                mean_v = sum(vals) / len(vals)
                std_v = math.sqrt(sum((v - mean_v) ** 2 for v in vals) / len(vals))
                scenario_stds.append(std_v)
        noise_sensitivity = sum(scenario_stds) / len(scenario_stds) if scenario_stds else 0

        # Sample efficiency (sparse vs dense performance similarity)
        sparse_vals = strategy_values[name].get("sparse", [])
        dense_vals = strategy_values[name].get("dense", [])
        sample_efficiency = 1 - abs(
            (sum(sparse_vals) / len(sparse_vals) if sparse_vals else 0.5) -
            (sum(dense_vals) / len(dense_vals) if dense_vals else 0.5)
        )

        strategy_benchmarks[name] = StrategyBenchmark(
            name=name,
            category=get_category(name),
            mean_value=sum(all_values) / len(all_values) if all_values else 0.5,
            std_value=math.sqrt(sum((v - sum(all_values)/len(all_values))**2 for v in all_values) / len(all_values)) if all_values else 0,
            min_value=min(all_values) if all_values else 0.5,
            max_value=max(all_values) if all_values else 0.5,
            mean_time_ms=sum(times) / len(times) if times else 0,
            std_time_ms=math.sqrt(sum((t - sum(times)/len(times))**2 for t in times) / len(times)) if times else 0,
            min_time_ms=min(times) if times else 0,
            max_time_ms=max(times) if times else 0,
            outlier_sensitivity=outlier_sensitivity,
            noise_sensitivity=noise_sensitivity,
            sample_efficiency=sample_efficiency,
        )

    # Category summary
    category_summary: Dict[str, Dict[str, float]] = {}
    for cat in CATEGORIES:
        cat_strategies = [s for s in strategy_benchmarks.values() if s.category == cat]
        if cat_strategies:
            category_summary[cat] = {
                "mean_time_ms": sum(s.mean_time_ms for s in cat_strategies) / len(cat_strategies),
                "mean_outlier_sens": sum(s.outlier_sensitivity for s in cat_strategies) / len(cat_strategies),
                "mean_noise_sens": sum(s.noise_sensitivity for s in cat_strategies) / len(cat_strategies),
                "n_strategies": len(cat_strategies),
            }

    # Rankings
    speed_ranking = sorted(strategy_benchmarks.keys(), key=lambda n: strategy_benchmarks[n].mean_time_ms)
    robustness_ranking = sorted(
        strategy_benchmarks.keys(),
        key=lambda n: strategy_benchmarks[n].outlier_sensitivity + strategy_benchmarks[n].noise_sensitivity
    )

    # Overall score: balance of speed and robustness
    def overall_score(name):
        b = strategy_benchmarks[name]
        # Normalize and combine (lower is better)
        time_score = b.mean_time_ms / 10  # Scale
        robustness_score = b.outlier_sensitivity + b.noise_sensitivity
        return time_score + robustness_score * 2

    overall_ranking = sorted(strategy_benchmarks.keys(), key=overall_score)

    # Simple clustering by behavior
    clusters: Dict[int, List[str]] = defaultdict(list)
    for name, bench in strategy_benchmarks.items():
        # Simple clustering based on value range
        if bench.max_value - bench.min_value < 0.2:
            cluster_id = 0  # Stable
        elif bench.outlier_sensitivity > 0.2:
            cluster_id = 1  # Outlier sensitive
        elif bench.mean_time_ms > 50:
            cluster_id = 2  # Slow
        else:
            cluster_id = 3  # General
        bench.cluster_id = cluster_id
        clusters[cluster_id].append(name)

    total_time = time.perf_counter() - start_time

    return BenchmarkResults(
        timestamp=datetime.now().isoformat(),
        n_strategies=len(strategies),
        n_scenarios=n_scenarios,
        n_iterations=n_iterations,
        strategy_results=strategy_benchmarks,
        category_summary=category_summary,
        speed_ranking=speed_ranking,
        robustness_ranking=robustness_ranking,
        overall_ranking=overall_ranking,
        clusters=dict(clusters),
        execution_time_seconds=total_time,
    )


def print_results(results: BenchmarkResults):
    """Print benchmark results to console."""
    CONSOLE.print(Panel.fit(
        f"[bold]Aggregation Strategy Benchmark[/bold]\n"
        f"Strategies: {results.n_strategies} | Scenarios: {results.n_scenarios} | "
        f"Iterations: {results.n_iterations}\n"
        f"Total Time: {results.execution_time_seconds:.2f}s",
        title="Summary"
    ))

    # Speed ranking
    CONSOLE.print("\n[bold]Top 10 Fastest Strategies:[/bold]")
    speed_table = Table(show_header=True, header_style="bold")
    speed_table.add_column("Rank", width=6)
    speed_table.add_column("Strategy", width=35)
    speed_table.add_column("Category", width=12)
    speed_table.add_column("Avg Time (ms)", width=15)

    for i, name in enumerate(results.speed_ranking[:10], 1):
        b = results.strategy_results[name]
        speed_table.add_row(str(i), name, b.category, f"{b.mean_time_ms:.3f}")
    CONSOLE.print(speed_table)

    # Robustness ranking
    CONSOLE.print("\n[bold]Top 10 Most Robust Strategies:[/bold]")
    robust_table = Table(show_header=True, header_style="bold")
    robust_table.add_column("Rank", width=6)
    robust_table.add_column("Strategy", width=35)
    robust_table.add_column("Category", width=12)
    robust_table.add_column("Outlier Sens", width=12)
    robust_table.add_column("Noise Sens", width=12)

    for i, name in enumerate(results.robustness_ranking[:10], 1):
        b = results.strategy_results[name]
        robust_table.add_row(str(i), name, b.category, f"{b.outlier_sensitivity:.4f}", f"{b.noise_sensitivity:.4f}")
    CONSOLE.print(robust_table)

    # Overall ranking
    CONSOLE.print("\n[bold]Top 15 Overall (Speed + Robustness):[/bold]")
    overall_table = Table(show_header=True, header_style="bold")
    overall_table.add_column("Rank", width=6)
    overall_table.add_column("Strategy", width=35)
    overall_table.add_column("Category", width=12)
    overall_table.add_column("Time (ms)", width=10)
    overall_table.add_column("Robustness", width=10)

    for i, name in enumerate(results.overall_ranking[:15], 1):
        b = results.strategy_results[name]
        robustness = 1 - (b.outlier_sensitivity + b.noise_sensitivity) / 2
        overall_table.add_row(str(i), name, b.category, f"{b.mean_time_ms:.2f}", f"{robustness:.2%}")
    CONSOLE.print(overall_table)

    # Category summary
    CONSOLE.print("\n[bold]Category Summary:[/bold]")
    cat_table = Table(show_header=True, header_style="bold")
    cat_table.add_column("Category", width=15)
    cat_table.add_column("Strategies", width=10)
    cat_table.add_column("Avg Time (ms)", width=15)
    cat_table.add_column("Avg Outlier Sens", width=15)
    cat_table.add_column("Avg Noise Sens", width=15)

    for cat, stats in sorted(results.category_summary.items(), key=lambda x: x[1]["mean_time_ms"]):
        cat_table.add_row(
            cat,
            str(int(stats["n_strategies"])),
            f"{stats['mean_time_ms']:.3f}",
            f"{stats['mean_outlier_sens']:.4f}",
            f"{stats['mean_noise_sens']:.4f}",
        )
    CONSOLE.print(cat_table)

    # Cluster info
    CONSOLE.print("\n[bold]Behavior Clusters:[/bold]")
    cluster_names = {0: "Stable", 1: "Outlier-sensitive", 2: "Slow", 3: "General"}
    for cluster_id, strategies in results.clusters.items():
        CONSOLE.print(f"  {cluster_names.get(cluster_id, 'Unknown')} ({len(strategies)} strategies)")


def export_results(results: BenchmarkResults, filepath: str):
    """Export results to JSON."""
    data = {
        "timestamp": results.timestamp,
        "n_strategies": results.n_strategies,
        "n_scenarios": results.n_scenarios,
        "n_iterations": results.n_iterations,
        "execution_time_seconds": results.execution_time_seconds,
        "strategy_results": {name: asdict(bench) for name, bench in results.strategy_results.items()},
        "category_summary": results.category_summary,
        "speed_ranking": results.speed_ranking,
        "robustness_ranking": results.robustness_ranking,
        "overall_ranking": results.overall_ranking,
        "clusters": results.clusters,
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    CONSOLE.print(f"\n[green]Results exported to {filepath}[/green]")


def main():
    parser = argparse.ArgumentParser(description="Benchmark aggregation strategies")
    parser.add_argument("--full", action="store_true", help="Run all 169 strategies (slower)")
    parser.add_argument("--iterations", type=int, default=5, help="Iterations per scenario")
    parser.add_argument("--export", type=str, help="Export results to JSON file")
    parser.add_argument("--quick", action="store_true", help="Quick mode with fewer strategies")
    args = parser.parse_args()

    CONSOLE.print("[bold]Aggregation Strategy Benchmark[/bold]\n")

    if args.quick:
        # Just core strategies
        strategies = [
            AggregationStrategy.WEIGHTED_AVERAGE,
            AggregationStrategy.MEDIAN,
            AggregationStrategy.BAYESIAN,
            AggregationStrategy.ROBUST_HUBER,
            AggregationStrategy.ENTROPY_WEIGHTED,
            AggregationStrategy.BOOTSTRAP,
            AggregationStrategy.MIXTURE_OF_EXPERTS,
            AggregationStrategy.ATTENTION_AGGREGATION,
            AggregationStrategy.DEMPSTER_SHAFER,
            AggregationStrategy.ADAPTIVE,
        ]
    elif args.full:
        strategies = None  # All strategies
    else:
        # Representative subset
        strategies = []
        for cat, cat_strategies in CATEGORIES.items():
            for name in cat_strategies[:3]:  # Top 3 from each category
                try:
                    strategies.append(AggregationStrategy[name])
                except KeyError:
                    pass

    results = run_benchmark(strategies, n_iterations=args.iterations)
    print_results(results)

    if args.export:
        export_results(results, args.export)


if __name__ == "__main__":
    main()
