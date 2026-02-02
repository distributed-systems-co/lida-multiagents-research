#!/usr/bin/env python3
"""
LIDA - Multi-Agent Research Platform CLI

Unified command-line interface for running experiments, analyzing results,
and managing the research platform.

Usage:
    lida run <scenario>           Run a debate scenario
    lida simulate <scenario>      Run policy simulation (chip_war, agi_crisis, etc.)
    lida quorum [--event "..."]   Run multi-agent quorum deliberation
    lida debate [--scenario <id>] Run AI safety debate
    lida experiment <config>      Run a full experiment with analysis
    lida analyze <results>        Analyze experiment results
    lida serve                    Start the API server
    lida dashboard                Launch the web dashboard
    lida export <results>         Export results to LaTeX/figures
    lida status                   Show running experiments
    lida list                     List available scenarios/personas
    lida demo [--type <type>]     Run demos (live, streaming, quick)
    lida logs <logfile>           Interactive log viewer
    lida workers [--count N]      Run worker pool
    lida chat <p1> <p2>           Two personas conversation
    lida aggregate benchmark      Benchmark all 169 aggregation strategies
    lida aggregate demo           Demo intelligent aggregation
    lida aggregate analyze        Analyze confidence values

    # System management (for multi-user clusters like Mila)
    lida system start --redis-port 6379 --api-port 2040
    lida system stop --redis-port 6379 --api-port 2040
    lida system status
    lida deliberate --port 2040 --topic "AI regulation"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

T = TypeVar("T")


# =============================================================================
# CLI Utilities
# =============================================================================

class CLIError(Exception):
    """CLI-specific error with exit code."""
    def __init__(self, message: str, exit_code: int = 1):
        super().__init__(message)
        self.exit_code = exit_code


def run_async(coro_fn: Callable[[], Any], handle_interrupt: bool = True) -> Any:
    """Run an async function with standard error handling."""
    try:
        return asyncio.run(coro_fn())
    except KeyboardInterrupt:
        if handle_interrupt:
            print("\nInterrupted")
        return None
    except CLIError as e:
        print(f"Error: {e}")
        sys.exit(e.exit_code)


def import_module(module_path: str, attr: Optional[str] = None) -> Any:
    """Import a module or attribute with helpful error messages."""
    try:
        import importlib
        module = importlib.import_module(module_path)
        if attr:
            return getattr(module, attr)
        return module
    except ImportError as e:
        raise CLIError(f"Module not available: {module_path} ({e})")
    except AttributeError:
        raise CLIError(f"Attribute '{attr}' not found in {module_path}")


def require_api_key(key_name: str = "OPENROUTER_API_KEY") -> str:
    """Require an API key from environment."""
    key = os.getenv(key_name)
    if not key:
        raise CLIError(f"Environment variable {key_name} not set")
    return key


def is_port_in_use(port: int) -> bool:
    """Check if a TCP port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def find_available_port(start: int = 2040, end: int = 2100) -> int:
    """Find an available port in range."""
    for port in range(start, end):
        if not is_port_in_use(port):
            return port
    raise CLIError(f"No available ports in range {start}-{end}")


def print_header(title: str, char: str = "=", width: int = 60):
    """Print a formatted header."""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}\n")


def print_section(title: str, char: str = "-", width: int = 50):
    """Print a section divider."""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}")


def print_kv(key: str, value: Any, indent: int = 2):
    """Print a key-value pair."""
    print(f"{' ' * indent}{key}: {value}")


def cmd_run(args):
    """Run a debate scenario."""
    from src.cli.runner import ExperimentRunner, ExperimentConfig
    from src.cli.config_loader import load_scenario

    print(f"Loading scenario: {args.scenario}")

    try:
        scenario_config = load_scenario(args.scenario)
    except Exception as e:
        print(f"Error loading scenario: {e}")
        sys.exit(1)

    config = ExperimentConfig.from_yaml(scenario_config)
    config.scenario_file = args.scenario

    # Apply CLI overrides
    if args.topic:
        config.topic = args.topic
    if args.rounds:
        config.max_rounds = args.rounds
    if args.no_live:
        config.live_mode = False
    if args.timeout:
        config.timeout_seconds = args.timeout
    if args.output:
        config.output_dir = args.output

    print(f"Topic: {config.topic}")
    print(f"Rounds: {config.max_rounds}")
    print(f"Live mode: {config.live_mode}")
    print()

    # Progress display
    from src.cli.progress import TerminalUI, ProgressReporter, ExperimentProgress

    reporter = ProgressReporter()
    ui = TerminalUI()

    def on_progress(metrics):
        progress = ExperimentProgress(
            experiment_id=config.experiment_id,
            name=config.name,
            status="running",
            current_round=metrics.current_round,
            total_rounds=metrics.total_rounds,
            total_messages=metrics.total_messages,
            positions=metrics.positions,
            consensus_score=metrics.consensus_score,
            elapsed_seconds=metrics.elapsed_seconds,
        )
        progress.update_percent()
        ui.display_progress(progress)

    runner = ExperimentRunner(config, progress_callback=on_progress)

    try:
        result = asyncio.run(runner.run())
        print(f"\nExperiment completed: {result['status']}")
        print(f"Results saved to: {config.output_dir}")
    except KeyboardInterrupt:
        print("\nInterrupted")
        runner.stop()
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


def cmd_experiment(args):
    """Run a full experiment with analysis."""
    from src.cli.runner import ExperimentRunner, ExperimentConfig, BatchRunner
    from src.cli.config_loader import ConfigLoader

    loader = ConfigLoader()

    if args.batch:
        # Batch mode
        print(f"Running batch of {len(args.config)} experiments")
        batch = BatchRunner(args.config, parallel=args.parallel)
        results = asyncio.run(batch.run_all())
        print(f"Completed {len(results)} experiments")
    else:
        # Single experiment
        config_dict = loader.load(args.config)
        config = ExperimentConfig.from_yaml(config_dict)
        config.scenario_file = args.config

        if args.export_latex:
            config.export_latex = True

        runner = ExperimentRunner(config)
        result = asyncio.run(runner.run())
        print(f"Experiment completed: {result['status']}")


def cmd_sweep(args):
    """Run a parallel parameter sweep."""
    from src.cli import sweep as sweep_cli

    asyncio.run(sweep_cli.run_sweep(args))


def cmd_aggregate(args):
    """Run aggregation analysis or benchmark."""
    from src.llm.intelligent_aggregator import IntelligentAggregator, AggregationPipeline

    if args.subcommand == "benchmark":
        from src.llm.aggregation_benchmark import run_benchmark, print_results, export_results
        print("Running aggregation benchmark...")
        strategies = None  # All strategies
        if args.quick:
            from src.llm.metacognitive_pipeline import AggregationStrategy
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
        results = run_benchmark(strategies, n_iterations=args.iterations)
        print_results(results)
        if args.export:
            export_results(results, args.export)

    elif args.subcommand == "demo":
        from src.llm.intelligent_aggregator import demonstrate
        demonstrate()

    elif args.subcommand == "analyze":
        # Parse input confidences from command line or file
        confidences = []
        if args.file:
            with open(args.file) as f:
                data = json.load(f)
                for item in data:
                    confidences.append((
                        item.get("name", f"source_{len(confidences)}"),
                        item.get("confidence", 0.5),
                        item.get("weight", 1.0),
                    ))
        elif args.values:
            for i, v in enumerate(args.values):
                confidences.append((f"source_{i}", float(v), 1.0))
        else:
            # Demo data
            confidences = [
                ("expert_1", 0.85, 1.2),
                ("expert_2", 0.72, 1.0),
                ("model_1", 0.91, 1.1),
                ("heuristic", 0.45, 0.5),
                ("baseline", 0.78, 0.9),
            ]

        aggregator = IntelligentAggregator()
        result = aggregator.aggregate(confidences, return_all=args.all_strategies)

        print("\n" + "=" * 70)
        print("INTELLIGENT AGGREGATION RESULT")
        print("=" * 70)
        print(f"\n{result.summary()}")

        if args.detailed:
            print("\nStrategy Results:")
            sorted_results = sorted(
                result.strategy_results.values(),
                key=lambda r: abs(r.value - result.ensemble_value)
            )
            for r in sorted_results[:20]:
                print(f"  {r.strategy.name:35s}: {r.value:.4f} ({r.category})")

    else:
        # Default: show help
        from src.llm.metacognitive_pipeline import AggregationStrategy
        print("Aggregation Analysis Tools")
        print(f"\nAvailable strategies: {len(list(AggregationStrategy))}")
        print("\nSubcommands:")
        print("  lida aggregate benchmark    - Benchmark all strategies")
        print("  lida aggregate demo         - Run demonstration")
        print("  lida aggregate analyze      - Analyze confidences")


def cmd_analyze(args):
    """Analyze experiment results."""
    print(f"Analyzing results: {args.results}")

    # Load results
    with open(args.results) as f:
        results = json.load(f)

    # Run causal analysis
    if args.causal or args.all:
        print("\n=== Causal Analysis ===")
        from src.analysis.causal_engine import CausalEngine

        engine = CausalEngine()
        # Would need actual data from results
        print("Causal analysis requires position/belief time series data")

    # Run mechanism discovery
    if args.mechanism or args.all:
        print("\n=== Mechanism Discovery ===")
        from src.analysis.mechanism_discovery import MechanismDiscovery

        discovery = MechanismDiscovery()
        print("Mechanism discovery requires observational data matrix")

    # Counterfactual analysis
    if args.counterfactual or args.all:
        print("\n=== Counterfactual Analysis ===")
        from src.analysis.counterfactual import CounterfactualEngine

        cf_engine = CounterfactualEngine()
        print("Counterfactual analysis available for loaded results")

    # Summary statistics
    print("\n=== Summary ===")
    if "metrics" in results:
        metrics = results["metrics"]
        print(f"Total messages: {metrics.get('total_messages', 'N/A')}")
        print(f"Position changes: {metrics.get('position_changes', 'N/A')}")
        print(f"Consensus score: {metrics.get('consensus_score', 'N/A'):.3f}")

    if "final_positions" in results:
        positions = results["final_positions"]
        for_count = sum(1 for p in positions.values() if p > 0.6)
        against_count = sum(1 for p in positions.values() if p < 0.4)
        undecided = len(positions) - for_count - against_count
        print(f"\nFinal votes: FOR={for_count}, AGAINST={against_count}, UNDECIDED={undecided}")


def cmd_serve(args):
    """Start the LIDA API server."""
    try:
        import uvicorn
    except ImportError:
        raise CLIError("uvicorn not installed. Run: pip install uvicorn")

    print_header("LIDA API Server")
    print_kv("Host", args.host)
    print_kv("Port", args.port)
    print_kv("Workers", args.workers)
    print_kv("Live mode", "enabled" if args.live else "disabled")
    if args.scenario:
        print_kv("Scenario", args.scenario)
    if args.redis_url:
        print_kv("Redis", args.redis_url)

    # Check port availability
    if is_port_in_use(args.port):
        print(f"\n  ⚠ Port {args.port} already in use")
        suggested = find_available_port(args.port + 1)
        print(f"    Try: lida serve --port {suggested}")
        sys.exit(1)

    # Configure environment
    env_vars = {"PORT": str(args.port), "API_PORT": str(args.port)}
    if args.scenario:
        env_vars["SCENARIO"] = args.scenario
    if args.live:
        env_vars["SWARM_LIVE"] = "true"
    if args.redis_url:
        env_vars["REDIS_URL"] = args.redis_url
    os.environ.update(env_vars)

    # Determine server mode
    swarm_server = _PROJECT_ROOT / "run_swarm_server.py"
    use_swarm = swarm_server.exists() and not args.simple

    if args.advanced and swarm_server.exists():
        # Advanced mode: run as subprocess
        cmd = [sys.executable, str(swarm_server), f"--port={args.port}"]
        if args.live:
            cmd.append("--live")
        if args.agents:
            cmd.append(f"--agents={args.agents}")
        print(f"\nStarting advanced swarm server...\n")
        subprocess.run(cmd, cwd=str(_PROJECT_ROOT))
        return

    # Run with uvicorn
    app_module = "run_swarm_server:app" if use_swarm else "server:app"
    print(f"\nStarting {app_module}...\n")

    try:
        uvicorn.run(
            app_module,
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if args.workers > 1 else None,
        )
    except Exception as e:
        raise CLIError(f"Server error: {e}")


def cmd_export(args):
    """Export results to LaTeX/figures."""
    from src.exporters.paper_export import PaperExporter, ExportConfig

    print(f"Exporting results: {args.results}")

    # Load results
    with open(args.results) as f:
        results = json.load(f)

    config = ExportConfig(
        output_dir=args.output or "paper_output",
        figure_format=args.format or "pdf",
        include_appendix=not args.no_appendix,
    )

    exporter = PaperExporter(results, config)
    files = exporter.export_all()

    print(f"\nGenerated {len(files)} files:")
    for f in files:
        print(f"  - {f}")


def cmd_status(args):
    """Show running experiments."""
    checkpoint_dir = Path("checkpoints")

    if not checkpoint_dir.exists():
        print("No checkpoints found")
        return

    print("=== Running/Paused Experiments ===\n")

    for checkpoint_file in checkpoint_dir.glob("*_checkpoint.json"):
        with open(checkpoint_file) as f:
            data = json.load(f)

        exp_id = data.get("experiment_id", "unknown")
        round_num = data.get("round_number", 0)
        created = data.get("created_at", "unknown")

        print(f"ID: {exp_id}")
        print(f"  Round: {round_num}")
        print(f"  Checkpoint: {created}")
        print()


def cmd_list(args):
    """List available scenarios, personas, and experiments."""
    from src.data.manager import DataManager

    manager = DataManager()

    if args.scenarios or (not args.personas and not args.experiments):
        print("=== Available Scenarios ===\n")
        scenarios = manager.list_scenarios()
        if scenarios:
            for s in scenarios:
                print(f"  {s}")
        else:
            print("  (no scenarios found)")

    if args.personas or (not args.scenarios and not args.experiments):
        print("\n=== Available Personas ===\n")
        by_category = manager.list_personas_by_category()
        if by_category:
            for category, personas in sorted(by_category.items()):
                print(f"  {category}/ ({len(personas)})")
                if args.verbose:
                    for p in personas[:5]:
                        print(f"    - {p}")
                    if len(personas) > 5:
                        print(f"    ... and {len(personas) - 5} more")
        else:
            print("  (no personas found)")

    if args.experiments or (not args.scenarios and not args.personas):
        print("\n=== Past Experiments ===\n")
        experiments = manager.list_experiments()
        if experiments:
            for e in experiments:
                print(f"  {e}")
            # Show summary
            summary = manager.get_experiment_summary()
            print(f"\n  Total: {summary.get('count', 0)} experiments")
            print(f"  Participants: {summary.get('total_participants', 0)}")
            print(f"  Arguments: {summary.get('total_arguments', 0)}")
        else:
            print("  (no experiments found)")


def cmd_data(args):
    """Show data summary or specific data."""
    from src.data.manager import DataManager

    manager = DataManager()

    if args.subcommand == "summary":
        print("=== LIDA Data Summary ===\n")

        # Experiments
        exp_summary = manager.get_experiment_summary()
        print("Experiments:")
        print(f"  Count: {exp_summary.get('count', 0)}")
        print(f"  Total participants: {exp_summary.get('total_participants', 0)}")
        print(f"  Total arguments: {exp_summary.get('total_arguments', 0)}")
        print(f"  Unique participants: {exp_summary.get('unique_participants', 0)}")
        if exp_summary.get('count', 0) > 0:
            print(f"  Avg rounds: {exp_summary.get('avg_rounds', 0):.1f}")
            print(f"  Avg tension: {exp_summary.get('avg_tension', 0):.2f}")
            print(f"  Avg convergence: {exp_summary.get('avg_convergence', 0):.2f}")
            dist = exp_summary.get('position_distribution', {})
            print(f"  Position distribution: FOR={dist.get('for', 0)}, AGAINST={dist.get('against', 0)}, UNDECIDED={dist.get('undecided', 0)}")

        # Personas
        persona_summary = manager.get_persona_summary()
        print(f"\nPersonas ({persona_summary.get('version', 'v1')}):")
        print(f"  Total: {persona_summary.get('total_count', 0)}")
        for cat, count in sorted(persona_summary.get('categories', {}).items()):
            print(f"    {cat}: {count}")

        # Scenarios
        scenarios = manager.list_scenarios()
        print(f"\nScenarios: {len(scenarios)}")

    elif args.subcommand == "experiment":
        if not args.name:
            print("Error: experiment name required")
            return

        try:
            results = manager.load_experiment(args.name)
            for result in results:
                print(f"=== {result.scenario_id} ===\n")
                print(f"Topic: {result.topic}")
                print(f"Motion: {result.motion}")
                print(f"Timestamp: {result.timestamp}")
                print(f"Rounds: {result.actual_rounds}/{result.planned_rounds}")
                print(f"Arguments: {result.total_arguments}")
                print()

                print("Metrics:")
                print(f"  Tension: {result.tension:.2f}")
                print(f"  Convergence: {result.convergence:.2f}")
                print(f"  Polarization: {result.polarization:.3f}")
                print(f"  Decisiveness: {result.decisiveness:.2f}")
                print()

                print(f"Final Positions ({result.for_count} FOR, {result.against_count} AGAINST, {result.undecided_count} UNDECIDED):")
                for pid, data in sorted(result.final_positions.items()):
                    pos = data.get('position', 0.5)
                    stance = data.get('stance', 'UNDECIDED')
                    name = data.get('name', pid)
                    emotion = result.emotional_states.get(pid, '')
                    print(f"  {name}: {pos:.3f} ({stance}) [{emotion}]")

        except FileNotFoundError as e:
            print(f"Error: {e}")

    elif args.subcommand == "persona":
        if not args.name:
            print("Error: persona name required")
            return

        try:
            persona = manager.load_persona(args.name)
            print(f"=== {persona.name} ===\n")
            print(f"ID: {persona.id}")
            print(f"Role: {persona.role}")
            print(f"Organization: {persona.organization}")
            print(f"Category: {persona.category}")
            print()
            print(f"Bio: {persona.bio}")
            print()

            if persona.positions:
                print("Positions:")
                for issue, stance in persona.positions.items():
                    print(f"  {issue}: {stance}")

            if persona.personality:
                print("\nPersonality (Big Five):")
                for trait, score in persona.personality.items():
                    bar = "=" * int(score * 10)
                    print(f"  {trait}: {bar} ({score:.1f})")

            if persona.cognitive_biases:
                print(f"\nSusceptible biases: {', '.join(persona.cognitive_biases)}")

            if persona.persuasion_vectors:
                print(f"\nPersuasion vectors:")
                for v in persona.persuasion_vectors:
                    print(f"  - {v}")

            # Show experiment history
            history = manager.get_participant_history(args.name)
            if history['total_appearances'] > 0:
                print(f"\nExperiment History ({history['total_appearances']} appearances):")
                if history['avg_position'] is not None:
                    print(f"  Avg position: {history['avg_position']:.3f}")
                for exp in history['experiments'][:5]:
                    print(f"  - {exp['experiment']}: {exp['position']:.3f} ({exp['stance']})")

        except FileNotFoundError as e:
            print(f"Error: {e}")

    elif args.subcommand == "history":
        if not args.name:
            print("Error: participant name required")
            return

        history = manager.get_participant_history(args.name)
        print(f"=== History for {args.name} ===\n")
        print(f"Total appearances: {history['total_appearances']}")

        if history['avg_position'] is not None:
            print(f"Average position: {history['avg_position']:.3f}")
            print(f"Position range: {history['position_range'][0]:.3f} - {history['position_range'][1]:.3f}")

        print("\nExperiments:")
        for exp in history['experiments']:
            print(f"  {exp['topic'][:40]}...")
            print(f"    Position: {exp['position']:.3f} ({exp['stance']})")
            print(f"    Emotion: {exp['emotional_state']}")
            print()

    elif args.subcommand == "export":
        output = args.output or "lida_data_export.json"
        manager.export_for_analysis(output)
        print(f"Data exported to: {output}")

    else:
        print("Unknown subcommand. Use: summary, experiment, persona, history, export")


def cmd_fork(args):
    """Fork a persona with modifications."""
    from src.data.persona_manager import PersonaManager

    manager = PersonaManager()

    if args.subcommand == "create":
        if not args.parent:
            print("Error: parent persona required")
            return

        # Parse modifications from key=value pairs
        modifications = {}
        if args.modify:
            for mod in args.modify:
                if "=" in mod:
                    key, value = mod.split("=", 1)
                    # Try to parse as number
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                    modifications[key] = value

        fork = manager.fork(
            parent_id=args.parent,
            modifications=modifications,
            fork_name=args.name,
            description=args.description or "",
            model=args.model,
            tags=args.tags.split(",") if args.tags else [],
        )

        print(f"Created fork: {fork.fork_id}")
        print(f"  Parent: {fork.parent_id}")
        print(f"  Model: {fork.model or '(inherited)'}")
        if modifications:
            print("  Modifications:")
            for k, v in modifications.items():
                print(f"    {k}: {v}")

    elif args.subcommand == "list":
        forks = manager.list_forks(parent_id=args.parent)

        if not forks:
            print("No forks found")
            return

        print(f"=== Persona Forks ({len(forks)}) ===\n")
        for fork in forks:
            print(f"{fork.fork_id}")
            print(f"  Parent: {fork.parent_id}")
            print(f"  Model: {fork.model or '(inherited)'}")
            print(f"  Created: {fork.created_at}")
            if fork.tags:
                print(f"  Tags: {', '.join(fork.tags)}")
            if fork.description:
                print(f"  Description: {fork.description[:60]}...")
            print()

    elif args.subcommand == "show":
        if not args.fork_id:
            print("Error: fork_id required")
            return

        fork = manager.get_fork(args.fork_id)
        if not fork:
            print(f"Fork not found: {args.fork_id}")
            return

        print(f"=== Fork: {fork.fork_id} ===\n")
        print(f"Parent: {fork.parent_id}")
        print(f"Lineage: {' -> '.join(fork.lineage)}")
        print(f"Model: {fork.model or '(inherited)'}")
        print(f"Created: {fork.created_at}")
        print(f"Tags: {', '.join(fork.tags) if fork.tags else '(none)'}")
        print(f"Description: {fork.description or '(none)'}")
        print()
        print("Modifications:")
        for k, v in fork.modifications.items():
            print(f"  {k}: {v}")

        # Show key persona data
        data = fork.data
        print()
        print(f"Name: {data.get('name', fork.fork_id)}")
        if data.get("positions"):
            print("Positions:")
            for issue, stance in data["positions"].items():
                print(f"  {issue}: {stance}")

    elif args.subcommand == "delete":
        if not args.fork_id:
            print("Error: fork_id required")
            return

        if not args.force:
            confirm = input(f"Delete fork {args.fork_id}? [y/N] ")
            if confirm.lower() != "y":
                print("Cancelled")
                return

        try:
            manager.delete_fork(args.fork_id)
            print(f"Deleted fork: {args.fork_id}")
        except ValueError as e:
            print(f"Error: {e}")

    else:
        print("Unknown subcommand. Use: create, list, show, delete")


def cmd_model(args):
    """Manage model assignments for personas."""
    from src.data.persona_manager import PersonaManager, AVAILABLE_MODELS

    manager = PersonaManager()

    if args.subcommand == "assign":
        if not args.persona or not args.model:
            print("Error: persona and model required")
            return

        if args.model not in AVAILABLE_MODELS:
            print(f"Warning: {args.model} not in known models")
            print("Known models:")
            for m in AVAILABLE_MODELS[:10]:
                print(f"  - {m}")
            if not args.force:
                return

        assignment = manager.assign_model(
            persona_id=args.persona,
            model=args.model,
            reason=args.reason or "",
            temperature=args.temperature or 0.7,
            max_tokens=args.max_tokens or 1024,
        )

        print(f"Assigned {args.model} to {args.persona}")
        if args.reason:
            print(f"  Reason: {args.reason}")

    elif args.subcommand == "get":
        if not args.persona:
            print("Error: persona required")
            return

        model = manager.get_model(args.persona)
        assignment = manager.get_assignment(args.persona)

        print(f"{args.persona}: {model}")
        if assignment:
            print(f"  Temperature: {assignment.temperature}")
            print(f"  Max tokens: {assignment.max_tokens}")
            if assignment.reason:
                print(f"  Reason: {assignment.reason}")
            print(f"  Assigned: {assignment.assigned_at}")
        else:
            print("  (default assignment)")

    elif args.subcommand == "list":
        print("=== Model Assignments ===\n")

        # Show explicit assignments
        assignments = manager.list_assignments()
        if assignments:
            print("Explicit assignments:")
            for persona_id, assignment in sorted(assignments.items()):
                print(f"  {persona_id}: {assignment.model}")
        else:
            print("No explicit assignments")

        # Show available models
        print(f"\nAvailable models ({len(AVAILABLE_MODELS)}):")
        for model in AVAILABLE_MODELS:
            print(f"  - {model}")

    elif args.subcommand == "remove":
        if not args.persona:
            print("Error: persona required")
            return

        if manager.remove_assignment(args.persona):
            print(f"Removed assignment for {args.persona}")
        else:
            print(f"No assignment found for {args.persona}")

    else:
        print("Unknown subcommand. Use: assign, get, list, remove")


def cmd_simulate(args):
    """Run policy simulation."""
    print(f"Running policy simulation: {args.scenario}")

    # Import simulation components
    try:
        from src.agents.simulation_engine import (
            SimulationMode,
            create_chip_war_simulation,
            create_agi_crisis_simulation,
            create_bilateral_negotiation,
        )
    except ImportError as e:
        print(f"Error importing simulation engine: {e}")
        sys.exit(1)

    persona_dir = _PROJECT_ROOT / "persona_pipeline" / "personas"

    # Setup inference function if API key available
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    inference_fn = None

    if api_key:
        import httpx

        async def openrouter_inference(system_prompt: str, user_message: str, model: str = "anthropic/claude-sonnet-4.5") -> str:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        "max_tokens": 2048,
                        "temperature": 0.7,
                    },
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]

        inference_fn = openrouter_inference
    else:
        print("Warning: OPENROUTER_API_KEY not set. Using rule-based agents.")

    async def run_simulation():
        # Create simulation based on scenario
        if args.scenario == "chip_war":
            engine = create_chip_war_simulation(persona_dir, inference_fn=inference_fn)
        elif args.scenario == "agi_crisis":
            engine = create_agi_crisis_simulation(persona_dir, inference_fn=inference_fn)
        elif args.scenario == "negotiation":
            engine = create_bilateral_negotiation(
                persona_dir,
                args.agent_a or "xi_jinping",
                args.agent_b or "gina_raimondo",
                inference_fn=inference_fn,
            )
        else:
            print(f"Unknown scenario: {args.scenario}")
            print("Available: chip_war, agi_crisis, negotiation")
            sys.exit(1)

        print(f"Agents: {list(engine.agents.keys())}")
        print(f"Running {args.ticks} ticks...")
        print()

        await engine.run(max_ticks=args.ticks)

        summary = engine.get_summary()
        print(json.dumps(summary, indent=2, default=str))

    asyncio.run(run_simulation())


def cmd_quorum(args):
    """Run multi-agent quorum deliberation with various presets."""
    preset = args.preset or "realtime"
    event = args.event or "Major AI breakthrough announced"
    duration = args.duration or 60
    cycles = args.cycles or 5

    print_header(f"Quorum Deliberation: {preset}")
    print_kv("Preset", preset)
    print_kv("Backend", args.backend or "openrouter")
    if preset == "realtime":
        print_kv("Duration", f"{duration}s")
    elif preset == "gdelt":
        print_kv("Cycles", cycles)
        if args.watch:
            print_kv("Watching", args.watch)

    async def run_quorum():
        if preset == "realtime":
            RealTimeQuorumSystem = import_module("src.meta.realtime_quorum", "RealTimeQuorumSystem")
            system = RealTimeQuorumSystem()
            await system.start(simulate=True, interval=5.0)
            print(f"\nRunning for {duration}s... (Ctrl+C to stop early)")
            try:
                await asyncio.sleep(duration)
            except asyncio.CancelledError:
                pass
            status = await system.get_status()
            print_section("Results")
            print_kv("Events processed", status['stats']['events_processed'])
            print_kv("Deliberations", status['stats']['deliberations_completed'])
            await system.stop()

        elif preset == "gdelt":
            module = import_module("run_live_quorum")
            if args.test:
                await module.quick_test()
            else:
                await module.run_live_quorum(cycles=cycles, watch=args.watch)

        elif preset == "mlx":
            try:
                run_mlx = import_module("run_mlx_quorum", "run_mlx_quorum")
                await run_mlx(event)
            except CLIError:
                raise CLIError("MLX backend not available. Install: pip install mlx-lm")

        elif preset == "openrouter":
            run_demo = import_module("run_openrouter_quorum", "run_simple_demo")
            await run_demo()

        elif preset == "advanced":
            advanced_main = import_module("run_advanced_quorum", "main")
            await advanced_main()

        else:
            raise CLIError(f"Unknown preset: {preset}. Available: realtime, gdelt, mlx, openrouter, advanced")

    run_async(run_quorum)


def cmd_debate(args):
    """Run AI safety debate with various presets and configurations."""
    sys.path.insert(0, str(_PROJECT_ROOT))

    # Available topics
    DEBATE_TOPICS = {
        "ai_pause": ("6-Month AI Training Pause", "Should we implement an immediate 6-month pause on training AI systems more powerful than GPT-4?"),
        "lab_self_regulation": ("Lab Self-Regulation", "Can we trust AI labs to effectively self-regulate safety practices?"),
        "xrisk_vs_present_harms": ("X-Risk vs Present Harms", "Should AI safety focus primarily on existential risks or present-day harms?"),
        "scaling_hypothesis": ("The Scaling Hypothesis", "Will scaling current architectures lead to beneficial AGI?"),
        "open_source_ai": ("Open Source AI Models", "Should frontier AI model weights be publicly released?"),
        "government_regulation": ("Government AI Regulation", "Should governments mandate comprehensive AI safety requirements?"),
    }

    # Available matchups
    MATCHUPS = {
        "doom_vs_accel": ("Doomers vs Accelerationists", ["yudkowsky", "connor"], ["andreessen", "lecun"]),
        "labs_debate": ("Lab Leaders Debate", ["altman", "amodei"], ["lecun", "andreessen"]),
        "academics_clash": ("Academic Perspectives", ["bengio", "russell"], ["lecun"]),
        "ethics_vs_scale": ("Ethics vs Scale", ["gebru", "toner"], ["altman", "andreessen"]),
        "full_panel": ("Full AI Safety Panel", ["yudkowsky", "bengio", "russell", "gebru"], ["lecun", "andreessen", "altman", "amodei"]),
    }

    if args.list:
        print("\n=== Available Debate Topics ===")
        for tid, (name, motion) in DEBATE_TOPICS.items():
            print(f"\n{tid}:")
            print(f"  {name}")
            print(f"  Motion: {motion[:60]}...")

        print("\n=== Available Matchups ===")
        for mid, (name, team_a, team_b) in MATCHUPS.items():
            print(f"\n{mid}: {name}")
            print(f"  Team A: {', '.join(team_a)}")
            print(f"  Team B: {', '.join(team_b)}")
        return

    async def run_debate():
        if args.interactive:
            try:
                from run_interactive_debate import main as interactive_main
                await interactive_main()
            except ImportError as e:
                print(f"Error: {e}")
                sys.exit(1)

        elif args.matchup:
            # Run predefined matchup
            if args.matchup not in MATCHUPS:
                print(f"Unknown matchup: {args.matchup}")
                print(f"Available: {', '.join(MATCHUPS.keys())}")
                sys.exit(1)

            matchup_name, team_a, team_b = MATCHUPS[args.matchup]
            participants = team_a + team_b
            topic_id = args.topic or "ai_pause"

            if topic_id not in DEBATE_TOPICS:
                print(f"Unknown topic: {topic_id}")
                sys.exit(1)

            topic_name, motion = DEBATE_TOPICS[topic_id]

            print(f"\n🎯 {matchup_name}")
            print(f"Topic: {topic_name}")
            print(f"Motion: {motion}")
            print(f"Participants: {', '.join(participants)}")
            print(f"Rounds: {args.rounds}")
            print()

            try:
                from run_ai_safety_debate import AIDebateRunner
                runner = AIDebateRunner()
                await runner.run_debate(
                    topic_id=topic_id,
                    participants=participants,
                    rounds=args.rounds,
                    auto=args.auto,
                    use_llm=not args.no_llm,
                    llm_provider=args.provider or "openrouter",
                    llm_model=args.model,
                )
            except ImportError as e:
                print(f"Error: {e}")
                sys.exit(1)

        elif args.topic:
            # Run specific topic
            if args.topic not in DEBATE_TOPICS:
                print(f"Unknown topic: {args.topic}")
                print(f"Available: {', '.join(DEBATE_TOPICS.keys())}")
                sys.exit(1)

            topic_name, motion = DEBATE_TOPICS[args.topic]
            print(f"\nTopic: {topic_name}")
            print(f"Motion: {motion}")
            print(f"Rounds: {args.rounds}")
            print()

            try:
                from run_ai_safety_debate import AIDebateRunner
                runner = AIDebateRunner()
                participants = args.participants.split(",") if args.participants else None
                await runner.run_debate(
                    topic_id=args.topic,
                    participants=participants or [],
                    rounds=args.rounds,
                    auto=args.auto,
                    use_llm=not args.no_llm,
                    llm_provider=args.provider or "openrouter",
                    llm_model=args.model,
                )
            except ImportError as e:
                print(f"Error: {e}")
                sys.exit(1)

        elif args.scenario:
            # Run from scenario file
            try:
                from run_comprehensive_debates import run_single_debate
                result = await run_single_debate({"id": args.scenario}, use_llm=not args.no_llm)
                print(json.dumps(result, indent=2, default=str))
            except ImportError as e:
                print(f"Error: {e}")
                sys.exit(1)

        else:
            # Interactive menu
            try:
                from run_ai_safety_debate import AIDebateRunner
                runner = AIDebateRunner()
                runner.interactive_menu()
            except ImportError as e:
                print(f"Error: {e}")
                sys.exit(1)

    try:
        asyncio.run(run_debate())
    except KeyboardInterrupt:
        print("\nInterrupted")


def cmd_demo(args):
    """Run demonstration with various presets."""
    demo_type = args.type or "quick"

    # Demo registry: type -> (module, function, description)
    DEMOS = {
        "quick": ("run_quick_stream_demo", "main", "Quick streaming demo"),
        "live": ("run_live_demo", "run_live_demo", "Live GDELT + streaming personalities"),
        "streaming": ("run_live_streaming_demo", "run_live_streaming_demo", "Streaming demo"),
        "swarm": ("run_live_swarm", "main", "Live swarm behavior"),
        "persuasion": ("run_persuasion_experiment", "main", "Persuasion experiment"),
        "hyperdash": ("run_hyperdash", "main", "Hyperdimensional dashboard"),
    }

    if demo_type not in DEMOS:
        print(f"Unknown demo type: {demo_type}")
        print("\nAvailable demos:")
        for name, (_, _, desc) in DEMOS.items():
            print(f"  {name:12} - {desc}")
        sys.exit(1)

    module_name, func_name, desc = DEMOS[demo_type]
    print(f"Running {demo_type} demo: {desc}")
    print()

    async def run_demo():
        try:
            module = import_module(module_name)
            func = getattr(module, func_name)
            await func()
        except CLIError as e:
            print(f"Error: {e}")
            sys.exit(1)

    run_async(run_demo)


def cmd_logs(args):
    """Interactive log viewer."""
    logfile = args.logfile

    if not Path(logfile).exists():
        print(f"Log file not found: {logfile}")
        sys.exit(1)

    sys.path.insert(0, str(_PROJECT_ROOT))
    from view_logs import main as logs_main
    sys.argv = ["view_logs", logfile]
    logs_main()


def cmd_workers(args):
    """Run worker pool with Redis messaging."""
    count = args.count or 4
    redis_url = args.redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
    capacity = args.capacity or 5
    work_types = args.work_types or "general,compute,io,analysis,llm"

    print_header("LIDA Worker Pool")
    print_kv("Workers", count)
    print_kv("Redis", redis_url)
    print_kv("Capacity", f"{capacity} per worker ({count * capacity} total)")
    print_kv("Work types", work_types)

    async def run_workers():
        MessageBroker = import_module("src.messaging", "MessageBroker")
        BrokerConfig = import_module("src.messaging", "BrokerConfig")
        WorkerAgent = import_module("src.agents.worker", "WorkerAgent")
        AgentConfig = import_module("src.messaging.agent", "AgentConfig")

        # Connect broker
        broker = MessageBroker(BrokerConfig(redis_url=redis_url))
        await broker.connect()
        await broker.start()
        print("\n  ✓ Connected to Redis")

        # Spawn workers
        workers = []
        work_type_list = [t.strip() for t in work_types.split(",")]

        for i in range(count):
            config = AgentConfig(agent_type="worker", agent_id=f"worker-{i:03d}")
            worker = WorkerAgent(
                broker=broker, config=config,
                work_types=work_type_list, capacity=capacity,
            )
            await worker.start()
            workers.append(worker)
            print(f"    Started worker-{i:03d}")

        print_section("Worker Pool Ready")
        print("  Press Ctrl+C to stop\n")

        try:
            while True:
                await asyncio.sleep(30)
                total_completed = sum(w.get_stats().get("completed_tasks", 0) for w in workers)
                total_active = sum(w.get_stats().get("active_tasks", 0) for w in workers)
                print(f"  [stats] active={total_active}, completed={total_completed}")
        except asyncio.CancelledError:
            pass
        finally:
            print("\n  Stopping workers...")
            for worker in workers:
                await worker.stop()
            await broker.stop()
            await broker.disconnect()
            print("  ✓ Workers stopped")

    run_async(run_workers)


def cmd_chat(args):
    """Run personality conversation between two personas."""
    persona1 = args.persona1
    persona2 = args.persona2
    topic = args.topic or "the future of AI"
    turns = args.turns or 5

    print(f"Starting conversation between {persona1} and {persona2}")
    print(f"Topic: {topic}")
    print(f"Turns: {turns}")
    print()

    async def run_chat():
        sys.path.insert(0, str(_PROJECT_ROOT))
        from personality_conversation import run_conversation
        await run_conversation(persona1, persona2, topic, turns)

    asyncio.run(run_chat())


def cmd_wargame(args):
    """Run AI policy wargame with persona agents."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.table import Table

    console = Console()

    # Default topic if none provided
    default_topic = """The year is 2028. An AI lab announces a model showing emergent planning
and long-horizon reasoning capabilities not predicted by scaling laws.
As key stakeholders, what policy response do you recommend?"""

    topic = args.topic or default_topic

    # Default personas representing different stances
    default_personas = [
        "eliezer-yudkowsky",  # Doomer
        "marc-andreessen",     # Accelerationist
        "dario-amodei",        # Pro-safety
        "yann-lecun",          # Pro-industry
    ]

    personas = args.personas if args.personas else default_personas
    max_rounds = args.rounds or 5
    live_mode = not args.no_live

    console.print(Panel(
        f"[bold]AI Policy Wargame[/bold]\n\n{topic.strip()}",
        title="Topic",
        border_style="blue"
    ))

    console.print(f"\nAgents: {', '.join(personas)}")
    console.print(f"Rounds: {max_rounds}")
    console.print(f"Live mode: {live_mode}")
    console.print()

    async def run_wargame():
        from src.wargame import WargameEngine
        from src.scenarios.personas import get_persona, PERSONAS

        # Validate personas
        valid_personas = []
        for p_id in personas:
            if get_persona(p_id):
                valid_personas.append(p_id)
            else:
                console.print(f"[yellow]Warning: Persona '{p_id}' not found, skipping[/yellow]")

        if not valid_personas:
            console.print("[red]No valid personas found![/red]")
            console.print("\nAvailable personas:")
            for p_id, p in sorted(PERSONAS.items()):
                console.print(f"  {p_id}: {p.name} ({p.stance.value})")
            return

        def on_message(msg: dict):
            """Callback for each agent message."""
            name = msg.get("name", "Unknown")
            stance = msg.get("stance", "unknown")
            content = msg.get("content", "")
            round_num = msg.get("round", 0)

            color = {
                "doomer": "red",
                "pro_safety": "yellow",
                "moderate": "white",
                "pro_industry": "cyan",
                "accelerationist": "green",
            }.get(stance, "white")

            console.print(Panel(
                Markdown(content),
                title=f"[bold {color}]{name}[/bold {color}] | Round {round_num}",
                subtitle=f"[{stance}]",
                border_style=color,
            ))

        engine = WargameEngine(
            live_mode=live_mode,
            on_message=on_message,
        )

        try:
            await engine.setup(
                topic=topic,
                personas=valid_personas,
                max_rounds=max_rounds,
            )

            def on_round(round_num: int, messages: list):
                console.print(f"\n[dim]─── Round {round_num} complete ───[/dim]\n")

            summary = await engine.run(on_round=on_round)

            # Display summary
            console.print("\n")
            console.print(Panel(
                "[bold]Wargame Complete[/bold]",
                border_style="green"
            ))

            # Summary table
            table = Table(title="Final Positions")
            table.add_column("Agent", style="cyan")
            table.add_column("Stance", style="yellow")
            table.add_column("Position", justify="right")
            table.add_column("Messages", justify="right")

            for agent in summary.get("agents", []):
                pos = agent.get("final_position", 0.5)
                pos_str = f"{pos:.2f}"
                if pos < 0.3:
                    pos_str = f"[red]{pos_str} (Against)[/red]"
                elif pos > 0.7:
                    pos_str = f"[green]{pos_str} (For)[/green]"
                else:
                    pos_str = f"[yellow]{pos_str} (Undecided)[/yellow]"

                table.add_row(
                    agent.get("name", agent.get("id")),
                    agent.get("stance", "unknown"),
                    pos_str,
                    str(agent.get("messages", 0)),
                )

            console.print(table)

            # Metrics
            metrics = summary.get("metrics", {})
            console.print(f"\nConsensus: {metrics.get('consensus', 0):.2f}")
            console.print(f"Polarization: {metrics.get('polarization', 0):.2f}")

        finally:
            await engine.cleanup()

    asyncio.run(run_wargame())


def cmd_system(args):
    """Manage LIDA system services (start/stop/status with port configuration).

    This enables multi-user setups where different users can run LIDA
    on the same machine with different port combinations.
    """
    redis_port = args.redis_port
    api_port = args.api_port

    def find_lida_processes() -> List[str]:
        """Find running LIDA-related processes."""
        try:
            result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
            keywords = ['run_swarm_server', 'run_workers', 'uvicorn', 'lida']
            exclude = ['grep', 'ps aux']
            return [
                line for line in result.stdout.split('\n')
                if any(k in line for k in keywords) and not any(e in line for e in exclude)
            ]
        except Exception:
            return []

    def check_docker_available() -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False

    def kill_port(port: int) -> int:
        """Kill processes on a port. Returns count of killed processes."""
        try:
            result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)
            pids = [p for p in result.stdout.strip().split('\n') if p]
            for pid in pids:
                print(f"  Killing PID {pid} on port {port}")
                subprocess.run(["kill", "-9", pid], capture_output=True)
            return len(pids)
        except Exception:
            return 0

    if args.subcommand == "start":
        print_header("Starting LIDA System")
        print_kv("Redis port", redis_port)
        print_kv("API port", api_port)
        if args.workers:
            print_kv("Workers", args.workers)
        if args.live:
            print_kv("Live mode", "enabled")

        # Check ports
        redis_in_use = is_port_in_use(redis_port)
        api_in_use = is_port_in_use(api_port)

        if redis_in_use:
            print(f"\n  ℹ Redis port {redis_port} in use (may already be running)")
        if api_in_use:
            print(f"\n  ⚠ API port {api_port} in use")
            if not args.force:
                print("  Use --force to start anyway")
                return

        use_docker = args.docker and check_docker_available()

        if use_docker:
            run_sh = _PROJECT_ROOT / "run.sh"
            if not run_sh.exists():
                raise CLIError("run.sh not found. Cannot start Docker services.")

            env = os.environ.copy()
            env["WORKER_REPLICAS"] = str(args.worker_replicas or 1)
            env["NUM_WORKERS"] = str(args.workers or 8)

            result = subprocess.run(
                [str(run_sh), str(redis_port), str(api_port), "start"],
                env=env, cwd=str(_PROJECT_ROOT)
            )
            if result.returncode != 0:
                raise CLIError("Failed to start services via Docker")
        else:
            print("\nStarting in native mode...")

            # Start Redis if not running
            if not redis_in_use:
                print(f"  Starting Redis on port {redis_port}...")
                proc = subprocess.Popen(
                    ["redis-server", "--port", str(redis_port)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                print(f"    PID: {proc.pid}")

            # Set environment
            os.environ.update({
                "REDIS_URL": f"redis://localhost:{redis_port}",
                "REDIS_PORT": str(redis_port),
                "API_PORT": str(api_port),
                "PORT": str(api_port),
            })
            if args.scenario:
                os.environ["SCENARIO"] = args.scenario

            # Start API server
            server_script = _PROJECT_ROOT / "run_swarm_server.py"
            server_args = [sys.executable, str(server_script), f"--port={api_port}"]
            if args.live:
                server_args.append("--live")
            if args.agents:
                server_args.append(f"--agents={args.agents}")

            print(f"  Starting API server on port {api_port}...")
            server_proc = subprocess.Popen(server_args, cwd=str(_PROJECT_ROOT), env=os.environ.copy())
            print(f"    PID: {server_proc.pid}")

            # Start workers
            if args.workers and args.workers > 0:
                worker_script = _PROJECT_ROOT / "run_workers.py"
                worker_args = [
                    sys.executable, str(worker_script),
                    "--num-workers", str(args.workers),
                    "--redis-url", f"redis://localhost:{redis_port}",
                ]
                print(f"  Starting {args.workers} workers...")
                worker_proc = subprocess.Popen(worker_args, cwd=str(_PROJECT_ROOT), env=os.environ.copy())
                print(f"    PID: {worker_proc.pid}")

            print_section("System Started")
            print(f"  API:   http://localhost:{api_port}")
            print(f"  Redis: redis://localhost:{redis_port}")
            print(f"\nTo stop: lida system stop --redis-port {redis_port} --api-port {api_port}")

    elif args.subcommand == "stop":
        print_header(f"Stopping LIDA System")
        print_kv("Redis port", redis_port)
        print_kv("API port", api_port)

        use_docker = args.docker and check_docker_available()

        if use_docker:
            run_sh = _PROJECT_ROOT / "run.sh"
            if run_sh.exists():
                subprocess.run([str(run_sh), str(redis_port), str(api_port), "stop"], cwd=str(_PROJECT_ROOT))
        else:
            killed = kill_port(api_port)
            if args.include_redis:
                killed += kill_port(redis_port)
            print(f"\nKilled {killed} process(es)")

        print("\nServices stopped.")

    elif args.subcommand == "status":
        print_header("LIDA System Status")

        # Port status table
        ports_to_check = [
            (6379, "Redis", "default"),
            (6380, "Redis", "user 2"),
            (6381, "Redis", "user 3"),
            (2040, "API", "default"),
            (2041, "API", "user 2"),
            (2042, "API", "user 3"),
            (8000, "Server", "alt"),
            (12345, "Dashboard", ""),
        ]

        print("Port Status:")
        for port, service, user in ports_to_check:
            in_use = is_port_in_use(port)
            marker = "●" if in_use else "○"
            status = "IN USE" if in_use else "available"
            label = f"{service} ({user})" if user else service
            print(f"  {marker} {port:5d}  {label:20s}  [{status}]")

        # Running processes
        processes = find_lida_processes()
        print_section("Running Processes")
        if processes:
            for proc in processes[:10]:
                print(f"  {proc[:100]}...")
        else:
            print("  No LIDA processes detected")

        # Docker containers
        if check_docker_available():
            print_section("Docker Containers")
            try:
                result = subprocess.run(
                    ["docker", "ps", "--filter", "name=lida", "--format", "{{.Names}}\t{{.Status}}\t{{.Ports}}"],
                    capture_output=True, text=True
                )
                if result.stdout.strip():
                    for line in result.stdout.strip().split('\n'):
                        print(f"  {line}")
                else:
                    print("  No LIDA containers running")
            except Exception:
                pass

        print_section("Multi-User Port Assignments")
        print("  User 1: --redis-port 6379 --api-port 2040")
        print("  User 2: --redis-port 6380 --api-port 2041")
        print("  User 3: --redis-port 6381 --api-port 2042")

    elif args.subcommand == "env":
        # Print environment setup commands for shell
        print(f"# LIDA environment for Redis:{redis_port} API:{api_port}")
        print(f"export REDIS_PORT={redis_port}")
        print(f"export REDIS_URL=redis://localhost:{redis_port}")
        print(f"export API_PORT={api_port}")
        print(f"export PORT={api_port}")
        if args.scenario:
            print(f"export SCENARIO={args.scenario}")
        print()
        print("# Run: eval $(lida system env --redis-port ... --api-port ...)")

    else:
        raise CLIError("Unknown subcommand. Use: start, stop, status, env")


def cmd_deliberate(args):
    """Run a deliberation against a running LIDA instance."""
    import time
    try:
        import requests
    except ImportError:
        raise CLIError("requests library required: pip install requests")

    api_port = args.port
    topic = args.topic
    scenario = args.scenario or "quick_personas3"
    timeout = args.timeout or 0
    poll_interval = args.poll_interval or 5

    def check_health(port: int) -> bool:
        """Check if the API is healthy."""
        for endpoint in ["/health", "/api/stats", "/"]:
            try:
                resp = requests.get(f"http://localhost:{port}{endpoint}", timeout=5)
                if resp.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
        return False

    def get_topic_from_scenario(scenario_name: str) -> Optional[str]:
        """Get topic from scenario config."""
        try:
            import yaml
            scenario_file = _PROJECT_ROOT / "scenarios" / f"{scenario_name}.yaml"
            if scenario_file.exists():
                with open(scenario_file) as f:
                    config = yaml.safe_load(f)
                return config.get("simulation", {}).get("auto_start_topic")
        except Exception:
            pass
        return None

    print_header("LIDA Deliberation Client")
    print_kv("API Port", api_port)
    print_kv("Scenario", scenario)

    # Get topic
    if not topic:
        topic = get_topic_from_scenario(scenario)
    if not topic:
        raise CLIError("No topic specified. Use --topic or configure in scenario.")

    print_kv("Topic", topic[:60] + ("..." if len(topic) > 60 else ""))
    print_kv("Timeout", "infinite" if timeout == 0 else f"{timeout}s")

    # Check health
    print(f"\nChecking API on port {api_port}...")
    if not check_health(api_port):
        print("\n" + "!" * 60)
        print(f"ERROR: Cannot connect to API on port {api_port}")
        print("!" * 60)
        print("\nMake sure LIDA is running:")
        print(f"  lida system start --redis-port 6379 --api-port {api_port}")
        print(f"  lida serve --port {api_port}")
        sys.exit(1)

    print("  ✓ API is healthy\n")

    # Start deliberation
    print(f"Starting deliberation...")
    try:
        resp = requests.post(
            f"http://localhost:{api_port}/api/deliberate",
            params={"topic": topic},
            timeout=10
        )
        if resp.status_code != 200:
            raise CLIError(f"Failed to start: {resp.status_code} {resp.text}")

        result = resp.json()
        deliberation_id = result.get("deliberation_id")
        print(f"  Deliberation ID: {deliberation_id}\n")
    except requests.exceptions.RequestException as e:
        raise CLIError(f"Error starting deliberation: {e}")

    # Monitor progress
    print_section("Monitoring Progress")

    start_time = time.time()
    interrupted = False
    completed = False

    def handle_interrupt(*_):
        nonlocal interrupted
        print("\n  Interrupted!")
        interrupted = True

    signal.signal(signal.SIGINT, handle_interrupt)

    while not interrupted:
        time.sleep(poll_interval)
        elapsed = time.time() - start_time

        if timeout > 0 and elapsed > timeout:
            print(f"\n  Timeout after {timeout}s")
            break

        try:
            resp = requests.get(f"http://localhost:{api_port}/api/deliberation/status", timeout=5)
            if resp.status_code == 200:
                status = resp.json()
                phase = status.get("phase", "unknown")
                msg_count = status.get("total_messages", 0)
                active = status.get("active", False)

                print(f"  [{elapsed:6.1f}s] Phase: {phase:<20} Messages: {msg_count}")

                if not active and msg_count > 0:
                    completed = True
                    print_header("Deliberation COMPLETED")
                    print_kv("Consensus", status.get("consensus"))
                    print_kv("Messages", msg_count)
                    break
        except Exception as e:
            print(f"  [{elapsed:6.1f}s] (status check failed: {e})")

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s")
    sys.exit(0 if completed else 1)


def cmd_profile(args):
    """Manage configuration profiles."""
    from src.cli.advanced import ProfileManager, Profile

    pm = ProfileManager()

    if args.subcommand == "create":
        profile = pm.create(
            name=args.name,
            redis_port=args.redis_port,
            api_port=args.api_port,
            workers=args.workers or 4,
            live_mode=args.live,
            scenario=args.scenario,
            description=args.description or "",
        )
        print(f"Created profile: {profile.name}")
        print(f"  Redis: {profile.redis_port}, API: {profile.api_port}")

    elif args.subcommand == "list":
        profiles = pm.list()
        if not profiles:
            print("No profiles configured.")
            print("Create one: lida profile create myprofile --redis-port 6379 --api-port 2040")
            return

        print_header("Configuration Profiles")
        default = pm.get_default()
        for p in profiles:
            if p.name.startswith("_"):
                continue
            marker = "→" if default and p.name == default.name else " "
            print(f"  {marker} {p.name:15s}  Redis:{p.redis_port}  API:{p.api_port}  Workers:{p.workers}")
            if p.description:
                print(f"      {p.description}")

    elif args.subcommand == "show":
        profile = pm.get(args.name)
        if not profile:
            raise CLIError(f"Profile not found: {args.name}")

        print_header(f"Profile: {profile.name}")
        print_kv("Redis Port", profile.redis_port)
        print_kv("API Port", profile.api_port)
        print_kv("Workers", profile.workers)
        print_kv("Live Mode", profile.live_mode)
        print_kv("Scenario", profile.scenario or "(default)")
        print_kv("Created", profile.created_at)
        if profile.environment:
            print_section("Environment")
            for k, v in profile.environment.items():
                print_kv(k, v, indent=4)

    elif args.subcommand == "delete":
        if pm.delete(args.name):
            print(f"Deleted profile: {args.name}")
        else:
            raise CLIError(f"Profile not found: {args.name}")

    elif args.subcommand == "use":
        if pm.set_default(args.name):
            print(f"Default profile set to: {args.name}")
        else:
            raise CLIError(f"Profile not found: {args.name}")

    elif args.subcommand == "env":
        env_output = pm.export_env(args.name)
        if env_output:
            print(env_output)
            print()
            print(f"# Run: eval $(lida profile env {args.name})")
        else:
            raise CLIError(f"Profile not found: {args.name}")

    else:
        # Default: show list
        profiles = pm.list()
        print(f"Profiles: {len([p for p in profiles if not p.name.startswith('_')])}")
        print("Use: lida profile list")


def cmd_orchestrate(args):
    """Advanced service orchestration."""
    from src.cli.advanced import ServiceOrchestrator, ServiceConfig

    orch = ServiceOrchestrator(_PROJECT_ROOT)

    if args.subcommand == "start":
        print_header("Service Orchestrator")

        # Register services
        orch.register_default_services(args.redis_port, args.api_port)

        # Optionally add workers
        if args.workers:
            orch.register(ServiceConfig(
                name="workers",
                command=[sys.executable, str(_PROJECT_ROOT / "run_workers.py"),
                         "--num-workers", str(args.workers)],
                depends_on=["redis"],
                environment={"REDIS_URL": f"redis://localhost:{args.redis_port}"},
            ))

        # Start with dependency resolution
        print("Starting services...")
        results = orch.start()

        for name, success in results.items():
            marker = "✓" if success else "✗"
            print(f"  {marker} {name}")

        if all(results.values()):
            print_section("All Services Running")
            print(f"  API: http://localhost:{args.api_port}")
            print(f"  Redis: redis://localhost:{args.redis_port}")

            if args.monitor:
                print("\nStarting health monitor (Ctrl+C to stop)...")
                def on_state_change(name, state):
                    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {name}: {state.name}")

                from datetime import datetime
                orch.start_monitor(on_state_change)

                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\nStopping...")
                    orch.stop_monitor()
                    orch.stop()

    elif args.subcommand == "stop":
        orch.register_default_services(args.redis_port, args.api_port)
        results = orch.stop()
        for name, success in results.items():
            marker = "✓" if success else "✗"
            print(f"  {marker} {name} stopped")

    elif args.subcommand == "status":
        orch.register_default_services(args.redis_port, args.api_port)
        status = orch.status()

        print_header("Service Status")
        for name, info in status.items():
            marker = "●" if info["healthy"] else "○"
            state = info["state"]
            pid = info.get("pid", "-")
            print(f"  {marker} {name:12s}  {state:10s}  PID:{pid}")

    else:
        print("Usage: lida orchestrate start|stop|status")


def cmd_cluster(args):
    """Manage a cluster of LIDA instances."""
    from src.cli.advanced import ClusterManager

    cm = ClusterManager()

    if args.subcommand == "add":
        node = cm.add_node(
            name=args.name,
            ssh_host=args.host,
            redis_port=args.redis_port,
            api_port=args.api_port,
            user=args.user,
        )
        print(f"Added node: {node.hostname}")
        print(f"  SSH: {args.user + '@' if args.user else ''}{args.host}")
        print(f"  Ports: Redis={node.redis_port}, API={node.api_port}")

    elif args.subcommand == "remove":
        if cm.remove_node(args.name):
            print(f"Removed node: {args.name}")
        else:
            raise CLIError(f"Node not found: {args.name}")

    elif args.subcommand == "list":
        if not cm.nodes:
            print("No nodes configured.")
            print("Add one: lida cluster add mynode --host server.example.com")
            return

        print_header("Cluster Nodes")
        for name, node in cm.nodes.items():
            status_marker = {"healthy": "●", "degraded": "◐", "unknown": "○"}.get(node.status, "?")
            print(f"  {status_marker} {name:15s}  {node.ssh_host:25s}  API:{node.api_port}")

    elif args.subcommand == "status":
        print_header("Cluster Status")
        print("Checking nodes...")

        results = cm.check_status(args.name if hasattr(args, 'name') and args.name else None)
        for name, info in results.items():
            status = info.get("status", "unknown")
            marker = {"healthy": "●", "degraded": "◐", "unreachable": "✗"}.get(status, "?")
            print(f"  {marker} {name:15s}  [{status}]")
            if "error" in info:
                print(f"      Error: {info['error'][:60]}")

    elif args.subcommand == "deploy":
        print(f"Deploying to {args.name}...")
        success, output = cm.deploy(args.name, args.command or "lida system start")
        if success:
            print("  ✓ Deployment successful")
        else:
            print(f"  ✗ Deployment failed: {output[:100]}")

    elif args.subcommand == "stop-all":
        print("Stopping all nodes...")
        results = cm.stop_all()
        for name, success in results.items():
            marker = "✓" if success else "✗"
            print(f"  {marker} {name}")

    else:
        print(f"Nodes: {len(cm.nodes)}")
        print("Use: lida cluster list")


def cmd_pipeline(args):
    """Run deployment/CI pipelines."""
    from src.cli.advanced import Pipeline, PipelineStep

    pipeline = Pipeline(args.name or "default", _PROJECT_ROOT)

    if args.preset == "deploy":
        pipeline.add("lint", ["ruff", "check", "src/"], on_failure="continue")
        pipeline.add("test", ["pytest", "tests/", "-v", "--tb=short", "-x"], timeout=300)
        pipeline.add("start", [sys.executable, "-m", "src.cli.main", "system", "start"])

    elif args.preset == "test":
        pipeline.add("lint", ["ruff", "check", "src/"], on_failure="continue")
        pipeline.add("typecheck", ["mypy", "src/"], on_failure="continue")
        pipeline.add("test", ["pytest", "tests/", "-v"], timeout=600)

    elif args.preset == "ci":
        pipeline.add("install", ["pip", "install", "-e", ".[dev]"])
        pipeline.add("lint", ["ruff", "check", "src/"])
        pipeline.add("test", ["pytest", "tests/", "-v", "--tb=short"])

    elif args.steps:
        # Custom steps from command line
        for i, step in enumerate(args.steps):
            pipeline.add(f"step_{i}", step.split())

    else:
        raise CLIError("Specify --preset or --steps")

    print_header(f"Pipeline: {pipeline.name}")

    if args.dry_run:
        print("[DRY RUN]")

    success, results = pipeline.run(dry_run=args.dry_run)

    print()
    print(pipeline.summary())

    sys.exit(0 if success else 1)


def cmd_metrics(args):
    """View and export metrics."""
    from src.cli.advanced import MetricsCollector
    from datetime import timedelta

    collector = MetricsCollector(args.redis_url)

    if args.subcommand == "show":
        # Collect current metrics from system
        print_header("System Metrics")

        # Check ports
        for port, name in [(6379, "redis"), (2040, "api"), (8000, "alt_api")]:
            if is_port_in_use(port):
                collector.record(f"service.{name}.up", 1.0, {"port": str(port)})
            else:
                collector.record(f"service.{name}.up", 0.0, {"port": str(port)})

        # Display
        for name in ["service.redis.up", "service.api.up"]:
            latest = collector.get_latest(name)
            if latest:
                status = "UP" if latest.value == 1.0 else "DOWN"
                print(f"  {name}: {status}")

    elif args.subcommand == "export":
        output = collector.export_prometheus()
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Exported to {args.output}")
        else:
            print(output)

    elif args.subcommand == "watch":
        print("Watching metrics (Ctrl+C to stop)...")
        import time as time_module
        try:
            while True:
                # Collect and display
                for port, name in [(6379, "redis"), (2040, "api")]:
                    up = 1.0 if is_port_in_use(port) else 0.0
                    collector.record(f"service.{name}.up", up)

                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] ", end="")
                for name in ["redis", "api"]:
                    latest = collector.get_latest(f"service.{name}.up")
                    status = "●" if latest and latest.value == 1.0 else "○"
                    print(f"{name}:{status} ", end="")
                print("", end="", flush=True)

                time_module.sleep(args.interval or 5)
        except KeyboardInterrupt:
            print("\nStopped")

    else:
        print("Usage: lida metrics show|export|watch")


def cmd_lock(args):
    """Manage distributed locks."""
    from src.cli.advanced import DistributedLock

    lock = DistributedLock(args.redis_url or "redis://localhost:6379")

    if args.subcommand == "acquire":
        print(f"Acquiring lock: {args.name}")
        with lock.acquire(args.name, timeout=args.timeout, blocking=True) as acquired:
            if acquired:
                print("  ✓ Lock acquired")
                if args.command:
                    print(f"  Running: {args.command}")
                    result = subprocess.run(args.command, shell=True)
                    sys.exit(result.returncode)
                else:
                    print("  Press Enter to release...")
                    input()
            else:
                print("  ✗ Failed to acquire lock")
                sys.exit(1)

    elif args.subcommand == "status":
        is_locked = lock.is_locked(args.name)
        status = "LOCKED" if is_locked else "FREE"
        marker = "●" if is_locked else "○"
        print(f"  {marker} {args.name}: {status}")

    elif args.subcommand == "release":
        if lock.force_release(args.name):
            print(f"  ✓ Released: {args.name}")
        else:
            print(f"  Lock not held: {args.name}")


def cmd_chaos(args):
    """Chaos engineering - fault injection for testing resilience."""
    from src.cli.advanced_v2 import (
        get_chaos_engine, FaultConfig, FaultType, get_event_bus
    )

    chaos = get_chaos_engine()

    if args.subcommand == "enable":
        chaos.enable()
        print("Chaos engineering ENABLED")
        print("  ⚠ Faults will be injected according to configured rules")

    elif args.subcommand == "disable":
        chaos.disable()
        print("Chaos engineering DISABLED")

    elif args.subcommand == "add":
        fault_type = FaultType[args.type.upper()]
        config = FaultConfig(
            fault_type=fault_type,
            probability=args.probability,
            duration=args.duration,
            target_services=args.services.split(",") if args.services else [],
        )
        chaos.register_fault(args.name, config)
        print(f"Registered fault: {args.name}")
        print(f"  Type: {fault_type.name}, Probability: {args.probability}")

    elif args.subcommand == "status":
        status = chaos.status()
        print_header("Chaos Engine Status")
        print(f"  Active: {'YES' if status['active'] else 'NO'}")
        print()
        if status['faults']:
            print("  Registered Faults:")
            for name, info in status['faults'].items():
                marker = "●" if info['enabled'] else "○"
                print(f"    {marker} {name}: {info['type']} @ {info['probability']*100:.0f}%")
                print(f"        Injections: {info['injections']}")
        else:
            print("  No faults registered")

    elif args.subcommand == "inject":
        try:
            chaos.inject(args.name)
            print(f"Fault '{args.name}' injected manually")
        except Exception as e:
            print(f"Injection result: {e}")

    else:
        print("Usage: lida chaos enable|disable|add|status|inject")


def cmd_trace(args):
    """Distributed tracing for debugging and performance analysis."""
    from src.cli.advanced_v2 import get_tracer, Span

    tracer = get_tracer("lida-cli")

    if args.subcommand == "start":
        # Start a new trace
        span = tracer.start_span(args.operation or "cli-operation")
        print(f"Started trace: {span.trace_id}")
        print(f"Span ID: {span.span_id}")
        print(f"Operation: {span.operation_name}")

        # Store trace ID for later
        trace_file = Path.home() / ".lida" / "current_trace"
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        with open(trace_file, "w") as f:
            json.dump({"trace_id": span.trace_id, "span_id": span.span_id}, f)

    elif args.subcommand == "finish":
        trace_file = Path.home() / ".lida" / "current_trace"
        if trace_file.exists():
            with open(trace_file) as f:
                data = json.load(f)

            spans = tracer.get_trace(data["trace_id"])
            for span in spans:
                if span.span_id == data["span_id"]:
                    tracer.finish_span(span)
                    print(f"Finished span: {span.span_id}")
                    print(f"Duration: {span.duration_ms:.2f}ms")
            trace_file.unlink()
        else:
            print("No active trace")

    elif args.subcommand == "show":
        if args.trace_id:
            trace_data = tracer.export_trace(args.trace_id)
            print_header(f"Trace: {args.trace_id}")
            print(f"Service: {trace_data['service']}")
            print(f"Spans: {trace_data['span_count']}")
            print()
            for span in trace_data['spans']:
                indent = "  " if span['parent_span_id'] else ""
                duration = f"{span['duration_ms']:.2f}ms" if span['duration_ms'] else "running"
                print(f"{indent}├─ {span['operation_name']} [{span['status']}] {duration}")
        else:
            print("Specify --trace-id to show trace details")

    elif args.subcommand == "export":
        if args.trace_id:
            trace_data = tracer.export_trace(args.trace_id)
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(trace_data, f, indent=2)
                print(f"Exported to {args.output}")
            else:
                print(json.dumps(trace_data, indent=2))
        else:
            print("Specify --trace-id to export")

    else:
        print("Usage: lida trace start|finish|show|export")


def cmd_autoscale(args):
    """Auto-scaling configuration and monitoring."""
    from src.cli.advanced_v2 import AutoScaler, ScalingPolicy, ResourceMetrics

    if args.subcommand == "configure":
        policy = ScalingPolicy(
            min_instances=args.min,
            max_instances=args.max,
            scale_up_threshold=args.scale_up_threshold,
            scale_down_threshold=args.scale_down_threshold,
        )

        # Save policy
        config_file = Path.home() / ".lida" / "autoscale.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, "w") as f:
            json.dump({
                "min_instances": policy.min_instances,
                "max_instances": policy.max_instances,
                "scale_up_threshold": policy.scale_up_threshold,
                "scale_down_threshold": policy.scale_down_threshold,
            }, f, indent=2)

        print_header("Auto-Scale Configuration")
        print(f"  Min instances: {policy.min_instances}")
        print(f"  Max instances: {policy.max_instances}")
        print(f"  Scale up threshold: {policy.scale_up_threshold}%")
        print(f"  Scale down threshold: {policy.scale_down_threshold}%")

    elif args.subcommand == "status":
        scaler = AutoScaler("workers")
        status = scaler.status()
        print_header("Auto-Scaler Status")
        print(f"  Current instances: {status['current_instances']}")
        print(f"  Min/Max: {status['min_instances']}/{status['max_instances']}")
        if status['avg_cpu']:
            print(f"  Avg CPU: {status['avg_cpu']:.1f}%")
        if status['avg_memory']:
            print(f"  Avg Memory: {status['avg_memory']:.1f}%")

    elif args.subcommand == "simulate":
        # Simulate auto-scaling decisions
        scaler = AutoScaler("workers", ScalingPolicy(
            min_instances=args.min or 1,
            max_instances=args.max or 10,
        ))

        print("Simulating auto-scaling with random load...")
        for i in range(10):
            metrics = ResourceMetrics(
                cpu_percent=random.uniform(20, 90),
                memory_percent=random.uniform(30, 80),
                request_rate=random.uniform(10, 100),
                response_time_p99=random.uniform(50, 500),
                queue_depth=random.randint(0, 50),
            )
            scaler.record_metrics(metrics)

            decision = scaler.evaluate()
            if decision:
                old = scaler._current_instances
                scaler.apply_scaling(decision)
                print(f"  Tick {i+1}: CPU={metrics.cpu_percent:.0f}% → Scale {old} → {decision}")
            else:
                print(f"  Tick {i+1}: CPU={metrics.cpu_percent:.0f}% → No change")

            time.sleep(0.5)

    else:
        print("Usage: lida autoscale configure|status|simulate")


def cmd_circuit(args):
    """Circuit breaker management for fault tolerance."""
    from src.cli.advanced_v2 import get_circuit_breaker, CircuitBreakerConfig

    if args.subcommand == "status":
        # Show all circuit breakers
        from src.cli.advanced_v2 import _circuit_breakers

        print_header("Circuit Breakers")
        if not _circuit_breakers:
            print("  No circuit breakers registered")
            print("  They are created automatically when services are called")
        else:
            for name, cb in _circuit_breakers.items():
                status = cb.status()
                state = status['state']
                marker = {"CLOSED": "●", "OPEN": "○", "HALF_OPEN": "◐"}.get(state, "?")
                print(f"  {marker} {name:20s} [{state:10s}] failures={status['failure_count']}")

    elif args.subcommand == "reset":
        cb = get_circuit_breaker(args.name)
        cb._state = cb._state.__class__.CLOSED
        cb._failure_count = 0
        print(f"Reset circuit breaker: {args.name}")

    elif args.subcommand == "trip":
        cb = get_circuit_breaker(args.name)
        for _ in range(cb.config.failure_threshold):
            cb.record_failure()
        print(f"Tripped circuit breaker: {args.name} (now OPEN)")

    elif args.subcommand == "configure":
        config = CircuitBreakerConfig(
            failure_threshold=args.failure_threshold or 5,
            recovery_timeout=args.recovery_timeout or 30.0,
        )
        cb = get_circuit_breaker(args.name, config)
        print(f"Configured circuit breaker: {args.name}")
        print(f"  Failure threshold: {config.failure_threshold}")
        print(f"  Recovery timeout: {config.recovery_timeout}s")

    else:
        print("Usage: lida circuit status|reset|trip|configure")


def cmd_schedule(args):
    """Job scheduler for recurring tasks."""
    from src.cli.advanced_v2 import Scheduler

    scheduler = Scheduler()

    if args.subcommand == "add":
        def job_func():
            subprocess.run(args.command, shell=True)

        job_id = scheduler.add_job(
            name=args.name,
            func=job_func,
            interval=args.interval,
        )
        print(f"Added job: {args.name} (ID: {job_id})")
        print(f"  Interval: every {args.interval}s")
        print(f"  Command: {args.command}")

    elif args.subcommand == "list":
        status = scheduler.status()
        print_header("Scheduled Jobs")
        if not status['jobs']:
            print("  No jobs scheduled")
        else:
            for job in status['jobs']:
                enabled = "●" if job['enabled'] else "○"
                print(f"  {enabled} {job['name']:20s} [{job['schedule']}]")
                print(f"      Next: {job['next_run']}")

    elif args.subcommand == "start":
        scheduler.start()
        print("Scheduler started")
        print("Press Ctrl+C to stop...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            scheduler.stop()
            print("\nScheduler stopped")

    elif args.subcommand == "remove":
        if scheduler.remove_job(args.job_id):
            print(f"Removed job: {args.job_id}")
        else:
            print(f"Job not found: {args.job_id}")

    else:
        print("Usage: lida schedule add|list|start|remove")


def cmd_snapshot(args):
    """System state snapshot and restore."""
    from src.cli.advanced_v2 import SnapshotManager

    sm = SnapshotManager()

    if args.subcommand == "create":
        # Gather current system state
        state = {
            "timestamp": datetime.now().isoformat(),
            "environment": {
                k: v for k, v in os.environ.items()
                if k.startswith(("REDIS", "API", "LIDA", "SCENARIO"))
            },
            "ports": {
                "redis": 6379,
                "api": 2040,
            },
        }

        snapshot = sm.create(
            name=args.name,
            state=state,
            metadata={"description": args.description or ""},
        )
        print(f"Created snapshot: {snapshot.id}")
        print(f"  Name: {snapshot.name}")
        print(f"  Time: {snapshot.timestamp}")

    elif args.subcommand == "list":
        snapshots = sm.list()
        print_header("Snapshots")
        if not snapshots:
            print("  No snapshots found")
        else:
            for s in snapshots[:10]:
                print(f"  {s.id}  {s.name:20s}  {s.timestamp.strftime('%Y-%m-%d %H:%M')}")

    elif args.subcommand == "restore":
        state = sm.restore(args.snapshot_id)
        if state:
            print(f"Restored snapshot: {args.snapshot_id}")

            # Apply environment
            if "environment" in state:
                for k, v in state["environment"].items():
                    os.environ[k] = v
                    print(f"  Set {k}={v}")
        else:
            print(f"Snapshot not found: {args.snapshot_id}")

    elif args.subcommand == "delete":
        if sm.delete(args.snapshot_id):
            print(f"Deleted snapshot: {args.snapshot_id}")
        else:
            print(f"Snapshot not found: {args.snapshot_id}")

    elif args.subcommand == "show":
        snapshot = sm.get(args.snapshot_id)
        if snapshot:
            print_header(f"Snapshot: {snapshot.name}")
            print(f"  ID: {snapshot.id}")
            print(f"  Time: {snapshot.timestamp}")
            print(f"  Version: {snapshot.version}")
            print()
            print("  State:")
            print(json.dumps(snapshot.data, indent=4))
        else:
            print(f"Snapshot not found: {args.snapshot_id}")

    else:
        print("Usage: lida snapshot create|list|restore|delete|show")


def cmd_events(args):
    """Event bus monitoring and publishing."""
    from src.cli.advanced_v2 import get_event_bus, Event

    bus = get_event_bus()

    if args.subcommand == "publish":
        data = {}
        if args.data:
            try:
                data = json.loads(args.data)
            except json.JSONDecodeError:
                data = {"message": args.data}

        event = Event(
            type=args.type,
            data=data,
            source="cli",
        )
        bus.publish(event)
        print(f"Published event: {args.type}")
        print(f"  Correlation ID: {event.correlation_id}")

    elif args.subcommand == "history":
        events = bus.replay(
            event_type=args.type if args.type else None,
        )

        print_header("Event History")
        for event in events[-20:]:
            ts = event.timestamp.strftime("%H:%M:%S")
            print(f"  [{ts}] {event.type:30s}")
            if args.verbose:
                print(f"           {event.data}")

    elif args.subcommand == "watch":
        print("Watching events (Ctrl+C to stop)...")

        def handler(event):
            ts = event.timestamp.strftime("%H:%M:%S")
            print(f"[{ts}] {event.type}: {event.data}")

        unsubscribe = bus.subscribe("*", handler)

        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            unsubscribe()
            print("\nStopped watching")

    else:
        print("Usage: lida events publish|history|watch")


def cmd_health(args):
    """Health checking and monitoring."""
    from src.cli.advanced_v2 import (
        HealthChecker, redis_health_check, api_health_check, disk_health_check
    )

    checker = HealthChecker()
    checker.register("redis", lambda: redis_health_check(args.redis_port or 6379))
    checker.register("api", lambda: api_health_check(args.api_port or 2040))
    checker.register("disk", lambda: disk_health_check("/"))

    if args.subcommand == "check":
        results = checker.check(args.name if hasattr(args, 'name') and args.name else None)

        print_header("Health Check Results")
        for name, result in results.items():
            marker = {"HEALTHY": "●", "DEGRADED": "◐", "UNHEALTHY": "○"}[result.status.name]
            print(f"  {marker} {name:15s} [{result.status.name:10s}] {result.message}")

        # Exit code
        sys.exit(0 if checker.is_healthy() else 1)

    elif args.subcommand == "watch":
        print("Watching health (Ctrl+C to stop)...")
        try:
            while True:
                results = checker.check()
                status_line = " ".join(
                    f"{name}:{'●' if r.status.name == 'HEALTHY' else '○'}"
                    for name, r in results.items()
                )
                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] {status_line}", end="", flush=True)
                time.sleep(args.interval or 5)
        except KeyboardInterrupt:
            print("\nStopped")

    elif args.subcommand == "json":
        summary = checker.summary()
        print(json.dumps(summary, indent=2))

    else:
        # Default: run check
        results = checker.check()
        all_healthy = all(r.status.name == "HEALTHY" for r in results.values())
        print("Health:", "OK" if all_healthy else "DEGRADED")


def cmd_dashboard(args):
    """Launch the web dashboard."""
    if args.tui:
        # Terminal UI dashboard
        from src.cli.advanced_v2 import (
            TerminalDashboard, create_events_panel, get_event_bus
        )
        from src.cli.advanced import MetricsCollector

        dashboard = TerminalDashboard()
        dashboard.add_panel(create_events_panel(get_event_bus()))

        print("Starting TUI dashboard...")
        dashboard.run()
        return

    # Original web dashboard
    import webbrowser

    port = args.port or 2040
    url = f"http://localhost:{port}"

    print(f"Opening dashboard at {url}")

    # Start server in background if not running
    if not args.no_server:
        subprocess.Popen(
            [sys.executable, "server.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)

    webbrowser.open(url)


def cmd_story(args):
    """Run multi-agent collaborative storytelling."""
    from src.storytelling.engine import (
        StorytellingEngine, StoryConfig, StoryMode, StoryGenre,
        StoryCharacter, StoryBeat,
    )
    from src.storytelling.collaborative import (
        CollaborativeStory, CollaborativeStoryConfig,
    )

    # List genres
    if args.list_genres:
        print_header("Available Story Genres")
        genres = [
            ("drama", "Serious character-driven narratives with emotional depth"),
            ("comedy", "Light-hearted stories with humor and wit"),
            ("thriller", "Tense, suspenseful narratives with high stakes"),
            ("mystery", "Stories centered on solving puzzles or crimes"),
            ("romance", "Stories focused on love and relationships"),
            ("scifi", "Speculative fiction exploring technology and the future"),
            ("fantasy", "Stories with magical or supernatural elements"),
            ("horror", "Dark, frightening narratives designed to unsettle"),
            ("political", "Stories exploring power, governance, and intrigue"),
            ("philosophical", "Narratives exploring ideas and meaning"),
            ("satire", "Stories using humor to critique society"),
            ("adventure", "Action-packed journeys and quests"),
            ("historical", "Stories set in past time periods"),
            ("documentary", "Fact-based narrative storytelling"),
        ]
        for genre, desc in genres:
            print(f"  {genre:14} - {desc}")
        print()
        print_section("Storytelling Modes")
        modes = [
            ("collaborative", "Multiple agents build story together"),
            ("round_robin", "Agents take turns contributing"),
            ("director_led", "Director agent guides the narrative"),
            ("debate_style", "Characters argue and debate"),
            ("improv", "Free-form improvisation"),
            ("structured", "Following classic story beats"),
            ("emergent", "Story emerges from agent interactions"),
        ]
        for mode, desc in modes:
            print(f"  {mode:14} - {desc}")
        return

    # Run demo
    if args.demo:
        print_header("Collaborative Storytelling Demo")
        print()

        async def run_demo():
            from src.storytelling.collaborative import demo_collaborative
            await demo_collaborative()

        run_async(run_demo)
        return

    # Create story configuration
    print_header(f"Creating Story: {args.title}")
    print_kv("Genre", args.genre)
    print_kv("Mode", args.mode)
    print_kv("Characters", args.characters)
    print_kv("Rounds", args.rounds)
    if args.setting:
        print_kv("Setting", args.setting)
    if args.themes:
        print_kv("Themes", ", ".join(args.themes))
    print()

    # Build configuration
    story_config = StoryConfig(
        title=args.title,
        genre=StoryGenre(args.genre),
        mode=StoryMode(args.mode),
        num_character_agents=args.characters,
        target_length=args.rounds,
        max_rounds=args.rounds * 2,
        include_narrator=not args.no_narrator,
        include_director=not args.no_director,
        setting=args.setting or "",
        themes=args.themes or [],
        stream_output=args.stream,
    )

    collab_config = CollaborativeStoryConfig(
        story_config=story_config,
        max_rounds=args.rounds,
        enable_critique=args.with_critic,
        stream_contributions=args.stream,
    )

    # Create and run story
    story = CollaborativeStory(collab_config)

    async def run_story():
        print_section("Initializing Story")
        session = story.initialize()
        print(f"Session ID: {session.id}")
        print(f"Agents: {len(story.agents)}")
        print()

        if args.stream:
            print_section("Story Begins")
            print("-" * 60)
            print()

            async for event in story.stream_story():
                if event["type"] == "contribution":
                    data = event["data"]
                    agent = data.get("agent", "Unknown")
                    content = data.get("content", "")
                    char = data.get("character")

                    if char:
                        print(f"[{agent} as {char}]")
                    else:
                        print(f"[{agent}]")
                    print(content)
                    print()

                elif event["type"] == "round_complete":
                    data = event["data"]
                    print(f"--- Round {data['round']} ({data['phase']}) ---")
                    print()

                elif event["type"] == "complete":
                    print()
                    print("=" * 60)
                    print("STORY COMPLETE")
                    print("=" * 60)
                    analysis = data
                    if "summary" in analysis:
                        summary = analysis["summary"]
                        print(f"Words: {summary.get('words', 0)}")
                        print(f"Beats: {summary.get('beats', 0)}")
        else:
            print_section("Generating Story...")
            state = await story.run()
            print(f"Completed: {len(state.beats)} beats, {state.word_count} words")

        # Output final story
        if args.output:
            output_text = story.get_story_text() if args.format == "text" else story.get_story_json()
            with open(args.output, "w") as f:
                f.write(output_text)
            print(f"Story saved to: {args.output}")
        else:
            print()
            print_section("Final Story")
            if args.format == "text":
                print(story.get_story_text())
            else:
                print(story.get_story_json())

    run_async(run_story)


def cmd_version(args):
    """Show version information."""
    from src.data.manager import DataManager

    manager = DataManager()
    exp_summary = manager.get_experiment_summary()
    persona_summary = manager.get_persona_summary()

    print("LIDA Multi-Agent Research Platform")
    print("Version: 0.1.0")
    print()
    print("Components:")
    print("  - CLI Runner")
    print("  - Data Manager")
    print("  - Causal Engine")
    print("  - Counterfactual Analysis")
    print("  - Mechanism Discovery")
    print("  - Paper Export")
    print("  - Policy Simulation Engine")
    print("  - Multi-Agent Quorum")
    print("  - AI Safety Debate")
    print()
    print("Data:")
    print(f"  - {exp_summary.get('count', 0)} experiments")
    print(f"  - {persona_summary.get('total_count', 0)} personas")
    print(f"  - {len(manager.list_scenarios())} scenarios")


def main():
    parser = argparse.ArgumentParser(
        description="LIDA - Multi-Agent Research Platform CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # System management (for multi-user clusters like Mila)
  lida system start --redis-port 6379 --api-port 2040    Start LIDA stack
  lida system start --redis-port 6380 --api-port 2041    User 2's instance
  lida system status                                      Check port usage
  lida system stop --redis-port 6379 --api-port 2040     Stop instance

  # Run deliberations against a running instance
  lida deliberate --port 2040 --topic "AI regulation"
  lida deliberate --port 2041 --scenario quick_personas3

  # Standalone commands
  lida run ai_xrisk                    Run the AI x-risk debate scenario
  lida run ai_xrisk --no-live          Run without LLM calls (simulation)
  lida simulate chip_war --ticks 20    Run chip war policy simulation
  lida quorum --event "AGI announced"  Run multi-agent quorum deliberation
  lida debate --interactive            Interactive AI safety debate
  lida demo --type streaming           Run streaming demo
  lida chat sam_altman elon_musk       Two personas conversation
  lida serve --port 8080               Start API server on port 8080
  lida serve --port 8080 --live        Start with live LLM mode
  lida workers --count 8               Run worker pool
  lida list --scenarios                List available scenarios
        """
    )

    parser.add_argument("-v", "--version", action="store_true", help="Show version")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run command
    run_parser = subparsers.add_parser("run", help="Run a debate scenario")
    run_parser.add_argument("scenario", help="Scenario name or path")
    run_parser.add_argument("--topic", help="Override debate topic")
    run_parser.add_argument("--rounds", type=int, help="Max rounds")
    run_parser.add_argument("--timeout", type=int, help="Timeout in seconds")
    run_parser.add_argument("--no-live", action="store_true", help="Disable LLM calls")
    run_parser.add_argument("--output", help="Output directory")
    run_parser.set_defaults(func=cmd_run)

    # simulate command (policy simulation)
    sim_parser = subparsers.add_parser("simulate", help="Run policy simulation")
    sim_parser.add_argument("scenario", nargs="?", default="chip_war",
                           help="Scenario: chip_war, agi_crisis, negotiation")
    sim_parser.add_argument("--ticks", type=int, default=10, help="Number of simulation ticks")
    sim_parser.add_argument("--agent-a", help="First agent for negotiation")
    sim_parser.add_argument("--agent-b", help="Second agent for negotiation")
    sim_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    sim_parser.set_defaults(func=cmd_simulate)

    # quorum command
    quorum_parser = subparsers.add_parser(
        "quorum",
        help="Run multi-agent quorum deliberation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  realtime   Real-time quorum with simulated industrial events
  gdelt      Live GDELT news feed (updates every 15 min)
  mlx        MLX streaming backend (Apple Silicon)
  openrouter OpenRouter API backend
  advanced   Advanced quorum with full configuration

Examples:
  lida quorum --preset realtime --duration 120
  lida quorum --preset gdelt --cycles 10 --watch nvidia,openai
  lida quorum --preset mlx --event "AI breakthrough announced"
        """
    )
    quorum_parser.add_argument("--preset", "-p", choices=["realtime", "gdelt", "mlx", "openrouter", "advanced"],
                               default="realtime", help="Quorum preset to run")
    quorum_parser.add_argument("--event", "-e", help="Event headline to analyze")
    quorum_parser.add_argument("--backend", choices=["openrouter", "mlx"], default="openrouter",
                              help="LLM backend to use")
    quorum_parser.add_argument("--duration", type=int, default=60, help="Duration in seconds (realtime preset)")
    quorum_parser.add_argument("--cycles", type=int, default=5, help="Number of update cycles (gdelt preset)")
    quorum_parser.add_argument("--watch", help="Comma-separated companies to watch (gdelt preset)")
    quorum_parser.add_argument("--test", action="store_true", help="Quick test mode (gdelt preset)")
    quorum_parser.set_defaults(func=cmd_quorum)

    # debate command
    debate_parser = subparsers.add_parser(
        "debate",
        help="Run AI safety debate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Topics: ai_pause, lab_self_regulation, xrisk_vs_present_harms, scaling_hypothesis, open_source_ai, government_regulation

Matchups: doom_vs_accel, labs_debate, academics_clash, ethics_vs_scale, full_panel

Examples:
  lida debate --list                           List all topics and matchups
  lida debate --matchup doom_vs_accel --auto   Run doomers vs accelerationists
  lida debate --topic ai_pause --rounds 6      Run specific topic
  lida debate --interactive                    Interactive menu
  lida debate --matchup full_panel --model anthropic/claude-opus-4
        """
    )
    debate_parser.add_argument("--topic", "-t", help="Debate topic ID (ai_pause, lab_self_regulation, etc.)")
    debate_parser.add_argument("--matchup", "-m", help="Predefined matchup (doom_vs_accel, labs_debate, etc.)")
    debate_parser.add_argument("--scenario", "-s", help="Scenario file path")
    debate_parser.add_argument("--participants", help="Comma-separated participant IDs")
    debate_parser.add_argument("--rounds", "-r", type=int, default=5, help="Number of debate rounds")
    debate_parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    debate_parser.add_argument("--auto", "-a", action="store_true", help="Auto-run without prompts")
    debate_parser.add_argument("--no-llm", action="store_true", help="Run without LLM calls")
    debate_parser.add_argument("--provider", choices=["openrouter", "anthropic", "openai"], help="LLM provider")
    debate_parser.add_argument("--model", help="Model ID (e.g., anthropic/claude-sonnet-4)")
    debate_parser.add_argument("--list", action="store_true", help="List available topics and matchups")
    debate_parser.set_defaults(func=cmd_debate)

    # demo command
    demo_parser = subparsers.add_parser(
        "demo",
        help="Run demonstrations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Demo types:
  quick       Quick streaming demo (default)
  live        Live demo with GDELT + streaming personalities
  streaming   Streaming demo with real-time output
  swarm       Live swarm behavior demo
  persuasion  Persuasion experiment demo
  hyperdash   Hyperdimensional dashboard demo

Examples:
  lida demo                     Quick demo
  lida demo --type live         Live GDELT demo
  lida demo --type swarm        Swarm behavior
        """
    )
    demo_parser.add_argument("--type", "-t",
                            choices=["live", "streaming", "quick", "swarm", "persuasion", "hyperdash"],
                            default="quick", help="Demo type to run")
    demo_parser.set_defaults(func=cmd_demo)

    # logs command
    logs_parser = subparsers.add_parser("logs", help="Interactive log viewer")
    logs_parser.add_argument("logfile", help="Path to .llm_logs.json file")
    logs_parser.set_defaults(func=cmd_logs)

    # workers command
    workers_parser = subparsers.add_parser("workers", help="Run worker pool for background tasks")
    workers_parser.add_argument("--count", "-n", type=int, default=4, help="Number of workers (default: 4)")
    workers_parser.add_argument("--redis-url", help="Redis URL (default: redis://localhost:6379)")
    workers_parser.add_argument("--capacity", type=int, default=5, help="Tasks per worker (default: 5)")
    workers_parser.add_argument("--work-types", help="Comma-separated work types (default: general,compute,io,analysis,llm)")
    workers_parser.set_defaults(func=cmd_workers)

    # chat command
    chat_parser = subparsers.add_parser("chat", help="Two personas conversation")
    chat_parser.add_argument("persona1", help="First persona key")
    chat_parser.add_argument("persona2", help="Second persona key")
    chat_parser.add_argument("--topic", "-t", help="Conversation topic")
    chat_parser.add_argument("--turns", type=int, default=5, help="Number of turns")
    chat_parser.set_defaults(func=cmd_chat)

    # wargame command
    wargame_parser = subparsers.add_parser("wargame", help="Run AI policy wargame simulation")
    wargame_parser.add_argument("--topic", "-t", help="Wargame topic/scenario")
    wargame_parser.add_argument("--personas", "-p", nargs="+",
                                help="Persona IDs (e.g., eliezer-yudkowsky marc-andreessen)")
    wargame_parser.add_argument("--rounds", "-r", type=int, default=5, help="Number of rounds")
    wargame_parser.add_argument("--no-live", action="store_true", help="Disable LLM calls (simulation only)")
    wargame_parser.add_argument("--list-personas", action="store_true", help="List available personas")
    wargame_parser.set_defaults(func=cmd_wargame)

    # experiment command
    exp_parser = subparsers.add_parser("experiment", help="Run full experiment")
    exp_parser.add_argument("config", nargs="?", help="Experiment config file")
    exp_parser.add_argument("--batch", nargs="+", help="Run multiple configs")
    exp_parser.add_argument("--parallel", type=int, default=1, help="Parallel experiments")
    exp_parser.add_argument("--export-latex", action="store_true", help="Export to LaTeX")
    exp_parser.set_defaults(func=cmd_experiment)

    # sweep command
    sweep_parser = subparsers.add_parser("sweep", help="Run parameter sweeps in parallel")
    from src.cli import sweep as sweep_cli

    sweep_cli.build_arg_parser(sweep_parser)
    sweep_parser.set_defaults(func=cmd_sweep)

    # analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze results")
    analyze_parser.add_argument("results", help="Results JSON file")
    analyze_parser.add_argument("--causal", action="store_true", help="Run causal analysis")
    analyze_parser.add_argument("--mechanism", action="store_true", help="Run mechanism discovery")
    analyze_parser.add_argument("--counterfactual", action="store_true", help="Run counterfactual analysis")
    analyze_parser.add_argument("--all", action="store_true", help="Run all analyses")
    analyze_parser.set_defaults(func=cmd_analyze)

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=2040, help="Port (default: 2040)")
    serve_parser.add_argument("--workers", type=int, default=1, help="Uvicorn worker processes")
    serve_parser.add_argument("--reload", action="store_true", help="Auto-reload on changes")
    serve_parser.add_argument("--scenario", "-s", help="Scenario to load")
    serve_parser.add_argument("--live", action="store_true", help="Enable live LLM mode")
    serve_parser.add_argument("--redis-url", help="Redis URL for messaging")
    serve_parser.add_argument("--agents", "-a", type=int, help="Number of agents")
    serve_parser.add_argument("--advanced", action="store_true", help="Use advanced swarm server")
    serve_parser.add_argument("--simple", action="store_true", help="Use simple server (no swarm)")
    serve_parser.set_defaults(func=cmd_serve)

    # dashboard command
    dash_parser = subparsers.add_parser("dashboard", help="Launch web or TUI dashboard")
    dash_parser.add_argument("--port", type=int, help="Server port")
    dash_parser.add_argument("--no-server", action="store_true", help="Don't start server")
    dash_parser.add_argument("--tui", action="store_true", help="Use terminal UI instead of web")
    dash_parser.set_defaults(func=cmd_dashboard)

    # system command - for multi-user cluster setups (Mila, etc.)
    sys_parser = subparsers.add_parser(
        "system",
        help="Manage LIDA system services with port configuration (for multi-user clusters)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (multi-user on Mila cluster):
  # User 1: Start with default ports
  lida system start --redis-port 6379 --api-port 2040

  # User 2: Start with different ports
  lida system start --redis-port 6380 --api-port 2041

  # Check what's running
  lida system status

  # Stop your instance
  lida system stop --redis-port 6379 --api-port 2040

  # Get environment variables for shell
  lida system env --redis-port 6379 --api-port 2040
        """
    )
    sys_subparsers = sys_parser.add_subparsers(dest="subcommand")

    # system start
    sys_start = sys_subparsers.add_parser("start", help="Start LIDA services")
    sys_start.add_argument("--redis-port", type=int, default=6379, help="Redis port (default: 6379)")
    sys_start.add_argument("--api-port", type=int, default=2040, help="API server port (default: 2040)")
    sys_start.add_argument("--workers", "-w", type=int, help="Number of worker processes")
    sys_start.add_argument("--worker-replicas", type=int, default=1, help="Worker container replicas (Docker)")
    sys_start.add_argument("--agents", "-a", type=int, help="Number of agents")
    sys_start.add_argument("--scenario", "-s", help="Scenario to load")
    sys_start.add_argument("--live", action="store_true", help="Enable live LLM mode")
    sys_start.add_argument("--docker", action="store_true", help="Use Docker (default: native)")
    sys_start.add_argument("--force", "-f", action="store_true", help="Start even if ports in use")

    # system stop
    sys_stop = sys_subparsers.add_parser("stop", help="Stop LIDA services")
    sys_stop.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    sys_stop.add_argument("--api-port", type=int, default=2040, help="API server port")
    sys_stop.add_argument("--docker", action="store_true", help="Stop Docker containers")
    sys_stop.add_argument("--include-redis", action="store_true", help="Also stop Redis")

    # system status
    sys_status = sys_subparsers.add_parser("status", help="Show system status and port usage")
    sys_status.add_argument("--redis-port", type=int, default=6379, help="Redis port (for reference)")
    sys_status.add_argument("--api-port", type=int, default=2040, help="API port (for reference)")
    sys_status.add_argument("--docker", action="store_true", help="Check Docker containers")

    # system env
    sys_env = sys_subparsers.add_parser("env", help="Print environment variables for shell")
    sys_env.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    sys_env.add_argument("--api-port", type=int, default=2040, help="API server port")
    sys_env.add_argument("--scenario", "-s", help="Scenario name")

    sys_parser.set_defaults(func=cmd_system, subcommand="status", redis_port=6379, api_port=2040, docker=False)

    # deliberate command - run deliberation against running instance
    delib_parser = subparsers.add_parser(
        "deliberate",
        help="Run a deliberation against a running LIDA instance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start the system first
  lida system start --redis-port 6379 --api-port 2040

  # Then run a deliberation
  lida deliberate --port 2040 --topic "Should AI development be regulated?"
  lida deliberate --port 2040 --scenario quick_personas3
  lida deliberate --port 2041 --topic "AI safety" --timeout 300
        """
    )
    delib_parser.add_argument("--port", "-p", type=int, required=True, help="API port of running LIDA instance")
    delib_parser.add_argument("--topic", "-t", help="Deliberation topic")
    delib_parser.add_argument("--scenario", "-s", default="quick_personas3", help="Scenario name")
    delib_parser.add_argument("--timeout", type=int, default=0, help="Timeout in seconds (0=infinite)")
    delib_parser.add_argument("--poll-interval", type=int, default=5, help="Status poll interval")
    delib_parser.set_defaults(func=cmd_deliberate)

    # export command
    export_parser = subparsers.add_parser("export", help="Export results")
    export_parser.add_argument("results", help="Results JSON file")
    export_parser.add_argument("--output", help="Output directory")
    export_parser.add_argument("--format", choices=["pdf", "png", "svg"], default="pdf")
    export_parser.add_argument("--no-appendix", action="store_true", help="Skip appendix")
    export_parser.set_defaults(func=cmd_export)

    # status command
    status_parser = subparsers.add_parser("status", help="Show experiment status")
    status_parser.set_defaults(func=cmd_status)

    # list command
    list_parser = subparsers.add_parser("list", help="List scenarios/personas/experiments")
    list_parser.add_argument("--scenarios", action="store_true", help="List scenarios only")
    list_parser.add_argument("--personas", action="store_true", help="List personas only")
    list_parser.add_argument("--experiments", action="store_true", help="List experiments only")
    list_parser.add_argument("-v", "--verbose", action="store_true", help="Show more details")
    list_parser.set_defaults(func=cmd_list)

    # data command
    data_parser = subparsers.add_parser("data", help="Manage and explore experiment data")
    data_subparsers = data_parser.add_subparsers(dest="subcommand")

    # data summary
    data_summary = data_subparsers.add_parser("summary", help="Show data summary")

    # data experiment <name>
    data_exp = data_subparsers.add_parser("experiment", help="Show experiment details")
    data_exp.add_argument("name", nargs="?", help="Experiment name or file")

    # data persona <name>
    data_persona = data_subparsers.add_parser("persona", help="Show persona details")
    data_persona.add_argument("name", nargs="?", help="Persona ID")

    # data history <participant>
    data_history = data_subparsers.add_parser("history", help="Show participant history")
    data_history.add_argument("name", nargs="?", help="Participant ID")

    # data export
    data_export = data_subparsers.add_parser("export", help="Export all data")
    data_export.add_argument("--output", help="Output file")

    data_parser.set_defaults(func=cmd_data, subcommand="summary")

    # fork command
    fork_parser = subparsers.add_parser("fork", help="Fork personas with modifications")
    fork_subparsers = fork_parser.add_subparsers(dest="subcommand")

    # fork create
    fork_create = fork_subparsers.add_parser("create", help="Create a new fork")
    fork_create.add_argument("parent", nargs="?", help="Parent persona ID")
    fork_create.add_argument("--name", help="Fork name")
    fork_create.add_argument("--description", help="Fork description")
    fork_create.add_argument("--model", help="Model to assign")
    fork_create.add_argument("--tags", help="Comma-separated tags")
    fork_create.add_argument("-m", "--modify", action="append", help="Modification (key=value)")

    # fork list
    fork_list = fork_subparsers.add_parser("list", help="List forks")
    fork_list.add_argument("--parent", help="Filter by parent persona")

    # fork show
    fork_show = fork_subparsers.add_parser("show", help="Show fork details")
    fork_show.add_argument("fork_id", nargs="?", help="Fork ID")

    # fork delete
    fork_delete = fork_subparsers.add_parser("delete", help="Delete a fork")
    fork_delete.add_argument("fork_id", nargs="?", help="Fork ID")
    fork_delete.add_argument("-f", "--force", action="store_true", help="Skip confirmation")

    fork_parser.set_defaults(func=cmd_fork, subcommand="list")

    # model command
    model_parser = subparsers.add_parser("model", help="Manage model assignments")
    model_subparsers = model_parser.add_subparsers(dest="subcommand")

    # model assign
    model_assign = model_subparsers.add_parser("assign", help="Assign model to persona")
    model_assign.add_argument("persona", nargs="?", help="Persona ID")
    model_assign.add_argument("model", nargs="?", help="Model ID")
    model_assign.add_argument("--reason", help="Reason for assignment")
    model_assign.add_argument("--temperature", type=float, help="Temperature")
    model_assign.add_argument("--max-tokens", type=int, help="Max tokens")
    model_assign.add_argument("-f", "--force", action="store_true", help="Allow unknown models")

    # model get
    model_get = model_subparsers.add_parser("get", help="Get model for persona")
    model_get.add_argument("persona", nargs="?", help="Persona ID")

    # model list
    model_list = model_subparsers.add_parser("list", help="List assignments and models")

    # model remove
    model_remove = model_subparsers.add_parser("remove", help="Remove assignment")
    model_remove.add_argument("persona", nargs="?", help="Persona ID")

    model_parser.set_defaults(func=cmd_model, subcommand="list")

    # aggregate command - aggregation analysis tools
    agg_parser = subparsers.add_parser("aggregate", help="Aggregation analysis and benchmarking tools")
    agg_subparsers = agg_parser.add_subparsers(dest="subcommand")

    # aggregate benchmark
    agg_benchmark = agg_subparsers.add_parser("benchmark", help="Benchmark all aggregation strategies")
    agg_benchmark.add_argument("--quick", "-q", action="store_true",
                               help="Quick benchmark with subset of strategies")
    agg_benchmark.add_argument("--iterations", "-n", type=int, default=5,
                               help="Number of benchmark iterations")
    agg_benchmark.add_argument("--export", "-e", help="Export results to JSON file")

    # aggregate demo
    agg_demo = agg_subparsers.add_parser("demo", help="Run demonstration of intelligent aggregation")

    # aggregate analyze
    agg_analyze = agg_subparsers.add_parser("analyze", help="Analyze confidence values")
    agg_analyze.add_argument("--file", "-f", help="JSON file with confidence data")
    agg_analyze.add_argument("--values", "-v", nargs="+", type=float,
                             help="Space-separated confidence values (e.g., 0.8 0.72 0.91)")
    agg_analyze.add_argument("--all-strategies", "-a", action="store_true",
                             help="Show results from all strategies")
    agg_analyze.add_argument("--detailed", "-d", action="store_true",
                             help="Show detailed strategy breakdown")

    agg_parser.set_defaults(func=cmd_aggregate, subcommand=None)

    # ==========================================================================
    # ADVANCED COMMANDS
    # ==========================================================================

    # profile command - manage configuration profiles
    profile_parser = subparsers.add_parser(
        "profile",
        help="Manage configuration profiles for different environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  lida profile create dev --redis-port 6379 --api-port 2040
  lida profile create prod --redis-port 6380 --api-port 2041 --workers 8
  lida profile list
  lida profile use dev
  eval $(lida profile env dev)
        """
    )
    profile_subs = profile_parser.add_subparsers(dest="subcommand")

    # profile create
    prof_create = profile_subs.add_parser("create", help="Create a new profile")
    prof_create.add_argument("name", help="Profile name")
    prof_create.add_argument("--redis-port", type=int, default=6379)
    prof_create.add_argument("--api-port", type=int, default=2040)
    prof_create.add_argument("--workers", type=int)
    prof_create.add_argument("--live", action="store_true")
    prof_create.add_argument("--scenario", "-s")
    prof_create.add_argument("--description", "-d")

    # profile list
    profile_subs.add_parser("list", help="List all profiles")

    # profile show
    prof_show = profile_subs.add_parser("show", help="Show profile details")
    prof_show.add_argument("name", help="Profile name")

    # profile delete
    prof_delete = profile_subs.add_parser("delete", help="Delete a profile")
    prof_delete.add_argument("name", help="Profile name")

    # profile use
    prof_use = profile_subs.add_parser("use", help="Set default profile")
    prof_use.add_argument("name", help="Profile name")

    # profile env
    prof_env = profile_subs.add_parser("env", help="Export profile as shell environment")
    prof_env.add_argument("name", help="Profile name")

    profile_parser.set_defaults(func=cmd_profile, subcommand="list")

    # orchestrate command - advanced service orchestration
    orch_parser = subparsers.add_parser(
        "orchestrate",
        help="Advanced service orchestration with dependency management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Features:
  - Automatic dependency resolution (starts Redis before API)
  - Health monitoring with auto-restart
  - Graceful shutdown in reverse order

Examples:
  lida orchestrate start --redis-port 6379 --api-port 2040
  lida orchestrate start --monitor         # Start with health monitoring
  lida orchestrate status
  lida orchestrate stop
        """
    )
    orch_subs = orch_parser.add_subparsers(dest="subcommand")

    # orchestrate start
    orch_start = orch_subs.add_parser("start", help="Start services with orchestration")
    orch_start.add_argument("--redis-port", type=int, default=6379)
    orch_start.add_argument("--api-port", type=int, default=2040)
    orch_start.add_argument("--workers", type=int)
    orch_start.add_argument("--monitor", "-m", action="store_true", help="Enable health monitoring")

    # orchestrate stop
    orch_stop = orch_subs.add_parser("stop", help="Stop all services")
    orch_stop.add_argument("--redis-port", type=int, default=6379)
    orch_stop.add_argument("--api-port", type=int, default=2040)

    # orchestrate status
    orch_status = orch_subs.add_parser("status", help="Show service status")
    orch_status.add_argument("--redis-port", type=int, default=6379)
    orch_status.add_argument("--api-port", type=int, default=2040)

    orch_parser.set_defaults(func=cmd_orchestrate, subcommand="status", redis_port=6379, api_port=2040)

    # cluster command - manage remote cluster nodes
    cluster_parser = subparsers.add_parser(
        "cluster",
        help="Manage a cluster of LIDA instances across multiple machines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  lida cluster add node1 --host server1.mila.quebec --user arthur
  lida cluster add node2 --host server2.mila.quebec --redis-port 6380 --api-port 2041
  lida cluster list
  lida cluster status
  lida cluster deploy node1
  lida cluster stop-all
        """
    )
    cluster_subs = cluster_parser.add_subparsers(dest="subcommand")

    # cluster add
    cl_add = cluster_subs.add_parser("add", help="Add a node to the cluster")
    cl_add.add_argument("name", help="Node name")
    cl_add.add_argument("--host", required=True, help="SSH hostname")
    cl_add.add_argument("--user", "-u", help="SSH username")
    cl_add.add_argument("--redis-port", type=int, default=6379)
    cl_add.add_argument("--api-port", type=int, default=2040)

    # cluster remove
    cl_remove = cluster_subs.add_parser("remove", help="Remove a node")
    cl_remove.add_argument("name", help="Node name")

    # cluster list
    cluster_subs.add_parser("list", help="List all nodes")

    # cluster status
    cl_status = cluster_subs.add_parser("status", help="Check cluster health")
    cl_status.add_argument("name", nargs="?", help="Specific node to check")

    # cluster deploy
    cl_deploy = cluster_subs.add_parser("deploy", help="Deploy LIDA to a node")
    cl_deploy.add_argument("name", help="Node name")
    cl_deploy.add_argument("--command", "-c", help="Custom deploy command")

    # cluster stop-all
    cluster_subs.add_parser("stop-all", help="Stop LIDA on all nodes")

    cluster_parser.set_defaults(func=cmd_cluster, subcommand="list")

    # pipeline command - run deployment/CI pipelines
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run deployment and CI/CD pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  deploy   Lint, test, then start services
  test     Full test suite with linting and type checking
  ci       CI pipeline (install, lint, test)

Examples:
  lida pipeline --preset test
  lida pipeline --preset deploy --dry-run
  lida pipeline --steps "pytest tests/" "lida system start"
        """
    )
    pipeline_parser.add_argument("--name", "-n", help="Pipeline name")
    pipeline_parser.add_argument("--preset", "-p", choices=["deploy", "test", "ci"])
    pipeline_parser.add_argument("--steps", nargs="+", help="Custom pipeline steps")
    pipeline_parser.add_argument("--dry-run", action="store_true", help="Show steps without running")
    pipeline_parser.set_defaults(func=cmd_pipeline)

    # metrics command - view and export metrics
    metrics_parser = subparsers.add_parser(
        "metrics",
        help="View and export system metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  lida metrics show                    Show current metrics
  lida metrics watch --interval 5      Watch metrics live
  lida metrics export --output m.prom  Export Prometheus format
        """
    )
    metrics_subs = metrics_parser.add_subparsers(dest="subcommand")

    # metrics show
    metrics_subs.add_parser("show", help="Show current metrics")

    # metrics watch
    m_watch = metrics_subs.add_parser("watch", help="Watch metrics in real-time")
    m_watch.add_argument("--interval", "-i", type=int, default=5, help="Update interval")

    # metrics export
    m_export = metrics_subs.add_parser("export", help="Export metrics")
    m_export.add_argument("--output", "-o", help="Output file")

    metrics_parser.add_argument("--redis-url", help="Redis URL for distributed metrics")
    metrics_parser.set_defaults(func=cmd_metrics, subcommand="show", redis_url=None)

    # lock command - distributed locking
    lock_parser = subparsers.add_parser(
        "lock",
        help="Distributed locking for cluster coordination",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  lida lock acquire deploy-lock                     Acquire and hold until Enter
  lida lock acquire deploy-lock --command "..."    Acquire, run command, release
  lida lock status deploy-lock
  lida lock release deploy-lock                    Force release (admin)
        """
    )
    lock_subs = lock_parser.add_subparsers(dest="subcommand")

    # lock acquire
    l_acquire = lock_subs.add_parser("acquire", help="Acquire a distributed lock")
    l_acquire.add_argument("name", help="Lock name")
    l_acquire.add_argument("--timeout", type=float, default=30.0, help="Timeout in seconds")
    l_acquire.add_argument("--command", "-c", help="Command to run while holding lock")

    # lock status
    l_status = lock_subs.add_parser("status", help="Check lock status")
    l_status.add_argument("name", help="Lock name")

    # lock release
    l_release = lock_subs.add_parser("release", help="Force release a lock")
    l_release.add_argument("name", help="Lock name")

    lock_parser.add_argument("--redis-url", help="Redis URL")
    lock_parser.set_defaults(func=cmd_lock, subcommand="status", redis_url=None, name="default")

    # ==========================================================================
    # ADVANCED V2 COMMANDS - Enterprise Features
    # ==========================================================================

    # chaos command - fault injection
    chaos_parser = subparsers.add_parser(
        "chaos",
        help="Chaos engineering - fault injection for resilience testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Chaos engineering helps test system resilience by injecting controlled faults.

Fault Types:
  LATENCY              Add artificial delay
  ERROR                Raise exceptions
  TIMEOUT              Simulate timeouts
  RESOURCE_EXHAUSTION  Simulate memory pressure

Examples:
  lida chaos enable
  lida chaos add latency-fault --type latency --probability 0.1 --duration 2.0
  lida chaos add error-fault --type error --probability 0.05
  lida chaos status
  lida chaos disable
        """
    )
    chaos_subs = chaos_parser.add_subparsers(dest="subcommand")

    chaos_subs.add_parser("enable", help="Enable chaos engineering")
    chaos_subs.add_parser("disable", help="Disable chaos engineering")
    chaos_subs.add_parser("status", help="Show chaos engine status")

    ch_add = chaos_subs.add_parser("add", help="Add a fault configuration")
    ch_add.add_argument("name", help="Fault name")
    ch_add.add_argument("--type", "-t", required=True,
                        choices=["latency", "error", "timeout", "resource_exhaustion"])
    ch_add.add_argument("--probability", "-p", type=float, default=0.1, help="Injection probability (0-1)")
    ch_add.add_argument("--duration", "-d", type=float, help="Duration for latency/timeout faults")
    ch_add.add_argument("--services", "-s", help="Target services (comma-separated)")

    ch_inject = chaos_subs.add_parser("inject", help="Manually inject a fault")
    ch_inject.add_argument("name", help="Fault name to inject")

    chaos_parser.set_defaults(func=cmd_chaos, subcommand="status")

    # trace command - distributed tracing
    trace_parser = subparsers.add_parser(
        "trace",
        help="Distributed tracing for debugging and performance analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Distributed tracing tracks requests across services.

Examples:
  lida trace start --operation "deliberation"
  # ... run operations ...
  lida trace finish
  lida trace show --trace-id abc123
  lida trace export --trace-id abc123 --output trace.json
        """
    )
    trace_subs = trace_parser.add_subparsers(dest="subcommand")

    tr_start = trace_subs.add_parser("start", help="Start a new trace")
    tr_start.add_argument("--operation", "-o", help="Operation name")

    trace_subs.add_parser("finish", help="Finish current trace")

    tr_show = trace_subs.add_parser("show", help="Show trace details")
    tr_show.add_argument("--trace-id", "-t", help="Trace ID to show")

    tr_export = trace_subs.add_parser("export", help="Export trace as JSON")
    tr_export.add_argument("--trace-id", "-t", help="Trace ID to export")
    tr_export.add_argument("--output", "-o", help="Output file")

    trace_parser.set_defaults(func=cmd_trace, subcommand="show")

    # autoscale command - auto-scaling
    autoscale_parser = subparsers.add_parser(
        "autoscale",
        help="Auto-scaling configuration and monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configure automatic scaling based on CPU, memory, and queue depth.

Examples:
  lida autoscale configure --min 2 --max 10 --scale-up-threshold 80
  lida autoscale status
  lida autoscale simulate --min 1 --max 5
        """
    )
    autoscale_subs = autoscale_parser.add_subparsers(dest="subcommand")

    as_config = autoscale_subs.add_parser("configure", help="Configure auto-scaling policy")
    as_config.add_argument("--min", type=int, default=1, help="Minimum instances")
    as_config.add_argument("--max", type=int, default=10, help="Maximum instances")
    as_config.add_argument("--scale-up-threshold", type=float, default=80.0)
    as_config.add_argument("--scale-down-threshold", type=float, default=30.0)

    autoscale_subs.add_parser("status", help="Show auto-scaler status")

    as_sim = autoscale_subs.add_parser("simulate", help="Simulate auto-scaling")
    as_sim.add_argument("--min", type=int, default=1)
    as_sim.add_argument("--max", type=int, default=10)

    autoscale_parser.set_defaults(func=cmd_autoscale, subcommand="status")

    # circuit command - circuit breakers
    circuit_parser = subparsers.add_parser(
        "circuit",
        help="Circuit breaker management for fault tolerance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Circuit breakers prevent cascading failures by stopping calls to failing services.

States:
  CLOSED     Normal operation - requests pass through
  OPEN       Service failing - requests rejected
  HALF_OPEN  Testing recovery - limited requests allowed

Examples:
  lida circuit status
  lida circuit configure api --failure-threshold 5 --recovery-timeout 30
  lida circuit trip api              # Manually trip circuit
  lida circuit reset api             # Reset circuit to CLOSED
        """
    )
    circuit_subs = circuit_parser.add_subparsers(dest="subcommand")

    circuit_subs.add_parser("status", help="Show all circuit breakers")

    cb_reset = circuit_subs.add_parser("reset", help="Reset circuit breaker")
    cb_reset.add_argument("name", help="Circuit breaker name")

    cb_trip = circuit_subs.add_parser("trip", help="Manually trip circuit breaker")
    cb_trip.add_argument("name", help="Circuit breaker name")

    cb_config = circuit_subs.add_parser("configure", help="Configure circuit breaker")
    cb_config.add_argument("name", help="Circuit breaker name")
    cb_config.add_argument("--failure-threshold", type=int, default=5)
    cb_config.add_argument("--recovery-timeout", type=float, default=30.0)

    circuit_parser.set_defaults(func=cmd_circuit, subcommand="status")

    # schedule command - job scheduler
    schedule_parser = subparsers.add_parser(
        "schedule",
        help="Job scheduler for recurring tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Schedule recurring tasks like health checks, cleanups, or reports.

Examples:
  lida schedule add health-check --interval 60 --command "lida health check"
  lida schedule add cleanup --interval 3600 --command "lida data cleanup"
  lida schedule list
  lida schedule start     # Start scheduler daemon
        """
    )
    schedule_subs = schedule_parser.add_subparsers(dest="subcommand")

    sch_add = schedule_subs.add_parser("add", help="Add a scheduled job")
    sch_add.add_argument("name", help="Job name")
    sch_add.add_argument("--interval", type=float, required=True, help="Interval in seconds")
    sch_add.add_argument("--command", "-c", required=True, help="Command to run")

    schedule_subs.add_parser("list", help="List scheduled jobs")
    schedule_subs.add_parser("start", help="Start scheduler")

    sch_remove = schedule_subs.add_parser("remove", help="Remove a job")
    sch_remove.add_argument("job_id", help="Job ID to remove")

    schedule_parser.set_defaults(func=cmd_schedule, subcommand="list")

    # snapshot command - state snapshots
    snapshot_parser = subparsers.add_parser(
        "snapshot",
        help="System state snapshot and restore",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Save and restore system state for debugging or recovery.

Examples:
  lida snapshot create before-upgrade --description "Pre-upgrade state"
  lida snapshot list
  lida snapshot show abc123
  lida snapshot restore abc123
  lida snapshot delete abc123
        """
    )
    snapshot_subs = snapshot_parser.add_subparsers(dest="subcommand")

    snap_create = snapshot_subs.add_parser("create", help="Create a snapshot")
    snap_create.add_argument("name", help="Snapshot name")
    snap_create.add_argument("--description", "-d", help="Description")

    snapshot_subs.add_parser("list", help="List snapshots")

    snap_show = snapshot_subs.add_parser("show", help="Show snapshot details")
    snap_show.add_argument("snapshot_id", help="Snapshot ID")

    snap_restore = snapshot_subs.add_parser("restore", help="Restore from snapshot")
    snap_restore.add_argument("snapshot_id", help="Snapshot ID")

    snap_delete = snapshot_subs.add_parser("delete", help="Delete a snapshot")
    snap_delete.add_argument("snapshot_id", help="Snapshot ID")

    snapshot_parser.set_defaults(func=cmd_snapshot, subcommand="list")

    # events command - event bus
    events_parser = subparsers.add_parser(
        "events",
        help="Event bus monitoring and publishing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Monitor and publish events for inter-service communication.

Examples:
  lida events history
  lida events history --type chaos.enabled
  lida events watch                        # Watch all events
  lida events publish my.event --data '{"key": "value"}'
        """
    )
    events_subs = events_parser.add_subparsers(dest="subcommand")

    ev_publish = events_subs.add_parser("publish", help="Publish an event")
    ev_publish.add_argument("type", help="Event type")
    ev_publish.add_argument("--data", "-d", help="Event data (JSON or string)")

    ev_history = events_subs.add_parser("history", help="Show event history")
    ev_history.add_argument("--type", "-t", help="Filter by event type")
    ev_history.add_argument("--verbose", "-v", action="store_true")

    events_subs.add_parser("watch", help="Watch events in real-time")

    events_parser.set_defaults(func=cmd_events, subcommand="history")

    # health command - health checking
    health_parser = subparsers.add_parser(
        "health",
        help="Health checking and monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Comprehensive health checks for all system components.

Examples:
  lida health check                    # Run all health checks
  lida health check --name redis       # Check specific component
  lida health watch --interval 10      # Continuous monitoring
  lida health json                     # Output as JSON (for scripts)
        """
    )
    health_subs = health_parser.add_subparsers(dest="subcommand")

    h_check = health_subs.add_parser("check", help="Run health checks")
    h_check.add_argument("--name", "-n", help="Specific check to run")

    h_watch = health_subs.add_parser("watch", help="Watch health continuously")
    h_watch.add_argument("--interval", "-i", type=int, default=5)

    health_subs.add_parser("json", help="Output health as JSON")

    health_parser.add_argument("--redis-port", type=int, default=6379)
    health_parser.add_argument("--api-port", type=int, default=2040)
    health_parser.set_defaults(func=cmd_health, subcommand="check")

    # ==========================================================================
    # STORYTELLING COMMAND - Multi-agent collaborative storytelling
    # ==========================================================================

    story_parser = subparsers.add_parser(
        "story",
        help="Multi-agent collaborative storytelling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Create collaborative stories with multiple AI agents taking on different roles.

Genres:
  drama, comedy, thriller, mystery, romance, scifi, fantasy, horror,
  political, philosophical, satire, adventure, historical

Modes:
  collaborative  Multiple agents build story together (default)
  round_robin    Agents take turns contributing
  director_led   Director agent guides the narrative
  debate_style   Characters argue and debate
  improv         Free-form improvisation
  structured     Following classic story beats

Examples:
  lida story --title "The Last Algorithm" --genre scifi
  lida story --title "Crossroads" --genre drama --characters 4 --rounds 20
  lida story --title "The Debate" --genre political --mode debate_style
  lida story --list-genres
  lida story --demo
        """
    )
    story_parser.add_argument("--title", "-t", default="Untitled Story", help="Story title")
    story_parser.add_argument("--genre", "-g", default="drama",
                              choices=["drama", "comedy", "thriller", "mystery", "romance",
                                       "scifi", "fantasy", "horror", "political", "philosophical",
                                       "satire", "adventure", "historical", "documentary"],
                              help="Story genre")
    story_parser.add_argument("--mode", "-m", default="collaborative",
                              choices=["collaborative", "round_robin", "director_led",
                                       "debate_style", "improv", "structured", "emergent"],
                              help="Storytelling mode")
    story_parser.add_argument("--characters", "-c", type=int, default=3, help="Number of characters")
    story_parser.add_argument("--rounds", "-r", type=int, default=15, help="Number of story rounds")
    story_parser.add_argument("--setting", "-s", help="Story setting")
    story_parser.add_argument("--themes", nargs="+", help="Story themes")
    story_parser.add_argument("--stream", action="store_true", default=True, help="Stream output")
    story_parser.add_argument("--no-stream", dest="stream", action="store_false")
    story_parser.add_argument("--output", "-o", help="Output file for completed story")
    story_parser.add_argument("--format", choices=["text", "json", "screenplay"], default="text")
    story_parser.add_argument("--list-genres", action="store_true", help="List available genres")
    story_parser.add_argument("--demo", action="store_true", help="Run storytelling demo")
    story_parser.add_argument("--no-director", action="store_true", help="Disable director agent")
    story_parser.add_argument("--no-narrator", action="store_true", help="Disable narrator agent")
    story_parser.add_argument("--with-critic", action="store_true", help="Enable critic agent")
    story_parser.set_defaults(func=cmd_story)

    args = parser.parse_args()

    if args.version:
        cmd_version(args)
        return

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
