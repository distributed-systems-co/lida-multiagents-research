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
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


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
    """Start the API server."""
    print(f"Starting LIDA API server on port {args.port}...")

    # Check if uvicorn is available
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)

    # Find the server module
    server_path = Path(__file__).parent / "server.py"
    if not server_path.exists():
        print("Error: server.py not found")
        sys.exit(1)

    os.environ["SCENARIO"] = args.scenario or "quick_personas3"
    if args.live:
        os.environ["SWARM_LIVE"] = "true"

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
    )


def cmd_dashboard(args):
    """Launch the web dashboard."""
    import webbrowser

    port = args.port or 2040
    url = f"http://localhost:{port}"

    print(f"Opening dashboard at {url}")

    # Start server in background if not running
    if not args.no_server:
        import subprocess
        subprocess.Popen(
            [sys.executable, "server.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        import time
        time.sleep(2)

    webbrowser.open(url)


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
    """Run multi-agent quorum deliberation."""
    print("Running quorum deliberation...")

    event = args.event or "Major AI breakthrough announced"
    backend = args.backend or "openrouter"

    async def run_quorum():
        if backend == "mlx":
            # Try MLX backend
            try:
                # Import from run_mlx_quorum
                sys.path.insert(0, str(_PROJECT_ROOT))
                from run_mlx_quorum import run_mlx_quorum as _run_mlx
                await _run_mlx(event)
            except ImportError:
                print("MLX backend not available. Install mlx-lm.")
                sys.exit(1)
        else:
            # OpenRouter backend
            try:
                sys.path.insert(0, str(_PROJECT_ROOT))
                from run_openrouter_quorum import run_simple_demo
                await run_simple_demo()
            except ImportError as e:
                print(f"Error loading quorum module: {e}")
                sys.exit(1)

    asyncio.run(run_quorum())


def cmd_debate(args):
    """Run AI safety debate."""
    print("Running AI safety debate...")

    async def run_debate():
        sys.path.insert(0, str(_PROJECT_ROOT))

        if args.interactive:
            from run_interactive_debate import main as interactive_main
            await interactive_main()
        elif args.scenario:
            from run_comprehensive_debates import run_single_debate
            # Load scenario
            result = await run_single_debate({"id": args.scenario}, use_llm=not args.no_llm)
            print(json.dumps(result, indent=2, default=str))
        else:
            from run_ai_safety_debate import main as debate_main
            debate_main()

    asyncio.run(run_debate())


def cmd_demo(args):
    """Run demonstration."""
    demo_type = args.type or "quick"
    print(f"Running {demo_type} demo...")

    async def run_demo():
        sys.path.insert(0, str(_PROJECT_ROOT))

        if demo_type == "live":
            from run_live_demo import run_live_demo
            await run_live_demo()
        elif demo_type == "streaming":
            from run_live_streaming_demo import run_live_streaming_demo
            await run_live_streaming_demo()
        elif demo_type == "quick":
            from run_quick_stream_demo import main as quick_main
            await quick_main()
        elif demo_type == "swarm":
            from run_live_swarm import main as swarm_main
            await swarm_main()
        else:
            print(f"Unknown demo type: {demo_type}")
            print("Available: live, streaming, quick, swarm")
            sys.exit(1)

    asyncio.run(run_demo())


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
    """Run worker pool."""
    count = args.count or 4
    print(f"Starting worker pool with {count} workers...")

    async def run_workers():
        sys.path.insert(0, str(_PROJECT_ROOT))
        from run_workers import main as workers_main
        sys.argv = ["run_workers", "--count", str(count)]
        await workers_main()

    asyncio.run(run_workers())


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
  lida run ai_xrisk                    Run the AI x-risk debate scenario
  lida run ai_xrisk --no-live          Run without LLM calls (simulation)
  lida simulate chip_war --ticks 20    Run chip war policy simulation
  lida simulate negotiation --agent-a xi_jinping --agent-b gina_raimondo
  lida quorum --event "AGI announced"  Run multi-agent quorum deliberation
  lida debate --interactive            Interactive AI safety debate
  lida demo --type streaming           Run streaming demo
  lida chat sam_altman elon_musk       Two personas conversation
  lida experiment config.yaml          Run full experiment from config
  lida analyze results.json --all      Run all analyses on results
  lida serve --port 8080               Start API server on port 8080
  lida export results.json             Export to LaTeX tables/figures
  lida list --scenarios                List available scenarios
  lida logs .llm_logs.json             Interactive log viewer
  lida workers --count 8               Run worker pool
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
    quorum_parser = subparsers.add_parser("quorum", help="Run multi-agent quorum deliberation")
    quorum_parser.add_argument("--event", "-e", help="Event headline to analyze")
    quorum_parser.add_argument("--backend", choices=["openrouter", "mlx"], default="openrouter",
                              help="LLM backend to use")
    quorum_parser.add_argument("--cycles", type=int, default=5, help="Number of deliberation cycles")
    quorum_parser.set_defaults(func=cmd_quorum)

    # debate command
    debate_parser = subparsers.add_parser("debate", help="Run AI safety debate")
    debate_parser.add_argument("--scenario", "-s", help="Specific debate scenario ID")
    debate_parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    debate_parser.add_argument("--no-llm", action="store_true", help="Run without LLM calls")
    debate_parser.add_argument("--list", action="store_true", help="List available scenarios")
    debate_parser.set_defaults(func=cmd_debate)

    # demo command
    demo_parser = subparsers.add_parser("demo", help="Run demonstrations")
    demo_parser.add_argument("--type", "-t", choices=["live", "streaming", "quick", "swarm"],
                            default="quick", help="Demo type to run")
    demo_parser.set_defaults(func=cmd_demo)

    # logs command
    logs_parser = subparsers.add_parser("logs", help="Interactive log viewer")
    logs_parser.add_argument("logfile", help="Path to .llm_logs.json file")
    logs_parser.set_defaults(func=cmd_logs)

    # workers command
    workers_parser = subparsers.add_parser("workers", help="Run worker pool")
    workers_parser.add_argument("--count", "-n", type=int, default=4, help="Number of workers")
    workers_parser.set_defaults(func=cmd_workers)

    # chat command
    chat_parser = subparsers.add_parser("chat", help="Two personas conversation")
    chat_parser.add_argument("persona1", help="First persona key")
    chat_parser.add_argument("persona2", help="Second persona key")
    chat_parser.add_argument("--topic", "-t", help="Conversation topic")
    chat_parser.add_argument("--turns", type=int, default=5, help="Number of turns")
    chat_parser.set_defaults(func=cmd_chat)

    # experiment command
    exp_parser = subparsers.add_parser("experiment", help="Run full experiment")
    exp_parser.add_argument("config", nargs="?", help="Experiment config file")
    exp_parser.add_argument("--batch", nargs="+", help="Run multiple configs")
    exp_parser.add_argument("--parallel", type=int, default=1, help="Parallel experiments")
    exp_parser.add_argument("--export-latex", action="store_true", help="Export to LaTeX")
    exp_parser.set_defaults(func=cmd_experiment)

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
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    serve_parser.add_argument("--port", type=int, default=2040, help="Port")
    serve_parser.add_argument("--workers", type=int, default=1, help="Worker processes")
    serve_parser.add_argument("--reload", action="store_true", help="Auto-reload on changes")
    serve_parser.add_argument("--scenario", help="Default scenario")
    serve_parser.add_argument("--live", action="store_true", help="Enable live LLM mode")
    serve_parser.set_defaults(func=cmd_serve)

    # dashboard command
    dash_parser = subparsers.add_parser("dashboard", help="Launch web dashboard")
    dash_parser.add_argument("--port", type=int, help="Server port")
    dash_parser.add_argument("--no-server", action="store_true", help="Don't start server")
    dash_parser.set_defaults(func=cmd_dashboard)

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
