#!/usr/bin/env python3
"""
Policy Wargaming Simulation Runner

Multi-agent policy simulation for post-AGI scenarios using OpenRouter.

Examples:
    # Turn-based chip war simulation
    python run_policy_sim.py --scenario chip_war --ticks 20 -v

    # AGI crisis simulation
    python run_policy_sim.py --scenario agi_crisis --ticks 30

    # Bilateral negotiation
    python run_policy_sim.py --scenario negotiation --agent-a xi_jinping --agent-b gina_raimondo

    # Great power competition with coalition dynamics
    python run_policy_sim.py --scenario great_power --ticks 25

    # Custom personas
    python run_policy_sim.py --personas xi_jinping,jake_sullivan,sam_altman --ticks 15

    # Interactive mode
    python run_policy_sim.py --scenario interactive
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import httpx

# Add project root to path for proper imports
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# OpenRouter config
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"


# =============================================================================
# OpenRouter Inference
# =============================================================================


async def openrouter_inference(
    system_prompt: str,
    user_message: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """Make an inference call via OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/distributed-systems-co/lida-multiagents-research",
                "X-Title": "LIDA Policy Simulation",
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
        data = response.json()

        return data["choices"][0]["message"]["content"]


# =============================================================================
# Lazy Imports (avoid import errors if modules not ready)
# =============================================================================


def get_simulation_modules():
    """Lazily import simulation modules."""
    from src.agents.simulation_engine import (
        SimulationMode,
        ScenarioLibrary,
        create_simulation,
        create_chip_war_simulation,
        create_agi_crisis_simulation,
        create_bilateral_negotiation,
    )
    return {
        "SimulationMode": SimulationMode,
        "ScenarioLibrary": ScenarioLibrary,
        "create_simulation": create_simulation,
        "create_chip_war_simulation": create_chip_war_simulation,
        "create_agi_crisis_simulation": create_agi_crisis_simulation,
        "create_bilateral_negotiation": create_bilateral_negotiation,
    }


def get_coalition_modules():
    """Lazily import coalition modules."""
    try:
        from src.agents.coalition_dynamics import (
            EnhancedSimulationEngine,
            create_great_power_competition,
        )
        return {
            "EnhancedSimulationEngine": EnhancedSimulationEngine,
            "create_great_power_competition": create_great_power_competition,
        }
    except ImportError as e:
        logger.warning(f"Coalition dynamics not available: {e}")
        return {}


def get_orchestrator():
    """Lazily import orchestrator."""
    try:
        from src.agents.orchestrator import (
            PolicySimulationOrchestrator,
            OrchestratorConfig,
            create_and_run_simulation,
        )
        return {
            "PolicySimulationOrchestrator": PolicySimulationOrchestrator,
            "OrchestratorConfig": OrchestratorConfig,
            "create_and_run_simulation": create_and_run_simulation,
        }
    except ImportError as e:
        logger.warning(f"Orchestrator not available: {e}")
        return {}


# =============================================================================
# Scenario Runners
# =============================================================================


async def run_chip_war(persona_dir: Path, ticks: int, verbose: bool = False):
    """Run chip war scenario."""
    logger.info("Starting Chip War Simulation...")

    mods = get_simulation_modules()
    create_chip_war_simulation = mods["create_chip_war_simulation"]
    ScenarioLibrary = mods["ScenarioLibrary"]

    engine = create_chip_war_simulation(persona_dir, inference_fn=openrouter_inference)

    def on_action(action):
        if verbose:
            print(f"\n[Tick {action.tick}] {action.agent_name} ({action.action_type.value}):")
            print(f"  → {action.content[:200]}")
            if action.escalation_delta != 0:
                sign = '+' if action.escalation_delta > 0 else ''
                print(f"  ↑ Escalation: {sign}{action.escalation_delta}")

    def on_escalation(event, level):
        print(f"\n⚠️  ESCALATION to {level.name} triggered by {event.source}")

    engine.on_action(on_action)
    engine.on_escalation(on_escalation)

    # Inject scenario
    name, desc, effects = ScenarioLibrary.compute_embargo("US", "China")
    engine.inject_scenario(name, desc, effects)

    print(f"\n{'='*60}")
    print("SCENARIO: US-China Compute Embargo")
    print(f"{'='*60}")
    print(f"Agents: {list(engine.agents.keys())}")
    print(f"Running {ticks} ticks...")
    print(f"{'='*60}\n")

    summary = await engine.run(max_ticks=ticks)

    print(f"\n{'='*60}")
    print("SIMULATION COMPLETE")
    print(f"{'='*60}")
    print(json.dumps(summary, indent=2, default=str))

    return summary


async def run_agi_crisis(persona_dir: Path, ticks: int, verbose: bool = False):
    """Run AGI crisis scenario."""
    logger.info("Starting AGI Crisis Simulation...")

    mods = get_simulation_modules()
    create_agi_crisis_simulation = mods["create_agi_crisis_simulation"]

    engine = create_agi_crisis_simulation(persona_dir, inference_fn=openrouter_inference)

    def on_action(action):
        if verbose:
            print(f"\n[Tick {action.tick}] {action.agent_name} ({action.action_type.value}):")
            print(f"  → {action.content[:200]}")

    engine.on_action(on_action)

    print(f"\n{'='*60}")
    print("SCENARIO: AGI Announcement Crisis")
    print(f"{'='*60}")
    print(f"Agents: {list(engine.agents.keys())}")
    print(f"Running {ticks} ticks...")
    print(f"{'='*60}\n")

    summary = await engine.run(max_ticks=ticks)

    print(f"\n{'='*60}")
    print("AGI CRISIS SIMULATION COMPLETE")
    print(f"{'='*60}")
    print(json.dumps(summary, indent=2, default=str))

    return summary


async def run_negotiation(
    persona_dir: Path,
    agent_a: str,
    agent_b: str,
    proposal: str,
):
    """Run bilateral negotiation."""
    logger.info(f"Starting Negotiation: {agent_a} ↔ {agent_b}")

    mods = get_simulation_modules()
    create_bilateral_negotiation = mods["create_bilateral_negotiation"]

    engine = create_bilateral_negotiation(
        persona_dir,
        agent_a,
        agent_b,
        inference_fn=openrouter_inference,
    )

    agent_names = list(engine.agents.keys())
    if len(agent_names) < 2:
        print(f"Error: Could not load both agents. Found: {agent_names}")
        print("Make sure persona JSON files exist for both agents.")
        return None

    print(f"\n{'='*60}")
    print(f"NEGOTIATION: {agent_names[0]} ↔ {agent_names[1]}")
    print(f"{'='*60}")
    print(f"Topic: {proposal[:100]}...")
    print(f"{'='*60}\n")

    result = await engine.run_negotiation(
        agent_names[0],
        agent_names[1],
        proposal,
    )

    print(f"\n{'='*60}")
    print("NEGOTIATION COMPLETE")
    print(f"{'='*60}")
    print(f"Outcome: {result['outcome']}")
    print(f"Rounds: {result['rounds']}")
    print("\nExchange:")
    for action in result['actions']:
        content = action['content'][:150]
        print(f"  [{action['agent']}]: {content}...")

    return result


async def run_great_power(persona_dir: Path, ticks: int, verbose: bool = False):
    """Run great power competition with coalition dynamics."""
    logger.info("Starting Great Power Competition...")

    coalition_mods = get_coalition_modules()
    if not coalition_mods:
        print("Coalition dynamics module not available.")
        return None

    create_great_power_competition = coalition_mods["create_great_power_competition"]

    engine = create_great_power_competition(persona_dir, inference_fn=openrouter_inference)

    def on_action(action):
        if verbose:
            print(f"\n[Tick {action.tick}] {action.agent_name} ({action.action_type.value}):")
            print(f"  → {action.content[:200]}")

    engine.on_action(on_action)

    print(f"\n{'='*60}")
    print("SCENARIO: Great Power Competition + Taiwan Crisis")
    print(f"{'='*60}")
    print(f"Agents: {list(engine.agents.keys())}")
    print(f"Running {ticks} ticks...")
    print(f"{'='*60}\n")

    await engine.run(max_ticks=ticks)

    summary = engine.get_strategic_summary() if hasattr(engine, 'get_strategic_summary') else engine.get_summary()

    print(f"\n{'='*60}")
    print("GREAT POWER COMPETITION COMPLETE")
    print(f"{'='*60}")
    print(json.dumps(summary, indent=2, default=str))

    return summary


async def run_custom(
    persona_dir: Path,
    personas: list,
    mode: str,
    ticks: int,
    scenario: str | None = None,
    verbose: bool = False,
):
    """Run custom simulation."""
    logger.info(f"Starting Custom Simulation with {personas}")

    mods = get_simulation_modules()
    create_simulation = mods["create_simulation"]
    SimulationMode = mods["SimulationMode"]
    ScenarioLibrary = mods["ScenarioLibrary"]

    mode_map = {
        "turn_based": SimulationMode.TURN_BASED,
        "continuous": SimulationMode.CONTINUOUS,
        "negotiation": SimulationMode.NEGOTIATION,
    }

    engine = create_simulation(
        mode=mode_map.get(mode, SimulationMode.TURN_BASED),
        persona_dir=persona_dir,
        persona_names=personas,
        inference_fn=openrouter_inference,
        max_ticks=ticks,
    )

    def on_action(action):
        if verbose:
            print(f"\n[Tick {action.tick}] {action.agent_name} ({action.action_type.value}):")
            print(f"  → {action.content[:200]}")

    engine.on_action(on_action)

    # Inject scenario
    if scenario == "agi":
        name, desc, effects = ScenarioLibrary.agi_announcement()
        engine.inject_scenario(name, desc, effects)
    elif scenario == "embargo":
        name, desc, effects = ScenarioLibrary.compute_embargo()
        engine.inject_scenario(name, desc, effects)
    elif scenario == "taiwan":
        name, desc, effects = ScenarioLibrary.taiwan_crisis()
        engine.inject_scenario(name, desc, effects)
    elif scenario == "safety":
        name, desc, effects = ScenarioLibrary.safety_incident("major")
        engine.inject_scenario(name, desc, effects)

    print(f"\n{'='*60}")
    print(f"CUSTOM SIMULATION: {', '.join(personas)}")
    print(f"{'='*60}")
    print(f"Mode: {mode}")
    print(f"Scenario: {scenario or 'None'}")
    print(f"Running {ticks} ticks...")
    print(f"{'='*60}\n")

    summary = await engine.run(max_ticks=ticks)

    print(f"\n{'='*60}")
    print("SIMULATION COMPLETE")
    print(f"{'='*60}")
    print(json.dumps(summary, indent=2, default=str))

    return summary


# =============================================================================
# Interactive Mode
# =============================================================================


async def interactive_mode(persona_dir: Path):
    """Run simulation in interactive mode."""
    print(f"\n{'='*60}")
    print("INTERACTIVE POLICY SIMULATION")
    print(f"{'='*60}")

    # Find available personas
    enhanced_dir = persona_dir / "enhanced"
    finalized_dir = persona_dir / "finalized"

    personas = []
    for d in [enhanced_dir, finalized_dir, persona_dir]:
        if d.exists():
            personas.extend([f.stem for f in d.glob("*.json") if not f.name.startswith(".")])
    personas = sorted(set(personas))

    print(f"\nFound {len(personas)} personas:")
    for i, p in enumerate(personas[:20], 1):
        print(f"  {i:2}. {p.replace('_', ' ').title()}")
    if len(personas) > 20:
        print(f"  ... and {len(personas) - 20} more")

    print("\nCommands:")
    print("  add <n>           - Add persona by number")
    print("  remove <n>        - Remove agent by number")
    print("  list              - List current agents and world state")
    print("  inject <scenario> - Inject: agi, embargo, taiwan, safety")
    print("  step              - Run one tick")
    print("  run <n>           - Run n ticks")
    print("  summary           - Show summary")
    print("  quit              - Exit")

    # Import what we need
    coalition_mods = get_coalition_modules()
    if coalition_mods and "EnhancedSimulationEngine" in coalition_mods:
        EnhancedSimulationEngine = coalition_mods["EnhancedSimulationEngine"]
        engine = EnhancedSimulationEngine(inference_fn=openrouter_inference)
    else:
        mods = get_simulation_modules()
        from src.agents.simulation_engine import SimulationEngine
        engine = SimulationEngine(inference_fn=openrouter_inference)

    mods = get_simulation_modules()
    ScenarioLibrary = mods["ScenarioLibrary"]

    while True:
        try:
            cmd = input("\n> ").strip()
            if not cmd:
                continue

            parts = cmd.split(maxsplit=1)
            command = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if command == "quit":
                break

            elif command == "add":
                try:
                    idx = int(arg) - 1
                    persona_name = personas[idx]

                    for d in [enhanced_dir, finalized_dir, persona_dir]:
                        path = d / f"{persona_name}.json"
                        if path.exists():
                            engine.add_agent_from_persona(path)
                            print(f"✓ Added: {persona_name}")
                            break
                    else:
                        print(f"✗ Not found: {persona_name}")
                except (ValueError, IndexError) as e:
                    print(f"✗ Invalid: {e}")

            elif command == "remove":
                try:
                    names = list(engine.agents.keys())
                    idx = int(arg) - 1
                    name = names[idx]
                    del engine.agents[name]
                    print(f"✓ Removed: {name}")
                except (ValueError, IndexError) as e:
                    print(f"✗ Invalid: {e}")

            elif command == "list":
                print("\nAgents:")
                for i, name in enumerate(engine.agents.keys(), 1):
                    agent = engine.agents[name]
                    role = agent.persona.current_role[:40] if agent.persona.current_role else "Unknown"
                    print(f"  {i}. {name} - {role}")

                print(f"\nWorld State:")
                print(f"  Tick: {engine.world.tick}")
                print(f"  Escalation: {engine.world.geopolitics.escalation_level.name}")
                print(f"  AI Capability: {engine.world.ai.frontier_model_capability:.1f}/100")
                print(f"  AGI Proximity: {engine.world.ai.agi_proximity:.1%}")

            elif command == "inject":
                scenarios = {
                    "agi": ScenarioLibrary.agi_announcement,
                    "embargo": ScenarioLibrary.compute_embargo,
                    "taiwan": ScenarioLibrary.taiwan_crisis,
                    "safety": ScenarioLibrary.safety_incident,
                }
                if arg in scenarios:
                    name, desc, effects = scenarios[arg]()
                    engine.inject_scenario(name, desc, effects)
                    print(f"✓ Injected: {desc[:80]}...")
                else:
                    print(f"✗ Unknown scenario. Options: {', '.join(scenarios.keys())}")

            elif command == "step":
                if not engine.agents:
                    print("✗ No agents loaded")
                    continue
                actions = await engine.step()
                if actions:
                    for a in actions:
                        print(f"[{a.agent_name}] {a.action_type.value}: {a.content[:100]}...")
                else:
                    print("(No actions this tick)")

            elif command == "run":
                try:
                    n = int(arg) if arg else 5
                    if not engine.agents:
                        print("✗ No agents loaded")
                        continue
                    for i in range(n):
                        actions = await engine.step()
                        for a in actions:
                            print(f"[T{engine.world.tick}] {a.agent_name}: {a.content[:80]}...")
                except ValueError:
                    print("✗ Invalid number")

            elif command == "summary":
                if hasattr(engine, 'get_strategic_summary'):
                    summary = engine.get_strategic_summary()
                else:
                    summary = engine.get_summary()
                print(json.dumps(summary, indent=2, default=str))

            else:
                print(f"✗ Unknown command: {command}")

        except KeyboardInterrupt:
            print("\n(Use 'quit' to exit)")
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

    print("\nExiting.")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Policy Wargaming Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_policy_sim.py --scenario chip_war --ticks 20 -v
  python run_policy_sim.py --scenario negotiation --agent-a xi_jinping --agent-b gina_raimondo
  python run_policy_sim.py --scenario interactive
        """
    )

    parser.add_argument(
        "--scenario",
        choices=["chip_war", "agi_crisis", "negotiation", "great_power", "custom", "interactive"],
        default="interactive",
        help="Scenario to run (default: interactive)",
    )
    parser.add_argument(
        "--mode",
        choices=["turn_based", "continuous", "negotiation"],
        default="turn_based",
        help="Simulation mode for custom scenario",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=10,
        help="Number of simulation ticks (default: 10)",
    )
    parser.add_argument(
        "--personas",
        type=str,
        default="",
        help="Comma-separated persona names for custom scenario",
    )
    parser.add_argument(
        "--agent-a",
        type=str,
        default="xi_jinping",
        help="First agent for negotiation",
    )
    parser.add_argument(
        "--agent-b",
        type=str,
        default="gina_raimondo",
        help="Second agent for negotiation",
    )
    parser.add_argument(
        "--proposal",
        type=str,
        default="Let us discuss a framework for managing AI competition that preserves both nations' security interests while enabling continued technological progress.",
        help="Initial proposal for negotiation",
    )
    parser.add_argument(
        "--inject-scenario",
        choices=["agi", "embargo", "taiwan", "safety"],
        help="Scenario to inject (for custom)",
    )
    parser.add_argument(
        "--persona-dir",
        type=str,
        default="persona_pipeline/personas",
        help="Directory containing persona JSON files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenRouter model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    persona_dir = Path(args.persona_dir)
    if not persona_dir.exists():
        print(f"Error: Persona directory not found: {persona_dir}")
        sys.exit(1)

    # Check API key
    if not OPENROUTER_API_KEY:
        print("Warning: OPENROUTER_API_KEY not set. LLM agents will not function.")
        print("Set it with: export OPENROUTER_API_KEY='your-key-here'")

    # Run appropriate scenario
    if args.scenario == "chip_war":
        asyncio.run(run_chip_war(persona_dir, args.ticks, args.verbose))

    elif args.scenario == "agi_crisis":
        asyncio.run(run_agi_crisis(persona_dir, args.ticks, args.verbose))

    elif args.scenario == "negotiation":
        asyncio.run(run_negotiation(
            persona_dir,
            args.agent_a,
            args.agent_b,
            args.proposal,
        ))

    elif args.scenario == "great_power":
        asyncio.run(run_great_power(persona_dir, args.ticks, args.verbose))

    elif args.scenario == "custom":
        personas = [p.strip() for p in args.personas.split(",") if p.strip()]
        if not personas:
            print("Error: --personas required for custom scenario")
            print("Example: --personas xi_jinping,jake_sullivan,sam_altman")
            sys.exit(1)

        asyncio.run(run_custom(
            persona_dir,
            personas,
            args.mode,
            args.ticks,
            args.inject_scenario,
            args.verbose,
        ))

    elif args.scenario == "interactive":
        asyncio.run(interactive_mode(persona_dir))


if __name__ == "__main__":
    main()
