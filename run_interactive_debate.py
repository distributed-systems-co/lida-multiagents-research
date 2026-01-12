#!/usr/bin/env python3
"""
Interactive AI X-Risk Debate Runner

Features:
- Support for 6-8+ personas per debate
- Real-time hooks for intervention
- Inject arguments mid-debate
- Pause/resume functionality
- Live position tracking
- Save/load debate state
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass, field

sys.path.insert(0, ".")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich import print as rprint

from src.simulation.advanced_debate_engine import (
    AdvancedDebateEngine,
    DebateState,
    DebaterState,
    Argument,
    EmotionalState,
    ArgumentType,
    EXTENDED_PERSONAS,
)

console = Console()


# ============================================================================
# HOOKS SYSTEM
# ============================================================================

@dataclass
class DebateHooks:
    """Hooks for real-time debate intervention and monitoring."""

    # Called before each round starts
    on_round_start: Optional[Callable[[int, DebateState], None]] = None

    # Called after each round ends
    on_round_end: Optional[Callable[[int, DebateState, List[Argument]], None]] = None

    # Called when any debater speaks
    on_speech: Optional[Callable[[str, str, Argument], None]] = None

    # Called when a belief changes
    on_belief_change: Optional[Callable[[str, str, float, float], None]] = None

    # Called when emotional state shifts
    on_emotional_shift: Optional[Callable[[str, EmotionalState, EmotionalState], None]] = None

    # Called to check if user wants to intervene
    on_intervention_check: Optional[Callable[[int, DebateState], Optional[Dict]], None] = None

    # Called when debate completes
    on_debate_complete: Optional[Callable[[DebateState], None]] = None


# ============================================================================
# INTERACTIVE DEBATE RUNNER
# ============================================================================

class InteractiveDebateRunner:
    """
    Enhanced debate runner with hooks, interaction, and support for many personas.
    """

    # All available personas
    ALL_PERSONAS = list(EXTENDED_PERSONAS.keys())

    def __init__(
        self,
        topic: str,
        motion: str,
        participants: List[str],
        rounds: int = 10,
        use_llm: bool = True,
        llm_provider: str = "openrouter",
        llm_model: Optional[str] = None,
        hooks: Optional[DebateHooks] = None,
    ):
        self.topic = topic
        self.motion = motion
        self.participants = participants
        self.rounds = rounds
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.hooks = hooks or DebateHooks()

        # State
        self.engine: Optional[AdvancedDebateEngine] = None
        self.round_history: List[Dict[str, Any]] = []
        self.paused = False
        self.aborted = False

    def _init_engine(self):
        """Initialize the debate engine."""
        self.engine = AdvancedDebateEngine(
            topic=self.topic,
            motion=self.motion,
            participants=self.participants,
            llm_provider=self.llm_provider,
            llm_model=self.llm_model,
            use_llm=self.use_llm,
        )

        # Wire up engine callbacks to our hooks
        if self.hooks.on_speech:
            self.engine.on_speech = self.hooks.on_speech
        if self.hooks.on_belief_change:
            self.engine.on_belief_change = self.hooks.on_belief_change
        if self.hooks.on_emotional_shift:
            self.engine.on_emotional_shift = self.hooks.on_emotional_shift

    async def run(self) -> DebateState:
        """Run the full debate with hooks."""
        self._init_engine()

        console.print(Panel(
            f"[bold cyan]{self.topic}[/bold cyan]\n"
            f"[yellow]Motion:[/yellow] {self.motion}\n"
            f"[green]Participants:[/green] {len(self.participants)}\n"
            f"[blue]Rounds:[/blue] {self.rounds}",
            title="[bold]AI X-Risk Debate[/bold]",
            expand=False
        ))

        # Show participants
        self._show_participants()

        # Run rounds
        for round_num in range(1, self.rounds + 1):
            if self.aborted:
                break

            # Check for pause
            while self.paused:
                await asyncio.sleep(0.1)

            # Hook: round start
            if self.hooks.on_round_start:
                self.hooks.on_round_start(round_num, self.engine.state)

            # Check for intervention
            intervention = None
            if self.hooks.on_intervention_check:
                intervention = self.hooks.on_intervention_check(round_num, self.engine.state)
                if intervention:
                    self.engine.intervention_queue.append(intervention)

            # Run the round
            console.print(f"\n[bold cyan]═══ Round {round_num}/{self.rounds} ═══[/bold cyan]")
            arguments = await self.engine.run_round()

            # Display arguments
            for arg in arguments:
                debater = self.engine.state.debaters[arg.speaker]
                self._display_argument(debater, arg)

            # Show positions after round
            self._show_positions_compact()

            # Hook: round end
            if self.hooks.on_round_end:
                self.hooks.on_round_end(round_num, self.engine.state, arguments)

            # Store history
            self.round_history.append({
                "round": round_num,
                "arguments": [self._arg_to_dict(a) for a in arguments],
                "positions": {
                    d_id: d.beliefs.get("support_motion", 0.5)
                    for d_id, d in self.engine.state.debaters.items()
                },
                "tension": self.engine.state.tension_level,
                "convergence": self.engine.state.convergence,
            })

            # Brief pause
            await asyncio.sleep(0.1)

        # Hook: debate complete
        if self.hooks.on_debate_complete:
            self.hooks.on_debate_complete(self.engine.state)

        # Final summary
        self._show_final_summary()

        return self.engine.state

    def _show_participants(self):
        """Display participant table."""
        table = Table(title="Debate Participants", show_header=True)
        table.add_column("Name", style="cyan")
        table.add_column("Role", style="white")
        table.add_column("Archetype", style="yellow")
        table.add_column("Initial Position", style="green")

        for p_id in self.participants:
            if p_id in EXTENDED_PERSONAS:
                persona = EXTENDED_PERSONAS[p_id]
                debater = self.engine.state.debaters.get(p_id)
                pos = debater.beliefs.get("support_motion", 0.5) if debater else 0.5
                pos_str = f"{pos:.0%} " + ("FOR" if pos > 0.6 else "AGAINST" if pos < 0.4 else "UNDECIDED")
                table.add_row(
                    persona["name"],
                    persona["title"][:40] + "..." if len(persona["title"]) > 40 else persona["title"],
                    persona.get("archetype", "unknown"),
                    pos_str
                )

        console.print(table)

    def _display_argument(self, debater: DebaterState, arg: Argument):
        """Display a single argument."""
        # Color based on position
        pos = debater.beliefs.get("support_motion", 0.5)
        if pos > 0.6:
            color = "green"
        elif pos < 0.4:
            color = "red"
        else:
            color = "yellow"

        # Truncate content for display
        content = arg.content[:200] + "..." if len(arg.content) > 200 else arg.content

        console.print(f"  [{color}]{debater.name}[/{color}] ({debater.emotional_state.value}):")
        console.print(f"    \"{content}\"")

    def _show_positions_compact(self):
        """Show current positions in compact format."""
        positions = []
        for d_id, debater in self.engine.state.debaters.items():
            pos = debater.beliefs.get("support_motion", 0.5)
            name = debater.name.split()[0]  # First name only
            if pos > 0.6:
                positions.append(f"[green]{name}:{pos:.0%}[/green]")
            elif pos < 0.4:
                positions.append(f"[red]{name}:{pos:.0%}[/red]")
            else:
                positions.append(f"[yellow]{name}:{pos:.0%}[/yellow]")

        console.print(f"  [dim]Positions: {' | '.join(positions)}[/dim]")

    def _show_final_summary(self):
        """Show final debate summary."""
        console.print("\n" + "=" * 60)
        console.print("[bold]DEBATE COMPLETE[/bold]")
        console.print("=" * 60)

        table = Table(title="Final Positions", show_header=True)
        table.add_column("Debater", style="cyan")
        table.add_column("Position", justify="right")
        table.add_column("Stance", style="bold")
        table.add_column("Emotional State")

        for d_id, debater in self.engine.state.debaters.items():
            pos = debater.beliefs.get("support_motion", 0.5)
            if pos > 0.6:
                stance = "[green]FOR[/green]"
            elif pos < 0.4:
                stance = "[red]AGAINST[/red]"
            else:
                stance = "[yellow]UNDECIDED[/yellow]"

            table.add_row(
                debater.name,
                f"{pos:.0%}",
                stance,
                debater.emotional_state.value
            )

        console.print(table)

        # Metrics
        console.print(f"\n[cyan]Tension Level:[/cyan] {self.engine.state.tension_level:.0%}")
        console.print(f"[cyan]Convergence:[/cyan] {self.engine.state.convergence:.0%}")
        console.print(f"[cyan]Total Arguments:[/cyan] {len(self.engine.state.arguments)}")

    def _arg_to_dict(self, arg: Argument) -> Dict:
        """Convert argument to dict for storage."""
        return {
            "id": arg.id,
            "speaker": arg.speaker,
            "content": arg.content,
            "type": arg.argument_type.value,
            "strength": arg.strength,
            "target": arg.target,
        }

    def inject_argument(self, speaker_id: str, content: str, target_id: Optional[str] = None):
        """Inject an argument into the debate."""
        if self.engine:
            self.engine.intervention_queue.append({
                "type": "inject_argument",
                "speaker": speaker_id,
                "content": content,
                "target": target_id,
            })

    def pause(self):
        """Pause the debate."""
        self.paused = True
        console.print("[yellow]Debate paused[/yellow]")

    def resume(self):
        """Resume the debate."""
        self.paused = False
        console.print("[green]Debate resumed[/green]")

    def abort(self):
        """Abort the debate."""
        self.aborted = True
        console.print("[red]Debate aborted[/red]")

    def save_state(self, filepath: str):
        """Save debate state to file."""
        state = {
            "topic": self.topic,
            "motion": self.motion,
            "participants": self.participants,
            "round_history": self.round_history,
            "timestamp": datetime.now().isoformat(),
        }
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)
        console.print(f"[green]State saved to {filepath}[/green]")

    @classmethod
    def list_personas(cls):
        """List all available personas."""
        table = Table(title="Available Personas", show_header=True)
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Archetype", style="yellow")

        for p_id, persona in EXTENDED_PERSONAS.items():
            table.add_row(
                p_id,
                persona["name"],
                persona.get("archetype", "unknown")
            )

        console.print(table)


# ============================================================================
# PRESET DEBATES
# ============================================================================

DEBATE_PRESETS = {
    "full_xrisk": {
        "topic": "AI Existential Risk - Full Panel",
        "motion": "AI development poses an existential risk to humanity and should be heavily regulated",
        "participants": [
            "yudkowsky", "sam_altman", "geoffrey_hinton", "yann_lecun",
            "dario_amodei", "stuart_russell", "connor", "demis_hassabis"
        ],
        "rounds": 15,
    },
    "lab_governance_extended": {
        "topic": "AI Lab Governance - Extended Panel",
        "motion": "AI labs cannot be trusted to self-regulate on safety",
        "participants": [
            "timnit_gebru", "dario_amodei", "stuart_russell", "demis_hassabis",
            "sam_altman", "yoshua_bengio"
        ],
        "rounds": 12,
    },
    "open_vs_closed": {
        "topic": "Open Source AI - Full Debate",
        "motion": "Frontier AI models should be open sourced for safety through transparency",
        "participants": [
            "yann_lecun", "yoshua_bengio", "andrew_ng", "connor",
            "sam_altman", "timnit_gebru"
        ],
        "rounds": 12,
    },
    "turing_winners": {
        "topic": "Turing Award Winners on AI Future",
        "motion": "Current AI development trajectories will lead to transformative AI within 10 years",
        "participants": [
            "geoffrey_hinton", "yann_lecun", "yoshua_bengio",
            "fei_fei_li", "andrew_ng", "demis_hassabis"
        ],
        "rounds": 10,
    },
    "ceo_clash": {
        "topic": "AI Lab CEOs Debate Safety",
        "motion": "Commercial AI labs are doing enough to ensure AI safety",
        "participants": [
            "sam_altman", "dario_amodei", "demis_hassabis",
            "elon_musk", "stuart_russell", "timnit_gebru"
        ],
        "rounds": 12,
    },
    "safety_maximalists": {
        "topic": "Safety Maximalists vs Accelerationists",
        "motion": "AI development should be paused until we solve alignment",
        "participants": [
            "yudkowsky", "connor", "stuart_russell", "geoffrey_hinton",
            "yann_lecun", "andrew_ng", "sam_altman", "elon_musk"
        ],
        "rounds": 15,
    },
}


# ============================================================================
# INTERACTIVE CLI
# ============================================================================

async def interactive_menu():
    """Run interactive menu for debate selection."""
    console.print(Panel(
        "[bold cyan]AI X-Risk Debate Simulator[/bold cyan]\n"
        "[dim]Interactive multi-agent debate with real personas[/dim]",
        expand=False
    ))

    while True:
        console.print("\n[bold]Options:[/bold]")
        console.print("  1. Run preset debate")
        console.print("  2. Custom debate")
        console.print("  3. List all personas")
        console.print("  4. Quit")

        choice = Prompt.ask("Select option", choices=["1", "2", "3", "4"], default="1")

        if choice == "1":
            await run_preset_debate()
        elif choice == "2":
            await run_custom_debate()
        elif choice == "3":
            InteractiveDebateRunner.list_personas()
        elif choice == "4":
            console.print("[dim]Goodbye![/dim]")
            break


async def run_preset_debate():
    """Run a preset debate."""
    console.print("\n[bold]Available Presets:[/bold]")
    for i, (key, preset) in enumerate(DEBATE_PRESETS.items(), 1):
        console.print(f"  {i}. [cyan]{key}[/cyan]: {preset['topic']} ({len(preset['participants'])} participants)")

    preset_names = list(DEBATE_PRESETS.keys())
    choice = Prompt.ask(
        "Select preset",
        choices=[str(i) for i in range(1, len(preset_names) + 1)],
        default="1"
    )

    preset_key = preset_names[int(choice) - 1]
    preset = DEBATE_PRESETS[preset_key]

    # Check for LLM
    use_llm = bool(os.environ.get("OPENROUTER_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))
    if not use_llm:
        console.print("[yellow]Warning: No API key found. Using template responses.[/yellow]")

    # Create hooks for interactivity
    hooks = DebateHooks(
        on_round_start=lambda r, s: console.print(f"[dim]Starting round {r}...[/dim]"),
    )

    # Run debate
    runner = InteractiveDebateRunner(
        topic=preset["topic"],
        motion=preset["motion"],
        participants=preset["participants"],
        rounds=preset["rounds"],
        use_llm=use_llm,
        hooks=hooks,
    )

    state = await runner.run()

    # Save option
    if Confirm.ask("Save debate state?", default=False):
        filename = f"debate_{preset_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        runner.save_state(f"experiment_results/{filename}")


async def run_custom_debate():
    """Run a custom debate with user-selected personas."""
    # List personas
    InteractiveDebateRunner.list_personas()

    # Get topic
    topic = Prompt.ask("Enter debate topic", default="AI Safety")
    motion = Prompt.ask("Enter motion to debate", default="AI development should be regulated")

    # Select personas
    console.print("\n[bold]Select participants (comma-separated IDs):[/bold]")
    console.print(f"[dim]Available: {', '.join(EXTENDED_PERSONAS.keys())}[/dim]")

    participants_str = Prompt.ask(
        "Participants",
        default="yudkowsky,sam_altman,geoffrey_hinton,yann_lecun,dario_amodei,stuart_russell"
    )
    participants = [p.strip() for p in participants_str.split(",")]

    # Validate
    valid_participants = [p for p in participants if p in EXTENDED_PERSONAS]
    if len(valid_participants) < 2:
        console.print("[red]Need at least 2 valid participants![/red]")
        return

    # Rounds
    rounds = int(Prompt.ask("Number of rounds", default="10"))

    # Check for LLM
    use_llm = bool(os.environ.get("OPENROUTER_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))

    # Run
    runner = InteractiveDebateRunner(
        topic=topic,
        motion=motion,
        participants=valid_participants,
        rounds=rounds,
        use_llm=use_llm,
    )

    await runner.run()


# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Interactive AI X-Risk Debate Simulator")
    parser.add_argument("--preset", type=str, help="Run a preset debate directly")
    parser.add_argument("--list-presets", action="store_true", help="List available presets")
    parser.add_argument("--list-personas", action="store_true", help="List available personas")
    parser.add_argument("--rounds", type=int, default=None, help="Override number of rounds")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM (use templates)")

    args = parser.parse_args()

    if args.list_presets:
        console.print("[bold]Available Presets:[/bold]")
        for key, preset in DEBATE_PRESETS.items():
            console.print(f"  [cyan]{key}[/cyan]: {preset['topic']} ({len(preset['participants'])} participants, {preset['rounds']} rounds)")
        return

    if args.list_personas:
        InteractiveDebateRunner.list_personas()
        return

    if args.preset:
        if args.preset not in DEBATE_PRESETS:
            console.print(f"[red]Unknown preset: {args.preset}[/red]")
            return

        preset = DEBATE_PRESETS[args.preset]
        rounds = args.rounds or preset["rounds"]
        use_llm = not args.no_llm and bool(os.environ.get("OPENROUTER_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))

        runner = InteractiveDebateRunner(
            topic=preset["topic"],
            motion=preset["motion"],
            participants=preset["participants"],
            rounds=rounds,
            use_llm=use_llm,
        )

        await runner.run()
        return

    # Interactive menu
    await interactive_menu()


if __name__ == "__main__":
    asyncio.run(main())
