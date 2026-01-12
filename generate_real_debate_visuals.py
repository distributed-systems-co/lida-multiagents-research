#!/usr/bin/env python3
"""
Generate visualizations from REAL AI x-risk debate simulations.

This runs actual debates with real personas and captures real data for visualization.
No fake data, no placeholders - everything comes from actual simulation runs.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

sys.path.insert(0, ".")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Error: matplotlib required. Install with: pip install matplotlib numpy")
    sys.exit(1)

from src.simulation.advanced_debate_engine import (
    AdvancedDebateEngine,
    DebateState,
    DebaterState,
    Argument,
    EmotionalState,
    ArgumentType,
    EXTENDED_PERSONAS,
)


# ============================================================================
# REAL DATA CAPTURE
# ============================================================================

class DebateDataCapture:
    """Captures real data from debate rounds for visualization."""

    def __init__(self, engine: AdvancedDebateEngine):
        self.engine = engine
        self.rounds: List[int] = []

        # Per-debater time series
        self.positions: Dict[str, List[float]] = {}  # support_motion over time
        self.emotional_states: Dict[str, List[str]] = {}
        self.arguments_count: Dict[str, List[int]] = {}

        # Per-round aggregates
        self.tension_history: List[float] = []
        self.convergence_history: List[float] = []
        self.polarization_history: List[float] = []
        self.decisiveness_history: List[float] = []
        self.momentum_history: List[float] = []

        # Arguments
        self.all_arguments: List[Dict[str, Any]] = []
        self.argument_types_by_speaker: Dict[str, Dict[str, int]] = {}

        # Initialize
        for debater_id, debater in engine.state.debaters.items():
            self.positions[debater_id] = []
            self.emotional_states[debater_id] = []
            self.arguments_count[debater_id] = []
            self.argument_types_by_speaker[debater_id] = {}

    def capture_round(self, round_num: int, arguments: List[Argument]):
        """Capture data after a round completes."""
        self.rounds.append(round_num)

        # Capture per-debater state
        for debater_id, debater in self.engine.state.debaters.items():
            self.positions[debater_id].append(debater.beliefs.get("support_motion", 0.5))
            self.emotional_states[debater_id].append(debater.emotional_state.value)
            self.arguments_count[debater_id].append(len(debater.arguments_made))

        # Capture round metrics
        self.tension_history.append(self.engine.state.tension_level)
        self.convergence_history.append(self.engine.state.convergence)
        # Handle new metrics (with backwards compatibility for older engine versions)
        self.polarization_history.append(getattr(self.engine.state, 'polarization', 0.0))
        self.decisiveness_history.append(getattr(self.engine.state, 'decisiveness', 0.0))
        self.momentum_history.append(getattr(self.engine.state, 'debate_momentum', 0.5))

        # Capture arguments
        for arg in arguments:
            self.all_arguments.append({
                "round": round_num,
                "speaker": arg.speaker,
                "speaker_name": self.engine.state.debaters[arg.speaker].name,
                "content": arg.content,
                "type": arg.argument_type.value,
                "strength": arg.strength,
                "target": arg.target,
            })

            # Track argument types per speaker
            if arg.argument_type.value not in self.argument_types_by_speaker[arg.speaker]:
                self.argument_types_by_speaker[arg.speaker][arg.argument_type.value] = 0
            self.argument_types_by_speaker[arg.speaker][arg.argument_type.value] += 1

    def get_debater_names(self) -> Dict[str, str]:
        """Get mapping of debater IDs to display names."""
        return {
            debater_id: debater.name
            for debater_id, debater in self.engine.state.debaters.items()
        }


# ============================================================================
# VISUALIZATION
# ============================================================================

# Color scheme for personas (expanded)
PERSONA_COLORS = {
    # Safety maximalists - reds/oranges
    "yudkowsky": "#E63946",
    "connor": "#D62828",
    "nick_bostrom": "#C1121F",
    "max_tegmark": "#F77F00",
    "jaan_tallinn": "#E85D04",
    "stuart_russell": "#E76F51",
    "toby_ord": "#FF4D4D",
    # Concerned pioneers - yellows/ambers
    "geoffrey_hinton": "#E9C46A",
    "ilya_sutskever": "#FFBA08",
    "yoshua_bengio": "#F4A261",
    # Safety-conscious builders - blues
    "dario_amodei": "#457B9D",
    "jan_leike": "#1D3557",
    "paul_christiano": "#2A6F97",
    # Pragmatic/accelerationist - teals/greens
    "sam_altman": "#2A9D8F",
    "demis_hassabis": "#264653",
    "mustafa_suleyman": "#168AAD",
    "kai_fu_lee": "#38A3A5",
    # Innovation advocates - light blues/greens
    "yann_lecun": "#A8DADC",
    "andrew_ng": "#90BE6D",
    # Centrists/empiricists - purples/pinks
    "gary_marcus": "#7B2CBF",
    "melanie_mitchell": "#9D4EDD",
    "katja_grace": "#C77DFF",
    "scott_aaronson": "#B388FF",
    "robin_hanson": "#EA80FC",
    # Ethics focused - magentas
    "timnit_gebru": "#9B5DE5",
    "fei_fei_li": "#F15BB5",
    "rumman_chowdhury": "#AD1457",
    "francesca_rossi": "#EC407A",
    # Technical experts - cyans
    "judea_pearl": "#00BCD4",
    "dawn_song": "#00ACC1",
    # Wild cards
    "elon_musk": "#FF6B6B",
    # Dynamic Persuaders - unique gradients
    "the_moderator": "#78909C",      # Gray-blue (neutral mediator)
    "the_provocateur": "#FF5722",     # Deep orange (provocative)
    "the_empiricist": "#00897B",      # Teal (data/evidence)
    "the_futurist": "#7E57C2",        # Deep purple (future-focused)
    "the_historian": "#8D6E63",       # Brown (historical)
    "the_ethicist": "#AB47BC",        # Purple (ethical)
    "the_synthesizer": "#26A69A",     # Cyan-teal (integrative)
    "the_pragmatist": "#5C6BC0",      # Indigo (practical)
}

ARGUMENT_TYPE_COLORS = {
    "empirical": "#4CAF50",
    "logical": "#2196F3",
    "ethical": "#9C27B0",
    "practical": "#FF9800",
    "emotional": "#E91E63",
    "authority": "#795548",
    "precedent": "#607D8B",
    "hypothetical": "#00BCD4",
    "reductio": "#F44336",
    "concession": "#8BC34A",
}


def create_position_chart(capture: DebateDataCapture, ax: plt.Axes, motion: str):
    """Create chart showing position changes over rounds."""
    names = capture.get_debater_names()

    for debater_id, positions in capture.positions.items():
        color = PERSONA_COLORS.get(debater_id, "#888888")
        name = names[debater_id]
        ax.plot(capture.rounds, positions,
                marker='o', linewidth=2.5, markersize=8,
                color=color, label=name)

    # Add reference lines with labels
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0.7, color='green', linestyle=':', alpha=0.4)
    ax.axhline(y=0.3, color='red', linestyle=':', alpha=0.4)

    # Add zone labels
    ax.text(capture.rounds[-1] + 0.1, 0.85, 'FOR', fontsize=9, color='green', alpha=0.7)
    ax.text(capture.rounds[-1] + 0.1, 0.15, 'AGAINST', fontsize=9, color='red', alpha=0.7)

    ax.set_xlabel("Debate Round", fontsize=11)
    ax.set_ylabel("Support for Motion", fontsize=11)
    ax.set_title(f"Position Evolution", fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_xlim(capture.rounds[0] - 0.3, capture.rounds[-1] + 0.8)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)


def create_tension_convergence_chart(capture: DebateDataCapture, ax: plt.Axes):
    """Create chart showing debate dynamics with all key metrics."""
    # Main metrics
    ax.plot(capture.rounds, capture.tension_history,
            'r-o', linewidth=2, label='Tension', markersize=4)
    ax.plot(capture.rounds, capture.convergence_history,
            'g-s', linewidth=2, label='True Convergence', markersize=4)
    ax.plot(capture.rounds, capture.polarization_history,
            'b-^', linewidth=2, label='Polarization', markersize=4)
    ax.plot(capture.rounds, capture.decisiveness_history,
            'm-d', linewidth=1.5, label='Decisiveness', markersize=3, alpha=0.7)

    # Light fills for key metrics
    ax.fill_between(capture.rounds, capture.tension_history, alpha=0.1, color='red')
    ax.fill_between(capture.rounds, capture.polarization_history, alpha=0.1, color='blue')

    ax.set_xlabel("Debate Round", fontsize=10)
    ax.set_ylabel("Level (0-1)", fontsize=10)
    ax.set_title("Debate Dynamics\n(True convergence requires agreement on FOR/AGAINST, not 50%)", fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Add annotation explaining metrics
    final_conv = capture.convergence_history[-1] if capture.convergence_history else 0
    final_polar = capture.polarization_history[-1] if capture.polarization_history else 0
    final_dec = capture.decisiveness_history[-1] if capture.decisiveness_history else 0
    if final_conv < 0.4 and final_polar > 0.3:
        ax.annotate("⚡ Debate unresolved", xy=(0.5, 0.02), xycoords='axes fraction',
                    fontsize=8, ha='center', color='red')


def create_argument_types_chart(capture: DebateDataCapture, ax: plt.Axes):
    """Create stacked bar chart of argument types by speaker."""
    names = capture.get_debater_names()
    speakers = list(capture.argument_types_by_speaker.keys())

    # Get all unique argument types
    all_types = set()
    for types_dict in capture.argument_types_by_speaker.values():
        all_types.update(types_dict.keys())
    all_types = sorted(all_types)

    # Build data matrix
    x = np.arange(len(speakers))
    width = 0.6
    bottom = np.zeros(len(speakers))

    for arg_type in all_types:
        counts = [capture.argument_types_by_speaker[s].get(arg_type, 0) for s in speakers]
        color = ARGUMENT_TYPE_COLORS.get(arg_type, "#888888")
        ax.bar(x, counts, width, bottom=bottom, label=arg_type.capitalize(), color=color)
        bottom += counts

    ax.set_xlabel("Debater", fontsize=10)
    ax.set_ylabel("Argument Count", fontsize=10)
    ax.set_title("Argument Types Used", fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([names[s].split()[0] for s in speakers], rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper right', fontsize=7, ncol=2)


def create_final_positions_chart(capture: DebateDataCapture, ax: plt.Axes):
    """Create bar chart of final positions."""
    names = capture.get_debater_names()

    debaters = list(capture.positions.keys())
    final_positions = [capture.positions[d][-1] for d in debaters]
    initial_positions = [capture.positions[d][0] for d in debaters]
    colors = [PERSONA_COLORS.get(d, "#888888") for d in debaters]

    x = np.arange(len(debaters))
    width = 0.35

    # Initial positions (lighter)
    bars1 = ax.bar(x - width/2, initial_positions, width,
                   label='Initial', alpha=0.5, color=colors, edgecolor='black')
    # Final positions
    bars2 = ax.bar(x + width/2, final_positions, width,
                   label='Final', color=colors, edgecolor='black')

    # Add change arrows
    for i, (init, final) in enumerate(zip(initial_positions, final_positions)):
        change = final - init
        if abs(change) > 0.05:
            arrow = '↑' if change > 0 else '↓'
            color = 'green' if change > 0 else 'red'
            ax.annotate(f'{arrow}{abs(change):.0%}',
                       xy=(x[i] + width/2, final + 0.02),
                       ha='center', fontsize=8, color=color, fontweight='bold')

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Debater", fontsize=10)
    ax.set_ylabel("Position", fontsize=10)
    ax.set_title("Position Change (Initial → Final)", fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([names[d].split()[0] for d in debaters], rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9)


def create_emotional_timeline(capture: DebateDataCapture, ax: plt.Axes):
    """Create timeline of emotional states."""
    names = capture.get_debater_names()

    # Emotional state to numeric value for visualization
    emotion_values = {
        "calm": 0, "reflective": 1, "conciliatory": 2,
        "passionate": 3, "skeptical": 4,
        "frustrated": 5, "defensive": 6, "aggressive": 7
    }

    for i, (debater_id, states) in enumerate(capture.emotional_states.items()):
        y_values = [emotion_values.get(s, 0) for s in states]
        color = PERSONA_COLORS.get(debater_id, "#888888")
        name = names[debater_id].split()[0]
        ax.plot(capture.rounds, y_values,
                marker='s', linewidth=1.5, markersize=5,
                color=color, label=name, alpha=0.8)

    ax.set_xlabel("Debate Round", fontsize=10)
    ax.set_ylabel("Emotional State", fontsize=10)
    ax.set_title("Emotional Dynamics", fontsize=11, fontweight='bold')
    ax.set_yticks(list(emotion_values.values()))
    ax.set_yticklabels([s.capitalize() for s in emotion_values.keys()], fontsize=7)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    ax.grid(True, alpha=0.3)


def create_transcript_panel(capture: DebateDataCapture, ax: plt.Axes):
    """Create panel showing key transcript excerpts."""
    ax.axis('off')

    # Get most impactful arguments (highest strength)
    sorted_args = sorted(capture.all_arguments, key=lambda x: x['strength'], reverse=True)[:4]

    text_lines = ["KEY ARGUMENTS FROM DEBATE:\n\n"]
    for arg in sorted_args:
        speaker = arg['speaker_name']
        content = arg['content'][:120] + "..." if len(arg['content']) > 120 else arg['content']
        text_lines.append(f"[Round {arg['round']}] {speaker}:\n\"{content}\"\n\n")

    ax.text(0.02, 0.95, ''.join(text_lines),
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='sans-serif',
            wrap=True, linespacing=1.3)


def generate_dashboard(capture: DebateDataCapture, engine: AdvancedDebateEngine, output_path: str):
    """Generate full dashboard from real debate data."""

    num_participants = len(engine.state.debaters)
    # Larger figure for more participants
    fig_width = 18 if num_participants > 6 else 16
    fig_height = 14 if num_participants > 6 else 12

    fig = plt.figure(figsize=(fig_width, fig_height))
    motion_display = engine.motion[:70] + "..." if len(engine.motion) > 70 else engine.motion
    fig.suptitle(f"AI X-Risk Debate: {engine.topic}\nMotion: {motion_display}",
                 fontsize=13, fontweight='bold', y=0.98)

    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Position evolution (large, top-left spanning 2 cols)
    ax1 = fig.add_subplot(gs[0, :2])
    create_position_chart(capture, ax1, engine.motion)

    # Tension/Convergence (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    create_tension_convergence_chart(capture, ax2)

    # Argument types (middle-left)
    ax3 = fig.add_subplot(gs[1, 0])
    create_argument_types_chart(capture, ax3)

    # Final positions (middle-center)
    ax4 = fig.add_subplot(gs[1, 1])
    create_final_positions_chart(capture, ax4)

    # Emotional timeline (middle-right)
    ax5 = fig.add_subplot(gs[1, 2])
    create_emotional_timeline(capture, ax5)

    # Transcript panel (bottom, spanning all cols)
    ax6 = fig.add_subplot(gs[2, :])
    create_transcript_panel(capture, ax6)

    # Add metadata
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    participant_names = [d.name.split()[0] for d in engine.state.debaters.values()]  # First names only
    fig.text(0.02, 0.01, f"Generated: {timestamp} | Participants: {', '.join(participant_names)} | Rounds: {len(capture.rounds)}",
             fontsize=7, alpha=0.7)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Dashboard saved to: {output_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def run_real_debate(
    topic: str,
    motion: str,
    participants: List[str],
    rounds: int = 5,
    use_llm: bool = False,
    llm_provider: str = "openrouter",
) -> tuple[AdvancedDebateEngine, DebateDataCapture]:
    """Run a real debate and capture data."""

    print(f"\n{'='*60}")
    print(f"AI X-RISK DEBATE SIMULATION")
    print(f"{'='*60}")
    print(f"Topic: {topic}")
    print(f"Motion: {motion}")
    print(f"Participants: {', '.join(participants)}")
    print(f"Rounds: {rounds}")
    print(f"LLM Enabled: {use_llm}")
    print(f"{'='*60}\n")

    # Create engine
    engine = AdvancedDebateEngine(
        topic=topic,
        motion=motion,
        participants=participants,
        llm_provider=llm_provider,
        use_llm=use_llm,
    )

    # Create data capture
    capture = DebateDataCapture(engine)

    # Capture initial state (round 0)
    capture.capture_round(0, [])

    # Run debate rounds
    for round_num in range(1, rounds + 1):
        print(f"\n--- Round {round_num} ---")
        arguments = await engine.run_round()

        for arg in arguments:
            speaker_name = engine.state.debaters[arg.speaker].name
            content_preview = arg.content[:80] + "..." if len(arg.content) > 80 else arg.content
            print(f"  {speaker_name}: \"{content_preview}\"")

        capture.capture_round(round_num, arguments)

        # Brief pause between rounds
        await asyncio.sleep(0.1)

    print(f"\n{'='*60}")
    print("DEBATE COMPLETE")
    print(f"{'='*60}")

    # Print final positions
    print("\nFinal Positions:")
    for debater_id, debater in engine.state.debaters.items():
        pos = debater.beliefs.get("support_motion", 0.5)
        stance = "FOR" if pos > 0.6 else "AGAINST" if pos < 0.4 else "UNDECIDED"
        print(f"  {debater.name}: {pos:.0%} ({stance})")

    return engine, capture


async def main():
    """Run debates and generate visualizations."""

    # Ensure images directory exists
    Path("images").mkdir(exist_ok=True)

    # Check for LLM availability
    use_llm = bool(os.environ.get("OPENROUTER_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))
    if not use_llm:
        print("Note: No API key found. Running with template responses.")
        print("Set OPENROUTER_API_KEY or ANTHROPIC_API_KEY for LLM-powered debates.\n")

    # =========================================================================
    # DEBATE 1: Full X-Risk Panel (12 participants)
    # =========================================================================
    engine1, capture1 = await run_real_debate(
        topic="AI Existential Risk - Full Expert Panel",
        motion="AI development poses an existential risk and should be heavily regulated",
        participants=[
            "yudkowsky", "sam_altman", "geoffrey_hinton", "yann_lecun",
            "dario_amodei", "stuart_russell", "connor", "nick_bostrom",
            "ilya_sutskever", "max_tegmark", "paul_christiano", "gary_marcus"
        ],
        rounds=10,
        use_llm=use_llm,
    )

    generate_dashboard(
        capture1, engine1,
        "images/xrisk_debate_dashboard.png"
    )

    # =========================================================================
    # DEBATE 2: Lab Governance (8 participants)
    # =========================================================================
    engine2, capture2 = await run_real_debate(
        topic="AI Lab Governance - Extended Panel",
        motion="AI labs cannot be trusted to self-regulate on safety",
        participants=[
            "timnit_gebru", "dario_amodei", "stuart_russell", "demis_hassabis",
            "sam_altman", "jan_leike", "mustafa_suleyman", "jaan_tallinn"
        ],
        rounds=10,
        use_llm=use_llm,
    )

    generate_dashboard(
        capture2, engine2,
        "images/lab_governance_debate.png"
    )

    # =========================================================================
    # DEBATE 3: Open Source AI (8 participants)
    # =========================================================================
    engine3, capture3 = await run_real_debate(
        topic="Open Source AI Safety - Extended Panel",
        motion="Frontier AI models should be open sourced for safety through transparency",
        participants=[
            "yann_lecun", "yoshua_bengio", "andrew_ng", "connor",
            "melanie_mitchell", "katja_grace", "nick_bostrom", "ilya_sutskever"
        ],
        rounds=10,
        use_llm=use_llm,
    )

    generate_dashboard(
        capture3, engine3,
        "images/open_source_ai_debate.png"
    )

    print("\n" + "="*60)
    print("ALL VISUALIZATIONS GENERATED")
    print("="*60)
    print("\nImages saved to images/ directory:")
    print("  - xrisk_debate_dashboard.png")
    print("  - lab_governance_debate.png")
    print("  - open_source_ai_debate.png")
    print("\nThese visualizations show REAL data from actual debate simulations.")


if __name__ == "__main__":
    asyncio.run(main())
