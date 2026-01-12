"""
Visualization Module for Sophisticated Simulations

Provides real-time and post-hoc visualization of:
- Narrative tension curves
- Strategic game outcomes
- Economic market dynamics
- Consciousness metrics
- Information propagation
- Metacognitive states
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
import math

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


# ============================================================================
# DATA COLLECTORS
# ============================================================================

@dataclass
class SimulationMetrics:
    """Collects metrics from all simulation modules"""

    # Time tracking
    rounds: List[int] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)

    # Narrative metrics
    tension_values: List[float] = field(default_factory=list)
    beat_types: List[str] = field(default_factory=list)
    character_arcs: Dict[str, List[float]] = field(default_factory=dict)

    # Strategic metrics
    actions: Dict[str, List[str]] = field(default_factory=dict)
    payoffs: Dict[str, List[float]] = field(default_factory=dict)
    cumulative_payoffs: Dict[str, List[float]] = field(default_factory=dict)

    # Economic metrics
    prices: Dict[str, List[float]] = field(default_factory=dict)
    order_book_depth: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)  # (bids, asks)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    volumes: Dict[str, List[float]] = field(default_factory=dict)

    # Consciousness metrics
    phi_values: Dict[str, List[float]] = field(default_factory=dict)
    qualia_counts: Dict[str, List[int]] = field(default_factory=dict)
    attention_focus: Dict[str, List[str]] = field(default_factory=dict)

    # Information metrics
    belief_counts: Dict[str, List[int]] = field(default_factory=dict)
    belief_credences: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    information_spread: List[float] = field(default_factory=list)

    # Metacognitive metrics
    cognitive_loads: Dict[str, List[float]] = field(default_factory=dict)
    strategy_selections: Dict[str, List[str]] = field(default_factory=dict)
    confidence_levels: Dict[str, List[float]] = field(default_factory=dict)

    def record_round(self, round_num: int):
        """Record a new round"""
        self.rounds.append(round_num)
        self.timestamps.append(datetime.now())

    def record_narrative(self, tension: float, beat_type: str,
                        character_progress: Dict[str, float]):
        """Record narrative metrics"""
        self.tension_values.append(tension)
        self.beat_types.append(beat_type)
        for char_id, progress in character_progress.items():
            if char_id not in self.character_arcs:
                self.character_arcs[char_id] = []
            self.character_arcs[char_id].append(progress)

    def record_strategy(self, agent_actions: Dict[str, str],
                       agent_payoffs: Dict[str, float]):
        """Record strategic metrics"""
        for agent, action in agent_actions.items():
            if agent not in self.actions:
                self.actions[agent] = []
                self.payoffs[agent] = []
                self.cumulative_payoffs[agent] = []
            self.actions[agent].append(action)
            self.payoffs[agent].append(agent_payoffs.get(agent, 0))
            prev = self.cumulative_payoffs[agent][-1] if self.cumulative_payoffs[agent] else 0
            self.cumulative_payoffs[agent].append(prev + agent_payoffs.get(agent, 0))

    def record_economy(self, good_id: str, price: float,
                      bids: int, asks: int, volume: float = 0):
        """Record economic metrics"""
        if good_id not in self.prices:
            self.prices[good_id] = []
            self.order_book_depth[good_id] = []
            self.volumes[good_id] = []
        self.prices[good_id].append(price)
        self.order_book_depth[good_id].append((bids, asks))
        self.volumes[good_id].append(volume)

    def record_consciousness(self, agent_id: str, phi: float,
                            qualia_count: int, focus: str = ""):
        """Record consciousness metrics"""
        if agent_id not in self.phi_values:
            self.phi_values[agent_id] = []
            self.qualia_counts[agent_id] = []
            self.attention_focus[agent_id] = []
        self.phi_values[agent_id].append(phi)
        self.qualia_counts[agent_id].append(qualia_count)
        self.attention_focus[agent_id].append(focus)

    def record_information(self, agent_id: str, belief_count: int,
                          key_beliefs: Dict[str, float] = None):
        """Record information dynamics metrics"""
        if agent_id not in self.belief_counts:
            self.belief_counts[agent_id] = []
            self.belief_credences[agent_id] = {}
        self.belief_counts[agent_id].append(belief_count)
        if key_beliefs:
            for belief_id, credence in key_beliefs.items():
                if belief_id not in self.belief_credences[agent_id]:
                    self.belief_credences[agent_id][belief_id] = []
                self.belief_credences[agent_id][belief_id].append(credence)

    def record_metacognition(self, agent_id: str, cognitive_load: float,
                            strategy: str = "", confidence: float = 0.5):
        """Record metacognitive metrics"""
        if agent_id not in self.cognitive_loads:
            self.cognitive_loads[agent_id] = []
            self.strategy_selections[agent_id] = []
            self.confidence_levels[agent_id] = []
        self.cognitive_loads[agent_id].append(cognitive_load)
        self.strategy_selections[agent_id].append(strategy)
        self.confidence_levels[agent_id].append(confidence)


# ============================================================================
# VISUALIZATION ENGINE
# ============================================================================

class SimulationVisualizer:
    """
    Comprehensive visualization for simulation metrics.
    """

    # Color schemes
    COLORS = {
        'alice': '#E74C3C',  # Red
        'bob': '#3498DB',    # Blue
        'charlie': '#2ECC71', # Green
        'diana': '#9B59B6',  # Purple
        'default': '#95A5A6', # Gray
    }

    BEAT_COLORS = {
        'conflict': '#E74C3C',
        'action': '#F39C12',
        'reaction': '#3498DB',
        'revelation': '#9B59B6',
        'decision': '#2ECC71',
        'resolution': '#1ABC9C',
    }

    def __init__(self, metrics: SimulationMetrics, figsize: Tuple[int, int] = (16, 12)):
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for visualization")

        self.metrics = metrics
        self.figsize = figsize
        self.fig = None
        self.axes = {}

    def get_agent_color(self, agent_id: str) -> str:
        """Get consistent color for an agent"""
        return self.COLORS.get(agent_id.lower(), self.COLORS['default'])

    def create_dashboard(self) -> plt.Figure:
        """Create a comprehensive 6-panel dashboard"""
        self.fig = plt.figure(figsize=self.figsize)
        self.fig.suptitle('Sophisticated Simulation Dashboard', fontsize=16, fontweight='bold')

        gs = GridSpec(3, 2, figure=self.fig, hspace=0.3, wspace=0.25)

        # Create subplots
        self.axes['narrative'] = self.fig.add_subplot(gs[0, 0])
        self.axes['strategy'] = self.fig.add_subplot(gs[0, 1])
        self.axes['economy'] = self.fig.add_subplot(gs[1, 0])
        self.axes['consciousness'] = self.fig.add_subplot(gs[1, 1])
        self.axes['information'] = self.fig.add_subplot(gs[2, 0])
        self.axes['metacognition'] = self.fig.add_subplot(gs[2, 1])

        # Plot each section
        self._plot_narrative()
        self._plot_strategy()
        self._plot_economy()
        self._plot_consciousness()
        self._plot_information()
        self._plot_metacognition()

        return self.fig

    def _plot_narrative(self):
        """Plot narrative tension and character arcs"""
        ax = self.axes['narrative']
        ax.set_title('📖 Narrative Dynamics', fontweight='bold')

        rounds = self.metrics.rounds
        if not rounds:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            return

        # Plot tension curve
        if self.metrics.tension_values:
            ax.plot(rounds, self.metrics.tension_values, 'k-', linewidth=2,
                   label='Tension', marker='o')

            # Color background by beat type
            for i, (r, beat) in enumerate(zip(rounds, self.metrics.beat_types)):
                color = self.BEAT_COLORS.get(beat, '#CCCCCC')
                if i < len(rounds) - 1:
                    ax.axvspan(r - 0.4, r + 0.4, alpha=0.2, color=color)

        # Plot character arcs
        ax2 = ax.twinx()
        for char_id, progress in self.metrics.character_arcs.items():
            if len(progress) == len(rounds):
                ax2.plot(rounds, progress, '--', linewidth=1.5,
                        color=self.get_agent_color(char_id),
                        label=f'{char_id.title()} arc')

        ax.set_xlabel('Round')
        ax.set_ylabel('Tension')
        ax2.set_ylabel('Arc Progress')
        ax.set_ylim(0, 1)
        ax2.set_ylim(0, 1)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

        ax.grid(True, alpha=0.3)

    def _plot_strategy(self):
        """Plot strategic game outcomes"""
        ax = self.axes['strategy']
        ax.set_title('🎯 Strategic Outcomes', fontweight='bold')

        rounds = self.metrics.rounds
        if not rounds or not self.metrics.cumulative_payoffs:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            return

        # Plot cumulative payoffs
        for agent, payoffs in self.metrics.cumulative_payoffs.items():
            if len(payoffs) == len(rounds):
                ax.plot(rounds, payoffs, '-o', linewidth=2,
                       color=self.get_agent_color(agent),
                       label=f'{agent.title()}', markersize=8)

                # Annotate actions
                for i, (r, action) in enumerate(zip(rounds, self.metrics.actions.get(agent, []))):
                    symbol = '✓' if action == 'cooperate' else '✗'
                    ax.annotate(symbol, (r, payoffs[i]),
                               textcoords="offset points", xytext=(0, 10),
                               ha='center', fontsize=10,
                               color='green' if action == 'cooperate' else 'red')

        ax.set_xlabel('Round')
        ax.set_ylabel('Cumulative Payoff')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Add action legend
        ax.text(0.98, 0.02, '✓=Cooperate  ✗=Defect', transform=ax.transAxes,
               ha='right', va='bottom', fontsize=8, style='italic')

    def _plot_economy(self):
        """Plot economic market dynamics"""
        ax = self.axes['economy']
        ax.set_title('💰 Market Dynamics', fontweight='bold')

        rounds = self.metrics.rounds
        if not rounds or not self.metrics.order_book_depth:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            return

        # Get first good's data
        good_id = list(self.metrics.order_book_depth.keys())[0]
        depths = self.metrics.order_book_depth[good_id]

        if len(depths) == len(rounds):
            bids = [d[0] for d in depths]
            asks = [d[1] for d in depths]

            width = 0.35
            x = range(len(rounds))

            ax.bar([i - width/2 for i in x], bids, width, label='Bids', color='#2ECC71', alpha=0.8)
            ax.bar([i + width/2 for i in x], asks, width, label='Asks', color='#E74C3C', alpha=0.8)

            ax.set_xlabel('Round')
            ax.set_ylabel('Order Count')
            ax.set_xticks(x)
            ax.set_xticklabels(rounds)
            ax.legend()

            # Add price line on secondary axis if available
            if good_id in self.metrics.prices:
                ax2 = ax.twinx()
                prices = self.metrics.prices[good_id]
                if prices:
                    ax2.plot(x, prices[:len(x)], 'k--', linewidth=2, marker='s', label='Price')
                    ax2.set_ylabel('Price ($)')
                    ax2.legend(loc='upper right')

        ax.grid(True, alpha=0.3, axis='y')

    def _plot_consciousness(self):
        """Plot consciousness metrics (Phi, qualia)"""
        ax = self.axes['consciousness']
        ax.set_title('🧠 Consciousness Metrics', fontweight='bold')

        rounds = self.metrics.rounds
        if not rounds or not self.metrics.phi_values:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            return

        # Plot Phi values
        for agent, phis in self.metrics.phi_values.items():
            if len(phis) == len(rounds):
                ax.plot(rounds, phis, '-o', linewidth=2,
                       color=self.get_agent_color(agent),
                       label=f'{agent.title()} Φ')

        # Plot qualia counts on secondary axis
        ax2 = ax.twinx()
        for agent, counts in self.metrics.qualia_counts.items():
            if len(counts) == len(rounds):
                ax2.bar([r + 0.1 * list(self.metrics.qualia_counts.keys()).index(agent)
                        for r in rounds],
                       counts, width=0.15, alpha=0.5,
                       color=self.get_agent_color(agent),
                       label=f'{agent.title()} qualia')

        ax.set_xlabel('Round')
        ax.set_ylabel('Integrated Information (Φ)')
        ax2.set_ylabel('Qualia Count')

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

        ax.grid(True, alpha=0.3)

    def _plot_information(self):
        """Plot information dynamics"""
        ax = self.axes['information']
        ax.set_title('📡 Information Dynamics', fontweight='bold')

        rounds = self.metrics.rounds
        if not rounds or not self.metrics.belief_counts:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            return

        # Plot belief counts as stacked area
        agents = list(self.metrics.belief_counts.keys())
        if agents:
            belief_data = []
            for agent in agents:
                counts = self.metrics.belief_counts[agent]
                if len(counts) == len(rounds):
                    belief_data.append(counts)

            if belief_data:
                ax.stackplot(rounds, belief_data, labels=agents,
                            colors=[self.get_agent_color(a) for a in agents],
                            alpha=0.7)

        ax.set_xlabel('Round')
        ax.set_ylabel('Belief Count')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    def _plot_metacognition(self):
        """Plot metacognitive states"""
        ax = self.axes['metacognition']
        ax.set_title('🔍 Metacognitive Load', fontweight='bold')

        rounds = self.metrics.rounds
        if not rounds or not self.metrics.cognitive_loads:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            return

        # Plot cognitive load
        for agent, loads in self.metrics.cognitive_loads.items():
            if len(loads) == len(rounds):
                ax.fill_between(rounds, loads, alpha=0.3,
                               color=self.get_agent_color(agent))
                ax.plot(rounds, loads, '-o', linewidth=2,
                       color=self.get_agent_color(agent),
                       label=f'{agent.title()}')

        ax.set_xlabel('Round')
        ax.set_ylabel('Cognitive Load')
        ax.set_ylim(0, 1)
        ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Strain threshold')
        ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='Overload threshold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    def show(self):
        """Display the visualization"""
        if self.fig is None:
            self.create_dashboard()
        plt.tight_layout()
        plt.show()

    def save(self, filename: str, dpi: int = 150):
        """Save the visualization to file"""
        if self.fig is None:
            self.create_dashboard()
        plt.tight_layout()
        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Visualization saved to {filename}")


# ============================================================================
# ASCII VISUALIZATION (for terminal output)
# ============================================================================

class ASCIIVisualizer:
    """
    ASCII-based visualization for terminal output.
    Works without matplotlib.
    """

    BARS = "▁▂▃▄▅▆▇█"

    def __init__(self, metrics: SimulationMetrics, width: int = 60):
        self.metrics = metrics
        self.width = width

    def _bar_char(self, value: float, max_val: float = 1.0) -> str:
        """Convert value to bar character"""
        if max_val == 0:
            return self.BARS[0]
        normalized = min(1.0, max(0.0, value / max_val))
        index = int(normalized * (len(self.BARS) - 1))
        return self.BARS[index]

    def _horizontal_bar(self, value: float, max_val: float, width: int = 20) -> str:
        """Create horizontal bar"""
        if max_val == 0:
            filled = 0
        else:
            filled = int((value / max_val) * width)
        return "█" * filled + "░" * (width - filled)

    def render_tension_curve(self) -> str:
        """Render ASCII tension curve"""
        lines = []
        lines.append("╔══════════════════════════════════════════════════════════════╗")
        lines.append("║                    📖 NARRATIVE TENSION                      ║")
        lines.append("╠══════════════════════════════════════════════════════════════╣")

        if not self.metrics.tension_values:
            lines.append("║                       No data available                      ║")
        else:
            # Render sparkline
            spark = "║ "
            for t in self.metrics.tension_values:
                spark += self._bar_char(t)
            spark = spark.ljust(65) + "║"
            lines.append(spark)

            # Render values
            vals = "║ "
            for i, t in enumerate(self.metrics.tension_values):
                vals += f"R{i+1}:{t:.2f} "
            vals = vals.ljust(65) + "║"
            lines.append(vals)

        lines.append("╚══════════════════════════════════════════════════════════════╝")
        return "\n".join(lines)

    def render_payoff_comparison(self) -> str:
        """Render ASCII payoff comparison"""
        lines = []
        lines.append("╔══════════════════════════════════════════════════════════════╗")
        lines.append("║                    🎯 STRATEGIC PAYOFFS                      ║")
        lines.append("╠══════════════════════════════════════════════════════════════╣")

        if not self.metrics.cumulative_payoffs:
            lines.append("║                       No data available                      ║")
        else:
            max_payoff = max(max(p) for p in self.metrics.cumulative_payoffs.values() if p)
            max_payoff = max(max_payoff, 1)

            for agent, payoffs in self.metrics.cumulative_payoffs.items():
                if payoffs:
                    final = payoffs[-1]
                    bar = self._horizontal_bar(final, max_payoff, 30)
                    line = f"║ {agent.title():10} {bar} {final:6.1f} ║"
                    lines.append(line.ljust(65) + "║" if len(line) < 65 else line[:65] + "║")

        lines.append("╚══════════════════════════════════════════════════════════════╝")
        return "\n".join(lines)

    def render_action_history(self) -> str:
        """Render ASCII action history"""
        lines = []
        lines.append("╔══════════════════════════════════════════════════════════════╗")
        lines.append("║                    📋 ACTION HISTORY                         ║")
        lines.append("╠══════════════════════════════════════════════════════════════╣")

        if not self.metrics.actions:
            lines.append("║                       No data available                      ║")
        else:
            # Header
            header = "║ Round  │ "
            for agent in self.metrics.actions.keys():
                header += f"{agent.title():12}"
            lines.append(header.ljust(65) + "║")
            lines.append("║────────┼" + "─" * 53 + "║")

            # Actions
            num_rounds = len(self.metrics.rounds)
            for i in range(num_rounds):
                row = f"║   {i+1}    │ "
                for agent, actions in self.metrics.actions.items():
                    if i < len(actions):
                        symbol = "✓ COOP" if actions[i] == "cooperate" else "✗ DEFECT"
                        row += f"{symbol:12}"
                lines.append(row.ljust(65) + "║")

        lines.append("╚══════════════════════════════════════════════════════════════╝")
        return "\n".join(lines)

    def render_consciousness(self) -> str:
        """Render ASCII consciousness metrics"""
        lines = []
        lines.append("╔══════════════════════════════════════════════════════════════╗")
        lines.append("║                    🧠 CONSCIOUSNESS (Φ)                      ║")
        lines.append("╠══════════════════════════════════════════════════════════════╣")

        if not self.metrics.phi_values:
            lines.append("║                       No data available                      ║")
        else:
            for agent, phis in self.metrics.phi_values.items():
                if phis:
                    spark = f"║ {agent.title():8} │ "
                    for p in phis:
                        spark += self._bar_char(p, max(phis) if max(phis) > 0 else 1)
                    spark += f" │ Final: {phis[-1]:.3f}"
                    lines.append(spark.ljust(65) + "║")

        lines.append("╚══════════════════════════════════════════════════════════════╝")
        return "\n".join(lines)

    def render_economy(self) -> str:
        """Render ASCII economy metrics"""
        lines = []
        lines.append("╔══════════════════════════════════════════════════════════════╗")
        lines.append("║                    💰 MARKET ORDER BOOK                      ║")
        lines.append("╠══════════════════════════════════════════════════════════════╣")

        if not self.metrics.order_book_depth:
            lines.append("║                       No data available                      ║")
        else:
            for good_id, depths in self.metrics.order_book_depth.items():
                lines.append(f"║ {good_id}:".ljust(65) + "║")
                for i, (bids, asks) in enumerate(depths):
                    bid_bar = "█" * min(bids, 10)
                    ask_bar = "█" * min(asks, 10)
                    row = f"║   R{i+1}: Bids [{bid_bar:10}] {bids:2} │ Asks [{ask_bar:10}] {asks:2}"
                    lines.append(row.ljust(65) + "║")

        lines.append("╚══════════════════════════════════════════════════════════════╝")
        return "\n".join(lines)

    def render_all(self) -> str:
        """Render all ASCII visualizations"""
        sections = [
            self.render_tension_curve(),
            self.render_payoff_comparison(),
            self.render_action_history(),
            self.render_consciousness(),
            self.render_economy(),
        ]
        return "\n\n".join(sections)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_metrics() -> SimulationMetrics:
    """Create a new metrics collector"""
    return SimulationMetrics()


def visualize_simulation(metrics: SimulationMetrics,
                        use_matplotlib: bool = True,
                        save_path: str = None) -> None:
    """
    Visualize simulation metrics.

    Args:
        metrics: SimulationMetrics object with recorded data
        use_matplotlib: If True, use matplotlib; if False, use ASCII
        save_path: Optional path to save the visualization
    """
    if use_matplotlib and MATPLOTLIB_AVAILABLE:
        viz = SimulationVisualizer(metrics)
        viz.create_dashboard()
        if save_path:
            viz.save(save_path)
        else:
            viz.show()
    else:
        viz = ASCIIVisualizer(metrics)
        print(viz.render_all())


def demo_visualization():
    """Demo the visualization with sample data"""
    print("Creating sample simulation data...")

    metrics = SimulationMetrics()

    # Simulate 5 rounds
    for r in range(1, 6):
        metrics.record_round(r)

        # Narrative
        tension = 0.2 + r * 0.15 if r < 4 else 0.8 - (r - 4) * 0.3
        beat = "conflict" if r < 3 else "action" if r < 4 else "resolution"
        metrics.record_narrative(
            tension=tension,
            beat_type=beat,
            character_progress={"alice": r * 0.18, "bob": r * 0.15}
        )

        # Strategy
        alice_action = "defect" if r < 3 else "cooperate"
        bob_action = "cooperate" if r < 4 else "defect"
        alice_payoff = 5 if alice_action == "defect" and bob_action == "cooperate" else 3 if alice_action == bob_action == "cooperate" else 1
        bob_payoff = 0 if alice_action == "defect" and bob_action == "cooperate" else 3 if alice_action == bob_action == "cooperate" else 1
        metrics.record_strategy(
            {"alice": alice_action, "bob": bob_action},
            {"alice": alice_payoff, "bob": bob_payoff}
        )

        # Economy
        metrics.record_economy("AITECH", 100 + r * 5, r, r + 1)

        # Consciousness
        metrics.record_consciousness("alice", 0.1 + r * 0.05, r * 2)
        metrics.record_consciousness("bob", 0.08 + r * 0.04, r * 2 + 1)

        # Information
        metrics.record_information("alice", r * 2, {"market_up": 0.5 + r * 0.1})
        metrics.record_information("bob", r * 2, {"alice_coop": 0.7 - r * 0.1})

        # Metacognition
        metrics.record_metacognition("alice", 0.3 + r * 0.05)
        metrics.record_metacognition("bob", 0.25 + r * 0.08)

    print("Sample data created. Generating visualization...")
    print()

    # Show ASCII visualization
    ascii_viz = ASCIIVisualizer(metrics)
    print(ascii_viz.render_all())

    # Try matplotlib if available
    if MATPLOTLIB_AVAILABLE:
        print("\nGenerating matplotlib dashboard...")
        viz = SimulationVisualizer(metrics)
        viz.create_dashboard()
        viz.save("/tmp/simulation_dashboard.png")
        print("Dashboard saved to /tmp/simulation_dashboard.png")


if __name__ == "__main__":
    demo_visualization()
