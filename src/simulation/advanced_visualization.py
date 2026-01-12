"""
Advanced Visualization Module for Sophisticated Simulations

Features:
- Network graphs for agent relationships and information flow
- Phase space portraits for dynamic systems
- Entropy and complexity metrics
- Correlation heatmaps
- Multi-dimensional radar charts
- Animated time evolution
- Statistical analysis and trend detection
- Emergence and pattern detection
- Interactive dashboards
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
from enum import Enum
from collections import defaultdict
import math
import random

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    import matplotlib.animation as animation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ============================================================================
# ADVANCED METRICS AND ANALYSIS
# ============================================================================

class ComplexityMetric(Enum):
    """Types of complexity metrics"""
    SHANNON_ENTROPY = "shannon_entropy"
    KOLMOGOROV = "kolmogorov"
    STATISTICAL = "statistical"
    EMERGENCE = "emergence"
    INTEGRATION = "integration"


@dataclass
class TimeSeriesAnalysis:
    """Analysis results for a time series"""
    mean: float = 0.0
    std: float = 0.0
    trend: float = 0.0  # Slope of linear fit
    volatility: float = 0.0
    autocorrelation: float = 0.0
    entropy: float = 0.0
    stationarity: bool = True
    change_points: List[int] = field(default_factory=list)
    anomalies: List[int] = field(default_factory=list)


@dataclass
class RelationshipMetrics:
    """Metrics for agent relationships"""
    cooperation_rate: float = 0.0
    exploitation_rate: float = 0.0
    reciprocity: float = 0.0
    trust_level: float = 0.5
    influence_strength: float = 0.0
    information_flow: float = 0.0


@dataclass
class EmergenceMetrics:
    """Metrics for detecting emergent behavior"""
    collective_complexity: float = 0.0
    downward_causation: float = 0.0
    synergy: float = 0.0
    redundancy: float = 0.0
    unique_information: float = 0.0
    emergence_score: float = 0.0


@dataclass
class AdvancedSimulationMetrics:
    """Extended metrics collection with advanced analytics"""

    # Basic tracking
    rounds: List[int] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)

    # Agent data
    agents: List[str] = field(default_factory=list)

    # Multi-dimensional state vectors per agent per round
    state_vectors: Dict[str, List[List[float]]] = field(default_factory=dict)

    # Relationship matrix (agent x agent x round)
    relationships: Dict[Tuple[str, str], List[RelationshipMetrics]] = field(default_factory=dict)

    # Action sequences for pattern detection
    action_sequences: Dict[str, List[str]] = field(default_factory=dict)

    # Payoff matrices over time
    payoff_history: List[Dict[str, Dict[str, float]]] = field(default_factory=list)

    # Information flow graph (who influenced whom)
    information_flow: List[Dict[Tuple[str, str], float]] = field(default_factory=list)

    # Belief divergence between agents
    belief_divergence: List[Dict[Tuple[str, str], float]] = field(default_factory=list)

    # System-level metrics
    system_entropy: List[float] = field(default_factory=list)
    system_complexity: List[float] = field(default_factory=list)
    emergence_scores: List[float] = field(default_factory=list)

    # Phase space trajectories
    phase_trajectories: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)

    # Event log with rich metadata
    events: List[Dict[str, Any]] = field(default_factory=list)

    # Narrative metrics
    tension_curve: List[float] = field(default_factory=list)
    dramatic_beats: List[Dict[str, Any]] = field(default_factory=list)
    character_arcs: Dict[str, List[float]] = field(default_factory=dict)

    # Economic metrics
    market_prices: Dict[str, List[float]] = field(default_factory=dict)
    order_flow: Dict[str, List[Tuple[int, int, int]]] = field(default_factory=dict)  # (bids, asks, trades)
    wealth_distribution: List[Dict[str, float]] = field(default_factory=list)

    # Consciousness metrics
    phi_values: Dict[str, List[float]] = field(default_factory=dict)
    qualia_complexity: Dict[str, List[float]] = field(default_factory=dict)
    attention_entropy: Dict[str, List[float]] = field(default_factory=dict)

    # Strategic metrics
    strategy_entropy: Dict[str, List[float]] = field(default_factory=dict)
    nash_distance: List[float] = field(default_factory=list)
    pareto_efficiency: List[float] = field(default_factory=list)

    # Metacognitive metrics
    cognitive_load: Dict[str, List[float]] = field(default_factory=dict)
    confidence_calibration: Dict[str, List[float]] = field(default_factory=dict)
    learning_rate: Dict[str, List[float]] = field(default_factory=dict)

    def register_agent(self, agent_id: str):
        """Register a new agent"""
        if agent_id not in self.agents:
            self.agents.append(agent_id)
            self.state_vectors[agent_id] = []
            self.action_sequences[agent_id] = []
            self.phase_trajectories[agent_id] = []
            self.character_arcs[agent_id] = []
            self.phi_values[agent_id] = []
            self.qualia_complexity[agent_id] = []
            self.attention_entropy[agent_id] = []
            self.strategy_entropy[agent_id] = []
            self.cognitive_load[agent_id] = []
            self.confidence_calibration[agent_id] = []
            self.learning_rate[agent_id] = []

    def record_round(self, round_num: int):
        """Start recording a new round"""
        self.rounds.append(round_num)
        self.timestamps.append(datetime.now())

    def record_state_vector(self, agent_id: str, state: List[float]):
        """Record multi-dimensional state for an agent"""
        if agent_id not in self.state_vectors:
            self.register_agent(agent_id)
        self.state_vectors[agent_id].append(state)

        # Also record phase trajectory (first two dimensions)
        if len(state) >= 2:
            self.phase_trajectories[agent_id].append((state[0], state[1]))

    def record_action(self, agent_id: str, action: str):
        """Record an action for pattern analysis"""
        if agent_id not in self.action_sequences:
            self.register_agent(agent_id)
        self.action_sequences[agent_id].append(action)

    def record_relationship(self, agent1: str, agent2: str, metrics: RelationshipMetrics):
        """Record relationship metrics between two agents"""
        key = (agent1, agent2)
        if key not in self.relationships:
            self.relationships[key] = []
        self.relationships[key].append(metrics)

    def record_information_flow(self, flows: Dict[Tuple[str, str], float]):
        """Record information flow between agents"""
        self.information_flow.append(flows)

    def record_system_metrics(self, entropy: float, complexity: float, emergence: float):
        """Record system-level metrics"""
        self.system_entropy.append(entropy)
        self.system_complexity.append(complexity)
        self.emergence_scores.append(emergence)

    def record_event(self, event_type: str, agents: List[str],
                    description: str, metadata: Dict[str, Any] = None):
        """Record a significant event"""
        self.events.append({
            'round': self.rounds[-1] if self.rounds else 0,
            'timestamp': datetime.now(),
            'type': event_type,
            'agents': agents,
            'description': description,
            'metadata': metadata or {}
        })

    def record_narrative(self, tension: float, beat: Dict[str, Any],
                        arc_progress: Dict[str, float]):
        """Record narrative metrics"""
        self.tension_curve.append(tension)
        self.dramatic_beats.append(beat)
        for agent, progress in arc_progress.items():
            if agent not in self.character_arcs:
                self.character_arcs[agent] = []
            self.character_arcs[agent].append(progress)

    def record_market(self, good_id: str, price: float,
                     bids: int, asks: int, trades: int,
                     wealth: Dict[str, float]):
        """Record market metrics"""
        if good_id not in self.market_prices:
            self.market_prices[good_id] = []
            self.order_flow[good_id] = []
        self.market_prices[good_id].append(price)
        self.order_flow[good_id].append((bids, asks, trades))
        self.wealth_distribution.append(wealth)

    def record_consciousness(self, agent_id: str, phi: float,
                            qualia_comp: float, attn_entropy: float):
        """Record consciousness metrics"""
        if agent_id not in self.phi_values:
            self.register_agent(agent_id)
        self.phi_values[agent_id].append(phi)
        self.qualia_complexity[agent_id].append(qualia_comp)
        self.attention_entropy[agent_id].append(attn_entropy)

    def record_strategy(self, agent_id: str, strategy_ent: float,
                       nash_dist: float = 0, pareto_eff: float = 0):
        """Record strategic metrics"""
        if agent_id not in self.strategy_entropy:
            self.register_agent(agent_id)
        self.strategy_entropy[agent_id].append(strategy_ent)
        if nash_dist > 0:
            self.nash_distance.append(nash_dist)
        if pareto_eff > 0:
            self.pareto_efficiency.append(pareto_eff)

    def record_metacognition(self, agent_id: str, load: float,
                            calibration: float, lr: float):
        """Record metacognitive metrics"""
        if agent_id not in self.cognitive_load:
            self.register_agent(agent_id)
        self.cognitive_load[agent_id].append(load)
        self.confidence_calibration[agent_id].append(calibration)
        self.learning_rate[agent_id].append(lr)


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

class MetricsAnalyzer:
    """Analyzes simulation metrics for patterns and insights"""

    def __init__(self, metrics: AdvancedSimulationMetrics):
        self.metrics = metrics

    def calculate_entropy(self, sequence: List[Any]) -> float:
        """Calculate Shannon entropy of a sequence"""
        if not sequence:
            return 0.0

        counts = defaultdict(int)
        for item in sequence:
            counts[item] += 1

        total = len(sequence)
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        return entropy

    def calculate_mutual_information(self, seq1: List[Any], seq2: List[Any]) -> float:
        """Calculate mutual information between two sequences"""
        if not seq1 or not seq2 or len(seq1) != len(seq2):
            return 0.0

        # Joint distribution
        joint_counts = defaultdict(int)
        for a, b in zip(seq1, seq2):
            joint_counts[(a, b)] += 1

        # Marginal distributions
        counts1 = defaultdict(int)
        counts2 = defaultdict(int)
        for a in seq1:
            counts1[a] += 1
        for b in seq2:
            counts2[b] += 1

        total = len(seq1)
        mi = 0.0

        for (a, b), count in joint_counts.items():
            p_joint = count / total
            p_a = counts1[a] / total
            p_b = counts2[b] / total
            if p_joint > 0 and p_a > 0 and p_b > 0:
                mi += p_joint * math.log2(p_joint / (p_a * p_b))

        return mi

    def detect_patterns(self, sequence: List[str], max_length: int = 5) -> Dict[str, int]:
        """Detect recurring patterns in a sequence"""
        patterns = defaultdict(int)

        for length in range(2, min(max_length + 1, len(sequence))):
            for i in range(len(sequence) - length + 1):
                pattern = tuple(sequence[i:i+length])
                patterns[pattern] += 1

        # Filter to patterns that occur more than once
        return {str(k): v for k, v in patterns.items() if v > 1}

    def analyze_time_series(self, data: List[float]) -> TimeSeriesAnalysis:
        """Comprehensive time series analysis"""
        if not data or len(data) < 2:
            return TimeSeriesAnalysis()

        analysis = TimeSeriesAnalysis()

        # Basic statistics
        analysis.mean = sum(data) / len(data)
        analysis.std = math.sqrt(sum((x - analysis.mean) ** 2 for x in data) / len(data))

        # Trend (simple linear regression slope)
        n = len(data)
        x_mean = (n - 1) / 2
        numerator = sum((i - x_mean) * (data[i] - analysis.mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        analysis.trend = numerator / denominator if denominator != 0 else 0

        # Volatility (standard deviation of returns)
        if len(data) > 1:
            returns = [(data[i] - data[i-1]) / data[i-1] if data[i-1] != 0 else 0
                      for i in range(1, len(data))]
            if returns:
                ret_mean = sum(returns) / len(returns)
                analysis.volatility = math.sqrt(sum((r - ret_mean) ** 2 for r in returns) / len(returns))

        # Autocorrelation (lag-1)
        if len(data) > 2:
            data_centered = [x - analysis.mean for x in data]
            numerator = sum(data_centered[i] * data_centered[i+1] for i in range(len(data) - 1))
            denominator = sum(x ** 2 for x in data_centered)
            analysis.autocorrelation = numerator / denominator if denominator != 0 else 0

        # Entropy (binned)
        if analysis.std > 0:
            bins = 10
            bin_width = (max(data) - min(data)) / bins if max(data) != min(data) else 1
            binned = [int((x - min(data)) / bin_width) for x in data]
            binned = [min(b, bins - 1) for b in binned]
            analysis.entropy = self.calculate_entropy(binned)

        # Change point detection (simple threshold-based)
        if len(data) > 3:
            threshold = 2 * analysis.std
            for i in range(1, len(data) - 1):
                if abs(data[i] - data[i-1]) > threshold:
                    analysis.change_points.append(i)

        # Anomaly detection (simple z-score)
        if analysis.std > 0:
            for i, x in enumerate(data):
                z_score = abs(x - analysis.mean) / analysis.std
                if z_score > 2.5:
                    analysis.anomalies.append(i)

        return analysis

    def calculate_emergence(self) -> EmergenceMetrics:
        """Calculate emergence metrics for the system"""
        emergence = EmergenceMetrics()

        if not self.metrics.agents or not self.metrics.action_sequences:
            return emergence

        # Calculate individual entropies
        individual_entropies = []
        for agent in self.metrics.agents:
            if agent in self.metrics.action_sequences:
                ent = self.calculate_entropy(self.metrics.action_sequences[agent])
                individual_entropies.append(ent)

        if not individual_entropies:
            return emergence

        avg_individual = sum(individual_entropies) / len(individual_entropies)

        # Calculate joint entropy (all agents combined)
        if len(self.metrics.agents) >= 2:
            combined = []
            min_len = min(len(self.metrics.action_sequences.get(a, []))
                         for a in self.metrics.agents)
            for i in range(min_len):
                joint_action = tuple(self.metrics.action_sequences[a][i]
                                    for a in self.metrics.agents
                                    if a in self.metrics.action_sequences)
                combined.append(joint_action)

            joint_entropy = self.calculate_entropy(combined)

            # Synergy: Joint entropy that can't be explained by parts
            emergence.synergy = max(0, joint_entropy - sum(individual_entropies))

            # Redundancy: Shared information
            emergence.redundancy = max(0, sum(individual_entropies) - joint_entropy)

        # Collective complexity
        emergence.collective_complexity = avg_individual * (1 + emergence.synergy)

        # Emergence score (normalized)
        max_entropy = math.log2(len(set(self.metrics.action_sequences.get(
            self.metrics.agents[0], ['a', 'b']))))
        if max_entropy > 0:
            emergence.emergence_score = emergence.synergy / max_entropy

        return emergence

    def get_relationship_summary(self, agent1: str, agent2: str) -> Dict[str, float]:
        """Get summary of relationship between two agents"""
        key = (agent1, agent2)
        if key not in self.metrics.relationships:
            return {}

        rels = self.metrics.relationships[key]
        if not rels:
            return {}

        return {
            'avg_cooperation': sum(r.cooperation_rate for r in rels) / len(rels),
            'avg_exploitation': sum(r.exploitation_rate for r in rels) / len(rels),
            'avg_reciprocity': sum(r.reciprocity for r in rels) / len(rels),
            'final_trust': rels[-1].trust_level,
            'influence_trend': rels[-1].influence_strength - rels[0].influence_strength
        }


# ============================================================================
# ADVANCED VISUALIZER
# ============================================================================

class AdvancedSimulationVisualizer:
    """
    Sophisticated visualization with advanced charts and analytics.
    """

    AGENT_COLORS = {
        'alice': '#E74C3C',
        'bob': '#3498DB',
        'charlie': '#2ECC71',
        'diana': '#9B59B6',
        'eve': '#F39C12',
        'frank': '#1ABC9C',
    }

    CMAP_DIVERGING = 'RdYlBu_r'
    CMAP_SEQUENTIAL = 'viridis'

    def __init__(self, metrics: AdvancedSimulationMetrics):
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for visualization")

        self.metrics = metrics
        self.analyzer = MetricsAnalyzer(metrics)
        self.fig = None
        self.axes = {}

    def get_color(self, agent: str) -> str:
        """Get consistent color for agent"""
        if agent.lower() in self.AGENT_COLORS:
            return self.AGENT_COLORS[agent.lower()]
        # Generate consistent color from hash
        h = hash(agent) % 360
        return f'#{int(128 + 127 * math.cos(h * math.pi / 180)):02x}' \
               f'{int(128 + 127 * math.cos((h + 120) * math.pi / 180)):02x}' \
               f'{int(128 + 127 * math.cos((h + 240) * math.pi / 180)):02x}'

    def create_comprehensive_dashboard(self, figsize: Tuple[int, int] = (20, 16)) -> plt.Figure:
        """Create a comprehensive 12-panel dashboard"""
        self.fig = plt.figure(figsize=figsize)
        self.fig.suptitle('Advanced Simulation Analytics Dashboard',
                         fontsize=18, fontweight='bold', y=0.98)

        # Create 4x3 grid
        gs = GridSpec(4, 3, figure=self.fig, hspace=0.35, wspace=0.3,
                     left=0.05, right=0.95, top=0.93, bottom=0.05)

        # Row 1: Overview
        self.axes['phase_space'] = self.fig.add_subplot(gs[0, 0])
        self.axes['tension_arc'] = self.fig.add_subplot(gs[0, 1])
        self.axes['emergence'] = self.fig.add_subplot(gs[0, 2])

        # Row 2: Strategic
        self.axes['payoff_evolution'] = self.fig.add_subplot(gs[1, 0])
        self.axes['action_patterns'] = self.fig.add_subplot(gs[1, 1])
        self.axes['strategy_radar'] = self.fig.add_subplot(gs[1, 2], projection='polar')

        # Row 3: Information & Consciousness
        self.axes['information_flow'] = self.fig.add_subplot(gs[2, 0])
        self.axes['consciousness'] = self.fig.add_subplot(gs[2, 1])
        self.axes['belief_heatmap'] = self.fig.add_subplot(gs[2, 2])

        # Row 4: Economic & Meta
        self.axes['market_depth'] = self.fig.add_subplot(gs[3, 0])
        self.axes['wealth_gini'] = self.fig.add_subplot(gs[3, 1])
        self.axes['metacognitive'] = self.fig.add_subplot(gs[3, 2])

        # Plot all sections
        self._plot_phase_space()
        self._plot_tension_arc()
        self._plot_emergence()
        self._plot_payoff_evolution()
        self._plot_action_patterns()
        self._plot_strategy_radar()
        self._plot_information_flow()
        self._plot_consciousness()
        self._plot_belief_heatmap()
        self._plot_market_depth()
        self._plot_wealth_gini()
        self._plot_metacognitive()

        return self.fig

    def _plot_phase_space(self):
        """Plot phase space trajectories"""
        ax = self.axes['phase_space']
        ax.set_title('Phase Space Trajectories', fontweight='bold')

        if not self.metrics.phase_trajectories:
            ax.text(0.5, 0.5, 'No trajectory data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            return

        for agent, trajectory in self.metrics.phase_trajectories.items():
            if len(trajectory) < 2:
                continue

            x = [p[0] for p in trajectory]
            y = [p[1] for p in trajectory]
            color = self.get_color(agent)

            # Plot trajectory with gradient
            points = list(zip(x, y))
            segments = [(points[i], points[i+1]) for i in range(len(points)-1)]

            if segments:
                lc = LineCollection(segments, cmap='plasma', alpha=0.7)
                lc.set_array(list(range(len(segments))))
                lc.set_linewidth(2)
                ax.add_collection(lc)

            # Mark start and end
            ax.scatter(x[0], y[0], c=color, s=100, marker='o',
                      edgecolors='black', linewidths=2, label=f'{agent} start', zorder=5)
            ax.scatter(x[-1], y[-1], c=color, s=100, marker='s',
                      edgecolors='black', linewidths=2, zorder=5)

        ax.set_xlabel('Dimension 1 (e.g., Cooperation)')
        ax.set_ylabel('Dimension 2 (e.g., Trust)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.autoscale()

    def _plot_tension_arc(self):
        """Plot narrative tension with dramatic structure"""
        ax = self.axes['tension_arc']
        ax.set_title('Narrative Tension & Character Arcs', fontweight='bold')

        rounds = self.metrics.rounds
        if not rounds or not self.metrics.tension_curve:
            ax.text(0.5, 0.5, 'No narrative data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            return

        # Tension curve with area fill
        tension = self.metrics.tension_curve[:len(rounds)]
        ax.fill_between(rounds[:len(tension)], tension, alpha=0.3, color='purple')
        ax.plot(rounds[:len(tension)], tension, 'purple', linewidth=3,
               label='Tension', marker='o')

        # Character arcs on secondary axis
        ax2 = ax.twinx()
        for agent, arcs in self.metrics.character_arcs.items():
            if len(arcs) >= len(rounds):
                ax2.plot(rounds, arcs[:len(rounds)], '--', linewidth=2,
                        color=self.get_color(agent), label=f'{agent.title()} arc')

        # Mark dramatic beats
        for i, beat in enumerate(self.metrics.dramatic_beats[:len(rounds)]):
            if i < len(rounds):
                beat_type = beat.get('type', 'unknown')
                if beat_type == 'conflict':
                    ax.axvline(rounds[i], color='red', alpha=0.3, linestyle=':')
                elif beat_type == 'resolution':
                    ax.axvline(rounds[i], color='green', alpha=0.3, linestyle=':')

        ax.set_xlabel('Round')
        ax.set_ylabel('Tension', color='purple')
        ax2.set_ylabel('Arc Progress')
        ax.set_ylim(0, 1)
        ax2.set_ylim(0, 1)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_emergence(self):
        """Plot emergence and complexity metrics"""
        ax = self.axes['emergence']
        ax.set_title('System Emergence & Complexity', fontweight='bold')

        rounds = self.metrics.rounds
        if not rounds:
            ax.text(0.5, 0.5, 'No emergence data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            return

        # System metrics
        if self.metrics.system_entropy:
            entropy = self.metrics.system_entropy[:len(rounds)]
            ax.plot(rounds[:len(entropy)], entropy, 'b-', linewidth=2,
                   label='Entropy', marker='o')

        if self.metrics.system_complexity:
            complexity = self.metrics.system_complexity[:len(rounds)]
            ax.plot(rounds[:len(complexity)], complexity, 'g-', linewidth=2,
                   label='Complexity', marker='s')

        if self.metrics.emergence_scores:
            emergence = self.metrics.emergence_scores[:len(rounds)]
            ax.plot(rounds[:len(emergence)], emergence, 'r-', linewidth=2,
                   label='Emergence', marker='^')

        # Calculate and display current emergence
        em = self.analyzer.calculate_emergence()
        ax.text(0.98, 0.98, f'Synergy: {em.synergy:.3f}\nEmergence: {em.emergence_score:.3f}',
               transform=ax.transAxes, ha='right', va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Round')
        ax.set_ylabel('Metric Value')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_payoff_evolution(self):
        """Plot payoff evolution with analysis"""
        ax = self.axes['payoff_evolution']
        ax.set_title('Strategic Payoff Evolution', fontweight='bold')

        rounds = self.metrics.rounds
        if not rounds or not self.metrics.payoff_history:
            ax.text(0.5, 0.5, 'No payoff data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            return

        # Cumulative payoffs
        cumulative = defaultdict(list)
        for agent in self.metrics.agents:
            total = 0
            for payoffs in self.metrics.payoff_history:
                for other_agent, agent_payoffs in payoffs.items():
                    if agent in agent_payoffs:
                        total += agent_payoffs[agent]
                cumulative[agent].append(total)

        for agent, payoffs in cumulative.items():
            color = self.get_color(agent)
            ax.plot(rounds[:len(payoffs)], payoffs, '-o', linewidth=2,
                   color=color, label=agent.title(), markersize=6)

            # Add trend analysis
            if len(payoffs) > 2:
                analysis = self.analyzer.analyze_time_series(payoffs)
                # Show trend arrow
                if analysis.trend > 0.1:
                    ax.annotate('', xy=(rounds[-1], payoffs[-1]),
                               xytext=(rounds[-1] - 1, payoffs[-1] - analysis.trend),
                               arrowprops=dict(arrowstyle='->', color=color, lw=2))

        ax.set_xlabel('Round')
        ax.set_ylabel('Cumulative Payoff')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_action_patterns(self):
        """Plot action sequence patterns"""
        ax = self.axes['action_patterns']
        ax.set_title('Action Sequence Patterns', fontweight='bold')

        if not self.metrics.action_sequences:
            ax.text(0.5, 0.5, 'No action data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            return

        # Create action matrix visualization
        agents = list(self.metrics.action_sequences.keys())
        max_len = max(len(seq) for seq in self.metrics.action_sequences.values())

        # Map actions to numbers
        all_actions = set()
        for seq in self.metrics.action_sequences.values():
            all_actions.update(seq)
        action_map = {a: i for i, a in enumerate(sorted(all_actions))}

        # Create matrix
        matrix = []
        for agent in agents:
            seq = self.metrics.action_sequences[agent]
            row = [action_map.get(a, -1) for a in seq]
            row.extend([-1] * (max_len - len(row)))
            matrix.append(row)

        if matrix:
            im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn',
                          interpolation='nearest')
            ax.set_yticks(range(len(agents)))
            ax.set_yticklabels([a.title() for a in agents])
            ax.set_xlabel('Round')
            ax.set_ylabel('Agent')

            # Add colorbar with action labels
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_ticks(list(action_map.values()))
            cbar.set_ticklabels(list(action_map.keys()))

        # Annotate patterns
        for i, agent in enumerate(agents):
            patterns = self.analyzer.detect_patterns(self.metrics.action_sequences[agent])
            if patterns:
                most_common = max(patterns.items(), key=lambda x: x[1])
                ax.text(max_len + 0.5, i, f'Pattern: {most_common[0][:20]}',
                       va='center', fontsize=7)

    def _plot_strategy_radar(self):
        """Plot multi-dimensional strategy profile as radar chart"""
        ax = self.axes['strategy_radar']
        ax.set_title('Agent Strategy Profiles', fontweight='bold', pad=20)

        # Define strategy dimensions
        dimensions = ['Cooperation', 'Aggression', 'Adaptability',
                     'Risk-taking', 'Exploitation']
        num_dims = len(dimensions)

        # Calculate angles
        angles = [n / float(num_dims) * 2 * math.pi for n in range(num_dims)]
        angles += angles[:1]  # Complete the loop

        for agent in self.metrics.agents:
            # Calculate dimension values from actions
            actions = self.metrics.action_sequences.get(agent, [])
            if not actions:
                continue

            coop_rate = actions.count('cooperate') / len(actions) if actions else 0.5
            defect_rate = actions.count('defect') / len(actions) if actions else 0.5

            # Simulate other dimensions
            values = [
                coop_rate,  # Cooperation
                defect_rate,  # Aggression
                self.analyzer.calculate_entropy(actions) / 2,  # Adaptability
                0.5 + random.uniform(-0.2, 0.2),  # Risk-taking
                defect_rate * 0.8,  # Exploitation
            ]
            values += values[:1]  # Complete the loop

            color = self.get_color(agent)
            ax.plot(angles, values, 'o-', linewidth=2, color=color,
                   label=agent.title(), markersize=4)
            ax.fill(angles, values, alpha=0.25, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimensions, size=8)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)

    def _plot_information_flow(self):
        """Plot information flow network"""
        ax = self.axes['information_flow']
        ax.set_title('Information Flow Network', fontweight='bold')

        if not self.metrics.agents or len(self.metrics.agents) < 2:
            ax.text(0.5, 0.5, 'Need multiple agents', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            return

        # Position agents in a circle
        n_agents = len(self.metrics.agents)
        positions = {}
        for i, agent in enumerate(self.metrics.agents):
            angle = 2 * math.pi * i / n_agents
            positions[agent] = (math.cos(angle), math.sin(angle))

        # Draw agents as circles
        for agent, (x, y) in positions.items():
            circle = Circle((x, y), 0.15, color=self.get_color(agent), alpha=0.8)
            ax.add_patch(circle)
            ax.text(x, y, agent[:3].upper(), ha='center', va='center',
                   fontweight='bold', fontsize=10, color='white')

        # Draw information flow arrows
        if self.metrics.information_flow:
            latest_flow = self.metrics.information_flow[-1]
            max_flow = max(latest_flow.values()) if latest_flow else 1

            for (src, dst), flow in latest_flow.items():
                if src in positions and dst in positions:
                    src_pos = positions[src]
                    dst_pos = positions[dst]

                    # Arrow with width proportional to flow
                    width = 0.02 + 0.08 * (flow / max_flow)
                    ax.annotate('', xy=dst_pos, xytext=src_pos,
                               arrowprops=dict(arrowstyle='->', color='gray',
                                             lw=width * 20, alpha=0.6))

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')

    def _plot_consciousness(self):
        """Plot consciousness metrics (Phi, qualia)"""
        ax = self.axes['consciousness']
        ax.set_title('Consciousness Metrics (IIT)', fontweight='bold')

        rounds = self.metrics.rounds
        if not rounds or not self.metrics.phi_values:
            ax.text(0.5, 0.5, 'No consciousness data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            return

        # Plot Phi values
        for agent, phis in self.metrics.phi_values.items():
            if phis:
                color = self.get_color(agent)
                ax.plot(rounds[:len(phis)], phis, '-o', linewidth=2,
                       color=color, label=f'{agent.title()} Φ')

        # Qualia complexity on secondary axis
        ax2 = ax.twinx()
        for agent, quals in self.metrics.qualia_complexity.items():
            if quals:
                color = self.get_color(agent)
                ax2.bar([r + 0.1 * self.metrics.agents.index(agent) for r in rounds[:len(quals)]],
                       quals, width=0.15, alpha=0.4, color=color)

        ax.set_xlabel('Round')
        ax.set_ylabel('Integrated Information (Φ)')
        ax2.set_ylabel('Qualia Complexity')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_belief_heatmap(self):
        """Plot belief divergence heatmap"""
        ax = self.axes['belief_heatmap']
        ax.set_title('Belief Divergence Matrix', fontweight='bold')

        agents = self.metrics.agents
        if len(agents) < 2:
            ax.text(0.5, 0.5, 'Need multiple agents', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            return

        # Create divergence matrix
        n = len(agents)
        matrix = [[0.0] * n for _ in range(n)]

        if self.metrics.belief_divergence:
            latest = self.metrics.belief_divergence[-1]
            for (a1, a2), div in latest.items():
                if a1 in agents and a2 in agents:
                    i, j = agents.index(a1), agents.index(a2)
                    matrix[i][j] = div
                    matrix[j][i] = div

        im = ax.imshow(matrix, cmap=self.CMAP_DIVERGING, aspect='equal')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([a.title() for a in agents], rotation=45, ha='right')
        ax.set_yticklabels([a.title() for a in agents])

        # Add values
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f'{matrix[i][j]:.2f}', ha='center', va='center',
                       fontsize=9, color='white' if matrix[i][j] > 0.5 else 'black')

        plt.colorbar(im, ax=ax, shrink=0.8, label='Divergence')

    def _plot_market_depth(self):
        """Plot market depth and order flow"""
        ax = self.axes['market_depth']
        ax.set_title('Market Depth & Price', fontweight='bold')

        rounds = self.metrics.rounds
        if not rounds or not self.metrics.order_flow:
            ax.text(0.5, 0.5, 'No market data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            return

        # Get first market's data
        good_id = list(self.metrics.order_flow.keys())[0]
        order_data = self.metrics.order_flow[good_id]

        bids = [d[0] for d in order_data]
        asks = [d[1] for d in order_data]
        trades = [d[2] for d in order_data]

        x = list(range(len(bids)))

        # Stacked bar for order book
        ax.bar(x, bids, label='Bids', color='#2ECC71', alpha=0.8)
        ax.bar(x, [-a for a in asks], label='Asks', color='#E74C3C', alpha=0.8)

        # Price line on secondary axis
        ax2 = ax.twinx()
        if good_id in self.metrics.market_prices:
            prices = self.metrics.market_prices[good_id]
            ax2.plot(x[:len(prices)], prices, 'k-', linewidth=2,
                    marker='D', label='Price', markersize=4)
            ax2.set_ylabel('Price')

        ax.set_xlabel('Round')
        ax.set_ylabel('Order Count')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_wealth_gini(self):
        """Plot wealth distribution and Gini coefficient"""
        ax = self.axes['wealth_gini']
        ax.set_title('Wealth Distribution & Inequality', fontweight='bold')

        if not self.metrics.wealth_distribution:
            ax.text(0.5, 0.5, 'No wealth data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            return

        rounds = self.metrics.rounds
        gini_values = []

        for wealth in self.metrics.wealth_distribution:
            values = sorted(wealth.values())
            if values and sum(values) > 0:
                n = len(values)
                numerator = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(values))
                gini = numerator / (n * sum(values))
                gini_values.append(gini)
            else:
                gini_values.append(0)

        # Plot Gini over time
        ax.plot(rounds[:len(gini_values)], gini_values, 'r-o', linewidth=2,
               label='Gini Coefficient')
        ax.axhline(0.3, color='orange', linestyle='--', alpha=0.5, label='Moderate inequality')
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='High inequality')

        ax.set_xlabel('Round')
        ax.set_ylabel('Gini Coefficient')
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_metacognitive(self):
        """Plot metacognitive metrics"""
        ax = self.axes['metacognitive']
        ax.set_title('Metacognitive Load & Calibration', fontweight='bold')

        rounds = self.metrics.rounds
        if not rounds or not self.metrics.cognitive_load:
            ax.text(0.5, 0.5, 'No metacognitive data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            return

        # Plot cognitive load
        for agent, loads in self.metrics.cognitive_load.items():
            if loads:
                color = self.get_color(agent)
                ax.fill_between(rounds[:len(loads)], loads, alpha=0.3, color=color)
                ax.plot(rounds[:len(loads)], loads, '-', linewidth=2,
                       color=color, label=f'{agent.title()} load')

        # Thresholds
        ax.axhline(0.7, color='orange', linestyle='--', alpha=0.5)
        ax.axhline(0.9, color='red', linestyle='--', alpha=0.5)

        # Confidence calibration on secondary axis
        ax2 = ax.twinx()
        for agent, calibs in self.metrics.confidence_calibration.items():
            if calibs:
                color = self.get_color(agent)
                ax2.plot(rounds[:len(calibs)], calibs, ':', linewidth=2,
                        color=color, alpha=0.7)

        ax.set_xlabel('Round')
        ax.set_ylabel('Cognitive Load')
        ax2.set_ylabel('Calibration', alpha=0.7)
        ax.set_ylim(0, 1)
        ax2.set_ylim(0, 1)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    def show(self):
        """Display the visualization"""
        if self.fig is None:
            self.create_comprehensive_dashboard()
        plt.show()

    def save(self, filename: str, dpi: int = 150):
        """Save visualization to file"""
        if self.fig is None:
            self.create_comprehensive_dashboard()
        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
        print(f"Advanced dashboard saved to {filename}")


# ============================================================================
# ADVANCED ASCII VISUALIZER
# ============================================================================

class AdvancedASCIIVisualizer:
    """
    Sophisticated ASCII visualization with Unicode graphics.
    """

    BLOCKS = " ▁▂▃▄▅▆▇█"
    SHADES = " ░▒▓█"
    ARROWS = "←↑→↓↖↗↘↙"

    def __init__(self, metrics: AdvancedSimulationMetrics, width: int = 80):
        self.metrics = metrics
        self.width = width
        self.analyzer = MetricsAnalyzer(metrics)

    def _bar(self, value: float, max_val: float = 1.0) -> str:
        """Convert value to block character"""
        if max_val == 0:
            return self.BLOCKS[0]
        idx = int(min(1, max(0, value / max_val)) * (len(self.BLOCKS) - 1))
        return self.BLOCKS[idx]

    def _horizontal_bar(self, value: float, max_val: float, width: int = 20,
                       fill: str = "█", empty: str = "░") -> str:
        """Create horizontal bar"""
        if max_val == 0:
            filled = 0
        else:
            filled = int((value / max_val) * width)
        return fill * filled + empty * (width - filled)

    def _sparkline(self, data: List[float], width: int = 30) -> str:
        """Create sparkline visualization"""
        if not data:
            return "No data"

        # Resample if needed
        if len(data) > width:
            step = len(data) / width
            data = [data[int(i * step)] for i in range(width)]

        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val if max_val != min_val else 1

        return ''.join(self._bar((v - min_val) / range_val) for v in data)

    def _mini_heatmap(self, matrix: List[List[float]], labels: List[str]) -> List[str]:
        """Create ASCII heatmap"""
        lines = []
        max_val = max(max(row) for row in matrix) if matrix else 1

        # Header
        header = "    " + " ".join(f"{l[:3]:>3}" for l in labels)
        lines.append(header)
        lines.append("    " + "─" * (len(labels) * 4))

        for i, (row, label) in enumerate(zip(matrix, labels)):
            row_str = f"{label[:3]:>3}│"
            for val in row:
                shade_idx = int(min(1, val / max_val) * (len(self.SHADES) - 1))
                row_str += f" {self.SHADES[shade_idx]} "
            lines.append(row_str)

        return lines

    def render_dashboard(self) -> str:
        """Render comprehensive ASCII dashboard"""
        lines = []

        # Header
        lines.append("╔" + "═" * (self.width - 2) + "╗")
        title = "ADVANCED SIMULATION ANALYTICS"
        lines.append("║" + title.center(self.width - 2) + "║")
        lines.append("╠" + "═" * (self.width - 2) + "╣")

        # System Overview
        lines.extend(self._render_system_overview())
        lines.append("╠" + "─" * (self.width - 2) + "╣")

        # Strategic Analysis
        lines.extend(self._render_strategic_analysis())
        lines.append("╠" + "─" * (self.width - 2) + "╣")

        # Emergence Metrics
        lines.extend(self._render_emergence())
        lines.append("╠" + "─" * (self.width - 2) + "╣")

        # Action Patterns
        lines.extend(self._render_action_patterns())
        lines.append("╠" + "─" * (self.width - 2) + "╣")

        # Market Analysis
        lines.extend(self._render_market_analysis())
        lines.append("╠" + "─" * (self.width - 2) + "╣")

        # Agent Comparison
        lines.extend(self._render_agent_comparison())

        # Footer
        lines.append("╚" + "═" * (self.width - 2) + "╝")

        return "\n".join(lines)

    def _render_system_overview(self) -> List[str]:
        """Render system overview section"""
        lines = []
        lines.append("║ 📊 SYSTEM OVERVIEW".ljust(self.width - 1) + "║")
        lines.append("║" + " " * (self.width - 2) + "║")

        # Rounds and agents
        info = f"  Rounds: {len(self.metrics.rounds)} │ Agents: {len(self.metrics.agents)}"
        events = f" │ Events: {len(self.metrics.events)}"
        lines.append("║" + (info + events).ljust(self.width - 2) + "║")

        # Tension sparkline
        if self.metrics.tension_curve:
            spark = self._sparkline(self.metrics.tension_curve, 40)
            lines.append("║" + f"  Tension: {spark} [{self.metrics.tension_curve[-1]:.2f}]".ljust(self.width - 2) + "║")

        # System entropy
        if self.metrics.system_entropy:
            spark = self._sparkline(self.metrics.system_entropy, 40)
            lines.append("║" + f"  Entropy: {spark} [{self.metrics.system_entropy[-1]:.2f}]".ljust(self.width - 2) + "║")

        lines.append("║" + " " * (self.width - 2) + "║")
        return lines

    def _render_strategic_analysis(self) -> List[str]:
        """Render strategic analysis section"""
        lines = []
        lines.append("║ 🎯 STRATEGIC ANALYSIS".ljust(self.width - 1) + "║")
        lines.append("║" + " " * (self.width - 2) + "║")

        if not self.metrics.payoff_history:
            lines.append("║" + "  No payoff data available".ljust(self.width - 2) + "║")
            return lines

        # Calculate cumulative payoffs
        cumulative = defaultdict(float)
        for payoffs in self.metrics.payoff_history:
            for other, agent_payoffs in payoffs.items():
                for agent, payoff in agent_payoffs.items():
                    cumulative[agent] += payoff

        max_payoff = max(cumulative.values()) if cumulative else 1

        for agent, total in sorted(cumulative.items(), key=lambda x: -x[1]):
            bar = self._horizontal_bar(total, max_payoff, 30)
            line = f"  {agent.title():10} {bar} {total:6.1f}"
            lines.append("║" + line.ljust(self.width - 2) + "║")

        # Nash distance
        if self.metrics.nash_distance:
            avg_nash = sum(self.metrics.nash_distance) / len(self.metrics.nash_distance)
            lines.append("║" + f"  Avg Nash Distance: {avg_nash:.3f}".ljust(self.width - 2) + "║")

        lines.append("║" + " " * (self.width - 2) + "║")
        return lines

    def _render_emergence(self) -> List[str]:
        """Render emergence metrics section"""
        lines = []
        lines.append("║ 🌀 EMERGENCE METRICS".ljust(self.width - 1) + "║")
        lines.append("║" + " " * (self.width - 2) + "║")

        em = self.analyzer.calculate_emergence()

        metrics = [
            ("Collective Complexity", em.collective_complexity),
            ("Synergy", em.synergy),
            ("Redundancy", em.redundancy),
            ("Emergence Score", em.emergence_score),
        ]

        for name, value in metrics:
            bar = self._horizontal_bar(min(1, value), 1, 20)
            line = f"  {name:22} {bar} {value:.3f}"
            lines.append("║" + line.ljust(self.width - 2) + "║")

        lines.append("║" + " " * (self.width - 2) + "║")
        return lines

    def _render_action_patterns(self) -> List[str]:
        """Render action pattern analysis"""
        lines = []
        lines.append("║ 📋 ACTION PATTERNS".ljust(self.width - 1) + "║")
        lines.append("║" + " " * (self.width - 2) + "║")

        if not self.metrics.action_sequences:
            lines.append("║" + "  No action data available".ljust(self.width - 2) + "║")
            return lines

        # Action timeline
        for agent, actions in self.metrics.action_sequences.items():
            symbols = ''.join('✓' if a == 'cooperate' else '✗' for a in actions[-20:])
            entropy = self.analyzer.calculate_entropy(actions)
            line = f"  {agent.title():10} {symbols:20} H={entropy:.2f}"
            lines.append("║" + line.ljust(self.width - 2) + "║")

        # Detected patterns
        lines.append("║" + " " * (self.width - 2) + "║")
        lines.append("║" + "  Detected Patterns:".ljust(self.width - 2) + "║")

        for agent, actions in self.metrics.action_sequences.items():
            patterns = self.analyzer.detect_patterns(actions, max_length=4)
            if patterns:
                top_pattern = max(patterns.items(), key=lambda x: x[1])
                line = f"    {agent.title()}: {top_pattern[0][:30]} (×{top_pattern[1]})"
                lines.append("║" + line.ljust(self.width - 2) + "║")

        lines.append("║" + " " * (self.width - 2) + "║")
        return lines

    def _render_market_analysis(self) -> List[str]:
        """Render market analysis section"""
        lines = []
        lines.append("║ 💰 MARKET ANALYSIS".ljust(self.width - 1) + "║")
        lines.append("║" + " " * (self.width - 2) + "║")

        if not self.metrics.market_prices:
            lines.append("║" + "  No market data available".ljust(self.width - 2) + "║")
            return lines

        for good_id, prices in self.metrics.market_prices.items():
            if not prices:
                continue

            analysis = self.analyzer.analyze_time_series(prices)
            spark = self._sparkline(prices, 30)

            lines.append("║" + f"  {good_id}:".ljust(self.width - 2) + "║")
            lines.append("║" + f"    Price: {spark} ${prices[-1]:.2f}".ljust(self.width - 2) + "║")

            trend_arrow = "↑" if analysis.trend > 0.01 else "↓" if analysis.trend < -0.01 else "→"
            stats = f"    Trend: {trend_arrow} │ Vol: {analysis.volatility:.2%} │ Entropy: {analysis.entropy:.2f}"
            lines.append("║" + stats.ljust(self.width - 2) + "║")

            if analysis.anomalies:
                lines.append("║" + f"    ⚠ Anomalies at rounds: {analysis.anomalies[:5]}".ljust(self.width - 2) + "║")

        lines.append("║" + " " * (self.width - 2) + "║")
        return lines

    def _render_agent_comparison(self) -> List[str]:
        """Render agent comparison matrix"""
        lines = []
        lines.append("║ 👥 AGENT COMPARISON".ljust(self.width - 1) + "║")
        lines.append("║" + " " * (self.width - 2) + "║")

        agents = self.metrics.agents
        if len(agents) < 2:
            lines.append("║" + "  Need multiple agents for comparison".ljust(self.width - 2) + "║")
            return lines

        # Create mutual information matrix
        mi_matrix = []
        for a1 in agents:
            row = []
            for a2 in agents:
                if a1 == a2:
                    row.append(1.0)
                else:
                    seq1 = self.metrics.action_sequences.get(a1, [])
                    seq2 = self.metrics.action_sequences.get(a2, [])
                    mi = self.analyzer.calculate_mutual_information(seq1, seq2)
                    row.append(mi)
            mi_matrix.append(row)

        # Render mini heatmap
        heatmap_lines = self._mini_heatmap(mi_matrix, agents)
        for hline in heatmap_lines:
            lines.append("║  " + hline.ljust(self.width - 4) + "║")

        lines.append("║" + "  (Mutual Information between agent strategies)".ljust(self.width - 2) + "║")
        lines.append("║" + " " * (self.width - 2) + "║")
        return lines


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_advanced_metrics() -> AdvancedSimulationMetrics:
    """Create a new advanced metrics collector"""
    return AdvancedSimulationMetrics()


def visualize_advanced(metrics: AdvancedSimulationMetrics,
                      use_matplotlib: bool = True,
                      save_path: str = None) -> None:
    """Visualize with advanced analytics"""
    if use_matplotlib and MATPLOTLIB_AVAILABLE:
        viz = AdvancedSimulationVisualizer(metrics)
        viz.create_comprehensive_dashboard()
        if save_path:
            viz.save(save_path)
        else:
            viz.show()
    else:
        viz = AdvancedASCIIVisualizer(metrics)
        print(viz.render_dashboard())


def demo_advanced_visualization():
    """Demo the advanced visualization"""
    print("Creating advanced simulation data...")

    metrics = AdvancedSimulationMetrics()

    # Register agents
    for agent in ['alice', 'bob', 'charlie']:
        metrics.register_agent(agent)

    # Simulate 10 rounds
    for r in range(1, 11):
        metrics.record_round(r)

        # State vectors (cooperation tendency, trust, wealth)
        metrics.record_state_vector('alice', [0.3 + r * 0.05, 0.5 + r * 0.02, 100 + r * 10])
        metrics.record_state_vector('bob', [0.7 - r * 0.03, 0.6 - r * 0.02, 100 - r * 5])
        metrics.record_state_vector('charlie', [0.5 + 0.1 * math.sin(r), 0.5, 100])

        # Actions
        alice_action = 'defect' if r < 4 else 'cooperate'
        bob_action = 'cooperate' if r < 6 else 'defect'
        charlie_action = 'cooperate' if r % 2 == 0 else 'defect'

        metrics.record_action('alice', alice_action)
        metrics.record_action('bob', bob_action)
        metrics.record_action('charlie', charlie_action)

        # Payoffs
        metrics.payoff_history.append({
            'round': {
                'alice': 5 if alice_action == 'defect' and bob_action == 'cooperate' else 3 if alice_action == bob_action == 'cooperate' else 1,
                'bob': 0 if alice_action == 'defect' and bob_action == 'cooperate' else 3 if alice_action == bob_action == 'cooperate' else 1,
                'charlie': random.randint(1, 4)
            }
        })

        # Information flow
        metrics.record_information_flow({
            ('alice', 'bob'): 0.5 + r * 0.03,
            ('bob', 'alice'): 0.3 + r * 0.02,
            ('alice', 'charlie'): 0.2,
            ('charlie', 'bob'): 0.4,
        })

        # System metrics
        metrics.record_system_metrics(
            entropy=0.5 + 0.1 * math.sin(r * 0.5),
            complexity=0.3 + r * 0.05,
            emergence=0.1 + r * 0.08
        )

        # Narrative
        metrics.record_narrative(
            tension=0.2 + r * 0.08 if r < 7 else 0.9 - (r - 7) * 0.2,
            beat={'type': 'conflict' if r < 5 else 'action' if r < 8 else 'resolution'},
            arc_progress={'alice': r * 0.1, 'bob': r * 0.08, 'charlie': r * 0.12}
        )

        # Market
        metrics.record_market(
            'AITECH', 100 + r * 3 + random.uniform(-5, 5),
            bids=r + random.randint(0, 3),
            asks=r + random.randint(0, 2),
            trades=random.randint(0, 3),
            wealth={'alice': 1000 + r * 50, 'bob': 1000 - r * 20, 'charlie': 1000}
        )

        # Consciousness
        for agent in ['alice', 'bob', 'charlie']:
            metrics.record_consciousness(
                agent,
                phi=0.1 + random.uniform(0, 0.3),
                qualia_comp=r * 0.5,
                attn_entropy=random.uniform(0.3, 0.7)
            )

        # Metacognition
        for agent in ['alice', 'bob', 'charlie']:
            metrics.record_metacognition(
                agent,
                load=0.3 + random.uniform(0, 0.4),
                calibration=0.5 + random.uniform(-0.2, 0.2),
                lr=0.1 * (1 - r / 15)
            )

        # Belief divergence
        metrics.belief_divergence.append({
            ('alice', 'bob'): 0.3 + r * 0.05,
            ('alice', 'charlie'): 0.2 + r * 0.03,
            ('bob', 'charlie'): 0.4 - r * 0.02,
        })

    print("Sample data created.")
    print()

    # ASCII visualization
    ascii_viz = AdvancedASCIIVisualizer(metrics)
    print(ascii_viz.render_dashboard())

    # Matplotlib visualization
    if MATPLOTLIB_AVAILABLE:
        print("\nGenerating advanced matplotlib dashboard...")
        viz = AdvancedSimulationVisualizer(metrics)
        viz.create_comprehensive_dashboard()
        viz.save("/tmp/advanced_simulation_dashboard.png")
        print("Dashboard saved to /tmp/advanced_simulation_dashboard.png")


if __name__ == "__main__":
    demo_advanced_visualization()
