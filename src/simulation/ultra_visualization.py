"""
Ultra-Sophisticated Visualization Module

Features:
- 3D phase space trajectories with attractor detection
- Causal inference using transfer entropy and Granger causality
- Recurrence plots and quantification analysis
- Spectral analysis with wavelet decomposition
- Agent embedding space with t-SNE/UMAP-style dimensionality reduction
- Lyapunov exponent estimation for chaos detection
- Bifurcation analysis
- Information-theoretic decomposition (synergy, redundancy, unique info)
- Multi-scale temporal analysis
- Cross-correlation with lag analysis
- Dynamic network evolution
- Anomaly detection with isolation forests
- Predictive state estimation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
from datetime import datetime
from enum import Enum
from collections import defaultdict, deque
import math
import random
import itertools

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Create minimal numpy-like functionality
    class np:
        @staticmethod
        def array(x): return list(x)
        @staticmethod
        def zeros(shape):
            if isinstance(shape, tuple):
                return [[0]*shape[1] for _ in range(shape[0])]
            return [0]*shape
        @staticmethod
        def mean(x): return sum(x)/len(x) if x else 0
        @staticmethod
        def std(x):
            if not x: return 0
            m = sum(x)/len(x)
            return math.sqrt(sum((xi-m)**2 for xi in x)/len(x))
        pi = math.pi
        @staticmethod
        def sin(x): return math.sin(x)
        @staticmethod
        def cos(x): return math.cos(x)
        @staticmethod
        def exp(x): return math.exp(x)
        @staticmethod
        def log(x): return math.log(x) if x > 0 else -float('inf')

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.collections import LineCollection
    import matplotlib.cm as cm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ============================================================================
# ADVANCED MATHEMATICAL ANALYSIS
# ============================================================================

class DynamicsType(Enum):
    """Types of dynamical behavior"""
    FIXED_POINT = "fixed_point"
    LIMIT_CYCLE = "limit_cycle"
    QUASI_PERIODIC = "quasi_periodic"
    CHAOTIC = "chaotic"
    TRANSIENT = "transient"
    UNKNOWN = "unknown"


class CausalityMethod(Enum):
    """Methods for causal inference"""
    GRANGER = "granger"
    TRANSFER_ENTROPY = "transfer_entropy"
    CONVERGENT_CROSS_MAPPING = "ccm"
    PARTIAL_CORRELATION = "partial_correlation"


@dataclass
class AttractorAnalysis:
    """Results of attractor detection"""
    dynamics_type: DynamicsType = DynamicsType.UNKNOWN
    fixed_points: List[Tuple[float, ...]] = field(default_factory=list)
    limit_cycle_period: Optional[float] = None
    correlation_dimension: float = 0.0
    lyapunov_exponent: float = 0.0
    basin_of_attraction: List[Tuple[float, float]] = field(default_factory=list)
    recurrence_rate: float = 0.0


@dataclass
class CausalLink:
    """A causal relationship between two variables"""
    source: str
    target: str
    strength: float
    lag: int
    method: CausalityMethod
    p_value: float = 1.0
    confidence: float = 0.0


@dataclass
class SpectralAnalysis:
    """Results of spectral/frequency analysis"""
    dominant_frequencies: List[float] = field(default_factory=list)
    power_spectrum: List[Tuple[float, float]] = field(default_factory=list)
    spectral_entropy: float = 0.0
    bandwidth: float = 0.0
    peak_frequency: float = 0.0


@dataclass
class RecurrenceAnalysis:
    """Recurrence quantification analysis results"""
    recurrence_rate: float = 0.0
    determinism: float = 0.0
    average_diagonal_length: float = 0.0
    max_diagonal_length: int = 0
    entropy_diagonal: float = 0.0
    laminarity: float = 0.0
    trapping_time: float = 0.0


@dataclass
class InformationDecomposition:
    """Partial information decomposition"""
    redundancy: float = 0.0
    unique_x: float = 0.0
    unique_y: float = 0.0
    synergy: float = 0.0
    total_information: float = 0.0


@dataclass
class AgentEmbedding:
    """Reduced-dimensional representation of an agent"""
    agent_id: str
    coordinates_2d: Tuple[float, float] = (0.0, 0.0)
    coordinates_3d: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    cluster_id: int = 0
    trajectory: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class AnomalyDetection:
    """Anomaly detection results"""
    anomaly_indices: List[int] = field(default_factory=list)
    anomaly_scores: List[float] = field(default_factory=list)
    threshold: float = 0.0
    isolation_depths: List[float] = field(default_factory=list)


@dataclass
class MultiscaleAnalysis:
    """Multi-scale temporal analysis"""
    scales: List[int] = field(default_factory=list)
    wavelet_coefficients: Dict[int, List[float]] = field(default_factory=dict)
    scale_entropy: Dict[int, float] = field(default_factory=dict)
    dominant_scale: int = 1
    cross_scale_correlations: Dict[Tuple[int, int], float] = field(default_factory=dict)


# ============================================================================
# ULTRA METRICS COLLECTION
# ============================================================================

@dataclass
class UltraSimulationMetrics:
    """Comprehensive metrics collection for ultra-sophisticated analysis"""

    # Basic tracking
    rounds: List[int] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)

    # High-dimensional state vectors per agent per round
    # Each state is [cooperation, aggression, trust, wealth, influence, phi, entropy]
    state_vectors: Dict[str, List[List[float]]] = field(default_factory=lambda: defaultdict(list))

    # Full interaction matrix per round
    interactions: List[Dict[Tuple[str, str], Dict[str, float]]] = field(default_factory=list)

    # Belief states per agent
    belief_states: Dict[str, List[Dict[str, float]]] = field(default_factory=lambda: defaultdict(list))

    # Action sequences (for pattern mining)
    action_sequences: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))

    # Continuous signals for spectral analysis
    tension_signal: List[float] = field(default_factory=list)
    entropy_signal: List[float] = field(default_factory=list)
    complexity_signal: List[float] = field(default_factory=list)

    # Phase space coordinates per agent
    phase_coordinates: Dict[str, List[Tuple[float, float, float]]] = field(default_factory=lambda: defaultdict(list))

    # Network snapshots
    network_snapshots: List[Dict[str, Any]] = field(default_factory=list)

    # Causal links discovered
    causal_links: List[CausalLink] = field(default_factory=list)

    # Attractor analysis per agent
    attractor_analyses: Dict[str, AttractorAnalysis] = field(default_factory=dict)

    # Global system state
    global_phi: List[float] = field(default_factory=list)
    global_entropy: List[float] = field(default_factory=list)
    global_complexity: List[float] = field(default_factory=list)

    def record_round(self, round_num: int):
        """Start recording a new round"""
        self.rounds.append(round_num)
        self.timestamps.append(datetime.now())
        self.interactions.append({})
        self.network_snapshots.append({"nodes": [], "edges": []})

    def record_agent_state(self, agent_id: str, state_vector: List[float],
                          phase_coords: Tuple[float, float, float],
                          beliefs: Dict[str, float], action: str):
        """Record comprehensive agent state"""
        self.state_vectors[agent_id].append(state_vector)
        self.phase_coordinates[agent_id].append(phase_coords)
        self.belief_states[agent_id].append(beliefs)
        self.action_sequences[agent_id].append(action)

    def record_interaction(self, agent1: str, agent2: str,
                          interaction_type: str, strength: float,
                          information_flow: float = 0.0):
        """Record interaction between agents"""
        if self.interactions:
            self.interactions[-1][(agent1, agent2)] = {
                "type": interaction_type,
                "strength": strength,
                "information_flow": information_flow
            }

    def record_global_state(self, tension: float, entropy: float,
                           complexity: float, phi: float):
        """Record global system state"""
        self.tension_signal.append(tension)
        self.entropy_signal.append(entropy)
        self.complexity_signal.append(complexity)
        self.global_phi.append(phi)
        self.global_entropy.append(entropy)
        self.global_complexity.append(complexity)

    def record_network_snapshot(self, nodes: List[Dict], edges: List[Dict]):
        """Record network state"""
        if self.network_snapshots:
            self.network_snapshots[-1] = {"nodes": nodes, "edges": edges}


# ============================================================================
# ADVANCED MATHEMATICAL ANALYZERS
# ============================================================================

class CausalInferenceEngine:
    """Performs causal inference between time series"""

    def __init__(self, max_lag: int = 5):
        self.max_lag = max_lag

    def granger_causality(self, x: List[float], y: List[float], lag: int = 1) -> Tuple[float, float]:
        """
        Test if x Granger-causes y
        Returns (F-statistic, pseudo p-value)
        """
        if len(x) < lag + 2 or len(y) < lag + 2:
            return 0.0, 1.0

        # Restricted model: y_t = a + b*y_{t-1} + ... + b_lag*y_{t-lag}
        # Full model: y_t = a + b*y_{t-1} + ... + c*x_{t-1} + ... + c_lag*x_{t-lag}

        n = len(y) - lag

        # Calculate residuals for restricted model (AR on y only)
        y_restricted_residuals = []
        for t in range(lag, len(y)):
            y_pred = sum(y[t-i-1] for i in range(lag)) / lag
            y_restricted_residuals.append((y[t] - y_pred) ** 2)

        # Calculate residuals for full model (AR on y + x)
        y_full_residuals = []
        for t in range(lag, len(y)):
            y_pred = sum(y[t-i-1] for i in range(lag)) / lag
            x_effect = sum(x[t-i-1] for i in range(min(lag, len(x)-t+lag))) / lag if t <= len(x) else 0
            y_pred_full = 0.5 * y_pred + 0.5 * x_effect
            y_full_residuals.append((y[t] - y_pred_full) ** 2)

        rss_restricted = sum(y_restricted_residuals)
        rss_full = sum(y_full_residuals)

        if rss_full == 0:
            return 0.0, 1.0

        # F-statistic
        f_stat = ((rss_restricted - rss_full) / lag) / (rss_full / max(1, n - 2*lag))

        # Pseudo p-value (simplified)
        p_value = math.exp(-abs(f_stat) / 2) if f_stat > 0 else 1.0

        return f_stat, p_value

    def transfer_entropy(self, source: List[float], target: List[float],
                        lag: int = 1, bins: int = 4) -> float:
        """
        Calculate transfer entropy from source to target
        TE = H(Y_t | Y_{t-1:t-k}) - H(Y_t | Y_{t-1:t-k}, X_{t-1:t-k})
        """
        if len(source) < lag + 1 or len(target) < lag + 1:
            return 0.0

        # Discretize signals
        def discretize(signal, n_bins):
            if not signal:
                return []
            min_val, max_val = min(signal), max(signal)
            if max_val == min_val:
                return [0] * len(signal)
            return [min(n_bins-1, int((x - min_val) / (max_val - min_val + 1e-10) * n_bins))
                    for x in signal]

        src_disc = discretize(source, bins)
        tgt_disc = discretize(target, bins)

        # Count joint occurrences
        joint_counts = defaultdict(int)
        marginal_y_past = defaultdict(int)
        marginal_y_past_x_past = defaultdict(int)

        for t in range(lag, min(len(src_disc), len(tgt_disc))):
            y_t = tgt_disc[t]
            y_past = tuple(tgt_disc[t-i-1] for i in range(lag))
            x_past = tuple(src_disc[t-i-1] for i in range(lag))

            joint_counts[(y_t, y_past, x_past)] += 1
            marginal_y_past[y_past] += 1
            marginal_y_past_x_past[(y_past, x_past)] += 1

        # Calculate transfer entropy
        te = 0.0
        total = sum(joint_counts.values())

        if total == 0:
            return 0.0

        for (y_t, y_past, x_past), count in joint_counts.items():
            p_joint = count / total
            p_y_past = marginal_y_past[y_past] / total
            p_y_past_x_past = marginal_y_past_x_past[(y_past, x_past)] / total

            if p_joint > 0 and p_y_past > 0 and p_y_past_x_past > 0:
                # TE contribution
                p_y_given_past = p_joint / p_y_past_x_past if p_y_past_x_past > 0 else 0
                p_y_given_y_past = count / marginal_y_past[y_past] if marginal_y_past[y_past] > 0 else 0

                if p_y_given_past > 0 and p_y_given_y_past > 0:
                    te += p_joint * math.log2(p_y_given_past / p_y_given_y_past + 1e-10)

        return max(0, te)

    def find_causal_links(self, time_series: Dict[str, List[float]],
                         method: CausalityMethod = CausalityMethod.TRANSFER_ENTROPY) -> List[CausalLink]:
        """Find all significant causal links between time series"""
        links = []
        variables = list(time_series.keys())

        for source, target in itertools.permutations(variables, 2):
            src_data = time_series[source]
            tgt_data = time_series[target]

            best_strength = 0.0
            best_lag = 1

            for lag in range(1, self.max_lag + 1):
                if method == CausalityMethod.TRANSFER_ENTROPY:
                    strength = self.transfer_entropy(src_data, tgt_data, lag)
                else:  # Granger
                    f_stat, p_value = self.granger_causality(src_data, tgt_data, lag)
                    strength = f_stat

                if strength > best_strength:
                    best_strength = strength
                    best_lag = lag

            if best_strength > 0.1:  # Threshold for significance
                links.append(CausalLink(
                    source=source,
                    target=target,
                    strength=best_strength,
                    lag=best_lag,
                    method=method,
                    confidence=min(1.0, best_strength)
                ))

        return sorted(links, key=lambda x: x.strength, reverse=True)


class AttractorDetector:
    """Detects and characterizes attractors in dynamical systems"""

    def __init__(self, embedding_dim: int = 3, time_delay: int = 1):
        self.embedding_dim = embedding_dim
        self.time_delay = time_delay

    def takens_embedding(self, time_series: List[float]) -> List[Tuple[float, ...]]:
        """Create delay embedding of time series"""
        n = len(time_series) - (self.embedding_dim - 1) * self.time_delay
        if n <= 0:
            return []

        embedded = []
        for i in range(n):
            point = tuple(time_series[i + j * self.time_delay]
                         for j in range(self.embedding_dim))
            embedded.append(point)

        return embedded

    def estimate_correlation_dimension(self, points: List[Tuple[float, ...]],
                                       eps_range: Tuple[float, float] = (0.01, 1.0),
                                       n_eps: int = 20) -> float:
        """Estimate correlation dimension using Grassberger-Procaccia algorithm"""
        if len(points) < 10:
            return 0.0

        n = len(points)

        def distance(p1, p2):
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

        # Calculate correlation sum for different epsilon values
        eps_values = []
        c_values = []

        for i in range(n_eps):
            eps = eps_range[0] * (eps_range[1] / eps_range[0]) ** (i / (n_eps - 1))

            count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if distance(points[i], points[j]) < eps:
                        count += 1

            c = 2 * count / (n * (n - 1)) if n > 1 else 0

            if c > 0:
                eps_values.append(math.log(eps))
                c_values.append(math.log(c))

        # Linear regression to estimate slope (correlation dimension)
        if len(eps_values) < 3:
            return 0.0

        n_pts = len(eps_values)
        sum_x = sum(eps_values)
        sum_y = sum(c_values)
        sum_xy = sum(x * y for x, y in zip(eps_values, c_values))
        sum_x2 = sum(x ** 2 for x in eps_values)

        slope = (n_pts * sum_xy - sum_x * sum_y) / (n_pts * sum_x2 - sum_x ** 2 + 1e-10)

        return slope

    def estimate_lyapunov(self, time_series: List[float], dt: float = 1.0) -> float:
        """Estimate largest Lyapunov exponent using Rosenstein method"""
        if len(time_series) < 20:
            return 0.0

        embedded = self.takens_embedding(time_series)
        if len(embedded) < 10:
            return 0.0

        def distance(p1, p2):
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

        # Find nearest neighbors (excluding temporal neighbors)
        divergence_rates = []

        for i in range(len(embedded) - 1):
            min_dist = float('inf')
            min_j = -1

            for j in range(len(embedded)):
                if abs(i - j) > self.time_delay * 2:  # Exclude temporal neighbors
                    d = distance(embedded[i], embedded[j])
                    if d < min_dist and d > 0:
                        min_dist = d
                        min_j = j

            if min_j >= 0 and min_j < len(embedded) - 1 and i < len(embedded) - 1:
                # Track divergence
                d_later = distance(embedded[i + 1], embedded[min_j + 1]) if min_j + 1 < len(embedded) else min_dist
                if min_dist > 0 and d_later > 0:
                    divergence_rates.append(math.log(d_later / min_dist) / dt)

        if not divergence_rates:
            return 0.0

        return sum(divergence_rates) / len(divergence_rates)

    def detect_fixed_points(self, points: List[Tuple[float, ...]],
                           tolerance: float = 0.1) -> List[Tuple[float, ...]]:
        """Detect fixed points in the trajectory"""
        if not points:
            return []

        # Look for points where trajectory slows down
        fixed_points = []
        velocities = []

        for i in range(1, len(points)):
            v = sum((points[i][j] - points[i-1][j]) ** 2 for j in range(len(points[i])))
            velocities.append(math.sqrt(v))

        if not velocities:
            return []

        mean_v = sum(velocities) / len(velocities)

        # Find regions of low velocity
        in_fixed_region = False
        region_points = []

        for i, v in enumerate(velocities):
            if v < tolerance * mean_v:
                region_points.append(points[i])
                in_fixed_region = True
            elif in_fixed_region and region_points:
                # Calculate centroid of region
                centroid = tuple(sum(p[j] for p in region_points) / len(region_points)
                               for j in range(len(region_points[0])))
                fixed_points.append(centroid)
                region_points = []
                in_fixed_region = False

        return fixed_points

    def analyze(self, time_series: List[float]) -> AttractorAnalysis:
        """Perform full attractor analysis"""
        analysis = AttractorAnalysis()

        if len(time_series) < 20:
            return analysis

        # Create embedding
        embedded = self.takens_embedding(time_series)
        if not embedded:
            return analysis

        # Detect fixed points
        analysis.fixed_points = self.detect_fixed_points(embedded)

        # Estimate dimensions and exponents
        analysis.correlation_dimension = self.estimate_correlation_dimension(embedded)
        analysis.lyapunov_exponent = self.estimate_lyapunov(time_series)

        # Classify dynamics
        if analysis.lyapunov_exponent > 0.1:
            analysis.dynamics_type = DynamicsType.CHAOTIC
        elif len(analysis.fixed_points) > 0 and analysis.lyapunov_exponent < 0:
            analysis.dynamics_type = DynamicsType.FIXED_POINT
        elif analysis.correlation_dimension < 1.5:
            analysis.dynamics_type = DynamicsType.LIMIT_CYCLE
        else:
            analysis.dynamics_type = DynamicsType.QUASI_PERIODIC

        # Calculate recurrence rate
        def distance(p1, p2):
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

        threshold = 0.1 * max(max(p) - min(p) for p in zip(*embedded)) if embedded else 0.1
        recurrence_count = 0
        total_pairs = 0

        for i in range(len(embedded)):
            for j in range(i + 1, len(embedded)):
                total_pairs += 1
                if distance(embedded[i], embedded[j]) < threshold:
                    recurrence_count += 1

        analysis.recurrence_rate = recurrence_count / total_pairs if total_pairs > 0 else 0

        return analysis


class RecurrencePlotAnalyzer:
    """Creates and analyzes recurrence plots"""

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold

    def create_recurrence_matrix(self, time_series: List[float]) -> List[List[int]]:
        """Create binary recurrence matrix"""
        n = len(time_series)
        if n == 0:
            return []

        # Normalize threshold
        series_range = max(time_series) - min(time_series) if time_series else 1
        eps = self.threshold * series_range if series_range > 0 else self.threshold

        matrix = [[0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if abs(time_series[i] - time_series[j]) < eps:
                    matrix[i][j] = 1

        return matrix

    def quantify_recurrence(self, matrix: List[List[int]]) -> RecurrenceAnalysis:
        """Perform recurrence quantification analysis"""
        analysis = RecurrenceAnalysis()

        if not matrix:
            return analysis

        n = len(matrix)

        # Recurrence rate
        total_recurrence = sum(sum(row) for row in matrix)
        analysis.recurrence_rate = total_recurrence / (n * n) if n > 0 else 0

        # Find diagonal lines
        diagonal_lengths = []

        # Main diagonals above main
        for k in range(1, n):
            length = 0
            for i in range(n - k):
                if matrix[i][i + k] == 1:
                    length += 1
                else:
                    if length > 1:
                        diagonal_lengths.append(length)
                    length = 0
            if length > 1:
                diagonal_lengths.append(length)

        # Main diagonals below main
        for k in range(1, n):
            length = 0
            for i in range(n - k):
                if matrix[i + k][i] == 1:
                    length += 1
                else:
                    if length > 1:
                        diagonal_lengths.append(length)
                    length = 0
            if length > 1:
                diagonal_lengths.append(length)

        if diagonal_lengths:
            analysis.average_diagonal_length = sum(diagonal_lengths) / len(diagonal_lengths)
            analysis.max_diagonal_length = max(diagonal_lengths)

            # Determinism: fraction of recurrence points in diagonals
            points_in_diagonals = sum(diagonal_lengths)
            analysis.determinism = points_in_diagonals / total_recurrence if total_recurrence > 0 else 0

            # Diagonal entropy
            hist = defaultdict(int)
            for l in diagonal_lengths:
                hist[l] += 1
            total = sum(hist.values())
            analysis.entropy_diagonal = -sum(
                (c / total) * math.log2(c / total)
                for c in hist.values() if c > 0
            ) if total > 0 else 0

        # Find vertical lines for laminarity
        vertical_lengths = []
        for j in range(n):
            length = 0
            for i in range(n):
                if matrix[i][j] == 1:
                    length += 1
                else:
                    if length > 1:
                        vertical_lengths.append(length)
                    length = 0
            if length > 1:
                vertical_lengths.append(length)

        if vertical_lengths:
            points_in_verticals = sum(vertical_lengths)
            analysis.laminarity = points_in_verticals / total_recurrence if total_recurrence > 0 else 0
            analysis.trapping_time = sum(vertical_lengths) / len(vertical_lengths)

        return analysis


class SpectralAnalyzer:
    """Performs spectral analysis on time series"""

    def fft_power_spectrum(self, signal: List[float],
                          sampling_rate: float = 1.0) -> List[Tuple[float, float]]:
        """Compute power spectrum using FFT-like algorithm"""
        n = len(signal)
        if n < 4:
            return []

        # Remove DC component
        mean_val = sum(signal) / n
        centered = [x - mean_val for x in signal]

        # Simple DFT (not FFT, but works for small signals)
        spectrum = []
        for k in range(n // 2):
            freq = k * sampling_rate / n

            real = sum(centered[t] * math.cos(2 * math.pi * k * t / n) for t in range(n))
            imag = sum(centered[t] * math.sin(2 * math.pi * k * t / n) for t in range(n))

            power = (real ** 2 + imag ** 2) / n
            spectrum.append((freq, power))

        return spectrum

    def wavelet_transform(self, signal: List[float],
                         scales: List[int] = None) -> Dict[int, List[float]]:
        """Simple wavelet-like multi-scale decomposition using moving averages"""
        if scales is None:
            scales = [2, 4, 8, 16, 32]

        coefficients = {}

        for scale in scales:
            if scale > len(signal):
                continue

            # Compute difference between signal and smoothed version
            smoothed = []
            detail = []

            for i in range(len(signal)):
                start = max(0, i - scale // 2)
                end = min(len(signal), i + scale // 2 + 1)
                window = signal[start:end]
                smoothed_val = sum(window) / len(window)
                smoothed.append(smoothed_val)
                detail.append(signal[i] - smoothed_val)

            coefficients[scale] = detail

        return coefficients

    def analyze(self, signal: List[float]) -> SpectralAnalysis:
        """Perform full spectral analysis"""
        analysis = SpectralAnalysis()

        if len(signal) < 4:
            return analysis

        # Power spectrum
        spectrum = self.fft_power_spectrum(signal)
        analysis.power_spectrum = spectrum

        if spectrum:
            # Find dominant frequencies
            sorted_spectrum = sorted(spectrum, key=lambda x: x[1], reverse=True)
            analysis.dominant_frequencies = [f for f, p in sorted_spectrum[:3]]

            # Peak frequency
            analysis.peak_frequency = sorted_spectrum[0][0] if sorted_spectrum else 0

            # Spectral entropy
            total_power = sum(p for _, p in spectrum)
            if total_power > 0:
                probs = [p / total_power for _, p in spectrum if p > 0]
                analysis.spectral_entropy = -sum(p * math.log2(p) for p in probs if p > 0)

            # Bandwidth (frequency range containing 90% of power)
            cumulative = 0
            low_freq, high_freq = 0, 0
            for f, p in sorted(spectrum):
                if cumulative < 0.05 * total_power:
                    low_freq = f
                cumulative += p
                if cumulative < 0.95 * total_power:
                    high_freq = f
            analysis.bandwidth = high_freq - low_freq

        return analysis


class DimensionalityReducer:
    """Reduces high-dimensional data for visualization"""

    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def pca_reduce(self, data: List[List[float]]) -> List[Tuple[float, ...]]:
        """Simple PCA-like dimensionality reduction"""
        if not data or len(data[0]) < self.n_components:
            return [(0.0,) * self.n_components] * len(data)

        n_samples = len(data)
        n_features = len(data[0])

        # Center data
        means = [sum(data[i][j] for i in range(n_samples)) / n_samples
                for j in range(n_features)]
        centered = [[data[i][j] - means[j] for j in range(n_features)]
                   for i in range(n_samples)]

        # Compute covariance matrix
        cov = [[0.0] * n_features for _ in range(n_features)]
        for i in range(n_features):
            for j in range(n_features):
                cov[i][j] = sum(centered[k][i] * centered[k][j]
                               for k in range(n_samples)) / max(1, n_samples - 1)

        # Power iteration to find principal components (simplified)
        components = []
        for _ in range(self.n_components):
            # Random starting vector
            v = [random.gauss(0, 1) for _ in range(n_features)]
            norm = math.sqrt(sum(x**2 for x in v))
            v = [x / norm for x in v]

            # Power iteration
            for _ in range(50):
                # Matrix-vector multiply
                v_new = [sum(cov[i][j] * v[j] for j in range(n_features))
                        for i in range(n_features)]
                norm = math.sqrt(sum(x**2 for x in v_new))
                if norm > 0:
                    v = [x / norm for x in v_new]

            components.append(v)

            # Deflate covariance matrix
            for i in range(n_features):
                for j in range(n_features):
                    eigenvalue = sum(cov[k][l] * v[k] * v[l]
                                   for k in range(n_features)
                                   for l in range(n_features))
                    cov[i][j] -= eigenvalue * v[i] * v[j]

        # Project data onto components
        reduced = []
        for point in centered:
            coords = tuple(sum(point[j] * components[c][j] for j in range(n_features))
                          for c in range(self.n_components))
            reduced.append(coords)

        return reduced

    def tsne_like_reduce(self, data: List[List[float]],
                        perplexity: float = 5.0,
                        n_iter: int = 200) -> List[Tuple[float, float]]:
        """Simplified t-SNE-like embedding"""
        n = len(data)
        if n < 2:
            return [(0.0, 0.0)] * n

        # Compute pairwise distances
        def dist(p1, p2):
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

        distances = [[dist(data[i], data[j]) for j in range(n)] for i in range(n)]

        # Compute affinities (simplified)
        sigma = sum(sum(row) for row in distances) / (n * n) + 1e-10
        P = [[math.exp(-distances[i][j] ** 2 / (2 * sigma ** 2)) if i != j else 0
             for j in range(n)] for i in range(n)]

        # Normalize
        for i in range(n):
            row_sum = sum(P[i]) + 1e-10
            P[i] = [p / row_sum for p in P[i]]

        # Symmetrize
        for i in range(n):
            for j in range(i + 1, n):
                avg = (P[i][j] + P[j][i]) / (2 * n)
                P[i][j] = P[j][i] = avg

        # Initialize embedding randomly
        Y = [[random.gauss(0, 0.1), random.gauss(0, 0.1)] for _ in range(n)]

        # Gradient descent
        learning_rate = 100.0
        momentum = 0.5
        gains = [[1.0, 1.0] for _ in range(n)]
        velocities = [[0.0, 0.0] for _ in range(n)]

        for iteration in range(n_iter):
            # Compute Q (t-distribution affinities in low-dim)
            Q_unnorm = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i != j:
                        d_sq = sum((Y[i][k] - Y[j][k]) ** 2 for k in range(2))
                        Q_unnorm[i][j] = 1 / (1 + d_sq)

            Q_sum = sum(sum(row) for row in Q_unnorm) + 1e-10
            Q = [[q / Q_sum for q in row] for row in Q_unnorm]

            # Compute gradients
            gradients = [[0.0, 0.0] for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i != j:
                        pq_diff = P[i][j] - Q[i][j]
                        d_sq = sum((Y[i][k] - Y[j][k]) ** 2 for k in range(2))
                        mult = 4 * pq_diff / (1 + d_sq)
                        for k in range(2):
                            gradients[i][k] += mult * (Y[i][k] - Y[j][k])

            # Update with momentum
            for i in range(n):
                for k in range(2):
                    # Adaptive gains
                    if (gradients[i][k] > 0) != (velocities[i][k] > 0):
                        gains[i][k] = min(gains[i][k] + 0.2, 4.0)
                    else:
                        gains[i][k] = max(gains[i][k] * 0.8, 0.01)

                    velocities[i][k] = momentum * velocities[i][k] - learning_rate * gains[i][k] * gradients[i][k]
                    Y[i][k] += velocities[i][k]

            # Center
            mean_y = [sum(Y[i][k] for i in range(n)) / n for k in range(2)]
            for i in range(n):
                for k in range(2):
                    Y[i][k] -= mean_y[k]

        return [(y[0], y[1]) for y in Y]


class AnomalyDetector:
    """Detects anomalies in time series using isolation-forest-like method"""

    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination

    def compute_isolation_scores(self, data: List[List[float]],
                                 n_trees: int = 10) -> List[float]:
        """Compute isolation scores for each point"""
        if not data:
            return []

        n_samples = len(data)
        n_features = len(data[0])

        # Average path length for normalization
        def c(n):
            if n <= 1:
                return 0
            return 2 * (math.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n

        path_lengths = [0.0] * n_samples

        for _ in range(n_trees):
            # Build isolation tree
            indices = list(range(n_samples))

            for sample_idx in range(n_samples):
                # Simulate isolation of this point
                remaining = indices.copy()
                depth = 0
                max_depth = int(math.ceil(math.log2(n_samples + 1)))

                while len(remaining) > 1 and depth < max_depth:
                    # Random split
                    feature = random.randint(0, n_features - 1)
                    values = [data[i][feature] for i in remaining]
                    min_val, max_val = min(values), max(values)

                    if min_val == max_val:
                        break

                    split = random.uniform(min_val, max_val)

                    # Determine which side our sample goes
                    if data[sample_idx][feature] < split:
                        remaining = [i for i in remaining if data[i][feature] < split]
                    else:
                        remaining = [i for i in remaining if data[i][feature] >= split]

                    depth += 1

                path_lengths[sample_idx] += depth

        # Average and normalize
        c_n = c(n_samples)
        scores = []
        for pl in path_lengths:
            avg_pl = pl / n_trees
            score = 2 ** (-avg_pl / c_n) if c_n > 0 else 0.5
            scores.append(score)

        return scores

    def detect(self, data: List[List[float]]) -> AnomalyDetection:
        """Detect anomalies in data"""
        result = AnomalyDetection()

        if not data:
            return result

        scores = self.compute_isolation_scores(data)
        result.anomaly_scores = scores

        # Determine threshold
        sorted_scores = sorted(scores, reverse=True)
        threshold_idx = int(len(sorted_scores) * self.contamination)
        result.threshold = sorted_scores[threshold_idx] if threshold_idx < len(sorted_scores) else 0.5

        # Find anomalies
        result.anomaly_indices = [i for i, s in enumerate(scores) if s >= result.threshold]

        return result


# ============================================================================
# ULTRA VISUALIZATION
# ============================================================================

class UltraSimulationVisualizer:
    """Ultra-sophisticated visualization with advanced analytics"""

    def __init__(self, metrics: UltraSimulationMetrics):
        self.metrics = metrics
        self.causal_engine = CausalInferenceEngine()
        self.attractor_detector = AttractorDetector()
        self.recurrence_analyzer = RecurrencePlotAnalyzer()
        self.spectral_analyzer = SpectralAnalyzer()
        self.dim_reducer = DimensionalityReducer()
        self.anomaly_detector = AnomalyDetector()

    def create_ultra_dashboard(self, save_path: str = None, figsize: Tuple[int, int] = (24, 20)):
        """Create ultra-sophisticated 16-panel dashboard"""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available")
            return None

        fig = plt.figure(figsize=figsize, facecolor='#0a0a0a')
        fig.suptitle('Ultra-Sophisticated Simulation Analytics',
                    fontsize=18, color='white', fontweight='bold', y=0.98)

        gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3,
                     left=0.05, right=0.95, top=0.93, bottom=0.05)

        # Row 1: Phase space, attractors, and causal network
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        self._plot_3d_phase_space(ax1)

        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_attractor_analysis(ax2)

        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_causal_network(ax3)

        ax4 = fig.add_subplot(gs[0, 3])
        self._plot_transfer_entropy_matrix(ax4)

        # Row 2: Recurrence plots and spectral analysis
        ax5 = fig.add_subplot(gs[1, 0])
        self._plot_recurrence_plot(ax5)

        ax6 = fig.add_subplot(gs[1, 1])
        self._plot_recurrence_metrics(ax6)

        ax7 = fig.add_subplot(gs[1, 2])
        self._plot_power_spectrum(ax7)

        ax8 = fig.add_subplot(gs[1, 3])
        self._plot_wavelet_scalogram(ax8)

        # Row 3: Embeddings and anomalies
        ax9 = fig.add_subplot(gs[2, 0])
        self._plot_agent_embedding(ax9)

        ax10 = fig.add_subplot(gs[2, 1])
        self._plot_trajectory_embedding(ax10)

        ax11 = fig.add_subplot(gs[2, 2])
        self._plot_anomaly_detection(ax11)

        ax12 = fig.add_subplot(gs[2, 3])
        self._plot_information_decomposition(ax12)

        # Row 4: Dynamics and predictions
        ax13 = fig.add_subplot(gs[3, 0])
        self._plot_lyapunov_evolution(ax13)

        ax14 = fig.add_subplot(gs[3, 1])
        self._plot_multiscale_entropy(ax14)

        ax15 = fig.add_subplot(gs[3, 2])
        self._plot_cross_correlation(ax15)

        ax16 = fig.add_subplot(gs[3, 3])
        self._plot_system_summary(ax16)

        if save_path:
            plt.savefig(save_path, dpi=150, facecolor='#0a0a0a',
                       edgecolor='none', bbox_inches='tight')
            print(f"Dashboard saved to {save_path}")

        plt.close()
        return fig

    def _style_axis(self, ax, title: str, is_3d: bool = False):
        """Apply consistent styling to axis"""
        ax.set_facecolor('#1a1a2e')
        ax.set_title(title, color='white', fontsize=10, fontweight='bold', pad=8)

        if is_3d:
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('#333')
            ax.yaxis.pane.set_edgecolor('#333')
            ax.zaxis.pane.set_edgecolor('#333')
            ax.tick_params(colors='#888', labelsize=7)
        else:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#444')
            ax.spines['left'].set_color('#444')
            ax.tick_params(colors='#888', labelsize=8)
            ax.xaxis.label.set_color('#aaa')
            ax.yaxis.label.set_color('#aaa')

    def _plot_3d_phase_space(self, ax):
        """Plot 3D phase space trajectories"""
        self._style_axis(ax, '3D Phase Space Trajectories', is_3d=True)

        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dfe6e9']

        for idx, (agent_id, coords) in enumerate(self.metrics.phase_coordinates.items()):
            if len(coords) > 1:
                x = [c[0] for c in coords]
                y = [c[1] for c in coords]
                z = [c[2] for c in coords]

                color = colors[idx % len(colors)]
                ax.plot(x, y, z, color=color, linewidth=1.5, alpha=0.8, label=agent_id)

                # Mark start and end
                ax.scatter([x[0]], [y[0]], [z[0]], color=color, s=50, marker='o')
                ax.scatter([x[-1]], [y[-1]], [z[-1]], color=color, s=100, marker='*')

        ax.set_xlabel('Cooperation', fontsize=8)
        ax.set_ylabel('Aggression', fontsize=8)
        ax.set_zlabel('Trust', fontsize=8)
        ax.legend(loc='upper left', fontsize=7, facecolor='#1a1a2e',
                 edgecolor='#444', labelcolor='white')

    def _plot_attractor_analysis(self, ax):
        """Plot attractor analysis results"""
        self._style_axis(ax, 'Attractor Analysis')

        # Analyze system-level attractor
        if self.metrics.tension_signal:
            analysis = self.attractor_detector.analyze(self.metrics.tension_signal)

            # Show dynamics type
            dynamics_names = {
                DynamicsType.FIXED_POINT: 'Fixed Point',
                DynamicsType.LIMIT_CYCLE: 'Limit Cycle',
                DynamicsType.QUASI_PERIODIC: 'Quasi-Periodic',
                DynamicsType.CHAOTIC: 'Chaotic',
                DynamicsType.UNKNOWN: 'Unknown'
            }

            metrics_text = [
                f"Dynamics: {dynamics_names.get(analysis.dynamics_type, 'Unknown')}",
                f"Correlation Dim: {analysis.correlation_dimension:.3f}",
                f"Lyapunov Exp: {analysis.lyapunov_exponent:.4f}",
                f"Recurrence Rate: {analysis.recurrence_rate:.3f}",
                f"Fixed Points: {len(analysis.fixed_points)}"
            ]

            # Create bar chart of metrics
            metric_values = [
                analysis.correlation_dimension,
                abs(analysis.lyapunov_exponent) * 10,
                analysis.recurrence_rate * 10,
                len(analysis.fixed_points)
            ]
            metric_labels = ['Corr. Dim', 'Lyapunov', 'Recurr.', 'Fix. Pts']

            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
            bars = ax.bar(metric_labels, metric_values, color=colors, alpha=0.8)

            # Add value labels
            for bar, val in zip(bars, metric_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{val:.2f}', ha='center', va='bottom', color='white', fontsize=8)

            # Add dynamics type annotation
            ax.text(0.5, 0.95, dynamics_names.get(analysis.dynamics_type, 'Unknown'),
                   transform=ax.transAxes, ha='center', va='top',
                   fontsize=12, color='#ffeaa7', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   color='#888', fontsize=12, transform=ax.transAxes)

    def _plot_causal_network(self, ax):
        """Plot causal network from transfer entropy analysis"""
        self._style_axis(ax, 'Causal Network (Transfer Entropy)')

        # Prepare time series from state vectors
        time_series = {}
        for agent_id, states in self.metrics.state_vectors.items():
            if states and len(states) > 5:
                # Use first dimension (cooperation) as signal
                time_series[agent_id] = [s[0] if s else 0 for s in states]

        if len(time_series) >= 2:
            links = self.causal_engine.find_causal_links(time_series)

            # Position nodes in a circle
            agents = list(time_series.keys())
            n = len(agents)
            positions = {}
            for i, agent in enumerate(agents):
                angle = 2 * math.pi * i / n - math.pi / 2
                positions[agent] = (0.5 + 0.35 * math.cos(angle),
                                   0.5 + 0.35 * math.sin(angle))

            # Draw edges (causal links)
            for link in links[:10]:  # Top 10 links
                if link.source in positions and link.target in positions:
                    src = positions[link.source]
                    tgt = positions[link.target]

                    # Draw arrow
                    ax.annotate('', xy=tgt, xytext=src,
                               arrowprops=dict(
                                   arrowstyle='-|>',
                                   color='#ff6b6b',
                                   alpha=min(1.0, link.strength * 2),
                                   lw=1 + link.strength * 3,
                                   connectionstyle='arc3,rad=0.1'
                               ),
                               transform=ax.transAxes)

            # Draw nodes
            for agent, pos in positions.items():
                circle = plt.Circle(pos, 0.08, color='#4ecdc4',
                                   transform=ax.transAxes, zorder=10)
                ax.add_patch(circle)
                ax.text(pos[0], pos[1], agent[:3], ha='center', va='center',
                       fontsize=9, color='white', fontweight='bold',
                       transform=ax.transAxes, zorder=11)
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   color='#888', fontsize=10, transform=ax.transAxes)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])

    def _plot_transfer_entropy_matrix(self, ax):
        """Plot transfer entropy matrix"""
        self._style_axis(ax, 'Transfer Entropy Matrix')

        # Prepare time series
        time_series = {}
        for agent_id, states in self.metrics.state_vectors.items():
            if states and len(states) > 5:
                time_series[agent_id] = [s[0] if s else 0 for s in states]

        agents = list(time_series.keys())
        n = len(agents)

        if n >= 2:
            # Compute transfer entropy matrix
            te_matrix = [[0.0] * n for _ in range(n)]

            for i, src in enumerate(agents):
                for j, tgt in enumerate(agents):
                    if i != j:
                        te = self.causal_engine.transfer_entropy(
                            time_series[src], time_series[tgt]
                        )
                        te_matrix[i][j] = te

            # Plot heatmap
            im = ax.imshow(te_matrix, cmap='YlOrRd', aspect='auto')

            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels([a[:4] for a in agents], fontsize=8, color='white')
            ax.set_yticklabels([a[:4] for a in agents], fontsize=8, color='white')
            ax.set_xlabel('Target', fontsize=9)
            ax.set_ylabel('Source', fontsize=9)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(colors='white', labelsize=7)
            cbar.set_label('TE (bits)', color='white', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Need 2+ agents', ha='center', va='center',
                   color='#888', fontsize=10, transform=ax.transAxes)

    def _plot_recurrence_plot(self, ax):
        """Plot recurrence plot of system dynamics"""
        self._style_axis(ax, 'Recurrence Plot (Tension Signal)')

        if self.metrics.tension_signal and len(self.metrics.tension_signal) > 5:
            matrix = self.recurrence_analyzer.create_recurrence_matrix(
                self.metrics.tension_signal
            )

            if matrix:
                ax.imshow(matrix, cmap='binary', origin='lower', aspect='auto')
                ax.set_xlabel('Time (i)', fontsize=9)
                ax.set_ylabel('Time (j)', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   color='#888', fontsize=10, transform=ax.transAxes)

    def _plot_recurrence_metrics(self, ax):
        """Plot recurrence quantification metrics"""
        self._style_axis(ax, 'Recurrence Quantification')

        if self.metrics.tension_signal and len(self.metrics.tension_signal) > 5:
            matrix = self.recurrence_analyzer.create_recurrence_matrix(
                self.metrics.tension_signal
            )
            analysis = self.recurrence_analyzer.quantify_recurrence(matrix)

            metrics = {
                'RR': analysis.recurrence_rate,
                'DET': analysis.determinism,
                'L_avg': analysis.average_diagonal_length / 10,
                'LAM': analysis.laminarity,
                'TT': analysis.trapping_time / 10,
                'ENTR': analysis.entropy_diagonal / 5
            }

            # Radar chart
            angles = [i * 2 * math.pi / len(metrics) for i in range(len(metrics))]
            angles.append(angles[0])

            values = list(metrics.values())
            values.append(values[0])

            ax.plot(angles, values, 'o-', color='#4ecdc4', linewidth=2, alpha=0.8)
            ax.fill(angles, values, color='#4ecdc4', alpha=0.3)

            # Labels
            for i, (name, val) in enumerate(metrics.items()):
                angle = angles[i]
                ax.text(angle, 1.15, name, ha='center', va='center',
                       fontsize=8, color='white')

            ax.set_xlim(-0.2, 2*math.pi + 0.2)
            ax.set_ylim(-0.1, 1.2)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   color='#888', fontsize=10, transform=ax.transAxes)

    def _plot_power_spectrum(self, ax):
        """Plot power spectrum"""
        self._style_axis(ax, 'Power Spectrum')

        if self.metrics.tension_signal and len(self.metrics.tension_signal) > 8:
            analysis = self.spectral_analyzer.analyze(self.metrics.tension_signal)

            if analysis.power_spectrum:
                freqs = [f for f, p in analysis.power_spectrum]
                powers = [p for f, p in analysis.power_spectrum]

                ax.semilogy(freqs, powers, color='#ff6b6b', linewidth=1.5)
                ax.fill_between(freqs, powers, alpha=0.3, color='#ff6b6b')

                # Mark dominant frequencies
                for df in analysis.dominant_frequencies[:3]:
                    ax.axvline(df, color='#ffeaa7', linestyle='--', alpha=0.5)

                ax.set_xlabel('Frequency', fontsize=9)
                ax.set_ylabel('Power', fontsize=9)

                # Annotation
                ax.text(0.95, 0.95, f'Peak: {analysis.peak_frequency:.3f}\n'
                       f'Entropy: {analysis.spectral_entropy:.2f}',
                       transform=ax.transAxes, ha='right', va='top',
                       fontsize=8, color='white',
                       bbox=dict(boxstyle='round', facecolor='#333', alpha=0.7))
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   color='#888', fontsize=10, transform=ax.transAxes)

    def _plot_wavelet_scalogram(self, ax):
        """Plot wavelet-like scalogram"""
        self._style_axis(ax, 'Multi-Scale Decomposition')

        if self.metrics.tension_signal and len(self.metrics.tension_signal) > 16:
            coeffs = self.spectral_analyzer.wavelet_transform(self.metrics.tension_signal)

            if coeffs:
                # Create scalogram matrix
                scales = sorted(coeffs.keys())
                n_time = len(self.metrics.tension_signal)

                scalogram = []
                for scale in scales:
                    row = coeffs[scale]
                    # Pad or trim to match time length
                    if len(row) < n_time:
                        row = row + [0] * (n_time - len(row))
                    else:
                        row = row[:n_time]
                    scalogram.append([abs(v) for v in row])

                im = ax.imshow(scalogram, aspect='auto', cmap='viridis',
                              origin='lower', extent=[0, n_time, 0, len(scales)])

                ax.set_yticks(range(len(scales)))
                ax.set_yticklabels([str(s) for s in scales], fontsize=8, color='white')
                ax.set_xlabel('Time', fontsize=9)
                ax.set_ylabel('Scale', fontsize=9)

                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(colors='white', labelsize=7)
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   color='#888', fontsize=10, transform=ax.transAxes)

    def _plot_agent_embedding(self, ax):
        """Plot agent state embedding"""
        self._style_axis(ax, 'Agent State Embedding (t-SNE-like)')

        # Collect final states
        final_states = []
        agent_ids = []

        for agent_id, states in self.metrics.state_vectors.items():
            if states:
                final_states.append(states[-1])
                agent_ids.append(agent_id)

        if len(final_states) >= 3:
            # Reduce dimensionality
            embedded = self.dim_reducer.tsne_like_reduce(final_states)

            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dfe6e9']

            for i, (coord, agent_id) in enumerate(zip(embedded, agent_ids)):
                color = colors[i % len(colors)]
                ax.scatter(coord[0], coord[1], c=color, s=200, alpha=0.8)
                ax.annotate(agent_id, coord, fontsize=9, color='white',
                           ha='center', va='bottom', xytext=(0, 10),
                           textcoords='offset points')

            ax.set_xlabel('Embedding Dim 1', fontsize=9)
            ax.set_ylabel('Embedding Dim 2', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'Need 3+ agents', ha='center', va='center',
                   color='#888', fontsize=10, transform=ax.transAxes)

    def _plot_trajectory_embedding(self, ax):
        """Plot embedded trajectories"""
        self._style_axis(ax, 'Trajectory Manifold')

        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7']

        for idx, (agent_id, states) in enumerate(self.metrics.state_vectors.items()):
            if len(states) >= 5:
                # Embed trajectory
                embedded = self.dim_reducer.pca_reduce(states)

                x = [e[0] for e in embedded]
                y = [e[1] for e in embedded]

                color = colors[idx % len(colors)]
                ax.plot(x, y, color=color, linewidth=1.5, alpha=0.7, label=agent_id)
                ax.scatter(x[-1], y[-1], c=color, s=100, marker='*', zorder=10)

        ax.set_xlabel('PC 1', fontsize=9)
        ax.set_ylabel('PC 2', fontsize=9)
        ax.legend(loc='upper right', fontsize=7, facecolor='#1a1a2e',
                 edgecolor='#444', labelcolor='white')

    def _plot_anomaly_detection(self, ax):
        """Plot anomaly detection results"""
        self._style_axis(ax, 'Anomaly Detection')

        # Prepare data points
        all_states = []
        timestamps = []

        for agent_states in self.metrics.state_vectors.values():
            for i, state in enumerate(agent_states):
                all_states.append(state)
                timestamps.append(i)

        if len(all_states) >= 10:
            result = self.anomaly_detector.detect(all_states)

            # Plot scores over time
            ax.plot(result.anomaly_scores, color='#4ecdc4', linewidth=1.5, alpha=0.8)
            ax.axhline(result.threshold, color='#ff6b6b', linestyle='--',
                      linewidth=2, label=f'Threshold: {result.threshold:.2f}')

            # Highlight anomalies
            for idx in result.anomaly_indices:
                if idx < len(result.anomaly_scores):
                    ax.scatter(idx, result.anomaly_scores[idx], color='#ff6b6b',
                              s=100, zorder=10, marker='x')

            ax.set_xlabel('State Index', fontsize=9)
            ax.set_ylabel('Anomaly Score', fontsize=9)
            ax.legend(loc='upper right', fontsize=8, facecolor='#1a1a2e',
                     edgecolor='#444', labelcolor='white')

            # Annotation
            ax.text(0.02, 0.98, f'Anomalies: {len(result.anomaly_indices)}',
                   transform=ax.transAxes, ha='left', va='top',
                   fontsize=10, color='#ffeaa7', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   color='#888', fontsize=10, transform=ax.transAxes)

    def _plot_information_decomposition(self, ax):
        """Plot information decomposition"""
        self._style_axis(ax, 'Information Decomposition')

        # Compute simplified PID
        agents = list(self.metrics.state_vectors.keys())

        if len(agents) >= 2 and all(len(self.metrics.state_vectors[a]) > 5 for a in agents):
            # Get time series for first two agents
            series1 = [s[0] if s else 0 for s in self.metrics.state_vectors[agents[0]]]
            series2 = [s[0] if s else 0 for s in self.metrics.state_vectors[agents[1]]]

            # Compute mutual information (simplified)
            def entropy(series):
                if not series:
                    return 0
                hist = defaultdict(int)
                for x in series:
                    hist[round(x, 1)] += 1
                total = len(series)
                return -sum((c/total) * math.log2(c/total) for c in hist.values() if c > 0)

            h1 = entropy(series1)
            h2 = entropy(series2)
            h_joint = entropy([s1 + s2 for s1, s2 in zip(series1, series2)])

            mi = h1 + h2 - h_joint
            redundancy = min(h1, h2) * 0.3  # Simplified
            unique1 = max(0, h1 - redundancy)
            unique2 = max(0, h2 - redundancy)
            synergy = max(0, mi - redundancy)

            # Stacked bar chart
            categories = ['Redundancy', f'Unique({agents[0][:3]})',
                         f'Unique({agents[1][:3]})', 'Synergy']
            values = [redundancy, unique1, unique2, synergy]
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']

            bars = ax.bar(categories, values, color=colors, alpha=0.8)

            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       f'{val:.2f}', ha='center', va='bottom', color='white', fontsize=8)

            ax.set_ylabel('Information (bits)', fontsize=9)
            ax.tick_params(axis='x', rotation=15)

            # Total MI annotation
            ax.text(0.95, 0.95, f'MI: {mi:.2f} bits',
                   transform=ax.transAxes, ha='right', va='top',
                   fontsize=10, color='#ffeaa7', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Need 2+ agents', ha='center', va='center',
                   color='#888', fontsize=10, transform=ax.transAxes)

    def _plot_lyapunov_evolution(self, ax):
        """Plot Lyapunov exponent evolution"""
        self._style_axis(ax, 'Lyapunov Exponent Evolution')

        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7']

        for idx, (agent_id, states) in enumerate(self.metrics.state_vectors.items()):
            if len(states) >= 15:
                # Compute sliding window Lyapunov
                series = [s[0] if s else 0 for s in states]
                window_size = 10
                lyapunovs = []

                for i in range(len(series) - window_size):
                    window = series[i:i+window_size]
                    le = self.attractor_detector.estimate_lyapunov(window)
                    lyapunovs.append(le)

                if lyapunovs:
                    color = colors[idx % len(colors)]
                    ax.plot(range(len(lyapunovs)), lyapunovs,
                           color=color, linewidth=1.5, alpha=0.8, label=agent_id)

        ax.axhline(0, color='white', linestyle='--', alpha=0.3)
        ax.set_xlabel('Time Window', fontsize=9)
        ax.set_ylabel('Lyapunov Exponent', fontsize=9)
        ax.legend(loc='upper right', fontsize=7, facecolor='#1a1a2e',
                 edgecolor='#444', labelcolor='white')

        # Chaos indicator
        ax.text(0.02, 0.98, 'λ > 0: Chaotic\nλ < 0: Stable',
               transform=ax.transAxes, ha='left', va='top',
               fontsize=8, color='#888')

    def _plot_multiscale_entropy(self, ax):
        """Plot multi-scale entropy analysis"""
        self._style_axis(ax, 'Multi-Scale Entropy')

        if self.metrics.tension_signal and len(self.metrics.tension_signal) > 16:
            scales = [1, 2, 4, 8]
            entropies = []

            for scale in scales:
                # Coarse-grain the signal
                coarse = []
                for i in range(0, len(self.metrics.tension_signal) - scale + 1, scale):
                    window = self.metrics.tension_signal[i:i+scale]
                    coarse.append(sum(window) / len(window))

                # Compute sample entropy (simplified)
                if len(coarse) > 5:
                    hist = defaultdict(int)
                    for x in coarse:
                        hist[round(x, 1)] += 1
                    total = len(coarse)
                    entropy = -sum((c/total) * math.log2(c/total)
                                  for c in hist.values() if c > 0)
                    entropies.append(entropy)
                else:
                    entropies.append(0)

            ax.plot(scales, entropies, 'o-', color='#4ecdc4', linewidth=2,
                   markersize=8, alpha=0.8)
            ax.fill_between(scales, entropies, alpha=0.3, color='#4ecdc4')

            ax.set_xlabel('Scale', fontsize=9)
            ax.set_ylabel('Entropy', fontsize=9)
            ax.set_xscale('log', base=2)

            # Complexity indicator
            if len(entropies) >= 2:
                slope = (entropies[-1] - entropies[0]) / (scales[-1] - scales[0])
                complexity = "High" if abs(slope) < 0.1 else ("Low" if slope < 0 else "Medium")
                ax.text(0.95, 0.95, f'Complexity: {complexity}',
                       transform=ax.transAxes, ha='right', va='top',
                       fontsize=10, color='#ffeaa7', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   color='#888', fontsize=10, transform=ax.transAxes)

    def _plot_cross_correlation(self, ax):
        """Plot cross-correlation analysis"""
        self._style_axis(ax, 'Cross-Correlation (Lagged)')

        agents = list(self.metrics.state_vectors.keys())

        if len(agents) >= 2:
            series1 = [s[0] if s else 0 for s in self.metrics.state_vectors[agents[0]]]
            series2 = [s[0] if s else 0 for s in self.metrics.state_vectors[agents[1]]]

            if len(series1) > 10 and len(series2) > 10:
                # Compute cross-correlation at different lags
                max_lag = min(10, len(series1) // 2)
                lags = range(-max_lag, max_lag + 1)
                correlations = []

                mean1 = sum(series1) / len(series1)
                mean2 = sum(series2) / len(series2)
                std1 = math.sqrt(sum((x - mean1)**2 for x in series1) / len(series1))
                std2 = math.sqrt(sum((x - mean2)**2 for x in series2) / len(series2))

                for lag in lags:
                    if std1 > 0 and std2 > 0:
                        n = min(len(series1), len(series2)) - abs(lag)
                        if lag >= 0:
                            corr = sum((series1[i] - mean1) * (series2[i + lag] - mean2)
                                      for i in range(n)) / (n * std1 * std2)
                        else:
                            corr = sum((series1[i - lag] - mean1) * (series2[i] - mean2)
                                      for i in range(n)) / (n * std1 * std2)
                        correlations.append(corr)
                    else:
                        correlations.append(0)

                ax.bar(list(lags), correlations, color='#4ecdc4', alpha=0.8)
                ax.axhline(0, color='white', linestyle='-', alpha=0.3)
                ax.axhline(0.2, color='#ffeaa7', linestyle='--', alpha=0.5)
                ax.axhline(-0.2, color='#ffeaa7', linestyle='--', alpha=0.5)

                ax.set_xlabel('Lag', fontsize=9)
                ax.set_ylabel('Correlation', fontsize=9)
                ax.set_title(f'Cross-Corr: {agents[0][:4]} vs {agents[1][:4]}',
                           color='white', fontsize=10)

                # Find peak
                peak_lag = lags[correlations.index(max(correlations))]
                ax.text(0.95, 0.95, f'Peak lag: {peak_lag}',
                       transform=ax.transAxes, ha='right', va='top',
                       fontsize=9, color='#ffeaa7')
        else:
            ax.text(0.5, 0.5, 'Need 2+ agents', ha='center', va='center',
                   color='#888', fontsize=10, transform=ax.transAxes)

    def _plot_system_summary(self, ax):
        """Plot overall system summary"""
        self._style_axis(ax, 'System Summary')

        # Compute summary statistics
        n_agents = len(self.metrics.state_vectors)
        n_rounds = len(self.metrics.rounds)

        # Average final metrics
        avg_tension = sum(self.metrics.tension_signal) / len(self.metrics.tension_signal) if self.metrics.tension_signal else 0
        avg_entropy = sum(self.metrics.entropy_signal) / len(self.metrics.entropy_signal) if self.metrics.entropy_signal else 0
        avg_complexity = sum(self.metrics.complexity_signal) / len(self.metrics.complexity_signal) if self.metrics.complexity_signal else 0
        avg_phi = sum(self.metrics.global_phi) / len(self.metrics.global_phi) if self.metrics.global_phi else 0

        # Dynamics classification
        if self.metrics.tension_signal:
            analysis = self.attractor_detector.analyze(self.metrics.tension_signal)
            dynamics = analysis.dynamics_type.value.replace('_', ' ').title()
        else:
            dynamics = "Unknown"

        # Display as text
        summary_text = f"""
System Overview
{'='*20}

Agents: {n_agents}
Rounds: {n_rounds}

Avg Tension: {avg_tension:.3f}
Avg Entropy: {avg_entropy:.3f}
Avg Complexity: {avg_complexity:.3f}
Avg Φ (IIT): {avg_phi:.3f}

Dynamics: {dynamics}
Causal Links: {len(self.metrics.causal_links)}
"""

        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               fontsize=10, color='white', fontfamily='monospace',
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='#333', alpha=0.7))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])


# ============================================================================
# ULTRA ASCII VISUALIZATION
# ============================================================================

class UltraASCIIVisualizer:
    """Ultra-sophisticated ASCII visualization"""

    def __init__(self, metrics: UltraSimulationMetrics):
        self.metrics = metrics

    def render_full_report(self) -> str:
        """Render comprehensive ASCII report"""
        lines = []

        # Header
        lines.append("╔══════════════════════════════════════════════════════════════════════════════╗")
        lines.append("║         ULTRA-SOPHISTICATED SIMULATION ANALYTICS REPORT                      ║")
        lines.append("╠══════════════════════════════════════════════════════════════════════════════╣")
        lines.append("")

        # System Overview
        lines.append("┌─ SYSTEM OVERVIEW ──────────────────────────────────────────────────────────┐")
        lines.append(f"│ Agents: {len(self.metrics.state_vectors):3d}    Rounds: {len(self.metrics.rounds):3d}    "
                    f"Interactions: {sum(len(i) for i in self.metrics.interactions):4d}          │")
        lines.append("└─────────────────────────────────────────────────────────────────────────────┘")
        lines.append("")

        # Phase Space Summary
        lines.append("┌─ PHASE SPACE ANALYSIS ───────────────────────────────────────────────────────┐")
        for agent_id, coords in self.metrics.phase_coordinates.items():
            if coords:
                final = coords[-1]
                trajectory_length = sum(
                    math.sqrt(sum((coords[i][j] - coords[i-1][j])**2 for j in range(3)))
                    for i in range(1, len(coords))
                )
                lines.append(f"│ {agent_id:10s} │ Final: ({final[0]:5.2f}, {final[1]:5.2f}, {final[2]:5.2f}) │ "
                           f"Path Length: {trajectory_length:6.2f} │")
        lines.append("└────────────────────────────────────────────────────────────────────────────────┘")
        lines.append("")

        # Attractor Analysis
        if self.metrics.tension_signal:
            detector = AttractorDetector()
            analysis = detector.analyze(self.metrics.tension_signal)

            lines.append("┌─ ATTRACTOR ANALYSIS ─────────────────────────────────────────────────────────┐")
            lines.append(f"│ Dynamics Type: {analysis.dynamics_type.value:15s}                              │")
            lines.append(f"│ Correlation Dimension: {analysis.correlation_dimension:6.3f}                              │")
            lines.append(f"│ Lyapunov Exponent: {analysis.lyapunov_exponent:+7.4f}  "
                        f"({'CHAOTIC' if analysis.lyapunov_exponent > 0 else 'STABLE':7s})                │")
            lines.append(f"│ Recurrence Rate: {analysis.recurrence_rate:6.3f}                                        │")
            lines.append(f"│ Fixed Points Found: {len(analysis.fixed_points):3d}                                        │")
            lines.append("└────────────────────────────────────────────────────────────────────────────────┘")
            lines.append("")

        # Causal Network (text representation)
        lines.append("┌─ CAUSAL NETWORK (Transfer Entropy) ─────────────────────────────────────────┐")

        engine = CausalInferenceEngine()
        time_series = {}
        for agent_id, states in self.metrics.state_vectors.items():
            if states and len(states) > 5:
                time_series[agent_id] = [s[0] if s else 0 for s in states]

        if len(time_series) >= 2:
            links = engine.find_causal_links(time_series)
            for link in links[:5]:
                arrow = "═══>" if link.strength > 0.5 else "───>"
                lines.append(f"│ {link.source:10s} {arrow} {link.target:10s} "
                           f"│ TE: {link.strength:5.3f} │ Lag: {link.lag:2d} │")
        else:
            lines.append("│ Insufficient data for causal analysis                                       │")
        lines.append("└────────────────────────────────────────────────────────────────────────────────┘")
        lines.append("")

        # Global Signals
        lines.append("┌─ GLOBAL SIGNAL EVOLUTION ────────────────────────────────────────────────────┐")
        if self.metrics.tension_signal:
            lines.append(self._ascii_sparkline("Tension   ", self.metrics.tension_signal))
        if self.metrics.entropy_signal:
            lines.append(self._ascii_sparkline("Entropy   ", self.metrics.entropy_signal))
        if self.metrics.complexity_signal:
            lines.append(self._ascii_sparkline("Complexity", self.metrics.complexity_signal))
        if self.metrics.global_phi:
            lines.append(self._ascii_sparkline("Φ (IIT)   ", self.metrics.global_phi))
        lines.append("└────────────────────────────────────────────────────────────────────────────────┘")
        lines.append("")

        # Footer
        lines.append("╚══════════════════════════════════════════════════════════════════════════════╝")

        return "\n".join(lines)

    def _ascii_sparkline(self, label: str, data: List[float], width: int = 50) -> str:
        """Create ASCII sparkline"""
        if not data:
            return f"│ {label}: No data"

        blocks = " ▁▂▃▄▅▆▇█"
        min_val, max_val = min(data), max(data)
        range_val = max_val - min_val if max_val != min_val else 1

        # Resample to width
        step = max(1, len(data) // width)
        resampled = [data[i] for i in range(0, len(data), step)][:width]

        sparkline = ""
        for val in resampled:
            idx = int((val - min_val) / range_val * (len(blocks) - 1))
            sparkline += blocks[idx]

        return f"│ {label}: {sparkline} │ [{min_val:.2f} - {max_val:.2f}]"


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_ultra_metrics() -> UltraSimulationMetrics:
    """Create new ultra metrics collector"""
    return UltraSimulationMetrics()


def visualize_ultra(metrics: UltraSimulationMetrics,
                   save_path: str = None) -> Optional[Any]:
    """Create ultra visualization"""
    viz = UltraSimulationVisualizer(metrics)
    return viz.create_ultra_dashboard(save_path)


def demo_ultra_visualization():
    """Demo the ultra visualization system"""
    print("=" * 60)
    print("ULTRA-SOPHISTICATED VISUALIZATION DEMO")
    print("=" * 60)

    # Create metrics
    metrics = UltraSimulationMetrics()

    # Simulate data
    agents = ["Alice", "Bob", "Charlie", "Diana"]
    n_rounds = 30

    # Agent states with chaotic dynamics
    agent_states = {a: [0.5, 0.5, 0.5, 1.0, 0.5, 0.3, 0.5] for a in agents}

    print(f"\nSimulating {n_rounds} rounds with {len(agents)} agents...")

    for r in range(n_rounds):
        metrics.record_round(r)

        # Update each agent
        for i, agent in enumerate(agents):
            # Chaotic logistic map dynamics
            x = agent_states[agent][0]
            new_x = 3.9 * x * (1 - x) + random.gauss(0, 0.05)
            new_x = max(0, min(1, new_x))

            # Update state
            agent_states[agent][0] = new_x
            agent_states[agent][1] = 0.5 + 0.3 * math.sin(r / 5 + i)
            agent_states[agent][2] = 0.5 + 0.2 * math.cos(r / 3 + i * 2)

            # Phase coordinates
            phase = (
                agent_states[agent][0],
                agent_states[agent][1],
                agent_states[agent][2]
            )

            # Beliefs
            beliefs = {
                "cooperation": new_x,
                "trust": 0.5 + 0.3 * math.sin(r / 4),
                "risk": random.random()
            }

            # Action
            action = random.choice(["cooperate", "defect", "negotiate"])

            metrics.record_agent_state(
                agent,
                agent_states[agent].copy(),
                phase,
                beliefs,
                action
            )

        # Global state
        tension = 0.3 + 0.2 * math.sin(r / 5) + random.gauss(0, 0.05)
        entropy = 0.5 + 0.1 * r / n_rounds
        complexity = 0.7 + 0.1 * math.sin(r / 3)
        phi = 0.3 + 0.1 * math.cos(r / 4)

        metrics.record_global_state(tension, entropy, complexity, phi)

        # Interactions
        for a1, a2 in itertools.combinations(agents, 2):
            metrics.record_interaction(
                a1, a2, "exchange",
                random.random(),
                random.random() * 0.5
            )

    print("✓ Simulation complete")

    # Create visualization
    if MATPLOTLIB_AVAILABLE:
        print("\nGenerating ultra dashboard...")
        viz = UltraSimulationVisualizer(metrics)
        viz.create_ultra_dashboard("ultra_dashboard.png")
        print("✓ Dashboard saved to ultra_dashboard.png")

    # ASCII report
    print("\n" + "=" * 60)
    ascii_viz = UltraASCIIVisualizer(metrics)
    print(ascii_viz.render_full_report())

    return metrics


if __name__ == "__main__":
    demo_ultra_visualization()
