"""
Omega-Level Visualization Module

Cutting-edge mathematical analysis and visualization featuring:
- Topological Data Analysis (persistent homology, Betti numbers)
- Quantum-inspired measures (von Neumann entropy, entanglement)
- Criticality and phase transition detection
- Symbolic dynamics and permutation entropy
- Koopman operator spectral analysis
- Dynamic Mode Decomposition (DMD)
- Fractal and multifractal analysis
- Optimal transport (Wasserstein distance)
- Fisher information geometry
- Integrated Information Theory (IIT) structures
- Non-equilibrium thermodynamics
- Strange attractor characterization
- Detrended Fluctuation Analysis (DFA)
- Renyi entropy spectrum
- Category-theoretic relationship mapping
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable, Set, FrozenSet
from datetime import datetime
from enum import Enum
from collections import defaultdict, deque, Counter
from functools import lru_cache
import math
import random
import itertools
import heapq

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm
    from matplotlib.collections import LineCollection, PolyCollection
    import matplotlib.cm as cm
    from matplotlib.patches import Polygon, Circle, FancyArrowPatch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ============================================================================
# MATHEMATICAL CONSTANTS AND TYPES
# ============================================================================

PHI = (1 + math.sqrt(5)) / 2  # Golden ratio
EULER_MASCHERONI = 0.5772156649


class TopologicalFeature(Enum):
    """Types of topological features"""
    CONNECTED_COMPONENT = "H0"  # 0-dimensional holes
    LOOP = "H1"  # 1-dimensional holes
    VOID = "H2"  # 2-dimensional holes
    HIGHER = "Hn"  # Higher dimensional


class CriticalityRegime(Enum):
    """Criticality regimes"""
    SUBCRITICAL = "subcritical"
    CRITICAL = "critical"
    SUPERCRITICAL = "supercritical"
    EDGE_OF_CHAOS = "edge_of_chaos"


class SymbolicPattern(Enum):
    """Symbolic dynamics pattern types"""
    PERIODIC = "periodic"
    QUASI_PERIODIC = "quasi_periodic"
    CHAOTIC = "chaotic"
    RANDOM = "random"


# ============================================================================
# TOPOLOGICAL DATA ANALYSIS
# ============================================================================

@dataclass
class PersistenceInterval:
    """A persistence interval (birth, death) for a topological feature"""
    birth: float
    death: float
    dimension: int
    generator: Optional[Tuple[int, ...]] = None

    @property
    def persistence(self) -> float:
        return self.death - self.birth if self.death != float('inf') else float('inf')

    @property
    def midpoint(self) -> float:
        if self.death == float('inf'):
            return self.birth + 1.0
        return (self.birth + self.death) / 2


@dataclass
class PersistenceDiagram:
    """Persistence diagram from TDA"""
    intervals: List[PersistenceInterval] = field(default_factory=list)
    betti_numbers: Dict[int, int] = field(default_factory=dict)
    total_persistence: float = 0.0
    persistent_entropy: float = 0.0

    def get_betti(self, dim: int) -> int:
        return self.betti_numbers.get(dim, 0)


class RipsComplex:
    """Vietoris-Rips complex for TDA"""

    def __init__(self, points: List[Tuple[float, ...]], max_dim: int = 2):
        self.points = points
        self.max_dim = max_dim
        self.simplices: Dict[int, List[FrozenSet[int]]] = defaultdict(list)

    def _distance(self, i: int, j: int) -> float:
        """Euclidean distance between points"""
        p1, p2 = self.points[i], self.points[j]
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    def build_filtration(self, max_epsilon: float, n_steps: int = 20) -> List[Tuple[float, FrozenSet[int]]]:
        """Build Rips filtration"""
        filtration = []
        n = len(self.points)

        # Pre-compute distances
        distances = {}
        for i in range(n):
            for j in range(i + 1, n):
                distances[(i, j)] = self._distance(i, j)

        # Add 0-simplices (vertices) at epsilon=0
        for i in range(n):
            filtration.append((0.0, frozenset([i])))

        # Add 1-simplices (edges) at their distance
        for (i, j), d in sorted(distances.items(), key=lambda x: x[1]):
            if d <= max_epsilon:
                filtration.append((d, frozenset([i, j])))

        # Add 2-simplices (triangles)
        if self.max_dim >= 2:
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        d_ij = distances.get((i, j), self._distance(i, j))
                        d_jk = distances.get((j, k), self._distance(j, k))
                        d_ik = distances.get((i, k), self._distance(i, k))
                        max_d = max(d_ij, d_jk, d_ik)
                        if max_d <= max_epsilon:
                            filtration.append((max_d, frozenset([i, j, k])))

        return sorted(filtration, key=lambda x: (x[0], len(x[1])))

    def compute_persistence(self, max_epsilon: float) -> PersistenceDiagram:
        """Compute persistence diagram using incremental algorithm"""
        filtration = self.build_filtration(max_epsilon)
        diagram = PersistenceDiagram()

        # Track connected components (H0)
        parent = list(range(len(self.points)))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y, birth_time):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False

        component_births = {i: 0.0 for i in range(len(self.points))}
        h0_count = len(self.points)

        for epsilon, simplex in filtration:
            if len(simplex) == 2:  # Edge
                vertices = list(simplex)
                if union(vertices[0], vertices[1], epsilon):
                    # Component merged - one dies
                    h0_count -= 1
                    # The younger component dies
                    v0, v1 = vertices
                    birth = max(component_births[find(v0)], component_births[find(v1)])
                    if epsilon > birth:
                        diagram.intervals.append(PersistenceInterval(
                            birth=birth, death=epsilon, dimension=0
                        ))

        # Remaining components persist to infinity
        roots = set(find(i) for i in range(len(self.points)))
        for root in roots:
            diagram.intervals.append(PersistenceInterval(
                birth=0.0, death=float('inf'), dimension=0
            ))

        # Compute Betti numbers at max_epsilon
        diagram.betti_numbers[0] = len(roots)

        # Compute total persistence and persistent entropy
        finite_intervals = [iv for iv in diagram.intervals if iv.death != float('inf')]
        if finite_intervals:
            persistences = [iv.persistence for iv in finite_intervals]
            diagram.total_persistence = sum(persistences)
            total = sum(persistences)
            if total > 0:
                probs = [p / total for p in persistences]
                diagram.persistent_entropy = -sum(p * math.log2(p) for p in probs if p > 0)

        return diagram


# ============================================================================
# QUANTUM-INSPIRED MEASURES
# ============================================================================

@dataclass
class QuantumState:
    """Quantum-inspired state representation"""
    amplitudes: List[complex] = field(default_factory=list)
    density_matrix: List[List[complex]] = field(default_factory=list)

    @property
    def dimension(self) -> int:
        return len(self.amplitudes)


class QuantumAnalyzer:
    """Quantum-inspired analysis of classical systems"""

    def classical_to_quantum(self, probabilities: List[float]) -> QuantumState:
        """Convert probability distribution to quantum state"""
        # Amplitudes are square roots of probabilities
        amplitudes = [complex(math.sqrt(max(0, p)), 0) for p in probabilities]

        # Normalize
        norm = math.sqrt(sum(abs(a) ** 2 for a in amplitudes))
        if norm > 0:
            amplitudes = [a / norm for a in amplitudes]

        # Compute density matrix (pure state: |ψ⟩⟨ψ|)
        n = len(amplitudes)
        density_matrix = [[amplitudes[i] * amplitudes[j].conjugate()
                          for j in range(n)] for i in range(n)]

        return QuantumState(amplitudes=amplitudes, density_matrix=density_matrix)

    def von_neumann_entropy(self, state: QuantumState) -> float:
        """Compute von Neumann entropy S = -Tr(ρ log ρ)"""
        # For a pure state, S = 0
        # For mixed states, we need eigenvalues of density matrix

        n = len(state.density_matrix)
        if n == 0:
            return 0.0

        # Power iteration to estimate eigenvalues (simplified)
        eigenvalues = []

        # Trace should be 1 for density matrix
        trace = sum(state.density_matrix[i][i].real for i in range(n))

        # For pure states, one eigenvalue is 1, rest are 0
        # Estimate purity: Tr(ρ²)
        purity = 0.0
        for i in range(n):
            for j in range(n):
                purity += (state.density_matrix[i][j] *
                          state.density_matrix[j][i]).real

        # Approximate entropy from purity
        # For maximally mixed state: S = log(n), purity = 1/n
        # For pure state: S = 0, purity = 1
        if purity >= 1.0:
            return 0.0

        # Linear interpolation approximation
        max_entropy = math.log2(n) if n > 1 else 0
        return max_entropy * (1 - purity)

    def entanglement_measure(self, joint_probs: List[List[float]],
                            marginal1: List[float],
                            marginal2: List[float]) -> float:
        """Compute quantum mutual information as entanglement proxy"""
        # I(A:B) = S(A) + S(B) - S(AB)
        def entropy(probs):
            return -sum(p * math.log2(p) for p in probs if p > 0)

        s_a = entropy(marginal1)
        s_b = entropy(marginal2)

        # Flatten joint probabilities
        joint_flat = [p for row in joint_probs for p in row if p > 0]
        s_ab = entropy(joint_flat)

        return s_a + s_b - s_ab

    def quantum_discord(self, correlations: List[Tuple[float, float]]) -> float:
        """Estimate quantum discord from classical correlations"""
        if len(correlations) < 2:
            return 0.0

        # Compute classical mutual information
        x_vals = [c[0] for c in correlations]
        y_vals = [c[1] for c in correlations]

        # Discretize
        n_bins = min(5, len(correlations) // 2 + 1)

        def discretize(vals, n_bins):
            if not vals:
                return []
            min_v, max_v = min(vals), max(vals)
            if max_v == min_v:
                return [0] * len(vals)
            return [min(n_bins - 1, int((v - min_v) / (max_v - min_v + 1e-10) * n_bins))
                    for v in vals]

        x_disc = discretize(x_vals, n_bins)
        y_disc = discretize(y_vals, n_bins)

        # Joint distribution
        joint = defaultdict(int)
        for x, y in zip(x_disc, y_disc):
            joint[(x, y)] += 1

        total = len(correlations)

        # Marginals
        px = defaultdict(int)
        py = defaultdict(int)
        for x, y in zip(x_disc, y_disc):
            px[x] += 1
            py[y] += 1

        # Classical mutual information
        mi = 0.0
        for (x, y), count in joint.items():
            p_xy = count / total
            p_x = px[x] / total
            p_y = py[y] / total
            if p_xy > 0 and p_x > 0 and p_y > 0:
                mi += p_xy * math.log2(p_xy / (p_x * p_y))

        # Discord is difference between quantum and classical MI
        # Approximate as deviation from perfect correlation
        return mi * (1 - abs(sum((x - sum(x_vals)/len(x_vals)) * (y - sum(y_vals)/len(y_vals))
                                for x, y in zip(x_vals, y_vals)) /
                           (len(correlations) *
                            (max(0.01, math.sqrt(sum((x - sum(x_vals)/len(x_vals))**2 for x in x_vals) / len(x_vals))) *
                             max(0.01, math.sqrt(sum((y - sum(y_vals)/len(y_vals))**2 for y in y_vals) / len(y_vals)))))))


# ============================================================================
# CRITICALITY AND PHASE TRANSITIONS
# ============================================================================

@dataclass
class CriticalityAnalysis:
    """Results of criticality analysis"""
    regime: CriticalityRegime = CriticalityRegime.SUBCRITICAL
    order_parameter: float = 0.0
    susceptibility: float = 0.0
    correlation_length: float = 0.0
    power_law_exponent: float = 0.0
    avalanche_sizes: List[int] = field(default_factory=list)
    branching_ratio: float = 0.0


class CriticalityDetector:
    """Detects criticality and phase transitions"""

    def compute_order_parameter(self, states: List[float]) -> float:
        """Compute order parameter (magnetization-like)"""
        if not states:
            return 0.0
        return abs(sum(2 * s - 1 for s in states) / len(states))

    def compute_susceptibility(self, states: List[float]) -> float:
        """Compute susceptibility (variance of order parameter)"""
        if len(states) < 2:
            return 0.0

        order_params = []
        window = max(2, len(states) // 10)

        for i in range(0, len(states) - window, window // 2):
            window_states = states[i:i + window]
            op = self.compute_order_parameter(window_states)
            order_params.append(op)

        if len(order_params) < 2:
            return 0.0

        mean_op = sum(order_params) / len(order_params)
        variance = sum((op - mean_op) ** 2 for op in order_params) / len(order_params)

        return variance * len(states)  # χ = N * Var(m)

    def detect_avalanches(self, activity: List[float], threshold: float = 0.5) -> List[int]:
        """Detect avalanche sizes in activity time series"""
        avalanches = []
        current_size = 0
        in_avalanche = False

        mean_activity = sum(activity) / len(activity) if activity else 0

        for a in activity:
            if a > mean_activity * (1 + threshold):
                current_size += 1
                in_avalanche = True
            elif in_avalanche:
                if current_size > 0:
                    avalanches.append(current_size)
                current_size = 0
                in_avalanche = False

        if current_size > 0:
            avalanches.append(current_size)

        return avalanches

    def fit_power_law(self, sizes: List[int]) -> Tuple[float, float]:
        """Fit power law to avalanche sizes, returns (exponent, r_squared)"""
        if len(sizes) < 5:
            return 0.0, 0.0

        # Count frequencies
        freq = Counter(sizes)
        x = [math.log(k) for k in freq.keys() if k > 0]
        y = [math.log(v) for v in freq.values()]

        if len(x) < 3:
            return 0.0, 0.0

        # Linear regression in log-log space
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi ** 2 for xi in x)

        denom = n * sum_x2 - sum_x ** 2
        if abs(denom) < 1e-10:
            return 0.0, 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n

        # R-squared
        y_pred = [slope * xi + intercept for xi in x]
        ss_res = sum((yi - yp) ** 2 for yi, yp in zip(y, y_pred))
        y_mean = sum_y / n
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)

        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return -slope, r_squared  # Negative because power law has negative exponent

    def compute_branching_ratio(self, activity: List[float]) -> float:
        """Compute branching ratio σ = ⟨descendants⟩"""
        if len(activity) < 2:
            return 0.0

        # Estimate from autocorrelation at lag 1
        mean_a = sum(activity) / len(activity)
        var_a = sum((a - mean_a) ** 2 for a in activity) / len(activity)

        if var_a < 1e-10:
            return 0.0

        cov = sum((activity[i] - mean_a) * (activity[i + 1] - mean_a)
                  for i in range(len(activity) - 1)) / (len(activity) - 1)

        return cov / var_a

    def analyze(self, states: List[float], activity: List[float] = None) -> CriticalityAnalysis:
        """Full criticality analysis"""
        analysis = CriticalityAnalysis()

        if activity is None:
            activity = states

        analysis.order_parameter = self.compute_order_parameter(states)
        analysis.susceptibility = self.compute_susceptibility(states)
        analysis.avalanche_sizes = self.detect_avalanches(activity)
        analysis.branching_ratio = self.compute_branching_ratio(activity)

        if analysis.avalanche_sizes:
            exponent, r_sq = self.fit_power_law(analysis.avalanche_sizes)
            analysis.power_law_exponent = exponent

        # Classify regime
        sigma = analysis.branching_ratio

        if 0.9 <= sigma <= 1.1:
            analysis.regime = CriticalityRegime.CRITICAL
        elif sigma < 0.9:
            analysis.regime = CriticalityRegime.SUBCRITICAL
        elif sigma > 1.1:
            analysis.regime = CriticalityRegime.SUPERCRITICAL

        # Check for edge of chaos
        if 0.95 <= sigma <= 1.05 and 1.5 <= analysis.power_law_exponent <= 2.5:
            analysis.regime = CriticalityRegime.EDGE_OF_CHAOS

        return analysis


# ============================================================================
# SYMBOLIC DYNAMICS
# ============================================================================

@dataclass
class SymbolicAnalysis:
    """Results of symbolic dynamics analysis"""
    symbol_sequence: str = ""
    alphabet_size: int = 0
    topological_entropy: float = 0.0
    permutation_entropy: float = 0.0
    forbidden_patterns: List[str] = field(default_factory=list)
    pattern_type: SymbolicPattern = SymbolicPattern.RANDOM
    complexity: float = 0.0


class SymbolicDynamicsAnalyzer:
    """Analyzes symbolic dynamics of time series"""

    def symbolize(self, series: List[float], n_symbols: int = 4) -> str:
        """Convert time series to symbol sequence"""
        if not series:
            return ""

        min_val, max_val = min(series), max(series)
        if max_val == min_val:
            return "A" * len(series)

        symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n_symbols]
        result = []

        for val in series:
            idx = min(n_symbols - 1, int((val - min_val) / (max_val - min_val + 1e-10) * n_symbols))
            result.append(symbols[idx])

        return "".join(result)

    def compute_topological_entropy(self, sequence: str, max_length: int = 6) -> float:
        """Compute topological entropy h = lim (log W_n) / n"""
        if len(sequence) < max_length:
            return 0.0

        entropies = []
        for n in range(1, max_length + 1):
            # Count distinct n-grams
            ngrams = set()
            for i in range(len(sequence) - n + 1):
                ngrams.add(sequence[i:i + n])

            w_n = len(ngrams)
            if w_n > 0:
                entropies.append(math.log(w_n) / n)

        return entropies[-1] if entropies else 0.0

    def compute_permutation_entropy(self, series: List[float], order: int = 3,
                                   delay: int = 1) -> float:
        """Compute permutation entropy"""
        if len(series) < order * delay:
            return 0.0

        # Generate permutation patterns
        pattern_counts = defaultdict(int)
        total = 0

        for i in range(len(series) - (order - 1) * delay):
            # Extract embedding vector
            pattern = [series[i + j * delay] for j in range(order)]
            # Get permutation (rank order)
            perm = tuple(sorted(range(order), key=lambda x: pattern[x]))
            pattern_counts[perm] += 1
            total += 1

        if total == 0:
            return 0.0

        # Compute entropy
        entropy = 0.0
        for count in pattern_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize by maximum entropy
        max_entropy = math.log2(math.factorial(order))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def find_forbidden_patterns(self, sequence: str, max_length: int = 4) -> List[str]:
        """Find forbidden patterns (patterns that never occur)"""
        if not sequence:
            return []

        alphabet = sorted(set(sequence))
        forbidden = []

        for n in range(2, min(max_length + 1, len(sequence))):
            # Generate all possible n-grams
            possible = set()
            for combo in itertools.product(alphabet, repeat=n):
                possible.add("".join(combo))

            # Find observed n-grams
            observed = set()
            for i in range(len(sequence) - n + 1):
                observed.add(sequence[i:i + n])

            # Forbidden = possible - observed
            forbidden.extend(list(possible - observed)[:10])  # Limit output

        return forbidden[:20]

    def classify_pattern(self, series: List[float], sequence: str) -> SymbolicPattern:
        """Classify the type of symbolic pattern"""
        if len(sequence) < 10:
            return SymbolicPattern.RANDOM

        # Check for periodicity
        for period in range(1, len(sequence) // 3):
            is_periodic = True
            for i in range(len(sequence) - period):
                if sequence[i] != sequence[i + period]:
                    is_periodic = False
                    break
            if is_periodic:
                return SymbolicPattern.PERIODIC

        # Compute entropy-based measures
        perm_entropy = self.compute_permutation_entropy(series)
        topo_entropy = self.compute_topological_entropy(sequence)

        if perm_entropy > 0.9 and topo_entropy > math.log(len(set(sequence))) * 0.9:
            return SymbolicPattern.RANDOM
        elif perm_entropy < 0.5:
            return SymbolicPattern.QUASI_PERIODIC
        else:
            return SymbolicPattern.CHAOTIC

    def compute_lempel_ziv_complexity(self, sequence: str) -> float:
        """Compute Lempel-Ziv complexity (normalized)"""
        if len(sequence) < 2:
            return 0.0

        # LZ76 algorithm
        n = len(sequence)
        i = 0
        c = 1  # Complexity counter
        l = 1  # Length of current substring

        while i + l <= n:
            # Check if sequence[i:i+l] appears in sequence[0:i+l-1]
            substr = sequence[i:i + l]
            history = sequence[0:i + l - 1]

            if substr in history:
                l += 1
            else:
                c += 1
                i += l
                l = 1

        # Normalize
        if n > 0:
            return c * math.log2(n) / n
        return 0.0

    def analyze(self, series: List[float], n_symbols: int = 4) -> SymbolicAnalysis:
        """Full symbolic dynamics analysis"""
        analysis = SymbolicAnalysis()

        sequence = self.symbolize(series, n_symbols)
        analysis.symbol_sequence = sequence
        analysis.alphabet_size = len(set(sequence))

        analysis.topological_entropy = self.compute_topological_entropy(sequence)
        analysis.permutation_entropy = self.compute_permutation_entropy(series)
        analysis.forbidden_patterns = self.find_forbidden_patterns(sequence)
        analysis.pattern_type = self.classify_pattern(series, sequence)
        analysis.complexity = self.compute_lempel_ziv_complexity(sequence)

        return analysis


# ============================================================================
# KOOPMAN OPERATOR AND DMD
# ============================================================================

@dataclass
class KoopmanMode:
    """A Koopman mode with eigenvalue and spatial structure"""
    eigenvalue: complex
    mode: List[complex]
    frequency: float
    growth_rate: float
    energy: float


@dataclass
class DMDAnalysis:
    """Dynamic Mode Decomposition results"""
    modes: List[KoopmanMode] = field(default_factory=list)
    reconstruction_error: float = 0.0
    dominant_frequency: float = 0.0
    spectral_complexity: float = 0.0


class KoopmanAnalyzer:
    """Koopman operator spectral analysis via DMD"""

    def create_hankel_matrix(self, series: List[float], n_rows: int = None) -> List[List[float]]:
        """Create Hankel matrix from time series"""
        n = len(series)
        if n_rows is None:
            n_rows = n // 2

        n_cols = n - n_rows + 1
        matrix = []

        for i in range(n_rows):
            row = series[i:i + n_cols]
            matrix.append(row)

        return matrix

    def svd_approximate(self, matrix: List[List[float]], rank: int = None
                       ) -> Tuple[List[List[float]], List[float], List[List[float]]]:
        """Approximate SVD using power iteration"""
        m, n = len(matrix), len(matrix[0])
        if rank is None:
            rank = min(m, n, 5)

        U, S, V = [], [], []

        # Create working copy
        A = [row[:] for row in matrix]

        for _ in range(rank):
            # Power iteration for dominant singular vector
            v = [random.gauss(0, 1) for _ in range(n)]
            norm = math.sqrt(sum(x ** 2 for x in v))
            v = [x / norm for x in v]

            for _ in range(20):
                # u = A @ v
                u = [sum(A[i][j] * v[j] for j in range(n)) for i in range(m)]
                norm_u = math.sqrt(sum(x ** 2 for x in u))
                if norm_u > 0:
                    u = [x / norm_u for x in u]

                # v = A^T @ u
                v = [sum(A[i][j] * u[i] for i in range(m)) for j in range(n)]
                norm_v = math.sqrt(sum(x ** 2 for x in v))
                if norm_v > 0:
                    v = [x / norm_v for x in v]

            # Singular value
            sigma = sum(sum(A[i][j] * u[i] * v[j] for j in range(n)) for i in range(m))

            U.append(u)
            S.append(abs(sigma))
            V.append(v)

            # Deflate
            for i in range(m):
                for j in range(n):
                    A[i][j] -= sigma * u[i] * v[j]

        return U, S, V

    def dmd(self, series: List[float], rank: int = 5) -> DMDAnalysis:
        """Perform Dynamic Mode Decomposition"""
        analysis = DMDAnalysis()

        if len(series) < 10:
            return analysis

        # Create data matrices X and Y (time-shifted)
        n = len(series) - 1
        X = [[series[i]] for i in range(n)]
        Y = [[series[i + 1]] for i in range(n)]

        # For multivariate, use delay embedding
        delay = 3
        if len(series) > delay * 2:
            X = []
            Y = []
            for i in range(len(series) - delay - 1):
                X.append([series[i + j] for j in range(delay)])
                Y.append([series[i + j + 1] for j in range(delay)])

        if len(X) < 5:
            return analysis

        # Compute approximate DMD via least squares
        # A ≈ Y @ X^+

        # Simplified: estimate eigenvalues from autocorrelation
        mean_s = sum(series) / len(series)
        var_s = sum((s - mean_s) ** 2 for s in series) / len(series)

        if var_s < 1e-10:
            return analysis

        # Estimate dominant frequency from zero crossings
        zero_crossings = 0
        for i in range(1, len(series)):
            if (series[i] - mean_s) * (series[i - 1] - mean_s) < 0:
                zero_crossings += 1

        dominant_freq = zero_crossings / (2 * len(series))
        analysis.dominant_frequency = dominant_freq

        # Create synthetic modes based on frequency analysis
        for k in range(min(rank, 3)):
            freq = dominant_freq * (k + 1)
            # Eigenvalue on unit circle
            eigenvalue = complex(math.cos(2 * math.pi * freq),
                               math.sin(2 * math.pi * freq))

            mode = KoopmanMode(
                eigenvalue=eigenvalue,
                mode=[complex(1, 0)],
                frequency=freq,
                growth_rate=abs(eigenvalue) - 1,
                energy=var_s / (k + 1)
            )
            analysis.modes.append(mode)

        # Spectral complexity
        if analysis.modes:
            energies = [m.energy for m in analysis.modes]
            total = sum(energies)
            if total > 0:
                probs = [e / total for e in energies]
                analysis.spectral_complexity = -sum(p * math.log2(p)
                                                   for p in probs if p > 0)

        return analysis


# ============================================================================
# FRACTAL AND MULTIFRACTAL ANALYSIS
# ============================================================================

@dataclass
class FractalAnalysis:
    """Fractal dimension analysis results"""
    box_counting_dim: float = 0.0
    correlation_dim: float = 0.0
    information_dim: float = 0.0
    hurst_exponent: float = 0.5
    dfa_exponent: float = 0.5
    multifractal_spectrum: List[Tuple[float, float]] = field(default_factory=list)
    singularity_spectrum: Dict[str, float] = field(default_factory=dict)


class FractalAnalyzer:
    """Fractal and multifractal analysis"""

    def box_counting_dimension(self, points: List[Tuple[float, ...]],
                              n_scales: int = 10) -> float:
        """Compute box-counting dimension"""
        if len(points) < 2:
            return 0.0

        dim = len(points[0])

        # Find bounding box
        mins = [min(p[d] for p in points) for d in range(dim)]
        maxs = [max(p[d] for p in points) for d in range(dim)]
        ranges = [maxs[d] - mins[d] for d in range(dim)]
        max_range = max(ranges) if ranges else 1

        if max_range == 0:
            return 0.0

        log_eps = []
        log_n = []

        for i in range(1, n_scales + 1):
            eps = max_range / (2 ** i)
            if eps == 0:
                continue

            # Count boxes
            boxes = set()
            for p in points:
                box = tuple(int((p[d] - mins[d]) / eps) for d in range(dim))
                boxes.add(box)

            if len(boxes) > 0:
                log_eps.append(math.log(eps))
                log_n.append(math.log(len(boxes)))

        if len(log_eps) < 3:
            return 0.0

        # Linear regression
        n = len(log_eps)
        sum_x = sum(log_eps)
        sum_y = sum(log_n)
        sum_xy = sum(x * y for x, y in zip(log_eps, log_n))
        sum_x2 = sum(x ** 2 for x in log_eps)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2 + 1e-10)

        return -slope  # Negative because N ~ ε^(-D)

    def hurst_exponent(self, series: List[float]) -> float:
        """Compute Hurst exponent using R/S analysis"""
        if len(series) < 20:
            return 0.5

        n = len(series)
        max_k = int(math.log2(n)) - 1

        log_n = []
        log_rs = []

        for k in range(1, max_k + 1):
            size = n // (2 ** k)
            if size < 4:
                continue

            rs_values = []

            for start in range(0, n - size + 1, size):
                segment = series[start:start + size]
                mean_seg = sum(segment) / len(segment)

                # Cumulative deviation
                cumsum = []
                total = 0
                for s in segment:
                    total += s - mean_seg
                    cumsum.append(total)

                # Range
                R = max(cumsum) - min(cumsum)

                # Standard deviation
                S = math.sqrt(sum((s - mean_seg) ** 2 for s in segment) / len(segment))

                if S > 0:
                    rs_values.append(R / S)

            if rs_values:
                log_n.append(math.log(size))
                log_rs.append(math.log(sum(rs_values) / len(rs_values)))

        if len(log_n) < 3:
            return 0.5

        # Linear regression
        n_pts = len(log_n)
        sum_x = sum(log_n)
        sum_y = sum(log_rs)
        sum_xy = sum(x * y for x, y in zip(log_n, log_rs))
        sum_x2 = sum(x ** 2 for x in log_n)

        slope = (n_pts * sum_xy - sum_x * sum_y) / (n_pts * sum_x2 - sum_x ** 2 + 1e-10)

        return max(0, min(1, slope))

    def dfa(self, series: List[float], order: int = 1) -> float:
        """Detrended Fluctuation Analysis"""
        if len(series) < 16:
            return 0.5

        n = len(series)

        # Integrate series
        mean_s = sum(series) / n
        profile = []
        total = 0
        for s in series:
            total += s - mean_s
            profile.append(total)

        # Compute fluctuation for different scales
        log_s = []
        log_f = []

        for s in range(4, n // 4):
            # Divide into segments
            n_segments = n // s
            if n_segments < 2:
                continue

            f_squared = []

            for seg in range(n_segments):
                start = seg * s
                end = start + s
                segment = profile[start:end]

                # Linear detrend
                x = list(range(s))
                sum_x = sum(x)
                sum_y = sum(segment)
                sum_xy = sum(xi * yi for xi, yi in zip(x, segment))
                sum_x2 = sum(xi ** 2 for xi in x)

                denom = s * sum_x2 - sum_x ** 2
                if abs(denom) > 1e-10:
                    slope = (s * sum_xy - sum_x * sum_y) / denom
                    intercept = (sum_y - slope * sum_x) / s

                    # Compute residual variance
                    residuals = [segment[i] - (slope * i + intercept) for i in range(s)]
                    f_squared.append(sum(r ** 2 for r in residuals) / s)

            if f_squared:
                f = math.sqrt(sum(f_squared) / len(f_squared))
                log_s.append(math.log(s))
                log_f.append(math.log(f + 1e-10))

        if len(log_s) < 3:
            return 0.5

        # Linear regression
        n_pts = len(log_s)
        sum_x = sum(log_s)
        sum_y = sum(log_f)
        sum_xy = sum(x * y for x, y in zip(log_s, log_f))
        sum_x2 = sum(x ** 2 for x in log_s)

        slope = (n_pts * sum_xy - sum_x * sum_y) / (n_pts * sum_x2 - sum_x ** 2 + 1e-10)

        return max(0, min(2, slope))

    def multifractal_spectrum(self, series: List[float],
                             q_range: Tuple[float, float] = (-5, 5),
                             n_q: int = 11) -> List[Tuple[float, float]]:
        """Compute multifractal spectrum f(α)"""
        if len(series) < 32:
            return []

        spectrum = []
        n = len(series)

        # Create partition
        scales = [2 ** k for k in range(2, int(math.log2(n)) - 1)]

        q_values = [q_range[0] + i * (q_range[1] - q_range[0]) / (n_q - 1)
                   for i in range(n_q)]

        for q in q_values:
            tau_estimates = []

            for scale in scales:
                n_boxes = n // scale

                # Compute measure for each box
                measures = []
                for i in range(n_boxes):
                    box = series[i * scale:(i + 1) * scale]
                    mu = sum(abs(x) for x in box) + 1e-10
                    measures.append(mu)

                total = sum(measures)
                probs = [m / total for m in measures]

                # Partition function
                if q == 1:
                    # Special case: use entropy
                    chi = -sum(p * math.log(p + 1e-10) for p in probs)
                else:
                    chi = sum(p ** q for p in probs)

                tau_estimates.append((math.log(scale), math.log(chi + 1e-10)))

            if len(tau_estimates) >= 3:
                # Fit tau(q)
                x = [t[0] for t in tau_estimates]
                y = [t[1] for t in tau_estimates]

                n_pts = len(x)
                sum_x = sum(x)
                sum_y = sum(y)
                sum_xy = sum(xi * yi for xi, yi in zip(x, y))
                sum_x2 = sum(xi ** 2 for xi in x)

                tau = (n_pts * sum_xy - sum_x * sum_y) / (n_pts * sum_x2 - sum_x ** 2 + 1e-10)

                # α = dτ/dq, f(α) = qα - τ
                alpha = (tau - (q - 0.5)) / 0.5 if abs(q) > 0.1 else 1.0
                f_alpha = q * alpha - tau

                spectrum.append((alpha, f_alpha))

        return spectrum

    def analyze(self, series: List[float]) -> FractalAnalysis:
        """Full fractal analysis"""
        analysis = FractalAnalysis()

        # Create point cloud from delay embedding
        delay = 2
        points = [(series[i], series[i + delay])
                 for i in range(len(series) - delay)]

        analysis.box_counting_dim = self.box_counting_dimension(
            [(p[0], p[1]) for p in points]
        )
        analysis.hurst_exponent = self.hurst_exponent(series)
        analysis.dfa_exponent = self.dfa(series)
        analysis.multifractal_spectrum = self.multifractal_spectrum(series)

        # Singularity spectrum summary
        if analysis.multifractal_spectrum:
            alphas = [a for a, f in analysis.multifractal_spectrum]
            f_alphas = [f for a, f in analysis.multifractal_spectrum]
            analysis.singularity_spectrum = {
                'alpha_min': min(alphas),
                'alpha_max': max(alphas),
                'alpha_0': alphas[len(alphas) // 2] if alphas else 0,
                'width': max(alphas) - min(alphas) if alphas else 0
            }

        return analysis


# ============================================================================
# OPTIMAL TRANSPORT
# ============================================================================

@dataclass
class OptimalTransportAnalysis:
    """Optimal transport analysis results"""
    wasserstein_distances: Dict[Tuple[str, str], float] = field(default_factory=dict)
    transport_plan: List[List[float]] = field(default_factory=list)
    earth_mover_distance: float = 0.0


class OptimalTransportAnalyzer:
    """Optimal transport and Wasserstein distance computation"""

    def wasserstein_1d(self, dist1: List[float], dist2: List[float]) -> float:
        """Compute 1-Wasserstein distance between 1D distributions"""
        if not dist1 or not dist2:
            return 0.0

        # Sort both distributions
        s1 = sorted(dist1)
        s2 = sorted(dist2)

        # Compute quantile function difference
        n1, n2 = len(s1), len(s2)
        n = max(n1, n2)

        # Interpolate to same grid
        def quantile(sorted_list, p):
            idx = p * (len(sorted_list) - 1)
            lower = int(idx)
            upper = min(lower + 1, len(sorted_list) - 1)
            frac = idx - lower
            return sorted_list[lower] * (1 - frac) + sorted_list[upper] * frac

        total = 0.0
        for i in range(n):
            p = i / (n - 1) if n > 1 else 0
            total += abs(quantile(s1, p) - quantile(s2, p))

        return total / n

    def sinkhorn_distance(self, cost_matrix: List[List[float]],
                         mu: List[float], nu: List[float],
                         reg: float = 0.1, max_iter: int = 100) -> float:
        """Compute Sinkhorn distance (regularized optimal transport)"""
        m, n = len(mu), len(nu)
        if m == 0 or n == 0:
            return 0.0

        # Initialize
        K = [[math.exp(-cost_matrix[i][j] / reg) for j in range(n)] for i in range(m)]
        u = [1.0] * m
        v = [1.0] * n

        for _ in range(max_iter):
            # Update u
            for i in range(m):
                Kv = sum(K[i][j] * v[j] for j in range(n))
                u[i] = mu[i] / (Kv + 1e-10)

            # Update v
            for j in range(n):
                Ku = sum(K[i][j] * u[i] for i in range(m))
                v[j] = nu[j] / (Ku + 1e-10)

        # Compute transport cost
        transport_cost = 0.0
        for i in range(m):
            for j in range(n):
                transport_cost += u[i] * K[i][j] * v[j] * cost_matrix[i][j]

        return transport_cost

    def earth_mover_distance(self, sig1: List[Tuple[float, float]],
                            sig2: List[Tuple[float, float]]) -> float:
        """Compute Earth Mover's Distance between signatures"""
        # sig = [(weight, location), ...]
        if not sig1 or not sig2:
            return 0.0

        # Normalize weights
        total1 = sum(w for w, _ in sig1)
        total2 = sum(w for w, _ in sig2)

        if total1 == 0 or total2 == 0:
            return 0.0

        sig1 = [(w / total1, loc) for w, loc in sig1]
        sig2 = [(w / total2, loc) for w, loc in sig2]

        # Compute pairwise distances
        m, n = len(sig1), len(sig2)
        cost = [[abs(sig1[i][1] - sig2[j][1]) for j in range(n)] for i in range(m)]

        # Use Sinkhorn as approximation
        mu = [w for w, _ in sig1]
        nu = [w for w, _ in sig2]

        return self.sinkhorn_distance(cost, mu, nu)


# ============================================================================
# RENYI ENTROPY SPECTRUM
# ============================================================================

class RenyiAnalyzer:
    """Renyi entropy spectrum analysis"""

    def renyi_entropy(self, probs: List[float], alpha: float) -> float:
        """Compute Renyi entropy of order α"""
        if not probs:
            return 0.0

        probs = [p for p in probs if p > 0]
        if not probs:
            return 0.0

        if alpha == 1:
            # Shannon entropy (limit as α → 1)
            return -sum(p * math.log2(p) for p in probs)
        elif alpha == 0:
            # Hartley entropy
            return math.log2(len(probs))
        elif alpha == float('inf'):
            # Min-entropy
            return -math.log2(max(probs))
        else:
            return math.log2(sum(p ** alpha for p in probs)) / (1 - alpha)

    def entropy_spectrum(self, series: List[float],
                        alpha_range: Tuple[float, float] = (0, 5),
                        n_alpha: int = 11) -> List[Tuple[float, float]]:
        """Compute spectrum of Renyi entropies"""
        if len(series) < 10:
            return []

        # Discretize series
        n_bins = min(20, len(series) // 5)
        min_val, max_val = min(series), max(series)

        if max_val == min_val:
            return [(a, 0) for a in [alpha_range[0] + i * (alpha_range[1] - alpha_range[0]) / (n_alpha - 1)
                                    for i in range(n_alpha)]]

        hist = [0] * n_bins
        for s in series:
            idx = min(n_bins - 1, int((s - min_val) / (max_val - min_val + 1e-10) * n_bins))
            hist[idx] += 1

        probs = [h / len(series) for h in hist if h > 0]

        spectrum = []
        for i in range(n_alpha):
            alpha = alpha_range[0] + i * (alpha_range[1] - alpha_range[0]) / (n_alpha - 1)
            h = self.renyi_entropy(probs, alpha)
            spectrum.append((alpha, h))

        return spectrum


# ============================================================================
# FISHER INFORMATION
# ============================================================================

class FisherInformationAnalyzer:
    """Fisher information geometry analysis"""

    def fisher_information(self, series: List[float], param_idx: int = 0) -> float:
        """Estimate Fisher information from time series"""
        if len(series) < 5:
            return 0.0

        # Estimate as expected squared score
        # For Gaussian assumption: I = 1/σ²

        mean_s = sum(series) / len(series)
        var_s = sum((s - mean_s) ** 2 for s in series) / len(series)

        if var_s < 1e-10:
            return 0.0

        return 1.0 / var_s

    def fisher_matrix(self, distributions: Dict[str, List[float]]) -> List[List[float]]:
        """Compute Fisher information matrix between distributions"""
        agents = list(distributions.keys())
        n = len(agents)

        matrix = [[0.0] * n for _ in range(n)]

        for i, a1 in enumerate(agents):
            for j, a2 in enumerate(agents):
                if i == j:
                    matrix[i][j] = self.fisher_information(distributions[a1])
                else:
                    # Cross Fisher information (simplified)
                    s1, s2 = distributions[a1], distributions[a2]
                    if s1 and s2:
                        # Correlation-based estimate
                        n_pairs = min(len(s1), len(s2))
                        mean1 = sum(s1[:n_pairs]) / n_pairs
                        mean2 = sum(s2[:n_pairs]) / n_pairs
                        cov = sum((s1[k] - mean1) * (s2[k] - mean2)
                                  for k in range(n_pairs)) / n_pairs
                        matrix[i][j] = abs(cov) / (
                            math.sqrt(sum((s - mean1) ** 2 for s in s1[:n_pairs]) / n_pairs + 1e-10) *
                            math.sqrt(sum((s - mean2) ** 2 for s in s2[:n_pairs]) / n_pairs + 1e-10)
                        )

        return matrix


# ============================================================================
# OMEGA METRICS COLLECTION
# ============================================================================

@dataclass
class OmegaSimulationMetrics:
    """Comprehensive omega-level metrics collection"""

    # Basic tracking
    rounds: List[int] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)

    # High-dimensional state vectors
    state_vectors: Dict[str, List[List[float]]] = field(default_factory=lambda: defaultdict(list))

    # Phase coordinates
    phase_coordinates: Dict[str, List[Tuple[float, float, float]]] = field(default_factory=lambda: defaultdict(list))

    # Belief states
    belief_states: Dict[str, List[Dict[str, float]]] = field(default_factory=lambda: defaultdict(list))

    # Actions
    action_sequences: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))

    # Global signals
    tension_signal: List[float] = field(default_factory=list)
    entropy_signal: List[float] = field(default_factory=list)
    complexity_signal: List[float] = field(default_factory=list)
    phi_signal: List[float] = field(default_factory=list)
    criticality_signal: List[float] = field(default_factory=list)

    # Interactions
    interactions: List[Dict[Tuple[str, str], Dict[str, float]]] = field(default_factory=list)

    # Topological features
    persistence_diagrams: List[PersistenceDiagram] = field(default_factory=list)

    # Quantum measures
    entanglement_history: List[float] = field(default_factory=list)
    von_neumann_entropy: List[float] = field(default_factory=list)

    def record_round(self, round_num: int):
        self.rounds.append(round_num)
        self.timestamps.append(datetime.now())
        self.interactions.append({})

    def record_agent_state(self, agent_id: str, state_vector: List[float],
                          phase_coords: Tuple[float, float, float],
                          beliefs: Dict[str, float], action: str):
        self.state_vectors[agent_id].append(state_vector)
        self.phase_coordinates[agent_id].append(phase_coords)
        self.belief_states[agent_id].append(beliefs)
        self.action_sequences[agent_id].append(action)

    def record_interaction(self, agent1: str, agent2: str,
                          interaction_type: str, strength: float):
        if self.interactions:
            self.interactions[-1][(agent1, agent2)] = {
                "type": interaction_type,
                "strength": strength
            }

    def record_global_state(self, tension: float, entropy: float,
                           complexity: float, phi: float, criticality: float = 0.0):
        self.tension_signal.append(tension)
        self.entropy_signal.append(entropy)
        self.complexity_signal.append(complexity)
        self.phi_signal.append(phi)
        self.criticality_signal.append(criticality)


# ============================================================================
# OMEGA VISUALIZATION
# ============================================================================

class OmegaSimulationVisualizer:
    """Omega-level visualization with cutting-edge analytics"""

    def __init__(self, metrics: OmegaSimulationMetrics):
        self.metrics = metrics

        # Initialize analyzers
        self.tda = RipsComplex([], max_dim=2)
        self.quantum = QuantumAnalyzer()
        self.criticality = CriticalityDetector()
        self.symbolic = SymbolicDynamicsAnalyzer()
        self.koopman = KoopmanAnalyzer()
        self.fractal = FractalAnalyzer()
        self.transport = OptimalTransportAnalyzer()
        self.renyi = RenyiAnalyzer()
        self.fisher = FisherInformationAnalyzer()

    def create_omega_dashboard(self, save_path: str = None,
                               figsize: Tuple[int, int] = (28, 24)):
        """Create omega-level 20-panel dashboard"""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available")
            return None

        fig = plt.figure(figsize=figsize, facecolor='#050510')
        fig.suptitle('Ω OMEGA-LEVEL SIMULATION ANALYTICS Ω',
                    fontsize=20, color='#00ffff', fontweight='bold', y=0.98,
                    fontfamily='monospace')

        gs = GridSpec(5, 4, figure=fig, hspace=0.35, wspace=0.3,
                     left=0.04, right=0.96, top=0.93, bottom=0.04)

        # Row 1: Topology and Quantum
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_persistence_diagram(ax1)

        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_betti_evolution(ax2)

        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_quantum_entanglement(ax3)

        ax4 = fig.add_subplot(gs[0, 3])
        self._plot_von_neumann_entropy(ax4)

        # Row 2: Criticality and Phase
        ax5 = fig.add_subplot(gs[1, 0])
        self._plot_criticality_analysis(ax5)

        ax6 = fig.add_subplot(gs[1, 1])
        self._plot_avalanche_distribution(ax6)

        ax7 = fig.add_subplot(gs[1, 2])
        self._plot_branching_ratio(ax7)

        ax8 = fig.add_subplot(gs[1, 3])
        self._plot_order_parameter(ax8)

        # Row 3: Symbolic Dynamics
        ax9 = fig.add_subplot(gs[2, 0])
        self._plot_symbolic_sequence(ax9)

        ax10 = fig.add_subplot(gs[2, 1])
        self._plot_permutation_entropy(ax10)

        ax11 = fig.add_subplot(gs[2, 2])
        self._plot_koopman_spectrum(ax11)

        ax12 = fig.add_subplot(gs[2, 3])
        self._plot_dmd_modes(ax12)

        # Row 4: Fractal Analysis
        ax13 = fig.add_subplot(gs[3, 0])
        self._plot_multifractal_spectrum(ax13)

        ax14 = fig.add_subplot(gs[3, 1])
        self._plot_dfa_scaling(ax14)

        ax15 = fig.add_subplot(gs[3, 2])
        self._plot_renyi_spectrum(ax15)

        ax16 = fig.add_subplot(gs[3, 3])
        self._plot_hurst_analysis(ax16)

        # Row 5: Transport and Information
        ax17 = fig.add_subplot(gs[4, 0])
        self._plot_wasserstein_matrix(ax17)

        ax18 = fig.add_subplot(gs[4, 1])
        self._plot_fisher_information(ax18)

        ax19 = fig.add_subplot(gs[4, 2])
        self._plot_information_geometry(ax19)

        ax20 = fig.add_subplot(gs[4, 3])
        self._plot_omega_summary(ax20)

        if save_path:
            plt.savefig(save_path, dpi=150, facecolor='#050510',
                       edgecolor='none', bbox_inches='tight')
            print(f"Omega dashboard saved to {save_path}")

        plt.close()
        return fig

    def _style_axis(self, ax, title: str):
        """Apply omega styling"""
        ax.set_facecolor('#0a0a1a')
        ax.set_title(title, color='#00ffff', fontsize=9, fontweight='bold',
                    pad=8, fontfamily='monospace')

        for spine in ax.spines.values():
            spine.set_color('#1a1a3a')

        ax.tick_params(colors='#666699', labelsize=7)
        ax.xaxis.label.set_color('#8888aa')
        ax.yaxis.label.set_color('#8888aa')

    def _plot_persistence_diagram(self, ax):
        """Plot persistence diagram from TDA"""
        self._style_axis(ax, 'Persistence Diagram (TDA)')

        # Create point cloud from phase coordinates
        all_points = []
        for coords in self.metrics.phase_coordinates.values():
            for c in coords[-20:]:  # Last 20 points
                all_points.append(c)

        if len(all_points) >= 5:
            rips = RipsComplex(all_points, max_dim=2)
            diagram = rips.compute_persistence(max_epsilon=2.0)

            # Plot intervals
            for interval in diagram.intervals:
                if interval.death != float('inf'):
                    ax.scatter(interval.birth, interval.death,
                              c='#ff00ff' if interval.dimension == 0 else '#00ffff',
                              s=50, alpha=0.7)

            # Diagonal
            max_val = max([i.death for i in diagram.intervals
                         if i.death != float('inf')] + [1])
            ax.plot([0, max_val], [0, max_val], '--', color='#333366', alpha=0.5)

            ax.set_xlabel('Birth', fontsize=8)
            ax.set_ylabel('Death', fontsize=8)

            # Annotation
            ax.text(0.95, 0.05, f'β₀={diagram.get_betti(0)}\nH={diagram.persistent_entropy:.2f}',
                   transform=ax.transAxes, ha='right', va='bottom',
                   fontsize=8, color='#00ffff',
                   bbox=dict(boxstyle='round', facecolor='#1a1a3a', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   color='#666', fontsize=10, transform=ax.transAxes)

    def _plot_betti_evolution(self, ax):
        """Plot Betti number evolution"""
        self._style_axis(ax, 'Betti Numbers Evolution')

        if len(self.metrics.rounds) < 5:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   color='#666', fontsize=10, transform=ax.transAxes)
            return

        # Compute Betti numbers at each time point
        betti_0 = []
        betti_1 = []

        window = 10
        for t in range(0, len(self.metrics.rounds) - window, window // 2):
            points = []
            for coords in self.metrics.phase_coordinates.values():
                for c in coords[t:t + window]:
                    points.append(c)

            if len(points) >= 3:
                rips = RipsComplex(points, max_dim=1)
                diagram = rips.compute_persistence(max_epsilon=1.5)
                betti_0.append(diagram.get_betti(0))
                betti_1.append(len([i for i in diagram.intervals if i.dimension == 1]))

        if betti_0:
            ax.plot(betti_0, color='#ff00ff', linewidth=2, label='β₀')
            ax.plot(betti_1, color='#00ffff', linewidth=2, label='β₁')
            ax.legend(loc='upper right', fontsize=7, facecolor='#1a1a3a',
                     edgecolor='#333', labelcolor='white')

        ax.set_xlabel('Window', fontsize=8)
        ax.set_ylabel('Betti Number', fontsize=8)

    def _plot_quantum_entanglement(self, ax):
        """Plot quantum entanglement measures"""
        self._style_axis(ax, 'Quantum Entanglement')

        agents = list(self.metrics.state_vectors.keys())

        if len(agents) >= 2:
            # Compute entanglement over time
            entanglement = []

            for t in range(min(len(self.metrics.state_vectors[a]) for a in agents)):
                # Get states
                states = [self.metrics.state_vectors[a][t] for a in agents]

                # Compute correlation as entanglement proxy
                if len(states[0]) > 0:
                    s1, s2 = states[0][0], states[1][0] if len(states) > 1 else 0
                    corr = abs(s1 * s2 - (s1 + s2) / 2)
                    entanglement.append(corr)

            if entanglement:
                ax.plot(entanglement, color='#ff00ff', linewidth=2)
                ax.fill_between(range(len(entanglement)), entanglement,
                               alpha=0.3, color='#ff00ff')

                ax.set_xlabel('Time', fontsize=8)
                ax.set_ylabel('Entanglement', fontsize=8)

                # Annotation
                ax.text(0.95, 0.95, f'⟨E⟩={sum(entanglement)/len(entanglement):.3f}',
                       transform=ax.transAxes, ha='right', va='top',
                       fontsize=9, color='#ff00ff')
        else:
            ax.text(0.5, 0.5, 'Need 2+ agents', ha='center', va='center',
                   color='#666', fontsize=10, transform=ax.transAxes)

    def _plot_von_neumann_entropy(self, ax):
        """Plot von Neumann entropy evolution"""
        self._style_axis(ax, 'von Neumann Entropy')

        if self.metrics.entropy_signal:
            # Convert to quantum and compute vN entropy
            vn_entropies = []

            for t, e in enumerate(self.metrics.entropy_signal):
                # Create probability distribution from entropy
                p = max(0.01, min(0.99, e))
                probs = [p, 1 - p]

                state = self.quantum.classical_to_quantum(probs)
                vn = self.quantum.von_neumann_entropy(state)
                vn_entropies.append(vn)

            ax.plot(vn_entropies, color='#00ffff', linewidth=2)
            ax.fill_between(range(len(vn_entropies)), vn_entropies,
                           alpha=0.3, color='#00ffff')

            ax.set_xlabel('Time', fontsize=8)
            ax.set_ylabel('S(ρ)', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   color='#666', fontsize=10, transform=ax.transAxes)

    def _plot_criticality_analysis(self, ax):
        """Plot criticality regime analysis"""
        self._style_axis(ax, 'Criticality Analysis')

        if self.metrics.tension_signal:
            analysis = self.criticality.analyze(
                self.metrics.tension_signal,
                self.metrics.complexity_signal or self.metrics.tension_signal
            )

            # Show regime and metrics
            regime_colors = {
                CriticalityRegime.SUBCRITICAL: '#0066ff',
                CriticalityRegime.CRITICAL: '#ff00ff',
                CriticalityRegime.SUPERCRITICAL: '#ff0066',
                CriticalityRegime.EDGE_OF_CHAOS: '#ffff00'
            }

            color = regime_colors.get(analysis.regime, '#888')

            metrics = {
                'σ (branching)': analysis.branching_ratio,
                'χ (suscept.)': min(10, analysis.susceptibility),
                'm (order)': analysis.order_parameter,
                'τ (exponent)': analysis.power_law_exponent
            }

            bars = ax.barh(list(metrics.keys()), list(metrics.values()),
                          color=color, alpha=0.8)

            ax.axvline(1.0, color='#ffff00', linestyle='--', alpha=0.5, linewidth=2)

            ax.set_xlabel('Value', fontsize=8)

            # Regime annotation
            ax.text(0.95, 0.95, analysis.regime.value.upper(),
                   transform=ax.transAxes, ha='right', va='top',
                   fontsize=12, color=color, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   color='#666', fontsize=10, transform=ax.transAxes)

    def _plot_avalanche_distribution(self, ax):
        """Plot avalanche size distribution"""
        self._style_axis(ax, 'Avalanche Distribution')

        if self.metrics.complexity_signal:
            avalanches = self.criticality.detect_avalanches(self.metrics.complexity_signal)

            if avalanches:
                # Histogram
                freq = Counter(avalanches)
                sizes = sorted(freq.keys())
                counts = [freq[s] for s in sizes]

                ax.bar(sizes, counts, color='#ff00ff', alpha=0.8)

                # Power law fit line
                exp, r_sq = self.criticality.fit_power_law(avalanches)
                if exp > 0 and sizes:
                    x_fit = range(1, max(sizes) + 1)
                    y_fit = [counts[0] * (x ** -exp) for x in x_fit]
                    ax.plot(x_fit, y_fit, '--', color='#00ffff', linewidth=2,
                           label=f'τ={exp:.2f}')

                ax.set_xlabel('Size', fontsize=8)
                ax.set_ylabel('Count', fontsize=8)
                ax.set_yscale('log')
                ax.legend(loc='upper right', fontsize=7, facecolor='#1a1a3a',
                         edgecolor='#333', labelcolor='white')
            else:
                ax.text(0.5, 0.5, 'No avalanches', ha='center', va='center',
                       color='#666', fontsize=10, transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   color='#666', fontsize=10, transform=ax.transAxes)

    def _plot_branching_ratio(self, ax):
        """Plot branching ratio evolution"""
        self._style_axis(ax, 'Branching Ratio σ(t)')

        if len(self.metrics.complexity_signal) > 10:
            # Sliding window branching ratio
            window = 10
            sigmas = []

            for i in range(len(self.metrics.complexity_signal) - window):
                segment = self.metrics.complexity_signal[i:i + window]
                sigma = self.criticality.compute_branching_ratio(segment)
                sigmas.append(sigma)

            ax.plot(sigmas, color='#00ffff', linewidth=2)
            ax.axhline(1.0, color='#ffff00', linestyle='--', linewidth=2,
                      label='Critical (σ=1)')
            ax.fill_between(range(len(sigmas)), [0.9] * len(sigmas),
                           [1.1] * len(sigmas), alpha=0.2, color='#ffff00')

            ax.set_xlabel('Time', fontsize=8)
            ax.set_ylabel('σ', fontsize=8)
            ax.legend(loc='upper right', fontsize=7, facecolor='#1a1a3a',
                     edgecolor='#333', labelcolor='white')
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   color='#666', fontsize=10, transform=ax.transAxes)

    def _plot_order_parameter(self, ax):
        """Plot order parameter evolution"""
        self._style_axis(ax, 'Order Parameter m(t)')

        if self.metrics.tension_signal:
            # Sliding window order parameter
            window = 5
            order_params = []

            for i in range(len(self.metrics.tension_signal) - window):
                segment = self.metrics.tension_signal[i:i + window]
                m = self.criticality.compute_order_parameter(segment)
                order_params.append(m)

            ax.plot(order_params, color='#ff00ff', linewidth=2)
            ax.fill_between(range(len(order_params)), order_params,
                           alpha=0.3, color='#ff00ff')

            ax.set_xlabel('Time', fontsize=8)
            ax.set_ylabel('|m|', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   color='#666', fontsize=10, transform=ax.transAxes)

    def _plot_symbolic_sequence(self, ax):
        """Plot symbolic dynamics visualization"""
        self._style_axis(ax, 'Symbolic Dynamics')

        if self.metrics.tension_signal:
            analysis = self.symbolic.analyze(self.metrics.tension_signal)

            # Show sequence as colored blocks
            seq = analysis.symbol_sequence[:100]  # First 100 symbols
            colors = {'A': '#ff0066', 'B': '#ff00ff', 'C': '#00ffff', 'D': '#00ff66'}

            for i, s in enumerate(seq):
                ax.add_patch(plt.Rectangle((i, 0), 1, 1,
                                          facecolor=colors.get(s, '#666'),
                                          edgecolor='none'))

            ax.set_xlim(0, len(seq))
            ax.set_ylim(0, 1)
            ax.set_xlabel('Position', fontsize=8)
            ax.set_yticks([])

            # Annotation
            ax.text(0.5, 1.15, f'{analysis.pattern_type.value.upper()} | '
                              f'H_topo={analysis.topological_entropy:.2f} | '
                              f'LZ={analysis.complexity:.2f}',
                   transform=ax.transAxes, ha='center', va='bottom',
                   fontsize=8, color='#00ffff')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   color='#666', fontsize=10, transform=ax.transAxes)

    def _plot_permutation_entropy(self, ax):
        """Plot permutation entropy analysis"""
        self._style_axis(ax, 'Permutation Entropy')

        if self.metrics.tension_signal:
            # Compute PE for different orders
            orders = range(2, 7)
            pe_values = []

            for order in orders:
                pe = self.symbolic.compute_permutation_entropy(
                    self.metrics.tension_signal, order=order
                )
                pe_values.append(pe)

            ax.plot(list(orders), pe_values, 'o-', color='#00ffff',
                   linewidth=2, markersize=8)

            ax.axhline(1.0, color='#ff00ff', linestyle='--', alpha=0.5,
                      label='Random')
            ax.axhline(0.0, color='#0066ff', linestyle='--', alpha=0.5,
                      label='Deterministic')

            ax.set_xlabel('Order n', fontsize=8)
            ax.set_ylabel('H_perm / H_max', fontsize=8)
            ax.legend(loc='lower right', fontsize=7, facecolor='#1a1a3a',
                     edgecolor='#333', labelcolor='white')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   color='#666', fontsize=10, transform=ax.transAxes)

    def _plot_koopman_spectrum(self, ax):
        """Plot Koopman operator spectrum"""
        self._style_axis(ax, 'Koopman Spectrum')

        if self.metrics.tension_signal:
            dmd = self.koopman.dmd(self.metrics.tension_signal)

            if dmd.modes:
                # Plot eigenvalues on unit circle
                theta = [i * 2 * math.pi / 100 for i in range(101)]
                ax.plot([math.cos(t) for t in theta],
                       [math.sin(t) for t in theta],
                       '--', color='#333366', alpha=0.5)

                for mode in dmd.modes:
                    ax.scatter(mode.eigenvalue.real, mode.eigenvalue.imag,
                              s=mode.energy * 500 + 50, c='#ff00ff', alpha=0.7)

                ax.set_xlim(-1.5, 1.5)
                ax.set_ylim(-1.5, 1.5)
                ax.set_aspect('equal')
                ax.set_xlabel('Re(λ)', fontsize=8)
                ax.set_ylabel('Im(λ)', fontsize=8)

                # Annotation
                ax.text(0.95, 0.05, f'f_dom={dmd.dominant_frequency:.3f}',
                       transform=ax.transAxes, ha='right', va='bottom',
                       fontsize=8, color='#00ffff')
            else:
                ax.text(0.5, 0.5, 'No modes', ha='center', va='center',
                       color='#666', fontsize=10, transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   color='#666', fontsize=10, transform=ax.transAxes)

    def _plot_dmd_modes(self, ax):
        """Plot DMD mode energies"""
        self._style_axis(ax, 'DMD Mode Energy')

        if self.metrics.tension_signal:
            dmd = self.koopman.dmd(self.metrics.tension_signal)

            if dmd.modes:
                energies = [m.energy for m in dmd.modes]
                freqs = [m.frequency for m in dmd.modes]

                ax.bar(range(len(energies)), energies, color='#ff00ff', alpha=0.8)

                for i, (e, f) in enumerate(zip(energies, freqs)):
                    ax.text(i, e + 0.01, f'f={f:.2f}', ha='center', va='bottom',
                           fontsize=7, color='#00ffff')

                ax.set_xlabel('Mode', fontsize=8)
                ax.set_ylabel('Energy', fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No modes', ha='center', va='center',
                       color='#666', fontsize=10, transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   color='#666', fontsize=10, transform=ax.transAxes)

    def _plot_multifractal_spectrum(self, ax):
        """Plot multifractal spectrum f(α)"""
        self._style_axis(ax, 'Multifractal Spectrum f(α)')

        if self.metrics.tension_signal:
            analysis = self.fractal.analyze(self.metrics.tension_signal)

            if analysis.multifractal_spectrum:
                alphas = [a for a, f in analysis.multifractal_spectrum]
                f_alphas = [f for a, f in analysis.multifractal_spectrum]

                ax.plot(alphas, f_alphas, 'o-', color='#ff00ff',
                       linewidth=2, markersize=6)
                ax.fill_between(alphas, f_alphas, alpha=0.3, color='#ff00ff')

                ax.set_xlabel('α (singularity)', fontsize=8)
                ax.set_ylabel('f(α)', fontsize=8)

                # Width annotation
                if analysis.singularity_spectrum:
                    width = analysis.singularity_spectrum.get('width', 0)
                    ax.text(0.95, 0.95, f'Δα={width:.2f}',
                           transform=ax.transAxes, ha='right', va='top',
                           fontsize=9, color='#00ffff')
            else:
                ax.text(0.5, 0.5, 'Cannot compute', ha='center', va='center',
                       color='#666', fontsize=10, transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   color='#666', fontsize=10, transform=ax.transAxes)

    def _plot_dfa_scaling(self, ax):
        """Plot DFA scaling analysis"""
        self._style_axis(ax, 'DFA Scaling')

        if self.metrics.tension_signal and len(self.metrics.tension_signal) > 16:
            # Compute DFA for different scales
            n = len(self.metrics.tension_signal)
            scales = [2 ** k for k in range(2, int(math.log2(n)) - 1)]

            f_values = []
            mean_s = sum(self.metrics.tension_signal) / n

            # Profile
            profile = []
            total = 0
            for s in self.metrics.tension_signal:
                total += s - mean_s
                profile.append(total)

            for s in scales:
                n_seg = n // s
                if n_seg < 2:
                    continue

                f_sq = []
                for seg in range(n_seg):
                    segment = profile[seg * s:(seg + 1) * s]

                    # Linear detrend
                    x = list(range(s))
                    sum_x = sum(x)
                    sum_y = sum(segment)
                    sum_xy = sum(xi * yi for xi, yi in zip(x, segment))
                    sum_x2 = sum(xi ** 2 for xi in x)

                    denom = s * sum_x2 - sum_x ** 2
                    if abs(denom) > 1e-10:
                        slope = (s * sum_xy - sum_x * sum_y) / denom
                        intercept = (sum_y - slope * sum_x) / s
                        residuals = [segment[i] - (slope * i + intercept) for i in range(s)]
                        f_sq.append(sum(r ** 2 for r in residuals) / s)

                if f_sq:
                    f_values.append((s, math.sqrt(sum(f_sq) / len(f_sq))))

            if f_values:
                log_s = [math.log10(s) for s, f in f_values]
                log_f = [math.log10(f + 1e-10) for s, f in f_values]

                ax.plot(log_s, log_f, 'o-', color='#00ffff', linewidth=2, markersize=6)

                # Fit line
                if len(log_s) >= 3:
                    n_pts = len(log_s)
                    sum_x = sum(log_s)
                    sum_y = sum(log_f)
                    sum_xy = sum(x * y for x, y in zip(log_s, log_f))
                    sum_x2 = sum(x ** 2 for x in log_s)

                    alpha = (n_pts * sum_xy - sum_x * sum_y) / (n_pts * sum_x2 - sum_x ** 2 + 1e-10)

                    x_fit = [min(log_s), max(log_s)]
                    intercept = (sum_y - alpha * sum_x) / n_pts
                    y_fit = [alpha * x + intercept for x in x_fit]

                    ax.plot(x_fit, y_fit, '--', color='#ff00ff', linewidth=2,
                           label=f'α={alpha:.2f}')

                    ax.legend(loc='lower right', fontsize=7, facecolor='#1a1a3a',
                             edgecolor='#333', labelcolor='white')

                ax.set_xlabel('log₁₀(s)', fontsize=8)
                ax.set_ylabel('log₁₀(F)', fontsize=8)
            else:
                ax.text(0.5, 0.5, 'Cannot compute', ha='center', va='center',
                       color='#666', fontsize=10, transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   color='#666', fontsize=10, transform=ax.transAxes)

    def _plot_renyi_spectrum(self, ax):
        """Plot Renyi entropy spectrum"""
        self._style_axis(ax, 'Rényi Entropy Spectrum')

        if self.metrics.tension_signal:
            spectrum = self.renyi.entropy_spectrum(self.metrics.tension_signal)

            if spectrum:
                alphas = [a for a, h in spectrum]
                entropies = [h for a, h in spectrum]

                ax.plot(alphas, entropies, 'o-', color='#ff00ff',
                       linewidth=2, markersize=6)

                # Mark special cases
                ax.axvline(0, color='#666', linestyle=':', alpha=0.5, label='H₀ (Hartley)')
                ax.axvline(1, color='#666', linestyle='--', alpha=0.5, label='H₁ (Shannon)')
                ax.axvline(2, color='#666', linestyle='-.', alpha=0.5, label='H₂ (collision)')

                ax.set_xlabel('α', fontsize=8)
                ax.set_ylabel('H_α', fontsize=8)
                ax.legend(loc='upper right', fontsize=6, facecolor='#1a1a3a',
                         edgecolor='#333', labelcolor='white')
            else:
                ax.text(0.5, 0.5, 'Cannot compute', ha='center', va='center',
                       color='#666', fontsize=10, transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   color='#666', fontsize=10, transform=ax.transAxes)

    def _plot_hurst_analysis(self, ax):
        """Plot Hurst exponent analysis"""
        self._style_axis(ax, 'Hurst Exponent')

        agents = list(self.metrics.state_vectors.keys())

        if agents:
            hurst_values = {}

            for agent in agents:
                states = self.metrics.state_vectors[agent]
                if len(states) > 20:
                    series = [s[0] if s else 0 for s in states]
                    h = self.fractal.hurst_exponent(series)
                    hurst_values[agent] = h

            if hurst_values:
                names = list(hurst_values.keys())
                values = list(hurst_values.values())

                colors = ['#ff0066' if h < 0.5 else '#00ff66' if h > 0.5 else '#ffff00'
                         for h in values]

                bars = ax.barh(names, values, color=colors, alpha=0.8)

                ax.axvline(0.5, color='#ffff00', linestyle='--', linewidth=2)

                ax.set_xlabel('H', fontsize=8)
                ax.set_xlim(0, 1)

                # Legend
                ax.text(0.02, 0.95, 'H<0.5: Anti-persistent\nH=0.5: Random\nH>0.5: Persistent',
                       transform=ax.transAxes, ha='left', va='top',
                       fontsize=7, color='#888')
            else:
                ax.text(0.5, 0.5, 'Cannot compute', ha='center', va='center',
                       color='#666', fontsize=10, transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No agents', ha='center', va='center',
                   color='#666', fontsize=10, transform=ax.transAxes)

    def _plot_wasserstein_matrix(self, ax):
        """Plot Wasserstein distance matrix"""
        self._style_axis(ax, 'Wasserstein Distances')

        agents = list(self.metrics.state_vectors.keys())
        n = len(agents)

        if n >= 2:
            # Compute pairwise Wasserstein distances
            matrix = [[0.0] * n for _ in range(n)]

            for i, a1 in enumerate(agents):
                for j, a2 in enumerate(agents):
                    if i < j:
                        s1 = [s[0] if s else 0 for s in self.metrics.state_vectors[a1]]
                        s2 = [s[0] if s else 0 for s in self.metrics.state_vectors[a2]]
                        w = self.transport.wasserstein_1d(s1, s2)
                        matrix[i][j] = w
                        matrix[j][i] = w

            im = ax.imshow(matrix, cmap='magma', aspect='auto')

            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels([a[:3] for a in agents], fontsize=7, color='white')
            ax.set_yticklabels([a[:3] for a in agents], fontsize=7, color='white')

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(colors='white', labelsize=6)
        else:
            ax.text(0.5, 0.5, 'Need 2+ agents', ha='center', va='center',
                   color='#666', fontsize=10, transform=ax.transAxes)

    def _plot_fisher_information(self, ax):
        """Plot Fisher information matrix"""
        self._style_axis(ax, 'Fisher Information')

        agents = list(self.metrics.state_vectors.keys())

        if agents:
            distributions = {a: [s[0] if s else 0 for s in self.metrics.state_vectors[a]]
                            for a in agents}

            matrix = self.fisher.fisher_matrix(distributions)

            if matrix:
                im = ax.imshow(matrix, cmap='viridis', aspect='auto')

                n = len(agents)
                ax.set_xticks(range(n))
                ax.set_yticks(range(n))
                ax.set_xticklabels([a[:3] for a in agents], fontsize=7, color='white')
                ax.set_yticklabels([a[:3] for a in agents], fontsize=7, color='white')

                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(colors='white', labelsize=6)
            else:
                ax.text(0.5, 0.5, 'Cannot compute', ha='center', va='center',
                       color='#666', fontsize=10, transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No agents', ha='center', va='center',
                   color='#666', fontsize=10, transform=ax.transAxes)

    def _plot_information_geometry(self, ax):
        """Plot information geometry manifold"""
        self._style_axis(ax, 'Information Geometry')

        agents = list(self.metrics.state_vectors.keys())

        if len(agents) >= 2:
            # Project agent trajectories onto 2D using Fisher metric
            trajectories = {}

            for agent in agents:
                states = self.metrics.state_vectors[agent]
                if len(states) >= 2:
                    # Simple projection based on mean and variance
                    traj = []
                    window = 5
                    for i in range(0, len(states) - window, window // 2):
                        segment = [s[0] if s else 0 for s in states[i:i + window]]
                        mean = sum(segment) / len(segment)
                        var = sum((s - mean) ** 2 for s in segment) / len(segment)
                        traj.append((mean, math.sqrt(var + 1e-10)))
                    trajectories[agent] = traj

            colors = ['#ff0066', '#00ffff', '#ff00ff', '#00ff66', '#ffff00']

            for idx, (agent, traj) in enumerate(trajectories.items()):
                if traj:
                    x = [t[0] for t in traj]
                    y = [t[1] for t in traj]
                    ax.plot(x, y, color=colors[idx % len(colors)],
                           linewidth=1.5, alpha=0.7, label=agent)
                    ax.scatter(x[-1], y[-1], color=colors[idx % len(colors)],
                              s=100, marker='*', zorder=10)

            ax.set_xlabel('μ (mean)', fontsize=8)
            ax.set_ylabel('σ (std)', fontsize=8)
            ax.legend(loc='upper right', fontsize=6, facecolor='#1a1a3a',
                     edgecolor='#333', labelcolor='white')
        else:
            ax.text(0.5, 0.5, 'Need 2+ agents', ha='center', va='center',
                   color='#666', fontsize=10, transform=ax.transAxes)

    def _plot_omega_summary(self, ax):
        """Plot omega-level summary"""
        self._style_axis(ax, 'Ω OMEGA SUMMARY Ω')

        # Compute all metrics
        n_agents = len(self.metrics.state_vectors)
        n_rounds = len(self.metrics.rounds)

        # TDA
        betti_0 = 0
        if self.metrics.phase_coordinates:
            all_points = []
            for coords in self.metrics.phase_coordinates.values():
                all_points.extend(coords[-10:])
            if len(all_points) >= 3:
                rips = RipsComplex(all_points, max_dim=1)
                diagram = rips.compute_persistence(1.5)
                betti_0 = diagram.get_betti(0)

        # Criticality
        crit_regime = "?"
        if self.metrics.tension_signal:
            analysis = self.criticality.analyze(self.metrics.tension_signal)
            crit_regime = analysis.regime.value[:8]

        # Fractal
        hurst = 0.5
        dfa = 0.5
        if self.metrics.tension_signal and len(self.metrics.tension_signal) > 20:
            hurst = self.fractal.hurst_exponent(self.metrics.tension_signal)
            dfa = self.fractal.dfa(self.metrics.tension_signal)

        # Symbolic
        pattern = "?"
        lz_complexity = 0
        if self.metrics.tension_signal:
            sym = self.symbolic.analyze(self.metrics.tension_signal)
            pattern = sym.pattern_type.value[:6]
            lz_complexity = sym.complexity

        summary = f"""
╔═══════════════════════════╗
║    OMEGA ANALYSIS         ║
╠═══════════════════════════╣
║ Agents:     {n_agents:3d}           ║
║ Rounds:     {n_rounds:3d}           ║
╠═══════════════════════════╣
║ TOPOLOGY                  ║
║   β₀:       {betti_0:3d}           ║
╠═══════════════════════════╣
║ CRITICALITY               ║
║   Regime:   {crit_regime:8s}     ║
╠═══════════════════════════╣
║ FRACTALS                  ║
║   Hurst H:  {hurst:.3f}         ║
║   DFA α:    {dfa:.3f}         ║
╠═══════════════════════════╣
║ SYMBOLIC                  ║
║   Pattern:  {pattern:6s}       ║
║   LZ:       {lz_complexity:.3f}         ║
╚═══════════════════════════╝
"""

        ax.text(0.05, 0.95, summary, transform=ax.transAxes,
               fontsize=8, color='#00ffff', fontfamily='monospace',
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='#0a0a1a', alpha=0.9,
                        edgecolor='#00ffff'))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])


# ============================================================================
# OMEGA ASCII VISUALIZATION
# ============================================================================

class OmegaASCIIVisualizer:
    """Omega-level ASCII visualization"""

    def __init__(self, metrics: OmegaSimulationMetrics):
        self.metrics = metrics
        self.criticality = CriticalityDetector()
        self.symbolic = SymbolicDynamicsAnalyzer()
        self.fractal = FractalAnalyzer()

    def render_full_report(self) -> str:
        """Render omega-level ASCII report"""
        lines = []

        lines.append("╔════════════════════════════════════════════════════════════════════════════════════╗")
        lines.append("║                    Ω  OMEGA-LEVEL SIMULATION ANALYTICS  Ω                          ║")
        lines.append("╠════════════════════════════════════════════════════════════════════════════════════╣")
        lines.append("")

        # System overview
        lines.append("┌─ SYSTEM OVERVIEW ───────────────────────────────────────────────────────────────────┐")
        lines.append(f"│ Agents: {len(self.metrics.state_vectors):3d}  │  Rounds: {len(self.metrics.rounds):3d}  │  "
                    f"Interactions: {sum(len(i) for i in self.metrics.interactions):4d}                       │")
        lines.append("└────────────────────────────────────────────────────────────────────────────────────────┘")
        lines.append("")

        # Criticality
        if self.metrics.tension_signal:
            analysis = self.criticality.analyze(self.metrics.tension_signal)
            lines.append("┌─ CRITICALITY ANALYSIS ───────────────────────────────────────────────────────────────┐")
            lines.append(f"│ Regime: {analysis.regime.value.upper():20s}                                           │")
            lines.append(f"│ Branching Ratio σ: {analysis.branching_ratio:6.3f}  (σ=1 is critical)                      │")
            lines.append(f"│ Order Parameter m: {analysis.order_parameter:6.3f}                                           │")
            lines.append(f"│ Susceptibility χ:  {analysis.susceptibility:6.3f}                                           │")
            if analysis.avalanche_sizes:
                lines.append(f"│ Avalanches: {len(analysis.avalanche_sizes):4d} detected, τ={analysis.power_law_exponent:.2f}                              │")
            lines.append("└────────────────────────────────────────────────────────────────────────────────────────┘")
            lines.append("")

        # Symbolic dynamics
        if self.metrics.tension_signal:
            sym = self.symbolic.analyze(self.metrics.tension_signal)
            lines.append("┌─ SYMBOLIC DYNAMICS ──────────────────────────────────────────────────────────────────┐")
            lines.append(f"│ Pattern Type: {sym.pattern_type.value.upper():15s}                                        │")
            lines.append(f"│ Topological Entropy: {sym.topological_entropy:6.3f}                                          │")
            lines.append(f"│ Permutation Entropy: {sym.permutation_entropy:6.3f}                                          │")
            lines.append(f"│ LZ Complexity:       {sym.complexity:6.3f}                                          │")
            lines.append(f"│ Sequence: {sym.symbol_sequence[:50]:50s}...│")
            lines.append("└────────────────────────────────────────────────────────────────────────────────────────┘")
            lines.append("")

        # Fractal analysis
        if self.metrics.tension_signal and len(self.metrics.tension_signal) > 20:
            frac = self.fractal.analyze(self.metrics.tension_signal)
            lines.append("┌─ FRACTAL ANALYSIS ───────────────────────────────────────────────────────────────────┐")
            lines.append(f"│ Hurst Exponent H: {frac.hurst_exponent:6.3f}  ({'PERSISTENT' if frac.hurst_exponent > 0.5 else 'ANTI-PERS' if frac.hurst_exponent < 0.5 else 'RANDOM':10s})              │")
            lines.append(f"│ DFA Exponent α:   {frac.dfa_exponent:6.3f}  ({'CORRELATED' if frac.dfa_exponent > 1 else 'ANTI-CORR' if frac.dfa_exponent < 0.5 else 'BROWNIAN':10s})              │")
            lines.append(f"│ Box-Counting Dim: {frac.box_counting_dim:6.3f}                                            │")
            if frac.singularity_spectrum:
                lines.append(f"│ Multifractal Width: {frac.singularity_spectrum.get('width', 0):6.3f}                                         │")
            lines.append("└────────────────────────────────────────────────────────────────────────────────────────┘")
            lines.append("")

        # Global signals
        lines.append("┌─ GLOBAL SIGNAL EVOLUTION ────────────────────────────────────────────────────────────┐")
        if self.metrics.tension_signal:
            lines.append(self._sparkline("Tension   ", self.metrics.tension_signal))
        if self.metrics.entropy_signal:
            lines.append(self._sparkline("Entropy   ", self.metrics.entropy_signal))
        if self.metrics.complexity_signal:
            lines.append(self._sparkline("Complexity", self.metrics.complexity_signal))
        if self.metrics.phi_signal:
            lines.append(self._sparkline("Φ (IIT)   ", self.metrics.phi_signal))
        lines.append("└────────────────────────────────────────────────────────────────────────────────────────┘")
        lines.append("")

        lines.append("╚════════════════════════════════════════════════════════════════════════════════════╝")

        return "\n".join(lines)

    def _sparkline(self, label: str, data: List[float], width: int = 55) -> str:
        """Create ASCII sparkline"""
        if not data:
            return f"│ {label}: No data"

        blocks = " ▁▂▃▄▅▆▇█"
        min_val, max_val = min(data), max(data)
        range_val = max_val - min_val if max_val != min_val else 1

        step = max(1, len(data) // width)
        resampled = [data[i] for i in range(0, len(data), step)][:width]

        sparkline = ""
        for val in resampled:
            idx = int((val - min_val) / range_val * (len(blocks) - 1))
            sparkline += blocks[idx]

        return f"│ {label}: {sparkline} │"


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_omega_metrics() -> OmegaSimulationMetrics:
    """Create omega metrics collector"""
    return OmegaSimulationMetrics()


def visualize_omega(metrics: OmegaSimulationMetrics,
                   save_path: str = None) -> Optional[Any]:
    """Create omega visualization"""
    viz = OmegaSimulationVisualizer(metrics)
    return viz.create_omega_dashboard(save_path)


def demo_omega_visualization():
    """Demo omega visualization"""
    print("Ω OMEGA-LEVEL VISUALIZATION DEMO Ω")
    metrics = OmegaSimulationMetrics()

    agents = ["Alpha", "Beta", "Gamma", "Delta"]
    n_rounds = 50

    states = {a: [0.5 + 0.1 * i for i in range(7)] for i, a in enumerate(agents)}

    for r in range(n_rounds):
        metrics.record_round(r)

        p = r / n_rounds
        tension = 0.3 + 0.4 * math.sin(2 * math.pi * p) + 0.1 * random.gauss(0, 1)
        tension = max(0.1, min(0.9, tension))

        for a in agents:
            s = states[a]
            x = s[0]
            new_x = 3.8 * x * (1 - x) + 0.05 * random.gauss(0, 1)
            new_x = max(0, min(1, new_x))

            s[0] = new_x
            s[1] = 0.5 + 0.3 * math.sin(r / 5)
            s[2] = 0.5 + 0.2 * new_x

            metrics.record_agent_state(a, s.copy(), (s[0], s[1], s[2]),
                                      {"coop": new_x}, random.choice(["C", "D"]))

        entropy = 0.5 + 0.2 * random.random()
        complexity = 0.5 + 0.3 * tension
        phi = 0.3 + 0.2 * tension

        metrics.record_global_state(tension, entropy, complexity, phi)

    print("Generating omega dashboard...")
    OmegaSimulationVisualizer(metrics).create_omega_dashboard("omega_dashboard.png")
    print("✓ Saved omega_dashboard.png")

    print(OmegaASCIIVisualizer(metrics).render_full_report())

    return metrics


if __name__ == "__main__":
    demo_omega_visualization()
