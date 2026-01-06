"""Emergent behavior analysis and swarm intelligence.

Provides:
- Swarm coordination patterns (flocking, foraging, clustering)
- Emergence detection and measurement
- Collective intelligence metrics
- Stigmergy-based coordination
- Self-organization dynamics
- Phase transition detection
- Fitness landscape analysis
- Novelty search and diversity metrics
"""
from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable, Set
from enum import Enum
from datetime import datetime, timezone
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    from scipy.spatial import distance
    from scipy.cluster import hierarchy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class SwarmBehavior(Enum):
    """Types of swarm behaviors."""
    FLOCKING = "flocking"  # Coordinated movement
    FORAGING = "foraging"  # Resource gathering
    CLUSTERING = "clustering"  # Spatial aggregation
    DISPERSAL = "dispersal"  # Spatial spreading
    QUORUM_SENSING = "quorum_sensing"  # Density-dependent behavior
    DIVISION_OF_LABOR = "division_of_labor"  # Task specialization
    COLLECTIVE_DECISION = "collective_decision"  # Group decision-making
    SELF_ASSEMBLY = "self_assembly"  # Structure formation


class EmergenceType(Enum):
    """Types of emergent phenomena."""
    WEAK = "weak"  # Aggregate properties (e.g., average)
    STRONG = "strong"  # Novel properties not predictable from components
    PHASE_TRANSITION = "phase_transition"  # Sudden qualitative change
    PATTERN_FORMATION = "pattern_formation"  # Spatial/temporal patterns
    SYNCHRONIZATION = "synchronization"  # Coordinated oscillations
    SELF_ORGANIZATION = "self_organization"  # Order without central control


@dataclass
class Agent:
    """A simple agent in the swarm."""

    id: str
    position: np.ndarray  # Spatial position
    velocity: np.ndarray  # Velocity vector
    state: Dict[str, Any] = field(default_factory=dict)  # Internal state

    # Behavioral parameters
    perception_radius: float = 1.0
    max_speed: float = 1.0

    def distance_to(self, other: "Agent") -> float:
        """Compute distance to another agent."""
        return np.linalg.norm(self.position - other.position)

    def neighbors(self, agents: List["Agent"]) -> List["Agent"]:
        """Find neighbors within perception radius."""
        return [a for a in agents if a.id != self.id and self.distance_to(a) <= self.perception_radius]


# ═══════════════════════════════════════════════════════════════════════════
# SWARM COORDINATION PATTERNS
# ═══════════════════════════════════════════════════════════════════════════

class FlockingBehavior:
    """Reynolds' flocking behavior (boids algorithm)."""

    def __init__(
        self,
        separation_weight: float = 1.5,
        alignment_weight: float = 1.0,
        cohesion_weight: float = 1.0,
    ):
        self.separation_weight = separation_weight
        self.alignment_weight = alignment_weight
        self.cohesion_weight = cohesion_weight

    def update_agent(self, agent: Agent, neighbors: List[Agent]) -> np.ndarray:
        """Compute velocity update for agent based on three rules."""
        if not neighbors:
            return agent.velocity

        # Rule 1: Separation - avoid crowding
        separation = np.zeros_like(agent.position)
        for neighbor in neighbors:
            diff = agent.position - neighbor.position
            dist = np.linalg.norm(diff)
            if dist > 0:
                separation += diff / dist

        # Rule 2: Alignment - steer towards average heading
        alignment = np.mean([n.velocity for n in neighbors], axis=0)

        # Rule 3: Cohesion - steer towards average position
        center_of_mass = np.mean([n.position for n in neighbors], axis=0)
        cohesion = center_of_mass - agent.position

        # Weighted combination
        new_velocity = (
            self.separation_weight * separation +
            self.alignment_weight * alignment +
            self.cohesion_weight * cohesion
        )

        # Limit speed
        speed = np.linalg.norm(new_velocity)
        if speed > agent.max_speed:
            new_velocity = (new_velocity / speed) * agent.max_speed

        return new_velocity


class StigmergyCoordination:
    """Stigmergy-based coordination (indirect communication via environment)."""

    def __init__(self, grid_size: Tuple[int, int] = (100, 100)):
        self.grid_size = grid_size
        # Pheromone grid
        self.pheromones = np.zeros(grid_size)
        # Decay rate per timestep
        self.decay_rate = 0.01
        # Diffusion coefficient
        self.diffusion = 0.1

    def deposit_pheromone(self, position: np.ndarray, amount: float = 1.0):
        """Deposit pheromone at position."""
        x, y = int(position[0]) % self.grid_size[0], int(position[1]) % self.grid_size[1]
        self.pheromones[x, y] += amount

    def sense_pheromone(self, position: np.ndarray, radius: int = 3) -> float:
        """Sense pheromone concentration around position."""
        x, y = int(position[0]) % self.grid_size[0], int(position[1]) % self.grid_size[1]

        # Get local neighborhood
        x_min, x_max = max(0, x - radius), min(self.grid_size[0], x + radius + 1)
        y_min, y_max = max(0, y - radius), min(self.grid_size[1], y + radius + 1)

        return self.pheromones[x_min:x_max, y_min:y_max].mean()

    def get_gradient(self, position: np.ndarray) -> np.ndarray:
        """Get pheromone gradient at position (for chemotaxis)."""
        x, y = int(position[0]) % self.grid_size[0], int(position[1]) % self.grid_size[1]

        # Finite difference gradient
        dx = self.pheromones[(x+1) % self.grid_size[0], y] - \
             self.pheromones[(x-1) % self.grid_size[0], y]
        dy = self.pheromones[x, (y+1) % self.grid_size[1]] - \
             self.pheromones[x, (y-1) % self.grid_size[1]]

        return np.array([dx, dy])

    def update(self):
        """Update pheromone field (decay and diffusion)."""
        # Decay
        self.pheromones *= (1 - self.decay_rate)

        # Diffusion (simple 2D convolution)
        if self.diffusion > 0:
            from scipy.ndimage import gaussian_filter
            self.pheromones = gaussian_filter(self.pheromones, sigma=self.diffusion)


class QuorumSensing:
    """Quorum sensing for density-dependent behavior."""

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    def check_quorum(self, agent: Agent, neighbors: List[Agent], total_agents: int) -> bool:
        """Check if local density exceeds quorum threshold."""
        if total_agents == 0:
            return False

        local_density = len(neighbors) / total_agents
        return local_density >= self.threshold


# ═══════════════════════════════════════════════════════════════════════════
# EMERGENCE DETECTION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EmergentProperty:
    """A detected emergent property."""

    property_name: str
    emergence_type: EmergenceType
    strength: float  # How strongly emergent
    description: str
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class EmergenceDetector:
    """Detect and measure emergent phenomena in swarms."""

    def __init__(self):
        self.history: List[Dict[str, Any]] = []
        self.detected_properties: List[EmergentProperty] = []

    def record_state(self, agents: List[Agent], metadata: Optional[Dict] = None):
        """Record current swarm state."""
        state = {
            "timestamp": datetime.now(timezone.utc),
            "num_agents": len(agents),
            "positions": np.array([a.position for a in agents]),
            "velocities": np.array([a.velocity for a in agents]),
            "metadata": metadata or {},
        }
        self.history.append(state)

    def detect_synchronization(self, agents: List[Agent], threshold: float = 0.8) -> Optional[EmergentProperty]:
        """Detect velocity synchronization (alignment)."""
        if len(agents) < 2:
            return None

        velocities = np.array([a.velocity for a in agents])

        # Compute pairwise alignment
        alignments = []
        for i in range(len(velocities)):
            for j in range(i + 1, len(velocities)):
                v1, v2 = velocities[i], velocities[j]
                norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)

                if norm1 > 0 and norm2 > 0:
                    alignment = np.dot(v1, v2) / (norm1 * norm2)
                    alignments.append(alignment)

        if not alignments:
            return None

        avg_alignment = np.mean(alignments)

        if avg_alignment >= threshold:
            prop = EmergentProperty(
                property_name="velocity_synchronization",
                emergence_type=EmergenceType.SYNCHRONIZATION,
                strength=avg_alignment,
                description=f"{avg_alignment:.2%} velocity alignment across swarm",
                metrics={"avg_alignment": avg_alignment, "std_alignment": np.std(alignments)},
            )
            self.detected_properties.append(prop)
            return prop

        return None

    def detect_clustering(self, agents: List[Agent], method: str = "dbscan") -> Optional[EmergentProperty]:
        """Detect spatial clustering."""
        if not SCIPY_AVAILABLE or len(agents) < 3:
            return None

        positions = np.array([a.position for a in agents])

        if method == "hierarchical":
            # Hierarchical clustering
            linkage = hierarchy.linkage(positions, method='ward')

            # Compute cophenetic correlation (measure of clustering quality)
            from scipy.spatial.distance import pdist
            from scipy.cluster.hierarchy import cophenet

            c, _ = cophenet(linkage, pdist(positions))

            if c >= 0.7:  # Strong clustering
                prop = EmergentProperty(
                    property_name="spatial_clustering",
                    emergence_type=EmergenceType.PATTERN_FORMATION,
                    strength=c,
                    description=f"Hierarchical clustering detected (cophenetic={c:.3f})",
                    metrics={"cophenetic_correlation": c},
                )
                self.detected_properties.append(prop)
                return prop

        return None

    def detect_phase_transition(
        self,
        metric_name: str,
        metric_fn: Callable[[List[Agent]], float],
        agents: List[Agent],
        window_size: int = 10,
        threshold: float = 2.0,
    ) -> Optional[EmergentProperty]:
        """Detect phase transitions in system dynamics.

        Phase transition = sudden change in order parameter.
        """
        if len(self.history) < window_size * 2:
            return None

        # Compute metric over recent history
        recent_values = []
        for state in self.history[-window_size*2:]:
            # Reconstruct agents from state (simplified)
            value = metric_fn(agents)  # In practice, would use historical state
            recent_values.append(value)

        # Split into before/after windows
        before = recent_values[:window_size]
        after = recent_values[window_size:]

        # Compute change
        mean_before = np.mean(before)
        mean_after = np.mean(after)
        std_before = np.std(before)

        if std_before > 0:
            change_magnitude = abs(mean_after - mean_before) / std_before

            if change_magnitude >= threshold:
                prop = EmergentProperty(
                    property_name=f"phase_transition_{metric_name}",
                    emergence_type=EmergenceType.PHASE_TRANSITION,
                    strength=change_magnitude,
                    description=f"Phase transition in {metric_name} (magnitude={change_magnitude:.2f}σ)",
                    metrics={
                        "metric": metric_name,
                        "before_mean": mean_before,
                        "after_mean": mean_after,
                        "change_sigma": change_magnitude,
                    },
                )
                self.detected_properties.append(prop)
                return prop

        return None

    def compute_entropy(self, agents: List[Agent], bins: int = 10) -> float:
        """Compute spatial entropy (measure of disorder)."""
        if len(agents) == 0:
            return 0.0

        positions = np.array([a.position for a in agents])

        # 2D histogram
        H, _, _ = np.histogram2d(
            positions[:, 0],
            positions[:, 1],
            bins=bins,
        )

        # Normalize to probability distribution
        H = H / H.sum()

        # Compute Shannon entropy
        H_flat = H.flatten()
        H_flat = H_flat[H_flat > 0]  # Remove zeros

        entropy = -np.sum(H_flat * np.log(H_flat))

        return float(entropy)


# ═══════════════════════════════════════════════════════════════════════════
# COLLECTIVE INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════

class CollectiveIntelligence:
    """Measure and harness collective intelligence of agent swarms."""

    def __init__(self):
        self.decision_history: List[Dict] = []

    def collective_decision(
        self,
        agents: List[Agent],
        options: List[Any],
        voting_fn: Callable[[Agent, List[Any]], Any],
    ) -> Tuple[Any, Dict[str, float]]:
        """Make collective decision via voting.

        Args:
            agents: Agents participating in decision
            options: Available options
            voting_fn: Function that takes (agent, options) and returns vote

        Returns:
            (winning_option, vote_distribution)
        """
        votes = defaultdict(int)

        for agent in agents:
            vote = voting_fn(agent, options)
            votes[vote] += 1

        # Find winner
        winner = max(votes.items(), key=lambda x: x[1])[0]

        # Compute distribution
        total_votes = sum(votes.values())
        distribution = {opt: votes[opt] / total_votes for opt in options}

        # Record decision
        self.decision_history.append({
            "timestamp": datetime.now(timezone.utc),
            "num_agents": len(agents),
            "options": options,
            "winner": winner,
            "distribution": distribution,
        })

        return winner, distribution

    def wisdom_of_crowds(
        self,
        estimates: List[float],
        method: str = "median",
        trim_percent: float = 0.1,
    ) -> float:
        """Aggregate individual estimates into collective estimate.

        Args:
            estimates: Individual agent estimates
            method: Aggregation method (mean, median, trimmed_mean)
            trim_percent: Percent to trim from each end (for trimmed_mean)

        Returns:
            Collective estimate
        """
        estimates = np.array(estimates)

        if method == "mean":
            return float(np.mean(estimates))
        elif method == "median":
            return float(np.median(estimates))
        elif method == "trimmed_mean":
            # Remove outliers
            n_trim = int(len(estimates) * trim_percent)
            sorted_est = np.sort(estimates)
            trimmed = sorted_est[n_trim:-n_trim] if n_trim > 0 else sorted_est
            return float(np.mean(trimmed))
        else:
            raise ValueError(f"Unknown method: {method}")

    def diversity_metric(self, agents: List[Agent], feature_fn: Callable[[Agent], np.ndarray]) -> float:
        """Compute diversity of agent population.

        Higher diversity → more varied perspectives → better collective intelligence.
        """
        if len(agents) < 2:
            return 0.0

        # Extract features
        features = np.array([feature_fn(a) for a in agents])

        # Compute pairwise distances
        if SCIPY_AVAILABLE:
            from scipy.spatial.distance import pdist
            distances = pdist(features)
            return float(np.mean(distances))
        else:
            # Manual pairwise distances
            total_dist = 0.0
            count = 0
            for i in range(len(features)):
                for j in range(i + 1, len(features)):
                    total_dist += np.linalg.norm(features[i] - features[j])
                    count += 1

            return total_dist / count if count > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# FITNESS LANDSCAPE & EVOLUTION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Individual:
    """An individual in evolutionary search."""

    genome: np.ndarray  # Genotype
    fitness: float  # Fitness score
    novelty: float = 0.0  # Novelty score (for novelty search)
    age: int = 0  # Age in generations


class NoveltySearch:
    """Novelty search for discovering diverse solutions."""

    def __init__(self, k_nearest: int = 15):
        self.k_nearest = k_nearest
        self.archive: List[np.ndarray] = []  # Archive of novel behaviors
        self.archive_threshold = 0.1  # Minimum novelty to enter archive

    def compute_novelty(self, behavior: np.ndarray, population_behaviors: List[np.ndarray]) -> float:
        """Compute novelty as average distance to k-nearest neighbors."""
        if not population_behaviors and not self.archive:
            return 0.0

        # Combine population and archive
        all_behaviors = population_behaviors + self.archive

        # Compute distances
        if SCIPY_AVAILABLE:
            from scipy.spatial.distance import cdist
            distances = cdist([behavior], all_behaviors)[0]
        else:
            distances = [np.linalg.norm(behavior - b) for b in all_behaviors]

        # Sort and take k nearest
        distances.sort()
        k = min(self.k_nearest, len(distances))

        return float(np.mean(distances[:k]))

    def update_archive(self, behavior: np.ndarray, novelty: float):
        """Add behavior to archive if sufficiently novel."""
        if novelty >= self.archive_threshold:
            self.archive.append(behavior)


class FitnessLandscape:
    """Analyze fitness landscape properties."""

    def __init__(self):
        self.samples: List[Tuple[np.ndarray, float]] = []  # (genome, fitness) pairs

    def add_sample(self, genome: np.ndarray, fitness: float):
        """Add fitness sample."""
        self.samples.append((genome, fitness))

    def estimate_ruggedness(self, sample_size: int = 100) -> float:
        """Estimate landscape ruggedness (autocorrelation).

        Rugged landscapes have many local optima.
        """
        if len(self.samples) < sample_size:
            return 0.0

        # Sample random pairs
        correlations = []
        for _ in range(sample_size):
            i, j = np.random.choice(len(self.samples), 2, replace=False)
            genome1, fit1 = self.samples[i]
            genome2, fit2 = self.samples[j]

            # Distance between genomes
            dist = np.linalg.norm(genome1 - genome2)

            # Fitness difference
            fit_diff = abs(fit1 - fit2)

            correlations.append((dist, fit_diff))

        # Compute correlation
        if SCIPY_AVAILABLE:
            from scipy.stats import spearmanr
            dists = [c[0] for c in correlations]
            fit_diffs = [c[1] for c in correlations]
            corr, _ = spearmanr(dists, fit_diffs)

            # Ruggedness = 1 - correlation (low correlation → high ruggedness)
            return 1.0 - abs(corr)

        return 0.5  # Default

    def find_local_optima(self, epsilon: float = 0.01) -> List[Tuple[np.ndarray, float]]:
        """Find local optima (fitness higher than all neighbors within epsilon)."""
        optima = []

        for i, (genome, fitness) in enumerate(self.samples):
            is_optimum = True

            # Check neighbors
            for j, (other_genome, other_fitness) in enumerate(self.samples):
                if i == j:
                    continue

                dist = np.linalg.norm(genome - other_genome)

                if dist < epsilon and other_fitness > fitness:
                    is_optimum = False
                    break

            if is_optimum:
                optima.append((genome, fitness))

        return optima


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED EMERGENT INTELLIGENCE SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

class EmergentIntelligenceSystem:
    """Unified system for emergent behavior and swarm intelligence."""

    def __init__(self):
        self.flocking = FlockingBehavior()
        self.stigmergy = StigmergyCoordination()
        self.quorum = QuorumSensing()
        self.emergence_detector = EmergenceDetector()
        self.collective_intelligence = CollectiveIntelligence()
        self.novelty_search = NoveltySearch()
        self.fitness_landscape = FitnessLandscape()

        # Active agents
        self.agents: Dict[str, Agent] = {}

    def add_agent(self, agent: Agent):
        """Add agent to system."""
        self.agents[agent.id] = agent

    def remove_agent(self, agent_id: str):
        """Remove agent."""
        if agent_id in self.agents:
            del self.agents[agent_id]

    async def update(self, behavior: SwarmBehavior = SwarmBehavior.FLOCKING):
        """Update swarm for one timestep."""
        agents_list = list(self.agents.values())

        if behavior == SwarmBehavior.FLOCKING:
            # Update velocities using flocking rules
            for agent in agents_list:
                neighbors = agent.neighbors(agents_list)
                new_velocity = self.flocking.update_agent(agent, neighbors)
                agent.velocity = new_velocity

        # Update positions
        for agent in agents_list:
            agent.position += agent.velocity

        # Update stigmergy
        self.stigmergy.update()

        # Record state for emergence detection
        self.emergence_detector.record_state(agents_list)

    def detect_emergent_phenomena(self) -> List[EmergentProperty]:
        """Detect all types of emergent phenomena."""
        agents_list = list(self.agents.values())
        detected = []

        # Check synchronization
        sync_prop = self.emergence_detector.detect_synchronization(agents_list)
        if sync_prop:
            detected.append(sync_prop)

        # Check clustering
        cluster_prop = self.emergence_detector.detect_clustering(agents_list)
        if cluster_prop:
            detected.append(cluster_prop)

        return detected

    def measure_collective_performance(self, task_fn: Callable[[List[Agent]], float]) -> float:
        """Measure collective performance on a task."""
        agents_list = list(self.agents.values())
        return task_fn(agents_list)


# Global instance
_emergent_system: Optional[EmergentIntelligenceSystem] = None


def get_emergent_system() -> EmergentIntelligenceSystem:
    """Get global emergent intelligence system."""
    global _emergent_system
    if _emergent_system is None:
        _emergent_system = EmergentIntelligenceSystem()
    return _emergent_system
