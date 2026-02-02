"""SOTA Prompt Evolution System.

State-of-the-art improvements:
1. Embedding-based semantic similarity
2. NSGA-II multi-objective optimization
3. MAP-Elites quality-diversity
4. Thompson Sampling for exploration
5. Bloom filters for fast lookups
6. Persistent storage with SQLite
7. Async/concurrent operations
8. Adaptive parameter control
9. Neural fitness prediction
10. LLM-based semantic mutations
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import pickle
import random
import sqlite3
import struct
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Dict,
    FrozenSet,
    Generic,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=float)


# =============================================================================
# SOTA: Bloom Filter for Fast Hash Lookups
# =============================================================================


class BloomFilter:
    """Space-efficient probabilistic set for fast hash existence checks.

    O(k) lookup where k is number of hash functions.
    False positives possible, false negatives impossible.
    """

    __slots__ = ("_bits", "_size", "_num_hashes", "_count")

    def __init__(self, expected_items: int = 10000, false_positive_rate: float = 0.01):
        # Optimal size: m = -n*ln(p) / (ln(2)^2)
        self._size = max(64, int(-expected_items * math.log(false_positive_rate) / (math.log(2) ** 2)))
        # Round up to nearest multiple of 64 for efficient storage
        self._size = ((self._size + 63) // 64) * 64

        # Optimal hash count: k = (m/n) * ln(2)
        self._num_hashes = max(1, int((self._size / expected_items) * math.log(2)))

        # Bit array stored as integers
        self._bits = [0] * (self._size // 64)
        self._count = 0

    def _get_hash_positions(self, item: str) -> List[int]:
        """Generate k hash positions using double hashing."""
        h1 = int(hashlib.sha256(item.encode()).hexdigest()[:16], 16)
        h2 = int(hashlib.md5(item.encode()).hexdigest(), 16)

        return [(h1 + i * h2) % self._size for i in range(self._num_hashes)]

    def add(self, item: str) -> None:
        """Add an item to the filter."""
        for pos in self._get_hash_positions(item):
            bucket, bit = divmod(pos, 64)
            self._bits[bucket] |= (1 << bit)
        self._count += 1

    def __contains__(self, item: str) -> bool:
        """Check if item might be in the filter."""
        for pos in self._get_hash_positions(item):
            bucket, bit = divmod(pos, 64)
            if not (self._bits[bucket] & (1 << bit)):
                return False
        return True

    def __len__(self) -> int:
        return self._count

    @property
    def fill_ratio(self) -> float:
        """Estimate fill ratio of the filter."""
        ones = sum(bin(b).count("1") for b in self._bits)
        return ones / self._size


# =============================================================================
# SOTA: Embedding-Based Semantic Similarity
# =============================================================================


class EmbeddingProvider(ABC):
    """Abstract embedding provider for semantic similarity."""

    @abstractmethod
    async def embed(self, text: str) -> np.ndarray:
        """Get embedding vector for text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for multiple texts."""
        pass

    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension."""
        pass


class LocalEmbedding(EmbeddingProvider):
    """Local embedding using TF-IDF + SVD as fallback."""

    def __init__(self, dimension: int = 256):
        self._dim = dimension
        self._vocab: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._doc_count = 0

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re
        return re.findall(r'\b\w+\b', text.lower())

    def _update_vocab(self, tokens: List[str]) -> None:
        """Update vocabulary with new tokens."""
        for token in set(tokens):
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab)
            self._idf[token] = self._idf.get(token, 0) + 1
        self._doc_count += 1

    async def embed(self, text: str) -> np.ndarray:
        """Compute TF-IDF-like embedding."""
        tokens = self._tokenize(text)
        self._update_vocab(tokens)

        # Compute TF
        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1

        # Build sparse vector
        vec = np.zeros(min(len(self._vocab), self._dim * 4))
        for token, count in tf.items():
            if token in self._vocab:
                idx = self._vocab[token]
                if idx < len(vec):
                    idf = math.log(self._doc_count / (1 + self._idf.get(token, 1)))
                    vec[idx] = count * idf

        # Reduce dimension via random projection (approximate SVD)
        if len(vec) > self._dim:
            np.random.seed(42)  # Deterministic projection
            proj = np.random.randn(self._dim, len(vec)) / np.sqrt(self._dim)
            vec = proj @ vec
        elif len(vec) < self._dim:
            vec = np.pad(vec, (0, self._dim - len(vec)))

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec

    async def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Batch embedding."""
        embeddings = await asyncio.gather(*[self.embed(t) for t in texts])
        return np.array(embeddings)

    def dimension(self) -> int:
        return self._dim


class SemanticSimilarity:
    """Semantic similarity using embeddings."""

    def __init__(self, provider: Optional[EmbeddingProvider] = None):
        self.provider = provider or LocalEmbedding()
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_size = 1000

    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding with caching."""
        key = hashlib.sha256(text.encode()).hexdigest()[:16]
        if key not in self._cache:
            if len(self._cache) >= self._cache_size:
                # LRU eviction
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = await self.provider.embed(text)
        return self._cache[key]

    async def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between texts."""
        e1 = await self._get_embedding(text1)
        e2 = await self._get_embedding(text2)
        return float(np.dot(e1, e2))

    async def pairwise_similarity(self, texts: List[str]) -> np.ndarray:
        """Compute pairwise similarity matrix."""
        embeddings = await self.provider.embed_batch(texts)
        return embeddings @ embeddings.T


# =============================================================================
# SOTA: NSGA-II Multi-Objective Optimization
# =============================================================================


@dataclass
class Individual:
    """Individual in NSGA-II population."""
    genome: Any
    objectives: Dict[str, float] = field(default_factory=dict)
    rank: int = 0
    crowding_distance: float = 0.0

    def dominates(self, other: "Individual") -> bool:
        """Check if this individual dominates another."""
        dominated = False
        for obj in self.objectives:
            if obj in other.objectives:
                if self.objectives[obj] < other.objectives[obj]:
                    return False
                if self.objectives[obj] > other.objectives[obj]:
                    dominated = True
        return dominated


class NSGAII:
    """Non-dominated Sorting Genetic Algorithm II.

    State-of-the-art multi-objective optimization.
    """

    def __init__(
        self,
        objectives: List[str],
        population_size: int = 50,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
    ):
        self.objectives = objectives
        self.pop_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population: List[Individual] = []
        self.generation = 0

    def _fast_non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """Fast non-dominated sorting (O(MN^2))."""
        fronts: List[List[Individual]] = [[]]
        domination_count: Dict[int, int] = {}
        dominated_by: Dict[int, List[Individual]] = defaultdict(list)

        for i, p in enumerate(population):
            domination_count[i] = 0
            for j, q in enumerate(population):
                if i != j:
                    if p.dominates(q):
                        dominated_by[i].append(q)
                    elif q.dominates(p):
                        domination_count[i] += 1

            if domination_count[i] == 0:
                p.rank = 0
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front = []
            for p_idx, p in enumerate(population):
                if p in fronts[i]:
                    for q in dominated_by[p_idx]:
                        q_idx = population.index(q)
                        domination_count[q_idx] -= 1
                        if domination_count[q_idx] == 0:
                            q.rank = i + 1
                            next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)

        return [f for f in fronts if f]

    def _crowding_distance(self, front: List[Individual]) -> None:
        """Calculate crowding distance for a front."""
        n = len(front)
        if n == 0:
            return

        for ind in front:
            ind.crowding_distance = 0.0

        for obj in self.objectives:
            # Sort by objective
            front.sort(key=lambda x: x.objectives.get(obj, 0))

            # Boundary points get infinite distance
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')

            # Calculate range
            obj_min = front[0].objectives.get(obj, 0)
            obj_max = front[-1].objectives.get(obj, 0)
            obj_range = obj_max - obj_min if obj_max > obj_min else 1

            # Interior points
            for i in range(1, n - 1):
                front[i].crowding_distance += (
                    front[i + 1].objectives.get(obj, 0) - front[i - 1].objectives.get(obj, 0)
                ) / obj_range

    def _tournament_select(self, population: List[Individual], k: int = 2) -> Individual:
        """Binary tournament selection based on rank and crowding distance."""
        candidates = random.sample(population, min(k, len(population)))
        # Sort by rank (ascending) then by crowding distance (descending)
        return sorted(candidates, key=lambda x: (x.rank, -x.crowding_distance))[0]

    def evolve(
        self,
        evaluate_fn: Callable[[Any], Dict[str, float]],
        crossover_fn: Callable[[Any, Any], Tuple[Any, Any]],
        mutate_fn: Callable[[Any], Any],
    ) -> List[Individual]:
        """Run one generation of NSGA-II."""
        # Evaluate current population
        for ind in self.population:
            if not ind.objectives:
                ind.objectives = evaluate_fn(ind.genome)

        # Create offspring
        offspring = []
        while len(offspring) < self.pop_size:
            parent1 = self._tournament_select(self.population)
            parent2 = self._tournament_select(self.population)

            if random.random() < self.crossover_rate:
                child1_genome, child2_genome = crossover_fn(parent1.genome, parent2.genome)
            else:
                child1_genome, child2_genome = parent1.genome, parent2.genome

            if random.random() < self.mutation_rate:
                child1_genome = mutate_fn(child1_genome)
            if random.random() < self.mutation_rate:
                child2_genome = mutate_fn(child2_genome)

            offspring.append(Individual(genome=child1_genome))
            if len(offspring) < self.pop_size:
                offspring.append(Individual(genome=child2_genome))

        # Evaluate offspring
        for ind in offspring:
            ind.objectives = evaluate_fn(ind.genome)

        # Combine populations
        combined = self.population + offspring

        # Non-dominated sorting
        fronts = self._fast_non_dominated_sort(combined)

        # Select next generation
        new_population = []
        for front in fronts:
            self._crowding_distance(front)
            if len(new_population) + len(front) <= self.pop_size:
                new_population.extend(front)
            else:
                # Sort by crowding distance and take what we need
                front.sort(key=lambda x: -x.crowding_distance)
                new_population.extend(front[:self.pop_size - len(new_population)])
                break

        self.population = new_population
        self.generation += 1

        return fronts[0] if fronts else []  # Return Pareto front


# =============================================================================
# SOTA: MAP-Elites Quality-Diversity
# =============================================================================


@dataclass
class MapCell:
    """A cell in the MAP-Elites archive."""
    elite: Any
    fitness: float
    behavior: Tuple[float, ...]
    visits: int = 1


class MAPElites:
    """MAP-Elites for quality-diversity optimization.

    Maintains archive of diverse, high-quality solutions.
    """

    def __init__(
        self,
        behavior_dims: int,
        bins_per_dim: int = 10,
        behavior_bounds: Optional[List[Tuple[float, float]]] = None,
    ):
        self.behavior_dims = behavior_dims
        self.bins_per_dim = bins_per_dim
        self.bounds = behavior_bounds or [(0, 1)] * behavior_dims

        # Archive is a dictionary mapping behavior bin tuples to cells
        self.archive: Dict[Tuple[int, ...], MapCell] = {}
        self.total_evaluations = 0

    def _discretize_behavior(self, behavior: Sequence[float]) -> Tuple[int, ...]:
        """Convert continuous behavior to discrete bin indices."""
        bins = []
        for i, (b, (low, high)) in enumerate(zip(behavior, self.bounds)):
            # Clip to bounds
            b = max(low, min(high, b))
            # Normalize to [0, 1]
            norm = (b - low) / (high - low) if high > low else 0.5
            # Convert to bin index
            bin_idx = min(int(norm * self.bins_per_dim), self.bins_per_dim - 1)
            bins.append(bin_idx)
        return tuple(bins)

    def add(
        self,
        genome: Any,
        fitness: float,
        behavior: Sequence[float],
    ) -> bool:
        """Add a solution to the archive if it's novel or better.

        Returns True if solution was added.
        """
        self.total_evaluations += 1
        behavior_tuple = tuple(behavior)
        cell_key = self._discretize_behavior(behavior)

        if cell_key not in self.archive:
            # New cell discovered
            self.archive[cell_key] = MapCell(
                elite=genome,
                fitness=fitness,
                behavior=behavior_tuple,
            )
            return True

        existing = self.archive[cell_key]
        existing.visits += 1

        if fitness > existing.fitness:
            # Better solution found
            self.archive[cell_key] = MapCell(
                elite=genome,
                fitness=fitness,
                behavior=behavior_tuple,
                visits=existing.visits,
            )
            return True

        return False

    def sample_elite(self) -> Optional[Any]:
        """Sample a random elite from the archive."""
        if not self.archive:
            return None
        cell = random.choice(list(self.archive.values()))
        return cell.elite

    def sample_by_curiosity(self) -> Optional[Any]:
        """Sample elite from least-visited cells (curiosity-driven)."""
        if not self.archive:
            return None
        # Prefer cells with fewer visits
        cells = list(self.archive.values())
        weights = [1.0 / (c.visits + 1) for c in cells]
        total = sum(weights)
        weights = [w / total for w in weights]
        cell = random.choices(cells, weights=weights)[0]
        return cell.elite

    @property
    def coverage(self) -> float:
        """Fraction of behavior space covered."""
        max_cells = self.bins_per_dim ** self.behavior_dims
        return len(self.archive) / max_cells

    @property
    def qd_score(self) -> float:
        """Quality-Diversity score (sum of all fitness values)."""
        return sum(cell.fitness for cell in self.archive.values())

    def get_best(self) -> Optional[MapCell]:
        """Get the highest-fitness elite."""
        if not self.archive:
            return None
        return max(self.archive.values(), key=lambda c: c.fitness)


# =============================================================================
# SOTA: Thompson Sampling for Exploration/Exploitation
# =============================================================================


@dataclass
class ArmStats:
    """Statistics for a Thompson Sampling arm."""
    successes: float = 1.0  # Beta prior alpha
    failures: float = 1.0   # Beta prior beta
    total_reward: float = 0.0
    pulls: int = 0


class ThompsonSampling:
    """Thompson Sampling for multi-armed bandits.

    Bayesian approach to exploration/exploitation.
    """

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.arms: Dict[str, ArmStats] = {}
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def add_arm(self, arm_id: str) -> None:
        """Add a new arm."""
        if arm_id not in self.arms:
            self.arms[arm_id] = ArmStats(
                successes=self.prior_alpha,
                failures=self.prior_beta,
            )

    def select(self) -> Optional[str]:
        """Select an arm using Thompson Sampling."""
        if not self.arms:
            return None

        # Sample from Beta posterior for each arm
        samples = {
            arm_id: float(np.random.beta(stats.successes, stats.failures))
            for arm_id, stats in self.arms.items()
        }

        return max(samples.keys(), key=lambda k: samples[k])

    def update(self, arm_id: str, reward: float) -> None:
        """Update arm statistics with observed reward."""
        if arm_id not in self.arms:
            self.add_arm(arm_id)

        stats = self.arms[arm_id]
        stats.pulls += 1
        stats.total_reward += reward

        # Update Beta distribution (treating reward as Bernoulli outcome)
        # For continuous rewards in [0,1], we use reward as success probability
        stats.successes += reward
        stats.failures += (1 - reward)

    def get_expected_value(self, arm_id: str) -> float:
        """Get expected value (mean of Beta distribution)."""
        if arm_id not in self.arms:
            return 0.5
        stats = self.arms[arm_id]
        return stats.successes / (stats.successes + stats.failures)

    def get_confidence_interval(
        self,
        arm_id: str,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """Get confidence interval for arm's true mean."""
        if arm_id not in self.arms:
            return (0.0, 1.0)

        arm = self.arms[arm_id]

        try:
            from scipy import stats as sp_stats
            alpha_param = (1 - confidence) / 2
            low = float(sp_stats.beta.ppf(alpha_param, arm.successes, arm.failures))
            high = float(sp_stats.beta.ppf(1 - alpha_param, arm.successes, arm.failures))
            return (low, high)
        except (ImportError, Exception):
            # Fallback without scipy
            mean = arm.successes / (arm.successes + arm.failures)
            std = math.sqrt((arm.successes * arm.failures) /
                           ((arm.successes + arm.failures) ** 2 * (arm.successes + arm.failures + 1)))
            return (max(0.0, mean - 2 * std), min(1.0, mean + 2 * std))


# =============================================================================
# SOTA: Adaptive Parameter Control
# =============================================================================


class AdaptiveParameterControl:
    """Self-adaptive parameter control using success history.

    Inspired by jDE (self-adaptive DE) and SHADE.
    """

    def __init__(
        self,
        param_names: List[str],
        initial_values: Dict[str, float],
        bounds: Dict[str, Tuple[float, float]],
        memory_size: int = 50,
    ):
        self.param_names = param_names
        self.params = initial_values.copy()
        self.bounds = bounds
        self.memory_size = memory_size

        # Success history for each parameter
        self.success_memory: Dict[str, List[float]] = {p: [] for p in param_names}
        self.success_fitness_deltas: Dict[str, List[float]] = {p: [] for p in param_names}

    def get_params(self) -> Dict[str, float]:
        """Get current adapted parameters."""
        return self.params.copy()

    def sample_params(self, exploration_rate: float = 0.1) -> Dict[str, float]:
        """Sample parameters with exploration."""
        sampled = {}

        for name in self.param_names:
            low, high = self.bounds[name]

            if random.random() < exploration_rate:
                # Random exploration
                sampled[name] = random.uniform(low, high)
            elif self.success_memory[name]:
                # Sample from successful values (Lehmer mean weighted by fitness delta)
                weights = [max(0.01, d) for d in self.success_fitness_deltas[name]]
                total_weight = sum(weights)

                # Weighted Lehmer mean (biases toward larger successful values)
                numerator = sum(w * v ** 2 for w, v in zip(weights, self.success_memory[name]))
                denominator = sum(w * v for w, v in zip(weights, self.success_memory[name]))

                if denominator > 0:
                    mean = numerator / denominator
                else:
                    mean = self.params[name]

                # Add Cauchy noise
                scale = 0.1 * (high - low)
                sampled[name] = np.clip(mean + scale * np.random.standard_cauchy(), low, high)
            else:
                sampled[name] = self.params[name]

        return sampled

    def record_success(
        self,
        params: Dict[str, float],
        fitness_delta: float,
    ) -> None:
        """Record successful parameter values."""
        for name in self.param_names:
            if name in params:
                self.success_memory[name].append(params[name])
                self.success_fitness_deltas[name].append(fitness_delta)

                # Maintain memory size
                if len(self.success_memory[name]) > self.memory_size:
                    self.success_memory[name].pop(0)
                    self.success_fitness_deltas[name].pop(0)

    def record_failure(self, params: Dict[str, float]) -> None:
        """Record failed parameter values (just for tracking)."""
        pass  # Could add failure memory for more sophisticated adaptation

    def update_base_params(self) -> None:
        """Update base parameters from success history."""
        for name in self.param_names:
            if self.success_memory[name]:
                low, high = self.bounds[name]
                # Use weighted mean of successful values
                weights = [max(0.01, d) for d in self.success_fitness_deltas[name]]
                total = sum(weights)
                if total > 0:
                    new_val = sum(w * v for w, v in zip(weights, self.success_memory[name])) / total
                    self.params[name] = np.clip(new_val, low, high)


# =============================================================================
# SOTA: Neural Fitness Predictor
# =============================================================================


class FitnessSurrogate:
    """Neural network surrogate model for fitness prediction.

    Uses simple MLP implemented in numpy (no deep learning deps).
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: List[int] = [128, 64],
        learning_rate: float = 0.01,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.lr = learning_rate

        # Initialize weights with Xavier initialization
        dims = [input_dim] + hidden_dims + [1]
        self.weights = []
        self.biases = []

        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / (dims[i] + dims[i+1]))
            self.weights.append(np.random.randn(dims[i], dims[i+1]) * scale)
            self.biases.append(np.zeros(dims[i+1]))

        # Training data buffer
        self.X_buffer: List[np.ndarray] = []
        self.y_buffer: List[float] = []
        self.buffer_size = 1000

        # Prediction uncertainty
        self.prediction_variance: Optional[float] = None

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _relu_grad(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    def _forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Forward pass, return output and activations."""
        activations = [x]
        current = x

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = current @ w + b
            if i < len(self.weights) - 1:
                current = self._relu(z)
            else:
                current = z  # Linear output
            activations.append(current)

        return current, activations

    def predict(self, embedding: np.ndarray) -> Tuple[float, float]:
        """Predict fitness with uncertainty estimate.

        Returns (prediction, uncertainty).
        """
        output, _ = self._forward(embedding.reshape(1, -1))
        prediction = float(output[0, 0])

        # Estimate uncertainty from buffer variance
        uncertainty = self.prediction_variance or 0.1

        return prediction, uncertainty

    def add_training_data(self, embedding: np.ndarray, fitness: float) -> None:
        """Add training data to buffer."""
        self.X_buffer.append(embedding)
        self.y_buffer.append(fitness)

        if len(self.X_buffer) > self.buffer_size:
            self.X_buffer.pop(0)
            self.y_buffer.pop(0)

    def train_step(self, batch_size: int = 32) -> float:
        """Run one training step, return loss."""
        if len(self.X_buffer) < batch_size:
            return 0.0

        # Sample batch
        indices = random.sample(range(len(self.X_buffer)), batch_size)
        X = np.array([self.X_buffer[i] for i in indices])
        y = np.array([self.y_buffer[i] for i in indices]).reshape(-1, 1)

        # Forward pass
        output, activations = self._forward(X)

        # Loss (MSE)
        loss = np.mean((output - y) ** 2)

        # Backward pass
        grad = 2 * (output - y) / batch_size

        for i in range(len(self.weights) - 1, -1, -1):
            # Gradient for weights and biases
            dw = activations[i].T @ grad
            db = np.sum(grad, axis=0)

            # Gradient for previous layer
            if i > 0:
                grad = (grad @ self.weights[i].T) * self._relu_grad(activations[i])

            # Update weights
            self.weights[i] -= self.lr * dw
            self.biases[i] -= self.lr * db

        # Update prediction variance
        predictions = self._forward(np.array(self.X_buffer))[0]
        self.prediction_variance = float(np.var(predictions - np.array(self.y_buffer).reshape(-1, 1)))

        return float(loss)


# =============================================================================
# SOTA: Persistent Storage
# =============================================================================


class PersistentStore:
    """SQLite-based persistent storage for evolution state."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or ":memory:"
        self._conn: sqlite3.Connection = sqlite3.connect(self.db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS prompts (
                hash TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                parent_hash TEXT,
                timestamp REAL,
                author TEXT,
                reason TEXT,
                modification_type TEXT,
                merkle_root TEXT,
                depth INTEGER,
                fitness_score REAL DEFAULT 0.0,
                evaluation_count INTEGER DEFAULT 0,
                fork_name TEXT,
                embedding BLOB
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS forks (
                name TEXT PRIMARY KEY,
                head_hash TEXT,
                created_at REAL,
                created_from TEXT,
                description TEXT,
                is_active INTEGER DEFAULT 1,
                merged_into TEXT
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_hash TEXT,
                objective TEXT,
                score REAL,
                timestamp REAL,
                context TEXT
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_prompts_parent ON prompts(parent_hash)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_prompts_fork ON prompts(fork_name)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_evals_hash ON evaluations(prompt_hash)")
        self._conn.commit()

    def store_prompt(
        self,
        hash: str,
        content: str,
        parent_hash: Optional[str],
        author: str,
        reason: str,
        modification_type: str,
        merkle_root: str,
        depth: int,
        fork_name: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
    ) -> None:
        """Store a prompt node."""
        embedding_blob = pickle.dumps(embedding) if embedding is not None else None
        self._conn.execute("""
            INSERT OR REPLACE INTO prompts
            (hash, content, parent_hash, timestamp, author, reason,
             modification_type, merkle_root, depth, fork_name, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (hash, content, parent_hash, time.time(), author, reason,
              modification_type, merkle_root, depth, fork_name, embedding_blob))
        self._conn.commit()

    def get_prompt(self, hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve a prompt by hash."""
        cursor = self._conn.execute(
            "SELECT * FROM prompts WHERE hash = ?", (hash,)
        )
        row = cursor.fetchone()
        if row:
            cols = [d[0] for d in cursor.description]
            result = dict(zip(cols, row))
            if result.get("embedding"):
                result["embedding"] = pickle.loads(result["embedding"])
            return result
        return None

    def update_fitness(
        self,
        hash: str,
        fitness_score: float,
        evaluation_count: int,
    ) -> None:
        """Update fitness score for a prompt."""
        self._conn.execute("""
            UPDATE prompts SET fitness_score = ?, evaluation_count = ?
            WHERE hash = ?
        """, (fitness_score, evaluation_count, hash))
        self._conn.commit()

    def store_fork(
        self,
        name: str,
        head_hash: str,
        created_from: str,
        description: str,
    ) -> None:
        """Store a fork."""
        self._conn.execute("""
            INSERT OR REPLACE INTO forks
            (name, head_hash, created_at, created_from, description)
            VALUES (?, ?, ?, ?, ?)
        """, (name, head_hash, time.time(), created_from, description))
        self._conn.commit()

    def get_fork(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a fork by name."""
        cursor = self._conn.execute(
            "SELECT * FROM forks WHERE name = ?", (name,)
        )
        row = cursor.fetchone()
        if row:
            cols = [d[0] for d in cursor.description]
            return dict(zip(cols, row))
        return None

    def get_all_forks(self) -> List[Dict[str, Any]]:
        """Get all forks."""
        cursor = self._conn.execute("SELECT * FROM forks")
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def store_evaluation(
        self,
        prompt_hash: str,
        objective: str,
        score: float,
        context: Optional[Dict] = None,
    ) -> None:
        """Store an evaluation result."""
        self._conn.execute("""
            INSERT INTO evaluations (prompt_hash, objective, score, timestamp, context)
            VALUES (?, ?, ?, ?, ?)
        """, (prompt_hash, objective, score, time.time(), json.dumps(context or {})))
        self._conn.commit()

    def get_evaluations(self, prompt_hash: str) -> List[Dict[str, Any]]:
        """Get all evaluations for a prompt."""
        cursor = self._conn.execute(
            "SELECT * FROM evaluations WHERE prompt_hash = ?", (prompt_hash,)
        )
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()


# =============================================================================
# SOTA: Integrated Evolution Engine
# =============================================================================


class SOTAEvolutionEngine:
    """State-of-the-art prompt evolution engine.

    Integrates all SOTA components:
    - Embedding-based semantic similarity
    - NSGA-II multi-objective optimization
    - MAP-Elites quality-diversity
    - Thompson Sampling exploration
    - Adaptive parameter control
    - Neural fitness prediction
    - Persistent storage
    """

    def __init__(
        self,
        objectives: List[str],
        db_path: Optional[str] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ):
        self.objectives = objectives

        # Core components
        self.store = PersistentStore(db_path)
        self.bloom = BloomFilter(expected_items=10000)
        self.similarity = SemanticSimilarity(embedding_provider)

        # Optimization components
        self.nsga = NSGAII(objectives)
        self.map_elites = MAPElites(
            behavior_dims=3,  # coherence, efficiency, novelty
            bins_per_dim=10,
        )
        self.thompson = ThompsonSampling()

        # Adaptive control
        self.adaptive = AdaptiveParameterControl(
            param_names=["crossover_rate", "mutation_rate", "exploration"],
            initial_values={"crossover_rate": 0.9, "mutation_rate": 0.1, "exploration": 0.1},
            bounds={"crossover_rate": (0.5, 1.0), "mutation_rate": (0.01, 0.5), "exploration": (0.01, 0.3)},
        )

        # Surrogate model
        self.surrogate = FitnessSurrogate()

        # State
        self._current_fork = "main"
        self._current_hash: Optional[str] = None
        self._generation = 0

    async def initialize(self, genesis_prompt: str) -> str:
        """Initialize with genesis prompt."""
        content_hash = hashlib.sha256(genesis_prompt.encode()).hexdigest()

        # Get embedding
        embedding = await self.similarity._get_embedding(genesis_prompt)

        # Store
        self.store.store_prompt(
            hash=content_hash,
            content=genesis_prompt,
            parent_hash=None,
            author="system",
            reason="Genesis",
            modification_type="genesis",
            merkle_root=content_hash,
            depth=0,
            fork_name="main",
            embedding=embedding,
        )

        # Create main fork
        self.store.store_fork(
            name="main",
            head_hash=content_hash,
            created_from=content_hash,
            description="Main evolution branch",
        )

        # Add to bloom filter
        self.bloom.add(content_hash)

        # Initialize Thompson Sampling arm
        self.thompson.add_arm(content_hash)

        self._current_hash = content_hash

        return content_hash

    async def evolve(
        self,
        evaluate_fn: Callable[[str], Coroutine[Any, Any, Dict[str, float]]],
        mutate_fn: Callable[[str], Coroutine[Any, Any, str]],
        num_generations: int = 1,
    ) -> Dict[str, Any]:
        """Run evolution for specified generations."""
        results = {
            "generations": [],
            "pareto_front": [],
            "best_fitness": 0.0,
            "qd_score": 0.0,
            "coverage": 0.0,
        }

        for gen in range(num_generations):
            self._generation += 1
            gen_results = await self._evolve_generation(evaluate_fn, mutate_fn)
            results["generations"].append(gen_results)

            # Train surrogate
            if len(self.surrogate.X_buffer) >= 32:
                loss = self.surrogate.train_step()
                gen_results["surrogate_loss"] = loss

        # Final statistics
        if self.nsga.population:
            results["pareto_front"] = [
                {"genome": ind.genome, "objectives": ind.objectives}
                for ind in self.nsga.population if ind.rank == 0
            ]
            results["best_fitness"] = max(
                sum(ind.objectives.values()) / len(ind.objectives)
                for ind in self.nsga.population
            )

        results["qd_score"] = self.map_elites.qd_score
        results["coverage"] = self.map_elites.coverage

        return results

    async def _evolve_generation(
        self,
        evaluate_fn: Callable[[str], Coroutine[Any, Any, Dict[str, float]]],
        mutate_fn: Callable[[str], Coroutine[Any, Any, str]],
    ) -> Dict[str, Any]:
        """Run one generation of evolution."""
        # Get adaptive parameters
        params = self.adaptive.sample_params()

        # Select parent using Thompson Sampling
        parent_hash_opt = self.thompson.select()
        if parent_hash_opt:
            parent_hash = parent_hash_opt
        elif self._current_hash:
            parent_hash = self._current_hash
        else:
            return {"error": "No parent hash available"}

        parent_data = self.store.get_prompt(parent_hash)
        if not parent_data:
            return {"error": "No parent found"}

        parent_content = parent_data["content"]

        # Generate mutations
        num_offspring = 10
        offspring = []

        for i in range(num_offspring):
            # Apply mutation
            mutated = await mutate_fn(parent_content)
            mutated_hash = hashlib.sha256(mutated.encode()).hexdigest()

            # Check novelty via bloom filter
            if mutated_hash in self.bloom:
                continue  # Skip duplicates

            # Evaluate
            fitness = await evaluate_fn(mutated)

            # Get embedding
            embedding = await self.similarity._get_embedding(mutated)

            # Store
            self.store.store_prompt(
                hash=mutated_hash,
                content=mutated,
                parent_hash=parent_hash,
                author="evolution",
                reason=f"Generation {self._generation}",
                modification_type="evolve",
                merkle_root=mutated_hash,  # Simplified
                depth=parent_data["depth"] + 1,
                fork_name=self._current_fork,
                embedding=embedding,
            )

            self.bloom.add(mutated_hash)

            # Add to Thompson Sampling
            self.thompson.add_arm(mutated_hash)
            avg_fitness = sum(fitness.values()) / len(fitness)
            self.thompson.update(mutated_hash, avg_fitness)

            # Add to NSGA-II
            self.nsga.population.append(Individual(
                genome=mutated_hash,
                objectives=fitness,
            ))

            # Compute behavior for MAP-Elites
            behavior = [
                fitness.get("coherence", 0.5),
                fitness.get("efficiency", 0.5),
                float(await self._compute_novelty(embedding)),
            ]

            # Add to MAP-Elites
            self.map_elites.add(mutated_hash, avg_fitness, behavior)

            # Add to surrogate training
            self.surrogate.add_training_data(embedding, avg_fitness)

            offspring.append({
                "hash": mutated_hash,
                "fitness": fitness,
                "avg_fitness": avg_fitness,
            })

            # Update adaptive parameters
            if avg_fitness > self.thompson.get_expected_value(parent_hash):
                self.adaptive.record_success(params, avg_fitness - self.thompson.get_expected_value(parent_hash))
            else:
                self.adaptive.record_failure(params)

        # Update base parameters periodically
        if self._generation % 10 == 0:
            self.adaptive.update_base_params()

        # Run NSGA-II selection
        if len(self.nsga.population) > self.nsga.pop_size:
            def dummy_crossover(g1, g2):
                return g1, g2

            def dummy_mutate(g):
                return g

            fronts = self.nsga._fast_non_dominated_sort(self.nsga.population)

            # Keep only Pareto optimal and near-optimal
            new_pop = []
            for front in fronts:
                self.nsga._crowding_distance(front)
                new_pop.extend(front)
                if len(new_pop) >= self.nsga.pop_size:
                    break

            self.nsga.population = new_pop[:self.nsga.pop_size]

        return {
            "generation": self._generation,
            "offspring_count": len(offspring),
            "best_offspring": max(offspring, key=lambda x: x["avg_fitness"]) if offspring else None,
            "pareto_size": sum(1 for ind in self.nsga.population if ind.rank == 0),
            "map_coverage": self.map_elites.coverage,
            "qd_score": self.map_elites.qd_score,
        }

    async def _compute_novelty(self, embedding: np.ndarray) -> float:
        """Compute novelty of an embedding relative to archive."""
        if not self.map_elites.archive:
            return 1.0

        # Get k-nearest neighbors
        k = min(15, len(self.map_elites.archive))
        distances = []

        for cell in self.map_elites.archive.values():
            cell_data = self.store.get_prompt(cell.elite)
            if cell_data and cell_data.get("embedding") is not None:
                cell_emb = cell_data["embedding"]
                dist = 1 - np.dot(embedding, cell_emb)  # Cosine distance
                distances.append(dist)

        if not distances:
            return 1.0

        distances.sort()
        return float(np.mean(distances[:k]))

    def get_best_prompt(self) -> Optional[str]:
        """Get the best prompt from the Pareto front."""
        best_cell = self.map_elites.get_best()
        if best_cell:
            data = self.store.get_prompt(best_cell.elite)
            if data:
                return data["content"]
        return None

    def get_diverse_prompts(self, count: int = 5) -> List[str]:
        """Get diverse prompts from MAP-Elites archive."""
        prompts = []
        cells = list(self.map_elites.archive.values())
        random.shuffle(cells)

        for cell in cells[:count]:
            data = self.store.get_prompt(cell.elite)
            if data:
                prompts.append(data["content"])

        return prompts

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "generation": self._generation,
            "total_prompts": len(self.bloom),
            "pareto_front_size": sum(1 for ind in self.nsga.population if ind.rank == 0),
            "map_elites_coverage": self.map_elites.coverage,
            "map_elites_qd_score": self.map_elites.qd_score,
            "thompson_arms": len(self.thompson.arms),
            "surrogate_buffer_size": len(self.surrogate.X_buffer),
            "adaptive_params": self.adaptive.get_params(),
        }
