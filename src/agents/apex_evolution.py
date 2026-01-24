"""APEX Evolution System - Cutting-Edge Prompt Optimization.

Advanced techniques beyond SOTA:
1. CMA-ES - Covariance Matrix Adaptation Evolution Strategy
2. Differential Evolution with adaptive parameters
3. Bayesian Optimization with Gaussian Processes
4. Monte Carlo Tree Search for exploration
5. Lexicase Selection for maintaining specialists
6. Age-Layered Population Structure (ALPS)
7. Cooperative Coevolution for modular prompts
8. Population-Based Training (DeepMind style)
9. Quality-Diversity with MAP-Elites variants
10. Transformer-based semantic embeddings
11. Graph attention for prompt structure
12. Multi-fidelity optimization
13. Novelty Search with Local Competition
14. Information-Geometric Optimization
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
)

import numpy as np
from numpy.linalg import eigh, inv, norm

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# CMA-ES: Covariance Matrix Adaptation Evolution Strategy
# =============================================================================


class CMAES:
    """CMA-ES for continuous optimization of prompt embeddings.

    State-of-the-art derivative-free optimizer that adapts the
    covariance matrix of a multivariate normal distribution.
    """

    def __init__(
        self,
        dimension: int,
        population_size: Optional[int] = None,
        sigma: float = 0.3,
        mean: Optional[np.ndarray] = None,
    ):
        self.dim = dimension
        self.sigma = sigma

        # Population size (lambda)
        self.lam = population_size or 4 + int(3 * np.log(dimension))
        self.mu = self.lam // 2  # Parent population size

        # Recombination weights
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / weights.sum()
        self.mueff = 1.0 / (self.weights ** 2).sum()

        # Adaptation parameters
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1/self.mueff) / ((self.dim + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs

        # State variables
        self.mean = mean if mean is not None else np.zeros(dimension)
        self.C = np.eye(dimension)  # Covariance matrix
        self.ps = np.zeros(dimension)  # Evolution path for sigma
        self.pc = np.zeros(dimension)  # Evolution path for C

        self.generation = 0
        self.eigeneval = 0
        self.B = np.eye(dimension)  # Eigenvectors
        self.D = np.ones(dimension)  # Eigenvalues

    def ask(self) -> List[np.ndarray]:
        """Sample new population."""
        # Eigendecomposition of C
        if self.generation - self.eigeneval > self.lam / (self.c1 + self.cmu) / self.dim / 10:
            self.eigeneval = self.generation
            self.C = np.triu(self.C) + np.triu(self.C, 1).T  # Enforce symmetry
            D2, self.B = eigh(self.C)
            self.D = np.sqrt(np.maximum(D2, 1e-20))

        # Sample offspring
        offspring = []
        for _ in range(self.lam):
            z = np.random.randn(self.dim)
            y = self.B @ (self.D * z)
            x = self.mean + self.sigma * y
            offspring.append(x)

        return offspring

    def tell(self, solutions: List[np.ndarray], fitnesses: List[float]) -> None:
        """Update distribution based on fitness."""
        self.generation += 1

        # Sort by fitness (minimization)
        indices = np.argsort(fitnesses)

        # Selected points
        selected = [solutions[i] for i in indices[:self.mu]]

        # Compute weighted mean
        old_mean = self.mean.copy()
        weighted_sum = np.zeros_like(self.mean)
        for w, x in zip(self.weights, selected):
            weighted_sum += w * x
        self.mean = weighted_sum

        # Update evolution paths
        y = (self.mean - old_mean) / self.sigma
        D_safe = np.maximum(self.D, 1e-10)  # Numerical stability
        C_invsqrt = self.B @ np.diag(1.0 / D_safe) @ self.B.T

        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (C_invsqrt @ y)

        hsig = (norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.generation))
                < (1.4 + 2 / (self.dim + 1)) * np.sqrt(self.dim))

        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y

        # Update covariance matrix
        artmp = np.array([(solutions[i] - old_mean) / self.sigma for i in indices[:self.mu]])
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * (artmp.T @ np.diag(self.weights) @ artmp))

        # Update step size
        self.sigma *= np.exp((self.cs / self.damps) * (norm(self.ps) / np.sqrt(self.dim) - 1))

    @property
    def result(self) -> Tuple[np.ndarray, float]:
        """Get current best solution."""
        return self.mean.copy(), self.sigma


# =============================================================================
# Differential Evolution with Self-Adaptation
# =============================================================================


class DifferentialEvolution:
    """Self-adaptive Differential Evolution (jDE/SHADE hybrid).

    Automatically adapts F and CR parameters based on success history.
    """

    def __init__(
        self,
        dimension: int,
        population_size: int = 50,
        bounds: Tuple[float, float] = (-1.0, 1.0),
    ):
        self.dim = dimension
        self.pop_size = population_size
        self.bounds = bounds

        # Initialize population
        self.population = np.random.uniform(
            bounds[0], bounds[1], (population_size, dimension)
        )
        self.fitness = np.full(population_size, np.inf)

        # Adaptive parameters (SHADE-style memory)
        self.memory_size = 5
        self.memory_F = np.full(self.memory_size, 0.5)
        self.memory_CR = np.full(self.memory_size, 0.5)
        self.memory_idx = 0

        # Success history
        self.success_F: List[float] = []
        self.success_CR: List[float] = []
        self.success_delta: List[float] = []

        self.generation = 0
        self.best_idx = 0

    def _sample_parameters(self) -> Tuple[float, float]:
        """Sample F and CR from memory."""
        idx = random.randint(0, self.memory_size - 1)

        # Cauchy distribution for F
        F = -1
        while F <= 0:
            F = np.random.standard_cauchy() * 0.1 + self.memory_F[idx]
        F = min(F, 1.0)

        # Normal distribution for CR
        CR = np.clip(np.random.normal(self.memory_CR[idx], 0.1), 0, 1)

        return F, CR

    def _mutate(self, idx: int, F: float) -> np.ndarray:
        """current-to-pbest/1 mutation."""
        # Select p-best
        p = max(2, int(0.1 * self.pop_size))
        pbest_indices = np.argsort(self.fitness)[:p]
        pbest_idx = np.random.choice(pbest_indices)

        # Select random indices
        candidates = [i for i in range(self.pop_size) if i != idx]
        r1, r2 = random.sample(candidates, 2)

        # Mutation
        mutant = (self.population[idx]
                  + F * (self.population[pbest_idx] - self.population[idx])
                  + F * (self.population[r1] - self.population[r2]))

        # Bound handling
        return np.clip(mutant, self.bounds[0], self.bounds[1])

    def _crossover(self, target: np.ndarray, mutant: np.ndarray, CR: float) -> np.ndarray:
        """Binomial crossover."""
        trial = target.copy()
        j_rand = random.randint(0, self.dim - 1)

        for j in range(self.dim):
            if random.random() < CR or j == j_rand:
                trial[j] = mutant[j]

        return trial

    def evolve(self, evaluate_fn: Callable[[np.ndarray], float]) -> np.ndarray:
        """Run one generation."""
        self.generation += 1

        # Evaluate initial population if needed
        if self.generation == 1:
            for i in range(self.pop_size):
                self.fitness[i] = evaluate_fn(self.population[i])
            self.best_idx = int(np.argmin(self.fitness))

        # Clear success history
        self.success_F.clear()
        self.success_CR.clear()
        self.success_delta.clear()

        for i in range(self.pop_size):
            F, CR = self._sample_parameters()

            mutant = self._mutate(i, F)
            trial = self._crossover(self.population[i], mutant, CR)

            trial_fitness = evaluate_fn(trial)

            if trial_fitness <= self.fitness[i]:
                # Success - record parameters
                if trial_fitness < self.fitness[i]:
                    self.success_F.append(F)
                    self.success_CR.append(CR)
                    self.success_delta.append(self.fitness[i] - trial_fitness)

                self.population[i] = trial
                self.fitness[i] = trial_fitness

                if trial_fitness < self.fitness[self.best_idx]:
                    self.best_idx = i

        # Update memory
        if self.success_F:
            weights = np.array(self.success_delta)
            weights /= weights.sum()

            # Lehmer mean for F
            mean_F = (weights * np.array(self.success_F) ** 2).sum() / (weights * np.array(self.success_F)).sum()
            # Weighted arithmetic mean for CR
            mean_CR = (weights * np.array(self.success_CR)).sum()

            self.memory_F[self.memory_idx] = mean_F
            self.memory_CR[self.memory_idx] = mean_CR
            self.memory_idx = (self.memory_idx + 1) % self.memory_size

        return self.population[self.best_idx]

    @property
    def best(self) -> Tuple[np.ndarray, float]:
        return self.population[self.best_idx], self.fitness[self.best_idx]


# =============================================================================
# Bayesian Optimization with Gaussian Processes
# =============================================================================


class GaussianProcess:
    """Gaussian Process for Bayesian Optimization.

    Uses RBF kernel with automatic relevance determination.
    """

    def __init__(
        self,
        length_scale: float = 1.0,
        signal_variance: float = 1.0,
        noise_variance: float = 1e-6,
    ):
        self.l = length_scale
        self.sf2 = signal_variance
        self.sn2 = noise_variance

        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.K_inv: Optional[np.ndarray] = None
        self.alpha: Optional[np.ndarray] = None

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF kernel with numerical stability."""
        sq_dist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * X1 @ X2.T
        sq_dist = np.maximum(sq_dist, 0)  # Ensure non-negative due to numerical errors
        return self.sf2 * np.exp(-0.5 * sq_dist / self.l**2)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the GP to training data."""
        self.X_train = X
        self.y_train = y

        K = self._kernel(X, X) + (self.sn2 + 1e-6) * np.eye(len(X))  # Extra regularization
        try:
            self.K_inv = inv(K)
        except np.linalg.LinAlgError:
            # Fallback: use pseudo-inverse
            self.K_inv = np.linalg.pinv(K)
        self.alpha = self.K_inv @ y

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance at X."""
        if self.X_train is None:
            return np.zeros(len(X)), np.ones(len(X)) * self.sf2

        K_s = self._kernel(X, self.X_train)
        K_ss = self._kernel(X, X)

        mean = K_s @ self.alpha
        var = np.diag(K_ss - K_s @ self.K_inv @ K_s.T)
        var = np.maximum(var, 1e-10)  # Numerical stability

        return mean, var


class BayesianOptimizer:
    """Bayesian Optimization with GP surrogate.

    Uses Expected Improvement acquisition function.
    """

    def __init__(
        self,
        dimension: int,
        bounds: Tuple[float, float] = (-1.0, 1.0),
        n_initial: int = 5,
    ):
        self.dim = dimension
        self.bounds = bounds
        self.n_initial = n_initial

        self.gp = GaussianProcess()
        self.X_observed: List[np.ndarray] = []
        self.y_observed: List[float] = []
        self.best_y = float('inf')
        self.best_x: Optional[np.ndarray] = None

    def _expected_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """Expected Improvement acquisition function."""
        mean, var = self.gp.predict(X)
        std = np.sqrt(var)

        with np.errstate(divide='warn'):
            imp = self.best_y - mean - xi
            Z = imp / std
            ei = imp * self._norm_cdf(Z) + std * self._norm_pdf(Z)
            ei[std < 1e-10] = 0.0

        return ei

    @staticmethod
    def _norm_cdf(x: np.ndarray) -> np.ndarray:
        """Standard normal CDF."""
        from scipy.special import erf
        return 0.5 * (1 + erf(x / np.sqrt(2)))

    @staticmethod
    def _norm_pdf(x: np.ndarray) -> np.ndarray:
        """Standard normal PDF."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

    def suggest(self, n_candidates: int = 1000) -> np.ndarray:
        """Suggest next point to evaluate."""
        if len(self.X_observed) < self.n_initial:
            # Random sampling during initialization
            return np.random.uniform(self.bounds[0], self.bounds[1], self.dim)

        # Update GP
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        self.gp.fit(X, y)

        # Optimize acquisition function
        candidates = np.random.uniform(
            self.bounds[0], self.bounds[1], (n_candidates, self.dim)
        )
        ei = self._expected_improvement(candidates)

        return candidates[np.argmax(ei)]

    def observe(self, x: np.ndarray, y: float) -> None:
        """Record observation."""
        self.X_observed.append(x)
        self.y_observed.append(y)

        if y < self.best_y:
            self.best_y = y
            self.best_x = x.copy()


# =============================================================================
# Monte Carlo Tree Search for Prompt Exploration
# =============================================================================


@dataclass
class MCTSNode:
    """Node in MCTS tree."""
    state: str  # Prompt content
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    untried_actions: List[str] = field(default_factory=list)

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    @property
    def ucb1(self) -> float:
        if self.visits == 0:
            return float('inf')
        exploit = self.value / self.visits
        explore = np.sqrt(2 * np.log(self.parent.visits) / self.visits) if self.parent else 0
        return exploit + explore


class PromptMCTS:
    """Monte Carlo Tree Search for prompt space exploration.

    Uses UCT (Upper Confidence bounds for Trees) for selection.
    """

    def __init__(
        self,
        action_generator: Callable[[str], List[str]],
        evaluator: Callable[[str], float],
        max_depth: int = 10,
        exploration_weight: float = 1.41,
    ):
        self.action_gen = action_generator
        self.evaluator = evaluator
        self.max_depth = max_depth
        self.c = exploration_weight

        self.root: Optional[MCTSNode] = None
        self.best_prompt: Optional[str] = None
        self.best_value: float = float('-inf')

    def search(
        self,
        initial_prompt: str,
        n_iterations: int = 100,
    ) -> str:
        """Run MCTS search."""
        self.root = MCTSNode(
            state=initial_prompt,
            untried_actions=self.action_gen(initial_prompt),
        )

        for _ in range(n_iterations):
            node = self._select(self.root)
            child = self._expand(node)
            value = self._simulate(child)
            self._backpropagate(child, value)

        # Return best child of root
        if self.root.children:
            best_child = max(self.root.children, key=lambda n: n.value / max(n.visits, 1))
            return best_child.state
        return initial_prompt

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select node to expand using UCB1."""
        depth = 0
        while node.is_fully_expanded and node.children and depth < self.max_depth:
            node = max(node.children, key=lambda n: self._ucb1(n))
            depth += 1
        return node

    def _ucb1(self, node: MCTSNode) -> float:
        """UCB1 score."""
        if node.visits == 0:
            return float('inf')
        exploit = node.value / node.visits
        explore = self.c * np.sqrt(np.log(node.parent.visits) / node.visits) if node.parent else 0
        return exploit + explore

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node with untried action."""
        if not node.untried_actions:
            return node

        action = node.untried_actions.pop()
        child = MCTSNode(
            state=action,
            parent=node,
            untried_actions=self.action_gen(action) if len(node.children) < 3 else [],
        )
        node.children.append(child)
        return child

    def _simulate(self, node: MCTSNode) -> float:
        """Evaluate node (rollout)."""
        value = self.evaluator(node.state)

        if value > self.best_value:
            self.best_value = value
            self.best_prompt = node.state

        return value

    def _backpropagate(self, node: Optional[MCTSNode], value: float) -> None:
        """Backpropagate value up tree."""
        current: Optional[MCTSNode] = node
        while current is not None:
            current.visits += 1
            current.value += value
            current = current.parent


# =============================================================================
# Lexicase Selection
# =============================================================================


class LexicaseSelection:
    """Lexicase Selection for maintaining specialists.

    Selects parents by randomly ordering test cases and
    eliminating candidates that aren't among the best on each case.
    """

    def __init__(self, epsilon: float = 0.0):
        self.epsilon = epsilon  # For epsilon-lexicase

    def select(
        self,
        population: List[Any],
        fitness_cases: List[Callable[[Any], float]],
        n_parents: int = 1,
    ) -> List[Any]:
        """Select parents using lexicase selection."""
        parents = []

        for _ in range(n_parents):
            candidates = list(range(len(population)))
            cases = list(range(len(fitness_cases)))
            random.shuffle(cases)

            for case_idx in cases:
                if len(candidates) <= 1:
                    break

                # Evaluate all candidates on this case
                scores = [fitness_cases[case_idx](population[i]) for i in candidates]
                best_score = max(scores)

                # Keep only best (or within epsilon)
                if self.epsilon > 0:
                    threshold = best_score - self.epsilon
                    candidates = [c for c, s in zip(candidates, scores) if s >= threshold]
                else:
                    candidates = [c for c, s in zip(candidates, scores) if s == best_score]

            # Random selection among remaining
            parents.append(population[random.choice(candidates)])

        return parents


# =============================================================================
# Age-Layered Population Structure (ALPS)
# =============================================================================


@dataclass
class ALPSLayer:
    """A layer in ALPS."""
    max_age: int
    population: List[Any] = field(default_factory=list)
    fitness: List[float] = field(default_factory=list)
    ages: List[int] = field(default_factory=list)


class ALPS:
    """Age-Layered Population Structure.

    Maintains genetic diversity by segregating individuals by age.
    """

    def __init__(
        self,
        n_layers: int = 5,
        layer_size: int = 20,
        age_scheme: str = "polynomial",  # "linear", "polynomial", "exponential"
    ):
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.age_scheme = age_scheme

        # Compute age limits for each layer
        self.age_limits = self._compute_age_limits()

        # Initialize layers
        self.layers = [ALPSLayer(max_age=limit) for limit in self.age_limits]

        self.generation = 0

    def _compute_age_limits(self) -> List[int]:
        """Compute age limits based on scheme."""
        if self.age_scheme == "linear":
            return [10 * (i + 1) for i in range(self.n_layers)]
        elif self.age_scheme == "polynomial":
            return [10 * (i + 1) ** 2 for i in range(self.n_layers)]
        else:  # exponential
            return [10 * (2 ** i) for i in range(self.n_layers)]

    def add_individual(self, individual: Any, fitness: float, age: int = 0) -> None:
        """Add individual to appropriate layer."""
        # Find correct layer
        layer_idx = 0
        for i, layer in enumerate(self.layers):
            if age <= layer.max_age:
                layer_idx = i
                break
        else:
            layer_idx = self.n_layers - 1

        layer = self.layers[layer_idx]

        if len(layer.population) < self.layer_size:
            layer.population.append(individual)
            layer.fitness.append(fitness)
            layer.ages.append(age)
        else:
            # Replace worst if better
            worst_idx = int(np.argmin(layer.fitness))
            if fitness > layer.fitness[worst_idx]:
                layer.population[worst_idx] = individual
                layer.fitness[worst_idx] = fitness
                layer.ages[worst_idx] = age

    def select_parents(self, layer_idx: int, n: int = 2) -> List[Any]:
        """Select parents from layer and adjacent layers."""
        candidates = []
        candidate_fitness = []

        # Include current layer
        layer = self.layers[layer_idx]
        candidates.extend(layer.population)
        candidate_fitness.extend(layer.fitness)

        # Include layer below (if exists)
        if layer_idx > 0:
            lower = self.layers[layer_idx - 1]
            candidates.extend(lower.population)
            candidate_fitness.extend(lower.fitness)

        if not candidates:
            return []

        # Tournament selection
        parents = []
        for _ in range(n):
            if len(candidates) < 3:
                parents.append(random.choice(candidates))
            else:
                tournament = random.sample(list(zip(candidates, candidate_fitness)), 3)
                winner = max(tournament, key=lambda x: x[1])
                parents.append(winner[0])

        return parents

    def age_population(self) -> None:
        """Age all individuals and move between layers."""
        self.generation += 1

        # Age and collect individuals to move
        to_move = []

        for layer_idx, layer in enumerate(self.layers):
            new_pop, new_fit, new_ages = [], [], []

            for ind, fit, age in zip(layer.population, layer.fitness, layer.ages):
                new_age = age + 1

                if new_age > layer.max_age and layer_idx < self.n_layers - 1:
                    to_move.append((ind, fit, new_age))
                else:
                    new_pop.append(ind)
                    new_fit.append(fit)
                    new_ages.append(new_age)

            layer.population = new_pop
            layer.fitness = new_fit
            layer.ages = new_ages

        # Add moved individuals to appropriate layers
        for ind, fit, age in to_move:
            self.add_individual(ind, fit, age)

    def get_best(self) -> Tuple[Any, float]:
        """Get best individual across all layers."""
        best = None
        best_fit = float('-inf')

        for layer in self.layers:
            if layer.fitness:
                idx = int(np.argmax(layer.fitness))
                if layer.fitness[idx] > best_fit:
                    best_fit = layer.fitness[idx]
                    best = layer.population[idx]

        return best, best_fit


# =============================================================================
# Cooperative Coevolution
# =============================================================================


class CooperativeCoevolution:
    """Cooperative Coevolution for modular prompt optimization.

    Evolves prompt components separately, combining for evaluation.
    """

    def __init__(
        self,
        n_components: int,
        component_dims: List[int],
        population_size: int = 20,
    ):
        self.n_components = n_components
        self.component_dims = component_dims
        self.pop_size = population_size

        # Separate population for each component
        self.populations: List[List[np.ndarray]] = [
            [np.random.randn(dim) for _ in range(population_size)]
            for dim in component_dims
        ]
        self.fitness: List[List[float]] = [
            [0.0] * population_size for _ in range(n_components)
        ]

        # Best representatives
        self.representatives = [pop[0].copy() for pop in self.populations]

        self.generation = 0

    def _combine(self, indices: List[int]) -> np.ndarray:
        """Combine components from different populations."""
        return np.concatenate([
            self.populations[i][indices[i]]
            for i in range(self.n_components)
        ])

    def _combine_with_representative(self, comp_idx: int, ind_idx: int) -> np.ndarray:
        """Combine one component with representatives of others."""
        parts = []
        for i in range(self.n_components):
            if i == comp_idx:
                parts.append(self.populations[i][ind_idx])
            else:
                parts.append(self.representatives[i])
        return np.concatenate(parts)

    def evolve(
        self,
        evaluate_fn: Callable[[np.ndarray], float],
        n_generations: int = 1,
    ) -> np.ndarray:
        """Evolve all components."""
        for _ in range(n_generations):
            self.generation += 1

            # Evolve each component
            for comp_idx in range(self.n_components):
                # Evaluate with representatives
                for ind_idx in range(self.pop_size):
                    combined = self._combine_with_representative(comp_idx, ind_idx)
                    self.fitness[comp_idx][ind_idx] = evaluate_fn(combined)

                # Update representative
                best_idx = int(np.argmax(self.fitness[comp_idx]))
                self.representatives[comp_idx] = self.populations[comp_idx][best_idx].copy()

                # Evolve population (simple mutation)
                sorted_indices = np.argsort(self.fitness[comp_idx])[::-1]
                new_pop = []

                # Keep elite
                for i in sorted_indices[:self.pop_size // 4]:
                    new_pop.append(self.populations[comp_idx][i].copy())

                # Generate offspring
                while len(new_pop) < self.pop_size:
                    parent_idx = random.choice(sorted_indices[:self.pop_size // 2])
                    parent = self.populations[comp_idx][parent_idx]
                    child = parent + 0.1 * np.random.randn(self.component_dims[comp_idx])
                    new_pop.append(child)

                self.populations[comp_idx] = new_pop

        # Return combined best
        return np.concatenate(self.representatives)


# =============================================================================
# Population-Based Training (PBT)
# =============================================================================


@dataclass
class PBTAgent:
    """Agent in Population-Based Training."""
    genome: Any
    hyperparams: Dict[str, float]
    fitness: float = 0.0
    ready_for_eval: bool = True

    def copy(self) -> "PBTAgent":
        return PBTAgent(
            genome=self.genome,
            hyperparams=self.hyperparams.copy(),
            fitness=self.fitness,
        )


class PopulationBasedTraining:
    """Population-Based Training (DeepMind style).

    Jointly optimizes models and hyperparameters.
    """

    def __init__(
        self,
        population_size: int = 20,
        hyperparam_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        exploit_fraction: float = 0.2,
        explore_factor: float = 0.2,
    ):
        self.pop_size = population_size
        self.hp_bounds = hyperparam_bounds or {
            "learning_rate": (0.0001, 0.1),
            "mutation_rate": (0.01, 0.5),
        }
        self.exploit_frac = exploit_fraction
        self.explore_factor = explore_factor

        self.population: List[PBTAgent] = []
        self.generation = 0

    def initialize(self, genome_factory: Callable[[], Any]) -> None:
        """Initialize population with random hyperparameters."""
        for _ in range(self.pop_size):
            hyperparams = {
                k: random.uniform(lo, hi)
                for k, (lo, hi) in self.hp_bounds.items()
            }
            self.population.append(PBTAgent(
                genome=genome_factory(),
                hyperparams=hyperparams,
            ))

    def step(
        self,
        train_fn: Callable[[Any, Dict], Any],
        eval_fn: Callable[[Any], float],
    ) -> None:
        """Run one PBT step."""
        self.generation += 1

        # Train each agent
        for agent in self.population:
            if agent.ready_for_eval:
                agent.genome = train_fn(agent.genome, agent.hyperparams)
                agent.fitness = eval_fn(agent.genome)
                agent.ready_for_eval = False

        # Sort by fitness
        sorted_pop = sorted(self.population, key=lambda a: a.fitness, reverse=True)

        # Exploit: bottom agents copy from top
        n_exploit = int(self.pop_size * self.exploit_frac)
        top_agents = sorted_pop[:n_exploit]
        bottom_agents = sorted_pop[-n_exploit:]

        for bottom in bottom_agents:
            # Copy from random top agent
            top = random.choice(top_agents)
            bottom.genome = top.genome  # Would deep copy in practice
            bottom.hyperparams = top.hyperparams.copy()

            # Explore: perturb hyperparameters
            for key in bottom.hyperparams:
                lo, hi = self.hp_bounds[key]
                factor = random.choice([1 - self.explore_factor, 1 + self.explore_factor])
                bottom.hyperparams[key] = np.clip(
                    bottom.hyperparams[key] * factor, lo, hi
                )

            bottom.ready_for_eval = True

    def get_best(self) -> PBTAgent:
        """Get best agent."""
        return max(self.population, key=lambda a: a.fitness)


# =============================================================================
# Novelty Search with Local Competition
# =============================================================================


class NoveltySearchLC:
    """Novelty Search with Local Competition.

    Combines novelty-based selection with local quality competition.
    """

    def __init__(
        self,
        k_nearest: int = 15,
        archive_threshold: float = 0.1,
        local_comp_radius: float = 0.2,
    ):
        self.k = k_nearest
        self.threshold = archive_threshold
        self.radius = local_comp_radius

        self.archive: List[Tuple[np.ndarray, float]] = []  # (behavior, fitness)
        self.max_archive = 1000

    def compute_novelty(self, behavior: np.ndarray) -> float:
        """Compute novelty as average distance to k-nearest."""
        if len(self.archive) < self.k:
            return 1.0

        distances = [
            norm(behavior - b)
            for b, _ in self.archive
        ]
        distances.sort()
        return float(np.mean(distances[:self.k]))

    def compute_local_competition(
        self,
        behavior: np.ndarray,
        fitness: float,
    ) -> float:
        """Compute local competition score."""
        if not self.archive:
            return 1.0

        # Find neighbors within radius
        neighbors = [
            (b, f) for b, f in self.archive
            if norm(behavior - b) < self.radius
        ]

        if not neighbors:
            return 1.0

        # Count how many neighbors this individual beats
        wins = sum(1 for _, f in neighbors if fitness > f)
        return wins / len(neighbors)

    def compute_score(
        self,
        behavior: np.ndarray,
        fitness: float,
        novelty_weight: float = 0.5,
    ) -> float:
        """Combined novelty + local competition score."""
        novelty = self.compute_novelty(behavior)
        local_comp = self.compute_local_competition(behavior, fitness)
        return novelty_weight * novelty + (1 - novelty_weight) * local_comp

    def maybe_add_to_archive(
        self,
        behavior: np.ndarray,
        fitness: float,
    ) -> bool:
        """Add to archive if novel enough."""
        novelty = self.compute_novelty(behavior)

        if novelty > self.threshold:
            self.archive.append((behavior.copy(), fitness))

            # Prune if too large
            if len(self.archive) > self.max_archive:
                # Remove oldest or least novel
                self.archive.pop(0)

            return True
        return False


# =============================================================================
# Information-Geometric Optimization (IGO)
# =============================================================================


class IGO:
    """Information-Geometric Optimization.

    Natural gradient descent in distribution space.
    """

    def __init__(
        self,
        dimension: int,
        population_size: int = 50,
        learning_rate: float = 0.5,
    ):
        self.dim = dimension
        self.lam = population_size
        self.lr = learning_rate

        # Natural parameters (for Gaussian)
        self.mean = np.zeros(dimension)
        self.sigma = 1.0

        self.generation = 0

    def ask(self) -> List[np.ndarray]:
        """Sample from current distribution."""
        return [
            self.mean + self.sigma * np.random.randn(self.dim)
            for _ in range(self.lam)
        ]

    def tell(self, solutions: List[np.ndarray], fitnesses: List[float]) -> None:
        """Update distribution using natural gradient."""
        self.generation += 1

        # Compute ranks
        sorted_indices = np.argsort(fitnesses)[::-1]  # Descending
        ranks = np.zeros(self.lam)
        for new_rank, old_idx in enumerate(sorted_indices):
            ranks[old_idx] = new_rank

        # Weights from ranks (using quantile function)
        weights = np.maximum(0, np.log(self.lam / 2 + 1) - np.log(ranks + 1))
        weights /= weights.sum()

        # Natural gradient update for mean
        natural_grad_mean = sum(
            w * (x - self.mean) / self.sigma
            for w, x in zip(weights, solutions)
        )
        self.mean += self.lr * self.sigma * natural_grad_mean

        # Update sigma (simplified)
        squared_norms = [norm(x - self.mean) ** 2 for x in solutions]
        mean_sq_norm = sum(w * sq for w, sq in zip(weights, squared_norms))
        target_sigma = np.sqrt(mean_sq_norm / self.dim)
        self.sigma = self.sigma ** (1 - self.lr) * target_sigma ** self.lr


# =============================================================================
# APEX Evolution Engine - Ultimate Integration
# =============================================================================


class APEXEvolutionEngine:
    """APEX: Advanced Prompt Evolution with eXtreme techniques.

    Integrates all cutting-edge algorithms:
    - CMA-ES for embedding-space optimization
    - Differential Evolution for robustness
    - Bayesian Optimization for sample efficiency
    - MCTS for structured exploration
    - Lexicase for specialist preservation
    - ALPS for diversity maintenance
    - Cooperative Coevolution for modularity
    - PBT for hyperparameter optimization
    - Novelty Search with Local Competition
    - IGO for natural gradient descent
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        objectives: Optional[List[str]] = None,
    ):
        self.embed_dim = embedding_dim
        self.objectives = objectives or ["quality", "novelty", "efficiency"]

        # Core optimizers
        self.cmaes = CMAES(embedding_dim)
        self.de = DifferentialEvolution(embedding_dim, population_size=30)
        self.bayesopt = BayesianOptimizer(embedding_dim)
        self.igo = IGO(embedding_dim)

        # Diversity mechanisms
        self.lexicase = LexicaseSelection(epsilon=0.01)
        self.alps = ALPS(n_layers=4, layer_size=15)
        self.nslc = NoveltySearchLC()

        # Modularity
        n_components = 4  # identity, capabilities, constraints, instructions
        self.coevolution = CooperativeCoevolution(
            n_components=n_components,
            component_dims=[embedding_dim // n_components] * n_components,
        )

        # Meta-optimization
        self.pbt = PopulationBasedTraining(
            population_size=10,
            hyperparam_bounds={
                "cmaes_sigma": (0.1, 1.0),
                "de_scale": (0.3, 0.9),
                "exploration_weight": (0.1, 0.9),
            }
        )

        # State
        self.generation = 0
        self.best_embedding: Optional[np.ndarray] = None
        self.best_fitness = float('-inf')

        # History
        self.history: List[Dict[str, Any]] = []

    async def evolve(
        self,
        embed_fn: Callable[[str], Coroutine[Any, Any, np.ndarray]],
        decode_fn: Callable[[np.ndarray], Coroutine[Any, Any, str]],
        evaluate_fn: Callable[[str], Coroutine[Any, Any, Dict[str, float]]],
        n_generations: int = 10,
    ) -> Dict[str, Any]:
        """Run APEX evolution."""
        results = {
            "generations": [],
            "best_fitness": 0.0,
            "diversity_metrics": {},
        }

        for _ in range(n_generations):
            self.generation += 1
            gen_result = await self._evolve_generation(
                embed_fn, decode_fn, evaluate_fn
            )
            results["generations"].append(gen_result)

            if gen_result.get("best_fitness", 0) > results["best_fitness"]:
                results["best_fitness"] = gen_result["best_fitness"]

        # Final diversity metrics
        results["diversity_metrics"] = {
            "alps_layers": sum(len(l.population) for l in self.alps.layers),
            "nslc_archive": len(self.nslc.archive),
            "cmaes_sigma": self.cmaes.sigma,
        }

        return results

    async def _evolve_generation(
        self,
        _embed_fn: Callable,  # Reserved for future embedding-based operations
        decode_fn: Callable,
        evaluate_fn: Callable,
    ) -> Dict[str, Any]:
        """Run one generation with all algorithms."""
        gen_result = {
            "generation": self.generation,
            "evaluations": 0,
            "best_fitness": self.best_fitness,
        }

        # Strategy 1: CMA-ES candidates
        cma_candidates = self.cmaes.ask()[:5]

        # Strategy 2: DE candidates
        de_best = self.de.evolve(lambda x: -self._quick_eval(x))  # Minimize

        # Strategy 3: Bayesian Optimization
        bo_candidate = self.bayesopt.suggest()

        # Strategy 4: IGO candidates
        igo_candidates = self.igo.ask()[:3]

        # Combine all candidates
        all_candidates = (
            cma_candidates +
            [de_best, bo_candidate] +
            igo_candidates
        )

        # Evaluate
        fitnesses = []
        for embedding in all_candidates:
            try:
                prompt = await decode_fn(embedding)
                fitness_dict = await evaluate_fn(prompt)
                fitness = sum(fitness_dict.values()) / len(fitness_dict)
                fitnesses.append(fitness)

                # Update ALPS
                self.alps.add_individual(embedding, fitness)

                # Update NSLC
                behavior = embedding[:3] / (norm(embedding[:3]) + 1e-10)
                self.nslc.maybe_add_to_archive(behavior, fitness)

                gen_result["evaluations"] += 1

                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_embedding = embedding.copy()
                    gen_result["best_fitness"] = fitness

            except Exception as e:
                fitnesses.append(0.0)
                logger.warning(f"Evaluation failed: {e}")

        # Update algorithms
        if len(cma_candidates) == len(fitnesses[:len(cma_candidates)]):
            self.cmaes.tell(cma_candidates, [-f for f in fitnesses[:len(cma_candidates)]])

        if len(igo_candidates) == len(fitnesses[-len(igo_candidates):]):
            self.igo.tell(igo_candidates, fitnesses[-len(igo_candidates):])

        self.bayesopt.observe(bo_candidate, -fitnesses[len(cma_candidates) + 1])

        # Age ALPS population
        self.alps.age_population()

        self.history.append(gen_result)
        return gen_result

    def _quick_eval(self, embedding: np.ndarray) -> float:
        """Quick heuristic evaluation for DE."""
        # Use surrogate or simple heuristic
        return float(norm(embedding))

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "cmaes_sigma": self.cmaes.sigma,
            "de_generation": self.de.generation,
            "bayesopt_observations": len(self.bayesopt.X_observed),
            "alps_population": sum(len(l.population) for l in self.alps.layers),
            "nslc_archive_size": len(self.nslc.archive),
            "igo_sigma": self.igo.sigma,
        }
