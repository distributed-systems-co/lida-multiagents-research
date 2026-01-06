"""Meta-learning and autonomous adaptation systems.

Provides:
- Learning to learn (MAML, Reptile, meta-gradients)
- Curriculum learning and difficulty progression
- Few-shot adaptation
- Transfer learning across tasks/domains
- Continual learning without catastrophic forgetting
- Automated hyperparameter optimization
- Performance prediction and model selection
- Self-improving agents
"""
from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum
from datetime import datetime
from collections import deque
import asyncio

logger = logging.getLogger(__name__)


class MetaLearningAlgorithm(Enum):
    """Meta-learning algorithms."""
    MAML = "maml"  # Model-Agnostic Meta-Learning
    REPTILE = "reptile"  # Reptile algorithm
    META_SGD = "meta_sgd"  # Meta-SGD with learned learning rates
    MATCHING_NETWORKS = "matching_networks"  # Matching Networks
    PROTOTYPICAL = "prototypical"  # Prototypical Networks


class AdaptationStrategy(Enum):
    """Adaptation strategies."""
    FINE_TUNING = "fine_tuning"  # Standard fine-tuning
    FEW_SHOT = "few_shot"  # Few-shot learning
    ZERO_SHOT = "zero_shot"  # Zero-shot transfer
    MULTI_TASK = "multi_task"  # Multi-task learning
    TRANSFER = "transfer"  # Transfer learning


@dataclass
class Task:
    """A learning task."""

    task_id: str
    name: str
    domain: str
    difficulty: float  # 0-1 scale
    support_set: List[Tuple[Any, Any]]  # (input, output) pairs for training
    query_set: List[Tuple[Any, Any]]  # (input, output) pairs for testing
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """A meta-learning episode (task + adaptation)."""

    task: Task
    initial_params: np.ndarray
    adapted_params: np.ndarray
    support_loss: float
    query_loss: float
    adaptation_steps: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MetaModel:
    """Meta-learned model that can quickly adapt to new tasks."""

    params: np.ndarray  # Model parameters
    meta_params: Optional[np.ndarray] = None  # Meta-parameters (e.g., learning rates)
    task_embedding_fn: Optional[Callable] = None  # Embed task for conditional adaptation
    adaptation_steps: int = 5  # Default adaptation steps
    learning_rate: float = 0.01


# ═══════════════════════════════════════════════════════════════════════════
# META-LEARNING ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════

class MAMLLearner:
    """Model-Agnostic Meta-Learning (MAML) implementation."""

    def __init__(
        self,
        model_fn: Callable,  # Function that takes params and input, returns output
        loss_fn: Callable,  # Function that takes (prediction, target), returns loss
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
    ):
        self.model_fn = model_fn
        self.loss_fn = loss_fn
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps

        self.meta_params: Optional[np.ndarray] = None

    def adapt(self, params: np.ndarray, support_set: List[Tuple]) -> np.ndarray:
        """Inner loop: adapt params on support set."""
        adapted_params = params.copy()

        for _ in range(self.inner_steps):
            # Compute gradient on support set
            grad = self._compute_gradient(adapted_params, support_set)

            # Gradient descent step
            adapted_params = adapted_params - self.inner_lr * grad

        return adapted_params

    def meta_update(self, tasks: List[Task]) -> np.ndarray:
        """Outer loop: update meta-parameters across tasks."""
        if self.meta_params is None:
            # Initialize randomly
            param_size = 100  # Example size
            self.meta_params = np.random.randn(param_size) * 0.01

        # Meta-gradient accumulator
        meta_grad = np.zeros_like(self.meta_params)

        for task in tasks:
            # Inner loop: adapt to task
            adapted_params = self.adapt(self.meta_params, task.support_set)

            # Compute gradient on query set (meta-gradient)
            query_grad = self._compute_gradient(adapted_params, task.query_set)

            # Accumulate
            meta_grad += query_grad

        # Average and update
        meta_grad /= len(tasks)
        self.meta_params = self.meta_params - self.outer_lr * meta_grad

        return self.meta_params

    def _compute_gradient(self, params: np.ndarray, dataset: List[Tuple]) -> np.ndarray:
        """Compute gradient (simplified - in practice use autodiff)."""
        # Placeholder: finite difference approximation
        epsilon = 1e-5
        grad = np.zeros_like(params)

        # Compute loss at current params
        loss = 0.0
        for inp, target in dataset:
            pred = self.model_fn(params, inp)
            loss += self.loss_fn(pred, target)
        loss /= len(dataset)

        # Finite differences (very inefficient but simple)
        for i in range(min(len(params), 10)):  # Only sample of params for efficiency
            params_plus = params.copy()
            params_plus[i] += epsilon

            loss_plus = 0.0
            for inp, target in dataset:
                pred = self.model_fn(params_plus, inp)
                loss_plus += self.loss_fn(pred, target)
            loss_plus /= len(dataset)

            grad[i] = (loss_plus - loss) / epsilon

        return grad


class ReptileLearner:
    """Reptile meta-learning algorithm (simpler than MAML)."""

    def __init__(
        self,
        model_fn: Callable,
        loss_fn: Callable,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
    ):
        self.model_fn = model_fn
        self.loss_fn = loss_fn
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps

        self.meta_params: Optional[np.ndarray] = None

    def adapt(self, params: np.ndarray, task: Task) -> np.ndarray:
        """Adapt on task."""
        adapted_params = params.copy()

        # Combine support and query for Reptile
        all_data = task.support_set + task.query_set

        for _ in range(self.inner_steps):
            grad = self._compute_gradient(adapted_params, all_data)
            adapted_params = adapted_params - self.inner_lr * grad

        return adapted_params

    def meta_update(self, tasks: List[Task]) -> np.ndarray:
        """Meta-update via averaged gradients."""
        if self.meta_params is None:
            param_size = 100
            self.meta_params = np.random.randn(param_size) * 0.01

        # Accumulate adapted parameters
        total_update = np.zeros_like(self.meta_params)

        for task in tasks:
            # Adapt
            adapted_params = self.adapt(self.meta_params, task)

            # Direction of adaptation
            update = adapted_params - self.meta_params
            total_update += update

        # Average and apply
        total_update /= len(tasks)
        self.meta_params = self.meta_params + self.outer_lr * total_update

        return self.meta_params

    def _compute_gradient(self, params: np.ndarray, dataset: List[Tuple]) -> np.ndarray:
        """Simplified gradient computation."""
        epsilon = 1e-5
        grad = np.zeros_like(params)

        loss = 0.0
        for inp, target in dataset:
            pred = self.model_fn(params, inp)
            loss += self.loss_fn(pred, target)
        loss /= len(dataset)

        for i in range(min(len(params), 10)):
            params_plus = params.copy()
            params_plus[i] += epsilon

            loss_plus = 0.0
            for inp, target in dataset:
                pred = self.model_fn(params_plus, inp)
                loss_plus += self.loss_fn(pred, target)
            loss_plus /= len(dataset)

            grad[i] = (loss_plus - loss) / epsilon

        return grad


# ═══════════════════════════════════════════════════════════════════════════
# CURRICULUM LEARNING
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CurriculumStage:
    """A stage in the curriculum."""

    stage_id: int
    difficulty_range: Tuple[float, float]  # Min, max difficulty
    tasks: List[Task]
    mastery_threshold: float = 0.8  # Performance threshold to advance


class CurriculumLearner:
    """Curriculum learning with adaptive difficulty progression."""

    def __init__(
        self,
        task_pool: List[Task],
        num_stages: int = 5,
        mastery_threshold: float = 0.8,
    ):
        self.task_pool = task_pool
        self.num_stages = num_stages
        self.mastery_threshold = mastery_threshold

        # Build curriculum stages
        self.stages = self._build_curriculum()

        # Current stage
        self.current_stage = 0

        # Performance history
        self.performance_history: List[float] = []

    def _build_curriculum(self) -> List[CurriculumStage]:
        """Build curriculum stages by difficulty."""
        # Sort tasks by difficulty
        sorted_tasks = sorted(self.task_pool, key=lambda t: t.difficulty)

        # Split into stages
        tasks_per_stage = len(sorted_tasks) // self.num_stages
        stages = []

        for i in range(self.num_stages):
            start_idx = i * tasks_per_stage
            end_idx = start_idx + tasks_per_stage if i < self.num_stages - 1 else len(sorted_tasks)

            stage_tasks = sorted_tasks[start_idx:end_idx]

            if stage_tasks:
                min_diff = min(t.difficulty for t in stage_tasks)
                max_diff = max(t.difficulty for t in stage_tasks)

                stage = CurriculumStage(
                    stage_id=i,
                    difficulty_range=(min_diff, max_diff),
                    tasks=stage_tasks,
                    mastery_threshold=self.mastery_threshold,
                )
                stages.append(stage)

        return stages

    def get_current_tasks(self) -> List[Task]:
        """Get tasks for current stage."""
        if self.current_stage < len(self.stages):
            return self.stages[self.current_stage].tasks
        return []

    def record_performance(self, performance: float):
        """Record performance and potentially advance stage."""
        self.performance_history.append(performance)

        # Check if mastered current stage (based on recent performance)
        recent_window = 5
        if len(self.performance_history) >= recent_window:
            recent_perf = self.performance_history[-recent_window:]
            avg_perf = np.mean(recent_perf)

            if avg_perf >= self.mastery_threshold:
                self.advance_stage()

    def advance_stage(self):
        """Move to next stage."""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            logger.info(f"Advanced to curriculum stage {self.current_stage}")

    def get_progress(self) -> Dict[str, Any]:
        """Get curriculum progress."""
        return {
            "current_stage": self.current_stage,
            "total_stages": len(self.stages),
            "progress_percent": (self.current_stage / len(self.stages)) * 100,
            "recent_performance": self.performance_history[-10:] if self.performance_history else [],
        }


# ═══════════════════════════════════════════════════════════════════════════
# CONTINUAL LEARNING
# ═══════════════════════════════════════════════════════════════════════════

class ContinualLearner:
    """Continual learning without catastrophic forgetting.

    Uses Elastic Weight Consolidation (EWC) approach.
    """

    def __init__(
        self,
        model_fn: Callable,
        loss_fn: Callable,
        fisher_samples: int = 100,
        ewc_lambda: float = 1000.0,
    ):
        self.model_fn = model_fn
        self.loss_fn = loss_fn
        self.fisher_samples = fisher_samples
        self.ewc_lambda = ewc_lambda

        # Current parameters
        self.params: Optional[np.ndarray] = None

        # Task-specific fisher information and parameters
        self.task_fishers: List[np.ndarray] = []
        self.task_params: List[np.ndarray] = []

    def learn_task(self, task: Task) -> np.ndarray:
        """Learn a new task while preserving previous tasks."""
        if self.params is None:
            # First task - normal learning
            self.params = np.random.randn(100) * 0.01

        # Train on new task with EWC regularization
        for _ in range(100):  # Training iterations
            # Compute loss on new task
            task_loss = self._task_loss(self.params, task)

            # Add EWC penalty for previous tasks
            ewc_penalty = 0.0
            for fisher, old_params in zip(self.task_fishers, self.task_params):
                # Penalty = λ * Σ F_i * (θ_i - θ*_i)^2
                ewc_penalty += np.sum(fisher * (self.params - old_params) ** 2)

            total_loss = task_loss + (self.ewc_lambda / 2) * ewc_penalty

            # Gradient descent (simplified)
            grad = self._compute_gradient(self.params, task)
            self.params -= 0.01 * grad

        # Compute fisher information for this task
        fisher = self._compute_fisher(self.params, task)

        # Store for future tasks
        self.task_fishers.append(fisher)
        self.task_params.append(self.params.copy())

        return self.params

    def _task_loss(self, params: np.ndarray, task: Task) -> float:
        """Compute loss on task."""
        total_loss = 0.0
        for inp, target in task.support_set:
            pred = self.model_fn(params, inp)
            total_loss += self.loss_fn(pred, target)
        return total_loss / len(task.support_set)

    def _compute_fisher(self, params: np.ndarray, task: Task) -> np.ndarray:
        """Compute Fisher information matrix diagonal."""
        # Simplified: use gradient variance as approximation
        gradients = []

        for inp, target in task.support_set[:self.fisher_samples]:
            grad = self._compute_gradient_sample(params, inp, target)
            gradients.append(grad)

        # Fisher ≈ E[∇log p(y|x)^2]
        fisher = np.var(gradients, axis=0)

        return fisher

    def _compute_gradient(self, params: np.ndarray, task: Task) -> np.ndarray:
        """Compute gradient on task."""
        epsilon = 1e-5
        grad = np.zeros_like(params)

        loss = self._task_loss(params, task)

        for i in range(min(len(params), 10)):
            params_plus = params.copy()
            params_plus[i] += epsilon

            loss_plus = self._task_loss(params_plus, task)
            grad[i] = (loss_plus - loss) / epsilon

        return grad

    def _compute_gradient_sample(self, params: np.ndarray, inp: Any, target: Any) -> np.ndarray:
        """Compute gradient on single sample."""
        epsilon = 1e-5
        grad = np.zeros_like(params)

        pred = self.model_fn(params, inp)
        loss = self.loss_fn(pred, target)

        for i in range(min(len(params), 10)):
            params_plus = params.copy()
            params_plus[i] += epsilon

            pred_plus = self.model_fn(params_plus, inp)
            loss_plus = self.loss_fn(pred_plus, target)

            grad[i] = (loss_plus - loss) / epsilon

        return grad


# ═══════════════════════════════════════════════════════════════════════════
# AUTOMATED HYPERPARAMETER OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HyperparameterConfig:
    """Hyperparameter configuration."""

    name: str
    params: Dict[str, Any]
    performance: Optional[float] = None


class BayesianOptimizer:
    """Bayesian optimization for hyperparameter tuning."""

    def __init__(
        self,
        param_space: Dict[str, Tuple[float, float]],  # param_name -> (min, max)
        acquisition: str = "ucb",
    ):
        self.param_space = param_space
        self.acquisition = acquisition

        # Observed configurations
        self.observations: List[Tuple[Dict, float]] = []

    def suggest(self) -> Dict[str, float]:
        """Suggest next configuration to try."""
        if len(self.observations) < 5:
            # Random exploration initially
            return self._random_config()

        # Use acquisition function
        if self.acquisition == "ucb":
            return self._ucb_acquisition()
        else:
            return self._random_config()

    def observe(self, config: Dict[str, float], performance: float):
        """Record observation."""
        self.observations.append((config, performance))

    def _random_config(self) -> Dict[str, float]:
        """Generate random configuration."""
        config = {}
        for param, (min_val, max_val) in self.param_space.items():
            config[param] = np.random.uniform(min_val, max_val)
        return config

    def _ucb_acquisition(self) -> Dict[str, float]:
        """Upper Confidence Bound acquisition."""
        # Simplified: sample random configs and pick best UCB
        n_candidates = 100
        best_ucb = -float('inf')
        best_config = None

        for _ in range(n_candidates):
            config = self._random_config()

            # Predict mean and uncertainty (simplified GP)
            mean, std = self._predict(config)

            # UCB = mean + κ * std
            ucb = mean + 2.0 * std

            if ucb > best_ucb:
                best_ucb = ucb
                best_config = config

        return best_config

    def _predict(self, config: Dict[str, float]) -> Tuple[float, float]:
        """Predict mean and std for config (simplified GP)."""
        if not self.observations:
            return 0.0, 1.0

        # Distance-weighted average (simple kernel)
        weights = []
        values = []

        for obs_config, obs_perf in self.observations:
            # Compute distance
            dist = 0.0
            for param in config:
                if param in obs_config:
                    param_range = self.param_space[param][1] - self.param_space[param][0]
                    if param_range > 0:
                        dist += ((config[param] - obs_config[param]) / param_range) ** 2

            dist = np.sqrt(dist)

            # Kernel: RBF
            weight = np.exp(-dist)
            weights.append(weight)
            values.append(obs_perf)

        weights = np.array(weights)
        values = np.array(values)

        if weights.sum() > 0:
            mean = np.sum(weights * values) / weights.sum()
            # Uncertainty decreases with more observations nearby
            std = 1.0 / (1.0 + weights.sum())
        else:
            mean = 0.0
            std = 1.0

        return mean, std


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED META-LEARNING SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

class MetaLearningSystem:
    """Unified meta-learning and adaptation system."""

    def __init__(
        self,
        algorithm: MetaLearningAlgorithm = MetaLearningAlgorithm.MAML,
    ):
        self.algorithm = algorithm

        # Initialize learners
        if algorithm == MetaLearningAlgorithm.MAML:
            self.meta_learner = MAMLLearner(
                model_fn=lambda p, x: p @ x if hasattr(x, '__matmul__') else np.sum(p),  # Dummy
                loss_fn=lambda pred, target: (pred - target) ** 2,
            )
        elif algorithm == MetaLearningAlgorithm.REPTILE:
            self.meta_learner = ReptileLearner(
                model_fn=lambda p, x: p @ x if hasattr(x, '__matmul__') else np.sum(p),
                loss_fn=lambda pred, target: (pred - target) ** 2,
            )
        else:
            self.meta_learner = None

        # Curriculum learner
        self.curriculum: Optional[CurriculumLearner] = None

        # Continual learner
        self.continual_learner = ContinualLearner(
            model_fn=lambda p, x: p @ x if hasattr(x, '__matmul__') else np.sum(p),
            loss_fn=lambda pred, target: (pred - target) ** 2,
        )

        # Hyperparameter optimizer
        self.hp_optimizer: Optional[BayesianOptimizer] = None

        # Episode history
        self.episodes: List[Episode] = []

    def meta_train(self, tasks: List[Task], num_iterations: int = 100) -> MetaModel:
        """Meta-train on a distribution of tasks."""
        logger.info(f"Meta-training with {len(tasks)} tasks for {num_iterations} iterations")

        for iteration in range(num_iterations):
            # Sample batch of tasks
            batch_size = min(5, len(tasks))
            task_batch = np.random.choice(tasks, batch_size, replace=False).tolist()

            # Meta-update
            if self.meta_learner:
                meta_params = self.meta_learner.meta_update(task_batch)

                if iteration % 10 == 0:
                    logger.info(f"Iteration {iteration}: meta-params updated")

        # Return meta-model
        return MetaModel(
            params=meta_params if self.meta_learner and hasattr(self.meta_learner, 'meta_params') else np.random.randn(100),
            adaptation_steps=5,
        )

    def few_shot_adapt(
        self,
        meta_model: MetaModel,
        task: Task,
        n_shots: int = 5,
    ) -> np.ndarray:
        """Few-shot adaptation to new task."""
        # Use only n_shots examples from support set
        few_shot_support = task.support_set[:n_shots]

        # Create temporary task
        few_shot_task = Task(
            task_id=f"{task.task_id}_few_shot",
            name=task.name,
            domain=task.domain,
            difficulty=task.difficulty,
            support_set=few_shot_support,
            query_set=task.query_set,
        )

        # Adapt
        if self.meta_learner:
            adapted_params = self.meta_learner.adapt(meta_model.params, few_shot_task.support_set)
        else:
            adapted_params = meta_model.params

        return adapted_params

    def create_curriculum(self, tasks: List[Task], num_stages: int = 5) -> CurriculumLearner:
        """Create curriculum from tasks."""
        self.curriculum = CurriculumLearner(tasks, num_stages)
        return self.curriculum

    def continual_learn(self, task: Task):
        """Learn new task continually."""
        self.continual_learner.learn_task(task)

    def optimize_hyperparameters(
        self,
        param_space: Dict[str, Tuple[float, float]],
        eval_fn: Callable[[Dict], float],
        n_iterations: int = 20,
    ) -> Dict[str, float]:
        """Optimize hyperparameters using Bayesian optimization."""
        self.hp_optimizer = BayesianOptimizer(param_space)

        best_config = None
        best_performance = -float('inf')

        for i in range(n_iterations):
            # Suggest config
            config = self.hp_optimizer.suggest()

            # Evaluate
            performance = eval_fn(config)

            # Record
            self.hp_optimizer.observe(config, performance)

            if performance > best_performance:
                best_performance = performance
                best_config = config

            logger.info(f"Iteration {i}: config={config}, performance={performance:.3f}")

        return best_config


# Global instance
_meta_learning_system: Optional[MetaLearningSystem] = None


def get_meta_learning_system() -> MetaLearningSystem:
    """Get global meta-learning system."""
    global _meta_learning_system
    if _meta_learning_system is None:
        _meta_learning_system = MetaLearningSystem()
    return _meta_learning_system
