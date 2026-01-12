"""
Meta-Cognitive Architecture

Advanced meta-cognition including:
- Learning to Learn (meta-learning)
- Self-Monitoring and Introspection
- Cognitive Strategy Selection
- Confidence Calibration
- Error Detection and Recovery
- Knowledge about Knowledge (epistemic awareness)
- Self-Modification and Improvement
- Cognitive Load Management

Based on:
- Metacognition Research
- Self-Regulated Learning
- Meta-Learning in AI
- Cognitive Architecture Theory
"""

import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Callable, TypeVar
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import statistics


# ============================================================================
# COGNITIVE MONITORING
# ============================================================================

class ConfidenceLevel(Enum):
    """Levels of confidence in knowledge/beliefs"""
    CERTAIN = "certain"
    CONFIDENT = "confident"
    PROBABLE = "probable"
    UNCERTAIN = "uncertain"
    SPECULATIVE = "speculative"
    UNKNOWN = "unknown"


class KnowledgeStatus(Enum):
    """Status of knowledge items"""
    VERIFIED = "verified"
    BELIEVED = "believed"
    HYPOTHESIZED = "hypothesized"
    QUESTIONED = "questioned"
    DEPRECATED = "deprecated"


class CognitiveState(Enum):
    """Cognitive states for self-monitoring"""
    FOCUSED = "focused"
    DIFFUSE = "diffuse"
    OVERLOADED = "overloaded"
    FATIGUED = "fatigued"
    CONFUSED = "confused"
    FLOW = "flow"
    BORED = "bored"


@dataclass
class ConfidenceEstimate:
    """Calibrated confidence estimate"""
    value: float  # 0-1
    level: ConfidenceLevel
    justification: str
    calibration_history: List[Tuple[float, bool]] = field(default_factory=list)

    @property
    def calibration_error(self) -> float:
        """Calculate calibration error (Brier score)"""
        if not self.calibration_history:
            return 0.5
        errors = [(conf - (1.0 if correct else 0.0))**2
                  for conf, correct in self.calibration_history]
        return sum(errors) / len(errors)

    def update_calibration(self, was_correct: bool):
        """Update calibration with outcome"""
        self.calibration_history.append((self.value, was_correct))
        if len(self.calibration_history) > 100:
            self.calibration_history.pop(0)


@dataclass
class MetaKnowledge:
    """Knowledge about a knowledge item"""
    content_id: str
    confidence: ConfidenceEstimate
    status: KnowledgeStatus
    source: str
    acquisition_method: str  # learned, inferred, told, observed
    last_verified: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    usefulness_score: float = 0.5
    related_items: Set[str] = field(default_factory=set)


class CognitiveMonitor:
    """
    Monitors cognitive state and performance.
    """

    def __init__(self):
        self.current_state = CognitiveState.FOCUSED
        self.cognitive_load = 0.3  # 0-1
        self.attention_span = 1.0  # Decreases with fatigue
        self.working_memory_usage = 0.0
        self.working_memory_capacity = 7  # Miller's magic number

        # Performance tracking
        self.task_performance: List[Dict[str, Any]] = []
        self.error_rate = 0.0
        self.response_times: List[float] = []

        # State history
        self.state_history: List[Tuple[datetime, CognitiveState]] = []
        self.load_history: List[Tuple[datetime, float]] = []

    def update_cognitive_load(self, items_in_wm: int, task_complexity: float):
        """Update cognitive load estimate"""
        wm_load = items_in_wm / self.working_memory_capacity
        self.working_memory_usage = wm_load
        self.cognitive_load = (wm_load + task_complexity) / 2

        # Update state based on load
        if self.cognitive_load > 0.9:
            self.current_state = CognitiveState.OVERLOADED
        elif self.cognitive_load > 0.7:
            if self.attention_span > 0.5:
                self.current_state = CognitiveState.FOCUSED
            else:
                self.current_state = CognitiveState.FATIGUED
        elif self.cognitive_load < 0.2:
            self.current_state = CognitiveState.BORED
        elif 0.4 <= self.cognitive_load <= 0.7 and self.attention_span > 0.7:
            self.current_state = CognitiveState.FLOW

        self.load_history.append((datetime.now(), self.cognitive_load))
        self.state_history.append((datetime.now(), self.current_state))

    def record_task_outcome(self, task_id: str, success: bool,
                           duration: float, difficulty: float):
        """Record task performance"""
        self.task_performance.append({
            "task_id": task_id,
            "success": success,
            "duration": duration,
            "difficulty": difficulty,
            "cognitive_load": self.cognitive_load,
            "state": self.current_state,
            "timestamp": datetime.now()
        })

        self.response_times.append(duration)

        # Update error rate
        recent = self.task_performance[-20:]
        self.error_rate = sum(1 for t in recent if not t["success"]) / len(recent)

        # Update attention span based on performance
        if success:
            self.attention_span = min(1.0, self.attention_span + 0.02)
        else:
            self.attention_span = max(0.0, self.attention_span - 0.05)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of cognitive performance"""
        if not self.task_performance:
            return {"status": "no_data"}

        recent = self.task_performance[-50:]
        success_rate = sum(1 for t in recent if t["success"]) / len(recent)
        avg_duration = statistics.mean(t["duration"] for t in recent)

        return {
            "current_state": self.current_state.value,
            "cognitive_load": self.cognitive_load,
            "attention_span": self.attention_span,
            "success_rate": success_rate,
            "avg_response_time": avg_duration,
            "error_rate": self.error_rate,
            "tasks_completed": len(self.task_performance)
        }

    def should_take_break(self) -> Tuple[bool, str]:
        """Determine if a break is needed"""
        if self.current_state == CognitiveState.OVERLOADED:
            return True, "Cognitive overload detected"
        if self.current_state == CognitiveState.FATIGUED:
            return True, "Fatigue detected"
        if self.error_rate > 0.3:
            return True, "High error rate"
        if self.attention_span < 0.3:
            return True, "Attention depleted"
        return False, ""


# ============================================================================
# STRATEGY SELECTION
# ============================================================================

class StrategyType(Enum):
    """Types of cognitive strategies"""
    ANALYTICAL = "analytical"  # Step-by-step logical analysis
    INTUITIVE = "intuitive"  # Quick pattern-based
    CREATIVE = "creative"  # Novel combinations
    SYSTEMATIC = "systematic"  # Exhaustive search
    HEURISTIC = "heuristic"  # Rule of thumb
    ANALOGICAL = "analogical"  # Mapping from known to unknown
    DECOMPOSITION = "decomposition"  # Break into subproblems
    ABSTRACTION = "abstraction"  # Find general principles


@dataclass
class CognitiveStrategy:
    """A cognitive strategy with performance tracking"""
    id: str
    strategy_type: StrategyType
    name: str
    description: str

    # Applicability
    suitable_tasks: List[str] = field(default_factory=list)
    cognitive_load: float = 0.5
    time_cost: float = 0.5

    # Performance tracking
    uses: int = 0
    successes: int = 0
    failures: int = 0
    avg_time: float = 0.0
    performance_by_task: Dict[str, float] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        if self.uses == 0:
            return 0.5
        return self.successes / self.uses

    def record_use(self, task_type: str, success: bool, time: float):
        """Record strategy use"""
        self.uses += 1
        if success:
            self.successes += 1
        else:
            self.failures += 1
        self.avg_time = (self.avg_time * (self.uses - 1) + time) / self.uses

        if task_type not in self.performance_by_task:
            self.performance_by_task[task_type] = 0.5
        current = self.performance_by_task[task_type]
        self.performance_by_task[task_type] = current * 0.9 + (1.0 if success else 0.0) * 0.1


class StrategySelector:
    """
    Selects appropriate cognitive strategies based on context.
    Implements meta-level strategy learning.
    """

    def __init__(self):
        self.strategies: Dict[str, CognitiveStrategy] = {}
        self.current_strategy: Optional[CognitiveStrategy] = None
        self.strategy_history: List[Tuple[str, str, bool]] = []  # (task, strategy, success)

        # Task-strategy performance matrix
        self.task_strategy_matrix: Dict[Tuple[str, str], float] = defaultdict(lambda: 0.5)

        # Initialize default strategies
        self._init_default_strategies()

    def _init_default_strategies(self):
        """Initialize default cognitive strategies"""
        defaults = [
            CognitiveStrategy(
                id="analytical",
                strategy_type=StrategyType.ANALYTICAL,
                name="Analytical Reasoning",
                description="Step-by-step logical analysis",
                suitable_tasks=["logic", "math", "debugging"],
                cognitive_load=0.7,
                time_cost=0.8
            ),
            CognitiveStrategy(
                id="intuitive",
                strategy_type=StrategyType.INTUITIVE,
                name="Intuitive Pattern Matching",
                description="Quick recognition-based reasoning",
                suitable_tasks=["classification", "estimation", "familiar_problems"],
                cognitive_load=0.3,
                time_cost=0.2
            ),
            CognitiveStrategy(
                id="creative",
                strategy_type=StrategyType.CREATIVE,
                name="Creative Exploration",
                description="Generate novel combinations and ideas",
                suitable_tasks=["brainstorming", "design", "novel_problems"],
                cognitive_load=0.6,
                time_cost=0.6
            ),
            CognitiveStrategy(
                id="systematic",
                strategy_type=StrategyType.SYSTEMATIC,
                name="Systematic Search",
                description="Exhaustive enumeration of possibilities",
                suitable_tasks=["verification", "complete_search", "enumeration"],
                cognitive_load=0.8,
                time_cost=0.9
            ),
            CognitiveStrategy(
                id="heuristic",
                strategy_type=StrategyType.HEURISTIC,
                name="Heuristic Shortcuts",
                description="Apply rules of thumb",
                suitable_tasks=["estimation", "satisficing", "time_pressure"],
                cognitive_load=0.2,
                time_cost=0.1
            ),
            CognitiveStrategy(
                id="analogical",
                strategy_type=StrategyType.ANALOGICAL,
                name="Analogical Transfer",
                description="Map from known to unknown domain",
                suitable_tasks=["novel_domains", "learning", "explanation"],
                cognitive_load=0.5,
                time_cost=0.5
            ),
            CognitiveStrategy(
                id="decomposition",
                strategy_type=StrategyType.DECOMPOSITION,
                name="Problem Decomposition",
                description="Break complex problems into subproblems",
                suitable_tasks=["complex_problems", "planning", "divide_conquer"],
                cognitive_load=0.6,
                time_cost=0.7
            ),
            CognitiveStrategy(
                id="abstraction",
                strategy_type=StrategyType.ABSTRACTION,
                name="Abstraction and Generalization",
                description="Find underlying principles",
                suitable_tasks=["pattern_finding", "theory_building", "transfer"],
                cognitive_load=0.7,
                time_cost=0.6
            )
        ]

        for strategy in defaults:
            self.strategies[strategy.id] = strategy

    def select_strategy(self, task_type: str, time_available: float,
                       cognitive_load_available: float) -> CognitiveStrategy:
        """Select best strategy for given context"""
        candidates = []

        for strategy in self.strategies.values():
            # Check constraints
            if strategy.time_cost > time_available:
                continue
            if strategy.cognitive_load > cognitive_load_available:
                continue

            # Calculate expected utility
            if task_type in strategy.suitable_tasks:
                suitability = 0.8
            else:
                suitability = 0.4

            performance = self.task_strategy_matrix[(task_type, strategy.id)]
            utility = suitability * 0.3 + performance * 0.5 + strategy.success_rate * 0.2

            candidates.append((strategy, utility))

        if not candidates:
            # Fall back to heuristic (lowest cost)
            return self.strategies.get("heuristic", list(self.strategies.values())[0])

        # Select best with some exploration
        candidates.sort(key=lambda x: -x[1])
        if random.random() < 0.1:  # 10% exploration
            return random.choice([c[0] for c in candidates])
        else:
            return candidates[0][0]

    def record_outcome(self, task_type: str, strategy_id: str,
                      success: bool, time: float):
        """Record strategy use outcome"""
        if strategy_id in self.strategies:
            self.strategies[strategy_id].record_use(task_type, success, time)

        # Update task-strategy matrix
        key = (task_type, strategy_id)
        current = self.task_strategy_matrix[key]
        self.task_strategy_matrix[key] = current * 0.9 + (1.0 if success else 0.0) * 0.1

        self.strategy_history.append((task_type, strategy_id, success))

    def get_strategy_recommendation(self, task_type: str) -> Dict[str, Any]:
        """Get recommendation with explanation"""
        time_avail = 1.0
        load_avail = 1.0

        strategy = self.select_strategy(task_type, time_avail, load_avail)

        return {
            "strategy": strategy.name,
            "strategy_id": strategy.id,
            "reasoning": f"Selected based on {strategy.success_rate:.0%} success rate for similar tasks",
            "expected_time": strategy.time_cost,
            "cognitive_load": strategy.cognitive_load,
            "alternatives": [
                s.name for s in self.strategies.values()
                if s.id != strategy.id and task_type in s.suitable_tasks
            ][:3]
        }


# ============================================================================
# SELF-MODELING
# ============================================================================

@dataclass
class Capability:
    """A capability the agent has"""
    id: str
    name: str
    domain: str
    proficiency: float = 0.5  # 0-1
    confidence_in_proficiency: float = 0.5
    practice_count: int = 0
    last_practiced: datetime = field(default_factory=datetime.now)
    learning_rate: float = 0.1


@dataclass
class Limitation:
    """A known limitation"""
    id: str
    description: str
    domain: str
    severity: float = 0.5
    workarounds: List[str] = field(default_factory=list)
    improving: bool = False


@dataclass
class SelfModel:
    """Agent's model of itself"""
    # Capabilities and limitations
    capabilities: Dict[str, Capability] = field(default_factory=dict)
    limitations: Dict[str, Limitation] = field(default_factory=dict)

    # Personality/preferences (learned)
    preferences: Dict[str, float] = field(default_factory=dict)
    biases: Dict[str, float] = field(default_factory=dict)

    # Current resources
    energy_level: float = 1.0
    focus_level: float = 1.0
    stress_level: float = 0.0

    # Goals and values
    goals: List[str] = field(default_factory=list)
    values: Dict[str, float] = field(default_factory=dict)

    # Meta-knowledge
    known_unknowns: Set[str] = field(default_factory=set)
    blind_spots: Set[str] = field(default_factory=set)  # Suspected but unconfirmed


class SelfModeler:
    """
    Builds and maintains a model of the agent itself.
    Enables self-awareness and self-improvement.
    """

    def __init__(self):
        self.model = SelfModel()
        self.observation_history: List[Dict[str, Any]] = []
        self.self_predictions: List[Tuple[str, bool]] = []  # (prediction, correct)
        self.self_model_accuracy = 0.5

    def observe_behavior(self, context: Dict[str, Any], behavior: str,
                        outcome: Dict[str, Any]):
        """Observe own behavior for self-modeling"""
        observation = {
            "context": context,
            "behavior": behavior,
            "outcome": outcome,
            "timestamp": datetime.now(),
            "energy": self.model.energy_level,
            "stress": self.model.stress_level
        }
        self.observation_history.append(observation)

        # Update capabilities based on outcome
        domain = context.get("domain", "general")
        success = outcome.get("success", False)
        self._update_capability(domain, success)

        # Detect patterns in behavior
        self._detect_patterns()

    def _update_capability(self, domain: str, success: bool):
        """Update capability estimate for a domain"""
        if domain not in self.model.capabilities:
            self.model.capabilities[domain] = Capability(
                id=domain,
                name=domain,
                domain=domain
            )

        cap = self.model.capabilities[domain]
        cap.practice_count += 1
        cap.last_practiced = datetime.now()

        # Update proficiency with learning
        if success:
            cap.proficiency = min(1.0, cap.proficiency + cap.learning_rate * (1 - cap.proficiency))
        else:
            cap.proficiency = max(0.0, cap.proficiency - cap.learning_rate * 0.5)

        # Update confidence in proficiency estimate
        cap.confidence_in_proficiency = min(0.95, cap.confidence_in_proficiency + 0.02)

    def _detect_patterns(self):
        """Detect patterns in own behavior"""
        if len(self.observation_history) < 10:
            return

        recent = self.observation_history[-50:]

        # Detect preferences
        context_outcomes = defaultdict(list)
        for obs in recent:
            for key, value in obs["context"].items():
                context_outcomes[f"{key}:{value}"].append(obs["outcome"].get("success", False))

        for context_key, outcomes in context_outcomes.items():
            if len(outcomes) >= 5:
                success_rate = sum(outcomes) / len(outcomes)
                if success_rate > 0.7:
                    self.model.preferences[context_key] = success_rate
                elif success_rate < 0.3:
                    self.model.biases[context_key] = 1 - success_rate

        # Detect limitations
        failure_contexts = defaultdict(int)
        for obs in recent:
            if not obs["outcome"].get("success", True):
                domain = obs["context"].get("domain", "unknown")
                failure_contexts[domain] += 1

        for domain, failures in failure_contexts.items():
            if failures >= 3:
                if domain not in self.model.limitations:
                    self.model.limitations[domain] = Limitation(
                        id=domain,
                        description=f"Difficulty with {domain} tasks",
                        domain=domain,
                        severity=failures / 10
                    )

    def predict_own_performance(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Predict own performance on a task"""
        domain = task.get("domain", "general")
        difficulty = task.get("difficulty", 0.5)

        # Base prediction on capability
        if domain in self.model.capabilities:
            cap = self.model.capabilities[domain]
            base_prob = cap.proficiency
            confidence = cap.confidence_in_proficiency
        else:
            base_prob = 0.5
            confidence = 0.3

        # Adjust for current state
        state_factor = (
            self.model.energy_level * 0.3 +
            self.model.focus_level * 0.4 +
            (1 - self.model.stress_level) * 0.3
        )

        # Adjust for difficulty
        adjusted_prob = base_prob * state_factor * (1 - difficulty * 0.5)

        return {
            "success_probability": adjusted_prob,
            "confidence": confidence,
            "domain_proficiency": base_prob,
            "state_factor": state_factor,
            "limitations": [
                lim.description for lim in self.model.limitations.values()
                if lim.domain == domain
            ]
        }

    def verify_prediction(self, prediction: Dict[str, Any], actual_success: bool):
        """Verify a self-prediction against outcome"""
        predicted_success = prediction["success_probability"] > 0.5
        correct = predicted_success == actual_success

        self.self_predictions.append((str(prediction["success_probability"]), correct))

        # Update self-model accuracy
        recent = self.self_predictions[-30:]
        self.self_model_accuracy = sum(1 for _, c in recent if c) / len(recent)

    def identify_blind_spots(self) -> List[str]:
        """Identify potential blind spots in self-knowledge"""
        blind_spots = []

        # Low confidence capabilities
        for cap in self.model.capabilities.values():
            if cap.confidence_in_proficiency < 0.4:
                blind_spots.append(f"Uncertain about {cap.domain} capability")

        # Unexplored areas
        if len(self.observation_history) > 20:
            domains_explored = set()
            for obs in self.observation_history:
                domains_explored.add(obs["context"].get("domain", "unknown"))

            common_domains = {"reasoning", "planning", "memory", "learning", "social", "creative"}
            unexplored = common_domains - domains_explored
            for domain in unexplored:
                blind_spots.append(f"Unexplored domain: {domain}")

        # Systematic biases
        for bias_key, bias_strength in self.model.biases.items():
            if bias_strength > 0.6:
                blind_spots.append(f"Possible bias: {bias_key}")

        self.model.blind_spots = set(blind_spots)
        return blind_spots

    def suggest_improvements(self) -> List[Dict[str, Any]]:
        """Suggest self-improvements"""
        suggestions = []

        # Improve weak capabilities
        for cap in self.model.capabilities.values():
            if cap.proficiency < 0.5 and cap.practice_count > 5:
                suggestions.append({
                    "type": "practice",
                    "target": cap.domain,
                    "reason": f"Low proficiency ({cap.proficiency:.0%}) despite practice",
                    "action": f"Focused practice on {cap.domain}"
                })

        # Address limitations
        for lim in self.model.limitations.values():
            if lim.severity > 0.5 and not lim.improving:
                suggestions.append({
                    "type": "address_limitation",
                    "target": lim.domain,
                    "reason": lim.description,
                    "action": f"Develop workarounds for {lim.domain}"
                })

        # Explore blind spots
        for blind_spot in self.model.blind_spots:
            suggestions.append({
                "type": "explore",
                "target": blind_spot,
                "reason": "Unknown territory",
                "action": f"Investigate: {blind_spot}"
            })

        # Calibration
        if self.self_model_accuracy < 0.7:
            suggestions.append({
                "type": "calibration",
                "target": "self_model",
                "reason": f"Self-model accuracy is {self.self_model_accuracy:.0%}",
                "action": "Collect more feedback on predictions"
            })

        return suggestions


# ============================================================================
# META-LEARNING
# ============================================================================

@dataclass
class LearningEpisode:
    """Record of a learning episode"""
    id: str
    domain: str
    initial_performance: float
    final_performance: float
    duration: timedelta
    strategy_used: str
    success: bool
    transfer_potential: float = 0.5


class MetaLearner:
    """
    Learns how to learn more effectively.
    Implements meta-learning capabilities.
    """

    def __init__(self):
        self.learning_episodes: List[LearningEpisode] = []
        self.learning_strategies: Dict[str, Dict[str, Any]] = {}
        self.domain_learning_rates: Dict[str, float] = defaultdict(lambda: 0.1)
        self.transfer_matrix: Dict[Tuple[str, str], float] = defaultdict(lambda: 0.0)

        # Meta-learning parameters
        self.meta_learning_rate = 0.1
        self.exploration_rate = 0.2
        self.curriculum: List[str] = []

    def record_learning(self, domain: str, initial_perf: float, final_perf: float,
                       duration: timedelta, strategy: str):
        """Record a learning episode"""
        improvement = final_perf - initial_perf

        episode = LearningEpisode(
            id=f"ep_{len(self.learning_episodes)}",
            domain=domain,
            initial_performance=initial_perf,
            final_performance=final_perf,
            duration=duration,
            strategy_used=strategy,
            success=improvement > 0
        )
        self.learning_episodes.append(episode)

        # Update learning rate for domain
        if improvement > 0:
            hours = duration.total_seconds() / 3600
            effective_rate = improvement / max(1, hours)
            current_rate = self.domain_learning_rates[domain]
            self.domain_learning_rates[domain] = current_rate * 0.9 + effective_rate * 0.1

        # Update strategy effectiveness
        if strategy not in self.learning_strategies:
            self.learning_strategies[strategy] = {
                "uses": 0, "successes": 0, "total_improvement": 0
            }
        self.learning_strategies[strategy]["uses"] += 1
        self.learning_strategies[strategy]["total_improvement"] += improvement
        if improvement > 0:
            self.learning_strategies[strategy]["successes"] += 1

    def detect_transfer(self, source_domain: str, target_domain: str,
                       transfer_success: bool, transfer_amount: float):
        """Record transfer learning outcome"""
        key = (source_domain, target_domain)
        current = self.transfer_matrix[key]
        self.transfer_matrix[key] = current * 0.8 + (transfer_amount if transfer_success else 0) * 0.2

    def get_learning_strategy(self, domain: str, time_available: float) -> Dict[str, Any]:
        """Get recommended learning strategy for a domain"""
        # Check if we have relevant experience
        relevant_episodes = [
            ep for ep in self.learning_episodes
            if ep.domain == domain and ep.success
        ]

        if relevant_episodes:
            # Use successful past strategy
            best_episode = max(relevant_episodes, key=lambda e: e.final_performance - e.initial_performance)
            return {
                "strategy": best_episode.strategy_used,
                "expected_rate": self.domain_learning_rates[domain],
                "based_on": "past_success"
            }

        # Check for transferable knowledge
        best_transfer = None
        best_transfer_score = 0
        for (source, target), score in self.transfer_matrix.items():
            if target == domain and score > best_transfer_score:
                best_transfer = source
                best_transfer_score = score

        if best_transfer and best_transfer_score > 0.3:
            return {
                "strategy": "transfer_learning",
                "source_domain": best_transfer,
                "expected_transfer": best_transfer_score,
                "based_on": "transfer_potential"
            }

        # Default to exploration
        best_overall = max(
            self.learning_strategies.items(),
            key=lambda x: x[1]["total_improvement"] / max(1, x[1]["uses"]),
            default=("incremental_practice", {"uses": 0})
        )

        return {
            "strategy": best_overall[0],
            "expected_rate": 0.1,
            "based_on": "general_effectiveness"
        }

    def generate_curriculum(self, target_domain: str,
                           current_capabilities: Dict[str, float]) -> List[str]:
        """Generate learning curriculum using transfer knowledge"""
        curriculum = []

        # Find prerequisite domains (those that transfer well)
        prereqs = []
        for (source, target), score in self.transfer_matrix.items():
            if target == target_domain and score > 0.3:
                if source not in current_capabilities or current_capabilities[source] < 0.7:
                    prereqs.append((source, score))

        # Sort by transfer potential
        prereqs.sort(key=lambda x: -x[1])

        # Add prerequisites first
        for prereq, _ in prereqs[:3]:
            curriculum.append(f"prerequisite:{prereq}")

        # Add main domain
        curriculum.append(f"target:{target_domain}")

        # Add practice phases
        curriculum.append(f"practice:{target_domain}:basic")
        curriculum.append(f"practice:{target_domain}:intermediate")
        curriculum.append(f"practice:{target_domain}:advanced")

        self.curriculum = curriculum
        return curriculum

    def optimize_learning(self, available_time: float,
                         target_capabilities: Dict[str, float],
                         current_capabilities: Dict[str, float]) -> List[Dict[str, Any]]:
        """Optimize learning schedule"""
        schedule = []
        remaining_time = available_time

        for domain, target_level in sorted(
            target_capabilities.items(),
            key=lambda x: x[1] - current_capabilities.get(x[0], 0),
            reverse=True
        ):
            current = current_capabilities.get(domain, 0)
            gap = target_level - current

            if gap <= 0:
                continue

            # Estimate time needed
            learning_rate = self.domain_learning_rates[domain]
            estimated_time = gap / learning_rate

            if estimated_time <= remaining_time:
                strategy = self.get_learning_strategy(domain, estimated_time)
                schedule.append({
                    "domain": domain,
                    "current": current,
                    "target": target_level,
                    "estimated_time": estimated_time,
                    "strategy": strategy
                })
                remaining_time -= estimated_time

        return schedule


# ============================================================================
# INTEGRATED META-COGNITIVE SYSTEM
# ============================================================================

class MetaCognitiveSystem:
    """
    Integrated meta-cognitive system combining all components.
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.monitor = CognitiveMonitor()
        self.strategy_selector = StrategySelector()
        self.self_modeler = SelfModeler()
        self.meta_learner = MetaLearner()

        # Meta-level state
        self.meta_attention: str = "normal"  # What meta-cognition is focused on
        self.reflections: List[Dict[str, Any]] = []
        self.insights: List[str] = []

    def before_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Meta-cognitive processing before starting a task"""
        # Check cognitive state
        should_break, reason = self.monitor.should_take_break()
        if should_break:
            return {
                "action": "take_break",
                "reason": reason,
                "proceed": False
            }

        # Select strategy
        strategy_rec = self.strategy_selector.get_strategy_recommendation(
            task.get("type", "general")
        )

        # Predict performance
        prediction = self.self_modeler.predict_own_performance(task)

        # Adjust approach based on prediction
        if prediction["success_probability"] < 0.5:
            # Low confidence - consider alternatives
            limitations = prediction.get("limitations", [])
            return {
                "proceed": True,
                "strategy": strategy_rec,
                "prediction": prediction,
                "warnings": limitations,
                "recommendation": "Consider decomposing task or seeking help"
            }

        return {
            "proceed": True,
            "strategy": strategy_rec,
            "prediction": prediction,
            "confidence": prediction["confidence"]
        }

    def after_task(self, task: Dict[str, Any], outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Meta-cognitive processing after completing a task"""
        # Record performance
        self.monitor.record_task_outcome(
            task.get("id", "unknown"),
            outcome.get("success", False),
            outcome.get("duration", 1.0),
            task.get("difficulty", 0.5)
        )

        # Record strategy outcome
        strategy_used = task.get("strategy_used", "unknown")
        self.strategy_selector.record_outcome(
            task.get("type", "general"),
            strategy_used,
            outcome.get("success", False),
            outcome.get("duration", 1.0)
        )

        # Update self-model
        self.self_modeler.observe_behavior(
            {"domain": task.get("type", "general"), **task},
            task.get("action", "performed_task"),
            outcome
        )

        # Record learning if applicable
        if task.get("is_learning_task", False):
            self.meta_learner.record_learning(
                task.get("domain", "general"),
                task.get("initial_performance", 0.0),
                outcome.get("final_performance", 0.5),
                timedelta(seconds=outcome.get("duration", 60) * 60),
                strategy_used
            )

        # Generate reflection
        reflection = self._reflect(task, outcome)
        self.reflections.append(reflection)

        return {
            "reflection": reflection,
            "performance_summary": self.monitor.get_performance_summary(),
            "improvements_suggested": self.self_modeler.suggest_improvements()[:3]
        }

    def _reflect(self, task: Dict[str, Any], outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Generate reflection on task performance"""
        success = outcome.get("success", False)

        reflection = {
            "task": task.get("id", "unknown"),
            "success": success,
            "timestamp": datetime.now()
        }

        if success:
            reflection["lesson"] = "Strategy was effective for this task type"
            reflection["reinforcement"] = task.get("strategy_used", "unknown")
        else:
            # Analyze failure
            prediction = self.self_modeler.predict_own_performance(task)
            if prediction["success_probability"] > 0.5:
                reflection["lesson"] = "Overconfident - need to improve self-calibration"
                reflection["action"] = "collect_more_data"
            else:
                reflection["lesson"] = "Known difficulty - expected failure"
                reflection["action"] = "improve_capability"

        return reflection

    def introspect(self) -> Dict[str, Any]:
        """Deep introspection on current state"""
        return {
            "cognitive_state": self.monitor.get_performance_summary(),
            "self_model": {
                "capabilities": {
                    cap.domain: cap.proficiency
                    for cap in self.self_modeler.model.capabilities.values()
                },
                "limitations": [
                    lim.description
                    for lim in self.self_modeler.model.limitations.values()
                ],
                "energy": self.self_modeler.model.energy_level,
                "stress": self.self_modeler.model.stress_level
            },
            "meta_learning": {
                "learning_rates": dict(self.meta_learner.domain_learning_rates),
                "best_strategies": sorted(
                    self.meta_learner.learning_strategies.items(),
                    key=lambda x: x[1]["total_improvement"],
                    reverse=True
                )[:3]
            },
            "blind_spots": self.self_modeler.identify_blind_spots(),
            "insights": self.insights[-5:]
        }

    def generate_insight(self) -> Optional[str]:
        """Generate meta-cognitive insight from patterns"""
        # Analyze reflection patterns
        recent_reflections = self.reflections[-20:]
        if not recent_reflections:
            return None

        failure_count = sum(1 for r in recent_reflections if not r.get("success", True))
        success_count = len(recent_reflections) - failure_count

        if failure_count > success_count * 2:
            insight = "High failure rate detected - consider changing approach or taking a break"
            self.insights.append(insight)
            return insight

        # Check for strategy patterns
        strategy_successes = defaultdict(list)
        for r in recent_reflections:
            strategy = r.get("reinforcement", "unknown")
            strategy_successes[strategy].append(r.get("success", False))

        for strategy, outcomes in strategy_successes.items():
            if len(outcomes) >= 3:
                rate = sum(outcomes) / len(outcomes)
                if rate > 0.8:
                    insight = f"Strategy '{strategy}' is highly effective - use more often"
                    self.insights.append(insight)
                    return insight
                elif rate < 0.2:
                    insight = f"Strategy '{strategy}' is ineffective - avoid or modify"
                    self.insights.append(insight)
                    return insight

        return None


# Convenience functions
def create_metacognitive_system(agent_id: str) -> MetaCognitiveSystem:
    """Create a new meta-cognitive system"""
    return MetaCognitiveSystem(agent_id)


def demo_metacognition():
    """Demonstrate meta-cognitive system"""
    meta = create_metacognitive_system("agent_1")

    print("=== Meta-Cognition Demo ===")

    # Simulate some tasks
    for i in range(10):
        task = {
            "id": f"task_{i}",
            "type": random.choice(["reasoning", "planning", "memory"]),
            "difficulty": random.uniform(0.3, 0.8)
        }

        # Before task
        prep = meta.before_task(task)
        print(f"\nTask {i}: {task['type']}")
        print(f"  Strategy: {prep.get('strategy', {}).get('strategy', 'N/A')}")
        print(f"  Prediction: {prep.get('prediction', {}).get('success_probability', 'N/A'):.0%}")

        # Simulate outcome
        outcome = {
            "success": random.random() > 0.4,
            "duration": random.uniform(0.5, 2.0)
        }
        task["strategy_used"] = prep.get("strategy", {}).get("strategy_id", "heuristic")

        # After task
        result = meta.after_task(task, outcome)
        print(f"  Outcome: {'Success' if outcome['success'] else 'Failure'}")
        print(f"  Reflection: {result['reflection'].get('lesson', 'N/A')}")

    # Final introspection
    print("\n=== Introspection ===")
    intro = meta.introspect()
    print(f"Cognitive state: {intro['cognitive_state']['current_state']}")
    print(f"Capabilities: {intro['self_model']['capabilities']}")
    print(f"Blind spots: {intro['blind_spots']}")

    return meta


if __name__ == "__main__":
    demo_metacognition()
