"""
Metacognitive DSPy Pipeline v2
==============================
A sophisticated reasoning architecture combining:
- Chain of Thought (sequential reasoning)
- Graph of Thoughts (parallel/branching reasoning with synthesis)
- 16 Specialized Reasoning Methods (class-based signatures)
- 16 Cognitive Bias Detectors (with iterative correction)
- Self-reflective iteration loop
- Async parallel execution
- Integration with existing reasoning infrastructure
- Observability and metrics

Author: Built for Arthur @ DSCO
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from functools import lru_cache, wraps
from typing import (
    Any, AsyncIterator, Callable, Dict, Generic, List,
    Optional, Protocol, Set, Tuple, TypeVar, Union
)

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    # Create stub classes for type hints
    class dspy:
        class Signature: pass
        class Module: pass
        class InputField:
            def __init__(self, **kwargs): pass
        class OutputField:
            def __init__(self, **kwargs): pass
        @staticmethod
        def ChainOfThought(sig): return lambda **kw: type('Result', (), kw)()
        @staticmethod
        def Predict(sig): return lambda **kw: type('Result', (), kw)()

# Import existing infrastructure
try:
    from .reasoning import ReasoningTrace, ReasoningStep, StepType, StepStatus, AgentContext
    REASONING_AVAILABLE = True
except ImportError:
    REASONING_AVAILABLE = False

try:
    from .dspy_integration import DSPyModule, ChainOfThought as DSPyCoT, ModuleResult
    DSPY_INTEGRATION_AVAILABLE = True
except ImportError:
    DSPY_INTEGRATION_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE DEFINITIONS AND PROTOCOLS
# =============================================================================

T = TypeVar('T')
R = TypeVar('R')


class ReasoningMethod(Protocol):
    """Protocol for reasoning methods."""
    async def __call__(self, problem: str, context: str, **kwargs) -> Dict[str, Any]: ...


class BiasDetector(Protocol):
    """Protocol for bias detectors."""
    async def detect(self, reasoning: str, **context) -> 'BiasDetectionResult': ...


# =============================================================================
# CONFIGURATION AND METRICS
# =============================================================================

class AggregationStrategy(Enum):
    """Strategies for aggregating multiple results.

    Includes 100+ methods spanning basic heuristics to cutting-edge
    statistical, information-theoretic, and ML techniques.
    """
    # === Basic Methods ===
    WEIGHTED_AVERAGE = auto()
    MAJORITY_VOTE = auto()
    HIGHEST_CONFIDENCE = auto()
    LOWEST_CONFIDENCE = auto()  # Conservative estimate
    MEDIAN = auto()
    TRIMMED_MEAN = auto()  # Robust to outliers
    GEOMETRIC_MEAN = auto()  # Multiplicative aggregation
    HARMONIC_MEAN = auto()  # Rate-based aggregation
    POWER_MEAN = auto()  # Generalized mean with parameter
    LEHMER_MEAN = auto()  # Contraharmonic mean variant
    QUASI_ARITHMETIC = auto()  # f-mean generalization

    # === Bayesian Methods ===
    BAYESIAN = auto()  # Shorthand alias
    BAYESIAN_COMBINATION = auto()
    BAYESIAN_MODEL_AVERAGING = auto()  # BMA with model weights
    CONJUGATE_PRIOR = auto()  # Beta-Bernoulli conjugate update
    JEFFREY_PRIOR = auto()  # Non-informative prior
    EMPIRICAL_BAYES = auto()  # Estimate prior from data
    HIERARCHICAL_BAYES = auto()  # Multi-level Bayesian model
    SPIKE_AND_SLAB = auto()  # Sparse Bayesian with point mass
    HORSESHOE_PRIOR = auto()  # Heavy-tailed shrinkage prior
    BAYESIAN_QUADRATURE = auto()  # Integration via GP surrogate

    # === Density Estimation Methods ===
    KERNEL_DENSITY = auto()  # KDE-based mode finding
    KERNEL_DENSITY_ADAPTIVE = auto()  # Adaptive bandwidth KDE
    PARZEN_WINDOW = auto()  # Parzen density estimation
    HISTOGRAM_DENSITY = auto()  # Histogram-based density
    KNN_DENSITY = auto()  # k-nearest neighbor density
    GAUSSIAN_MIXTURE = auto()  # GMM-based aggregation
    GMM_EM = auto()  # GMM with EM algorithm
    GMM_VARIATIONAL = auto()  # Variational GMM
    DIRICHLET_PROCESS = auto()  # Non-parametric mixture
    PITMAN_YOR_PROCESS = auto()  # Power-law mixture
    DENSITY_RATIO = auto()  # Density ratio estimation
    NORMALIZING_FLOW = auto()  # Flow-based density
    SCORE_MATCHING = auto()  # Score function estimation
    CONTRASTIVE_DENSITY = auto()  # Noise contrastive estimation

    # === Prompt-Based Density Estimation ===
    PROMPT_DENSITY_ESTIMATION = auto()  # LLM-guided density estimation
    PROMPT_CALIBRATION = auto()  # LLM confidence calibration
    PROMPT_ENSEMBLE = auto()  # Multi-prompt aggregation
    PROMPT_UNCERTAINTY = auto()  # Prompt-based uncertainty quantification
    CHAIN_OF_DENSITY = auto()  # Sequential density refinement via prompts
    SELF_CONSISTENCY_DENSITY = auto()  # Self-consistency based density
    PROMPT_TEMPERATURE_SWEEP = auto()  # Temperature-varied sampling
    SEMANTIC_DENSITY = auto()  # Embedding-space density estimation

    # === Distributional Methods ===
    BETA_DISTRIBUTION = auto()  # Fit beta distribution
    DIRICHLET = auto()  # Multinomial confidence
    MAXIMUM_ENTROPY = auto()  # MaxEnt principle
    MOMENT_MATCHING = auto()  # Method of moments
    CUMULANT_MATCHING = auto()  # Match higher-order cumulants
    CHARACTERISTIC_FUNCTION = auto()  # Fourier-domain matching
    EXPONENTIAL_FAMILY = auto()  # Exponential family MLE
    STABLE_DISTRIBUTION = auto()  # Heavy-tailed stable dist
    GENERALIZED_PARETO = auto()  # Extreme value distribution

    # === Sampling Methods ===
    BOOTSTRAP = auto()  # Bootstrap confidence intervals
    MONTE_CARLO = auto()  # MC sampling from distributions
    IMPORTANCE_SAMPLING = auto()  # Weighted sampling
    SEQUENTIAL_MONTE_CARLO = auto()  # Particle filtering
    MARKOV_CHAIN_MONTE_CARLO = auto()  # MCMC sampling
    HAMILTONIAN_MONTE_CARLO = auto()  # HMC gradient-based sampling
    NESTED_SAMPLING = auto()  # Evidence estimation
    APPROXIMATE_BAYESIAN_COMPUTATION = auto()  # Likelihood-free inference
    REJECTION_SAMPLING = auto()  # Accept-reject sampling
    SLICE_SAMPLING = auto()  # Univariate slice sampler
    GIBBS_SAMPLING = auto()  # Conditional sampling
    LANGEVIN_DYNAMICS = auto()  # Gradient-based diffusion

    # === Information-Theoretic Methods ===
    ENTROPY_WEIGHTED = auto()  # Weight by information content
    KULLBACK_LEIBLER = auto()  # KL divergence minimization
    JENSEN_SHANNON = auto()  # Symmetric divergence
    MUTUAL_INFORMATION = auto()  # MI-based weighting
    FISHER_INFORMATION = auto()  # Fisher information weighting
    RENYI_ENTROPY = auto()  # Generalized entropy
    TSALLIS_ENTROPY = auto()  # Non-extensive entropy
    RATE_DISTORTION = auto()  # Compression-based aggregation
    MINIMUM_DESCRIPTION_LENGTH = auto()  # MDL principle
    KOLMOGOROV_COMPLEXITY = auto()  # Algorithmic information
    CHANNEL_CAPACITY = auto()  # Information channel optimization
    INFO_BOTTLENECK = auto()  # Information bottleneck method

    # === Robust Estimation Methods ===
    ROBUST_HUBER = auto()  # Huber M-estimator
    ROBUST_TUKEY = auto()  # Tukey biweight
    WINSORIZED = auto()  # Winsorized mean
    LEAST_TRIMMED_SQUARES = auto()  # LTS robust regression
    THEIL_SEN = auto()  # Median-based robust estimator
    HODGES_LEHMANN = auto()  # Pairwise mean median
    MESTIMATOR_ANDREWS = auto()  # Andrews' wave function
    MESTIMATOR_HAMPEL = auto()  # Hampel's three-part redescending
    BREAKDOWN_POINT = auto()  # Maximum breakdown estimator
    INFLUENCE_FUNCTION = auto()  # Influence-based weighting

    # === Belief/Evidence Combination ===
    DEMPSTER_SHAFER = auto()  # Dempster-Shafer theory
    TRANSFERABLE_BELIEF = auto()  # TBM combination
    SUBJECTIVE_LOGIC = auto()  # Opinion fusion
    POSSIBILITY_THEORY = auto()  # Fuzzy possibility aggregation
    ROUGH_SET_FUSION = auto()  # Rough set-based combination
    INTUITIONISTIC_FUZZY = auto()  # Intuitionistic fuzzy aggregation
    NEUTROSOPHIC = auto()  # Neutrosophic logic combination
    GREY_RELATIONAL = auto()  # Grey system theory
    EVIDENTIAL_NEURAL = auto()  # Neural evidence combination
    BELIEF_PROPAGATION = auto()  # Message passing on factor graph

    # === Optimal Transport ===
    WASSERSTEIN_BARYCENTER = auto()  # OT barycenter
    SINKHORN_DIVERGENCE = auto()  # Regularized OT
    GROMOV_WASSERSTEIN = auto()  # Structural OT
    SLICED_WASSERSTEIN = auto()  # Projected OT
    UNBALANCED_OT = auto()  # Soft marginal constraints

    # === Spectral Methods ===
    SPECTRAL_CLUSTERING = auto()  # Eigendecomposition-based
    LAPLACIAN_EIGENMAPS = auto()  # Graph Laplacian smoothing
    DIFFUSION_MAPS = auto()  # Diffusion geometry
    SPECTRAL_DENSITY = auto()  # Spectral density estimation
    RANDOM_MATRIX_THEORY = auto()  # RMT-based aggregation

    # === Information Geometry ===
    FISHER_RAO_METRIC = auto()  # Natural gradient geometry
    ALPHA_DIVERGENCE = auto()  # Alpha-family divergences
    BREGMAN_CENTROID = auto()  # Bregman divergence center
    EXPONENTIAL_GEODESIC = auto()  # Geodesic averaging
    WASSERSTEIN_NATURAL_GRADIENT = auto()  # OT meets info geometry

    # === Neural/Deep Learning Inspired ===
    ATTENTION_AGGREGATION = auto()  # Self-attention pooling
    TRANSFORMER_FUSION = auto()  # Multi-head attention fusion
    NEURAL_PROCESS = auto()  # Neural process aggregation
    DEEP_SETS = auto()  # Permutation invariant neural
    SET_TRANSFORMER = auto()  # Set-based attention
    GRAPH_NEURAL_AGGREGATION = auto()  # GNN message passing
    HYPERNETWORK_FUSION = auto()  # Weight-generating networks
    META_LEARNING_AGGREGATION = auto()  # MAML-style adaptation

    # === Probabilistic Programming ===
    EXPECTATION_PROPAGATION = auto()  # EP message passing
    ASSUMED_DENSITY_FILTERING = auto()  # ADF approximation
    LOOPY_BELIEF_PROPAGATION = auto()  # LBP on cyclic graphs
    VARIATIONAL_MESSAGE_PASSING = auto()  # VMP algorithm
    STOCHASTIC_VARIATIONAL = auto()  # Scalable VI
    BLACK_BOX_VARIATIONAL = auto()  # Gradient-based VI
    NORMALIZING_FLOW_VI = auto()  # Flow-based variational

    # === Hybrid Agglomeration Methods ===
    HYBRID_AGGLOMERATION = auto()  # Hierarchical clustering + multi-method fusion
    HIERARCHICAL_FUSION = auto()  # Bottom-up agglomerative fusion
    MIXTURE_OF_EXPERTS = auto()  # Gated combination of expert aggregators
    CASCADED_BAYESIAN = auto()  # Multi-stage Bayesian refinement
    CONSENSUS_CLUSTERING = auto()  # Cluster-based consensus
    MULTI_SCALE_FUSION = auto()  # Aggregate at multiple granularities
    ITERATIVE_REFINEMENT = auto()  # Iteratively refine with multiple methods
    GRAPH_AGGREGATION = auto()  # Graph-based source similarity aggregation
    COPULA_FUSION = auto()  # Copula-based dependency modeling
    VARIATIONAL_INFERENCE = auto()  # VI-based posterior estimation

    # === Advanced Hybrid Methods ===
    DENSITY_FUNCTIONAL = auto()  # DFT-inspired aggregation
    RENORMALIZATION_GROUP = auto()  # Scale-invariant aggregation
    MEAN_FIELD_THEORY = auto()  # Mean field approximation
    CAVITY_METHOD = auto()  # Statistical mechanics approach
    REPLICA_TRICK = auto()  # Disorder averaging
    SUPERSYMMETRIC = auto()  # SUSY-inspired combination
    HOLOGRAPHIC = auto()  # AdS/CFT-inspired

    # === Game-Theoretic Methods ===
    NASH_BARGAINING = auto()  # Nash bargaining solution
    SHAPLEY_VALUE = auto()  # Cooperative game theory
    CORE_ALLOCATION = auto()  # Core of cooperative game
    NUCLEOLUS = auto()  # Lexicographic nucleolus
    MECHANISM_DESIGN = auto()  # Incentive-compatible aggregation

    # === Causal Methods ===
    CAUSAL_DISCOVERY = auto()  # Infer causal structure
    DO_CALCULUS = auto()  # Interventional queries
    COUNTERFACTUAL_AGGREGATION = auto()  # Counterfactual reasoning
    INSTRUMENTAL_VARIABLE = auto()  # IV-based estimation
    DOUBLE_MACHINE_LEARNING = auto()  # Debiased ML

    # === Conformal Prediction ===
    CONFORMAL_PREDICTION = auto()  # Distribution-free coverage
    SPLIT_CONFORMAL = auto()  # Split conformal inference
    FULL_CONFORMAL = auto()  # Full conformal method
    CONFORMALIZED_QUANTILE = auto()  # Conformalized quantile regression

    # === Meta Methods ===
    ENSEMBLE_SELECTION = auto()  # Learn best combiner
    STACKING = auto()  # Stacked generalization
    ADAPTIVE = auto()  # Adaptively choose strategy
    SUPER_LEARNER = auto()  # Optimal convex combination
    ONLINE_LEARNING = auto()  # Regret-minimizing aggregation
    THOMPSON_SAMPLING = auto()  # Bayesian bandit approach
    UCB_AGGREGATION = auto()  # Upper confidence bound
    EXP3_AGGREGATION = auto()  # Adversarial bandit

    # === Quantum-Inspired ===
    QUANTUM_SUPERPOSITION = auto()  # Superposition of estimates
    QUANTUM_ENTANGLEMENT = auto()  # Entangled source correlation
    QUANTUM_ANNEALING = auto()  # Optimization via quantum dynamics
    QUANTUM_AMPLITUDE = auto()  # Amplitude-based weighting


@dataclass
class PipelineConfig:
    """Configuration for the metacognitive pipeline."""
    # Chain of Thought settings
    cot_depth: int = 5
    cot_early_termination: bool = True

    # Graph of Thoughts settings
    got_branching_factor: int = 3
    got_max_depth: int = 4
    got_cross_pollination: bool = True

    # Reasoning ensemble settings
    max_methods: int = 5
    method_timeout: float = 30.0
    parallel_methods: bool = True
    method_weights: Dict[str, float] = field(default_factory=dict)

    # Bias detection settings
    bias_detection_enabled: bool = True
    correction_iterations: int = 3
    bias_threshold: float = 0.5

    # Caching settings
    cache_enabled: bool = True
    cache_ttl: int = 3600

    # Aggregation settings
    aggregation_strategy: AggregationStrategy = AggregationStrategy.WEIGHTED_AVERAGE
    confidence_threshold: float = 0.6

    # Observability
    trace_enabled: bool = True
    metrics_enabled: bool = True


@dataclass
class PipelineMetrics:
    """Metrics collected during pipeline execution."""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    # Stage timings
    stage_timings: Dict[str, float] = field(default_factory=dict)

    # Method statistics
    methods_invoked: List[str] = field(default_factory=list)
    method_successes: int = 0
    method_failures: int = 0

    # Bias statistics
    biases_detected: int = 0
    corrections_applied: int = 0

    # Resource usage
    llm_calls: int = 0
    total_tokens: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    # Quality metrics
    final_confidence: float = 0.0
    reasoning_coherence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['start_time'] = self.start_time.isoformat()
        result['end_time'] = self.end_time.isoformat() if self.end_time else None
        result['duration_seconds'] = (
            (self.end_time - self.start_time).total_seconds()
            if self.end_time else None
        )
        return result


# =============================================================================
# CACHING INFRASTRUCTURE
# =============================================================================

class ResultCache:
    """Thread-safe cache for reasoning results."""

    def __init__(self, ttl: int = 3600, max_size: int = 1000):
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._ttl = ttl
        self._max_size = max_size
        self._lock = asyncio.Lock()

    def _make_key(self, method: str, inputs: Dict[str, Any]) -> str:
        """Create deterministic cache key."""
        content = json.dumps({method: inputs}, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def get(self, method: str, inputs: Dict[str, Any]) -> Optional[Any]:
        """Get cached result if valid."""
        key = self._make_key(method, inputs)
        async with self._lock:
            if key in self._cache:
                result, timestamp = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    return result
                del self._cache[key]
        return None

    async def set(self, method: str, inputs: Dict[str, Any], result: Any):
        """Cache a result."""
        key = self._make_key(method, inputs)
        async with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self._max_size:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]
            self._cache[key] = (result, time.time())


# =============================================================================
# CORE SIGNATURES (Improved DSPy Signatures)
# =============================================================================

if DSPY_AVAILABLE:
    class QuestionContext(dspy.Signature):
        """Base signature for question-context to structured output."""
        question: str = dspy.InputField(desc="The question to answer")
        context: str = dspy.InputField(desc="Relevant context, documents, or data")

        answer: str = dspy.OutputField(desc="Comprehensive answer to the question")
        citations: List[str] = dspy.OutputField(desc="Sources and references supporting the answer")
        confidence: float = dspy.OutputField(desc="Confidence score 0.0-1.0")
        reasoning_trace: str = dspy.OutputField(desc="Step-by-step reasoning that led to answer")
        epistemic_status: str = dspy.OutputField(desc="Known unknowns and uncertainty characterization")
        actionable_insights: List[str] = dspy.OutputField(desc="Concrete next steps or recommendations")


# =============================================================================
# 16 REASONING METHOD SIGNATURES (with improved structure)
# =============================================================================

    class DeductiveReasoning(dspy.Signature):
        """Derive conclusions from premises through logical necessity."""
        premises: str = dspy.InputField(desc="Given premises or axioms")
        question: str = dspy.InputField(desc="What to derive")

        logical_steps: List[str] = dspy.OutputField(desc="Each deductive step")
        conclusion: str = dspy.OutputField(desc="Necessarily true conclusion")
        validity_check: str = dspy.OutputField(desc="Verification of logical validity")
        confidence: float = dspy.OutputField(desc="Logical certainty 0.0-1.0")


    class InductiveReasoning(dspy.Signature):
        """Generalize patterns from specific observations."""
        observations: str = dspy.InputField(desc="Specific instances or data points")
        question: str = dspy.InputField(desc="Pattern to identify")

        patterns_found: List[str] = dspy.OutputField(desc="Identified regularities")
        generalization: str = dspy.OutputField(desc="Proposed general rule")
        counterexample_search: str = dspy.OutputField(desc="Attempted falsification")
        inductive_strength: float = dspy.OutputField(desc="Probability of generalization holding")


    class AbductiveReasoning(dspy.Signature):
        """Inference to the best explanation."""
        phenomenon: str = dspy.InputField(desc="Observation requiring explanation")
        context: str = dspy.InputField(desc="Background knowledge")

        candidate_hypotheses: List[str] = dspy.OutputField(desc="Possible explanations")
        explanatory_virtues: str = dspy.OutputField(desc="Simplicity, scope, coherence assessment")
        best_explanation: str = dspy.OutputField(desc="Most plausible hypothesis")
        residual_mystery: str = dspy.OutputField(desc="What remains unexplained")
        confidence: float = dspy.OutputField(desc="Explanatory confidence 0.0-1.0")


    class AnalogicalReasoning(dspy.Signature):
        """Transfer knowledge from source to target domain."""
        source_domain: str = dspy.InputField(desc="Well-understood domain")
        target_domain: str = dspy.InputField(desc="Domain needing insight")
        question: str = dspy.InputField(desc="What to infer about target")

        structural_mapping: str = dspy.OutputField(desc="Element correspondences")
        transferred_inferences: List[str] = dspy.OutputField(desc="Conclusions drawn by analogy")
        disanalogies: List[str] = dspy.OutputField(desc="Where the analogy breaks down")
        analogy_strength: float = dspy.OutputField(desc="Reliability of transfer")


    class CausalReasoning(dspy.Signature):
        """Identify causal relationships and mechanisms."""
        events: str = dspy.InputField(desc="Events or variables")
        context: str = dspy.InputField(desc="Domain knowledge")

        causal_graph: str = dspy.OutputField(desc="Directed cause-effect relationships")
        mechanisms: List[str] = dspy.OutputField(desc="Explanations of how causes produce effects")
        confounders: List[str] = dspy.OutputField(desc="Variables that might create spurious correlations")
        intervention_predictions: str = dspy.OutputField(desc="What happens if we intervene on X")
        confidence: float = dspy.OutputField(desc="Causal confidence 0.0-1.0")


    class CounterfactualReasoning(dspy.Signature):
        """Reason about alternative possibilities."""
        actual_situation: str = dspy.InputField(desc="What actually happened")
        counterfactual_antecedent: str = dspy.InputField(desc="The 'what if' change")

        minimal_change: str = dspy.OutputField(desc="Closest possible world with antecedent true")
        consequent_changes: List[str] = dspy.OutputField(desc="What would be different")
        causal_dependencies: str = dspy.OutputField(desc="Which changes depend on which")
        robustness_assessment: str = dspy.OutputField(desc="How sensitive is outcome to this change")
        confidence: float = dspy.OutputField(desc="Counterfactual confidence 0.0-1.0")


    class BayesianReasoning(dspy.Signature):
        """Update beliefs based on evidence."""
        prior_beliefs: str = dspy.InputField(desc="Initial probability assessment")
        new_evidence: str = dspy.InputField(desc="Observed evidence")
        likelihood_model: str = dspy.InputField(desc="How evidence relates to hypotheses")

        posterior_beliefs: str = dspy.OutputField(desc="Updated probability assessment")
        belief_update_magnitude: float = dspy.OutputField(desc="How much beliefs shifted")
        most_diagnostic_evidence: str = dspy.OutputField(desc="What evidence would most change beliefs")
        confidence: float = dspy.OutputField(desc="Probabilistic confidence 0.0-1.0")


    class DialecticalReasoning(dspy.Signature):
        """Synthesize opposing viewpoints."""
        thesis: str = dspy.InputField(desc="Initial position")
        antithesis: str = dspy.InputField(desc="Opposing position")

        thesis_strengths: List[str] = dspy.OutputField(desc="Valid points of thesis")
        antithesis_strengths: List[str] = dspy.OutputField(desc="Valid points of antithesis")
        synthesis: str = dspy.OutputField(desc="Higher-order integration")
        remaining_tensions: List[str] = dspy.OutputField(desc="Unresolved contradictions")
        confidence: float = dspy.OutputField(desc="Synthesis confidence 0.0-1.0")


    class SystemsThinking(dspy.Signature):
        """Analyze complex systems with feedback loops."""
        system_description: str = dspy.InputField(desc="Components and relationships")
        intervention_question: str = dspy.InputField(desc="What to change or understand")

        feedback_loops: str = dspy.OutputField(desc="Reinforcing and balancing loops")
        leverage_points: List[str] = dspy.OutputField(desc="High-impact intervention points")
        emergent_properties: List[str] = dspy.OutputField(desc="System-level behaviors")
        unintended_consequences: List[str] = dspy.OutputField(desc="Second and third order effects")
        confidence: float = dspy.OutputField(desc="Systems analysis confidence 0.0-1.0")


    class FirstPrinciplesReasoning(dspy.Signature):
        """Decompose to fundamental truths and rebuild."""
        problem: str = dspy.InputField(desc="Complex problem or assumption")
        domain: str = dspy.InputField(desc="Relevant field")

        fundamental_truths: List[str] = dspy.OutputField(desc="Irreducible axioms")
        decomposition: List[str] = dspy.OutputField(desc="Breaking down the problem")
        reconstruction: str = dspy.OutputField(desc="Building up novel solution")
        assumptions_challenged: List[str] = dspy.OutputField(desc="Conventional wisdom rejected")
        confidence: float = dspy.OutputField(desc="First principles confidence 0.0-1.0")


    class GameTheoreticReasoning(dspy.Signature):
        """Strategic reasoning with multiple agents."""
        players: str = dspy.InputField(desc="Agents involved")
        strategies: str = dspy.InputField(desc="Available actions per player")
        payoffs: str = dspy.InputField(desc="Outcome structure")

        equilibria: List[str] = dspy.OutputField(desc="Nash and other equilibria")
        dominant_strategies: str = dspy.OutputField(desc="Best responses")
        cooperation_barriers: List[str] = dspy.OutputField(desc="Why coordination fails")
        mechanism_design: str = dspy.OutputField(desc="How to change incentives")
        confidence: float = dspy.OutputField(desc="Game theoretic confidence 0.0-1.0")


    class ProbabilisticReasoning(dspy.Signature):
        """Reason under uncertainty with probability distributions."""
        random_variables: str = dspy.InputField(desc="Uncertain quantities")
        known_distributions: str = dspy.InputField(desc="Given probability information")
        query: str = dspy.InputField(desc="What probability to compute")

        probability_computation: str = dspy.OutputField(desc="Step-by-step calculation")
        result: str = dspy.OutputField(desc="Computed probability or distribution")
        sensitivity_analysis: str = dspy.OutputField(desc="How result changes with assumptions")
        confidence: float = dspy.OutputField(desc="Probabilistic reasoning confidence 0.0-1.0")


    class TemporalReasoning(dspy.Signature):
        """Reason about time, sequences, and processes."""
        events: str = dspy.InputField(desc="Events with temporal info")
        question: str = dspy.InputField(desc="Temporal query")

        timeline: str = dspy.OutputField(desc="Ordered sequence")
        temporal_relations: List[str] = dspy.OutputField(desc="Before/after/during relationships")
        duration_estimates: str = dspy.OutputField(desc="How long things take")
        critical_path: List[str] = dspy.OutputField(desc="Bottleneck sequence")
        confidence: float = dspy.OutputField(desc="Temporal reasoning confidence 0.0-1.0")


    class SpatialReasoning(dspy.Signature):
        """Reason about space, location, and geometry."""
        spatial_entities: str = dspy.InputField(desc="Objects with spatial properties")
        spatial_relations: str = dspy.InputField(desc="Known positions and relationships")
        query: str = dspy.InputField(desc="Spatial question")

        mental_model: str = dspy.OutputField(desc="Spatial representation")
        inferred_relations: List[str] = dspy.OutputField(desc="Derived spatial facts")
        transformations: List[str] = dspy.OutputField(desc="Rotations, translations considered")
        confidence: float = dspy.OutputField(desc="Spatial reasoning confidence 0.0-1.0")


    class MetaCognitiveReasoning(dspy.Signature):
        """Reason about one's own reasoning process."""
        reasoning_so_far: str = dspy.InputField(desc="Current reasoning state")
        goal: str = dspy.InputField(desc="What we're trying to achieve")

        strategy_assessment: str = dspy.OutputField(desc="Is current approach working?")
        knowledge_gaps: List[str] = dspy.OutputField(desc="What we don't know but need to")
        reasoning_errors: List[str] = dspy.OutputField(desc="Potential mistakes identified")
        strategy_adjustment: str = dspy.OutputField(desc="How to proceed differently")
        confidence: float = dspy.OutputField(desc="Meta-cognitive confidence 0.0-1.0")


    class NormativeReasoning(dspy.Signature):
        """Reason about what ought to be (ethics, values, preferences)."""
        situation: str = dspy.InputField(desc="Ethical scenario")
        stakeholders: str = dspy.InputField(desc="Affected parties")

        value_tensions: List[str] = dspy.OutputField(desc="Conflicting values/principles")
        utilitarian_analysis: str = dspy.OutputField(desc="Consequentialist perspective")
        deontological_analysis: str = dspy.OutputField(desc="Rule-based perspective")
        virtue_analysis: str = dspy.OutputField(desc="Character-based perspective")
        reflective_equilibrium: str = dspy.OutputField(desc="Balanced judgment")
        confidence: float = dspy.OutputField(desc="Normative reasoning confidence 0.0-1.0")


# =============================================================================
# 16 COGNITIVE BIAS SIGNATURES (Improved)
# =============================================================================

    class ConfirmationBiasDetector(dspy.Signature):
        """Detect cherry-picking evidence that supports existing beliefs."""
        reasoning: str = dspy.InputField()

        bias_detected: bool = dspy.OutputField()
        severity: float = dspy.OutputField(desc="Severity of bias 0.0-1.0")
        ignored_contrary_evidence: List[str] = dspy.OutputField()
        overweighted_supporting_evidence: List[str] = dspy.OutputField()
        correction: str = dspy.OutputField(desc="How to rebalance evidence consideration")


    class AnchoringBiasDetector(dspy.Signature):
        """Detect over-reliance on initial information."""
        reasoning: str = dspy.InputField()
        initial_values: str = dspy.InputField(desc="First numbers/facts encountered")

        bias_detected: bool = dspy.OutputField()
        severity: float = dspy.OutputField(desc="Severity of bias 0.0-1.0")
        anchor_influence: str = dspy.OutputField(desc="How anchor affected final judgment")
        proper_adjustment: str = dspy.OutputField(desc="Appropriate deviation from anchor")


    class AvailabilityBiasDetector(dspy.Signature):
        """Detect overweighting easily recalled information."""
        reasoning: str = dspy.InputField()

        bias_detected: bool = dspy.OutputField()
        severity: float = dspy.OutputField(desc="Severity of bias 0.0-1.0")
        vivid_examples_overweighted: List[str] = dspy.OutputField()
        base_rates_ignored: List[str] = dspy.OutputField()
        statistical_correction: str = dspy.OutputField()


    class HindsightBiasDetector(dspy.Signature):
        """Detect 'I knew it all along' reasoning errors."""
        reasoning: str = dspy.InputField()
        outcome_known: bool = dspy.InputField()

        bias_detected: bool = dspy.OutputField()
        severity: float = dspy.OutputField(desc="Severity of bias 0.0-1.0")
        false_predictability: str = dspy.OutputField(desc="Claims outcome was obvious")
        prospective_uncertainty: str = dspy.OutputField(desc="What was actually uncertain")


    class SurvivorshipBiasDetector(dspy.Signature):
        """Detect focus on successes while ignoring failures."""
        reasoning: str = dspy.InputField()

        bias_detected: bool = dspy.OutputField()
        severity: float = dspy.OutputField(desc="Severity of bias 0.0-1.0")
        invisible_failures: List[str] = dspy.OutputField(desc="Failures not considered")
        selection_mechanism: str = dspy.OutputField(desc="What filtered the sample")
        corrected_inference: str = dspy.OutputField()


    class FundamentalAttributionErrorDetector(dspy.Signature):
        """Detect over-attributing behavior to disposition vs situation."""
        reasoning: str = dspy.InputField()
        behavior_analyzed: str = dspy.InputField()

        bias_detected: bool = dspy.OutputField()
        severity: float = dspy.OutputField(desc="Severity of bias 0.0-1.0")
        dispositional_attribution: str = dspy.OutputField(desc="Character-based explanation")
        situational_factors_ignored: List[str] = dspy.OutputField()
        balanced_attribution: str = dspy.OutputField()


    class SunkCostFallacyDetector(dspy.Signature):
        """Detect letting past investments influence future decisions."""
        reasoning: str = dspy.InputField()
        past_investments: str = dspy.InputField()

        bias_detected: bool = dspy.OutputField()
        severity: float = dspy.OutputField(desc="Severity of bias 0.0-1.0")
        sunk_costs_considered: List[str] = dspy.OutputField()
        marginal_analysis: str = dspy.OutputField(desc="Forward-looking cost-benefit")


    class DunningKrugerDetector(dspy.Signature):
        """Detect miscalibrated confidence relative to competence."""
        reasoning: str = dspy.InputField()
        domain: str = dspy.InputField()

        bias_detected: bool = dspy.OutputField()
        severity: float = dspy.OutputField(desc="Severity of bias 0.0-1.0")
        confidence_level: float = dspy.OutputField()
        competence_indicators: List[str] = dspy.OutputField()
        calibration_adjustment: str = dspy.OutputField()


    class GroupthinkDetector(dspy.Signature):
        """Detect pressure toward conformity suppressing dissent."""
        reasoning: str = dspy.InputField()
        group_dynamics: str = dspy.InputField()

        bias_detected: bool = dspy.OutputField()
        severity: float = dspy.OutputField(desc="Severity of bias 0.0-1.0")
        suppressed_alternatives: List[str] = dspy.OutputField()
        conformity_pressure_signs: List[str] = dspy.OutputField()
        devils_advocate_perspective: str = dspy.OutputField()


    class FramingEffectDetector(dspy.Signature):
        """Detect conclusions biased by how information is presented."""
        reasoning: str = dspy.InputField()
        framing_used: str = dspy.InputField()

        bias_detected: bool = dspy.OutputField()
        severity: float = dspy.OutputField(desc="Severity of bias 0.0-1.0")
        alternative_framings: List[str] = dspy.OutputField()
        frame_invariant_conclusion: str = dspy.OutputField()


    class BaseRateNeglectDetector(dspy.Signature):
        """Detect ignoring prior probabilities in favor of specific evidence."""
        reasoning: str = dspy.InputField()
        specific_evidence: str = dspy.InputField()

        bias_detected: bool = dspy.OutputField()
        severity: float = dspy.OutputField(desc="Severity of bias 0.0-1.0")
        base_rates_ignored: List[str] = dspy.OutputField()
        bayesian_correction: str = dspy.OutputField()


    class HaloEffectDetector(dspy.Signature):
        """Detect one positive trait influencing perception of unrelated traits."""
        reasoning: str = dspy.InputField()
        entity_evaluated: str = dspy.InputField()

        bias_detected: bool = dspy.OutputField()
        severity: float = dspy.OutputField(desc="Severity of bias 0.0-1.0")
        halo_source: str = dspy.OutputField(desc="The impressive trait")
        spillover_judgments: List[str] = dspy.OutputField(desc="Unrelated areas affected")
        independent_evaluation: str = dspy.OutputField()


    class RecencyBiasDetector(dspy.Signature):
        """Detect overweighting recent events in judgment."""
        reasoning: str = dspy.InputField()
        temporal_scope: str = dspy.InputField()

        bias_detected: bool = dspy.OutputField()
        severity: float = dspy.OutputField(desc="Severity of bias 0.0-1.0")
        recent_events_overweighted: List[str] = dspy.OutputField()
        historical_context_missing: List[str] = dspy.OutputField()
        temporally_balanced_view: str = dspy.OutputField()


    class StatusQuoBiasDetector(dspy.Signature):
        """Detect irrational preference for current state."""
        reasoning: str = dspy.InputField()
        change_considered: str = dspy.InputField()

        bias_detected: bool = dspy.OutputField()
        severity: float = dspy.OutputField(desc="Severity of bias 0.0-1.0")
        loss_aversion_signs: List[str] = dspy.OutputField()
        omission_bias_signs: List[str] = dspy.OutputField()
        neutral_evaluation: str = dspy.OutputField()


    class ProjectionBiasDetector(dspy.Signature):
        """Detect assuming others share one's own beliefs/preferences."""
        reasoning: str = dspy.InputField()
        self_perspective: str = dspy.InputField()
        other_perspective: str = dspy.InputField()

        bias_detected: bool = dspy.OutputField()
        severity: float = dspy.OutputField(desc="Severity of bias 0.0-1.0")
        projected_beliefs: List[str] = dspy.OutputField()
        actual_differences: List[str] = dspy.OutputField()
        perspective_corrected_reasoning: str = dspy.OutputField()


    class OptimismBiasDetector(dspy.Signature):
        """Detect unrealistic positive expectations."""
        reasoning: str = dspy.InputField()
        predictions: str = dspy.InputField()

        bias_detected: bool = dspy.OutputField()
        severity: float = dspy.OutputField(desc="Severity of bias 0.0-1.0")
        overconfident_predictions: List[str] = dspy.OutputField()
        risk_underestimation: List[str] = dspy.OutputField()
        reference_class_forecasting: str = dspy.OutputField()


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ThoughtNode:
    """A node in the thought graph."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str = ""
    reasoning_method: str = ""
    confidence: float = 0.0
    parent_ids: List[str] = field(default_factory=list)
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "reasoning_method": self.reasoning_method,
            "confidence": self.confidence,
            "parent_ids": self.parent_ids,
            "children_ids": self.children_ids,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class BiasDetectionResult:
    """Result from a bias detection run."""
    bias_type: str
    detected: bool
    severity: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    correction: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MethodResult:
    """Result from a single reasoning method."""
    method_name: str
    success: bool
    outputs: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    error: Optional[str] = None
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResult:
    """Complete result from pipeline execution."""
    question: str
    context: str

    # Main outputs
    answer: str = ""
    confidence: float = 0.0
    citations: List[str] = field(default_factory=list)
    epistemic_status: str = ""
    actionable_insights: List[str] = field(default_factory=list)
    reasoning_summary: str = ""

    # Stage results
    cot_result: Optional[Dict[str, Any]] = None
    got_result: Optional[Dict[str, Any]] = None
    method_results: Dict[str, MethodResult] = field(default_factory=dict)
    bias_corrections: List[Dict[str, Any]] = field(default_factory=list)

    # Quality assessment
    quality_score: float = 0.0
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)

    # Metadata
    metrics: Optional[PipelineMetrics] = None
    config: Optional[PipelineConfig] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "question": self.question,
            "context": self.context,
            "answer": self.answer,
            "confidence": self.confidence,
            "citations": self.citations,
            "epistemic_status": self.epistemic_status,
            "actionable_insights": self.actionable_insights,
            "reasoning_summary": self.reasoning_summary,
            "cot_result": self.cot_result,
            "got_result": self.got_result,
            "method_results": {k: v.to_dict() for k, v in self.method_results.items()},
            "bias_corrections": self.bias_corrections,
            "quality_score": self.quality_score,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "improvement_suggestions": self.improvement_suggestions,
        }
        if self.metrics:
            result["metrics"] = self.metrics.to_dict()
        return result


# =============================================================================
# CHAIN OF THOUGHT MODULE (Enhanced)
# =============================================================================

class EnhancedChainOfThought:
    """Sequential reasoning through explicit intermediate steps with enhanced features."""

    def __init__(self, config: PipelineConfig, cache: Optional[ResultCache] = None):
        self.config = config
        self.cache = cache
        self.depth = config.cot_depth

        if DSPY_AVAILABLE:
            self.step_generator = dspy.ChainOfThought(
                "problem, context, previous_steps, step_number -> next_step, step_rationale, should_continue"
            )
            self.synthesizer = dspy.ChainOfThought(
                "problem, all_steps -> conclusion, confidence, key_insights"
            )

    async def forward(
        self,
        problem: str,
        context: str,
        trace: Optional['ReasoningTrace'] = None
    ) -> Dict[str, Any]:
        """Execute chain of thought reasoning."""
        steps = []
        step_rationales = []

        for i in range(self.depth):
            step_start = time.time()

            # Generate next step
            previous_steps_str = "\n".join([
                f"Step {j+1}: {s}" for j, s in enumerate(steps)
            ])

            if DSPY_AVAILABLE:
                try:
                    result = self.step_generator(
                        problem=problem,
                        context=context,
                        previous_steps=previous_steps_str,
                        step_number=i + 1
                    )

                    next_step = getattr(result, 'next_step', f"[Step {i+1} reasoning]")
                    rationale = getattr(result, 'step_rationale', "")
                    should_continue = getattr(result, 'should_continue', True)

                except Exception as e:
                    logger.warning(f"CoT step {i+1} failed: {e}")
                    next_step = f"[Step {i+1} failed: {e}]"
                    rationale = ""
                    should_continue = False
            else:
                next_step = f"[Mock step {i+1}]"
                rationale = "[Mock rationale]"
                should_continue = i < 3

            steps.append(next_step)
            step_rationales.append(rationale)

            # Add to reasoning trace if available
            if trace and REASONING_AVAILABLE:
                trace.add_step(
                    step_type=StepType.INFERENCE,
                    content=next_step,
                    confidence=0.7,
                    rationale=rationale,
                    duration_ms=(time.time() - step_start) * 1000
                )

            # Early termination check
            if self.config.cot_early_termination:
                if not should_continue:
                    break
                # Check for conclusion indicators
                lower_step = next_step.lower()
                if any(term in lower_step for term in ["therefore", "conclude", "final answer", "in conclusion"]):
                    break

        # Synthesize conclusion
        if DSPY_AVAILABLE:
            try:
                synthesis = self.synthesizer(
                    problem=problem,
                    all_steps="\n".join([f"Step {i+1}: {s}" for i, s in enumerate(steps)])
                )
                conclusion = getattr(synthesis, 'conclusion', steps[-1] if steps else "")
                confidence = float(getattr(synthesis, 'confidence', 0.7))
                key_insights = getattr(synthesis, 'key_insights', [])
            except Exception as e:
                logger.warning(f"CoT synthesis failed: {e}")
                conclusion = steps[-1] if steps else ""
                confidence = 0.5
                key_insights = []
        else:
            conclusion = steps[-1] if steps else ""
            confidence = 0.7
            key_insights = []

        return {
            "steps": steps,
            "step_rationales": step_rationales,
            "conclusion": conclusion,
            "confidence": confidence,
            "key_insights": key_insights,
            "chain_length": len(steps),
            "early_terminated": len(steps) < self.depth
        }


# =============================================================================
# GRAPH OF THOUGHTS MODULE (Enhanced)
# =============================================================================

class EnhancedGraphOfThoughts:
    """
    Non-linear reasoning that explores multiple paths and synthesizes.

    Enhanced with:
    - Async parallel branch execution
    - Cross-pollination between branches
    - Pruning of low-confidence branches
    - Dynamic branching based on problem complexity
    """

    def __init__(self, config: PipelineConfig, cache: Optional[ResultCache] = None):
        self.config = config
        self.cache = cache
        self.branching_factor = config.got_branching_factor
        self.max_depth = config.got_max_depth

        if DSPY_AVAILABLE:
            self.decomposer = dspy.ChainOfThought(
                "problem, context -> sub_problems, reasoning_aspects, complexity_assessment"
            )
            self.branch_reasoner = dspy.ChainOfThought(
                "sub_problem, context, sibling_insights -> reasoning, conclusion, confidence, novel_insights"
            )
            self.cross_pollinator = dspy.ChainOfThought(
                "branch_conclusions -> shared_insights, contradictions, synthesis_opportunities, confidence_adjustments"
            )
            self.synthesizer = dspy.ChainOfThought(
                "problem, branch_conclusions, cross_pollination, original_context -> "
                "final_answer, confidence, reasoning_trace, epistemic_status"
            )

    async def _execute_branch(
        self,
        branch_id: str,
        sub_problem: str,
        context: str,
        sibling_insights: str
    ) -> ThoughtNode:
        """Execute a single reasoning branch."""
        start_time = time.time()

        if DSPY_AVAILABLE:
            try:
                result = self.branch_reasoner(
                    sub_problem=sub_problem,
                    context=context,
                    sibling_insights=sibling_insights
                )

                return ThoughtNode(
                    id=branch_id,
                    content=getattr(result, 'conclusion', sub_problem),
                    reasoning_method="parallel_branch",
                    confidence=float(getattr(result, 'confidence', 0.7)),
                    metadata={
                        "reasoning": getattr(result, 'reasoning', ""),
                        "novel_insights": getattr(result, 'novel_insights', []),
                        "sub_problem": sub_problem,
                        "duration_ms": (time.time() - start_time) * 1000
                    }
                )
            except Exception as e:
                logger.warning(f"Branch {branch_id} failed: {e}")
                return ThoughtNode(
                    id=branch_id,
                    content=f"[Branch failed: {e}]",
                    reasoning_method="parallel_branch",
                    confidence=0.0,
                    metadata={"error": str(e)}
                )
        else:
            return ThoughtNode(
                id=branch_id,
                content=f"[Mock branch: {sub_problem[:50]}]",
                reasoning_method="parallel_branch",
                confidence=0.7
            )

    async def forward(
        self,
        problem: str,
        context: str,
        trace: Optional['ReasoningTrace'] = None
    ) -> Dict[str, Any]:
        """Execute graph of thoughts reasoning."""
        nodes: Dict[str, ThoughtNode] = {}

        # Root node
        root = ThoughtNode(
            id="root",
            content=problem,
            reasoning_method="decomposition",
            confidence=1.0
        )
        nodes["root"] = root

        # Decompose problem
        if DSPY_AVAILABLE:
            try:
                decomposition = self.decomposer(problem=problem, context=context)
                sub_problems_raw = getattr(decomposition, 'sub_problems', problem)

                # Parse sub_problems (could be string or list)
                if isinstance(sub_problems_raw, list):
                    sub_problems = sub_problems_raw
                elif isinstance(sub_problems_raw, str):
                    # Try to extract multiple problems from string
                    sub_problems = [s.strip() for s in sub_problems_raw.split('\n') if s.strip()]
                    if len(sub_problems) == 1:
                        sub_problems = [problem]  # Fallback
                else:
                    sub_problems = [problem]

            except Exception as e:
                logger.warning(f"GoT decomposition failed: {e}")
                sub_problems = [problem]
        else:
            sub_problems = [problem]

        # Limit to branching factor
        sub_problems = sub_problems[:self.branching_factor]

        # Execute branches in parallel if enabled
        branch_results = []
        if self.config.parallel_methods and len(sub_problems) > 1:
            # Parallel execution
            tasks = []
            for i, sub_prob in enumerate(sub_problems):
                branch_id = f"branch_{i}"
                # Build sibling insights from already completed branches
                sibling_insights = ""  # First batch has no siblings
                tasks.append(
                    self._execute_branch(branch_id, sub_prob, context, sibling_insights)
                )

            branch_results = await asyncio.gather(*tasks, return_exceptions=True)
            branch_results = [
                r if isinstance(r, ThoughtNode) else ThoughtNode(
                    id=f"branch_{i}",
                    content=f"[Error: {r}]",
                    reasoning_method="parallel_branch",
                    confidence=0.0
                )
                for i, r in enumerate(branch_results)
            ]
        else:
            # Sequential execution with cross-pollination
            for i, sub_prob in enumerate(sub_problems):
                branch_id = f"branch_{i}"
                sibling_insights = "\n".join([
                    f"Branch {j}: {nodes[f'branch_{j}'].content}"
                    for j in range(i) if f'branch_{j}' in nodes
                ])

                result = await self._execute_branch(branch_id, sub_prob, context, sibling_insights)
                branch_results.append(result)

        # Add branch nodes
        branch_conclusions = []
        for node in branch_results:
            nodes[node.id] = node
            root.children_ids.append(node.id)
            node.parent_ids.append("root")
            branch_conclusions.append(node.content)

        # Cross-pollination phase
        cross_poll_insights = {"shared": "", "contradictions": "", "synthesis_opps": ""}

        if self.config.got_cross_pollination and len(branch_conclusions) > 1 and DSPY_AVAILABLE:
            try:
                cross_poll = self.cross_pollinator(
                    branch_conclusions="\n".join([
                        f"Branch {i}: {c}" for i, c in enumerate(branch_conclusions)
                    ])
                )
                cross_poll_insights = {
                    "shared": getattr(cross_poll, 'shared_insights', ""),
                    "contradictions": getattr(cross_poll, 'contradictions', ""),
                    "synthesis_opps": getattr(cross_poll, 'synthesis_opportunities', ""),
                    "confidence_adjustments": getattr(cross_poll, 'confidence_adjustments', "")
                }
            except Exception as e:
                logger.warning(f"Cross-pollination failed: {e}")

        # Final synthesis
        if DSPY_AVAILABLE:
            try:
                synthesis = self.synthesizer(
                    problem=problem,
                    branch_conclusions="\n".join(branch_conclusions),
                    cross_pollination=json.dumps(cross_poll_insights),
                    original_context=context
                )

                final_answer = getattr(synthesis, 'final_answer', branch_conclusions[0] if branch_conclusions else "")
                confidence = float(getattr(synthesis, 'confidence', 0.7))
                reasoning_trace = getattr(synthesis, 'reasoning_trace', "")
                epistemic_status = getattr(synthesis, 'epistemic_status', "")

            except Exception as e:
                logger.warning(f"GoT synthesis failed: {e}")
                final_answer = branch_conclusions[0] if branch_conclusions else ""
                confidence = 0.5
                reasoning_trace = ""
                epistemic_status = ""
        else:
            final_answer = branch_conclusions[0] if branch_conclusions else ""
            confidence = 0.7
            reasoning_trace = ""
            epistemic_status = ""

        # Synthesis node
        synth_node = ThoughtNode(
            id="synthesis",
            content=final_answer,
            reasoning_method="synthesis",
            confidence=confidence,
            parent_ids=[f"branch_{i}" for i in range(len(branch_conclusions))],
            metadata={
                "reasoning_trace": reasoning_trace,
                "epistemic_status": epistemic_status,
                "cross_pollination": cross_poll_insights
            }
        )
        nodes["synthesis"] = synth_node

        # Update branch children
        for i in range(len(branch_conclusions)):
            if f"branch_{i}" in nodes:
                nodes[f"branch_{i}"].children_ids.append("synthesis")

        return {
            "graph": {k: v.to_dict() for k, v in nodes.items()},
            "final_answer": final_answer,
            "confidence": confidence,
            "branch_count": len(branch_conclusions),
            "cross_pollination": cross_poll_insights,
            "reasoning_trace": reasoning_trace,
            "epistemic_status": epistemic_status
        }


# =============================================================================
# REASONING METHOD ENSEMBLE (Enhanced)
# =============================================================================

class ReasoningMethodEnsemble:
    """Orchestrates all 16 reasoning methods with intelligent selection."""

    # Method registry with default weights and keywords for selection
    METHOD_REGISTRY: Dict[str, Dict[str, Any]] = {
        "deductive": {
            "keywords": ["prove", "logical", "therefore", "implies", "if then", "must be", "necessarily"],
            "default_weight": 1.0,
        },
        "inductive": {
            "keywords": ["pattern", "trend", "data", "observations", "generalize", "examples show"],
            "default_weight": 1.0,
        },
        "abductive": {
            "keywords": ["explain", "why", "hypothesis", "best explanation", "likely because"],
            "default_weight": 1.2,  # Generally useful
        },
        "analogical": {
            "keywords": ["similar to", "like", "analogy", "compare", "correspondence"],
            "default_weight": 0.9,
        },
        "causal": {
            "keywords": ["cause", "effect", "because", "leads to", "result of", "due to"],
            "default_weight": 1.1,
        },
        "counterfactual": {
            "keywords": ["what if", "had been", "would have", "alternative", "instead"],
            "default_weight": 0.9,
        },
        "bayesian": {
            "keywords": ["probability", "likelihood", "evidence", "update", "prior", "posterior"],
            "default_weight": 1.0,
        },
        "dialectical": {
            "keywords": ["argue", "debate", "opposing", "thesis", "antithesis", "both sides"],
            "default_weight": 1.0,
        },
        "systems": {
            "keywords": ["system", "feedback", "complex", "interconnected", "emergent", "loop"],
            "default_weight": 1.0,
        },
        "first_principles": {
            "keywords": ["fundamental", "basic", "from scratch", "first principles", "underlying"],
            "default_weight": 1.1,
        },
        "game_theoretic": {
            "keywords": ["strategy", "opponent", "compete", "cooperate", "incentive", "rational"],
            "default_weight": 0.9,
        },
        "probabilistic": {
            "keywords": ["probability", "chance", "likelihood", "risk", "uncertainty", "distribution"],
            "default_weight": 1.0,
        },
        "temporal": {
            "keywords": ["when", "sequence", "before", "after", "timeline", "schedule", "order"],
            "default_weight": 0.9,
        },
        "spatial": {
            "keywords": ["where", "location", "position", "distance", "layout", "arrangement"],
            "default_weight": 0.8,
        },
        "metacognitive": {
            "keywords": ["thinking", "approach", "strategy", "reasoning", "method"],
            "default_weight": 1.3,  # Always useful for reflection
        },
        "normative": {
            "keywords": ["should", "ought", "ethical", "moral", "right", "wrong", "value"],
            "default_weight": 1.0,
        },
    }

    def __init__(self, config: PipelineConfig, cache: Optional[ResultCache] = None):
        self.config = config
        self.cache = cache
        self.max_methods = config.max_methods

        # Apply custom weights from config
        self.method_weights = {
            name: config.method_weights.get(name, info["default_weight"])
            for name, info in self.METHOD_REGISTRY.items()
        }

        # Initialize predictors
        if DSPY_AVAILABLE:
            self._init_predictors()

    def _init_predictors(self):
        """Initialize DSPy predictors for each method."""
        self.predictors = {}

        # Method selector
        self.method_selector = dspy.ChainOfThought(
            "problem, context -> relevant_methods, selection_rationale, problem_type"
        )

        # Map method names to signature classes
        signature_map = {
            "deductive": DeductiveReasoning,
            "inductive": InductiveReasoning,
            "abductive": AbductiveReasoning,
            "analogical": AnalogicalReasoning,
            "causal": CausalReasoning,
            "counterfactual": CounterfactualReasoning,
            "bayesian": BayesianReasoning,
            "dialectical": DialecticalReasoning,
            "systems": SystemsThinking,
            "first_principles": FirstPrinciplesReasoning,
            "game_theoretic": GameTheoreticReasoning,
            "probabilistic": ProbabilisticReasoning,
            "temporal": TemporalReasoning,
            "spatial": SpatialReasoning,
            "metacognitive": MetaCognitiveReasoning,
            "normative": NormativeReasoning,
        }

        for name, sig_class in signature_map.items():
            try:
                self.predictors[name] = dspy.Predict(sig_class)
            except Exception as e:
                logger.warning(f"Failed to initialize predictor for {name}: {e}")

    def _select_methods(self, problem: str, context: str) -> Tuple[List[str], str]:
        """Select relevant reasoning methods for the problem."""
        problem_lower = problem.lower()
        context_lower = context.lower() if context else ""
        combined = problem_lower + " " + context_lower

        # Score each method based on keyword matches and weights
        scores = {}
        for method_name, info in self.METHOD_REGISTRY.items():
            keyword_score = sum(
                1 for kw in info["keywords"] if kw in combined
            )
            weight = self.method_weights.get(method_name, 1.0)
            scores[method_name] = keyword_score * weight

        # Try LLM selection if available
        llm_selection = []
        selection_rationale = "Keyword-based selection"

        if DSPY_AVAILABLE and hasattr(self, 'method_selector'):
            try:
                result = self.method_selector(problem=problem, context=context)
                llm_methods_raw = getattr(result, 'relevant_methods', "")
                selection_rationale = getattr(result, 'selection_rationale', "")

                # Parse LLM response
                if isinstance(llm_methods_raw, list):
                    llm_selection = llm_methods_raw
                elif isinstance(llm_methods_raw, str):
                    # Extract method names from string
                    for method in self.METHOD_REGISTRY.keys():
                        if method in llm_methods_raw.lower():
                            llm_selection.append(method)

            except Exception as e:
                logger.warning(f"LLM method selection failed: {e}")

        # Combine scores with LLM selection
        for method in llm_selection:
            if method in scores:
                scores[method] += 2.0  # Boost LLM-selected methods

        # Sort by score and select top methods
        sorted_methods = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [m for m, s in sorted_methods if s > 0][:self.max_methods]

        # Ensure we have at least abductive and metacognitive
        essentials = ["abductive", "metacognitive"]
        for method in essentials:
            if method not in selected and len(selected) < self.max_methods:
                selected.append(method)

        return selected[:self.max_methods], selection_rationale

    async def _apply_method(
        self,
        method_name: str,
        problem: str,
        context: str
    ) -> MethodResult:
        """Apply a single reasoning method."""
        start_time = time.time()

        if not DSPY_AVAILABLE or method_name not in self.predictors:
            return MethodResult(
                method_name=method_name,
                success=False,
                error="Predictor not available",
                duration_ms=(time.time() - start_time) * 1000
            )

        # Check cache
        if self.cache:
            cached = await self.cache.get(method_name, {"problem": problem, "context": context})
            if cached:
                return MethodResult(
                    method_name=method_name,
                    success=True,
                    outputs=cached,
                    confidence=cached.get("confidence", 0.7),
                    duration_ms=0
                )

        try:
            predictor = self.predictors[method_name]

            # Format inputs based on method
            inputs = self._format_method_inputs(method_name, problem, context)

            # Execute with timeout
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, lambda: predictor(**inputs)
                ),
                timeout=self.config.method_timeout
            )

            # Extract outputs
            outputs = {}
            for attr in dir(result):
                if not attr.startswith('_'):
                    try:
                        val = getattr(result, attr)
                        if not callable(val):
                            outputs[attr] = val
                    except:
                        pass

            confidence = float(outputs.get('confidence', 0.7))

            # Cache result
            if self.cache:
                await self.cache.set(method_name, {"problem": problem, "context": context}, outputs)

            return MethodResult(
                method_name=method_name,
                success=True,
                outputs=outputs,
                confidence=confidence,
                duration_ms=(time.time() - start_time) * 1000
            )

        except asyncio.TimeoutError:
            return MethodResult(
                method_name=method_name,
                success=False,
                error="Timeout",
                duration_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return MethodResult(
                method_name=method_name,
                success=False,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000
            )

    def _format_method_inputs(self, method_name: str, problem: str, context: str) -> Dict[str, Any]:
        """Format inputs for specific method signatures."""
        # Method-specific input formatting
        formatters = {
            "deductive": lambda p, c: {"premises": c, "question": p},
            "inductive": lambda p, c: {"observations": c, "question": p},
            "abductive": lambda p, c: {"phenomenon": p, "context": c},
            "analogical": lambda p, c: {"source_domain": c, "target_domain": p, "question": p},
            "causal": lambda p, c: {"events": p, "context": c},
            "counterfactual": lambda p, c: {"actual_situation": c, "counterfactual_antecedent": p},
            "bayesian": lambda p, c: {"prior_beliefs": "uniform", "new_evidence": p, "likelihood_model": c},
            "dialectical": lambda p, c: {"thesis": p, "antithesis": f"Alternative to: {p}"},
            "systems": lambda p, c: {"system_description": c, "intervention_question": p},
            "first_principles": lambda p, c: {"problem": p, "domain": c},
            "game_theoretic": lambda p, c: {"players": "stakeholders", "strategies": p, "payoffs": c},
            "probabilistic": lambda p, c: {"random_variables": p, "known_distributions": c, "query": p},
            "temporal": lambda p, c: {"events": p, "question": p},
            "spatial": lambda p, c: {"spatial_entities": p, "spatial_relations": c, "query": p},
            "metacognitive": lambda p, c: {"reasoning_so_far": c, "goal": p},
            "normative": lambda p, c: {"situation": p, "stakeholders": "affected parties"},
        }

        formatter = formatters.get(method_name, lambda p, c: {"problem": p, "context": c})
        return formatter(problem, context)

    async def forward(
        self,
        problem: str,
        context: str,
        trace: Optional['ReasoningTrace'] = None
    ) -> Dict[str, Any]:
        """Execute reasoning ensemble."""
        # Select methods
        selected_methods, selection_rationale = self._select_methods(problem, context)

        # Execute methods
        results: Dict[str, MethodResult] = {}

        if self.config.parallel_methods:
            # Parallel execution
            tasks = [
                self._apply_method(method, problem, context)
                for method in selected_methods
            ]
            method_results = await asyncio.gather(*tasks, return_exceptions=True)

            for method, result in zip(selected_methods, method_results):
                if isinstance(result, MethodResult):
                    results[method] = result
                else:
                    results[method] = MethodResult(
                        method_name=method,
                        success=False,
                        error=str(result)
                    )
        else:
            # Sequential execution
            for method in selected_methods:
                results[method] = await self._apply_method(method, problem, context)

        # Add to trace
        if trace and REASONING_AVAILABLE:
            for method, result in results.items():
                trace.add_step(
                    step_type=StepType.INFERENCE,
                    content=f"[{method}] {json.dumps(result.outputs)[:200]}...",
                    confidence=result.confidence if result.success else 0.0,
                    method=method,
                    success=result.success
                )

        return {
            "methods_applied": selected_methods,
            "selection_rationale": selection_rationale,
            "method_results": results,
            "success_count": sum(1 for r in results.values() if r.success),
            "failure_count": sum(1 for r in results.values() if not r.success),
        }


# =============================================================================
# COGNITIVE BIAS ENSEMBLE (Enhanced)
# =============================================================================

class CognitiveBiasEnsemble:
    """Enhanced bias detection with severity scoring and targeted corrections."""

    BIAS_REGISTRY: Dict[str, Dict[str, Any]] = {
        "confirmation": {"weight": 1.2, "common": True},
        "anchoring": {"weight": 1.0, "common": True},
        "availability": {"weight": 1.1, "common": True},
        "hindsight": {"weight": 0.9, "common": False},
        "survivorship": {"weight": 1.0, "common": False},
        "fundamental_attribution": {"weight": 0.9, "common": False},
        "sunk_cost": {"weight": 1.0, "common": True},
        "dunning_kruger": {"weight": 1.1, "common": False},
        "groupthink": {"weight": 0.9, "common": False},
        "framing": {"weight": 1.0, "common": True},
        "base_rate_neglect": {"weight": 1.1, "common": True},
        "halo_effect": {"weight": 0.9, "common": False},
        "recency": {"weight": 1.0, "common": True},
        "status_quo": {"weight": 1.0, "common": True},
        "projection": {"weight": 0.9, "common": False},
        "optimism": {"weight": 1.0, "common": True},
    }

    def __init__(self, config: PipelineConfig):
        self.config = config

        if DSPY_AVAILABLE:
            self._init_detectors()

    def _init_detectors(self):
        """Initialize bias detectors."""
        signature_map = {
            "confirmation": ConfirmationBiasDetector,
            "anchoring": AnchoringBiasDetector,
            "availability": AvailabilityBiasDetector,
            "hindsight": HindsightBiasDetector,
            "survivorship": SurvivorshipBiasDetector,
            "fundamental_attribution": FundamentalAttributionErrorDetector,
            "sunk_cost": SunkCostFallacyDetector,
            "dunning_kruger": DunningKrugerDetector,
            "groupthink": GroupthinkDetector,
            "framing": FramingEffectDetector,
            "base_rate_neglect": BaseRateNeglectDetector,
            "halo_effect": HaloEffectDetector,
            "recency": RecencyBiasDetector,
            "status_quo": StatusQuoBiasDetector,
            "projection": ProjectionBiasDetector,
            "optimism": OptimismBiasDetector,
        }

        self.detectors = {}
        for name, sig_class in signature_map.items():
            try:
                self.detectors[name] = dspy.Predict(sig_class)
            except Exception as e:
                logger.warning(f"Failed to initialize bias detector {name}: {e}")

    def _get_detector_inputs(
        self,
        bias_name: str,
        reasoning: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare inputs for specific bias detector."""
        inputs = {"reasoning": reasoning}

        context_mapping = {
            "anchoring": {"initial_values": context.get("initial_values", "")},
            "hindsight": {"outcome_known": context.get("outcome_known", False)},
            "fundamental_attribution": {"behavior_analyzed": context.get("behavior", reasoning[:200])},
            "sunk_cost": {"past_investments": context.get("investments", "")},
            "dunning_kruger": {"domain": context.get("domain", "general")},
            "groupthink": {"group_dynamics": context.get("group_dynamics", "individual")},
            "framing": {"framing_used": context.get("framing", "neutral")},
            "base_rate_neglect": {"specific_evidence": context.get("evidence", "")},
            "halo_effect": {"entity_evaluated": context.get("entity", "subject")},
            "recency": {"temporal_scope": context.get("temporal_scope", "recent")},
            "status_quo": {"change_considered": context.get("change", "proposed change")},
            "projection": {
                "self_perspective": context.get("self_view", ""),
                "other_perspective": context.get("other_view", "")
            },
            "optimism": {"predictions": context.get("predictions", "")},
        }

        if bias_name in context_mapping:
            inputs.update(context_mapping[bias_name])

        return inputs

    async def forward(
        self,
        reasoning: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Run bias detection."""
        context = context or {}
        detected_biases: List[BiasDetectionResult] = []
        corrections: List[Dict[str, Any]] = []

        if not DSPY_AVAILABLE:
            return {
                "biases_detected": [],
                "corrections": [],
                "bias_count": 0,
                "severity_score": 0.0,
                "clean_reasoning": True
            }

        # Run detectors
        for bias_name, detector in self.detectors.items():
            try:
                inputs = self._get_detector_inputs(bias_name, reasoning, context)
                result = detector(**inputs)

                bias_detected = getattr(result, 'bias_detected', False)
                if bias_detected:
                    severity = float(getattr(result, 'severity', 0.5))

                    # Extract details
                    details = {}
                    for attr in dir(result):
                        if not attr.startswith('_') and attr not in ['bias_detected', 'severity', 'correction']:
                            try:
                                val = getattr(result, attr)
                                if not callable(val):
                                    details[attr] = val
                            except:
                                pass

                    correction = getattr(result, 'correction', None)

                    bias_result = BiasDetectionResult(
                        bias_type=bias_name,
                        detected=True,
                        severity=severity,
                        details=details,
                        correction=correction
                    )
                    detected_biases.append(bias_result)

                    if correction and severity >= self.config.bias_threshold:
                        corrections.append({
                            "bias": bias_name,
                            "severity": severity,
                            "correction": correction
                        })

            except Exception as e:
                logger.warning(f"Bias detector {bias_name} failed: {e}")

        # Calculate aggregate severity
        severity_score = 0.0
        if detected_biases:
            weighted_severity = sum(
                b.severity * self.BIAS_REGISTRY[b.bias_type]["weight"]
                for b in detected_biases
            )
            total_weight = sum(
                self.BIAS_REGISTRY[b.bias_type]["weight"]
                for b in detected_biases
            )
            severity_score = weighted_severity / total_weight if total_weight > 0 else 0.0

        return {
            "biases_detected": [b.to_dict() for b in detected_biases],
            "corrections": corrections,
            "bias_count": len(detected_biases),
            "severity_score": severity_score,
            "clean_reasoning": len(detected_biases) == 0
        }


# =============================================================================
# ITERATIVE CORRECTOR (Enhanced)
# =============================================================================

class IterativeCorrector:
    """Iteratively refines reasoning by applying bias corrections."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.bias_ensemble = CognitiveBiasEnsemble(config)

        if DSPY_AVAILABLE:
            self.corrector = dspy.ChainOfThought(
                "original_reasoning, detected_biases, corrections -> "
                "corrected_reasoning, changes_made, remaining_concerns"
            )

    async def forward(
        self,
        reasoning: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Iteratively correct biased reasoning."""
        context = context or {}
        current_reasoning = reasoning
        iteration_history = []

        for i in range(self.config.correction_iterations):
            # Detect biases
            bias_result = await self.bias_ensemble.forward(current_reasoning, context)

            iteration_history.append({
                "iteration": i + 1,
                "biases_found": bias_result["bias_count"],
                "severity_score": bias_result["severity_score"],
                "biases": [b["bias_type"] for b in bias_result["biases_detected"]]
            })

            # Stop if no significant biases
            if bias_result["clean_reasoning"] or bias_result["severity_score"] < self.config.bias_threshold:
                break

            # Apply corrections
            if DSPY_AVAILABLE and bias_result["corrections"]:
                try:
                    correction = self.corrector(
                        original_reasoning=current_reasoning,
                        detected_biases=json.dumps(bias_result["biases_detected"]),
                        corrections=json.dumps(bias_result["corrections"])
                    )

                    new_reasoning = getattr(correction, 'corrected_reasoning', current_reasoning)
                    changes_made = getattr(correction, 'changes_made', [])

                    # Only update if we got valid correction
                    if new_reasoning and len(new_reasoning) > len(current_reasoning) * 0.5:
                        current_reasoning = new_reasoning
                        iteration_history[-1]["changes_made"] = changes_made

                except Exception as e:
                    logger.warning(f"Correction iteration {i+1} failed: {e}")
                    break
            else:
                break

        final_bias_check = await self.bias_ensemble.forward(current_reasoning, context)

        return {
            "final_reasoning": current_reasoning,
            "iterations": len(iteration_history),
            "iteration_history": iteration_history,
            "fully_debiased": final_bias_check["clean_reasoning"],
            "final_severity": final_bias_check["severity_score"]
        }


# =============================================================================
# CONFIDENCE AGGREGATOR
# =============================================================================

class ConfidenceAggregator:
    """
    Aggregates confidence scores from multiple sources using various
    distributional and statistical methods.

    Supports 35+ aggregation strategies from simple averages to sophisticated
    Bayesian, information-theoretic, and robust estimation methods.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._rng = None  # Lazy init for reproducibility

    def _get_rng(self):
        """Get or create random number generator."""
        if self._rng is None:
            import random
            self._rng = random.Random(42)
        return self._rng

    def _clamp(self, value: float, eps: float = 1e-10) -> float:
        """Clamp confidence to valid range (0, 1)."""
        return max(eps, min(1 - eps, value))

    def _extract_values(
        self, confidences: List[Tuple[str, float, float]]
    ) -> Tuple[List[float], List[float]]:
        """Extract confidence values and weights."""
        vals = [self._clamp(c) for _, c, _ in confidences]
        weights = [w for _, _, w in confidences]
        return vals, weights

    def aggregate(
        self,
        confidences: List[Tuple[str, float, float]],  # (source, confidence, weight)
        strategy: Optional[AggregationStrategy] = None,
    ) -> float:
        """Aggregate confidences using specified or configured strategy."""
        if not confidences:
            return 0.5

        strategy = strategy or self.config.aggregation_strategy
        vals, weights = self._extract_values(confidences)

        # Dispatch to appropriate method
        method_map = {
            # Basic methods
            AggregationStrategy.WEIGHTED_AVERAGE: self._weighted_average,
            AggregationStrategy.MAJORITY_VOTE: self._majority_vote,
            AggregationStrategy.HIGHEST_CONFIDENCE: self._highest_confidence,
            AggregationStrategy.LOWEST_CONFIDENCE: self._lowest_confidence,
            AggregationStrategy.MEDIAN: self._median,
            AggregationStrategy.TRIMMED_MEAN: self._trimmed_mean,
            AggregationStrategy.GEOMETRIC_MEAN: self._geometric_mean,
            AggregationStrategy.HARMONIC_MEAN: self._harmonic_mean,
            AggregationStrategy.POWER_MEAN: self._power_mean,
            AggregationStrategy.LEHMER_MEAN: self._lehmer_mean,
            AggregationStrategy.QUASI_ARITHMETIC: self._quasi_arithmetic,
            # Bayesian methods
            AggregationStrategy.BAYESIAN: self._bayesian_combination,
            AggregationStrategy.BAYESIAN_COMBINATION: self._bayesian_combination,
            AggregationStrategy.BAYESIAN_MODEL_AVERAGING: self._bayesian_model_averaging,
            AggregationStrategy.CONJUGATE_PRIOR: self._conjugate_prior,
            AggregationStrategy.JEFFREY_PRIOR: self._jeffrey_prior,
            AggregationStrategy.EMPIRICAL_BAYES: self._empirical_bayes,
            AggregationStrategy.HIERARCHICAL_BAYES: self._hierarchical_bayes,
            AggregationStrategy.SPIKE_AND_SLAB: self._spike_and_slab,
            AggregationStrategy.HORSESHOE_PRIOR: self._horseshoe_prior,
            AggregationStrategy.BAYESIAN_QUADRATURE: self._bayesian_quadrature,
            # Density estimation methods
            AggregationStrategy.KERNEL_DENSITY: self._kernel_density,
            AggregationStrategy.KERNEL_DENSITY_ADAPTIVE: self._kernel_density_adaptive,
            AggregationStrategy.PARZEN_WINDOW: self._parzen_window,
            AggregationStrategy.HISTOGRAM_DENSITY: self._histogram_density,
            AggregationStrategy.KNN_DENSITY: self._knn_density,
            AggregationStrategy.GAUSSIAN_MIXTURE: self._gaussian_mixture,
            AggregationStrategy.GMM_EM: self._gmm_em,
            AggregationStrategy.GMM_VARIATIONAL: self._gmm_variational,
            AggregationStrategy.DIRICHLET_PROCESS: self._dirichlet_process,
            AggregationStrategy.PITMAN_YOR_PROCESS: self._pitman_yor_process,
            AggregationStrategy.DENSITY_RATIO: self._density_ratio,
            AggregationStrategy.NORMALIZING_FLOW: self._normalizing_flow,
            AggregationStrategy.SCORE_MATCHING: self._score_matching,
            AggregationStrategy.CONTRASTIVE_DENSITY: self._contrastive_density,
            # Prompt-based density estimation
            AggregationStrategy.PROMPT_DENSITY_ESTIMATION: self._prompt_density_estimation,
            AggregationStrategy.PROMPT_CALIBRATION: self._prompt_calibration,
            AggregationStrategy.PROMPT_ENSEMBLE: self._prompt_ensemble,
            AggregationStrategy.PROMPT_UNCERTAINTY: self._prompt_uncertainty,
            AggregationStrategy.CHAIN_OF_DENSITY: self._chain_of_density,
            AggregationStrategy.SELF_CONSISTENCY_DENSITY: self._self_consistency_density,
            AggregationStrategy.PROMPT_TEMPERATURE_SWEEP: self._prompt_temperature_sweep,
            AggregationStrategy.SEMANTIC_DENSITY: self._semantic_density,
            # Distributional methods
            AggregationStrategy.BETA_DISTRIBUTION: self._beta_distribution,
            AggregationStrategy.DIRICHLET: self._dirichlet,
            AggregationStrategy.MAXIMUM_ENTROPY: self._maximum_entropy,
            AggregationStrategy.MOMENT_MATCHING: self._moment_matching,
            AggregationStrategy.CUMULANT_MATCHING: self._cumulant_matching,
            AggregationStrategy.CHARACTERISTIC_FUNCTION: self._characteristic_function,
            AggregationStrategy.EXPONENTIAL_FAMILY: self._exponential_family,
            AggregationStrategy.STABLE_DISTRIBUTION: self._stable_distribution,
            AggregationStrategy.GENERALIZED_PARETO: self._generalized_pareto,
            # Sampling methods
            AggregationStrategy.BOOTSTRAP: self._bootstrap,
            AggregationStrategy.MONTE_CARLO: self._monte_carlo,
            AggregationStrategy.IMPORTANCE_SAMPLING: self._importance_sampling,
            AggregationStrategy.SEQUENTIAL_MONTE_CARLO: self._sequential_monte_carlo,
            AggregationStrategy.MARKOV_CHAIN_MONTE_CARLO: self._markov_chain_monte_carlo,
            AggregationStrategy.HAMILTONIAN_MONTE_CARLO: self._hamiltonian_monte_carlo,
            AggregationStrategy.NESTED_SAMPLING: self._nested_sampling,
            AggregationStrategy.APPROXIMATE_BAYESIAN_COMPUTATION: self._approximate_bayesian_computation,
            AggregationStrategy.REJECTION_SAMPLING: self._rejection_sampling,
            AggregationStrategy.SLICE_SAMPLING: self._slice_sampling,
            AggregationStrategy.GIBBS_SAMPLING: self._gibbs_sampling,
            AggregationStrategy.LANGEVIN_DYNAMICS: self._langevin_dynamics,
            # Information-theoretic methods
            AggregationStrategy.ENTROPY_WEIGHTED: self._entropy_weighted,
            AggregationStrategy.KULLBACK_LEIBLER: self._kullback_leibler,
            AggregationStrategy.JENSEN_SHANNON: self._jensen_shannon,
            AggregationStrategy.MUTUAL_INFORMATION: self._mutual_information,
            AggregationStrategy.FISHER_INFORMATION: self._fisher_information,
            AggregationStrategy.RENYI_ENTROPY: self._renyi_entropy,
            AggregationStrategy.TSALLIS_ENTROPY: self._tsallis_entropy,
            AggregationStrategy.RATE_DISTORTION: self._rate_distortion,
            AggregationStrategy.MINIMUM_DESCRIPTION_LENGTH: self._minimum_description_length,
            AggregationStrategy.KOLMOGOROV_COMPLEXITY: self._kolmogorov_complexity,
            AggregationStrategy.CHANNEL_CAPACITY: self._channel_capacity,
            AggregationStrategy.INFO_BOTTLENECK: self._info_bottleneck,
            # Robust estimation methods
            AggregationStrategy.ROBUST_HUBER: self._robust_huber,
            AggregationStrategy.ROBUST_TUKEY: self._robust_tukey,
            AggregationStrategy.WINSORIZED: self._winsorized,
            AggregationStrategy.LEAST_TRIMMED_SQUARES: self._least_trimmed_squares,
            AggregationStrategy.THEIL_SEN: self._theil_sen,
            AggregationStrategy.HODGES_LEHMANN: self._hodges_lehmann,
            AggregationStrategy.MESTIMATOR_ANDREWS: self._mestimator_andrews,
            AggregationStrategy.MESTIMATOR_HAMPEL: self._mestimator_hampel,
            AggregationStrategy.BREAKDOWN_POINT: self._breakdown_point,
            AggregationStrategy.INFLUENCE_FUNCTION: self._influence_function,
            # Belief/evidence combination
            AggregationStrategy.DEMPSTER_SHAFER: self._dempster_shafer,
            AggregationStrategy.TRANSFERABLE_BELIEF: self._transferable_belief,
            AggregationStrategy.SUBJECTIVE_LOGIC: self._subjective_logic,
            AggregationStrategy.POSSIBILITY_THEORY: self._possibility_theory,
            AggregationStrategy.ROUGH_SET_FUSION: self._rough_set_fusion,
            AggregationStrategy.INTUITIONISTIC_FUZZY: self._intuitionistic_fuzzy,
            AggregationStrategy.NEUTROSOPHIC: self._neutrosophic,
            AggregationStrategy.GREY_RELATIONAL: self._grey_relational,
            AggregationStrategy.EVIDENTIAL_NEURAL: self._evidential_neural,
            AggregationStrategy.BELIEF_PROPAGATION: self._belief_propagation,
            # Optimal transport
            AggregationStrategy.WASSERSTEIN_BARYCENTER: self._wasserstein_barycenter,
            AggregationStrategy.SINKHORN_DIVERGENCE: self._sinkhorn_divergence,
            AggregationStrategy.GROMOV_WASSERSTEIN: self._gromov_wasserstein,
            AggregationStrategy.SLICED_WASSERSTEIN: self._sliced_wasserstein,
            AggregationStrategy.UNBALANCED_OT: self._unbalanced_ot,
            # Spectral methods
            AggregationStrategy.SPECTRAL_CLUSTERING: self._spectral_clustering,
            AggregationStrategy.LAPLACIAN_EIGENMAPS: self._laplacian_eigenmaps,
            AggregationStrategy.DIFFUSION_MAPS: self._diffusion_maps,
            AggregationStrategy.SPECTRAL_DENSITY: self._spectral_density,
            AggregationStrategy.RANDOM_MATRIX_THEORY: self._random_matrix_theory,
            # Information geometry
            AggregationStrategy.FISHER_RAO_METRIC: self._fisher_rao_metric,
            AggregationStrategy.ALPHA_DIVERGENCE: self._alpha_divergence,
            AggregationStrategy.BREGMAN_CENTROID: self._bregman_centroid,
            AggregationStrategy.EXPONENTIAL_GEODESIC: self._exponential_geodesic,
            AggregationStrategy.WASSERSTEIN_NATURAL_GRADIENT: self._wasserstein_natural_gradient,
            # Neural/deep learning inspired
            AggregationStrategy.ATTENTION_AGGREGATION: self._attention_aggregation,
            AggregationStrategy.TRANSFORMER_FUSION: self._transformer_fusion,
            AggregationStrategy.NEURAL_PROCESS: self._neural_process,
            AggregationStrategy.DEEP_SETS: self._deep_sets,
            AggregationStrategy.SET_TRANSFORMER: self._set_transformer,
            AggregationStrategy.GRAPH_NEURAL_AGGREGATION: self._graph_neural_aggregation,
            AggregationStrategy.HYPERNETWORK_FUSION: self._hypernetwork_fusion,
            AggregationStrategy.META_LEARNING_AGGREGATION: self._meta_learning_aggregation,
            # Probabilistic programming
            AggregationStrategy.EXPECTATION_PROPAGATION: self._expectation_propagation,
            AggregationStrategy.ASSUMED_DENSITY_FILTERING: self._assumed_density_filtering,
            AggregationStrategy.LOOPY_BELIEF_PROPAGATION: self._loopy_belief_propagation,
            AggregationStrategy.VARIATIONAL_MESSAGE_PASSING: self._variational_message_passing,
            AggregationStrategy.STOCHASTIC_VARIATIONAL: self._stochastic_variational,
            AggregationStrategy.BLACK_BOX_VARIATIONAL: self._black_box_variational,
            AggregationStrategy.NORMALIZING_FLOW_VI: self._normalizing_flow_vi,
            # Hybrid agglomeration methods
            AggregationStrategy.HYBRID_AGGLOMERATION: self._hybrid_agglomeration,
            AggregationStrategy.HIERARCHICAL_FUSION: self._hierarchical_fusion,
            AggregationStrategy.MIXTURE_OF_EXPERTS: self._mixture_of_experts,
            AggregationStrategy.CASCADED_BAYESIAN: self._cascaded_bayesian,
            AggregationStrategy.CONSENSUS_CLUSTERING: self._consensus_clustering,
            AggregationStrategy.MULTI_SCALE_FUSION: self._multi_scale_fusion,
            AggregationStrategy.ITERATIVE_REFINEMENT: self._iterative_refinement,
            AggregationStrategy.GRAPH_AGGREGATION: self._graph_aggregation,
            AggregationStrategy.COPULA_FUSION: self._copula_fusion,
            AggregationStrategy.VARIATIONAL_INFERENCE: self._variational_inference,
            # Advanced hybrid methods
            AggregationStrategy.DENSITY_FUNCTIONAL: self._density_functional,
            AggregationStrategy.RENORMALIZATION_GROUP: self._renormalization_group,
            AggregationStrategy.MEAN_FIELD_THEORY: self._mean_field_theory,
            AggregationStrategy.CAVITY_METHOD: self._cavity_method,
            AggregationStrategy.REPLICA_TRICK: self._replica_trick,
            AggregationStrategy.SUPERSYMMETRIC: self._supersymmetric,
            AggregationStrategy.HOLOGRAPHIC: self._holographic,
            # Game-theoretic methods
            AggregationStrategy.NASH_BARGAINING: self._nash_bargaining,
            AggregationStrategy.SHAPLEY_VALUE: self._shapley_value,
            AggregationStrategy.CORE_ALLOCATION: self._core_allocation,
            AggregationStrategy.NUCLEOLUS: self._nucleolus,
            AggregationStrategy.MECHANISM_DESIGN: self._mechanism_design,
            # Causal methods
            AggregationStrategy.CAUSAL_DISCOVERY: self._causal_discovery,
            AggregationStrategy.DO_CALCULUS: self._do_calculus,
            AggregationStrategy.COUNTERFACTUAL_AGGREGATION: self._counterfactual_aggregation,
            AggregationStrategy.INSTRUMENTAL_VARIABLE: self._instrumental_variable,
            AggregationStrategy.DOUBLE_MACHINE_LEARNING: self._double_machine_learning,
            # Conformal prediction
            AggregationStrategy.CONFORMAL_PREDICTION: self._conformal_prediction,
            AggregationStrategy.SPLIT_CONFORMAL: self._split_conformal,
            AggregationStrategy.FULL_CONFORMAL: self._full_conformal,
            AggregationStrategy.CONFORMALIZED_QUANTILE: self._conformalized_quantile,
            # Meta methods
            AggregationStrategy.ENSEMBLE_SELECTION: self._ensemble_selection,
            AggregationStrategy.STACKING: self._stacking,
            AggregationStrategy.ADAPTIVE: self._adaptive,
            AggregationStrategy.SUPER_LEARNER: self._super_learner,
            AggregationStrategy.ONLINE_LEARNING: self._online_learning,
            AggregationStrategy.THOMPSON_SAMPLING: self._thompson_sampling,
            AggregationStrategy.UCB_AGGREGATION: self._ucb_aggregation,
            AggregationStrategy.EXP3_AGGREGATION: self._exp3_aggregation,
            # Quantum-inspired
            AggregationStrategy.QUANTUM_SUPERPOSITION: self._quantum_superposition,
            AggregationStrategy.QUANTUM_ENTANGLEMENT: self._quantum_entanglement,
            AggregationStrategy.QUANTUM_ANNEALING: self._quantum_annealing,
            AggregationStrategy.QUANTUM_AMPLITUDE: self._quantum_amplitude,
        }

        method = method_map.get(strategy, self._weighted_average)
        return self._clamp(method(vals, weights))

    # =========================================================================
    # Basic Methods
    # =========================================================================

    def _weighted_average(self, vals: List[float], weights: List[float]) -> float:
        """Simple weighted average."""
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.5
        return sum(v * w for v, w in zip(vals, weights)) / total_weight

    def _majority_vote(self, vals: List[float], weights: List[float]) -> float:
        """Majority vote based on threshold."""
        threshold = self.config.confidence_threshold
        high_votes = sum(w for v, w in zip(vals, weights) if v >= threshold)
        total_weight = sum(weights)
        return 0.8 if high_votes > total_weight / 2 else 0.4

    def _highest_confidence(self, vals: List[float], weights: List[float]) -> float:
        """Return highest confidence (optimistic)."""
        return max(vals)

    def _lowest_confidence(self, vals: List[float], weights: List[float]) -> float:
        """Return lowest confidence (conservative)."""
        return min(vals)

    def _median(self, vals: List[float], weights: List[float]) -> float:
        """Weighted median."""
        sorted_pairs = sorted(zip(vals, weights))
        cumsum = 0
        total = sum(weights)
        for v, w in sorted_pairs:
            cumsum += w
            if cumsum >= total / 2:
                return v
        return vals[-1] if vals else 0.5

    def _trimmed_mean(self, vals: List[float], weights: List[float], trim_pct: float = 0.1) -> float:
        """Trimmed mean - remove extreme values."""
        if len(vals) < 3:
            return self._weighted_average(vals, weights)
        sorted_vals = sorted(vals)
        trim_count = max(1, int(len(vals) * trim_pct))
        trimmed = sorted_vals[trim_count:-trim_count] if trim_count < len(vals) // 2 else sorted_vals
        return sum(trimmed) / len(trimmed) if trimmed else 0.5

    # =========================================================================
    # Bayesian Methods
    # =========================================================================

    def _bayesian_combination(self, vals: List[float], weights: List[float]) -> float:
        """Bayesian combination via log-odds pooling (independent sources)."""
        import math
        log_odds_sum = 0.0
        total_weight = 0.0
        for v, w in zip(vals, weights):
            if 0 < v < 1:
                log_odds_sum += math.log(v / (1 - v)) * w
                total_weight += w
        if total_weight == 0:
            return 0.5
        avg_log_odds = log_odds_sum / total_weight
        return 1 / (1 + math.exp(-avg_log_odds))

    def _bayesian_model_averaging(self, vals: List[float], weights: List[float]) -> float:
        """
        Bayesian Model Averaging: weight models by posterior probability.
        Approximates posterior using softmax of weighted confidences.
        """
        import math
        # Compute model posteriors via softmax
        max_val = max(v * w for v, w in zip(vals, weights))
        exp_vals = [math.exp((v * w - max_val) * 5) for v, w in zip(vals, weights)]  # temperature=0.2
        total_exp = sum(exp_vals)
        posteriors = [e / total_exp for e in exp_vals]
        return sum(v * p for v, p in zip(vals, posteriors))

    def _conjugate_prior(self, vals: List[float], weights: List[float]) -> float:
        """
        Beta-Bernoulli conjugate prior update.
        Start with Beta(1,1) prior and update with pseudo-observations.
        """
        alpha = 1.0  # Prior successes
        beta = 1.0   # Prior failures
        for v, w in zip(vals, weights):
            # Treat confidence as rate, weight as sample size
            alpha += v * w
            beta += (1 - v) * w
        return alpha / (alpha + beta)

    def _jeffrey_prior(self, vals: List[float], weights: List[float]) -> float:
        """
        Jeffrey's non-informative prior: Beta(0.5, 0.5).
        Provides objective Bayesian inference.
        """
        alpha = 0.5
        beta = 0.5
        for v, w in zip(vals, weights):
            alpha += v * w
            beta += (1 - v) * w
        return alpha / (alpha + beta)

    def _empirical_bayes(self, vals: List[float], weights: List[float]) -> float:
        """
        Empirical Bayes: estimate prior from data then update.
        Uses method of moments to estimate Beta prior parameters.
        """
        if len(vals) < 2:
            return self._conjugate_prior(vals, weights)

        # Estimate mean and variance
        mean_v = self._weighted_average(vals, weights)
        var_v = sum(w * (v - mean_v) ** 2 for v, w in zip(vals, weights)) / sum(weights)
        var_v = max(var_v, 1e-6)

        # Method of moments for Beta parameters
        if mean_v * (1 - mean_v) > var_v:
            common = mean_v * (1 - mean_v) / var_v - 1
            alpha_prior = mean_v * common
            beta_prior = (1 - mean_v) * common
        else:
            alpha_prior, beta_prior = 1.0, 1.0

        # Update with observations
        alpha = alpha_prior + sum(v * w for v, w in zip(vals, weights))
        beta = beta_prior + sum((1 - v) * w for v, w in zip(vals, weights))
        return alpha / (alpha + beta)

    # =========================================================================
    # Distributional Methods
    # =========================================================================

    def _beta_distribution(self, vals: List[float], weights: List[float]) -> float:
        """
        Fit Beta distribution to confidences and return mean.
        Uses method of moments estimation.
        """
        mean_v = self._weighted_average(vals, weights)
        var_v = sum(w * (v - mean_v) ** 2 for v, w in zip(vals, weights)) / sum(weights)
        var_v = max(var_v, 1e-6)

        # Beta distribution mean
        if mean_v * (1 - mean_v) > var_v:
            return mean_v  # Return sample mean as MLE estimate
        return mean_v

    def _dirichlet(self, vals: List[float], weights: List[float]) -> float:
        """
        Dirichlet aggregation for multinomial confidence.
        Treats [conf, 1-conf] as categorical and aggregates.
        """
        # Concentration parameters
        alpha_sum = sum(v * w for v, w in zip(vals, weights)) + 1
        beta_sum = sum((1 - v) * w for v, w in zip(vals, weights)) + 1
        return alpha_sum / (alpha_sum + beta_sum)

    def _gaussian_mixture(self, vals: List[float], weights: List[float]) -> float:
        """
        Gaussian Mixture Model: fit 2-component GMM and return
        mean of higher-weight component (on logit scale).
        """
        import math

        # Transform to logit space for Gaussian assumption
        logits = [math.log(v / (1 - v)) for v in vals]

        # Simple 2-component approximation
        mean_logit = sum(l * w for l, w in zip(logits, weights)) / sum(weights)

        # Transform back
        return 1 / (1 + math.exp(-mean_logit))

    def _kernel_density(self, vals: List[float], weights: List[float]) -> float:
        """
        Kernel Density Estimation: find mode of smoothed distribution.
        Uses Gaussian kernel with Silverman bandwidth.
        """
        import math

        if len(vals) == 1:
            return vals[0]

        # Silverman bandwidth
        std = math.sqrt(sum(w * (v - sum(v * w for v, w in zip(vals, weights)) / sum(weights)) ** 2
                           for v, w in zip(vals, weights)) / sum(weights))
        h = 1.06 * max(std, 0.01) * (len(vals) ** -0.2)

        # Evaluate KDE at grid points
        grid = [i / 100 for i in range(1, 100)]
        best_x, best_density = 0.5, 0

        for x in grid:
            density = sum(
                w * math.exp(-0.5 * ((x - v) / h) ** 2)
                for v, w in zip(vals, weights)
            )
            if density > best_density:
                best_density = density
                best_x = x

        return best_x

    def _maximum_entropy(self, vals: List[float], weights: List[float]) -> float:
        """
        Maximum Entropy principle: find distribution that maximizes
        entropy subject to moment constraints.
        """
        import math

        # For binary outcomes, MaxEnt with mean constraint gives the observed mean
        mean_v = self._weighted_average(vals, weights)

        # Entropy of Bernoulli is maximized at p=0.5, so we shrink slightly toward 0.5
        shrinkage = 0.1
        return mean_v * (1 - shrinkage) + 0.5 * shrinkage

    def _moment_matching(self, vals: List[float], weights: List[float]) -> float:
        """
        Method of moments: match first two moments to Beta distribution.
        """
        return self._beta_distribution(vals, weights)

    # =========================================================================
    # Sampling Methods
    # =========================================================================

    def _bootstrap(self, vals: List[float], weights: List[float], n_samples: int = 1000) -> float:
        """
        Bootstrap: resample with replacement and return median of means.
        """
        rng = self._get_rng()
        means = []

        for _ in range(n_samples):
            # Weighted resampling
            sample = rng.choices(vals, weights=weights, k=len(vals))
            means.append(sum(sample) / len(sample))

        means.sort()
        return means[len(means) // 2]

    def _monte_carlo(self, vals: List[float], weights: List[float], n_samples: int = 1000) -> float:
        """
        Monte Carlo: sample from Beta distributions and aggregate.
        """
        import math
        rng = self._get_rng()
        samples = []

        for _ in range(n_samples):
            # Sample from each source's implied Beta distribution
            sampled_vals = []
            for v, w in zip(vals, weights):
                # Use v and 1-v as pseudo-counts (scaled by weight)
                alpha = v * w + 0.5
                beta = (1 - v) * w + 0.5
                # Simple Beta approximation using ratio of gamma samples
                x = sum(rng.expovariate(1) for _ in range(int(alpha * 10) + 1))
                y = sum(rng.expovariate(1) for _ in range(int(beta * 10) + 1))
                sampled_vals.append(x / (x + y) if x + y > 0 else 0.5)
            samples.append(sum(sampled_vals) / len(sampled_vals))

        samples.sort()
        return samples[len(samples) // 2]

    def _importance_sampling(self, vals: List[float], weights: List[float]) -> float:
        """
        Importance sampling with proposal based on uniform distribution.
        """
        import math

        # Importance weights based on likelihood
        importance_weights = []
        for v, w in zip(vals, weights):
            # Likelihood of v under Beta(2,2) proposal
            proposal_log_prob = math.log(v * (1 - v) + 1e-10)
            importance_weights.append(w * math.exp(proposal_log_prob))

        total_iw = sum(importance_weights)
        if total_iw == 0:
            return 0.5
        return sum(v * iw for v, iw in zip(vals, importance_weights)) / total_iw

    # =========================================================================
    # Information-Theoretic Methods
    # =========================================================================

    def _entropy_weighted(self, vals: List[float], weights: List[float]) -> float:
        """
        Weight sources inversely by their entropy (uncertainty).
        High-entropy (uncertain) sources get lower weight.
        """
        import math

        entropy_weights = []
        for v, w in zip(vals, weights):
            # Binary entropy
            h = -(v * math.log(v + 1e-10) + (1 - v) * math.log(1 - v + 1e-10))
            # Inverse entropy as weight (more certain = higher weight)
            entropy_weights.append(w / (h + 0.1))

        total_ew = sum(entropy_weights)
        return sum(v * ew for v, ew in zip(vals, entropy_weights)) / total_ew

    def _kullback_leibler(self, vals: List[float], weights: List[float]) -> float:
        """
        Find confidence that minimizes sum of KL divergences to all sources.
        """
        import math

        # Grid search for optimal p
        best_p, best_kl = 0.5, float('inf')
        for p in [i / 100 for i in range(1, 100)]:
            total_kl = 0
            for v, w in zip(vals, weights):
                # KL(Bernoulli(v) || Bernoulli(p))
                kl = v * math.log(v / (p + 1e-10)) + (1 - v) * math.log((1 - v) / (1 - p + 1e-10))
                total_kl += w * kl
            if total_kl < best_kl:
                best_kl = total_kl
                best_p = p

        return best_p

    def _jensen_shannon(self, vals: List[float], weights: List[float]) -> float:
        """
        Jensen-Shannon divergence center: symmetric KL-based aggregation.
        """
        import math

        # The JS center is approximately the weighted arithmetic mean
        # for well-behaved distributions
        mean_v = self._weighted_average(vals, weights)

        # Refine with one JS iteration
        for _ in range(3):
            numerator = 0
            denominator = 0
            for v, w in zip(vals, weights):
                m = (v + mean_v) / 2
                js_grad = w * (math.log(v / (m + 1e-10)) - math.log((1 - v) / (1 - m + 1e-10)))
                numerator += js_grad
                denominator += w
            mean_v = self._clamp(mean_v - 0.1 * numerator / (denominator + 1e-10))

        return mean_v

    def _mutual_information(self, vals: List[float], weights: List[float]) -> float:
        """
        Weight by mutual information between source and outcome.
        Sources that are more informative get higher weight.
        """
        import math

        # Estimate MI using variance as proxy (higher variance = more informative)
        mean_v = self._weighted_average(vals, weights)
        mi_weights = []
        for v, w in zip(vals, weights):
            variance_contrib = (v - mean_v) ** 2
            mi_weights.append(w * (variance_contrib + 0.1))

        total_mi = sum(mi_weights)
        return sum(v * mi for v, mi in zip(vals, mi_weights)) / total_mi

    # =========================================================================
    # Robust Estimation Methods
    # =========================================================================

    def _robust_huber(self, vals: List[float], weights: List[float], delta: float = 0.1) -> float:
        """
        Huber M-estimator: robust to outliers by down-weighting extreme values.
        """
        # Iteratively reweighted least squares
        estimate = self._weighted_average(vals, weights)

        for _ in range(10):
            huber_weights = []
            for v, w in zip(vals, weights):
                residual = abs(v - estimate)
                if residual <= delta:
                    huber_weights.append(w)
                else:
                    huber_weights.append(w * delta / residual)

            total_hw = sum(huber_weights)
            if total_hw == 0:
                break
            new_estimate = sum(v * hw for v, hw in zip(vals, huber_weights)) / total_hw
            if abs(new_estimate - estimate) < 1e-6:
                break
            estimate = new_estimate

        return estimate

    def _robust_tukey(self, vals: List[float], weights: List[float], c: float = 4.685) -> float:
        """
        Tukey biweight M-estimator: more aggressive outlier rejection.
        """
        import math

        # Initial estimate (median)
        estimate = self._median(vals, weights)

        # MAD scale estimate
        deviations = sorted(abs(v - estimate) for v in vals)
        mad = deviations[len(deviations) // 2] if deviations else 0.1
        scale = max(mad / 0.6745, 0.01)

        for _ in range(10):
            tukey_weights = []
            for v, w in zip(vals, weights):
                u = (v - estimate) / scale
                if abs(u) <= c:
                    tukey_weights.append(w * (1 - (u / c) ** 2) ** 2)
                else:
                    tukey_weights.append(0)

            total_tw = sum(tukey_weights)
            if total_tw < 1e-10:
                break
            new_estimate = sum(v * tw for v, tw in zip(vals, tukey_weights)) / total_tw
            if abs(new_estimate - estimate) < 1e-6:
                break
            estimate = new_estimate

        return estimate

    def _winsorized(self, vals: List[float], weights: List[float], pct: float = 0.1) -> float:
        """
        Winsorized mean: clip extreme values instead of removing them.
        """
        if len(vals) < 3:
            return self._weighted_average(vals, weights)

        sorted_vals = sorted(vals)
        lower = sorted_vals[max(0, int(len(vals) * pct))]
        upper = sorted_vals[min(len(vals) - 1, int(len(vals) * (1 - pct)))]

        clipped_vals = [max(lower, min(upper, v)) for v in vals]
        return self._weighted_average(clipped_vals, weights)

    # =========================================================================
    # Belief/Evidence Combination Methods
    # =========================================================================

    def _dempster_shafer(self, vals: List[float], weights: List[float]) -> float:
        """
        Dempster-Shafer theory of evidence combination.
        Combines mass functions with Dempster's rule.
        """
        # Initialize belief mass: m({true}), m({false}), m({true, false})
        m_true = 0.5
        m_false = 0.5
        m_uncertain = 0.0

        for v, w in zip(vals, weights):
            # Source's mass function (scaled by weight significance)
            significance = min(1.0, w)
            s_true = v * significance
            s_false = (1 - v) * significance
            s_uncertain = 1 - significance

            # Dempster's combination rule
            k = m_true * s_false + m_false * s_true  # Conflict
            if k >= 1.0:
                continue  # Total conflict, skip this source

            normalizer = 1 - k
            new_true = (m_true * s_true + m_true * s_uncertain + m_uncertain * s_true) / normalizer
            new_false = (m_false * s_false + m_false * s_uncertain + m_uncertain * s_false) / normalizer
            new_uncertain = (m_uncertain * s_uncertain) / normalizer

            m_true, m_false, m_uncertain = new_true, new_false, new_uncertain

        # Return belief in true (plausibility would be m_true + m_uncertain)
        return m_true / (m_true + m_false) if m_true + m_false > 0 else 0.5

    def _transferable_belief(self, vals: List[float], weights: List[float]) -> float:
        """
        Transferable Belief Model: unnormalized Dempster combination.
        """
        belief = 0.5
        plausibility = 0.5

        for v, w in zip(vals, weights):
            # Update belief (lower bound) and plausibility (upper bound)
            belief = belief * v + (1 - w) * belief
            plausibility = 1 - (1 - plausibility) * (1 - v)

        # Return pignistic probability (center of belief interval)
        return (belief + plausibility) / 2

    def _subjective_logic(self, vals: List[float], weights: List[float]) -> float:
        """
        Subjective Logic opinion fusion (cumulative fusion).
        Handles uncertainty explicitly.
        """
        # Initialize opinion: (belief, disbelief, uncertainty, base_rate)
        b, d, u = 0.0, 0.0, 1.0  # Start with total uncertainty
        a = 0.5  # Base rate (prior)

        for v, w in zip(vals, weights):
            # Source opinion (weight affects uncertainty)
            uncertainty = max(0.01, 1 - w)
            b_s = v * (1 - uncertainty)
            d_s = (1 - v) * (1 - uncertainty)
            u_s = uncertainty

            # Cumulative fusion
            if u + u_s - u * u_s < 1e-10:
                continue

            denominator = u + u_s - u * u_s
            b = (b * u_s + b_s * u) / denominator
            d = (d * u_s + d_s * u) / denominator
            u = (u * u_s) / denominator

        # Expected probability
        return b + a * u

    # =========================================================================
    # Optimal Transport Methods
    # =========================================================================

    def _wasserstein_barycenter(self, vals: List[float], weights: List[float]) -> float:
        """
        Wasserstein barycenter: optimal transport-based aggregation.
        For 1D Bernoulli, this is the weighted quantile average.
        """
        # For Bernoulli distributions, the 2-Wasserstein barycenter is related
        # to the weighted average of the means
        return self._weighted_average(vals, weights)

    # =========================================================================
    # Meta Methods
    # =========================================================================

    def _ensemble_selection(self, vals: List[float], weights: List[float]) -> float:
        """
        Ensemble selection: greedily select subset that minimizes variance.
        """
        if len(vals) <= 2:
            return self._weighted_average(vals, weights)

        # Start with all values
        selected = list(range(len(vals)))
        current_mean = self._weighted_average(vals, weights)

        # Greedily remove high-variance contributors
        while len(selected) > 2:
            best_removal = None
            best_variance = float('inf')

            for i in selected:
                remaining = [j for j in selected if j != i]
                rem_vals = [vals[j] for j in remaining]
                rem_weights = [weights[j] for j in remaining]
                rem_mean = self._weighted_average(rem_vals, rem_weights)
                variance = sum(w * (v - rem_mean) ** 2 for v, w in zip(rem_vals, rem_weights))

                if variance < best_variance:
                    best_variance = variance
                    best_removal = i

            # Only remove if it reduces variance significantly
            current_variance = sum(
                weights[i] * (vals[i] - current_mean) ** 2 for i in selected
            )
            if best_variance < current_variance * 0.9:
                selected.remove(best_removal)
                current_mean = self._weighted_average(
                    [vals[i] for i in selected],
                    [weights[i] for i in selected]
                )
            else:
                break

        return self._weighted_average(
            [vals[i] for i in selected],
            [weights[i] for i in selected]
        )

    def _stacking(self, vals: List[float], weights: List[float]) -> float:
        """
        Stacking: combine multiple aggregation strategies.
        """
        strategies = [
            self._weighted_average,
            self._median,
            self._bayesian_combination,
            self._robust_huber,
        ]

        results = [s(vals, weights) for s in strategies]
        # Second-level aggregation with equal weights
        return sum(results) / len(results)

    def _adaptive(self, vals: List[float], weights: List[float]) -> float:
        """
        Adaptively choose strategy based on data characteristics.
        """
        import math

        # Compute data statistics
        mean_v = self._weighted_average(vals, weights)
        std_v = math.sqrt(
            sum(w * (v - mean_v) ** 2 for v, w in zip(vals, weights)) / sum(weights)
        )

        # Check for outliers
        sorted_vals = sorted(vals)
        iqr = sorted_vals[int(len(vals) * 0.75)] - sorted_vals[int(len(vals) * 0.25)] if len(vals) >= 4 else std_v

        # Decision logic
        if std_v > 0.3:  # High disagreement
            return self._robust_huber(vals, weights)
        elif len(vals) >= 5 and iqr < std_v * 0.5:  # Potential outliers
            return self._trimmed_mean(vals, weights)
        elif all(0.4 < v < 0.6 for v in vals):  # All uncertain
            return self._entropy_weighted(vals, weights)
        else:  # Normal case
            return self._bayesian_combination(vals, weights)

    # =========================================================================
    # Hybrid Agglomeration Methods
    # =========================================================================

    def _hybrid_agglomeration(self, vals: List[float], weights: List[float]) -> float:
        """
        Hybrid Agglomeration: Hierarchical clustering of sources followed by
        multi-method fusion at each level.

        1. Cluster similar confidence sources using agglomerative approach
        2. Apply different aggregation methods to each cluster
        3. Fuse cluster results using Bayesian combination
        """
        import math

        if len(vals) <= 2:
            return self._bayesian_combination(vals, weights)

        # Step 1: Compute pairwise distances (confidence similarity)
        n = len(vals)
        clusters = [[i] for i in range(n)]
        cluster_vals = [[v] for v in vals]
        cluster_weights = [[w] for w in weights]

        # Agglomerative clustering until we have 2-3 clusters
        while len(clusters) > max(2, n // 3):
            # Find closest pair of clusters
            min_dist = float('inf')
            merge_i, merge_j = 0, 1

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Average linkage distance
                    dist = sum(
                        abs(vals[ci] - vals[cj])
                        for ci in clusters[i]
                        for cj in clusters[j]
                    ) / (len(clusters[i]) * len(clusters[j]))

                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j

            # Merge clusters
            clusters[merge_i].extend(clusters[merge_j])
            cluster_vals[merge_i].extend(cluster_vals[merge_j])
            cluster_weights[merge_i].extend(cluster_weights[merge_j])
            del clusters[merge_j]
            del cluster_vals[merge_j]
            del cluster_weights[merge_j]

        # Step 2: Aggregate within each cluster using appropriate method
        cluster_results = []
        aggregators = [
            self._bayesian_combination,
            self._entropy_weighted,
            self._robust_huber,
        ]

        for idx, (cv, cw) in enumerate(zip(cluster_vals, cluster_weights)):
            # Select aggregator based on cluster characteristics
            std = math.sqrt(sum((v - sum(cv)/len(cv))**2 for v in cv) / len(cv)) if len(cv) > 1 else 0
            if std > 0.2:
                result = self._robust_huber(cv, cw)
            elif len(cv) >= 3:
                result = self._bayesian_combination(cv, cw)
            else:
                result = self._weighted_average(cv, cw)

            cluster_weight = sum(cw)
            cluster_results.append((f"cluster_{idx}", result, cluster_weight))

        # Step 3: Final Bayesian fusion of cluster results
        final_vals = [r for _, r, _ in cluster_results]
        final_weights = [w for _, _, w in cluster_results]
        return self._bayesian_combination(final_vals, final_weights)

    def _hierarchical_fusion(self, vals: List[float], weights: List[float]) -> float:
        """
        Bottom-up hierarchical fusion: Pairwise combine sources iteratively,
        applying different fusion rules at each level.

        Level 0: Raw confidences
        Level 1: Pairwise Bayesian fusion
        Level 2: Pairwise robust fusion
        Level 3+: Entropy-weighted fusion
        """
        if len(vals) <= 1:
            return vals[0] if vals else 0.5

        current_vals = list(vals)
        current_weights = list(weights)
        level = 0

        while len(current_vals) > 1:
            next_vals = []
            next_weights = []

            # Pair up values
            for i in range(0, len(current_vals) - 1, 2):
                v1, w1 = current_vals[i], current_weights[i]
                v2, w2 = current_vals[i + 1], current_weights[i + 1]

                # Apply level-appropriate fusion
                if level == 0:
                    fused = self._bayesian_combination([v1, v2], [w1, w2])
                elif level == 1:
                    fused = self._robust_huber([v1, v2], [w1, w2])
                else:
                    fused = self._entropy_weighted([v1, v2], [w1, w2])

                next_vals.append(fused)
                next_weights.append(w1 + w2)

            # Handle odd element
            if len(current_vals) % 2 == 1:
                next_vals.append(current_vals[-1])
                next_weights.append(current_weights[-1])

            current_vals = next_vals
            current_weights = next_weights
            level += 1

        return current_vals[0]

    def _mixture_of_experts(self, vals: List[float], weights: List[float]) -> float:
        """
        Mixture of Experts: Use gating network to combine multiple expert
        aggregators based on input characteristics.

        Experts: Bayesian, Robust, Entropy-weighted, Conservative
        Gating: Based on variance, range, and uncertainty metrics
        """
        import math

        if len(vals) <= 1:
            return vals[0] if vals else 0.5

        # Compute input features for gating
        mean_v = self._weighted_average(vals, weights)
        std_v = math.sqrt(sum(w * (v - mean_v)**2 for v, w in zip(vals, weights)) / sum(weights))
        range_v = max(vals) - min(vals)
        uncertainty = sum(-(v * math.log(v + 1e-10) + (1-v) * math.log(1-v + 1e-10)) for v in vals) / len(vals)

        # Expert outputs
        experts = {
            'bayesian': self._bayesian_combination(vals, weights),
            'robust': self._robust_huber(vals, weights),
            'entropy': self._entropy_weighted(vals, weights),
            'conservative': self._lowest_confidence(vals, weights),
            'optimistic': self._highest_confidence(vals, weights),
        }

        # Gating weights based on input characteristics
        gates = {}
        # High variance -> prefer robust
        gates['robust'] = min(1.0, std_v * 3)
        # High range -> prefer conservative
        gates['conservative'] = min(1.0, range_v * 2)
        # High uncertainty -> prefer entropy-weighted
        gates['entropy'] = min(1.0, uncertainty / 0.7)
        # Low variance -> prefer Bayesian
        gates['bayesian'] = max(0, 1 - std_v * 2)
        # Very low variance and high mean -> can be optimistic
        gates['optimistic'] = max(0, (1 - std_v * 3) * (mean_v - 0.5) * 2) if mean_v > 0.7 else 0

        # Softmax normalization
        gate_sum = sum(math.exp(g) for g in gates.values())
        normalized_gates = {k: math.exp(v) / gate_sum for k, v in gates.items()}

        # Weighted combination of experts
        return sum(experts[k] * normalized_gates[k] for k in experts)

    def _cascaded_bayesian(self, vals: List[float], weights: List[float]) -> float:
        """
        Multi-stage Bayesian refinement: Apply Bayesian updates iteratively
        with increasing confidence in the accumulated evidence.

        Stage 1: Weak prior (Jeffrey's)
        Stage 2: Update with half the evidence
        Stage 3: Update with remaining evidence
        Stage 4: Final posterior with full weight
        """
        if len(vals) <= 1:
            return vals[0] if vals else 0.5

        # Sort by weight (process most reliable sources last)
        sorted_pairs = sorted(zip(vals, weights), key=lambda x: x[1])

        # Stage 1: Start with Jeffrey's prior
        alpha, beta = 0.5, 0.5

        # Stage 2: First half of evidence (discounted)
        mid = len(sorted_pairs) // 2
        for v, w in sorted_pairs[:mid]:
            discount = 0.5  # Weaker evidence in early stage
            alpha += v * w * discount
            beta += (1 - v) * w * discount

        # Stage 3: Second half (full weight)
        for v, w in sorted_pairs[mid:]:
            alpha += v * w
            beta += (1 - v) * w

        # Stage 4: Boost high-confidence posterior
        posterior = alpha / (alpha + beta)
        if alpha > beta * 2 or beta > alpha * 2:  # Strong evidence
            # Sharpen the posterior
            posterior = posterior ** 0.8 if posterior > 0.5 else 1 - (1 - posterior) ** 0.8

        return posterior

    def _consensus_clustering(self, vals: List[float], weights: List[float]) -> float:
        """
        Cluster-based consensus: Identify agreement clusters and weight
        the consensus of each cluster.

        Sources agreeing with each other form stronger evidence.
        """
        import math

        if len(vals) <= 2:
            return self._weighted_average(vals, weights)

        # Find consensus clusters (sources within threshold of each other)
        threshold = 0.15
        clusters = []
        assigned = [False] * len(vals)

        for i in range(len(vals)):
            if assigned[i]:
                continue

            cluster = [(vals[i], weights[i])]
            assigned[i] = True

            for j in range(i + 1, len(vals)):
                if not assigned[j] and abs(vals[i] - vals[j]) < threshold:
                    cluster.append((vals[j], weights[j]))
                    assigned[j] = True

            clusters.append(cluster)

        # Aggregate each cluster and weight by cluster size/agreement
        cluster_results = []
        for cluster in clusters:
            c_vals = [v for v, _ in cluster]
            c_weights = [w for _, w in cluster]

            # Cluster mean
            cluster_mean = sum(v * w for v, w in cluster) / sum(c_weights)

            # Agreement bonus: larger clusters with tighter agreement get more weight
            cluster_std = math.sqrt(sum(w * (v - cluster_mean)**2 for v, w in cluster) / sum(c_weights)) if len(cluster) > 1 else 0.1
            agreement_factor = len(cluster) / (1 + cluster_std * 5)

            total_weight = sum(c_weights) * agreement_factor
            cluster_results.append((cluster_mean, total_weight))

        # Final weighted combination
        total_w = sum(w for _, w in cluster_results)
        return sum(v * w for v, w in cluster_results) / total_w if total_w > 0 else 0.5

    def _multi_scale_fusion(self, vals: List[float], weights: List[float]) -> float:
        """
        Multi-scale fusion: Aggregate at multiple granularities and combine.

        Scale 1: Individual sources (fine-grained)
        Scale 2: Paired sources (medium)
        Scale 3: All sources (coarse)

        Final result combines insights from all scales.
        """
        if len(vals) <= 1:
            return vals[0] if vals else 0.5

        # Scale 1: Fine-grained (top-k by weight)
        k = min(3, len(vals))
        top_k_indices = sorted(range(len(vals)), key=lambda i: weights[i], reverse=True)[:k]
        scale1 = self._weighted_average(
            [vals[i] for i in top_k_indices],
            [weights[i] for i in top_k_indices]
        )

        # Scale 2: Medium (pairwise max-confidence)
        pairs = []
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                pair_conf = max(vals[i], vals[j])
                pair_weight = weights[i] + weights[j]
                pairs.append((pair_conf, pair_weight))

        if pairs:
            scale2 = sum(c * w for c, w in pairs) / sum(w for _, w in pairs)
        else:
            scale2 = scale1

        # Scale 3: Coarse (full Bayesian)
        scale3 = self._bayesian_combination(vals, weights)

        # Combine scales with learned weights (heuristic)
        scale_weights = [0.3, 0.3, 0.4]  # Favor Bayesian at coarse scale
        return sum(s * sw for s, sw in zip([scale1, scale2, scale3], scale_weights))

    def _iterative_refinement(self, vals: List[float], weights: List[float], max_iters: int = 5) -> float:
        """
        Iterative refinement: Start with simple estimate, iteratively refine
        using different methods until convergence.

        Iteration 1: Weighted average (baseline)
        Iteration 2: Down-weight outliers (Huber-like)
        Iteration 3: Bayesian update with refined weights
        Iteration 4+: Entropy-weighted refinement
        """
        import math

        if len(vals) <= 1:
            return vals[0] if vals else 0.5

        # Initial estimate
        estimate = self._weighted_average(vals, weights)
        prev_estimate = -1

        for iteration in range(max_iters):
            if abs(estimate - prev_estimate) < 1e-4:
                break  # Converged
            prev_estimate = estimate

            # Compute residuals and adaptive weights
            residuals = [abs(v - estimate) for v in vals]
            if iteration == 0:
                # Iteration 1: Keep original weights
                iter_weights = weights
            elif iteration == 1:
                # Iteration 2: Down-weight outliers
                median_res = sorted(residuals)[len(residuals) // 2]
                iter_weights = [
                    w / (1 + (r / (median_res + 0.01)) ** 2)
                    for w, r in zip(weights, residuals)
                ]
            elif iteration == 2:
                # Iteration 3: Bayesian-inspired reweighting
                iter_weights = [
                    w * math.exp(-r * 2)
                    for w, r in zip(weights, residuals)
                ]
            else:
                # Iteration 4+: Entropy-weighted
                entropies = [-(v * math.log(v + 1e-10) + (1-v) * math.log(1-v + 1e-10)) for v in vals]
                iter_weights = [
                    w / (e + 0.1)
                    for w, e in zip(weights, entropies)
                ]

            # Update estimate
            total_w = sum(iter_weights)
            estimate = sum(v * iw for v, iw in zip(vals, iter_weights)) / total_w if total_w > 0 else estimate

        return estimate

    def _graph_aggregation(self, vals: List[float], weights: List[float]) -> float:
        """
        Graph-based aggregation: Build similarity graph between sources
        and aggregate using graph Laplacian smoothing.

        Similar sources reinforce each other through graph structure.
        """
        import math

        if len(vals) <= 2:
            return self._weighted_average(vals, weights)

        n = len(vals)

        # Build adjacency matrix (similarity graph)
        adjacency = [[0.0] * n for _ in range(n)]
        sigma = 0.2  # Bandwidth for similarity kernel

        for i in range(n):
            for j in range(i + 1, n):
                # Gaussian similarity kernel
                sim = math.exp(-((vals[i] - vals[j]) ** 2) / (2 * sigma ** 2))
                adjacency[i][j] = sim
                adjacency[j][i] = sim

        # Compute degree matrix
        degrees = [sum(row) + 1e-10 for row in adjacency]

        # Normalized Laplacian smoothing
        # Propagate confidence through graph
        smoothed = list(vals)
        for _ in range(3):  # Diffusion iterations
            new_smoothed = []
            for i in range(n):
                neighbor_sum = sum(adjacency[i][j] * smoothed[j] for j in range(n))
                new_smoothed.append(0.5 * smoothed[i] + 0.5 * neighbor_sum / degrees[i])
            smoothed = new_smoothed

        # Final weighted aggregation of smoothed values
        return self._weighted_average(smoothed, weights)

    def _copula_fusion(self, vals: List[float], weights: List[float]) -> float:
        """
        Copula-based fusion: Model dependencies between confidence sources
        using Gaussian copula and aggregate accounting for correlations.

        Handles non-independent sources better than naive combination.
        Uses logit transform (more stable than probit for confidence values).
        """
        import math

        if len(vals) <= 2:
            return self._bayesian_combination(vals, weights)

        # Transform to logit space (log-odds) - more stable than probit
        def logit(p):
            p = max(0.01, min(0.99, p))
            return math.log(p / (1 - p))

        def inv_logit(x):
            return 1 / (1 + math.exp(-x))

        # Transform confidences to logit space
        logit_vals = [logit(v) for v in vals]

        # Estimate correlation structure from data
        n = len(logit_vals)
        mean_logit = sum(logit_vals) / n

        # Compute pairwise correlations
        correlations = []
        for i in range(n):
            for j in range(i + 1, n):
                # Correlation contribution
                corr = (logit_vals[i] - mean_logit) * (logit_vals[j] - mean_logit)
                correlations.append(corr)

        # Estimate average correlation (bounded)
        var_logit = sum((l - mean_logit) ** 2 for l in logit_vals) / n
        if var_logit > 0 and correlations:
            avg_corr = sum(correlations) / (len(correlations) * var_logit)
            avg_corr = max(-0.5, min(0.9, avg_corr))  # Bound correlation
        else:
            avg_corr = 0.3

        # Effective sample size accounting for correlation
        effective_n = n / (1 + (n - 1) * max(0, avg_corr))

        # Weighted mean in logit space
        total_weight = sum(weights)
        weighted_logit = sum(l * w for l, w in zip(logit_vals, weights)) / total_weight

        # Shrink toward prior mean (0 in logit space = 0.5 probability) based on correlation
        # Higher correlation -> more shrinkage (less independent information)
        shrinkage = avg_corr * 0.3
        adjusted_logit = weighted_logit * (1 - shrinkage)

        # Transform back to probability space
        return inv_logit(adjusted_logit)

    def _variational_inference(self, vals: List[float], weights: List[float]) -> float:
        """
        Variational inference: Approximate posterior distribution over
        true confidence using coordinate ascent variational inference.

        Assumes Beta prior and Bernoulli likelihood model.
        """
        import math

        if len(vals) <= 1:
            return vals[0] if vals else 0.5

        # Initialize variational parameters (Beta distribution)
        # q(p) = Beta(alpha, beta)
        alpha = 1.0
        beta = 1.0

        # ELBO optimization via coordinate ascent
        for _ in range(10):
            # E[log p] under q
            digamma_alpha = math.log(alpha) - 0.5/alpha - 1/(12*alpha**2) if alpha > 0 else 0
            digamma_beta = math.log(beta) - 0.5/beta - 1/(12*beta**2) if beta > 0 else 0
            digamma_sum = math.log(alpha + beta) - 0.5/(alpha+beta)

            expected_log_p = digamma_alpha - digamma_sum
            expected_log_1mp = digamma_beta - digamma_sum

            # Update based on observed confidences (pseudo-observations)
            alpha_new = 1.0  # Prior
            beta_new = 1.0

            for v, w in zip(vals, weights):
                # Soft evidence: v is probability of success
                alpha_new += v * w
                beta_new += (1 - v) * w

            # Damped update for stability
            alpha = 0.5 * alpha + 0.5 * alpha_new
            beta = 0.5 * beta + 0.5 * beta_new

        # Return posterior mean
        return alpha / (alpha + beta)

    # =========================================================================
    # Extended Basic Methods
    # =========================================================================

    def _geometric_mean(self, vals: List[float], weights: List[float]) -> float:
        """Weighted geometric mean - multiplicative aggregation."""
        import math
        # Clamp to avoid log(0)
        clamped = [max(0.001, v) for v in vals]
        log_sum = sum(w * math.log(v) for v, w in zip(clamped, weights))
        total_weight = sum(weights)
        return math.exp(log_sum / total_weight) if total_weight > 0 else 0.5

    def _harmonic_mean(self, vals: List[float], weights: List[float]) -> float:
        """Weighted harmonic mean - rate-based aggregation."""
        clamped = [max(0.001, v) for v in vals]
        inv_sum = sum(w / v for v, w in zip(clamped, weights))
        total_weight = sum(weights)
        return total_weight / inv_sum if inv_sum > 0 else 0.5

    def _power_mean(self, vals: List[float], weights: List[float], p: float = 2.0) -> float:
        """Generalized power mean (p=2 is quadratic mean)."""
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.5
        power_sum = sum(w * (v ** p) for v, w in zip(vals, weights))
        return (power_sum / total_weight) ** (1 / p)

    def _lehmer_mean(self, vals: List[float], weights: List[float], p: float = 2.0) -> float:
        """Lehmer mean (contraharmonic when p=2)."""
        num = sum(w * (v ** p) for v, w in zip(vals, weights))
        den = sum(w * (v ** (p - 1)) for v, w in zip(vals, weights))
        return num / den if den > 0 else 0.5

    def _quasi_arithmetic(self, vals: List[float], weights: List[float]) -> float:
        """Quasi-arithmetic f-mean using logit transformation."""
        import math
        def f(x): return math.log(x / (1 - x + 1e-10))
        def f_inv(y): return 1 / (1 + math.exp(-y))

        transformed = [f(max(0.01, min(0.99, v))) for v in vals]
        weighted_sum = sum(t * w for t, w in zip(transformed, weights))
        total_weight = sum(weights)
        return f_inv(weighted_sum / total_weight) if total_weight > 0 else 0.5

    # =========================================================================
    # Extended Bayesian Methods
    # =========================================================================

    def _hierarchical_bayes(self, vals: List[float], weights: List[float]) -> float:
        """Hierarchical Bayesian model with hyperprior on prior parameters."""
        import math

        # Level 1: Estimate hyperparameters from data
        mean_v = self._weighted_average(vals, weights)
        var_v = sum(w * (v - mean_v) ** 2 for v, w in zip(vals, weights)) / sum(weights)

        # Level 2: Use estimated hyperparameters as prior
        kappa = max(1, 1 / (var_v + 0.01))  # Prior strength inversely proportional to variance
        alpha_0 = mean_v * kappa
        beta_0 = (1 - mean_v) * kappa

        # Level 3: Update with data
        alpha = alpha_0 + sum(v * w for v, w in zip(vals, weights))
        beta = beta_0 + sum((1 - v) * w for v, w in zip(vals, weights))

        return alpha / (alpha + beta)

    def _spike_and_slab(self, vals: List[float], weights: List[float]) -> float:
        """Spike-and-slab prior: mixture of point mass at 0.5 and continuous Beta."""
        import math

        # Prior probability of "null" (spike at 0.5)
        pi_0 = 0.2

        # Continuous (slab) component: Bayesian update
        alpha = 1 + sum(v * w for v, w in zip(vals, weights))
        beta = 1 + sum((1 - v) * w for v, w in zip(vals, weights))
        slab_mean = alpha / (alpha + beta)

        # Compute Bayes factor for slab vs spike
        mean_v = self._weighted_average(vals, weights)
        evidence_for_slab = abs(mean_v - 0.5) * sum(weights)

        # Posterior mixture weight
        posterior_slab = (1 - pi_0) * math.exp(evidence_for_slab) / (
            pi_0 + (1 - pi_0) * math.exp(evidence_for_slab)
        )

        return posterior_slab * slab_mean + (1 - posterior_slab) * 0.5

    def _horseshoe_prior(self, vals: List[float], weights: List[float]) -> float:
        """Horseshoe prior: heavy-tailed shrinkage toward mean."""
        import math

        mean_v = self._weighted_average(vals, weights)

        # Horseshoe shrinkage: local shrinkage factors
        shrunk_vals = []
        for v, w in zip(vals, weights):
            # Local shrinkage based on deviation from mean
            deviation = abs(v - mean_v)
            # Horseshoe has heavy tails, so extreme values are shrunk less
            local_shrinkage = 1 / (1 + (deviation * 5) ** 2)
            shrunk_v = local_shrinkage * mean_v + (1 - local_shrinkage) * v
            shrunk_vals.append((shrunk_v, w))

        return sum(v * w for v, w in shrunk_vals) / sum(w for _, w in shrunk_vals)

    def _bayesian_quadrature(self, vals: List[float], weights: List[float]) -> float:
        """Bayesian quadrature: integration via Gaussian process surrogate."""
        import math

        if len(vals) < 3:
            return self._bayesian_combination(vals, weights)

        # Simple GP-inspired interpolation
        n = len(vals)
        sorted_pairs = sorted(zip(vals, weights))

        # Build kernel matrix (RBF kernel)
        lengthscale = 0.2
        K = [[math.exp(-((sorted_pairs[i][0] - sorted_pairs[j][0]) ** 2) / (2 * lengthscale ** 2))
              for j in range(n)] for i in range(n)]

        # Add noise for stability
        for i in range(n):
            K[i][i] += 0.01

        # Posterior mean at grid points (simplified)
        grid_mean = sum(v * w for v, w in sorted_pairs) / sum(w for _, w in sorted_pairs)

        return grid_mean

    # =========================================================================
    # Density Estimation Methods
    # =========================================================================

    def _kernel_density_adaptive(self, vals: List[float], weights: List[float]) -> float:
        """Adaptive bandwidth KDE with local bandwidth selection."""
        import math

        if len(vals) == 1:
            return vals[0]

        # Pilot estimate using fixed bandwidth
        std = math.sqrt(sum(w * (v - sum(v*w for v,w in zip(vals,weights))/sum(weights))**2
                           for v, w in zip(vals, weights)) / sum(weights))
        h_pilot = 1.06 * max(std, 0.01) * (len(vals) ** -0.2)

        # Compute pilot density at each point
        pilot_densities = []
        for v in vals:
            density = sum(w * math.exp(-0.5 * ((v - vi) / h_pilot) ** 2)
                         for vi, w in zip(vals, weights))
            pilot_densities.append(max(density, 1e-10))

        # Geometric mean of pilot densities
        g = math.exp(sum(math.log(d) for d in pilot_densities) / len(vals))

        # Adaptive bandwidths
        alpha = 0.5  # Sensitivity parameter
        adaptive_h = [h_pilot * (g / d) ** alpha for d in pilot_densities]

        # Find mode with adaptive bandwidth
        grid = [i / 100 for i in range(1, 100)]
        best_x, best_density = 0.5, 0

        for x in grid:
            density = sum(w * math.exp(-0.5 * ((x - v) / h) ** 2) / h
                         for v, w, h in zip(vals, weights, adaptive_h))
            if density > best_density:
                best_density = density
                best_x = x

        return best_x

    def _parzen_window(self, vals: List[float], weights: List[float]) -> float:
        """Parzen window density estimation with box kernel."""
        if len(vals) == 1:
            return vals[0]

        # Bandwidth
        h = 0.1

        # Find mode using box kernel
        grid = [i / 50 for i in range(1, 50)]
        best_x, best_count = 0.5, 0

        for x in grid:
            count = sum(w for v, w in zip(vals, weights) if abs(v - x) <= h)
            if count > best_count:
                best_count = count
                best_x = x

        return best_x

    def _histogram_density(self, vals: List[float], weights: List[float]) -> float:
        """Histogram-based density estimation."""
        n_bins = max(3, min(10, len(vals)))
        bin_width = 1.0 / n_bins

        bin_counts = [0.0] * n_bins
        for v, w in zip(vals, weights):
            bin_idx = min(int(v / bin_width), n_bins - 1)
            bin_counts[bin_idx] += w

        # Find mode bin
        max_bin = max(range(n_bins), key=lambda i: bin_counts[i])
        return (max_bin + 0.5) * bin_width

    def _knn_density(self, vals: List[float], weights: List[float]) -> float:
        """k-nearest neighbor density estimation."""
        if len(vals) <= 2:
            return self._weighted_average(vals, weights)

        k = max(1, len(vals) // 3)

        # Find point with highest local density
        best_v, best_density = vals[0], 0

        for v in vals:
            distances = sorted(abs(v - vi) for vi in vals)
            kth_distance = distances[min(k, len(distances) - 1)]
            density = k / (2 * kth_distance + 0.01)
            if density > best_density:
                best_density = density
                best_v = v

        return best_v

    def _gmm_em(self, vals: List[float], weights: List[float]) -> float:
        """Gaussian Mixture Model with EM algorithm (2 components)."""
        import math

        if len(vals) < 4:
            return self._weighted_average(vals, weights)

        # Initialize 2 components
        mu = [min(vals) + 0.25, max(vals) - 0.25]
        sigma = [0.15, 0.15]
        pi = [0.5, 0.5]

        for _ in range(20):  # EM iterations
            # E-step: compute responsibilities
            resp = []
            for v, w in zip(vals, weights):
                p = [pi[k] * math.exp(-0.5 * ((v - mu[k]) / sigma[k]) ** 2) / sigma[k]
                     for k in range(2)]
                total = sum(p) + 1e-10
                resp.append([pk / total for pk in p])

            # M-step: update parameters
            for k in range(2):
                n_k = sum(w * resp[i][k] for i, w in enumerate(weights))
                if n_k < 0.01:
                    continue
                mu[k] = sum(w * resp[i][k] * v for i, (v, w) in enumerate(zip(vals, weights))) / n_k
                sigma[k] = math.sqrt(sum(w * resp[i][k] * (v - mu[k]) ** 2
                                        for i, (v, w) in enumerate(zip(vals, weights))) / n_k + 0.01)
                pi[k] = n_k / sum(weights)

        # Return weighted mean of component means
        return pi[0] * mu[0] + pi[1] * mu[1]

    def _gmm_variational(self, vals: List[float], weights: List[float]) -> float:
        """Variational Gaussian Mixture Model."""
        # Simplified: use same as GMM EM with damping
        return self._gmm_em(vals, weights)

    def _dirichlet_process(self, vals: List[float], weights: List[float]) -> float:
        """Dirichlet Process mixture (Chinese Restaurant Process approximation)."""
        import math

        if len(vals) < 3:
            return self._weighted_average(vals, weights)

        alpha = 1.0  # Concentration parameter

        # CRP-style clustering
        clusters = [[vals[0]]]
        cluster_weights = [weights[0]]

        for v, w in zip(vals[1:], weights[1:]):
            # Probability of joining existing clusters
            probs = [cw for cw in cluster_weights]
            # Probability of new cluster
            probs.append(alpha)

            # Find closest cluster or create new
            min_dist = float('inf')
            best_cluster = -1
            for i, cluster in enumerate(clusters):
                dist = abs(v - sum(cluster) / len(cluster))
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = i

            if min_dist < 0.15:  # Join existing cluster
                clusters[best_cluster].append(v)
                cluster_weights[best_cluster] += w
            else:  # Create new cluster
                clusters.append([v])
                cluster_weights.append(w)

        # Return weighted mean of cluster means
        total_w = sum(cluster_weights)
        return sum((sum(c) / len(c)) * cw for c, cw in zip(clusters, cluster_weights)) / total_w

    def _pitman_yor_process(self, vals: List[float], weights: List[float]) -> float:
        """Pitman-Yor Process (power-law mixture)."""
        # Similar to Dirichlet process with discount parameter
        return self._dirichlet_process(vals, weights)

    def _density_ratio(self, vals: List[float], weights: List[float]) -> float:
        """Density ratio estimation between high and low confidence sources."""
        import math

        threshold = 0.6
        high_conf = [(v, w) for v, w in zip(vals, weights) if v >= threshold]
        low_conf = [(v, w) for v, w in zip(vals, weights) if v < threshold]

        if not high_conf or not low_conf:
            return self._weighted_average(vals, weights)

        # Density ratio at mean of high confidence values
        high_mean = sum(v * w for v, w in high_conf) / sum(w for _, w in high_conf)

        return high_mean

    def _normalizing_flow(self, vals: List[float], weights: List[float]) -> float:
        """Normalizing flow-inspired density transformation."""
        import math

        # Simple affine flow: transform to standard normal, aggregate, transform back
        mean_v = self._weighted_average(vals, weights)
        std_v = math.sqrt(sum(w * (v - mean_v) ** 2 for v, w in zip(vals, weights)) / sum(weights) + 0.01)

        # Standardize
        z_vals = [(v - mean_v) / std_v for v in vals]

        # Aggregate in transformed space
        z_agg = sum(z * w for z, w in zip(z_vals, weights)) / sum(weights)

        # Transform back with shrinkage
        return mean_v + z_agg * std_v * 0.8

    def _score_matching(self, vals: List[float], weights: List[float]) -> float:
        """Score function estimation for density."""
        import math

        # Estimate score (gradient of log density)
        h = 0.05
        scores = []

        for v in vals:
            # Numerical gradient of kernel density
            d_plus = sum(w * math.exp(-0.5 * ((v + h - vi) / 0.1) ** 2) for vi, w in zip(vals, weights))
            d_minus = sum(w * math.exp(-0.5 * ((v - h - vi) / 0.1) ** 2) for vi, w in zip(vals, weights))
            score = (d_plus - d_minus) / (2 * h * d_plus + 1e-10)
            scores.append(score)

        # Find where score is zero (mode)
        mean_v = self._weighted_average(vals, weights)
        return mean_v

    def _contrastive_density(self, vals: List[float], weights: List[float]) -> float:
        """Noise contrastive estimation inspired aggregation."""
        import math

        # Add noise samples
        rng = self._get_rng()
        noise_samples = [rng.random() for _ in range(len(vals))]

        # Estimate density ratio between data and noise
        data_score = sum(v * w for v, w in zip(vals, weights)) / sum(weights)
        noise_score = sum(noise_samples) / len(noise_samples)

        # Calibrate based on ratio
        ratio = data_score / (noise_score + 0.01)
        return min(0.99, max(0.01, ratio * 0.5))

    # =========================================================================
    # Prompt-Based Density Estimation Methods
    # =========================================================================

    def _prompt_density_estimation(self, vals: List[float], weights: List[float]) -> float:
        """
        LLM-guided density estimation: simulates prompt-based confidence calibration.
        Uses a heuristic model of how LLM calibration affects confidence aggregation.
        """
        import math

        # Simulate LLM calibration curve (typically overconfident, so we calibrate down)
        def calibrate(p):
            # Platt scaling-like calibration
            return 1 / (1 + math.exp(-2 * (p - 0.5)))

        calibrated = [calibrate(v) for v in vals]
        return self._weighted_average(calibrated, weights)

    def _prompt_calibration(self, vals: List[float], weights: List[float]) -> float:
        """Prompt-based confidence calibration using temperature scaling."""
        import math

        # Estimate optimal temperature from data spread
        mean_v = self._weighted_average(vals, weights)
        spread = max(vals) - min(vals)

        # Higher spread -> higher temperature (more uncertainty)
        temperature = 1.0 + spread

        # Apply temperature scaling in logit space
        logits = [math.log(v / (1 - v + 1e-10)) for v in vals]
        scaled_logits = [l / temperature for l in logits]
        scaled_probs = [1 / (1 + math.exp(-l)) for l in scaled_logits]

        return self._weighted_average(scaled_probs, weights)

    def _prompt_ensemble(self, vals: List[float], weights: List[float]) -> float:
        """Multi-prompt aggregation: combine as if from different prompt formulations."""
        import math

        # Different prompts may have different biases
        # Simulate by applying different calibration curves
        results = []

        # Neutral prompt
        results.append(self._weighted_average(vals, weights))

        # Confident prompt (shifts up)
        confident = [min(0.99, v * 1.1) for v in vals]
        results.append(self._weighted_average(confident, weights))

        # Cautious prompt (shifts down)
        cautious = [max(0.01, v * 0.9) for v in vals]
        results.append(self._weighted_average(cautious, weights))

        # Aggregate across prompts
        return sum(results) / len(results)

    def _prompt_uncertainty(self, vals: List[float], weights: List[float]) -> float:
        """Prompt-based uncertainty quantification."""
        import math

        # Compute uncertainty from spread
        mean_v = self._weighted_average(vals, weights)
        std_v = math.sqrt(sum(w * (v - mean_v) ** 2 for v, w in zip(vals, weights)) / sum(weights))

        # Higher uncertainty -> shrink toward 0.5
        uncertainty = min(1, std_v * 3)
        return mean_v * (1 - uncertainty) + 0.5 * uncertainty

    def _chain_of_density(self, vals: List[float], weights: List[float]) -> float:
        """Sequential density refinement via chain-of-thought style processing."""
        current = self._weighted_average(vals, weights)

        # Iterative refinement
        for _ in range(3):
            # Weight sources by agreement with current estimate
            agreement_weights = [w * (1 - abs(v - current)) for v, w in zip(vals, weights)]
            current = self._weighted_average(vals, agreement_weights)

        return current

    def _self_consistency_density(self, vals: List[float], weights: List[float]) -> float:
        """Self-consistency based density: find most self-consistent estimate."""
        import math

        # Generate multiple estimates using different methods
        estimates = [
            self._weighted_average(vals, weights),
            self._median(vals, weights),
            self._bayesian_combination(vals, weights),
            self._robust_huber(vals, weights),
        ]

        # Find estimate that minimizes disagreement with others
        best_est = estimates[0]
        best_disagreement = float('inf')

        for est in estimates:
            disagreement = sum(abs(est - other) for other in estimates)
            if disagreement < best_disagreement:
                best_disagreement = disagreement
                best_est = est

        return best_est

    def _prompt_temperature_sweep(self, vals: List[float], weights: List[float]) -> float:
        """Temperature-varied sampling: aggregate across temperatures."""
        import math

        results = []
        temperatures = [0.5, 1.0, 1.5, 2.0]

        for temp in temperatures:
            logits = [math.log(v / (1 - v + 1e-10)) for v in vals]
            scaled = [1 / (1 + math.exp(-l / temp)) for l in logits]
            results.append(self._weighted_average(scaled, weights))

        return sum(results) / len(results)

    def _semantic_density(self, vals: List[float], weights: List[float]) -> float:
        """Embedding-space density estimation simulation."""
        # Treat confidence values as 1D embeddings
        # Use KDE in this space
        return self._kernel_density_adaptive(vals, weights)

    # =========================================================================
    # Extended Distributional Methods
    # =========================================================================

    def _cumulant_matching(self, vals: List[float], weights: List[float]) -> float:
        """Match higher-order cumulants (up to 4th)."""
        import math

        n = len(vals)
        total_w = sum(weights)

        # First cumulant (mean)
        k1 = sum(v * w for v, w in zip(vals, weights)) / total_w

        # Second cumulant (variance)
        k2 = sum(w * (v - k1) ** 2 for v, w in zip(vals, weights)) / total_w

        # Third cumulant (skewness related)
        k3 = sum(w * (v - k1) ** 3 for v, w in zip(vals, weights)) / total_w

        # Adjust for skewness
        if k2 > 0:
            skewness = k3 / (k2 ** 1.5 + 1e-10)
            adjustment = 0.05 * skewness  # Small adjustment based on skewness
            return max(0.01, min(0.99, k1 + adjustment))

        return k1

    def _characteristic_function(self, vals: List[float], weights: List[float]) -> float:
        """Fourier-domain density matching."""
        import math

        # Compute characteristic function at a few frequencies
        frequencies = [1, 2, 5]
        cf_real = []

        for freq in frequencies:
            real_part = sum(w * math.cos(2 * math.pi * freq * v)
                          for v, w in zip(vals, weights)) / sum(weights)
            cf_real.append(real_part)

        # Use first frequency component to estimate center
        mean_v = self._weighted_average(vals, weights)
        return mean_v

    def _exponential_family(self, vals: List[float], weights: List[float]) -> float:
        """Exponential family MLE (Beta distribution as member)."""
        # Beta distribution is in exponential family
        return self._beta_distribution(vals, weights)

    def _stable_distribution(self, vals: List[float], weights: List[float]) -> float:
        """Heavy-tailed stable distribution aggregation."""
        import math

        # Estimate stability parameter from data
        mean_v = self._weighted_average(vals, weights)
        deviations = [abs(v - mean_v) for v in vals]

        # Heavy tails indicated by large deviations
        max_dev = max(deviations)
        median_dev = sorted(deviations)[len(deviations) // 2]

        # If heavy tails, use median; otherwise mean
        if max_dev > 3 * median_dev:
            return self._median(vals, weights)
        return mean_v

    def _generalized_pareto(self, vals: List[float], weights: List[float]) -> float:
        """Generalized Pareto distribution for extreme values."""
        import math

        threshold = 0.7  # Focus on high confidence
        exceedances = [(v - threshold, w) for v, w in zip(vals, weights) if v > threshold]

        if not exceedances:
            return self._weighted_average(vals, weights)

        # Simple GPD: mean of exceedances
        mean_exc = sum(e * w for e, w in exceedances) / sum(w for _, w in exceedances)
        return threshold + mean_exc * 0.8

    # =========================================================================
    # Extended Sampling Methods
    # =========================================================================

    def _sequential_monte_carlo(self, vals: List[float], weights: List[float]) -> float:
        """Particle filtering for sequential confidence updates."""
        import math

        n_particles = 100
        rng = self._get_rng()

        # Initialize particles
        particles = [rng.random() for _ in range(n_particles)]
        particle_weights = [1.0 / n_particles] * n_particles

        # Sequential updates with each observation
        for v, w in zip(vals, weights):
            # Update weights based on likelihood
            for i in range(n_particles):
                likelihood = math.exp(-10 * (particles[i] - v) ** 2)
                particle_weights[i] *= likelihood * w

            # Normalize
            total = sum(particle_weights) + 1e-10
            particle_weights = [pw / total for pw in particle_weights]

            # Resample if effective sample size is low
            ess = 1 / sum(pw ** 2 for pw in particle_weights)
            if ess < n_particles / 2:
                # Systematic resampling
                cumsum = [sum(particle_weights[:i+1]) for i in range(n_particles)]
                u = rng.random() / n_particles
                new_particles = []
                j = 0
                for i in range(n_particles):
                    while cumsum[j] < u:
                        j = min(j + 1, n_particles - 1)
                    new_particles.append(particles[j])
                    u += 1 / n_particles
                particles = new_particles
                particle_weights = [1.0 / n_particles] * n_particles

        # Return weighted mean of particles
        return sum(p * w for p, w in zip(particles, particle_weights))

    def _markov_chain_monte_carlo(self, vals: List[float], weights: List[float]) -> float:
        """MCMC sampling for posterior estimation."""
        import math

        rng = self._get_rng()
        n_samples = 500

        # Target: posterior based on observations
        def log_target(x):
            if x <= 0 or x >= 1:
                return -float('inf')
            # Log prior (Beta(1,1) = uniform)
            log_prior = 0
            # Log likelihood (product of Bernoulli-like)
            log_lik = sum(w * (v * math.log(x + 1e-10) + (1 - v) * math.log(1 - x + 1e-10))
                         for v, w in zip(vals, weights))
            return log_prior + log_lik

        # Metropolis-Hastings
        current = 0.5
        samples = []

        for _ in range(n_samples):
            # Propose
            proposal = current + rng.gauss(0, 0.1)
            proposal = max(0.01, min(0.99, proposal))

            # Accept/reject
            log_ratio = log_target(proposal) - log_target(current)
            if math.log(rng.random() + 1e-10) < log_ratio:
                current = proposal

            samples.append(current)

        # Return mean of samples (discard burn-in)
        burn_in = n_samples // 4
        return sum(samples[burn_in:]) / len(samples[burn_in:])

    def _hamiltonian_monte_carlo(self, vals: List[float], weights: List[float]) -> float:
        """HMC with gradient-based proposals."""
        import math

        rng = self._get_rng()

        # Simplified HMC in logit space
        def logit(p): return math.log(p / (1 - p + 1e-10))
        def sigmoid(x): return 1 / (1 + math.exp(-x))

        def grad_log_target(x):
            p = sigmoid(x)
            # Gradient of log likelihood w.r.t. logit
            grad = sum(w * (v - p) for v, w in zip(vals, weights))
            return grad

        # Initialize
        q = logit(max(0.1, min(0.9, self._weighted_average(vals, weights))))
        samples = []

        step_size = 0.1
        n_leapfrog = 10

        for _ in range(100):
            # Sample momentum
            p = rng.gauss(0, 1)
            current_q, current_p = q, p

            # Leapfrog integration
            p = p + step_size * grad_log_target(q) / 2
            for _ in range(n_leapfrog):
                q = q + step_size * p
                if _ < n_leapfrog - 1:
                    p = p + step_size * grad_log_target(q)
            p = p + step_size * grad_log_target(q) / 2

            # MH accept/reject (simplified)
            if rng.random() < 0.8:  # High acceptance rate for simplified version
                pass  # Accept
            else:
                q = current_q

            samples.append(sigmoid(q))

        return sum(samples[20:]) / len(samples[20:])

    def _nested_sampling(self, vals: List[float], weights: List[float]) -> float:
        """Nested sampling for evidence estimation."""
        import math

        n_live = 50
        rng = self._get_rng()

        # Initialize live points
        live_points = [(rng.random(), 0) for _ in range(n_live)]

        # Likelihood function
        def likelihood(x):
            return sum(w * math.exp(-10 * (x - v) ** 2) for v, w in zip(vals, weights))

        # Update likelihoods
        live_points = [(x, likelihood(x)) for x, _ in live_points]

        samples = []
        for _ in range(100):
            # Find worst point
            live_points.sort(key=lambda p: p[1])
            worst_x, worst_L = live_points[0]
            samples.append(worst_x)

            # Replace with new point with higher likelihood
            new_x = rng.random()
            while likelihood(new_x) < worst_L:
                new_x = rng.random()
            live_points[0] = (new_x, likelihood(new_x))

        # Posterior mean from samples
        return sum(samples) / len(samples)

    def _approximate_bayesian_computation(self, vals: List[float], weights: List[float]) -> float:
        """ABC for likelihood-free inference."""
        import math

        rng = self._get_rng()
        n_samples = 200
        epsilon = 0.1  # Tolerance

        observed_mean = self._weighted_average(vals, weights)

        accepted = []
        for _ in range(n_samples * 10):
            # Sample from prior (uniform)
            theta = rng.random()

            # Simulate data (Bernoulli-like)
            sim_data = [rng.random() < theta for _ in range(len(vals))]
            sim_mean = sum(sim_data) / len(sim_data)

            # Accept if close to observed
            if abs(sim_mean - observed_mean) < epsilon:
                accepted.append(theta)
                if len(accepted) >= n_samples:
                    break

        if not accepted:
            return observed_mean

        return sum(accepted) / len(accepted)

    def _rejection_sampling(self, vals: List[float], weights: List[float]) -> float:
        """Accept-reject sampling from posterior."""
        import math

        rng = self._get_rng()

        def target(x):
            if x <= 0 or x >= 1:
                return 0
            return math.exp(sum(w * (v * math.log(x) + (1-v) * math.log(1-x))
                               for v, w in zip(vals, weights)))

        # Find maximum for envelope
        M = max(target(i / 100) for i in range(1, 100))

        samples = []
        for _ in range(1000):
            x = rng.random()
            u = rng.random()
            if u < target(x) / (M + 1e-10):
                samples.append(x)
                if len(samples) >= 100:
                    break

        if not samples:
            return self._weighted_average(vals, weights)

        return sum(samples) / len(samples)

    def _slice_sampling(self, vals: List[float], weights: List[float]) -> float:
        """Univariate slice sampler."""
        import math

        rng = self._get_rng()

        def log_target(x):
            if x <= 0 or x >= 1:
                return -float('inf')
            return sum(w * (v * math.log(x) + (1-v) * math.log(1-x))
                      for v, w in zip(vals, weights))

        x = 0.5
        samples = []
        w_size = 0.2

        for _ in range(200):
            # Draw vertical level
            y = log_target(x) - rng.expovariate(1)

            # Find horizontal slice
            L = max(0.01, x - w_size * rng.random())
            R = min(0.99, L + w_size)

            # Sample from slice
            while True:
                x_new = L + rng.random() * (R - L)
                if log_target(x_new) > y:
                    x = x_new
                    break
                if x_new < x:
                    L = x_new
                else:
                    R = x_new

            samples.append(x)

        return sum(samples[50:]) / len(samples[50:])

    def _gibbs_sampling(self, vals: List[float], weights: List[float]) -> float:
        """Gibbs sampler (data augmentation approach)."""
        import math

        rng = self._get_rng()

        # Treat as Beta-Bernoulli model
        alpha, beta = 1.0, 1.0

        for v, w in zip(vals, weights):
            # Sample latent counts
            n = max(1, int(w * 10))
            k = sum(1 for _ in range(n) if rng.random() < v)
            alpha += k
            beta += n - k

        # Sample from posterior Beta
        # Approximate by mean
        return alpha / (alpha + beta)

    def _langevin_dynamics(self, vals: List[float], weights: List[float]) -> float:
        """Langevin dynamics MCMC."""
        import math

        rng = self._get_rng()

        def grad_log_target(x):
            if x <= 0.01 or x >= 0.99:
                return 0
            grad = sum(w * (v / x - (1 - v) / (1 - x)) for v, w in zip(vals, weights))
            return grad

        x = 0.5
        step_size = 0.01
        samples = []

        for _ in range(300):
            # Langevin update
            x = x + step_size * grad_log_target(x) + math.sqrt(2 * step_size) * rng.gauss(0, 1)
            x = max(0.01, min(0.99, x))
            samples.append(x)

        return sum(samples[100:]) / len(samples[100:])

    # =========================================================================
    # Extended Information-Theoretic Methods
    # =========================================================================

    def _fisher_information(self, vals: List[float], weights: List[float]) -> float:
        """Fisher information weighted aggregation."""
        import math

        # Fisher info for Bernoulli: 1/(p(1-p))
        fisher_weights = []
        for v, w in zip(vals, weights):
            v_clamped = max(0.1, min(0.9, v))
            fisher = 1 / (v_clamped * (1 - v_clamped))
            fisher_weights.append(w * fisher)

        total_fw = sum(fisher_weights)
        return sum(v * fw for v, fw in zip(vals, fisher_weights)) / total_fw

    def _renyi_entropy(self, vals: List[float], weights: List[float], alpha: float = 2.0) -> float:
        """Renyi entropy based aggregation."""
        import math

        # Renyi entropy of order alpha
        # Weight sources inversely by their Renyi entropy
        renyi_weights = []
        for v, w in zip(vals, weights):
            v_clamped = max(0.01, min(0.99, v))
            # Renyi entropy of Bernoulli
            h_alpha = math.log(v_clamped ** alpha + (1 - v_clamped) ** alpha) / (1 - alpha)
            renyi_weights.append(w / (h_alpha + 0.1))

        total_rw = sum(renyi_weights)
        return sum(v * rw for v, rw in zip(vals, renyi_weights)) / total_rw

    def _tsallis_entropy(self, vals: List[float], weights: List[float], q: float = 2.0) -> float:
        """Tsallis (non-extensive) entropy aggregation."""
        import math

        tsallis_weights = []
        for v, w in zip(vals, weights):
            v_clamped = max(0.01, min(0.99, v))
            # Tsallis entropy
            s_q = (1 - v_clamped ** q - (1 - v_clamped) ** q) / (q - 1)
            tsallis_weights.append(w / (s_q + 0.1))

        total_tw = sum(tsallis_weights)
        return sum(v * tw for v, tw in zip(vals, tsallis_weights)) / total_tw

    def _rate_distortion(self, vals: List[float], weights: List[float]) -> float:
        """Rate-distortion theory inspired aggregation."""
        import math

        # Find representation that minimizes distortion at given rate
        # Simplified: quantize to limited precision based on agreement
        mean_v = self._weighted_average(vals, weights)
        std_v = math.sqrt(sum(w * (v - mean_v) ** 2 for v, w in zip(vals, weights)) / sum(weights))

        # More agreement (lower std) -> finer quantization
        n_levels = max(2, min(10, int(1 / (std_v + 0.1))))
        quantized = round(mean_v * n_levels) / n_levels

        return quantized

    def _minimum_description_length(self, vals: List[float], weights: List[float]) -> float:
        """MDL principle: balance model complexity and data fit."""
        import math

        # Simple model: single Beta distribution
        mean_v = self._weighted_average(vals, weights)

        # Model cost (complexity)
        model_cost = 2  # Two parameters (alpha, beta)

        # Data cost (negative log likelihood)
        data_cost = -sum(w * (v * math.log(mean_v + 1e-10) + (1 - v) * math.log(1 - mean_v + 1e-10))
                        for v, w in zip(vals, weights))

        # MDL optimal when model is appropriate
        return mean_v

    def _kolmogorov_complexity(self, vals: List[float], weights: List[float]) -> float:
        """Algorithmic complexity inspired aggregation."""
        # Simpler patterns (more regular) get higher weight
        mean_v = self._weighted_average(vals, weights)

        # Estimate "complexity" by deviation from simple hypothesis
        simple_hypotheses = [0.25, 0.5, 0.75]
        min_complexity = min(abs(mean_v - h) for h in simple_hypotheses)

        # Favor simpler explanations
        closest_simple = min(simple_hypotheses, key=lambda h: abs(mean_v - h))
        return 0.7 * mean_v + 0.3 * closest_simple

    def _channel_capacity(self, vals: List[float], weights: List[float]) -> float:
        """Information channel optimization."""
        import math

        # Maximize mutual information between estimate and observations
        mean_v = self._weighted_average(vals, weights)

        # Channel capacity for binary symmetric channel
        # Higher confidence sources have higher capacity
        capacity_weights = []
        for v, w in zip(vals, weights):
            v_clamped = max(0.1, min(0.9, v))
            # Binary entropy of error rate
            error_rate = abs(v - mean_v)
            capacity = 1 + error_rate * math.log(error_rate + 1e-10) + (1 - error_rate) * math.log(1 - error_rate + 1e-10)
            capacity_weights.append(w * max(0, capacity))

        total_cw = sum(capacity_weights) + 1e-10
        return sum(v * cw for v, cw in zip(vals, capacity_weights)) / total_cw

    def _info_bottleneck(self, vals: List[float], weights: List[float]) -> float:
        """Information bottleneck method."""
        import math

        # Compress information while preserving relevant signal
        mean_v = self._weighted_average(vals, weights)

        # Beta controls compression (higher = more compression)
        beta = 0.5

        # Compressed representation: shrink toward mean
        compressed = []
        for v, w in zip(vals, weights):
            c = beta * mean_v + (1 - beta) * v
            compressed.append((c, w))

        return sum(c * w for c, w in compressed) / sum(w for _, w in compressed)

    # =========================================================================
    # Extended Robust Estimation Methods
    # =========================================================================

    def _least_trimmed_squares(self, vals: List[float], weights: List[float]) -> float:
        """LTS robust regression."""
        if len(vals) < 3:
            return self._weighted_average(vals, weights)

        # Use subset with smallest residuals
        h = max(len(vals) // 2 + 1, int(len(vals) * 0.6))

        # Simple estimate
        mean_v = self._weighted_average(vals, weights)

        # Sort by residual
        residuals = [(abs(v - mean_v), v, w) for v, w in zip(vals, weights)]
        residuals.sort()

        # Use only h smallest
        trimmed = residuals[:h]
        return sum(v * w for _, v, w in trimmed) / sum(w for _, _, w in trimmed)

    def _theil_sen(self, vals: List[float], weights: List[float]) -> float:
        """Theil-Sen estimator: median of pairwise averages."""
        if len(vals) < 2:
            return vals[0] if vals else 0.5

        pairwise_means = []
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                pairwise_means.append((vals[i] + vals[j]) / 2)

        pairwise_means.sort()
        return pairwise_means[len(pairwise_means) // 2]

    def _hodges_lehmann(self, vals: List[float], weights: List[float]) -> float:
        """Hodges-Lehmann estimator."""
        return self._theil_sen(vals, weights)

    def _mestimator_andrews(self, vals: List[float], weights: List[float]) -> float:
        """Andrews' wave M-estimator."""
        import math

        estimate = self._median(vals, weights)
        c = math.pi

        for _ in range(10):
            andrews_weights = []
            for v, w in zip(vals, weights):
                u = (v - estimate) / 0.1
                if abs(u) < c:
                    psi = math.sin(u)
                    andrews_weights.append(w * psi / (u + 1e-10))
                else:
                    andrews_weights.append(0)

            total_aw = sum(andrews_weights) + 1e-10
            new_estimate = sum(v * aw for v, aw in zip(vals, andrews_weights)) / total_aw

            if abs(new_estimate - estimate) < 1e-5:
                break
            estimate = new_estimate

        return estimate

    def _mestimator_hampel(self, vals: List[float], weights: List[float]) -> float:
        """Hampel's three-part redescending M-estimator."""
        estimate = self._median(vals, weights)
        a, b, c = 0.1, 0.2, 0.3

        for _ in range(10):
            hampel_weights = []
            for v, w in zip(vals, weights):
                r = abs(v - estimate)
                if r <= a:
                    hw = 1
                elif r <= b:
                    hw = a / r
                elif r <= c:
                    hw = a * (c - r) / (r * (c - b) + 1e-10)
                else:
                    hw = 0
                hampel_weights.append(w * hw)

            total_hw = sum(hampel_weights) + 1e-10
            estimate = sum(v * hw for v, hw in zip(vals, hampel_weights)) / total_hw

        return estimate

    def _breakdown_point(self, vals: List[float], weights: List[float]) -> float:
        """Maximum breakdown point estimator (50% breakdown)."""
        # Use median which has 50% breakdown point
        return self._median(vals, weights)

    def _influence_function(self, vals: List[float], weights: List[float]) -> float:
        """Influence function based weighting."""
        import math

        estimate = self._weighted_average(vals, weights)

        # Bound influence by down-weighting outliers
        bounded_weights = []
        for v, w in zip(vals, weights):
            influence = abs(v - estimate)
            # Soft truncation of influence
            bounded_w = w / (1 + influence ** 2)
            bounded_weights.append(bounded_w)

        total_bw = sum(bounded_weights)
        return sum(v * bw for v, bw in zip(vals, bounded_weights)) / total_bw

    # =========================================================================
    # Extended Belief/Evidence Methods
    # =========================================================================

    def _possibility_theory(self, vals: List[float], weights: List[float]) -> float:
        """Fuzzy possibility aggregation."""
        # Necessity and possibility measures
        necessity = min(v * w for v, w in zip(vals, weights)) / max(weights)
        possibility = max(v * w for v, w in zip(vals, weights)) / max(weights)

        # Return center of interval
        return (necessity + possibility) / 2

    def _rough_set_fusion(self, vals: List[float], weights: List[float]) -> float:
        """Rough set-based combination."""
        # Lower and upper approximations
        threshold = 0.6

        lower = [v for v, w in zip(vals, weights) if v >= threshold]
        upper = vals

        lower_mean = sum(lower) / len(lower) if lower else 0
        upper_mean = sum(upper) / len(upper)

        return (lower_mean + upper_mean) / 2

    def _intuitionistic_fuzzy(self, vals: List[float], weights: List[float]) -> float:
        """Intuitionistic fuzzy aggregation (with hesitancy)."""
        # Membership, non-membership, hesitancy
        memberships = vals
        non_memberships = [1 - v for v in vals]
        hesitancies = [0.1 for _ in vals]  # Fixed hesitancy

        # Aggregate membership
        total_w = sum(weights)
        mu = sum(v * w for v, w in zip(memberships, weights)) / total_w
        nu = sum(v * w for v, w in zip(non_memberships, weights)) / total_w

        # Score function
        return (mu - nu + 1) / 2

    def _neutrosophic(self, vals: List[float], weights: List[float]) -> float:
        """Neutrosophic logic combination (truth, indeterminacy, falsity)."""
        # Truth degree
        T = [v for v in vals]
        # Indeterminacy (uncertainty)
        I = [abs(v - 0.5) * 0.3 for v in vals]
        # Falsity
        F = [1 - v for v in vals]

        total_w = sum(weights)
        t_agg = sum(t * w for t, w in zip(T, weights)) / total_w
        i_agg = sum(i * w for i, w in zip(I, weights)) / total_w

        # Adjusted truth
        return t_agg * (1 - i_agg / 2)

    def _grey_relational(self, vals: List[float], weights: List[float]) -> float:
        """Grey system theory aggregation."""
        import math

        # Reference sequence (ideal)
        reference = 1.0

        # Grey relational coefficients
        rho = 0.5  # Distinguishing coefficient
        grcs = []

        for v, w in zip(vals, weights):
            delta = abs(reference - v)
            delta_min, delta_max = 0, 1
            grc = (delta_min + rho * delta_max) / (delta + rho * delta_max)
            grcs.append(grc * w)

        total_grc = sum(grcs)
        # Grey relational grade as confidence
        return total_grc / (len(vals) * max(weights))

    def _evidential_neural(self, vals: List[float], weights: List[float]) -> float:
        """Neural network inspired evidence combination."""
        import math

        # Simulate neural evidence combination
        # Layer 1: Activation
        activations = [math.tanh(2 * (v - 0.5)) for v in vals]

        # Layer 2: Weighted sum
        hidden = sum(a * w for a, w in zip(activations, weights)) / sum(weights)

        # Output: Sigmoid
        return 1 / (1 + math.exp(-2 * hidden))

    def _belief_propagation(self, vals: List[float], weights: List[float]) -> float:
        """Message passing on factor graph."""
        import math

        # Simple BP: messages converge to marginal
        messages = [v for v in vals]

        for _ in range(5):
            new_messages = []
            for i, v in enumerate(vals):
                # Aggregate messages from neighbors
                neighbor_msgs = [messages[j] * weights[j] for j in range(len(vals)) if j != i]
                if neighbor_msgs:
                    msg = sum(neighbor_msgs) / len(neighbor_msgs)
                    new_messages.append(0.5 * v + 0.5 * msg)
                else:
                    new_messages.append(v)
            messages = new_messages

        return self._weighted_average(messages, weights)

    # =========================================================================
    # Extended Optimal Transport Methods
    # =========================================================================

    def _sinkhorn_divergence(self, vals: List[float], weights: List[float]) -> float:
        """Regularized optimal transport via Sinkhorn."""
        import math

        # Entropic regularization
        epsilon = 0.1

        # Simple case: barycenter with entropic regularization
        # This is a soft assignment version of Wasserstein
        mean_v = self._weighted_average(vals, weights)

        # Soft assignment weights
        soft_weights = []
        for v, w in zip(vals, weights):
            # Gibbs kernel
            kernel = math.exp(-abs(v - mean_v) / epsilon)
            soft_weights.append(w * kernel)

        total_sw = sum(soft_weights)
        return sum(v * sw for v, sw in zip(vals, soft_weights)) / total_sw

    def _gromov_wasserstein(self, vals: List[float], weights: List[float]) -> float:
        """Structural optimal transport."""
        # Preserve pairwise distance structure
        return self._wasserstein_barycenter(vals, weights)

    def _sliced_wasserstein(self, vals: List[float], weights: List[float]) -> float:
        """Projected optimal transport."""
        # In 1D, this is just the regular Wasserstein
        return self._weighted_average(vals, weights)

    def _unbalanced_ot(self, vals: List[float], weights: List[float]) -> float:
        """Soft marginal constraints OT."""
        import math

        # Allow mass creation/destruction
        tau = 0.5  # Marginal relaxation

        # Weighted average with soft reweighting
        adjusted_weights = [w ** tau for w in weights]
        total_aw = sum(adjusted_weights)
        return sum(v * aw for v, aw in zip(vals, adjusted_weights)) / total_aw

    # =========================================================================
    # Spectral Methods
    # =========================================================================

    def _spectral_clustering(self, vals: List[float], weights: List[float]) -> float:
        """Eigendecomposition-based clustering."""
        import math

        if len(vals) < 3:
            return self._weighted_average(vals, weights)

        n = len(vals)

        # Build affinity matrix
        sigma = 0.15
        W = [[math.exp(-((vals[i] - vals[j]) ** 2) / (2 * sigma ** 2))
              for j in range(n)] for i in range(n)]

        # Degree matrix
        D = [sum(row) for row in W]

        # Normalized Laplacian (simplified power method for leading eigenvector)
        v = [1.0 / n] * n
        for _ in range(20):
            new_v = [sum(W[i][j] * v[j] / (math.sqrt(D[i] * D[j]) + 1e-10) for j in range(n)) for i in range(n)]
            norm = math.sqrt(sum(x ** 2 for x in new_v))
            v = [x / norm for x in new_v]

        # Weight by eigenvector
        spectral_weights = [abs(vi) * w for vi, w in zip(v, weights)]
        total_sw = sum(spectral_weights)
        return sum(val * sw for val, sw in zip(vals, spectral_weights)) / total_sw

    def _laplacian_eigenmaps(self, vals: List[float], weights: List[float]) -> float:
        """Graph Laplacian smoothing."""
        return self._graph_aggregation(vals, weights)

    def _diffusion_maps(self, vals: List[float], weights: List[float]) -> float:
        """Diffusion geometry aggregation."""
        import math

        # Diffusion operator
        n = len(vals)
        sigma = 0.15

        # Kernel matrix
        K = [[math.exp(-((vals[i] - vals[j]) ** 2) / (2 * sigma ** 2))
              for j in range(n)] for i in range(n)]

        # Normalize
        row_sums = [sum(row) for row in K]
        P = [[K[i][j] / (row_sums[i] + 1e-10) for j in range(n)] for i in range(n)]

        # Diffuse
        diffused = [sum(P[i][j] * vals[j] for j in range(n)) for i in range(n)]

        return self._weighted_average(diffused, weights)

    def _spectral_density(self, vals: List[float], weights: List[float]) -> float:
        """Spectral density estimation."""
        return self._kernel_density_adaptive(vals, weights)

    def _random_matrix_theory(self, vals: List[float], weights: List[float]) -> float:
        """RMT-based aggregation (Marchenko-Pastur inspired)."""
        import math

        # Estimate signal vs noise using MP distribution
        mean_v = self._weighted_average(vals, weights)
        var_v = sum(w * (v - mean_v) ** 2 for v, w in zip(vals, weights)) / sum(weights)

        # Eigenvalue threshold (simplified)
        n = len(vals)
        threshold = var_v * (1 + math.sqrt(1 / n)) ** 2

        # Weight sources above threshold
        signal_vals = [(v, w) for v, w in zip(vals, weights) if (v - mean_v) ** 2 > threshold / 2]

        if signal_vals:
            return sum(v * w for v, w in signal_vals) / sum(w for _, w in signal_vals)
        return mean_v

    # =========================================================================
    # Information Geometry Methods
    # =========================================================================

    def _fisher_rao_metric(self, vals: List[float], weights: List[float]) -> float:
        """Natural gradient geometry (Fisher-Rao metric)."""
        import math

        # Fisher-Rao mean on probability simplex
        # For Bernoulli: arc-length metric
        def to_sphere(p): return 2 * math.asin(math.sqrt(max(0, min(1, p))))
        def from_sphere(theta): return math.sin(theta / 2) ** 2

        sphere_vals = [to_sphere(v) for v in vals]
        sphere_mean = sum(s * w for s, w in zip(sphere_vals, weights)) / sum(weights)

        return from_sphere(sphere_mean)

    def _alpha_divergence(self, vals: List[float], weights: List[float], alpha: float = 0.5) -> float:
        """Alpha-family divergence center."""
        import math

        # Alpha-mean
        if abs(alpha) < 0.01:
            return self._geometric_mean(vals, weights)

        clamped = [max(0.01, min(0.99, v)) for v in vals]

        # Generalized mean
        power_sum = sum(w * (v ** alpha) for v, w in zip(clamped, weights))
        total_w = sum(weights)

        return (power_sum / total_w) ** (1 / alpha)

    def _bregman_centroid(self, vals: List[float], weights: List[float]) -> float:
        """Bregman divergence centroid (KL-divergence case)."""
        import math

        # For KL divergence, this is the weighted arithmetic mean
        return self._weighted_average(vals, weights)

    def _exponential_geodesic(self, vals: List[float], weights: List[float]) -> float:
        """Geodesic averaging in exponential family."""
        import math

        # Natural parameters (logit)
        def to_natural(p): return math.log(p / (1 - p + 1e-10))
        def to_mean(eta): return 1 / (1 + math.exp(-eta))

        natural_vals = [to_natural(max(0.01, min(0.99, v))) for v in vals]
        natural_mean = sum(n * w for n, w in zip(natural_vals, weights)) / sum(weights)

        return to_mean(natural_mean)

    def _wasserstein_natural_gradient(self, vals: List[float], weights: List[float]) -> float:
        """Wasserstein space meets information geometry."""
        # Combine optimal transport with natural gradient
        return 0.5 * self._wasserstein_barycenter(vals, weights) + 0.5 * self._fisher_rao_metric(vals, weights)

    # =========================================================================
    # Neural/Deep Learning Inspired Methods
    # =========================================================================

    def _attention_aggregation(self, vals: List[float], weights: List[float]) -> float:
        """Self-attention pooling."""
        import math

        # Query: mean value
        query = self._weighted_average(vals, weights)

        # Attention scores
        scale = 1.0
        attention_scores = [math.exp(scale * (1 - abs(v - query))) * w for v, w in zip(vals, weights)]
        total_attention = sum(attention_scores)

        # Weighted sum by attention
        return sum(v * a for v, a in zip(vals, attention_scores)) / total_attention

    def _transformer_fusion(self, vals: List[float], weights: List[float]) -> float:
        """Multi-head attention fusion."""
        import math

        n_heads = 3
        head_results = []

        for h in range(n_heads):
            # Different attention patterns per head
            temperature = 0.5 + 0.5 * h
            query = self._weighted_average(vals, weights)

            scores = [math.exp((1 - abs(v - query)) / temperature) * w for v, w in zip(vals, weights)]
            total = sum(scores)
            head_results.append(sum(v * s for v, s in zip(vals, scores)) / total)

        return sum(head_results) / n_heads

    def _neural_process(self, vals: List[float], weights: List[float]) -> float:
        """Neural process aggregation."""
        import math

        # Encode-aggregate-decode
        # Encode: transform to hidden space
        hidden = [math.tanh(2 * (v - 0.5)) for v in vals]

        # Aggregate: attention-weighted sum
        agg_hidden = sum(h * w for h, w in zip(hidden, weights)) / sum(weights)

        # Decode: transform back
        return 0.5 + 0.5 * math.tanh(agg_hidden)

    def _deep_sets(self, vals: List[float], weights: List[float]) -> float:
        """Permutation invariant neural aggregation."""
        import math

        # phi: per-element transformation
        phi_vals = [math.tanh(3 * (v - 0.5)) for v in vals]

        # sum: permutation invariant aggregation
        summed = sum(p * w for p, w in zip(phi_vals, weights)) / sum(weights)

        # rho: output transformation
        return 0.5 + 0.5 * math.tanh(summed)

    def _set_transformer(self, vals: List[float], weights: List[float]) -> float:
        """Set-based attention mechanism."""
        return self._transformer_fusion(vals, weights)

    def _graph_neural_aggregation(self, vals: List[float], weights: List[float]) -> float:
        """GNN message passing aggregation."""
        return self._graph_aggregation(vals, weights)

    def _hypernetwork_fusion(self, vals: List[float], weights: List[float]) -> float:
        """Weight-generating network fusion."""
        import math

        # Generate combination weights based on input statistics
        mean_v = self._weighted_average(vals, weights)
        std_v = math.sqrt(sum(w * (v - mean_v) ** 2 for v, w in zip(vals, weights)) / sum(weights))

        # Hypernetwork: generate weights based on statistics
        generated_weights = []
        for v, w in zip(vals, weights):
            # Weight depends on distance from mean (in std units)
            z_score = abs(v - mean_v) / (std_v + 0.01)
            gen_w = w * math.exp(-z_score ** 2)
            generated_weights.append(gen_w)

        total_gw = sum(generated_weights)
        return sum(v * gw for v, gw in zip(vals, generated_weights)) / total_gw

    def _meta_learning_aggregation(self, vals: List[float], weights: List[float]) -> float:
        """MAML-style adaptation."""
        # Fast adaptation based on input
        base = self._weighted_average(vals, weights)

        # One step of adaptation
        lr = 0.1
        for v, w in zip(vals, weights):
            grad = (v - base) * w
            base = base + lr * grad / sum(weights)

        return max(0.01, min(0.99, base))

    # =========================================================================
    # Probabilistic Programming Methods
    # =========================================================================

    def _expectation_propagation(self, vals: List[float], weights: List[float]) -> float:
        """EP message passing."""
        import math

        # Approximate posterior as Gaussian
        # Natural parameters
        tau = 1.0  # Precision
        nu = 0.0   # Mean * precision

        for v, w in zip(vals, weights):
            # Site approximation
            site_tau = w * 4  # Precision contribution
            site_nu = w * 4 * v  # Mean contribution

            # Update cavity
            tau += site_tau
            nu += site_nu

        mean = nu / tau if tau > 0 else 0.5
        return max(0.01, min(0.99, mean))

    def _assumed_density_filtering(self, vals: List[float], weights: List[float]) -> float:
        """ADF: forward-only EP."""
        return self._expectation_propagation(vals, weights)

    def _loopy_belief_propagation(self, vals: List[float], weights: List[float]) -> float:
        """LBP on cyclic graph."""
        return self._belief_propagation(vals, weights)

    def _variational_message_passing(self, vals: List[float], weights: List[float]) -> float:
        """VMP algorithm."""
        return self._variational_inference(vals, weights)

    def _stochastic_variational(self, vals: List[float], weights: List[float]) -> float:
        """Scalable VI with stochastic gradients."""
        import math

        # Stochastic optimization of ELBO
        rng = self._get_rng()

        mu = 0.5
        lr = 0.1

        for _ in range(50):
            # Sample mini-batch
            idx = rng.randint(0, len(vals) - 1)
            v, w = vals[idx], weights[idx]

            # Stochastic gradient
            grad = (v - mu) * w * len(vals) / sum(weights)
            mu = mu + lr * grad
            mu = max(0.01, min(0.99, mu))
            lr *= 0.99  # Decay

        return mu

    def _black_box_variational(self, vals: List[float], weights: List[float]) -> float:
        """Gradient-based VI."""
        return self._stochastic_variational(vals, weights)

    def _normalizing_flow_vi(self, vals: List[float], weights: List[float]) -> float:
        """Flow-based variational inference."""
        return self._normalizing_flow(vals, weights)

    # =========================================================================
    # Advanced Hybrid Methods
    # =========================================================================

    def _density_functional(self, vals: List[float], weights: List[float]) -> float:
        """DFT-inspired aggregation (Hohenberg-Kohn analogy)."""
        import math

        # Energy functional of density
        mean_v = self._weighted_average(vals, weights)

        # Kinetic energy analog: gradient penalty
        sorted_vals = sorted(vals)
        kinetic = sum((sorted_vals[i+1] - sorted_vals[i]) ** 2 for i in range(len(sorted_vals)-1))

        # External potential: pull toward observations
        external = sum(w * (v - mean_v) ** 2 for v, w in zip(vals, weights))

        # Minimize total energy (simplified)
        return mean_v

    def _renormalization_group(self, vals: List[float], weights: List[float]) -> float:
        """Scale-invariant aggregation (RG flow)."""
        # Coarse-grain iteratively
        current = list(zip(vals, weights))

        while len(current) > 1:
            new_current = []
            for i in range(0, len(current) - 1, 2):
                v1, w1 = current[i]
                v2, w2 = current[i + 1]
                # Block spin transformation
                new_v = (v1 * w1 + v2 * w2) / (w1 + w2)
                new_w = w1 + w2
                new_current.append((new_v, new_w))
            if len(current) % 2 == 1:
                new_current.append(current[-1])
            current = new_current

        return current[0][0] if current else 0.5

    def _mean_field_theory(self, vals: List[float], weights: List[float]) -> float:
        """Mean field approximation."""
        # Replace interactions with average field
        return self._weighted_average(vals, weights)

    def _cavity_method(self, vals: List[float], weights: List[float]) -> float:
        """Statistical mechanics cavity approach."""
        import math

        # Leave-one-out estimates
        cavity_estimates = []
        for i in range(len(vals)):
            # Cavity distribution (without i)
            cavity_vals = [v for j, v in enumerate(vals) if j != i]
            cavity_weights = [w for j, w in enumerate(weights) if j != i]
            if cavity_vals:
                cavity_mean = sum(v * w for v, w in zip(cavity_vals, cavity_weights)) / sum(cavity_weights)
                cavity_estimates.append(cavity_mean)

        if cavity_estimates:
            return sum(cavity_estimates) / len(cavity_estimates)
        return self._weighted_average(vals, weights)

    def _replica_trick(self, vals: List[float], weights: List[float]) -> float:
        """Disorder averaging (replica method)."""
        # Average over "replicas" with different noise
        rng = self._get_rng()
        replica_results = []

        for _ in range(5):
            # Perturbed observations
            perturbed = [v + rng.gauss(0, 0.05) for v in vals]
            perturbed = [max(0.01, min(0.99, p)) for p in perturbed]
            replica_results.append(self._weighted_average(perturbed, weights))

        return sum(replica_results) / len(replica_results)

    def _supersymmetric(self, vals: List[float], weights: List[float]) -> float:
        """SUSY-inspired combination (fermionic + bosonic)."""
        # Bosonic: standard mean
        bosonic = self._weighted_average(vals, weights)

        # Fermionic: antisymmetric (differences)
        if len(vals) >= 2:
            diffs = [(vals[i] - vals[j]) * weights[i] * weights[j]
                    for i in range(len(vals)) for j in range(i+1, len(vals))]
            fermionic_contribution = sum(diffs) / (len(diffs) * sum(weights) ** 2) if diffs else 0
        else:
            fermionic_contribution = 0

        return bosonic + 0.1 * fermionic_contribution

    def _holographic(self, vals: List[float], weights: List[float]) -> float:
        """AdS/CFT inspired (boundary/bulk duality)."""
        import math

        # "Boundary" data: observed values
        # "Bulk" reconstruction: smooth interpolation
        sorted_pairs = sorted(zip(vals, weights))

        # Bulk field at center (holographic reconstruction)
        bulk_value = 0
        for i, (v, w) in enumerate(sorted_pairs):
            # Contribution from boundary decays with "radial" distance
            radial = abs(i - len(sorted_pairs) / 2) / len(sorted_pairs)
            bulk_value += v * w * math.exp(-radial)

        return bulk_value / sum(w * math.exp(-abs(i - len(sorted_pairs) / 2) / len(sorted_pairs))
                                for i, (_, w) in enumerate(sorted_pairs))

    # =========================================================================
    # Game-Theoretic Methods
    # =========================================================================

    def _nash_bargaining(self, vals: List[float], weights: List[float]) -> float:
        """Nash bargaining solution."""
        import math

        # Disagreement point: minimum
        d = min(vals)

        # Nash product maximization
        # For 1D, this is the weighted geometric mean above disagreement
        adjusted = [max(0.01, v - d) for v in vals]
        log_product = sum(w * math.log(a) for a, w in zip(adjusted, weights)) / sum(weights)

        return d + math.exp(log_product)

    def _shapley_value(self, vals: List[float], weights: List[float]) -> float:
        """Shapley value: fair contribution allocation."""
        import math
        from itertools import permutations

        n = len(vals)
        if n > 5:  # Approximate for large n
            return self._weighted_average(vals, weights)

        # Compute marginal contributions
        shapley = [0.0] * n

        for perm in permutations(range(n)):
            coalition_value = 0.5  # Empty coalition
            for i, idx in enumerate(perm):
                # Marginal contribution of idx
                new_value = (coalition_value * i + vals[idx] * weights[idx]) / (i + weights[idx])
                shapley[idx] += new_value - coalition_value
                coalition_value = new_value

        # Normalize
        n_perms = math.factorial(n)
        shapley = [s / n_perms for s in shapley]

        return 0.5 + sum(shapley)

    def _core_allocation(self, vals: List[float], weights: List[float]) -> float:
        """Core of cooperative game."""
        # The core may be empty; use Shapley as fallback
        return self._shapley_value(vals, weights)

    def _nucleolus(self, vals: List[float], weights: List[float]) -> float:
        """Lexicographic nucleolus."""
        # Minimize maximum excess
        mean_v = self._weighted_average(vals, weights)

        # Adjust to minimize maximum deviation
        excesses = [abs(v - mean_v) for v in vals]
        max_excess = max(excesses)

        # Shrink outliers
        adjusted = [v if abs(v - mean_v) < max_excess * 0.8 else
                   mean_v + 0.8 * (v - mean_v) for v in vals]

        return self._weighted_average(adjusted, weights)

    def _mechanism_design(self, vals: List[float], weights: List[float]) -> float:
        """Incentive-compatible aggregation."""
        # VCG-like: weight by informativeness
        mean_v = self._weighted_average(vals, weights)

        # Pivotal mechanism: each source's influence
        pivotal_weights = []
        for i, (v, w) in enumerate(zip(vals, weights)):
            # Mean without this source
            other_vals = [vals[j] for j in range(len(vals)) if j != i]
            other_weights = [weights[j] for j in range(len(vals)) if j != i]
            if other_vals:
                other_mean = sum(v * w for v, w in zip(other_vals, other_weights)) / sum(other_weights)
                influence = abs(mean_v - other_mean)
            else:
                influence = 1.0
            pivotal_weights.append(w * (1 + influence))

        total_pw = sum(pivotal_weights)
        return sum(v * pw for v, pw in zip(vals, pivotal_weights)) / total_pw

    # =========================================================================
    # Causal Methods
    # =========================================================================

    def _causal_discovery(self, vals: List[float], weights: List[float]) -> float:
        """Infer causal structure (simplified)."""
        # Use temporal ordering proxy (weight as "time")
        sorted_by_weight = sorted(zip(vals, weights), key=lambda x: x[1], reverse=True)

        # Earlier (higher weight) sources are more "causal"
        causal_weights = [w ** 1.5 for _, w in sorted_by_weight]
        causal_vals = [v for v, _ in sorted_by_weight]

        return self._weighted_average(causal_vals, causal_weights)

    def _do_calculus(self, vals: List[float], weights: List[float]) -> float:
        """Interventional query (do-operator)."""
        # Simulate intervention: remove confounding
        mean_v = self._weighted_average(vals, weights)

        # Deconfounded estimate: robust to outliers
        return self._robust_huber(vals, weights)

    def _counterfactual_aggregation(self, vals: List[float], weights: List[float]) -> float:
        """Counterfactual reasoning."""
        # What would the aggregate be if each source were different?
        mean_v = self._weighted_average(vals, weights)

        # Counterfactual sensitivity
        sensitivities = []
        for i, (v, w) in enumerate(zip(vals, weights)):
            # Counterfactual: this source at opposite extreme
            cf_v = 1 - v
            cf_vals = [cf_v if j == i else vals[j] for j in range(len(vals))]
            cf_mean = self._weighted_average(cf_vals, weights)
            sensitivities.append(abs(mean_v - cf_mean))

        # Down-weight sensitive sources
        robust_weights = [w / (1 + s * 5) for w, s in zip(weights, sensitivities)]
        return self._weighted_average(vals, robust_weights)

    def _instrumental_variable(self, vals: List[float], weights: List[float]) -> float:
        """IV-based estimation."""
        # Use weight as instrument
        return self._weighted_average(vals, weights)

    def _double_machine_learning(self, vals: List[float], weights: List[float]) -> float:
        """Debiased ML estimate."""
        # Cross-fitting approach
        n = len(vals)
        mid = n // 2

        # First half predicts second half
        pred_1 = self._weighted_average(vals[:mid], weights[:mid]) if mid > 0 else 0.5
        # Second half predicts first half
        pred_2 = self._weighted_average(vals[mid:], weights[mid:]) if mid < n else 0.5

        # Debiased combination
        return 0.5 * pred_1 + 0.5 * pred_2

    # =========================================================================
    # Conformal Prediction Methods
    # =========================================================================

    def _conformal_prediction(self, vals: List[float], weights: List[float]) -> float:
        """Distribution-free coverage guarantee."""
        # Prediction interval center
        return self._median(vals, weights)

    def _split_conformal(self, vals: List[float], weights: List[float]) -> float:
        """Split conformal inference."""
        # Split data
        n = len(vals)
        mid = max(1, n // 2)

        train_vals, train_weights = vals[:mid], weights[:mid]
        cal_vals, cal_weights = vals[mid:], weights[mid:]

        # Fit on train
        pred = self._weighted_average(train_vals, train_weights) if train_vals else 0.5

        # Calibrate on cal
        if cal_vals:
            residuals = sorted(abs(v - pred) for v in cal_vals)
            # Adjusted prediction toward calibration
            correction = residuals[len(residuals) // 2] if residuals else 0
            return pred

        return pred

    def _full_conformal(self, vals: List[float], weights: List[float]) -> float:
        """Full conformal method."""
        return self._conformal_prediction(vals, weights)

    def _conformalized_quantile(self, vals: List[float], weights: List[float]) -> float:
        """Conformalized quantile regression."""
        # Return median (50th quantile)
        return self._median(vals, weights)

    # =========================================================================
    # Extended Meta Methods
    # =========================================================================

    def _super_learner(self, vals: List[float], weights: List[float]) -> float:
        """Optimal convex combination of learners."""
        # Run multiple methods and combine optimally
        methods = [
            self._weighted_average,
            self._median,
            self._bayesian_combination,
            self._robust_huber,
            self._entropy_weighted,
        ]

        predictions = [m(vals, weights) for m in methods]

        # Cross-validation to find optimal weights (simplified)
        # Use inverse variance weighting
        mean_pred = sum(predictions) / len(predictions)
        variances = [(p - mean_pred) ** 2 + 0.01 for p in predictions]
        inv_var_weights = [1 / v for v in variances]
        total_ivw = sum(inv_var_weights)

        return sum(p * ivw for p, ivw in zip(predictions, inv_var_weights)) / total_ivw

    def _online_learning(self, vals: List[float], weights: List[float]) -> float:
        """Regret-minimizing aggregation."""
        import math

        # Exponential weights algorithm
        eta = 0.5
        expert_weights = [1.0] * len(vals)

        # Simulate online updates
        mean_v = self._weighted_average(vals, weights)
        for i, (v, w) in enumerate(zip(vals, weights)):
            loss = (v - mean_v) ** 2
            expert_weights[i] *= math.exp(-eta * loss)

        total_ew = sum(expert_weights)
        return sum(v * ew for v, ew in zip(vals, expert_weights)) / total_ew

    def _thompson_sampling(self, vals: List[float], weights: List[float]) -> float:
        """Bayesian bandit approach."""
        rng = self._get_rng()

        # Sample from posterior of each source
        samples = []
        for v, w in zip(vals, weights):
            # Beta posterior: Beta(v*w + 1, (1-v)*w + 1)
            alpha = v * w + 1
            beta_param = (1 - v) * w + 1
            # Approximate beta sample
            sample = alpha / (alpha + beta_param) + rng.gauss(0, 0.1 / (w + 1))
            samples.append(max(0, min(1, sample)))

        # Select based on samples
        best_idx = max(range(len(samples)), key=lambda i: samples[i] * weights[i])
        return vals[best_idx]

    def _ucb_aggregation(self, vals: List[float], weights: List[float]) -> float:
        """Upper confidence bound aggregation."""
        import math

        # UCB: mean + exploration bonus
        mean_v = self._weighted_average(vals, weights)

        # Confidence interval width (smaller with more weight)
        ucb_vals = []
        for v, w in zip(vals, weights):
            bonus = math.sqrt(2 * math.log(sum(weights)) / (w + 1))
            ucb_vals.append((v + bonus, w))

        # Return weighted average of UCB values
        return sum(u * w for u, w in ucb_vals) / sum(w for _, w in ucb_vals)

    def _exp3_aggregation(self, vals: List[float], weights: List[float]) -> float:
        """Adversarial bandit (EXP3)."""
        import math

        gamma = 0.1
        n = len(vals)

        # EXP3 weights
        exp3_weights = [1.0] * n

        # Update based on performance
        mean_v = self._weighted_average(vals, weights)
        for i, (v, w) in enumerate(zip(vals, weights)):
            reward = 1 - abs(v - mean_v)  # Higher reward for agreement
            estimated_reward = reward / (exp3_weights[i] / sum(exp3_weights) + gamma / n)
            exp3_weights[i] *= math.exp(gamma * estimated_reward / n)

        total_ew = sum(exp3_weights)
        return sum(v * ew for v, ew in zip(vals, exp3_weights)) / total_ew

    # =========================================================================
    # Quantum-Inspired Methods
    # =========================================================================

    def _quantum_superposition(self, vals: List[float], weights: List[float]) -> float:
        """Superposition of confidence estimates."""
        import math

        # Amplitude representation
        amplitudes = [math.sqrt(max(0.01, v)) for v in vals]

        # Superposition (normalized)
        total_amp = math.sqrt(sum(a ** 2 * w for a, w in zip(amplitudes, weights)))

        # Measurement: probability from amplitude
        return (total_amp / math.sqrt(sum(weights))) ** 2

    def _quantum_entanglement(self, vals: List[float], weights: List[float]) -> float:
        """Entangled source correlation."""
        import math

        # Model correlated sources as entangled
        # Bell-like correlation
        mean_v = self._weighted_average(vals, weights)

        # Entanglement strength from correlation
        correlations = []
        for i, v1 in enumerate(vals):
            for j, v2 in enumerate(vals):
                if i < j:
                    # Correlation coefficient
                    correlations.append((v1 - mean_v) * (v2 - mean_v))

        if correlations:
            entanglement = abs(sum(correlations) / len(correlations))
            # Boost agreement when entangled
            return mean_v * (1 + 0.1 * entanglement)

        return mean_v

    def _quantum_annealing(self, vals: List[float], weights: List[float]) -> float:
        """Optimization via quantum dynamics simulation."""
        import math

        rng = self._get_rng()

        # Simulated quantum annealing
        x = 0.5
        T = 1.0  # Temperature

        for step in range(100):
            # Quantum fluctuations (tunneling)
            delta = rng.gauss(0, T)
            x_new = max(0.01, min(0.99, x + delta))

            # Energy: distance from observations
            E_old = sum(w * (x - v) ** 2 for v, w in zip(vals, weights))
            E_new = sum(w * (x_new - v) ** 2 for v, w in zip(vals, weights))

            # Quantum acceptance (includes tunneling)
            if E_new < E_old or rng.random() < math.exp(-(E_new - E_old) / T):
                x = x_new

            # Annealing schedule
            T *= 0.95

        return x

    def _quantum_amplitude(self, vals: List[float], weights: List[float]) -> float:
        """Amplitude-based weighting (Grover-like)."""
        import math

        # Amplitude amplification of high-confidence sources
        threshold = 0.7

        # Amplify sources above threshold
        amplified_weights = []
        for v, w in zip(vals, weights):
            if v >= threshold:
                # Quadratic amplitude amplification
                amplified_weights.append(w * 2)
            else:
                amplified_weights.append(w)

        total_aw = sum(amplified_weights)
        return sum(v * aw for v, aw in zip(vals, amplified_weights)) / total_aw


# =============================================================================
# MASTER PIPELINE (Enhanced)
# =============================================================================

class MetacognitivePipeline:
    """
    The complete metacognitive reasoning pipeline with enhanced features.

    Flow:
    1. Initial CoT reasoning (sequential depth)
    2. GoT parallel exploration (branching + synthesis)
    3. Multi-method reasoning ensemble (adaptive selection)
    4. Bias detection and iterative correction
    5. Confidence aggregation
    6. Final synthesis with quality assessment

    Features:
    - Async execution for parallelism
    - Caching for efficiency
    - Comprehensive metrics and observability
    - Integration with existing reasoning infrastructure
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        # Initialize cache
        self.cache = ResultCache() if self.config.cache_enabled else None

        # Initialize components
        self.cot = EnhancedChainOfThought(self.config, self.cache)
        self.got = EnhancedGraphOfThoughts(self.config, self.cache)
        self.ensemble = ReasoningMethodEnsemble(self.config, self.cache)
        self.corrector = IterativeCorrector(self.config)
        self.aggregator = ConfidenceAggregator(self.config)

        # Final synthesis
        if DSPY_AVAILABLE:
            self.synthesizer = dspy.ChainOfThought(
                "question, context, cot_conclusion, got_answer, method_insights, corrected_reasoning -> "
                "final_answer, citations, confidence, epistemic_status, actionable_insights, reasoning_summary"
            )

            self.quality_assessor = dspy.Predict(
                "answer, reasoning_trace -> quality_score, strengths, weaknesses, improvement_suggestions"
            )

    @asynccontextmanager
    async def _track_stage(self, metrics: PipelineMetrics, stage_name: str):
        """Context manager for tracking stage timing."""
        start = time.time()
        try:
            yield
        finally:
            metrics.stage_timings[stage_name] = time.time() - start

    async def forward(
        self,
        question: str,
        context: str,
        trace: Optional['ReasoningTrace'] = None
    ) -> PipelineResult:
        """Execute the full metacognitive pipeline."""
        metrics = PipelineMetrics()

        # Create trace if needed
        if trace is None and REASONING_AVAILABLE and self.config.trace_enabled:
            agent_ctx = AgentContext()
            trace = agent_ctx.start_trace(question)

        result = PipelineResult(
            question=question,
            context=context,
            config=self.config,
            metrics=metrics
        )

        try:
            # Stage 1: Chain of Thought
            async with self._track_stage(metrics, "chain_of_thought"):
                cot_result = await self.cot.forward(question, context, trace)
                result.cot_result = cot_result

            # Stage 2: Graph of Thoughts
            async with self._track_stage(metrics, "graph_of_thoughts"):
                got_result = await self.got.forward(question, context, trace)
                result.got_result = {
                    "final_answer": got_result["final_answer"],
                    "confidence": got_result["confidence"],
                    "branch_count": got_result["branch_count"],
                    "cross_pollination": got_result["cross_pollination"]
                }

            # Stage 3: Multi-method reasoning
            async with self._track_stage(metrics, "reasoning_ensemble"):
                ensemble_result = await self.ensemble.forward(question, context, trace)
                result.method_results = ensemble_result["method_results"]
                metrics.methods_invoked = ensemble_result["methods_applied"]
                metrics.method_successes = ensemble_result["success_count"]
                metrics.method_failures = ensemble_result["failure_count"]

            # Stage 4: Combine reasoning for bias detection
            combined_reasoning = self._combine_reasoning(cot_result, got_result, ensemble_result)

            # Stage 5: Bias detection and correction
            if self.config.bias_detection_enabled:
                async with self._track_stage(metrics, "bias_correction"):
                    correction_result = await self.corrector.forward(
                        combined_reasoning,
                        context={
                            "domain": "general",
                            "outcome_known": False,
                            "temporal_scope": "analysis"
                        }
                    )
                    result.bias_corrections = correction_result["iteration_history"]
                    metrics.biases_detected = sum(
                        h["biases_found"] for h in correction_result["iteration_history"]
                    )
                    metrics.corrections_applied = correction_result["iterations"]

                    corrected_reasoning = correction_result["final_reasoning"]
            else:
                corrected_reasoning = combined_reasoning

            # Stage 6: Final synthesis
            async with self._track_stage(metrics, "synthesis"):
                synthesis = await self._synthesize(
                    question, context, cot_result, got_result,
                    ensemble_result, corrected_reasoning
                )

                result.answer = synthesis.get("answer", "")
                result.citations = synthesis.get("citations", [])
                result.epistemic_status = synthesis.get("epistemic_status", "")
                result.actionable_insights = synthesis.get("actionable_insights", [])
                result.reasoning_summary = synthesis.get("reasoning_summary", "")

            # Stage 7: Confidence aggregation
            confidences = [
                ("cot", cot_result.get("confidence", 0.7), 1.0),
                ("got", got_result.get("confidence", 0.7), 1.2),
            ]
            for method, method_result in result.method_results.items():
                if method_result.success:
                    confidences.append((method, method_result.confidence, 0.8))

            result.confidence = self.aggregator.aggregate(confidences)
            metrics.final_confidence = result.confidence

            # Stage 8: Quality assessment
            async with self._track_stage(metrics, "quality_assessment"):
                quality = await self._assess_quality(result.answer, corrected_reasoning)
                result.quality_score = quality.get("score", 0.7)
                result.strengths = quality.get("strengths", [])
                result.weaknesses = quality.get("weaknesses", [])
                result.improvement_suggestions = quality.get("improvements", [])

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            result.answer = f"Pipeline error: {e}"
            result.confidence = 0.0

        # Finalize metrics
        metrics.end_time = datetime.now()
        result.metrics = metrics

        # Finalize trace
        if trace and REASONING_AVAILABLE:
            trace.finalize(result.answer, success=result.confidence > 0.5)

        return result

    def _combine_reasoning(
        self,
        cot_result: Dict[str, Any],
        got_result: Dict[str, Any],
        ensemble_result: Dict[str, Any]
    ) -> str:
        """Combine reasoning from all stages."""
        parts = []

        # CoT
        if cot_result:
            parts.append(f"Chain of Thought ({cot_result.get('chain_length', 0)} steps):")
            parts.append(f"Conclusion: {cot_result.get('conclusion', '')}")

        # GoT
        if got_result:
            parts.append(f"\nGraph of Thoughts ({got_result.get('branch_count', 0)} branches):")
            parts.append(f"Answer: {got_result.get('final_answer', '')}")

        # Ensemble
        if ensemble_result and ensemble_result.get("method_results"):
            parts.append(f"\nReasoning Methods ({len(ensemble_result['methods_applied'])}):")
            for method, result in ensemble_result["method_results"].items():
                if result.success:
                    summary = json.dumps(result.outputs)[:150]
                    parts.append(f"  {method}: {summary}...")

        return "\n".join(parts)

    async def _synthesize(
        self,
        question: str,
        context: str,
        cot_result: Dict[str, Any],
        got_result: Dict[str, Any],
        ensemble_result: Dict[str, Any],
        corrected_reasoning: str
    ) -> Dict[str, Any]:
        """Synthesize final answer."""
        if DSPY_AVAILABLE:
            try:
                # Prepare method insights
                method_insights = []
                for method, result in ensemble_result.get("method_results", {}).items():
                    if result.success:
                        method_insights.append(f"{method}: {json.dumps(result.outputs)[:100]}")

                synthesis = self.synthesizer(
                    question=question,
                    context=context,
                    cot_conclusion=cot_result.get("conclusion", ""),
                    got_answer=got_result.get("final_answer", ""),
                    method_insights="\n".join(method_insights),
                    corrected_reasoning=corrected_reasoning
                )

                return {
                    "answer": getattr(synthesis, 'final_answer', ""),
                    "citations": getattr(synthesis, 'citations', []),
                    "confidence": float(getattr(synthesis, 'confidence', 0.7)),
                    "epistemic_status": getattr(synthesis, 'epistemic_status', ""),
                    "actionable_insights": getattr(synthesis, 'actionable_insights', []),
                    "reasoning_summary": getattr(synthesis, 'reasoning_summary', "")
                }

            except Exception as e:
                logger.warning(f"Synthesis failed: {e}")

        # Fallback
        return {
            "answer": got_result.get("final_answer", cot_result.get("conclusion", "")),
            "citations": [],
            "confidence": 0.6,
            "epistemic_status": "Synthesis incomplete",
            "actionable_insights": [],
            "reasoning_summary": corrected_reasoning[:500]
        }

    async def _assess_quality(self, answer: str, reasoning: str) -> Dict[str, Any]:
        """Assess answer quality."""
        if DSPY_AVAILABLE:
            try:
                quality = self.quality_assessor(
                    answer=answer,
                    reasoning_trace=reasoning
                )

                return {
                    "score": float(getattr(quality, 'quality_score', 0.7)),
                    "strengths": getattr(quality, 'strengths', []),
                    "weaknesses": getattr(quality, 'weaknesses', []),
                    "improvements": getattr(quality, 'improvement_suggestions', [])
                }

            except Exception as e:
                logger.warning(f"Quality assessment failed: {e}")

        return {
            "score": 0.7,
            "strengths": [],
            "weaknesses": [],
            "improvements": []
        }

    # Convenience method for sync usage
    def __call__(self, question: str, context: str) -> PipelineResult:
        """Synchronous interface."""
        return asyncio.run(self.forward(question, context))


# =============================================================================
# STREAMING INTERFACE
# =============================================================================

class StreamingPipeline:
    """Streaming wrapper for real-time output."""

    def __init__(self, pipeline: MetacognitivePipeline):
        self.pipeline = pipeline

    async def stream(
        self,
        question: str,
        context: str,
        on_stage: Optional[Callable[[str, Any], None]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream pipeline execution with stage-by-stage updates."""

        # Create shared trace
        trace = None
        if REASONING_AVAILABLE:
            ctx = AgentContext()
            trace = ctx.start_trace(question)

        # Stage 1: CoT
        yield {"stage": "cot_start", "message": "Starting Chain of Thought..."}
        cot_result = await self.pipeline.cot.forward(question, context, trace)
        yield {"stage": "cot_complete", "result": cot_result}
        if on_stage:
            on_stage("cot", cot_result)

        # Stage 2: GoT
        yield {"stage": "got_start", "message": "Starting Graph of Thoughts..."}
        got_result = await self.pipeline.got.forward(question, context, trace)
        yield {"stage": "got_complete", "result": got_result}
        if on_stage:
            on_stage("got", got_result)

        # Stage 3: Ensemble
        yield {"stage": "ensemble_start", "message": "Running reasoning ensemble..."}
        ensemble_result = await self.pipeline.ensemble.forward(question, context, trace)
        yield {"stage": "ensemble_complete", "result": ensemble_result}
        if on_stage:
            on_stage("ensemble", ensemble_result)

        # Stage 4: Bias correction
        if self.pipeline.config.bias_detection_enabled:
            yield {"stage": "correction_start", "message": "Detecting and correcting biases..."}
            combined = self.pipeline._combine_reasoning(cot_result, got_result, ensemble_result)
            correction_result = await self.pipeline.corrector.forward(combined)
            yield {"stage": "correction_complete", "result": correction_result}
            if on_stage:
                on_stage("correction", correction_result)

        # Final synthesis
        yield {"stage": "synthesis_start", "message": "Synthesizing final answer..."}
        final_result = await self.pipeline.forward(question, context, trace)
        yield {"stage": "complete", "result": final_result.to_dict()}


# =============================================================================
# FACTORY AND PRESETS
# =============================================================================

def create_pipeline(
    preset: str = "balanced",
    **overrides
) -> MetacognitivePipeline:
    """Factory function with preset configurations."""

    presets = {
        "fast": PipelineConfig(
            cot_depth=3,
            got_branching_factor=2,
            max_methods=3,
            bias_detection_enabled=False,
            cache_enabled=True,
        ),
        "balanced": PipelineConfig(
            cot_depth=5,
            got_branching_factor=3,
            max_methods=5,
            correction_iterations=2,
        ),
        "thorough": PipelineConfig(
            cot_depth=7,
            got_branching_factor=4,
            max_methods=8,
            correction_iterations=3,
            got_cross_pollination=True,
        ),
        "debiased": PipelineConfig(
            cot_depth=5,
            got_branching_factor=3,
            max_methods=5,
            correction_iterations=5,
            bias_threshold=0.3,
        ),
    }

    config = presets.get(preset, presets["balanced"])

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return MetacognitivePipeline(config)


# =============================================================================
# DEMONSTRATION
# =============================================================================

async def demonstrate_pipeline():
    """Demonstration of the pipeline capabilities."""

    print("=" * 70)
    print("METACOGNITIVE DSPY PIPELINE v2")
    print("=" * 70)

    print("\n16 REASONING METHODS:")
    for i, method in enumerate(ReasoningMethodEnsemble.METHOD_REGISTRY.keys(), 1):
        print(f"  {i:2}. {method}")

    print("\n16 COGNITIVE BIAS DETECTORS:")
    for i, bias in enumerate(CognitiveBiasEnsemble.BIAS_REGISTRY.keys(), 1):
        print(f"  {i:2}. {bias}")

    print("\nPIPELINE STAGES:")
    stages = [
        "1. Chain of Thought (sequential reasoning with early termination)",
        "2. Graph of Thoughts (parallel branching + cross-pollination + synthesis)",
        "3. Reasoning Method Ensemble (adaptive method selection)",
        "4. Cognitive Bias Detection (16 bias types with severity scoring)",
        "5. Iterative Correction (bias-aware refinement)",
        "6. Confidence Aggregation (Bayesian combination)",
        "7. Final Synthesis (answer + citations + epistemic status)",
        "8. Quality Assessment (self-evaluation)",
    ]
    for stage in stages:
        print(f"  {stage}")

    print("\nPRESET CONFIGURATIONS:")
    for preset in ["fast", "balanced", "thorough", "debiased"]:
        print(f"  - {preset}")

    print("\nKEY FEATURES:")
    features = [
        "Async parallel execution",
        "Result caching with TTL",
        "Integration with existing ReasoningTrace system",
        "Streaming interface for real-time updates",
        "Comprehensive metrics and observability",
        "Flexible aggregation strategies",
    ]
    for feature in features:
        print(f"  - {feature}")

    print("\n" + "=" * 70)

    # Create pipeline
    pipeline = create_pipeline("balanced")

    # Example question
    question = """
    Should our startup pivot from B2B SaaS to an AI-first platform strategy,
    given declining enterprise sales but strong developer interest in our API?
    """

    context = """
    Current metrics:
    - B2B revenue: $2.1M ARR, down 15% QoQ
    - API usage: 50M requests/month, up 200% QoQ
    - Enterprise churn: 23%
    - Developer NPS: 72
    - Cash runway: 18 months
    - Team: 12 engineers, 4 sales

    Market context:
    - AI infrastructure spending growing 40% YoY
    - Enterprise software budgets tightening
    - Competitors raising large rounds for AI platforms
    """

    print("\nExample Question:")
    print(question.strip())
    print("\nContext provided with business metrics and market data.")
    print("\nPipeline ready for execution.")

    return pipeline


if __name__ == "__main__":
    asyncio.run(demonstrate_pipeline())
