"""Advanced reasoning patterns beyond ReAct.

Includes:
- Analogical reasoning (structure mapping, case-based reasoning)
- Probabilistic reasoning (Bayesian inference, uncertainty propagation)
- Causal reasoning (causal graphs, counterfactual analysis, intervention)
- Constraint satisfaction (CSP solving, optimization)
- Abductive reasoning (inference to best explanation)
- Dialectical reasoning (thesis-antithesis-synthesis)
"""
from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from enum import Enum
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# ANALOGICAL REASONING
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StructuralMapping:
    """Represents a mapping between two domains (source → target)."""

    source_domain: str
    target_domain: str

    # Entity mappings: source_entity -> target_entity
    entity_map: Dict[str, str] = field(default_factory=dict)

    # Relation mappings: (source_rel, source_entities) -> (target_rel, target_entities)
    relation_map: Dict[Tuple, Tuple] = field(default_factory=dict)

    # Systematicity score (higher-order relations preferred)
    systematicity: float = 0.0

    # Structural similarity score
    similarity: float = 0.0

    def __post_init__(self):
        """Compute similarity and systematicity."""
        self.similarity = len(self.entity_map) * 0.3 + len(self.relation_map) * 0.7
        self.systematicity = sum(1 for rel in self.relation_map.keys() if self._is_higher_order(rel))

    def _is_higher_order(self, relation: Tuple) -> bool:
        """Check if relation is higher-order (relations between relations)."""
        # Simple heuristic: check if relation name contains meta-indicators
        rel_name = str(relation[0]).lower()
        return any(keyword in rel_name for keyword in ["cause", "enable", "prevent", "implies"])


class AnalogicalReasoner:
    """Structure-mapping engine for analogical reasoning (SME-inspired)."""

    def __init__(self):
        self.case_base: List[Dict[str, Any]] = []  # Library of cases for CBR

    def find_analogy(
        self,
        source: Dict[str, Any],
        target: Dict[str, Any],
        max_candidates: int = 5,
    ) -> List[StructuralMapping]:
        """Find analogical mappings from source to target.

        Args:
            source: Source domain representation with entities and relations
            target: Target domain representation
            max_candidates: Maximum candidate mappings to return

        Returns:
            List of structural mappings sorted by quality
        """
        # Extract entities and relations
        source_entities = source.get("entities", [])
        target_entities = target.get("entities", [])
        source_relations = source.get("relations", [])
        target_relations = target.get("relations", [])

        # Generate candidate mappings
        candidates = []

        # Try all entity mapping combinations (simplified - in practice use constraint satisfaction)
        from itertools import combinations, permutations

        for target_subset in combinations(target_entities, min(len(source_entities), len(target_entities))):
            for perm in permutations(target_subset):
                if len(perm) != len(source_entities):
                    continue

                entity_map = {s: t for s, t in zip(source_entities, perm)}

                # Try to map relations
                relation_map = {}
                for src_rel in source_relations:
                    src_rel_name = src_rel.get("name")
                    src_rel_entities = src_rel.get("entities", [])

                    # Find matching target relation
                    for tgt_rel in target_relations:
                        tgt_rel_name = tgt_rel.get("name")
                        tgt_rel_entities = tgt_rel.get("entities", [])

                        # Check if relation names match (or are similar)
                        if self._relations_compatible(src_rel_name, tgt_rel_name):
                            # Check if entity mapping is consistent
                            mapped_entities = [entity_map.get(e) for e in src_rel_entities]
                            if all(e in tgt_rel_entities for e in mapped_entities if e):
                                relation_map[(src_rel_name, tuple(src_rel_entities))] = \
                                    (tgt_rel_name, tuple(tgt_rel_entities))

                # Create mapping
                mapping = StructuralMapping(
                    source_domain=source.get("name", "source"),
                    target_domain=target.get("name", "target"),
                    entity_map=entity_map,
                    relation_map=relation_map,
                )

                candidates.append(mapping)

        # Sort by combined score
        candidates.sort(
            key=lambda m: m.systematicity * 0.6 + m.similarity * 0.4,
            reverse=True
        )

        return candidates[:max_candidates]

    def _relations_compatible(self, rel1: str, rel2: str) -> bool:
        """Check if two relations are compatible for mapping."""
        # Simple string matching - in practice use semantic similarity
        return rel1.lower() == rel2.lower() or \
               rel1.lower() in rel2.lower() or \
               rel2.lower() in rel1.lower()

    def transfer_inference(
        self,
        mapping: StructuralMapping,
        source_knowledge: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Transfer knowledge from source to target via mapping."""
        target_knowledge = {}

        # Transfer facts
        for fact, value in source_knowledge.items():
            # Map entities in fact to target
            mapped_fact = fact
            for src, tgt in mapping.entity_map.items():
                mapped_fact = mapped_fact.replace(src, tgt)
            target_knowledge[mapped_fact] = value

        return target_knowledge

    # Case-Based Reasoning (CBR)
    def add_case(self, case: Dict[str, Any]):
        """Add a case to the case base."""
        self.case_base.append(case)

    def retrieve_cases(
        self,
        query: Dict[str, Any],
        k: int = 5,
        similarity_fn: Optional[Callable] = None,
    ) -> List[Tuple[Dict, float]]:
        """Retrieve similar cases."""
        if similarity_fn is None:
            similarity_fn = self._default_case_similarity

        scored_cases = [
            (case, similarity_fn(query, case))
            for case in self.case_base
        ]

        scored_cases.sort(key=lambda x: x[1], reverse=True)
        return scored_cases[:k]

    def _default_case_similarity(self, case1: Dict, case2: Dict) -> float:
        """Default case similarity (Jaccard similarity on keys)."""
        keys1 = set(case1.keys())
        keys2 = set(case2.keys())

        if not keys1 or not keys2:
            return 0.0

        intersection = keys1.intersection(keys2)
        union = keys1.union(keys2)

        return len(intersection) / len(union)


# ═══════════════════════════════════════════════════════════════════════════
# PROBABILISTIC REASONING
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BayesianVariable:
    """A variable in a Bayesian network."""
    name: str
    domain: List[Any]  # Possible values
    prior: Optional[np.ndarray] = None  # Prior probability distribution

    def __post_init__(self):
        if self.prior is None:
            # Uniform prior
            self.prior = np.ones(len(self.domain)) / len(self.domain)


@dataclass
class ConditionalProbTable:
    """Conditional probability table P(child|parents)."""
    child: str
    parents: List[str]
    table: np.ndarray  # Multi-dimensional array indexed by parent values


class BayesianNetwork:
    """Bayesian network for probabilistic reasoning."""

    def __init__(self):
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX required for Bayesian networks")

        self.variables: Dict[str, BayesianVariable] = {}
        self.cpts: Dict[str, ConditionalProbTable] = {}
        self.graph = nx.DiGraph()

    def add_variable(self, var: BayesianVariable):
        """Add a variable to the network."""
        self.variables[var.name] = var
        self.graph.add_node(var.name)

    def add_edge(self, parent: str, child: str, cpt: Optional[ConditionalProbTable] = None):
        """Add directed edge from parent to child."""
        self.graph.add_edge(parent, child)
        if cpt:
            self.cpts[child] = cpt

    def infer(
        self,
        query_var: str,
        evidence: Dict[str, Any],
        method: str = "enumeration",
    ) -> np.ndarray:
        """Compute posterior probability P(query_var | evidence).

        Args:
            query_var: Variable to compute posterior for
            evidence: Observed variable values
            method: Inference method (enumeration, variable_elimination, gibbs)

        Returns:
            Posterior probability distribution
        """
        if method == "enumeration":
            return self._enumerate_ask(query_var, evidence)
        else:
            raise NotImplementedError(f"Inference method {method} not implemented")

    def _enumerate_ask(self, query_var: str, evidence: Dict[str, Any]) -> np.ndarray:
        """Exact inference via enumeration (inefficient but simple)."""
        query_domain = self.variables[query_var].domain
        distribution = []

        for value in query_domain:
            # Extend evidence with query variable
            extended_evidence = {**evidence, query_var: value}
            # Enumerate all hidden variables
            prob = self._enumerate_all(list(self.variables.keys()), extended_evidence)
            distribution.append(prob)

        # Normalize
        distribution = np.array(distribution)
        total = distribution.sum()
        if total > 0:
            distribution /= total

        return distribution

    def _enumerate_all(self, vars: List[str], evidence: Dict[str, Any]) -> float:
        """Recursive enumeration of all variables."""
        if not vars:
            return 1.0

        var = vars[0]

        if var in evidence:
            # Variable is observed
            prob = self._get_probability(var, evidence)
            return prob * self._enumerate_all(vars[1:], evidence)
        else:
            # Sum over possible values
            total = 0.0
            for value in self.variables[var].domain:
                extended_evidence = {**evidence, var: value}
                prob = self._get_probability(var, extended_evidence)
                total += prob * self._enumerate_all(vars[1:], extended_evidence)
            return total

    def _get_probability(self, var: str, evidence: Dict[str, Any]) -> float:
        """Get P(var|parents) given evidence."""
        if var in self.cpts:
            cpt = self.cpts[var]
            # Index into CPT based on parent values
            indices = tuple(
                self.variables[p].domain.index(evidence.get(p))
                for p in cpt.parents
            )
            var_idx = self.variables[var].domain.index(evidence[var])
            return cpt.table[indices + (var_idx,)]
        else:
            # No parents - use prior
            var_idx = self.variables[var].domain.index(evidence[var])
            return self.variables[var].prior[var_idx]


class ProbabilisticReasoner:
    """High-level probabilistic reasoning interface."""

    def __init__(self):
        self.networks: Dict[str, BayesianNetwork] = {}

    def create_network(self, name: str) -> BayesianNetwork:
        """Create a new Bayesian network."""
        network = BayesianNetwork()
        self.networks[name] = network
        return network

    def update_belief(
        self,
        network_name: str,
        query_var: str,
        evidence: Dict[str, Any],
    ) -> Dict[Any, float]:
        """Update belief about variable given evidence."""
        network = self.networks.get(network_name)
        if not network:
            raise ValueError(f"Network {network_name} not found")

        posterior = network.infer(query_var, evidence)
        var_domain = network.variables[query_var].domain

        return {value: prob for value, prob in zip(var_domain, posterior)}

    def uncertainty_propagation(
        self,
        values: List[float],
        uncertainties: List[float],
        operation: str = "sum",
    ) -> Tuple[float, float]:
        """Propagate uncertainty through operation.

        Uses simple error propagation formulas.
        """
        if operation == "sum":
            result = sum(values)
            result_uncertainty = np.sqrt(sum(u**2 for u in uncertainties))
        elif operation == "product":
            result = np.prod(values)
            # Relative uncertainties add in quadrature
            rel_uncertainties = [u/v if v != 0 else 0 for u, v in zip(uncertainties, values)]
            result_uncertainty = result * np.sqrt(sum(r**2 for r in rel_uncertainties))
        else:
            raise ValueError(f"Unknown operation: {operation}")

        return result, result_uncertainty


# ═══════════════════════════════════════════════════════════════════════════
# CAUSAL REASONING
# ═══════════════════════════════════════════════════════════════════════════

class CausalEdgeType(Enum):
    """Types of causal edges."""
    CAUSES = "causes"  # Direct causation
    PREVENTS = "prevents"  # Preventive causation
    ENABLES = "enables"  # Enabling condition
    CONFOUNDS = "confounds"  # Confounding variable


@dataclass
class Intervention:
    """Represents an intervention on a causal graph."""
    variable: str
    value: Any


@dataclass
class Counterfactual:
    """Represents a counterfactual question: 'What if X had been Y?'"""
    intervention: Intervention
    query_variable: str
    actual_world: Dict[str, Any]  # Actual values of variables


class CausalGraph:
    """Causal graph for causal reasoning and counterfactual analysis."""

    def __init__(self):
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX required for causal graphs")

        self.graph = nx.DiGraph()
        self.structural_equations: Dict[str, Callable] = {}  # Variable -> function of parents

    def add_variable(self, name: str, structural_eq: Optional[Callable] = None):
        """Add a variable to the causal graph."""
        self.graph.add_node(name)
        if structural_eq:
            self.structural_equations[name] = structural_eq

    def add_causal_edge(
        self,
        cause: str,
        effect: str,
        edge_type: CausalEdgeType = CausalEdgeType.CAUSES,
        strength: float = 1.0,
    ):
        """Add a causal edge."""
        self.graph.add_edge(cause, effect, type=edge_type, strength=strength)

    def intervene(self, intervention: Intervention) -> "CausalGraph":
        """Perform do-calculus intervention: create graph with intervention.

        This implements Pearl's do-operator by:
        1. Removing all edges into the intervened variable
        2. Setting the variable to the intervention value
        """
        # Create modified graph
        intervened_graph = CausalGraph()
        intervened_graph.graph = self.graph.copy()
        intervened_graph.structural_equations = self.structural_equations.copy()

        # Remove incoming edges to intervened variable
        intervened_graph.graph.remove_edges_from(
            list(intervened_graph.graph.in_edges(intervention.variable))
        )

        # Set structural equation to constant
        intervened_graph.structural_equations[intervention.variable] = \
            lambda *args: intervention.value

        return intervened_graph

    def counterfactual_inference(self, counterfactual: Counterfactual) -> Any:
        """Answer counterfactual question using three-step procedure:

        1. Abduction: Infer exogenous variables from actual world
        2. Action: Apply intervention
        3. Prediction: Compute counterfactual outcome
        """
        # Step 1: Abduction - infer exogenous variables (background factors)
        # In full implementation, would solve for unobserved variables
        # For now, assume actual_world contains all needed info

        # Step 2: Action - apply intervention
        intervened_graph = self.intervene(counterfactual.intervention)

        # Step 3: Prediction - compute outcome in intervened world
        # Topological sort to compute variables in causal order
        topo_order = list(nx.topological_sort(intervened_graph.graph))

        counterfactual_world = counterfactual.actual_world.copy()
        counterfactual_world[counterfactual.intervention.variable] = \
            counterfactual.intervention.value

        for var in topo_order:
            if var == counterfactual.intervention.variable:
                continue

            # Compute value from structural equation
            if var in intervened_graph.structural_equations:
                parents = list(intervened_graph.graph.predecessors(var))
                parent_values = [counterfactual_world.get(p) for p in parents]
                counterfactual_world[var] = \
                    intervened_graph.structural_equations[var](*parent_values)

        return counterfactual_world.get(counterfactual.query_variable)

    def find_confounders(self, treatment: str, outcome: str) -> List[str]:
        """Find confounding variables (common causes of treatment and outcome)."""
        # Find all ancestors of both variables
        treatment_ancestors = nx.ancestors(self.graph, treatment)
        outcome_ancestors = nx.ancestors(self.graph, outcome)

        # Confounders are common ancestors
        confounders = treatment_ancestors.intersection(outcome_ancestors)

        return list(confounders)

    def causal_effect(self, treatment: str, outcome: str) -> float:
        """Estimate causal effect of treatment on outcome.

        Uses simplified approach - in practice would need data and estimation.
        """
        # Find direct path strength
        if self.graph.has_edge(treatment, outcome):
            return self.graph[treatment][outcome].get("strength", 0.0)

        # Find indirect paths
        total_effect = 0.0
        for path in nx.all_simple_paths(self.graph, treatment, outcome):
            # Multiply strengths along path
            path_strength = 1.0
            for i in range(len(path) - 1):
                edge_strength = self.graph[path[i]][path[i+1]].get("strength", 0.0)
                path_strength *= edge_strength
            total_effect += path_strength

        return total_effect


class CausalReasoner:
    """High-level interface for causal reasoning."""

    def __init__(self):
        self.graphs: Dict[str, CausalGraph] = {}

    def create_graph(self, name: str) -> CausalGraph:
        """Create a new causal graph."""
        graph = CausalGraph()
        self.graphs[name] = graph
        return graph

    def estimate_effect(
        self,
        graph_name: str,
        treatment: str,
        outcome: str,
        confounders: Optional[List[str]] = None,
    ) -> float:
        """Estimate causal effect with confounder adjustment."""
        graph = self.graphs.get(graph_name)
        if not graph:
            raise ValueError(f"Graph {graph_name} not found")

        if confounders is None:
            confounders = graph.find_confounders(treatment, outcome)

        # In practice, would use backdoor adjustment or other identification strategy
        # For now, return direct causal effect
        return graph.causal_effect(treatment, outcome)

    def what_if(
        self,
        graph_name: str,
        intervention_var: str,
        intervention_value: Any,
        query_var: str,
        actual_values: Dict[str, Any],
    ) -> Any:
        """Answer 'what if' counterfactual question."""
        graph = self.graphs.get(graph_name)
        if not graph:
            raise ValueError(f"Graph {graph_name} not found")

        counterfactual = Counterfactual(
            intervention=Intervention(intervention_var, intervention_value),
            query_variable=query_var,
            actual_world=actual_values,
        )

        return graph.counterfactual_inference(counterfactual)


# ═══════════════════════════════════════════════════════════════════════════
# ABDUCTIVE REASONING
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Explanation:
    """A candidate explanation for observations."""
    hypothesis: str
    explains: List[str]  # What observations this explains
    probability: float  # Prior probability
    parsimony: float  # Simplicity score (Occam's razor)
    coherence: float  # Internal consistency

    def score(self) -> float:
        """Overall explanation quality."""
        return 0.4 * self.probability + 0.3 * self.parsimony + 0.3 * self.coherence


class AbductiveReasoner:
    """Inference to the best explanation."""

    def __init__(self):
        self.hypotheses: List[Explanation] = []

    def add_hypothesis(self, explanation: Explanation):
        """Add a candidate explanation."""
        self.hypotheses.append(explanation)

    def find_best_explanation(
        self,
        observations: List[str],
        min_coverage: float = 0.7,
    ) -> Optional[Explanation]:
        """Find the best explanation for observations.

        Args:
            observations: List of observed facts
            min_coverage: Minimum fraction of observations to explain

        Returns:
            Best explanation that covers enough observations
        """
        valid_explanations = []

        for explanation in self.hypotheses:
            # Check coverage
            explained = set(explanation.explains)
            obs_set = set(observations)
            coverage = len(explained.intersection(obs_set)) / len(obs_set)

            if coverage >= min_coverage:
                valid_explanations.append((explanation, coverage))

        if not valid_explanations:
            return None

        # Sort by score
        valid_explanations.sort(
            key=lambda x: x[0].score() * x[1],  # Score weighted by coverage
            reverse=True
        )

        return valid_explanations[0][0]


# ═══════════════════════════════════════════════════════════════════════════
# CONSTRAINT SATISFACTION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Variable:
    """A CSP variable."""
    name: str
    domain: List[Any]


@dataclass
class Constraint:
    """A CSP constraint."""
    variables: List[str]
    predicate: Callable[..., bool]  # Function that returns True if constraint satisfied


class CSPSolver:
    """Constraint Satisfaction Problem solver using backtracking."""

    def __init__(self):
        self.variables: Dict[str, Variable] = {}
        self.constraints: List[Constraint] = []

    def add_variable(self, var: Variable):
        """Add a variable."""
        self.variables[var.name] = var

    def add_constraint(self, constraint: Constraint):
        """Add a constraint."""
        self.constraints.append(constraint)

    def solve(self) -> Optional[Dict[str, Any]]:
        """Solve CSP using backtracking."""
        return self._backtrack({})

    def _backtrack(self, assignment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Recursive backtracking search."""
        # Check if assignment is complete
        if len(assignment) == len(self.variables):
            return assignment

        # Select unassigned variable
        var_name = self._select_unassigned_variable(assignment)

        # Try values in order
        for value in self._order_domain_values(var_name, assignment):
            # Check consistency
            assignment[var_name] = value

            if self._is_consistent(var_name, assignment):
                result = self._backtrack(assignment)
                if result is not None:
                    return result

            # Backtrack
            del assignment[var_name]

        return None

    def _select_unassigned_variable(self, assignment: Dict[str, Any]) -> str:
        """Select next variable to assign (MRV heuristic)."""
        unassigned = [v for v in self.variables if v not in assignment]
        # Could implement minimum-remaining-values heuristic here
        return unassigned[0] if unassigned else None

    def _order_domain_values(self, var_name: str, assignment: Dict[str, Any]) -> List[Any]:
        """Order domain values (could implement least-constraining-value heuristic)."""
        return self.variables[var_name].domain

    def _is_consistent(self, var_name: str, assignment: Dict[str, Any]) -> bool:
        """Check if current assignment is consistent with constraints."""
        for constraint in self.constraints:
            # Check if all constraint variables are assigned
            if not all(v in assignment for v in constraint.variables):
                continue

            # Evaluate constraint
            values = [assignment[v] for v in constraint.variables]
            if not constraint.predicate(*values):
                return False

        return True


# ═══════════════════════════════════════════════════════════════════════════
# DIALECTICAL REASONING
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DialecticalTriad:
    """Thesis-Antithesis-Synthesis triad."""
    thesis: str
    antithesis: str
    synthesis: str

    # Evidence/arguments for each
    thesis_support: List[str] = field(default_factory=list)
    antithesis_support: List[str] = field(default_factory=list)
    synthesis_rationale: str = ""


class DialecticalReasoner:
    """Hegelian dialectical reasoning for resolving contradictions."""

    def __init__(self):
        self.triads: List[DialecticalTriad] = []

    async def synthesize(
        self,
        thesis: str,
        antithesis: str,
        llm_function: Optional[Callable] = None,
    ) -> DialecticalTriad:
        """Create synthesis from thesis and antithesis.

        Args:
            thesis: Initial proposition
            antithesis: Contradicting proposition
            llm_function: Optional LLM function to generate synthesis

        Returns:
            Dialectical triad with synthesis
        """
        if llm_function:
            # Use LLM to generate synthesis
            prompt = f"""Given these contradicting positions:

Thesis: {thesis}
Antithesis: {antithesis}

Generate a synthesis that resolves the contradiction by finding a higher-order perspective that incorporates insights from both."""

            synthesis = await llm_function(prompt)
        else:
            synthesis = f"Synthesis of '{thesis}' and '{antithesis}' (requires LLM)"

        triad = DialecticalTriad(
            thesis=thesis,
            antithesis=antithesis,
            synthesis=synthesis,
        )

        self.triads.append(triad)
        return triad


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED ADVANCED REASONING INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

class AdvancedReasoningEngine:
    """Unified interface for all advanced reasoning capabilities."""

    def __init__(self):
        self.analogical = AnalogicalReasoner()
        self.probabilistic = ProbabilisticReasoner()
        self.causal = CausalReasoner()
        self.abductive = AbductiveReasoner()
        self.csp = CSPSolver()
        self.dialectical = DialecticalReasoner()

    # Facade methods for easy access

    def find_analogy(self, source: Dict, target: Dict) -> List[StructuralMapping]:
        """Find analogical mappings."""
        return self.analogical.find_analogy(source, target)

    def update_belief(self, network: str, query: str, evidence: Dict) -> Dict:
        """Bayesian belief update."""
        return self.probabilistic.update_belief(network, query, evidence)

    def estimate_causal_effect(self, graph: str, treatment: str, outcome: str) -> float:
        """Estimate causal effect."""
        return self.causal.estimate_effect(graph, treatment, outcome)

    def counterfactual(self, graph: str, what_if: Dict, query: str, actual: Dict) -> Any:
        """Answer counterfactual question."""
        return self.causal.what_if(graph, what_if["var"], what_if["value"], query, actual)

    def best_explanation(self, observations: List[str]) -> Optional[Explanation]:
        """Find best explanation."""
        return self.abductive.find_best_explanation(observations)

    def solve_constraints(self) -> Optional[Dict[str, Any]]:
        """Solve CSP."""
        return self.csp.solve()

    async def synthesize_positions(self, thesis: str, antithesis: str, llm=None) -> DialecticalTriad:
        """Dialectical synthesis."""
        return await self.dialectical.synthesize(thesis, antithesis, llm)


# Global instance
_reasoning_engine: Optional[AdvancedReasoningEngine] = None


def get_reasoning_engine() -> AdvancedReasoningEngine:
    """Get global reasoning engine."""
    global _reasoning_engine
    if _reasoning_engine is None:
        _reasoning_engine = AdvancedReasoningEngine()
    return _reasoning_engine
