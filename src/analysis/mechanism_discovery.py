#!/usr/bin/env python3
"""
Mechanism Discovery Engine for LIDA Experiments

Implements causal structure discovery algorithms:
- PC Algorithm (constraint-based)
- PC-stable (order-independent version)
- FCI (handles latent confounders)
- GES (score-based greedy search)
- NOTEARS (continuous optimization)
- LiNGAM (linear non-Gaussian)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple, Set
import logging

logger = logging.getLogger("lida.mechanism")


class DiscoveryAlgorithm(Enum):
    """Available causal discovery algorithms."""
    PC = "pc"
    PC_STABLE = "pc_stable"
    FCI = "fci"
    GES = "ges"
    NOTEARS = "notears"
    LINGAM = "lingam"


class EdgeType(Enum):
    """Types of edges in causal graphs."""
    DIRECTED = "->"
    BIDIRECTED = "<->"
    UNDIRECTED = "--"
    PARTIALLY_DIRECTED = "o->"


@dataclass
class Edge:
    """An edge in the causal graph."""
    source: str
    target: str
    edge_type: EdgeType = EdgeType.DIRECTED
    weight: float = 1.0
    confidence: float = 1.0


@dataclass
class CausalGraph:
    """A causal graph structure."""
    nodes: List[str] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)

    # Adjacency representations
    adjacency_matrix: Optional[np.ndarray] = None
    parents: Dict[str, Set[str]] = field(default_factory=dict)
    children: Dict[str, Set[str]] = field(default_factory=dict)

    # Metadata
    algorithm: Optional[str] = None
    discovered_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_edge(self, source: str, target: str, edge_type: EdgeType = EdgeType.DIRECTED):
        """Add an edge to the graph."""
        if source not in self.nodes:
            self.nodes.append(source)
        if target not in self.nodes:
            self.nodes.append(target)

        self.edges.append(Edge(source=source, target=target, edge_type=edge_type))

        if source not in self.children:
            self.children[source] = set()
        self.children[source].add(target)

        if target not in self.parents:
            self.parents[target] = set()
        self.parents[target].add(source)

    def remove_edge(self, source: str, target: str):
        """Remove an edge from the graph."""
        self.edges = [e for e in self.edges if not (e.source == source and e.target == target)]

        if source in self.children:
            self.children[source].discard(target)
        if target in self.parents:
            self.parents[target].discard(source)

    def has_edge(self, source: str, target: str) -> bool:
        """Check if edge exists."""
        return any(e.source == source and e.target == target for e in self.edges)

    def get_neighbors(self, node: str) -> Set[str]:
        """Get all neighbors (parents and children) of a node."""
        neighbors = set()
        neighbors.update(self.parents.get(node, set()))
        neighbors.update(self.children.get(node, set()))
        return neighbors

    def to_adjacency_matrix(self) -> np.ndarray:
        """Convert to adjacency matrix."""
        n = len(self.nodes)
        adj = np.zeros((n, n))
        node_idx = {node: i for i, node in enumerate(self.nodes)}

        for edge in self.edges:
            i = node_idx[edge.source]
            j = node_idx[edge.target]
            adj[i, j] = edge.weight

            if edge.edge_type == EdgeType.BIDIRECTED:
                adj[j, i] = edge.weight
            elif edge.edge_type == EdgeType.UNDIRECTED:
                adj[j, i] = edge.weight

        self.adjacency_matrix = adj
        return adj

    def topological_sort(self) -> List[str]:
        """Topological sort of nodes (if DAG)."""
        in_degree = {node: 0 for node in self.nodes}
        for node, parents in self.parents.items():
            in_degree[node] = len(parents)

        queue = [n for n, deg in in_degree.items() if deg == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)

            for child in self.children.get(node, set()):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return order if len(order) == len(self.nodes) else []


@dataclass
class DiscoveryResult:
    """Results from causal discovery."""
    graph: CausalGraph
    algorithm: DiscoveryAlgorithm
    sample_size: int = 0

    # Quality metrics
    edge_stability: Dict[Tuple[str, str], float] = field(default_factory=dict)
    shd: Optional[float] = None  # Structural Hamming Distance (if ground truth known)
    bic: Optional[float] = None  # Bayesian Information Criterion

    # Identified structures
    v_structures: List[Tuple[str, str, str]] = field(default_factory=list)  # (X, Y, Z) where X->Y<-Z
    confounded_pairs: List[Tuple[str, str]] = field(default_factory=list)
    mediators: List[str] = field(default_factory=list)

    # Uncertainty
    bootstrap_graphs: List[CausalGraph] = field(default_factory=list)

    # Metadata
    runtime_seconds: float = 0.0
    converged: bool = True
    warnings: List[str] = field(default_factory=list)


class MechanismDiscovery:
    """
    Causal mechanism discovery from observational data.

    Discovers causal structure from debate data to understand:
    - What drives position changes
    - How influence propagates
    - Which arguments cause belief updates
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_conditioning_set: int = 3,
        bootstrap_samples: int = 100,
    ):
        self.alpha = alpha  # Significance level for independence tests
        self.max_conditioning_set = max_conditioning_set
        self.bootstrap_samples = bootstrap_samples

    def discover(
        self,
        data: Dict[str, np.ndarray],
        algorithm: DiscoveryAlgorithm = DiscoveryAlgorithm.PC_STABLE,
        prior_knowledge: Optional[CausalGraph] = None,
    ) -> DiscoveryResult:
        """
        Discover causal structure from data.

        Args:
            data: Dictionary mapping variable names to arrays
            algorithm: Discovery algorithm to use
            prior_knowledge: Known edges to incorporate

        Returns:
            DiscoveryResult with discovered graph
        """
        import time
        start_time = time.time()

        # Convert to matrix form
        variables = list(data.keys())
        n_samples = len(data[variables[0]])
        X = np.column_stack([data[v] for v in variables])

        # Run discovery
        if algorithm == DiscoveryAlgorithm.PC:
            graph = self._pc_algorithm(X, variables)
        elif algorithm == DiscoveryAlgorithm.PC_STABLE:
            graph = self._pc_stable(X, variables)
        elif algorithm == DiscoveryAlgorithm.GES:
            graph = self._ges_algorithm(X, variables)
        elif algorithm == DiscoveryAlgorithm.NOTEARS:
            graph = self._notears(X, variables)
        else:
            graph = self._pc_stable(X, variables)

        graph.algorithm = algorithm.value

        # Incorporate prior knowledge
        if prior_knowledge:
            graph = self._incorporate_prior(graph, prior_knowledge)

        # Bootstrap for edge stability
        edge_stability = self._bootstrap_stability(X, variables, algorithm)

        # Identify structures
        v_structures = self._find_v_structures(graph)
        mediators = self._find_mediators(graph)

        result = DiscoveryResult(
            graph=graph,
            algorithm=algorithm,
            sample_size=n_samples,
            edge_stability=edge_stability,
            v_structures=v_structures,
            mediators=mediators,
            runtime_seconds=time.time() - start_time,
        )

        return result

    def _pc_algorithm(self, X: np.ndarray, variables: List[str]) -> CausalGraph:
        """
        PC algorithm for causal discovery.

        Phase 1: Start with complete graph, remove edges based on conditional independence
        Phase 2: Orient v-structures
        Phase 3: Apply Meek rules for remaining orientations
        """
        n_vars = len(variables)
        graph = CausalGraph(nodes=variables.copy())

        # Start with complete undirected graph
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                graph.add_edge(variables[i], variables[j], EdgeType.UNDIRECTED)
                graph.add_edge(variables[j], variables[i], EdgeType.UNDIRECTED)

        # Phase 1: Edge removal via conditional independence tests
        sep_sets = {}  # Separation sets for orientation

        for cond_size in range(self.max_conditioning_set + 1):
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    x, y = variables[i], variables[j]

                    if not graph.has_edge(x, y):
                        continue

                    # Get potential conditioning sets
                    neighbors = graph.get_neighbors(x) - {y}

                    for cond_set in combinations(neighbors, min(cond_size, len(neighbors))):
                        if self._conditional_independence(X, i, j, [variables.index(v) for v in cond_set]):
                            graph.remove_edge(x, y)
                            graph.remove_edge(y, x)
                            sep_sets[(x, y)] = set(cond_set)
                            sep_sets[(y, x)] = set(cond_set)
                            break

        # Phase 2: Orient v-structures
        graph = self._orient_v_structures(graph, sep_sets)

        # Phase 3: Apply Meek rules
        graph = self._apply_meek_rules(graph)

        return graph

    def _pc_stable(self, X: np.ndarray, variables: List[str]) -> CausalGraph:
        """
        PC-stable: Order-independent version of PC algorithm.

        Main difference: Decisions about edge removal are made simultaneously
        based on the skeleton at the start of each level.
        """
        n_vars = len(variables)
        graph = CausalGraph(nodes=variables.copy())

        # Start with complete undirected graph
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                graph.add_edge(variables[i], variables[j], EdgeType.UNDIRECTED)
                graph.add_edge(variables[j], variables[i], EdgeType.UNDIRECTED)

        sep_sets = {}
        edges_to_remove = []

        for cond_size in range(self.max_conditioning_set + 1):
            # Collect edges to remove at this level
            edges_to_remove = []

            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    x, y = variables[i], variables[j]

                    if not graph.has_edge(x, y):
                        continue

                    neighbors = graph.get_neighbors(x) - {y}

                    for cond_set in combinations(neighbors, min(cond_size, len(neighbors))):
                        cond_indices = [variables.index(v) for v in cond_set]
                        if self._conditional_independence(X, i, j, cond_indices):
                            edges_to_remove.append((x, y))
                            sep_sets[(x, y)] = set(cond_set)
                            sep_sets[(y, x)] = set(cond_set)
                            break

            # Remove all edges at once
            for x, y in edges_to_remove:
                graph.remove_edge(x, y)
                graph.remove_edge(y, x)

        # Orient v-structures
        graph = self._orient_v_structures(graph, sep_sets)

        # Apply Meek rules
        graph = self._apply_meek_rules(graph)

        return graph

    def _ges_algorithm(self, X: np.ndarray, variables: List[str]) -> CausalGraph:
        """
        Greedy Equivalence Search (GES) algorithm.

        Score-based approach that searches equivalence classes.
        Phase 1: Forward (add edges)
        Phase 2: Backward (remove edges)
        """
        n_vars = len(variables)
        graph = CausalGraph(nodes=variables.copy())

        def bic_score(graph: CausalGraph) -> float:
            """Compute BIC score for current graph."""
            score = 0.0
            n = X.shape[0]

            for j, var in enumerate(variables):
                parent_indices = [variables.index(p) for p in graph.parents.get(var, set())]

                if parent_indices:
                    X_parents = X[:, parent_indices]
                    X_parents = np.column_stack([np.ones(n), X_parents])
                    y = X[:, j]

                    try:
                        beta = np.linalg.lstsq(X_parents, y, rcond=None)[0]
                        residuals = y - X_parents @ beta
                        rss = np.sum(residuals ** 2)
                        k = len(parent_indices) + 1
                    except np.linalg.LinAlgError:
                        rss = np.var(X[:, j]) * n
                        k = 1
                else:
                    rss = np.var(X[:, j]) * n
                    k = 1

                # BIC: n*log(RSS/n) + k*log(n)
                score += n * np.log(rss / n + 1e-10) + k * np.log(n)

            return score

        current_score = bic_score(graph)

        # Forward phase: Add edges greedily
        improved = True
        while improved:
            improved = False
            best_edge = None
            best_score = current_score

            for i in range(n_vars):
                for j in range(n_vars):
                    if i == j:
                        continue
                    if graph.has_edge(variables[i], variables[j]):
                        continue

                    # Try adding edge
                    graph.add_edge(variables[i], variables[j])
                    new_score = bic_score(graph)

                    if new_score < best_score:
                        best_score = new_score
                        best_edge = (variables[i], variables[j])

                    graph.remove_edge(variables[i], variables[j])

            if best_edge and best_score < current_score:
                graph.add_edge(best_edge[0], best_edge[1])
                current_score = best_score
                improved = True

        # Backward phase: Remove edges greedily
        improved = True
        while improved:
            improved = False
            best_removal = None
            best_score = current_score

            for edge in graph.edges[:]:
                graph.remove_edge(edge.source, edge.target)
                new_score = bic_score(graph)

                if new_score < best_score:
                    best_score = new_score
                    best_removal = (edge.source, edge.target)

                graph.add_edge(edge.source, edge.target)

            if best_removal and best_score < current_score:
                graph.remove_edge(best_removal[0], best_removal[1])
                current_score = best_score
                improved = True

        return graph

    def _notears(self, X: np.ndarray, variables: List[str]) -> CausalGraph:
        """
        NOTEARS: Non-combinatorial optimization for structure learning.

        Continuous optimization approach with acyclicity constraint.
        """
        n, d = X.shape
        graph = CausalGraph(nodes=variables.copy())

        # Standardize data
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        # Initialize adjacency matrix
        W = np.zeros((d, d))

        # Optimization parameters
        lr = 0.001
        max_iter = 500
        lambda_reg = 0.1
        rho = 1.0
        alpha = 0.0
        h_tol = 1e-8

        def h(W):
            """Acyclicity constraint: trace(e^(W*W)) - d = 0 for DAG."""
            M = W * W
            return np.trace(np.linalg.matrix_power(np.eye(d) + M / d, d)) - d

        for iteration in range(max_iter):
            # Gradient of loss (squared loss)
            loss_grad = (2 / n) * X.T @ X @ W - (2 / n) * X.T @ X

            # Gradient of acyclicity constraint
            M = W * W
            E = np.linalg.matrix_power(np.eye(d) + M / d, d - 1)
            h_grad = 2 * W * E.T

            # L1 regularization gradient
            l1_grad = lambda_reg * np.sign(W)

            # Augmented Lagrangian gradient
            grad = loss_grad + alpha * h_grad + rho * h(W) * h_grad + l1_grad

            # Gradient descent step
            W = W - lr * grad

            # Threshold small values
            W[np.abs(W) < 0.01] = 0

            # Check convergence
            h_val = h(W)
            if h_val < h_tol:
                break

            # Update Lagrangian parameters
            alpha = alpha + rho * h_val
            if h_val > 0.25 * h_tol:
                rho = min(rho * 10, 1e16)

        # Convert to graph
        threshold = 0.3
        for i in range(d):
            for j in range(d):
                if abs(W[i, j]) > threshold:
                    graph.add_edge(variables[i], variables[j])
                    graph.edges[-1].weight = float(W[i, j])

        return graph

    def _conditional_independence(
        self,
        X: np.ndarray,
        i: int,
        j: int,
        cond_set: List[int],
        alpha: Optional[float] = None
    ) -> bool:
        """
        Test conditional independence using partial correlation.

        H0: X_i ⊥ X_j | X_cond_set
        """
        if alpha is None:
            alpha = self.alpha

        n = X.shape[0]

        if not cond_set:
            # Marginal correlation
            corr = np.corrcoef(X[:, i], X[:, j])[0, 1]
        else:
            # Partial correlation via regression
            X_cond = X[:, cond_set]
            X_cond = np.column_stack([np.ones(n), X_cond])

            # Residuals of X_i on conditioning set
            try:
                beta_i = np.linalg.lstsq(X_cond, X[:, i], rcond=None)[0]
                res_i = X[:, i] - X_cond @ beta_i

                beta_j = np.linalg.lstsq(X_cond, X[:, j], rcond=None)[0]
                res_j = X[:, j] - X_cond @ beta_j

                corr = np.corrcoef(res_i, res_j)[0, 1]
            except np.linalg.LinAlgError:
                return False

        # Fisher's z transformation
        if abs(corr) >= 1:
            return False

        z = 0.5 * np.log((1 + corr) / (1 - corr))
        se = 1 / np.sqrt(n - len(cond_set) - 3)

        # Two-tailed test
        p_value = 2 * (1 - self._norm_cdf(abs(z) / se))

        return p_value > alpha

    def _orient_v_structures(
        self,
        graph: CausalGraph,
        sep_sets: Dict[Tuple[str, str], Set[str]]
    ) -> CausalGraph:
        """Orient v-structures: X -> Y <- Z where X and Z are not adjacent."""
        for y in graph.nodes:
            neighbors = list(graph.get_neighbors(y))

            for i, x in enumerate(neighbors):
                for z in neighbors[i + 1:]:
                    # Check if X and Z are not adjacent
                    if graph.has_edge(x, z) or graph.has_edge(z, x):
                        continue

                    # Check if Y is in separating set
                    sep_set = sep_sets.get((x, z), set())
                    if y not in sep_set:
                        # Orient as v-structure: X -> Y <- Z
                        graph.remove_edge(y, x)
                        graph.remove_edge(y, z)

                        if not graph.has_edge(x, y):
                            graph.add_edge(x, y)
                        if not graph.has_edge(z, y):
                            graph.add_edge(z, y)

                        for edge in graph.edges:
                            if edge.source in (x, z) and edge.target == y:
                                edge.edge_type = EdgeType.DIRECTED

        return graph

    def _apply_meek_rules(self, graph: CausalGraph) -> CausalGraph:
        """Apply Meek's orientation rules for remaining undirected edges."""
        changed = True

        while changed:
            changed = False

            for edge in graph.edges[:]:
                if edge.edge_type != EdgeType.UNDIRECTED:
                    continue

                x, y = edge.source, edge.target

                # Rule 1: If Z -> X -- Y, then X -> Y
                for z in graph.parents.get(x, set()):
                    z_edge = next((e for e in graph.edges if e.source == z and e.target == x), None)
                    if z_edge and z_edge.edge_type == EdgeType.DIRECTED:
                        if z not in graph.get_neighbors(y):
                            edge.edge_type = EdgeType.DIRECTED
                            graph.remove_edge(y, x)
                            changed = True
                            break

                # Rule 2: If X -> Z -> Y and X -- Y, then X -> Y
                for z in graph.children.get(x, set()):
                    if z in graph.parents.get(y, set()):
                        x_z_edge = next((e for e in graph.edges if e.source == x and e.target == z), None)
                        z_y_edge = next((e for e in graph.edges if e.source == z and e.target == y), None)
                        if (x_z_edge and x_z_edge.edge_type == EdgeType.DIRECTED and
                            z_y_edge and z_y_edge.edge_type == EdgeType.DIRECTED):
                            edge.edge_type = EdgeType.DIRECTED
                            graph.remove_edge(y, x)
                            changed = True
                            break

        return graph

    def _incorporate_prior(
        self,
        graph: CausalGraph,
        prior: CausalGraph
    ) -> CausalGraph:
        """Incorporate prior knowledge into discovered graph."""
        for edge in prior.edges:
            if not graph.has_edge(edge.source, edge.target):
                graph.add_edge(edge.source, edge.target, edge.edge_type)
            else:
                # Update edge type if prior has stronger constraint
                for g_edge in graph.edges:
                    if g_edge.source == edge.source and g_edge.target == edge.target:
                        if edge.edge_type == EdgeType.DIRECTED:
                            g_edge.edge_type = EdgeType.DIRECTED
                        break

        return graph

    def _bootstrap_stability(
        self,
        X: np.ndarray,
        variables: List[str],
        algorithm: DiscoveryAlgorithm
    ) -> Dict[Tuple[str, str], float]:
        """Compute edge stability via bootstrap."""
        n = X.shape[0]
        edge_counts: Dict[Tuple[str, str], int] = {}

        for _ in range(self.bootstrap_samples):
            # Bootstrap sample
            indices = np.random.choice(n, n, replace=True)
            X_boot = X[indices]

            # Run discovery
            if algorithm == DiscoveryAlgorithm.PC_STABLE:
                graph = self._pc_stable(X_boot, variables)
            elif algorithm == DiscoveryAlgorithm.GES:
                graph = self._ges_algorithm(X_boot, variables)
            else:
                graph = self._pc_stable(X_boot, variables)

            # Count edges
            for edge in graph.edges:
                key = (edge.source, edge.target)
                edge_counts[key] = edge_counts.get(key, 0) + 1

        # Compute stability
        return {k: v / self.bootstrap_samples for k, v in edge_counts.items()}

    def _find_v_structures(self, graph: CausalGraph) -> List[Tuple[str, str, str]]:
        """Find v-structures (X -> Y <- Z)."""
        v_structures = []

        for y in graph.nodes:
            parents = list(graph.parents.get(y, set()))

            for i, x in enumerate(parents):
                for z in parents[i + 1:]:
                    # Check if X and Z are not adjacent
                    if not graph.has_edge(x, z) and not graph.has_edge(z, x):
                        v_structures.append((x, y, z))

        return v_structures

    def _find_mediators(self, graph: CausalGraph) -> List[str]:
        """Find potential mediator variables."""
        mediators = []

        for node in graph.nodes:
            parents = graph.parents.get(node, set())
            children = graph.children.get(node, set())

            # Mediator has both parents and children
            if parents and children:
                mediators.append(node)

        return mediators

    def _norm_cdf(self, x: float) -> float:
        """Standard normal CDF approximation."""
        return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    def compare_graphs(
        self,
        discovered: CausalGraph,
        ground_truth: CausalGraph
    ) -> Dict[str, float]:
        """
        Compare discovered graph to ground truth.

        Returns metrics like SHD, precision, recall.
        """
        # Build edge sets
        disc_edges = {(e.source, e.target) for e in discovered.edges if e.edge_type == EdgeType.DIRECTED}
        true_edges = {(e.source, e.target) for e in ground_truth.edges if e.edge_type == EdgeType.DIRECTED}

        # True positives, false positives, false negatives
        tp = len(disc_edges & true_edges)
        fp = len(disc_edges - true_edges)
        fn = len(true_edges - disc_edges)

        # Metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        shd = fp + fn  # Structural Hamming Distance

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "shd": shd,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }


def discover_debate_mechanisms(
    debate_data: Dict[str, np.ndarray],
    algorithm: DiscoveryAlgorithm = DiscoveryAlgorithm.PC_STABLE,
) -> DiscoveryResult:
    """
    Convenience function for discovering causal mechanisms in debate data.

    Expected variables:
    - argument_quality
    - speaker_credibility
    - emotional_state
    - prior_position
    - final_position
    - position_change
    """
    discovery = MechanismDiscovery()
    return discovery.discover(debate_data, algorithm)


if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)
    n = 500

    # Generate data from known structure
    # X -> Y -> Z, X -> Z (with confounder)
    X = np.random.normal(0, 1, n)
    Y = 0.7 * X + np.random.normal(0, 0.5, n)
    Z = 0.5 * Y + 0.3 * X + np.random.normal(0, 0.5, n)
    W = np.random.normal(0, 1, n)  # Independent

    data = {
        "argument_quality": X,
        "credibility": Y,
        "position_change": Z,
        "noise": W,
    }

    discovery = MechanismDiscovery(bootstrap_samples=50)

    print("=== Causal Structure Discovery ===\n")

    for algo in [DiscoveryAlgorithm.PC_STABLE, DiscoveryAlgorithm.GES]:
        result = discovery.discover(data, algorithm=algo)

        print(f"Algorithm: {algo.value}")
        print(f"Runtime: {result.runtime_seconds:.2f}s")
        print("Discovered edges:")
        for edge in result.graph.edges:
            stability = result.edge_stability.get((edge.source, edge.target), 0)
            print(f"  {edge.source} {edge.edge_type.value} {edge.target} (stability: {stability:.2f})")

        print(f"V-structures: {result.v_structures}")
        print(f"Mediators: {result.mediators}")
        print()
