"""Hypergraph for storing capabilities and multi-way relationships.

A hypergraph extends traditional graphs by allowing edges (hyperedges)
to connect any number of nodes, not just two. This is ideal for
representing complex capability relationships.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Iterator, Optional, Set, Tuple

from .structures import MetaStructure, Capability


class EdgeType(str, Enum):
    """Types of hyperedges."""
    COMPOSITION = "composition"     # A + B -> C
    DERIVATION = "derivation"       # A derives B
    DEPENDENCY = "dependency"       # A depends on B
    TRANSFORMATION = "transformation"  # A transforms to B
    EMERGENCE = "emergence"         # A, B, C emerge D
    SIMILARITY = "similarity"       # A similar to B
    SUBSUMPTION = "subsumption"     # A subsumes B
    CONFLICT = "conflict"           # A conflicts with B
    ENHANCEMENT = "enhancement"     # A enhances B
    CONTEXT = "context"             # Contextual relationship


@dataclass
class HyperNode:
    """A node in the hypergraph.

    Can represent:
    - Capabilities
    - MetaStructures
    - Templates
    - Abstract concepts
    """

    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    node_type: str = "capability"  # capability, structure, template, concept
    data: Any = None  # The actual object

    # Cached properties for fast lookup
    name: str = ""
    signature: str = ""
    tags: set = field(default_factory=set)

    # Embeddings for semantic search (optional)
    embedding: Optional[list[float]] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    metadata: dict = field(default_factory=dict)

    def __hash__(self):
        return hash(self.node_id)

    def __eq__(self, other):
        if isinstance(other, HyperNode):
            return self.node_id == other.node_id
        return False

    def touch(self):
        """Update access tracking."""
        self.accessed_at = datetime.now()
        self.access_count += 1


@dataclass
class HyperEdge:
    """A hyperedge connecting multiple nodes.

    Unlike regular edges, hyperedges can connect any number of nodes,
    representing n-ary relationships.
    """

    edge_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    edge_type: EdgeType = EdgeType.COMPOSITION

    # Nodes in this hyperedge
    source_nodes: list[str] = field(default_factory=list)  # Node IDs
    target_nodes: list[str] = field(default_factory=list)  # Node IDs

    # Edge properties
    weight: float = 1.0
    confidence: float = 1.0
    bidirectional: bool = False

    # For ordered relationships
    ordered: bool = False

    # Edge data
    data: dict = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    def __hash__(self):
        return hash(self.edge_id)

    def nodes(self) -> set[str]:
        """Get all nodes in this edge."""
        return set(self.source_nodes) | set(self.target_nodes)

    def matches_pattern(
        self,
        sources: Optional[set[str]] = None,
        targets: Optional[set[str]] = None,
        edge_type: Optional[EdgeType] = None,
    ) -> bool:
        """Check if edge matches a pattern."""
        if edge_type and self.edge_type != edge_type:
            return False

        if sources and not sources.issubset(set(self.source_nodes)):
            return False

        if targets and not targets.issubset(set(self.target_nodes)):
            return False

        return True


class Hypergraph:
    """A hypergraph for storing and querying capabilities.

    Supports:
    - Multi-way relationships between capabilities
    - Efficient traversal and querying
    - Caching of computed structures
    - Semantic similarity search (with embeddings)
    """

    def __init__(self):
        self._nodes: dict[str, HyperNode] = {}
        self._edges: dict[str, HyperEdge] = {}

        # Indexes for fast lookup
        self._node_to_edges: dict[str, set[str]] = defaultdict(set)
        self._type_index: dict[str, set[str]] = defaultdict(set)
        self._tag_index: dict[str, set[str]] = defaultdict(set)
        self._name_index: dict[str, str] = {}  # name -> node_id

        # Cache for computed structures
        self._cache: dict[str, Any] = {}
        self._cache_timestamps: dict[str, datetime] = {}

        # Statistics
        self._stats = {
            "nodes_added": 0,
            "edges_added": 0,
            "queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Node Operations
    # ─────────────────────────────────────────────────────────────────────────

    def add_node(self, node: HyperNode) -> str:
        """Add a node to the hypergraph."""
        self._nodes[node.node_id] = node

        # Update indexes
        self._type_index[node.node_type].add(node.node_id)
        for tag in node.tags:
            self._tag_index[tag].add(node.node_id)
        if node.name:
            self._name_index[node.name] = node.node_id

        self._stats["nodes_added"] += 1
        return node.node_id

    def add_capability(self, capability: Capability) -> str:
        """Add a capability as a node."""
        node = HyperNode(
            node_id=capability.capability_id,
            node_type="capability",
            data=capability,
            name=capability.name,
            signature=capability.signature(),
            tags=set(capability.inputs + capability.outputs),
        )
        return self.add_node(node)

    def add_structure(self, structure: MetaStructure) -> str:
        """Add a meta-structure as a node."""
        node = HyperNode(
            node_id=structure.structure_id,
            node_type="structure",
            data=structure,
            name=structure.name,
            signature=structure.signature(),
            tags={structure.structure_type.value},
        )
        node_id = self.add_node(node)

        # Also add capabilities within the structure
        for cap in structure.capabilities:
            cap_node_id = self.add_capability(cap)
            # Create contains edge
            self.add_edge(HyperEdge(
                edge_type=EdgeType.SUBSUMPTION,
                source_nodes=[node_id],
                target_nodes=[cap_node_id],
            ))

        return node_id

    def get_node(self, node_id: str) -> Optional[HyperNode]:
        """Get a node by ID."""
        node = self._nodes.get(node_id)
        if node:
            node.touch()
        return node

    def get_by_name(self, name: str) -> Optional[HyperNode]:
        """Get a node by name."""
        node_id = self._name_index.get(name)
        if node_id:
            return self.get_node(node_id)
        return None

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and its edges."""
        if node_id not in self._nodes:
            return False

        node = self._nodes.pop(node_id)

        # Remove from indexes
        self._type_index[node.node_type].discard(node_id)
        for tag in node.tags:
            self._tag_index[tag].discard(node_id)
        if node.name in self._name_index:
            del self._name_index[node.name]

        # Remove edges involving this node
        for edge_id in list(self._node_to_edges[node_id]):
            self.remove_edge(edge_id)

        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Edge Operations
    # ─────────────────────────────────────────────────────────────────────────

    def add_edge(self, edge: HyperEdge) -> str:
        """Add a hyperedge."""
        self._edges[edge.edge_id] = edge

        # Update node-to-edge index
        for node_id in edge.nodes():
            self._node_to_edges[node_id].add(edge.edge_id)

        self._stats["edges_added"] += 1
        return edge.edge_id

    def create_edge(
        self,
        sources: list[str],
        targets: list[str],
        edge_type: EdgeType,
        **kwargs,
    ) -> str:
        """Create and add a hyperedge."""
        edge = HyperEdge(
            edge_type=edge_type,
            source_nodes=sources,
            target_nodes=targets,
            **kwargs,
        )
        return self.add_edge(edge)

    def get_edge(self, edge_id: str) -> Optional[HyperEdge]:
        """Get an edge by ID."""
        return self._edges.get(edge_id)

    def remove_edge(self, edge_id: str) -> bool:
        """Remove an edge."""
        if edge_id not in self._edges:
            return False

        edge = self._edges.pop(edge_id)

        # Update node-to-edge index
        for node_id in edge.nodes():
            self._node_to_edges[node_id].discard(edge_id)

        return True

    def get_edges_for_node(self, node_id: str) -> list[HyperEdge]:
        """Get all edges involving a node."""
        edge_ids = self._node_to_edges.get(node_id, set())
        return [self._edges[eid] for eid in edge_ids if eid in self._edges]

    # ─────────────────────────────────────────────────────────────────────────
    # Querying
    # ─────────────────────────────────────────────────────────────────────────

    def query_nodes(
        self,
        node_type: Optional[str] = None,
        tags: Optional[set[str]] = None,
        predicate: Optional[Callable[[HyperNode], bool]] = None,
    ) -> list[HyperNode]:
        """Query nodes by type, tags, or custom predicate."""
        self._stats["queries"] += 1

        candidates = set(self._nodes.keys())

        if node_type:
            candidates &= self._type_index.get(node_type, set())

        if tags:
            for tag in tags:
                candidates &= self._tag_index.get(tag, set())

        nodes = [self._nodes[nid] for nid in candidates]

        if predicate:
            nodes = [n for n in nodes if predicate(n)]

        return nodes

    def query_edges(
        self,
        edge_type: Optional[EdgeType] = None,
        sources: Optional[set[str]] = None,
        targets: Optional[set[str]] = None,
    ) -> list[HyperEdge]:
        """Query edges by type, sources, or targets."""
        self._stats["queries"] += 1

        edges = list(self._edges.values())

        if edge_type or sources or targets:
            edges = [
                e for e in edges
                if e.matches_pattern(sources, targets, edge_type)
            ]

        return edges

    def find_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5,
    ) -> Optional[list[str]]:
        """Find a path between two nodes through edges."""
        if start_id not in self._nodes or end_id not in self._nodes:
            return None

        visited = set()
        queue = [(start_id, [start_id])]

        while queue:
            current, path = queue.pop(0)

            if current == end_id:
                return path

            if len(path) > max_depth:
                continue

            if current in visited:
                continue

            visited.add(current)

            for edge in self.get_edges_for_node(current):
                for next_node in edge.nodes():
                    if next_node not in visited:
                        queue.append((next_node, path + [next_node]))

        return None

    def find_related(
        self,
        node_id: str,
        edge_types: Optional[set[EdgeType]] = None,
        depth: int = 1,
    ) -> set[str]:
        """Find all nodes related to a given node."""
        if node_id not in self._nodes:
            return set()

        related = set()
        frontier = {node_id}

        for _ in range(depth):
            next_frontier = set()
            for nid in frontier:
                for edge in self.get_edges_for_node(nid):
                    if edge_types and edge.edge_type not in edge_types:
                        continue
                    next_frontier |= edge.nodes()

            related |= next_frontier
            frontier = next_frontier - related

        related.discard(node_id)
        return related

    # ─────────────────────────────────────────────────────────────────────────
    # Caching
    # ─────────────────────────────────────────────────────────────────────────

    def cache_put(self, key: str, value: Any, ttl_seconds: int = 3600):
        """Put a value in the cache."""
        self._cache[key] = value
        self._cache_timestamps[key] = datetime.now()

    def cache_get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        if key in self._cache:
            self._stats["cache_hits"] += 1
            return self._cache[key]
        self._stats["cache_misses"] += 1
        return None

    def cache_invalidate(self, key: str):
        """Invalidate a cache entry."""
        self._cache.pop(key, None)
        self._cache_timestamps.pop(key, None)

    def cache_capability_composition(
        self,
        source_ids: list[str],
        result: MetaStructure,
    ):
        """Cache the result of composing capabilities."""
        key = f"compose:{':'.join(sorted(source_ids))}"
        self.cache_put(key, result)

    def get_cached_composition(
        self,
        source_ids: list[str],
    ) -> Optional[MetaStructure]:
        """Get a cached composition result."""
        key = f"compose:{':'.join(sorted(source_ids))}"
        return self.cache_get(key)

    # ─────────────────────────────────────────────────────────────────────────
    # Analysis
    # ─────────────────────────────────────────────────────────────────────────

    def get_composition_candidates(
        self,
        node_id: str,
    ) -> list[Tuple[str, float]]:
        """Find nodes that can compose with the given node."""
        node = self.get_node(node_id)
        if not node or not isinstance(node.data, (Capability, MetaStructure)):
            return []

        candidates = []

        for other in self._nodes.values():
            if other.node_id == node_id:
                continue

            if isinstance(other.data, (Capability, MetaStructure)):
                # Check output-input compatibility
                if isinstance(node.data, MetaStructure):
                    outputs = set()
                    for cap in node.data.capabilities:
                        outputs.update(cap.outputs)
                else:
                    outputs = set(node.data.outputs)

                if isinstance(other.data, MetaStructure):
                    inputs = set()
                    for cap in other.data.capabilities:
                        inputs.update(cap.inputs)
                else:
                    inputs = set(other.data.inputs)

                overlap = outputs & inputs
                if overlap:
                    score = len(overlap) / max(len(outputs), len(inputs))
                    candidates.append((other.node_id, score))

        return sorted(candidates, key=lambda x: -x[1])

    def get_stats(self) -> dict:
        """Get hypergraph statistics."""
        return {
            **self._stats,
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "cache_size": len(self._cache),
            "nodes_by_type": {
                t: len(ids) for t, ids in self._type_index.items()
            },
        }

    def to_dict(self) -> dict:
        """Export hypergraph to dictionary."""
        return {
            "nodes": [
                {
                    "id": n.node_id,
                    "type": n.node_type,
                    "name": n.name,
                    "tags": list(n.tags),
                }
                for n in self._nodes.values()
            ],
            "edges": [
                {
                    "id": e.edge_id,
                    "type": e.edge_type.value,
                    "sources": e.source_nodes,
                    "targets": e.target_nodes,
                    "weight": e.weight,
                }
                for e in self._edges.values()
            ],
        }


# Global hypergraph instance
_capability_graph: Optional[Hypergraph] = None


def get_capability_graph() -> Hypergraph:
    """Get or create the global capability hypergraph."""
    global _capability_graph
    if _capability_graph is None:
        _capability_graph = Hypergraph()
    return _capability_graph
