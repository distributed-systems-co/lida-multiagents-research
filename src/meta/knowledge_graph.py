"""Knowledge graph and ontology system for structured knowledge representation.

Provides:
- RDF-style triple store (subject-predicate-object)
- Ontology definition and reasoning (RDFS/OWL-inspired)
- SPARQL-like query language
- Knowledge graph embeddings
- Reasoning via rules and inference
- Entity linking and resolution
- Temporal knowledge graphs
- Multi-hop reasoning
"""
from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from enum import Enum
from datetime import datetime
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


class EntityType(Enum):
    """Types of entities in knowledge graph."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    EVENT = "event"
    CONCEPT = "concept"
    OBJECT = "object"
    TIME = "time"


class RelationType(Enum):
    """Types of relations."""
    IS_A = "is_a"  # Type/class relationship
    PART_OF = "part_of"
    HAS_PROPERTY = "has_property"
    LOCATED_IN = "located_in"
    OCCURRED_AT = "occurred_at"
    CAUSED_BY = "caused_by"
    RELATED_TO = "related_to"
    INSTANCE_OF = "instance_of"


@dataclass
class Entity:
    """An entity in the knowledge graph."""

    id: str
    label: str
    entity_type: EntityType
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    aliases: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "type": self.entity_type.value,
            "properties": self.properties,
            "aliases": self.aliases,
            "metadata": self.metadata,
        }


@dataclass
class Relation:
    """A relation (triple) in the knowledge graph."""

    subject: str  # Entity ID
    predicate: str  # Relation type
    object: str  # Entity ID or literal value
    confidence: float = 1.0
    source: Optional[str] = None  # Source of this knowledge
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Temporal validity
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    def is_valid_at(self, time: datetime) -> bool:
        """Check if relation is valid at given time."""
        if self.valid_from and time < self.valid_from:
            return False
        if self.valid_to and time > self.valid_to:
            return False
        return True


@dataclass
class OntologyClass:
    """A class in the ontology."""

    name: str
    parent_classes: List[str] = field(default_factory=list)
    properties: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class InferenceRule:
    """An inference rule for reasoning."""

    name: str
    conditions: List[Tuple[str, str, str]]  # List of (subject, predicate, object) patterns
    conclusions: List[Tuple[str, str, str]]  # Inferred triples
    description: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH
# ═══════════════════════════════════════════════════════════════════════════

class KnowledgeGraph:
    """In-memory knowledge graph with reasoning capabilities."""

    def __init__(self):
        # Entity store
        self.entities: Dict[str, Entity] = {}

        # Triple store: (subject, predicate) -> list of objects
        self.triples: Dict[Tuple[str, str], List[Relation]] = defaultdict(list)

        # Reverse index: (predicate, object) -> list of subjects
        self.reverse_index: Dict[Tuple[str, str], List[str]] = defaultdict(list)

        # Graph representation (if NetworkX available)
        self.graph: Optional[nx.MultiDiGraph] = None
        if NETWORKX_AVAILABLE:
            self.graph = nx.MultiDiGraph()

        # Ontology
        self.ontology: Dict[str, OntologyClass] = {}

        # Inference rules
        self.rules: List[InferenceRule] = []

    def add_entity(self, entity: Entity):
        """Add entity to knowledge graph."""
        self.entities[entity.id] = entity

        if self.graph is not None:
            self.graph.add_node(entity.id, **entity.to_dict())

        logger.debug(f"Added entity: {entity.label} ({entity.id})")

    def add_relation(self, relation: Relation):
        """Add relation (triple) to knowledge graph."""
        key = (relation.subject, relation.predicate)
        self.triples[key].append(relation)

        # Update reverse index
        rev_key = (relation.predicate, relation.object)
        if relation.subject not in self.reverse_index[rev_key]:
            self.reverse_index[rev_key].append(relation.subject)

        # Add to graph
        if self.graph is not None:
            self.graph.add_edge(
                relation.subject,
                relation.object,
                predicate=relation.predicate,
                confidence=relation.confidence,
            )

        logger.debug(f"Added relation: {relation.subject} -{relation.predicate}-> {relation.object}")

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self.entities.get(entity_id)

    def find_entities(
        self,
        entity_type: Optional[EntityType] = None,
        label_pattern: Optional[str] = None,
    ) -> List[Entity]:
        """Find entities matching criteria."""
        results = []

        for entity in self.entities.values():
            if entity_type and entity.entity_type != entity_type:
                continue

            if label_pattern and label_pattern.lower() not in entity.label.lower():
                continue

            results.append(entity)

        return results

    def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
    ) -> List[Relation]:
        """Query for relations matching pattern.

        Use None as wildcard.
        """
        results = []

        if subject and predicate:
            # Direct lookup
            key = (subject, predicate)
            relations = self.triples.get(key, [])
            if object:
                results = [r for r in relations if r.object == object]
            else:
                results = relations

        elif predicate and object:
            # Reverse lookup
            rev_key = (predicate, object)
            subjects = self.reverse_index.get(rev_key, [])
            for subj in subjects:
                relations = self.triples.get((subj, predicate), [])
                results.extend([r for r in relations if r.object == object])

        else:
            # Full scan
            for key, relations in self.triples.items():
                for relation in relations:
                    if subject and relation.subject != subject:
                        continue
                    if predicate and relation.predicate != predicate:
                        continue
                    if object and relation.object != object:
                        continue
                    results.append(relation)

        return results

    def multi_hop_query(
        self,
        start_entity: str,
        relation_path: List[str],
        max_results: int = 100,
    ) -> List[List[str]]:
        """Multi-hop reasoning: follow a path of relations.

        Args:
            start_entity: Starting entity ID
            relation_path: List of relation types to follow
            max_results: Maximum paths to return

        Returns:
            List of entity paths
        """
        if not relation_path:
            return [[start_entity]]

        paths = [[start_entity]]

        for relation_type in relation_path:
            new_paths = []

            for path in paths:
                current_entity = path[-1]

                # Find all entities connected by this relation
                relations = self.query(subject=current_entity, predicate=relation_type)

                for relation in relations:
                    new_path = path + [relation.object]
                    new_paths.append(new_path)

                if len(new_paths) >= max_results:
                    break

            paths = new_paths

            if not paths:
                break

        return paths[:max_results]

    def shortest_path(self, start_entity: str, end_entity: str) -> Optional[List[str]]:
        """Find shortest path between two entities."""
        if not NETWORKX_AVAILABLE or self.graph is None:
            logger.warning("NetworkX not available for path finding")
            return None

        try:
            path = nx.shortest_path(self.graph, start_entity, end_entity)
            return path
        except nx.NetworkXNoPath:
            return None

    def get_neighbors(
        self,
        entity_id: str,
        relation_type: Optional[str] = None,
        direction: str = "outgoing",
    ) -> List[str]:
        """Get neighboring entities.

        Args:
            entity_id: Entity to find neighbors for
            relation_type: Optional relation type filter
            direction: 'outgoing', 'incoming', or 'both'

        Returns:
            List of neighbor entity IDs
        """
        neighbors = []

        if direction in ("outgoing", "both"):
            # Outgoing edges
            for key, relations in self.triples.items():
                subject, predicate = key
                if subject == entity_id:
                    if relation_type is None or predicate == relation_type:
                        neighbors.extend([r.object for r in relations])

        if direction in ("incoming", "both"):
            # Incoming edges
            for relations in self.triples.values():
                for relation in relations:
                    if relation.object == entity_id:
                        if relation_type is None or relation.predicate == relation_type:
                            neighbors.append(relation.subject)

        return list(set(neighbors))  # Deduplicate


# ═══════════════════════════════════════════════════════════════════════════
# ONTOLOGY & REASONING
# ═══════════════════════════════════════════════════════════════════════════

class OntologyReasoner:
    """Reasoning engine for ontologies and rules."""

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def add_class(self, ontology_class: OntologyClass):
        """Add class to ontology."""
        self.kg.ontology[ontology_class.name] = ontology_class
        logger.debug(f"Added ontology class: {ontology_class.name}")

    def add_rule(self, rule: InferenceRule):
        """Add inference rule."""
        self.kg.rules.append(rule)
        logger.debug(f"Added inference rule: {rule.name}")

    def infer_type_hierarchy(self):
        """Infer type relationships from ontology.

        If A is-a B and B is-a C, then A is-a C.
        """
        changed = True
        iterations = 0
        max_iterations = 100

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1

            for class_name, ontology_class in self.kg.ontology.items():
                for parent in ontology_class.parent_classes:
                    # Get parent's parents
                    if parent in self.kg.ontology:
                        grandparents = self.kg.ontology[parent].parent_classes

                        for grandparent in grandparents:
                            if grandparent not in ontology_class.parent_classes:
                                ontology_class.parent_classes.append(grandparent)
                                changed = True

        logger.info(f"Type hierarchy inference completed in {iterations} iterations")

    def apply_rules(self, max_iterations: int = 10) -> int:
        """Apply inference rules to derive new knowledge.

        Returns:
            Number of new triples inferred
        """
        inferred_count = 0

        for iteration in range(max_iterations):
            new_triples = []

            for rule in self.kg.rules:
                # Try to match rule conditions
                matches = self._match_rule_conditions(rule.conditions)

                # For each match, generate conclusions
                for match in matches:
                    for conclusion_pattern in rule.conclusions:
                        # Instantiate conclusion with matched variables
                        triple = self._instantiate_pattern(conclusion_pattern, match)

                        if triple and not self._triple_exists(triple):
                            new_triples.append(triple)

            # Add new triples
            for triple in new_triples:
                subject, predicate, object_ = triple
                relation = Relation(
                    subject=subject,
                    predicate=predicate,
                    object=object_,
                    confidence=0.8,  # Inferred triples have lower confidence
                    source="inference",
                )
                self.kg.add_relation(relation)
                inferred_count += 1

            if not new_triples:
                break  # Fixed point reached

        logger.info(f"Inferred {inferred_count} new triples")
        return inferred_count

    def _match_rule_conditions(
        self,
        conditions: List[Tuple[str, str, str]],
    ) -> List[Dict[str, str]]:
        """Match rule conditions against knowledge graph.

        Variables start with '?'.
        """
        # Start with all possible bindings for first condition
        if not conditions:
            return [{}]

        first_condition = conditions[0]
        bindings = self._match_pattern(first_condition, {})

        # Iteratively refine bindings with remaining conditions
        for condition in conditions[1:]:
            new_bindings = []

            for binding in bindings:
                matches = self._match_pattern(condition, binding)
                new_bindings.extend(matches)

            bindings = new_bindings

        return bindings

    def _match_pattern(
        self,
        pattern: Tuple[str, str, str],
        existing_binding: Dict[str, str],
    ) -> List[Dict[str, str]]:
        """Match a single triple pattern.

        Returns list of variable bindings.
        """
        subject, predicate, object_ = pattern

        # Instantiate pattern with existing bindings
        subject = existing_binding.get(subject, subject) if subject.startswith('?') else subject
        predicate = existing_binding.get(predicate, predicate) if predicate.startswith('?') else predicate
        object_ = existing_binding.get(object_, object_) if object_.startswith('?') else object_

        # Query knowledge graph
        query_subject = None if subject.startswith('?') else subject
        query_predicate = None if predicate.startswith('?') else predicate
        query_object = None if object_.startswith('?') else object_

        relations = self.kg.query(query_subject, query_predicate, query_object)

        # Generate bindings
        bindings = []

        for relation in relations:
            binding = existing_binding.copy()

            if pattern[0].startswith('?'):
                binding[pattern[0]] = relation.subject
            if pattern[1].startswith('?'):
                binding[pattern[1]] = relation.predicate
            if pattern[2].startswith('?'):
                binding[pattern[2]] = relation.object

            bindings.append(binding)

        return bindings

    def _instantiate_pattern(
        self,
        pattern: Tuple[str, str, str],
        binding: Dict[str, str],
    ) -> Optional[Tuple[str, str, str]]:
        """Instantiate a pattern with variable bindings."""
        subject, predicate, object_ = pattern

        subject = binding.get(subject, subject) if subject.startswith('?') else subject
        predicate = binding.get(predicate, predicate) if predicate.startswith('?') else predicate
        object_ = binding.get(object_, object_) if object_.startswith('?') else object_

        # Check all variables are bound
        if subject.startswith('?') or predicate.startswith('?') or object_.startswith('?'):
            return None

        return (subject, predicate, object_)

    def _triple_exists(self, triple: Tuple[str, str, str]) -> bool:
        """Check if triple already exists in knowledge graph."""
        subject, predicate, object_ = triple
        relations = self.kg.query(subject, predicate, object_)
        return len(relations) > 0


# ═══════════════════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════════

class KGEmbedding:
    """Knowledge graph embedding for vector-based reasoning.

    Implements TransE-style embedding: h + r ≈ t
    """

    def __init__(self, kg: KnowledgeGraph, embedding_dim: int = 128):
        self.kg = kg
        self.embedding_dim = embedding_dim

        # Embeddings
        self.entity_embeddings: Dict[str, np.ndarray] = {}
        self.relation_embeddings: Dict[str, np.ndarray] = {}

        # Initialize randomly
        self._initialize_embeddings()

    def _initialize_embeddings(self):
        """Initialize embeddings randomly."""
        # Entity embeddings
        for entity_id in self.kg.entities.keys():
            self.entity_embeddings[entity_id] = np.random.randn(self.embedding_dim) * 0.01

        # Relation embeddings
        relation_types = set()
        for relations in self.kg.triples.values():
            for relation in relations:
                relation_types.add(relation.predicate)

        for relation_type in relation_types:
            self.relation_embeddings[relation_type] = np.random.randn(self.embedding_dim) * 0.01

    def train(self, num_epochs: int = 100, learning_rate: float = 0.01, margin: float = 1.0):
        """Train embeddings using TransE loss.

        Loss = max(0, margin + d(h+r, t) - d(h+r, t'))
        where t' is a corrupted tail.
        """
        logger.info(f"Training KG embeddings for {num_epochs} epochs...")

        # Collect all triples
        all_triples = []
        for relations in self.kg.triples.values():
            for relation in relations:
                all_triples.append((relation.subject, relation.predicate, relation.object))

        for epoch in range(num_epochs):
            total_loss = 0.0

            # Shuffle triples
            np.random.shuffle(all_triples)

            for head, relation, tail in all_triples:
                # Skip if embedding doesn't exist
                if head not in self.entity_embeddings or tail not in self.entity_embeddings:
                    continue
                if relation not in self.relation_embeddings:
                    continue

                # Get embeddings
                h = self.entity_embeddings[head]
                r = self.relation_embeddings[relation]
                t = self.entity_embeddings[tail]

                # Positive score
                pos_score = np.linalg.norm(h + r - t)

                # Generate negative sample (corrupt tail)
                neg_tail = np.random.choice(list(self.entity_embeddings.keys()))
                t_neg = self.entity_embeddings[neg_tail]
                neg_score = np.linalg.norm(h + r - t_neg)

                # Hinge loss
                loss = max(0, margin + pos_score - neg_score)
                total_loss += loss

                if loss > 0:
                    # Gradient descent (simplified)
                    # ∇h = (h + r - t) / ||h + r - t|| - (h + r - t') / ||h + r - t'||
                    pos_grad = (h + r - t) / (pos_score + 1e-8)
                    neg_grad = (h + r - t_neg) / (neg_score + 1e-8)

                    grad_h = pos_grad - neg_grad
                    grad_r = pos_grad - neg_grad
                    grad_t = -pos_grad

                    # Update embeddings
                    self.entity_embeddings[head] -= learning_rate * grad_h
                    self.relation_embeddings[relation] -= learning_rate * grad_r
                    self.entity_embeddings[tail] -= learning_rate * grad_t

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: loss = {total_loss:.3f}")

    def predict_tail(self, head: str, relation: str, k: int = 10) -> List[Tuple[str, float]]:
        """Predict tail entities for (head, relation, ?)."""
        if head not in self.entity_embeddings or relation not in self.relation_embeddings:
            return []

        h = self.entity_embeddings[head]
        r = self.relation_embeddings[relation]

        # Compute scores for all entities
        scores = []
        for entity_id, t in self.entity_embeddings.items():
            score = -np.linalg.norm(h + r - t)  # Negative distance (higher is better)
            scores.append((entity_id, score))

        # Sort and return top k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED KNOWLEDGE GRAPH SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

class KnowledgeGraphSystem:
    """Unified knowledge graph system with reasoning and embeddings."""

    def __init__(self, embedding_dim: int = 128):
        self.kg = KnowledgeGraph()
        self.reasoner = OntologyReasoner(self.kg)
        self.embedding: Optional[KGEmbedding] = None
        self.embedding_dim = embedding_dim

    def add_entity(self, entity: Entity):
        """Add entity."""
        self.kg.add_entity(entity)

    def add_relation(self, relation: Relation):
        """Add relation."""
        self.kg.add_relation(relation)

    def query(self, subject=None, predicate=None, object=None) -> List[Relation]:
        """Query knowledge graph."""
        return self.kg.query(subject, predicate, object)

    def multi_hop(self, start: str, path: List[str]) -> List[List[str]]:
        """Multi-hop reasoning."""
        return self.kg.multi_hop_query(start, path)

    def add_ontology_class(self, ontology_class: OntologyClass):
        """Add ontology class."""
        self.reasoner.add_class(ontology_class)

    def add_rule(self, rule: InferenceRule):
        """Add inference rule."""
        self.reasoner.add_rule(rule)

    def apply_reasoning(self) -> int:
        """Apply ontology and rule-based reasoning."""
        self.reasoner.infer_type_hierarchy()
        return self.reasoner.apply_rules()

    def train_embeddings(self, num_epochs: int = 100):
        """Train knowledge graph embeddings."""
        self.embedding = KGEmbedding(self.kg, self.embedding_dim)
        self.embedding.train(num_epochs=num_epochs)

    def predict_relation(self, head: str, relation: str, k: int = 10) -> List[Tuple[str, float]]:
        """Predict tail entities using embeddings."""
        if not self.embedding:
            raise ValueError("Embeddings not trained")
        return self.embedding.predict_tail(head, relation, k)

    def export_to_json(self, filepath: str):
        """Export knowledge graph to JSON."""
        data = {
            "entities": {eid: e.to_dict() for eid, e in self.kg.entities.items()},
            "relations": [],
        }

        for relations in self.kg.triples.values():
            data["relations"].extend([r.to_dict() for r in relations])

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported knowledge graph to {filepath}")

    def import_from_json(self, filepath: str):
        """Import knowledge graph from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Import entities
        for entity_data in data.get("entities", {}).values():
            entity = Entity(
                id=entity_data["id"],
                label=entity_data["label"],
                entity_type=EntityType(entity_data["type"]),
                properties=entity_data.get("properties", {}),
                aliases=entity_data.get("aliases", []),
                metadata=entity_data.get("metadata", {}),
            )
            self.add_entity(entity)

        # Import relations
        for relation_data in data.get("relations", []):
            relation = Relation(
                subject=relation_data["subject"],
                predicate=relation_data["predicate"],
                object=relation_data["object"],
                confidence=relation_data.get("confidence", 1.0),
                source=relation_data.get("source"),
                metadata=relation_data.get("metadata", {}),
            )
            self.add_relation(relation)

        logger.info(f"Imported knowledge graph from {filepath}")


# Global registry
_knowledge_graphs: Dict[str, KnowledgeGraphSystem] = {}


def create_knowledge_graph(name: str = "default", embedding_dim: int = 128) -> KnowledgeGraphSystem:
    """Create a knowledge graph system."""
    kg_system = KnowledgeGraphSystem(embedding_dim)
    _knowledge_graphs[name] = kg_system
    return kg_system


def get_knowledge_graph(name: str = "default") -> Optional[KnowledgeGraphSystem]:
    """Get knowledge graph by name."""
    return _knowledge_graphs.get(name)
