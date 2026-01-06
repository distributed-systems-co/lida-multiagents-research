"""Advanced semantic memory system with vector database integration.

Supports:
- Multi-modal embeddings (text, images, structured data)
- Hierarchical memory organization (episodic, semantic, procedural)
- Temporal decay and importance weighting
- Cross-modal retrieval
- Memory consolidation and forgetting
- Associative recall with spreading activation
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum
import json

logger = logging.getLogger(__name__)

# Vector database backend support
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("ChromaDB not available - install with: pip install chromadb")

try:
    import qdrant_client
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class MemoryType(Enum):
    """Types of memory in cognitive hierarchy."""
    EPISODIC = "episodic"  # Specific experiences and events
    SEMANTIC = "semantic"  # General knowledge and facts
    PROCEDURAL = "procedural"  # Skills and how-to knowledge
    WORKING = "working"  # Temporary, active information
    PROSPECTIVE = "prospective"  # Future intentions and plans


class MemoryImportance(Enum):
    """Importance levels for memory retention."""
    CRITICAL = 1.0  # Never forget
    HIGH = 0.8  # Remember for long time
    MEDIUM = 0.5  # Normal retention
    LOW = 0.3  # Fast decay
    EPHEMERAL = 0.1  # Very fast decay


class ConsolidationStrategy(Enum):
    """Strategies for memory consolidation."""
    FREQUENCY_BASED = "frequency"  # Consolidate frequently accessed memories
    RECENCY_BASED = "recency"  # Consolidate recent memories
    IMPORTANCE_BASED = "importance"  # Consolidate important memories
    ASSOCIATIVE = "associative"  # Consolidate strongly connected memories
    PREDICTIVE = "predictive"  # Consolidate memories useful for prediction


@dataclass
class MemoryTrace:
    """A single memory trace with metadata."""

    id: str
    content: Any
    embedding: np.ndarray
    memory_type: MemoryType
    importance: float
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    decay_rate: float = 0.1  # Per day
    associations: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Multi-modal support
    modality: str = "text"  # text, image, structured, audio, video

    # Consolidation tracking
    consolidated: bool = False
    consolidation_strength: float = 0.0

    def __post_init__(self):
        if isinstance(self.embedding, list):
            self.embedding = np.array(self.embedding)

    def compute_activation(self, current_time: Optional[datetime] = None) -> float:
        """Compute current activation level using ACT-R inspired formula.

        Activation = BaseLevel + Spreading + Noise

        BaseLevel = ln(sum(t_i^-d)) where t_i is time since access i, d is decay
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Base-level activation (recency and frequency)
        time_since_creation = (current_time - self.created_at).total_seconds() / 86400  # days
        time_since_access = (current_time - self.last_accessed).total_seconds() / 86400

        # Avoid log(0)
        time_since_creation = max(time_since_creation, 0.0001)
        time_since_access = max(time_since_access, 0.0001)

        base_level = np.log(
            time_since_creation ** (-self.decay_rate) +
            (self.access_count * time_since_access ** (-self.decay_rate))
        )

        # Importance boost
        importance_boost = self.importance * 2.0

        # Consolidation boost
        consolidation_boost = self.consolidation_strength * 1.5

        activation = base_level + importance_boost + consolidation_boost

        return float(activation)

    def should_forget(self, threshold: float = -2.0, current_time: Optional[datetime] = None) -> bool:
        """Determine if memory should be forgotten based on activation."""
        return self.compute_activation(current_time) < threshold

    def access(self):
        """Record memory access."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding.tolist() if isinstance(self.embedding, np.ndarray) else self.embedding,
            "memory_type": self.memory_type.value,
            "importance": self.importance,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "decay_rate": self.decay_rate,
            "associations": list(self.associations),
            "tags": list(self.tags),
            "metadata": self.metadata,
            "modality": self.modality,
            "consolidated": self.consolidated,
            "consolidation_strength": self.consolidation_strength,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MemoryTrace:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            embedding=np.array(data["embedding"]),
            memory_type=MemoryType(data["memory_type"]),
            importance=data["importance"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            access_count=data.get("access_count", 0),
            decay_rate=data.get("decay_rate", 0.1),
            associations=set(data.get("associations", [])),
            tags=set(data.get("tags", [])),
            metadata=data.get("metadata", {}),
            modality=data.get("modality", "text"),
            consolidated=data.get("consolidated", False),
            consolidation_strength=data.get("consolidation_strength", 0.0),
        )


class VectorStore:
    """Abstract vector store interface."""

    async def add(self, traces: List[MemoryTrace]):
        """Add memory traces."""
        raise NotImplementedError

    async def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filter_dict: Optional[Dict] = None,
    ) -> List[Tuple[MemoryTrace, float]]:
        """Search for similar memories."""
        raise NotImplementedError

    async def delete(self, trace_ids: List[str]):
        """Delete memory traces."""
        raise NotImplementedError

    async def update(self, trace: MemoryTrace):
        """Update a memory trace."""
        raise NotImplementedError

    async def get_by_id(self, trace_id: str) -> Optional[MemoryTrace]:
        """Get a specific memory trace."""
        raise NotImplementedError


class ChromaDBStore(VectorStore):
    """ChromaDB backend for vector storage."""

    def __init__(self, collection_name: str = "agent_memory", persist_directory: str = "./.chroma_db"):
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB not available")

        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory
        ))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    async def add(self, traces: List[MemoryTrace]):
        """Add memory traces to ChromaDB."""
        self.collection.add(
            embeddings=[t.embedding.tolist() for t in traces],
            documents=[json.dumps(t.to_dict()) for t in traces],
            ids=[t.id for t in traces],
            metadatas=[{
                "memory_type": t.memory_type.value,
                "importance": t.importance,
                "modality": t.modality,
            } for t in traces],
        )

    async def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filter_dict: Optional[Dict] = None,
    ) -> List[Tuple[MemoryTrace, float]]:
        """Search ChromaDB."""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where=filter_dict,
        )

        traces = []
        if results["documents"]:
            for doc, distance in zip(results["documents"][0], results["distances"][0]):
                trace_dict = json.loads(doc)
                trace = MemoryTrace.from_dict(trace_dict)
                # Convert distance to similarity score
                similarity = 1.0 / (1.0 + distance)
                traces.append((trace, similarity))

        return traces

    async def delete(self, trace_ids: List[str]):
        """Delete from ChromaDB."""
        self.collection.delete(ids=trace_ids)

    async def update(self, trace: MemoryTrace):
        """Update in ChromaDB."""
        # ChromaDB doesn't have direct update, so delete and re-add
        await self.delete([trace.id])
        await self.add([trace])

    async def get_by_id(self, trace_id: str) -> Optional[MemoryTrace]:
        """Get by ID from ChromaDB."""
        results = self.collection.get(ids=[trace_id])
        if results["documents"]:
            trace_dict = json.loads(results["documents"][0])
            return MemoryTrace.from_dict(trace_dict)
        return None


class FAISSStore(VectorStore):
    """FAISS backend for high-performance vector search."""

    def __init__(self, dimension: int = 768, index_type: str = "IVF"):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available")

        self.dimension = dimension

        # Create index based on type
        if index_type == "Flat":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "IVF":
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
            # Need to train IVF index
            self.trained = False
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        # Store metadata separately (FAISS only stores vectors)
        self.id_to_trace: Dict[str, MemoryTrace] = {}
        self.idx_to_id: Dict[int, str] = {}
        self.next_idx = 0

    async def add(self, traces: List[MemoryTrace]):
        """Add to FAISS index."""
        if not traces:
            return

        # Convert embeddings to numpy array
        embeddings = np.array([t.embedding for t in traces]).astype('float32')

        # Train IVF index if needed
        if hasattr(self, 'trained') and not self.trained and len(self.id_to_trace) + len(traces) >= 100:
            # Collect all embeddings for training
            all_embeddings = embeddings
            if self.id_to_trace:
                existing = np.array([t.embedding for t in self.id_to_trace.values()]).astype('float32')
                all_embeddings = np.vstack([all_embeddings, existing])
            self.index.train(all_embeddings)
            self.trained = True

        # Add to index
        self.index.add(embeddings)

        # Store metadata
        for trace in traces:
            self.id_to_trace[trace.id] = trace
            self.idx_to_id[self.next_idx] = trace.id
            self.next_idx += 1

    async def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filter_dict: Optional[Dict] = None,
    ) -> List[Tuple[MemoryTrace, float]]:
        """Search FAISS index."""
        if self.index.ntotal == 0:
            return []

        # Search
        query = query_embedding.astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query, min(k, self.index.ntotal))

        # Convert to traces
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for missing results
                continue

            trace_id = self.idx_to_id.get(int(idx))
            if not trace_id:
                continue

            trace = self.id_to_trace.get(trace_id)
            if not trace:
                continue

            # Apply filters if specified
            if filter_dict:
                if not self._matches_filter(trace, filter_dict):
                    continue

            # Convert L2 distance to similarity
            similarity = 1.0 / (1.0 + float(dist))
            results.append((trace, similarity))

        return results

    def _matches_filter(self, trace: MemoryTrace, filter_dict: Dict) -> bool:
        """Check if trace matches filter criteria."""
        for key, value in filter_dict.items():
            if key == "memory_type" and trace.memory_type.value != value:
                return False
            if key == "modality" and trace.modality != value:
                return False
            if key == "importance_min" and trace.importance < value:
                return False
            if key == "importance_max" and trace.importance > value:
                return False
        return True

    async def delete(self, trace_ids: List[str]):
        """Delete from FAISS (requires rebuilding index)."""
        for trace_id in trace_ids:
            if trace_id in self.id_to_trace:
                del self.id_to_trace[trace_id]

        # FAISS doesn't support deletion - would need to rebuild index
        # For simplicity, just remove from metadata
        logger.warning("FAISS deletion requires index rebuild - only removed from metadata")

    async def update(self, trace: MemoryTrace):
        """Update in FAISS."""
        if trace.id in self.id_to_trace:
            self.id_to_trace[trace.id] = trace

    async def get_by_id(self, trace_id: str) -> Optional[MemoryTrace]:
        """Get by ID."""
        return self.id_to_trace.get(trace_id)


class SemanticMemory:
    """Advanced semantic memory system with consolidation and forgetting.

    Features:
    - Hierarchical memory types (episodic, semantic, procedural)
    - Activation-based retrieval
    - Spreading activation for associative recall
    - Memory consolidation
    - Intelligent forgetting based on activation
    - Cross-modal retrieval
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_function: Any = None,  # Function that takes text/content and returns embedding
        forget_threshold: float = -2.0,
        consolidation_interval: int = 3600,  # seconds
    ):
        self.vector_store = vector_store
        self.embedding_function = embedding_function
        self.forget_threshold = forget_threshold
        self.consolidation_interval = consolidation_interval

        # Local cache for fast access
        self._cache: Dict[str, MemoryTrace] = {}

        # Consolidation state
        self._last_consolidation = datetime.now(timezone.utc)
        self._consolidation_task: Optional[asyncio.Task] = None

    async def store(
        self,
        content: Any,
        memory_type: MemoryType,
        importance: float = 0.5,
        tags: Optional[Set[str]] = None,
        metadata: Optional[Dict] = None,
        embedding: Optional[np.ndarray] = None,
        modality: str = "text",
    ) -> MemoryTrace:
        """Store a new memory."""
        # Generate embedding if not provided
        if embedding is None and self.embedding_function:
            embedding = await self._generate_embedding(content, modality)
        elif embedding is None:
            # Dummy embedding if no function provided
            embedding = np.random.randn(768)

        # Create memory trace
        trace_id = self._generate_id(content)
        trace = MemoryTrace(
            id=trace_id,
            content=content,
            embedding=embedding,
            memory_type=memory_type,
            importance=importance,
            created_at=datetime.now(timezone.utc),
            last_accessed=datetime.now(timezone.utc),
            tags=tags or set(),
            metadata=metadata or {},
            modality=modality,
        )

        # Store in vector DB
        await self.vector_store.add([trace])

        # Cache locally
        self._cache[trace_id] = trace

        return trace

    async def recall(
        self,
        query: Union[str, np.ndarray],
        k: int = 10,
        memory_type: Optional[MemoryType] = None,
        modality: Optional[str] = None,
        min_activation: float = -1.0,
        spreading_activation: bool = True,
        spreading_depth: int = 2,
    ) -> List[Tuple[MemoryTrace, float]]:
        """Recall memories similar to query.

        Args:
            query: Text query or embedding vector
            k: Number of memories to retrieve
            memory_type: Filter by memory type
            modality: Filter by modality
            min_activation: Minimum activation threshold
            spreading_activation: Whether to use spreading activation
            spreading_depth: Depth of spreading activation

        Returns:
            List of (MemoryTrace, combined_score) tuples
        """
        # Generate query embedding
        if isinstance(query, str):
            query_embedding = await self._generate_embedding(query, modality or "text")
        else:
            query_embedding = query

        # Build filter
        filter_dict = {}
        if memory_type:
            filter_dict["memory_type"] = memory_type.value
        if modality:
            filter_dict["modality"] = modality

        # Search vector store
        results = await self.vector_store.search(query_embedding, k=k*2, filter_dict=filter_dict)

        # Compute combined scores (similarity + activation)
        current_time = datetime.now(timezone.utc)
        scored_results = []

        for trace, similarity in results:
            # Update access
            trace.access()

            # Compute activation
            activation = trace.compute_activation(current_time)

            # Filter by minimum activation
            if activation < min_activation:
                continue

            # Combined score: weighted sum of similarity and activation
            combined_score = 0.7 * similarity + 0.3 * (activation + 3.0) / 6.0  # Normalize activation

            scored_results.append((trace, combined_score))

        # Sort by combined score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        scored_results = scored_results[:k]

        # Apply spreading activation if enabled
        if spreading_activation and scored_results:
            scored_results = await self._spread_activation(scored_results, spreading_depth, k)

        return scored_results

    async def _spread_activation(
        self,
        initial_traces: List[Tuple[MemoryTrace, float]],
        depth: int,
        k: int,
    ) -> List[Tuple[MemoryTrace, float]]:
        """Apply spreading activation to find associated memories."""
        if depth <= 0:
            return initial_traces

        activated = {t.id: score for t, score in initial_traces}

        for trace, score in initial_traces:
            # Spread to associated memories
            for assoc_id in trace.associations:
                if assoc_id not in activated:
                    assoc_trace = await self.vector_store.get_by_id(assoc_id)
                    if assoc_trace:
                        # Activation spreads with decay
                        spread_score = score * 0.7
                        activated[assoc_id] = max(activated.get(assoc_id, 0.0), spread_score)

        # Get all activated traces
        all_traces = []
        for trace_id, score in activated.items():
            if trace_id in self._cache:
                all_traces.append((self._cache[trace_id], score))
            else:
                trace = await self.vector_store.get_by_id(trace_id)
                if trace:
                    all_traces.append((trace, score))

        # Sort and limit
        all_traces.sort(key=lambda x: x[1], reverse=True)
        return all_traces[:k]

    async def associate(self, trace_id1: str, trace_id2: str):
        """Create bidirectional association between memories."""
        trace1 = await self.vector_store.get_by_id(trace_id1)
        trace2 = await self.vector_store.get_by_id(trace_id2)

        if trace1 and trace2:
            trace1.associations.add(trace_id2)
            trace2.associations.add(trace_id1)

            await self.vector_store.update(trace1)
            await self.vector_store.update(trace2)

    async def consolidate(self, strategy: ConsolidationStrategy = ConsolidationStrategy.FREQUENCY_BASED):
        """Consolidate memories based on strategy."""
        logger.info(f"Starting memory consolidation with strategy: {strategy.value}")

        # This would retrieve all memories and consolidate based on strategy
        # For now, just update consolidation timestamp
        self._last_consolidation = datetime.now(timezone.utc)

        # In a full implementation:
        # 1. Retrieve memories based on strategy criteria
        # 2. Strengthen connections between related memories
        # 3. Increase consolidation_strength
        # 4. Reduce decay_rate for consolidated memories

    async def forget(self):
        """Remove low-activation memories."""
        # Get all cached memories
        to_forget = []
        current_time = datetime.now(timezone.utc)

        for trace in self._cache.values():
            if trace.should_forget(self.forget_threshold, current_time):
                to_forget.append(trace.id)

        if to_forget:
            logger.info(f"Forgetting {len(to_forget)} low-activation memories")
            await self.vector_store.delete(to_forget)
            for trace_id in to_forget:
                del self._cache[trace_id]

    async def _generate_embedding(self, content: Any, modality: str) -> np.ndarray:
        """Generate embedding for content."""
        if self.embedding_function:
            return await self.embedding_function(content, modality)
        else:
            # Dummy embedding
            return np.random.randn(768)

    def _generate_id(self, content: Any) -> str:
        """Generate unique ID for content."""
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    async def start_consolidation_loop(self):
        """Start background consolidation task."""
        if self._consolidation_task:
            return

        async def consolidation_loop():
            while True:
                await asyncio.sleep(self.consolidation_interval)
                await self.consolidate()
                await self.forget()

        self._consolidation_task = asyncio.create_task(consolidation_loop())

    async def stop_consolidation_loop(self):
        """Stop background consolidation task."""
        if self._consolidation_task:
            self._consolidation_task.cancel()
            try:
                await self._consolidation_task
            except asyncio.CancelledError:
                pass
            self._consolidation_task = None


# Global registry
_memory_systems: Dict[str, SemanticMemory] = {}


def get_memory_system(name: str = "default") -> Optional[SemanticMemory]:
    """Get a memory system by name."""
    return _memory_systems.get(name)


def create_memory_system(
    name: str = "default",
    backend: str = "chroma",
    **kwargs,
) -> SemanticMemory:
    """Create and register a memory system."""
    if backend == "chroma":
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB not available")
        store = ChromaDBStore(**kwargs)
    elif backend == "faiss":
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available")
        store = FAISSStore(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    memory = SemanticMemory(vector_store=store, **kwargs)
    _memory_systems[name] = memory
    return memory
