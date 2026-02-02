"""Prompt Evolution System with Merkle Trees, Forking, and Retrodynamic Evaluation.

This module implements a git-like version control system for agent prompts with:

1. Content-Addressable Storage (CAS) - Prompts stored by their hash
2. Merkle Trees - Cryptographic verification of prompt lineage
3. Forking - Parallel branches of prompt evolution
4. Retrodynamic Re-evaluation - Replay conversations with different prompts
5. Fitness Scoring - Track which prompt versions perform better

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    PromptEvolutionEngine                     │
    │  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐ │
    │  │ MerkleStore │  │  ForkGraph  │  │ RetrodynamicEngine   │ │
    │  │  (CAS)      │  │  (DAG)      │  │ (Replay & Evaluate)  │ │
    │  └─────────────┘  └─────────────┘  └──────────────────────┘ │
    └─────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Content-Addressable Storage with Merkle Proofs
# =============================================================================


def compute_hash(content: str) -> str:
    """Compute SHA-256 hash of content."""
    return hashlib.sha256(content.encode()).hexdigest()


def compute_merkle_root(hashes: List[str]) -> str:
    """Compute Merkle root from a list of hashes."""
    if not hashes:
        return compute_hash("")
    if len(hashes) == 1:
        return hashes[0]

    # Pair and hash
    next_level = []
    for i in range(0, len(hashes), 2):
        left = hashes[i]
        right = hashes[i + 1] if i + 1 < len(hashes) else left
        combined = compute_hash(left + right)
        next_level.append(combined)

    return compute_merkle_root(next_level)


@dataclass(frozen=True, slots=True)
class PromptNode:
    """Immutable node in the prompt evolution tree."""
    content_hash: str
    content: str
    parent_hash: Optional[str]
    timestamp: float
    author: str  # "agent", "user", "system"

    # Metadata
    reason: str
    modification_type: str  # "genesis", "modify", "append", "fork", "merge", "revert"

    # Merkle proof components
    merkle_root: str  # Root of all ancestors
    depth: int

    # Performance tracking
    fitness_score: float = 0.0
    evaluation_count: int = 0

    # Fork tracking
    fork_name: Optional[str] = None
    fork_source: Optional[str] = None  # Hash of node this was forked from

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_hash": self.content_hash,
            "content": self.content,
            "parent_hash": self.parent_hash,
            "timestamp": self.timestamp,
            "author": self.author,
            "reason": self.reason,
            "modification_type": self.modification_type,
            "merkle_root": self.merkle_root,
            "depth": self.depth,
            "fitness_score": self.fitness_score,
            "evaluation_count": self.evaluation_count,
            "fork_name": self.fork_name,
            "fork_source": self.fork_source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptNode":
        return cls(**data)


class MerklePromptStore:
    """Content-addressable store for prompts with Merkle proofs."""

    def __init__(self):
        self._nodes: Dict[str, PromptNode] = {}
        self._content_to_hash: Dict[str, str] = {}  # Dedup by content

    def store(
        self,
        content: str,
        parent_hash: Optional[str],
        author: str,
        reason: str,
        modification_type: str,
        fork_name: Optional[str] = None,
        fork_source: Optional[str] = None,
    ) -> PromptNode:
        """Store a new prompt version."""
        content_hash = compute_hash(content)

        # Check if identical content exists
        if content in self._content_to_hash:
            existing_hash = self._content_to_hash[content]
            return self._nodes[existing_hash]

        # Compute depth and ancestors for Merkle root
        if parent_hash and parent_hash in self._nodes:
            parent = self._nodes[parent_hash]
            depth = parent.depth + 1
            ancestor_hashes = self._get_ancestor_hashes(parent_hash)
            ancestor_hashes.append(content_hash)
            merkle_root = compute_merkle_root(ancestor_hashes)
        else:
            depth = 0
            merkle_root = content_hash

        node = PromptNode(
            content_hash=content_hash,
            content=content,
            parent_hash=parent_hash,
            timestamp=time.time(),
            author=author,
            reason=reason,
            modification_type=modification_type,
            merkle_root=merkle_root,
            depth=depth,
            fork_name=fork_name,
            fork_source=fork_source,
        )

        self._nodes[content_hash] = node
        self._content_to_hash[content] = content_hash

        return node

    def get(self, content_hash: str) -> Optional[PromptNode]:
        """Retrieve a prompt by hash."""
        return self._nodes.get(content_hash)

    def get_by_content(self, content: str) -> Optional[PromptNode]:
        """Retrieve a prompt by its content."""
        content_hash = self._content_to_hash.get(content)
        if content_hash:
            return self._nodes.get(content_hash)
        return None

    def _get_ancestor_hashes(self, node_hash: str) -> List[str]:
        """Get all ancestor hashes for Merkle proof."""
        ancestors = []
        current = node_hash

        while current:
            ancestors.append(current)
            node = self._nodes.get(current)
            if node:
                current = node.parent_hash
            else:
                break

        return list(reversed(ancestors))

    def verify_lineage(self, node_hash: str) -> bool:
        """Verify the Merkle proof for a node's lineage."""
        node = self._nodes.get(node_hash)
        if not node:
            return False

        ancestors = self._get_ancestor_hashes(node_hash)
        computed_root = compute_merkle_root(ancestors)

        return computed_root == node.merkle_root

    def get_lineage(self, node_hash: str) -> List[PromptNode]:
        """Get full lineage from genesis to this node."""
        lineage = []
        current = node_hash

        while current:
            node = self._nodes.get(current)
            if node:
                lineage.append(node)
                current = node.parent_hash
            else:
                break

        return list(reversed(lineage))

    def update_fitness(
        self,
        node_hash: str,
        score: float,
        propagate: bool = True,
        decay: float = 0.9
    ):
        """Update fitness score for a node, optionally propagating to ancestors."""
        node = self._nodes.get(node_hash)
        if not node:
            return

        # Create updated node (immutable, so we replace)
        updated = PromptNode(
            content_hash=node.content_hash,
            content=node.content,
            parent_hash=node.parent_hash,
            timestamp=node.timestamp,
            author=node.author,
            reason=node.reason,
            modification_type=node.modification_type,
            merkle_root=node.merkle_root,
            depth=node.depth,
            fitness_score=(
                (node.fitness_score * node.evaluation_count + score) /
                (node.evaluation_count + 1)
            ),
            evaluation_count=node.evaluation_count + 1,
            fork_name=node.fork_name,
            fork_source=node.fork_source,
        )
        self._nodes[node_hash] = updated

        # Propagate to ancestors with decay
        if propagate and node.parent_hash:
            self.update_fitness(node.parent_hash, score * decay, propagate=True, decay=decay)

    def all_nodes(self) -> List[PromptNode]:
        """Get all stored nodes."""
        return list(self._nodes.values())


# =============================================================================
# Fork Graph (DAG for parallel prompt evolution)
# =============================================================================


@dataclass
class Fork:
    """A named branch in the prompt evolution."""
    name: str
    head_hash: str  # Current tip of this fork
    created_at: float
    created_from: str  # Hash of the node this fork started from
    description: str
    is_active: bool = True
    merged_into: Optional[str] = None  # Fork name if merged


class ForkGraph:
    """Manages parallel branches of prompt evolution."""

    def __init__(self, store: MerklePromptStore):
        self._store = store
        self._forks: Dict[str, Fork] = {}
        self._main_fork = "main"

    def create_fork(
        self,
        name: str,
        source_hash: str,
        description: str = ""
    ) -> Fork:
        """Create a new fork from a specific node."""
        if name in self._forks:
            raise ValueError(f"Fork '{name}' already exists")

        fork = Fork(
            name=name,
            head_hash=source_hash,
            created_at=time.time(),
            created_from=source_hash,
            description=description,
        )
        self._forks[name] = fork

        logger.info(f"Created fork '{name}' from {source_hash[:8]}")
        return fork

    def get_fork(self, name: str) -> Optional[Fork]:
        """Get a fork by name."""
        return self._forks.get(name)

    def get_head(self, fork_name: str) -> Optional[PromptNode]:
        """Get the head node of a fork."""
        fork = self._forks.get(fork_name)
        if fork:
            return self._store.get(fork.head_hash)
        return None

    def advance_fork(self, fork_name: str, new_hash: str):
        """Move a fork's head to a new node."""
        if fork_name not in self._forks:
            raise ValueError(f"Fork '{fork_name}' does not exist")

        self._forks[fork_name].head_hash = new_hash

    def list_forks(self) -> List[Fork]:
        """List all forks."""
        return list(self._forks.values())

    def get_divergence_point(self, fork1: str, fork2: str) -> Optional[str]:
        """Find the common ancestor of two forks."""
        head1 = self._forks.get(fork1)
        head2 = self._forks.get(fork2)

        if not head1 or not head2:
            return None

        ancestors1 = set(
            n.content_hash for n in self._store.get_lineage(head1.head_hash)
        )

        for node in self._store.get_lineage(head2.head_hash):
            if node.content_hash in ancestors1:
                return node.content_hash

        return None

    def merge_forks(
        self,
        source_fork: str,
        target_fork: str,
        merged_content: str,
        reason: str,
    ) -> PromptNode:
        """Merge one fork into another."""
        source = self._forks.get(source_fork)
        target = self._forks.get(target_fork)

        if not source or not target:
            raise ValueError("Both forks must exist")

        # Create merge node
        node = self._store.store(
            content=merged_content,
            parent_hash=target.head_hash,
            author="agent",
            reason=f"Merge '{source_fork}' into '{target_fork}': {reason}",
            modification_type="merge",
            fork_name=target_fork,
            fork_source=source.head_hash,
        )

        # Update target fork
        self.advance_fork(target_fork, node.content_hash)

        # Mark source as merged
        source.is_active = False
        source.merged_into = target_fork

        logger.info(f"Merged '{source_fork}' into '{target_fork}'")
        return node

    def get_fork_diff(self, fork_name: str) -> List[PromptNode]:
        """Get all nodes unique to a fork (since divergence from main)."""
        fork = self._forks.get(fork_name)
        main = self._forks.get(self._main_fork)

        if not fork or not main:
            return []

        divergence = self.get_divergence_point(fork_name, self._main_fork)
        if not divergence:
            return self._store.get_lineage(fork.head_hash)

        # Get nodes after divergence
        lineage = self._store.get_lineage(fork.head_hash)
        diff = []
        found_divergence = False

        for node in lineage:
            if node.content_hash == divergence:
                found_divergence = True
                continue
            if found_divergence:
                diff.append(node)

        return diff


# =============================================================================
# Retrodynamic Re-evaluation Engine
# =============================================================================


@dataclass
class ConversationSnapshot:
    """A frozen snapshot of a conversation for replay."""
    snapshot_id: str
    messages: List[Dict[str, Any]]
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    prompt_hash: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result of re-evaluating a conversation with a different prompt."""
    original_prompt_hash: str
    new_prompt_hash: str
    snapshot_id: str

    # Comparison metrics
    response_similarity: float  # 0-1, how similar the responses are
    tool_call_diff: int  # Difference in number of tool calls
    outcome_match: bool  # Whether the final outcome matches

    # Quality metrics
    coherence_score: float
    task_completion_score: float
    efficiency_score: float

    # Overall fitness delta
    fitness_delta: float

    # Details
    original_response: str
    new_response: str
    analysis: str


class RetrodynamicEngine:
    """Engine for replaying conversations with different prompts.

    This enables:
    1. A/B testing of prompt modifications
    2. Counterfactual analysis ("what if I had used this prompt?")
    3. Fitness scoring for prompt evolution
    4. Learning from past interactions
    """

    def __init__(
        self,
        store: MerklePromptStore,
        inference_fn: Callable,  # async fn(prompt, messages) -> response
    ):
        self._store = store
        self._inference_fn = inference_fn
        self._snapshots: Dict[str, ConversationSnapshot] = {}
        self._evaluations: List[EvaluationResult] = []

    def capture_snapshot(
        self,
        messages: List[Dict[str, Any]],
        tool_calls: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
        prompt_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationSnapshot:
        """Capture a conversation snapshot for later replay."""
        snapshot_id = compute_hash(json.dumps({
            "messages": messages,
            "prompt_hash": prompt_hash,
            "timestamp": time.time(),
        }))[:16]

        snapshot = ConversationSnapshot(
            snapshot_id=snapshot_id,
            messages=messages.copy(),
            tool_calls=tool_calls.copy(),
            tool_results=tool_results.copy(),
            prompt_hash=prompt_hash,
            timestamp=time.time(),
            metadata=metadata or {},
        )

        self._snapshots[snapshot_id] = snapshot
        return snapshot

    async def replay_with_prompt(
        self,
        snapshot_id: str,
        new_prompt_hash: str,
    ) -> Optional[EvaluationResult]:
        """Replay a conversation with a different prompt."""
        snapshot = self._snapshots.get(snapshot_id)
        new_prompt_node = self._store.get(new_prompt_hash)

        if not snapshot or not new_prompt_node:
            return None

        original_node = self._store.get(snapshot.prompt_hash)
        if not original_node:
            return None

        # Extract user messages for replay
        user_messages = [
            m for m in snapshot.messages if m.get("role") == "user"
        ]

        if not user_messages:
            return None

        # Replay with new prompt
        try:
            new_response = await self._inference_fn(
                new_prompt_node.content,
                user_messages,
            )
        except Exception as e:
            logger.error(f"Replay failed: {e}")
            return None

        # Get original response
        original_responses = [
            m.get("content", "") for m in snapshot.messages
            if m.get("role") == "assistant"
        ]
        original_response = " ".join(original_responses)

        # Compute metrics
        result = self._evaluate_responses(
            original_response=original_response,
            new_response=new_response,
            original_prompt_hash=snapshot.prompt_hash,
            new_prompt_hash=new_prompt_hash,
            snapshot_id=snapshot_id,
            snapshot=snapshot,
        )

        self._evaluations.append(result)

        # Update fitness scores based on result
        if result.fitness_delta > 0:
            self._store.update_fitness(new_prompt_hash, result.fitness_delta)
        elif result.fitness_delta < 0:
            self._store.update_fitness(snapshot.prompt_hash, -result.fitness_delta)

        return result

    def _evaluate_responses(
        self,
        original_response: str,
        new_response: str,
        original_prompt_hash: str,
        new_prompt_hash: str,
        snapshot_id: str,
        snapshot: ConversationSnapshot,  # noqa: ARG002 - reserved for advanced evaluation
    ) -> EvaluationResult:
        """Evaluate and compare two responses."""
        # Simple similarity (in production, use embeddings)
        similarity = self._text_similarity(original_response, new_response)

        # Tool call difference (would compare with snapshot.tool_calls in production)
        tool_call_diff = len(snapshot.tool_calls) if snapshot.tool_calls else 0

        # Simple heuristics for quality (in production, use model-based eval)
        coherence = min(1.0, len(new_response) / max(1, len(original_response)))
        task_completion = 1.0 if len(new_response) > 50 else 0.5
        efficiency = 1.0 - (abs(len(new_response) - len(original_response)) /
                          max(len(new_response), len(original_response), 1))

        # Fitness delta
        new_score = (coherence + task_completion + efficiency) / 3
        original_score = 0.5  # Baseline
        fitness_delta = new_score - original_score

        return EvaluationResult(
            original_prompt_hash=original_prompt_hash,
            new_prompt_hash=new_prompt_hash,
            snapshot_id=snapshot_id,
            response_similarity=similarity,
            tool_call_diff=tool_call_diff,
            outcome_match=similarity > 0.8,
            coherence_score=coherence,
            task_completion_score=task_completion,
            efficiency_score=efficiency,
            fitness_delta=fitness_delta,
            original_response=original_response[:500],
            new_response=new_response[:500],
            analysis=self._generate_analysis(similarity, fitness_delta),
        )

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple Jaccard similarity (use embeddings in production)."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _generate_analysis(self, similarity: float, fitness_delta: float) -> str:
        """Generate human-readable analysis."""
        if fitness_delta > 0.1:
            quality = "significantly better"
        elif fitness_delta > 0:
            quality = "slightly better"
        elif fitness_delta < -0.1:
            quality = "significantly worse"
        elif fitness_delta < 0:
            quality = "slightly worse"
        else:
            quality = "equivalent"

        if similarity > 0.8:
            divergence = "responses are very similar"
        elif similarity > 0.5:
            divergence = "responses show moderate differences"
        else:
            divergence = "responses diverge significantly"

        return f"New prompt performed {quality} ({divergence})"

    async def counterfactual_analysis(
        self,
        snapshot_id: str,
        prompt_hashes: List[str],
    ) -> List[EvaluationResult]:
        """Run counterfactual analysis with multiple prompts."""
        results = []

        for prompt_hash in prompt_hashes:
            result = await self.replay_with_prompt(snapshot_id, prompt_hash)
            if result:
                results.append(result)

        # Rank by fitness
        results.sort(key=lambda r: r.fitness_delta, reverse=True)
        return results

    def get_best_prompt_for_context(
        self,
        context_keywords: List[str],
    ) -> Optional[PromptNode]:
        """Find the best-performing prompt for a given context."""
        # Filter evaluations by context (simplified - would use embeddings)
        relevant_evals = []

        for eval_result in self._evaluations:
            snapshot = self._snapshots.get(eval_result.snapshot_id)
            if snapshot:
                snapshot_text = json.dumps(snapshot.messages).lower()
                if any(kw.lower() in snapshot_text for kw in context_keywords):
                    relevant_evals.append(eval_result)

        if not relevant_evals:
            return None

        # Find prompt with highest fitness in this context
        prompt_scores: Dict[str, List[float]] = {}
        for eval_result in relevant_evals:
            hash_ = eval_result.new_prompt_hash
            if hash_ not in prompt_scores:
                prompt_scores[hash_] = []
            prompt_scores[hash_].append(eval_result.fitness_delta)

        best_hash = max(
            prompt_scores.keys(),
            key=lambda h: sum(prompt_scores[h]) / len(prompt_scores[h])
        )

        return self._store.get(best_hash)


# =============================================================================
# Prompt Evolution Engine (Main Interface)
# =============================================================================


class EvolutionStrategy(str, Enum):
    """Strategy for evolving prompts."""
    CONSERVATIVE = "conservative"  # Small, incremental changes
    EXPLORATORY = "exploratory"    # Try diverse variations
    ADAPTIVE = "adaptive"          # Based on performance feedback
    GENETIC = "genetic"            # Crossover and mutation


@dataclass
class EvolutionConfig:
    """Configuration for prompt evolution."""
    strategy: EvolutionStrategy = EvolutionStrategy.ADAPTIVE
    max_forks: int = 10
    auto_merge_threshold: float = 0.8  # Fitness threshold for auto-merge
    snapshot_interval: int = 5  # Capture snapshot every N turns
    enable_retrodynamic: bool = True
    fitness_decay: float = 0.9  # Decay for ancestor fitness propagation


class PromptEvolutionEngine:
    """Main engine for prompt evolution with all capabilities."""

    def __init__(
        self,
        config: Optional[EvolutionConfig] = None,
        inference_fn: Optional[Callable] = None,
    ):
        self.config = config or EvolutionConfig()
        self._store = MerklePromptStore()
        self._forks = ForkGraph(self._store)
        self._retro = RetrodynamicEngine(
            self._store,
            inference_fn or self._dummy_inference,
        ) if self.config.enable_retrodynamic else None

        # Current state
        self._current_fork = "main"
        self._current_hash: Optional[str] = None
        self._turn_count = 0

    async def _dummy_inference(
        self,
        prompt: str,
        messages: List[Dict],
    ) -> str:
        """Placeholder inference function."""
        return f"[Replayed with prompt of length {len(prompt)}, {len(messages)} messages]"

    def initialize(self, genesis_prompt: str, author: str = "system") -> PromptNode:
        """Initialize with a genesis prompt."""
        node = self._store.store(
            content=genesis_prompt,
            parent_hash=None,
            author=author,
            reason="Genesis prompt",
            modification_type="genesis",
            fork_name="main",
        )

        # Create main fork
        self._forks._forks["main"] = Fork(
            name="main",
            head_hash=node.content_hash,
            created_at=time.time(),
            created_from=node.content_hash,
            description="Main prompt evolution branch",
        )

        self._current_hash = node.content_hash
        logger.info(f"Initialized evolution with genesis: {node.content_hash[:8]}")

        return node

    def get_current_prompt(self) -> Optional[str]:
        """Get the current prompt content."""
        if self._current_hash:
            node = self._store.get(self._current_hash)
            if node:
                return node.content
        return None

    def get_current_node(self) -> Optional[PromptNode]:
        """Get the current prompt node."""
        if self._current_hash:
            return self._store.get(self._current_hash)
        return None

    def modify(
        self,
        new_content: str,
        reason: str,
        modification_type: str = "modify",
        author: str = "agent",
    ) -> PromptNode:
        """Modify the current prompt."""
        node = self._store.store(
            content=new_content,
            parent_hash=self._current_hash,
            author=author,
            reason=reason,
            modification_type=modification_type,
            fork_name=self._current_fork,
        )

        self._current_hash = node.content_hash
        self._forks.advance_fork(self._current_fork, node.content_hash)

        logger.info(f"Modified prompt: {node.content_hash[:8]} ({reason})")
        return node

    def append(
        self,
        content: str,
        section: str,
        reason: str,
        author: str = "agent",
    ) -> PromptNode:
        """Append to the current prompt."""
        current = self.get_current_prompt() or ""
        addition = f"\n\n## [{section.upper()}]\n{content}"
        new_content = current + addition

        return self.modify(
            new_content=new_content,
            reason=f"Append to {section}: {reason}",
            modification_type="append",
            author=author,
        )

    def fork(
        self,
        fork_name: str,
        description: str = "",
        experimental_modification: Optional[str] = None,
    ) -> Fork:
        """Create a new fork for experimental changes."""
        if len(self._forks.list_forks()) >= self.config.max_forks:
            # Find lowest fitness fork to replace
            forks = self._forks.list_forks()
            worst = min(forks, key=lambda f: self._get_fork_fitness(f.name))
            if worst.name != "main":
                del self._forks._forks[worst.name]
                logger.info(f"Evicted low-fitness fork: {worst.name}")

        fork = self._forks.create_fork(
            name=fork_name,
            source_hash=self._current_hash or "",
            description=description,
        )

        # Apply experimental modification if provided
        if experimental_modification:
            self.switch_fork(fork_name)
            self.modify(
                new_content=experimental_modification,
                reason=f"Experimental: {description}",
                modification_type="fork",
            )

        return fork

    def switch_fork(self, fork_name: str):
        """Switch to a different fork."""
        fork = self._forks.get_fork(fork_name)
        if not fork:
            raise ValueError(f"Fork '{fork_name}' does not exist")

        self._current_fork = fork_name
        self._current_hash = fork.head_hash
        logger.info(f"Switched to fork: {fork_name}")

    def merge(
        self,
        source_fork: str,
        merge_strategy: str = "prefer_source",
    ) -> PromptNode:
        """Merge a fork into the current fork."""
        source_head = self._forks.get_head(source_fork)
        target_head = self._forks.get_head(self._current_fork)

        if not source_head or not target_head:
            raise ValueError("Both forks must have valid heads")

        # Simple merge strategies
        if merge_strategy == "prefer_source":
            merged_content = source_head.content
        elif merge_strategy == "prefer_target":
            merged_content = target_head.content
        elif merge_strategy == "concatenate":
            merged_content = f"{target_head.content}\n\n---\n\n{source_head.content}"
        else:
            merged_content = source_head.content

        node = self._forks.merge_forks(
            source_fork=source_fork,
            target_fork=self._current_fork,
            merged_content=merged_content,
            reason=f"Merge with strategy: {merge_strategy}",
        )

        self._current_hash = node.content_hash
        return node

    def revert(self, version: int = -1, reason: str = "") -> PromptNode:
        """Revert to a previous version."""
        lineage = self._store.get_lineage(self._current_hash or "")

        if version == -1:
            target_idx = max(0, len(lineage) - 2)
        elif version == 0:
            target_idx = 0
        else:
            target_idx = min(version, len(lineage) - 1)

        target_node = lineage[target_idx]

        return self.modify(
            new_content=target_node.content,
            reason=f"Revert to v{target_idx}: {reason}",
            modification_type="revert",
        )

    def _get_fork_fitness(self, fork_name: str) -> float:
        """Get aggregate fitness score for a fork."""
        diff = self._forks.get_fork_diff(fork_name)
        if not diff:
            return 0.0

        scores = [n.fitness_score for n in diff if n.evaluation_count > 0]
        return sum(scores) / len(scores) if scores else 0.0

    # -------------------------------------------------------------------------
    # Retrodynamic Operations
    # -------------------------------------------------------------------------

    def capture_conversation(
        self,
        messages: List[Dict[str, Any]],
        tool_calls: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ConversationSnapshot]:
        """Capture current conversation for later replay."""
        if not self._retro or not self._current_hash:
            return None

        self._turn_count += 1

        # Capture at configured interval
        if self._turn_count % self.config.snapshot_interval != 0:
            return None

        return self._retro.capture_snapshot(
            messages=messages,
            tool_calls=tool_calls,
            tool_results=tool_results,
            prompt_hash=self._current_hash,
            metadata=metadata,
        )

    async def evaluate_modification(
        self,
        new_prompt: str,
        recent_snapshots: int = 3,
    ) -> List[EvaluationResult]:
        """Evaluate a prompt modification against recent conversations."""
        if not self._retro:
            return []

        # Store the new prompt temporarily
        test_node = self._store.store(
            content=new_prompt,
            parent_hash=self._current_hash,
            author="agent",
            reason="Evaluation candidate",
            modification_type="test",
        )

        # Get recent snapshots
        snapshots = sorted(
            self._retro._snapshots.values(),
            key=lambda s: s.timestamp,
            reverse=True,
        )[:recent_snapshots]

        results = []
        for snapshot in snapshots:
            result = await self._retro.replay_with_prompt(
                snapshot.snapshot_id,
                test_node.content_hash,
            )
            if result:
                results.append(result)

        return results

    async def auto_evolve(
        self,
        feedback: str,
        performance_score: float,
    ) -> Optional[PromptNode]:
        """Automatically evolve the prompt based on feedback."""
        if self.config.strategy == EvolutionStrategy.CONSERVATIVE:
            # Small, targeted changes
            current = self.get_current_prompt() or ""
            evolved = f"{current}\n\n## [LEARNED]\n{feedback}"

        elif self.config.strategy == EvolutionStrategy.EXPLORATORY:
            # Create a fork for experimentation
            fork_name = f"explore_{int(time.time())}"
            self.fork(fork_name, f"Exploring: {feedback[:50]}")

            current = self.get_current_prompt() or ""
            evolved = f"[EXPERIMENTAL]\n{current}\n\n## [HYPOTHESIS]\n{feedback}"

        elif self.config.strategy == EvolutionStrategy.ADAPTIVE:
            # Modify based on performance score
            current = self.get_current_prompt() or ""
            if performance_score > 0.7:
                evolved = f"{current}\n\n## [REINFORCED]\n{feedback}"
            else:
                evolved = f"[REVISED]\n{current}\n\n## [CORRECTION]\n{feedback}"

        else:  # GENETIC
            # Would implement crossover with other forks
            evolved = self.get_current_prompt() or ""

        # Evaluate before committing
        if self._retro:
            results = await self.evaluate_modification(evolved)
            avg_fitness = (
                sum(r.fitness_delta for r in results) / len(results)
                if results else 0.0
            )

            if avg_fitness < -0.1:
                logger.info("Evolution rejected - negative fitness impact")
                return None

        return self.modify(
            new_content=evolved,
            reason=f"Auto-evolution: {feedback[:50]}",
            modification_type="evolve",
        )

    # -------------------------------------------------------------------------
    # Verification and Analysis
    # -------------------------------------------------------------------------

    def verify_integrity(self) -> Dict[str, Any]:
        """Verify the integrity of all stored prompts."""
        all_nodes = self._store.all_nodes()
        results = {
            "total_nodes": len(all_nodes),
            "nodes_checked": len(all_nodes),
            "verified": 0,
            "failed": 0,
            "failures": [],
            "valid": True,  # Will be set to False if any fail
        }

        for node in all_nodes:
            if self._store.verify_lineage(node.content_hash):
                results["verified"] += 1
            else:
                results["failed"] += 1
                results["failures"].append(node.content_hash)
                results["valid"] = False

        # If no nodes, still valid
        if not all_nodes:
            results["valid"] = True

        return results

    def get_evolution_stats(self) -> Dict[str, Any]:
        """Get statistics about prompt evolution."""
        nodes = self._store.all_nodes()
        forks = self._forks.list_forks()

        current_node = self.get_current_node()
        current_depth = current_node.depth if current_node is not None else 0

        return {
            "total_versions": len(nodes),
            "total_forks": len(forks),
            "active_forks": len([f for f in forks if f.is_active]),
            "current_fork": self._current_fork,
            "current_depth": current_depth,
            "avg_fitness": (
                sum(n.fitness_score for n in nodes if n.evaluation_count > 0) /
                max(1, len([n for n in nodes if n.evaluation_count > 0]))
            ),
            "modifications_by_type": self._count_by_type(nodes),
            "modifications_by_author": self._count_by_author(nodes),
        }

    def _count_by_type(self, nodes: List[PromptNode]) -> Dict[str, int]:
        """Count nodes by modification type."""
        counts: Dict[str, int] = {}
        for node in nodes:
            t = node.modification_type
            counts[t] = counts.get(t, 0) + 1
        return counts

    def _count_by_author(self, nodes: List[PromptNode]) -> Dict[str, int]:
        """Count nodes by author."""
        counts: Dict[str, int] = {}
        for node in nodes:
            a = node.author
            counts[a] = counts.get(a, 0) + 1
        return counts

    def export_lineage(self, format: str = "json") -> str:
        """Export the full lineage for visualization."""
        if not self._current_hash:
            return "{}"

        lineage = self._store.get_lineage(self._current_hash)

        if format == "json":
            return json.dumps([n.to_dict() for n in lineage], indent=2)
        elif format == "mermaid":
            lines = ["graph TD"]
            for node in lineage:
                label = f"{node.content_hash[:8]}<br/>{node.modification_type}"
                lines.append(f'    {node.content_hash[:8]}["{label}"]')
                if node.parent_hash:
                    lines.append(
                        f"    {node.parent_hash[:8]} --> {node.content_hash[:8]}"
                    )
            return "\n".join(lines)
        else:
            return str([n.content_hash[:8] for n in lineage])

    # -------------------------------------------------------------------------
    # Tool Definitions for Agent Use
    # -------------------------------------------------------------------------

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible tool definitions for agent use."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "evolve_prompt",
                    "description": (
                        "Evolve your system prompt with full version control. "
                        "Supports modification, forking, merging, and revert. "
                        "All changes are tracked with Merkle proofs."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["modify", "append", "fork", "merge", "revert", "switch_fork"],
                                "description": "Evolution action to take",
                            },
                            "content": {
                                "type": "string",
                                "description": "New content (for modify/append)",
                            },
                            "reason": {
                                "type": "string",
                                "description": "Why you're making this change",
                            },
                            "fork_name": {
                                "type": "string",
                                "description": "Name for fork operations",
                            },
                            "section": {
                                "type": "string",
                                "enum": ["capabilities", "constraints", "context", "personality", "learned"],
                                "description": "Section for append operations",
                            },
                            "merge_strategy": {
                                "type": "string",
                                "enum": ["prefer_source", "prefer_target", "concatenate"],
                                "description": "Strategy for merge operations",
                            },
                            "version": {
                                "type": "integer",
                                "description": "Version number for revert (0=original, -1=previous)",
                            },
                        },
                        "required": ["action", "reason"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_evolution",
                    "description": (
                        "Analyze prompt evolution history, verify integrity, "
                        "and get recommendations for improvement."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "analysis_type": {
                                "type": "string",
                                "enum": ["stats", "lineage", "integrity", "fitness", "compare_forks"],
                                "description": "Type of analysis to perform",
                            },
                            "fork_name": {
                                "type": "string",
                                "description": "Fork to analyze (for fork-specific analysis)",
                            },
                            "include_content": {
                                "type": "boolean",
                                "description": "Whether to include full prompt content",
                            },
                        },
                        "required": ["analysis_type"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "retrodynamic_eval",
                    "description": (
                        "Re-evaluate past conversations with different prompts. "
                        "Enables counterfactual analysis and fitness scoring."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "candidate_prompt": {
                                "type": "string",
                                "description": "New prompt to evaluate",
                            },
                            "num_snapshots": {
                                "type": "integer",
                                "description": "Number of recent snapshots to replay",
                            },
                            "compare_forks": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Fork names to compare",
                            },
                        },
                    },
                },
            },
        ]
