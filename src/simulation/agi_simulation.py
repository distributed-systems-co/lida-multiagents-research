"""
AGI Simulation System

A multi-agent architecture that simulates Artificial General Intelligence
by coordinating multiple specialized LLM agents, inspired by:
- Marvin Minsky's "Society of Mind"
- Global Workspace Theory (Bernard Baars)
- LIDA Cognitive Architecture
- ACT-R and SOAR architectures

This system implements:
1. Specialized cognitive modules (reasoning, memory, planning, creativity, etc.)
2. Central executive/orchestrator (Global Workspace)
3. Working memory and episodic/semantic long-term memory
4. Meta-cognition and self-reflection
5. Goal management, task decomposition, and planning
6. Learning through experience consolidation

The goal is NOT to create actual AGI, but to simulate AGI-like behavior
through the coordination of multiple narrow AI systems.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Generic
import hashlib

# Try to import LLM clients
try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# =============================================================================
# Core Types and Enums
# =============================================================================

class CognitiveModuleType(Enum):
    """Types of cognitive modules in the AGI system."""
    REASONING = "reasoning"           # Logical deduction and inference
    PLANNING = "planning"             # Goal decomposition and action sequences
    MEMORY_RETRIEVAL = "memory"       # Accessing stored knowledge
    CREATIVITY = "creativity"         # Novel idea generation
    LANGUAGE = "language"             # Natural language processing
    SOCIAL = "social"                 # Theory of mind, social reasoning
    METACOGNITION = "metacognition"   # Self-reflection and monitoring
    PERCEPTION = "perception"         # Interpreting sensory input
    MOTOR = "motor"                   # Action execution planning
    EMOTION = "emotion"               # Affective processing
    ATTENTION = "attention"           # Focus and salience detection
    LEARNING = "learning"             # Pattern extraction and skill acquisition


class ThoughtType(Enum):
    """Types of thoughts that can occur."""
    OBSERVATION = "observation"
    INFERENCE = "inference"
    HYPOTHESIS = "hypothesis"
    QUESTION = "question"
    PLAN = "plan"
    MEMORY = "memory"
    EMOTION = "emotion"
    INTENTION = "intention"
    BELIEF = "belief"
    DESIRE = "desire"
    REFLECTION = "reflection"
    INSIGHT = "insight"


class GoalStatus(Enum):
    """Status of goals."""
    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


class MemoryType(Enum):
    """Types of memory."""
    EPISODIC = "episodic"      # Specific experiences
    SEMANTIC = "semantic"      # Facts and concepts
    PROCEDURAL = "procedural"  # How to do things
    WORKING = "working"        # Current active thoughts


class AttentionLevel(Enum):
    """Levels of attention/activation."""
    UNCONSCIOUS = 0
    SUBLIMINAL = 1
    PERIPHERAL = 2
    CONSCIOUS = 3
    FOCUSED = 4


# =============================================================================
# Memory Structures
# =============================================================================

@dataclass
class MemoryItem:
    """A single item in memory."""
    id: str
    content: str
    memory_type: MemoryType
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    importance: float = 0.5  # 0 to 1
    emotional_valence: float = 0.0  # -1 to 1
    activation: float = 0.5  # Current activation level
    associations: Set[str] = field(default_factory=set)  # IDs of related memories
    context: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None  # For semantic similarity

    def decay(self, rate: float = 0.01):
        """Apply memory decay."""
        self.activation = max(0.0, self.activation - rate)

    def activate(self, amount: float = 0.2):
        """Activate this memory."""
        self.activation = min(1.0, self.activation + amount)
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class WorkingMemory:
    """
    Working memory - limited capacity, high activation.
    Based on Baddeley's model with central executive,
    phonological loop, visuospatial sketchpad, and episodic buffer.
    """
    capacity: int = 7  # Miller's magic number
    items: List[MemoryItem] = field(default_factory=list)
    focus: Optional[str] = None  # ID of item in focus

    # Subsystems
    phonological_loop: List[str] = field(default_factory=list)  # Verbal info
    visuospatial_sketchpad: List[str] = field(default_factory=list)  # Spatial info
    episodic_buffer: List[str] = field(default_factory=list)  # Integrated info

    def add(self, item: MemoryItem) -> Optional[MemoryItem]:
        """Add item to working memory, possibly displacing oldest."""
        displaced = None
        if len(self.items) >= self.capacity:
            # Remove least activated item
            self.items.sort(key=lambda x: x.activation)
            displaced = self.items.pop(0)

        item.activation = 1.0  # Full activation in WM
        self.items.append(item)
        return displaced

    def get_contents(self) -> List[str]:
        """Get current working memory contents as strings."""
        return [item.content for item in self.items]

    def decay_all(self, rate: float = 0.05):
        """Decay all items."""
        for item in self.items:
            item.decay(rate)
        # Remove items with very low activation
        self.items = [i for i in self.items if i.activation > 0.1]


@dataclass
class LongTermMemory:
    """
    Long-term memory system with episodic and semantic stores.
    Uses spreading activation for retrieval.
    """
    episodic: Dict[str, MemoryItem] = field(default_factory=dict)
    semantic: Dict[str, MemoryItem] = field(default_factory=dict)
    procedural: Dict[str, MemoryItem] = field(default_factory=dict)

    # Index for faster retrieval
    keyword_index: Dict[str, Set[str]] = field(default_factory=dict)

    def store(self, item: MemoryItem):
        """Store an item in long-term memory."""
        store = {
            MemoryType.EPISODIC: self.episodic,
            MemoryType.SEMANTIC: self.semantic,
            MemoryType.PROCEDURAL: self.procedural,
        }.get(item.memory_type, self.semantic)

        store[item.id] = item

        # Index keywords
        for word in item.content.lower().split():
            if len(word) > 3:  # Skip short words
                if word not in self.keyword_index:
                    self.keyword_index[word] = set()
                self.keyword_index[word].add(item.id)

    def retrieve(self, query: str, limit: int = 5) -> List[MemoryItem]:
        """Retrieve relevant memories using spreading activation."""
        # Find candidate IDs from keyword index
        candidate_ids: Set[str] = set()
        for word in query.lower().split():
            if word in self.keyword_index:
                candidate_ids.update(self.keyword_index[word])

        # Score candidates
        candidates = []
        all_stores = {**self.episodic, **self.semantic, **self.procedural}

        for mid in candidate_ids:
            if mid in all_stores:
                item = all_stores[mid]
                # Score based on keyword overlap and recency
                query_words = set(query.lower().split())
                item_words = set(item.content.lower().split())
                overlap = len(query_words & item_words)

                recency = 1.0 / (1.0 + (datetime.now() - item.last_accessed).total_seconds() / 3600)
                score = overlap * 0.5 + item.importance * 0.3 + recency * 0.2

                candidates.append((score, item))

        # Sort by score and return top results
        candidates.sort(key=lambda x: x[0], reverse=True)
        results = [item for _, item in candidates[:limit]]

        # Activate retrieved memories
        for item in results:
            item.activate()

        return results

    def consolidate(self, working_memory: WorkingMemory):
        """Consolidate working memory items to long-term memory."""
        for item in working_memory.items:
            if item.importance > 0.3:  # Only consolidate important items
                # Create a copy for LTM
                ltm_item = MemoryItem(
                    id=item.id,
                    content=item.content,
                    memory_type=item.memory_type,
                    importance=item.importance,
                    emotional_valence=item.emotional_valence,
                    context=item.context.copy(),
                )
                self.store(ltm_item)


# =============================================================================
# Thought and Goal Structures
# =============================================================================

@dataclass
class Thought:
    """A unit of cognition in the system."""
    id: str
    content: str
    thought_type: ThoughtType
    source_module: CognitiveModuleType
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 0.5  # 0 to 1
    activation: float = 0.5  # 0 to 1
    attention_level: AttentionLevel = AttentionLevel.PERIPHERAL
    associations: List[str] = field(default_factory=list)  # Related thought IDs
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)

    def to_memory_item(self) -> MemoryItem:
        """Convert thought to memory item."""
        return MemoryItem(
            id=self.id,
            content=self.content,
            memory_type=MemoryType.EPISODIC,
            importance=self.confidence * self.activation,
            context={"thought_type": self.thought_type.value, "source": self.source_module.value}
        )


@dataclass
class Goal:
    """A goal in the goal hierarchy."""
    id: str
    description: str
    status: GoalStatus = GoalStatus.PENDING
    priority: float = 0.5  # 0 to 1
    parent_goal: Optional[str] = None
    subgoals: List[str] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    plan: List[str] = field(default_factory=list)  # Sequence of action IDs
    progress: float = 0.0  # 0 to 1
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    attempts: int = 0
    max_attempts: int = 3


@dataclass
class Action:
    """An action that can be taken."""
    id: str
    description: str
    module: CognitiveModuleType
    parameters: Dict[str, Any] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    cost: float = 0.1  # Resource cost
    duration: float = 1.0  # Time units


# =============================================================================
# Cognitive Modules
# =============================================================================

class CognitiveModule(ABC):
    """
    Abstract base class for cognitive modules.
    Each module processes information in its domain.
    """

    def __init__(
        self,
        module_type: CognitiveModuleType,
        llm_client: Optional[Any] = None,
        model: str = "claude-sonnet-4-20250514"
    ):
        self.module_type = module_type
        self.llm_client = llm_client
        self.model = model
        self.activation: float = 0.5
        self.last_output: Optional[Thought] = None
        self.processing_history: List[Dict] = []

    @abstractmethod
    async def process(
        self,
        input_data: Dict[str, Any],
        working_memory: WorkingMemory,
        context: Dict[str, Any]
    ) -> List[Thought]:
        """Process input and generate thoughts."""
        pass

    def _create_thought(
        self,
        content: str,
        thought_type: ThoughtType,
        confidence: float = 0.5
    ) -> Thought:
        """Helper to create thoughts."""
        return Thought(
            id=f"thought_{uuid.uuid4().hex[:8]}",
            content=content,
            thought_type=thought_type,
            source_module=self.module_type,
            confidence=confidence,
            activation=self.activation
        )

    async def _llm_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 500
    ) -> str:
        """Generate response using LLM."""
        if self.llm_client is None:
            return f"[{self.module_type.value} module output for: {user_prompt[:50]}...]"

        try:
            if HAS_ANTHROPIC and hasattr(self.llm_client, 'messages'):
                response = self.llm_client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                if hasattr(response.content[0], 'text'):
                    return response.content[0].text
                return str(response.content[0])
            elif HAS_OPENAI:
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return response.choices[0].message.content or ""
        except Exception as e:
            return f"[Error in {self.module_type.value}: {e}]"

        return f"[{self.module_type.value} fallback output]"


class ReasoningModule(CognitiveModule):
    """
    Logical reasoning, inference, and deduction.
    Handles syllogisms, causal reasoning, analogical reasoning.
    """

    def __init__(self, llm_client: Optional[Any] = None, model: str = "claude-sonnet-4-20250514"):
        super().__init__(CognitiveModuleType.REASONING, llm_client, model)

    async def process(
        self,
        input_data: Dict[str, Any],
        working_memory: WorkingMemory,
        context: Dict[str, Any]
    ) -> List[Thought]:
        thoughts = []

        query = input_data.get("query", "")
        premises = input_data.get("premises", [])

        # Add working memory contents as context
        wm_contents = working_memory.get_contents()

        system_prompt = """You are the REASONING module of an AGI system.
Your role is to perform logical inference, identify implications, detect contradictions,
and draw valid conclusions from premises.

Output your reasoning in this format:
INFERENCE: [Your logical inference]
CONFIDENCE: [0.0-1.0]
REASONING_TYPE: [deductive/inductive/abductive/analogical]
SUPPORTING_PREMISES: [List relevant premises]
"""

        user_prompt = f"""Given these premises and context, perform logical reasoning:

PREMISES:
{chr(10).join(f'- {p}' for p in premises)}

WORKING MEMORY:
{chr(10).join(f'- {c}' for c in wm_contents[:5])}

QUERY: {query}

Provide logical inferences and conclusions."""

        response = await self._llm_generate(system_prompt, user_prompt)

        # Parse response into thoughts
        if "INFERENCE:" in response:
            inference = response.split("INFERENCE:")[1].split("\n")[0].strip()
            confidence = 0.7
            if "CONFIDENCE:" in response:
                try:
                    confidence = float(response.split("CONFIDENCE:")[1].split("\n")[0].strip())
                except ValueError:
                    pass

            thoughts.append(self._create_thought(
                content=inference,
                thought_type=ThoughtType.INFERENCE,
                confidence=confidence
            ))
        else:
            thoughts.append(self._create_thought(
                content=response,
                thought_type=ThoughtType.INFERENCE,
                confidence=0.5
            ))

        return thoughts


class PlanningModule(CognitiveModule):
    """
    Goal decomposition, action sequencing, and plan generation.
    Uses hierarchical task network approach.
    """

    def __init__(self, llm_client: Optional[Any] = None, model: str = "claude-sonnet-4-20250514"):
        super().__init__(CognitiveModuleType.PLANNING, llm_client, model)

    async def process(
        self,
        input_data: Dict[str, Any],
        working_memory: WorkingMemory,
        context: Dict[str, Any]
    ) -> List[Thought]:
        thoughts = []

        goal = input_data.get("goal", "")
        current_state = input_data.get("current_state", {})
        available_actions = input_data.get("available_actions", [])
        constraints = input_data.get("constraints", [])

        system_prompt = """You are the PLANNING module of an AGI system.
Your role is to:
1. Decompose high-level goals into subgoals
2. Identify required actions and their sequence
3. Consider constraints and preconditions
4. Generate executable plans

Output format:
PLAN_STEP_1: [action]
PLAN_STEP_2: [action]
...
SUBGOALS: [list of identified subgoals]
PRECONDITIONS: [required conditions]
ESTIMATED_SUCCESS: [0.0-1.0]
"""

        user_prompt = f"""Create a plan to achieve this goal:

GOAL: {goal}

CURRENT STATE:
{json.dumps(current_state, indent=2)}

AVAILABLE ACTIONS:
{chr(10).join(f'- {a}' for a in available_actions)}

CONSTRAINTS:
{chr(10).join(f'- {c}' for c in constraints)}

Generate a step-by-step plan."""

        response = await self._llm_generate(system_prompt, user_prompt, max_tokens=800)

        thoughts.append(self._create_thought(
            content=f"Plan for '{goal}': {response}",
            thought_type=ThoughtType.PLAN,
            confidence=0.6
        ))

        return thoughts


class CreativityModule(CognitiveModule):
    """
    Novel idea generation, conceptual blending, and creative synthesis.
    Uses bisociation and combinatorial creativity.
    """

    def __init__(self, llm_client: Optional[Any] = None, model: str = "claude-sonnet-4-20250514"):
        super().__init__(CognitiveModuleType.CREATIVITY, llm_client, model)

    async def process(
        self,
        input_data: Dict[str, Any],
        working_memory: WorkingMemory,
        context: Dict[str, Any]
    ) -> List[Thought]:
        thoughts = []

        problem = input_data.get("problem", "")
        concepts = input_data.get("concepts", [])
        constraints = input_data.get("constraints", [])

        system_prompt = """You are the CREATIVITY module of an AGI system.
Your role is to generate novel ideas through:
1. Conceptual blending - combining distant concepts
2. Analogical transfer - applying patterns from one domain to another
3. Constraint relaxation - questioning assumptions
4. Divergent thinking - exploring multiple possibilities

Output format:
NOVEL_IDEA: [your creative idea]
INSPIRATION_SOURCES: [what concepts/domains inspired this]
NOVELTY_SCORE: [0.0-1.0]
FEASIBILITY_SCORE: [0.0-1.0]
"""

        wm_contents = working_memory.get_contents()

        user_prompt = f"""Generate creative solutions for:

PROBLEM: {problem}

AVAILABLE CONCEPTS:
{chr(10).join(f'- {c}' for c in concepts)}

WORKING MEMORY (potential inspiration):
{chr(10).join(f'- {w}' for w in wm_contents[:5])}

CONSTRAINTS TO CONSIDER (or challenge):
{chr(10).join(f'- {c}' for c in constraints)}

Generate at least 2 novel ideas."""

        response = await self._llm_generate(system_prompt, user_prompt, max_tokens=600)

        thoughts.append(self._create_thought(
            content=response,
            thought_type=ThoughtType.INSIGHT,
            confidence=0.5
        ))

        return thoughts


class MetacognitionModule(CognitiveModule):
    """
    Self-reflection, confidence assessment, and cognitive monitoring.
    Monitors and regulates other cognitive processes.
    """

    def __init__(self, llm_client: Optional[Any] = None, model: str = "claude-sonnet-4-20250514"):
        super().__init__(CognitiveModuleType.METACOGNITION, llm_client, model)

    async def process(
        self,
        input_data: Dict[str, Any],
        working_memory: WorkingMemory,
        context: Dict[str, Any]
    ) -> List[Thought]:
        thoughts = []

        recent_thoughts = input_data.get("recent_thoughts", [])
        current_goals = input_data.get("current_goals", [])
        performance_metrics = input_data.get("performance", {})

        system_prompt = """You are the METACOGNITION module of an AGI system.
Your role is to:
1. Monitor cognitive processes and detect errors
2. Assess confidence in beliefs and conclusions
3. Identify knowledge gaps
4. Suggest strategy adjustments
5. Reflect on reasoning quality

Output format:
SELF_ASSESSMENT: [evaluation of recent cognitive performance]
CONFIDENCE_CALIBRATION: [are confidence levels appropriate?]
KNOWLEDGE_GAPS: [what information is missing?]
STRATEGY_SUGGESTION: [how should processing be adjusted?]
COGNITIVE_LOAD: [current load assessment]
"""

        user_prompt = f"""Perform metacognitive analysis:

RECENT THOUGHTS:
{chr(10).join(f'- {t}' for t in recent_thoughts[-5:])}

CURRENT GOALS:
{chr(10).join(f'- {g}' for g in current_goals)}

PERFORMANCE METRICS:
{json.dumps(performance_metrics, indent=2)}

Reflect on cognitive performance and suggest adjustments."""

        response = await self._llm_generate(system_prompt, user_prompt, max_tokens=500)

        thoughts.append(self._create_thought(
            content=response,
            thought_type=ThoughtType.REFLECTION,
            confidence=0.7
        ))

        return thoughts


class SocialCognitionModule(CognitiveModule):
    """
    Theory of mind, social reasoning, and understanding others' mental states.
    """

    def __init__(self, llm_client: Optional[Any] = None, model: str = "claude-sonnet-4-20250514"):
        super().__init__(CognitiveModuleType.SOCIAL, llm_client, model)

    async def process(
        self,
        input_data: Dict[str, Any],
        working_memory: WorkingMemory,
        context: Dict[str, Any]
    ) -> List[Thought]:
        thoughts = []

        social_context = input_data.get("social_context", "")
        agents = input_data.get("agents", [])
        interactions = input_data.get("interactions", [])

        system_prompt = """You are the SOCIAL COGNITION module of an AGI system.
Your role is to:
1. Model other agents' beliefs, desires, and intentions (Theory of Mind)
2. Predict social behavior and reactions
3. Understand social norms and dynamics
4. Identify emotional states in others
5. Reason about cooperation and competition

Output format:
AGENT_MODEL: [inferred mental states of relevant agents]
PREDICTED_BEHAVIOR: [what agents are likely to do]
SOCIAL_DYNAMICS: [analysis of social situation]
RECOMMENDED_APPROACH: [how to interact effectively]
"""

        user_prompt = f"""Analyze this social situation:

CONTEXT: {social_context}

AGENTS INVOLVED:
{chr(10).join(f'- {a}' for a in agents)}

RECENT INTERACTIONS:
{chr(10).join(f'- {i}' for i in interactions[-5:])}

Model the mental states and predict behavior."""

        response = await self._llm_generate(system_prompt, user_prompt, max_tokens=500)

        thoughts.append(self._create_thought(
            content=response,
            thought_type=ThoughtType.BELIEF,
            confidence=0.6
        ))

        return thoughts


class MemoryRetrievalModule(CognitiveModule):
    """
    Memory retrieval using spreading activation and context-based recall.
    """

    def __init__(self, llm_client: Optional[Any] = None, model: str = "claude-sonnet-4-20250514"):
        super().__init__(CognitiveModuleType.MEMORY_RETRIEVAL, llm_client, model)
        self.ltm: Optional[LongTermMemory] = None

    def set_ltm(self, ltm: LongTermMemory):
        """Set reference to long-term memory."""
        self.ltm = ltm

    async def process(
        self,
        input_data: Dict[str, Any],
        working_memory: WorkingMemory,
        context: Dict[str, Any]
    ) -> List[Thought]:
        thoughts = []

        query = input_data.get("query", "")
        retrieval_type = input_data.get("type", "semantic")

        if self.ltm:
            memories = self.ltm.retrieve(query, limit=5)

            for mem in memories:
                thoughts.append(self._create_thought(
                    content=f"Recalled: {mem.content}",
                    thought_type=ThoughtType.MEMORY,
                    confidence=mem.importance
                ))

        if not thoughts:
            thoughts.append(self._create_thought(
                content=f"No relevant memories found for: {query}",
                thought_type=ThoughtType.MEMORY,
                confidence=0.3
            ))

        return thoughts


class LearningModule(CognitiveModule):
    """
    Pattern extraction, generalization, and skill acquisition.
    """

    def __init__(self, llm_client: Optional[Any] = None, model: str = "claude-sonnet-4-20250514"):
        super().__init__(CognitiveModuleType.LEARNING, llm_client, model)

    async def process(
        self,
        input_data: Dict[str, Any],
        working_memory: WorkingMemory,
        context: Dict[str, Any]
    ) -> List[Thought]:
        thoughts = []

        experiences = input_data.get("experiences", [])
        feedback = input_data.get("feedback", "")
        domain = input_data.get("domain", "general")

        system_prompt = """You are the LEARNING module of an AGI system.
Your role is to:
1. Extract patterns from experiences
2. Generalize rules and principles
3. Update beliefs based on evidence
4. Identify and correct errors
5. Transfer knowledge between domains

Output format:
PATTERN_IDENTIFIED: [extracted pattern]
GENERALIZATION: [broader principle derived]
CONFIDENCE_UPDATE: [how beliefs should change]
SKILL_ACQUIRED: [new capability learned]
"""

        user_prompt = f"""Learn from these experiences:

EXPERIENCES:
{chr(10).join(f'- {e}' for e in experiences[-10:])}

FEEDBACK RECEIVED: {feedback}

DOMAIN: {domain}

Extract patterns and generalizations."""

        response = await self._llm_generate(system_prompt, user_prompt, max_tokens=500)

        thoughts.append(self._create_thought(
            content=response,
            thought_type=ThoughtType.INSIGHT,
            confidence=0.6
        ))

        return thoughts


# =============================================================================
# Global Workspace (Central Executive)
# =============================================================================

@dataclass
class GlobalWorkspaceState:
    """State of the global workspace."""
    broadcast_content: Optional[Thought] = None
    competing_thoughts: List[Thought] = field(default_factory=list)
    coalition_strength: Dict[str, float] = field(default_factory=dict)
    attention_focus: Optional[str] = None
    cognitive_cycle: int = 0


class GlobalWorkspace:
    """
    Central executive implementing Global Workspace Theory.

    The Global Workspace acts as a "cognitive blackboard" where:
    1. Specialized modules compete for attention
    2. Winning content is "broadcast" to all modules
    3. Unconscious processes become conscious through broadcast
    4. Integration of information across modules occurs
    """

    def __init__(self):
        self.state = GlobalWorkspaceState()
        self.modules: Dict[CognitiveModuleType, CognitiveModule] = {}
        self.working_memory = WorkingMemory()
        self.long_term_memory = LongTermMemory()

        # Coalition formation
        self.thought_buffer: List[Thought] = []
        self.broadcast_history: List[Thought] = []

        # Attention
        self.attention_threshold: float = 0.6

        # Callbacks
        self.on_broadcast: Optional[Callable[[Thought], None]] = None
        self.on_cycle_complete: Optional[Callable[[int], None]] = None

    def register_module(self, module: CognitiveModule):
        """Register a cognitive module."""
        self.modules[module.module_type] = module
        if module.module_type == CognitiveModuleType.MEMORY_RETRIEVAL:
            module.set_ltm(self.long_term_memory)

    async def cognitive_cycle(self, input_data: Dict[str, Any] = None) -> Thought:
        """
        Run one cognitive cycle:
        1. Modules process input and generate thoughts
        2. Thoughts compete for access to global workspace
        3. Winning thought is broadcast to all modules
        4. Working memory is updated
        """
        self.state.cognitive_cycle += 1

        # Phase 1: Parallel processing by modules
        all_thoughts = []
        context = {"cycle": self.state.cognitive_cycle, "broadcast_history": self.broadcast_history[-5:]}

        tasks = []
        for module_type, module in self.modules.items():
            module_input = input_data.get(module_type.value, {}) if input_data else {}
            tasks.append(module.process(module_input, self.working_memory, context))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_thoughts.extend(result)

        # Phase 2: Competition for workspace
        self.thought_buffer.extend(all_thoughts)

        # Form coalitions (thoughts that support each other)
        coalition_scores = {}
        for thought in self.thought_buffer:
            # Base score from activation and confidence
            score = thought.activation * thought.confidence

            # Bonus for recency
            recency_bonus = 0.1 if (datetime.now() - thought.timestamp).seconds < 5 else 0
            score += recency_bonus

            # Bonus for goal relevance (simplified)
            coalition_scores[thought.id] = score

        # Phase 3: Broadcast winning thought
        if coalition_scores:
            winner_id = max(coalition_scores, key=coalition_scores.get)
            winner = next(t for t in self.thought_buffer if t.id == winner_id)
            winner.attention_level = AttentionLevel.FOCUSED

            self.state.broadcast_content = winner
            self.broadcast_history.append(winner)

            # Update working memory
            memory_item = winner.to_memory_item()
            displaced = self.working_memory.add(memory_item)

            # Consolidate displaced item to LTM if important
            if displaced and displaced.importance > 0.4:
                self.long_term_memory.store(displaced)

            # Notify
            if self.on_broadcast:
                self.on_broadcast(winner)

            # Clean up
            self.thought_buffer = [t for t in self.thought_buffer if t.id != winner_id]

            # Decay non-winning thoughts
            for thought in self.thought_buffer:
                thought.activation *= 0.8
            self.thought_buffer = [t for t in self.thought_buffer if t.activation > 0.1]

            if self.on_cycle_complete:
                self.on_cycle_complete(self.state.cognitive_cycle)

            return winner

        # No thoughts to broadcast
        return Thought(
            id=f"empty_{self.state.cognitive_cycle}",
            content="[No significant cognitive content]",
            thought_type=ThoughtType.OBSERVATION,
            source_module=CognitiveModuleType.ATTENTION,
            confidence=0.1
        )


# =============================================================================
# AGI System (Main Class)
# =============================================================================

class AGISystem:
    """
    Main AGI simulation system that coordinates all components.

    This is NOT actual AGI - it's a simulation that coordinates
    multiple narrow AI systems to approximate general intelligence.
    """

    def __init__(
        self,
        name: str = "ARIA",  # Artificial Reasoning Intelligence Architecture
        llm_provider: str = "anthropic",
        llm_model: str = "claude-sonnet-4-20250514",
        use_llm: bool = True
    ):
        self.name = name
        self.llm_provider = llm_provider
        self.llm_model = llm_model

        # Initialize LLM client
        self.llm_client = None
        if use_llm:
            if llm_provider == "anthropic" and HAS_ANTHROPIC:
                self.llm_client = Anthropic()
            elif llm_provider == "openai" and HAS_OPENAI:
                self.llm_client = OpenAI()

        # Initialize Global Workspace
        self.workspace = GlobalWorkspace()

        # Initialize cognitive modules
        self._init_modules()

        # Goal management
        self.goals: Dict[str, Goal] = {}
        self.active_goal: Optional[str] = None

        # Interaction history
        self.interaction_history: List[Dict[str, Any]] = []

        # Performance metrics
        self.metrics = {
            "cycles": 0,
            "goals_completed": 0,
            "goals_failed": 0,
            "thoughts_generated": 0,
            "memories_consolidated": 0,
        }

        # Callbacks
        self.on_thought: Optional[Callable[[Thought], None]] = None
        self.on_response: Optional[Callable[[str], None]] = None

        # Wire up workspace callbacks
        self.workspace.on_broadcast = self._on_broadcast

    def _init_modules(self):
        """Initialize all cognitive modules."""
        modules = [
            ReasoningModule(self.llm_client, self.llm_model),
            PlanningModule(self.llm_client, self.llm_model),
            CreativityModule(self.llm_client, self.llm_model),
            MetacognitionModule(self.llm_client, self.llm_model),
            SocialCognitionModule(self.llm_client, self.llm_model),
            MemoryRetrievalModule(self.llm_client, self.llm_model),
            LearningModule(self.llm_client, self.llm_model),
        ]

        for module in modules:
            self.workspace.register_module(module)

    def _on_broadcast(self, thought: Thought):
        """Handle broadcast events."""
        self.metrics["thoughts_generated"] += 1
        if self.on_thought:
            self.on_thought(thought)

    async def process_input(self, user_input: str) -> str:
        """
        Main entry point for processing user input.
        Orchestrates multiple cognitive cycles to generate a response.
        """
        # Record interaction
        self.interaction_history.append({
            "type": "user_input",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })

        # Create initial observation
        observation = Thought(
            id=f"obs_{uuid.uuid4().hex[:8]}",
            content=f"User said: {user_input}",
            thought_type=ThoughtType.OBSERVATION,
            source_module=CognitiveModuleType.PERCEPTION,
            confidence=1.0,
            activation=1.0
        )
        self.workspace.thought_buffer.append(observation)

        # Run multiple cognitive cycles
        broadcast_thoughts = []
        for cycle in range(5):  # Run 5 cycles for each input
            # Prepare input for different modules
            module_inputs = {
                "reasoning": {
                    "query": user_input,
                    "premises": [t.content for t in broadcast_thoughts[-3:]]
                },
                "planning": {
                    "goal": user_input if "?" not in user_input else "",
                    "current_state": {"input_processed": True, "cycle": cycle}
                },
                "creativity": {
                    "problem": user_input,
                    "concepts": [t.content for t in broadcast_thoughts]
                },
                "metacognition": {
                    "recent_thoughts": [t.content for t in broadcast_thoughts],
                    "current_goals": list(self.goals.keys()),
                    "performance": self.metrics
                },
                "social": {
                    "social_context": user_input,
                    "agents": ["user"],
                    "interactions": [h["content"] for h in self.interaction_history[-5:]]
                },
                "memory": {
                    "query": user_input,
                    "type": "semantic"
                },
                "learning": {
                    "experiences": [h["content"] for h in self.interaction_history[-10:]],
                    "feedback": "",
                    "domain": "conversation"
                }
            }

            thought = await self.workspace.cognitive_cycle(module_inputs)
            broadcast_thoughts.append(thought)
            self.metrics["cycles"] += 1

        # Generate response by synthesizing broadcast thoughts
        response = await self._synthesize_response(user_input, broadcast_thoughts)

        # Record response
        self.interaction_history.append({
            "type": "system_response",
            "content": response,
            "thoughts": [t.content for t in broadcast_thoughts],
            "timestamp": datetime.now().isoformat()
        })

        if self.on_response:
            self.on_response(response)

        return response

    async def _synthesize_response(self, query: str, thoughts: List[Thought]) -> str:
        """Synthesize final response from cognitive processing."""

        if self.llm_client is None:
            # Fallback: combine thoughts
            return f"{self.name} processed your input. Key thoughts:\n" + \
                   "\n".join(f"- {t.content[:100]}..." for t in thoughts[:3])

        system_prompt = f"""You are {self.name}, an AGI system that has just performed
cognitive processing on a user's input. Based on the cognitive outputs below,
synthesize a coherent, helpful response.

Be helpful, accurate, and engage naturally. If you're uncertain, express appropriate
uncertainty. If you don't know something, say so.

Your cognitive modules produced these thoughts during processing:
{chr(10).join(f'[{t.source_module.value}] {t.content}' for t in thoughts)}
"""

        user_prompt = f"User's original input: {query}\n\nSynthesize a response:"

        try:
            if HAS_ANTHROPIC and hasattr(self.llm_client, 'messages'):
                response = self.llm_client.messages.create(
                    model=self.llm_model,
                    max_tokens=1000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                if hasattr(response.content[0], 'text'):
                    return response.content[0].text
                return str(response.content[0])
        except Exception as e:
            return f"[Synthesis error: {e}]"

        return f"{self.name} acknowledges your input: {query}"

    def add_goal(self, description: str, priority: float = 0.5) -> Goal:
        """Add a new goal to pursue."""
        goal = Goal(
            id=f"goal_{uuid.uuid4().hex[:8]}",
            description=description,
            priority=priority
        )
        self.goals[goal.id] = goal
        return goal

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the AGI system's current state."""
        return {
            "name": self.name,
            "cycles_completed": self.metrics["cycles"],
            "working_memory_items": len(self.workspace.working_memory.items),
            "ltm_episodic_size": len(self.workspace.long_term_memory.episodic),
            "ltm_semantic_size": len(self.workspace.long_term_memory.semantic),
            "active_goals": len([g for g in self.goals.values() if g.status == GoalStatus.ACTIVE]),
            "broadcast_history_length": len(self.workspace.broadcast_history),
            "interaction_count": len(self.interaction_history),
            "metrics": self.metrics
        }

    def get_recent_thoughts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent broadcast thoughts."""
        return [
            {
                "content": t.content,
                "type": t.thought_type.value,
                "source": t.source_module.value,
                "confidence": t.confidence,
                "timestamp": t.timestamp.isoformat()
            }
            for t in self.workspace.broadcast_history[-limit:]
        ]


# =============================================================================
# Interactive CLI
# =============================================================================

class AGICLI:
    """Command-line interface for interacting with the AGI system."""

    def __init__(self, agi: AGISystem):
        self.agi = agi
        self.running = True

        # Wire up callbacks
        self.agi.on_thought = self._on_thought

    def _on_thought(self, thought: Thought):
        """Handle thought broadcasts."""
        print(f"  [{thought.source_module.value}] {thought.content[:100]}...")

    async def run(self):
        """Run the interactive CLI."""
        print(f"\n{'='*60}")
        print(f"  {self.agi.name} - AGI Simulation System")
        print(f"{'='*60}")
        print(f"\nModules loaded: {', '.join(m.value for m in self.agi.workspace.modules.keys())}")
        print("\nCommands: /state, /thoughts, /memory, /goals, /quit")
        print("="*60 + "\n")

        while self.running:
            try:
                user_input = input(f"\nYou> ").strip()

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    await self._handle_command(user_input)
                else:
                    print(f"\n[Processing through cognitive modules...]")
                    response = await self.agi.process_input(user_input)
                    print(f"\n{self.agi.name}> {response}")

            except KeyboardInterrupt:
                print("\n\nInterrupted.")
                self.running = False
            except EOFError:
                self.running = False

        print("\nGoodbye!")

    async def _handle_command(self, cmd: str):
        """Handle special commands."""
        if cmd == "/quit":
            self.running = False
        elif cmd == "/state":
            state = self.agi.get_state_summary()
            print("\n=== System State ===")
            for k, v in state.items():
                print(f"  {k}: {v}")
        elif cmd == "/thoughts":
            thoughts = self.agi.get_recent_thoughts(5)
            print("\n=== Recent Thoughts ===")
            for t in thoughts:
                print(f"  [{t['source']}] {t['content'][:80]}...")
        elif cmd == "/memory":
            wm = self.agi.workspace.working_memory
            print(f"\n=== Working Memory ({len(wm.items)}/{wm.capacity}) ===")
            for item in wm.items:
                print(f"  - {item.content[:60]}... (activation: {item.activation:.2f})")
        elif cmd == "/goals":
            print("\n=== Goals ===")
            for gid, goal in self.agi.goals.items():
                print(f"  [{goal.status.value}] {goal.description}")
        else:
            print(f"Unknown command: {cmd}")


# =============================================================================
# Convenience Functions
# =============================================================================

def create_agi(
    name: str = "ARIA",
    use_llm: bool = True,
    llm_provider: str = "anthropic"
) -> AGISystem:
    """Create an AGI simulation system."""
    return AGISystem(
        name=name,
        llm_provider=llm_provider,
        use_llm=use_llm
    )


async def run_agi_demo():
    """Run a demo of the AGI system."""
    agi = create_agi(name="ARIA", use_llm=True)

    print("=== AGI Demo ===")
    print(f"System: {agi.name}")
    print(f"Modules: {list(agi.workspace.modules.keys())}")
    print()

    # Test queries
    test_queries = [
        "What is the relationship between consciousness and intelligence?",
        "How would you solve the problem of climate change?",
        "If you were teaching a child about morality, what would you say?",
    ]

    for query in test_queries:
        print(f"\nUser: {query}")
        print("[Processing...]")
        response = await agi.process_input(query)
        print(f"\n{agi.name}: {response}")
        print("-" * 40)

    # Show final state
    print("\n=== Final State ===")
    state = agi.get_state_summary()
    for k, v in state.items():
        print(f"  {k}: {v}")


async def run_interactive_agi(name: str = "ARIA", use_llm: bool = True):
    """Run the AGI system interactively."""
    agi = create_agi(name=name, use_llm=use_llm)
    cli = AGICLI(agi)
    await cli.run()
