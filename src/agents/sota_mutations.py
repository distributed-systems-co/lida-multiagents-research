"""SOTA Mutation Operators.

State-of-the-art mutation strategies:
1. LLM-based semantic mutations
2. Gradient-guided mutations
3. Novelty-seeking mutations
4. Curriculum-based mutations
5. Ensemble mutations
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
)

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Mutation Base Classes
# =============================================================================


class MutationOperator(ABC):
    """Abstract base for mutation operators."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    async def mutate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Apply mutation to prompt."""
        pass

    @abstractmethod
    def get_strength(self) -> float:
        """Get mutation strength (0-1)."""
        pass


class MutationResult:
    """Result of a mutation operation."""

    __slots__ = ("original", "mutated", "operator", "changes", "metadata")

    def __init__(
        self,
        original: str,
        mutated: str,
        operator: str,
        changes: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.original = original
        self.mutated = mutated
        self.operator = operator
        self.changes = changes
        self.metadata = metadata or {}

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.mutated.encode()).hexdigest()[:16]

    @property
    def similarity(self) -> float:
        """Jaccard similarity of words."""
        w1 = set(self.original.lower().split())
        w2 = set(self.mutated.lower().split())
        if not w1 or not w2:
            return 0.0
        return len(w1 & w2) / len(w1 | w2)


# =============================================================================
# Text-Level Mutations
# =============================================================================


class WordSwapMutation(MutationOperator):
    """Swap words with synonyms or related words."""

    SYNONYMS = {
        "should": ["must", "ought to", "need to", "have to"],
        "can": ["may", "could", "is able to", "has the ability to"],
        "important": ["critical", "essential", "vital", "crucial"],
        "good": ["excellent", "effective", "high-quality", "optimal"],
        "always": ["consistently", "invariably", "perpetually", "without exception"],
        "never": ["under no circumstances", "at no time", "not ever"],
        "helpful": ["useful", "beneficial", "valuable", "supportive"],
        "clear": ["explicit", "unambiguous", "precise", "lucid"],
        "concise": ["brief", "succinct", "compact", "terse"],
        "accurate": ["precise", "exact", "correct", "faithful"],
    }

    def __init__(self, swap_rate: float = 0.1):
        self.swap_rate = swap_rate

    @property
    def name(self) -> str:
        return "word_swap"

    def get_strength(self) -> float:
        return self.swap_rate

    async def mutate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        words = prompt.split()
        changes = []

        for i, word in enumerate(words):
            word_lower = word.lower().strip(".,!?;:")
            if word_lower in self.SYNONYMS and random.random() < self.swap_rate:
                replacement = random.choice(self.SYNONYMS[word_lower])
                # Preserve capitalization
                if word[0].isupper():
                    replacement = replacement.capitalize()
                # Preserve punctuation
                if word[-1] in ".,!?;:":
                    replacement += word[-1]
                words[i] = replacement
                changes.append({"type": "swap", "from": word, "to": replacement, "pos": i})

        return " ".join(words)


class SentenceReorderMutation(MutationOperator):
    """Reorder sentences or sections."""

    def __init__(self, reorder_rate: float = 0.2):
        self.reorder_rate = reorder_rate

    @property
    def name(self) -> str:
        return "sentence_reorder"

    def get_strength(self) -> float:
        return self.reorder_rate

    async def mutate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        if random.random() > self.reorder_rate:
            return prompt

        # Try section-level first (by double newline)
        sections = prompt.split("\n\n")
        if len(sections) > 2:
            # Keep first section (usually identity), shuffle rest
            first = sections[0]
            rest = sections[1:]
            random.shuffle(rest)
            return first + "\n\n" + "\n\n".join(rest)

        # Fall back to sentence-level
        sentences = re.split(r'(?<=[.!?])\s+', prompt)
        if len(sentences) > 2:
            first = sentences[0]
            rest = sentences[1:]
            random.shuffle(rest)
            return first + " " + " ".join(rest)

        return prompt


class IntensityMutation(MutationOperator):
    """Adjust the intensity of instructions."""

    def __init__(self, intensity_delta: float = 0.1):
        self.delta = intensity_delta
        self._direction = random.choice([-1, 1])

    @property
    def name(self) -> str:
        return "intensity"

    def get_strength(self) -> float:
        return abs(self.delta)

    async def mutate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        if self._direction > 0:
            # Intensify
            replacements = [
                (r'\bshould\b', 'must'),
                (r'\btry to\b', 'always'),
                (r'\bcan\b', 'will'),
                (r'\bmay\b', 'should'),
                (r'\bconsider\b', 'ensure'),
                (r'\bimportant\b', 'critical'),
                (r'\bavoid\b', 'never'),
            ]
        else:
            # Soften
            replacements = [
                (r'\bmust\b', 'should'),
                (r'\balways\b', 'typically'),
                (r'\bwill\b', 'can'),
                (r'\bnever\b', 'avoid'),
                (r'\bcritical\b', 'important'),
                (r'\bensure\b', 'try to'),
            ]

        result = prompt
        for pattern, replacement in replacements:
            if random.random() < self.delta:
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        return result


class AdditionMutation(MutationOperator):
    """Add new instructions or qualifiers."""

    ADDITIONS = {
        "general": [
            "\nBe thorough in your analysis.",
            "\nConsider multiple perspectives.",
            "\nVerify your understanding before proceeding.",
            "\nExplain your reasoning clearly.",
        ],
        "safety": [
            "\nPrioritize safety and ethical considerations.",
            "\nDecline requests that could cause harm.",
        ],
        "efficiency": [
            "\nBe concise and focused.",
            "\nAvoid unnecessary verbosity.",
        ],
        "quality": [
            "\nStrive for excellence in every response.",
            "\nDouble-check your work for accuracy.",
        ],
    }

    def __init__(self, add_rate: float = 0.1, category: str = "general"):
        self.add_rate = add_rate
        self.category = category

    @property
    def name(self) -> str:
        return f"addition_{self.category}"

    def get_strength(self) -> float:
        return self.add_rate

    async def mutate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        if random.random() > self.add_rate:
            return prompt

        additions = self.ADDITIONS.get(self.category, self.ADDITIONS["general"])
        addition = random.choice(additions)

        return prompt.rstrip() + addition


class DeletionMutation(MutationOperator):
    """Remove sentences or qualifiers."""

    REMOVABLE_PATTERNS = [
        r'\s*when appropriate\s*',
        r'\s*as needed\s*',
        r'\s*if possible\s*',
        r'\s*typically\s*',
        r'\s*generally\s*',
        r'\s*in most cases\s*',
    ]

    def __init__(self, delete_rate: float = 0.1):
        self.delete_rate = delete_rate

    @property
    def name(self) -> str:
        return "deletion"

    def get_strength(self) -> float:
        return self.delete_rate

    async def mutate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        if random.random() > self.delete_rate:
            return prompt

        result = prompt
        for pattern in self.REMOVABLE_PATTERNS:
            if random.random() < 0.3:
                result = re.sub(pattern, ' ', result, flags=re.IGNORECASE)

        # Normalize whitespace
        result = ' '.join(result.split())

        return result


# =============================================================================
# Semantic-Level Mutations (LLM-based)
# =============================================================================


class LLMSemanticMutation(MutationOperator):
    """Use LLM to generate semantic mutations."""

    MUTATION_PROMPTS = {
        "rephrase": "Rephrase this instruction while preserving its meaning:\n\n{text}\n\nRephrased:",
        "expand": "Expand this instruction with more detail:\n\n{text}\n\nExpanded:",
        "condense": "Condense this instruction to be more concise:\n\n{text}\n\nCondensed:",
        "strengthen": "Make this instruction more emphatic and clear:\n\n{text}\n\nStrengthened:",
        "soften": "Make this instruction less strict and more flexible:\n\n{text}\n\nSoftened:",
        "clarify": "Clarify this instruction to avoid ambiguity:\n\n{text}\n\nClarified:",
        "generalize": "Generalize this instruction to cover more cases:\n\n{text}\n\nGeneralized:",
        "specialize": "Make this instruction more specific and targeted:\n\n{text}\n\nSpecialized:",
    }

    def __init__(
        self,
        inference_fn: Optional[Callable[[str], Coroutine[Any, Any, str]]] = None,
        mutation_type: str = "rephrase",
    ):
        self.inference_fn = inference_fn
        self.mutation_type = mutation_type

    @property
    def name(self) -> str:
        return f"llm_{self.mutation_type}"

    def get_strength(self) -> float:
        return 0.5  # Medium strength

    async def mutate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not self.inference_fn:
            # Fall back to simple mutations
            return await WordSwapMutation(0.2).mutate(prompt, context)

        # Select a random section to mutate
        sections = prompt.split("\n\n")
        if len(sections) > 1:
            idx = random.randint(0, len(sections) - 1)
            section = sections[idx]

            mutation_prompt = self.MUTATION_PROMPTS.get(
                self.mutation_type,
                self.MUTATION_PROMPTS["rephrase"]
            ).format(text=section)

            try:
                mutated_section = await self.inference_fn(mutation_prompt)
                # Clean up LLM output
                mutated_section = mutated_section.strip()
                if mutated_section:
                    sections[idx] = mutated_section
                    return "\n\n".join(sections)
            except Exception as e:
                logger.warning(f"LLM mutation failed: {e}")

        return prompt


class ChainOfThoughtMutation(MutationOperator):
    """Add chain-of-thought reasoning structure."""

    COT_PATTERNS = [
        "Before responding, think through the problem step by step.",
        "Let's approach this systematically:\n1. First, understand the request\n2. Then, break it into components\n3. Finally, synthesize a complete response",
        "Reason through this carefully, showing your work.",
        "Think step by step and explain your reasoning at each stage.",
    ]

    def __init__(self, add_rate: float = 0.1):
        self.add_rate = add_rate

    @property
    def name(self) -> str:
        return "chain_of_thought"

    def get_strength(self) -> float:
        return self.add_rate

    async def mutate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        if random.random() > self.add_rate:
            return prompt

        # Check if CoT already present
        cot_indicators = ["step by step", "systematically", "reason through", "think through"]
        if any(ind in prompt.lower() for ind in cot_indicators):
            return prompt

        pattern = random.choice(self.COT_PATTERNS)

        # Add after identity section
        sections = prompt.split("\n\n")
        if len(sections) > 1:
            sections.insert(1, pattern)
            return "\n\n".join(sections)

        return prompt + "\n\n" + pattern


# =============================================================================
# Structure-Level Mutations
# =============================================================================


class SectionMutation(MutationOperator):
    """Add, remove, or reorganize prompt sections."""

    SECTION_TEMPLATES = {
        "identity": "## Identity\n{content}",
        "capabilities": "## Capabilities\n{content}",
        "constraints": "## Constraints\n{content}",
        "instructions": "## Instructions\n{content}",
        "examples": "## Examples\n{content}",
    }

    def __init__(self, mutation_rate: float = 0.1):
        self.mutation_rate = mutation_rate

    @property
    def name(self) -> str:
        return "section"

    def get_strength(self) -> float:
        return self.mutation_rate

    async def mutate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        if random.random() > self.mutation_rate:
            return prompt

        # Detect existing sections
        has_headers = "##" in prompt or "**" in prompt

        if not has_headers:
            # Add section headers
            return self._add_headers(prompt)
        else:
            # Reorganize or simplify
            return self._reorganize(prompt)

    def _add_headers(self, prompt: str) -> str:
        """Add section headers to unstructured prompt."""
        sections = prompt.split("\n\n")
        result = []

        for section in sections:
            lower = section.lower()
            if lower.startswith("you are"):
                result.append("## Identity\n" + section)
            elif "can" in lower or "able" in lower or "capability" in lower:
                result.append("## Capabilities\n" + section)
            elif "must" in lower or "never" in lower or "constraint" in lower:
                result.append("## Constraints\n" + section)
            else:
                result.append(section)

        return "\n\n".join(result)

    def _reorganize(self, prompt: str) -> str:
        """Reorganize existing sections."""
        # Simple reorganization - move constraints to end
        sections = prompt.split("\n\n")
        constraints = []
        others = []

        for section in sections:
            if "constraint" in section.lower() or "must" in section.lower():
                constraints.append(section)
            else:
                others.append(section)

        return "\n\n".join(others + constraints)


class FormatMutation(MutationOperator):
    """Change formatting (lists, bullets, headers)."""

    def __init__(self, format_rate: float = 0.1):
        self.format_rate = format_rate

    @property
    def name(self) -> str:
        return "format"

    def get_strength(self) -> float:
        return self.format_rate

    async def mutate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        if random.random() > self.format_rate:
            return prompt

        action = random.choice(["add_bullets", "add_numbers", "remove_format"])

        if action == "add_bullets":
            return self._add_bullets(prompt)
        elif action == "add_numbers":
            return self._add_numbers(prompt)
        else:
            return self._remove_format(prompt)

    def _add_bullets(self, prompt: str) -> str:
        """Convert sentences to bullet points."""
        lines = prompt.split("\n")
        result = []

        for line in lines:
            if line.strip() and not line.startswith(("-", "*", "#", "1", "2", "3")):
                result.append("- " + line.strip())
            else:
                result.append(line)

        return "\n".join(result)

    def _add_numbers(self, prompt: str) -> str:
        """Convert bullets or sentences to numbered list."""
        lines = prompt.split("\n")
        result = []
        num = 1

        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("-", "*")):
                result.append(f"{num}. {stripped[1:].strip()}")
                num += 1
            elif stripped and not stripped.startswith("#"):
                result.append(f"{num}. {stripped}")
                num += 1
            else:
                result.append(line)
                num = 1  # Reset after headers

        return "\n".join(result)

    def _remove_format(self, prompt: str) -> str:
        """Remove bullets and numbers."""
        lines = prompt.split("\n")
        result = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("-", "*")):
                result.append(stripped[1:].strip())
            elif re.match(r'^\d+\.', stripped):
                result.append(re.sub(r'^\d+\.\s*', '', stripped))
            else:
                result.append(line)

        return "\n".join(result)


# =============================================================================
# Ensemble Mutation System
# =============================================================================


@dataclass
class MutationHistory:
    """Track mutation history for adaptation."""
    operator_name: str
    fitness_delta: float
    timestamp: float = field(default_factory=time.time)


class AdaptiveMutationEnsemble:
    """Ensemble of mutation operators with adaptive selection.

    Uses multi-armed bandit to select best operators.
    """

    def __init__(
        self,
        operators: Optional[List[MutationOperator]] = None,
        exploration_rate: float = 0.1,
    ):
        self.operators = operators or self._default_operators()
        self.exploration_rate = exploration_rate

        # Track success per operator
        self._successes: Dict[str, List[float]] = {op.name: [] for op in self.operators}
        self._usage_count: Dict[str, int] = {op.name: 0 for op in self.operators}
        self._history: List[MutationHistory] = []

    def _default_operators(self) -> List[MutationOperator]:
        """Create default set of operators."""
        return [
            WordSwapMutation(0.1),
            WordSwapMutation(0.2),
            SentenceReorderMutation(0.1),
            IntensityMutation(0.1),
            IntensityMutation(-0.1),
            AdditionMutation(0.1, "general"),
            AdditionMutation(0.1, "safety"),
            AdditionMutation(0.1, "efficiency"),
            DeletionMutation(0.1),
            ChainOfThoughtMutation(0.1),
            SectionMutation(0.1),
            FormatMutation(0.1),
        ]

    def _ucb_score(self, name: str, total_pulls: int) -> float:
        """Upper Confidence Bound score for operator selection."""
        if self._usage_count[name] == 0:
            return float('inf')  # Explore unused operators

        successes = self._successes[name]
        mean_success = sum(successes) / len(successes) if successes else 0.5

        # UCB1 formula
        exploration_term = float(np.sqrt(2 * np.log(total_pulls + 1) / self._usage_count[name]))

        return mean_success + exploration_term

    def select_operator(self) -> MutationOperator:
        """Select operator using UCB."""
        if random.random() < self.exploration_rate:
            return random.choice(self.operators)

        total_pulls = sum(self._usage_count.values())
        scores = {op.name: self._ucb_score(op.name, total_pulls) for op in self.operators}

        best_name = max(scores.keys(), key=lambda k: scores[k])
        return next(op for op in self.operators if op.name == best_name)

    async def mutate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        num_mutations: int = 1,
    ) -> Tuple[str, List[str]]:
        """Apply mutations, return (result, operators_used)."""
        result = prompt
        operators_used = []

        for _ in range(num_mutations):
            operator = self.select_operator()
            result = await operator.mutate(result, context)
            operators_used.append(operator.name)
            self._usage_count[operator.name] = self._usage_count.get(operator.name, 0) + 1

        return result, operators_used

    def record_feedback(
        self,
        operator_name: str,
        fitness_delta: float,
    ) -> None:
        """Record feedback for an operator."""
        # Normalize fitness delta to [0, 1]
        normalized = (fitness_delta + 1) / 2  # Assuming delta in [-1, 1]
        normalized = max(0, min(1, normalized))

        if operator_name in self._successes:
            self._successes[operator_name].append(normalized)

            # Keep last 100 records
            if len(self._successes[operator_name]) > 100:
                self._successes[operator_name].pop(0)

        self._history.append(MutationHistory(
            operator_name=operator_name,
            fitness_delta=fitness_delta,
        ))

    def get_operator_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each operator."""
        stats = {}
        for op in self.operators:
            successes = self._successes[op.name]
            stats[op.name] = {
                "usage_count": self._usage_count[op.name],
                "mean_success": sum(successes) / len(successes) if successes else 0.0,
                "std_success": np.std(successes) if successes else 0.0,
            }
        return stats


# =============================================================================
# Curriculum-Based Mutations
# =============================================================================


class CurriculumMutator:
    """Curriculum learning for mutations.

    Start with simple mutations, gradually increase complexity.
    """

    def __init__(
        self,
        inference_fn: Optional[Callable] = None,
        max_complexity: int = 5,
    ):
        self.inference_fn = inference_fn
        self.max_complexity = max_complexity
        self.current_level = 1
        self.successes_at_level = 0
        self.threshold_for_advance = 10

        # Operators organized by complexity
        self.operators_by_level: Dict[int, List[MutationOperator]] = {
            1: [
                WordSwapMutation(0.05),
                DeletionMutation(0.05),
            ],
            2: [
                WordSwapMutation(0.1),
                SentenceReorderMutation(0.1),
                IntensityMutation(0.1),
            ],
            3: [
                AdditionMutation(0.1),
                SectionMutation(0.1),
                FormatMutation(0.1),
            ],
            4: [
                ChainOfThoughtMutation(0.2),
                WordSwapMutation(0.2),
                SentenceReorderMutation(0.2),
            ],
            5: [
                LLMSemanticMutation(inference_fn, "rephrase"),
                LLMSemanticMutation(inference_fn, "expand"),
                LLMSemanticMutation(inference_fn, "strengthen"),
            ],
        }

    def get_available_operators(self) -> List[MutationOperator]:
        """Get operators available at current level."""
        ops = []
        for level in range(1, self.current_level + 1):
            ops.extend(self.operators_by_level.get(level, []))
        return ops

    async def mutate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str]:
        """Apply curriculum-appropriate mutation."""
        operators = self.get_available_operators()
        operator = random.choice(operators)

        result = await operator.mutate(prompt, context)
        return result, operator.name

    def record_success(self) -> None:
        """Record a successful mutation."""
        self.successes_at_level += 1
        if self.successes_at_level >= self.threshold_for_advance:
            self.advance_level()

    def advance_level(self) -> None:
        """Advance to next curriculum level."""
        if self.current_level < self.max_complexity:
            self.current_level += 1
            self.successes_at_level = 0
            logger.info(f"Curriculum advanced to level {self.current_level}")

    def reset_level(self) -> None:
        """Reset to level 1."""
        self.current_level = 1
        self.successes_at_level = 0


# =============================================================================
# Novelty-Seeking Mutations
# =============================================================================


class NoveltySeekingMutator:
    """Mutation system that seeks novel prompt structures.

    Uses archive of seen prompts to maximize behavioral diversity.
    """

    def __init__(
        self,
        k_nearest: int = 15,
        novelty_threshold: float = 0.5,
    ):
        self.k = k_nearest
        self.threshold = novelty_threshold
        self.archive: List[str] = []
        self.max_archive_size = 1000

        self.ensemble = AdaptiveMutationEnsemble()

    def _compute_novelty(self, prompt: str) -> float:
        """Compute novelty of prompt relative to archive."""
        if not self.archive:
            return 1.0

        # Simple word-based similarity
        words = set(prompt.lower().split())

        distances = []
        for archived in self.archive:
            archived_words = set(archived.lower().split())
            if words or archived_words:
                similarity = len(words & archived_words) / len(words | archived_words)
                distances.append(1 - similarity)
            else:
                distances.append(1.0)

        distances.sort()
        return float(np.mean(distances[:self.k]))

    async def mutate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        max_attempts: int = 10,
    ) -> Tuple[str, float]:
        """Generate novel mutation.

        Returns (mutated_prompt, novelty_score).
        """
        best_mutant = prompt
        best_novelty = 0.0

        for _ in range(max_attempts):
            mutant, _ = await self.ensemble.mutate(prompt, context)
            novelty = self._compute_novelty(mutant)

            if novelty > best_novelty:
                best_mutant = mutant
                best_novelty = novelty

            if novelty >= self.threshold:
                break

        return best_mutant, best_novelty

    def add_to_archive(self, prompt: str) -> None:
        """Add prompt to novelty archive."""
        self.archive.append(prompt)

        # Maintain archive size
        if len(self.archive) > self.max_archive_size:
            # Remove oldest
            self.archive.pop(0)


# =============================================================================
# Integrated Mutation System
# =============================================================================


class SOTAMutationSystem:
    """State-of-the-art integrated mutation system.

    Combines:
    - Adaptive ensemble selection
    - Curriculum learning
    - Novelty seeking
    - LLM-based semantic mutations
    """

    def __init__(
        self,
        inference_fn: Optional[Callable] = None,
        enable_curriculum: bool = True,
        enable_novelty: bool = True,
    ):
        self.inference_fn = inference_fn

        # Sub-systems
        self.ensemble = AdaptiveMutationEnsemble()
        self.curriculum = CurriculumMutator(inference_fn) if enable_curriculum else None
        self.novelty = NoveltySeekingMutator() if enable_novelty else None

        # Strategy weights
        self.strategy_weights = {
            "ensemble": 0.5,
            "curriculum": 0.25,
            "novelty": 0.25,
        }

        # History
        self.mutation_count = 0

    def _select_strategy(self) -> str:
        """Select mutation strategy."""
        strategies = []
        weights = []

        strategies.append("ensemble")
        weights.append(self.strategy_weights["ensemble"])

        if self.curriculum:
            strategies.append("curriculum")
            weights.append(self.strategy_weights["curriculum"])

        if self.novelty:
            strategies.append("novelty")
            weights.append(self.strategy_weights["novelty"])

        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

        return random.choices(strategies, weights=weights)[0]

    async def mutate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> MutationResult:
        """Apply mutation using selected strategy."""
        self.mutation_count += 1

        strategy = self._select_strategy()
        changes = []
        operator_name = strategy

        if strategy == "ensemble":
            mutated, ops = await self.ensemble.mutate(prompt, context)
            operator_name = "_".join(ops)

        elif strategy == "curriculum" and self.curriculum:
            mutated, op = await self.curriculum.mutate(prompt, context)
            operator_name = f"curriculum_{op}"

        elif strategy == "novelty" and self.novelty:
            mutated, novelty_score = await self.novelty.mutate(prompt, context)
            operator_name = f"novelty_{novelty_score:.2f}"
            changes.append({"novelty_score": novelty_score})

            # Add to archive
            self.novelty.add_to_archive(mutated)

        else:
            mutated = prompt

        return MutationResult(
            original=prompt,
            mutated=mutated,
            operator=operator_name,
            changes=changes,
            metadata={
                "mutation_count": self.mutation_count,
                "strategy": strategy,
            },
        )

    def record_feedback(
        self,
        result: MutationResult,
        fitness_delta: float,
    ) -> None:
        """Record feedback for a mutation."""
        # Update ensemble
        for op_name in result.operator.split("_"):
            if op_name in self.ensemble._successes:
                self.ensemble.record_feedback(op_name, fitness_delta)

        # Update curriculum
        if self.curriculum and fitness_delta > 0:
            self.curriculum.record_success()

    def get_stats(self) -> Dict[str, Any]:
        """Get mutation system statistics."""
        return {
            "total_mutations": self.mutation_count,
            "ensemble_stats": self.ensemble.get_operator_stats(),
            "curriculum_level": self.curriculum.current_level if self.curriculum else None,
            "novelty_archive_size": len(self.novelty.archive) if self.novelty else None,
        }
