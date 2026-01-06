"""Reasoning traces and step tracking for cognitive agents."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class StepType(str, Enum):
    """Types of reasoning steps."""

    SUBGOAL = "subgoal"
    BACKWARD_CHAIN = "backward_chain"
    VERIFICATION = "verification"
    BACKTRACK = "backtrack"
    INFERENCE = "inference"
    OBSERVATION = "observation"
    ACTION = "action"
    SYNTHESIS = "synthesis"
    HYPOTHESIS = "hypothesis"
    REFINEMENT = "refinement"


class StepStatus(str, Enum):
    """Status of a reasoning step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BACKTRACKED = "backtracked"


@dataclass
class ReasoningStep:
    """A single step in the reasoning process."""

    step_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    step_type: StepType = StepType.INFERENCE
    content: str = ""
    status: StepStatus = StepStatus.PENDING
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "step_type": self.step_type.value,
            "content": self.content,
            "status": self.status.value,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
        }


@dataclass
class ReasoningTrace:
    """Complete trace of a reasoning process."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task: str = ""
    steps: list[ReasoningStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    success: bool = False
    total_confidence: float = 0.0
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def add_step(
        self,
        step_type: StepType,
        content: str,
        confidence: float = 0.0,
        parent_id: Optional[str] = None,
        **metadata,
    ) -> ReasoningStep:
        """Add a reasoning step."""
        step = ReasoningStep(
            step_type=step_type,
            content=content,
            confidence=confidence,
            parent_id=parent_id,
            metadata=metadata,
            status=StepStatus.IN_PROGRESS,
        )

        if parent_id:
            for s in self.steps:
                if s.step_id == parent_id:
                    s.children_ids.append(step.step_id)
                    break

        self.steps.append(step)
        return step

    def complete_step(
        self,
        step_id: str,
        status: StepStatus = StepStatus.COMPLETED,
        confidence: Optional[float] = None,
    ):
        """Mark a step as completed."""
        for step in self.steps:
            if step.step_id == step_id:
                step.status = status
                if confidence is not None:
                    step.confidence = confidence
                step.duration_ms = (datetime.now() - step.timestamp).total_seconds() * 1000
                break

    def get_step(self, step_id: str) -> Optional[ReasoningStep]:
        """Get a step by ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_steps_by_type(self, step_type: StepType) -> list[ReasoningStep]:
        """Get all steps of a given type."""
        return [s for s in self.steps if s.step_type == step_type]

    def get_active_steps(self) -> list[ReasoningStep]:
        """Get steps that are in progress."""
        return [s for s in self.steps if s.status == StepStatus.IN_PROGRESS]

    def get_completed_steps(self) -> list[ReasoningStep]:
        """Get completed steps."""
        return [s for s in self.steps if s.status == StepStatus.COMPLETED]

    def get_failed_steps(self) -> list[ReasoningStep]:
        """Get failed steps."""
        return [s for s in self.steps if s.status == StepStatus.FAILED]

    def compute_confidence(self) -> float:
        """Compute overall confidence from step confidences."""
        completed = self.get_completed_steps()
        if not completed:
            return 0.0

        # Weight by step type importance
        weights = {
            StepType.VERIFICATION: 1.5,
            StepType.SYNTHESIS: 1.3,
            StepType.INFERENCE: 1.0,
            StepType.SUBGOAL: 0.8,
            StepType.OBSERVATION: 0.7,
        }

        total_weight = 0.0
        weighted_conf = 0.0

        for step in completed:
            weight = weights.get(step.step_type, 1.0)
            weighted_conf += step.confidence * weight
            total_weight += weight

        self.total_confidence = weighted_conf / total_weight if total_weight > 0 else 0.0
        return self.total_confidence

    def finalize(self, answer: str, success: bool = True):
        """Finalize the trace with an answer."""
        self.final_answer = answer
        self.success = success
        self.completed_at = datetime.now()
        self.compute_confidence()

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "task": self.task,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "success": self.success,
            "total_confidence": self.total_confidence,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "step_count": len(self.steps),
            "completed_count": len(self.get_completed_steps()),
            "failed_count": len(self.get_failed_steps()),
        }

    def to_markdown(self) -> str:
        """Convert trace to markdown for display."""
        lines = [f"# Reasoning Trace: {self.trace_id[:8]}"]
        lines.append(f"\n**Task:** {self.task}")
        lines.append(f"**Status:** {'✅ Success' if self.success else '❌ Failed'}")
        try:
            conf = float(self.total_confidence)
        except (ValueError, TypeError):
            conf = 0.0
        lines.append(f"**Confidence:** {conf:.1%}")
        lines.append(f"\n## Steps ({len(self.steps)})")

        for i, step in enumerate(self.steps, 1):
            status_icon = {
                StepStatus.COMPLETED: "✅",
                StepStatus.FAILED: "❌",
                StepStatus.IN_PROGRESS: "🔄",
                StepStatus.SKIPPED: "⏭️",
                StepStatus.BACKTRACKED: "↩️",
                StepStatus.PENDING: "⏳",
            }.get(step.status, "•")

            lines.append(f"\n### {i}. {status_icon} {step.step_type.value.title()}")
            try:
                step_conf = float(step.confidence)
            except (ValueError, TypeError):
                step_conf = 0.0
            lines.append(f"**ID:** `{step.step_id}` | **Confidence:** {step_conf:.1%}")
            lines.append(f"\n{step.content}")

            if step.metadata:
                lines.append(f"\n*Metadata:* `{step.metadata}`")

        if self.final_answer:
            lines.append(f"\n## Final Answer\n{self.final_answer}")

        return "\n".join(lines)


@dataclass
class AgentContext:
    """Shared context for cognitive agent execution."""

    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: Optional[str] = None
    task: str = ""

    # Current state
    current_trace: Optional[ReasoningTrace] = None
    traces: list[ReasoningTrace] = field(default_factory=list)

    # Knowledge and memory
    observations: list[dict] = field(default_factory=list)
    hypotheses: list[dict] = field(default_factory=list)
    conclusions: list[dict] = field(default_factory=list)

    # Execution state
    iteration: int = 0
    max_iterations: int = 10
    backtrack_count: int = 0
    max_backtracks: int = 3

    # Shared data
    variables: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def start_trace(self, task: str) -> ReasoningTrace:
        """Start a new reasoning trace."""
        self.task = task
        self.current_trace = ReasoningTrace(task=task)
        self.traces.append(self.current_trace)
        return self.current_trace

    def add_observation(self, content: str, source: str = "unknown", **metadata):
        """Add an observation to context."""
        self.observations.append({
            "content": content,
            "source": source,
            "timestamp": datetime.now().isoformat(),
            **metadata,
        })

    def add_hypothesis(self, content: str, confidence: float = 0.5, **metadata):
        """Add a hypothesis."""
        try:
            conf = float(confidence)
        except (ValueError, TypeError):
            conf = 0.5
        self.hypotheses.append({
            "content": content,
            "confidence": conf,
            "timestamp": datetime.now().isoformat(),
            "verified": False,
            **metadata,
        })

    def add_conclusion(self, content: str, confidence: float = 0.8, **metadata):
        """Add a conclusion."""
        try:
            conf = float(confidence)
        except (ValueError, TypeError):
            conf = 0.8
        self.conclusions.append({
            "content": content,
            "confidence": conf,
            "timestamp": datetime.now().isoformat(),
            **metadata,
        })

    def can_continue(self) -> bool:
        """Check if agent can continue executing."""
        return (
            self.iteration < self.max_iterations
            and self.backtrack_count < self.max_backtracks
        )

    def increment_iteration(self):
        """Increment iteration counter."""
        self.iteration += 1

    def increment_backtrack(self):
        """Increment backtrack counter."""
        self.backtrack_count += 1

    def get_summary(self) -> dict:
        """Get context summary."""
        return {
            "context_id": self.context_id,
            "agent_id": self.agent_id,
            "task": self.task,
            "iteration": self.iteration,
            "backtrack_count": self.backtrack_count,
            "observation_count": len(self.observations),
            "hypothesis_count": len(self.hypotheses),
            "conclusion_count": len(self.conclusions),
            "trace_count": len(self.traces),
            "current_trace_steps": len(self.current_trace.steps) if self.current_trace else 0,
        }

    def to_prompt_context(self) -> str:
        """Convert context to string for LLM prompts."""
        lines = []

        if self.observations:
            lines.append("## Recent Observations")
            for obs in self.observations[-5:]:  # Last 5
                lines.append(f"- {obs['content']} (source: {obs['source']})")

        if self.hypotheses:
            lines.append("\n## Current Hypotheses")
            for hyp in self.hypotheses:
                verified = "✓" if hyp.get("verified") else "?"
                try:
                    conf = float(hyp.get('confidence', 0.5))
                except (ValueError, TypeError):
                    conf = 0.5
                lines.append(f"- [{verified}] {hyp['content']} (confidence: {conf:.0%})")

        if self.conclusions:
            lines.append("\n## Established Conclusions")
            for conc in self.conclusions[-5:]:
                lines.append(f"- {conc['content']}")

        return "\n".join(lines)
