"""Cognitive behavior modules for sophisticated reasoning."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from .dspy_integration import DSPyModule, create_module
from .openrouter import OpenRouterClient, get_client
from .reasoning import AgentContext, ReasoningTrace, StepType, StepStatus
from .signatures import SignatureBuilder

logger = logging.getLogger(__name__)


class BehaviorModule(ABC):
    """Base class for cognitive behavior modules."""

    name: str = "base_behavior"
    description: str = "Base behavior module"

    def __init__(
        self,
        client: Optional[OpenRouterClient] = None,
        model: str = "anthropic/claude-3.5-sonnet",
    ):
        self.client = client or get_client()
        self.model = model
        self._module: Optional[DSPyModule] = None

    @abstractmethod
    def _build_signature(self) -> Any:
        """Build the signature for this behavior."""
        pass

    def get_module(self) -> DSPyModule:
        """Get or create the DSPy module."""
        if self._module is None:
            sig = self._build_signature()
            self._module = DSPyModule(sig, client=self.client, model=self.model)
        return self._module

    @abstractmethod
    async def execute(
        self,
        context: AgentContext,
        **kwargs,
    ) -> dict:
        """Execute the behavior and return results."""
        pass


class SubgoalSettingModule(BehaviorModule):
    """
    Break down complex tasks into manageable subgoals.

    Uses hierarchical decomposition to create actionable subtasks.
    """

    name = "subgoal_setting"
    description = "Decompose complex tasks into subgoals"

    def _build_signature(self):
        return (
            SignatureBuilder("SubgoalSetting")
            .describe("Break down a complex task into manageable subgoals.")
            .instruct(
                "Analyze the task and create a hierarchical breakdown of subgoals. "
                "Each subgoal should be specific, measurable, and achievable. "
                "Order subgoals by dependency - prerequisites first."
            )
            .input("task", "The main task to decompose")
            .input("context", "Current context and constraints", required=False)
            .input("depth", "Maximum depth of subgoal hierarchy", field_type="int", required=False)
            .output("subgoals", "List of subgoals with dependencies", field_type="list")
            .output("reasoning", "Explanation of the decomposition strategy")
            .output("critical_path", "The sequence of must-complete subgoals", field_type="list")
            .build()
        )

    async def execute(
        self,
        context: AgentContext,
        task: Optional[str] = None,
        depth: int = 3,
        **kwargs,
    ) -> dict:
        """Execute subgoal decomposition."""
        task = task or context.task
        module = self.get_module()

        # Add step to trace
        step = None
        if context.current_trace:
            step = context.current_trace.add_step(
                StepType.SUBGOAL,
                f"Decomposing task: {task[:100]}...",
            )

        try:
            result = await module(
                task=task,
                context=context.to_prompt_context(),
                depth=depth,
            )

            subgoals = result.get("subgoals", [])
            critical_path = result.get("critical_path", [])

            # Store in context
            context.variables["subgoals"] = subgoals
            context.variables["critical_path"] = critical_path

            if step:
                context.current_trace.complete_step(
                    step.step_id,
                    StepStatus.COMPLETED,
                    confidence=0.8,
                )
                step.metadata["subgoal_count"] = len(subgoals)

            return {
                "success": True,
                "subgoals": subgoals,
                "critical_path": critical_path,
                "reasoning": result.get("reasoning", ""),
            }

        except Exception as e:
            logger.error(f"Subgoal setting failed: {e}")
            if step:
                context.current_trace.complete_step(step.step_id, StepStatus.FAILED)
            return {"success": False, "error": str(e)}


class BackwardChainingModule(BehaviorModule):
    """
    Work backwards from goal to determine required steps.

    Useful for planning when the goal is clear but the path isn't.
    """

    name = "backward_chaining"
    description = "Reason backwards from goal to initial conditions"

    def _build_signature(self):
        return (
            SignatureBuilder("BackwardChaining")
            .describe("Reason backwards from a goal state to determine required preconditions.")
            .instruct(
                "Start from the desired goal and work backwards. "
                "For each state, identify what must be true immediately before. "
                "Continue until you reach the current state or initial conditions."
            )
            .input("goal", "The desired end state or outcome")
            .input("current_state", "The current situation or starting point")
            .input("constraints", "Any constraints or limitations", required=False)
            .output("chain", "Sequence of states from goal back to start", field_type="list")
            .output("preconditions", "Required preconditions at each step", field_type="list")
            .output("blockers", "Potential obstacles identified", field_type="list")
            .output("reasoning", "Explanation of the backward reasoning")
            .build()
        )

    async def execute(
        self,
        context: AgentContext,
        goal: Optional[str] = None,
        current_state: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """Execute backward chaining."""
        goal = goal or context.task
        current_state = current_state or context.to_prompt_context()
        module = self.get_module()

        step = None
        if context.current_trace:
            step = context.current_trace.add_step(
                StepType.BACKWARD_CHAIN,
                f"Backward chaining from goal: {goal[:100]}...",
            )

        try:
            result = await module(
                goal=goal,
                current_state=current_state,
                constraints=kwargs.get("constraints", ""),
            )

            chain = result.get("chain", [])
            preconditions = result.get("preconditions", [])
            blockers = result.get("blockers", [])

            context.variables["backward_chain"] = chain
            context.variables["preconditions"] = preconditions

            if step:
                context.current_trace.complete_step(
                    step.step_id,
                    StepStatus.COMPLETED,
                    confidence=0.75,
                )
                step.metadata["chain_length"] = len(chain)

            return {
                "success": True,
                "chain": chain,
                "preconditions": preconditions,
                "blockers": blockers,
                "reasoning": result.get("reasoning", ""),
            }

        except Exception as e:
            logger.error(f"Backward chaining failed: {e}")
            if step:
                context.current_trace.complete_step(step.step_id, StepStatus.FAILED)
            return {"success": False, "error": str(e)}


class VerificationModule(BehaviorModule):
    """
    Verify conclusions and check for errors.

    Implements self-consistency checking and validation.
    """

    name = "verification"
    description = "Verify conclusions and identify errors"

    def _build_signature(self):
        return (
            SignatureBuilder("Verification")
            .describe("Verify a conclusion or result for correctness and consistency.")
            .instruct(
                "Critically examine the claim. Look for logical errors, "
                "unsupported assumptions, and contradictions. "
                "Consider edge cases and alternative interpretations."
            )
            .input("claim", "The claim or conclusion to verify")
            .input("evidence", "Supporting evidence or reasoning")
            .input("context", "Relevant context", required=False)
            .output("is_valid", "Whether the claim appears valid", field_type="bool")
            .output("confidence", "Confidence in the verification", field_type="float")
            .output("issues", "List of issues found", field_type="list")
            .output("suggestions", "Suggestions for improvement", field_type="list")
            .output("reasoning", "Detailed verification reasoning")
            .build()
        )

    async def execute(
        self,
        context: AgentContext,
        claim: str,
        evidence: str = "",
        **kwargs,
    ) -> dict:
        """Execute verification."""
        module = self.get_module()

        step = None
        if context.current_trace:
            step = context.current_trace.add_step(
                StepType.VERIFICATION,
                f"Verifying: {claim[:100]}...",
            )

        try:
            result = await module(
                claim=claim,
                evidence=evidence,
                context=context.to_prompt_context(),
            )

            is_valid = result.get("is_valid", False)
            confidence = float(result.get("confidence", 0.5))
            issues = result.get("issues", [])

            if step:
                status = StepStatus.COMPLETED if is_valid else StepStatus.FAILED
                context.current_trace.complete_step(
                    step.step_id,
                    status,
                    confidence=confidence,
                )
                step.metadata["is_valid"] = is_valid
                step.metadata["issue_count"] = len(issues)

            return {
                "success": True,
                "is_valid": is_valid,
                "confidence": confidence,
                "issues": issues,
                "suggestions": result.get("suggestions", []),
                "reasoning": result.get("reasoning", ""),
            }

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            if step:
                context.current_trace.complete_step(step.step_id, StepStatus.FAILED)
            return {"success": False, "error": str(e)}


class BacktrackingModule(BehaviorModule):
    """
    Backtrack when stuck and try alternative approaches.

    Implements intelligent backtracking with learning.
    """

    name = "backtracking"
    description = "Backtrack and explore alternative approaches"

    def _build_signature(self):
        return (
            SignatureBuilder("Backtracking")
            .describe("Analyze a failed approach and suggest alternatives.")
            .instruct(
                "Examine why the current approach failed. "
                "Identify the decision point where things went wrong. "
                "Suggest alternative approaches that avoid the same pitfalls."
            )
            .input("failed_approach", "Description of what was tried")
            .input("failure_reason", "Why it didn't work")
            .input("history", "Previous attempts", required=False)
            .input("constraints", "Constraints that must be maintained", required=False)
            .output("backtrack_point", "Where to backtrack to")
            .output("alternatives", "Alternative approaches to try", field_type="list")
            .output("lessons", "Lessons learned from the failure", field_type="list")
            .output("recommended", "Most promising alternative")
            .output("reasoning", "Analysis of the failure and alternatives")
            .build()
        )

    async def execute(
        self,
        context: AgentContext,
        failed_approach: str,
        failure_reason: str,
        **kwargs,
    ) -> dict:
        """Execute backtracking analysis."""
        module = self.get_module()
        context.increment_backtrack()

        step = None
        if context.current_trace:
            step = context.current_trace.add_step(
                StepType.BACKTRACK,
                f"Backtracking due to: {failure_reason[:100]}...",
            )

        try:
            # Gather history from context
            history = ""
            if context.current_trace:
                failed_steps = context.current_trace.get_failed_steps()
                history = "\n".join([
                    f"- {s.content} (failed: {s.metadata.get('error', 'unknown')})"
                    for s in failed_steps
                ])

            result = await module(
                failed_approach=failed_approach,
                failure_reason=failure_reason,
                history=history,
                constraints=kwargs.get("constraints", ""),
            )

            alternatives = result.get("alternatives", [])
            lessons = result.get("lessons", [])

            # Store lessons in context
            for lesson in lessons:
                context.add_observation(
                    lesson,
                    source="backtracking",
                    type="lesson_learned",
                )

            if step:
                context.current_trace.complete_step(
                    step.step_id,
                    StepStatus.COMPLETED,
                    confidence=0.7,
                )
                step.metadata["alternatives_count"] = len(alternatives)

            return {
                "success": True,
                "backtrack_point": result.get("backtrack_point", ""),
                "alternatives": alternatives,
                "lessons": lessons,
                "recommended": result.get("recommended", ""),
                "reasoning": result.get("reasoning", ""),
            }

        except Exception as e:
            logger.error(f"Backtracking failed: {e}")
            if step:
                context.current_trace.complete_step(step.step_id, StepStatus.FAILED)
            return {"success": False, "error": str(e)}


class HypothesisGenerationModule(BehaviorModule):
    """Generate and rank hypotheses for a given observation."""

    name = "hypothesis_generation"
    description = "Generate hypotheses to explain observations"

    def _build_signature(self):
        return (
            SignatureBuilder("HypothesisGeneration")
            .describe("Generate plausible hypotheses to explain observations.")
            .instruct(
                "Consider multiple possible explanations. "
                "Rank by plausibility and testability. "
                "Include both obvious and non-obvious hypotheses."
            )
            .input("observation", "The observation to explain")
            .input("background", "Background knowledge", required=False)
            .input("constraints", "Known constraints", required=False)
            .output("hypotheses", "List of hypotheses with rankings", field_type="list")
            .output("best_hypothesis", "Most plausible hypothesis")
            .output("tests", "How to test each hypothesis", field_type="list")
            .output("reasoning", "Reasoning behind the hypotheses")
            .build()
        )

    async def execute(
        self,
        context: AgentContext,
        observation: str,
        **kwargs,
    ) -> dict:
        """Generate hypotheses."""
        module = self.get_module()

        step = None
        if context.current_trace:
            step = context.current_trace.add_step(
                StepType.HYPOTHESIS,
                f"Generating hypotheses for: {observation[:100]}...",
            )

        try:
            result = await module(
                observation=observation,
                background=context.to_prompt_context(),
                constraints=kwargs.get("constraints", ""),
            )

            hypotheses = result.get("hypotheses", [])

            # Add to context
            for hyp in hypotheses:
                if isinstance(hyp, dict):
                    try:
                        conf = float(hyp.get("plausibility", 0.5))
                    except (ValueError, TypeError):
                        conf = 0.5
                    context.add_hypothesis(
                        hyp.get("content", str(hyp)),
                        confidence=conf,
                    )
                else:
                    context.add_hypothesis(str(hyp))

            if step:
                context.current_trace.complete_step(
                    step.step_id,
                    StepStatus.COMPLETED,
                    confidence=0.7,
                )

            return {
                "success": True,
                "hypotheses": hypotheses,
                "best_hypothesis": result.get("best_hypothesis", ""),
                "tests": result.get("tests", []),
                "reasoning": result.get("reasoning", ""),
            }

        except Exception as e:
            logger.error(f"Hypothesis generation failed: {e}")
            if step:
                context.current_trace.complete_step(step.step_id, StepStatus.FAILED)
            return {"success": False, "error": str(e)}


class SynthesisModule(BehaviorModule):
    """Synthesize multiple inputs into a coherent conclusion."""

    name = "synthesis"
    description = "Synthesize information into conclusions"

    def _build_signature(self):
        return (
            SignatureBuilder("Synthesis")
            .describe("Synthesize multiple pieces of information into a coherent conclusion.")
            .instruct(
                "Integrate the provided information. "
                "Resolve any contradictions. "
                "Identify the most important insights and form a unified conclusion."
            )
            .input("inputs", "List of information pieces to synthesize", field_type="list")
            .input("focus", "What aspect to focus on", required=False)
            .input("format", "Desired output format", required=False)
            .output("synthesis", "The synthesized conclusion")
            .output("key_insights", "Most important insights", field_type="list")
            .output("contradictions", "Any contradictions found", field_type="list")
            .output("confidence", "Confidence in the synthesis", field_type="float")
            .output("reasoning", "How the synthesis was formed")
            .build()
        )

    async def execute(
        self,
        context: AgentContext,
        inputs: list,
        focus: str = "",
        **kwargs,
    ) -> dict:
        """Synthesize inputs."""
        module = self.get_module()

        step = None
        if context.current_trace:
            step = context.current_trace.add_step(
                StepType.SYNTHESIS,
                f"Synthesizing {len(inputs)} inputs...",
            )

        try:
            result = await module(
                inputs=inputs,
                focus=focus,
                format=kwargs.get("format", "paragraph"),
            )

            synthesis = result.get("synthesis", "")
            confidence = float(result.get("confidence", 0.7))

            # Add conclusion to context
            context.add_conclusion(synthesis, confidence=confidence)

            if step:
                context.current_trace.complete_step(
                    step.step_id,
                    StepStatus.COMPLETED,
                    confidence=confidence,
                )

            return {
                "success": True,
                "synthesis": synthesis,
                "key_insights": result.get("key_insights", []),
                "contradictions": result.get("contradictions", []),
                "confidence": confidence,
                "reasoning": result.get("reasoning", ""),
            }

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            if step:
                context.current_trace.complete_step(step.step_id, StepStatus.FAILED)
            return {"success": False, "error": str(e)}


# Registry of available behaviors
BEHAVIORS = {
    "subgoal_setting": SubgoalSettingModule,
    "backward_chaining": BackwardChainingModule,
    "verification": VerificationModule,
    "backtracking": BacktrackingModule,
    "hypothesis_generation": HypothesisGenerationModule,
    "synthesis": SynthesisModule,
}


def get_behavior(
    name: str,
    client: Optional[OpenRouterClient] = None,
    model: str = "anthropic/claude-3.5-sonnet",
) -> BehaviorModule:
    """Get a behavior module by name."""
    if name not in BEHAVIORS:
        raise ValueError(f"Unknown behavior: {name}. Available: {list(BEHAVIORS.keys())}")
    return BEHAVIORS[name](client=client, model=model)


def list_behaviors() -> list[dict]:
    """List all available behaviors."""
    return [
        {"name": name, "description": cls.description}
        for name, cls in BEHAVIORS.items()
    ]
