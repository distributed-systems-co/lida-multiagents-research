"""Cognitive agent with sophisticated reasoning capabilities."""

from __future__ import annotations

import asyncio
import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional

from .behaviors import (
    BehaviorModule,
    BacktrackingModule,
    BackwardChainingModule,
    HypothesisGenerationModule,
    SubgoalSettingModule,
    SynthesisModule,
    VerificationModule,
    get_behavior,
    list_behaviors,
)
from .dspy_integration import DSPyModule, ChainOfThought
from .openrouter import OpenRouterClient, get_client
from .reasoning import AgentContext, ReasoningTrace, StepType, StepStatus
from .signatures import SignatureBuilder

logger = logging.getLogger(__name__)


@dataclass
class PlanStep:
    """A step in an execution plan."""

    behavior: str
    params: dict = field(default_factory=dict)
    condition: Optional[str] = None  # Execute only if condition is met
    fallback: Optional[str] = None  # Behavior to run if this fails
    priority: int = 0


@dataclass
class ExecutionPlan:
    """A plan of behavior executions."""

    steps: list[PlanStep] = field(default_factory=list)
    name: str = ""
    description: str = ""

    def add_step(
        self,
        behavior: str,
        params: dict = None,
        condition: str = None,
        fallback: str = None,
        priority: int = 0,
    ) -> "ExecutionPlan":
        """Add a step to the plan."""
        self.steps.append(PlanStep(
            behavior=behavior,
            params=params or {},
            condition=condition,
            fallback=fallback,
            priority=priority,
        ))
        return self


class CartesianProductPlanner:
    """
    Explore behavior combinations using Cartesian product.

    Useful for finding optimal behavior sequences.
    """

    def __init__(
        self,
        behaviors: list[str] = None,
        max_depth: int = 3,
    ):
        self.behaviors = behaviors or [
            "subgoal_setting",
            "backward_chaining",
            "verification",
        ]
        self.max_depth = max_depth

    def generate_plans(self, depth: int = None) -> list[ExecutionPlan]:
        """Generate all possible plans up to given depth."""
        depth = min(depth or self.max_depth, self.max_depth)
        plans = []

        for d in range(1, depth + 1):
            for combo in itertools.product(self.behaviors, repeat=d):
                plan = ExecutionPlan(
                    name=f"plan_{'_'.join(combo)}",
                    description=f"Sequence: {' → '.join(combo)}",
                )
                for behavior in combo:
                    plan.add_step(behavior)
                plans.append(plan)

        return plans

    def generate_conditional_plans(self) -> list[ExecutionPlan]:
        """Generate plans with conditional logic."""
        plans = []

        # Decompose → Verify → Synthesize
        plan1 = ExecutionPlan(name="decompose_verify_synthesize")
        plan1.add_step("subgoal_setting")
        plan1.add_step("verification", condition="subgoals_generated")
        plan1.add_step("synthesis", condition="verified")
        plans.append(plan1)

        # Backward chain → Hypothesis → Verify
        plan2 = ExecutionPlan(name="backward_hypothesis_verify")
        plan2.add_step("backward_chaining")
        plan2.add_step("hypothesis_generation", condition="chain_found")
        plan2.add_step("verification")
        plans.append(plan2)

        # Hypothesis → Verify → Backtrack if failed
        plan3 = ExecutionPlan(name="hypothesis_verify_backtrack")
        plan3.add_step("hypothesis_generation")
        plan3.add_step("verification", fallback="backtracking")
        plan3.add_step("synthesis", condition="verified")
        plans.append(plan3)

        return plans


class CognitiveAgent:
    """
    A cognitive agent with sophisticated reasoning behaviors.

    Combines multiple cognitive behaviors with planning and
    execution tracking.
    """

    def __init__(
        self,
        client: Optional[OpenRouterClient] = None,
        model: str = "anthropic/claude-sonnet-4.5",
        behaviors: list[str] = None,
        max_iterations: int = 10,
        max_backtracks: int = 3,
    ):
        self.client = client or get_client()
        self.model = model
        self.max_iterations = max_iterations
        self.max_backtracks = max_backtracks

        # Initialize behavior modules
        self.behaviors: dict[str, BehaviorModule] = {}
        behavior_names = behaviors or [
            "subgoal_setting",
            "backward_chaining",
            "verification",
            "backtracking",
            "hypothesis_generation",
            "synthesis",
        ]
        for name in behavior_names:
            self.behaviors[name] = get_behavior(name, client=self.client, model=self.model)

        # Planner
        self.planner = CartesianProductPlanner(list(self.behaviors.keys()))

        # Main reasoning module
        self._reasoning_module = self._create_reasoning_module()

    def _create_reasoning_module(self) -> DSPyModule:
        """Create the main reasoning module."""
        sig = (
            SignatureBuilder("CognitiveReasoning")
            .describe("Perform sophisticated multi-step reasoning.")
            .instruct(
                "Analyze the task carefully. Use structured reasoning. "
                "Consider multiple perspectives. Verify conclusions."
            )
            .input("task", "The task to reason about")
            .input("context", "Current context and observations", required=False)
            .input("constraints", "Constraints to satisfy", required=False)
            .output("reasoning", "Step-by-step reasoning process")
            .output("conclusion", "Final conclusion")
            .output("confidence", "Confidence level (0-1)", field_type="float")
            .output("next_action", "Suggested next action", required=False)
            .build()
        )
        return ChainOfThought(sig, client=self.client, model=self.model)

    async def reason(
        self,
        task: str,
        context: Optional[AgentContext] = None,
        plan: Optional[ExecutionPlan] = None,
        on_step: Optional[Callable[[ReasoningTrace], None]] = None,
    ) -> dict:
        """
        Execute reasoning with cognitive behaviors.

        Args:
            task: The task to reason about
            context: Optional existing context
            plan: Optional execution plan
            on_step: Callback for each reasoning step

        Returns:
            Result dictionary with trace and conclusions
        """
        # Initialize context
        if context is None:
            context = AgentContext(
                max_iterations=self.max_iterations,
                max_backtracks=self.max_backtracks,
            )

        # Start trace
        trace = context.start_trace(task)

        # Use default plan if none provided
        if plan is None:
            plan = self._create_default_plan(task)

        # Execute plan
        results = []
        for step in plan.steps:
            if not context.can_continue():
                logger.warning("Context limit reached, stopping execution")
                break

            context.increment_iteration()

            # Check condition
            if step.condition and not self._check_condition(step.condition, context):
                continue

            # Get behavior
            behavior = self.behaviors.get(step.behavior)
            if not behavior:
                logger.warning(f"Unknown behavior: {step.behavior}")
                continue

            # Execute
            try:
                result = await behavior.execute(context, **step.params, task=task)
                results.append({
                    "behavior": step.behavior,
                    "result": result,
                    "success": result.get("success", False),
                })

                if on_step:
                    on_step(trace)

                # Handle failure with fallback
                if not result.get("success") and step.fallback:
                    fallback_behavior = self.behaviors.get(step.fallback)
                    if fallback_behavior:
                        fallback_result = await fallback_behavior.execute(
                            context,
                            failed_approach=step.behavior,
                            failure_reason=result.get("error", "Unknown"),
                        )
                        results.append({
                            "behavior": step.fallback,
                            "result": fallback_result,
                            "success": fallback_result.get("success", False),
                            "is_fallback": True,
                        })

            except Exception as e:
                logger.error(f"Behavior {step.behavior} failed: {e}")
                results.append({
                    "behavior": step.behavior,
                    "error": str(e),
                    "success": False,
                })

        # Final synthesis
        synthesis_result = await self._synthesize_results(context, results)

        # Finalize trace
        trace.finalize(
            answer=synthesis_result.get("synthesis", ""),
            success=synthesis_result.get("success", False),
        )

        return {
            "task": task,
            "trace": trace.to_dict(),
            "results": results,
            "synthesis": synthesis_result,
            "context_summary": context.get_summary(),
        }

    def _create_default_plan(self, task: str) -> ExecutionPlan:
        """Create a default execution plan based on task type."""
        plan = ExecutionPlan(name="default_plan")

        # Always start with subgoal decomposition
        plan.add_step("subgoal_setting", params={"task": task})

        # Generate hypotheses
        plan.add_step("hypothesis_generation", params={"observation": task})

        # Verify initial understanding
        plan.add_step(
            "verification",
            params={"claim": "Initial task understanding"},
            fallback="backtracking",
        )

        # Synthesize conclusions
        plan.add_step("synthesis")

        return plan

    def _check_condition(self, condition: str, context: AgentContext) -> bool:
        """Check if a condition is met."""
        conditions = {
            "subgoals_generated": lambda: bool(context.variables.get("subgoals")),
            "chain_found": lambda: bool(context.variables.get("backward_chain")),
            "verified": lambda: any(
                s.step_type == StepType.VERIFICATION and s.status == StepStatus.COMPLETED
                for s in (context.current_trace.steps if context.current_trace else [])
            ),
            "has_hypotheses": lambda: bool(context.hypotheses),
            "has_conclusions": lambda: bool(context.conclusions),
        }

        checker = conditions.get(condition)
        if checker:
            return checker()

        # Default: condition is variable name
        return bool(context.variables.get(condition))

    async def _synthesize_results(
        self,
        context: AgentContext,
        results: list[dict],
    ) -> dict:
        """Synthesize all results into final conclusion."""
        synthesis = self.behaviors.get("synthesis")
        if not synthesis:
            return {
                "success": True,
                "synthesis": "No synthesis behavior available",
            }

        # Gather all insights
        inputs = []
        for r in results:
            if r.get("success"):
                result_data = r.get("result", {})
                if "reasoning" in result_data:
                    inputs.append(result_data["reasoning"])
                if "synthesis" in result_data:
                    inputs.append(result_data["synthesis"])

        # Add observations and conclusions
        for obs in context.observations[-5:]:
            inputs.append(f"Observation: {obs['content']}")
        for conc in context.conclusions[-3:]:
            inputs.append(f"Conclusion: {conc['content']}")

        if not inputs:
            return {
                "success": False,
                "synthesis": "No data to synthesize",
            }

        return await synthesis.execute(
            context,
            inputs=inputs,
            focus=context.task,
        )

    async def stream_reason(
        self,
        task: str,
        context: Optional[AgentContext] = None,
    ) -> AsyncIterator[dict]:
        """
        Stream reasoning process step by step.

        Yields status updates and intermediate results.
        """
        context = context or AgentContext(
            max_iterations=self.max_iterations,
            max_backtracks=self.max_backtracks,
        )

        trace = context.start_trace(task)

        yield {
            "type": "start",
            "task": task,
            "trace_id": trace.trace_id,
        }

        plan = self._create_default_plan(task)

        for step in plan.steps:
            if not context.can_continue():
                yield {"type": "limit_reached", "reason": "max iterations"}
                break

            context.increment_iteration()

            behavior = self.behaviors.get(step.behavior)
            if not behavior:
                continue

            yield {
                "type": "behavior_start",
                "behavior": step.behavior,
                "iteration": context.iteration,
            }

            try:
                result = await behavior.execute(context, **step.params, task=task)

                yield {
                    "type": "behavior_complete",
                    "behavior": step.behavior,
                    "success": result.get("success", False),
                    "result": result,
                }

            except Exception as e:
                yield {
                    "type": "behavior_error",
                    "behavior": step.behavior,
                    "error": str(e),
                }

        # Final synthesis
        synthesis_result = await self._synthesize_results(context, [])
        trace.finalize(
            synthesis_result.get("synthesis", ""),
            success=synthesis_result.get("success", False),
        )

        yield {
            "type": "complete",
            "trace": trace.to_dict(),
            "synthesis": synthesis_result,
        }

    async def quick_reason(self, task: str) -> str:
        """Quick reasoning without full behavior execution."""
        result = await self._reasoning_module(task=task)
        return result.get("conclusion", result.get("reasoning", ""))

    def list_behaviors(self) -> list[dict]:
        """List available behaviors."""
        return [
            {
                "name": name,
                "description": behavior.description,
                "loaded": True,
            }
            for name, behavior in self.behaviors.items()
        ]

    def get_plans(self, depth: int = 2) -> list[dict]:
        """Get possible execution plans."""
        plans = self.planner.generate_plans(depth)
        return [
            {
                "name": p.name,
                "description": p.description,
                "steps": [s.behavior for s in p.steps],
            }
            for p in plans
        ]


# Factory function
def create_cognitive_agent(
    model: str = "anthropic/claude-sonnet-4.5",
    behaviors: list[str] = None,
    **kwargs,
) -> CognitiveAgent:
    """Create a cognitive agent with specified configuration."""
    return CognitiveAgent(
        model=model,
        behaviors=behaviors,
        **kwargs,
    )


# Convenience functions
async def reason(task: str, model: str = "anthropic/claude-sonnet-4.5") -> dict:
    """Quick reasoning with default agent."""
    agent = create_cognitive_agent(model=model)
    return await agent.reason(task)


async def quick_reason(task: str, model: str = "anthropic/claude-sonnet-4.5") -> str:
    """Very quick reasoning without behaviors."""
    agent = create_cognitive_agent(model=model)
    return await agent.quick_reason(task)
