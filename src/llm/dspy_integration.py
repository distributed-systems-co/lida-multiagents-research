"""DSPy-style module system with OpenRouter backend."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional

from .openrouter import OpenRouterClient, StreamingResponse, get_client
from .signatures import DynamicSignature, SignatureBuilder, get_signature

logger = logging.getLogger(__name__)


@dataclass
class ModuleResult:
    """Result from a DSPy module execution."""

    outputs: dict
    raw_response: str
    model: str
    signature: str
    success: bool = True
    error: Optional[str] = None

    def __getattr__(self, name: str) -> Any:
        if name in self.outputs:
            return self.outputs[name]
        raise AttributeError(f"No output field: {name}")

    def __getitem__(self, key: str) -> Any:
        return self.outputs[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.outputs.get(key, default)


class DSPyModule:
    """
    A DSPy-style module that executes signatures against an LLM.

    Example:
        sig = SignatureBuilder("QA").input("question").output("answer").build()
        module = DSPyModule(sig)
        result = await module(question="What is 2+2?")
        print(result.answer)
    """

    def __init__(
        self,
        signature: DynamicSignature | str,
        client: Optional[OpenRouterClient] = None,
        model: str = "anthropic/claude-sonnet-4.5",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        if isinstance(signature, str):
            signature = get_signature(signature)

        self.signature = signature
        self.client = client or get_client()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Execution history for optimization
        self.history: list[ModuleResult] = []

    async def __call__(self, **kwargs) -> ModuleResult:
        """Execute the module with given inputs."""
        return await self.forward(**kwargs)

    async def forward(self, **kwargs) -> ModuleResult:
        """Execute the signature with the given inputs."""
        # Validate inputs
        self.signature(**kwargs)

        # Build messages
        system_prompt = self.signature.to_system_prompt()
        user_prompt = self.signature.format_input(**kwargs)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await self.client.complete(
                messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            outputs = self.signature.parse_output(response.content)

            result = ModuleResult(
                outputs=outputs,
                raw_response=response.content,
                model=response.model,
                signature=self.signature.name,
            )

        except Exception as e:
            logger.error(f"Module execution failed: {e}")
            result = ModuleResult(
                outputs={},
                raw_response="",
                model=self.model,
                signature=self.signature.name,
                success=False,
                error=str(e),
            )

        self.history.append(result)
        return result

    async def stream(self, **kwargs) -> AsyncIterator[str]:
        """Stream the module output."""
        self.signature(**kwargs)

        system_prompt = self.signature.to_system_prompt()
        user_prompt = self.signature.format_input(**kwargs)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        async for chunk in await self.client.complete(
            messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        ):
            yield chunk

    async def stream_to_result(
        self,
        on_chunk: Optional[Callable[[str], None]] = None,
        **kwargs,
    ) -> ModuleResult:
        """Stream and collect into a result."""
        self.signature(**kwargs)

        system_prompt = self.signature.to_system_prompt()
        user_prompt = self.signature.format_input(**kwargs)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = await self.client.stream_to_response(
            messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            on_chunk=on_chunk,
        )

        outputs = self.signature.parse_output(response.content)

        result = ModuleResult(
            outputs=outputs,
            raw_response=response.content,
            model=response.model,
            signature=self.signature.name,
        )

        self.history.append(result)
        return result


class ChainOfThought(DSPyModule):
    """Chain of thought reasoning module."""

    def __init__(
        self,
        signature: DynamicSignature | str,
        **kwargs,
    ):
        # Wrap signature with CoT instructions
        if isinstance(signature, str):
            signature = get_signature(signature)

        # Add reasoning to outputs if not present
        has_reasoning = any(f.name == "reasoning" for f in signature.outputs)
        if not has_reasoning:
            from .signatures import Field

            signature.outputs.insert(
                0,
                Field(
                    name="reasoning",
                    description="Step-by-step reasoning process",
                    required=True,
                ),
            )

        # Add CoT instructions
        cot_instruct = (
            "Think step by step. First explain your reasoning process, "
            "then provide the final outputs."
        )
        if signature.instructions:
            signature.instructions = f"{cot_instruct}\n\n{signature.instructions}"
        else:
            signature.instructions = cot_instruct

        super().__init__(signature, **kwargs)


class Predict(DSPyModule):
    """Simple prediction module (alias for DSPyModule)."""

    pass


class MultiModule:
    """Execute multiple modules in parallel or sequence."""

    def __init__(self, modules: list[DSPyModule]):
        self.modules = modules

    async def parallel(self, inputs_list: list[dict]) -> list[ModuleResult]:
        """Execute all modules in parallel with corresponding inputs."""
        if len(inputs_list) != len(self.modules):
            raise ValueError("Number of inputs must match number of modules")

        tasks = [
            module(**inputs) for module, inputs in zip(self.modules, inputs_list)
        ]
        return await asyncio.gather(*tasks)

    async def sequential(self, initial_input: dict) -> list[ModuleResult]:
        """Execute modules sequentially, passing outputs to next module."""
        results = []
        current_input = initial_input

        for module in self.modules:
            result = await module(**current_input)
            results.append(result)

            # Merge outputs into next input
            current_input = {**current_input, **result.outputs}

        return results


def create_module(
    signature: DynamicSignature | str,
    chain_of_thought: bool = False,
    **kwargs,
) -> DSPyModule:
    """
    Factory function to create a module.

    Args:
        signature: Signature or signature name
        chain_of_thought: Whether to wrap with CoT
        **kwargs: Passed to module constructor

    Returns:
        DSPyModule instance
    """
    if chain_of_thought:
        return ChainOfThought(signature, **kwargs)
    return DSPyModule(signature, **kwargs)


# Convenience functions
async def predict(signature: str | DynamicSignature, **kwargs) -> ModuleResult:
    """Quick prediction with a signature."""
    module = DSPyModule(signature)
    return await module(**kwargs)


async def stream_predict(
    signature: str | DynamicSignature,
    **kwargs,
) -> AsyncIterator[str]:
    """Quick streaming prediction."""
    module = DSPyModule(signature)
    async for chunk in module.stream(**kwargs):
        yield chunk


# Agent-specific modules
def create_persona_module(
    persona_prompt: str,
    model: str = "anthropic/claude-sonnet-4.5",
) -> DSPyModule:
    """Create a module for a persona agent."""
    sig = (
        SignatureBuilder("PersonaResponse")
        .describe(f"You are embodying this persona:\n{persona_prompt}")
        .instruct(
            "Respond authentically from this persona's perspective. "
            "Draw on the persona's knowledge, values, and communication style."
        )
        .input("query", "The query or topic to respond to")
        .input("context", "Conversation context", required=False)
        .output("response", "Response from the persona's perspective")
        .output("internal_thought", "Your internal reasoning", required=False)
        .build()
    )
    return DSPyModule(sig, model=model)


def create_analysis_module(
    perspective: str,
    model: str = "anthropic/claude-sonnet-4.5",
) -> DSPyModule:
    """Create a module for analysis from a specific perspective."""
    sig = (
        SignatureBuilder("PerspectiveAnalysis")
        .describe(f"Analyze from this perspective: {perspective}")
        .input("content", "Content to analyze")
        .input("focus", "Specific aspects to examine", required=False)
        .output("analysis", "Detailed analysis")
        .output("key_insights", "Main insights", field_type="list")
        .output("blind_spots", "Potential blind spots in this perspective", field_type="list", required=False)
        .build()
    )
    return DSPyModule(sig, model=model)


def create_debate_module(
    position: str,
    model: str = "anthropic/claude-sonnet-4.5",
) -> DSPyModule:
    """Create a module for debate participation."""
    sig = (
        SignatureBuilder("DebateParticipant")
        .describe(f"You are arguing for: {position}")
        .instruct(
            "Make compelling arguments for your position. "
            "Acknowledge strong opposing points but defend your stance. "
            "Use evidence and logical reasoning."
        )
        .input("topic", "The debate topic")
        .input("opponent_point", "The opponent's latest argument", required=False)
        .input("debate_history", "Previous exchanges", required=False)
        .output("argument", "Your main argument")
        .output("evidence", "Supporting evidence", field_type="list")
        .output("rebuttal", "Response to opponent", required=False)
        .build()
    )
    return DSPyModule(sig, model=model)
