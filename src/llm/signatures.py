"""Dynamic DSPy-style signature builder."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Type, get_type_hints

logger = logging.getLogger(__name__)


@dataclass
class Field:
    """A field in a signature."""

    name: str
    description: str = ""
    field_type: str = "str"
    required: bool = True
    default: Any = None
    prefix: str = ""  # For output formatting

    def to_prompt_line(self, is_input: bool = True) -> str:
        """Convert to a line in the signature prompt."""
        type_hint = f" ({self.field_type})" if self.field_type != "str" else ""
        req = "" if self.required else " [optional]"
        desc = f": {self.description}" if self.description else ""
        return f"- {self.name}{type_hint}{req}{desc}"


@dataclass
class DynamicSignature:
    """
    A dynamic signature that can be created at runtime.

    Inspired by DSPy signatures but with runtime flexibility.
    """

    name: str
    description: str = ""
    inputs: list[Field] = field(default_factory=list)
    outputs: list[Field] = field(default_factory=list)
    instructions: str = ""

    def input_schema(self) -> dict:
        """Get JSON schema for inputs."""
        properties = {}
        required = []

        for f in self.inputs:
            properties[f.name] = {
                "type": self._type_to_json(f.field_type),
                "description": f.description,
            }
            if f.required:
                required.append(f.name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def output_schema(self) -> dict:
        """Get JSON schema for outputs."""
        properties = {}
        required = []

        for f in self.outputs:
            properties[f.name] = {
                "type": self._type_to_json(f.field_type),
                "description": f.description,
            }
            if f.required:
                required.append(f.name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _type_to_json(self, t: str) -> str:
        """Convert Python type to JSON schema type."""
        mapping = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
        }
        return mapping.get(t, "string")

    def to_system_prompt(self) -> str:
        """Generate a system prompt from the signature."""
        lines = []

        if self.description:
            lines.append(self.description)
            lines.append("")

        if self.instructions:
            lines.append("Instructions:")
            lines.append(self.instructions)
            lines.append("")

        if self.inputs:
            lines.append("Input Fields:")
            for f in self.inputs:
                lines.append(f.to_prompt_line(is_input=True))
            lines.append("")

        if self.outputs:
            lines.append("Output Fields (respond with these):")
            for f in self.outputs:
                lines.append(f.to_prompt_line(is_input=False))
            lines.append("")

        lines.append("Respond with a JSON object containing the output fields.")

        return "\n".join(lines)

    def format_input(self, **kwargs) -> str:
        """Format input values as a prompt."""
        lines = []
        for f in self.inputs:
            value = kwargs.get(f.name)
            if value is not None:
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, indent=2)
                lines.append(f"{f.name}: {value}")
        return "\n".join(lines)

    def parse_output(self, response: str) -> dict:
        """Parse LLM response into output fields."""
        # Try to extract JSON from the response
        try:
            # Look for JSON block
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
            if json_match:
                return json.loads(json_match.group(1))

            # Try parsing the whole response as JSON
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Fallback: extract fields manually
        result = {}
        for f in self.outputs:
            # Look for "field_name: value" pattern
            pattern = rf"{f.name}:\s*(.+?)(?:\n|$)"
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                result[f.name] = match.group(1).strip()

        return result

    def __call__(self, **kwargs) -> dict:
        """Make the signature callable for validation."""
        # Validate required inputs
        for f in self.inputs:
            if f.required and f.name not in kwargs:
                raise ValueError(f"Missing required input: {f.name}")

        return kwargs


class SignatureBuilder:
    """Builder for creating signatures dynamically."""

    def __init__(self, name: str):
        self.name = name
        self.description = ""
        self.instructions = ""
        self._inputs: list[Field] = []
        self._outputs: list[Field] = []

    def describe(self, description: str) -> "SignatureBuilder":
        """Set the signature description."""
        self.description = description
        return self

    def instruct(self, instructions: str) -> "SignatureBuilder":
        """Set detailed instructions."""
        self.instructions = instructions
        return self

    def input(
        self,
        name: str,
        description: str = "",
        field_type: str = "str",
        required: bool = True,
        default: Any = None,
    ) -> "SignatureBuilder":
        """Add an input field."""
        self._inputs.append(Field(
            name=name,
            description=description,
            field_type=field_type,
            required=required,
            default=default,
        ))
        return self

    def output(
        self,
        name: str,
        description: str = "",
        field_type: str = "str",
        required: bool = True,
        prefix: str = "",
    ) -> "SignatureBuilder":
        """Add an output field."""
        self._outputs.append(Field(
            name=name,
            description=description,
            field_type=field_type,
            required=required,
            prefix=prefix,
        ))
        return self

    def build(self) -> DynamicSignature:
        """Build the signature."""
        return DynamicSignature(
            name=self.name,
            description=self.description,
            instructions=self.instructions,
            inputs=self._inputs.copy(),
            outputs=self._outputs.copy(),
        )


# Predefined signatures for common tasks
def qa_signature() -> DynamicSignature:
    """Question-answering signature."""
    return (
        SignatureBuilder("QuestionAnswer")
        .describe("Answer questions based on context.")
        .input("context", "Background information")
        .input("question", "The question to answer")
        .output("answer", "Direct answer to the question")
        .output("confidence", "Confidence level (low/medium/high)", required=False)
        .build()
    )


def summarize_signature() -> DynamicSignature:
    """Text summarization signature."""
    return (
        SignatureBuilder("Summarize")
        .describe("Summarize text concisely.")
        .input("text", "Text to summarize")
        .input("max_length", "Maximum summary length", field_type="int", required=False)
        .output("summary", "Concise summary of the text")
        .output("key_points", "Main points as a list", field_type="list", required=False)
        .build()
    )


def analyze_signature() -> DynamicSignature:
    """Analysis signature."""
    return (
        SignatureBuilder("Analyze")
        .describe("Analyze content from a specific perspective.")
        .input("content", "Content to analyze")
        .input("perspective", "Analysis perspective or lens")
        .input("focus", "Specific aspects to focus on", required=False)
        .output("analysis", "Detailed analysis")
        .output("insights", "Key insights discovered", field_type="list")
        .output("recommendations", "Suggested actions", field_type="list", required=False)
        .build()
    )


def debate_signature() -> DynamicSignature:
    """Debate/discussion signature."""
    return (
        SignatureBuilder("Debate")
        .describe("Engage in structured debate on a topic.")
        .input("topic", "The topic being debated")
        .input("position", "Your assigned position")
        .input("opponent_argument", "The opponent's previous argument", required=False)
        .output("argument", "Your argument")
        .output("evidence", "Supporting evidence", field_type="list")
        .output("counter", "Counter to opponent's points", required=False)
        .build()
    )


def chain_of_thought_signature() -> DynamicSignature:
    """Chain of thought reasoning signature."""
    return (
        SignatureBuilder("ChainOfThought")
        .describe("Reason step-by-step to solve a problem.")
        .instruct(
            "Think through the problem step by step. "
            "Show your reasoning process explicitly before giving the final answer."
        )
        .input("problem", "The problem to solve")
        .input("context", "Additional context", required=False)
        .output("reasoning", "Step-by-step reasoning process")
        .output("answer", "Final answer after reasoning")
        .build()
    )


def persona_response_signature() -> DynamicSignature:
    """Persona-based response signature."""
    return (
        SignatureBuilder("PersonaResponse")
        .describe("Respond to a query from a specific persona's perspective.")
        .input("persona", "Description of the persona")
        .input("query", "The query or topic to respond to")
        .input("context", "Conversation context", required=False)
        .output("response", "Response from the persona's perspective")
        .output("reasoning", "Internal reasoning (hidden from output)", required=False)
        .output("follow_up", "Suggested follow-up questions", field_type="list", required=False)
        .build()
    )


# Registry of predefined signatures
SIGNATURES = {
    "qa": qa_signature,
    "summarize": summarize_signature,
    "analyze": analyze_signature,
    "debate": debate_signature,
    "cot": chain_of_thought_signature,
    "chain_of_thought": chain_of_thought_signature,
    "persona": persona_response_signature,
}


def get_signature(name: str) -> DynamicSignature:
    """Get a predefined signature by name."""
    if name not in SIGNATURES:
        raise ValueError(f"Unknown signature: {name}. Available: {list(SIGNATURES.keys())}")
    return SIGNATURES[name]()
