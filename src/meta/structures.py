"""Meta-structures: Composable building blocks for capabilities."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, Union


class StructureType(str, Enum):
    """Types of meta-structures."""
    PRIMITIVE = "primitive"       # Atomic operation
    COMPOSITE = "composite"       # Composed from others
    RECURSIVE = "recursive"       # Self-referential
    PARAMETRIC = "parametric"     # Parameterized template
    EMERGENT = "emergent"         # Emerged from combination
    REFLECTIVE = "reflective"     # Can inspect/modify self


class CapabilityType(str, Enum):
    """Types of capabilities."""
    PERCEPTION = "perception"     # Observe world
    REASONING = "reasoning"       # Think/analyze
    ACTION = "action"             # Affect world
    COMMUNICATION = "communication"  # Agent-to-agent
    LEARNING = "learning"         # Update from experience
    CREATION = "creation"         # Generate new structures
    COMPOSITION = "composition"   # Combine structures
    REFLECTION = "reflection"     # Self-examination
    META = "meta"                 # Meta-level operations


@dataclass
class Capability:
    """A capability that an agent can possess."""

    capability_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    name: str = ""
    capability_type: CapabilityType = CapabilityType.REASONING
    description: str = ""

    # The actual implementation
    implementation: Optional[Callable] = None

    # Structural properties
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)

    # Composition info
    composed_from: list[str] = field(default_factory=list)  # Capability IDs

    # Metadata
    version: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    def __hash__(self):
        return hash(self.capability_id)

    def signature(self) -> str:
        """Get unique signature based on structure."""
        sig = f"{self.name}:{','.join(sorted(self.inputs))}:{','.join(sorted(self.outputs))}"
        return hashlib.sha256(sig.encode()).hexdigest()[:16]


@dataclass
class MetaStructure:
    """A composable meta-structure for building capabilities.

    Meta-structures can:
    - Compose with other structures
    - Apply themselves recursively
    - Generate new structures
    - Transform existing structures
    """

    structure_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    name: str = ""
    structure_type: StructureType = StructureType.PRIMITIVE

    # Core structure
    capabilities: list[Capability] = field(default_factory=list)
    sub_structures: list["MetaStructure"] = field(default_factory=list)

    # Self-reference for recursive structures
    self_reference: Optional[str] = None  # Points to own structure_id
    recursion_depth: int = 0
    max_recursion: int = 10

    # Parameters for parametric structures
    parameters: dict[str, Any] = field(default_factory=dict)
    parameter_constraints: dict[str, Callable] = field(default_factory=dict)

    # Transformation functions
    transform: Optional[Callable[["MetaStructure"], "MetaStructure"]] = None
    compose_fn: Optional[Callable[["MetaStructure", "MetaStructure"], "MetaStructure"]] = None
    apply_fn: Optional[Callable[["MetaStructure", Any], Any]] = None

    # Emergence tracking
    emerged_from: list[str] = field(default_factory=list)
    emergence_conditions: dict = field(default_factory=dict)

    # Metadata
    version: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.structure_type == StructureType.RECURSIVE:
            self.self_reference = self.structure_id

    def __hash__(self):
        return hash(self.structure_id)

    def signature(self) -> str:
        """Get unique signature for this structure."""
        cap_sigs = sorted([c.signature() for c in self.capabilities])
        sub_sigs = sorted([s.signature() for s in self.sub_structures])
        sig = f"{self.name}:{self.structure_type.value}:{cap_sigs}:{sub_sigs}"
        return hashlib.sha256(sig.encode()).hexdigest()[:16]

    def can_compose_with(self, other: "MetaStructure") -> bool:
        """Check if this structure can compose with another."""
        # Check output-input compatibility
        my_outputs = set()
        for cap in self.capabilities:
            my_outputs.update(cap.outputs)

        other_inputs = set()
        for cap in other.capabilities:
            other_inputs.update(cap.inputs)

        return bool(my_outputs & other_inputs)

    def compose(self, other: "MetaStructure") -> "MetaStructure":
        """Compose this structure with another."""
        if self.compose_fn:
            return self.compose_fn(self, other)

        # Default composition: combine capabilities
        new_structure = MetaStructure(
            name=f"{self.name}+{other.name}",
            structure_type=StructureType.COMPOSITE,
            capabilities=self.capabilities + other.capabilities,
            sub_structures=[self, other],
            emerged_from=[self.structure_id, other.structure_id],
            metadata={"composition_type": "sequential"},
        )
        return new_structure

    def apply(self, target: Any) -> Any:
        """Apply this structure to a target."""
        if self.apply_fn:
            return self.apply_fn(self, target)

        # Default: execute capabilities in order
        result = target
        for cap in self.capabilities:
            if cap.implementation:
                result = cap.implementation(result)
        return result

    def recurse(self, depth: int = 0) -> "MetaStructure":
        """Apply structure recursively to itself."""
        if depth >= self.max_recursion:
            return self

        if self.structure_type != StructureType.RECURSIVE:
            return self

        # Apply transformation to self
        if self.transform:
            transformed = self.transform(self)
            transformed.recursion_depth = depth + 1
            return transformed.recurse(depth + 1)

        return self

    def instantiate(self, **params) -> "MetaStructure":
        """Instantiate a parametric structure with specific parameters."""
        if self.structure_type != StructureType.PARAMETRIC:
            return self

        # Validate parameters
        for name, constraint in self.parameter_constraints.items():
            if name in params and not constraint(params[name]):
                raise ValueError(f"Parameter {name} violates constraint")

        # Merge parameters
        new_params = {**self.parameters, **params}

        # Create new structure with resolved parameters
        new_structure = MetaStructure(
            name=f"{self.name}[{','.join(f'{k}={v}' for k,v in params.items())}]",
            structure_type=StructureType.COMPOSITE,
            capabilities=list(self.capabilities),
            sub_structures=list(self.sub_structures),
            parameters=new_params,
            emerged_from=[self.structure_id],
            metadata={"instantiated_from": self.structure_id, "params": params},
        )

        return new_structure

    def reflect(self) -> dict:
        """Reflect on own structure."""
        return {
            "id": self.structure_id,
            "name": self.name,
            "type": self.structure_type.value,
            "signature": self.signature(),
            "capability_count": len(self.capabilities),
            "capability_types": [c.capability_type.value for c in self.capabilities],
            "sub_structure_count": len(self.sub_structures),
            "is_recursive": self.structure_type == StructureType.RECURSIVE,
            "is_parametric": self.structure_type == StructureType.PARAMETRIC,
            "parameters": list(self.parameters.keys()),
            "version": self.version,
        }

    def evolve(self, mutation: Callable[["MetaStructure"], "MetaStructure"]) -> "MetaStructure":
        """Evolve structure through mutation."""
        evolved = mutation(self)
        evolved.version = self.version + 1
        evolved.emerged_from = [self.structure_id]
        evolved.metadata["evolved_from"] = self.structure_id
        return evolved


def compose_structures(*structures: MetaStructure) -> MetaStructure:
    """Compose multiple structures into one."""
    if not structures:
        raise ValueError("Need at least one structure")

    if len(structures) == 1:
        return structures[0]

    result = structures[0]
    for s in structures[1:]:
        result = result.compose(s)

    return result


def apply_meta(
    structure: MetaStructure,
    target: Union[MetaStructure, Any],
    recursive: bool = False,
) -> Union[MetaStructure, Any]:
    """Apply a meta-structure to a target.

    If target is a MetaStructure, produces a new structure.
    Otherwise, applies the structure's capabilities.
    """
    if recursive and structure.structure_type == StructureType.RECURSIVE:
        structure = structure.recurse()

    if isinstance(target, MetaStructure):
        # Meta-application: structure creates new structure
        return structure.compose(target)
    else:
        # Regular application
        return structure.apply(target)


# ─────────────────────────────────────────────────────────────────────────────
# Primitive Meta-Structures (building blocks)
# ─────────────────────────────────────────────────────────────────────────────

def create_identity_structure() -> MetaStructure:
    """Create identity structure (returns input unchanged)."""
    return MetaStructure(
        name="identity",
        structure_type=StructureType.PRIMITIVE,
        capabilities=[
            Capability(
                name="pass_through",
                capability_type=CapabilityType.ACTION,
                inputs=["any"],
                outputs=["any"],
                implementation=lambda x: x,
            )
        ],
    )


def create_composition_structure() -> MetaStructure:
    """Create a structure that composes other structures."""
    def compose_impl(structures: list[MetaStructure]) -> MetaStructure:
        return compose_structures(*structures)

    return MetaStructure(
        name="composer",
        structure_type=StructureType.META,
        capabilities=[
            Capability(
                name="compose",
                capability_type=CapabilityType.COMPOSITION,
                inputs=["structures"],
                outputs=["structure"],
                implementation=compose_impl,
            )
        ],
    )


def create_reflection_structure() -> MetaStructure:
    """Create a structure that reflects on other structures."""
    def reflect_impl(structure: MetaStructure) -> dict:
        return structure.reflect()

    return MetaStructure(
        name="reflector",
        structure_type=StructureType.REFLECTIVE,
        capabilities=[
            Capability(
                name="reflect",
                capability_type=CapabilityType.REFLECTION,
                inputs=["structure"],
                outputs=["reflection"],
                implementation=reflect_impl,
            )
        ],
    )


def create_recursive_applicator() -> MetaStructure:
    """Create a structure that recursively applies itself."""
    def recursive_apply(self: MetaStructure, target: Any, depth: int = 0) -> Any:
        if depth >= self.max_recursion:
            return target
        result = self.apply(target)
        if result != target:  # Changed, continue
            return recursive_apply(self, result, depth + 1)
        return result

    structure = MetaStructure(
        name="recursive_applicator",
        structure_type=StructureType.RECURSIVE,
        max_recursion=10,
    )
    structure.apply_fn = lambda s, t: recursive_apply(s, t)
    return structure


def create_generator_structure(
    generator: Callable[..., MetaStructure]
) -> MetaStructure:
    """Create a structure that generates other structures."""
    return MetaStructure(
        name="generator",
        structure_type=StructureType.META,
        capabilities=[
            Capability(
                name="generate",
                capability_type=CapabilityType.CREATION,
                inputs=["params"],
                outputs=["structure"],
                implementation=generator,
            )
        ],
    )
