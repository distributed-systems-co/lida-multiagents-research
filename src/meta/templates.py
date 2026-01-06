"""Agent templates: Parameterized patterns for instantiating agents.

Supports MCP server parameterization for injecting external tool capabilities.
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, Type, Union, TYPE_CHECKING

from .structures import MetaStructure, StructureType, Capability, CapabilityType

if TYPE_CHECKING:
    from .mcp_registry import MCPServerConfig, MCPServerCategory


class ConstraintType(str, Enum):
    """Types of parameter constraints."""
    TYPE = "type"           # Type constraint
    RANGE = "range"         # Numeric range
    ENUM = "enum"           # Enumerated values
    PATTERN = "pattern"     # Regex pattern
    PREDICATE = "predicate" # Custom predicate
    DEPENDENT = "dependent" # Depends on other params
    MCP_SERVER = "mcp_server"  # Must be a valid MCP server ID


@dataclass
class MCPBinding:
    """Binding between an agent capability and an MCP server tool."""
    server_id: str
    tool_name: str
    capability_name: str  # Name of the capability this maps to
    auto_invoke: bool = False  # Automatically invoke when capability is triggered
    transform_input: Optional[Callable[[dict], dict]] = None
    transform_output: Optional[Callable[[dict], Any]] = None


@dataclass
class MCPServerParam:
    """Parameter specification for MCP server injection."""
    server_id: str
    required: bool = False
    tool_filter: Optional[list[str]] = None  # Only bind these tools
    category_filter: Optional[list[str]] = None  # Only bind tools from these categories
    auto_bind: bool = True  # Automatically create capability bindings for tools
    custom_bindings: list[MCPBinding] = field(default_factory=list)


@dataclass
class TemplateConstraint:
    """Constraint on a template parameter."""

    constraint_type: ConstraintType
    value: Any  # The constraint value (type, range tuple, enum list, etc.)
    error_message: str = ""

    def validate(self, param_value: Any, context: dict = None) -> bool:
        """Validate a parameter value against this constraint."""
        try:
            if self.constraint_type == ConstraintType.TYPE:
                return isinstance(param_value, self.value)

            elif self.constraint_type == ConstraintType.RANGE:
                min_val, max_val = self.value
                return min_val <= param_value <= max_val

            elif self.constraint_type == ConstraintType.ENUM:
                return param_value in self.value

            elif self.constraint_type == ConstraintType.PATTERN:
                import re
                return bool(re.match(self.value, str(param_value)))

            elif self.constraint_type == ConstraintType.PREDICATE:
                return self.value(param_value)

            elif self.constraint_type == ConstraintType.DEPENDENT:
                # value is a function that takes (param_value, context)
                return self.value(param_value, context or {})

            return True

        except Exception:
            return False


@dataclass
class TemplateParameter:
    """A parameter in an agent template."""

    name: str
    param_type: Type = Any
    default: Any = None
    required: bool = False
    description: str = ""
    constraints: list[TemplateConstraint] = field(default_factory=list)

    # For nested/composite parameters
    nested_params: list["TemplateParameter"] = field(default_factory=list)

    def validate(self, value: Any, context: dict = None) -> tuple[bool, str]:
        """Validate a value for this parameter."""
        for constraint in self.constraints:
            if not constraint.validate(value, context):
                return False, constraint.error_message or f"Constraint failed for {self.name}"
        return True, ""

    def resolve(self, value: Any = None) -> Any:
        """Resolve parameter to final value."""
        if value is not None:
            return value
        if self.required and self.default is None:
            raise ValueError(f"Required parameter {self.name} not provided")
        return self.default


@dataclass
class AgentTemplate:
    """A template for creating agent instances.

    Templates are parameterized patterns that, when instantiated,
    produce agent structures with specific capabilities.

    Supports MCP server parameterization for external tool injection.
    """

    template_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    name: str = ""
    description: str = ""

    # Template structure
    base_structure: Optional[MetaStructure] = None
    parameters: list[TemplateParameter] = field(default_factory=list)

    # Capability templates (capabilities to add during instantiation)
    capability_templates: list[dict] = field(default_factory=list)

    # MCP server specifications
    mcp_servers: list[MCPServerParam] = field(default_factory=list)
    mcp_bindings: list[MCPBinding] = field(default_factory=list)

    # Sub-templates (for hierarchical composition)
    sub_templates: list["AgentTemplate"] = field(default_factory=list)

    # Composition rules
    composition_rules: list[Callable] = field(default_factory=list)

    # Post-instantiation hooks
    post_hooks: list[Callable[[MetaStructure, dict], MetaStructure]] = field(default_factory=list)

    # Derivation tracking
    derived_from: Optional[str] = None
    derivations: list[str] = field(default_factory=list)

    # Metadata
    version: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    def validate_params(self, params: dict) -> tuple[bool, list[str]]:
        """Validate all parameters."""
        errors = []

        for param in self.parameters:
            value = params.get(param.name)

            if value is None and param.required:
                errors.append(f"Missing required parameter: {param.name}")
                continue

            if value is not None:
                valid, error = param.validate(value, params)
                if not valid:
                    errors.append(error)

        return len(errors) == 0, errors

    def instantiate(self, **params) -> MetaStructure:
        """Instantiate the template with given parameters."""
        # Validate parameters
        valid, errors = self.validate_params(params)
        if not valid:
            raise ValueError(f"Invalid parameters: {errors}")

        # Resolve all parameters
        resolved = {}
        for param in self.parameters:
            resolved[param.name] = param.resolve(params.get(param.name))

        # Start with base structure or create new
        if self.base_structure:
            structure = copy.deepcopy(self.base_structure)
        else:
            structure = MetaStructure(
                name=f"{self.name}_instance",
                structure_type=StructureType.COMPOSITE,
            )

        # Apply capability templates
        for cap_template in self.capability_templates:
            cap = self._resolve_capability(cap_template, resolved)
            structure.capabilities.append(cap)

        # Inject MCP server capabilities
        mcp_caps, mcp_bindings = self._inject_mcp_capabilities(resolved)
        structure.capabilities.extend(mcp_caps)

        # Instantiate and compose sub-templates
        for sub_template in self.sub_templates:
            sub_params = self._extract_sub_params(sub_template.name, resolved)
            sub_structure = sub_template.instantiate(**sub_params)
            structure.sub_structures.append(sub_structure)

        # Apply composition rules
        for rule in self.composition_rules:
            structure = rule(structure, resolved)

        # Apply post-hooks
        for hook in self.post_hooks:
            structure = hook(structure, resolved)

        # Set metadata
        structure.metadata["template_id"] = self.template_id
        structure.metadata["template_name"] = self.name
        structure.metadata["instantiation_params"] = resolved
        structure.metadata["instantiated_at"] = datetime.now().isoformat()
        structure.metadata["mcp_servers"] = [s.server_id for s in self.mcp_servers]
        structure.metadata["mcp_bindings"] = [
            {"server": b.server_id, "tool": b.tool_name, "capability": b.capability_name}
            for b in mcp_bindings
        ]

        return structure

    def _inject_mcp_capabilities(self, params: dict) -> tuple[list[Capability], list[MCPBinding]]:
        """Inject capabilities from MCP servers."""
        from .mcp_registry import get_mcp_registry, MCPServerCategory

        capabilities = []
        bindings = []
        registry = get_mcp_registry()

        for mcp_param in self.mcp_servers:
            server = registry.get_server(mcp_param.server_id)
            if not server:
                if mcp_param.required:
                    raise ValueError(f"Required MCP server not found: {mcp_param.server_id}")
                continue

            # Filter tools
            tools = server.tools
            if mcp_param.tool_filter:
                tools = [t for t in tools if t.name in mcp_param.tool_filter]
            if mcp_param.category_filter:
                cats = [MCPServerCategory(c) for c in mcp_param.category_filter]
                tools = [t for t in tools if t.category in cats]

            # Create capabilities and bindings for each tool
            if mcp_param.auto_bind:
                for tool in tools:
                    cap = Capability(
                        name=f"mcp_{server.server_id}_{tool.name}",
                        capability_type=CapabilityType.ACTION,
                        inputs=list(tool.input_schema.keys()) if tool.input_schema else ["input"],
                        outputs=["result"],
                        metadata={
                            "mcp_server": server.server_id,
                            "mcp_tool": tool.name,
                            "mcp_description": tool.description,
                            "mcp_category": tool.category.value,
                        },
                    )
                    capabilities.append(cap)

                    binding = MCPBinding(
                        server_id=server.server_id,
                        tool_name=tool.name,
                        capability_name=cap.name,
                    )
                    bindings.append(binding)

        # Add explicit bindings
        bindings.extend(self.mcp_bindings)

        return capabilities, bindings

    def with_mcp_server(
        self,
        server_id: str,
        required: bool = False,
        tool_filter: Optional[list[str]] = None,
        category_filter: Optional[list[str]] = None,
    ) -> "AgentTemplate":
        """Return a new template with an MCP server added."""
        new_template = copy.deepcopy(self)
        new_template.mcp_servers.append(MCPServerParam(
            server_id=server_id,
            required=required,
            tool_filter=tool_filter,
            category_filter=category_filter,
        ))
        return new_template

    def _resolve_capability(self, template: dict, params: dict) -> Capability:
        """Resolve a capability template with parameters."""
        resolved = {}
        for key, value in template.items():
            if isinstance(value, str) and value.startswith("$"):
                param_name = value[1:]
                resolved[key] = params.get(param_name, value)
            elif callable(value):
                resolved[key] = value(params)
            else:
                resolved[key] = value

        return Capability(**resolved)

    def _extract_sub_params(self, prefix: str, params: dict) -> dict:
        """Extract parameters for a sub-template."""
        sub_params = {}
        prefix_dot = f"{prefix}."
        for key, value in params.items():
            if key.startswith(prefix_dot):
                sub_key = key[len(prefix_dot):]
                sub_params[sub_key] = value
        return sub_params

    def derive(
        self,
        name: str,
        modifications: Callable[["AgentTemplate"], None] = None,
        **extra_params,
    ) -> "AgentTemplate":
        """Create a derived template."""
        derived = copy.deepcopy(self)
        derived.template_id = str(uuid.uuid4())[:12]
        derived.name = name
        derived.derived_from = self.template_id
        derived.version = 1
        derived.created_at = datetime.now()

        # Add extra parameters
        for pname, pdef in extra_params.items():
            if isinstance(pdef, TemplateParameter):
                derived.parameters.append(pdef)
            else:
                derived.parameters.append(TemplateParameter(
                    name=pname,
                    default=pdef,
                ))

        # Apply modifications
        if modifications:
            modifications(derived)

        # Track derivation
        self.derivations.append(derived.template_id)

        return derived

    def compose_with(self, other: "AgentTemplate") -> "AgentTemplate":
        """Compose this template with another."""
        composed = AgentTemplate(
            name=f"{self.name}+{other.name}",
            description=f"Composition of {self.name} and {other.name}",
            parameters=self.parameters + other.parameters,
            capability_templates=self.capability_templates + other.capability_templates,
            sub_templates=self.sub_templates + other.sub_templates,
            composition_rules=self.composition_rules + other.composition_rules,
            post_hooks=self.post_hooks + other.post_hooks,
            derived_from=self.template_id,
            metadata={
                "composed_from": [self.template_id, other.template_id],
                "composition_type": "merge",
            },
        )
        return composed


def instantiate_template(
    template: AgentTemplate,
    params: dict = None,
    recursive: bool = False,
) -> MetaStructure:
    """Instantiate a template, optionally with recursive sub-template instantiation."""
    params = params or {}
    structure = template.instantiate(**params)

    if recursive:
        # Recursively instantiate any nested parametric structures
        for i, sub in enumerate(structure.sub_structures):
            if sub.structure_type == StructureType.PARAMETRIC:
                structure.sub_structures[i] = sub.instantiate()

    return structure


# ─────────────────────────────────────────────────────────────────────────────
# Pre-built Agent Templates
# ─────────────────────────────────────────────────────────────────────────────

def create_observer_template() -> AgentTemplate:
    """Template for observer agents that perceive world state."""
    return AgentTemplate(
        name="observer",
        description="Agent that observes and reports on world state",
        parameters=[
            TemplateParameter(
                name="focus_domains",
                param_type=list,
                default=["general"],
                description="Domains to focus observation on",
            ),
            TemplateParameter(
                name="sensitivity",
                param_type=float,
                default=0.5,
                constraints=[TemplateConstraint(ConstraintType.RANGE, (0.0, 1.0))],
            ),
        ],
        capability_templates=[
            {
                "name": "observe",
                "capability_type": CapabilityType.PERCEPTION,
                "inputs": ["world_state"],
                "outputs": ["observations"],
            },
            {
                "name": "filter",
                "capability_type": CapabilityType.REASONING,
                "inputs": ["observations", "$sensitivity"],
                "outputs": ["filtered_observations"],
            },
            {
                "name": "report",
                "capability_type": CapabilityType.COMMUNICATION,
                "inputs": ["filtered_observations"],
                "outputs": ["report"],
            },
        ],
    )


def create_reasoner_template() -> AgentTemplate:
    """Template for reasoning agents."""
    return AgentTemplate(
        name="reasoner",
        description="Agent that performs reasoning and analysis",
        parameters=[
            TemplateParameter(
                name="reasoning_depth",
                param_type=int,
                default=3,
                constraints=[TemplateConstraint(ConstraintType.RANGE, (1, 10))],
            ),
            TemplateParameter(
                name="reasoning_strategies",
                param_type=list,
                default=["deductive", "inductive"],
            ),
        ],
        capability_templates=[
            {
                "name": "analyze",
                "capability_type": CapabilityType.REASONING,
                "inputs": ["data", "context"],
                "outputs": ["analysis"],
            },
            {
                "name": "hypothesize",
                "capability_type": CapabilityType.REASONING,
                "inputs": ["analysis"],
                "outputs": ["hypotheses"],
            },
            {
                "name": "conclude",
                "capability_type": CapabilityType.REASONING,
                "inputs": ["hypotheses", "evidence"],
                "outputs": ["conclusions"],
            },
        ],
    )


def create_actor_template() -> AgentTemplate:
    """Template for actor agents that take actions."""
    return AgentTemplate(
        name="actor",
        description="Agent that executes actions in the world",
        parameters=[
            TemplateParameter(
                name="action_types",
                param_type=list,
                default=["modify", "create", "delete"],
            ),
            TemplateParameter(
                name="caution_level",
                param_type=float,
                default=0.5,
                constraints=[TemplateConstraint(ConstraintType.RANGE, (0.0, 1.0))],
            ),
        ],
        capability_templates=[
            {
                "name": "plan",
                "capability_type": CapabilityType.REASONING,
                "inputs": ["goal", "context"],
                "outputs": ["action_plan"],
            },
            {
                "name": "validate",
                "capability_type": CapabilityType.REASONING,
                "inputs": ["action_plan", "$caution_level"],
                "outputs": ["validated_plan"],
            },
            {
                "name": "execute",
                "capability_type": CapabilityType.ACTION,
                "inputs": ["validated_plan"],
                "outputs": ["action_result"],
            },
        ],
    )


def create_meta_agent_template() -> AgentTemplate:
    """Template for meta-agents that create/modify other agents."""
    return AgentTemplate(
        name="meta_agent",
        description="Agent that creates and modifies other agents",
        parameters=[
            TemplateParameter(
                name="creativity_level",
                param_type=float,
                default=0.7,
            ),
            TemplateParameter(
                name="templates_available",
                param_type=list,
                default=[],
            ),
        ],
        capability_templates=[
            {
                "name": "design",
                "capability_type": CapabilityType.CREATION,
                "inputs": ["requirements"],
                "outputs": ["agent_design"],
            },
            {
                "name": "compose",
                "capability_type": CapabilityType.COMPOSITION,
                "inputs": ["structures"],
                "outputs": ["composed_structure"],
            },
            {
                "name": "instantiate",
                "capability_type": CapabilityType.CREATION,
                "inputs": ["template", "params"],
                "outputs": ["agent_instance"],
            },
            {
                "name": "reflect",
                "capability_type": CapabilityType.REFLECTION,
                "inputs": ["structure"],
                "outputs": ["analysis"],
            },
            {
                "name": "evolve",
                "capability_type": CapabilityType.META,
                "inputs": ["structure", "mutation"],
                "outputs": ["evolved_structure"],
            },
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# MCP-Enabled Agent Templates
# ─────────────────────────────────────────────────────────────────────────────

def create_research_agent_template() -> AgentTemplate:
    """Template for research agents with Jina MCP search/reading capabilities."""
    return AgentTemplate(
        name="research_agent",
        description="Agent with web search, reading, and research capabilities via Jina MCP",
        parameters=[
            TemplateParameter(
                name="research_depth",
                param_type=int,
                default=3,
                constraints=[TemplateConstraint(ConstraintType.RANGE, (1, 10))],
                description="Depth of research to conduct",
            ),
            TemplateParameter(
                name="source_types",
                param_type=list,
                default=["web", "arxiv"],
                description="Types of sources to search",
            ),
        ],
        mcp_servers=[
            MCPServerParam(
                server_id="jina-mcp",
                required=False,
                category_filter=["search", "reading", "research"],
            ),
        ],
        capability_templates=[
            {
                "name": "formulate_query",
                "capability_type": CapabilityType.REASONING,
                "inputs": ["topic", "context"],
                "outputs": ["search_queries"],
            },
            {
                "name": "synthesize",
                "capability_type": CapabilityType.REASONING,
                "inputs": ["sources", "context"],
                "outputs": ["synthesis"],
            },
            {
                "name": "cite",
                "capability_type": CapabilityType.COMMUNICATION,
                "inputs": ["synthesis"],
                "outputs": ["cited_report"],
            },
        ],
    )


def create_embedding_agent_template() -> AgentTemplate:
    """Template for agents that work with embeddings and semantic similarity."""
    return AgentTemplate(
        name="embedding_agent",
        description="Agent with embedding and semantic similarity capabilities via Jina MCP",
        parameters=[
            TemplateParameter(
                name="embedding_model",
                param_type=str,
                default="jina-embeddings-v3",
            ),
            TemplateParameter(
                name="similarity_threshold",
                param_type=float,
                default=0.8,
                constraints=[TemplateConstraint(ConstraintType.RANGE, (0.0, 1.0))],
            ),
        ],
        mcp_servers=[
            MCPServerParam(
                server_id="jina-mcp",
                required=False,
                category_filter=["embedding", "reranking"],
            ),
        ],
        capability_templates=[
            {
                "name": "embed",
                "capability_type": CapabilityType.PERCEPTION,
                "inputs": ["content"],
                "outputs": ["embeddings"],
            },
            {
                "name": "compare",
                "capability_type": CapabilityType.REASONING,
                "inputs": ["embedding_a", "embedding_b"],
                "outputs": ["similarity_score"],
            },
            {
                "name": "cluster",
                "capability_type": CapabilityType.REASONING,
                "inputs": ["embeddings"],
                "outputs": ["clusters"],
            },
        ],
    )


def create_filesystem_agent_template() -> AgentTemplate:
    """Template for agents that operate on the filesystem."""
    return AgentTemplate(
        name="filesystem_agent",
        description="Agent with filesystem read/write capabilities via MCP",
        parameters=[
            TemplateParameter(
                name="root_path",
                param_type=str,
                default=".",
                description="Root path for filesystem operations",
            ),
            TemplateParameter(
                name="allowed_extensions",
                param_type=list,
                default=["*"],
                description="Allowed file extensions",
            ),
        ],
        mcp_servers=[
            MCPServerParam(
                server_id="filesystem-mcp",
                required=False,
                category_filter=["filesystem"],
            ),
        ],
        capability_templates=[
            {
                "name": "navigate",
                "capability_type": CapabilityType.PERCEPTION,
                "inputs": ["path"],
                "outputs": ["directory_contents"],
            },
            {
                "name": "read",
                "capability_type": CapabilityType.PERCEPTION,
                "inputs": ["file_path"],
                "outputs": ["file_contents"],
            },
            {
                "name": "write",
                "capability_type": CapabilityType.ACTION,
                "inputs": ["file_path", "content"],
                "outputs": ["write_result"],
            },
        ],
    )


def create_multimodal_research_agent_template() -> AgentTemplate:
    """Template for multimodal research agents with full Jina capabilities."""
    return AgentTemplate(
        name="multimodal_research_agent",
        description="Full-featured research agent with search, reading, embeddings, and vision",
        parameters=[
            TemplateParameter(
                name="research_depth",
                param_type=int,
                default=5,
                constraints=[TemplateConstraint(ConstraintType.RANGE, (1, 10))],
            ),
            TemplateParameter(
                name="include_images",
                param_type=bool,
                default=True,
            ),
            TemplateParameter(
                name="max_sources",
                param_type=int,
                default=20,
            ),
        ],
        mcp_servers=[
            MCPServerParam(
                server_id="jina-mcp",
                required=False,
                auto_bind=True,  # Bind ALL Jina tools
            ),
        ],
        capability_templates=[
            {
                "name": "plan_research",
                "capability_type": CapabilityType.REASONING,
                "inputs": ["topic", "constraints"],
                "outputs": ["research_plan"],
            },
            {
                "name": "gather_sources",
                "capability_type": CapabilityType.PERCEPTION,
                "inputs": ["research_plan"],
                "outputs": ["sources"],
            },
            {
                "name": "analyze_visual",
                "capability_type": CapabilityType.PERCEPTION,
                "inputs": ["images"],
                "outputs": ["visual_analysis"],
            },
            {
                "name": "synthesize",
                "capability_type": CapabilityType.REASONING,
                "inputs": ["sources", "visual_analysis"],
                "outputs": ["synthesis"],
            },
            {
                "name": "generate_report",
                "capability_type": CapabilityType.CREATION,
                "inputs": ["synthesis"],
                "outputs": ["research_report"],
            },
        ],
    )


def create_memory_agent_template() -> AgentTemplate:
    """Template for agents with persistent memory capabilities."""
    return AgentTemplate(
        name="memory_agent",
        description="Agent with persistent memory and knowledge graph capabilities",
        parameters=[
            TemplateParameter(
                name="memory_scope",
                param_type=str,
                default="session",
                description="Scope of memory: session, persistent, or shared",
            ),
            TemplateParameter(
                name="max_memories",
                param_type=int,
                default=1000,
            ),
        ],
        mcp_servers=[
            MCPServerParam(
                server_id="memory-mcp",
                required=False,
                category_filter=["memory"],
            ),
        ],
        capability_templates=[
            {
                "name": "remember",
                "capability_type": CapabilityType.ACTION,
                "inputs": ["information", "context"],
                "outputs": ["memory_id"],
            },
            {
                "name": "recall",
                "capability_type": CapabilityType.PERCEPTION,
                "inputs": ["query", "context"],
                "outputs": ["memories"],
            },
            {
                "name": "associate",
                "capability_type": CapabilityType.REASONING,
                "inputs": ["memory_a", "memory_b", "relation"],
                "outputs": ["association"],
            },
            {
                "name": "forget",
                "capability_type": CapabilityType.ACTION,
                "inputs": ["memory_id"],
                "outputs": ["forget_result"],
            },
        ],
    )


def create_composite_mcp_agent_template(
    name: str,
    description: str,
    mcp_server_ids: list[str],
    category_filters: Optional[dict[str, list[str]]] = None,
) -> AgentTemplate:
    """Create a custom agent template with multiple MCP servers.

    Args:
        name: Template name
        description: Template description
        mcp_server_ids: List of MCP server IDs to include
        category_filters: Optional dict mapping server_id -> category filter list
    """
    category_filters = category_filters or {}

    mcp_servers = [
        MCPServerParam(
            server_id=sid,
            required=False,
            category_filter=category_filters.get(sid),
            auto_bind=True,
        )
        for sid in mcp_server_ids
    ]

    return AgentTemplate(
        name=name,
        description=description,
        mcp_servers=mcp_servers,
        capability_templates=[
            {
                "name": "process",
                "capability_type": CapabilityType.REASONING,
                "inputs": ["input"],
                "outputs": ["output"],
            },
        ],
    )
