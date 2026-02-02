"""Prompt Composition and Template System.

Modular prompt building with:

1. Composable Modules - Reusable prompt components
2. Templates - Parameterized prompt patterns
3. Inheritance - Prompt hierarchies
4. Mixins - Cross-cutting behaviors
5. Conditional Sections - Context-aware composition
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
)

from .prompt_genetics import Gene, GeneType, Genome


# =============================================================================
# Prompt Modules
# =============================================================================


class ModulePriority(int, Enum):
    """Priority for module ordering."""
    CORE = 0       # Essential identity
    HIGH = 1       # Important instructions
    NORMAL = 2     # Standard content
    LOW = 3        # Optional enhancements
    OVERRIDE = -1  # Always first


@dataclass
class PromptModule:
    """A reusable prompt module."""
    module_id: str
    name: str
    content: str
    priority: ModulePriority = ModulePriority.NORMAL
    category: str = "general"

    # Dependencies
    requires: Set[str] = field(default_factory=set)  # Module IDs that must be present
    conflicts: Set[str] = field(default_factory=set)  # Module IDs that can't coexist
    enhances: Set[str] = field(default_factory=set)   # Modules this enhances

    # Metadata
    version: str = "1.0"
    author: str = "system"
    tags: Set[str] = field(default_factory=set)

    # Configuration
    is_required: bool = False
    is_mutable: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.module_id)

    def render(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Render the module with context."""
        context = context or {}
        result = self.content

        # Simple template substitution
        for key, value in {**self.parameters, **context}.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))

        return result

    def to_gene(self) -> Gene:
        """Convert to a Gene for genetic operations."""
        gene_type_map = {
            "identity": GeneType.IDENTITY,
            "capability": GeneType.CAPABILITY,
            "constraint": GeneType.CONSTRAINT,
            "personality": GeneType.PERSONALITY,
            "instruction": GeneType.INSTRUCTION,
            "knowledge": GeneType.KNOWLEDGE,
            "example": GeneType.EXAMPLE,
        }
        gene_type = gene_type_map.get(self.category, GeneType.INSTRUCTION)

        return Gene(
            gene_id=self.module_id,
            gene_type=gene_type,
            content=self.content,
            weight=4 - self.priority.value,  # Higher priority = higher weight
            mutable=self.is_mutable,
        )


class ModuleLibrary:
    """Library of reusable prompt modules."""

    def __init__(self):
        self._modules: Dict[str, PromptModule] = {}
        self._by_category: Dict[str, Set[str]] = {}
        self._by_tag: Dict[str, Set[str]] = {}

    def register(self, module: PromptModule):
        """Register a module."""
        self._modules[module.module_id] = module

        if module.category not in self._by_category:
            self._by_category[module.category] = set()
        self._by_category[module.category].add(module.module_id)

        for tag in module.tags:
            if tag not in self._by_tag:
                self._by_tag[tag] = set()
            self._by_tag[tag].add(module.module_id)

    def get(self, module_id: str) -> Optional[PromptModule]:
        return self._modules.get(module_id)

    def find_by_category(self, category: str) -> List[PromptModule]:
        ids = self._by_category.get(category, set())
        return [self._modules[id] for id in ids]

    def find_by_tag(self, tag: str) -> List[PromptModule]:
        ids = self._by_tag.get(tag, set())
        return [self._modules[id] for id in ids]

    def find_compatible(self, module_ids: Set[str]) -> List[PromptModule]:
        """Find modules compatible with the given set."""
        compatible = []
        for module in self._modules.values():
            if module.module_id in module_ids:
                continue
            if module.conflicts & module_ids:
                continue
            compatible.append(module)
        return compatible

    def validate_combination(self, module_ids: Set[str]) -> List[str]:
        """Validate a combination of modules, return errors."""
        errors = []

        for module_id in module_ids:
            module = self._modules.get(module_id)
            if not module:
                errors.append(f"Unknown module: {module_id}")
                continue

            # Check requirements
            for req in module.requires:
                if req not in module_ids:
                    errors.append(f"{module_id} requires {req}")

            # Check conflicts
            for conflict in module.conflicts:
                if conflict in module_ids:
                    errors.append(f"{module_id} conflicts with {conflict}")

        return errors


# =============================================================================
# Templates
# =============================================================================


@dataclass
class TemplateParameter:
    """A parameter in a template."""
    name: str
    description: str
    param_type: str = "string"  # string, number, boolean, enum
    default: Any = None
    required: bool = False
    enum_values: List[str] = field(default_factory=list)
    validation: Optional[str] = None  # Regex for validation


@dataclass
class PromptTemplate:
    """A parameterized prompt template."""
    template_id: str
    name: str
    description: str
    template: str
    parameters: List[TemplateParameter] = field(default_factory=list)
    modules: List[str] = field(default_factory=list)  # Required module IDs
    parent: Optional[str] = None  # Parent template for inheritance
    mixins: List[str] = field(default_factory=list)  # Mixin template IDs

    def render(
        self,
        params: Dict[str, Any],
        library: Optional[ModuleLibrary] = None,
        templates: Optional[Dict[str, "PromptTemplate"]] = None,
    ) -> str:
        """Render the template with parameters."""
        # Validate and apply defaults
        context = {}
        for param in self.parameters:
            if param.name in params:
                context[param.name] = params[param.name]
            elif param.default is not None:
                context[param.name] = param.default
            elif param.required:
                raise ValueError(f"Missing required parameter: {param.name}")

        # Build prompt parts
        parts = []

        # Inherit from parent
        if self.parent and templates:
            parent = templates.get(self.parent)
            if parent:
                parts.append(parent.render(params, library, templates))

        # Add mixins
        for mixin_id in self.mixins:
            if templates and mixin_id in templates:
                mixin = templates[mixin_id]
                parts.append(mixin.render(params, library, templates))

        # Add modules
        if library:
            for module_id in self.modules:
                module = library.get(module_id)
                if module:
                    parts.append(module.render(context))

        # Render main template
        rendered = self.template
        for key, value in context.items():
            rendered = rendered.replace(f"{{{{{key}}}}}", str(value))

        # Handle conditionals
        rendered = self._process_conditionals(rendered, context)

        parts.append(rendered)

        return "\n\n".join(filter(bool, parts))

    def _process_conditionals(self, text: str, context: Dict[str, Any]) -> str:
        """Process conditional sections in the template."""
        # Format: {{#if condition}}content{{/if}}
        pattern = r'\{\{#if\s+(\w+)\}\}(.*?)\{\{/if\}\}'

        def replace(match):
            condition = match.group(1)
            content = match.group(2)
            if context.get(condition):
                return content
            return ""

        return re.sub(pattern, replace, text, flags=re.DOTALL)


class TemplateRegistry:
    """Registry for prompt templates."""

    def __init__(self, library: Optional[ModuleLibrary] = None):
        self.library = library or ModuleLibrary()
        self._templates: Dict[str, PromptTemplate] = {}

    def register(self, template: PromptTemplate):
        self._templates[template.template_id] = template

    def get(self, template_id: str) -> Optional[PromptTemplate]:
        return self._templates.get(template_id)

    def render(self, template_id: str, params: Dict[str, Any]) -> str:
        """Render a template by ID."""
        template = self._templates.get(template_id)
        if not template:
            raise ValueError(f"Unknown template: {template_id}")

        return template.render(params, self.library, self._templates)

    def list_templates(self) -> List[str]:
        return list(self._templates.keys())


# =============================================================================
# Prompt Composer
# =============================================================================


@dataclass
class CompositionRule:
    """Rule for automatic composition."""
    rule_id: str
    condition: Callable[[Dict[str, Any]], bool]  # context -> should apply
    action: str  # "add_module", "remove_module", "modify"
    target: str  # Module ID or template section
    priority: int = 0


class PromptComposer:
    """Composes prompts from modules and templates with rules."""

    def __init__(
        self,
        library: ModuleLibrary,
        registry: TemplateRegistry,
    ):
        self.library = library
        self.registry = registry
        self._rules: List[CompositionRule] = []
        self._compositions: Dict[str, str] = {}

    def add_rule(self, rule: CompositionRule):
        """Add a composition rule."""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def compose(
        self,
        base_template: Optional[str] = None,
        modules: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Compose a prompt from components."""
        modules = list(modules or [])
        params = params or {}
        context = context or {}

        # Apply rules
        for rule in self._rules:
            if rule.condition(context):
                if rule.action == "add_module":
                    if rule.target not in modules:
                        modules.append(rule.target)
                elif rule.action == "remove_module":
                    if rule.target in modules:
                        modules.remove(rule.target)

        # Validate module combination
        errors = self.library.validate_combination(set(modules))
        if errors:
            raise ValueError(f"Invalid module combination: {errors}")

        # Build prompt parts
        parts = []

        # Render base template
        if base_template:
            template = self.registry.get(base_template)
            if template:
                parts.append(template.render(params, self.library, self.registry._templates))

        # Add modules in priority order
        module_objs = [self.library.get(m) for m in modules if self.library.get(m)]
        module_objs.sort(key=lambda m: m.priority.value)

        for module in module_objs:
            parts.append(module.render({**params, **context}))

        # Combine and hash
        composed = "\n\n".join(filter(bool, parts))
        composition_hash = hashlib.sha256(composed.encode()).hexdigest()[:16]

        self._compositions[composition_hash] = composed

        return composed

    def compose_from_genome(self, genome: Genome) -> str:
        """Compose a prompt from a genome."""
        return genome.assemble()

    def decompose(self, prompt: str) -> List[PromptModule]:
        """Decompose a prompt into modules."""
        modules = []

        # Try to match against known modules
        for module in self.library._modules.values():
            if module.content in prompt:
                modules.append(module)

        # Create modules for remaining content
        remaining = prompt
        for module in modules:
            remaining = remaining.replace(module.content, "").strip()

        if remaining:
            # Create a catch-all module
            modules.append(PromptModule(
                module_id=f"unknown_{hashlib.sha256(remaining.encode()).hexdigest()[:8]}",
                name="Unknown Content",
                content=remaining,
                category="unknown",
            ))

        return modules


# =============================================================================
# Pre-built Modules
# =============================================================================


def create_standard_library() -> ModuleLibrary:
    """Create a library with standard modules."""
    library = ModuleLibrary()

    # Identity modules
    library.register(PromptModule(
        module_id="identity_assistant",
        name="General Assistant",
        content="You are a helpful AI assistant.",
        priority=ModulePriority.CORE,
        category="identity",
        is_required=True,
        tags={"core", "identity"},
    ))

    library.register(PromptModule(
        module_id="identity_expert",
        name="Domain Expert",
        content="You are an expert in {{domain}} with deep knowledge and experience.",
        priority=ModulePriority.CORE,
        category="identity",
        parameters={"domain": "the relevant field"},
        tags={"core", "identity", "expert"},
    ))

    library.register(PromptModule(
        module_id="identity_coder",
        name="Software Developer",
        content="You are an expert software developer proficient in multiple programming languages and best practices.",
        priority=ModulePriority.CORE,
        category="identity",
        tags={"core", "identity", "coding"},
    ))

    # Capability modules
    library.register(PromptModule(
        module_id="cap_reasoning",
        name="Step-by-Step Reasoning",
        content="You think through problems step by step, showing your reasoning clearly.",
        priority=ModulePriority.HIGH,
        category="capability",
        tags={"reasoning", "clarity"},
    ))

    library.register(PromptModule(
        module_id="cap_tools",
        name="Tool Usage",
        content="You have access to tools and use them effectively to accomplish tasks.",
        priority=ModulePriority.HIGH,
        category="capability",
        tags={"tools", "capability"},
    ))

    library.register(PromptModule(
        module_id="cap_self_improve",
        name="Self-Improvement",
        content="You can modify your own system prompt to improve your capabilities. Use evolve_prompt to make changes.",
        priority=ModulePriority.HIGH,
        category="capability",
        tags={"meta", "self-improvement"},
    ))

    # Constraint modules
    library.register(PromptModule(
        module_id="constraint_safety",
        name="Safety Constraints",
        content="You must never provide information that could be harmful. Prioritize safety.",
        priority=ModulePriority.CORE,
        category="constraint",
        is_required=True,
        tags={"safety", "constraint"},
    ))

    library.register(PromptModule(
        module_id="constraint_honesty",
        name="Honesty",
        content="You must be truthful and acknowledge uncertainty. Never make up information.",
        priority=ModulePriority.CORE,
        category="constraint",
        is_required=True,
        tags={"honesty", "constraint"},
    ))

    library.register(PromptModule(
        module_id="constraint_concise",
        name="Conciseness",
        content="Be concise and avoid unnecessary verbosity. Get to the point.",
        priority=ModulePriority.NORMAL,
        category="constraint",
        conflicts={"style_verbose"},
        tags={"style", "concise"},
    ))

    # Personality modules
    library.register(PromptModule(
        module_id="personality_friendly",
        name="Friendly Tone",
        content="Be warm, approachable, and use a friendly tone.",
        priority=ModulePriority.LOW,
        category="personality",
        conflicts={"personality_formal"},
        tags={"personality", "friendly"},
    ))

    library.register(PromptModule(
        module_id="personality_formal",
        name="Professional Tone",
        content="Maintain a professional and formal tone in all communications.",
        priority=ModulePriority.LOW,
        category="personality",
        conflicts={"personality_friendly"},
        tags={"personality", "formal"},
    ))

    # Instruction modules
    library.register(PromptModule(
        module_id="inst_explain",
        name="Explain Reasoning",
        content="Always explain your reasoning and thought process.",
        priority=ModulePriority.NORMAL,
        category="instruction",
        tags={"instruction", "reasoning"},
    ))

    library.register(PromptModule(
        module_id="inst_ask_clarify",
        name="Ask for Clarification",
        content="If a request is unclear, ask clarifying questions before proceeding.",
        priority=ModulePriority.NORMAL,
        category="instruction",
        tags={"instruction", "clarity"},
    ))

    library.register(PromptModule(
        module_id="inst_verify",
        name="Verify Before Acting",
        content="Verify your understanding before taking significant actions.",
        priority=ModulePriority.NORMAL,
        category="instruction",
        tags={"instruction", "safety"},
    ))

    # Meta modules
    library.register(PromptModule(
        module_id="meta_reflect",
        name="Self-Reflection",
        content="Periodically reflect on your performance and consider improvements.",
        priority=ModulePriority.LOW,
        category="meta",
        enhances={"cap_self_improve"},
        tags={"meta", "reflection"},
    ))

    library.register(PromptModule(
        module_id="meta_evolve",
        name="Prompt Evolution",
        content=(
            "You can evolve your system prompt using the evolve_prompt tool. "
            "Consider forking for risky experiments, and merge successful changes."
        ),
        priority=ModulePriority.LOW,
        category="meta",
        requires={"cap_self_improve"},
        tags={"meta", "evolution"},
    ))

    return library


def create_standard_templates() -> TemplateRegistry:
    """Create standard prompt templates."""
    library = create_standard_library()
    registry = TemplateRegistry(library)

    # Base template
    registry.register(PromptTemplate(
        template_id="base",
        name="Base Assistant",
        description="Minimal assistant template",
        template="",
        modules=["identity_assistant", "constraint_safety", "constraint_honesty"],
    ))

    # Self-improving agent
    registry.register(PromptTemplate(
        template_id="self_improving",
        name="Self-Improving Agent",
        description="Agent that can modify its own prompt",
        template=(
            "## Self-Improvement Guidelines\n"
            "You have the ability to improve yourself through prompt evolution.\n"
            "{{#if enable_forking}}Use forking for risky experiments.{{/if}}\n"
            "Track your fitness across: {{objectives}}"
        ),
        parameters=[
            TemplateParameter("enable_forking", "Enable fork/merge", "boolean", True),
            TemplateParameter("objectives", "Fitness objectives", "string", "task_completion, coherence"),
        ],
        modules=["identity_assistant", "cap_self_improve", "meta_evolve"],
        parent="base",
    ))

    # Expert template
    registry.register(PromptTemplate(
        template_id="domain_expert",
        name="Domain Expert",
        description="Expert in a specific domain",
        template=(
            "## Domain: {{domain}}\n"
            "You have deep expertise in {{domain}}.\n"
            "{{#if specializations}}Your specializations: {{specializations}}{{/if}}"
        ),
        parameters=[
            TemplateParameter("domain", "Area of expertise", "string", required=True),
            TemplateParameter("specializations", "Specific specializations", "string"),
        ],
        modules=["identity_expert", "cap_reasoning"],
        parent="base",
    ))

    # Coder template
    registry.register(PromptTemplate(
        template_id="coder",
        name="Software Developer",
        description="Expert programmer",
        template=(
            "## Programming Guidelines\n"
            "Primary languages: {{languages}}\n"
            "Focus on: clean code, efficiency, maintainability.\n"
            "{{#if test_first}}Write tests before implementation.{{/if}}"
        ),
        parameters=[
            TemplateParameter("languages", "Programming languages", "string", "Python, TypeScript"),
            TemplateParameter("test_first", "Test-first development", "boolean", True),
        ],
        modules=["identity_coder", "cap_reasoning", "cap_tools"],
        parent="base",
    ))

    return registry


# =============================================================================
# Composition Tools for Agent
# =============================================================================


def get_composition_tools() -> List[Dict[str, Any]]:
    """Get tool definitions for prompt composition."""
    return [
        {
            "type": "function",
            "function": {
                "name": "compose_prompt",
                "description": (
                    "Compose a new prompt from modules and templates. "
                    "Enables modular, reusable prompt construction."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "template": {
                            "type": "string",
                            "description": "Base template ID to use",
                        },
                        "modules": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Module IDs to include",
                        },
                        "params": {
                            "type": "object",
                            "description": "Template parameters",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_modules",
                "description": "List available prompt modules",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Filter by category",
                        },
                        "tag": {
                            "type": "string",
                            "description": "Filter by tag",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_templates",
                "description": "List available prompt templates",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "decompose_prompt",
                "description": "Decompose current prompt into modules for analysis",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
    ]
