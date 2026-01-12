"""
Template Loading for Simulation Characters

Loads persona profiles and prompt templates for character instantiation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Iterator
from enum import Enum

logger = logging.getLogger(__name__)


class TemplateCategory(str, Enum):
    """Categories of persona templates."""
    TECH_CEO = "tech_ceo"
    RESEARCHER = "researcher"
    POLICYMAKER = "policymaker"
    SAFETY_ADVOCATE = "safety_advocate"
    INVESTOR = "investor"
    MEDIA = "media"
    CUSTOM = "custom"


@dataclass
class PersonaTemplate:
    """Template for creating a simulation character."""

    id: str
    name: str
    category: TemplateCategory

    # Psychological profile (from persona_profiles.py)
    dominant_archetypes: List[Any] = field(default_factory=list)
    shadow_archetypes: List[Any] = field(default_factory=list)
    primary_complex: Optional[Any] = None
    ifs_parts: List[Any] = field(default_factory=list)
    core_dialectics: List[Any] = field(default_factory=list)
    somatic_map: Dict[Any, str] = field(default_factory=dict)

    # Prompt template (from prompts loader)
    prompt_text: Optional[str] = None
    prompt_id: Optional[int] = None

    # Behavior configuration
    traits: Dict[str, float] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_system_prompt(self) -> str:
        """Generate a system prompt from this template."""
        parts = []

        # Name and role
        parts.append(f"You are {self.name}.")

        # Use explicit prompt if available
        if self.prompt_text:
            parts.append(self.prompt_text)

        # Add archetype context
        if self.dominant_archetypes:
            archetype_names = [a.value if hasattr(a, 'value') else str(a)
                             for a in self.dominant_archetypes]
            parts.append(f"Your core archetypes are: {', '.join(archetype_names)}.")

        # Add primary complex for depth
        if self.primary_complex:
            if hasattr(self.primary_complex, 'name'):
                parts.append(f"Your driving psychological pattern: {self.primary_complex.name}.")
                if hasattr(self.primary_complex, 'core_affect'):
                    parts.append(f"Core affect: {self.primary_complex.core_affect}.")

        return "\n\n".join(parts)

    def with_prompt(self, prompt_text: str, prompt_id: Optional[int] = None) -> "PersonaTemplate":
        """Return a copy with a prompt attached."""
        import copy
        new = copy.deepcopy(self)
        new.prompt_text = prompt_text
        new.prompt_id = prompt_id
        return new


class TemplateLoader:
    """Loads and manages persona templates."""

    def __init__(self):
        self._templates: Dict[str, PersonaTemplate] = {}
        self._loaded = False
        self._prompt_loader = None

    def _ensure_loaded(self):
        """Ensure templates are loaded."""
        if self._loaded:
            return
        self.load_all()

    def load_all(self) -> int:
        """Load all available templates."""
        count = 0

        # Load from persona_profiles
        try:
            from ..research.persona_profiles import ALL_PERSONA_PROFILES

            for persona_id, profile in ALL_PERSONA_PROFILES.items():
                template = self._profile_to_template(profile)
                self._templates[persona_id] = template
                count += 1

            logger.info(f"Loaded {count} persona templates from profiles")
        except ImportError as e:
            logger.warning(f"Could not load persona profiles: {e}")

        self._loaded = True
        return count

    def _profile_to_template(self, profile: dict) -> PersonaTemplate:
        """Convert a psychological profile to a template."""
        category_map = {
            "tech_ceo": TemplateCategory.TECH_CEO,
            "researcher": TemplateCategory.RESEARCHER,
            "policymaker": TemplateCategory.POLICYMAKER,
            "safety_advocate": TemplateCategory.SAFETY_ADVOCATE,
            "investor": TemplateCategory.INVESTOR,
            "media": TemplateCategory.MEDIA,
        }

        category = category_map.get(
            profile.get("category", "custom"),
            TemplateCategory.CUSTOM
        )

        return PersonaTemplate(
            id=profile.get("id", "unknown"),
            name=profile.get("name", "Unknown"),
            category=category,
            dominant_archetypes=profile.get("dominant_archetypes", []),
            shadow_archetypes=profile.get("shadow_archetypes", []),
            primary_complex=profile.get("primary_complex"),
            ifs_parts=profile.get("ifs_parts", []),
            core_dialectics=profile.get("core_dialectics", []),
            somatic_map=profile.get("somatic_map", {}),
        )

    def get(self, template_id: str) -> Optional[PersonaTemplate]:
        """Get a template by ID."""
        self._ensure_loaded()
        return self._templates.get(template_id)

    def get_by_category(self, category: TemplateCategory) -> List[PersonaTemplate]:
        """Get all templates in a category."""
        self._ensure_loaded()
        return [t for t in self._templates.values() if t.category == category]

    def list_ids(self) -> List[str]:
        """List all template IDs."""
        self._ensure_loaded()
        return list(self._templates.keys())

    def list_all(self) -> List[PersonaTemplate]:
        """Get all templates."""
        self._ensure_loaded()
        return list(self._templates.values())

    def iterate(self) -> Iterator[PersonaTemplate]:
        """Iterate over all templates."""
        self._ensure_loaded()
        yield from self._templates.values()

    def search(self, query: str) -> List[PersonaTemplate]:
        """Search templates by name or ID."""
        self._ensure_loaded()
        query_lower = query.lower()
        return [
            t for t in self._templates.values()
            if query_lower in t.id.lower() or query_lower in t.name.lower()
        ]

    def with_prompts_from_loader(self) -> "TemplateLoader":
        """Attach prompts from the PromptLoader to templates."""
        try:
            from ..prompts.loader import get_loader
            prompt_loader = get_loader()

            # Search for matching prompts and attach
            for template_id, template in self._templates.items():
                # Search for prompts mentioning this persona
                matching = prompt_loader.search(template.name, limit=1)
                if matching:
                    prompt = matching[0]
                    template.prompt_text = prompt.text
                    template.prompt_id = prompt.id

            logger.info("Attached prompts to templates")
        except ImportError as e:
            logger.warning(f"Could not load prompts: {e}")

        return self


# Global loader instance
_template_loader: Optional[TemplateLoader] = None


def get_template_loader() -> TemplateLoader:
    """Get the global template loader instance."""
    global _template_loader
    if _template_loader is None:
        _template_loader = TemplateLoader()
    return _template_loader


def load_all_templates() -> Dict[str, PersonaTemplate]:
    """Load all templates and return as dict."""
    loader = get_template_loader()
    loader.load_all()
    return {t.id: t for t in loader.list_all()}


def load_template(template_id: str) -> Optional[PersonaTemplate]:
    """Load a specific template by ID."""
    loader = get_template_loader()
    return loader.get(template_id)
