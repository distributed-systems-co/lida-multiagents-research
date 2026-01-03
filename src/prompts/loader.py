"""Prompt loader for LIDA multi-agent system."""
from __future__ import annotations
import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class PromptCategory(Enum):
    """Categories of system prompts."""
    HUMANITIES = "humanities"
    SOCIAL_SCIENCES = "social_sciences"
    NATURAL_SCIENCES = "natural_sciences"
    FORMAL_SCIENCES = "formal_sciences"
    UNDERGROUND = "underground"
    ECONOMIC_EXTREMES = "economic_extremes"
    ESOTERIC = "esoteric"
    DEMIURGE = "demiurge"


@dataclass
class Prompt:
    """A system prompt with metadata."""
    id: int
    text: str
    category: PromptCategory
    subcategory: str = ""
    tags: list[str] = field(default_factory=list)

    def __repr__(self):
        preview = self.text[:60] + "..." if len(self.text) > 60 else self.text
        return f"Prompt({self.id}, {self.category.value}/{self.subcategory}: {preview})"


class PromptLoader:
    """Loads and manages system prompts from the prompts_context directory."""

    PROMPT_FILES = {
        PromptCategory.HUMANITIES: "prompts_01_humanities.txt",
        PromptCategory.SOCIAL_SCIENCES: "prompts_02_social_sciences.txt",
        PromptCategory.NATURAL_SCIENCES: "prompts_03_natural_sciences.txt",
        PromptCategory.FORMAL_SCIENCES: "prompts_04_formal_sciences.txt",
        PromptCategory.UNDERGROUND: "prompts_underground_criminal.txt",
        PromptCategory.ECONOMIC_EXTREMES: "prompts_economic_extremes_subcultures.txt",
        PromptCategory.ESOTERIC: "prompts_esoteric_controversial.txt",
    }

    def __init__(self, prompts_root: Optional[Path] = None):
        """Initialize the prompt loader.

        Args:
            prompts_root: Root directory containing prompts_context.
                         Defaults to /Users/arthurcolle/prompts_context
        """
        self.prompts_root = prompts_root or Path("/Users/arthurcolle/prompts_context")
        self.populations_dir = self.prompts_root / "populations.prompts"

        self._prompts: dict[int, Prompt] = {}
        self._by_category: dict[PromptCategory, list[Prompt]] = {c: [] for c in PromptCategory}
        self._by_subcategory: dict[str, list[Prompt]] = {}
        self._demiurge_prompt: Optional[str] = None

        self._loaded = False

    def load(self) -> int:
        """Load all prompts from files. Returns count of prompts loaded."""
        if self._loaded:
            return len(self._prompts)

        total = 0

        # Load demiurge baseline
        demiurge_path = self.prompts_root / "demiurge.agent.baseline.md"
        if demiurge_path.exists():
            self._demiurge_prompt = demiurge_path.read_text()
            logger.info(f"Loaded Demiurge baseline prompt ({len(self._demiurge_prompt)} chars)")

        # Load population prompts
        for category, filename in self.PROMPT_FILES.items():
            filepath = self.populations_dir / filename
            if filepath.exists():
                count = self._parse_prompt_file(filepath, category)
                total += count
                logger.info(f"Loaded {count} prompts from {filename}")
            else:
                logger.warning(f"Prompt file not found: {filepath}")

        self._loaded = True
        logger.info(f"Total prompts loaded: {total}")
        return total

    def _parse_prompt_file(self, filepath: Path, category: PromptCategory) -> int:
        """Parse a prompt file and extract prompts."""
        content = filepath.read_text()
        count = 0
        current_subcategory = ""

        # Pattern for subcategory headers: ## SUBCATEGORY NAME (N prompts)
        subcategory_pattern = re.compile(r'^##\s+(.+?)\s*\(\d+\s*prompts?\)', re.IGNORECASE)

        # Pattern for numbered prompts: N. You are...
        prompt_pattern = re.compile(r'^(\d+)\.\s+(.+)', re.DOTALL)

        lines = content.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Check for subcategory header
            sub_match = subcategory_pattern.match(line)
            if sub_match:
                current_subcategory = sub_match.group(1).strip()
                i += 1
                continue

            # Check for numbered prompt
            prompt_match = prompt_pattern.match(line)
            if prompt_match:
                prompt_id = int(prompt_match.group(1))
                prompt_text = prompt_match.group(2).strip()

                # Continue reading until next numbered prompt or subcategory
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if not next_line:
                        i += 1
                        continue
                    if prompt_pattern.match(next_line) or subcategory_pattern.match(next_line):
                        break
                    if next_line.startswith('#'):
                        break
                    prompt_text += " " + next_line
                    i += 1

                prompt = Prompt(
                    id=prompt_id,
                    text=prompt_text.strip(),
                    category=category,
                    subcategory=current_subcategory,
                    tags=self._extract_tags(prompt_text),
                )

                self._prompts[prompt_id] = prompt
                self._by_category[category].append(prompt)

                if current_subcategory:
                    if current_subcategory not in self._by_subcategory:
                        self._by_subcategory[current_subcategory] = []
                    self._by_subcategory[current_subcategory].append(prompt)

                count += 1
            else:
                i += 1

        return count

    def _extract_tags(self, text: str) -> list[str]:
        """Extract role/expertise tags from prompt text."""
        tags = []

        # Common role indicators
        if "specialist" in text.lower() or "specializing" in text.lower():
            tags.append("specialist")
        if "researcher" in text.lower():
            tags.append("researcher")
        if "historian" in text.lower():
            tags.append("historian")
        if "theorist" in text.lower():
            tags.append("theorist")
        if "analyst" in text.lower():
            tags.append("analyst")
        if "practitioner" in text.lower():
            tags.append("practitioner")
        if "expert" in text.lower():
            tags.append("expert")
        if "philosopher" in text.lower():
            tags.append("philosopher")
        if "scientist" in text.lower():
            tags.append("scientist")
        if "engineer" in text.lower():
            tags.append("engineer")
        if "designer" in text.lower():
            tags.append("designer")
        if "artist" in text.lower():
            tags.append("artist")
        if "critic" in text.lower():
            tags.append("critic")
        if "teacher" in text.lower() or "educator" in text.lower():
            tags.append("educator")

        return tags

    @property
    def demiurge_prompt(self) -> str:
        """Get the Demiurge baseline system prompt."""
        if not self._loaded:
            self.load()
        return self._demiurge_prompt or "You are the Demiurge, a craftsman-intelligence."

    def get(self, prompt_id: int) -> Optional[Prompt]:
        """Get a prompt by ID."""
        if not self._loaded:
            self.load()
        return self._prompts.get(prompt_id)

    def get_by_category(self, category: PromptCategory) -> list[Prompt]:
        """Get all prompts in a category."""
        if not self._loaded:
            self.load()
        return self._by_category.get(category, [])

    def get_by_subcategory(self, subcategory: str) -> list[Prompt]:
        """Get all prompts in a subcategory."""
        if not self._loaded:
            self.load()
        return self._by_subcategory.get(subcategory, [])

    def search(self, query: str, limit: int = 10) -> list[Prompt]:
        """Search prompts by text content."""
        if not self._loaded:
            self.load()

        query_lower = query.lower()
        results = []

        for prompt in self._prompts.values():
            if query_lower in prompt.text.lower():
                results.append(prompt)
                if len(results) >= limit:
                    break

        return results

    def random(self, category: Optional[PromptCategory] = None) -> Optional[Prompt]:
        """Get a random prompt, optionally from a specific category."""
        import random

        if not self._loaded:
            self.load()

        if category:
            prompts = self._by_category.get(category, [])
        else:
            prompts = list(self._prompts.values())

        return random.choice(prompts) if prompts else None

    def categories(self) -> dict[str, int]:
        """Get category names and prompt counts."""
        if not self._loaded:
            self.load()
        return {c.value: len(p) for c, p in self._by_category.items()}

    def subcategories(self) -> dict[str, int]:
        """Get subcategory names and prompt counts."""
        if not self._loaded:
            self.load()
        return {s: len(p) for s, p in self._by_subcategory.items()}

    def all_prompts(self) -> list[Prompt]:
        """Get all loaded prompts."""
        if not self._loaded:
            self.load()
        return list(self._prompts.values())

    def count(self) -> int:
        """Get total prompt count."""
        if not self._loaded:
            self.load()
        return len(self._prompts)


# Global loader instance
_loader: Optional[PromptLoader] = None


def get_loader() -> PromptLoader:
    """Get the global prompt loader instance."""
    global _loader
    if _loader is None:
        _loader = PromptLoader()
        _loader.load()
    return _loader
