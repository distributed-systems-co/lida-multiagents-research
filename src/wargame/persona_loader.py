"""
Rich Persona Loader for Wargame Engine

Loads detailed persona profiles from the persona_pipeline JSON files
and converts them to system prompts for authentic roleplay.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default paths
PERSONA_PIPELINE_DIR = Path(__file__).parent.parent.parent / "persona_pipeline" / "personas"


def list_available_personas(persona_dir: Path = PERSONA_PIPELINE_DIR) -> list[str]:
    """List all available persona IDs from the pipeline."""
    if not persona_dir.exists():
        return []

    personas = []
    for f in persona_dir.glob("*.json"):
        # Skip dated versions, prefer base files
        name = f.stem
        if "_2026-" not in name and "_2025-" not in name:
            personas.append(name)

    return sorted(personas)


def load_persona(persona_id: str, persona_dir: Path = PERSONA_PIPELINE_DIR) -> Optional[dict]:
    """Load a persona JSON file by ID."""
    # Normalize ID (underscores)
    normalized = persona_id.replace("-", "_").lower()

    # Try exact match first
    path = persona_dir / f"{normalized}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)

    # Try with dashes
    dashed = persona_id.replace("_", "-").lower()
    path = persona_dir / f"{dashed}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)

    # Search for partial match
    for f in persona_dir.glob("*.json"):
        if normalized in f.stem.lower() or dashed in f.stem.lower():
            with open(f) as file:
                return json.load(file)

    return None


def build_system_prompt(persona: dict, topic: str = "", context: str = "") -> str:
    """
    Build a rich system prompt from a persona JSON profile.

    Uses:
    - Background for authority/expertise
    - Communication style and verbal tics
    - Worldview for positions
    - Simulation guide for roleplay instructions
    """
    lines = []

    # Identity
    bg = persona.get("background", {})
    name = persona.get("_metadata", {}).get("name", "Unknown")

    # Try to extract name from career arc or origin story
    if not name or name == "Unknown":
        career = bg.get("career_arc", "")
        if "CEO" in career:
            name = career.split(",")[0] if "," in career else "Executive"

    lines.append(f"# You are roleplaying as a specific individual")
    lines.append("")

    # Background
    if bg:
        career = bg.get("career_arc", "")
        if career:
            career_str = career if isinstance(career, str) else str(career)
            lines.append(f"**Career**: {career_str[:500]}")
        education = bg.get("education", "")
        if education:
            edu_str = education if isinstance(education, str) else str(education)
            lines.append(f"**Education**: {edu_str[:200]}")
        lines.append("")

    # Personality
    personality = persona.get("personality", {})
    if personality.get("core_traits"):
        traits = ", ".join(personality["core_traits"][:7])
        lines.append(f"**Core traits**: {traits}")
    if personality.get("triggers"):
        lines.append(f"**Sensitive topics**: {', '.join(personality['triggers'][:5])}")
    lines.append("")

    # Communication style
    comm = persona.get("communication", {})
    public_persona = comm.get("public_persona", "")
    if public_persona:
        pp_str = public_persona if isinstance(public_persona, str) else str(public_persona)
        lines.append(f"**Public persona**: {pp_str[:300]}")
    verbal_tics = comm.get("verbal_tics", [])
    if verbal_tics and isinstance(verbal_tics, list):
        tics = [str(t) if not isinstance(t, str) else t for t in verbal_tics[:5]]
        lines.append(f"**Speech patterns**: {', '.join(tics)}")
    lines.append("")

    # Sample quotes for voice calibration
    sample_quotes = comm.get("sample_quotes", [])
    if sample_quotes and isinstance(sample_quotes, list):
        lines.append("**Characteristic quotes** (match this voice):")
        for quote in sample_quotes[:3]:
            q_str = quote if isinstance(quote, str) else str(quote)
            lines.append(f'- "{q_str[:150]}"')
        lines.append("")

    # Worldview - critical for policy positions
    worldview = persona.get("worldview", {})
    if worldview:
        core_beliefs = worldview.get("core_beliefs", [])
        if core_beliefs and isinstance(core_beliefs, list):
            lines.append("**Core beliefs**:")
            for belief in core_beliefs[:5]:
                b_str = belief if isinstance(belief, str) else str(belief)
                lines.append(f"- {b_str}")
        ai_phil = worldview.get("ai_philosophy", "")
        if ai_phil:
            ap_str = ai_phil if isinstance(ai_phil, str) else str(ai_phil)
            lines.append(f"\n**AI philosophy**: {ap_str[:400]}")
        china = worldview.get("china_stance", "")
        if china:
            c_str = china if isinstance(china, str) else str(china)
            lines.append(f"\n**China stance**: {c_str[:300]}")
        reg = worldview.get("regulation_views", "")
        if reg:
            r_str = reg if isinstance(reg, str) else str(reg)
            lines.append(f"\n**Regulation views**: {r_str[:300]}")
        lines.append("")

    # Relationships for interaction dynamics
    rels = persona.get("relationships", {})
    if rels.get("allies"):
        allies = rels["allies"][:5] if isinstance(rels["allies"], list) else []
        if allies:
            ally_names = [a if isinstance(a, str) else a.get("name", "") for a in allies]
            lines.append(f"**Allies**: {', '.join(ally_names)}")
    if rels.get("enemies"):
        enemies = rels["enemies"][:5] if isinstance(rels["enemies"], list) else []
        if enemies:
            enemy_names = [e if isinstance(e, str) else e.get("name", "") for e in enemies]
            lines.append(f"**Adversaries**: {', '.join(enemy_names)}")
    lines.append("")

    # Simulation guide - the most important part for authentic roleplay
    guide = persona.get("simulation_guide", {})
    if guide:
        how_to = guide.get("how_to_embody", "")
        if how_to:
            hte_str = how_to if isinstance(how_to, str) else str(how_to)
            lines.append("## Roleplay Instructions")
            lines.append(hte_str[:800])
            lines.append("")

        never_say = guide.get("never_say", [])
        if never_say and isinstance(never_say, list):
            lines.append("**Never say**:")
            for item in never_say[:5]:
                i_str = item if isinstance(item, str) else str(item)
                lines.append(f"- {i_str}")
            lines.append("")

        hot_buttons = guide.get("hot_buttons", [])
        if hot_buttons and isinstance(hot_buttons, list):
            lines.append("**Hot buttons** (strong reactions):")
            for item in hot_buttons[:4]:
                i_str = item if isinstance(item, str) else str(item)
                lines.append(f"- {i_str[:100]}")
            lines.append("")

    # Current context
    lines.append("---")
    lines.append("")
    lines.append("## Current Situation")
    if topic:
        lines.append(f"**Topic under discussion**: {topic}")
    if context:
        lines.append(f"\n{context}")
    lines.append("")
    lines.append("Respond authentically as this person would. Stay in character.")
    lines.append("Keep responses focused and substantive (under 250 words unless detail needed).")

    return "\n".join(lines)


def get_persona_stance(persona: dict) -> str:
    """Infer stance from worldview, positions, and personality."""
    worldview = persona.get("worldview", {})
    personality = persona.get("personality", {})

    # Check AI philosophy
    ai_phil = worldview.get("ai_philosophy", "").lower()
    reg_views = worldview.get("regulation_views", "").lower()
    core_beliefs_list = worldview.get("core_beliefs", [])
    if isinstance(core_beliefs_list, list):
        core_beliefs = " ".join(str(b) for b in core_beliefs_list).lower()
    else:
        core_beliefs = str(core_beliefs_list).lower()
    traits_list = personality.get("core_traits", [])
    if isinstance(traits_list, list):
        traits = " ".join(str(t) for t in traits_list).lower()
    else:
        traits = str(traits_list).lower()

    # Accelerationist markers
    accel_markers = ["dismissive of", "opposes restriction", "move fast", "build", "democratiz",
                     "progress is", "optimistic about tech", "anti-regulation"]
    accel_score = sum(1 for m in accel_markers if m in ai_phil or m in reg_views or m in core_beliefs or m in traits)

    # Doomer markers
    doom_markers = ["existential risk", "catastroph", "extinction", "pause", "moratorium",
                    "shut down", "danger", "x-risk", "unsafe"]
    doom_score = sum(1 for m in doom_markers if m in ai_phil or m in core_beliefs or m in traits)

    # Pro-safety markers (not doomer, but careful)
    safety_markers = ["safety", "cautious", "careful", "responsible", "alignment",
                      "interpretab", "constitutional", "testing"]
    safety_score = sum(1 for m in safety_markers if m in ai_phil or m in core_beliefs or m in traits)

    # Pro-industry markers
    industry_markers = ["self-regulation", "industry", "innovation", "competitive",
                        "light touch", "voluntary", "market"]
    industry_score = sum(1 for m in industry_markers if m in reg_views or m in ai_phil or m in traits)

    # Determine stance based on scores
    scores = {
        "accelerationist": accel_score,
        "doomer": doom_score,
        "pro_safety": safety_score,
        "pro_industry": industry_score,
    }

    max_stance = max(scores, key=lambda k: scores[k])
    max_score = scores[max_stance]

    if max_score >= 2:
        return max_stance
    elif max_score == 1:
        # Check regulation views for tiebreaker
        if "strong" in reg_views or "strict" in reg_views or "mandate" in reg_views:
            return "pro_safety"
        if "oppose" in reg_views or "dismiss" in reg_views:
            return "pro_industry"

    return "moderate"


def get_persona_category(persona: dict) -> str:
    """Infer category from metadata or background."""
    meta = persona.get("_metadata", {})
    if meta.get("category"):
        return meta["category"]

    bg = persona.get("background", {})
    career = bg.get("career_arc", "").lower()

    if "ceo" in career or "founder" in career:
        return "tech_leader"
    if "president" in career or "senator" in career or "representative" in career:
        return "politician"
    if "professor" in career or "researcher" in career:
        return "researcher"
    if "investor" in career or "venture" in career:
        return "investor"

    return "other"


class RichPersona:
    """Wrapper for rich persona data with helper methods."""

    def __init__(self, data: dict, persona_id: str):
        self.data = data
        self.id = persona_id

    @property
    def name(self) -> str:
        # Try to extract from various places
        meta = self.data.get("_metadata", {})
        if meta.get("name"):
            return meta["name"]

        # Convert ID to proper name (e.g., "jensen_huang" -> "Jensen Huang")
        return self.id.replace("_", " ").title()

    @property
    def stance(self) -> str:
        return get_persona_stance(self.data)

    @property
    def category(self) -> str:
        return get_persona_category(self.data)

    def build_system_prompt(self, topic: str = "", context: str = "") -> str:
        return build_system_prompt(self.data, topic, context)

    def get_sample_quotes(self) -> list[str]:
        return self.data.get("communication", {}).get("sample_quotes", [])

    def get_triggers(self) -> list[str]:
        return self.data.get("personality", {}).get("triggers", [])

    def get_allies(self) -> list[str]:
        allies = self.data.get("relationships", {}).get("allies", [])
        return [a if isinstance(a, str) else a.get("name", "") for a in allies]

    def get_enemies(self) -> list[str]:
        enemies = self.data.get("relationships", {}).get("enemies", [])
        return [e if isinstance(e, str) else e.get("name", "") for e in enemies]


def load_rich_persona(persona_id: str, persona_dir: Path = PERSONA_PIPELINE_DIR) -> Optional[RichPersona]:
    """Load a persona and return a RichPersona wrapper."""
    data = load_persona(persona_id, persona_dir)
    if data:
        return RichPersona(data, persona_id)
    return None
