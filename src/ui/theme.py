"""Theme and styling for the multi-agent system UI."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class Color(str, Enum):
    """Color palette."""
    # Primary
    PRIMARY = "#7C3AED"       # Purple
    SECONDARY = "#06B6D4"     # Cyan
    ACCENT = "#F59E0B"        # Amber

    # Status
    SUCCESS = "#10B981"       # Emerald
    WARNING = "#F59E0B"       # Amber
    ERROR = "#EF4444"         # Red
    INFO = "#3B82F6"          # Blue

    # Agents
    DEMIURGE = "#8B5CF6"      # Violet
    PERSONA = "#06B6D4"       # Cyan
    WORKER = "#10B981"        # Emerald
    MONITOR = "#F59E0B"       # Amber

    # Messages
    BROADCAST = "#EC4899"     # Pink
    MULTICAST = "#8B5CF6"     # Violet
    DIRECT = "#06B6D4"        # Cyan
    REQUEST = "#3B82F6"       # Blue
    RESPONSE = "#10B981"      # Emerald
    EVENT = "#F59E0B"         # Amber

    # UI
    MUTED = "#6B7280"         # Gray
    BORDER = "#374151"        # Dark gray
    BG = "#111827"            # Near black
    FG = "#F9FAFB"            # Near white


# Unicode symbols for styling
SYMBOLS = {
    # Status
    "check": "✓",
    "cross": "✗",
    "warning": "⚠",
    "info": "ℹ",
    "dot": "●",
    "ring": "○",
    "star": "★",

    # Arrows
    "arrow_right": "→",
    "arrow_left": "←",
    "arrow_up": "↑",
    "arrow_down": "↓",
    "arrow_bidir": "↔",

    # Messages
    "broadcast": "📡",
    "multicast": "📢",
    "direct": "✉",
    "request": "❓",
    "response": "💬",
    "event": "⚡",

    # Agents
    "demiurge": "🔮",
    "persona": "👤",
    "worker": "⚙",
    "monitor": "👁",

    # UI
    "box_h": "─",
    "box_v": "│",
    "box_tl": "┌",
    "box_tr": "┐",
    "box_bl": "└",
    "box_br": "┘",
    "box_t": "┬",
    "box_b": "┴",
    "box_l": "├",
    "box_r": "┤",
    "box_x": "┼",

    # Double box
    "dbox_h": "═",
    "dbox_v": "║",
    "dbox_tl": "╔",
    "dbox_tr": "╗",
    "dbox_bl": "╚",
    "dbox_br": "╝",

    # Progress
    "bar_full": "█",
    "bar_half": "▓",
    "bar_light": "░",
    "spinner": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
}


@dataclass
class Theme:
    """UI Theme configuration."""
    name: str = "default"
    colors: Dict[str, str] = None

    def __post_init__(self):
        if self.colors is None:
            self.colors = {c.name.lower(): c.value for c in Color}

    @classmethod
    def dark(cls) -> "Theme":
        return cls(name="dark")

    @classmethod
    def light(cls) -> "Theme":
        return cls(
            name="light",
            colors={
                "bg": "#FFFFFF",
                "fg": "#1F2937",
                "muted": "#9CA3AF",
                "border": "#D1D5DB",
            }
        )

    def get_agent_color(self, agent_type: str) -> str:
        """Get color for agent type."""
        mapping = {
            "demiurge": Color.DEMIURGE.value,
            "persona": Color.PERSONA.value,
            "worker": Color.WORKER.value,
            "monitor": Color.MONITOR.value,
        }
        return mapping.get(agent_type.lower(), Color.SECONDARY.value)

    def get_message_color(self, msg_type: str) -> str:
        """Get color for message type."""
        mapping = {
            "broadcast": Color.BROADCAST.value,
            "multicast": Color.MULTICAST.value,
            "direct": Color.DIRECT.value,
            "request": Color.REQUEST.value,
            "response": Color.RESPONSE.value,
            "event": Color.EVENT.value,
        }
        return mapping.get(msg_type.lower(), Color.SECONDARY.value)

    def get_status_color(self, status: str) -> str:
        """Get color for status."""
        mapping = {
            "running": Color.SUCCESS.value,
            "ready": Color.SUCCESS.value,
            "busy": Color.WARNING.value,
            "suspended": Color.WARNING.value,
            "error": Color.ERROR.value,
            "dead": Color.ERROR.value,
            "terminated": Color.MUTED.value,
        }
        return mapping.get(status.lower(), Color.MUTED.value)
