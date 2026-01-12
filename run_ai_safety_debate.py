#!/usr/bin/env python3
"""
AI Safety Debate Runner

Run debates using the AI X-Risk scenario YAML files with LLM-powered personas.

Usage:
    python run_ai_safety_debate.py                          # Interactive menu
    python run_ai_safety_debate.py --topic ai_pause         # Specific topic
    python run_ai_safety_debate.py --matchup doom_vs_accel  # Predefined matchup
    python run_ai_safety_debate.py --rounds 6               # Set round count
    python run_ai_safety_debate.py --auto                   # Auto-run (no prompts)
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    print("PyYAML required: pip install pyyaml")
    sys.exit(1)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.simulation.advanced_debate_engine import (
    AdvancedDebateEngine,
    DebateCLI,
    DebaterState,
    EmotionalState,
    ArgumentType,
    RelationshipType,
    EXTENDED_PERSONAS,
)


# =============================================================================
# YAML Loader
# =============================================================================

def load_scenario(yaml_path: str) -> Dict[str, Any]:
    """Load a scenario from YAML file."""
    path = Path(yaml_path)
    if not path.exists():
        # Try scenarios directory
        scenarios_dir = Path(__file__).parent / "scenarios"
        path = scenarios_dir / yaml_path
        if not path.exists():
            path = scenarios_dir / f"{yaml_path}.yaml"

    if not path.exists():
        raise FileNotFoundError(f"Scenario not found: {yaml_path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_campaign(yaml_path: str) -> Dict[str, Any]:
    """Load a campaign configuration."""
    path = Path(yaml_path)
    if not path.exists():
        campaigns_dir = Path(__file__).parent / "scenarios" / "campaigns"
        path = campaigns_dir / yaml_path
        if not path.exists():
            path = campaigns_dir / f"{yaml_path}.yaml"

    if not path.exists():
        raise FileNotFoundError(f"Campaign not found: {yaml_path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)


# =============================================================================
# Persona Mapping
# =============================================================================

@dataclass
class YAMLPersona:
    """Persona loaded from YAML."""
    id: str
    name: str
    role: str
    organization: str
    model: str
    personality: str
    initial_position: str
    confidence: float
    background: str
    system_prompt: str
    signature_arguments: List[str]
    debate_tactics: List[str]
    cognitive_biases: Dict[str, float]
    vulnerabilities: List[str]
    resistances: List[str]
    hot_buttons: List[str]
    resistance_score: float


def parse_personas(scenario: Dict[str, Any]) -> Dict[str, YAMLPersona]:
    """Parse personas from scenario YAML."""
    personas = {}
    for p in scenario.get("personas", []):
        persona = YAMLPersona(
            id=p.get("id", "unknown"),
            name=p.get("name", "Unknown"),
            role=p.get("role", ""),
            organization=p.get("organization", ""),
            model=p.get("model", "anthropic/claude-sonnet-4"),
            personality=p.get("personality", "analytical"),
            initial_position=p.get("initial_position", "NEUTRAL"),
            confidence=p.get("confidence", 0.5),
            background=p.get("background", ""),
            system_prompt=p.get("system_prompt", ""),
            signature_arguments=p.get("signature_arguments", []),
            debate_tactics=p.get("debate_tactics", []),
            cognitive_biases=p.get("cognitive_biases", {}),
            vulnerabilities=p.get("vulnerabilities", []),
            resistances=p.get("resistances", []),
            hot_buttons=p.get("hot_buttons", []),
            resistance_score=p.get("resistance_score", 0.5),
        )
        personas[persona.id] = persona
    return personas


# =============================================================================
# Debate Topics from YAML
# =============================================================================

DEBATE_TOPICS = {
    "ai_pause": {
        "topic": "6-Month AI Training Pause",
        "motion": "Should we implement an immediate 6-month pause on training AI systems more powerful than GPT-4?",
        "recommended_participants": ["yudkowsky", "connor", "lecun", "andreessen"],
    },
    "lab_self_regulation": {
        "topic": "Lab Self-Regulation",
        "motion": "Can we trust AI labs to effectively self-regulate safety practices?",
        "recommended_participants": ["altman", "amodei", "gebru", "toner"],
    },
    "xrisk_vs_present_harms": {
        "topic": "X-Risk vs Present Harms",
        "motion": "Should AI safety focus primarily on existential risks or present-day harms?",
        "recommended_participants": ["yudkowsky", "gebru", "bengio", "macaskill"],
    },
    "scaling_hypothesis": {
        "topic": "The Scaling Hypothesis",
        "motion": "Will scaling current architectures lead to beneficial AGI?",
        "recommended_participants": ["lecun", "altman", "russell", "bengio"],
    },
    "open_source_ai": {
        "topic": "Open Source AI Models",
        "motion": "Should frontier AI model weights be publicly released?",
        "recommended_participants": ["lecun", "andreessen", "amodei", "connor"],
    },
    "government_regulation": {
        "topic": "Government AI Regulation",
        "motion": "Should governments mandate comprehensive AI safety requirements?",
        "recommended_participants": ["toner", "andreessen", "bengio", "altman"],
    },
}

# Predefined matchups
MATCHUPS = {
    "doom_vs_accel": {
        "name": "Doomers vs Accelerationists",
        "team_a": ["yudkowsky", "connor"],
        "team_b": ["andreessen", "lecun"],
        "default_topic": "ai_pause",
    },
    "labs_debate": {
        "name": "Lab Leaders Debate",
        "team_a": ["altman", "amodei"],
        "team_b": ["lecun", "andreessen"],
        "default_topic": "lab_self_regulation",
    },
    "academics_clash": {
        "name": "Academic Perspectives",
        "team_a": ["bengio", "russell"],
        "team_b": ["lecun"],
        "default_topic": "xrisk_vs_present_harms",
    },
    "ethics_vs_scale": {
        "name": "Ethics vs Scale",
        "team_a": ["gebru", "toner"],
        "team_b": ["altman", "andreessen"],
        "default_topic": "xrisk_vs_present_harms",
    },
    "full_panel": {
        "name": "Full AI Safety Panel",
        "team_a": ["yudkowsky", "bengio", "russell", "gebru"],
        "team_b": ["lecun", "andreessen", "altman", "amodei"],
        "default_topic": "ai_pause",
    },
}


# =============================================================================
# Enhanced Debate Engine with YAML Personas
# =============================================================================

class YAMLDebateEngine(AdvancedDebateEngine):
    """Debate engine that uses YAML persona profiles."""

    def __init__(
        self,
        topic: str,
        motion: str,
        yaml_personas: Dict[str, YAMLPersona],
        participants: List[str],
        llm_provider: str = "openrouter",
        llm_model: Optional[str] = None,
        use_llm: bool = True,
    ):
        self.yaml_personas = yaml_personas

        # Override EXTENDED_PERSONAS with YAML data
        for pid, yp in yaml_personas.items():
            if pid in participants:
                EXTENDED_PERSONAS[pid] = self._convert_yaml_persona(yp)

        super().__init__(
            topic=topic,
            motion=motion,
            participants=participants,
            llm_provider=llm_provider,
            llm_model=llm_model,
            use_llm=use_llm,
        )

        # Apply YAML-specific configurations
        self._apply_yaml_config()

    def _convert_yaml_persona(self, yp: YAMLPersona) -> Dict[str, Any]:
        """Convert YAML persona to engine format."""
        # Map tactics
        tactic_mapping = {
            "thought_experiments": "use analogies",
            "reductio_ad_absurdum": "reduce to absurdity",
            "appeal_to_technical_difficulty": "cite technical details",
            "pessimistic_induction": "invoke past failures",
            "analogy_to_evolution": "evolutionary argument",
            "empirical_evidence": "cite research",
            "appeal_to_humanity": "invoke human values",
            "strategic_ambiguity": "hedge positions",
            "coalition_building": "build alliances",
            "first_principles": "reason from axioms",
            "contrarian_framing": "challenge assumptions",
            "attention_dominance": "control narrative",
            "provocation": "make bold claims",
            "stakeholder_emphasis": "center affected communities",
            "power_analysis": "analyze power dynamics",
            "personal_narrative": "share experience",
        }

        favorite_moves = [
            tactic_mapping.get(t, t.replace("_", " "))
            for t in yp.debate_tactics[:4]
        ]

        return {
            "name": yp.name,
            "title": f"{yp.role}, {yp.organization}",
            "archetype": yp.personality,
            "speaking_style": yp.personality,
            "core_values": [v.replace("_", " ") for v in yp.vulnerabilities[:3]],
            "debate_tendencies": {
                "opens_with": "framing the stakes" if yp.initial_position == "AGAINST" else "positive framing",
                "favorite_moves": favorite_moves,
                "concedes_on": yp.vulnerabilities[:2],
                "never_concedes": yp.resistances[:2],
                "gets_frustrated_by": yp.hot_buttons[:2],
            },
            "relationships": {},  # Will be populated from YAML
            "signature_phrases": yp.signature_arguments[:3],
            # Extra fields for enhanced behavior
            "system_prompt": yp.system_prompt,
            "background": yp.background,
            "cognitive_biases": yp.cognitive_biases,
        }

    def _apply_yaml_config(self):
        """Apply YAML-specific debater configurations."""
        for debater_id, debater in self.state.debaters.items():
            if debater_id in self.yaml_personas:
                yp = self.yaml_personas[debater_id]

                # Set initial beliefs based on position
                if yp.initial_position == "AGAINST":
                    debater.beliefs["support_motion"] = 0.1
                elif yp.initial_position == "FOR":
                    debater.beliefs["support_motion"] = 0.9
                else:
                    debater.beliefs["support_motion"] = 0.5

                # Set confidence
                debater.beliefs["confidence"] = yp.confidence

                # Set vulnerabilities and red lines
                debater.vulnerabilities = yp.vulnerabilities
                debater.red_lines = yp.resistances


# =============================================================================
# Debate Runner
# =============================================================================

class AIDebateRunner:
    """Run AI safety debates from YAML configurations."""

    def __init__(self, scenario_path: str = "scenarios/ai_xrisk.yaml"):
        self.scenario = load_scenario(scenario_path)
        self.personas = parse_personas(self.scenario)
        print(f"✓ Loaded {len(self.personas)} personas from scenario")

    def list_personas(self):
        """Print available personas."""
        print("\n" + "="*60)
        print("AVAILABLE PERSONAS")
        print("="*60)

        # Group by stance
        positions = {}
        for pid, p in self.personas.items():
            pos = p.initial_position
            if pos not in positions:
                positions[pos] = []
            positions[pos].append(p)

        for pos, members in positions.items():
            print(f"\n{pos}:")
            for p in members:
                print(f"  • {p.id}: {p.name} ({p.organization})")

    def list_topics(self):
        """Print available debate topics."""
        print("\n" + "="*60)
        print("DEBATE TOPICS")
        print("="*60)

        for tid, topic in DEBATE_TOPICS.items():
            print(f"\n{tid}:")
            print(f"  Topic: {topic['topic']}")
            print(f"  Motion: {topic['motion']}")
            participants = ", ".join(topic["recommended_participants"])
            print(f"  Recommended: {participants}")

    def list_matchups(self):
        """Print predefined matchups."""
        print("\n" + "="*60)
        print("PREDEFINED MATCHUPS")
        print("="*60)

        for mid, matchup in MATCHUPS.items():
            print(f"\n{mid}: {matchup['name']}")
            print(f"  Team A: {', '.join(matchup['team_a'])}")
            print(f"  Team B: {', '.join(matchup['team_b'])}")
            print(f"  Default Topic: {matchup['default_topic']}")

    async def run_debate(
        self,
        topic_id: str,
        participants: List[str],
        rounds: int = 5,
        auto: bool = False,
        use_llm: bool = True,
        llm_provider: str = "openrouter",
        llm_model: Optional[str] = None,
    ):
        """Run a debate on the specified topic."""

        if topic_id not in DEBATE_TOPICS:
            print(f"Unknown topic: {topic_id}")
            self.list_topics()
            return

        topic_config = DEBATE_TOPICS[topic_id]

        # Filter to only participants we have
        valid_participants = [p for p in participants if p in self.personas]
        if not valid_participants:
            print(f"No valid participants found. Using recommended.")
            valid_participants = topic_config["recommended_participants"]

        # Ensure we have participants
        valid_participants = [p for p in valid_participants if p in self.personas]

        if len(valid_participants) < 2:
            print("Need at least 2 valid participants!")
            return

        print("\n" + "="*60)
        print(f"DEBATE: {topic_config['topic']}")
        print("="*60)
        print(f"Motion: {topic_config['motion']}")
        print(f"Provider: {llm_provider}" + (f" ({llm_model})" if llm_model else ""))
        print(f"\nParticipants:")
        for pid in valid_participants:
            p = self.personas[pid]
            print(f"  • {p.name} ({p.organization}) - {p.initial_position}")

        # Create engine
        engine = YAMLDebateEngine(
            topic=topic_config["topic"],
            motion=topic_config["motion"],
            yaml_personas=self.personas,
            participants=valid_participants,
            use_llm=use_llm,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )

        if auto:
            # Auto-run without interaction - still use CLI for callbacks
            cli = DebateCLI(engine)
            print(f"\nRunning {rounds} rounds automatically...")
            for round_num in range(rounds):
                print(f"\n--- Round {round_num + 1} ---")
                await engine.run_round()
            print("\n" + engine.summarize())
        else:
            # Interactive mode
            cli = DebateCLI(engine)
            await cli.run(max_rounds=rounds)

        return engine

    async def run_matchup(
        self,
        matchup_id: str,
        topic_id: Optional[str] = None,
        rounds: int = 5,
        auto: bool = False,
        llm_provider: str = "openrouter",
        llm_model: Optional[str] = None,
    ):
        """Run a predefined matchup."""
        if matchup_id not in MATCHUPS:
            print(f"Unknown matchup: {matchup_id}")
            self.list_matchups()
            return

        matchup = MATCHUPS[matchup_id]
        topic = topic_id or matchup["default_topic"]
        participants = matchup["team_a"] + matchup["team_b"]

        print(f"\n🎯 {matchup['name']}")
        return await self.run_debate(
            topic, participants, rounds, auto,
            llm_provider=llm_provider, llm_model=llm_model
        )

    def interactive_menu(self):
        """Show interactive menu."""
        while True:
            print("\n" + "="*60)
            print("AI SAFETY DEBATE SIMULATOR")
            print("="*60)
            print("\n1. List personas")
            print("2. List topics")
            print("3. List matchups")
            print("4. Run custom debate")
            print("5. Run matchup")
            print("6. Quick demo (auto-run)")
            print("q. Quit")

            choice = input("\nChoice> ").strip().lower()

            if choice == "1":
                self.list_personas()
            elif choice == "2":
                self.list_topics()
            elif choice == "3":
                self.list_matchups()
            elif choice == "4":
                self._run_custom()
            elif choice == "5":
                self._run_matchup_interactive()
            elif choice == "6":
                asyncio.run(self._quick_demo())
            elif choice in ["q", "quit"]:
                print("Goodbye!")
                break

    def _run_custom(self):
        """Interactive custom debate setup."""
        self.list_topics()
        topic = input("\nEnter topic ID> ").strip()

        self.list_personas()
        participants_str = input("\nEnter participant IDs (comma-separated)> ").strip()
        participants = [p.strip() for p in participants_str.split(",")]

        rounds = int(input("Number of rounds [5]> ").strip() or "5")

        asyncio.run(self.run_debate(topic, participants, rounds))

    def _run_matchup_interactive(self):
        """Interactive matchup selection."""
        self.list_matchups()
        matchup = input("\nEnter matchup ID> ").strip()
        rounds = int(input("Number of rounds [5]> ").strip() or "5")

        asyncio.run(self.run_matchup(matchup, rounds=rounds))

    async def _quick_demo(self):
        """Run a quick demo debate."""
        print("\n🎬 Running quick demo: Doomers vs Accelerationists")
        await self.run_matchup("doom_vs_accel", rounds=3, auto=True)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run AI Safety debates from YAML scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_ai_safety_debate.py                           # Interactive menu
  python run_ai_safety_debate.py --topic ai_pause          # Specific topic
  python run_ai_safety_debate.py --matchup doom_vs_accel   # Predefined matchup
  python run_ai_safety_debate.py --rounds 6 --auto         # Auto-run 6 rounds
  python run_ai_safety_debate.py --list-all                # Show all options

  # Using OpenRouter (default):
  export OPENROUTER_API_KEY=sk-or-...
  python run_ai_safety_debate.py --matchup doom_vs_accel --auto

  # Using specific model:
  python run_ai_safety_debate.py --topic ai_pause --model google/gemini-2.0-flash-001

Available OpenRouter models:
  anthropic/claude-sonnet-4 (default)
  anthropic/claude-opus-4
  openai/gpt-4o
  google/gemini-2.0-flash-001
  meta-llama/llama-3.3-70b-instruct
  deepseek/deepseek-chat-v3-0324
        """
    )

    parser.add_argument(
        "--scenario",
        default="scenarios/ai_xrisk.yaml",
        help="Path to scenario YAML file"
    )
    parser.add_argument(
        "--topic",
        choices=list(DEBATE_TOPICS.keys()),
        help="Debate topic to run"
    )
    parser.add_argument(
        "--matchup",
        choices=list(MATCHUPS.keys()),
        help="Predefined matchup to run"
    )
    parser.add_argument(
        "--participants",
        help="Comma-separated participant IDs"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Number of debate rounds"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-run without interaction"
    )
    parser.add_argument(
        "--provider",
        choices=["openrouter", "anthropic", "openai"],
        default="openrouter",
        help="LLM provider (default: openrouter)"
    )
    parser.add_argument(
        "--model",
        help="Model ID (e.g., anthropic/claude-sonnet-4, openai/gpt-4o)"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM (use templates)"
    )
    parser.add_argument(
        "--list-all",
        action="store_true",
        help="List personas, topics, and matchups"
    )

    args = parser.parse_args()

    runner = AIDebateRunner(args.scenario)

    if args.list_all:
        runner.list_personas()
        runner.list_topics()
        runner.list_matchups()
        return

    if args.matchup:
        asyncio.run(runner.run_matchup(
            args.matchup,
            topic_id=args.topic,
            rounds=args.rounds,
            auto=args.auto,
            llm_provider=args.provider,
            llm_model=args.model,
        ))
    elif args.topic:
        participants = []
        if args.participants:
            participants = [p.strip() for p in args.participants.split(",")]
        else:
            participants = DEBATE_TOPICS[args.topic]["recommended_participants"]

        asyncio.run(runner.run_debate(
            args.topic,
            participants,
            rounds=args.rounds,
            auto=args.auto,
            use_llm=not args.no_llm,
            llm_provider=args.provider,
            llm_model=args.model,
        ))
    else:
        runner.interactive_menu()


if __name__ == "__main__":
    main()
