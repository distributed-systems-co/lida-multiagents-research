#!/usr/bin/env python3
"""
MLX-Powered Emotional Quorum

Uses local MLX models on Apple Silicon to run agent deliberations
with actual LLM reasoning instead of simulated responses.

Each agent role gets a different personality and provides real analysis.

Usage:
    python run_mlx_quorum.py [--event "headline"] [--turns N]
"""

import asyncio
import argparse
import sys
import re
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

# Suppress warnings
logging.getLogger("root").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

from src.meta import MLXClient, MLXModelConfig, PERSONALITY_ARCHETYPES
from src.meta.industrial_intelligence import (
    IndustrialEvent,
    IndustrialEventType,
    IndustrialSector,
    EmotionalStance,
)


# ANSI colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


@dataclass
class MLXAgentRole:
    """An agent role with personality for MLX-based deliberation."""
    role_id: str
    role_name: str
    personality_key: str  # Key into PERSONALITY_ARCHETYPES
    system_prompt: str
    default_stance: EmotionalStance


# Agent roles mapped to personality types
MLX_AGENT_ROLES = [
    MLXAgentRole(
        role_id="market_analyst",
        role_name="Market Analyst",
        personality_key="mbti_intj",  # Strategic, analytical
        system_prompt="""You are a senior market analyst focused on competitive dynamics and valuations.
Analyze events for market impact, competitive positioning, and strategic implications.
Be direct and data-driven in your assessment.""",
        default_stance=EmotionalStance.ANALYTICAL,
    ),
    MLXAgentRole(
        role_id="risk_manager",
        role_name="Risk Manager",
        personality_key="enneagram_6",  # Security-focused, skeptical
        system_prompt="""You are a chief risk officer focused on identifying threats and vulnerabilities.
Look for regulatory risks, execution risks, and potential failure modes.
Be cautious and thorough in your risk assessment.""",
        default_stance=EmotionalStance.CAUTIOUS,
    ),
    MLXAgentRole(
        role_id="opportunity_scout",
        role_name="Opportunity Scout",
        personality_key="mbti_enfp",  # Enthusiastic, possibility-focused
        system_prompt="""You are a business development lead looking for opportunities.
Identify potential partnerships, acquisition targets, and strategic moves.
Be optimistic but grounded in your opportunity assessment.""",
        default_stance=EmotionalStance.OPPORTUNISTIC,
    ),
    MLXAgentRole(
        role_id="sector_specialist",
        role_name="Sector Specialist",
        personality_key="mbti_intp",  # Deep expertise, theoretical
        system_prompt="""You are a sector expert with deep technical knowledge.
Analyze the technology, competitive moat, and industry dynamics.
Be thorough and technically precise in your assessment.""",
        default_stance=EmotionalStance.ANALYTICAL,
    ),
    MLXAgentRole(
        role_id="contrarian",
        role_name="Devil's Advocate",
        personality_key="enneagram_8",  # Challenger, provocative
        system_prompt="""You are a contrarian analyst who challenges consensus views.
Look for what everyone else is missing. Question assumptions.
Be provocative but substantive in your counterarguments.""",
        default_stance=EmotionalStance.CONTRARIAN,
    ),
]


def format_stance(stance: EmotionalStance) -> str:
    """Format stance with color."""
    colors = {
        EmotionalStance.BULLISH: Colors.GREEN,
        EmotionalStance.EXCITED: Colors.GREEN + Colors.BOLD,
        EmotionalStance.BEARISH: Colors.RED,
        EmotionalStance.ALARMED: Colors.RED + Colors.BOLD,
        EmotionalStance.CAUTIOUS: Colors.YELLOW,
        EmotionalStance.SKEPTICAL: Colors.YELLOW,
        EmotionalStance.NEUTRAL: Colors.DIM,
        EmotionalStance.CONTRARIAN: Colors.BLUE,
        EmotionalStance.OPPORTUNISTIC: Colors.CYAN,
        EmotionalStance.ANALYTICAL: Colors.BLUE,
    }
    color = colors.get(stance, "")
    return f"{color}{stance.value.upper()}{Colors.ENDC}"


def clean_response(text: str) -> str:
    """Clean up model output."""
    text = re.sub(r'<\|[^>]+\|>', '', text)
    text = re.sub(r'\|<\|[^>]+\|>', '', text)
    text = re.sub(r'</s>', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    # Get first substantial paragraph
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 20]
    if paragraphs:
        return paragraphs[0][:500]  # Limit length
    return text[:500] if text else "(no response)"


def classify_stance(text: str) -> EmotionalStance:
    """Classify the stance from the response text."""
    text_lower = text.lower()

    # Check for explicit stance signals
    if any(w in text_lower for w in ["alarming", "dangerous", "threat", "serious risk", "very concerned"]):
        return EmotionalStance.ALARMED
    if any(w in text_lower for w in ["bullish", "optimistic", "strong buy", "very positive"]):
        return EmotionalStance.BULLISH
    if any(w in text_lower for w in ["exciting", "huge opportunity", "transformative", "game-changer"]):
        return EmotionalStance.EXCITED
    if any(w in text_lower for w in ["bearish", "pessimistic", "sell", "overvalued", "bubble"]):
        return EmotionalStance.BEARISH
    if any(w in text_lower for w in ["cautious", "careful", "wait and see", "uncertain"]):
        return EmotionalStance.CAUTIOUS
    if any(w in text_lower for w in ["skeptical", "doubt", "questionable", "unproven"]):
        return EmotionalStance.SKEPTICAL
    if any(w in text_lower for w in ["however", "but consider", "on the other hand", "disagree"]):
        return EmotionalStance.CONTRARIAN
    if any(w in text_lower for w in ["opportunity", "potential", "could benefit", "upside"]):
        return EmotionalStance.OPPORTUNISTIC

    return EmotionalStance.ANALYTICAL


@dataclass
class MLXAgentOpinion:
    """An opinion from an MLX-powered agent."""
    agent_role: MLXAgentRole
    raw_response: str
    cleaned_response: str
    stance: EmotionalStance
    confidence: float
    generation_time_ms: float


class MLXEmotionalQuorum:
    """Emotional quorum powered by MLX local models."""

    def __init__(
        self,
        model_path: str = "mlx-community/Josiefied-Qwen3-4B-abliterated-v1-4bit",
        max_tokens: int = 300,
        temperature: float = 0.7,
    ):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client: Optional[MLXClient] = None
        self._config: Optional[MLXModelConfig] = None

    def _get_client(self) -> MLXClient:
        """Lazy load the MLX client."""
        if self._client is None:
            self._config = MLXModelConfig(
                model_path=self.model_path,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            self._client = MLXClient(self._config)
        return self._client

    async def get_agent_opinion(
        self,
        role: MLXAgentRole,
        event: IndustrialEvent,
        context: str = "",
        stream: bool = True,
    ) -> MLXAgentOpinion:
        """Get an opinion from a single agent."""
        client = self._get_client()

        # Set personality
        if role.personality_key in PERSONALITY_ARCHETYPES:
            client.personality = PERSONALITY_ARCHETYPES[role.personality_key]()

        # Build prompt
        prompt = f"""{role.system_prompt}

EVENT: {event.title}
Company: {event.primary_company}
Type: {event.event_type.value if event.event_type else 'general'}
{f'Value: ${event.value_billions}B' if event.value_billions else ''}
{f'Context: {context}' if context else ''}

Provide your analysis in 2-3 sentences. State your stance (bullish/bearish/cautious/etc) and reasoning.
What's your assessment and recommended action?"""

        start_time = datetime.now()

        if stream:
            # Stream the response token by token
            full_response = ""
            print(f"{Colors.BOLD}[{role.role_name}]{Colors.ENDC}", end=" ", flush=True)

            async for token in client.generate_stream(prompt):
                print(token, end="", flush=True)
                full_response += token

            print()  # Newline after streaming
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

            cleaned = clean_response(full_response)
            stance = classify_stance(cleaned)

            # Estimate confidence from response strength
            confidence = 0.7
            if any(w in cleaned.lower() for w in ["very", "strongly", "clearly", "definitely"]):
                confidence = 0.85
            elif any(w in cleaned.lower() for w in ["might", "perhaps", "possibly", "uncertain"]):
                confidence = 0.5

            return MLXAgentOpinion(
                agent_role=role,
                raw_response=full_response,
                cleaned_response=cleaned,
                stance=stance,
                confidence=confidence,
                generation_time_ms=elapsed_ms,
            )
        else:
            # Non-streaming fallback
            response = await client.generate(prompt)
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

            cleaned = clean_response(response.text)
            stance = classify_stance(cleaned)

            # Estimate confidence from response strength
            confidence = 0.7
            if any(w in cleaned.lower() for w in ["very", "strongly", "clearly", "definitely"]):
                confidence = 0.85
            elif any(w in cleaned.lower() for w in ["might", "perhaps", "possibly", "uncertain"]):
                confidence = 0.5

            return MLXAgentOpinion(
                agent_role=role,
                raw_response=response.text,
                cleaned_response=cleaned,
                stance=stance,
                confidence=confidence,
                generation_time_ms=elapsed_ms,
            )

    async def deliberate(
        self,
        event: IndustrialEvent,
        roles: Optional[List[MLXAgentRole]] = None,
        context: str = "",
    ) -> List[MLXAgentOpinion]:
        """Run full quorum deliberation."""
        if roles is None:
            roles = MLX_AGENT_ROLES

        opinions = []
        for role in roles:
            opinion = await self.get_agent_opinion(role, event, context)
            opinions.append(opinion)

        return opinions

    def analyze_consensus(self, opinions: List[MLXAgentOpinion]) -> Dict[str, Any]:
        """Analyze the opinions for consensus."""
        if not opinions:
            return {"consensus": None, "strength": 0, "dissent": 1.0}

        # Count stances
        stance_counts: Dict[EmotionalStance, int] = {}
        total_confidence = 0.0

        for op in opinions:
            stance_counts[op.stance] = stance_counts.get(op.stance, 0) + 1
            total_confidence += op.confidence

        # Find majority
        max_stance = max(stance_counts.items(), key=lambda x: x[1])
        consensus_strength = max_stance[1] / len(opinions)
        dissent = 1.0 - consensus_strength

        return {
            "consensus_stance": max_stance[0],
            "consensus_strength": consensus_strength,
            "dissent_level": dissent,
            "stance_distribution": {s.value: c for s, c in stance_counts.items()},
            "avg_confidence": total_confidence / len(opinions),
        }


async def run_mlx_quorum(
    headline: str,
    event_type: Optional[IndustrialEventType] = None,
    value_billions: float = 0.0,
    turns: int = 1,
):
    """Run the MLX quorum on an event."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}")
    print(" MLX-POWERED EMOTIONAL QUORUM")
    print(f"{'='*70}{Colors.ENDC}\n")

    print(f"Model: mlx-community/Josiefied-Qwen3-4B-abliterated-v1-4bit")
    print(f"Agents: {len(MLX_AGENT_ROLES)}")
    print()

    # Create event
    event = IndustrialEvent(
        event_id="manual_1",
        event_type=event_type or IndustrialEventType.ACQUISITION_ANNOUNCED,
        timestamp=datetime.utcnow(),
        primary_company=headline.split()[0],  # First word as company
        title=headline,
        value_billions=value_billions,
    )

    print(f"{Colors.CYAN}{'─'*50}")
    print(f" EVENT: {headline}")
    print(f"{'─'*50}{Colors.ENDC}\n")

    quorum = MLXEmotionalQuorum()

    for turn in range(turns):
        if turns > 1:
            print(f"\n{Colors.BOLD}── Round {turn + 1}/{turns} ──{Colors.ENDC}\n")

        print(f"{Colors.DIM}Gathering agent opinions (streaming)...{Colors.ENDC}\n")

        opinions = await quorum.deliberate(event)

        # Display summary for each opinion
        print()
        for op in opinions:
            stance_str = format_stance(op.stance)
            conf_bar = "█" * int(op.confidence * 10) + "░" * (10 - int(op.confidence * 10))

            print(f"{Colors.DIM}[{op.agent_role.role_name}] {stance_str} | [{conf_bar}] {op.confidence:.0%} | {op.generation_time_ms:.0f}ms{Colors.ENDC}")

        # Analyze consensus
        analysis = quorum.analyze_consensus(opinions)

        print(f"{Colors.CYAN}{'─'*50}")
        print(f" CONSENSUS ANALYSIS")
        print(f"{'─'*50}{Colors.ENDC}\n")

        print(f"  Consensus: {format_stance(analysis['consensus_stance'])}")
        print(f"  Strength: {analysis['consensus_strength']:.0%}")
        print(f"  Dissent: {analysis['dissent_level']:.0%}")
        print(f"  Avg Confidence: {analysis['avg_confidence']:.0%}")

        print(f"\n  Stance Distribution:")
        for stance, count in analysis['stance_distribution'].items():
            bar = "█" * count
            print(f"    {stance:15} {bar} ({count})")

    print(f"\n{Colors.GREEN}Done.{Colors.ENDC}\n")


# Sample events for testing
SAMPLE_EVENTS = [
    ("Nvidia announces $20B acquisition of Cerebras Systems", IndustrialEventType.ACQUISITION_ANNOUNCED, 20.0),
    ("OpenAI raises $40B at $300B valuation from SoftBank", IndustrialEventType.FUNDING_ROUND, 40.0),
    ("Anthropic CEO warns of AGI risks, calls for regulation", None, 0.0),
    ("Figure AI lays off 30% of workforce amid funding crunch", IndustrialEventType.LAYOFFS, 0.0),
    ("Adobe completes $8B acquisition of Runway AI", IndustrialEventType.ACQUISITION_COMPLETED, 8.0),
    ("Tesla FSD achieves Level 4 autonomy in test markets", IndustrialEventType.PRODUCT_LAUNCH, 0.0),
    ("Microsoft considering $15B acquisition of Mistral AI", IndustrialEventType.ACQUISITION_ANNOUNCED, 15.0),
]


async def run_demo():
    """Run demo with sample events."""
    print(f"\n{Colors.HEADER}MLX Quorum Demo - Running 3 sample events{Colors.ENDC}\n")

    for headline, event_type, value in SAMPLE_EVENTS[:3]:
        await run_mlx_quorum(headline, event_type, value)
        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MLX-powered emotional quorum")
    parser.add_argument("--event", type=str, help="Event headline to analyze")
    parser.add_argument("--value", type=float, default=0.0, help="Deal value in billions")
    parser.add_argument("--turns", type=int, default=1, help="Number of deliberation rounds")
    parser.add_argument("--demo", action="store_true", help="Run demo with sample events")

    args = parser.parse_args()

    try:
        if args.demo:
            asyncio.run(run_demo())
        elif args.event:
            asyncio.run(run_mlx_quorum(args.event, value_billions=args.value, turns=args.turns))
        else:
            # Default: run one sample event
            asyncio.run(run_mlx_quorum(
                "Nvidia announces $20B acquisition of Cerebras Systems",
                IndustrialEventType.ACQUISITION_ANNOUNCED,
                20.0
            ))
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted.{Colors.ENDC}")
