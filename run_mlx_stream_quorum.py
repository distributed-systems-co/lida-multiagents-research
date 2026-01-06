#!/usr/bin/env python3
"""
MLX Streaming Emotional Quorum

Real-time streaming responses from MLX-powered agents.
Each agent's response streams token-by-token to the terminal.

Usage:
    python run_mlx_stream_quorum.py [--event "headline"] [--model MODEL]
"""

import asyncio
import argparse
import sys
import re
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncIterator
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
    MAGENTA = '\033[35m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


# Agent colors for visual distinction
AGENT_COLORS = [
    Colors.CYAN,
    Colors.GREEN,
    Colors.YELLOW,
    Colors.MAGENTA,
    Colors.BLUE,
]


@dataclass
class StreamingAgent:
    """An agent that streams responses."""
    agent_id: str
    name: str
    personality_key: str
    role_prompt: str
    color: str


STREAMING_AGENTS = [
    StreamingAgent(
        agent_id="analyst",
        name="Market Analyst",
        personality_key="mbti_intj",
        role_prompt="You are a strategic market analyst. Analyze competitive dynamics and valuations. Be direct.",
        color=Colors.CYAN,
    ),
    StreamingAgent(
        agent_id="risk",
        name="Risk Manager",
        personality_key="enneagram_6",
        role_prompt="You are a risk-focused analyst. Identify threats, regulatory risks, and failure modes. Be cautious.",
        color=Colors.YELLOW,
    ),
    StreamingAgent(
        agent_id="opportunity",
        name="Opportunity Scout",
        personality_key="mbti_enfp",
        role_prompt="You are an optimistic business developer. Find opportunities and strategic angles. Be enthusiastic.",
        color=Colors.GREEN,
    ),
    StreamingAgent(
        agent_id="technical",
        name="Tech Specialist",
        personality_key="mbti_intp",
        role_prompt="You are a deep technical expert. Analyze technology, moats, and industry dynamics. Be precise.",
        color=Colors.MAGENTA,
    ),
    StreamingAgent(
        agent_id="contrarian",
        name="Devil's Advocate",
        personality_key="enneagram_8",
        role_prompt="You challenge consensus views. Question assumptions. Find what others miss. Be provocative.",
        color=Colors.RED,
    ),
]


class MLXStreamingQuorum:
    """Quorum that streams agent responses in real-time."""

    def __init__(
        self,
        model_path: str = "mlx-community/Josiefied-Qwen3-4B-abliterated-v1-4bit",
        max_tokens: int = 250,
        temperature: float = 0.7,
    ):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._clients: Dict[str, MLXClient] = {}
        self._config: Optional[MLXModelConfig] = None

    def _get_client(self, agent: StreamingAgent) -> MLXClient:
        """Get or create client for agent."""
        if agent.agent_id not in self._clients:
            if self._config is None:
                self._config = MLXModelConfig(
                    model_path=self.model_path,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )

            client = MLXClient(self._config)

            # Set personality
            if agent.personality_key in PERSONALITY_ARCHETYPES:
                client.personality = PERSONALITY_ARCHETYPES[agent.personality_key]()

            self._clients[agent.agent_id] = client

        return self._clients[agent.agent_id]

    async def stream_agent_response(
        self,
        agent: StreamingAgent,
        event_headline: str,
        event_type: str = "",
        value: float = 0.0,
    ) -> AsyncIterator[str]:
        """Stream response from a single agent."""
        client = self._get_client(agent)

        prompt = f"""{agent.role_prompt}

EVENT: {event_headline}
{f'Type: {event_type}' if event_type else ''}
{f'Value: ${value}B' if value else ''}

Give your analysis in 2-3 sentences. State your stance and key reasoning."""

        async for token in client.generate_stream(prompt):
            yield token

    async def run_streaming_deliberation(
        self,
        event_headline: str,
        event_type: str = "",
        value: float = 0.0,
        agents: Optional[List[StreamingAgent]] = None,
    ):
        """Run full deliberation with streaming output."""
        if agents is None:
            agents = STREAMING_AGENTS

        print(f"\n{Colors.HEADER}{Colors.BOLD}{'═'*70}")
        print(f" MLX STREAMING QUORUM")
        print(f"{'═'*70}{Colors.ENDC}\n")

        print(f"{Colors.DIM}Model: {self.model_path}{Colors.ENDC}")
        print(f"{Colors.DIM}Agents: {len(agents)}{Colors.ENDC}\n")

        print(f"{Colors.CYAN}{'─'*50}")
        print(f" EVENT: {event_headline}")
        if value:
            print(f" Value: ${value}B")
        print(f"{'─'*50}{Colors.ENDC}\n")

        responses = []
        total_start = datetime.now()

        for i, agent in enumerate(agents):
            color = agent.color
            print(f"{color}{Colors.BOLD}┌─ [{agent.name}] ─────────────────────────────{Colors.ENDC}")
            print(f"{color}│{Colors.ENDC} ", end="", flush=True)

            start_time = datetime.now()
            full_response = ""
            char_count = 0
            line_width = 60

            async for token in self.stream_agent_response(agent, event_headline, event_type, value):
                # Clean token
                token = token.replace("<|", "").replace("|>", "").replace("</s>", "")
                if not token:
                    continue

                full_response += token

                # Print with word wrap
                for char in token:
                    if char == '\n':
                        print(f"\n{color}│{Colors.ENDC} ", end="", flush=True)
                        char_count = 0
                    else:
                        print(char, end="", flush=True)
                        char_count += 1
                        if char_count >= line_width and char == ' ':
                            print(f"\n{color}│{Colors.ENDC} ", end="", flush=True)
                            char_count = 0

            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"\n{color}└─ {Colors.DIM}({elapsed:.1f}s){Colors.ENDC}\n")

            responses.append({
                "agent": agent.name,
                "response": full_response,
                "time": elapsed,
            })

        # Summary
        total_elapsed = (datetime.now() - total_start).total_seconds()

        print(f"{Colors.CYAN}{'─'*50}")
        print(f" DELIBERATION COMPLETE")
        print(f"{'─'*50}{Colors.ENDC}\n")

        print(f"  Total time: {total_elapsed:.1f}s")
        print(f"  Agents: {len(responses)}")

        # Quick stance analysis
        stances = {"bullish": 0, "bearish": 0, "cautious": 0, "neutral": 0}
        for r in responses:
            text = r["response"].lower()
            if "bullish" in text or "optimistic" in text or "opportunity" in text:
                stances["bullish"] += 1
            elif "bearish" in text or "risk" in text or "concern" in text:
                stances["bearish"] += 1
            elif "cautious" in text or "careful" in text:
                stances["cautious"] += 1
            else:
                stances["neutral"] += 1

        print(f"\n  Stance summary:")
        for stance, count in stances.items():
            if count > 0:
                bar = "█" * count
                print(f"    {stance:10} {bar} ({count})")

        print()
        return responses


async def run_interactive():
    """Interactive mode - enter events and see streaming responses."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'═'*70}")
    print(f" MLX STREAMING QUORUM - INTERACTIVE MODE")
    print(f"{'═'*70}{Colors.ENDC}\n")

    print("Enter event headlines to analyze. Type 'quit' to exit.\n")

    quorum = MLXStreamingQuorum()

    while True:
        try:
            event = input(f"{Colors.CYAN}Event> {Colors.ENDC}").strip()

            if event.lower() in ['quit', 'exit', 'q']:
                break

            if not event:
                continue

            # Check for value
            value = 0.0
            if "$" in event:
                import re
                match = re.search(r'\$(\d+(?:\.\d+)?)\s*[Bb]', event)
                if match:
                    value = float(match.group(1))

            await quorum.run_streaming_deliberation(event, value=value)

        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Interrupted.{Colors.ENDC}")
            break
        except EOFError:
            break

    print(f"\n{Colors.GREEN}Goodbye.{Colors.ENDC}\n")


async def main(event: Optional[str] = None, value: float = 0.0, interactive: bool = False):
    """Main entry point."""
    if interactive:
        await run_interactive()
    elif event:
        quorum = MLXStreamingQuorum()
        await quorum.run_streaming_deliberation(event, value=value)
    else:
        # Default demo
        quorum = MLXStreamingQuorum()
        await quorum.run_streaming_deliberation(
            "Nvidia announces $20B acquisition of Cerebras Systems",
            event_type="acquisition",
            value=20.0,
        )


# Sample events
SAMPLE_EVENTS = [
    ("Nvidia announces $20B acquisition of Cerebras Systems", 20.0),
    ("OpenAI raises $40B at $300B valuation from SoftBank", 40.0),
    ("Anthropic CEO warns of AGI risks, calls for immediate regulation", 0.0),
    ("Figure AI lays off 30% of workforce amid funding difficulties", 0.0),
    ("Microsoft in talks to acquire Mistral AI for $15B", 15.0),
    ("Tesla FSD achieves full Level 4 autonomy certification", 0.0),
    ("Amazon announces $10B investment in Anthropic", 10.0),
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLX Streaming Quorum")
    parser.add_argument("--event", "-e", type=str, help="Event headline to analyze")
    parser.add_argument("--value", "-v", type=float, default=0.0, help="Deal value in billions")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--model", "-m", type=str,
                       default="mlx-community/Josiefied-Qwen3-4B-abliterated-v1-4bit",
                       help="MLX model path")

    args = parser.parse_args()

    try:
        asyncio.run(main(event=args.event, value=args.value, interactive=args.interactive))
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted.{Colors.ENDC}")
