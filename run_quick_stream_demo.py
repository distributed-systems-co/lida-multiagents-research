#!/usr/bin/env python3
"""
Quick Streaming Demo - Simplified for reliability

Shows streaming LLM analysis with GDELT feeds in a cleaner format.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.meta import MLXClient, MLXModelConfig, PERSONALITY_ARCHETYPES
from src.meta.industrial_intelligence import IndustrialEvent, IndustrialEventType
from run_live_quorum import GDELTLiveFeed


# Colors
class C:
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    MAG = '\033[35m'


async def stream_analysis(event: IndustrialEvent, agent_name: str, personality_key: str):
    """Stream a concise analysis."""
    config = MLXModelConfig(
        model_path="mlx-community/Josiefied-Qwen3-4B-abliterated-v1-4bit",
        max_tokens=150,  # Keep it SHORT
        temperature=0.7
    )
    client = MLXClient(config)

    # Set personality
    if personality_key in PERSONALITY_ARCHETYPES:
        client.personality = PERSONALITY_ARCHETYPES[personality_key]()

    prompt = f"""Analyze this corporate event in 1-2 sentences:

Event: {event.title}
Company: {event.primary_company}

Your analysis (be concise):"""

    print(f"{C.BOLD}{C.CYAN}{agent_name}:{C.END} ", end="", flush=True)

    full = ""
    async for token in client.generate_stream(prompt):
        print(token, end="", flush=True)
        full += token
        if len(full) > 200:  # Cap at 200 chars
            break

    print(f"{C.END}\n")
    return full


async def main():
    print(f"\n{C.BOLD}{C.MAG}{'='*60}")
    print(" LIVE STREAMING DEMO")
    print(f"{'='*60}{C.END}\n")

    print(f"{C.DIM}Fetching live GDELT data...{C.END}")
    gdelt = GDELTLiveFeed()
    events = await gdelt.fetch_latest_events(limit=100)

    if events:
        relevant = gdelt.filter_relevant_events(events)
        if relevant:
            gdelt_event = relevant[0]
            company = gdelt_event.get('matched_company', 'Google')
            headline = f"{company}: {gdelt_event.get('Actor1Name', 'COMPANY')} - {gdelt_event.get('Actor2Name', 'NEWS')}"

            event = IndustrialEvent(
                event_id=gdelt_event['GLOBALEVENTID'],
                event_type=None,  # General event
                timestamp=datetime.utcnow(),
                primary_company=company,
                title=headline[:80]
            )
            print(f"{C.GREEN}✓ Found live event: {event.title}{C.END}\n")
        else:
            print(f"{C.YELLOW}Using sample event{C.END}\n")
            event = IndustrialEvent(
                event_id="sample",
                event_type=IndustrialEventType.ACQUISITION_ANNOUNCED,
                timestamp=datetime.utcnow(),
                primary_company="Nvidia",
                title="Nvidia announces $20B acquisition of Cerebras Systems",
                value_billions=20.0
            )
    else:
        print(f"{C.YELLOW}Using sample event{C.END}\n")
        event = IndustrialEvent(
            event_id="sample",
            event_type=IndustrialEventType.ACQUISITION_ANNOUNCED,
            timestamp=datetime.utcnow(),
            primary_company="Nvidia",
            title="Nvidia announces $20B acquisition of Cerebras Systems",
            value_billions=20.0
        )

    print(f"{C.CYAN}{'─'*60}{C.END}")
    print(f"{C.BOLD}Event: {event.title}{C.END}")
    print(f"{C.CYAN}{'─'*60}{C.END}\n")

    print(f"{C.BOLD}Multi-agent streaming analysis:{C.END}\n")

    # Run 3 agents with streaming
    agents = [
        ("Market Analyst (INTJ)", "mbti_intj"),
        ("Risk Manager (Enneagram 6)", "enneagram_6"),
        ("Opportunity Scout (ENFP)", "mbti_enfp")
    ]

    tasks = [stream_analysis(event, name, personality) for name, personality in agents]
    await asyncio.gather(*tasks)

    print(f"\n{C.GREEN}{C.BOLD}✓ Live streaming complete!{C.END}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}Interrupted.{C.END}")
