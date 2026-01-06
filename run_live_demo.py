#!/usr/bin/env python3
"""
Live Demo: Streaming Personalities + GDELT Data

Combines:
- MLX streaming personality models
- Live GDELT news feeds
- Real-time quorum deliberation
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from run_mlx_quorum import MLXEmotionalQuorum, MLX_AGENT_ROLES, Colors
from run_live_quorum import GDELTLiveFeed, WATCHED_COMPANIES, EVENT_KEYWORDS
from src.meta.industrial_intelligence import IndustrialEvent, IndustrialEventType


async def run_live_demo():
    """Run live demo with streaming personalities and GDELT data."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}")
    print(" LIVE DEMO: STREAMING PERSONALITIES + GDELT NEWS")
    print(f"{'='*70}{Colors.ENDC}\n")

    print(f"{Colors.CYAN}Initializing systems...{Colors.ENDC}")
    print(f"  • MLX Streaming Personalities: {len(MLX_AGENT_ROLES)} agents")
    print(f"  • GDELT Live Feed: Monitoring {len(WATCHED_COMPANIES)} companies")
    print()

    # Initialize systems
    gdelt = GDELTLiveFeed()
    quorum = MLXEmotionalQuorum()

    print(f"{Colors.CYAN}{'─'*50}")
    print(" FETCHING LIVE GDELT DATA...")
    print(f"{'─'*50}{Colors.ENDC}\n")

    # Fetch latest events
    events = await gdelt.fetch_latest_events(limit=500)

    if not events:
        print(f"{Colors.YELLOW}No new GDELT data yet. Using sample events.{Colors.ENDC}\n")

        # Use sample events
        sample_events = [
            ("Nvidia announces $20B acquisition of Cerebras Systems", IndustrialEventType.ACQUISITION_ANNOUNCED, 20.0, "Nvidia"),
            ("Anthropic raises $5B Series C led by Google", IndustrialEventType.FUNDING_ROUND, 5.0, "Anthropic"),
            ("Microsoft considering $15B acquisition of Mistral AI", IndustrialEventType.ACQUISITION_ANNOUNCED, 15.0, "Microsoft"),
        ]

        for headline, event_type, value, company in sample_events:
            print(f"{Colors.BOLD}EVENT:{Colors.ENDC} {headline}\n")

            event = IndustrialEvent(
                event_id=f"demo_{company}",
                event_type=event_type,
                timestamp=datetime.utcnow(),
                primary_company=company,
                title=headline,
                value_billions=value,
            )

            print(f"{Colors.DIM}Deliberating with streaming personalities...{Colors.ENDC}\n")
            opinions = await quorum.deliberate(event)

            # Show consensus
            analysis = quorum.analyze_consensus(opinions)

            print(f"\n{Colors.CYAN}{'─'*50}")
            print(f" QUORUM RESULT")
            print(f"{'─'*50}{Colors.ENDC}\n")

            from run_mlx_quorum import format_stance
            print(f"  Consensus: {format_stance(analysis['consensus_stance'])}")
            print(f"  Strength: {analysis['consensus_strength']:.0%}")
            print(f"  Dissent: {analysis['dissent_level']:.0%}\n")

            print(f"{Colors.DIM}{'─'*70}{Colors.ENDC}\n")

            await asyncio.sleep(1)
    else:
        print(f"{Colors.GREEN}Fetched {len(events)} GDELT events{Colors.ENDC}\n")

        # Filter for relevant companies
        relevant = gdelt.filter_relevant_events(events)

        if relevant:
            print(f"{Colors.GREEN}Found {len(relevant)} relevant events!{Colors.ENDC}\n")

            # Process top 3
            for gdelt_event in relevant[:3]:
                company = gdelt_event.get('matched_company', 'Unknown')

                # Classify event type
                url_lower = gdelt_event.get('SOURCEURL', '').lower()
                event_type = None
                for keyword, etype in EVENT_KEYWORDS.items():
                    if keyword in url_lower:
                        event_type = etype
                        break

                headline = f"{company}: {gdelt_event.get('Actor1Name', '')} - {gdelt_event.get('Actor2Name', '')}"

                print(f"{Colors.BOLD}LIVE EVENT:{Colors.ENDC} {headline[:80]}")
                print(f"  Source: {gdelt_event.get('SOURCEURL', 'N/A')[:60]}...")
                print(f"  Tone: {gdelt_event.get('AvgTone', 0):.1f}\n")

                event = IndustrialEvent(
                    event_id=gdelt_event['GLOBALEVENTID'],
                    event_type=event_type,
                    timestamp=datetime.utcnow(),
                    primary_company=company,
                    title=headline,
                )

                print(f"{Colors.DIM}Streaming personality analysis...{Colors.ENDC}\n")
                opinions = await quorum.deliberate(event)

                # Show consensus
                analysis = quorum.analyze_consensus(opinions)

                print(f"\n{Colors.CYAN}{'─'*50}")
                print(f" QUORUM CONSENSUS")
                print(f"{'─'*50}{Colors.ENDC}\n")

                from run_mlx_quorum import format_stance
                print(f"  Consensus: {format_stance(analysis['consensus_stance'])}")
                print(f"  Agreement: {analysis['consensus_strength']:.0%}")
                print(f"  Dissent: {analysis['dissent_level']:.0%}\n")

                print(f"{Colors.DIM}{'─'*70}{Colors.ENDC}\n")

                await asyncio.sleep(1)
        else:
            print(f"{Colors.YELLOW}No events matching watched companies. Using samples.{Colors.ENDC}\n")

    print(f"{Colors.GREEN}{Colors.BOLD}DEMO COMPLETE{Colors.ENDC}")
    print(f"{Colors.DIM}All systems operational.{Colors.ENDC}\n")


if __name__ == "__main__":
    try:
        asyncio.run(run_live_demo())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Demo interrupted.{Colors.ENDC}")
