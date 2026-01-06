#!/usr/bin/env python3
"""
Live GDELT Quorum Runner

Connects to real GDELT data and runs the emotional quorum on live global events.
GDELT updates every 15 minutes with worldwide news events.

Usage:
    python run_live_quorum.py [--cycles N] [--watch COMPANY]
"""

import asyncio
import argparse
import json
import sys
import zipfile
import io
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import urllib.request

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from meta.realtime_quorum import (
    RealTimeQuorumSystem,
    RealTimeEvent,
    QuorumDeliberation,
)
from meta.industrial_intelligence import (
    IndustrialEventType,
    IndustrialSector,
    EmotionalStance,
    get_industrial_registry,
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


# Companies to watch for in GDELT
WATCHED_COMPANIES = {
    # AI Companies
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "nvidia": "Nvidia",
    "microsoft": "Microsoft",
    "google": "Google",
    "alphabet": "Alphabet",
    "meta": "Meta",
    "amazon": "Amazon",
    "apple": "Apple",
    "tesla": "Tesla",
    "spacex": "SpaceX",

    # AI Startups
    "cursor": "Cursor",
    "cerebras": "Cerebras",
    "groq": "Groq",
    "cohere": "Cohere",
    "mistral": "Mistral",
    "runway": "Runway",
    "elevenlabs": "ElevenLabs",
    "hugging face": "Hugging Face",
    "stability ai": "Stability AI",
    "midjourney": "Midjourney",

    # Robotics
    "figure ai": "Figure AI",
    "boston dynamics": "Boston Dynamics",
    "waymo": "Waymo",
    "cruise": "Cruise",
    "aurora": "Aurora",

    # Defense
    "anduril": "Anduril",
    "palantir": "Palantir",
    "lockheed": "Lockheed Martin",
    "raytheon": "Raytheon",

    # Enterprise
    "salesforce": "Salesforce",
    "servicenow": "ServiceNow",
    "databricks": "Databricks",
    "snowflake": "Snowflake",

    # Semiconductors
    "tsmc": "TSMC",
    "asml": "ASML",
    "amd": "AMD",
    "intel": "Intel",
    "qualcomm": "Qualcomm",
}

# Event keywords for classification
EVENT_KEYWORDS = {
    "acquire": IndustrialEventType.ACQUISITION_ANNOUNCED,
    "acquisition": IndustrialEventType.ACQUISITION_ANNOUNCED,
    "merger": IndustrialEventType.MERGER_ANNOUNCED,
    "ipo": IndustrialEventType.IPO_FILED,
    "public offering": IndustrialEventType.IPO_FILED,
    "funding": IndustrialEventType.FUNDING_ROUND,
    "investment": IndustrialEventType.FUNDING_ROUND,
    "layoff": IndustrialEventType.LAYOFFS,
    "job cuts": IndustrialEventType.LAYOFFS,
    "restructur": IndustrialEventType.LAYOFFS,
    "bankrupt": IndustrialEventType.BANKRUPTCY,
    "partnership": IndustrialEventType.PARTNERSHIP_ANNOUNCED,
    "collaboration": IndustrialEventType.PARTNERSHIP_ANNOUNCED,
    "ceo": IndustrialEventType.CEO_CHANGE,
    "executive": IndustrialEventType.KEY_HIRE,
    "appointed": IndustrialEventType.KEY_HIRE,
    "lawsuit": IndustrialEventType.REGULATORY_BLOCK,
    "antitrust": IndustrialEventType.REGULATORY_BLOCK,
    "regulation": IndustrialEventType.REGULATORY_BLOCK,
    "launch": IndustrialEventType.PRODUCT_LAUNCH,
    "release": IndustrialEventType.PRODUCT_LAUNCH,
    "announce": IndustrialEventType.PRODUCT_LAUNCH,
}


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


class GDELTLiveFeed:
    """Fetches live events from GDELT."""

    GDELT_LAST_UPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"

    def __init__(self):
        self.last_file_processed = None
        self.events_seen = set()

    async def fetch_latest_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch the latest GDELT events."""
        try:
            # Get latest file URL
            with urllib.request.urlopen(self.GDELT_LAST_UPDATE_URL, timeout=10) as response:
                content = response.read().decode('utf-8')

            # Parse to get export file URL
            lines = content.strip().split('\n')
            export_url = None
            for line in lines:
                if 'export' in line.lower() and '.zip' in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        export_url = parts[2]
                        break

            if not export_url:
                return []

            # Skip if already processed
            if export_url == self.last_file_processed:
                return []

            self.last_file_processed = export_url

            # Download and parse
            events = await self._download_and_parse(export_url, limit)
            return events

        except Exception as e:
            print(f"{Colors.DIM}GDELT fetch error: {e}{Colors.ENDC}")
            return []

    async def _download_and_parse(self, url: str, limit: int) -> List[Dict[str, Any]]:
        """Download and parse GDELT export file."""
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                zip_data = response.read()

            events = []
            with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
                for filename in zf.namelist():
                    if filename.endswith('.export.CSV') or filename.endswith('.csv'):
                        with zf.open(filename) as f:
                            content = f.read().decode('utf-8', errors='ignore')
                            for line in content.split('\n')[:limit * 2]:
                                event = self._parse_gdelt_line(line)
                                if event and event.get('GLOBALEVENTID') not in self.events_seen:
                                    self.events_seen.add(event['GLOBALEVENTID'])
                                    events.append(event)
                                    if len(events) >= limit:
                                        break

            return events

        except Exception as e:
            print(f"{Colors.DIM}Parse error: {e}{Colors.ENDC}")
            return []

    def _parse_gdelt_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single GDELT record."""
        try:
            fields = line.strip().split('\t')
            if len(fields) < 58:
                return None

            return {
                'GLOBALEVENTID': fields[0],
                'Day': fields[1],
                'Actor1Name': fields[6] if len(fields) > 6 else '',
                'Actor1CountryCode': fields[7] if len(fields) > 7 else '',
                'Actor1Type1Code': fields[12] if len(fields) > 12 else '',
                'Actor2Name': fields[16] if len(fields) > 16 else '',
                'Actor2CountryCode': fields[17] if len(fields) > 17 else '',
                'Actor2Type1Code': fields[22] if len(fields) > 22 else '',
                'EventCode': fields[26] if len(fields) > 26 else '',
                'EventRootCode': fields[28] if len(fields) > 28 else '',
                'GoldsteinScale': float(fields[30]) if len(fields) > 30 and fields[30] else 0.0,
                'NumMentions': int(fields[31]) if len(fields) > 31 and fields[31] else 1,
                'NumSources': int(fields[32]) if len(fields) > 32 and fields[32] else 1,
                'NumArticles': int(fields[33]) if len(fields) > 33 and fields[33] else 1,
                'AvgTone': float(fields[34]) if len(fields) > 34 and fields[34] else 0.0,
                'SOURCEURL': fields[57] if len(fields) > 57 else '',
            }
        except:
            return None

    def filter_relevant_events(self, events: List[Dict], watch_list: Optional[List[str]] = None) -> List[Dict]:
        """Filter events for relevance to watched companies."""
        if watch_list is None:
            watch_list = list(WATCHED_COMPANIES.keys())

        relevant = []
        for event in events:
            actor1 = (event.get('Actor1Name') or '').lower()
            actor2 = (event.get('Actor2Name') or '').lower()
            url = (event.get('SOURCEURL') or '').lower()

            for company_key in watch_list:
                if (company_key in actor1 or
                    company_key in actor2 or
                    company_key in url):
                    event['matched_company'] = WATCHED_COMPANIES.get(company_key, company_key)
                    relevant.append(event)
                    break

        return relevant


async def run_live_quorum(cycles: int = 10, watch: Optional[str] = None):
    """Run the quorum on live GDELT events."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}")
    print(" LIVE GDELT QUORUM SYSTEM")
    print(f"{'='*70}{Colors.ENDC}\n")

    # Set up watch list
    watch_list = None
    if watch:
        watch_list = [w.strip().lower() for w in watch.split(',')]
        print(f"Watching: {', '.join(watch_list)}")
    else:
        print(f"Watching {len(WATCHED_COMPANIES)} companies")

    print(f"GDELT updates every 15 minutes")
    print(f"Will run {cycles} update cycles\n")

    # Initialize systems
    gdelt = GDELTLiveFeed()
    system = RealTimeQuorumSystem()
    registry = get_industrial_registry()

    # Alert handler
    def handle_alert(deliberation: QuorumDeliberation):
        print(f"\n{Colors.RED}{'!'*60}")
        print(f"{Colors.BOLD}ALERT: {deliberation.event.title[:60]}{Colors.ENDC}")
        print(f"{Colors.RED}Company: {deliberation.event.primary_company}")
        if deliberation.consensus_stance:
            print(f"Consensus: {format_stance(deliberation.consensus_stance)}")
        print(f"{'!'*60}{Colors.ENDC}\n")

    system.set_alert_handler(handle_alert)
    await system.start(simulate=False)

    # Stats
    total_events = 0
    relevant_events = 0
    deliberations = 0

    print(f"{Colors.CYAN}{'─'*50}{Colors.ENDC}")
    print(f" Fetching live GDELT events...")
    print(f"{Colors.CYAN}{'─'*50}{Colors.ENDC}\n")

    for cycle in range(cycles):
        print(f"\n{Colors.BOLD}[Cycle {cycle + 1}/{cycles}] {datetime.now().strftime('%H:%M:%S')}{Colors.ENDC}")

        # Fetch latest events
        events = await gdelt.fetch_latest_events(limit=500)
        total_events += len(events)

        if not events:
            print(f"  {Colors.DIM}No new events yet...{Colors.ENDC}")
        else:
            print(f"  Fetched {len(events)} new GDELT events")

            # Filter for relevant companies
            relevant = gdelt.filter_relevant_events(events, watch_list)
            relevant_events += len(relevant)

            if relevant:
                print(f"  {Colors.GREEN}Found {len(relevant)} relevant events!{Colors.ENDC}\n")

                for event in relevant[:5]:  # Process top 5
                    # Convert to RealTimeEvent
                    companies = [event.get('matched_company', 'Unknown')]
                    if event.get('Actor2Name'):
                        companies.append(event['Actor2Name'])

                    # Classify event type
                    url_lower = event.get('SOURCEURL', '').lower()
                    event_type = None
                    for keyword, etype in EVENT_KEYWORDS.items():
                        if keyword in url_lower:
                            event_type = etype
                            break

                    headline = f"{event.get('matched_company', 'Company')}: {event.get('Actor1Name', '')} - {event.get('Actor2Name', '')}"

                    rt_event = RealTimeEvent(
                        event_id=event['GLOBALEVENTID'],
                        source="gdelt_live",
                        timestamp=datetime.utcnow(),
                        headline=headline[:100],
                        companies_mentioned=companies,
                        event_type=event_type,
                        source_url=event.get('SOURCEURL', ''),
                        sentiment=event.get('AvgTone', 0) / 10,
                        importance=min(1.0, event.get('NumMentions', 1) / 50),
                    )

                    # Run through quorum
                    _, delib = await system.inject_event(
                        headline=rt_event.headline,
                        companies=rt_event.companies_mentioned,
                        event_type=rt_event.event_type,
                        importance=rt_event.importance,
                        sentiment=rt_event.sentiment,
                    )

                    if delib:
                        deliberations += 1

                        print(f"  {Colors.BOLD}EVENT:{Colors.ENDC} {companies[0]}")
                        print(f"    Source: {event.get('SOURCEURL', 'N/A')[:60]}...")
                        print(f"    Goldstein: {event.get('GoldsteinScale', 0):.1f} | Tone: {event.get('AvgTone', 0):.1f}")

                        if delib.consensus_stance:
                            print(f"    {Colors.BOLD}Quorum:{Colors.ENDC} {format_stance(delib.consensus_stance)} "
                                  f"({delib.consensus_strength:.0%} agreement)")

                        if delib.key_insights:
                            print(f"    Insight: {delib.key_insights[0]}")

                        print()
            else:
                print(f"  {Colors.DIM}No events matching watched companies{Colors.ENDC}")

        # Wait for next GDELT update (15 minutes, but we check more frequently)
        if cycle < cycles - 1:
            wait_time = 60  # Check every minute
            print(f"\n  {Colors.DIM}Waiting {wait_time}s for next check...{Colors.ENDC}")
            await asyncio.sleep(wait_time)

    # Summary
    await system.stop()

    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}")
    print(" SESSION COMPLETE")
    print(f"{'='*70}{Colors.ENDC}\n")

    print(f"Total GDELT events processed: {total_events}")
    print(f"Relevant events found: {relevant_events}")
    print(f"Quorum deliberations: {deliberations}")

    # Show predictions
    predictions = await system.get_predictions(min_probability=0.3)
    if predictions:
        print(f"\n{Colors.CYAN}{'─'*50}")
        print(" PREDICTIONS GENERATED")
        print(f"{'─'*50}{Colors.ENDC}\n")

        for pred in predictions[:5]:
            print(f"{Colors.BOLD}{pred.prediction_type.value.upper()}: {pred.target_company}{Colors.ENDC}")
            print(f"  Probability: {pred.probability:.0%}")
            if pred.likely_acquirers:
                print(f"  Likely acquirers: {', '.join(pred.likely_acquirers[:3])}")
            print()

    print(f"{Colors.GREEN}Done.{Colors.ENDC}\n")


async def quick_test():
    """Quick test with simulated events."""
    print(f"\n{Colors.HEADER}Quick Test Mode{Colors.ENDC}\n")

    system = RealTimeQuorumSystem()

    def alert_handler(d):
        print(f"{Colors.RED}ALERT: {d.event.title}{Colors.ENDC}")

    system.set_alert_handler(alert_handler)
    await system.start(simulate=False)

    # Inject test events
    test_events = [
        ("Nvidia announces $20B acquisition of Cerebras", ["Nvidia", "Cerebras"], IndustrialEventType.ACQUISITION_ANNOUNCED),
        ("OpenAI raises $10B at $300B valuation", ["OpenAI"], IndustrialEventType.FUNDING_ROUND),
        ("Figure AI partners with BMW for humanoid deployment", ["Figure AI", "BMW"], IndustrialEventType.PARTNERSHIP_ANNOUNCED),
        ("Anthropic CEO discusses AGI timeline concerns", ["Anthropic"], None),
        ("Microsoft considering Mistral acquisition", ["Microsoft", "Mistral"], IndustrialEventType.ACQUISITION_ANNOUNCED),
    ]

    for headline, companies, event_type in test_events:
        print(f"\n{Colors.BOLD}Injecting:{Colors.ENDC} {headline}")
        event, delib = await system.inject_event(
            headline=headline,
            companies=companies,
            event_type=event_type,
            importance=0.8,
        )

        if delib:
            print(f"  Consensus: {format_stance(delib.consensus_stance) if delib.consensus_stance else 'mixed'}")
            print(f"  Agreement: {delib.consensus_strength:.0%}")
            print(f"  Urgency: {delib.urgency_level}")
            if delib.recommended_actions:
                print(f"  Action: {delib.recommended_actions[0]}")

        await asyncio.sleep(0.5)

    await system.stop()
    print(f"\n{Colors.GREEN}Test complete.{Colors.ENDC}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run live GDELT quorum")
    parser.add_argument("--cycles", type=int, default=5, help="Number of update cycles")
    parser.add_argument("--watch", type=str, help="Comma-separated companies to watch")
    parser.add_argument("--test", action="store_true", help="Run quick test mode")

    args = parser.parse_args()

    try:
        if args.test:
            asyncio.run(quick_test())
        else:
            asyncio.run(run_live_quorum(cycles=args.cycles, watch=args.watch))
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted.{Colors.ENDC}")
