#!/usr/bin/env python3
"""
Real-Time Quorum Demo Runner

Demonstrates the emotional quorum system processing events in real-time,
with agents deliberating and generating predictions.

Usage:
    python run_quorum_demo.py [--duration SECONDS] [--interval SECONDS]
"""

import asyncio
import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from meta.realtime_quorum import (
    RealTimeQuorumSystem,
    PredictionType,
    QuorumDeliberation,
)
from meta.industrial_intelligence import (
    IndustrialEventType,
    EmotionalStance,
)


# ANSI colors for terminal output
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


def print_header(text: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}")
    print(f" {text}")
    print(f"{'='*70}{Colors.ENDC}\n")


def print_section(text: str):
    print(f"\n{Colors.CYAN}{'-'*50}")
    print(f" {text}")
    print(f"{'-'*50}{Colors.ENDC}")


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


def format_urgency(urgency: str) -> str:
    """Format urgency with color."""
    colors = {
        "critical": Colors.RED + Colors.BOLD,
        "high": Colors.RED,
        "elevated": Colors.YELLOW,
        "normal": Colors.GREEN,
        "low": Colors.DIM,
    }
    color = colors.get(urgency, "")
    return f"{color}{urgency.upper()}{Colors.ENDC}"


async def main(duration: int = 60, interval: float = 5.0):
    """Run the demo."""
    print_header("REAL-TIME QUORUM SYSTEM")
    print(f"Duration: {duration}s | Event interval: {interval}s")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    system = RealTimeQuorumSystem()

    # Alert handler
    def handle_alert(deliberation: QuorumDeliberation):
        print(f"\n{Colors.RED}{'!'*60}")
        print(f"{Colors.BOLD}ALERT: {deliberation.event.title}{Colors.ENDC}")
        print(f"{Colors.RED}Urgency: {deliberation.urgency_level.upper()}")
        if deliberation.consensus_stance:
            print(f"Consensus: {format_stance(deliberation.consensus_stance)}")
        print(f"Actions: {', '.join(deliberation.recommended_actions[:2])}")
        print(f"{'!'*60}{Colors.ENDC}\n")

    system.set_alert_handler(handle_alert)

    # Start system
    await system.start(simulate=True, interval=interval)

    print_section("EVENT STREAM ACTIVE")
    print("Agents are deliberating on incoming events...\n")

    # Track for summary
    all_deliberations = []

    # Run for duration
    elapsed = 0
    tick = 0
    while elapsed < duration:
        await asyncio.sleep(min(interval, duration - elapsed))
        elapsed += interval
        tick += 1

        # Get status
        status = await system.get_status()
        deliberations = await system.get_recent_deliberations(limit=1)

        print(f"\n{Colors.DIM}[{datetime.now().strftime('%H:%M:%S')}] "
              f"Events: {status['stats']['events_processed']} | "
              f"Deliberations: {status['stats']['deliberations_completed']} | "
              f"Predictions: {status['active_predictions']}{Colors.ENDC}")

        if deliberations:
            d = deliberations[0]
            all_deliberations.append(d)

            # Event info
            print(f"\n{Colors.BOLD}EVENT:{Colors.ENDC} {d.event.title}")
            print(f"  Type: {d.event.event_type.value if d.event.event_type else 'unknown'}")
            print(f"  Company: {d.event.primary_company}")

            # Quorum result
            if d.consensus_stance:
                print(f"\n{Colors.BOLD}QUORUM CONSENSUS:{Colors.ENDC} {format_stance(d.consensus_stance)}")
                print(f"  Agreement: {d.consensus_strength:.0%}")
                print(f"  Dissent: {d.dissent_level:.0%}")
                print(f"  Urgency: {format_urgency(d.urgency_level)}")

            # Agent opinions summary
            if d.opinions:
                print(f"\n{Colors.BOLD}AGENT OPINIONS:{Colors.ENDC}")
                for op in d.opinions[:4]:
                    stance_str = format_stance(op.stance)
                    conf_bar = "█" * int(op.confidence * 10) + "░" * (10 - int(op.confidence * 10))
                    print(f"  {op.agent_role:18} {stance_str:20} [{conf_bar}] {op.confidence:.0%}")

            # Insights
            if d.key_insights:
                print(f"\n{Colors.BOLD}KEY INSIGHTS:{Colors.ENDC}")
                for insight in d.key_insights[:2]:
                    print(f"  • {insight}")

            # Risk/Opportunity
            print(f"\n{Colors.BOLD}ASSESSMENT:{Colors.ENDC}")
            print(f"  Risk: {d.risk_assessment}")
            print(f"  Opportunity: {d.opportunity_assessment}")

            # Actions
            if d.recommended_actions:
                print(f"\n{Colors.BOLD}RECOMMENDED ACTIONS:{Colors.ENDC}")
                for action in d.recommended_actions[:3]:
                    print(f"  → {action}")

    # Final summary
    await system.stop()

    print_header("SESSION SUMMARY")

    status = await system.get_status()
    print(f"Total events processed: {status['stats']['events_processed']}")
    print(f"Deliberations completed: {status['stats']['deliberations_completed']}")
    print(f"Predictions generated: {status['stats']['predictions_generated']}")
    print(f"Alerts triggered: {status['stats']['alerts_triggered']}")

    # Stance distribution
    if all_deliberations:
        print_section("STANCE DISTRIBUTION")
        stance_counts = {}
        for d in all_deliberations:
            if d.consensus_stance:
                stance_counts[d.consensus_stance] = stance_counts.get(d.consensus_stance, 0) + 1

        for stance, count in sorted(stance_counts.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * count
            print(f"  {format_stance(stance):30} {bar} ({count})")

    # Top predictions
    predictions = await system.get_predictions(min_probability=0.3)
    if predictions:
        print_section("TOP PREDICTIONS")
        for pred in predictions[:5]:
            print(f"\n{Colors.BOLD}{pred.prediction_type.value.upper()}: {pred.target_company}{Colors.ENDC}")
            print(f"  Probability: {pred.probability:.0%}")
            print(f"  Confidence: {pred.confidence.value}")
            if pred.likely_acquirers:
                print(f"  Likely acquirers: {', '.join(pred.likely_acquirers[:3])}")
            if pred.estimated_price_billions:
                print(f"  Est. price: ${pred.estimated_price_billions:.1f}B")

    print(f"\n{Colors.GREEN}Demo complete.{Colors.ENDC}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run real-time quorum demo")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    parser.add_argument("--interval", type=float, default=5.0, help="Event interval in seconds")

    args = parser.parse_args()

    try:
        asyncio.run(main(duration=args.duration, interval=args.interval))
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted.{Colors.ENDC}")
