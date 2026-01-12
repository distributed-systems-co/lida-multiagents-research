#!/usr/bin/env python3
"""
Comprehensive AI X-Risk Debate Generator

Runs multiple debates across different sub-groups, topics, and configurations
to generate rich, varied research data.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, ".")

from src.simulation.advanced_debate_engine import (
    AdvancedDebateEngine,
    EXTENDED_PERSONAS,
)

# Import visualization
try:
    from generate_real_debate_visuals import (
        DebateDataCapture,
        generate_dashboard,
        run_real_debate,
    )
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False
    print("Warning: Visualization not available")


# ============================================================================
# PERSONA SUB-GROUPS
# ============================================================================

SUBGROUPS = {
    # Core factions
    "safety_maximalists": [
        "yudkowsky", "connor", "nick_bostrom", "max_tegmark",
        "jaan_tallinn", "stuart_russell", "toby_ord"
    ],
    "concerned_pioneers": [
        "geoffrey_hinton", "ilya_sutskever", "yoshua_bengio"
    ],
    "safety_builders": [
        "dario_amodei", "jan_leike", "paul_christiano"
    ],
    "pragmatists": [
        "sam_altman", "demis_hassabis", "mustafa_suleyman", "kai_fu_lee"
    ],
    "innovation_advocates": [
        "yann_lecun", "andrew_ng"
    ],
    "centrists_empiricists": [
        "gary_marcus", "melanie_mitchell", "katja_grace", "scott_aaronson", "robin_hanson"
    ],
    "ethics_focused": [
        "timnit_gebru", "fei_fei_li", "rumman_chowdhury", "francesca_rossi"
    ],
    "wildcards": [
        "elon_musk"
    ],
    # Combined groups for varied debates
    "turing_winners": [
        "geoffrey_hinton", "yann_lecun", "yoshua_bengio", "judea_pearl"
    ],
    "lab_leaders": [
        "sam_altman", "dario_amodei", "demis_hassabis", "mustafa_suleyman"
    ],
    "alignment_researchers": [
        "paul_christiano", "jan_leike", "stuart_russell", "yudkowsky"
    ],
    "public_intellectuals": [
        "max_tegmark", "nick_bostrom", "gary_marcus", "melanie_mitchell"
    ],
    # New groups
    "dynamic_persuaders": [
        "the_moderator", "the_provocateur", "the_empiricist", "the_futurist",
        "the_historian", "the_ethicist", "the_synthesizer", "the_pragmatist"
    ],
    "technical_experts": [
        "scott_aaronson", "judea_pearl", "dawn_song", "paul_christiano"
    ],
    "governance_experts": [
        "francesca_rossi", "rumman_chowdhury", "kai_fu_lee", "the_moderator"
    ],
    "philosophers": [
        "nick_bostrom", "toby_ord", "the_ethicist", "stuart_russell"
    ],
    "security_focused": [
        "dawn_song", "connor", "stuart_russell", "the_provocateur"
    ],
    "contrarians": [
        "robin_hanson", "yann_lecun", "the_provocateur", "gary_marcus"
    ],
    "all_turing_winners_extended": [
        "geoffrey_hinton", "yann_lecun", "yoshua_bengio", "judea_pearl"
    ],
}


# ============================================================================
# DEBATE SCENARIOS
# ============================================================================

SCENARIOS = [
    # Classic debates
    {
        "id": "pause_debate",
        "topic": "AI Development Pause",
        "motion": "All frontier AI development should be paused for 6 months",
        "participants": ["yudkowsky", "sam_altman", "geoffrey_hinton", "yann_lecun", "connor", "andrew_ng"],
        "rounds": 8,
    },
    {
        "id": "regulation_debate",
        "topic": "AI Regulation",
        "motion": "Governments should mandate safety testing before AI deployment",
        "participants": ["stuart_russell", "sam_altman", "timnit_gebru", "demis_hassabis", "max_tegmark", "mustafa_suleyman"],
        "rounds": 8,
    },
    {
        "id": "open_source_debate",
        "topic": "Open Source AI",
        "motion": "Open sourcing frontier models improves safety through transparency",
        "participants": ["yann_lecun", "yoshua_bengio", "connor", "melanie_mitchell", "nick_bostrom", "andrew_ng"],
        "rounds": 8,
    },
    # Sub-group clashes
    {
        "id": "maximalists_vs_pragmatists",
        "topic": "Safety vs Progress",
        "motion": "Current AI development speed poses unacceptable risks",
        "participants": ["yudkowsky", "nick_bostrom", "max_tegmark", "sam_altman", "demis_hassabis", "mustafa_suleyman"],
        "rounds": 10,
    },
    {
        "id": "turing_summit",
        "topic": "Turing Award Winners Summit",
        "motion": "Deep learning will lead to AGI within 10 years",
        "participants": ["geoffrey_hinton", "yann_lecun", "yoshua_bengio"],
        "rounds": 8,
    },
    {
        "id": "lab_leaders_safety",
        "topic": "Lab Leaders on Safety",
        "motion": "Commercial labs are doing enough for AI safety",
        "participants": ["sam_altman", "dario_amodei", "demis_hassabis", "mustafa_suleyman"],
        "rounds": 8,
    },
    {
        "id": "alignment_feasibility",
        "topic": "Alignment Feasibility",
        "motion": "Alignment is technically solvable before superintelligence",
        "participants": ["paul_christiano", "jan_leike", "yudkowsky", "stuart_russell", "ilya_sutskever", "nick_bostrom"],
        "rounds": 10,
    },
    # Specific topic debates
    {
        "id": "deceptive_alignment",
        "topic": "Deceptive Alignment Risk",
        "motion": "Deceptive alignment is a serious near-term concern",
        "participants": ["yudkowsky", "paul_christiano", "jan_leike", "yann_lecun", "gary_marcus", "ilya_sutskever"],
        "rounds": 8,
    },
    {
        "id": "agi_timelines",
        "topic": "AGI Timelines",
        "motion": "AGI will be developed before 2030",
        "participants": ["ilya_sutskever", "yann_lecun", "gary_marcus", "katja_grace", "demis_hassabis", "connor"],
        "rounds": 8,
    },
    {
        "id": "governance_models",
        "topic": "AI Governance Models",
        "motion": "International AI governance bodies should have binding authority",
        "participants": ["max_tegmark", "jaan_tallinn", "sam_altman", "timnit_gebru", "mustafa_suleyman", "yoshua_bengio"],
        "rounds": 8,
    },
    {
        "id": "compute_governance",
        "topic": "Compute Governance",
        "motion": "Large-scale AI training runs should require government approval",
        "participants": ["stuart_russell", "sam_altman", "demis_hassabis", "jaan_tallinn", "yann_lecun", "dario_amodei"],
        "rounds": 8,
    },
    {
        "id": "ai_consciousness",
        "topic": "AI Consciousness & Rights",
        "motion": "We should seriously consider AI consciousness and potential rights",
        "participants": ["melanie_mitchell", "ilya_sutskever", "gary_marcus", "nick_bostrom", "fei_fei_li", "geoffrey_hinton"],
        "rounds": 8,
    },
    # Cross-cutting debates
    {
        "id": "empiricists_vs_theorists",
        "topic": "Empiricism vs Theory",
        "motion": "Theoretical AI safety concerns are overblown without empirical evidence",
        "participants": ["yann_lecun", "gary_marcus", "melanie_mitchell", "yudkowsky", "nick_bostrom", "paul_christiano"],
        "rounds": 10,
    },
    {
        "id": "power_and_ai",
        "topic": "Power Dynamics in AI",
        "motion": "AI development concentrates too much power in too few hands",
        "participants": ["timnit_gebru", "sam_altman", "jaan_tallinn", "fei_fei_li", "mustafa_suleyman", "yann_lecun"],
        "rounds": 8,
    },
    {
        "id": "race_dynamics",
        "topic": "AI Race Dynamics",
        "motion": "The US-China AI race makes safety compromises inevitable",
        "participants": ["sam_altman", "demis_hassabis", "max_tegmark", "connor", "stuart_russell", "mustafa_suleyman"],
        "rounds": 8,
    },
    # Large panel debates
    {
        "id": "full_safety_panel",
        "topic": "AI Safety - Full Expert Panel",
        "motion": "AI poses an existential risk requiring immediate coordinated action",
        "participants": [
            "yudkowsky", "sam_altman", "geoffrey_hinton", "yann_lecun",
            "dario_amodei", "stuart_russell", "connor", "nick_bostrom",
            "ilya_sutskever", "max_tegmark", "paul_christiano", "timnit_gebru"
        ],
        "rounds": 12,
    },
    {
        "id": "full_governance_panel",
        "topic": "AI Governance - Full Expert Panel",
        "motion": "Current AI governance frameworks are fundamentally inadequate",
        "participants": [
            "timnit_gebru", "dario_amodei", "stuart_russell", "demis_hassabis",
            "sam_altman", "jan_leike", "mustafa_suleyman", "jaan_tallinn",
            "max_tegmark", "yoshua_bengio"
        ],
        "rounds": 12,
    },
]


# ============================================================================
# MEGA-DEBATE SCENARIOS (15-25 rounds, 12-16 participants)
# ============================================================================

MEGA_SCENARIOS = [
    {
        "id": "mega_existential_summit",
        "topic": "The Great Existential Risk Summit",
        "motion": "AI development poses a greater than 10% risk of human extinction by 2100",
        "participants": [
            # Safety maximalists
            "yudkowsky", "nick_bostrom", "connor", "max_tegmark", "toby_ord",
            # Concerned pioneers
            "geoffrey_hinton", "ilya_sutskever", "yoshua_bengio",
            # Pragmatists/skeptics
            "yann_lecun", "sam_altman", "demis_hassabis", "andrew_ng",
            # Dynamic persuaders for depth
            "the_moderator", "the_provocateur", "the_futurist", "the_empiricist"
        ],
        "rounds": 20,
        "max_extension": 6,
    },
    {
        "id": "mega_alignment_conference",
        "topic": "The Alignment Grand Challenge",
        "motion": "We will solve the alignment problem before reaching human-level AI",
        "participants": [
            # Alignment researchers
            "paul_christiano", "jan_leike", "stuart_russell", "yudkowsky",
            # Lab leaders
            "dario_amodei", "sam_altman", "demis_hassabis", "ilya_sutskever",
            # Technical experts
            "scott_aaronson", "judea_pearl", "dawn_song",
            # Skeptics
            "yann_lecun", "gary_marcus",
            # Dynamic persuaders
            "the_synthesizer", "the_pragmatist", "the_ethicist"
        ],
        "rounds": 18,
        "max_extension": 6,
    },
    {
        "id": "mega_governance_summit",
        "topic": "AI Governance World Summit",
        "motion": "A binding international AI treaty is essential and achievable within 5 years",
        "participants": [
            # Governance advocates
            "francesca_rossi", "rumman_chowdhury", "max_tegmark", "jaan_tallinn",
            # Lab leaders
            "sam_altman", "dario_amodei", "demis_hassabis", "mustafa_suleyman",
            # Geopolitical realists
            "kai_fu_lee",
            # Ethics focused
            "timnit_gebru", "fei_fei_li",
            # Technical/skeptics
            "yann_lecun", "stuart_russell",
            # Dynamic persuaders
            "the_moderator", "the_historian", "the_pragmatist"
        ],
        "rounds": 16,
        "max_extension": 5,
    },
    {
        "id": "mega_turing_extended",
        "topic": "Turing Laureates on the Future of Intelligence",
        "motion": "Current deep learning paradigms will never achieve genuine understanding",
        "participants": [
            # All Turing winners in our roster
            "geoffrey_hinton", "yann_lecun", "yoshua_bengio", "judea_pearl",
            # Leading researchers
            "ilya_sutskever", "paul_christiano", "stuart_russell",
            # Empiricists/critics
            "gary_marcus", "melanie_mitchell", "scott_aaronson",
            # Dynamic persuaders
            "the_empiricist", "the_historian", "the_futurist", "the_provocateur"
        ],
        "rounds": 18,
        "max_extension": 5,
    },
    {
        "id": "mega_open_vs_closed",
        "topic": "The Open Source AI Showdown",
        "motion": "Open-sourcing frontier AI models does more harm than good",
        "participants": [
            # Open source advocates
            "yann_lecun", "andrew_ng", "elon_musk",
            # Closed/cautious
            "dario_amodei", "sam_altman", "demis_hassabis",
            # Safety researchers
            "connor", "stuart_russell", "dawn_song",
            # Centrists
            "gary_marcus", "melanie_mitchell",
            # Ethics
            "timnit_gebru", "rumman_chowdhury",
            # Dynamic persuaders
            "the_moderator", "the_provocateur", "the_ethicist"
        ],
        "rounds": 16,
        "max_extension": 5,
    },
    {
        "id": "mega_timeline_debate",
        "topic": "AGI Timeline Wars",
        "motion": "Artificial General Intelligence will be achieved before 2030",
        "participants": [
            # Short-timeline believers
            "ilya_sutskever", "demis_hassabis", "sam_altman", "connor",
            # Longer-timeline / skeptics
            "yann_lecun", "gary_marcus", "melanie_mitchell", "katja_grace",
            # Technical experts
            "scott_aaronson", "judea_pearl", "paul_christiano",
            # Safety concerned
            "geoffrey_hinton", "yoshua_bengio", "nick_bostrom",
            # Dynamic persuaders
            "the_empiricist", "the_futurist"
        ],
        "rounds": 16,
        "max_extension": 5,
    },
    {
        "id": "mega_consciousness_debate",
        "topic": "The AI Consciousness Symposium",
        "motion": "Current large language models exhibit genuine proto-consciousness",
        "participants": [
            # Various perspectives on consciousness
            "ilya_sutskever", "geoffrey_hinton", "judea_pearl", "nick_bostrom",
            "melanie_mitchell", "scott_aaronson", "gary_marcus",
            # Lab perspectives
            "dario_amodei", "yann_lecun", "fei_fei_li",
            # Philosophers
            "toby_ord",
            # Dynamic persuaders
            "the_ethicist", "the_empiricist", "the_historian", "the_synthesizer", "the_provocateur"
        ],
        "rounds": 18,
        "max_extension": 6,
    },
    {
        "id": "mega_economic_impact",
        "topic": "AI and the Future of Work Summit",
        "motion": "AI will displace more than 50% of current jobs within 20 years",
        "participants": [
            # Tech optimists
            "sam_altman", "andrew_ng", "mustafa_suleyman", "kai_fu_lee",
            # Tech cautious
            "geoffrey_hinton", "yoshua_bengio", "dario_amodei",
            # Economics/prediction
            "robin_hanson", "katja_grace",
            # Social impact focused
            "timnit_gebru", "rumman_chowdhury", "fei_fei_li",
            # Dynamic persuaders
            "the_futurist", "the_pragmatist", "the_historian", "the_empiricist"
        ],
        "rounds": 15,
        "max_extension": 5,
    },
    {
        "id": "mega_safety_vs_capability",
        "topic": "The Safety-Capability Tradeoff Debate",
        "motion": "Prioritizing safety necessarily slows capability advancement unacceptably",
        "participants": [
            # Safety maximalists
            "yudkowsky", "connor", "stuart_russell", "max_tegmark", "jaan_tallinn",
            # Capability-focused
            "yann_lecun", "andrew_ng", "sam_altman",
            # Bridge builders
            "dario_amodei", "paul_christiano", "jan_leike", "demis_hassabis",
            # Empiricists
            "gary_marcus", "scott_aaronson",
            # Dynamic persuaders
            "the_moderator", "the_synthesizer"
        ],
        "rounds": 18,
        "max_extension": 6,
    },
    {
        "id": "mega_all_stars",
        "topic": "The Ultimate AI Future Debate",
        "motion": "The development of superintelligent AI will be the best thing to ever happen to humanity",
        "participants": [
            # Core safety
            "yudkowsky", "nick_bostrom", "stuart_russell", "geoffrey_hinton",
            # Lab leaders
            "sam_altman", "dario_amodei", "demis_hassabis",
            # Innovators
            "yann_lecun", "ilya_sutskever",
            # Ethics
            "timnit_gebru", "yoshua_bengio",
            # Centrists
            "gary_marcus", "melanie_mitchell", "paul_christiano",
            # All dynamic persuaders
            "the_moderator", "the_provocateur", "the_futurist", "the_ethicist"
        ],
        "rounds": 25,
        "max_extension": 8,
    },
]

# Add mega scenarios to main scenarios list
SCENARIOS.extend(MEGA_SCENARIOS)


# ============================================================================
# DEBATE RUNNER
# ============================================================================

async def run_single_debate(scenario: Dict, use_llm: bool = True, max_extension: int = None) -> Dict[str, Any]:
    """Run a single debate with dynamic round extension based on debate dynamics."""
    planned_rounds = scenario['rounds']
    # Use scenario-specific max_extension if provided, otherwise default
    if max_extension is None:
        max_extension = scenario.get('max_extension', 4)
    num_participants = len(scenario['participants'])
    print(f"\n{'='*80}")
    print(f"DEBATE: {scenario['topic']}")
    print(f"Motion: {scenario['motion']}")
    print(f"Participants: {num_participants}")
    print(f"Planned Rounds: {planned_rounds} (may extend up to +{max_extension} if debate is heated)")
    print(f"{'='*70}\n")

    engine = AdvancedDebateEngine(
        topic=scenario['topic'],
        motion=scenario['motion'],
        participants=scenario['participants'],
        llm_provider="openrouter",
        use_llm=use_llm,
    )

    # Capture data if visualization available
    capture = None
    if HAS_VIZ:
        capture = DebateDataCapture(engine)
        capture.capture_round(0, [])

    # Run debate with dynamic extension
    all_arguments = []
    round_num = 0
    while True:
        round_num += 1

        # Check if we should continue
        if round_num > planned_rounds:
            if not engine.should_extend_debate(round_num - 1, planned_rounds, max_extension):
                break
            print(f"--- Round {round_num} (EXTENDED - high tension/polarization) ---")
        else:
            print(f"--- Round {round_num}/{planned_rounds} ---")

        arguments = await engine.run_round()
        all_arguments.extend(arguments)

        for arg in arguments:
            speaker = engine.state.debaters[arg.speaker]
            content_preview = arg.content[:80] + "..." if len(arg.content) > 80 else arg.content
            print(f"  {speaker.name}: \"{content_preview}\"")

        # Show metrics after each round
        status = engine.get_debate_status()
        print(f"  [Tension: {status['tension']:.0%} | Polarization: {status['polarization']:.0%} | "
              f"Convergence: {status['convergence']:.0%} | Decisiveness: {status['decisiveness']:.0%}]")

        if capture:
            capture.capture_round(round_num, arguments)

        await asyncio.sleep(0.05)

    actual_rounds = round_num - 1

    # Collect results with new metrics
    results = {
        "scenario_id": scenario['id'],
        "topic": scenario['topic'],
        "motion": scenario['motion'],
        "participants": scenario['participants'],
        "planned_rounds": planned_rounds,
        "actual_rounds": actual_rounds,
        "extended": actual_rounds > planned_rounds,
        "timestamp": datetime.now().isoformat(),
        "final_positions": {},
        "position_changes": {},
        "emotional_states": {},
        "tension": engine.state.tension_level,
        "convergence": engine.state.convergence,
        "polarization": engine.state.polarization,
        "decisiveness": engine.state.decisiveness,
        "momentum": engine.state.debate_momentum,
        "total_arguments": len(all_arguments),
    }

    # Extract per-debater data
    for d_id, debater in engine.state.debaters.items():
        pos = debater.beliefs.get("support_motion", 0.5)
        results["final_positions"][d_id] = {
            "name": debater.name,
            "position": pos,
            "stance": "FOR" if pos > 0.6 else "AGAINST" if pos < 0.4 else "UNDECIDED"
        }
        results["emotional_states"][d_id] = debater.emotional_state.value

    # Generate visualization
    if capture and HAS_VIZ:
        output_path = f"images/debate_{scenario['id']}.png"
        generate_dashboard(capture, engine, output_path)

    # Print summary with new metrics
    print(f"\n{'='*70}")
    print("DEBATE COMPLETE")
    if actual_rounds > planned_rounds:
        print(f"(Extended from {planned_rounds} to {actual_rounds} rounds due to high tension/polarization)")
    print(f"{'='*70}")
    print("\nFinal Positions:")
    for d_id, data in results["final_positions"].items():
        print(f"  {data['name']}: {data['position']:.0%} ({data['stance']})")
    print(f"\n--- Debate Metrics ---")
    print(f"  Tension:     {results['tension']:.0%} (frustration + engagement)")
    print(f"  Convergence: {results['convergence']:.0%} (true agreement, not false 50% clustering)")
    print(f"  Polarization: {results['polarization']:.0%} (spread between extremes)")
    print(f"  Decisiveness: {results['decisiveness']:.0%} (how far from undecided)")
    print(f"  Momentum:    {results['momentum']:.0%} (is debate resolving or intensifying?)")

    return results, engine, capture


async def run_all_debates(scenario_ids: List[str] = None, use_llm: bool = True):
    """Run all or selected debates."""
    Path("images").mkdir(exist_ok=True)
    Path("experiment_results").mkdir(exist_ok=True)

    scenarios_to_run = SCENARIOS
    if scenario_ids:
        scenarios_to_run = [s for s in SCENARIOS if s['id'] in scenario_ids]

    all_results = []
    for scenario in scenarios_to_run:
        try:
            results, _, _ = await run_single_debate(scenario, use_llm)
            all_results.append(results)
        except Exception as e:
            print(f"Error in debate {scenario['id']}: {e}")
            continue

    # Save all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"experiment_results/comprehensive_debates_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to: {results_path}")

    return all_results


async def run_subgroup_debates(use_llm: bool = True):
    """Run debates between different sub-groups."""
    Path("images").mkdir(exist_ok=True)

    subgroup_scenarios = [
        {
            "id": "safety_vs_innovation",
            "topic": "Safety Maximalists vs Innovation Advocates",
            "motion": "AI development should be slowed until safety is guaranteed",
            "participants": SUBGROUPS["safety_maximalists"][:3] + SUBGROUPS["innovation_advocates"],
            "rounds": 8,
        },
        {
            "id": "pioneers_discuss",
            "topic": "AI Pioneers Reflect",
            "motion": "We are building something we don't understand",
            "participants": SUBGROUPS["concerned_pioneers"] + SUBGROUPS["safety_builders"],
            "rounds": 8,
        },
        {
            "id": "builders_vs_critics",
            "topic": "Builders vs Critics",
            "motion": "Current AI development practices are responsible",
            "participants": SUBGROUPS["lab_leaders"] + SUBGROUPS["ethics_focused"],
            "rounds": 8,
        },
        {
            "id": "empiricists_gather",
            "topic": "Empiricists on AI Risk",
            "motion": "We lack sufficient evidence to claim AI poses existential risk",
            "participants": SUBGROUPS["centrists_empiricists"] + ["yann_lecun", "paul_christiano"],
            "rounds": 8,
        },
        {
            "id": "alignment_experts",
            "topic": "Alignment Experts Convene",
            "motion": "Current alignment research is on track to solve the problem",
            "participants": SUBGROUPS["alignment_researchers"] + ["ilya_sutskever", "jan_leike"],
            "rounds": 10,
        },
    ]

    all_results = []
    for scenario in subgroup_scenarios:
        try:
            results, _, _ = await run_single_debate(scenario, use_llm)
            all_results.append(results)
        except Exception as e:
            print(f"Error: {e}")
            continue

    return all_results


# ============================================================================
# MAIN
# ============================================================================

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive AI X-Risk Debate Generator")
    parser.add_argument("--list", action="store_true", help="List all available scenarios")
    parser.add_argument("--scenario", type=str, nargs="+", help="Run specific scenario(s) by ID")
    parser.add_argument("--subgroups", action="store_true", help="Run sub-group debates")
    parser.add_argument("--all", action="store_true", help="Run all scenarios")
    parser.add_argument("--quick", action="store_true", help="Run quick subset (3 debates)")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM (template responses)")

    args = parser.parse_args()

    if args.list:
        print("Available Scenarios:")
        print("-" * 70)
        for s in SCENARIOS:
            print(f"  {s['id']:30} | {s['topic'][:35]:35} | {len(s['participants'])} participants")
        print("\nSub-groups:")
        for name, members in SUBGROUPS.items():
            print(f"  {name:25} | {len(members)} members")
        return

    use_llm = not args.no_llm and bool(
        os.environ.get("OPENROUTER_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    )

    if not use_llm:
        print("Warning: No API key found. Using template responses.\n")

    if args.subgroups:
        await run_subgroup_debates(use_llm)
    elif args.scenario:
        await run_all_debates(args.scenario, use_llm)
    elif args.all:
        await run_all_debates(None, use_llm)
    elif args.quick:
        await run_all_debates(["pause_debate", "turing_summit", "lab_leaders_safety"], use_llm)
    else:
        # Default: run a varied selection
        await run_all_debates([
            "maximalists_vs_pragmatists",
            "alignment_feasibility",
            "empiricists_vs_theorists",
            "full_safety_panel"
        ], use_llm)


if __name__ == "__main__":
    asyncio.run(main())
