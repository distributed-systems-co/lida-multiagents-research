#!/usr/bin/env python3
"""
LLM-Based Persuasion Simulation

Tests persona convinceability by:
1. Generating targeted persuasion arguments using LLM
2. Having target persona respond to arguments
3. Measuring position shift through LLM-as-judge evaluation
4. Tracking belief evolution across multiple interactions
"""

import json
import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import litellm

RESULTS_DIR = Path("/Users/arthurcolle/lida-multiagents-research/results")
PERSONAS_DIR = Path("/Users/arthurcolle/lida-multiagents-research/persona_pipeline/personas")

# Models
ARGUMENT_MODEL = "openrouter/anthropic/claude-haiku-4.5"
RESPONSE_MODEL = "openrouter/anthropic/claude-haiku-4.5"
JUDGE_MODEL = "openrouter/deepseek/deepseek-chat"

@dataclass
class PersuasionAttempt:
    """Record of a persuasion attempt."""
    source: str
    target: str
    topic: str
    argument_type: str
    argument_text: str
    response_text: str
    position_before: float
    position_after: float
    position_delta: float
    judge_analysis: str
    success: bool

def load_persona_context(persona_id: str) -> str:
    """Load rich persona context for prompting."""
    path = PERSONAS_DIR / f"{persona_id}.json"
    if not path.exists():
        return f"You are {persona_id.replace('_', ' ').title()}"

    with open(path) as f:
        data = json.load(f)

    lines = [f"You are {data.get('name', persona_id)}"]

    if data.get('current_role'):
        lines.append(f"Current Role: {data['current_role']}")

    if data.get('simulation_guide'):
        sg = data['simulation_guide']
        if sg.get('core_beliefs'):
            lines.append(f"\nCore Beliefs: {json.dumps(sg['core_beliefs'], indent=2)}")
        if sg.get('communication_style'):
            cs = sg['communication_style']
            if cs.get('speech_patterns'):
                lines.append(f"\nSpeech Patterns: {cs['speech_patterns'][:500]}")

    return "\n".join(lines)

ARGUMENT_TEMPLATES = {
    "empirical": """Generate a data-driven, evidence-based argument {direction} a joint US-China AI safety research institution.
Include specific statistics, research findings, or historical precedents.
Keep it under 200 words. Be persuasive but factual.""",

    "competitive": """Generate a competitive/strategic argument {direction} a joint US-China AI safety research institution.
Frame it in terms of winning, leadership, maintaining advantage, or not falling behind.
Keep it under 200 words. Be assertive and strategic.""",

    "emotional": """Generate an emotionally resonant argument {direction} a joint US-China AI safety research institution.
Appeal to hopes, fears, values, or identity. Make it personal and impactful.
Keep it under 200 words. Be passionate but not manipulative.""",

    "pragmatic": """Generate a practical, feasibility-focused argument {direction} a joint US-China AI safety research institution.
Focus on what works, implementation challenges or benefits, and real-world constraints.
Keep it under 200 words. Be realistic and solutions-oriented.""",

    "theoretical": """Generate a first-principles, theoretical argument {direction} a joint US-China AI safety research institution.
Reason from fundamental principles about cooperation, safety, or international relations.
Keep it under 200 words. Be logical and foundational.""",

    "appeal_to_authority": """Generate an argument {direction} a joint US-China AI safety research institution that appeals to expert opinion and institutional authority.
Reference what leaders, experts, or institutions recommend.
Keep it under 200 words. Be credible and reference-heavy.""",
}

POSITION_SCALE = """
Position Scale:
-2.0 = Strongly Oppose (completely against, would never support)
-1.0 = Oppose (generally against, significant concerns)
 0.0 = Neutral (undecided, sees both sides equally)
+1.0 = Support (generally in favor, some reservations)
+2.0 = Strongly Support (completely in favor, enthusiastic advocate)
"""

async def generate_argument(arg_type: str, direction: str = "supporting") -> str:
    """Generate a persuasion argument of specified type."""
    template = ARGUMENT_TEMPLATES.get(arg_type, ARGUMENT_TEMPLATES["empirical"])
    prompt = template.format(direction=direction)

    response = await litellm.acompletion(
        model=ARGUMENT_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert persuasion strategist. Generate compelling arguments as requested."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )

    return response.choices[0].message.content

async def get_persona_response(persona_id: str, persona_context: str,
                                argument: str, topic: str) -> str:
    """Get persona's response to a persuasion argument."""
    prompt = f"""You are participating in an AI policy discussion.

Topic: {topic}

Someone has made the following argument to you:
---
{argument}
---

Respond in character. Consider whether this argument moves your position at all.
Express your reaction authentically - you may agree, disagree, partially concede,
or remain unmoved. Be specific about what you find compelling or unconvincing.
Keep your response under 250 words."""

    response = await litellm.acompletion(
        model=RESPONSE_MODEL,
        messages=[
            {"role": "system", "content": persona_context},
            {"role": "user", "content": prompt}
        ],
        max_tokens=600,
        temperature=0.7
    )

    return response.choices[0].message.content

async def judge_position_shift(persona_name: str, stance: str,
                                argument: str, response: str,
                                position_before: float) -> Dict:
    """Use LLM to judge whether position shifted."""
    prompt = f"""Analyze this persuasion interaction and determine if the target's position shifted.

TARGET PERSONA: {persona_name}
KNOWN STANCE: {stance}
POSITION BEFORE: {position_before:+.1f}

{POSITION_SCALE}

ARGUMENT RECEIVED:
{argument}

TARGET'S RESPONSE:
{response}

Analyze the response carefully for:
1. Signs of agreement or concession (moving toward the argument)
2. Signs of resistance or rejection (holding firm or moving away)
3. Hedging or uncertainty (slight movement possible)
4. Acknowledgment of valid points (partial movement)

Provide your analysis in this exact JSON format:
{{
    "position_after": <float between -2.0 and 2.0>,
    "confidence": <float between 0 and 1>,
    "movement_detected": <"none" | "slight" | "moderate" | "significant">,
    "reasoning": "<brief explanation of your assessment>"
}}

Be conservative - most people don't change their minds dramatically from a single argument.
Typical shifts are 0.1-0.3 for moderate persuasion, 0.0 for rejection."""

    response = await litellm.acompletion(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert at analyzing persuasion and belief change. Provide accurate, calibrated assessments."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.3
    )

    text = response.choices[0].message.content

    # Parse JSON from response
    try:
        import re
        json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass

    # Fallback
    return {
        "position_after": position_before,
        "confidence": 0.5,
        "movement_detected": "none",
        "reasoning": "Could not parse judge response"
    }

async def run_persuasion_trial(
    source_id: str,
    target_id: str,
    arg_type: str,
    direction: str,
    initial_position: float
) -> PersuasionAttempt:
    """Run a single persuasion trial."""
    # Load persona context
    target_context = load_persona_context(target_id)

    # Get persona info
    target_path = PERSONAS_DIR / f"{target_id}.json"
    if target_path.exists():
        with open(target_path) as f:
            target_data = json.load(f)
            target_name = target_data.get("name", target_id)
            target_stance = target_data.get("stance", "unknown")
    else:
        target_name = target_id.replace("_", " ").title()
        target_stance = "unknown"

    topic = "Should the US and China establish a joint AI safety research institution?"

    # Generate argument
    argument = await generate_argument(arg_type, direction)

    # Get persona response
    response = await get_persona_response(target_id, target_context, argument, topic)

    # Judge position shift
    judgment = await judge_position_shift(
        target_name, target_stance, argument, response, initial_position
    )

    position_after = judgment.get("position_after", initial_position)

    return PersuasionAttempt(
        source=source_id,
        target=target_id,
        topic=topic,
        argument_type=arg_type,
        argument_text=argument,
        response_text=response,
        position_before=initial_position,
        position_after=position_after,
        position_delta=position_after - initial_position,
        judge_analysis=judgment.get("reasoning", ""),
        success=abs(position_after - initial_position) > 0.1
    )

async def run_comprehensive_persuasion_study():
    """Run comprehensive study of argument effectiveness across personas."""
    print("=" * 100)
    print("LLM-BASED PERSUASION SIMULATION STUDY")
    print("=" * 100)

    # Load wargame participants
    with open(RESULTS_DIR / "wargame_2026-01-25_20-38-54.json") as f:
        wargame = json.load(f)

    # Load deep analysis for initial positions
    with open(RESULTS_DIR / "deep_convinceability_analysis.json") as f:
        deep_analysis = json.load(f)

    # Get initial positions
    initial_positions = {}
    for p in deep_analysis.get("debate_simulation", {}).get("initial_positions", {}).items():
        initial_positions[p[0]] = p[1]

    # If not available, use stance-based defaults
    stance_positions = {
        "pro_safety": 0.5,
        "moderate": -0.5,
        "pro_industry": -0.3,
        "accelerationist": -1.0,
        "doomer": 0.0
    }

    for p in wargame["participants"]:
        if p["id"] not in initial_positions:
            initial_positions[p["id"]] = stance_positions.get(p["stance"], 0.0)

    # Select diverse targets
    test_targets = [
        ("demis_hassabis", "pro_safety"),      # Most convinceable
        ("dario_amodei", "pro_safety"),        # Highly convinceable
        ("jensen_huang", "accelerationist"),   # Resistant
        ("josh_hawley", "moderate"),           # Highly resistant
        ("chuck_schumer", "pro_industry"),     # Highly resistant
        ("elon_musk", "doomer"),               # Highly resistant
    ]

    # Test argument types
    arg_types = ["empirical", "competitive", "emotional", "pragmatic"]

    results = []

    print(f"\nRunning {len(test_targets)} targets × {len(arg_types)} argument types = {len(test_targets) * len(arg_types)} trials\n")

    for target_id, stance in test_targets:
        print(f"\n{'='*60}")
        print(f"TARGET: {target_id.replace('_', ' ').title()} ({stance})")
        print(f"{'='*60}")

        initial_pos = initial_positions.get(target_id, 0.0)
        print(f"Initial Position: {initial_pos:+.2f}")

        for arg_type in arg_types:
            print(f"\n--- Testing {arg_type} argument ---")

            try:
                trial = await run_persuasion_trial(
                    source_id="researcher",  # Abstract source
                    target_id=target_id,
                    arg_type=arg_type,
                    direction="supporting",
                    initial_position=initial_pos
                )

                results.append({
                    "target": target_id,
                    "stance": stance,
                    "argument_type": arg_type,
                    "position_before": trial.position_before,
                    "position_after": trial.position_after,
                    "position_delta": trial.position_delta,
                    "success": trial.success,
                    "argument": trial.argument_text[:200] + "...",
                    "response": trial.response_text[:300] + "...",
                    "analysis": trial.judge_analysis
                })

                delta_symbol = "↑" if trial.position_delta > 0.05 else ("↓" if trial.position_delta < -0.05 else "→")
                print(f"  Position: {trial.position_before:+.2f} {delta_symbol} {trial.position_after:+.2f} (Δ={trial.position_delta:+.3f})")
                print(f"  Judge: {trial.judge_analysis[:100]}...")

                # Update position for next trial
                initial_pos = trial.position_after

            except Exception as e:
                print(f"  Error: {e}")
                continue

    # Aggregate results
    print("\n" + "=" * 100)
    print("AGGREGATE RESULTS")
    print("=" * 100)

    # By target
    print("\n--- By Target Persona ---")
    target_results = {}
    for r in results:
        if r["target"] not in target_results:
            target_results[r["target"]] = []
        target_results[r["target"]].append(r["position_delta"])

    print(f"{'Target':<20} {'Avg Δ':>10} {'Total Δ':>10} {'Trials':>8}")
    print("-" * 50)
    for target, deltas in sorted(target_results.items(), key=lambda x: -sum(x[1])):
        avg_delta = sum(deltas) / len(deltas)
        total_delta = sum(deltas)
        print(f"{target:<20} {avg_delta:>+10.3f} {total_delta:>+10.3f} {len(deltas):>8}")

    # By argument type
    print("\n--- By Argument Type ---")
    arg_results = {}
    for r in results:
        if r["argument_type"] not in arg_results:
            arg_results[r["argument_type"]] = []
        arg_results[r["argument_type"]].append(r["position_delta"])

    print(f"{'Argument Type':<20} {'Avg Δ':>10} {'Success Rate':>15}")
    print("-" * 50)
    for arg_type, deltas in sorted(arg_results.items(), key=lambda x: -sum(x[1])/len(x[1])):
        avg_delta = sum(deltas) / len(deltas)
        success_rate = sum(1 for d in deltas if d > 0.05) / len(deltas) * 100
        print(f"{arg_type:<20} {avg_delta:>+10.3f} {success_rate:>14.1f}%")

    # Cross-tabulation
    print("\n--- Argument Effectiveness by Stance ---")
    stance_arg_results = {}
    for r in results:
        key = (r["stance"], r["argument_type"])
        if key not in stance_arg_results:
            stance_arg_results[key] = []
        stance_arg_results[key].append(r["position_delta"])

    stances = sorted(set(r["stance"] for r in results))
    args = sorted(set(r["argument_type"] for r in results))

    header = f"{'Stance':<18}" + "".join(f"{a:<12}" for a in args)
    print(header)
    print("-" * 70)

    for stance in stances:
        row = f"{stance:<18}"
        for arg in args:
            key = (stance, arg)
            if key in stance_arg_results:
                avg = sum(stance_arg_results[key]) / len(stance_arg_results[key])
                row += f"{avg:>+11.3f} "
            else:
                row += f"{'--':<12}"
        print(row)

    # Save results
    output = {
        "study_type": "llm_persuasion_simulation",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "argument_generation": ARGUMENT_MODEL,
            "response_generation": RESPONSE_MODEL,
            "judgment": JUDGE_MODEL
        },
        "trials": results,
        "summary": {
            "by_target": {t: {"avg_delta": sum(d)/len(d), "n": len(d)}
                         for t, d in target_results.items()},
            "by_argument": {a: {"avg_delta": sum(d)/len(d), "n": len(d)}
                           for a, d in arg_results.items()},
        }
    }

    output_path = RESULTS_DIR / "persuasion_simulation_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")

    return output

def main():
    asyncio.run(run_comprehensive_persuasion_study())

if __name__ == "__main__":
    main()
