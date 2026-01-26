#!/usr/bin/env python3
"""Generate cross-persona persuasion influence matrix and strategy recommendations."""

import json
from pathlib import Path
from collections import defaultdict
from itertools import combinations

RESULTS_DIR = Path("/Users/arthurcolle/lida-multiagents-research/results")

# Persuasion vectors - what arguments work on what types of personas
PERSUASION_VECTORS = {
    "economic": {
        "keywords": ["market", "growth", "jobs", "investment", "gdp", "economy", "industry"],
        "effective_on": ["pro_industry", "accelerationist"],
        "description": "Market-based, economic growth arguments"
    },
    "safety": {
        "keywords": ["risk", "safety", "alignment", "catastrophic", "existential", "harm"],
        "effective_on": ["pro_safety", "doomer", "moderate"],
        "description": "Risk mitigation and safety-focused arguments"
    },
    "national_security": {
        "keywords": ["security", "defense", "adversary", "threat", "protect", "sovereignty"],
        "effective_on": ["moderate"],
        "description": "National security and defense arguments"
    },
    "competition": {
        "keywords": ["competition", "lead", "winning", "dominance", "edge", "advantage"],
        "effective_on": ["accelerationist", "pro_industry"],
        "description": "Competitive advantage and leadership arguments"
    },
    "cooperation": {
        "keywords": ["cooperation", "collaboration", "partnership", "together", "mutual", "shared"],
        "effective_on": ["pro_safety"],
        "description": "Collaborative and multilateral arguments"
    },
    "technical": {
        "keywords": ["research", "technical", "empirical", "evidence", "data", "methodology"],
        "effective_on": ["pro_safety", "doomer"],
        "description": "Evidence-based technical arguments"
    },
    "populist": {
        "keywords": ["workers", "families", "people", "citizens", "Americans", "jobs"],
        "effective_on": ["pro_industry", "moderate"],
        "description": "Worker/citizen-focused populist arguments"
    },
}

# Known relationships and alliances (based on real-world dynamics)
RELATIONSHIPS = {
    ("dario_amodei", "sam_altman"): {"type": "aligned", "strength": 0.7},
    ("dario_amodei", "demis_hassabis"): {"type": "aligned", "strength": 0.9},
    ("sam_altman", "elon_musk"): {"type": "adversarial", "strength": -0.6},
    ("jensen_huang", "mark_zuckerberg"): {"type": "aligned", "strength": 0.5},
    ("chuck_schumer", "josh_hawley"): {"type": "adversarial", "strength": -0.8},
    ("xi_jinping", "josh_hawley"): {"type": "adversarial", "strength": -0.9},
    ("gina_raimondo", "xi_jinping"): {"type": "adversarial", "strength": -0.5},
    ("rishi_sunak", "dario_amodei"): {"type": "aligned", "strength": 0.6},
}

def load_convinceability():
    """Load convinceability analysis results."""
    with open(RESULTS_DIR / "convinceability_analysis.json") as f:
        return json.load(f)

def load_wargame():
    """Load wargame transcript."""
    with open(RESULTS_DIR / "wargame_2026-01-25_20-38-54.json") as f:
        return json.load(f)

def analyze_argument_vectors(wargame):
    """Analyze what persuasion vectors each persona uses."""
    persona_vectors = defaultdict(lambda: defaultdict(int))

    for msg in wargame["transcript"]:
        text = msg["response"].lower()
        persona = msg["persona_id"]

        for vector_name, vector_data in PERSUASION_VECTORS.items():
            for keyword in vector_data["keywords"]:
                count = text.count(keyword.lower())
                persona_vectors[persona][vector_name] += count

    return dict(persona_vectors)

def calculate_influence_potential(source, target, convinceability_data, argument_vectors):
    """Calculate how effectively source can influence target."""
    # Get source's argument style
    source_data = next((p for p in convinceability_data["personas"] if p["persona_id"] == source), None)
    target_data = next((p for p in convinceability_data["personas"] if p["persona_id"] == target), None)

    if not source_data or not target_data:
        return 0

    # Base influence from target's convinceability
    base_influence = target_data["convinceability_score"] / 100

    # Modify by relationship
    rel_key = tuple(sorted([source, target]))
    relationship = RELATIONSHIPS.get(rel_key, RELATIONSHIPS.get((source, target), {"type": "neutral", "strength": 0}))

    rel_modifier = 1.0
    if relationship["type"] == "aligned":
        rel_modifier = 1.0 + (relationship["strength"] * 0.5)  # Boost for allies
    elif relationship["type"] == "adversarial":
        rel_modifier = 1.0 + (relationship["strength"] * 0.3)  # Penalty for adversaries (negative strength)

    # Modify by argument vector alignment
    target_stance = target_data["stance"]
    source_vectors = argument_vectors.get(source, {})

    vector_match = 0
    for vector_name, count in source_vectors.items():
        if count > 0 and target_stance in PERSUASION_VECTORS[vector_name]["effective_on"]:
            vector_match += count * 0.02

    vector_modifier = min(1.5, 1.0 + vector_match)

    final_influence = base_influence * rel_modifier * vector_modifier
    return round(min(1.0, final_influence), 3)

def generate_influence_matrix(convinceability_data, argument_vectors):
    """Generate persona-to-persona influence matrix."""
    personas = [p["persona_id"] for p in convinceability_data["personas"]]

    matrix = {}
    for source in personas:
        matrix[source] = {}
        for target in personas:
            if source != target:
                matrix[source][target] = calculate_influence_potential(
                    source, target, convinceability_data, argument_vectors
                )
            else:
                matrix[source][target] = 0  # Can't influence self

    return matrix

def identify_persuasion_strategies(convinceability_data, argument_vectors):
    """Generate personalized persuasion strategies for each persona."""
    strategies = {}

    for persona in convinceability_data["personas"]:
        pid = persona["persona_id"]
        stance = persona["stance"]
        score = persona["convinceability_score"]

        # Find effective vectors for this persona's stance
        effective_vectors = [
            (v_name, v_data["description"])
            for v_name, v_data in PERSUASION_VECTORS.items()
            if stance in v_data["effective_on"]
        ]

        # Identify key triggers based on markers
        triggers = []
        if persona["markers"]["openness"] > 5:
            triggers.append("Open to nuanced discussion - present tradeoffs")
        if persona["markers"]["concession"] > 0:
            triggers.append("Acknowledges valid points - build on common ground")
        if persona["markers"]["dogmatic"] > 3:
            triggers.append("Strongly committed - avoid direct contradiction")
        if persona["markers"]["anchoring"] > 2:
            triggers.append("Position anchored - frame as enhancement not change")

        # Determine difficulty
        if score >= 70:
            difficulty = "Low - highly receptive to well-reasoned arguments"
        elif score >= 50:
            difficulty = "Medium - open to persuasion with right framing"
        elif score >= 30:
            difficulty = "High - resistant, requires aligned messenger"
        else:
            difficulty = "Very High - deeply entrenched, focus on other targets"

        strategies[pid] = {
            "name": persona["name"],
            "convinceability_score": score,
            "difficulty": difficulty,
            "effective_vectors": effective_vectors,
            "key_triggers": triggers,
            "avoid": []
        }

        # Add avoidance recommendations
        if stance == "pro_safety":
            strategies[pid]["avoid"].append("Pure economic arguments without safety framing")
        elif stance == "accelerationist":
            strategies[pid]["avoid"].append("Fear-based or restrictive arguments")
        elif stance == "doomer":
            strategies[pid]["avoid"].append("Dismissing existential concerns")
        elif stance == "pro_industry":
            strategies[pid]["avoid"].append("Heavy regulatory proposals")
        elif stance == "moderate":
            strategies[pid]["avoid"].append("Extreme positions from either side")

    return strategies

def print_analysis(influence_matrix, strategies, convinceability_data):
    """Print comprehensive persuasion analysis."""
    personas = [p["persona_id"] for p in convinceability_data["personas"]]
    names = {p["persona_id"]: p["name"].split()[0] for p in convinceability_data["personas"]}

    print("=" * 120)
    print("CROSS-PERSONA INFLUENCE MATRIX")
    print("=" * 120)
    print("\nValues represent probability (0-1) that source can influence target's position\n")

    # Header
    header = "Source →      " + "".join(f"{names[t][:8]:<10}" for t in personas)
    print(header)
    print("-" * 120)

    for source in personas:
        row = f"{names[source][:12]:<14}"
        for target in personas:
            val = influence_matrix[source][target]
            if val == 0:
                row += f"{'--':<10}"
            elif val >= 0.5:
                row += f"{val:<10.2f}"  # High influence
            elif val >= 0.3:
                row += f"{val:<10.2f}"  # Medium influence
            else:
                row += f"{val:<10.2f}"  # Low influence
        print(row)

    print("\n" + "=" * 120)
    print("TOP INFLUENCE PAIRS (Who can persuade whom)")
    print("=" * 120)

    pairs = []
    for source in personas:
        for target in personas:
            if source != target:
                pairs.append((source, target, influence_matrix[source][target]))

    pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"\n{'Rank':<6} {'Influencer':<20} {'Target':<20} {'Influence':>10}")
    print("-" * 60)
    for i, (src, tgt, val) in enumerate(pairs[:15], 1):
        src_name = next(p["name"] for p in convinceability_data["personas"] if p["persona_id"] == src)
        tgt_name = next(p["name"] for p in convinceability_data["personas"] if p["persona_id"] == tgt)
        print(f"{i:<6} {src_name:<20} {tgt_name:<20} {val:>10.3f}")

    print("\n" + "=" * 120)
    print("PERSUASION STRATEGIES BY PERSONA")
    print("=" * 120)

    for pid, strat in strategies.items():
        print(f"\n--- {strat['name']} (Score: {strat['convinceability_score']}) ---")
        print(f"Difficulty: {strat['difficulty']}")
        print(f"Effective Arguments:")
        for v_name, v_desc in strat['effective_vectors']:
            print(f"  • {v_name}: {v_desc}")
        if strat['key_triggers']:
            print(f"Key Triggers:")
            for t in strat['key_triggers']:
                print(f"  • {t}")
        if strat['avoid']:
            print(f"Avoid:")
            for a in strat['avoid']:
                print(f"  • {a}")

    print("\n" + "=" * 120)
    print("COALITION BUILDING RECOMMENDATIONS")
    print("=" * 120)

    # Find natural coalitions based on influence patterns
    pro_safety_group = [p for p in convinceability_data["personas"] if p["stance"] == "pro_safety"]
    moderate_group = [p for p in convinceability_data["personas"] if p["stance"] == "moderate"]

    print("\nTo build consensus for joint US-China AI safety institution:")
    print("\n1. START WITH: Pro-safety tech leaders")
    for p in sorted(pro_safety_group, key=lambda x: -x["convinceability_score"]):
        print(f"   • {p['name']} (Score: {p['convinceability_score']}) - Natural ally")

    print("\n2. THEN TARGET: Moderates through security framing")
    for p in sorted(moderate_group, key=lambda x: -x["convinceability_score"]):
        print(f"   • {p['name']} (Score: {p['convinceability_score']})")

    print("\n3. HARDEST TO MOVE: Politicians and accelerationists")
    resistant = [p for p in convinceability_data["personas"] if p["convinceability_score"] < 30]
    for p in sorted(resistant, key=lambda x: x["convinceability_score"]):
        print(f"   • {p['name']} (Score: {p['convinceability_score']}) - Focus elsewhere")

def save_results(influence_matrix, strategies, output_path):
    """Save all persuasion analysis to JSON."""
    output = {
        "influence_matrix": influence_matrix,
        "persuasion_strategies": strategies,
        "persuasion_vectors": PERSUASION_VECTORS,
        "relationships": {str(k): v for k, v in RELATIONSHIPS.items()},
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

def main():
    convinceability_data = load_convinceability()
    wargame = load_wargame()

    argument_vectors = analyze_argument_vectors(wargame)
    influence_matrix = generate_influence_matrix(convinceability_data, argument_vectors)
    strategies = identify_persuasion_strategies(convinceability_data, argument_vectors)

    print_analysis(influence_matrix, strategies, convinceability_data)
    save_results(influence_matrix, strategies, RESULTS_DIR / "persuasion_matrix.json")

if __name__ == "__main__":
    main()
