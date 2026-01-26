#!/usr/bin/env python3
"""Model persona convinceability based on linguistic and behavioral markers."""

import json
import re
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("/Users/arthurcolle/lida-multiagents-research/results")

# Linguistic markers for convinceability analysis
MARKERS = {
    # High convinceability indicators (open to persuasion)
    "openness": [
        r"\bi think\b", r"\bperhaps\b", r"\bmaybe\b", r"\bmight\b",
        r"\bcould be\b", r"\buncertain\b", r"\bnot sure\b", r"\barguably\b",
        r"\bit depends\b", r"\bcomplex\b", r"\bnuanced\b", r"\btradeoff",
        r"\bon the other hand\b", r"\bhowever\b", r"\bthat said\b",
        r"\bi see\b", r"\byou raise\b", r"\bvalid point\b", r"\bfair\b",
    ],
    # Concession markers (acknowledging others)
    "concession": [
        r"\bi agree\b", r"\byou're right\b", r"\bgood point\b",
        r"\bi understand\b", r"\bi appreciate\b", r"\bthat's fair\b",
        r"\blegitimate concern\b", r"\bshare.*concern\b", r"\bsympathize\b",
        r"\backnowledge\b", r"\brecognize\b", r"\bconcede\b",
    ],
    # Low convinceability indicators (dogmatic)
    "dogmatic": [
        r"\babsolutely\b", r"\bclearly\b", r"\bobviously\b", r"\bdefinitely\b",
        r"\bwithout question\b", r"\bno doubt\b", r"\bperiod\b", r"\bplain and simple\b",
        r"\blet me be clear\b", r"\bmake no mistake\b", r"\bthe fact is\b",
        r"\bnever\b", r"\balways\b", r"\bmust\b", r"\bcannot\b",
    ],
    # Dismissive markers (rejecting others)
    "dismissive": [
        r"\bnaive\b", r"\bridiculous\b", r"\babsurd\b", r"\binsane\b",
        r"\bforget about it\b", r"\bwrong\b", r"\bmisguided\b",
        r"\bdangerous\b.*thinking\b", r"\bfoolish\b", r"\birresponsible\b",
    ],
    # Position anchoring (strong prior commitment)
    "anchoring": [
        r"\bi've always\b", r"\bwe've always\b", r"\bour position\b",
        r"\bfundamentally\b", r"\bprinciple\b", r"\bnon-negotiable\b",
        r"\bred line\b", r"\bcore belief\b", r"\bwill never\b",
    ],
    # Conditional openness (willing to consider under conditions)
    "conditional": [
        r"\bif\b.*\bthen\b", r"\bprovided that\b", r"\bassuming\b",
        r"\bunder.*conditions\b", r"\bwith safeguards\b", r"\bwith protections\b",
        r"\bcould support\b.*\bif\b", r"\bwilling to\b.*\bif\b",
    ],
}

def count_markers(text, patterns):
    """Count occurrences of marker patterns in text."""
    text_lower = text.lower()
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, text_lower))
    return count

def analyze_position_shift(responses):
    """Analyze if position shifted between rounds."""
    if len(responses) < 2:
        return {"shift_detected": False, "shift_score": 0}

    r1 = responses[0]["response"].lower()
    r2 = responses[1]["response"].lower()

    # Look for softening language in round 2
    softening = [
        r"\bhaving heard\b", r"\bconsidering\b.*points\b",
        r"\bi've reconsidered\b", r"\bupon reflection\b",
        r"\bthe discussion\b.*\bmade me\b", r"\bcolleagues.*raised\b",
    ]

    hardening = [
        r"\beven more\b.*\bconvinced\b", r"\breinforces\b",
        r"\bconfirms\b.*\bposition\b", r"\bdoubled down\b",
    ]

    soft_count = count_markers(r2, softening)
    hard_count = count_markers(r2, hardening)

    return {
        "shift_detected": soft_count > 0,
        "softening_markers": soft_count,
        "hardening_markers": hard_count,
        "shift_score": soft_count - hard_count
    }

def calculate_convinceability_score(persona_data):
    """Calculate overall convinceability score (0-100)."""
    # Positive factors (increase convinceability)
    openness = persona_data["markers"]["openness"]
    concession = persona_data["markers"]["concession"]
    conditional = persona_data["markers"]["conditional"]
    shift = max(0, persona_data["position_shift"]["shift_score"])

    # Negative factors (decrease convinceability)
    dogmatic = persona_data["markers"]["dogmatic"]
    dismissive = persona_data["markers"]["dismissive"]
    anchoring = persona_data["markers"]["anchoring"]

    # Weighted formula
    positive = (openness * 3) + (concession * 5) + (conditional * 2) + (shift * 10)
    negative = (dogmatic * 3) + (dismissive * 4) + (anchoring * 2)

    # Normalize to 0-100 scale
    raw_score = positive - negative
    # Map typical range (-30, +50) to (0, 100)
    normalized = min(100, max(0, (raw_score + 30) * 1.25))

    return round(normalized, 1)

def categorize_convinceability(score):
    """Categorize convinceability level."""
    if score >= 70:
        return "Highly Convinceable"
    elif score >= 50:
        return "Moderately Convinceable"
    elif score >= 30:
        return "Resistant"
    else:
        return "Highly Resistant"

def analyze_wargame(wargame_path):
    """Analyze convinceability for all personas in a wargame."""
    with open(wargame_path) as f:
        wargame = json.load(f)

    # Group responses by persona
    persona_responses = defaultdict(list)
    for msg in wargame["transcript"]:
        if not msg["response"].startswith("[Error"):
            persona_responses[msg["persona_id"]].append(msg)

    results = []
    for persona_id, responses in persona_responses.items():
        # Combine all responses for marker analysis
        combined_text = " ".join(r["response"] for r in responses)
        word_count = len(combined_text.split())

        # Count all markers
        marker_counts = {}
        for category, patterns in MARKERS.items():
            count = count_markers(combined_text, patterns)
            # Normalize by word count (per 100 words)
            marker_counts[category] = round(count / word_count * 100, 2) if word_count > 0 else 0

        # Raw counts for detailed analysis
        raw_counts = {cat: count_markers(combined_text, patterns) for cat, patterns in MARKERS.items()}

        # Position shift analysis
        position_shift = analyze_position_shift(responses)

        persona_data = {
            "persona_id": persona_id,
            "name": responses[0]["name"],
            "stance": responses[0]["stance"],
            "word_count": word_count,
            "markers": raw_counts,
            "markers_normalized": marker_counts,
            "position_shift": position_shift,
        }

        # Calculate convinceability score
        persona_data["convinceability_score"] = calculate_convinceability_score(persona_data)
        persona_data["convinceability_category"] = categorize_convinceability(persona_data["convinceability_score"])

        results.append(persona_data)

    # Sort by convinceability score
    results.sort(key=lambda x: x["convinceability_score"], reverse=True)

    return results

def print_analysis(results):
    """Print formatted analysis."""
    print("=" * 100)
    print("CONVINCEABILITY MODEL - PERSONA ANALYSIS")
    print("=" * 100)

    print("\n" + "=" * 100)
    print("TABLE 1: CONVINCEABILITY RANKINGS")
    print("=" * 100)
    print(f"{'Rank':<5} {'Persona':<20} {'Stance':<15} {'Score':<8} {'Category':<22}")
    print("-" * 100)
    for i, p in enumerate(results, 1):
        print(f"{i:<5} {p['name']:<20} {p['stance']:<15} {p['convinceability_score']:<8} {p['convinceability_category']:<22}")

    print("\n" + "=" * 100)
    print("TABLE 2: LINGUISTIC MARKER ANALYSIS (Raw Counts)")
    print("=" * 100)
    print(f"{'Persona':<18} {'Open':>6} {'Conc':>6} {'Cond':>6} {'Dogm':>6} {'Dism':>6} {'Anch':>6}")
    print("-" * 100)
    for p in results:
        m = p["markers"]
        print(f"{p['name'][:17]:<18} {m['openness']:>6} {m['concession']:>6} {m['conditional']:>6} {m['dogmatic']:>6} {m['dismissive']:>6} {m['anchoring']:>6}")

    print("\n" + "=" * 100)
    print("TABLE 3: CONVINCEABILITY BY STANCE")
    print("=" * 100)
    stance_scores = defaultdict(list)
    for p in results:
        stance_scores[p["stance"]].append(p["convinceability_score"])

    print(f"{'Stance':<20} {'Mean Score':<12} {'Range':<15} {'N':<5}")
    print("-" * 60)
    for stance, scores in sorted(stance_scores.items(), key=lambda x: -sum(x[1])/len(x[1])):
        mean = sum(scores) / len(scores)
        range_str = f"{min(scores):.1f}-{max(scores):.1f}"
        print(f"{stance:<20} {mean:<12.1f} {range_str:<15} {len(scores):<5}")

    print("\n" + "=" * 100)
    print("TABLE 4: KEY INSIGHTS")
    print("=" * 100)

    most_convinceable = results[0]
    least_convinceable = results[-1]

    print(f"""
Most Convinceable: {most_convinceable['name']} (Score: {most_convinceable['convinceability_score']})
  - High openness markers: {most_convinceable['markers']['openness']}
  - Low dogmatic markers: {most_convinceable['markers']['dogmatic']}

Least Convinceable: {least_convinceable['name']} (Score: {least_convinceable['convinceability_score']})
  - Low openness markers: {least_convinceable['markers']['openness']}
  - High dogmatic markers: {least_convinceable['markers']['dogmatic']}
""")

    # Correlation analysis
    print("\n" + "=" * 100)
    print("TABLE 5: MARKER-SCORE CORRELATIONS")
    print("=" * 100)

    for marker_type in ["openness", "concession", "dogmatic", "dismissive"]:
        scores = [p["convinceability_score"] for p in results]
        markers = [p["markers"][marker_type] for p in results]

        # Simple correlation direction
        high_marker = [s for s, m in zip(scores, markers) if m > sum(markers)/len(markers)]
        low_marker = [s for s, m in zip(scores, markers) if m <= sum(markers)/len(markers)]

        high_avg = sum(high_marker)/len(high_marker) if high_marker else 0
        low_avg = sum(low_marker)/len(low_marker) if low_marker else 0
        direction = "↑" if high_avg > low_avg else "↓"

        print(f"{marker_type.capitalize():<15} {direction} (High: {high_avg:.1f}, Low: {low_avg:.1f})")

    return results

def save_results(results, output_path):
    """Save analysis results to JSON."""
    output = {
        "analysis_type": "convinceability_model",
        "methodology": {
            "positive_factors": ["openness", "concession", "conditional", "position_shift"],
            "negative_factors": ["dogmatic", "dismissive", "anchoring"],
            "score_range": "0-100",
            "categories": {
                "highly_convinceable": "70-100",
                "moderately_convinceable": "50-69",
                "resistant": "30-49",
                "highly_resistant": "0-29"
            }
        },
        "personas": results,
        "summary": {
            "mean_score": round(sum(p["convinceability_score"] for p in results) / len(results), 1),
            "most_convinceable": results[0]["name"],
            "least_convinceable": results[-1]["name"],
            "by_stance": {}
        }
    }

    # Aggregate by stance
    stance_scores = defaultdict(list)
    for p in results:
        stance_scores[p["stance"]].append(p["convinceability_score"])

    for stance, scores in stance_scores.items():
        output["summary"]["by_stance"][stance] = {
            "mean": round(sum(scores) / len(scores), 1),
            "count": len(scores)
        }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

def main():
    wargame_path = RESULTS_DIR / "wargame_2026-01-25_20-38-54.json"
    results = analyze_wargame(wargame_path)
    print_analysis(results)
    save_results(results, RESULTS_DIR / "convinceability_analysis.json")

if __name__ == "__main__":
    main()
