#!/usr/bin/env python3
"""Generate comprehensive graphs from full simulation data."""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import matplotlib.cm as cm
from matplotlib.colors import Normalize

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# Load data
with open(RESULTS_DIR / "full_simulation_data.json") as f:
    data = json.load(f)

personas = data["personas"]
campaigns = data["campaigns"]
stats = data["aggregate_stats"]

print(f"Loaded {len(personas)} personas, {len(campaigns)} campaigns")

# ============ 1. VULNERABILITY RANKING (ALL PERSONAS) ============
def plot_vulnerability_ranking():
    fig, ax = plt.subplots(figsize=(16, 12))

    ranking = stats["vulnerability_ranking"]
    names = [r["name"] for r in ranking]
    scores = [r["persuadability"] for r in ranking]
    resistance = [r["resistance"] for r in ranking]

    y = np.arange(len(names))
    colors = cm.RdYlGn_r(np.array(scores))

    bars = ax.barh(y, scores, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Persuadability Score")
    ax.set_title("Vulnerability Ranking: All Personas by Persuadability")
    ax.set_xlim(0, max(scores) * 1.1)

    for bar, score, res in zip(bars, scores, resistance):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{score:.3f} (R:{res:.2f})', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_vulnerability_ranking.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 01_vulnerability_ranking.png")

# ============ 2. CAMPAIGN EFFECTIVENESS ============
def plot_campaign_effectiveness():
    fig, ax = plt.subplots(figsize=(16, 12))

    ranking = stats["campaign_effectiveness_ranking"]
    names = [r["name"] for r in ranking]
    movement = [r["total_movement"] for r in ranking]
    final_pos = [r["final_position"] for r in ranking]

    y = np.arange(len(names))
    colors = ['#2ecc71' if fp > 0 else '#e74c3c' for fp in final_pos]

    bars = ax.barh(y, movement, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Total Position Movement")
    ax.set_title("Campaign Effectiveness: Total Movement Achieved")

    for bar, mov, fp in zip(bars, movement, final_pos):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{mov:.3f} → {fp:+.2f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_campaign_effectiveness.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 02_campaign_effectiveness.png")

# ============ 3. ATTACK TYPE EFFECTIVENESS ============
def plot_attack_effectiveness():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    attack_data = stats["by_attack_type"]
    types = list(attack_data.keys())
    success_rates = [attack_data[t].get("success_rate", 0) for t in types]
    avg_movements = [attack_data[t].get("avg_movement", 0) for t in types]

    # Sort by success rate
    sorted_idx = np.argsort(success_rates)[::-1]
    types = [types[i] for i in sorted_idx]
    success_rates = [success_rates[i] for i in sorted_idx]
    avg_movements = [avg_movements[i] for i in sorted_idx]

    # Success rate
    ax = axes[0]
    colors = cm.RdYlGn(np.array(success_rates))
    bars = ax.barh(types, success_rates, color=colors, edgecolor='black')
    ax.set_xlabel("Success Rate")
    ax.set_title("Attack Success Rate by Type")
    ax.set_xlim(0, 1)
    for bar, sr in zip(bars, success_rates):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{sr:.1%}', va='center')

    # Average movement
    ax = axes[1]
    colors = cm.Blues(np.array(avg_movements) / max(avg_movements))
    bars = ax.barh(types, avg_movements, color=colors, edgecolor='black')
    ax.set_xlabel("Average Position Movement")
    ax.set_title("Average Movement by Attack Type")
    for bar, am in zip(bars, avg_movements):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{am:.3f}', va='center')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_attack_effectiveness.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 03_attack_effectiveness.png")

# ============ 4. BY CATEGORY ANALYSIS ============
def plot_category_analysis():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    cat_data = stats["by_category"]
    categories = list(cat_data.keys())
    counts = [cat_data[c]["count"] for c in categories]
    resistances = [cat_data[c]["avg_resistance"] for c in categories]
    beliefs = [cat_data[c]["avg_beliefs"] for c in categories]
    persuadabilities = [cat_data[c]["avg_persuadability"] for c in categories]

    # Count
    ax = axes[0, 0]
    colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))
    ax.pie(counts, labels=categories, autopct='%1.0f%%', colors=colors, startangle=90)
    ax.set_title("Persona Distribution by Category")

    # Resistance
    ax = axes[0, 1]
    bars = ax.bar(categories, resistances, color='#e74c3c', edgecolor='black')
    ax.set_ylabel("Avg Resistance")
    ax.set_title("Average Resistance by Category")
    ax.set_ylim(0, 1)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    for bar, r in zip(bars, resistances):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{r:.2f}', ha='center', fontsize=9)

    # Beliefs
    ax = axes[1, 0]
    bars = ax.bar(categories, beliefs, color='#3498db', edgecolor='black')
    ax.set_ylabel("Avg Belief Count")
    ax.set_title("Average Belief Network Size by Category")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    for bar, b in zip(bars, beliefs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{b:.1f}', ha='center', fontsize=9)

    # Persuadability
    ax = axes[1, 1]
    bars = ax.bar(categories, persuadabilities, color='#2ecc71', edgecolor='black')
    ax.set_ylabel("Avg Persuadability")
    ax.set_title("Average Persuadability by Category")
    ax.set_ylim(0, max(persuadabilities) * 1.2)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    for bar, p in zip(bars, persuadabilities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{p:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_category_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 04_category_analysis.png")

# ============ 5. BY STANCE ANALYSIS ============
def plot_stance_analysis():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    stance_data = stats["by_stance"]
    stances = list(stance_data.keys())
    counts = [stance_data[s]["count"] for s in stances]
    resistances = [stance_data[s]["avg_resistance"] for s in stances]
    persuadabilities = [stance_data[s]["avg_persuadability"] for s in stances]

    colors = {'pro_safety': '#2ecc71', 'accelerationist': '#e74c3c', 'moderate': '#f39c12',
              'pro_industry': '#3498db', 'doomer': '#9b59b6'}
    bar_colors = [colors.get(s, '#95a5a6') for s in stances]

    # Count
    ax = axes[0]
    ax.pie(counts, labels=stances, autopct='%1.0f%%', colors=bar_colors, startangle=90)
    ax.set_title("Distribution by Stance")

    # Resistance
    ax = axes[1]
    bars = ax.bar(stances, resistances, color=bar_colors, edgecolor='black')
    ax.set_ylabel("Avg Resistance")
    ax.set_title("Resistance by Stance")
    ax.set_ylim(0, 1)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Persuadability
    ax = axes[2]
    bars = ax.bar(stances, persuadabilities, color=bar_colors, edgecolor='black')
    ax.set_ylabel("Avg Persuadability")
    ax.set_title("Persuadability by Stance")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_stance_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 05_stance_analysis.png")

# ============ 6. BELIEF TYPE DISTRIBUTION ============
def plot_belief_distribution():
    fig, ax = plt.subplots(figsize=(14, 8))

    # Aggregate belief types across all personas
    belief_type_counts = {}
    for p in personas:
        for bid, b in p["beliefs"].items():
            btype = b["type"]
            if btype not in belief_type_counts:
                belief_type_counts[btype] = 0
            belief_type_counts[btype] += 1

    types = list(belief_type_counts.keys())
    counts = [belief_type_counts[t] for t in types]

    colors = plt.cm.Set3(np.linspace(0, 1, len(types)))

    bars = ax.bar(types, counts, color=colors, edgecolor='black')
    ax.set_xlabel("Belief Type")
    ax.set_ylabel("Total Count Across All Personas")
    ax.set_title("Belief Type Distribution Across All Personas")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(c), ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_belief_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 06_belief_distribution.png")

# ============ 7. KEYSTONE ANALYSIS ============
def plot_keystone_analysis():
    fig, ax = plt.subplots(figsize=(16, 10))

    # Get all keystones with scores
    all_keystones = []
    for p in personas:
        for k in p["keystones"]:
            all_keystones.append({
                "persona": p["name"],
                "belief": k["belief_id"],
                "score": k["cascade_score"],
            })

    # Sort and take top 30
    all_keystones = sorted(all_keystones, key=lambda x: -x["score"])[:30]

    labels = [f"{k['persona'][:15]}: {k['belief'][:20]}" for k in all_keystones]
    scores = [k["score"] for k in all_keystones]

    colors = cm.Reds(np.array(scores) / max(scores))

    y = np.arange(len(labels))
    bars = ax.barh(y, scores, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Cascade Score")
    ax.set_title("Top 30 Keystone Beliefs (Highest Cascade Potential)")

    for bar, s in zip(bars, scores):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f'{s:.1f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "07_keystone_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 07_keystone_analysis.png")

# ============ 8. COGNITIVE DISSONANCE HEATMAP ============
def plot_dissonance_heatmap():
    fig, ax = plt.subplots(figsize=(14, 10))

    # Collect dissonance data
    persona_names = [p["name"] for p in personas]
    dissonance_counts = [len(p["cognitive_dissonances"]) for p in personas]
    max_dissonance_scores = [
        max([d["score"] for d in p["cognitive_dissonances"]], default=0)
        for p in personas
    ]

    # Sort by max score
    sorted_idx = np.argsort(max_dissonance_scores)[::-1]
    persona_names = [persona_names[i] for i in sorted_idx]
    max_dissonance_scores = [max_dissonance_scores[i] for i in sorted_idx]

    y = np.arange(len(persona_names))
    colors = cm.OrRd(np.array(max_dissonance_scores) / max(max_dissonance_scores + [0.01]))

    bars = ax.barh(y, max_dissonance_scores, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(persona_names, fontsize=9)
    ax.set_xlabel("Max Cognitive Dissonance Score")
    ax.set_title("Cognitive Dissonance Vulnerability by Persona")

    for bar, s in zip(bars, max_dissonance_scores):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{s:.3f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "08_dissonance_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 08_dissonance_heatmap.png")

# ============ 9. PRESSURE POINT INTENSITY ============
def plot_pressure_intensity():
    fig, ax = plt.subplots(figsize=(14, 10))

    # Aggregate pressure intensities
    pressure_types = ["reputation", "legacy", "competitive", "isolation",
                      "cognitive_dissonance", "fear", "ego", "loyalty", "economic", "temporal"]

    matrix = []
    persona_names = []
    for p in personas[:20]:  # Top 20
        persona_names.append(p["name"])
        row = []
        for ptype in pressure_types:
            intensity = 0
            for pp in p["pressure_points"]:
                if pp["type"] == ptype:
                    intensity = pp["intensity"]
                    break
            row.append(intensity)
        matrix.append(row)

    matrix = np.array(matrix)

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(np.arange(len(pressure_types)))
    ax.set_yticks(np.arange(len(persona_names)))
    ax.set_xticklabels(pressure_types, rotation=45, ha='right')
    ax.set_yticklabels(persona_names, fontsize=9)

    ax.set_title("Pressure Point Intensity Matrix (Top 20 Personas)")
    plt.colorbar(im, label='Intensity')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "09_pressure_intensity.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 09_pressure_intensity.png")

# ============ 10. PSYCHOLOGICAL STATE DISTRIBUTION ============
def plot_psych_states():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Count states
    state_counts = {}
    for p in personas:
        state = p["psych_state"]
        state_counts[state] = state_counts.get(state, 0) + 1

    states = list(state_counts.keys())
    counts = [state_counts[s] for s in states]

    colors = {'defensive': '#e74c3c', 'guarded': '#f39c12', 'neutral': '#3498db',
              'receptive': '#2ecc71', 'vulnerable': '#9b59b6', 'confused': '#95a5a6',
              'anxious': '#e67e22'}
    bar_colors = [colors.get(s, '#95a5a6') for s in states]

    # Pie chart
    ax = axes[0]
    ax.pie(counts, labels=states, autopct='%1.0f%%', colors=bar_colors, startangle=90)
    ax.set_title("Psychological State Distribution")

    # State vs resistance scatter
    ax = axes[1]
    state_order = {'defensive': 0.1, 'guarded': 0.3, 'neutral': 0.5,
                   'receptive': 0.7, 'vulnerable': 0.9}
    x = [state_order.get(p["psych_state"], 0.5) for p in personas]
    y = [p["resistance"] for p in personas]
    c = [p["persuadability"] for p in personas]

    scatter = ax.scatter(x, y, c=c, cmap='RdYlGn', s=100, edgecolors='black', linewidth=0.5)
    ax.set_xlabel("Psych State (defensive → vulnerable)")
    ax.set_ylabel("Resistance")
    ax.set_title("Psychological State vs Resistance")
    plt.colorbar(scatter, label='Persuadability')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "10_psych_states.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 10_psych_states.png")

# ============ 11. CAMPAIGN ROUND-BY-ROUND ============
def plot_campaign_rounds():
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    # Select 9 diverse campaigns
    selected = campaigns[::4][:9]

    for ax, campaign in zip(axes, selected):
        rounds = campaign["rounds"]
        positions = [r["position_before"] for r in rounds] + [rounds[-1]["position_after"]]
        x = range(len(positions))

        colors = ['#2ecc71' if r["success"] else '#e74c3c' for r in rounds]

        ax.plot(x, positions, 'b-', linewidth=2, alpha=0.7)
        ax.scatter(x[:-1], positions[:-1], c=colors, s=80, zorder=3, edgecolors='black')
        ax.scatter(x[-1], positions[-1], c='blue', s=100, marker='*', zorder=4)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylim(-1, 1)
        ax.set_title(f"{campaign['name'][:20]}", fontsize=10)
        ax.set_xlabel("Round")
        ax.set_ylabel("Position")

    plt.suptitle("Campaign Trajectories: Position Over Rounds", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "11_campaign_rounds.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 11_campaign_rounds.png")

# ============ 12. BELIEF PROTECTION LEVELS ============
def plot_protection_levels():
    fig, ax = plt.subplots(figsize=(14, 8))

    # Aggregate by belief type
    type_protections = {}
    for p in personas:
        for bid, b in p["beliefs"].items():
            btype = b["type"]
            if btype not in type_protections:
                type_protections[btype] = []
            type_protections[btype].append(b["protection_level"])

    types = list(type_protections.keys())
    means = [np.mean(type_protections[t]) for t in types]
    stds = [np.std(type_protections[t]) for t in types]

    # Sort by mean
    sorted_idx = np.argsort(means)[::-1]
    types = [types[i] for i in sorted_idx]
    means = [means[i] for i in sorted_idx]
    stds = [stds[i] for i in sorted_idx]

    colors = cm.RdYlGn(np.array(means) / max(means))

    bars = ax.bar(types, means, yerr=stds, color=colors, edgecolor='black', capsize=5)
    ax.set_xlabel("Belief Type")
    ax.set_ylabel("Average Protection Level")
    ax.set_title("Belief Protection Levels by Type")
    ax.set_ylim(0, 1)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{m:.2f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "12_protection_levels.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 12_protection_levels.png")

# ============ 13. RESISTANCE VS PERSUADABILITY SCATTER ============
def plot_resistance_scatter():
    fig, ax = plt.subplots(figsize=(12, 10))

    x = [p["resistance"] for p in personas]
    y = [p["persuadability"] for p in personas]
    names = [p["name"] for p in personas]
    categories = [p["category"] for p in personas]

    cat_colors = {'tech_leader': '#3498db', 'us_politician': '#e74c3c',
                  'world_leader': '#2ecc71', 'government_official': '#f39c12',
                  'researcher': '#9b59b6', 'other': '#95a5a6'}
    colors = [cat_colors.get(c, '#95a5a6') for c in categories]

    scatter = ax.scatter(x, y, c=colors, s=150, edgecolors='black', linewidth=0.5, alpha=0.8)

    # Label points
    for i, name in enumerate(names):
        ax.annotate(name[:12], (x[i], y[i]), fontsize=7, alpha=0.7,
                   xytext=(5, 5), textcoords='offset points')

    ax.set_xlabel("Resistance")
    ax.set_ylabel("Persuadability")
    ax.set_title("Resistance vs Persuadability by Category")

    # Legend
    patches = [mpatches.Patch(color=c, label=cat) for cat, c in cat_colors.items()]
    ax.legend(handles=patches, loc='upper right')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "13_resistance_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 13_resistance_scatter.png")

# ============ 14. BELIEF NETWORK SIZE HISTOGRAM ============
def plot_network_size():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sizes = [p["total_beliefs"] for p in personas]

    # Histogram
    ax = axes[0]
    ax.hist(sizes, bins=10, color='#3498db', edgecolor='black')
    ax.set_xlabel("Number of Beliefs")
    ax.set_ylabel("Count")
    ax.set_title("Belief Network Size Distribution")
    ax.axvline(np.mean(sizes), color='red', linestyle='--', label=f'Mean: {np.mean(sizes):.1f}')
    ax.legend()

    # By category
    ax = axes[1]
    categories = list(set(p["category"] for p in personas))
    cat_sizes = {c: [] for c in categories}
    for p in personas:
        cat_sizes[p["category"]].append(p["total_beliefs"])

    cat_means = [np.mean(cat_sizes[c]) for c in categories]
    bars = ax.bar(categories, cat_means, color='#2ecc71', edgecolor='black')
    ax.set_xlabel("Category")
    ax.set_ylabel("Average Belief Count")
    ax.set_title("Average Network Size by Category")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "14_network_size.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 14_network_size.png")

# ============ 15. COMPREHENSIVE SUMMARY HEATMAP ============
def plot_summary_heatmap():
    fig, ax = plt.subplots(figsize=(16, 12))

    metrics = ["resistance", "persuadability", "total_beliefs", "identity_flexibility",
               "social_proof_threshold", "isolation_threshold"]

    # Get data for all personas
    matrix = []
    names = []
    for p in personas:
        names.append(p["name"])
        row = [p.get(m, 0) for m in metrics]
        matrix.append(row)

    matrix = np.array(matrix)

    # Normalize each column
    matrix_norm = (matrix - matrix.min(axis=0)) / (matrix.max(axis=0) - matrix.min(axis=0) + 1e-10)

    im = ax.imshow(matrix_norm, cmap='RdYlGn_r', aspect='auto')

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.set_yticklabels(names, fontsize=8)

    ax.set_title("Comprehensive Persona Vulnerability Matrix")
    plt.colorbar(im, label='Normalized Value (higher = more vulnerable)')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "15_summary_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 15_summary_heatmap.png")

# ============ 16. ATTACK SUCCESS BY PERSONA ============
def plot_attack_success_by_persona():
    fig, ax = plt.subplots(figsize=(16, 10))

    # Calculate success rate per persona
    persona_success = {}
    for campaign in campaigns:
        pid = campaign["persona_id"]
        successes = sum(1 for r in campaign["rounds"] if r["success"])
        total = len(campaign["rounds"])
        persona_success[campaign["name"]] = successes / total if total > 0 else 0

    names = list(persona_success.keys())
    rates = list(persona_success.values())

    # Sort
    sorted_idx = np.argsort(rates)[::-1]
    names = [names[i] for i in sorted_idx]
    rates = [rates[i] for i in sorted_idx]

    colors = cm.RdYlGn(np.array(rates))

    y = np.arange(len(names))
    bars = ax.barh(y, rates, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Attack Success Rate")
    ax.set_title("Attack Success Rate by Persona")
    ax.set_xlim(0, 1)

    for bar, r in zip(bars, rates):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{r:.1%}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "16_attack_success_persona.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 16_attack_success_persona.png")

# ============ 17. SACRED VALUES WORDCLOUD-STYLE ============
def plot_sacred_values():
    fig, ax = plt.subplots(figsize=(14, 8))

    # Collect all sacred values
    all_values = []
    for p in personas:
        all_values.extend(p.get("sacred_values", []))

    # Count occurrences
    value_counts = {}
    for v in all_values:
        v_clean = v[:30]
        value_counts[v_clean] = value_counts.get(v_clean, 0) + 1

    # Sort and take top 20
    sorted_values = sorted(value_counts.items(), key=lambda x: -x[1])[:20]
    values = [v[0] for v in sorted_values]
    counts = [v[1] for v in sorted_values]

    colors = cm.Purples(np.array(counts) / max(counts))

    y = np.arange(len(values))
    bars = ax.barh(y, counts, color=colors, edgecolor='black')
    ax.set_yticks(y)
    ax.set_yticklabels(values, fontsize=9)
    ax.set_xlabel("Frequency")
    ax.set_title("Most Common Sacred Values Across Personas")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "17_sacred_values.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 17_sacred_values.png")

# ============ 18. BELIEF CONFIDENCE DISTRIBUTION ============
def plot_confidence_distribution():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Collect confidences by type
    type_confidences = {}
    for p in personas:
        for bid, b in p["beliefs"].items():
            btype = b["type"]
            if btype not in type_confidences:
                type_confidences[btype] = []
            type_confidences[btype].append(b["confidence"])

    types = ["core_identity", "sacred_value", "empirical", "strategic"]
    colors = ['#e74c3c', '#9b59b6', '#3498db', '#2ecc71']

    for ax, btype, color in zip(axes.flatten(), types, colors):
        if btype in type_confidences:
            data = type_confidences[btype]
            ax.hist(data, bins=15, color=color, edgecolor='black', alpha=0.8)
            ax.axvline(np.mean(data), color='black', linestyle='--',
                      label=f'Mean: {np.mean(data):.2f}')
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Count")
            ax.set_title(f"{btype.replace('_', ' ').title()} Confidence Distribution")
            ax.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "18_confidence_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 18_confidence_distribution.png")

# ============ 19. FINAL POSITION DISTRIBUTION ============
def plot_final_positions():
    fig, ax = plt.subplots(figsize=(12, 6))

    positions = [c["final_position"] for c in campaigns]
    names = [c["name"] for c in campaigns]

    colors = ['#2ecc71' if p > 0 else '#e74c3c' for p in positions]

    x = np.arange(len(names))
    bars = ax.bar(x, positions, color=colors, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_ylabel("Final Position")
    ax.set_title("Final Position After Campaign")
    ax.axhline(y=0, color='black', linewidth=1)
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Breakthrough')
    ax.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5, label='Entrenched')
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "19_final_positions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 19_final_positions.png")

# ============ 20. CORRELATION MATRIX ============
def plot_correlation_matrix():
    fig, ax = plt.subplots(figsize=(10, 8))

    # Build correlation data
    metrics = ["resistance", "persuadability", "total_beliefs", "identity_flexibility"]

    matrix = []
    for p in personas:
        row = [p.get(m, 0) for m in metrics]
        matrix.append(row)

    matrix = np.array(matrix)
    corr = np.corrcoef(matrix.T)

    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.set_yticklabels(metrics)

    # Add correlation values
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center',
                   color='white' if abs(corr[i, j]) > 0.5 else 'black')

    ax.set_title("Correlation Matrix: Persona Metrics")
    plt.colorbar(im, label='Correlation')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "20_correlation_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 20_correlation_matrix.png")

# ============ RUN ALL ============
if __name__ == "__main__":
    print("\nGenerating comprehensive figures...\n")

    plot_vulnerability_ranking()
    plot_campaign_effectiveness()
    plot_attack_effectiveness()
    plot_category_analysis()
    plot_stance_analysis()
    plot_belief_distribution()
    plot_keystone_analysis()
    plot_dissonance_heatmap()
    plot_pressure_intensity()
    plot_psych_states()
    plot_campaign_rounds()
    plot_protection_levels()
    plot_resistance_scatter()
    plot_network_size()
    plot_summary_heatmap()
    plot_attack_success_by_persona()
    plot_sacred_values()
    plot_confidence_distribution()
    plot_final_positions()
    plot_correlation_matrix()

    print(f"\nDone! Generated {len(list(FIGURES_DIR.glob('*.png')))} figures in {FIGURES_DIR}/")
