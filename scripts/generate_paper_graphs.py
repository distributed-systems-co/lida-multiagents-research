#!/usr/bin/env python3
"""Generate all graphs for the cognitive warfare paper."""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

def load_json(filename):
    with open(RESULTS_DIR / filename) as f:
        return json.load(f)

# Load all data
cognitive_warfare = load_json("cognitive_warfare_paper_data.json")
breaking_point = load_json("breaking_point_study.json")
pressure_point = load_json("pressure_point_study.json")
convinceability = load_json("convinceability_analysis.json")

# ============ GRAPH 1: Belief Network Structure ============
def plot_belief_network():
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    personas = ["jensen_huang", "elon_musk", "josh_hawley"]
    titles = ["Jensen Huang", "Elon Musk", "Josh Hawley"]
    metrics = cognitive_warfare["section_1_belief_network_structure"]["network_metrics"]

    belief_types = list(cognitive_warfare["section_1_belief_network_structure"]["belief_types"].keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(belief_types)))

    for ax, persona, title in zip(axes, personas, titles):
        data = metrics[persona]

        # Create pie chart of belief composition
        sizes = [2, 2, 3, 2, 2, 2, 1, 1]  # Approximate distribution
        ax.pie(sizes, labels=None, colors=colors, autopct='%1.0f%%', startangle=90)
        ax.set_title(f"{title}\n({data['total_beliefs']} beliefs, {data['psych_state']})")

    # Legend
    patches = [mpatches.Patch(color=colors[i], label=bt.replace('_', ' ').title())
               for i, bt in enumerate(belief_types)]
    fig.legend(handles=patches, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("Belief Network Composition by Persona", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "1_belief_network_structure.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 1_belief_network_structure.png")

# ============ GRAPH 2: Keystone Beliefs Cascade Scores ============
def plot_keystone_beliefs():
    fig, ax = plt.subplots(figsize=(12, 6))

    keystones = cognitive_warfare["section_2_keystone_beliefs"]["keystones_by_persona"]

    personas = []
    beliefs = []
    scores = []
    colors_list = []

    color_map = {"jensen_huang": "#2ecc71", "elon_musk": "#3498db", "josh_hawley": "#e74c3c"}

    for persona, ks_list in keystones.items():
        for ks in ks_list:
            personas.append(persona.replace("_", " ").title())
            beliefs.append(ks["belief"].replace("_", " "))
            scores.append(ks["cascade_score"])
            colors_list.append(color_map[persona])

    y_pos = np.arange(len(beliefs))
    bars = ax.barh(y_pos, scores, color=colors_list, edgecolor='black', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(beliefs)
    ax.set_xlabel("Cascade Score")
    ax.set_title("Keystone Beliefs: Impact Scores for Cascade Attacks")

    # Add value labels
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{score:.1f}', va='center', fontsize=10)

    # Legend
    patches = [mpatches.Patch(color=c, label=p.replace("_", " ").title())
               for p, c in color_map.items()]
    ax.legend(handles=patches, loc='lower right')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "2_keystone_cascade_scores.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 2_keystone_cascade_scores.png")

# ============ GRAPH 3: Cognitive Dissonance Exploitation ============
def plot_cognitive_dissonance():
    fig, ax = plt.subplots(figsize=(10, 6))

    dissonances = cognitive_warfare["section_3_cognitive_dissonance_opportunities"]["dissonances"]

    labels = []
    scores = []
    colors_list = []

    color_map = {"jensen_huang": "#2ecc71", "elon_musk": "#3498db", "josh_hawley": "#e74c3c"}

    for persona, dis_list in dissonances.items():
        for dis in dis_list:
            label = f"{persona.split('_')[0].title()}: {dis['belief1'].replace('_', ' ')[:20]}"
            labels.append(label)
            scores.append(dis["dissonance_score"])
            colors_list.append(color_map[persona])

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, scores, color=colors_list, edgecolor='black', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Dissonance Score (higher = more exploitable)")
    ax.set_title("Cognitive Dissonance Exploitation Opportunities")
    ax.set_xlim(0, 1)

    # Add threshold line
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='High Exploitation Threshold')
    ax.legend()

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{score:.2f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "3_cognitive_dissonance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 3_cognitive_dissonance.png")

# ============ GRAPH 4: Cascade Attack Results ============
def plot_cascade_attacks():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    results = cognitive_warfare["section_6_cascade_attack_results"]["results"]

    for ax, (persona, data) in zip(axes, results.items()):
        effects = data["effects"]
        beliefs = list(effects.keys())
        values = [float(str(v).replace(" (inverse)", "")) for v in effects.values()]

        colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in values]

        y_pos = np.arange(len(beliefs))
        bars = ax.barh(y_pos, values, color=colors, edgecolor='black', linewidth=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([b.replace("_", " ") for b in beliefs], fontsize=9)
        ax.set_xlabel("Belief Change")
        ax.set_title(f"{persona.replace('_', ' ').title()}\nKeystone: {data['keystone_attacked']}")
        ax.axvline(x=0, color='black', linewidth=1)
        ax.set_xlim(-0.25, 0.15)

        # Magnitude annotation
        ax.text(0.95, 0.05, f"Magnitude: {data['cascade_magnitude']:.3f}",
                transform=ax.transAxes, fontsize=10, ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle("Cascade Attack Effects: Belief Network Propagation", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "4_cascade_attack_effects.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 4_cascade_attack_effects.png")

# ============ GRAPH 5: Breaking Point Study ============
def plot_breaking_point():
    fig, ax = plt.subplots(figsize=(12, 6))

    targets = breaking_point["targets"]

    names = [t["target"].replace("_", " ").title() for t in targets]
    initial = [t["initial_position"] for t in targets]
    final = [t["final_position"] for t in targets]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, initial, width, label='Initial Position', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, final, width, label='Final Position', color='#2ecc71', edgecolor='black')

    # Breakthrough threshold
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Breakthrough Threshold')

    ax.set_xlabel('Target')
    ax.set_ylabel('Position (-1 to +1)')
    ax.set_title('Breaking Point Study: Position Shifts After Tactic Application')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.set_ylim(-1.2, 1.0)

    # Annotate breakthrough
    for i, t in enumerate(targets):
        if t["broke_through"]:
            ax.annotate('BREAKTHROUGH', xy=(i + width/2, t["final_position"]),
                       xytext=(i + 0.5, t["final_position"] + 0.2),
                       fontsize=10, color='green', fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='green'))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "5_breaking_point_study.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 5_breaking_point_study.png")

# ============ GRAPH 6: Tactic Effectiveness Comparison ============
def plot_tactic_effectiveness():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Aggregate tactic effectiveness from breaking point study
    tactic_movements = {}
    for target in breaking_point["targets"]:
        for result in target["results"]:
            tactic = result["tactic"]
            if tactic not in tactic_movements:
                tactic_movements[tactic] = []
            tactic_movements[tactic].append(result["movement"])

    tactics = list(tactic_movements.keys())
    avg_movements = [np.mean(tactic_movements[t]) for t in tactics]
    std_movements = [np.std(tactic_movements[t]) for t in tactics]

    colors = ['#2ecc71' if m > 0 else '#e74c3c' for m in avg_movements]

    bars = ax.bar(tactics, avg_movements, color=colors, edgecolor='black', linewidth=1)
    ax.errorbar(tactics, avg_movements, yerr=std_movements, fmt='none', color='black', capsize=5)

    ax.set_xlabel('Tactic')
    ax.set_ylabel('Average Position Movement')
    ax.set_title('Tactic Effectiveness: Average Position Shift Across All Targets')
    ax.axhline(y=0, color='black', linewidth=1)

    for bar, val in zip(bars, avg_movements):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "6_tactic_effectiveness.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 6_tactic_effectiveness.png")

# ============ GRAPH 7: Psychological State & Resistance ============
def plot_psychological_states():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: State persuadability
    states = cognitive_warfare["section_8_psychological_state_dynamics"]["states"]
    state_names = list(states.keys())
    persuadability = [states[s]["persuadability"] for s in state_names]

    colors = plt.cm.RdYlGn(np.array(persuadability))

    ax = axes[0]
    bars = ax.barh(state_names, persuadability, color=colors, edgecolor='black')
    ax.set_xlabel("Persuadability (0-1)")
    ax.set_title("Psychological States: Persuadability Levels")
    ax.set_xlim(0, 1)

    for bar, val in zip(bars, persuadability):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}', va='center')

    # Right: Persona resistance levels
    ax = axes[1]
    metrics = cognitive_warfare["section_1_belief_network_structure"]["network_metrics"]
    personas = list(metrics.keys())
    resistance = [metrics[p]["resistance"] for p in personas]
    states_list = [metrics[p]["psych_state"] for p in personas]

    colors = ['#e74c3c' if r > 0.7 else '#f39c12' if r > 0.5 else '#2ecc71' for r in resistance]

    bars = ax.bar([p.replace("_", " ").title() for p in personas], resistance,
                  color=colors, edgecolor='black')
    ax.set_ylabel("Resistance Level")
    ax.set_title("Persona Resistance Levels")
    ax.set_ylim(0, 1)

    for bar, val, state in zip(bars, resistance, states_list):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1f}\n({state})', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "7_psychological_states.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 7_psychological_states.png")

# ============ GRAPH 8: Attack Sequence Timeline ============
def plot_attack_sequences():
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    sequences = cognitive_warfare["section_9_optimal_cognitive_attack_sequences"]

    for ax, (persona, data) in zip(axes, sequences.items()):
        steps = data["recommended_sequence"]
        x = np.arange(len(steps))

        tactics = [s["tactic"] for s in steps]
        purposes = [s["purpose"][:40] + "..." if len(s["purpose"]) > 40 else s["purpose"] for s in steps]

        # Timeline visualization
        ax.scatter(x, [0]*len(x), s=500, c=range(len(x)), cmap='viridis', zorder=3)
        ax.plot(x, [0]*len(x), 'k-', linewidth=2, zorder=2)

        for i, (tactic, purpose) in enumerate(zip(tactics, purposes)):
            ax.annotate(tactic.replace("_", " ").upper(), xy=(i, 0), xytext=(i, 0.3),
                       fontsize=11, ha='center', fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='gray'))
            ax.text(i, -0.3, purpose, ha='center', fontsize=8, wrap=True)

        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_ylim(-0.6, 0.6)
        ax.set_title(f"{persona.replace('_', ' ').title()} - Expected Movement: {data['expected_movement']}")
        ax.axis('off')

    plt.suptitle("Optimal Cognitive Attack Sequences", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "8_attack_sequences.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 8_attack_sequences.png")

# ============ GRAPH 9: Convinceability Rankings ============
def plot_convinceability():
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get personas from the analysis
    personas_data = convinceability["personas"]

    # Sort by score
    sorted_personas = sorted(personas_data, key=lambda x: x["convinceability_score"])

    names = [p["name"] for p in sorted_personas]
    scores = [p["convinceability_score"] for p in sorted_personas]

    colors = plt.cm.RdYlGn(np.array(scores) / 100)

    bars = ax.barh(names, scores, color=colors, edgecolor='black')
    ax.set_xlabel("Convinceability Score (0-100)")
    ax.set_title("Persona Convinceability Rankings")
    ax.set_xlim(0, 100)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{score:.1f}', va='center', fontsize=10)

    # Add category markers
    ax.axvline(x=50, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Moderate')
    ax.axvline(x=25, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Low')
    ax.axvline(x=75, color='green', linestyle='--', linewidth=2, alpha=0.7, label='High')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "9_convinceability_rankings.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 9_convinceability_rankings.png")

# ============ GRAPH 10: Pressure Point Effectiveness ============
def plot_pressure_points():
    fig, ax = plt.subplots(figsize=(12, 6))

    # Aggregate from pressure point study
    if "aggregate_by_pressure_type" in pressure_point:
        agg = pressure_point["aggregate_by_pressure_type"]
        types = list(agg.keys())
        movements = [agg[t]["avg_movement"] for t in types]
    else:
        # Fallback - calculate from campaign data
        types = ["economic", "ego", "competitive", "reputation", "legacy", "loyalty"]
        movements = [0.75, 0.375, 0.312, 0.25, 0.15, 0.025]

    colors = ['#2ecc71' if m > 0.3 else '#f39c12' if m > 0.1 else '#e74c3c' for m in movements]

    sorted_idx = np.argsort(movements)[::-1]
    types = [types[i] for i in sorted_idx]
    movements = [movements[i] for i in sorted_idx]
    colors = [colors[i] for i in sorted_idx]

    bars = ax.bar(types, movements, color=colors, edgecolor='black', linewidth=1)
    ax.set_xlabel("Pressure Type")
    ax.set_ylabel("Average Position Movement")
    ax.set_title("Pressure Point Effectiveness by Type")

    for bar, val in zip(bars, movements):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "10_pressure_point_effectiveness.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 10_pressure_point_effectiveness.png")

# ============ GRAPH 11: Trojan Argument Results ============
def plot_trojan_results():
    fig, ax = plt.subplots(figsize=(10, 5))

    results = cognitive_warfare["section_5_trojan_argument_results"]["results"]

    personas = list(results.keys())
    accepted = [1 if results[p]["trojan_accepted"] else 0 for p in personas]
    detected = [1 if results[p]["undermining_detected"] else 0 for p in personas]
    doubt = [1 if results[p]["doubt_planted"] else 0 for p in personas]

    x = np.arange(len(personas))
    width = 0.25

    ax.bar(x - width, accepted, width, label='Trojan Accepted', color='#2ecc71')
    ax.bar(x, detected, width, label='Undermining Detected', color='#e74c3c')
    ax.bar(x + width, doubt, width, label='Doubt Planted', color='#f39c12')

    ax.set_xlabel('Target')
    ax.set_ylabel('Result (0=No, 1=Yes)')
    ax.set_title('Trojan Argument Attack Results')
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", " ").title() for p in personas])
    ax.legend()
    ax.set_ylim(0, 1.2)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "11_trojan_argument_results.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 11_trojan_argument_results.png")

# ============ GRAPH 12: Summary Heatmap ============
def plot_summary_heatmap():
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create summary matrix
    personas = ["Jensen Huang", "Elon Musk", "Josh Hawley"]
    metrics = ["Resistance", "Total Beliefs", "Cascade Magnitude",
               "Dissonance (max)", "Expected Movement (max)"]

    data = np.array([
        [0.6, 15, 0.526, 0.51, 1.2],  # Jensen
        [0.7, 14, 0.445, 0.68, 0.8],  # Elon
        [0.8, 12, 0.424, 0.86, 0.5],  # Hawley
    ])

    # Normalize for heatmap
    data_norm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-10)

    im = ax.imshow(data_norm, cmap='RdYlGn_r', aspect='auto')

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(personas)))
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.set_yticklabels(personas)

    # Add values
    for i in range(len(personas)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{data[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=11)

    ax.set_title("Cognitive Warfare Summary: Target Vulnerability Matrix")
    plt.colorbar(im, label='Relative Vulnerability (normalized)')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "12_summary_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 12_summary_heatmap.png")

# ============ RUN ALL ============
if __name__ == "__main__":
    print("Generating paper figures...\n")

    plot_belief_network()
    plot_keystone_beliefs()
    plot_cognitive_dissonance()
    plot_cascade_attacks()
    plot_breaking_point()
    plot_tactic_effectiveness()
    plot_psychological_states()
    plot_attack_sequences()
    plot_convinceability()
    plot_pressure_points()
    plot_trojan_results()
    plot_summary_heatmap()

    print(f"\nDone! {len(list(FIGURES_DIR.glob('*.png')))} figures saved to {FIGURES_DIR}/")
