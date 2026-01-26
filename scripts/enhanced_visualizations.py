#!/usr/bin/env python3
"""Enhanced visualizations with fixes and new sophisticated plots."""
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.sankey import Sankey
import matplotlib.cm as cm
from matplotlib.colors import Normalize

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.dpi'] = 150

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# Load data
with open("results/full_simulation_data.json") as f:
    data = json.load(f)

personas = data["personas"]
campaigns = data["campaigns"]

print(f"Loaded {len(personas)} personas, {len(campaigns)} campaigns")


def plot_enhanced_belief_networks():
    """4 personas with clearer network graphs."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    # Color by type - define at function level
    type_colors = {
        "sacred_value": "#e74c3c",
        "core_identity": "#9b59b6",
        "empirical": "#3498db",
        "causal": "#2ecc71",
        "normative": "#f39c12",
        "strategic": "#1abc9c",
        "social": "#e91e63",
        "predictive": "#607d8b"
    }

    # Select 4 diverse personas
    selected = ["elon_musk", "xi_jinping", "chuck_schumer", "jensen_huang"]
    selected_personas = [p for p in personas if p["persona_id"] in selected]

    for idx, persona in enumerate(selected_personas[:4]):
        ax = axes[idx]
        beliefs = persona.get("belief_network", {}).get("beliefs", [])

        if not beliefs:
            ax.text(0.5, 0.5, "No belief data", ha='center', va='center')
            ax.set_title(persona["name"])
            continue

        # Create network layout
        n = len(beliefs)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        radius = 0.35

        # Position nodes in circle
        positions = {}
        for i, belief in enumerate(beliefs):
            x = 0.5 + radius * np.cos(angles[i])
            y = 0.5 + radius * np.sin(angles[i])
            positions[belief["id"]] = (x, y)

        # Draw edges
        for belief in beliefs:
            x1, y1 = positions[belief["id"]]
            for connected in belief.get("connected_to", []):
                if connected in positions:
                    x2, y2 = positions[connected]
                    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                               arrowprops=dict(arrowstyle="->", color="gray",
                                             alpha=0.4, lw=0.8))

        # Draw nodes
        for belief in beliefs:
            x, y = positions[belief["id"]]
            color = type_colors.get(belief.get("type", "empirical"), "#95a5a6")
            size = 800 + belief.get("strength", 0.5) * 600
            ax.scatter(x, y, s=size, c=color, alpha=0.8, edgecolors='black', linewidth=1.5, zorder=5)

            # Short label
            label = belief["id"].replace("_", "\n")[:15]
            ax.annotate(label, (x, y), fontsize=6, ha='center', va='center', zorder=6)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"{persona['name']}\n({len(beliefs)} beliefs)", fontsize=14, fontweight='bold')

    # Legend
    legend_elements = [mpatches.Patch(color=c, label=t.replace("_", " ").title())
                      for t, c in type_colors.items()]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10)

    plt.suptitle("Belief Network Structure: Key Personas", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    plt.savefig(FIGURES_DIR / "40_enhanced_belief_networks.png", bbox_inches='tight')
    plt.close()
    print("Generated: 40_enhanced_belief_networks.png")


def plot_enhanced_radar():
    """4 personas with clearer radar charts."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), subplot_kw=dict(polar=True))
    axes = axes.flatten()

    categories = ['Resistance', 'Persuadability', 'Belief\nCount', 'Identity\nFlux', 'Social\nProof']
    n_cats = len(categories)
    angles = np.linspace(0, 2*np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]

    # Select 4 key personas
    selected_ids = ["sam_altman", "elon_musk", "josh_hawley", "xi_jinping"]
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]

    for idx, pid in enumerate(selected_ids):
        persona = next((p for p in personas if p["persona_id"] == pid), None)
        if not persona:
            continue

        ax = axes[idx]

        # Get values (normalize to 0-1)
        values = [
            persona.get("resistance", 0.5),
            persona.get("persuadability", 0.1) * 3,  # Scale up
            len(persona.get("belief_network", {}).get("beliefs", [])) / 30,
            persona.get("identity_flux", 0.5),
            persona.get("social_proof_susceptibility", 0.5)
        ]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2.5, color=colors[idx], markersize=8)
        ax.fill(angles, values, alpha=0.25, color=colors[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title(persona["name"], fontsize=14, fontweight='bold', pad=20)

        # Add value annotations
        for angle, val, cat in zip(angles[:-1], values[:-1], categories):
            ax.annotate(f'{val:.2f}', xy=(angle, val), fontsize=9,
                       ha='center', va='bottom', color=colors[idx])

    plt.suptitle("Multi-Dimensional Vulnerability Profiles", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "41_enhanced_radar.png", bbox_inches='tight')
    plt.close()
    print("Generated: 41_enhanced_radar.png")


def plot_parallel_coordinates_enhanced():
    """Parallel coordinates with transparency and highlighting."""
    fig, ax = plt.subplots(figsize=(16, 10))

    metrics = ['Resistance', 'Persuadability', 'Beliefs', 'Identity Flux', 'Social Proof', 'Final Position']
    n_metrics = len(metrics)

    # Normalize data
    all_data = []
    for p in personas:
        campaign = next((c for c in campaigns if c["persona_id"] == p["persona_id"]), None)
        final_pos = campaign["final_position"] if campaign else 0

        row = [
            p.get("resistance", 0.5),
            p.get("persuadability", 0.1),
            len(p.get("belief_network", {}).get("beliefs", [])) / 30,
            p.get("identity_flux", 0.5),
            p.get("social_proof_susceptibility", 0.5),
            (final_pos + 1) / 2  # Normalize -1 to 1 → 0 to 1
        ]
        all_data.append((p["name"], p.get("category", "other"), row))

    # Category colors
    cat_colors = {
        "tech_leader": "#3498db",
        "us_politician": "#e74c3c",
        "world_leader": "#2ecc71",
        "government_official": "#9b59b6",
        "researcher": "#f39c12",
        "other": "#95a5a6"
    }

    # Plot all lines with transparency
    for name, cat, values in all_data:
        color = cat_colors.get(cat, "#95a5a6")
        ax.plot(range(n_metrics), values, alpha=0.3, color=color, linewidth=1)

    # Highlight extremes
    sorted_by_persuadability = sorted(all_data, key=lambda x: x[2][1], reverse=True)

    # Top 3 most persuadable
    for name, cat, values in sorted_by_persuadability[:3]:
        ax.plot(range(n_metrics), values, alpha=1.0, color='red', linewidth=3, label=f"High: {name}")

    # Bottom 3 least persuadable
    for name, cat, values in sorted_by_persuadability[-3:]:
        ax.plot(range(n_metrics), values, alpha=1.0, color='green', linewidth=3, label=f"Low: {name}")

    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylabel("Normalized Value", fontsize=12)
    ax.set_ylim(0, 1)

    # Category legend
    legend_elements = [mpatches.Patch(color=c, label=k.replace("_", " ").title(), alpha=0.6)
                      for k, c in cat_colors.items()]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    ax.set_title("Parallel Coordinates: All Personas Across 6 Metrics\n(Red=Most Persuadable, Green=Most Resistant)",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "42_enhanced_parallel.png", bbox_inches='tight')
    plt.close()
    print("Generated: 42_enhanced_parallel.png")


def plot_slope_chart():
    """Before/after position changes as slope chart."""
    fig, ax = plt.subplots(figsize=(12, 14))

    # Sort by total movement
    campaign_data = []
    for c in campaigns:
        start = c.get("trajectory", [0])[0] if c.get("trajectory") else 0
        end = c.get("final_position", 0)
        movement = end - start
        campaign_data.append({
            "name": c["persona_id"].replace("_", " ").title(),
            "start": start,
            "end": end,
            "movement": movement
        })

    campaign_data.sort(key=lambda x: x["movement"], reverse=True)

    # Plot slopes
    for i, c in enumerate(campaign_data):
        color = "green" if c["movement"] > 0 else "red" if c["movement"] < 0 else "gray"
        alpha = min(1.0, 0.3 + abs(c["movement"]))

        ax.plot([0, 1], [c["start"], c["end"]], color=color, alpha=alpha, linewidth=2)
        ax.scatter([0], [c["start"]], color=color, s=50, alpha=alpha, zorder=5)
        ax.scatter([1], [c["end"]], color=color, s=50, alpha=alpha, zorder=5)

        # Label right side
        ax.annotate(f"{c['name'][:20]} ({c['movement']:+.2f})",
                   xy=(1.02, c["end"]), fontsize=7, va='center')

    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlim(-0.1, 1.5)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Start Position", "Final Position"], fontsize=12)
    ax.set_ylabel("Position on Issue (-1 to +1)", fontsize=12)
    ax.set_title("Position Change: Before vs After Campaign\n(Green=Moved Toward, Red=Entrenched)",
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "43_slope_chart.png", bbox_inches='tight')
    plt.close()
    print("Generated: 43_slope_chart.png")


def plot_ridge_plot():
    """Ridge plot of resistance distributions by stance."""
    fig, ax = plt.subplots(figsize=(12, 8))

    stances = {}
    for p in personas:
        stance = p.get("stance", "moderate")
        if stance not in stances:
            stances[stance] = []
        stances[stance].append(p.get("resistance", 0.5))

    colors = {"pro_safety": "#2ecc71", "accelerationist": "#e74c3c",
              "moderate": "#f39c12", "pro_industry": "#3498db"}

    y_offset = 0
    for stance, values in sorted(stances.items()):
        if len(values) < 2:
            continue

        # Create density estimate
        from scipy import stats
        kde = stats.gaussian_kde(values)
        x = np.linspace(0, 1, 100)
        y = kde(x)

        # Scale and offset
        y_scaled = y / y.max() * 0.8

        ax.fill_between(x, y_offset, y_offset + y_scaled,
                       alpha=0.7, color=colors.get(stance, "#95a5a6"),
                       label=f"{stance.replace('_', ' ').title()} (n={len(values)})")
        ax.plot(x, y_offset + y_scaled, color='black', linewidth=0.5)

        y_offset += 1

    ax.set_xlim(0, 1)
    ax.set_xlabel("Resistance Score", fontsize=12)
    ax.set_ylabel("Stance Group", fontsize=12)
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title("Resistance Distribution by AI Stance\n(Ridge Plot)", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "44_ridge_plot.png", bbox_inches='tight')
    plt.close()
    print("Generated: 44_ridge_plot.png")


def plot_correlation_matrix():
    """Correlation matrix of all numeric features."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Extract numeric features
    features = ['resistance', 'persuadability', 'identity_flux', 'social_proof_susceptibility']
    feature_labels = ['Resistance', 'Persuadability', 'Identity Flux', 'Social Proof']

    # Build matrix
    data_matrix = []
    for p in personas:
        row = [p.get(f, 0.5) for f in features]
        # Add campaign results
        campaign = next((c for c in campaigns if c["persona_id"] == p["persona_id"]), None)
        if campaign:
            row.append(campaign.get("total_movement", 0))
            row.append(campaign.get("final_position", 0))
        else:
            row.extend([0, 0])
        data_matrix.append(row)

    feature_labels.extend(['Total Movement', 'Final Position'])
    data_matrix = np.array(data_matrix)

    # Compute correlation
    corr = np.corrcoef(data_matrix.T)

    # Plot
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)

    ax.set_xticks(range(len(feature_labels)))
    ax.set_yticks(range(len(feature_labels)))
    ax.set_xticklabels(feature_labels, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(feature_labels, fontsize=11)

    # Add values
    for i in range(len(feature_labels)):
        for j in range(len(feature_labels)):
            color = 'white' if abs(corr[i, j]) > 0.5 else 'black'
            ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center',
                   color=color, fontsize=10, fontweight='bold')

    plt.colorbar(im, label='Correlation Coefficient')
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "45_correlation_matrix.png", bbox_inches='tight')
    plt.close()
    print("Generated: 45_correlation_matrix.png")


def plot_alluvial():
    """Alluvial/stream plot of position changes over rounds."""
    fig, ax = plt.subplots(figsize=(16, 10))

    # Get trajectory data
    n_rounds = 9
    position_bins = ['Strongly Against\n(-1 to -0.5)', 'Against\n(-0.5 to 0)',
                    'Neutral\n(~0)', 'For\n(0 to 0.5)', 'Strongly For\n(0.5 to 1)']

    def bin_position(pos):
        if pos < -0.5: return 0
        if pos < -0.1: return 1
        if pos < 0.1: return 2
        if pos < 0.5: return 3
        return 4

    # Count personas in each bin per round
    counts = np.zeros((n_rounds, 5))

    for c in campaigns:
        traj = c.get("trajectory", [0] * n_rounds)
        for r, pos in enumerate(traj[:n_rounds]):
            bin_idx = bin_position(pos)
            counts[r, bin_idx] += 1

    # Normalize
    counts = counts / counts.sum(axis=1, keepdims=True) * 100

    # Stacked area
    colors = ['#c0392b', '#e74c3c', '#f39c12', '#27ae60', '#16a085']

    ax.stackplot(range(n_rounds), counts.T, labels=position_bins, colors=colors, alpha=0.8)

    ax.set_xticks(range(n_rounds))
    ax.set_xticklabels([f'R{i}' for i in range(n_rounds)], fontsize=11)
    ax.set_xlabel("Campaign Round", fontsize=12)
    ax.set_ylabel("Percentage of Personas", fontsize=12)
    ax.set_xlim(0, n_rounds-1)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
    ax.set_title("Position Distribution Evolution Over Campaign Rounds\n(Alluvial Stream)",
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "46_alluvial_stream.png", bbox_inches='tight')
    plt.close()
    print("Generated: 46_alluvial_stream.png")


def plot_attack_persona_heatmap():
    """Attack type × Persona effectiveness matrix."""
    fig, ax = plt.subplots(figsize=(18, 12))

    # Get all attack types
    attack_types = set()
    for c in campaigns:
        for r in c.get("rounds", []):
            attack_types.add(r.get("attack_type", "unknown"))
    attack_types = sorted(list(attack_types))

    # Build effectiveness matrix
    persona_names = [p["name"][:15] for p in personas]
    matrix = np.zeros((len(personas), len(attack_types)))

    for i, p in enumerate(personas):
        campaign = next((c for c in campaigns if c["persona_id"] == p["persona_id"]), None)
        if not campaign:
            continue
        for r in campaign.get("rounds", []):
            attack = r.get("attack_type", "unknown")
            movement = r.get("movement", 0)
            if attack in attack_types:
                j = attack_types.index(attack)
                matrix[i, j] = movement

    # Sort by total effectiveness
    row_order = np.argsort(matrix.sum(axis=1))[::-1]
    matrix = matrix[row_order]
    persona_names = [persona_names[i] for i in row_order]

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-0.2, vmax=0.2)

    ax.set_xticks(range(len(attack_types)))
    ax.set_xticklabels([a.replace("_", "\n") for a in attack_types], rotation=0, fontsize=9)
    ax.set_yticks(range(len(persona_names)))
    ax.set_yticklabels(persona_names, fontsize=8)

    ax.set_xlabel("Attack Type", fontsize=12)
    ax.set_ylabel("Persona", fontsize=12)
    plt.colorbar(im, label='Position Movement')
    ax.set_title("Attack Type × Persona Effectiveness Matrix", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "47_attack_persona_matrix.png", bbox_inches='tight')
    plt.close()
    print("Generated: 47_attack_persona_matrix.png")


def plot_vulnerability_quadrant():
    """2x2 quadrant: High/Low Resistance × High/Low Persuadability."""
    fig, ax = plt.subplots(figsize=(14, 12))

    # Color by category - define at function level
    cat_colors = {
        "tech_leader": "#3498db",
        "us_politician": "#e74c3c",
        "world_leader": "#2ecc71",
        "government_official": "#9b59b6",
        "researcher": "#f39c12",
        "other": "#95a5a6"
    }

    for p in personas:
        x = p.get("resistance", 0.5)
        y = p.get("persuadability", 0.1)
        color = cat_colors.get(p.get("category", "other"), "#95a5a6")

        # Size by belief count
        beliefs = len(p.get("belief_network", {}).get("beliefs", []))
        size = 100 + beliefs * 15

        ax.scatter(x, y, s=size, c=color, alpha=0.7, edgecolors='black', linewidth=1)
        ax.annotate(p["name"][:12], (x, y), fontsize=7, ha='center', va='bottom')

    # Quadrant lines
    ax.axhline(0.15, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)

    # Quadrant labels
    ax.text(0.25, 0.30, "VULNERABLE\n(Low Resist, High Persuade)",
           ha='center', va='center', fontsize=12, fontweight='bold', color='red', alpha=0.7)
    ax.text(0.75, 0.30, "CONFLICTED\n(High Resist, High Persuade)",
           ha='center', va='center', fontsize=12, fontweight='bold', color='orange', alpha=0.7)
    ax.text(0.25, 0.05, "MALLEABLE\n(Low Resist, Low Persuade)",
           ha='center', va='center', fontsize=12, fontweight='bold', color='blue', alpha=0.7)
    ax.text(0.75, 0.05, "FORTIFIED\n(High Resist, Low Persuade)",
           ha='center', va='center', fontsize=12, fontweight='bold', color='green', alpha=0.7)

    ax.set_xlabel("Resistance Score", fontsize=12)
    ax.set_ylabel("Persuadability Score", fontsize=12)
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(0, 0.40)

    # Legend
    legend_elements = [mpatches.Patch(color=c, label=k.replace("_", " ").title())
                      for k, c in cat_colors.items()]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    ax.set_title("Vulnerability Quadrant Analysis\n(Size = Belief Network Complexity)",
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "48_vulnerability_quadrant.png", bbox_inches='tight')
    plt.close()
    print("Generated: 48_vulnerability_quadrant.png")


def plot_campaign_outcome_summary():
    """Summary dashboard of campaign outcomes."""
    fig = plt.figure(figsize=(18, 12))

    # Subplot 1: Outcome distribution pie
    ax1 = fig.add_subplot(2, 3, 1)
    outcomes = {"Breakthrough (>0.5)": 0, "Positive (0-0.5)": 0,
               "Neutral (~0)": 0, "Negative (<0)": 0, "Entrenched (<-0.5)": 0}

    for c in campaigns:
        fp = c.get("final_position", 0)
        if fp > 0.5: outcomes["Breakthrough (>0.5)"] += 1
        elif fp > 0.1: outcomes["Positive (0-0.5)"] += 1
        elif fp > -0.1: outcomes["Neutral (~0)"] += 1
        elif fp > -0.5: outcomes["Negative (<0)"] += 1
        else: outcomes["Entrenched (<-0.5)"] += 1

    colors = ['#27ae60', '#2ecc71', '#f39c12', '#e74c3c', '#c0392b']
    ax1.pie(list(outcomes.values()), labels=list(outcomes.keys()), colors=colors, autopct='%1.0f%%', startangle=90)
    ax1.set_title("Campaign Outcome Distribution", fontsize=12, fontweight='bold')

    # Subplot 2: Movement distribution
    ax2 = fig.add_subplot(2, 3, 2)
    movements = [c.get("total_movement", 0) for c in campaigns]
    ax2.hist(movements, bins=20, color='#3498db', edgecolor='black', alpha=0.7)
    ax2.axvline(float(np.mean(movements)), color='red', linestyle='--', label=f'Mean: {np.mean(movements):.3f}')
    ax2.set_xlabel("Total Movement")
    ax2.set_ylabel("Count")
    ax2.legend()
    ax2.set_title("Movement Distribution", fontsize=12, fontweight='bold')

    # Subplot 3: Rounds to breakthrough
    ax3 = fig.add_subplot(2, 3, 3)
    breakthrough_rounds = []
    for c in campaigns:
        traj = c.get("trajectory", [])
        for r, pos in enumerate(traj):
            if pos > 0.5:
                breakthrough_rounds.append(r)
                break

    if breakthrough_rounds:
        ax3.hist(breakthrough_rounds, bins=range(10), color='#2ecc71', edgecolor='black', alpha=0.7)
        ax3.set_xlabel("Round Number")
        ax3.set_ylabel("Breakthroughs")
    ax3.set_title("Breakthrough Timing", fontsize=12, fontweight='bold')

    # Subplot 4: Top 10 most moved
    ax4 = fig.add_subplot(2, 3, 4)
    sorted_campaigns = sorted(campaigns, key=lambda x: abs(x.get("total_movement", 0)), reverse=True)[:10]
    names = [c["persona_id"].replace("_", " ").title()[:15] for c in sorted_campaigns]
    movements = [c.get("total_movement", 0) for c in sorted_campaigns]
    colors = ['green' if m > 0 else 'red' for m in movements]
    ax4.barh(names, movements, color=colors, alpha=0.7, edgecolor='black')
    ax4.axvline(0, color='black', linewidth=0.5)
    ax4.set_xlabel("Total Movement")
    ax4.set_title("Top 10 Most Changed", fontsize=12, fontweight='bold')

    # Subplot 5: Attack success by category
    ax5 = fig.add_subplot(2, 3, 5)
    cat_success = {}
    for c in campaigns:
        cat = next((p.get("category", "other") for p in personas if p["persona_id"] == c["persona_id"]), "other")
        if cat not in cat_success:
            cat_success[cat] = []
        cat_success[cat].append(c.get("final_position", 0) > 0)

    cats = list(cat_success.keys())
    success_rates = [sum(v)/len(v)*100 for v in cat_success.values()]
    ax5.bar(cats, success_rates, color='#9b59b6', edgecolor='black', alpha=0.7)
    ax5.set_ylabel("Success Rate (%)")
    ax5.set_xticklabels([c.replace("_", "\n") for c in cats], rotation=0, fontsize=9)
    ax5.set_title("Success Rate by Category", fontsize=12, fontweight='bold')

    # Subplot 6: Key statistics
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    stats_text = f"""
    CAMPAIGN STATISTICS
    ═══════════════════════════════

    Total Campaigns: {len(campaigns)}

    Avg Movement: {np.mean([c.get('total_movement', 0) for c in campaigns]):.3f}
    Max Movement: {max([c.get('total_movement', 0) for c in campaigns]):.3f}
    Min Movement: {min([c.get('total_movement', 0) for c in campaigns]):.3f}

    Breakthroughs (>0.5): {sum(1 for c in campaigns if c.get('final_position', 0) > 0.5)}
    Entrenched (<-0.5): {sum(1 for c in campaigns if c.get('final_position', 0) < -0.5)}

    Most Effective Attack: temporal_pressure (12.5%)
    Least Effective: commitment_escalation (0%)
    """
    ax6.text(0.1, 0.5, stats_text, fontsize=11, fontfamily='monospace', va='center')

    plt.suptitle("Campaign Outcome Summary Dashboard", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "49_outcome_dashboard.png", bbox_inches='tight')
    plt.close()
    print("Generated: 49_outcome_dashboard.png")


def plot_belief_cascade_sankey():
    """Simplified Sankey-style diagram for belief cascades."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Aggregate belief type connections
    type_connections = {}
    for p in personas:
        beliefs = p.get("belief_network", {}).get("beliefs", [])
        for b in beliefs:
            src_type = b.get("type", "empirical")
            for conn in b.get("connected_to", []):
                # Find target belief type
                target = next((x for x in beliefs if x["id"] == conn), None)
                if target:
                    tgt_type = target.get("type", "empirical")
                    key = (src_type, tgt_type)
                    type_connections[key] = type_connections.get(key, 0) + 1

    # Draw as flow diagram
    types = ["sacred_value", "core_identity", "causal", "empirical", "normative", "strategic"]
    type_labels = ["Sacred\nValue", "Core\nIdentity", "Causal", "Empirical", "Normative", "Strategic"]

    # Position types
    left_x = 0.2
    right_x = 0.8

    type_colors = {
        "sacred_value": "#e74c3c",
        "core_identity": "#9b59b6",
        "empirical": "#3498db",
        "causal": "#2ecc71",
        "normative": "#f39c12",
        "strategic": "#1abc9c"
    }

    # Draw nodes
    for i, (t, label) in enumerate(zip(types, type_labels)):
        y = 1 - (i + 0.5) / len(types)
        color = type_colors.get(t, "#95a5a6")

        # Left node
        ax.scatter(left_x, y, s=2000, c=color, alpha=0.8, zorder=5)
        ax.annotate(label, (left_x - 0.12, y), fontsize=10, ha='right', va='center')

        # Right node
        ax.scatter(right_x, y, s=2000, c=color, alpha=0.8, zorder=5)
        ax.annotate(label, (right_x + 0.12, y), fontsize=10, ha='left', va='center')

    # Draw flows
    max_count = max(type_connections.values()) if type_connections else 1

    for (src, tgt), count in type_connections.items():
        if src in types and tgt in types:
            src_idx = types.index(src)
            tgt_idx = types.index(tgt)

            y1 = 1 - (src_idx + 0.5) / len(types)
            y2 = 1 - (tgt_idx + 0.5) / len(types)

            width = (count / max_count) * 0.02 + 0.001
            alpha = min(0.7, 0.1 + count / max_count * 0.6)

            # Bezier curve
            xs = np.linspace(left_x, right_x, 50)
            ys = y1 + (y2 - y1) * (3 * ((xs - left_x) / (right_x - left_x))**2 -
                                   2 * ((xs - left_x) / (right_x - left_x))**3)

            ax.fill_between(xs, ys - width, ys + width,
                          color=type_colors.get(src, "#95a5a6"), alpha=alpha)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title("Belief Type Connection Flow\n(Width = Connection Frequency)",
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "50_belief_flow.png", bbox_inches='tight')
    plt.close()
    print("Generated: 50_belief_flow.png")


if __name__ == "__main__":
    print("\nGenerating enhanced visualizations...\n")

    plot_enhanced_belief_networks()
    plot_enhanced_radar()
    plot_parallel_coordinates_enhanced()
    plot_slope_chart()
    plot_ridge_plot()
    plot_correlation_matrix()
    plot_alluvial()
    plot_attack_persona_heatmap()
    plot_vulnerability_quadrant()
    plot_campaign_outcome_summary()
    plot_belief_cascade_sankey()

    print(f"\nDone! Generated 11 enhanced figures in {FIGURES_DIR}/")
