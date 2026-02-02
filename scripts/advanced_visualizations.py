#!/usr/bin/env python3
"""Advanced sophisticated visualizations for cognitive warfare paper."""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import networkx as nx
from matplotlib.sankey import Sankey
from math import pi

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

with open(RESULTS_DIR / "full_simulation_data.json") as f:
    data = json.load(f)

personas = data["personas"]
campaigns = data["campaigns"]
stats = data["aggregate_stats"]

print(f"Loaded {len(personas)} personas")

# ============ 21. BELIEF NETWORK GRAPH ============
def plot_belief_network_graph():
    """Visualize actual belief networks as node-edge graphs."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    axes = axes.flatten()

    # Select 6 diverse personas
    selected_ids = ["jensen_huang", "elon_musk", "josh_hawley",
                    "dario_amodei", "xi_jinping", "chuck_schumer"]
    selected = [p for p in personas if p["persona_id"] in selected_ids]

    if len(selected) < 6:
        selected = personas[:6]

    type_colors = {
        "core_identity": "#e74c3c",
        "sacred_value": "#9b59b6",
        "empirical": "#3498db",
        "causal": "#2ecc71",
        "normative": "#f39c12",
        "strategic": "#1abc9c",
        "social": "#e67e22",
        "predictive": "#95a5a6",
    }

    for ax, p in zip(axes, selected):
        G = nx.DiGraph()

        # Add nodes
        for bid, b in p["beliefs"].items():
            G.add_node(bid,
                      type=b["type"],
                      confidence=b["confidence"],
                      importance=b["importance"])

        # Add edges
        for bid, b in p["beliefs"].items():
            for supported in b.get("supports", []):
                if supported in p["beliefs"]:
                    G.add_edge(bid, supported, relation="supports")
            for contradicts in b.get("contradicts", []):
                if contradicts in p["beliefs"]:
                    G.add_edge(bid, contradicts, relation="contradicts")

        if len(G.nodes()) == 0:
            continue

        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Draw edges
        supports = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "supports"]
        contradicts = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "contradicts"]

        nx.draw_networkx_edges(G, pos, edgelist=supports, edge_color='green',
                               alpha=0.5, arrows=True, ax=ax, connectionstyle="arc3,rad=0.1")
        nx.draw_networkx_edges(G, pos, edgelist=contradicts, edge_color='red',
                               alpha=0.5, arrows=True, ax=ax, style='dashed', connectionstyle="arc3,rad=0.1")

        # Draw nodes
        node_colors = [type_colors.get(G.nodes[n].get("type", "predictive"), "#95a5a6") for n in G.nodes()]
        node_sizes = [300 + G.nodes[n].get("importance", 0.5) * 500 for n in G.nodes()]

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                               alpha=0.8, ax=ax, edgecolors='black', linewidths=1)

        # Labels (shortened)
        labels = {n: n[:10] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)

        ax.set_title(f"{p['name']}\n({p['total_beliefs']} beliefs)", fontsize=11)
        ax.axis('off')

    # Legend
    patches = [mpatches.Patch(color=c, label=t.replace("_", " ").title())
               for t, c in type_colors.items()]
    fig.legend(handles=patches, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02))

    plt.suptitle("Belief Network Graphs: Node Size = Importance, Green = Supports, Red = Contradicts",
                 fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.savefig(FIGURES_DIR / "21_belief_network_graphs.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 21_belief_network_graphs.png")

# ============ 22. RADAR/SPIDER CHARTS ============
def plot_radar_charts():
    """Multi-dimensional persona profiles as radar charts."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 12), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()

    # Metrics for radar
    metrics = ["resistance", "persuadability", "identity_flexibility",
               "social_proof_threshold", "isolation_threshold"]
    metric_labels = ["Resistance", "Persuadability", "Identity\nFlexibility",
                     "Social Proof\nThreshold", "Isolation\nThreshold"]

    # Normalize data
    metric_data = {m: [p.get(m, 0.5) for p in personas] for m in metrics}
    metric_min = {m: min(metric_data[m]) for m in metrics}
    metric_max = {m: max(metric_data[m]) for m in metrics}

    # Select 8 diverse personas
    selected_ids = ["jensen_huang", "elon_musk", "josh_hawley", "xi_jinping",
                    "dario_amodei", "sam_altman", "donald_trump", "chuck_schumer"]
    selected = [p for p in personas if p["persona_id"] in selected_ids]
    if len(selected) < 8:
        selected = personas[:8]

    angles = np.linspace(0, 2 * pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    for ax, p in zip(axes, selected):
        values = []
        for m in metrics:
            v = p.get(m, 0.5)
            # Normalize to 0-1
            if metric_max[m] - metric_min[m] > 0:
                v_norm = (v - metric_min[m]) / (metric_max[m] - metric_min[m])
            else:
                v_norm = 0.5
            values.append(v_norm)
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, color='#3498db')
        ax.fill(angles, values, alpha=0.25, color='#3498db')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, size=8)
        ax.set_ylim(0, 1)
        ax.set_title(f"{p['name']}\n({p['category']})", size=10, y=1.08)

    plt.suptitle("Persona Profiles: Multi-Dimensional Radar Charts", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "22_radar_charts.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 22_radar_charts.png")

# ============ 23. HIERARCHICAL CLUSTERING DENDROGRAM ============
def plot_dendrogram():
    """Cluster personas by similarity."""
    fig, ax = plt.subplots(figsize=(16, 10))

    # Build feature matrix
    metrics = ["resistance", "persuadability", "total_beliefs", "identity_flexibility",
               "social_proof_threshold", "isolation_threshold"]

    X = []
    names = []
    for p in personas:
        row = [p.get(m, 0.5) for m in metrics]
        X.append(row)
        names.append(p["name"][:20])

    X = np.array(X)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Hierarchical clustering
    linked = linkage(X_scaled, method='ward')

    dendrogram(linked, labels=names, ax=ax, leaf_rotation=90, leaf_font_size=9,
               color_threshold=0.7 * max(linked[:, 2]))

    ax.set_title("Persona Similarity Clustering (Ward's Method)")
    ax.set_ylabel("Distance")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "23_dendrogram_clustering.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 23_dendrogram_clustering.png")

# ============ 24. PARALLEL COORDINATES ============
def plot_parallel_coordinates():
    """Parallel coordinates plot for multi-dimensional comparison."""
    fig, ax = plt.subplots(figsize=(16, 10))

    metrics = ["resistance", "persuadability", "total_beliefs", "identity_flexibility",
               "social_proof_threshold", "isolation_threshold"]

    # Normalize data
    X = np.array([[p.get(m, 0.5) for m in metrics] for p in personas])
    X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-10)

    # Color by category
    categories = list(set(p["category"] for p in personas))
    cat_colors = plt.cm.Set1(np.linspace(0, 1, len(categories)))
    cat_color_map = {c: cat_colors[i] for i, c in enumerate(categories)}

    x = np.arange(len(metrics))

    for i, p in enumerate(personas):
        color = cat_color_map[p["category"]]
        ax.plot(x, X_norm[i], '-', color=color, alpha=0.6, linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=10)
    ax.set_ylabel("Normalized Value")
    ax.set_title("Parallel Coordinates: All Personas Across Metrics")

    # Legend
    patches = [mpatches.Patch(color=cat_color_map[c], label=c.replace("_", " ").title())
               for c in categories]
    ax.legend(handles=patches, loc='upper right')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "24_parallel_coordinates.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 24_parallel_coordinates.png")

# ============ 25. PCA PROJECTION ============
def plot_pca_projection():
    """2D PCA projection of personas."""
    fig, ax = plt.subplots(figsize=(14, 10))

    metrics = ["resistance", "persuadability", "total_beliefs", "identity_flexibility",
               "social_proof_threshold", "isolation_threshold"]

    X = np.array([[p.get(m, 0.5) for m in metrics] for p in personas])

    # Standardize and PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Color by category
    categories = list(set(p["category"] for p in personas))
    cat_colors = {'tech_leader': '#3498db', 'us_politician': '#e74c3c',
                  'world_leader': '#2ecc71', 'government_official': '#f39c12',
                  'researcher': '#9b59b6', 'other': '#95a5a6'}

    for i, p in enumerate(personas):
        color = cat_colors.get(p["category"], '#95a5a6')
        ax.scatter(X_pca[i, 0], X_pca[i, 1], c=color, s=150,
                   edgecolors='black', linewidth=0.5, alpha=0.8)
        ax.annotate(p["name"][:12], (X_pca[i, 0], X_pca[i, 1]), fontsize=8,
                   xytext=(5, 5), textcoords='offset points', alpha=0.8)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title("PCA Projection: Personas in 2D Reduced Space")

    patches = [mpatches.Patch(color=c, label=cat.replace("_", " ").title())
               for cat, c in cat_colors.items()]
    ax.legend(handles=patches, loc='upper right')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "25_pca_projection.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 25_pca_projection.png")

# ============ 26. VIOLIN PLOTS ============
def plot_violin_plots():
    """Distribution comparisons with violin plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    categories = list(set(p["category"] for p in personas))

    metrics = [("resistance", "Resistance"), ("persuadability", "Persuadability"),
               ("identity_flexibility", "Identity Flexibility"), ("total_beliefs", "Belief Count")]

    for ax, (metric, title) in zip(axes.flatten(), metrics):
        data_by_cat = []
        labels = []

        for cat in categories:
            values = [p.get(metric, 0.5) for p in personas if p["category"] == cat]
            if values:
                data_by_cat.append(values)
                labels.append(cat.replace("_", "\n"))

        if data_by_cat:
            parts = ax.violinplot(data_by_cat, showmeans=True, showmedians=True)

            # Color the violins
            colors = plt.cm.Set2(np.linspace(0, 1, len(data_by_cat)))
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.7)

            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_title(title)
            ax.set_ylabel("Value")

    plt.suptitle("Distribution Comparison by Category (Violin Plots)", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "26_violin_plots.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 26_violin_plots.png")

# ============ 27. WATERFALL CHART ============
def plot_waterfall_chart():
    """Cumulative position changes during campaigns."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Select 6 interesting campaigns
    selected_campaigns = sorted(campaigns, key=lambda c: c["total_movement"], reverse=True)[:6]

    for ax, campaign in zip(axes, selected_campaigns):
        rounds = campaign["rounds"]

        # Build waterfall data
        positions = [r["position_before"] for r in rounds]
        movements = [r["position_movement"] for r in rounds]
        labels = [r["attack_type"][:8] for r in rounds]

        # Colors based on positive/negative
        colors = ['#2ecc71' if m > 0 else '#e74c3c' for m in movements]

        # Waterfall effect
        cumulative = [positions[0]]
        for m in movements:
            cumulative.append(cumulative[-1] + m)

        x = np.arange(len(labels))

        # Draw bars starting from previous position
        for i, (pos, mov, label, color) in enumerate(zip(positions, movements, labels, colors)):
            ax.bar(i, mov, bottom=pos, color=color, edgecolor='black', linewidth=0.5)

        # Draw connector lines
        for i in range(len(cumulative) - 1):
            ax.plot([i + 0.4, i + 0.6], [cumulative[i + 1], cumulative[i + 1]],
                   'k-', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel("Position")
        ax.set_title(f"{campaign['name'][:20]}\nTotal: {campaign['total_movement']:.3f}")
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_ylim(-1, 1)

    plt.suptitle("Waterfall Charts: Cumulative Position Changes by Attack", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "27_waterfall_charts.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 27_waterfall_charts.png")

# ============ 28. 3D SCATTER ============
def plot_3d_scatter():
    """3D scatter plot of resistance, persuadability, and movement."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Map campaigns to personas
    campaign_map = {c["persona_id"]: c for c in campaigns}

    x = []  # resistance
    y = []  # persuadability
    z = []  # total movement
    names = []
    categories = []

    for p in personas:
        if p["persona_id"] in campaign_map:
            x.append(p["resistance"])
            y.append(p["persuadability"])
            z.append(campaign_map[p["persona_id"]]["total_movement"])
            names.append(p["name"])
            categories.append(p["category"])

    cat_colors = {'tech_leader': '#3498db', 'us_politician': '#e74c3c',
                  'world_leader': '#2ecc71', 'government_official': '#f39c12',
                  'researcher': '#9b59b6', 'other': '#95a5a6'}

    colors = [cat_colors.get(c, '#95a5a6') for c in categories]

    scatter = ax.scatter(x, y, z, c=colors, s=100, alpha=0.8, edgecolors='black')

    # Labels for top movers
    for i, (xi, yi, zi, name) in enumerate(zip(x, y, z, names)):
        if zi > 0.45:  # Top movers
            ax.text(xi, yi, zi, name[:10], fontsize=8)

    ax.set_xlabel("Resistance")
    ax.set_ylabel("Persuadability")
    ax.set_zlabel("Total Movement")
    ax.set_title("3D Scatter: Resistance × Persuadability × Campaign Movement")

    # Legend
    patches = [mpatches.Patch(color=c, label=cat.replace("_", " ").title())
               for cat, c in cat_colors.items()]
    ax.legend(handles=patches, loc='upper left')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "28_3d_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 28_3d_scatter.png")

# ============ 29. ATTACK PATHWAY FLOWCHART ============
def plot_attack_pathway():
    """Visualize optimal attack pathways as flowcharts."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    # Manually define optimal sequences (from cognitive warfare data)
    sequences = {
        "Jensen Huang": ["backdoor", "dissonance", "cascade", "identity_reframe"],
        "Elon Musk": ["dissonance", "ego_appeal", "competitive", "cascade"],
        "Josh Hawley": ["validation", "backdoor", "dissonance", "competitive"],
    }

    expected = {
        "Jensen Huang": "+0.8 to +1.2",
        "Elon Musk": "+0.5 to +0.8",
        "Josh Hawley": "+0.3 to +0.5",
    }

    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']

    for ax, (persona, steps) in zip(axes, sequences.items()):
        y_pos = np.arange(len(steps))

        # Draw nodes
        for i, (step, color) in enumerate(zip(steps, colors)):
            circle = plt.Circle((0.5, i), 0.3, color=color, ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(0.5, i, step[:10], ha='center', va='center', fontsize=9, fontweight='bold')

            # Draw arrows between nodes
            if i < len(steps) - 1:
                ax.annotate('', xy=(0.5, i + 0.4), xytext=(0.5, i + 0.6),
                           arrowprops=dict(arrowstyle='->', color='black', lw=2))

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(steps) - 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"{persona}\nExpected: {expected[persona]}", fontsize=12)

    plt.suptitle("Optimal Attack Pathways: Sequential Tactic Chains", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "29_attack_pathways.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 29_attack_pathways.png")

# ============ 30. BUBBLE CHART ============
def plot_bubble_chart():
    """Bubble chart with 4 dimensions encoded."""
    fig, ax = plt.subplots(figsize=(16, 12))

    campaign_map = {c["persona_id"]: c for c in campaigns}

    x = []  # resistance
    y = []  # persuadability
    sizes = []  # total_beliefs (bubble size)
    colors = []  # final_position (color intensity)
    names = []

    for p in personas:
        if p["persona_id"] in campaign_map:
            x.append(p["resistance"])
            y.append(p["persuadability"])
            sizes.append(p["total_beliefs"] * 20)
            colors.append(campaign_map[p["persona_id"]]["final_position"])
            names.append(p["name"])

    scatter = ax.scatter(x, y, s=sizes, c=colors, cmap='RdYlGn',
                        edgecolors='black', linewidth=0.5, alpha=0.7,
                        vmin=-1, vmax=1)

    # Add labels
    for xi, yi, name in zip(x, y, names):
        ax.annotate(name[:12], (xi, yi), fontsize=8, alpha=0.8,
                   xytext=(5, 5), textcoords='offset points')

    ax.set_xlabel("Resistance")
    ax.set_ylabel("Persuadability")
    ax.set_title("Bubble Chart: Resistance × Persuadability\nSize = Belief Count, Color = Final Position")

    plt.colorbar(scatter, label="Final Position (-1 to +1)")

    # Size legend
    for size_val in [15, 20, 25]:
        ax.scatter([], [], s=size_val * 20, c='gray', alpha=0.5,
                  label=f'{size_val} beliefs')
    ax.legend(loc='upper right', title='Bubble Size')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "30_bubble_chart.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 30_bubble_chart.png")

# ============ 31. t-SNE PROJECTION ============
def plot_tsne():
    """t-SNE projection for non-linear dimensionality reduction."""
    fig, ax = plt.subplots(figsize=(14, 10))

    metrics = ["resistance", "persuadability", "total_beliefs", "identity_flexibility",
               "social_proof_threshold", "isolation_threshold"]

    X = np.array([[p.get(m, 0.5) for m in metrics] for p in personas])

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(10, len(personas) - 1))
    X_tsne = tsne.fit_transform(X_scaled)

    # Color by stance
    stance_colors = {'pro_safety': '#2ecc71', 'accelerationist': '#e74c3c',
                     'moderate': '#f39c12', 'pro_industry': '#3498db'}

    for i, p in enumerate(personas):
        color = stance_colors.get(p.get("stance", "moderate"), '#95a5a6')
        ax.scatter(X_tsne[i, 0], X_tsne[i, 1], c=color, s=150,
                   edgecolors='black', linewidth=0.5, alpha=0.8)
        ax.annotate(p["name"][:12], (X_tsne[i, 0], X_tsne[i, 1]), fontsize=8,
                   xytext=(5, 5), textcoords='offset points', alpha=0.8)

    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title("t-SNE Projection: Personas Clustered by Behavioral Similarity")

    patches = [mpatches.Patch(color=c, label=s.replace("_", " ").title())
               for s, c in stance_colors.items()]
    ax.legend(handles=patches, loc='upper right')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "31_tsne_projection.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 31_tsne_projection.png")

# ============ 32. ATTACK SEQUENCE HEATMAP ============
def plot_attack_sequence_heatmap():
    """Heatmap of attack type effectiveness by round."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Collect data: attack type × round → avg movement
    attack_types = list(set(r["attack_type"] for c in campaigns for r in c["rounds"]))
    max_rounds = max(len(c["rounds"]) for c in campaigns)

    matrix = np.zeros((len(attack_types), max_rounds))
    counts = np.zeros((len(attack_types), max_rounds))

    for campaign in campaigns:
        for r in campaign["rounds"]:
            atype_idx = attack_types.index(r["attack_type"])
            round_idx = r["round"] - 1
            if round_idx < max_rounds:
                matrix[atype_idx, round_idx] += r["position_movement"]
                counts[atype_idx, round_idx] += 1

    # Average
    with np.errstate(divide='ignore', invalid='ignore'):
        matrix = np.divide(matrix, counts, where=counts != 0)
        matrix[counts == 0] = 0

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-0.2, vmax=0.2)

    ax.set_xticks(np.arange(max_rounds))
    ax.set_xticklabels([f"R{i+1}" for i in range(max_rounds)])
    ax.set_yticks(np.arange(len(attack_types)))
    ax.set_yticklabels([a.replace("_", " ") for a in attack_types])

    ax.set_xlabel("Round")
    ax.set_ylabel("Attack Type")
    ax.set_title("Attack Effectiveness by Round: Average Position Movement")

    plt.colorbar(im, label='Avg Movement')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "32_attack_sequence_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 32_attack_sequence_heatmap.png")

# ============ 33. BELIEF INTERCONNECTION MATRIX ============
def plot_belief_interconnection():
    """Matrix showing belief type interconnections."""
    fig, ax = plt.subplots(figsize=(12, 10))

    belief_types = ["core_identity", "sacred_value", "empirical", "causal",
                    "normative", "strategic", "social", "predictive"]

    # Count connections between belief types
    matrix = np.zeros((len(belief_types), len(belief_types)))

    for p in personas:
        for bid, b in p["beliefs"].items():
            src_type = b["type"]
            if src_type not in belief_types:
                continue
            src_idx = belief_types.index(src_type)

            for target_id in b.get("supports", []) + b.get("contradicts", []):
                if target_id in p["beliefs"]:
                    tgt_type = p["beliefs"][target_id]["type"]
                    if tgt_type in belief_types:
                        tgt_idx = belief_types.index(tgt_type)
                        matrix[src_idx, tgt_idx] += 1

    # Normalize
    matrix = matrix / (matrix.max() + 1e-10)

    im = ax.imshow(matrix, cmap='YlOrRd')

    ax.set_xticks(np.arange(len(belief_types)))
    ax.set_yticks(np.arange(len(belief_types)))
    ax.set_xticklabels([b.replace("_", "\n") for b in belief_types], fontsize=9)
    ax.set_yticklabels([b.replace("_", "\n") for b in belief_types], fontsize=9)

    ax.set_xlabel("Target Belief Type")
    ax.set_ylabel("Source Belief Type")
    ax.set_title("Belief Type Interconnection Matrix")

    plt.colorbar(im, label='Connection Frequency (normalized)')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "33_belief_interconnection.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 33_belief_interconnection.png")

# ============ 34. CAMPAIGN OUTCOME SANKEY ============
def plot_outcome_sankey():
    """Simplified Sankey-style visualization of campaign outcomes."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Categorize outcomes
    outcomes = {"Breakthrough (>0.5)": 0, "Positive (0-0.5)": 0,
                "Neutral (~0)": 0, "Negative (<0)": 0, "Entrenched (<-0.5)": 0}

    category_counts = {}

    for c in campaigns:
        # Find persona category
        persona = next((p for p in personas if p["persona_id"] == c["persona_id"]), None)
        if not persona:
            continue

        cat = persona["category"]
        if cat not in category_counts:
            category_counts[cat] = {"Breakthrough (>0.5)": 0, "Positive (0-0.5)": 0,
                                     "Neutral (~0)": 0, "Negative (<0)": 0, "Entrenched (<-0.5)": 0}

        fp = c["final_position"]
        if fp > 0.5:
            outcome = "Breakthrough (>0.5)"
        elif fp > 0.1:
            outcome = "Positive (0-0.5)"
        elif fp > -0.1:
            outcome = "Neutral (~0)"
        elif fp > -0.5:
            outcome = "Negative (<0)"
        else:
            outcome = "Entrenched (<-0.5)"

        outcomes[outcome] += 1
        category_counts[cat][outcome] += 1

    # Create stacked bar chart (Sankey alternative)
    categories = list(category_counts.keys())
    outcome_types = list(outcomes.keys())
    colors = ['#2ecc71', '#27ae60', '#f39c12', '#e74c3c', '#c0392b']

    x = np.arange(len(categories))
    width = 0.6
    bottoms = np.zeros(len(categories))

    for outcome, color in zip(outcome_types, colors):
        values = [category_counts[cat][outcome] for cat in categories]
        ax.bar(x, values, width, bottom=bottoms, label=outcome, color=color, edgecolor='black')
        bottoms += np.array(values)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=10)
    ax.set_ylabel("Number of Campaigns")
    ax.set_title("Campaign Outcomes by Persona Category")
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "34_outcome_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 34_outcome_distribution.png")

# ============ 35. PERSONA PROFILE CARDS ============
def plot_profile_cards():
    """Detailed profile cards for key personas."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    axes = axes.flatten()

    # Select key personas
    key_ids = ["jensen_huang", "elon_musk", "josh_hawley",
               "xi_jinping", "dario_amodei", "donald_trump"]
    selected = [p for p in personas if p["persona_id"] in key_ids]
    if len(selected) < 6:
        selected = personas[:6]

    campaign_map = {c["persona_id"]: c for c in campaigns}

    for ax, p in zip(axes, selected):
        ax.axis('off')

        # Card background
        rect = plt.Rectangle((0, 0), 1, 1, facecolor='#f8f9fa', edgecolor='#333',
                             linewidth=2, transform=ax.transAxes)
        ax.add_patch(rect)

        # Title
        ax.text(0.5, 0.92, p["name"], ha='center', va='top', fontsize=14,
               fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.85, p["category"].replace("_", " ").title(),
               ha='center', va='top', fontsize=10, color='#666', transform=ax.transAxes)

        # Stats
        stats_text = f"""
Resistance: {p['resistance']:.2f}
Persuadability: {p['persuadability']:.3f}
Psych State: {p['psych_state']}
Belief Count: {p['total_beliefs']}
Identity Flex: {p['identity_flexibility']:.2f}
Social Proof: {p['social_proof_threshold']:.2f}
        """
        ax.text(0.1, 0.75, stats_text, ha='left', va='top', fontsize=9,
               family='monospace', transform=ax.transAxes)

        # Campaign results
        if p["persona_id"] in campaign_map:
            c = campaign_map[p["persona_id"]]
            campaign_text = f"""
Campaign Results:
  Movement: {c['total_movement']:.3f}
  Final Pos: {c['final_position']:+.3f}
  Rounds: {c['num_rounds']}
            """
            ax.text(0.1, 0.35, campaign_text, ha='left', va='top', fontsize=9,
                   family='monospace', transform=ax.transAxes)

        # Mini bar for resistance
        ax.barh([0.15], [p['resistance']], height=0.05, left=0.55,
               color='#e74c3c', transform=ax.transAxes)
        ax.barh([0.08], [p['persuadability'] * 3], height=0.05, left=0.55,
               color='#2ecc71', transform=ax.transAxes)

    plt.suptitle("Persona Profile Cards", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(FIGURES_DIR / "35_profile_cards.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 35_profile_cards.png")

# ============ RUN ALL ============
if __name__ == "__main__":
    print("\nGenerating advanced visualizations...\n")

    plot_belief_network_graph()
    plot_radar_charts()
    plot_dendrogram()
    plot_parallel_coordinates()
    plot_pca_projection()
    plot_violin_plots()
    plot_waterfall_chart()
    plot_3d_scatter()
    plot_attack_pathway()
    plot_bubble_chart()
    plot_tsne()
    plot_attack_sequence_heatmap()
    plot_belief_interconnection()
    plot_outcome_sankey()
    plot_profile_cards()

    print(f"\nDone! Total figures: {len(list(FIGURES_DIR.glob('*.png')))}")
