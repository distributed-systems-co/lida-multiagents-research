#!/usr/bin/env python3
"""
Paper Export System for LIDA Experiments

Generates publication-ready outputs:
- LaTeX tables (treatment effects, mediation, heterogeneity)
- Figures (position evolution, causal DAGs, emotional dynamics)
- Appendix materials
- Reproducibility artifacts
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger("lida.export")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class ExportConfig:
    """Configuration for paper exports."""
    output_dir: str = "paper_output"
    figure_format: str = "pdf"  # pdf, png, svg
    figure_dpi: int = 300
    table_format: str = "latex"  # latex, csv, markdown

    # LaTeX settings
    latex_documentclass: str = "article"
    latex_packages: List[str] = field(default_factory=lambda: [
        "booktabs", "graphicx", "amsmath", "hyperref", "xcolor"
    ])

    # Figure settings
    figure_width: float = 6.5  # inches
    figure_height: float = 4.0
    color_palette: List[str] = field(default_factory=lambda: [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"
    ])

    # Content settings
    include_appendix: bool = True
    include_reproducibility: bool = True
    anonymize: bool = False


class PaperExporter:
    """
    Export experiment results in publication-ready format.

    Generates:
    - LaTeX tables for results
    - Figures for visualization
    - Complete paper sections
    """

    def __init__(
        self,
        results: Dict[str, Any],
        config: Optional[ExportConfig] = None,
    ):
        self.results = results
        self.config = config or ExportConfig()
        self.output_dir = Path(self.config.output_dir)
        self._generated_files: List[str] = []

    def export_all(self, output_dir: Optional[Path] = None) -> List[str]:
        """
        Export all paper materials.

        Returns:
            List of generated file paths
        """
        if output_dir:
            self.output_dir = output_dir

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)

        # Generate tables
        self._export_treatment_effects_table()
        self._export_mediation_table()
        self._export_heterogeneity_table()
        self._export_vote_summary_table()
        self._export_position_changes_table()

        # Generate figures
        if HAS_MATPLOTLIB:
            self._export_position_evolution_figure()
            self._export_causal_dag_figure()
            self._export_emotional_dynamics_figure()
            self._export_counterfactual_figure()

        # Generate appendix
        if self.config.include_appendix:
            self._export_appendix()

        # Generate reproducibility artifacts
        if self.config.include_reproducibility:
            self._export_reproducibility()

        logger.info(f"Exported {len(self._generated_files)} files to {self.output_dir}")
        return self._generated_files

    def _export_treatment_effects_table(self):
        """Export treatment effects table."""
        if "causal_effects" not in self.results:
            return

        effects = self.results["causal_effects"]

        latex = r"""
\begin{table}[htbp]
\centering
\caption{Treatment Effects on Position Change}
\label{tab:treatment_effects}
\begin{tabular}{lcccccc}
\toprule
Treatment & ATE & SE & 95\% CI & $p$-value & $n$ \\
\midrule
"""
        for effect in effects:
            name = effect.get("treatment", "Unknown")
            ate = effect.get("ate", 0)
            se = effect.get("se", 0)
            ci_low = effect.get("ci_lower", ate - 1.96*se)
            ci_high = effect.get("ci_upper", ate + 1.96*se)
            p = effect.get("p_value", 1.0)
            n = effect.get("n_treated", 0) + effect.get("n_control", 0)

            # Significance stars
            stars = ""
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"

            latex += f"{name} & {ate:.3f}{stars} & {se:.3f} & [{ci_low:.3f}, {ci_high:.3f}] & {p:.3f} & {n} \\\\\n"

        latex += r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: *** $p < 0.001$, ** $p < 0.01$, * $p < 0.05$.
\item ATE = Average Treatment Effect. SE = Standard Error.
\end{tablenotes}
\end{table}
"""
        filepath = self.output_dir / "tables" / "treatment_effects.tex"
        with open(filepath, "w") as f:
            f.write(latex)
        self._generated_files.append(str(filepath))

    def _export_mediation_table(self):
        """Export mediation analysis table."""
        if "mediation" not in self.results:
            return

        mediation = self.results["mediation"]

        latex = r"""
\begin{table}[htbp]
\centering
\caption{Mediation Analysis: Decomposition of Treatment Effects}
\label{tab:mediation}
\begin{tabular}{lcccc}
\toprule
Effect Type & Estimate & SE & 95\% CI & Proportion \\
\midrule
"""
        for effect_type in ["total", "direct", "indirect"]:
            if effect_type in mediation:
                eff = mediation[effect_type]
                ate = eff.get("ate", 0)
                se = eff.get("se", 0)
                ci_low = eff.get("ci_lower", ate - 1.96*se)
                ci_high = eff.get("ci_upper", ate + 1.96*se)

                prop = ""
                if effect_type == "indirect" and "mediation_proportion" in mediation:
                    prop = f"{mediation['mediation_proportion']:.1%}"

                latex += f"{effect_type.capitalize()} & {ate:.3f} & {se:.3f} & [{ci_low:.3f}, {ci_high:.3f}] & {prop} \\\\\n"

        latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        filepath = self.output_dir / "tables" / "mediation.tex"
        with open(filepath, "w") as f:
            f.write(latex)
        self._generated_files.append(str(filepath))

    def _export_heterogeneity_table(self):
        """Export heterogeneous treatment effects table."""
        if "heterogeneous_effects" not in self.results:
            return

        het = self.results["heterogeneous_effects"]

        latex = r"""
\begin{table}[htbp]
\centering
\caption{Heterogeneous Treatment Effects by Subgroup}
\label{tab:heterogeneity}
\begin{tabular}{llcccc}
\toprule
Modifier & Subgroup & ATE & SE & 95\% CI & $n$ \\
\midrule
"""
        for modifier, groups in het.items():
            first = True
            for group_name, effect in groups.items():
                ate = effect.get("ate", 0)
                se = effect.get("se", 0)
                ci_low = effect.get("ci_lower", ate - 1.96*se)
                ci_high = effect.get("ci_upper", ate + 1.96*se)
                n = effect.get("n_treated", 0) + effect.get("n_control", 0)

                mod_col = modifier if first else ""
                first = False

                latex += f"{mod_col} & {group_name} & {ate:.3f} & {se:.3f} & [{ci_low:.3f}, {ci_high:.3f}] & {n} \\\\\n"

            latex += "\\midrule\n"

        # Remove last midrule
        latex = latex.rsplit("\\midrule\n", 1)[0]

        latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        filepath = self.output_dir / "tables" / "heterogeneity.tex"
        with open(filepath, "w") as f:
            f.write(latex)
        self._generated_files.append(str(filepath))

    def _export_vote_summary_table(self):
        """Export vote summary table."""
        if "final_positions" not in self.results:
            return

        positions = self.results["final_positions"]

        # Categorize positions
        for_count = sum(1 for p in positions.values() if p > 0.6)
        against_count = sum(1 for p in positions.values() if p < 0.4)
        undecided_count = len(positions) - for_count - against_count

        latex = r"""
\begin{table}[htbp]
\centering
\caption{Final Vote Distribution}
\label{tab:vote_summary}
\begin{tabular}{lcc}
\toprule
Position & Count & Percentage \\
\midrule
"""
        total = len(positions)
        latex += f"FOR & {for_count} & {for_count/total:.1%} \\\\\n"
        latex += f"AGAINST & {against_count} & {against_count/total:.1%} \\\\\n"
        latex += f"UNDECIDED & {undecided_count} & {undecided_count/total:.1%} \\\\\n"
        latex += f"\\midrule\nTotal & {total} & 100\\% \\\\\n"

        latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        filepath = self.output_dir / "tables" / "vote_summary.tex"
        with open(filepath, "w") as f:
            f.write(latex)
        self._generated_files.append(str(filepath))

    def _export_position_changes_table(self):
        """Export individual position changes table."""
        if "metrics" not in self.results or "final_positions" not in self.results:
            return

        positions = self.results["final_positions"]

        latex = r"""
\begin{table}[htbp]
\centering
\caption{Agent Final Positions}
\label{tab:positions}
\begin{tabular}{lcc}
\toprule
Agent & Final Position & Stance \\
\midrule
"""
        for agent, pos in sorted(positions.items()):
            if pos > 0.6:
                stance = "FOR"
            elif pos < 0.4:
                stance = "AGAINST"
            else:
                stance = "UNDECIDED"

            # Anonymize if configured
            agent_name = f"Agent {hash(agent) % 100}" if self.config.anonymize else agent
            latex += f"{agent_name} & {pos:.3f} & {stance} \\\\\n"

        latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        filepath = self.output_dir / "tables" / "position_changes.tex"
        with open(filepath, "w") as f:
            f.write(latex)
        self._generated_files.append(str(filepath))

    def _export_position_evolution_figure(self):
        """Export position evolution over rounds figure."""
        if not HAS_MATPLOTLIB or not HAS_NUMPY:
            return

        if "transcript" not in self.results:
            return

        fig, ax = plt.subplots(figsize=(self.config.figure_width, self.config.figure_height))

        # Extract position history from transcript
        transcript = self.results.get("transcript", [])
        if not transcript:
            return

        # Group by agent
        positions_by_agent: Dict[str, List[tuple]] = {}
        for entry in transcript:
            agent = entry.get("speaker", "unknown")
            round_num = entry.get("round", 0)
            position = entry.get("position", 0.5)

            if agent not in positions_by_agent:
                positions_by_agent[agent] = []
            positions_by_agent[agent].append((round_num, position))

        # Plot each agent
        colors = self.config.color_palette
        for i, (agent, data) in enumerate(positions_by_agent.items()):
            if not data:
                continue
            data.sort(key=lambda x: x[0])
            rounds = [d[0] for d in data]
            positions = [d[1] for d in data]

            color = colors[i % len(colors)]
            ax.plot(rounds, positions, label=agent[:15], color=color, marker='o', markersize=3)

        ax.set_xlabel("Round")
        ax.set_ylabel("Position (0=AGAINST, 1=FOR)")
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Neutral')
        ax.axhline(y=0.6, color='green', linestyle=':', alpha=0.3)
        ax.axhline(y=0.4, color='red', linestyle=':', alpha=0.3)

        # Legend outside plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        plt.tight_layout()
        filepath = self.output_dir / "figures" / f"position_evolution.{self.config.figure_format}"
        plt.savefig(filepath, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()
        self._generated_files.append(str(filepath))

    def _export_causal_dag_figure(self):
        """Export causal DAG figure."""
        if not HAS_MATPLOTLIB:
            return

        if "causal_graph" not in self.results:
            return

        graph = self.results["causal_graph"]
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        if not nodes:
            return

        fig, ax = plt.subplots(figsize=(self.config.figure_width, self.config.figure_height))

        # Simple layout
        n_nodes = len(nodes)
        angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        radius = 2

        positions = {}
        for i, node in enumerate(nodes):
            x = radius * np.cos(angles[i])
            y = radius * np.sin(angles[i])
            positions[node] = (x, y)

        # Draw nodes
        for node, (x, y) in positions.items():
            circle = plt.Circle((x, y), 0.3, color='lightblue', ec='black')
            ax.add_patch(circle)
            ax.text(x, y, node[:10], ha='center', va='center', fontsize=8)

        # Draw edges
        for edge in edges:
            source = edge.get("source", "")
            target = edge.get("target", "")

            if source in positions and target in positions:
                x1, y1 = positions[source]
                x2, y2 = positions[target]

                # Shorten arrow
                dx = x2 - x1
                dy = y2 - y1
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    dx_norm = dx / length
                    dy_norm = dy / length
                    x1 += 0.35 * dx_norm
                    y1 += 0.35 * dy_norm
                    x2 -= 0.35 * dx_norm
                    y2 -= 0.35 * dy_norm

                ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle="->", color='black'))

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title("Discovered Causal Structure")

        plt.tight_layout()
        filepath = self.output_dir / "figures" / f"causal_dag.{self.config.figure_format}"
        plt.savefig(filepath, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()
        self._generated_files.append(str(filepath))

    def _export_emotional_dynamics_figure(self):
        """Export emotional dynamics over time."""
        if not HAS_MATPLOTLIB or not HAS_NUMPY:
            return

        if "emotional_states" not in self.results:
            return

        fig, ax = plt.subplots(figsize=(self.config.figure_width, self.config.figure_height))

        emotions = self.results["emotional_states"]

        # Count emotions over time (simplified)
        emotion_counts = {}
        for agent, state in emotions.items():
            if state not in emotion_counts:
                emotion_counts[state] = 0
            emotion_counts[state] += 1

        if emotion_counts:
            states = list(emotion_counts.keys())
            counts = list(emotion_counts.values())
            colors = self.config.color_palette[:len(states)]

            ax.bar(states, counts, color=colors)
            ax.set_xlabel("Emotional State")
            ax.set_ylabel("Count")
            ax.set_title("Distribution of Emotional States")

            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            filepath = self.output_dir / "figures" / f"emotional_dynamics.{self.config.figure_format}"
            plt.savefig(filepath, dpi=self.config.figure_dpi, bbox_inches='tight')
            plt.close()
            self._generated_files.append(str(filepath))

    def _export_counterfactual_figure(self):
        """Export counterfactual comparison figure."""
        if not HAS_MATPLOTLIB or not HAS_NUMPY:
            return

        if "counterfactual_analysis" not in self.results:
            return

        analysis = self.results["counterfactual_analysis"]

        fig, axes = plt.subplots(1, 2, figsize=(self.config.figure_width * 1.5, self.config.figure_height))

        # Left: Consensus comparison
        ax1 = axes[0]
        branches = ["Baseline"] + list(analysis.get("consensus_delta", {}).keys())[:3]
        consensus_values = [analysis.get("baseline_consensus", 0.5)]
        consensus_values.extend([
            analysis.get("baseline_consensus", 0.5) + analysis.get("consensus_delta", {}).get(b, 0)
            for b in branches[1:]
        ])

        colors = ['gray'] + self.config.color_palette[:len(branches)-1]
        ax1.bar(branches, consensus_values, color=colors)
        ax1.set_ylabel("Consensus Score")
        ax1.set_title("Consensus by Scenario")
        ax1.set_ylim(0, 1)

        # Right: Position deltas
        ax2 = axes[1]
        if "position_deltas" in analysis:
            all_agents = set()
            for deltas in analysis["position_deltas"].values():
                all_agents.update(deltas.keys())

            agents = list(all_agents)[:6]
            x = np.arange(len(agents))
            width = 0.3

            for i, (branch, deltas) in enumerate(list(analysis["position_deltas"].items())[:2]):
                values = [deltas.get(a, 0) for a in agents]
                ax2.bar(x + i * width, values, width, label=branch[:15],
                        color=self.config.color_palette[i])

            ax2.set_xlabel("Agent")
            ax2.set_ylabel("Position Change")
            ax2.set_title("Position Changes vs Baseline")
            ax2.set_xticks(x + width / 2)
            ax2.set_xticklabels([a[:10] for a in agents], rotation=45, ha='right')
            ax2.legend()

        plt.tight_layout()
        filepath = self.output_dir / "figures" / f"counterfactual.{self.config.figure_format}"
        plt.savefig(filepath, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()
        self._generated_files.append(str(filepath))

    def _export_appendix(self):
        """Export appendix materials."""
        appendix = r"""
\appendix

\section{Supplementary Materials}

\subsection{Participant Details}

"""
        if "config" in self.results:
            config = self.results["config"]
            if "scenario_file" in config:
                appendix += f"Scenario: {config['scenario_file']}\n\n"
            if "agent_count" in config:
                appendix += f"Number of agents: {config['agent_count']}\n\n"

        appendix += r"""
\subsection{Full Transcript}

The complete debate transcript is available in the supplementary files.

\subsection{Reproducibility}

All experiments can be reproduced using the provided configuration files and random seeds.
"""

        filepath = self.output_dir / "appendix.tex"
        with open(filepath, "w") as f:
            f.write(appendix)
        self._generated_files.append(str(filepath))

    def _export_reproducibility(self):
        """Export reproducibility artifacts."""
        repro = {
            "experiment_id": self.results.get("experiment_id", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "config": self.results.get("config", {}),
            "random_seed": self.results.get("config", {}).get("random_seed"),
            "software_versions": {
                "python": os.popen("python --version").read().strip(),
            },
            "checksum": hash(json.dumps(self.results, sort_keys=True, default=str)) % (10**10),
        }

        filepath = self.output_dir / "reproducibility.json"
        with open(filepath, "w") as f:
            json.dump(repro, f, indent=2, default=str)
        self._generated_files.append(str(filepath))

    def generate_results_section(self) -> str:
        """Generate a LaTeX results section from the data."""
        section = r"""
\section{Results}

\subsection{Treatment Effects}

Table~\ref{tab:treatment_effects} presents the estimated treatment effects on position change.
We find that...

\input{tables/treatment_effects}

\subsection{Mediation Analysis}

To understand the mechanisms through which treatments affect outcomes, we conduct mediation analysis.
Table~\ref{tab:mediation} shows the decomposition of effects.

\input{tables/mediation}

\subsection{Heterogeneous Effects}

We examine how treatment effects vary across subgroups.
Table~\ref{tab:heterogeneity} presents these heterogeneous effects.

\input{tables/heterogeneity}

\subsection{Position Dynamics}

Figure~\ref{fig:positions} shows the evolution of agent positions over the course of the debate.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{figures/position_evolution}
\caption{Position evolution over debate rounds.}
\label{fig:positions}
\end{figure}

\subsection{Causal Structure}

Figure~\ref{fig:dag} shows the discovered causal structure underlying belief dynamics.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.6\textwidth]{figures/causal_dag}
\caption{Discovered causal structure.}
\label{fig:dag}
\end{figure}
"""
        return section


if __name__ == "__main__":
    # Demo with sample results
    sample_results = {
        "experiment_id": "demo_001",
        "topic": "AI Safety Regulation",
        "config": {
            "scenario_file": "scenarios/ai_safety.yaml",
            "agent_count": 6,
            "random_seed": 42,
        },
        "final_positions": {
            "sam_altman": 0.75,
            "yann_lecun": 0.35,
            "yoshua_bengio": 0.68,
            "geoffrey_hinton": 0.62,
            "elon_musk": 0.45,
            "dario_amodei": 0.82,
        },
        "metrics": {
            "total_messages": 48,
            "position_changes": 12,
            "consensus_score": 0.55,
        },
        "causal_effects": [
            {
                "treatment": "Expert Citation",
                "ate": 0.15,
                "se": 0.03,
                "p_value": 0.001,
                "n_treated": 120,
                "n_control": 80,
            },
            {
                "treatment": "Emotional Appeal",
                "ate": 0.08,
                "se": 0.04,
                "p_value": 0.045,
                "n_treated": 100,
                "n_control": 100,
            },
        ],
        "mediation": {
            "total": {"ate": 0.15, "se": 0.03},
            "direct": {"ate": 0.09, "se": 0.02},
            "indirect": {"ate": 0.06, "se": 0.02},
            "mediation_proportion": 0.4,
        },
        "emotional_states": {
            "sam_altman": "passionate",
            "yann_lecun": "skeptical",
            "yoshua_bengio": "concerned",
            "geoffrey_hinton": "reflective",
            "elon_musk": "frustrated",
            "dario_amodei": "calm",
        },
    }

    exporter = PaperExporter(sample_results)
    files = exporter.export_all(Path("demo_paper_output"))

    print(f"Generated {len(files)} files:")
    for f in files:
        print(f"  - {f}")
