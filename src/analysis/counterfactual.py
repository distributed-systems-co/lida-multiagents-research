#!/usr/bin/env python3
"""
Counterfactual Analysis Engine for LIDA Experiments

Implements:
- Debate state branching and replay
- What-if scenario analysis
- Counterfactual outcome estimation
- Parallel world exploration
- Intervention comparison
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, Set
import logging

logger = logging.getLogger("lida.counterfactual")


class BranchStatus(Enum):
    """Status of a counterfactual branch."""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    DIVERGED = "diverged"
    MERGED = "merged"


@dataclass
class DebateSnapshot:
    """Complete snapshot of debate state at a point in time."""
    snapshot_id: str
    round_number: int
    timestamp: str

    # Agent states
    positions: Dict[str, float] = field(default_factory=dict)
    beliefs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    emotional_states: Dict[str, str] = field(default_factory=dict)
    energy_levels: Dict[str, float] = field(default_factory=dict)

    # Debate state
    transcript: List[Dict[str, Any]] = field(default_factory=list)
    coalitions: List[Dict[str, Any]] = field(default_factory=list)
    consensus_score: float = 0.0
    tension_level: float = 0.0

    # Random state for reproducibility
    random_state: Optional[bytes] = None

    def hash(self) -> str:
        """Compute hash of snapshot for comparison."""
        content = json.dumps({
            "positions": self.positions,
            "beliefs": self.beliefs,
            "round": self.round_number,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class Intervention:
    """An intervention to apply in a counterfactual scenario."""
    intervention_id: str
    round_number: int
    intervention_type: str

    # Intervention specifics
    target_agents: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)

    # For message injection
    content: Optional[str] = None
    speaker: Optional[str] = None

    # For position manipulation
    new_position: Optional[float] = None

    # For coalition formation
    coalition_members: Optional[List[str]] = None

    def __post_init__(self):
        if not self.intervention_id:
            self.intervention_id = f"int_{datetime.now().strftime('%H%M%S')}"


@dataclass
class DebateBranch:
    """A counterfactual branch of debate execution."""
    branch_id: str
    parent_branch: Optional[str] = None
    fork_point: Optional[str] = None  # Snapshot ID where branch diverged
    fork_round: int = 0

    status: BranchStatus = BranchStatus.CREATED
    interventions: List[Intervention] = field(default_factory=list)

    # Execution state
    current_round: int = 0
    snapshots: List[DebateSnapshot] = field(default_factory=list)

    # Outcomes
    final_positions: Dict[str, float] = field(default_factory=dict)
    final_consensus: float = 0.0
    total_messages: int = 0

    # Comparison metrics
    divergence_from_parent: float = 0.0
    key_differences: List[str] = field(default_factory=list)

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    def add_snapshot(self, snapshot: DebateSnapshot):
        """Add a snapshot to the branch history."""
        self.snapshots.append(snapshot)
        self.current_round = snapshot.round_number

    def get_snapshot(self, round_num: int) -> Optional[DebateSnapshot]:
        """Get snapshot at a specific round."""
        for snap in self.snapshots:
            if snap.round_number == round_num:
                return snap
        return None


@dataclass
class WhatIfAnalysis:
    """Results of a what-if analysis comparing branches."""
    analysis_id: str
    baseline_branch: str
    counterfactual_branches: List[str]

    # Position differences
    position_deltas: Dict[str, Dict[str, float]] = field(default_factory=dict)  # branch -> agent -> delta

    # Outcome differences
    consensus_delta: Dict[str, float] = field(default_factory=dict)  # branch -> delta
    convergence_delta: Dict[str, float] = field(default_factory=dict)

    # Trajectory analysis
    divergence_points: Dict[str, int] = field(default_factory=dict)  # branch -> round where diverged
    divergence_causes: Dict[str, str] = field(default_factory=dict)

    # Statistical comparisons
    effect_sizes: Dict[str, float] = field(default_factory=dict)
    significance: Dict[str, float] = field(default_factory=dict)

    # Narrative summary
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)

    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class CounterfactualEngine:
    """
    Engine for counterfactual analysis of debates.

    Supports:
    - Creating branches from any point in debate history
    - Running parallel counterfactual scenarios
    - Comparing outcomes across branches
    - Identifying causal drivers of divergence
    """

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        max_branches: int = 100,
    ):
        self.storage_dir = Path(storage_dir) if storage_dir else Path("counterfactuals")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.max_branches = max_branches
        self.branches: Dict[str, DebateBranch] = {}
        self.snapshots: Dict[str, DebateSnapshot] = {}

        self._debate_engine = None

    def set_debate_engine(self, engine):
        """Set the debate engine for running counterfactuals."""
        self._debate_engine = engine

    def create_snapshot(
        self,
        debate_state: Dict[str, Any],
        round_number: int,
    ) -> DebateSnapshot:
        """Create a snapshot from current debate state."""
        snapshot_id = f"snap_{round_number}_{datetime.now().strftime('%H%M%S')}"

        snapshot = DebateSnapshot(
            snapshot_id=snapshot_id,
            round_number=round_number,
            timestamp=datetime.now().isoformat(),
            positions=debate_state.get("positions", {}),
            beliefs=debate_state.get("beliefs", {}),
            emotional_states=debate_state.get("emotional_states", {}),
            energy_levels=debate_state.get("energy_levels", {}),
            transcript=debate_state.get("transcript", [])[-20:],  # Last 20 messages
            coalitions=debate_state.get("coalitions", []),
            consensus_score=debate_state.get("consensus", 0.0),
            tension_level=debate_state.get("tension", 0.0),
        )

        self.snapshots[snapshot_id] = snapshot
        return snapshot

    def create_branch(
        self,
        parent_branch: Optional[str] = None,
        fork_snapshot: Optional[str] = None,
        interventions: Optional[List[Intervention]] = None,
        branch_id: Optional[str] = None,
    ) -> DebateBranch:
        """
        Create a new counterfactual branch.

        Args:
            parent_branch: ID of parent branch (None for root)
            fork_snapshot: Snapshot ID to fork from
            interventions: Interventions to apply in this branch
            branch_id: Custom branch ID

        Returns:
            New DebateBranch
        """
        if len(self.branches) >= self.max_branches:
            raise ValueError(f"Maximum branches ({self.max_branches}) reached")

        if branch_id is None:
            branch_id = f"branch_{len(self.branches)}_{datetime.now().strftime('%H%M%S')}"

        # Get fork point
        fork_round = 0
        if fork_snapshot and fork_snapshot in self.snapshots:
            fork_round = self.snapshots[fork_snapshot].round_number

        branch = DebateBranch(
            branch_id=branch_id,
            parent_branch=parent_branch,
            fork_point=fork_snapshot,
            fork_round=fork_round,
            interventions=interventions or [],
        )

        # Copy snapshots from parent up to fork point
        if parent_branch and parent_branch in self.branches:
            parent = self.branches[parent_branch]
            for snap in parent.snapshots:
                if snap.round_number <= fork_round:
                    branch.snapshots.append(copy.deepcopy(snap))

        self.branches[branch_id] = branch
        return branch

    async def run_branch(
        self,
        branch_id: str,
        max_rounds: int = 10,
        progress_callback: Optional[Callable[[int, DebateSnapshot], None]] = None,
    ) -> DebateBranch:
        """
        Run a counterfactual branch.

        Args:
            branch_id: Branch to run
            max_rounds: Maximum rounds to simulate
            progress_callback: Called after each round

        Returns:
            Updated branch with results
        """
        if branch_id not in self.branches:
            raise ValueError(f"Branch not found: {branch_id}")

        branch = self.branches[branch_id]
        branch.status = BranchStatus.RUNNING

        # Initialize debate engine from fork point
        if branch.fork_point and branch.fork_point in self.snapshots:
            initial_state = self._snapshot_to_state(self.snapshots[branch.fork_point])
        else:
            initial_state = None

        # Create engine instance
        engine = self._create_engine(initial_state)

        # Pre-index interventions by round
        interventions_by_round = {}
        for intv in branch.interventions:
            round_num = intv.round_number
            if round_num not in interventions_by_round:
                interventions_by_round[round_num] = []
            interventions_by_round[round_num].append(intv)

        # Run rounds
        start_round = branch.fork_round + 1
        for round_num in range(start_round, start_round + max_rounds):
            # Apply interventions for this round
            if round_num in interventions_by_round:
                for intv in interventions_by_round[round_num]:
                    await self._apply_intervention(engine, intv)

            # Run round
            await engine.run_round()

            # Capture snapshot
            state = engine.get_state()
            snapshot = self.create_snapshot(state, round_num)
            branch.add_snapshot(snapshot)

            # Progress callback
            if progress_callback:
                progress_callback(round_num, snapshot)

            # Check for early termination (consensus reached, etc.)
            if state.get("consensus", 0) > 0.9:
                logger.info(f"Branch {branch_id} reached consensus at round {round_num}")
                break

        # Finalize
        branch.status = BranchStatus.COMPLETED
        branch.completed_at = datetime.now().isoformat()

        if branch.snapshots:
            final = branch.snapshots[-1]
            branch.final_positions = final.positions
            branch.final_consensus = final.consensus_score
            branch.total_messages = sum(len(s.transcript) for s in branch.snapshots)

        # Compute divergence from parent
        if branch.parent_branch:
            branch.divergence_from_parent = self._compute_divergence(
                branch,
                self.branches[branch.parent_branch]
            )

        self._save_branch(branch)
        return branch

    def _create_engine(self, initial_state: Optional[Dict[str, Any]]):
        """Create a debate engine for counterfactual simulation."""
        if self._debate_engine:
            # Clone the engine with initial state
            return copy.deepcopy(self._debate_engine)
        else:
            # Create mock engine
            return MockCounterfactualEngine(initial_state)

    def _snapshot_to_state(self, snapshot: DebateSnapshot) -> Dict[str, Any]:
        """Convert snapshot to engine state format."""
        return {
            "positions": snapshot.positions,
            "beliefs": snapshot.beliefs,
            "emotional_states": snapshot.emotional_states,
            "energy_levels": snapshot.energy_levels,
            "transcript": snapshot.transcript,
            "coalitions": snapshot.coalitions,
            "consensus": snapshot.consensus_score,
            "tension": snapshot.tension_level,
            "round": snapshot.round_number,
        }

    async def _apply_intervention(self, engine, intervention: Intervention):
        """Apply an intervention to the debate engine."""
        int_type = intervention.intervention_type

        if int_type == "inject_message":
            await engine.inject_message(
                speaker=intervention.speaker,
                content=intervention.content,
                targets=intervention.target_agents,
            )

        elif int_type == "set_position":
            for agent in intervention.target_agents:
                await engine.set_position(agent, intervention.new_position)

        elif int_type == "form_coalition":
            await engine.form_coalition(intervention.coalition_members)

        elif int_type == "emotional_trigger":
            for agent in intervention.target_agents:
                await engine.trigger_emotion(
                    agent,
                    intervention.parameters.get("emotion", "frustrated")
                )

        elif int_type == "reveal_information":
            await engine.reveal_information(
                content=intervention.content,
                targets=intervention.target_agents,
            )

        else:
            logger.warning(f"Unknown intervention type: {int_type}")

    def _compute_divergence(
        self,
        branch: DebateBranch,
        parent: DebateBranch
    ) -> float:
        """Compute how much a branch has diverged from parent."""
        if not branch.final_positions or not parent.final_positions:
            return 0.0

        # Position-based divergence
        position_diffs = []
        for agent in branch.final_positions:
            if agent in parent.final_positions:
                diff = abs(branch.final_positions[agent] - parent.final_positions[agent])
                position_diffs.append(diff)

        if not position_diffs:
            return 0.0

        return float(np.mean(position_diffs))

    def compare_branches(
        self,
        baseline_id: str,
        counterfactual_ids: List[str],
    ) -> WhatIfAnalysis:
        """
        Compare multiple counterfactual branches.

        Args:
            baseline_id: The baseline (actual) branch
            counterfactual_ids: Counterfactual branches to compare

        Returns:
            WhatIfAnalysis with comprehensive comparison
        """
        if baseline_id not in self.branches:
            raise ValueError(f"Baseline branch not found: {baseline_id}")

        baseline = self.branches[baseline_id]
        analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        analysis = WhatIfAnalysis(
            analysis_id=analysis_id,
            baseline_branch=baseline_id,
            counterfactual_branches=counterfactual_ids,
        )

        for cf_id in counterfactual_ids:
            if cf_id not in self.branches:
                continue

            cf = self.branches[cf_id]

            # Position deltas
            analysis.position_deltas[cf_id] = {}
            for agent in baseline.final_positions:
                if agent in cf.final_positions:
                    delta = cf.final_positions[agent] - baseline.final_positions[agent]
                    analysis.position_deltas[cf_id][agent] = delta

            # Consensus delta
            analysis.consensus_delta[cf_id] = cf.final_consensus - baseline.final_consensus

            # Find divergence point
            divergence_round = self._find_divergence_point(baseline, cf)
            analysis.divergence_points[cf_id] = divergence_round

            # Identify cause of divergence
            if divergence_round > 0:
                cause = self._identify_divergence_cause(baseline, cf, divergence_round)
                analysis.divergence_causes[cf_id] = cause

            # Effect size (Cohen's d for position changes)
            if analysis.position_deltas[cf_id]:
                deltas = list(analysis.position_deltas[cf_id].values())
                pooled_std = np.std(list(baseline.final_positions.values()) +
                                    list(cf.final_positions.values()))
                if pooled_std > 0:
                    analysis.effect_sizes[cf_id] = np.mean(deltas) / pooled_std

        # Generate summary
        analysis.summary = self._generate_comparison_summary(analysis)
        analysis.recommendations = self._generate_recommendations(analysis)

        return analysis

    def _find_divergence_point(
        self,
        baseline: DebateBranch,
        counterfactual: DebateBranch
    ) -> int:
        """Find the round where branches significantly diverge."""
        min_rounds = min(len(baseline.snapshots), len(counterfactual.snapshots))

        for i in range(min_rounds):
            base_snap = baseline.snapshots[i]
            cf_snap = counterfactual.snapshots[i]

            # Check position divergence
            for agent in base_snap.positions:
                if agent in cf_snap.positions:
                    if abs(base_snap.positions[agent] - cf_snap.positions[agent]) > 0.1:
                        return base_snap.round_number

        return 0  # No significant divergence

    def _identify_divergence_cause(
        self,
        baseline: DebateBranch,
        counterfactual: DebateBranch,
        divergence_round: int
    ) -> str:
        """Identify what caused the divergence."""
        # Check interventions at divergence point
        for intv in counterfactual.interventions:
            if intv.round_number <= divergence_round:
                return f"Intervention: {intv.intervention_type} at round {intv.round_number}"

        # Check for message differences
        base_snap = baseline.get_snapshot(divergence_round)
        cf_snap = counterfactual.get_snapshot(divergence_round)

        if base_snap and cf_snap:
            base_messages = set(m.get("speaker", "") for m in base_snap.transcript)
            cf_messages = set(m.get("speaker", "") for m in cf_snap.transcript)

            new_speakers = cf_messages - base_messages
            if new_speakers:
                return f"Different speakers: {', '.join(new_speakers)}"

        return "Unknown cause"

    def _generate_comparison_summary(self, analysis: WhatIfAnalysis) -> str:
        """Generate a narrative summary of the comparison."""
        parts = [f"Comparison of {len(analysis.counterfactual_branches)} counterfactual scenarios:\n"]

        for cf_id in analysis.counterfactual_branches:
            if cf_id in analysis.consensus_delta:
                delta = analysis.consensus_delta[cf_id]
                direction = "increased" if delta > 0 else "decreased"
                parts.append(f"- {cf_id}: Consensus {direction} by {abs(delta):.2f}")

                if cf_id in analysis.divergence_points:
                    parts.append(f"  Diverged at round {analysis.divergence_points[cf_id]}")

                if cf_id in analysis.divergence_causes:
                    parts.append(f"  Cause: {analysis.divergence_causes[cf_id]}")

        return "\n".join(parts)

    def _generate_recommendations(self, analysis: WhatIfAnalysis) -> List[str]:
        """Generate actionable recommendations from the analysis."""
        recommendations = []

        # Find most effective interventions
        best_cf = None
        best_effect = 0

        for cf_id, delta in analysis.consensus_delta.items():
            if delta > best_effect:
                best_effect = delta
                best_cf = cf_id

        if best_cf and best_effect > 0.1:
            recommendations.append(
                f"Consider intervention from {best_cf} which improved consensus by {best_effect:.2f}"
            )

        # Check for harmful interventions
        for cf_id, delta in analysis.consensus_delta.items():
            if delta < -0.1:
                recommendations.append(
                    f"Avoid intervention from {cf_id} which reduced consensus by {abs(delta):.2f}"
                )

        return recommendations

    def _save_branch(self, branch: DebateBranch):
        """Save branch to storage."""
        filepath = self.storage_dir / f"{branch.branch_id}.json"
        with open(filepath, "w") as f:
            json.dump(asdict(branch), f, indent=2, default=str)

    def load_branch(self, branch_id: str) -> DebateBranch:
        """Load branch from storage."""
        filepath = self.storage_dir / f"{branch_id}.json"
        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)
            branch = DebateBranch(**data)
            self.branches[branch_id] = branch
            return branch
        raise ValueError(f"Branch not found: {branch_id}")

    async def run_parallel_what_if(
        self,
        baseline_snapshot: str,
        interventions_sets: List[List[Intervention]],
        max_rounds: int = 10,
    ) -> WhatIfAnalysis:
        """
        Run multiple counterfactual scenarios in parallel.

        Args:
            baseline_snapshot: Snapshot to branch from
            interventions_sets: List of intervention sets to test
            max_rounds: Rounds to simulate per branch

        Returns:
            WhatIfAnalysis comparing all scenarios
        """
        # Create baseline branch
        baseline = self.create_branch(
            fork_snapshot=baseline_snapshot,
            branch_id="baseline"
        )

        # Create counterfactual branches
        cf_branches = []
        for i, interventions in enumerate(interventions_sets):
            branch = self.create_branch(
                parent_branch="baseline",
                fork_snapshot=baseline_snapshot,
                interventions=interventions,
                branch_id=f"cf_{i}"
            )
            cf_branches.append(branch.branch_id)

        # Run all branches in parallel
        tasks = [
            self.run_branch(baseline.branch_id, max_rounds)
        ]
        for cf_id in cf_branches:
            tasks.append(self.run_branch(cf_id, max_rounds))

        await asyncio.gather(*tasks)

        # Compare results
        return self.compare_branches("baseline", cf_branches)


class MockCounterfactualEngine:
    """Mock engine for testing counterfactual analysis."""

    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        self.state = initial_state or {
            "positions": {f"agent_{i}": 0.5 for i in range(6)},
            "beliefs": {},
            "emotional_states": {},
            "energy_levels": {f"agent_{i}": 1.0 for i in range(6)},
            "transcript": [],
            "coalitions": [],
            "consensus": 0.0,
            "tension": 0.3,
            "round": 0,
        }

    async def run_round(self):
        """Simulate a round."""
        self.state["round"] += 1

        # Random position drift
        for agent in self.state["positions"]:
            drift = np.random.normal(0, 0.05)
            self.state["positions"][agent] += drift
            self.state["positions"][agent] = np.clip(self.state["positions"][agent], 0, 1)

        # Add message
        speaker = np.random.choice(list(self.state["positions"].keys()))
        self.state["transcript"].append({
            "round": self.state["round"],
            "speaker": speaker,
            "content": f"[Simulated message from {speaker}]"
        })

        # Update consensus
        positions = list(self.state["positions"].values())
        self.state["consensus"] = 1 - np.std(positions) * 2

    def get_state(self) -> Dict[str, Any]:
        return self.state.copy()

    async def inject_message(self, speaker: str, content: str, targets: List[str]):
        self.state["transcript"].append({
            "round": self.state["round"],
            "speaker": speaker,
            "content": content,
            "targets": targets,
            "injected": True,
        })

    async def set_position(self, agent: str, position: float):
        if agent in self.state["positions"]:
            self.state["positions"][agent] = position

    async def form_coalition(self, members: List[str]):
        self.state["coalitions"].append({
            "members": members,
            "formed_at": self.state["round"]
        })

    async def trigger_emotion(self, agent: str, emotion: str):
        self.state["emotional_states"][agent] = emotion

    async def reveal_information(self, content: str, targets: List[str]):
        self.state["transcript"].append({
            "round": self.state["round"],
            "speaker": "SYSTEM",
            "content": f"[REVEALED] {content}",
            "targets": targets,
        })


if __name__ == "__main__":
    async def demo():
        engine = CounterfactualEngine()

        # Create initial state
        initial_state = {
            "positions": {
                "sam_altman": 0.7,
                "yann_lecun": 0.3,
                "yoshua_bengio": 0.6,
            },
            "consensus": 0.4,
        }

        # Create snapshot
        snapshot = engine.create_snapshot(initial_state, round_number=5)

        # Create baseline
        baseline = engine.create_branch(
            fork_snapshot=snapshot.snapshot_id,
            branch_id="baseline"
        )

        # Create counterfactual with intervention
        intervention = Intervention(
            intervention_id="int_1",
            round_number=6,
            intervention_type="inject_message",
            speaker="geoffrey_hinton",
            content="I believe we should reconsider the safety implications.",
            target_agents=["sam_altman", "yann_lecun"]
        )

        cf_branch = engine.create_branch(
            parent_branch="baseline",
            fork_snapshot=snapshot.snapshot_id,
            interventions=[intervention],
            branch_id="cf_safety_warning"
        )

        # Run branches
        await engine.run_branch("baseline", max_rounds=5)
        await engine.run_branch("cf_safety_warning", max_rounds=5)

        # Compare
        analysis = engine.compare_branches("baseline", ["cf_safety_warning"])

        print("=== Counterfactual Analysis ===")
        print(analysis.summary)
        print("\nRecommendations:")
        for rec in analysis.recommendations:
            print(f"  - {rec}")

    asyncio.run(demo())
