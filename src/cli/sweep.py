#!/usr/bin/env python3
"""
Parallel sweep runner for LIDA experiments.

Highlights
- Generate a run matrix (scenarios × topics × seeds/replicas)
- Limit concurrency so we can fire off many simulations without melting APIs
- Rich progress bars plus a compact summary table + JSON summary

Usage (once wired into `lida sweep`):
    python -m src.cli.sweep --scenarios ai_xrisk budget \
        --topics ai_pause open_source_ai --replicas 3 --parallel 4 --no-live
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from .config_loader import ConfigLoader
from .runner import ExperimentConfig, ExperimentRunner

# Paths / console helpers
CONSOLE = Console()
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _slugify(text: str) -> str:
    """Safe slug for filenames/ids."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", text).strip("-").lower()
    return cleaned or "run"


def _default_output_root() -> Path:
    """Default sweep output root."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "experiment_results" / f"sweep_{ts}"


def _resolve_scenario(raw: str) -> Path:
    """
    Resolve a scenario name/path to an existing YAML file.
    Tries common locations: absolute, project root, scenarios/, scenarios/presets/, scenarios/campaigns/.
    """
    candidates = []
    raw_path = Path(raw)

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(PROJECT_ROOT / raw)
        if raw_path.suffix != ".yaml":
            candidates.append(PROJECT_ROOT / "scenarios" / f"{raw}.yaml")
            candidates.append(PROJECT_ROOT / "scenarios" / "presets" / f"{raw}.yaml")
            candidates.append(PROJECT_ROOT / "scenarios" / "campaigns" / f"{raw}.yaml")
        candidates.append(PROJECT_ROOT / "scenarios" / raw)
        candidates.append(PROJECT_ROOT / "scenarios" / "presets" / raw)
        candidates.append(PROJECT_ROOT / "scenarios" / "campaigns" / raw)

    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    raise FileNotFoundError(f"Could not resolve scenario '{raw}' to a YAML file")


def _expand_scenarios(inputs: Iterable[str]) -> List[Path]:
    """
    Expand scenario inputs, honoring globs.
    """
    paths: List[Path] = []
    for raw in inputs:
        if any(ch in raw for ch in "*?[]"):
            pattern = raw if Path(raw).is_absolute() else str(PROJECT_ROOT / raw)
            for match in Path().glob(pattern):
                if match.is_file():
                    paths.append(match.resolve())
        else:
            paths.append(_resolve_scenario(raw))

    # Deduplicate while preserving order
    seen = set()
    unique_paths = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)
    return unique_paths


@dataclass
class SweepJob:
    """Single experiment run in the sweep matrix."""

    scenario_path: Path
    topic: Optional[str]
    seed: Optional[int]
    config: ExperimentConfig

    def label(self) -> str:
        topic_part = self.topic or self.config.topic or "default"
        seed_part = f"s{self.seed}" if self.seed is not None else "seedless"
        return f"{self.scenario_path.stem}:{topic_part}:{seed_part}"


def build_jobs(
    scenario_inputs: List[str],
    topics: Optional[List[str]],
    seeds: Optional[List[int]],
    replicas: int,
    rounds: Optional[int],
    live: Optional[bool],
    model: Optional[str],
    output_root: Optional[Path],
) -> list[SweepJob]:
    """
    Create SweepJob list from CLI args.
    """
    scenario_paths = _expand_scenarios(scenario_inputs)
    if not scenario_paths:
        raise ValueError("No scenarios matched the provided inputs")

    seed_list = seeds if seeds else list(range(replicas))
    if not seed_list:
        seed_list = [None]

    topic_list = topics or [None]

    output_base = Path(output_root) if output_root else _default_output_root()
    loader = ConfigLoader(base_dir=str(PROJECT_ROOT))

    jobs: list[SweepJob] = []
    for scenario_path in scenario_paths:
        config_dict = loader.load(str(scenario_path))

        for topic in topic_list:
            for seed in seed_list:
                cfg = ExperimentConfig.from_yaml(config_dict)
                cfg.scenario_file = str(scenario_path)

                if topic:
                    cfg.topic = topic
                if rounds:
                    cfg.max_rounds = rounds
                if live is not None:
                    cfg.live_mode = live
                if model:
                    cfg.model_override = model
                if seed is not None:
                    cfg.random_seed = seed

                topic_dir = _slugify(cfg.topic or topic or "default")
                job_output = output_base / scenario_path.stem / topic_dir / (
                    f"seed-{seed}" if seed is not None else "seedless"
                )
                cfg.output_dir = str(job_output)
                cfg.name = f"{scenario_path.stem}-{topic_dir}-s{seed if seed is not None else 'na'}"

                jobs.append(SweepJob(scenario_path, topic or cfg.topic, seed, cfg))

    return jobs


async def _run_single_job(
    job: SweepJob,
    semaphore: asyncio.Semaphore,
    progress: Progress,
    task_id: int,
) -> dict:
    """
    Run one experiment with optional concurrency limit.
    Returns lightweight result dict (no transcripts) to keep memory down.
    """

    def on_progress(metrics):
        progress.update(
            task_id,
            completed=metrics.current_round or 0,
            total=metrics.total_rounds or job.config.max_rounds or 1,
            description=f"{job.label()} r{metrics.current_round}/{metrics.total_rounds}",
        )

    async with semaphore:
        try:
            runner = ExperimentRunner(job.config, progress_callback=on_progress)
            full_result = await runner.run()
            metrics = full_result.get("metrics", {}) if isinstance(full_result, dict) else {}
            progress.update(
                task_id,
                completed=progress.tasks[task_id].total or metrics.get("total_rounds", 1),
                description=f"{job.label()} ✓",
            )
            light_result = {
                "experiment_id": full_result.get("experiment_id"),
                "status": full_result.get("status"),
                "metrics": metrics,
                "config": {
                    "scenario_file": job.config.scenario_file,
                    "topic": job.config.topic,
                    "max_rounds": job.config.max_rounds,
                    "live_mode": job.config.live_mode,
                    "model_override": job.config.model_override,
                    "random_seed": job.config.random_seed,
                    "output_dir": job.config.output_dir,
                },
            }
            return {"job": job, "status": "ok", "result": light_result}
        except Exception as exc:  # noqa: BLE001
            progress.update(task_id, description=f"{job.label()} ✗")
            return {
                "job": job,
                "status": "error",
                "error": str(exc),
                "result": None,
            }


async def execute_jobs(jobs: List[SweepJob], parallel: int) -> List[dict]:
    """
    Run all jobs with bounded concurrency and pretty progress bars.
    """
    semaphore = asyncio.Semaphore(max(1, parallel))
    results: List[dict] = []

    columns = [
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    ]

    with Progress(*columns, console=CONSOLE, expand=True, transient=False) as progress:
        tasks = []
        for job in jobs:
            task_id = progress.add_task(job.label(), total=job.config.max_rounds or 1)
            tasks.append(
                asyncio.create_task(_run_single_job(job, semaphore, progress, task_id))
            )

        for coro in asyncio.as_completed(tasks):
            results.append(await coro)

    return results


def _render_summary(results: List[dict], output_root: Path) -> Path:
    """Print a compact table and persist a JSON summary."""
    table = Table(
        "Run",
        "Status",
        "Rounds",
        "Messages",
        "LLM $",
        "Output",
        title="Sweep Summary",
        show_lines=False,
    )

    summary_rows = []
    success = 0

    for res in sorted(results, key=lambda r: r["job"].label()):
        job = res["job"]
        status = res["status"]
        metrics = res.get("result", {}).get("metrics", {}) if res.get("result") else {}
        rounds = metrics.get("current_round") or metrics.get("total_rounds") or "-"
        messages = metrics.get("total_messages", "-")
        cost = metrics.get("llm_cost_usd", 0.0)
        cost_display = f"${cost:.2f}" if isinstance(cost, (int, float)) else "-"

        if status == "ok":
            success += 1

        table.add_row(
            job.label(),
            status,
            str(rounds),
            str(messages),
            cost_display,
            job.config.output_dir,
        )

        summary_rows.append(
            {
                "run": job.label(),
                "status": status,
                "topic": job.topic,
                "seed": job.seed,
                "scenario_file": job.config.scenario_file,
                "output_dir": job.config.output_dir,
                "metrics": {
                    "current_round": metrics.get("current_round"),
                    "total_rounds": metrics.get("total_rounds"),
                    "total_messages": metrics.get("total_messages"),
                    "llm_cost_usd": metrics.get("llm_cost_usd"),
                    "elapsed_seconds": metrics.get("elapsed_seconds"),
                },
                "config": res.get("result", {}).get("config") if res.get("result") else {},
                "error": res.get("error"),
            }
        )

    CONSOLE.print(table)

    summary = {
        "created_at": datetime.now().isoformat(),
        "output_root": str(output_root),
        "total_runs": len(results),
        "succeeded": success,
        "failed": len(results) - success,
        "results": summary_rows,
    }

    summary_path = output_root / "sweep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    CONSOLE.print(f"[green]Summary saved to[/green] {summary_path}")
    return summary_path


async def run_sweep(args) -> None:
    """
    Entry point used by CLI.
    """
    output_base = Path(args.output) if args.output else _default_output_root()
    jobs = build_jobs(
        scenario_inputs=args.scenarios,
        topics=args.topics,
        seeds=args.seeds,
        replicas=args.replicas,
        rounds=args.rounds,
        live=args.live,
        model=args.model,
        output_root=output_base,
    )

    CONSOLE.print(
        f"Planned {len(jobs)} runs "
        f"(parallel={args.parallel}) -> {output_base}"
    )

    if args.dry_run:
        for job in jobs:
            CONSOLE.print(
                f"- {job.label()} "
                f"rounds={job.config.max_rounds} "
                f"live={'on' if job.config.live_mode else 'off'} "
                f"model={job.config.model_override or 'scenario-default'} "
                f"output={job.config.output_dir}"
            )
        return

    results = await execute_jobs(jobs, args.parallel)
    _render_summary(results, output_base)


def build_arg_parser(parser):
    """
    Allow reuse of argument definitions from main CLI.
    """
    parser.add_argument(
        "-s",
        "--scenarios",
        nargs="+",
        required=True,
        help="Scenario names/paths or globs (e.g. scenarios/presets/*.yaml)",
    )
    parser.add_argument(
        "-t",
        "--topics",
        nargs="+",
        help="Override topics (space-separated list)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        help="Override max rounds for all runs",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Explicit random seeds (overrides --replicas)",
    )
    parser.add_argument(
        "-r",
        "--replicas",
        type=int,
        default=1,
        help="Replicate each scenario/topic this many times (ignored if --seeds set)",
    )
    parser.add_argument(
        "-p",
        "--parallel",
        type=int,
        default=2,
        help="Max concurrent runs",
    )
    live_group = parser.add_mutually_exclusive_group()
    live_group.add_argument(
        "--live",
        dest="live",
        action="store_true",
        help="Force live LLM calls",
    )
    live_group.add_argument(
        "--no-live",
        dest="live",
        action="store_false",
        help="Disable LLM calls (fast mock mode)",
    )
    parser.add_argument(
        "--model",
        help="Override model for all agents (e.g. anthropic/claude-sonnet-4.1)",
    )
    parser.add_argument(
        "--output",
        help="Base output directory (default: experiment_results/sweep_<timestamp>)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the run matrix without executing",
    )
    parser.set_defaults(live=None)
    return parser


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run parameter sweeps for LIDA experiments")
    build_arg_parser(parser)
    args = parser.parse_args()
    asyncio.run(run_sweep(args))


if __name__ == "__main__":
    main()
