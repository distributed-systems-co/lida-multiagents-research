Title: Causal Mediation and Distributed Lag Models
Domain: metrics
Mode: careful

Prompt:
- Estimate mediation effects (A→M→Y) and distributed lag responses of interventions on outcomes; define DAGs and identification assumptions.
- Use DLM/VAR/IRF where appropriate; compare to simpler heuristics; quantify uncertainty.
- Produce policy‑relevant summaries: immediate, short‑lag, and long‑lag effects; show tradeoffs.
- Validate with backtests and sensitivity to unobserved confounders.
- Publish visualizations (impulse responses) with CIs; avoid overreach in narration.
- Integrate into orchestrator gates and resource pacing.
- Provide documentation and training.
- Schedule re‑estimation cadence.

Outputs:
- MODEL specs, identification notes, and IRFs.
- POLICY surfaces with lag‑aware guidance.
- VALIDATION and sensitivity reports.

Assumptions:
- Time series quality and coverage sufficient; stationarity issues manageable.
- DAG assumptions are credible; instrumentation feasible.
- Stakeholders accept uncertainty bands.

Metrics to watch:
- Forecast/backtest errors; calibration of lag effects.
- Policy adherence and impact alignment; guardrail breaches.
- Drift signals and re‑estimation outcomes.

Procedure:
- Draft DAG with mediator(s); declare identification strategy (front‑door/back‑door/instrument).
- Choose model class (DLM/VAR) and priors; define horizon and granularity.
- Estimate effects; compute IRFs and mediation shares with CIs.
- Validate with placebo and sensitivity (bounds on bias, M‑values).
- Generate lag‑aware policy surfaces (short/med/long) and attach gates.
- Build dashboards; train consumers on interpretation limits.
- Schedule re‑fits; monitor covariate shift and structural breaks.

Artifacts:
- DAG diagram; identification note; IRF plots; mediation table; policy memo.

Acceptance criteria:
- Identification justified; sensitivity documented; forecasts calibrated.
- Policy surfaces approved; dashboards live; retrain cadence set.
