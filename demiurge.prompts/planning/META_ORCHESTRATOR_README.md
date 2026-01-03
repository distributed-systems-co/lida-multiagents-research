Title: Meta‑Orchestrator Pack — 4‑Week Execution Plan

Week 1 — Foundations
- Draft temporal scales (T0/T1/T2) and RACI; sync clocks and correlation_ids.
- Author initial causal DAG with lags; attach guards and budgets; pick observability topics.
- Create Chronicle graph namespace; ingest a small pilot; run basic queries.
- Tabletop: one milestone cycle; record defects; fix contracts.

Week 2 — Simulation + Evidence
- Implement event algebra + causality kernel POC; adversarial schedule tests; IRF stability checks.
- Flesh trace graph schema; implement redaction tiers; index and query performance.
- Build counterfactual replay POC; package evidence; sign and archive.
- Freeze low‑risk windows; launch observe mode.

Week 3 — Gates + Safety
- Turn on warn mode for selected gates; attach lag‑aware policy surfaces (DL/IRF models).
- Publish risk propagation maps; run drills; reduce alarm fatigue.
- Add temporal policy diffs with sunsets; simulate boundary conflicts.
- Human‑facing temporal maps; accessibility and localization pass.

Week 4 — Enforce + Merge
- Promote to enforce for a subset; SLOs for rollback/merge; timewarp invariants tests.
- Chronicle coverage >95%; causal labeling audits; publish transparency summaries.
- Post‑mortem on defects; acceptance review; schedule quarterly re‑estimation and drills.

Artifacts to produce
- Orchestrator plan (schema‑valid); kernel spec; DAG + guards; budgets; calendars; Chronicle cookbook; audit tools.

Exit criteria
- Validated plan; gates and freezes tested; replay and counterfactuals functional; human maps published; audits pass.

