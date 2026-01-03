Title: Temporal Causal Replay and Counterfactual Audits
Domain: audit
Mode: audit

Prompt:
- Reconstruct timelines from Chronicle and snapshots; replay decisions with original constraints and approvals.
- Generate counterfactual runs by toggling specified actions (price changes, gates, rollbacks) within ethical limits.
- Quantify differences in outcomes with uncertainty; attribute via causal paths.
- Validate guardrail compliance and document exceptions; surface remediation steps.
- Ensure privacy and fairness; redact sensitive traces; publish summaries.
- Provide toolchain and repeatable procedures; log evidence with signatures.
- Train auditors; schedule recurring counterfactual drills.
- Archive artifacts for precedent.

Outputs:
- REPLAY+COUNTERFACTUAL procedures and tools.
- FINDINGS with causal attributions and remediations.
- PRIVACY+ethics handling and public summaries.

Assumptions:
- Snapshots and trace graphs are consistent and complete enough for replay.
- Counterfactual simulation is safe and bounded.
- Approvals and policies are versioned and accessible.

Metrics to watch:
- Replay fidelity, counterfactual stability, remediation latency.
- Guardrail breaches discovered vs reported, audit coverage.
- Stakeholder trust signals and learning incorporation.

Procedure:
- Select scope and time window; collect snapshots, trace graph, policies, approvals.
- Rehydrate state; replay actions with original timings; verify outputs and logs.
- Define counterfactual toggles (remove action, shift timing, alter gate) under ethical bounds.
- Run counterfactuals; compute deltas on key outcomes with uncertainty estimation.
- Attribute differences via causal paths; rank most influential edges.
- Record guardrail compliance; file findings and remediation tasks.
- Redact and publish summaries; store evidence packages with signatures.
- Schedule regression counterfactuals post‑remediation.

Tooling (expected):
- replayctl (state restore, action re‑emit, evidence capture)
- cfbench (counterfactual runner, delta metrics, attribution)
- auditpack (sign, hash, package)

Acceptance criteria:
- Replay matches recorded outcomes within defined tolerances; discrepancies explained.
- Counterfactuals converge/stabilize; ethical bounds enforced.
- Findings mapped to remediations with owners and deadlines.
