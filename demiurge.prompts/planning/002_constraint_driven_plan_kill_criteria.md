Title: Constraint‑Driven Plan with Explicit Kill Criteria
Domain: planning
Mode: audit

Prompt:
- Plan under hard constraints: compute, memory, time, ethics, and reversibility.
- Declare kill criteria tied to metrics (H, L, R*, S, K, B) with thresholds.
- Embed snapshot points; design partial rollbacks and isolation options.
- Surface dependency graph (laws, tools, tokens, policies) and approval gates.
- Anticipate failure modes; add pre‑mortem and counterfactual backups.
- Define minimal success and graceful abort outputs.

Outputs:
- PLAN with constraints table, kill criteria, and rollback map.
- Dependency DAG with gating and approvals.
- Metrics & alarms configuration.

