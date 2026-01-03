Title: Yield Management with Capacity and SLA Constraints
Domain: economy
Mode: careful

Prompt:
- Build a yield management policy (prices/allocations) under capacity and SLA constraints (compute, support, logistics).
- Formulate as a constrained optimization (e.g., MIP/LP), embedding fairness and minimum service guarantees.
- Simulate peak vs off‑peak policies (time‑of‑day/region) with guardrails.
- Add reservations/overbooking rules and penalties; ensure customer rights.
- Provide real‑time controls, prediction of demand spikes, and throttles.
- Publish SLOs, transparency notes, and opt‑out paths where applicable.
- Instrument for audits and post‑mortems.
- Train operators and review.

Outputs:
- POLICY optimization model with constraints and solver plan.
- SIMULATION results and guardrails.
- SLOs, throttles, and audit hooks.

Assumptions:
- Capacity metrics are observable in real time; penalties enforceable.
- Solver/runtime fits operational cadence; fallback heuristics available.
- Users accept transparent time‑varying policies when communicated well.

Metrics to watch:
- SLA adherence, queue delays, abandonment; margin per unit capacity.
- Throttle activations, fairness flags, complaints.
- Forecast error and control loop stability.

