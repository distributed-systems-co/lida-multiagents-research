Title: Meta‑Planning — Design the Planning System
Domain: planning
Mode: careful

Prompt:
- Specify a planning kernel with inputs, internal state, and outputs under MCCM.
- Define heuristics, search strategy, and anytime behavior with bounded compute.
- Encode safety envelopes and approvals into the planner itself.
- Support simulation rollouts, counterfactuals, and value of information.
- Expose inspection hooks for audit with privacy budget controls.
- Provide upgrade/migration path and test harness.

Outputs:
- SPEC of planner APIs, states, and invariants.
- TEST plan for correctness, performance, and safety.
- ROLLOUT plan with observe→warn→enforce gates.

