Title: Constraint Programming Planner with Resource Leveling
Domain: planning
Mode: careful

Prompt:
- Formulate a constraint programming (CP-SAT) model for the plan: tasks, durations, precedence, and resource constraints.
- Include calendars and blackouts; model multi-skill resources and alternate routings.
- Add leveling to minimize peaks; trade off makespan vs WIP vs context switches.
- Encode risk buffers (slack S) and reversibility R* checkpoints as explicit variables.
- Penalize gate failures and late milestones; add soft constraints with weights.
- Provide solver config (limits, gaps, restarts) and fallback heuristics.
- Output a baseline schedule, a “robust” schedule under perturbations, and a candidate freeze window plan.
- Generate deltas and Chronicle entries for approvals; wire regression tests for constraints.

Outputs:
- CP model spec, solver settings, and baseline/robust schedules.
- RESOURCE leveling report and gate timing with buffers.
- TESTS for constraints and regression harness.

Assumptions:
- Task graph, durations, and resource capacities are available or reasonably estimable.
- Calendars and blackouts are known; approvals can be scheduled.
- Solving time fits the cadence; heuristics are acceptable if optimality gaps persist.

Metrics to watch:
- Makespan, peak resource utilization, gate on-time %, replan frequency.
- Slack S consumption over time; R* distance at checkpoints.
- Constraint violations in regression, schedule stability index.

