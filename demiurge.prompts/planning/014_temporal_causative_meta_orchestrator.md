Title: Temporal‑Causative Meta‑Orchestrator
Domain: planning
Mode: careful

Prompt:
- Design a meta‑orchestrator that coordinates multi‑domain plans across time (Aion design phases → Chronos execution windows).
- Model causal structure explicitly: build DAGs of interventions→signals→outcomes with lags, feedbacks, and guardrails.
- Encode temporal operators (before/after/during/unless) and conflict resolution (priority, veto, freeze windows).
- Allocate compute/energy/memory budgets over time; enforce R* (reversibility reserve) at each step.
- Define calendars: strategic (quarterly), operational (weekly), realtime (minute/hour) with hand‑offs and SLAs.
- Integrate ABAC modes by phase (observe→warn→enforce) and approvals at gates; attach kill criteria to milestones.
- Instrument observability: correlation IDs stitched to Chronicle, metric bands, and anomaly tripwires.
- Provide rollback/branching protocol using snapshots; prove temporal consistency on merge.
- Publish API contracts for planners, simulators, auditors, and explainers; include schema for temporal constraints.
- Add failure playbooks for temporal deadlocks, priority inversions, and cascading delays.

Outputs:
- ORCHESTRATOR spec (APIs, DAG schema, calendars, guards).
- TEMPORAL plan templates (milestones, gates, rollback, branches).
- OBSERVABILITY and Chronicle linkage plan with queries.

Assumptions:
- Causal edges are empirically testable or well‑grounded priors; time lags are estimable.
- Teams can adhere to time windows and gate approvals; clocks are synchronized.
- Snapshot/restore and branch/merge are available and audited.

Metrics to watch:
- Gate success/abort rates, latency to decision, rollback frequency, violation of guards.
- Temporal drift (schedule slip), deadlock incidence, and recovery time.
- Outcome lift vs plan with attribution to causal paths; audit coverage.

Procedure:
- Define canonical time scales and name them (T0 strategic, T1 tactical, T2 realtime).
- Draft the causal DAG: enumerate nodes (actions, mediators, outcomes) and edges with lag annotations (Δt).
- Attach guards to nodes/edges (preconditions, invariants, approvals, kill criteria).
- Allocate budgets per time window; reserve R* (reversibility) and S (slack) buffers explicitly.
- Generate milestone plan with freeze/gate points and snapshot schedule; link correlation_ids.
- Configure observability (topic names, sampling, alarms) and Chronicle query templates.
- Simulate happy/sad/edge paths; record rollback branches and merge rules.
- Publish API contracts for sub‑planners (schema below) and register them.
- Run a dry‑run (tabletop) across one cycle; capture defects and fix contracts.
- Enter observe→warn→enforce phasing with exit/rollback criteria.

Interfaces (high‑level APIs):
- POST /orchestrator/plans {id, timescale, dag, guards, budgets, milestones}
- POST /orchestrator/run {plan_id, window} → {gate_status, diffs, metrics}
- GET  /orchestrator/state {plan_id} → {checkpoint, health, debts}
- POST /orchestrator/rollback {plan_id, to_checkpoint, reason}
- POST /orchestrator/freeze {scope, until, approvals}

Schema (sketch):
```
dag: {
  nodes: [{ id, kind: action|mediator|outcome, lag: "PT0S..P7D", guards: [guard_id] }],
  edges: [{ from, to, lag: "PT1H", note }]
}
guards: [{ id, type: approval|invariant|kill, expr, owner }]
milestones: [{ id, at: "2026-01-15T10:00Z", gate: { approvals: [role], kill_criteria: [id] } }]
budgets: { compute: { T0: 100, T1: 50 }, memory: {...}, energy: {...} }
```

Acceptance criteria:
- All actions/edges guarded; all lags annotated; gates have owners and approvals.
- Snapshots/branches defined with restore tests; Chronicle queries verified.
- Dry‑run completed with recorded defects and remediations.

Checklists:
- Clocks synced; correlation_ids end‑to‑end; privacy tiers applied.
- Ombuds/ethics sign‑off for high‑risk edges; freeze switch verified.
- Kill criteria alarmed and tested; rollback drills executed.
