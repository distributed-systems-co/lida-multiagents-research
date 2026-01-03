Title: Multi‑Scale Temporal Program Orchestration
Domain: planning
Mode: careful

Prompt:
- Orchestrate a program spanning quarters→sprints→daily ops, with explicit cross‑scale contracts and escalations.
- Define interfaces between strategic intents, tactical workstreams, and realtime controls; include pre/postconditions.
- Map causality across scales (policy→mechanism→metric) with lags and mediation effects; add tests at each boundary.
- Allocate slack S and reversibility R* by scale; track consumption and replenishment.
- Create cadence artifacts (QBRs, weekly reviews, daily standups) tied to Chronicle evidence.
- Embed ethics/safety gates at each scale; require ombuds involvement for high‑risk changes.
- Provide scenario branches and freeze points; rehearse with tabletop exercises.
- Publish role matrix and approvals; instrument operator load and alarm fatigue protections.

Outputs:
- PROGRAM map with cross‑scale interfaces and gates.
- CADENCE calendar, artifacts, and Chronicle queries.
- SCENARIO and freeze playbooks with drills.

Assumptions:
- Teams accept cross‑scale contracts and review discipline.
- Evidence capture and time sync are reliable.
- Resource budgets can be sliced by scale without starvation.

Metrics to watch:
- Cross‑scale defect leakage, escalation latency, gate rejections/overrides.
- Slack S trend, R* sufficiency, and alarm fatigue indicators.
- Outcome alignment (top‑down intents vs bottom‑up metrics).

Procedure:
- Identify scopes and owners per scale; define RACI and escalation ladders.
- Write interface contracts: inputs/outputs, pre/postconditions, evidence required at handoff.
- Build scale‑bridging causal maps (intent→mechanism→metric) with lags and mediators.
- Assign budgets and reserves (S, R*) per scale; define replenishment policies.
- Stand up cadences: QBR agenda, weekly review packet, daily ops dashboard.
- Pre‑define freeze windows and rollback responsibilities by scale.
- Run cross‑scale tabletop; measure friction and defect handoff quality.
- Close gaps; publish “contract tests” for recurring validation.
- Enter progressive enforcement with clear exit criteria.

Artifacts/Templates:
- RACI matrix; handoff checklist; evidence schema JSON; weekly review template.
- Cross‑scale DAG YAML; freeze window calendar; rollback playbooks.

Acceptance criteria:
- All scales have named owners, gates, and contract tests in CI.
- Evidence flows to Chronicle with correlation_ids linking scales.
- Freeze/rollback drills completed with time‑boxed SLOs.
