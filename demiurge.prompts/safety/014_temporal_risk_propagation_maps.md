Title: Temporal Risk Propagation Maps
Domain: safety
Mode: careful

Prompt:
- Chart how hazards propagate over time across components (technical, human, policy); include lags, mediators, and feedback.
- Define early warnings and thresholds per arc; add hysteresis.
- Map mitigations (nudge, quarantine, rollback) with time windows and owners.
- Integrate approvals and ombuds escalation for high‑risk arcs.
- Provide drills and synthetic incidents to validate maps.
- Record evidence in Chronicle for audits.
- Update maps post‑incident; publish safe summaries.
- Align with orchestrator gates.

Outputs:
- RISK propagation map with signals and actions.
- DRILLS and validation scenarios.
- AUDIT hooks and update cadence.

Assumptions:
- Signals are measurable or proxied; alarm fatigue mitigations exist.
- Teams can act within windows; rollback is available.
- Privacy constraints respected for sensitive signals.

Metrics to watch:
- Detection latency, false rates, mitigation success.
- Incident counts by arc and residual risk.
- Review cadence adherence and improvements.

Procedure:
- Enumerate components and interfaces; draw risk arcs with lags and mediators.
- Assign signals, thresholds, hysteresis per arc; specify owners and runbooks.
- Attach mitigations (nudge, throttle, quarantine, rollback) with windows and approvals.
- Simulate incidents; time responses; record evidence and outcomes.
- Reduce alarm fatigue: tiering, dedup, cooldowns; test integrity.
- Publish maps and runbooks; link to orchestrator gates and Chronicle.
- Review after incidents; update maps and drills.

Artifacts:
- Risk map (graph), threshold table, runbooks, drill calendar, Chronicle queries.

Acceptance criteria:
- All high‑risk arcs mapped; signals measurable; owners named.
- Drills pass within SLOs; fatigue mitigations effective; audits satisfied.
