Title: Prompt Injection and Tool Hijack Defense (Agents)
Domain: adversary
Mode: audit

Prompt:
- Enumerate agent/tool attack surfaces: prompt injection, data exfiltration, tool hijack, cross‑tool poisoning.
- Implement sandboxes, allow‑lists, strong tool contracts, and output validation.
- Add provenance tags and signature checks; quarantine suspicious flows.
- Build red team prompts and evaluation harness; measure bypass rates.
- Publish safe prompting patterns and examples; train users.
- Record incidents to Chronicle; add regression tests.
- Integrate with approvals and rate limits.
- Review and update defenses.

Outputs:
- DEFENSE controls and contracts.
- RED TEAM suite and bypass metrics.
- RUNBOOKS and regression tests.

Assumptions:
- Tool adapters can enforce contracts; sandboxing available.
- Telemetry sufficient to detect abuse.
- User education program in place.

Metrics to watch:
- Bypass rate, severity, and time‑to‑detect; false positives.
- Incident recurrence; regression test pass rate.
- User error reports and adherence to patterns.
