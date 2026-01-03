Title: Temporal Consistency and Sunset Governance
Domain: law
Mode: careful

Prompt:
- Ensure policies/laws remain temporally consistent across overlapping effective windows and dependencies.
- Define sunsets, grace periods, and freeze windows; manage precedence and conflicts.
- Version and diff policies with time facets; require approvals for emergency extensions.
- Provide redress for harms caused by temporal inconsistencies.
- Publish transparent calendars and change logs.
- Audit compliance and exceptions.
- Train policy owners; simulate boundary cases.
- Archive rationale for precedent.

Outputs:
- TEMPORAL policy framework and calendars.
- DIFF/precedence rules and tools.
- AUDIT procedures and remediation.

Assumptions:
- Policy engine supports effective‑from/until semantics.
- Owners can schedule changes and communicate adequately.
- Ombuds capacity exists for appeals.

Metrics to watch:
- Conflict incidents, late sunsets, emergency extensions.
- Policy diff accuracy and stakeholder notifications.
- Remediation latency and recurrence.

Procedure:
- Inventory policies with effective windows; encode precedence (global>org>env>project with exceptions).
- Define sunsets, freezes, grace windows; publish calendars and notifications.
- Implement diff tooling with time facets; require approvals for emergency changes.
- Simulate overlapping policies and boundary cases; document outcomes.
- Establish redress workflows for harms; log and report.
- Audit compliance and exceptions; publish summaries.
- Train owners; run drills for freeze/extension scenarios.

Artifacts:
- Temporal policy spec, calendars, diff examples, redress template, audit checklist.

Acceptance criteria:
- All effective windows encoded; precedence rules enforced; calendars visible.
- Exceptions tracked with approvals; redress active; audits green.
