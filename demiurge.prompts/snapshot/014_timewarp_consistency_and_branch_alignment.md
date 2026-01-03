Title: Timewarp Consistency and Branch Alignment
Domain: snapshot
Mode: careful

Prompt:
- Define consistent “timewarp” rules for snapshot restore/branch that preserve causal order and rights.
- Detect and resolve conflicts on merge with fairness and minimal harm.
- Require human review/audit for risky merges; log evidence and rationale.
- Provide user‑facing explanations; support appeals.
- Benchmark overhead and integrity; set SLOs.
- Simulate adversarial merges; record outcomes.
- Publish playbooks and public reports.
- Align with orchestrator freeze windows.

Outputs:
- TIMEWARP rules and merge protocol.
- AUDIT/review gates and evidence schema.
- BENCHMARKS and incident drills.

Assumptions:
- Snapshot lineage is accurate; diffs are comprehensible.
- Human review bandwidth available.
- Privacy/rights constraints enforceable during merge.

Metrics to watch:
- Merge conflict rate, rollback needed, harm incidents.
- Latency and success of reconciliations.
- Stakeholder satisfaction and appeal outcomes.

Procedure:
- Define timewarp invariants: no retroactive rights violation; causal order preserved across branches.
- Implement branch naming/versioning; encode lineage and diffs; sign metadata.
- Write merge protocol: detection, classification (soft/hard), resolution rules, and human review gates.
- Build restore/merge test harness; simulate adversarial merges and partial restores.
- Publish user‑facing comms templates; capture appeals and outcomes.
- Set SLOs for detection, review, and reconciliation; monitor and alert.
- Archive evidence and public summaries to Chronicle.

Artifacts:
- Timewarp spec; merge protocol; harness; comms templates; SLO dashboards.

Acceptance criteria:
- Invariants proven in harness; merges meet SLOs; appeals resolved within SLA.
- Public summaries published; evidence signed; audits satisfied.
