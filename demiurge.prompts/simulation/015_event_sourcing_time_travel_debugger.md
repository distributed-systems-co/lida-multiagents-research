Title: Event Sourcing Time‑Travel Debugger
Domain: simulation
Mode: careful

Prompt:
- Build a time‑travel debugger for event‑sourced systems; replay/step/branch with invariants.
- Provide breakpoints on conditions; inspect state deltas and causal traces.
- Support partial restores and differential snapshots; sign evidence.
- Integrate with orchestrator and Chronicle; propagate correlation_ids.
- Simulate adversarial schedules; verify determinism where required.
- Publish CLI/GUI and tutorials; ensure accessibility.
- Measure performance overhead and safety.
- Archive sessions and outcomes.

Outputs:
- DEBUGGER design and tooling.
- TESTS for determinism and invariants.
- DOCS and tutorials.

Assumptions:
- Event sourcing and snapshot infra exist.
- Invariants are codified; performance budget available.
- Privacy of debug data respected.

Metrics to watch:
- Determinism drift; invariant violations in debug.
- Overhead and usability scores.
- Incident remediation speed.

