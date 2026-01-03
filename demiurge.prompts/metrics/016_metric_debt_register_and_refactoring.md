Title: Metric Debt Register and Refactoring
Domain: metrics
Mode: careful

Prompt:
- Create a metric debt register: ambiguous, unowned, uncalibrated, or duplicative metrics.
- Propose refactors: deprecations, consolidations, ownership, and style guide updates.
- Add tests and calibration procedures; enforce schema/versioning.
- Tie debt to risk and prioritization; schedule work in sprints.
- Publish before/after dashboards and adoption plans.
- Enforce linting in CI and approvals for new metrics.
- Record changes in Chronicle and train consumers.
- Review quarterly and prevent regressions.

Outputs:
- DEBT register and prioritization.
- REFACTOR plan and style guide updates.
- TESTS, calibrations, and CI lint rules.

Assumptions:
- Discoverability of metrics and owners; buy‑in for refactors.
- CI can enforce lint; dashboards can be updated.
- Consumers accept change plans.

Metrics to watch:
- Debt burn‑down; calibration error reduction; adoption.
- Incident reduction tied to metric fixes.
- Lint violations and exception requests.
