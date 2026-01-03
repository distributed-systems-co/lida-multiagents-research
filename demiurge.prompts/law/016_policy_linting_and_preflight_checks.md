Title: Automated Policy Linting and Preflight Checks
Domain: law
Mode: careful

Prompt:
- Define policy lint rules (scope, mode, conflicts, missing approvals).
- Implement preflight checks for proposed changes; block on critical issues.
- Provide human‑readable diffs and remediation guidance.
- Log results to Chronicle; escalate on repeated failures.
- Integrate with CI/CD and orchestrator gates.
- Publish rules and governance; include change control.
- Review lint drift and false rates; update rules.
- Train policy authors.

Outputs:
- LINT rule set and preflight pipeline.
- DIFF formats and guidance.
- REPORTS and training plan.

Assumptions:
- Policies are versioned and machine‑readable.
- Teams accept preflight gates.
- Governance can evolve rules safely.

Metrics to watch:
- Lint failures by type; time‑to‑fix.
- False positive/negative rates; policy conflicts caught.
- Author training coverage and improvement.
