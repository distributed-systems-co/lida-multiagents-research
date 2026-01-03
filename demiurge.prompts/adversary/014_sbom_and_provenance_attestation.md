Title: SBOM and Provenance Attestation
Domain: adversary
Mode: careful

Prompt:
- Generate SBOMs for critical components; sign and publish with build provenance (SLSA).
- Enforce verified builds and dependency pinning; rotate keys; scan for tampering.
- Simulate supply chain compromise; test detection and rollback.
- Define quarantine for tainted artifacts; record evidence.
- Integrate with CI/CD and policy gates; fail safe on unknown provenance.
- Train teams; publish incident runbooks.
- Provide transparency reports.
- Audit periodically.

Outputs:
- SBOM/provenance pipeline and policies.
- COMPROMISE drills and rollback.
- AUDIT reports and transparency.

Assumptions:
- Build infra supports attestations; keys secured.
- Teams can adopt pinning and rotation.
- Vendors participate or are sandboxed.

Metrics to watch:
- Coverage of SBOMs, attestation validation rate.
- Time to detect/respond; rollback success.
- Drift in dependency risk scores.

