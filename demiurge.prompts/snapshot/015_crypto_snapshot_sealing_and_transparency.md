Title: Cryptographic Snapshot Sealing and Transparency Logs
Domain: snapshot
Mode: careful

Prompt:
- Seal snapshots with cryptographic hashes and publish to transparency logs.
- Support inclusion proofs and auditability; rotate keys.
- Integrate with restore/merge protocols and SLOs.
- Provide public verification tools and docs.
- Redact safely while preserving proofs.
- Train operators; drill verification.
- Report transparency metrics; handle incidents.
- Archive revocation and rotation events.

Outputs:
- SEALING pipeline and transparency log policy.
- TOOLS and documentation for verification.
- REPORTS and drills.

Assumptions:
- Crypto primitives and HSMs available.
- Legal allows publication; redaction feasible.
- Users can verify independently.

Metrics to watch:
- Coverage, verification success, key hygiene.
- Incident counts; drill outcomes.
- Public verification requests.

