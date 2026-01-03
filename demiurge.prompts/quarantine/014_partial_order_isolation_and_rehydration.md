Title: Partial‑Order Isolation with Dependency Rehydration
Domain: quarantine
Mode: careful

Prompt:
- Isolate subsystems based on partial order of dependencies; avoid unnecessary shutdowns.
- Detect minimal cut sets; compute safe rehydration sequences.
- Provide read‑only modes and decoys where safe.
- Track evidence and preserve integrity; log correlation_ids.
- Define re‑entry gates and verification checks.
- Publish playbooks; drill with scenarios.
- Measure impact and time to steady state.
- Integrate with orchestrator.

Outputs:
- ISOLATION plan by dependency partial order.
- REHYDRATION sequences and checks.
- PLAYBOOKS and drills.

Assumptions:
- Accurate dependency graphs and health signals.
- Safe decoy patterns exist for critical paths.
- Teams can execute sequences reliably.

Metrics to watch:
- Isolation scope vs minimal cut; recovery time; residual errors.
- Evidence integrity; read‑only mode stability.
- Drill performance and improvements.

