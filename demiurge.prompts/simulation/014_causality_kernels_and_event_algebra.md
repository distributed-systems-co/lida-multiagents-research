Title: Causality Kernels and Event Algebra
Domain: simulation
Mode: careful

Prompt:
- Define an event algebra (compose, delay, throttle, cancel) and a causality kernel (propagators, lags, mediation) for world updates.
- Specify concurrency semantics (ordering, tie‑breakers, determinism) and permissible non‑local correlations under no‑signalling.
- Model distributed lag responses (DL/IRF) and causal mediation; attach invariants and safety bounds.
- Provide instrumentation for causal traces and explainability hooks.
- Create a test harness with adversarial schedules; include snapshot/restore checks.
- Publish numerics and stability requirements; document time step interactions with kernels.
- Offer policy surfaces (allowed operations, quotas, proof obligations).
- Provide examples and failure analysis.

Outputs:
- KERNEL+ALGEBRA spec and proofs/sketches.
- TEST harness and adversarial scenarios.
- EXPLAIN and policy surfaces.

Assumptions:
- Event sourcing and snapshotting are available.
- Determinism and auditability are desired within defined bounds.
- Performance budgets accommodate tracing overhead.

Metrics to watch:
- Invariant violations, scheduling anomalies, determinism drift.
- Trace completeness/latency and explanation quality.
- Stability under stress (noise, burst events).

Procedure:
- Specify event algebra ops: compose(a,b), delay(a,t), throttle(a,r), cancel(a), guard(a,phi).
- Define causality kernel API: apply(event) → {effects[], lags, invariants_checked}.
- Choose ordering semantics: total vs partial; define tie‑breakers and fairness.
- Implement distributed lag responses (filters/IRFs); verify numerics vs step sizes.
- Build adversarial schedule suite (reorderings, bursts, near‑simultaneity) and assert invariants.
- Wire causal traces to Chronicle with minimal overhead; sample rates and backpressure.
- Document allowed non‑local correlations and proofs of no‑signalling.
- Publish extension points and policies; ship examples.

API sketch:
```
POST /kernel/event { id, type, payload, ts }
→ { trace_id, effects: [{id,type,ts}], checks: [{invariant, ok}], cost: {cpu,mem}}

POST /kernel/simulate_irf { impulse, horizon, dt } → { irf: [..], ci: [..] }
```

Acceptance criteria:
- Algebra closed under composition; invariants hold across adversarial schedules.
- IRFs stable across dt; tie‑breaks deterministic and documented.
- Traces queryable with correlation to upstream actions.
