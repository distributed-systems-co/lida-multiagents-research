Title: Temporal Causal Trace Graphs
Domain: chronicle
Mode: careful

Prompt:
- Build a temporal trace graph linking actions→effects via causal edges; distinguish correlation vs causation with evidence levels.
- Index by time, actor, scope, correlation_id; support path queries and “why”/“what‑if” traversals.
- Capture lags, mediation nodes, and counterfactual forks; record uncertainty and provenance.
- Provide write policies and signatures; ensure privacy with tiered redaction.
- Design fast queries and roll‑ups; expose APIs to auditors and explainers.
- Define compaction/retention without losing causal structure.
- Publish schemas and examples; add query cookbook.
- Integrate with snapshot lineage.

Outputs:
- GRAPH schema, APIs, and indexing plan.
- PRIVACY/redaction policy with examples.
- QUERY cookbook and performance SLOs.

Assumptions:
- Correlation IDs propagate reliably; clocks are within tolerated skew.
- Evidence standards allow causal labeling.
- Storage/index budgets can support graph features.

Metrics to watch:
- Query latency/throughput, link completeness, causal labeling accuracy.
- Privacy incidents, redaction failures, signature verification.
- Auditor satisfaction and coverage.

Procedure:
- Define graph schema: Action, Mediator, Outcome nodes; CAUSES edges with lag and evidence attrs.
- Implement ingestion from orchestrator/simulation with correlation_ids and signatures.
- Build path/query library: why(action→outcome?), what‑if(remove edge?), counterfactual forks.
- Implement privacy tiers and redaction with audit justifications; expose redacted views.
- Index for time/actor/scope plus reachability; benchmark and tune.
- Write retention/compaction preserving causal topology (supernodes/roll‑ups).
- Publish query cookbook and examples; add CI checks for graph integrity.

Schema (sketch):
```
node Action { id, ts, actor, scope, payload_hash, sig }
node Mediator { id, ts, kind, payload_hash }
node Outcome { id, ts, metric, value, ci }
edge CAUSES { from, to, lag_s, evidence: enum(correlation|ablation|randomized), strength }
```

Acceptance criteria:
- >95% of orchestrator actions link to downstream outcomes or documented as sinks.
- Redaction leaves topology analyzable; signatures validate.
- Cookbook queries pass and meet SLOs.
