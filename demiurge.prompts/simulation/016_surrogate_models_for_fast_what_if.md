Title: Surrogate Models for Fast What‑If Analysis
Domain: simulation
Mode: careful

Prompt:
- Train surrogate models (emulators) to approximate slow simulations for rapid what‑if analysis.
- Validate fidelity and uncertainty; detect covariate shift.
- Integrate with planners and explainers; expose APIs.
- Apply constraints to avoid exploitation outside training support.
- Schedule retraining; version artifacts; sign and archive.
- Publish benchmarks and governance.
- Provide fairness/ethics notes; avoid misleading conclusions.
- Link to Chronicle.

Outputs:
- SURROGATE design, validation, and APIs.
- BENCHMARKS and drift monitors.
- GOVERNANCE and archival.

Assumptions:
- Data covers relevant regimes; cost of labels manageable.
- Consumers understand limits; ethics approved.
- Infra supports deployment and versioning.

Metrics to watch:
- Error vs ground truth; uncertainty calibration; drift.
- Decision impact vs full sim; misuse incidents.
- Retrain cadence and artifact lineage integrity.
