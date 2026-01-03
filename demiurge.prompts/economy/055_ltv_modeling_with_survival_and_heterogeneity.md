Title: LTV Modeling with Survival and Heterogeneity
Domain: economy
Mode: careful

Prompt:
- Model LTV via survival analysis (Cox/parametric) with time‑varying covariates and heterogeneous treatment effects.
- Integrate revenue processes (renewals, expansion, churn) and cost‑to‑serve into cash LTV.
- Provide uncertainty bands; avoid over‑allocation to overfit segments.
- Validate with holdouts and temporal cross‑validation.
- Map LTV to CAC caps and budget allocation; enforce guardrails.
- Add drift monitoring and recalibration cadence.
- Publish assumptions and error sources.
- Provide templates and dashboards.

Outputs:
- LTV model spec and validation.
- CAC caps/targets and budget rules.
- DASHBOARDS and drift monitors.

Assumptions:
- Revenue/cost events are accurately captured and timestamped.
- Sufficient tenure to estimate tails; censoring handled properly.
- Finance alignment on cash LTV definition.

Metrics to watch:
- Calibration of survival and LTV predictions; realized vs predicted.
- Allocation efficiency and guardrail breaches.
- Drift in hazard rates by segment.

