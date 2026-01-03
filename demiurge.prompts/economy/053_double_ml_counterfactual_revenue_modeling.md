Title: Double‑ML Counterfactual Revenue Modeling
Domain: economy
Mode: careful

Prompt:
- Employ Double/Orthogonal ML to estimate treatment effects of pricing/promos/channels on revenue, controlling for high‑dim confounders.
- Validate with pre‑analysis plans, placebo tests, and sensitivity to unobserved confounding.
- Compare uplift modeling vs A/B; define when each is preferred.
- Use resulting CATEs to target offers under fairness constraints.
- Publish assumptions and uncertainty bands; avoid overfit deployment.
- Add governance for periodic re‑estimation.
- Integrate with attribution and forecasting.
- Provide code and data schema notes.

Outputs:
- CAUSAL model spec, validation, and deployment plan.
- TARGETING policy with fairness and caps.
- MONITORING for drift and misuse.

Assumptions:
- Sufficient covariates and sample size; stationarity manageable.
- Infra supports feature pipelines and re‑estimation cadence.
- Legal approves targeted offers under policy.

Metrics to watch:
- PEHE, calibration of uplift; realized vs predicted uplift.
- Fairness metrics; opt‑out/complaints.
- Drift and degradation rates.

