Title: Causal Uplift Monitoring in Production
Domain: metrics
Mode: careful

Prompt:
- Monitor treatment effects in production using uplift models and guard against drift and leakage.
- Implement shadow policies/holdouts to validate ongoing causality assumptions.
- Attach fairness constraints to targeting; audit disparate impact and opt‑outs.
- Calibrate uplift predictions; bound harms with conservative caps.
- Provide dashboards and alerts for effect decay or sign flips.
- Publish a re‑estimation schedule and pre‑analysis plans.
- Integrate with orchestrator gates for automatic rollbacks.
- Archive evidence to Chronicle.

Outputs:
- UPLIFT monitoring spec and dashboards.
- HOLDOUT/shadow policy design and audits.
- RE‑ESTIMATION cadence with governance.

Assumptions:
- Sufficient traffic and instrumentation; privacy constraints satisfied.
- Model infra supports online monitoring; rollback is available.
- Legal approves targeted experiments.

Metrics to watch:
- Realized vs predicted uplift; calibration error; sign stability.
- Fairness metrics; opt‑out/complaint rates.
- Drift indicators and rollback count.

