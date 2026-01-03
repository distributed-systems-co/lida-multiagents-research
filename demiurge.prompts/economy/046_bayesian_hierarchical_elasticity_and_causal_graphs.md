Title: Bayesian Hierarchical Elasticity and Causal Graphs
Domain: economy
Mode: careful

Prompt:
- Build a Bayesian hierarchical model of price elasticity by segment (product×geo×channel), with partial pooling and priors informed by domain knowledge.
- Encode causal structure via DAGs; identify confounders (seasonality, promo, competitor prices), instruments, and backdoor paths to close.
- Compare “reduced form” vs structural demand; quantify uncertainty; produce counterfactual demand curves.
- Stress-test with posterior predictive checks; test transportability across segments and time windows.
- Integrate fairness constraints (price parity, protected classes) into policy recommendations.
- Provide productionization plan: sampling cadence, model drift monitoring, and retraining triggers.
- Define how outputs map to pricing fences and guardrails (min margin, variance caps) and governance approvals.
- Document assumptions, priors, convergence diagnostics (R‑hat/ESS), and sensitivity to priors.

Outputs:
- MODEL spec (priors, structure, DAG) and inference plan.
- POLICY surfaces: elasticity curves with CIs and guardrail‑compliant price bands.
- VALIDATION: PPC, out‑of‑sample fit, and transport tests.

Assumptions:
- Adequate coverage of confounders; competitor pricing data is reliable or instrumentable.
- Traffic suffices for convergence; MCMC runtime acceptable for cadence.
- Governance accepts uncertainty bands in decisions.

Metrics to watch:
- Posterior calibration, R‑hat/ESS, OOS error; uplift vs baseline.
- Guardrail violations, fairness indicators, complaint rate.
- Model drift: covariate shift, parameter instability, retrain frequency.

