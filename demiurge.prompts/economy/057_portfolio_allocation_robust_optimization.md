Title: Portfolio Allocation via Robust Optimization (CVaR/Minimax Regret)
Domain: economy
Mode: careful

Prompt:
- Allocate budget across bets/channels under uncertainty using robust methods (CVaR, minimax regret) to bound downside while preserving upside.
- Encode constraints (CAC, payback, fairness, brand, capacity) and interdependencies.
- Run scenario generation (bootstraps/stress) and sensitivity to priors.
- Compare robust vs mean‑variance vs heuristic baselines; justify choice.
- Provide governance for rebalancing cadence and kill criteria.
- Integrate telemetry for violations and automatic caps.
- Publish transparent policy and rationale; review quarterly.
- Archive decisions and outcomes to the Chronicle.

Outputs:
- OPTIMIZATION spec, constraints, and solver plan.
- ALLOCATION results with risk bands; sensitivity analysis.
- GOVERNANCE for rebalancing and stops.

Assumptions:
- Return distributions and correlations are estimable within tolerances.
- Constraints can be enforced operationally; caps are actionable.
- Stakeholders accept robust framing.

Metrics to watch:
- Downside risk (CVaR), realized regret, variance vs targets.
- Frequency/severity of cap triggers; reallocation stability.
- Performance vs baselines across cycles.

