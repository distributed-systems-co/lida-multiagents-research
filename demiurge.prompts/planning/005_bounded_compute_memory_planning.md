Title: Planning Under Bounded Compute and Memory
Domain: planning
Mode: fast

Prompt:
- Produce a minimal but effective plan under strict compute and memory caps.
- Use coarse‑to‑fine refinement; show anytime intermediate outputs.
- Prioritize reversible micro‑actions; defer costly steps behind gates.
- Compress state via MDL; document lossy choices and error bounds.
- Specify what to drop when under duress (graceful degrade policy).
- Include quick sanity checks and fallback plan if time expires.

Outputs:
- PLAN with phased refinement and compute budgets per phase.
- Fallback path and degrade policy.
- Minimal test probes for correctness.

