Title: Human Factors and Safety‑Critical HMI Design
Domain: safety
Mode: careful

Prompt:
- Design interfaces for safety‑critical operations with human factors: error prevention, recognition over recall, clear affordances.
- Implement alarm tiering, debounce, and context to prevent fatigue and slips.
- Provide confirmation, undo, and safe defaults; enforce dual‑control where needed.
- Test with representative operators; run usability and failure drills.
- Quantify cognitive load and response times; set SLAs.
- Publish style guide and component library; enforce via design systems.
- Integrate accessibility and localization from the start.
- Link interactions to Chronicle with correlation_ids.

Outputs:
- HMI design guide and component library.
- USABILITY test plan and drill outcomes.
- OPERATOR SLAs and alarm policies.

Assumptions:
- Operator personas are known; usage contexts are mapped.
- Engineering can implement guard components platform‑wide.
- Testing environments are available without harm.

Metrics to watch:
- Error rates, near misses, alarm fatigue indicators.
- Response time distributions, cognitive load proxies.
- Accessibility conformance and localization quality.
