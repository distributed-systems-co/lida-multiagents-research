# Tools

Utilities for parsing/encoding agent names and validating world models.

## agent_name_parser.py

Parses normalized IDs and extras; decodes typed facet values; re‑encodes canonically.

Examples:

```
python tools/agent_name_parser.py parse \
  "agent-uuidv7-018f2c1e-9fd2-7d3a-8b10-9a4e3c2b1a44-domain-simulation-function-forecast-mode-careful-\
   world-hsha256-deadbeef-cosmology-e.euclidean-dims-i3-lattice-e.cubic-dt-f1e-3-bc-u(e.periodic|e.reflective)-\
   invariants-s(e.energy,e.momentum)-rng-m(alg:e.xoshiro;seed:i424242)-noise-m(type:e.gaussian;sigma:f0.01)-\
   resources-m(energy:f250.0;memory:i2147483648)-telos-u(e.robustness|e.knowledge)-g_weights-m(w_h:f2.0;w_f:f1.0;w_l:f1.5;w_r:f1.0;w_s:f0.5;w_k:f1.2;w_b:f0.7)-\
   psi_min-f0.02-kill_criteria-l(e.psi_breach,e.metric_critical,e.audit_fail)-lawset-sem-1.4.2-units-e.si-snapshot_cadence-dur-1d"
```

Round‑trip check:

```
python tools/agent_name_parser.py roundtrip "<id>"
```

## world_model_validator.py

Validates a world‑model YAML/JSON, emits a canonical hash and extras, and can compare against an agent id.

Examples:

```
# Validate and compute hash/extras
python tools/world_model_validator.py validate specs/world_models/world-example.yaml

# Validate against an agent id (checks extras alignment)
python tools/world_model_validator.py validate specs/world_models/world-example.yaml \
  --agent-id "agent-uuidv7-<uuid>-domain-simulation-function-forecast-mode-careful-..."
```

Notes:
- Canonical hash is `hsha256-<hex>` of the canonical JSON (sorted keys, no spaces, UTF‑8).
- Extras follow the typed grammar in §18 (e.g., `u(…|…)`, `m(k:v;...)`, `s(a,b)`).

