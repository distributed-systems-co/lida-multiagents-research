# World Model Manifest — Side-by-Side Example

This example shows how to take a compact world-model YAML and produce both:
- Encoded extras for an agent id, and
- A facets_typed JSON object suitable for the agent manifest.

## Source YAML
Path: `specs/world_models/world-example.yaml`

```
name: "World Example — Euclidean Lattice"
version: "1.4.2"
cosmology: euclidean
dims: 3
lattice: cubic
dt: 0.001
bc: [periodic, reflective]
invariants: [energy, momentum]
rng: { alg: xoshiro, seed: 424242 }
noise: { type: gaussian, sigma: 0.01 }
resources: { energy: 250.0, memory: 2147483648 }
telos: [robustness, knowledge]
g_weights: { w_h: 2.0, w_f: 1.0, w_l: 1.5, w_r: 1.0, w_s: 0.5, w_k: 1.2, w_b: 0.7 }
psi_min: 0.02
kill_criteria: [psi_breach, metric_critical, audit_fail]
lawset: 1.4.2
units: si
snapshot_cadence: 1d
region_bbox: { sw: { lon: -122.42, lat: 37.77 }, ne: { lon: -122.40, lat: 37.78 } }
```

## Encoded extras (as they’d appear in the agent id)

Note: `world` is a content hash of the canonicalized world model spec (`hsha256-<hex>`). Compute with the validator below.

```
world: hsha256-<hex>
cosmology: e.euclidean
dims: i3
lattice: e.cubic
dt: f0.001
bc: u(e.periodic|e.reflective)
invariants: s(e.energy,e.momentum)
rng: m(alg:e.xoshiro;seed:i424242)
noise: m(type:e.gaussian;sigma:f0.01)
resources: m(energy:f250.0;memory:i2147483648)
telos: u(e.robustness|e.knowledge)
g_weights: m(w_h:f2.0;w_f:f1.0;w_l:f1.5;w_r:f1.0;w_s:f0.5;w_k:f1.2;w_b:f0.7)
psi_min: f0.02
kill_criteria: l(e.psi_breach,e.metric_critical,e.audit_fail)
lawset: sem-1.4.2
units: e.si
snapshot_cadence: dur-1d
region_bbox: bb--122.42_37.77..-122.40_37.78
```

## facets_typed (structured object)

```
{
  "world": "hsha256-<hex>",
  "cosmology": "euclidean",
  "dims": 3,
  "lattice": "cubic",
  "dt": 0.001,
  "bc": ["periodic", "reflective"],
  "invariants": ["energy", "momentum"],
  "rng": { "alg": "xoshiro", "seed": 424242 },
  "noise": { "type": "gaussian", "sigma": 0.01 },
  "resources": { "energy": 250.0, "memory": 2147483648 },
  "telos": ["robustness", "knowledge"],
  "g_weights": { "w_h": 2.0, "w_f": 1.0, "w_l": 1.5, "w_r": 1.0, "w_s": 0.5, "w_k": 1.2, "w_b": 0.7 },
  "psi_min": 0.02,
  "kill_criteria": ["psi_breach", "metric_critical", "audit_fail"],
  "lawset": "1.4.2",
  "units": "si",
  "snapshot_cadence": "1d",
  "region_bbox": { "sw": { "lon": -122.42, "lat": 37.77 }, "ne": { "lon": -122.40, "lat": 37.78 } }
}
```

## Sample agent manifest (JSON5)

```
{
  id: "agent-uuidv7-<uuid>-domain-simulation-function-forecast-mode-careful-"
      + "world-hsha256-<hex>-cosmology-e.euclidean-dims-i3-lattice-e.cubic-dt-f0.001-"
      + "bc-u(e.periodic|e.reflective)-invariants-s(e.energy,e.momentum)-rng-m(alg:e.xoshiro;seed:i424242)-"
      + "noise-m(type:e.gaussian;sigma:f0.01)-resources-m(energy:f250.0;memory:i2147483648)-telos-u(e.robustness|e.knowledge)-"
      + "g_weights-m(w_h:f2.0;w_f:f1.0;w_l:f1.5;w_r:f1.0;w_s:f0.5;w_k:f1.2;w_b:f0.7)-psi_min-f0.02-"
      + "kill_criteria-l(e.psi_breach,e.metric_critical,e.audit_fail)-lawset-sem-1.4.2-units-e.si-snapshot_cadence-dur-1d-"
      + "region_bbox-bb--122.42_37.77..-122.40_37.78",
  label: "Simulation Forecaster",
  facets: { domain: "simulation", function: "forecast", mode: "careful" },
  facets_typed: { /* see above (structured object) */ },
  status: "ready",
  version: "vΘ.1.0",
  created_at: "2025-11-10T00:00:00Z"
}
```

## How to generate the extras and hash

Commands:
- Compute hash and extras: `python tools/world_model_validator.py validate specs/world_models/world-example.yaml`
- Use `world_hash` and `extras` to assemble your agent id.
- Optional consistency check with an existing id: `--agent-id '<id>'`

