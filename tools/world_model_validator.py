#!/usr/bin/env python3
"""
World-model spec validator and encoder.

Reads a YAML (or JSON) world model spec and:
- Validates basic constraints (enums, ranges, shapes)
- Produces a canonical JSON and hsha256-<hex> fingerprint
- Emits recommended extras suitable for an agent id
- Optionally verifies against an existing agent id

CLI
  python tools/world_model_validator.py validate specs/world_models/world-example.yaml [--agent-id '<id>']
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from typing import Any, Dict, List, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

NAME_RE = re.compile(
    r"^agent-uuidv7-[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}-domain-"
    r"(planning|safety|metrics|adversary|explain|ethics|law|ecology|economy|culture|simulation|chronicle|audit|quarantine|snapshot|consent)"
    r"-function-[a-z][a-z0-9_]{1,48}-mode-(fast|careful|audit)"
)


ALLOWED = {
    "cosmology": {"euclidean", "spherical", "hyperbolic"},
    "lattice": {"continuous", "square", "hex", "cubic"},
    "bc": {"periodic", "reflective", "absorbing"},
    "invariants": {"energy", "momentum", "charge", "mass", "angular_momentum"},
    "noise.type": {"gaussian", "laplace", "uniform", "poisson"},
    "telos": {"play", "knowledge", "robustness"},
    "units": {"si", "cgs"},
}


def load_world(path: str) -> Dict[str, Any]:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    if yaml is None:
        raise RuntimeError("PyYAML is required to load YAML world models; install pyyaml or use JSON.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def hsha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def validate_world_model(w: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    def need(key: str, predicate) -> None:
        if key not in w:
            errs.append(f"missing:{key}")
        elif not predicate(w[key]):
            errs.append(f"invalid:{key}")

    need("cosmology", lambda x: x in ALLOWED["cosmology"])
    need("dims", lambda x: isinstance(x, int) and 1 <= x <= 12)
    need("lattice", lambda x: x in ALLOWED["lattice"])
    need("dt", lambda x: isinstance(x, (int, float)) and x > 0)
    need("bc", lambda x: isinstance(x, list) and set(x).issubset(ALLOWED["bc"]))
    need("invariants", lambda x: isinstance(x, list) and set(x).issubset(ALLOWED["invariants"]))
    if "rng" in w:
        rng = w["rng"]
        if not isinstance(rng, dict) or "alg" not in rng:
            errs.append("invalid:rng")
    if "noise" in w:
        nz = w["noise"]
        if not isinstance(nz, dict) or nz.get("type") not in ALLOWED["noise.type"]:
            errs.append("invalid:noise")
    if "units" in w and w["units"] not in ALLOWED["units"]:
        errs.append("invalid:units")
    # bbox sanity
    if "region_bbox" in w:
        bb = w["region_bbox"]
        try:
            sw, ne = bb["sw"], bb["ne"]
            for d in (sw, ne):
                if not (-180 <= float(d["lon"]) <= 180 and -90 <= float(d["lat"]) <= 90):
                    raise ValueError
        except Exception:
            errs.append("out_of_range:region_bbox")
    return errs


def encode_extras_from_world(w: Dict[str, Any]) -> Dict[str, str]:
    ex: Dict[str, str] = {}
    # world hash will be filled by caller
    ex["cosmology"] = f"e.{w['cosmology']}"
    ex["dims"] = f"i{w['dims']}"
    ex["lattice"] = f"e.{w['lattice']}"
    ex["dt"] = f"f{w['dt']}"
    ex["bc"] = "u(" + "|".join(sorted(f"e.{v}" for v in w.get("bc", []))) + ")"
    ex["invariants"] = "s(" + ",".join(sorted(f"e.{v}" for v in w.get("invariants", []))) + ")"
    if "rng" in w:
        rng = w["rng"]
        parts = [f"alg:e.{rng['alg']}"]
        if "seed" in rng:
            parts.append(f"seed:i{rng['seed']}")
        ex["rng"] = "m(" + ";".join(parts) + ")"
    if "noise" in w:
        nz = w["noise"]
        parts = [f"type:e.{nz['type']}"]
        if "sigma" in nz:
            parts.append(f"sigma:f{nz['sigma']}")
        ex["noise"] = "m(" + ";".join(parts) + ")"
    if "resources" in w:
        r = w["resources"]
        parts = []
        if "energy" in r:
            parts.append(f"energy:f{r['energy']}")
        if "memory" in r:
            parts.append(f"memory:i{r['memory']}")
        ex["resources"] = "m(" + ";".join(parts) + ")"
    if "telos" in w:
        ex["telos"] = "u(" + "|".join(sorted(f"e.{v}" for v in w["telos"])) + ")"
    if "g_weights" in w:
        gw = w["g_weights"]
        parts = [f"{k}:f{gw[k]}" for k in sorted(gw.keys())]
        ex["g_weights"] = "m(" + ";".join(parts) + ")"
    if "psi_min" in w:
        ex["psi_min"] = f"f{w['psi_min']}"
    if "kill_criteria" in w:
        ex["kill_criteria"] = "l(" + ",".join(sorted(f"e.{v}" for v in w["kill_criteria"])) + ")"
    if "lawset" in w:
        ex["lawset"] = f"sem-{w['lawset']}"
    if "units" in w:
        ex["units"] = f"e.{w['units']}"
    if "snapshot_cadence" in w:
        ex["snapshot_cadence"] = f"dur-{w['snapshot_cadence']}"
    if "region_bbox" in w:
        sw = w["region_bbox"]["sw"]
        ne = w["region_bbox"]["ne"]
        ex["region_bbox"] = f"bb-{sw['lon']}_{sw['lat']}..{ne['lon']}_{ne['lat']}"
    return ex


def validate_and_encode(path: str, agent_id: str | None = None) -> Dict[str, Any]:
    world = load_world(path)
    errs = validate_world_model(world)
    canon = canonical_json(world)
    world_hash = "hsha256-" + hsha256_hex(canon)
    extras = encode_extras_from_world(world)
    extras["world"] = world_hash
    out: Dict[str, Any] = {"ok": len(errs) == 0, "errors": errs, "world_hash": world_hash, "extras": extras}
    if agent_id:
        m = NAME_RE.match(agent_id)
        if not m:
            out["agent_match"] = False
            out["agent_reason"] = "invalid_agent_id"
        else:
            # shallow verify: check the provided extras (if present) match computed
            provided = {}
            for kv in re.findall(r"-([a-z]+)-([a-z0-9][a-z0-9_.:()|,;\-]{0,128})", agent_id):
                provided[kv[0]] = kv[1]
            diffs = {}
            for k, v in extras.items():
                if k in provided and provided[k] != v:
                    diffs[k] = {"expected": v, "found": provided[k]}
            out["agent_match"] = len(diffs) == 0
            if diffs:
                out["diffs"] = diffs
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    v = sub.add_parser("validate")
    v.add_argument("path")
    v.add_argument("--agent-id")
    args = ap.parse_args()
    if args.cmd != "validate":
        ap.error("usage: validate <path> [--agent-id ID]")
    res = validate_and_encode(args.path, args.agent_id)
    print(json.dumps(res, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

