import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from tools.agent_name_parser import parse_agent_name, encode_agent_name


def test_basic_parse_roundtrip():
    s = (
        "agent-uuidv7-018f2c1e-9fd2-7d3a-8b10-9a4e3c2b1a00-"
        "domain-planning-function-orchestrate-mode-audit-"
        "region-gh-9q8yy-trust-e.l3-hazard-e.med-caps-l.embed_law.snapshot-"
        "ttl-dur-7d-version-sem-1.3.0"
    )
    p = parse_agent_name(s)
    assert p.domain == "planning"
    assert p.function == "orchestrate"
    assert p.mode == "audit"
    assert p.extras["region"] == "gh-9q8yy"
    assert p.extras["trust"].startswith("e.")
    r = encode_agent_name(p)
    assert r == s


def test_world_model_extended_example_roundtrip():
    s = (
        "agent-uuidv7-018f2c1e-9fd2-7d3a-8b10-9a4e3c2b1a44-"
        "domain-simulation-function-forecast-mode-careful-"
        "world-hsha256-deadbeef-cosmology-e.euclidean-dims-i3-"
        "lattice-e.cubic-dt-f1e-3-bc-u(e.periodic|e.reflective)-"
        "invariants-s(e.energy,e.momentum)-rng-m(alg:e.xoshiro;seed:i424242)-"
        "noise-m(type:e.gaussian;sigma:f0.01)-resources-m(energy:f250.0;memory:i2147483648)-"
        "telos-u(e.robustness|e.knowledge)-g_weights-m(w_h:f2.0;w_f:f1.0;w_l:f1.5;w_r:f1.0;w_s:f0.5;w_k:f1.2;w_b:f0.7)-"
        "psi_min-f0.02-kill_criteria-l(e.psi_breach,e.metric_critical,e.audit_fail)-"
        "lawset-sem-1.4.2-lawhash-hsha256-abcdef-snapshot_cadence-dur-1d-units-e.si"
    )
    p = parse_agent_name(s)
    # sanity checks on typed facets
    assert p.facets_typed["dims"] == 3
    assert p.facets_typed["psi_min"] == 0.02
    assert p.facets_typed["rng"]["_type"] == "map"
    r = encode_agent_name(p)
    assert r == s
