import os
import json
import sys
import pytest

yaml = pytest.importorskip("yaml")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from tools.world_model_validator import validate_and_encode


def test_world_model_validate_encode_matches_example(tmp_path):
    path = os.path.join("specs", "world_models", "world-example.yaml")
    res = validate_and_encode(path)
    assert res["ok"] is True
    assert res["world_hash"].startswith("hsha256-")
    ex = res["extras"]
    assert ex["cosmology"] == "e.euclidean"
    assert ex["dims"] == "i3"
    assert ex["lattice"] == "e.cubic"
    assert ex["psi_min"] == "f0.02"
    assert "region_bbox" in ex and ex["region_bbox"].startswith("bb-")


def test_world_model_detects_bad_bbox(tmp_path):
    bad = {
        "cosmology": "euclidean",
        "dims": 3,
        "lattice": "cubic",
        "dt": 0.001,
        "bc": ["periodic"],
        "invariants": ["energy"],
        "region_bbox": {"sw": {"lon": -999, "lat": 0}, "ne": {"lon": 0, "lat": 999}},
    }
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    res = validate_and_encode(str(p))
    assert res["ok"] is False
    assert any(e.startswith("out_of_range:region_bbox") for e in res["errors"])
