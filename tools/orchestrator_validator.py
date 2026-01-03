#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

def load(path: str):
    p = Path(path)
    if p.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML required for YAML. Install pyyaml or use JSON.")
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    return json.loads(p.read_text(encoding="utf-8"))

def main(argv):
    if len(argv) < 2:
        print("usage: orchestrator_validator.py <plan.(yaml|json)> [schema.json]", file=sys.stderr)
        sys.exit(2)
    plan = load(argv[1])
    schema_path = argv[2] if len(argv) > 2 else str(Path(__file__).parent.parent / "specs" / "orchestrator" / "orchestrator.schema.json")
    try:
        import jsonschema  # type: ignore
    except Exception:
        print(json.dumps({"ok": False, "error": "jsonschema_not_installed", "hint": "pip install jsonschema"}))
        sys.exit(0)
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    try:
        jsonschema.validate(instance=plan, schema=schema)
        print(json.dumps({"ok": True, "message": "valid"}))
    except jsonschema.ValidationError as e:  # type: ignore
        print(json.dumps({"ok": False, "error": "schema_validation_failed", "path": list(e.path), "message": e.message}))

if __name__ == "__main__":
    main(sys.argv)

