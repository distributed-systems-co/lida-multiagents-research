#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REQUIRED_HEADERS = ["Title:", "Domain:", "Mode:", "Prompt:", "Outputs:"]
SKIP_PATTERNS = ("INDEX", "README")

def lint_file(p: Path) -> list[str]:
    errs: list[str] = []
    text = p.read_text(encoding="utf-8")
    lines = text.strip("\n").splitlines()
    if len(lines) <= 17:
        errs.append("length<=17")
    for h in REQUIRED_HEADERS:
        if h not in text:
            errs.append(f"missing:{h}")
    return errs

def main(argv):
    root = Path(argv[1]) if len(argv) > 1 else Path("demiurge.prompts")
    failures = {}
    for p in root.rglob("*.md"):
        if any(s in p.name for s in SKIP_PATTERNS):
            continue
        errs = lint_file(p)
        if errs:
            failures[str(p)] = errs
    if failures:
        print("PROMPT LINT REPORT (warnings):")
        for k, v in failures.items():
            print(f"- {k}: {', '.join(v)}")
        sys.exit(0)
    print("OK: all prompts passed lint (no warnings)")

if __name__ == "__main__":
    main(sys.argv)
