#!/usr/bin/env python3
"""
Agent name parser/encoder for Demiurge normalized IDs and extras.

Features
- Parses normalized IDs like:
  agent-uuidv7-<uuid>-domain-<domain>-function-<func>-mode-<mode>[-key-value]*
- Extracts extras and decodes typed values per §18 mini‑grammar:
  iN, fN, b0|b1, e.token, r<T>..<T>o?, l(...)|l.a.b, s(...)|s.a.b, p(...), m(k:v;...), u(a|b),
  tYYYY-..., dur-..., sem-..., gh-..., bb-..., h<alg>-<hex>, jz-...
- Encodes typed values back to the canonical string form.

CLI
  python tools/agent_name_parser.py parse "<id>"
  python tools/agent_name_parser.py roundtrip "<id>"
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


NAME_RE = re.compile(
    r"^agent-uuidv7-"
    r"(?P<uuid>[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})"
    r"-domain-(?P<domain>planning|safety|metrics|adversary|explain|ethics|law|ecology|economy|culture|simulation|chronicle|audit|quarantine|snapshot|consent)"
    r"-function-(?P<function>[a-z][a-z0-9_]{1,48})"
    r"-mode-(?P<mode>fast|careful|audit)"
    r"(?P<extras>(?:-[a-z_]+-[a-z0-9][a-z0-9_.:()|,;\-]{0,256})*)$"
)

EXTRA_PAIR_RE = re.compile(
    r"-(?P<k>[a-z_]+)-(?P<v>(?:h[a-z0-9]+-[0-9a-f]+|[a-z0-9][a-z0-9_.:()|,;\-]{0,256}?))(?=-(?:[a-z_]+)-|$)"
)


@dataclass
class Parsed:
    uuid: str
    domain: str
    function: str
    mode: str
    extras: Dict[str, str]
    facets_typed: Dict[str, Any]


class Cursor:
    def __init__(self, s: str, i: int = 0) -> None:
        self.s = s
        self.i = i

    def peek(self, n: int = 1) -> str:
        return self.s[self.i : self.i + n]

    def get(self, n: int = 1) -> str:
        out = self.peek(n)
        self.i += n
        return out

    def done(self) -> bool:
        return self.i >= len(self.s)


def parse_agent_name(name: str) -> Parsed:
    m = NAME_RE.match(name)
    if not m:
        raise ValueError("Invalid agent name")
    extras_raw = m.group("extras") or ""
    extras: Dict[str, str] = {}
    for m_ex in EXTRA_PAIR_RE.finditer(extras_raw):
        k = m_ex.group("k")
        v = m_ex.group("v")
        if k in extras:
            raise ValueError(f"Duplicate extra key: {k}")
        extras[k] = v

    facets_typed: Dict[str, Any] = {}
    for k, v in extras.items():
        facets_typed[k] = decode_T(v)

    return Parsed(
        uuid=m.group("uuid"),
        domain=m.group("domain"),
        function=m.group("function"),
        mode=m.group("mode"),
        extras=extras,
        facets_typed=facets_typed,
    )


def encode_agent_name(p: Parsed) -> str:
    head = (
        f"agent-uuidv7-{p.uuid}-domain-{p.domain}-function-{p.function}-mode-{p.mode}"
    )
    if not p.extras:
        return head
    # Preserve original ordering and encoding of extras for round-trip equality
    parts: List[str] = []
    for k, v_str in p.extras.items():
        parts.append(f"-{k}-{v_str}")
    return head + "".join(parts)


def decode_T(s: str) -> Any:
    """Decode typed facet value string into JSON-serializable structure.

    Returns either a primitive (int/float/bool/str) or a structured object with
    a `_type` field to preserve round-trip fidelity.
    """

    # Dot-style list/set (legacy): l.a.b or s.a.b
    if s.startswith("l."):
        items = [decode_T(tok) for tok in s[2:].split(".") if tok]
        return {"_type": "list", "items": items}
    if s.startswith("s."):
        items = sorted(set(tok for tok in s[2:].split(".") if tok))
        return {"_type": "set", "items": [decode_T(tok) for tok in items]}

    # Simple scalars
    if s.startswith("i") and s[1:].isdigit():
        return int(s[1:])
    if s.startswith("f"):
        try:
            return float(s[1:])
        except ValueError:
            pass
    if s == "b0":
        return False
    if s == "b1":
        return True
    if s.startswith("e."):
        return {"_type": "enum", "value": s[2:]}
    if s.startswith("t") and "t" in s and s.endswith("z") and len(s) >= 2:
        # Loose time detection; keep as typed string
        return {"_type": "time", "value": s[1:]}
    if s.startswith("dur-"):
        return {"_type": "duration", "value": s[4:]}
    if s.startswith("sem-"):
        return {"_type": "semver", "value": s[4:]}
    if s.startswith("gh-"):
        return {"_type": "geohash", "value": s[3:]}
    if s.startswith("bb-"):
        try:
            rest = s[3:]
            sw, ne = rest.split("..", 1)
            lon1, lat1 = sw.split("_", 1)
            lon2, lat2 = ne.split("_", 1)
            return {
                "_type": "bbox",
                "sw": {"lon": float(lon1), "lat": float(lat1)},
                "ne": {"lon": float(lon2), "lat": float(lat2)},
            }
        except Exception:
            return s
    if s.startswith("h") and "-" in s:
        alg, hexval = s[1:].split("-", 1)
        return {"_type": "hash", "alg": alg, "hex": hexval}
    if s.startswith("jz-"):
        # Keep compressed canonical JSON blob as-is; decoding is optional/out-of-scope
        return {"_type": "json", "blob": s[3:]}

    # Parenthesized composites and range
    if s.startswith("r"):
        # r<T>..<T> with optional 'o' suffix on end
        left, right, open_end = _split_range_payload(s[1:])
        return {
            "_type": "range",
            "start": decode_T(left),
            "end": decode_T(right),
            "open_end": open_end,
        }

    for tag, tname in (("l", "list"), ("s", "set"), ("p", "tuple"), ("u", "union")):
        if s.startswith(tag + "(") and s.endswith(")"):
            inner = s[len(tag) + 1 : -1]
            sep = ","
            items = [x for x in _split_top_level(inner, sep) if x != ""]
            dec_items = [decode_T(x) for x in items]
            if tname == "set":
                # Canonicalize by encoded form
                dec_items = sorted(dec_items, key=encode_T)
            if tname == "tuple":
                return {"_type": tname, "items": dec_items}
            return {"_type": tname, "items": dec_items}

    if s.startswith("m(") and s.endswith(")"):
        inner = s[2:-1]
        parts = [x for x in _split_top_level(inner, ";") if x != ""]
        items: Dict[str, Any] = {}
        for kv in parts:
            if ":" not in kv:
                continue
            k, v = kv.split(":", 1)
            items[k] = decode_T(v)
        # Canonicalize by key sort during encoding
        return {"_type": "map", "items": items}

    # Fallback plain string
    return s


def encode_T(obj: Any) -> str:
    # Plain primitives
    if isinstance(obj, bool):
        return "b1" if obj else "b0"
    if isinstance(obj, int) and not isinstance(obj, bool):
        return f"i{obj}"
    if isinstance(obj, float):
        s = ("%g" % obj)
        return f"f{s}"
    if isinstance(obj, str):
        # Ambiguous string → return as-is (already encoded or enum token?)
        return obj

    if isinstance(obj, dict) and "_type" in obj:
        t = obj["_type"]
        if t == "enum":
            return f"e.{obj['value']}"
        if t == "range":
            start = encode_T(obj.get("start"))
            end = encode_T(obj.get("end"))
            suffix = "o" if obj.get("open_end") else ""
            return f"r{start}..{end}{suffix}"
        if t in ("list", "tuple", "set", "union"):
            items = obj.get("items", [])
            enc = [encode_T(x) for x in items]
            if t in ("set", "union"):
                enc = sorted(enc)
            return f"{t[0]}(" + ",".join(enc) + ")"
        if t == "map":
            items: Dict[str, Any] = obj.get("items", {})
            enc_pairs = [f"{k}:{encode_T(v)}" for k, v in sorted(items.items())]
            return "m(" + ";".join(enc_pairs) + ")"
        if t == "time":
            return "t" + obj.get("value", "")
        if t == "duration":
            return "dur-" + obj.get("value", "")
        if t == "semver":
            return "sem-" + obj.get("value", "")
        if t == "geohash":
            return "gh-" + obj.get("value", "")
        if t == "bbox":
            sw = obj.get("sw", {})
            ne = obj.get("ne", {})
            return f"bb-{sw.get('lon')}_{sw.get('lat')}..{ne.get('lon')}_{ne.get('lat')}"
        if t == "hash":
            return f"h{obj.get('alg')}-{obj.get('hex')}"
        if t == "json":
            return "jz-" + obj.get("blob", "")

    # Unknown structure → JSON dump as blob for stability (but prefer explicit _type)
    return "jz-" + json.dumps(obj, sort_keys=True, separators=(",", ":")).encode().hex()


def _split_range_payload(payload: str) -> Tuple[str, str, bool]:
    # Find top-level '..'
    depth = 0
    for i in range(len(payload)):
        c = payload[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "." and depth == 0 and payload[i : i + 2] == "..":
            left = payload[:i]
            right = payload[i + 2 :]
            open_end = right.endswith("o")
            if open_end:
                right = right[:-1]
            return left, right, open_end
    # Fallback: treat entire payload as right with empty left
    return "", payload, False


def _split_top_level(s: str, sep: str) -> List[str]:
    out: List[str] = []
    depth = 0
    cur = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "(":
            depth += 1
            cur.append(c)
        elif c == ")":
            depth -= 1
            cur.append(c)
        elif c == sep and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    out.append("".join(cur))
    return out


def main(argv: List[str]) -> None:
    if len(argv) < 2 or argv[1] not in {"parse", "roundtrip"}:
        print("Usage: agent_name_parser.py parse|roundtrip '<id>'", file=sys.stderr)
        sys.exit(2)
    cmd = argv[1]
    name = argv[2] if len(argv) > 2 else ""
    if not name:
        print("Missing id", file=sys.stderr)
        sys.exit(2)
    p = parse_agent_name(name)
    if cmd == "parse":
        print(json.dumps({
            "uuid": p.uuid,
            "domain": p.domain,
            "function": p.function,
            "mode": p.mode,
            "extras": p.extras,
            "facets_typed": p.facets_typed,
        }, indent=2, sort_keys=True))
    else:
        enc = encode_agent_name(p)
        ok = enc == name
        print(json.dumps({"reencoded": enc, "equal": ok}, indent=2))


if __name__ == "__main__":
    main(sys.argv)
