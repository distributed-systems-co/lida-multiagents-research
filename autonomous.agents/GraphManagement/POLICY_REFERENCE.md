# Graph Management Policy Reference

This document summarizes access control and policy enforcement across the Graph Management services, combining static/effective RBAC, ABAC policies, proposal approvals, and capability tokens.

## Overview
- Static RBAC: Role → permissions, enforced via `require_permissions(required)`.
- Effective RBAC: Graph-derived roles/permissions, enforced via `require_permissions_effective(required, org_param, project_param)`.
- ABAC: Policy rules (allow/deny) by scope with mode precedence, evaluated via `enforce_abac(action, …)`.
- Approvals: Per-change-type approval thresholds from policy.
- Capabilities: Fine-grained writer/linker tokens with daily budgets.

## Static RBAC (security.py)
- Roles and permissions (prefix semantics: `prefix:` matches `prefix:read|write|*`, `*` means all):
  - org_admin → ["*"]
  - schema_admin → [
    "schemas:", "proposals:", "policy:", "query:", "orgs:", "teams:", "projects:",
    "tools:", "datasets:", "secrets:", "agents:", "runs:", "budgets:", "events:"
  ]
  - schema_approver → ["schemas:read", "proposals:review", "proposals:apply"]
  - schema_reviewer → ["schemas:read", "proposals:review"]
  - developer → [
    "schemas:read", "proposals:create", "query:read", "query:write",
    "tools:read", "datasets:read", "runs:write", "runs:read", "events:read"
  ]
  - readonly → ["schemas:read", "query:read", "tools:read", "datasets:read", "runs:read", "events:read"]
  - service → ["schemas:read", "query:", "tools:read", "datasets:read", "runs:"]

- Enforcers:
  - `require_permissions(required)` at security.py:162
  - `has_permission(principal, permission)` at security.py:171

## Effective RBAC (graph-derived)
- Resolver: `_effective_roles_perms(principal, org_id, project_id)` at graph_management_service.py:320
  - Resolves via Cypher: `MATCH (u:User)-[HAS_ROLE]->(Role)-[:GRANTS]->(Permission)` scoped by `hr.scope ∈ {org:..., project:...}`.
  - Fallback: expands `_ROLE_GRANTS` into atomic permissions.
- Enforcer: `require_permissions_effective(required, org_param, project_param)` at graph_management_service.py:354

## Atomic Permissions and Role Grants
- Canonical atomics (graph_management_service.py:696):
  - `schemas:read|write`, `proposals:*`, `policy:*`, `query:*`, `orgs:*`, `teams:*`, `projects:*`, `tools:*`,
    `datasets:*`, `secrets:*`, `agents:*`, `runs:*`, `budgets:*`, `events:*`.
- Role grants (graph_management_service.py:733): mapping of role → patterns and explicit atomics.

## ABAC Policy Model
- Schema (schemas.py):
  - PolicyUpsertRequest (schemas.py:186)
    - scope_type: `global | org | env | project`
    - scope_id: string | null
    - mode: `observe | warn | enforce` (optional)
    - approvals: Dict[str,int] (e.g., `{ "add_node_schema": 1 }`)
    - rules: List[Rule]
  - Rule fields (enforced in graph_management_service.py:371):
    - effect: `deny | allow`
    - actions: List[string] (tested against `enforce_abac(action)`)
    - roles: List[string]
    - env_ids: List[string]
    - projects: List[string]
  - PolicyResponse (schemas.py:183): mode
  - PolicyDetailResponse (schemas.py:198): scope_type, scope_id, mode, approvals

- Evaluation order (graph_management_service.py:371):
  1) Any matching deny rule → deny
  2) Else any matching allow rule → allow
  3) Else default by highest-precedence mode along scopes (project > env > org > global):
     - `enforce` → deny by default
     - `warn/observe` → allow by default

## Approvals Policy (Proposals)
- Required approvals per change type (graph_management_service.py:1969):
  - `_approval_threshold_for_async(change_type, org_id)` reads `Policy{id='org:…' or 'global'}.approvals[change_type]`, defaults to 1.
- Endpoints:
  - GET `/v1/registry/proposals/{pid}/approvals` (graph_management_service.py:2208) → current/required approvals
  - POST `/v1/registry/proposals/{pid}/validate` (proposals:review) (graph_management_service.py:1912)
  - POST `/v1/registry/proposals/{pid}/apply` (proposals:apply) (graph_management_service.py:1929) → enforces approvals_current ≥ approvals_required

## Capability Tokens (Fine-Grained)
- Create (schemas.py:259): CapabilityTokenCreateRequest
  - kind: `writer | linker`
  - target_label: string (writer)
  - relationship_type: string (linker)
  - version: Optional[string]
  - budget_per_day: int (≤0 = unlimited)
  - expires_at: Optional[ISO datetime]
  - subject: Optional[string] (defaults to requester)
  - note: Optional[string]
- Response (schemas.py:278): CapabilityTokenResponse fields: id, kind, target_label, relationship_type, version, subject, budget_per_day, used_today, day_key, expires_at, note
- Service endpoints (graph_management_service.py):
  - POST `/v1/capabilities` (policy:write) (graph_management_service.py:2320)
    - Creates CapabilityToken node; optional expiry; daily budget counters
  - GET `/v1/capabilities/{cid}` (policy:read) (graph_management_service.py:2353)
- Consumption (graph_management_service.py:287): `_consume_capability(principal, kind, target_label?, relationship_type?)`
  - Validates presence, expiry, daily budget; increments used_today atomically
- Enforced on ingest endpoints:
  - Nodes ingest requires writer on label (graph_management_service.py:2294)
  - Relationships ingest requires linker on type (graph_management_service.py:2340)

## Policy Endpoints (Summary)
- POST `/v1/policies/upsert` (policy:write) (graph_management_service.py:948)
  - Upserts Policy{id=`global`|`org:…`|`env:…`|`project:…`}, sets mode/approvals/rules and links via HAS_POLICY
- GET `/v1/policies/global` (policy:read) (graph_management_service.py:981)
- GET `/v1/policies/{scope_type}/{scope_id}` (policy:read) (graph_management_service.py:990)
- GET `/v1/policy` (global mode convenience) (graph_management_service.py:2298)
- POST `/v1/policy` (set global mode only) (graph_management_service.py:2307)

## Enforcement Summary
- Static RBAC: `require_permissions([...])` checks `_ROLE_PERMISSIONS` (security.py).
- Effective RBAC: `require_permissions_effective([...])` checks graph‑derived grants.
- ABAC: `enforce_abac("action")` evaluates Policy.rules with allow/deny, scoped by env/org/project, and Policy.mode.
- Approvals: thresholds from Policy.approvals per change type.
- Capabilities: budgeted writer/linker tokens enforced during ingest.

## Examples
### Permission Patterns
- Prefix patterns: `schemas:` implies `schemas:read` and `schemas:write` (or `schemas:*` if applicable).
- Atomics: `proposals:review`, `proposals:apply`, `query:read`, `runs:write`, `events:read`.

### ABAC Rules (JSON)
```
{
  "mode": "enforce",
  "rules": [
    { "effect": "deny", "actions": ["datasets:write"], "projects": ["prod"] },
    { "effect": "allow", "actions": ["datasets:write"], "roles": ["schema_admin"], "projects": ["staging","dev"] }
  ],
  "approvals": { "add_node_schema": 2, "apply_proposal": 1 }
}
```

### Capability Tokens
- Writer for label `Person`, budget 100/day, auto‑expires:
```
{
  "kind": "writer",
  "target_label": "Person",
  "budget_per_day": 100,
  "expires_at": "2026-01-01T00:00:00Z",
  "note": "ETL batch writer"
}
```

## Operational Guidance
- Prefer `require_permissions_effective` for org/project‑aware checks.
- Set Policy.mode progressively: start with `observe` → `warn` → `enforce`.
- Define approvals per change type; surface thresholds via `/v1/registry/proposals/{pid}/approvals`.
- Issue capability tokens for automated ingestion with clear budgets and expirations.

## Concepts and Best Practices

### Philosophy and Design Goals
- Least privilege: default to the minimum rights necessary to perform work.
- Defense in depth: combine RBAC, ABAC, approvals, and capabilities for layered control.
- Progressive hardening: move from observe → warn → enforce only after telemetry confirms safety.
- Auditability: every allow/deny must be attributable to a rule, mode, or grant.
- Human legibility: permission strings and rules are simple, composable, and documented.

### Mental Model and Scopes
ABAC evaluates from most specific scope (project) to least (global):
```
global
  └─ org:{org_id}
       └─ env:{env_id}
            └─ project:{project_id}
```
- Any matching deny at a specific scope wins over allows at broader scopes.
- If no rules match, the most specific scope’s mode determines default allow/deny.

### Roles Explained (intent and typical holders)
- org_admin: superuser for an org; use sparingly for break‑glass and bootstrapping.
- schema_admin: manages schemas, proposals, policies; commonly platform owners.
- schema_approver: change control board; final sign‑off on proposals.
- schema_reviewer: peer review without apply powers.
- developer: proposes changes, runs queries/jobs under guardrails.
- readonly: analysts and auditors; broad read, no write.
- service: non‑human principals; pair with capabilities for writes.

### Permission Grammar
- `domain:read|write|*` are atomics; `domain:` grants both read and write.
- Examples: `datasets:` implies `datasets:read` and `datasets:write`.
- Prefer explicit atomics in production for clarity and least privilege.

### Approvals Recipes
- Production, breaking changes: 2 approvals minimum.
- Non‑breaking or dev/staging: 1 approval.
- Emergency rollback: pre‑authorized change type with 1 approval.

### Capability Token Best Practices
- One token per job; scope to a single label/type.
- Set budget_per_day with 20–50% headroom; alert on spikes.
- Short expirations with automated rotation; store in a secrets manager.
- Combine with ABAC deny rules for emergency stop without reissuing tokens.

### End‑to‑End Enforcement Flow
1) Identity + optional capability token received.
2) Static RBAC pre‑check.
3) Effective RBAC resolution (org/project scope).
4) ABAC rule evaluation (deny/allow/mode default).
5) Approvals threshold enforced for proposal apply.
6) Capability budget/expiry validated for ingest.
7) Decision logged with rationale and correlation_id.

Sequence (high‑level):
```
caller → gateway → static RBAC ✓ → effective RBAC ✓ → ABAC ✓/✗ → approvals ✓/✗ → capability ✓/✗ → handler
                                                    ↘ audit (allow/deny, scope, rule, mode)
```

### Threat Model and Pitfalls
- Too many org_admins: reduce blast radius; require break‑glass with time‑bound elevation.
- Policy drift: schedule periodic reviews; keep `observe` dashboards.
- Unlimited tokens: enforce budgets; revoke on anomalies.
- Surprising grants: prefer effective RBAC checks consistently.
- Hard default denies: make explicit allowlists for critical maintenance flows.

### Observability and Audit
- Log: action, principal, roles, required permission, scope, matched rule(s), mode, outcome, correlation_id.
- Chronicle all policy upserts with actor and diff.
- Example event:
```
{
  "ts": "2025-11-10T01:23:45Z",
  "principal": "user:alice",
  "action": "datasets:write",
  "org": "acme", "project": "prod",
  "effective_roles": ["developer"],
  "abac": { "mode": "enforce", "matched": [{"effect":"deny","scope":"project:prod"}] },
  "approvals": { "required": 2, "current": 1 },
  "capability": { "id": "cap:xyz", "ok": false, "reason": "budget_exhausted" },
  "outcome": "deny", "correlation_id": "req-7f9c"
}
```

### Rollout and Migration
- Phase 1: RBAC only; catalog atomics and grants.
- Phase 2: ABAC in `observe`; instrument dashboards.
- Phase 3: ABAC in `warn`; fix false positives.
- Phase 4: ABAC in `enforce`; add freeze recipe for emergencies.

### ABAC Rule Recipes
- Deny writes to datasets in prod except schema admins:
```
{
  "mode": "enforce",
  "rules": [
    { "effect": "deny", "actions": ["datasets:write"], "projects": ["prod"] },
    { "effect": "allow", "actions": ["datasets:write"], "roles": ["schema_admin"], "projects": ["prod"] }
  ]
}
```

- Allow reads everywhere; warn on writes in staging env:
```
{
  "mode": "warn",
  "rules": [
    { "effect": "allow", "actions": ["tools:read"] },
    { "effect": "deny", "actions": ["tools:write"], "env_ids": ["staging"] }
  ]
}
```

- Organization freeze window:
```
{
  "mode": "enforce",
  "rules": [
    { "effect": "deny", "actions": ["schemas:write","proposals:apply"], "roles": ["*"], "projects": ["*"] }
  ]
}
```

### FAQ
- Which check runs first? Static RBAC → Effective RBAC → ABAC; approvals/capabilities apply to specific endpoints.
- How do prefix permissions work? `domain:` expands to read+write; prefer explicit atomics for least privilege.
- Should services use roles or tokens? Both; minimal `service` role plus narrow capability tokens.
- How to simulate policy impact? Use `observe` and inspect decision logs before enforcing.

### Glossary
- RBAC: Role‑Based Access Control.
- ABAC: Attribute‑Based Access Control.
- Atomics: Minimal permission strings, e.g., `schemas:read`.
- Capability token: Budgeted credential for narrow write/link powers.
- Mode: Default decision when no rules match (observe/warn/enforce).

---
File index:
- security.py:89 (_ROLE_PERMISSIONS), 162 (require_permissions), 171 (has_permission)
- graph_management_service.py:320 (_effective_roles_perms), 354 (require_permissions_effective), 371 (enforce_abac)
- graph_management_service.py:696 (_ATOMIC_PERMISSIONS), 733 (_ROLE_GRANTS)
- graph_management_service.py:948, 981, 990, 2298, 2307 (policy endpoints)
- graph_management_service.py:1969, 2208, 1912, 1929 (approvals flow)
- graph_management_service.py:2320, 2353, 287, 2294, 2340 (capabilities + ingest enforcement)
- schemas.py:183, 186, 198 (policy schemas), 259, 278 (capability schemas)
