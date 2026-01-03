Title: Extended Facet Catalog (Typed Keys and Examples)

Legend
- i<number>, f<float>, b0|b1, e.<token>, r<..>, l(...), s(...), p(...), m(k:v;...), u(a|b), tYYYY-..., dur-..., sem-x.y.z, gh-..., bb-..., h<alg>-<hex>, jz-<b32>
- Additional: pr-<0..1>, pct-<0..100>, cur-<ISO>_<amt>, uuid-<uuid>, ip4-<a.b.c.d>, ip6-<hex_with_underscores>, cidr-<addr>_<mask>, host-<dots>, tz-<zone>, cron-jz-<b32>

Identity & Ownership
- identity.org: e.<org>
- identity.project: e.<proj>
- identity.env: e.dev|e.staging|e.prod
- identity.team: e.<team>
- identity.owner: e.<owner>
- identity.service: e.<svc>
- identity.component: e.<cmp>
- identity.region: e.<geo>
- identity.zone: e.<zone>
- identity.tier: e.core|e.edge|e.batch

Lifecycle & State
- lifecycle.phase: e.init|e.ready|e.active|e.quiescent|e.degraded|e.quarantined|e.retired
- lifecycle.ttl: dur-<...>
- lifecycle.freeze: u(e.soft|e.hard)
- lifecycle.reason: e.maintenance|e.incident|e.policy
- lifecycle.snapshot: b0|b1
- lifecycle.branch: e.<branch>
- lifecycle.merge.policy: e.human_review|e.auto_soft|e.block
- lifecycle.version: sem-x.y.z
- lifecycle.migration.id: e.<token>
- lifecycle.rollback.to: e.<snap_id>

Scheduling & Windows
- schedule.window: r<t..t>
- schedule.blackout: s(r<t..t>,...)
- schedule.tz: tz-<zone>
- schedule.cron: cron-jz-<b32>
- schedule.calendar: e.<cal>
- schedule.freeze_until: t<...>
- schedule.cooldown: dur-<...>
- schedule.warmup: dur-<...>
- schedule.priority: i<0..9>
- schedule.deadline: t<...>

Governance & Approvals
- gov.mode: e.observe|e.warn|e.enforce
- gov.approvals: m(<change>:iN;...)
- gov.reviewers: s(<role>,...)
- gov.owners: s(<role>,...)
- gov.freeze.switch: b0|b1
- gov.freeze.scope: s(<scope>,...)
- gov.kill.criteria: l(e.<crit>,...)
- gov.audit.level: e.full|e.sampled|e.off
- gov.audit.cf: b0|b1
- gov.charter: sem-x.y.z|hsha256-<hex>

Access & Security
- access.rbac: s(<role>,...)
- access.abac.mode: e.observe|e.warn|e.enforce
- access.cap.kind: e.writer|e.linker
- access.cap.budget: i<N>
- access.cap.expires: t<...>
- sec.authn: e.jwt|e.mtls|e.oidc
- sec.authz: e.rbac|e.abac|e.cap
- sec.secret.ref: e.<ref>
- sec.key.id: e.<key>
- sec.key.rotate: dur-<...>
- sec.token.rotate: dur-<...>
- sec.policy.hash: hsha256-<hex>
- sec.iam.scope: s(<scope>,...)
- sec.mfa: b0|b1
- sec.rate.limit: i<rps>

Privacy & Data Policy
- privacy.tier: e.public|e.internal|e.secret
- privacy.redaction: e.strict|e.lenient
- privacy.purpose: s(e.ops,e.research,e.product,...) 
- privacy.dp.epsilon: f<val>
- privacy.dp.delta: f<val>
- privacy.retention: dur-<...>
- privacy.subject.rights: s(e.access,e.delete,e.export)
- privacy.logging: e.minimal|e.normal|e.verbose

Observability & Logging
- obs.log.level: e.debug|e.info|e.warn|e.error
- obs.sample.rate: pr-<0..1>
- obs.trace: b0|b1
- obs.metrics.ns: e.<ns>
- obs.event.schema: hsha256-<hex>
- obs.retention: dur-<...>
- obs.privacy.tier: e.tier1|e.tier2
- obs.correlation: e.required|e.optional
- obs.topic: s(<topic>,...)
- obs.dashboard: e.<slug>

Performance & SLOs
- perf.latency.slo: dur-<...>
- perf.latency.p95: dur-<...>
- perf.throughput: i<qps>
- perf.error.budget: pr-<0..1>
- perf.uptime.target: pct-<...>
- perf.queue.depth: i<N>
- perf.concurrency: i<N>
- perf.cpu.util.max: pct-<...>
- perf.mem.util.max: pct-<...>
- perf.backoff: m(init:dur-<...>;mult:f<r>)

Networking
- net.protocol: e.http|e.grpc|e.tcp|e.kafka
- net.endpoint: host-<dots>
- net.port: i<N>
- net.timeout: dur-<...>
- net.retry: m(max:iN;backoff:f<r>)
- net.ip4.allow: s(ip4-<...>,...)
- net.ip6.allow: s(ip6-<...>,...)
- net.cidr.allow: s(cidr-<...>,...)
- net.tls: b0|b1
- net.sni: host-<dots>

Compute & Resources
- res.cpu.limit: f<cores>
- res.mem.limit: i<bytes>
- res.disk.limit: i<bytes>
- res.gpu.count: i<N>
- res.workers: i<N>
- res.shards: i<N>
- res.batch.size: i<N>
- res.queue.capacity: i<N>
- res.cache.size: i<N>
- res.eviction: e.lru|e.lfu|e.none

Datasets & Schemas
- data.dataset: e.<id>
- data.version: sem-x.y.z
- data.schema: hsha256-<hex>
- data.codec: e.parquet|e.orc|e.jsonl
- data.partition: s(e.ds,e.geo,e.hash)
- data.lineage: jz-<b32>
- data.retention: dur-<...>
- data.quality.min: pct-<...>
- data.pii: e.none|e.partial|e.full

ML & Models
- ml.model.id: e.<id>
- ml.model.ver: sem-x.y.z
- ml.model.hash: hsha256-<hex>
- ml.family: e.gpt|e.bert|e.tree|e.tabular
- ml.task: e.clf|e.reg|e.gen|e.rank
- ml.eval.policy: e.strict|e.fast
- ml.fairness: s(e.di,e.eod,e.dp)
- ml.dp.epsilon: f<val>
- ml.training.data: hsha256-<hex>
- ml.eval.dataset: e.<id>
- ml.ci: pr-<0..1>

Experiments & Flags
- exp.id: e.<id>
- exp.cohort: s(e.a,e.b,e.c)
- exp.rollout: pct-<...>
- exp.stage: e.dryrun|e.pilot|e.ramp|e.ga
- exp.kill: l(e.guardrail_breach,e.complaint_spike)
- flag.feature: e.<slug>
- flag.on: b0|b1

Economics & Pricing
- econ.price: cur-USD_<amt>
- econ.discount: pct-<...>
- econ.take_rate: pr-<...>
- econ.margin.min: pct-<...>
- econ.budget.daily: cur-USD_<amt>
- econ.cost.unit: cur-USD_<amt>
- econ.cash.payback.max: dur-<...>
- econ.revenue.ns: e.<ns>
- econ.promo.id: e.<id>

Ethics & Rights
- ethics.ria.id: e.<id>
- ethics.ria.mode: e.required|e.optional
- ethics.rights: s(e.access,e.opt_out,e.redress)
- ethics.redress: e.enabled|e.disabled
- ethics.ombuds: s(e.OMB_1,e.OMB_2)
- ethics.review.cadence: dur-<...>

Audit & Chronicle
- audit.level: e.full|e.sampled|e.off
- audit.cf.enabled: b0|b1
- audit.evidence.pkg: hsha256-<hex>
- audit.finding.sev: e.low|e.med|e.high
- audit.owner: e.<role>
- chron.topic: s(<topic>,...)
- chron.sig.required: b0|b1
- chron.retention: dur-<...>

Snapshots & Branching
- snap.cadence: dur-<...>
- snap.merge.policy: e.human_review|e.auto_soft
- snap.seal: b0|b1
- snap.branch.name: e.<name>
- snap.restore.to: e.<id>
- snap.delta.codec: e.zstd|e.lz4|e.snappy

Quarantine & Isolation
- quar.mode: e.region|e.service|e.account
- quar.whitelist: s(<api>,...)
- quar.rehydrate: e.auto|e.manual
- quar.freeze: e.soft|e.hard
- quar.honey: b0|b1
- quar.release.criteria: m(<metric>:f<thr>)

World Simulation
- world.cosmology: e.euclidean|e.spherical|e.hyperbolic
- world.dims: i2|i3|iN
- world.lattice: e.continuous|e.square|e.hex|e.cubic
- world.dt: f<sec>
- world.bc: u(e.periodic|e.reflective|e.absorbing)
- world.invariants: s(e.energy,e.momentum,e.charge)
- world.rng: m(alg:e.xoshiro;seed:iN)
- world.noise: m(type:e.gaussian;sigma:f<val>)
- world.resources: m(energy:f<val>;memory:i<bytes>)
- world.telos: u(e.play|e.knowledge|e.robustness)
- world.psi_min: f<0..1>
- world.kill: l(e.psi_breach,e.metric_critical,e.audit_fail)
- world.units: e.si|e.cgs
- world.region_bbox: bb-<lon1>_<lat1>..<lon2>_<lat2>

This catalog is not exhaustive; extend with namespaced keys and typed values using the grammar above. Prefer stable, legible keys; avoid collisions by namespacing (e.g., obs., perf., sec., econ., ml., world.).

