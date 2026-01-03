Title: Chronicle Causal Query Cookbook

Queries (pseudo-Cypher)

- Path from action to outcomes within lag window:
  MATCH (a:Action {id:$aid})-[:CAUSES*1..5]->(o:Outcome)
  WHERE duration.between(a.ts, o.ts) <= duration($maxLag)
  RETURN a,o

- Why did outcome Y change? Top k causal paths by strength:
  MATCH p=(a:Action)-[r:CAUSES*1..5]->(o:Outcome {id:$oid})
  WITH p, reduce(s=0, e IN relationships(p) | s + e.strength) AS score
  RETURN p ORDER BY score DESC LIMIT $k

- What if we remove edge e (counterfactual skeleton):
  MATCH (x)-[e:CAUSES {id:$eid}]->(y)
  CALL remove_edge_and_recompute($eid) YIELD delta
  RETURN delta

- Link completeness (coverage of actions to outcomes):
  MATCH (a:Action)
  OPTIONAL MATCH (a)-[:CAUSES*1..5]->(o:Outcome)
  WITH a, count(o) AS c
  RETURN avg(CASE WHEN c>0 THEN 1 ELSE 0 END) AS coverage

- Edge evidence audit (correlation vs ablation vs randomized):
  MATCH ()-[e:CAUSES]->()
  RETURN e.evidence AS evidence, count(*) AS n ORDER BY n DESC

Operational notes
- Enforce signature verification on Action nodes before ingestion.
- Provide redacted views for sensitive scopes; log redaction justifications.
- Indexes: (Action.ts), (Outcome.ts), (CAUSES.from,to), (Action.correlation_id)

