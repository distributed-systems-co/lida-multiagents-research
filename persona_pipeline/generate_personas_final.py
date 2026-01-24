#!/usr/bin/env python3
"""
FULL PERSONA GENERATION

Stack:
1. DSPy + OpenRouter for search term generation
2. Parallel AI for web searches
3. OpenRouter for persona synthesis
"""

import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import dspy
import httpx

from people import get_all_people, PEOPLE

# === CONFIG ===
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
PARALLEL_API_URL = os.getenv("PARALLEL_API_URL", "https://api.parallel.ai")
PARALLEL_API_KEY = os.getenv("PARALLEL_API_KEY", "")

OUTPUT_DIR = Path(__file__).parent / "personas"
CONCURRENCY = 2

# Models (via OpenRouter)
TERM_GEN_MODEL = "anthropic/claude-haiku-4.5"
SYNTHESIS_MODEL = "anthropic/claude-sonnet-4.5"


# === DSPY SETUP ===
def configure_dspy():
    """Configure DSPy with OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set")

    lm = dspy.LM(
        model=f"openrouter/{TERM_GEN_MODEL}",
        api_key=OPENROUTER_API_KEY,
        api_base=OPENROUTER_BASE_URL,
        max_tokens=4000,
    )
    dspy.configure(lm=lm)
    return lm


class DeepSearchTerms(dspy.Signature):
    """Generate exhaustive search terms for deep persona research."""

    name: str = dspy.InputField()
    role: str = dspy.InputField()
    category: str = dspy.InputField()

    # Each category gets 5 terms
    biography: list[str] = dspy.OutputField(desc="5 terms for bio, education, career")
    current: list[str] = dspy.OutputField(desc="5 terms for 2024-2025 news, current role")
    personality: list[str] = dspy.OutputField(desc="5 terms for personality, quirks, management style")
    communication: list[str] = dspy.OutputField(desc="5 terms for quotes, interviews, how they talk")
    allies: list[str] = dspy.OutputField(desc="5 terms for friends, allies, inner circle")
    enemies: list[str] = dspy.OutputField(desc="5 terms for rivals, enemies, people who hate them")
    feuds: list[str] = dspy.OutputField(desc="5 terms for feuds, beefs, fallings out, betrayals")
    family: list[str] = dspy.OutputField(desc="5 terms for family, spouse, romantic life")
    legal: list[str] = dspy.OutputField(desc="5 terms for lawsuits, legal troubles")
    criminal: list[str] = dspy.OutputField(desc="5 terms for criminal investigations, indictments, fraud")
    regulatory: list[str] = dspy.OutputField(desc="5 terms for SEC, FTC, DOJ, congressional")
    scandals: list[str] = dspy.OutputField(desc="5 terms for scandals, PR disasters")
    misconduct: list[str] = dspy.OutputField(desc="5 terms for harassment, discrimination allegations")
    financial: list[str] = dspy.OutputField(desc="5 terms for money troubles, conflicts of interest")
    vulnerabilities: list[str] = dspy.OutputField(desc="5 terms for weaknesses, insecurities, failures")
    leverage: list[str] = dspy.OutputField(desc="5 terms for what they owe, who controls them")
    twitter: list[str] = dspy.OutputField(desc="5 terms for twitter drama, deleted tweets")
    reddit: list[str] = dspy.OutputField(desc="5 terms for reddit/HN discussions")
    gossip: list[str] = dspy.OutputField(desc="5 terms for rumors, secrets, leaks")
    policy: list[str] = dspy.OutputField(desc="5 terms for AI/chip policy positions")


# Fallback templates if DSPy fails
FALLBACK_TEMPLATES = {
    "biography": ["{name} biography background", "{name} education career history", "{name} early life origin"],
    "current": ["{name} 2024 2025 news", "{name} latest recent", "{name} current role today"],
    "personality": ["{name} personality traits", "{name} management style", "{name} temper ego"],
    "communication": ["{name} quotes statements", "{name} interview podcast", '"{name}" said'],
    "allies": ["{name} friends allies", "{name} inner circle", "{name} mentor"],
    "enemies": ["{name} enemies rivals", "{name} critics opponents", "{name} hates"],
    "feuds": ["{name} feud beef", "{name} falling out", "{name} betrayed"],
    "family": ["{name} married spouse", "{name} family children", "{name} divorce affair"],
    "legal": ["{name} lawsuit sued", "{name} legal troubles", "{name} court case"],
    "criminal": ["{name} criminal investigation", "{name} indicted charges", "{name} fraud"],
    "regulatory": ["{name} SEC investigation", "{name} FTC DOJ", "{name} congressional hearing"],
    "scandals": ["{name} scandal controversy", "{name} PR disaster", "{name} resigned fired"],
    "misconduct": ["{name} harassment allegations", "{name} discrimination", "{name} toxic"],
    "financial": ["{name} financial troubles", "{name} conflict of interest", "{name} insider trading"],
    "vulnerabilities": ["{name} weakness failure", "{name} insecure defensive", "{name} regret mistake"],
    "leverage": ["{name} owes favor", "{name} controlled by", "{name} compromised"],
    "twitter": ["{name} twitter drama", "{name} deleted tweet", "{name} twitter fight"],
    "reddit": ['"{name}" reddit', '"{name}" hacker news', "{name} glassdoor"],
    "gossip": ["{name} rumors gossip", "{name} secretly actually", "{name} leaked"],
    "policy": ["{name} AI policy", "{name} China stance", "{name} regulation"],
}


def generate_search_terms(name: str, role: str, category: str) -> list[str]:
    """Generate search terms using DSPy or fallback."""
    all_terms = []

    try:
        configure_dspy()
        predictor = dspy.Predict(DeepSearchTerms)
        result = predictor(name=name, role=role, category=category)

        for field_name in FALLBACK_TEMPLATES.keys():
            terms = getattr(result, field_name, [])
            if terms:
                all_terms.extend(terms)

    except Exception as e:
        print(f"    DSPy failed ({e}), using templates")
        for templates in FALLBACK_TEMPLATES.values():
            for t in templates:
                all_terms.append(t.format(name=name, role=role))

    # Dedupe
    seen = set()
    unique = []
    for t in all_terms:
        if t.lower() not in seen:
            seen.add(t.lower())
            unique.append(t)

    return unique


# === PARALLEL AI SEARCH ===
class ParallelAISearcher:
    """Web search via Parallel AI."""

    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=60.0,
            headers={"Authorization": f"Bearer {PARALLEL_API_KEY}"} if PARALLEL_API_KEY else {},
        )

    async def close(self):
        await self.client.aclose()

    async def search(self, query: str, max_results: int = 10) -> list[dict]:
        """Run search via Parallel AI."""
        try:
            resp = await self.client.post(
                f"{PARALLEL_API_URL}/v1beta/search",
                json={"objective": query, "max_results": max_results},
                headers={"Parallel-Beta": "search-extract-2025-10-10"},
            )
            resp.raise_for_status()
            return resp.json().get("results", [])
        except Exception as e:
            return []

    async def batch_search(self, queries: list[str], max_per: int = 8) -> list[dict]:
        """Run many searches."""
        all_results = []
        seen_urls = set()
        batch_size = 8

        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            tasks = [self.search(q, max_per) for q in batch]
            results = await asyncio.gather(*tasks)

            for batch_results in results:
                for r in batch_results:
                    url = r.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append(r)

            print(f"    {min(i + batch_size, len(queries))}/{len(queries)} queries, {len(all_results)} sources")
            await asyncio.sleep(0.3)

        return all_results


# === OPENROUTER SYNTHESIS ===
async def synthesize_persona(name: str, role: str, category: str, research: str) -> dict:
    """Generate persona via OpenRouter."""

    prompt = f"""Create a COMPREHENSIVE persona dossier for {name}.

For multi-agent simulation. I need EVERYTHING - good, bad, ugly.

MUST INCLUDE:

1. LEGAL EXPOSURE - investigations, indictments, lawsuits, regulatory issues, settlements
2. CONTROVERSIES - scandals, misconduct allegations, cover-ups, hypocrisies, lies caught
3. RELATIONSHIPS - inner circle, allies, enemies, burned bridges, who they owe, who has leverage
4. PRESSURE POINTS - career vulnerabilities, reputation risks, legal leverage, psychological triggers, skeletons
5. PERSONALITY - dark traits, insecurities, ego needs, triggers, under-pressure behavior
6. COMMUNICATION - real voice vs PR, verbal tics, 5+ actual quotes, manipulation tactics
7. INTERNET - twitter beefs, deleted tweets, memes, reddit reputation, leaks

RESEARCH:
{research[:35000] if research else '[No research data - use your knowledge]'}

CONTEXT:
- Name: {name}
- Role: {role}
- Category: {category}

Be specific. Name names. Include dates. Don't sanitize.

Return JSON:
{{
  "name": "",
  "current_title": "",
  "background": {{"origin_story": "", "education": [], "career_arc": [], "net_worth": ""}},
  "personality": {{"core_traits": [], "dark_traits": [], "triggers": [], "insecurities": [], "ego_needs": []}},
  "communication": {{"public_persona": "", "private_persona": "", "verbal_tics": [], "sample_quotes": []}},
  "relationships": {{"inner_circle": [], "allies": [], "enemies": [], "burned_bridges": [], "owes_favors_to": [], "has_leverage_over": []}},
  "legal_exposure": {{"investigations": [], "indictments": [], "lawsuits": [], "settlements": [], "potential_exposure": []}},
  "controversies": {{"scandals": [], "misconduct": [], "hypocrisies": [], "lies_caught": []}},
  "pressure_points": {{"career_vulnerabilities": [], "reputation_risks": [], "legal_leverage": [], "psychological_triggers": [], "skeletons": [], "what_keeps_them_up": []}},
  "internet_presence": {{"twitter_handle": "", "twitter_style": "", "twitter_beefs": [], "meme_status": "", "reddit_reputation": ""}},
  "worldview": {{"ai_philosophy": "", "china_stance": "", "regulation_views": "", "blindspots": []}},
  "current_state": {{"priorities": [], "battles": [], "momentum": "", "stress_level": ""}},
  "simulation_guide": {{"how_to_embody": "", "never_say": [], "hot_buttons": [], "how_to_flatter": [], "how_to_provoke": []}}
}}"""

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": SYNTHESIS_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 8000,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]

            # Extract JSON
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {"name": name, "raw": text[:5000], "error": "json_parse_failed"}

        except Exception as e:
            return {"name": name, "error": str(e)}


# === MAIN PIPELINE ===
async def generate_persona(person: dict, searcher: ParallelAISearcher) -> dict:
    """Full pipeline for one person."""
    name = person["name"]
    role = person["role"]
    category = person["category_name"]

    print(f"\n{'='*70}")
    print(f"[{name}]")

    # 1. Generate search terms (DSPy)
    print("  [TERMS] Generating via DSPy...")
    terms = generate_search_terms(name, role, category)
    print(f"    {len(terms)} search terms")

    # 2. Run searches (Parallel AI)
    print("  [SEARCH] Running via Parallel AI...")
    results = await searcher.batch_search(terms[:80])  # Cap at 80

    # Format research
    research = ""
    for i, r in enumerate(results[:50]):
        research += f"\n[{i+1}] {r.get('title', '')}\n{r.get('url', '')}\n{' '.join(r.get('excerpts', []))[:400]}\n"

    # 3. Synthesize (OpenRouter)
    print(f"  [SYNTHESIZE] Generating via OpenRouter ({SYNTHESIS_MODEL})...")
    persona = await synthesize_persona(name, role, category, research)

    # Metadata
    persona["_metadata"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "role": role,
        "search_terms": len(terms),
        "sources": len(results),
        "model": SYNTHESIS_MODEL,
    }

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r'[^a-z0-9_]', '_', name.lower()).strip('_')
    safe_name = re.sub(r'_+', '_', safe_name)
    filepath = OUTPUT_DIR / f"{safe_name}.json"

    with open(filepath, "w") as f:
        json.dump(persona, f, indent=2, ensure_ascii=False)
    print(f"  [SAVED] {filepath.name}")

    return persona


async def run_pipeline(people: list[dict], concurrency: int = CONCURRENCY):
    """Run full pipeline."""
    print("=" * 70)
    print("PERSONA GENERATION")
    print("=" * 70)
    print(f"Stack: DSPy → Parallel AI → OpenRouter")
    print(f"People: {len(people)}")
    print(f"Term model: {TERM_GEN_MODEL}")
    print(f"Synthesis model: {SYNTHESIS_MODEL}")
    print("=" * 70)

    searcher = ParallelAISearcher()
    semaphore = asyncio.Semaphore(concurrency)

    async def process(p):
        async with semaphore:
            try:
                return await generate_persona(p, searcher)
            except Exception as e:
                print(f"[FAILED] {p['name']}: {e}")
                return {"name": p["name"], "error": str(e)}

    try:
        results = await asyncio.gather(*[process(p) for p in people])
    finally:
        await searcher.close()

    ok = sum(1 for r in results if "error" not in r)
    print(f"\n{'='*70}")
    print(f"DONE: {ok}/{len(people)}")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--person", type=str)
    parser.add_argument("-c", "--category", type=str)
    parser.add_argument("-n", "--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("-l", "--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for cid, cat in PEOPLE.items():
            print(f"\n{cat['name']} [{cid}]:")
            for p in cat["people"]:
                print(f"  {p['name']:30} — {p['role']}")
        print(f"\nTotal: {len(get_all_people())}")
        return

    if args.person:
        people = [p for p in get_all_people() if args.person.lower() in p["name"].lower()]
    elif args.category:
        cat = PEOPLE.get(args.category)
        if not cat:
            print(f"Categories: {list(PEOPLE.keys())}")
            return
        people = [{**p, "category_id": args.category, "category_name": cat["name"]} for p in cat["people"]]
    else:
        people = get_all_people()

    if not people:
        print("No people matched")
        return

    asyncio.run(run_pipeline(people, args.concurrency))


if __name__ == "__main__":
    main()
