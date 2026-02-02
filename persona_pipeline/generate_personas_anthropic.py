#!/usr/bin/env python3
"""
FULL PERSONA GENERATION - Anthropic Direct Version

Uses:
- Anthropic Claude directly for synthesis (no Parallel.ai dependency)
- Tavily/Serper/Brave for web search (configurable)
- Falls back to generating search queries if no search API

This version works standalone without requiring the Parallel.ai server.
"""

import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

# Anthropic
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic not installed, pip install anthropic")

from people import get_all_people, PEOPLE
from deep_search_terms import generate_deep_terms, flatten_terms

OUTPUT_DIR = Path(__file__).parent / "personas"
CONCURRENCY = 2

# Search API config (pick one)
SEARCH_API = os.getenv("SEARCH_API", "tavily")  # tavily, serper, brave, none
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")

# Anthropic config
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
SYNTHESIS_MODEL = os.getenv("SYNTHESIS_MODEL", "claude-sonnet-4-20250514")


class WebSearcher:
    """Multi-backend web search."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        await self.client.aclose()

    async def search(self, query: str, max_results: int = 10) -> list[dict]:
        """Search using configured backend."""
        if SEARCH_API == "tavily" and TAVILY_API_KEY:
            return await self._tavily_search(query, max_results)
        elif SEARCH_API == "serper" and SERPER_API_KEY:
            return await self._serper_search(query, max_results)
        elif SEARCH_API == "brave" and BRAVE_API_KEY:
            return await self._brave_search(query, max_results)
        else:
            return []

    async def _tavily_search(self, query: str, max_results: int) -> list[dict]:
        """Tavily search API."""
        try:
            resp = await self.client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "max_results": max_results,
                    "include_raw_content": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "excerpts": [r.get("content", "")[:500]],
                }
                for r in data.get("results", [])
            ]
        except Exception as e:
            print(f"      Tavily error: {e}")
            return []

    async def _serper_search(self, query: str, max_results: int) -> list[dict]:
        """Serper.dev search API."""
        try:
            resp = await self.client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": SERPER_API_KEY},
                json={"q": query, "num": max_results},
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("link", ""),
                    "excerpts": [r.get("snippet", "")],
                }
                for r in data.get("organic", [])
            ]
        except Exception as e:
            print(f"      Serper error: {e}")
            return []

    async def _brave_search(self, query: str, max_results: int) -> list[dict]:
        """Brave search API."""
        try:
            resp = await self.client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={"X-Subscription-Token": BRAVE_API_KEY},
                params={"q": query, "count": max_results},
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "excerpts": [r.get("description", "")],
                }
                for r in data.get("web", {}).get("results", [])
            ]
        except Exception as e:
            print(f"      Brave error: {e}")
            return []


class PersonaGenerator:
    """Generate personas using Anthropic."""

    def __init__(self):
        if not ANTHROPIC_AVAILABLE:
            raise RuntimeError("anthropic package not installed")
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.searcher = WebSearcher()

    async def close(self):
        await self.searcher.close()

    async def deep_research(self, name: str, role: str, category: str) -> tuple[str, list[str], int]:
        """Run comprehensive research."""
        print(f"  [TERMS] Generating search terms...")

        terms_by_cat = generate_deep_terms(name, role, category)
        all_terms = flatten_terms(terms_by_cat)
        print(f"    {len(all_terms)} terms across {len(terms_by_cat)} categories")

        # Check if we have a search API
        has_search = (
            (SEARCH_API == "tavily" and TAVILY_API_KEY) or
            (SEARCH_API == "serper" and SERPER_API_KEY) or
            (SEARCH_API == "brave" and BRAVE_API_KEY)
        )

        if not has_search:
            print(f"    No search API configured, using terms only")
            return "", all_terms, 0

        print(f"  [SEARCH] Running searches via {SEARCH_API}...")
        all_results = []
        seen_urls = set()
        batch_size = 5

        for i in range(0, min(len(all_terms), 60), batch_size):  # Cap at 60 queries
            batch = all_terms[i:i + batch_size]
            tasks = [self.searcher.search(q, 8) for q in batch]
            batch_results = await asyncio.gather(*tasks)

            for results in batch_results:
                for r in results:
                    url = r.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append(r)

            done = min(i + batch_size, min(len(all_terms), 60))
            print(f"    {done}/60 queries, {len(all_results)} sources")
            await asyncio.sleep(0.2)

        # Format research
        research_text = ""
        for i, r in enumerate(all_results[:50]):
            title = r.get("title", "")
            url = r.get("url", "")
            excerpts = " ".join(r.get("excerpts", []))[:400]
            research_text += f"\n[{i+1}] {title}\n{url}\n{excerpts}\n"

        return research_text, all_terms, len(all_results)

    def generate_persona(self, name: str, role: str, category: str, research: str, search_terms: list[str]) -> dict:
        """Generate persona using Claude."""
        print(f"  [SYNTHESIZE] Generating with {SYNTHESIS_MODEL}...")

        # If no research, include the search terms so Claude knows what to look for
        terms_context = ""
        if not research:
            terms_context = f"""
NOTE: No web search results available. Use your knowledge to fill in what you know.
These are the search queries that WOULD have been run - use them as a guide for what to include:

{chr(10).join(f'- {t}' for t in search_terms[:40])}
"""

        prompt = f"""Create a COMPREHENSIVE persona dossier for {name}.

This is for multi-agent simulation. I need EVERYTHING - good, bad, and ugly.

CRITICAL SECTIONS:

1. LEGAL EXPOSURE
   - Criminal investigations, indictments, arrests
   - Lawsuits (against them and filed by them)
   - Regulatory issues (SEC, FTC, DOJ, congressional)
   - Settlements paid
   - Potential future exposure

2. CONTROVERSIES & SCANDALS
   - Major public scandals
   - Misconduct allegations (harassment, discrimination, abuse)
   - Cover-ups
   - Hypocrisies and lies caught
   - Public failures

3. RELATIONSHIPS (be specific - name names)
   - Inner circle (who they ACTUALLY trust)
   - Allies and enemies
   - Burned bridges and betrayals
   - Who they owe favors to
   - Who has leverage over them
   - Family vulnerabilities

4. PRESSURE POINTS (CRITICAL)
   - What could end their career
   - Reputation risks
   - Legal leverage points
   - Financial vulnerabilities
   - Psychological triggers
   - Insider threats
   - Skeletons that could resurface
   - What keeps them up at night

5. PERSONALITY (for simulation)
   - Dark traits (narcissism, manipulation, temper)
   - Insecurities and ego needs
   - Triggers
   - Under pressure behavior
   - How they attack/defend

6. COMMUNICATION
   - How they really talk vs PR voice
   - Verbal tics, catchphrases
   - 5+ actual quotes that capture their voice
   - Manipulation tactics

7. INTERNET PRESENCE
   - Twitter style and beefs
   - Deleted tweets / regrets
   - How they're memed
   - Reddit/Glassdoor reputation
   - Leaked content

RESEARCH DATA:
{research if research else '[No web search data - use your knowledge]'}
{terms_context}

CONTEXT:
- Name: {name}
- Role: {role}
- Category: {category}

Be specific. Name names. Include dates. If something is rumor, include it but note it.
Don't sanitize - I need the REAL person for accurate simulation.

Return as JSON with these sections:
- name, current_title, category, role_in_policy
- background (origin_story, education, career_arc, wealth_source, net_worth)
- personality (core_traits, dark_traits, quirks, triggers, insecurities, ego_needs)
- communication (public_persona, private_persona, verbal_tics, manipulation_tactics, sample_quotes[])
- relationships (inner_circle[], allies[], enemies[], burned_bridges[], owes_favors_to[], family_dynamics)
- legal_exposure (investigations[], indictments[], lawsuits[], settlements[], potential_exposure[])
- controversies (scandals[], misconduct[], coverups[], hypocrisies[], lies_caught[])
- financial (wealth_sources[], vulnerabilities[], conflicts_of_interest[])
- pressure_points (career_vulnerabilities[], reputation_risks[], legal_leverage[], psychological_triggers[], skeletons[], what_keeps_them_up[])
- internet_presence (twitter_handle, twitter_style, twitter_beefs[], meme_status, reddit_reputation, leaked_content[])
- worldview (core_beliefs[], ai_philosophy, china_stance, regulation_views, political_leanings, blindspots[])
- current_state (battles[], priorities[], wins[], losses[], momentum, stress_indicators[])
- simulation_guide (how_to_embody, signature_behaviors[], never_say[], hot_buttons[], how_to_flatter[], how_to_provoke[], negotiation_style)
- sources[]"""

        try:
            response = self.client.messages.create(
                model=SYNTHESIS_MODEL,
                max_tokens=8000,
                messages=[{"role": "user", "content": prompt}]
            )

            text = response.content[0].text

            # Extract JSON
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass

            return {"name": name, "raw_output": text[:8000], "error": "json_parse_failed"}

        except Exception as e:
            print(f"    Synthesis error: {e}")
            return {"name": name, "error": str(e)}

    async def process_person(self, person: dict) -> dict:
        """Full pipeline for one person."""
        name = person["name"]
        role = person["role"]
        category = person["category_name"]

        print(f"\n{'='*70}")
        print(f"[{name}]")
        print(f"  Role: {role}")

        # Research
        research, terms, source_count = await self.deep_research(name, role, category)

        # Generate
        persona = self.generate_persona(name, role, category, research, terms)

        # Metadata
        persona["_metadata"] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "category_id": person["category_id"],
            "category_name": category,
            "role": role,
            "search_terms_used": len(terms),
            "sources_found": source_count,
            "model": SYNTHESIS_MODEL,
            "search_api": SEARCH_API if source_count > 0 else "none",
        }

        # Save
        self._save(name, persona)
        return persona

    def _save(self, name: str, persona: dict):
        """Save persona to file."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r'[^a-z0-9_]', '_', name.lower())
        safe_name = re.sub(r'_+', '_', safe_name).strip('_')
        filepath = OUTPUT_DIR / f"{safe_name}.json"

        with open(filepath, "w") as f:
            json.dump(persona, f, indent=2, default=str, ensure_ascii=False)
        print(f"  [SAVED] {filepath.name}")


async def run_pipeline(people: list[dict], concurrency: int = CONCURRENCY):
    """Run pipeline."""
    print("=" * 70)
    print("FULL PERSONA GENERATION - Anthropic Direct")
    print("=" * 70)
    print(f"People: {len(people)}")
    print(f"Model: {SYNTHESIS_MODEL}")
    print(f"Search API: {SEARCH_API}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)

    generator = PersonaGenerator()
    semaphore = asyncio.Semaphore(concurrency)

    async def process(person):
        async with semaphore:
            try:
                return await generator.process_person(person)
            except Exception as e:
                print(f"[FAILED] {person['name']}: {e}")
                return {"name": person["name"], "error": str(e)}

    try:
        results = await asyncio.gather(*[process(p) for p in people])
    finally:
        await generator.close()

    successful = sum(1 for r in results if "error" not in r)
    print(f"\n{'='*70}")
    print(f"COMPLETE: {successful}/{len(people)} personas")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--category", type=str)
    parser.add_argument("-p", "--person", type=str)
    parser.add_argument("-n", "--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("-l", "--list", action="store_true")
    parser.add_argument("--model", type=str, default=SYNTHESIS_MODEL)
    args = parser.parse_args()

    global SYNTHESIS_MODEL
    SYNTHESIS_MODEL = args.model

    if args.person:
        all_people = get_all_people()
        people = [p for p in all_people if args.person.lower() in p["name"].lower()]
        if not people:
            print(f"No match: {args.person}")
            sys.exit(1)
    elif args.category:
        if args.category not in PEOPLE:
            print(f"Categories: {list(PEOPLE.keys())}")
            sys.exit(1)
        cat = PEOPLE[args.category]
        people = [{**p, "category_id": args.category, "category_name": cat["name"]}
                  for p in cat["people"]]
    else:
        people = get_all_people()

    if args.list:
        for cat_id, cat in PEOPLE.items():
            print(f"\n{cat['name']} [{cat_id}]:")
            for p in cat["people"]:
                print(f"  {p['name']:30} — {p['role']}")
        print(f"\nTotal: {len(get_all_people())}")
        return

    asyncio.run(run_pipeline(people, args.concurrency))


if __name__ == "__main__":
    main()
