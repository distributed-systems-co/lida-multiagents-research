#!/usr/bin/env python3
"""
DEEP PERSONA GENERATION v2 - Cost-Optimized with DSPy

Uses DSPy to generate comprehensive search terms, then cheaper processors:
- Search term generation: Claude Haiku (cheap)
- Web searches: Parallel.ai lite processor
- Persona synthesis: Parallel.ai base/core processor

Generates 50-80 targeted search queries per person for comprehensive coverage.
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

from people import get_all_people, PEOPLE

# Try to import DSPy components
try:
    from search_term_generator import (
        configure_dspy,
        SearchTermGenerator,
        generate_fallback_terms,
    )
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    print("Warning: search_term_generator not available, using fallback templates")

PARALLEL_API_URL = os.getenv("PARALLEL_API_URL", "http://127.0.0.1:4002")
PARALLEL_API_KEY = os.getenv("PARALLEL_API_KEY", "")
OUTPUT_DIR = Path(__file__).parent / "personas"
CONCURRENCY = 3

# Cost-optimized processor choices
SEARCH_PROCESSOR = "lite"      # Cheapest for search
SYNTHESIS_PROCESSOR = "core"   # Good balance for final synthesis


PERSONA_TEMPLATE = """
# PERSONA: {name}

## Quick Reference
- **Current Title**:
- **Category**: {category}
- **Role in AI/Chip Policy**: {role}

## The Real Person

### Personality
- **Core Traits**:
- **Quirks**:
- **Pet Peeves**:
- **Insecurities**:
- **Ego Triggers**:
- **Under Pressure**:

### Communication Style
- **How They Talk**:
- **Verbal Tics/Catchphrases**:
- **Real Voice vs PR Voice**:
- **Sample Quotes**:
  -
  -
  -

### Relationships
- **Inner Circle**:
- **Close Friends**:
- **Mentors**:
- **Proteges**:
- **Rivals**:
- **Enemies**:
- **Notable Fallings Out**:

### Power Dynamics
- **Real Power Sources**:
- **Weaknesses**:
- **Who They Defer To**:
- **Reputation Among Peers**:

### Controversies & Drama
- **Public Scandals**:
- **Open Secrets**:
- **Rumors**:
- **Hypocrisies**:

### Internet Presence
- **Twitter Style**:
- **Notable Twitter Beefs**:
- **How They're Memed**:
- **Reddit/HN Reputation**:

### Worldview
- **Core Beliefs**:
- **AI Philosophy**:
- **China Views**:
- **Regulation Stance**:

### Background
- **Origin Story**:
- **Education**:
- **Career Arc**:
- **Lucky Breaks**:
- **Failures**:

### Current State (2024-2025)
- **Current Priorities**:
- **Current Battles**:
- **Momentum**:

## Simulation Guide
- **How to Embody**:
- **Things They'd Never Say**:
- **Hot Buttons**:
- **Negotiation Style**:

## Sources
-
"""


class DeepPersonaGeneratorV2:
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.client: Optional[httpx.AsyncClient] = None
        self.term_generator: Optional[SearchTermGenerator] = None

        # Try to initialize DSPy
        if DSPY_AVAILABLE:
            try:
                configure_dspy()
                self.term_generator = SearchTermGenerator()
                print("✓ DSPy initialized with Claude Haiku")
            except Exception as e:
                print(f"✗ DSPy init failed: {e}")
                self.term_generator = None

    async def __aenter__(self):
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        self.client = httpx.AsyncClient(
            base_url=self.api_url,
            timeout=httpx.Timeout(300.0),
            headers=headers,
        )
        return self

    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()

    def generate_search_terms(self, name: str, role: str, category: str) -> list[str]:
        """Generate comprehensive search terms using DSPy or fallback."""
        if self.term_generator:
            try:
                print(f"  [DSPy] Generating search terms...")
                terms = self.term_generator.generate_initial_terms(name, role, category)
                print(f"    Generated {len(terms)} terms via DSPy")
                return terms
            except Exception as e:
                print(f"    DSPy failed ({e}), using fallback")

        # Fallback to templates
        if DSPY_AVAILABLE:
            terms = generate_fallback_terms(name, role)
        else:
            terms = self._fallback_terms(name, role)

        print(f"    Generated {len(terms)} terms via templates")
        return terms

    def _fallback_terms(self, name: str, role: str) -> list[str]:
        """Inline fallback if search_term_generator not available."""
        templates = [
            # Biography
            f"{name} biography background",
            f"{name} education career history",
            f"{name} early life origin story",

            # Current
            f"{name} 2024 2025 recent news",
            f"{name} current role latest",

            # Personality
            f"{name} personality traits",
            f"{name} management style leadership",
            f"{name} interview personality quirks",
            f'"{name}" what is like',

            # Communication
            f"{name} quotes statements",
            f"{name} interview podcast transcript",
            f'"{name}" said',

            # Relationships
            f"{name} friends allies relationships",
            f"{name} mentor inner circle",
            f"{name} family married",

            # Conflicts
            f"{name} feud beef rivalry",
            f"{name} vs competitor",
            f"{name} enemies opponents conflict",
            f"{name} falling out former ally",

            # Controversies
            f"{name} controversy scandal",
            f"{name} criticism backlash",
            f"{name} accused allegations",

            # Gossip
            f"{name} rumors gossip",
            f'"{name}" actually secretly',

            # Twitter
            f"{name} twitter drama",
            f'"{name}" tweet ratio',
            f"{name} social media fight",

            # Reddit/HN
            f'"{name}" reddit',
            f'"{name}" hacker news',
            f"{name} overrated underrated",

            # Memes
            f"{name} meme viral",
            f"{name} jokes parody",

            # Policy
            f"{name} AI views position",
            f"{name} China policy",
            f"{name} regulation stance",
            f"{name} {role}",

            # Personal
            f"{name} personal life hobbies",
            f"{name} net worth wealth",

            # Psychology
            f"{name} motivations goals",
            f"{name} insecure weakness",
            f"{name} ego arrogant humble",
        ]
        return templates

    async def search(self, query: str, max_results: int = 10) -> list[dict]:
        """Run a web search using lite processor."""
        try:
            resp = await self.client.post(
                "/v1beta/search",
                json={"objective": query, "max_results": max_results},
            )
            resp.raise_for_status()
            return resp.json().get("results", [])
        except Exception as e:
            return []

    async def batch_search(self, queries: list[str], max_results_per: int = 8) -> list[dict]:
        """Run many searches in parallel batches."""
        all_results = []
        seen_urls = set()

        # Process in batches to avoid overwhelming the API
        batch_size = 10

        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            tasks = [self.search(q, max_results_per) for q in batch]
            batch_results = await asyncio.gather(*tasks)

            for results in batch_results:
                for r in results:
                    url = r.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append(r)

            # Small delay between batches
            if i + batch_size < len(queries):
                await asyncio.sleep(0.5)

        return all_results

    async def generate_persona(self, person: dict) -> dict:
        """Generate deep persona for one person."""
        name = person["name"]
        role = person["role"]
        category = person["category_name"]

        print(f"\n{'='*60}")
        print(f"[{name}]")
        print(f"  Role: {role}")

        # Step 1: Generate search terms (DSPy or fallback)
        search_terms = self.generate_search_terms(name, role, category)

        # Step 2: Run all searches
        print(f"  [SEARCH] Running {len(search_terms)} searches...")
        results = await self.batch_search(search_terms)
        print(f"    Found {len(results)} unique sources")

        # Step 3: Format research data
        research_text = self._format_research(results)

        # Step 4: Generate persona using core processor
        print(f"  [SYNTHESIZE] Generating persona with {SYNTHESIS_PROCESSOR}...")
        persona = await self._synthesize_persona(name, role, category, research_text)

        # Add metadata
        persona["_metadata"] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "category_id": person["category_id"],
            "category_name": category,
            "role": role,
            "search_queries": len(search_terms),
            "sources_found": len(results),
            "processor": SYNTHESIS_PROCESSOR,
        }

        # Save
        await self._save_persona(name, persona)
        return persona

    def _format_research(self, results: list[dict]) -> str:
        """Format search results for the synthesis prompt."""
        text = ""
        for i, r in enumerate(results[:50]):  # Top 50 results
            title = r.get("title", "")
            url = r.get("url", "")
            excerpts = " ".join(r.get("excerpts", []))[:600]
            text += f"\n[{i+1}] {title}\n{url}\n{excerpts}\n"
        return text

    async def _synthesize_persona(self, name: str, role: str, category: str, research: str) -> dict:
        """Synthesize persona from research using Parallel.ai task."""

        prompt = f"""Create a DEEP, UNFILTERED persona profile for {name}.

I need the REAL person for multi-agent simulation. Not Wikipedia. The stuff that makes them human:

MUST INCLUDE:
1. PERSONALITY - quirks, pet peeves, insecurities, ego triggers, how they handle stress
2. HOW THEY TALK - verbal tics, catchphrases, real voice vs PR voice, 3-5 actual quotes
3. RELATIONSHIPS - inner circle, real friends, mentors, rivals, ENEMIES, feuds, fallings out
4. POWER DYNAMICS - where their real power comes from, weaknesses, who they defer to
5. GOSSIP & DRAMA - scandals, open secrets, rumors, hypocrisies
6. INTERNET PRESENCE - Twitter style, beefs, how they're memed, Reddit/HN reputation
7. WORLDVIEW - AI philosophy, China views, what they really believe
8. PSYCHOLOGY - motivations, insecurities, blind spots, what drives them
9. CURRENT STATE - what they're fighting for now, momentum, stress level
10. SIMULATION GUIDE - how an AI should embody them, things they'd never say, hot buttons

RESEARCH DATA:
{research[:30000]}

CONTEXT:
- Name: {name}
- Role: {role}
- Category: {category}

Be specific. Include real examples and quotes. If something is rumor/gossip, still include it but note it. Don't sanitize - I need the real person.

Return as JSON with these top-level keys:
- name, current_title, category, role_in_policy
- personality (object with: core_traits, quirks, pet_peeves, insecurities, ego_triggers, under_pressure)
- communication (object with: speaking_style, verbal_tics, real_vs_pr_voice, sample_quotes array)
- relationships (object with: inner_circle, friends, mentors, proteges, rivals, enemies, fallings_out)
- power_dynamics (object with: power_sources, weaknesses, defers_to, reputation)
- controversies (object with: scandals, open_secrets, rumors, hypocrisies)
- internet_presence (object with: twitter_style, twitter_beefs, meme_status, reddit_reputation)
- worldview (object with: core_beliefs, ai_philosophy, china_views, regulation_stance)
- background (object with: origin_story, education, career_arc, lucky_breaks, failures)
- current_state (object with: priorities, battles, momentum, stress_level)
- simulation_guide (object with: how_to_embody, never_say, hot_buttons, negotiation_style)
- sources (array of key URLs)"""

        try:
            resp = await self.client.post(
                "/v1/tasks/runs/managed",
                json={
                    "processor": SYNTHESIS_PROCESSOR,
                    "input": prompt,
                },
                params={"poll_interval_seconds": 2, "timeout_seconds": 600},
            )
            resp.raise_for_status()
            result = resp.json()

            output = result.get("output_json") or result.get("output_text")

            if isinstance(output, dict):
                return output
            elif isinstance(output, str):
                # Try to extract JSON
                match = re.search(r'\{[\s\S]*\}', output)
                if match:
                    try:
                        return json.loads(match.group())
                    except json.JSONDecodeError:
                        pass
                return {"name": name, "raw_output": output[:5000], "error": "json_parse_failed"}
            else:
                return {"name": name, "error": "no_output"}

        except Exception as e:
            print(f"    Synthesis error: {e}")
            return {"name": name, "error": str(e)}

    async def _save_persona(self, name: str, persona: dict):
        """Save persona to file."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        safe_name = re.sub(r'[^a-z0-9_]', '_', name.lower())
        safe_name = re.sub(r'_+', '_', safe_name).strip('_')
        filepath = OUTPUT_DIR / f"{safe_name}.json"

        with open(filepath, "w") as f:
            json.dump(persona, f, indent=2, default=str, ensure_ascii=False)

        print(f"  [SAVED] {filepath.name}")


async def run_pipeline(people: list[dict], concurrency: int = CONCURRENCY):
    """Run the pipeline."""
    print("=" * 60)
    print("DEEP PERSONA GENERATION v2 (Cost-Optimized)")
    print("=" * 60)
    print(f"People: {len(people)}")
    print(f"Concurrency: {concurrency}")
    print(f"Search processor: {SEARCH_PROCESSOR}")
    print(f"Synthesis processor: {SYNTHESIS_PROCESSOR}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    semaphore = asyncio.Semaphore(concurrency)

    async with DeepPersonaGeneratorV2(PARALLEL_API_URL, PARALLEL_API_KEY) as gen:
        async def process(person):
            async with semaphore:
                try:
                    return await gen.generate_persona(person)
                except Exception as e:
                    print(f"[FAILED] {person['name']}: {e}")
                    return {"name": person["name"], "error": str(e)}

        results = await asyncio.gather(*[process(p) for p in people])

    successful = sum(1 for r in results if "error" not in r)
    print(f"\n{'='*60}")
    print(f"DONE: {successful}/{len(people)} personas generated")
    print(f"Output: {OUTPUT_DIR}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Deep persona generation v2")
    parser.add_argument("-c", "--category", type=str, help="Process one category")
    parser.add_argument("-p", "--person", type=str, help="Process one person")
    parser.add_argument("-n", "--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("-l", "--list", action="store_true", help="List people")
    parser.add_argument("--processor", type=str, default=SYNTHESIS_PROCESSOR,
                        help="Synthesis processor (lite/base/core)")
    args = parser.parse_args()

    global SYNTHESIS_PROCESSOR
    SYNTHESIS_PROCESSOR = args.processor

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
            print(f"\n{cat['name']} ({cat_id}):")
            for p in cat["people"]:
                print(f"  • {p['name']:25} — {p['role']}")
        print(f"\nTotal: {len(get_all_people())}")
        return

    asyncio.run(run_pipeline(people, args.concurrency))


if __name__ == "__main__":
    main()
