#!/usr/bin/env python3
"""
FULL PERSONA GENERATION - Maximum Depth

Generates comprehensive personas with focus on:
- Criminal/legal exposure
- Controversies and scandals
- Relationship dynamics (allies, enemies, leverage)
- Pressure points and vulnerabilities
- Financial entanglements
- What can be used against them

For multi-agent simulation where you need to understand the FULL picture.
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
from deep_search_terms import generate_deep_terms, flatten_terms, DEEP_QUERY_TEMPLATES

PARALLEL_API_URL = os.getenv("PARALLEL_API_URL", "http://127.0.0.1:4002")
PARALLEL_API_KEY = os.getenv("PARALLEL_API_KEY", "")
OUTPUT_DIR = Path(__file__).parent / "personas"
CONCURRENCY = 2

# Use base for cost, core for quality
SYNTHESIS_PROCESSOR = os.getenv("SYNTHESIS_PROCESSOR", "core")


PRESSURE_POINT_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "current_title": {"type": "string"},
        "category": {"type": "string"},
        "role_in_policy": {"type": "string"},

        # === IDENTITY & BACKGROUND ===
        "background": {
            "type": "object",
            "properties": {
                "origin_story": {"type": "string"},
                "education": {"type": "array", "items": {"type": "string"}},
                "career_arc": {"type": "array", "items": {"type": "string"}},
                "wealth_source": {"type": "string"},
                "net_worth_estimate": {"type": "string"},
            }
        },

        # === PERSONALITY ===
        "personality": {
            "type": "object",
            "properties": {
                "core_traits": {"type": "array", "items": {"type": "string"}},
                "dark_traits": {"type": "array", "items": {"type": "string"}, "description": "Narcissism, manipulation, temper, etc"},
                "quirks": {"type": "array", "items": {"type": "string"}},
                "triggers": {"type": "array", "items": {"type": "string"}, "description": "What sets them off"},
                "insecurities": {"type": "array", "items": {"type": "string"}},
                "ego_needs": {"type": "array", "items": {"type": "string"}},
            }
        },

        # === COMMUNICATION ===
        "communication": {
            "type": "object",
            "properties": {
                "public_persona": {"type": "string"},
                "private_persona": {"type": "string"},
                "verbal_tics": {"type": "array", "items": {"type": "string"}},
                "manipulation_tactics": {"type": "array", "items": {"type": "string"}},
                "tells_when_lying": {"type": "array", "items": {"type": "string"}},
                "sample_quotes": {"type": "array", "items": {"type": "string"}},
            }
        },

        # === RELATIONSHIPS ===
        "relationships": {
            "type": "object",
            "properties": {
                "inner_circle": {"type": "array", "items": {"type": "string"}, "description": "Who they actually trust"},
                "allies": {"type": "array", "items": {"type": "string"}},
                "mentors": {"type": "array", "items": {"type": "string"}},
                "proteges": {"type": "array", "items": {"type": "string"}},
                "rivals": {"type": "array", "items": {"type": "string"}},
                "enemies": {"type": "array", "items": {"type": "string"}, "description": "People who actively want them to fail"},
                "frenemies": {"type": "array", "items": {"type": "string"}},
                "burned_bridges": {"type": "array", "items": {"type": "string"}, "description": "Relationships they destroyed"},
                "owes_favors_to": {"type": "array", "items": {"type": "string"}, "description": "Political/social debts"},
                "has_leverage_over": {"type": "array", "items": {"type": "string"}},
                "family_dynamics": {"type": "string"},
                "romantic_vulnerabilities": {"type": "string"},
            }
        },

        # === LEGAL EXPOSURE ===
        "legal_exposure": {
            "type": "object",
            "properties": {
                "criminal_investigations": {"type": "array", "items": {"type": "string"}},
                "indictments": {"type": "array", "items": {"type": "string"}},
                "lawsuits_against": {"type": "array", "items": {"type": "string"}},
                "lawsuits_filed": {"type": "array", "items": {"type": "string"}},
                "regulatory_issues": {"type": "array", "items": {"type": "string"}, "description": "SEC, FTC, DOJ, etc"},
                "settlements": {"type": "array", "items": {"type": "string"}},
                "potential_exposure": {"type": "array", "items": {"type": "string"}, "description": "Things that could become legal issues"},
                "statute_of_limitations_concerns": {"type": "array", "items": {"type": "string"}},
            }
        },

        # === CONTROVERSIES ===
        "controversies": {
            "type": "object",
            "properties": {
                "major_scandals": {"type": "array", "items": {"type": "string"}},
                "misconduct_allegations": {"type": "array", "items": {"type": "string"}},
                "public_failures": {"type": "array", "items": {"type": "string"}},
                "covered_up_incidents": {"type": "array", "items": {"type": "string"}},
                "hypocrisies": {"type": "array", "items": {"type": "string"}},
                "lies_caught": {"type": "array", "items": {"type": "string"}},
                "things_they_regret": {"type": "array", "items": {"type": "string"}},
            }
        },

        # === FINANCIAL ===
        "financial": {
            "type": "object",
            "properties": {
                "wealth_sources": {"type": "array", "items": {"type": "string"}},
                "financial_vulnerabilities": {"type": "array", "items": {"type": "string"}},
                "conflicts_of_interest": {"type": "array", "items": {"type": "string"}},
                "questionable_transactions": {"type": "array", "items": {"type": "string"}},
                "debts_obligations": {"type": "array", "items": {"type": "string"}},
                "golden_parachutes": {"type": "array", "items": {"type": "string"}},
            }
        },

        # === PRESSURE POINTS ===
        "pressure_points": {
            "type": "object",
            "properties": {
                "career_vulnerabilities": {"type": "array", "items": {"type": "string"}, "description": "What could end their career"},
                "reputation_risks": {"type": "array", "items": {"type": "string"}, "description": "What would damage their reputation"},
                "legal_leverage": {"type": "array", "items": {"type": "string"}, "description": "Legal exposure that could be used"},
                "financial_pressure": {"type": "array", "items": {"type": "string"}, "description": "Financial vulnerabilities"},
                "relationship_leverage": {"type": "array", "items": {"type": "string"}, "description": "Relationships that could be exploited"},
                "psychological_triggers": {"type": "array", "items": {"type": "string"}, "description": "Emotional buttons to push"},
                "public_opinion_risks": {"type": "array", "items": {"type": "string"}, "description": "What would turn public against them"},
                "insider_threats": {"type": "array", "items": {"type": "string"}, "description": "People inside who could damage them"},
                "skeletons": {"type": "array", "items": {"type": "string"}, "description": "Past issues that could resurface"},
                "what_keeps_them_up_at_night": {"type": "array", "items": {"type": "string"}},
            }
        },

        # === INTERNET PRESENCE ===
        "internet_presence": {
            "type": "object",
            "properties": {
                "twitter_handle": {"type": "string"},
                "twitter_style": {"type": "string"},
                "deleted_tweets": {"type": "array", "items": {"type": "string"}},
                "twitter_beefs": {"type": "array", "items": {"type": "string"}},
                "viral_moments": {"type": "array", "items": {"type": "string"}},
                "meme_status": {"type": "string"},
                "reddit_reputation": {"type": "string"},
                "glassdoor_reputation": {"type": "string"},
                "leaked_content": {"type": "array", "items": {"type": "string"}},
            }
        },

        # === WORLDVIEW ===
        "worldview": {
            "type": "object",
            "properties": {
                "core_beliefs": {"type": "array", "items": {"type": "string"}},
                "ai_philosophy": {"type": "string"},
                "china_stance": {"type": "string"},
                "regulation_views": {"type": "string"},
                "political_leanings": {"type": "string"},
                "sacred_cows": {"type": "array", "items": {"type": "string"}},
                "blindspots": {"type": "array", "items": {"type": "string"}},
            }
        },

        # === CURRENT STATE ===
        "current_state": {
            "type": "object",
            "properties": {
                "current_battles": {"type": "array", "items": {"type": "string"}},
                "current_priorities": {"type": "array", "items": {"type": "string"}},
                "recent_wins": {"type": "array", "items": {"type": "string"}},
                "recent_losses": {"type": "array", "items": {"type": "string"}},
                "momentum": {"type": "string", "description": "Rising, falling, stable"},
                "stress_indicators": {"type": "array", "items": {"type": "string"}},
                "next_likely_moves": {"type": "array", "items": {"type": "string"}},
            }
        },

        # === SIMULATION GUIDE ===
        "simulation_guide": {
            "type": "object",
            "properties": {
                "how_to_embody": {"type": "string"},
                "signature_behaviors": {"type": "array", "items": {"type": "string"}},
                "things_they_would_never_say": {"type": "array", "items": {"type": "string"}},
                "hot_buttons": {"type": "array", "items": {"type": "string"}},
                "how_to_flatter_them": {"type": "array", "items": {"type": "string"}},
                "how_to_provoke_them": {"type": "array", "items": {"type": "string"}},
                "negotiation_style": {"type": "string"},
                "under_pressure_behavior": {"type": "string"},
                "how_they_attack": {"type": "string"},
                "how_they_defend": {"type": "string"},
            }
        },

        "sources": {"type": "array", "items": {"type": "string"}},
        "confidence_notes": {"type": "string"},
    }
}


class FullPersonaGenerator:
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        self.client = httpx.AsyncClient(
            base_url=self.api_url,
            timeout=httpx.Timeout(600.0),
            headers=headers,
        )
        return self

    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()

    async def search(self, query: str, max_results: int = 10) -> list[dict]:
        """Run web search."""
        try:
            resp = await self.client.post(
                "/v1beta/search",
                json={"objective": query, "max_results": max_results},
            )
            resp.raise_for_status()
            return resp.json().get("results", [])
        except:
            return []

    async def deep_research(self, name: str, role: str, category: str) -> tuple[str, int]:
        """Run comprehensive research with many targeted queries."""
        print(f"  [RESEARCH] Generating search terms...")

        # Generate categorized search terms
        terms_by_category = generate_deep_terms(name, role, category)
        all_terms = flatten_terms(terms_by_category)

        print(f"    {len(all_terms)} search terms across {len(terms_by_category)} categories")

        # Run searches in batches
        print(f"  [SEARCH] Running searches...")
        all_results = []
        seen_urls = set()
        batch_size = 8

        for i in range(0, len(all_terms), batch_size):
            batch = all_terms[i:i + batch_size]
            tasks = [self.search(q, 8) for q in batch]
            batch_results = await asyncio.gather(*tasks)

            for results in batch_results:
                for r in results:
                    url = r.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append(r)

            # Progress
            done = min(i + batch_size, len(all_terms))
            print(f"    {done}/{len(all_terms)} queries, {len(all_results)} unique sources")

            if i + batch_size < len(all_terms):
                await asyncio.sleep(0.3)

        # Format research
        research_text = ""
        for i, r in enumerate(all_results[:60]):
            title = r.get("title", "")
            url = r.get("url", "")
            excerpts = " ".join(r.get("excerpts", []))[:500]
            research_text += f"\n[{i+1}] {title}\nURL: {url}\n{excerpts}\n"

        return research_text, len(all_results)

    async def generate_persona(self, person: dict) -> dict:
        """Generate full persona with pressure points."""
        name = person["name"]
        role = person["role"]
        category = person["category_name"]

        print(f"\n{'='*70}")
        print(f"[{name}]")
        print(f"  Role: {role} | Category: {category}")

        # Deep research
        research, source_count = await self.deep_research(name, role, category)

        # Synthesize
        print(f"  [SYNTHESIZE] Generating full persona with {SYNTHESIS_PROCESSOR}...")
        persona = await self._synthesize(name, role, category, research)

        # Metadata
        persona["_metadata"] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "category_id": person["category_id"],
            "category_name": category,
            "role": role,
            "sources_found": source_count,
            "processor": SYNTHESIS_PROCESSOR,
        }

        await self._save(name, persona)
        return persona

    async def _synthesize(self, name: str, role: str, category: str, research: str) -> dict:
        """Synthesize full persona from research."""

        prompt = f"""Create a COMPREHENSIVE persona dossier for {name}.

This is for multi-agent simulation. I need EVERYTHING - the good, bad, and ugly:

CRITICAL SECTIONS TO FILL:

1. LEGAL EXPOSURE
   - Any criminal investigations, indictments, arrests
   - Lawsuits (against them and filed by them)
   - Regulatory issues (SEC, FTC, DOJ, congressional)
   - Settlements and what they paid
   - Potential future legal exposure

2. CONTROVERSIES & SCANDALS
   - Major public scandals
   - Misconduct allegations (harassment, discrimination, abuse of power)
   - Things they've covered up
   - Hypocrisies and lies caught
   - Public failures and embarrassments

3. RELATIONSHIPS
   - Inner circle (who they ACTUALLY trust)
   - Allies and enemies (be specific about who and why)
   - Burned bridges and betrayals
   - Who they owe favors to (political/social debts)
   - Who they have leverage over
   - Family and romantic vulnerabilities

4. PRESSURE POINTS (CRITICAL)
   - Career vulnerabilities (what could end them)
   - Reputation risks
   - Legal leverage points
   - Financial pressure points
   - Psychological triggers
   - Insider threats
   - Skeletons that could resurface
   - What keeps them up at night

5. FINANCIAL
   - Conflicts of interest
   - Questionable transactions
   - Debts and obligations

6. PERSONALITY (for simulation)
   - Dark traits (narcissism, manipulation, temper)
   - Insecurities and ego needs
   - What triggers them
   - How they behave under pressure
   - How they attack and defend

7. INTERNET
   - Twitter beefs and deleted tweets
   - Leaked content
   - Meme status and viral moments
   - Reddit/Glassdoor reputation

RESEARCH DATA:
{research[:35000]}

CONTEXT:
- Name: {name}
- Role: {role}
- Category: {category}

Be specific. Name names. Include dates where known. If something is rumor, include it but note it. Don't sanitize - this is for understanding the full person.

Return as JSON matching the schema."""

        try:
            resp = await self.client.post(
                "/v1/tasks/runs/managed",
                json={
                    "processor": SYNTHESIS_PROCESSOR,
                    "input": prompt,
                    "task_spec": {
                        "output_schema": {
                            "type": "json",
                            "json_schema": PRESSURE_POINT_SCHEMA
                        }
                    }
                },
                params={"poll_interval_seconds": 3, "timeout_seconds": 900},
            )
            resp.raise_for_status()
            result = resp.json()

            output = result.get("output_json") or result.get("output_text")

            if isinstance(output, dict):
                return output
            elif isinstance(output, str):
                match = re.search(r'\{[\s\S]*\}', output)
                if match:
                    try:
                        return json.loads(match.group())
                    except json.JSONDecodeError:
                        pass
                return {"name": name, "raw": output[:8000], "error": "json_parse_failed"}
            return {"name": name, "error": "no_output"}

        except Exception as e:
            print(f"    Synthesis error: {e}")
            return {"name": name, "error": str(e)}

    async def _save(self, name: str, persona: dict):
        """Save persona."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r'[^a-z0-9_]', '_', name.lower())
        safe_name = re.sub(r'_+', '_', safe_name).strip('_')
        filepath = OUTPUT_DIR / f"{safe_name}.json"

        with open(filepath, "w") as f:
            json.dump(persona, f, indent=2, default=str, ensure_ascii=False)
        print(f"  [SAVED] {filepath.name}")


async def run_pipeline(people: list[dict], concurrency: int = CONCURRENCY):
    """Run full pipeline."""
    print("=" * 70)
    print("FULL PERSONA GENERATION - Maximum Depth")
    print("=" * 70)
    print(f"People: {len(people)}")
    print(f"Concurrency: {concurrency}")
    print(f"Processor: {SYNTHESIS_PROCESSOR}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)

    semaphore = asyncio.Semaphore(concurrency)

    async with FullPersonaGenerator(PARALLEL_API_URL, PARALLEL_API_KEY) as gen:
        async def process(person):
            async with semaphore:
                try:
                    return await gen.generate_persona(person)
                except Exception as e:
                    print(f"[FAILED] {person['name']}: {e}")
                    return {"name": person["name"], "error": str(e)}

        results = await asyncio.gather(*[process(p) for p in people])

    successful = sum(1 for r in results if "error" not in r)
    print(f"\n{'='*70}")
    print(f"COMPLETE: {successful}/{len(people)} personas")
    return results


def main():
    global SYNTHESIS_PROCESSOR

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--category", type=str)
    parser.add_argument("-p", "--person", type=str)
    parser.add_argument("-n", "--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("-l", "--list", action="store_true")
    parser.add_argument("--processor", type=str, default=SYNTHESIS_PROCESSOR)
    args = parser.parse_args()

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
            print(f"\n{cat['name']} [{cat_id}]:")
            for p in cat["people"]:
                print(f"  {p['name']:30} — {p['role']}")
        print(f"\nTotal: {len(get_all_people())}")
        return

    asyncio.run(run_pipeline(people, args.concurrency))


if __name__ == "__main__":
    main()
