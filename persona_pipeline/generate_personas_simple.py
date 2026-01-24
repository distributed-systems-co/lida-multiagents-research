#!/usr/bin/env python3
"""
Simple Persona Generation Pipeline - Direct API Version

Uses basic Parallel.ai search + task APIs without orchestration layer.
Good fallback if orchestration endpoints aren't available.
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

from people import get_all_people, PEOPLE

PARALLEL_API_URL = os.getenv("PARALLEL_API_URL", "http://127.0.0.1:4002")
PARALLEL_API_KEY = os.getenv("PARALLEL_API_KEY", "")
OUTPUT_DIR = Path(__file__).parent / "personas"
CONCURRENCY = 5


class SimplePersonaGenerator:
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        self.client = httpx.AsyncClient(
            base_url=self.api_url,
            timeout=httpx.Timeout(300.0),
            headers=headers,
        )
        return self

    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()

    async def search(self, objective: str, max_results: int = 15) -> list[dict]:
        """Run a web search."""
        try:
            response = await self.client.post(
                "/v1beta/search",
                json={"objective": objective, "max_results": max_results},
            )
            response.raise_for_status()
            return response.json().get("results", [])
        except Exception as e:
            print(f"    Search error: {e}")
            return []

    async def run_task(self, prompt: str, processor: str = "base") -> dict:
        """Run a task and wait for result."""
        try:
            response = await self.client.post(
                "/v1/tasks/runs/managed",
                json={
                    "processor": processor,
                    "input": prompt,
                },
                params={"poll_interval_seconds": 2, "timeout_seconds": 300},
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"    Task error: {e}")
            return {"error": str(e)}

    async def generate_persona(self, person: dict) -> dict:
        """Generate persona for one person."""
        name = person["name"]
        role = person["role"]
        category = person["category_name"]

        print(f"\n[{name}]")

        # Multi-query search
        search_queries = [
            f"{name} AI policy positions 2024 2025",
            f"{name} chip semiconductor export controls",
            f"{name} recent statements speeches interviews",
            f"{name} biography background career",
        ]

        print("  Searching...")
        search_tasks = [self.search(q, max_results=10) for q in search_queries]
        search_results = await asyncio.gather(*search_tasks)

        # Flatten and dedupe results
        all_results = []
        seen_urls = set()
        for results in search_results:
            for r in results:
                url = r.get("url", "")
                if url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(r)

        # Format search results for prompt
        search_context = "\n".join([
            f"- {r.get('title', 'No title')}: {r.get('url', '')}\n  {' '.join(r.get('excerpts', []))[:500]}"
            for r in all_results[:20]
        ])

        print(f"  Found {len(all_results)} sources, generating persona...")

        # Generate persona
        prompt = f"""Create a comprehensive persona profile for {name} based on this research.

SEARCH RESULTS:
{search_context}

CONTEXT:
- Name: {name}
- Known Role: {role}
- Category: {category}

Generate a detailed JSON persona with these fields:
{{
  "name": "{name}",
  "current_title": "their current primary title and organization",
  "category": "{category}",
  "role_in_ai_policy": "specific influence on AI/chip policy",

  "background": {{
    "education": ["degree, institution, year"],
    "career_history": ["role at org (years)"],
    "age": "approximate or exact",
    "nationality": "country"
  }},

  "current_positions": ["all current roles"],

  "policy_stance": {{
    "on_china_decoupling": "their position",
    "on_export_controls": "their position",
    "on_ai_safety_regulation": "their position",
    "on_open_source_ai": "their position",
    "on_compute_governance": "their position"
  }},

  "key_actions": ["notable decisions, statements, actions 2023-2025"],

  "influence_network": {{
    "allies": ["key allies"],
    "adversaries": ["opponents or rivals"],
    "organizations": ["affiliated orgs"]
  }},

  "communication_style": {{
    "public_persona": "how they present themselves",
    "rhetorical_patterns": ["common phrases, arguments"],
    "media_preferences": ["where they communicate"]
  }},

  "motivations": {{
    "stated_goals": ["what they say they want"],
    "inferred_motivations": ["underlying drivers"],
    "pressure_points": ["vulnerabilities, concerns"]
  }},

  "predictive_indicators": {{
    "likely_future_actions": ["what they might do"],
    "red_lines": ["things they won't accept"],
    "wildcards": ["unexpected things they might do"]
  }},

  "simulation_notes": "key traits for accurately simulating this person",
  "sources": ["key urls used"]
}}

Return ONLY valid JSON, no other text."""

        result = await self.run_task(prompt, processor="core")

        # Parse result
        output = result.get("output_text") or result.get("output_json")
        if isinstance(output, str):
            # Try to extract JSON from response
            try:
                # Find JSON in response
                start = output.find("{")
                end = output.rfind("}") + 1
                if start >= 0 and end > start:
                    persona = json.loads(output[start:end])
                else:
                    persona = {"name": name, "raw_output": output, "parse_error": "no json found"}
            except json.JSONDecodeError as e:
                persona = {"name": name, "raw_output": output, "parse_error": str(e)}
        elif isinstance(output, dict):
            persona = output
        else:
            persona = {"name": name, "error": "no output", "raw": result}

        # Add metadata
        persona["_metadata"] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sources_found": len(all_results),
            "category_id": person["category_id"],
        }

        # Save
        await self.save_persona(name, persona)
        return persona

    async def save_persona(self, name: str, persona: dict):
        """Save persona to file."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = name.lower().replace(" ", "_").replace(".", "").replace("-", "_")
        filepath = OUTPUT_DIR / f"{safe_name}.json"

        with open(filepath, "w") as f:
            json.dump(persona, f, indent=2, default=str)
        print(f"  Saved: {filepath.name}")


async def run_pipeline(people: list[dict], concurrency: int = CONCURRENCY):
    """Run pipeline for all people."""
    print(f"Generating personas for {len(people)} people")
    print(f"Concurrency: {concurrency}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async with SimplePersonaGenerator(PARALLEL_API_URL, PARALLEL_API_KEY) as gen:
        async def process(person):
            async with semaphore:
                try:
                    return await gen.generate_persona(person)
                except Exception as e:
                    print(f"[FAILED] {person['name']}: {e}")
                    return {"name": person["name"], "error": str(e)}

        results = await asyncio.gather(*[process(p) for p in people])

    successful = sum(1 for r in results if "error" not in r and "parse_error" not in r)
    print(f"\n{'=' * 60}")
    print(f"Done: {successful}/{len(people)} successful")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str)
    parser.add_argument("--person", type=str)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.person:
        all_people = get_all_people()
        people = [p for p in all_people if args.person.lower() in p["name"].lower()]
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
        for p in people:
            print(f"{p['name']:30} | {p['role']}")
        print(f"\nTotal: {len(people)}")
        return

    asyncio.run(run_pipeline(people, args.concurrency))


if __name__ == "__main__":
    main()
