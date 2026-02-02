#!/usr/bin/env python3
"""
Persona Generation Pipeline using Parallel.ai APIs

Generates detailed, up-to-date personas for influential people in chip/GPU/AI policy.
Uses iterative search with processor escalation for comprehensive research.
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

# Configuration
PARALLEL_API_URL = os.getenv("PARALLEL_API_URL", "http://127.0.0.1:4002")
PARALLEL_API_KEY = os.getenv("PARALLEL_API_KEY", "")
OUTPUT_DIR = Path(__file__).parent / "personas"
CONCURRENCY = 3  # How many personas to generate in parallel


PERSONA_SCHEMA = {
    "type": "json",
    "json_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Full name"},
            "current_title": {"type": "string", "description": "Current job title and organization"},
            "category": {"type": "string", "description": "Category in the influence map"},
            "role_in_ai_policy": {"type": "string", "description": "Their specific role/influence in AI/chip policy"},

            "background": {
                "type": "object",
                "properties": {
                    "education": {"type": "array", "items": {"type": "string"}},
                    "career_history": {"type": "array", "items": {"type": "string"}},
                    "age": {"type": "string"},
                    "nationality": {"type": "string"},
                },
            },

            "current_positions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "All current roles and positions"
            },

            "policy_stance": {
                "type": "object",
                "properties": {
                    "on_china_decoupling": {"type": "string"},
                    "on_export_controls": {"type": "string"},
                    "on_ai_safety_regulation": {"type": "string"},
                    "on_open_source_ai": {"type": "string"},
                    "on_compute_governance": {"type": "string"},
                },
            },

            "key_actions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Notable actions, decisions, statements in 2023-2025"
            },

            "influence_network": {
                "type": "object",
                "properties": {
                    "allies": {"type": "array", "items": {"type": "string"}},
                    "adversaries": {"type": "array", "items": {"type": "string"}},
                    "organizations": {"type": "array", "items": {"type": "string"}},
                },
            },

            "communication_style": {
                "type": "object",
                "properties": {
                    "public_persona": {"type": "string"},
                    "rhetorical_patterns": {"type": "array", "items": {"type": "string"}},
                    "media_preferences": {"type": "array", "items": {"type": "string"}},
                },
            },

            "motivations": {
                "type": "object",
                "properties": {
                    "stated_goals": {"type": "array", "items": {"type": "string"}},
                    "inferred_motivations": {"type": "array", "items": {"type": "string"}},
                    "pressure_points": {"type": "array", "items": {"type": "string"}},
                },
            },

            "predictive_indicators": {
                "type": "object",
                "properties": {
                    "likely_future_actions": {"type": "array", "items": {"type": "string"}},
                    "red_lines": {"type": "array", "items": {"type": "string"}},
                    "wildcards": {"type": "array", "items": {"type": "string"}},
                },
            },

            "simulation_notes": {
                "type": "string",
                "description": "Notes for accurately simulating this person in multi-agent scenarios"
            },

            "last_updated": {"type": "string"},
            "sources": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["name", "current_title", "background", "policy_stance", "key_actions"]
    }
}


class PersonaGenerator:
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(
            base_url=self.api_url,
            timeout=httpx.Timeout(300.0),  # 5 min timeout for research tasks
            headers={"x-api-key": self.api_key} if self.api_key else {},
        )
        return self

    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()

    async def research_person(self, name: str, role: str, category: str) -> dict:
        """Use orchestrated search to research a person."""
        print(f"  [RESEARCH] Searching for {name}...")

        try:
            response = await self.client.post(
                "/v1/orchestration/search",
                json={
                    "objective": f"""Research {name} comprehensively for a detailed persona profile.
Focus on:
- Current role and recent activities (2024-2025)
- Their position on AI policy, chip export controls, and compute governance
- Key decisions, statements, and actions
- Their influence network and relationships
- Communication style and public persona
- Background and career history

Context: {name} is known for: {role}
Category: {category}""",
                    "max_iterations": 3,
                    "escalation_strategy": "adaptive",
                    "starting_processor": "base",
                    "final_processor": "core",
                    "enable_synthesis": True,
                },
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            print(f"  [ERROR] Research failed for {name}: {e.response.status_code}")
            return {"error": str(e), "synthesis": None}
        except Exception as e:
            print(f"  [ERROR] Research failed for {name}: {e}")
            return {"error": str(e), "synthesis": None}

    async def generate_persona(self, name: str, role: str, category: str, research: dict) -> dict:
        """Generate structured persona from research."""
        print(f"  [GENERATE] Creating persona for {name}...")

        synthesis = research.get("synthesis") or research.get("final_results", [])
        if isinstance(synthesis, list):
            synthesis = "\n".join(str(r) for r in synthesis[:10])

        try:
            response = await self.client.post(
                "/v1/tasks/runs/managed",
                json={
                    "processor": "core",
                    "input": {
                        "query": f"""Based on this research, create a comprehensive persona profile for {name}.

RESEARCH DATA:
{synthesis[:15000] if synthesis else 'No research data available - use your knowledge.'}

CONTEXT:
- Name: {name}
- Known Role: {role}
- Category: {category}

Generate a detailed, accurate persona that could be used to simulate this person in multi-agent policy scenarios. Be specific about their stances, relationships, and behavioral patterns. Include recent actions and statements from 2024-2025 where available."""
                    },
                    "task_spec": {
                        "output_schema": PERSONA_SCHEMA
                    },
                },
                params={"poll_interval_seconds": 3, "timeout_seconds": 600},
            )
            response.raise_for_status()
            result = response.json()

            if result.get("output_json"):
                persona = result["output_json"]
                persona["_metadata"] = {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "research_quality": research.get("quality_metrics", {}),
                    "api_calls": research.get("total_api_calls", 0),
                }
                return persona
            elif result.get("output_text"):
                # Try to parse as JSON
                try:
                    return json.loads(result["output_text"])
                except:
                    return {"name": name, "raw_output": result["output_text"], "error": "non-json response"}
            else:
                return {"name": name, "error": "no output", "raw": result}

        except httpx.HTTPStatusError as e:
            print(f"  [ERROR] Persona generation failed for {name}: {e.response.status_code}")
            return {"name": name, "error": str(e)}
        except Exception as e:
            print(f"  [ERROR] Persona generation failed for {name}: {e}")
            return {"name": name, "error": str(e)}

    async def process_person(self, person: dict) -> dict:
        """Full pipeline for one person."""
        name = person["name"]
        role = person["role"]
        category = person["category_name"]

        print(f"\n[PROCESSING] {name}")

        # Step 1: Research
        research = await self.research_person(name, role, category)

        # Step 2: Generate persona
        persona = await self.generate_persona(name, role, category, research)

        # Step 3: Save
        await self.save_persona(person, persona)

        return persona

    async def save_persona(self, person: dict, persona: dict):
        """Save persona to file."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Create filename from name
        safe_name = person["name"].lower().replace(" ", "_").replace(".", "")
        filepath = OUTPUT_DIR / f"{safe_name}.json"

        with open(filepath, "w") as f:
            json.dump(persona, f, indent=2, default=str)

        print(f"  [SAVED] {filepath}")


async def run_pipeline(
    people: list[dict],
    concurrency: int = CONCURRENCY,
    api_url: str = PARALLEL_API_URL,
    api_key: str = PARALLEL_API_KEY,
):
    """Run the full pipeline for all people."""
    print(f"Starting persona generation for {len(people)} people")
    print(f"Concurrency: {concurrency}")
    print(f"API URL: {api_url}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async with PersonaGenerator(api_url, api_key) as generator:
        async def process_with_semaphore(person):
            async with semaphore:
                try:
                    return await generator.process_person(person)
                except Exception as e:
                    print(f"[FAILED] {person['name']}: {e}")
                    return {"name": person["name"], "error": str(e)}

        tasks = [process_with_semaphore(p) for p in people]
        results = await asyncio.gather(*tasks)

    # Summary
    successful = sum(1 for r in results if "error" not in r)
    print("\n" + "=" * 60)
    print(f"COMPLETE: {successful}/{len(people)} personas generated successfully")
    print(f"Output directory: {OUTPUT_DIR}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate personas using Parallel.ai")
    parser.add_argument("--category", type=str, help="Only process one category")
    parser.add_argument("--person", type=str, help="Only process one person by name")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--api-url", type=str, default=PARALLEL_API_URL)
    parser.add_argument("--dry-run", action="store_true", help="List people without processing")
    args = parser.parse_args()

    # Get people to process
    if args.person:
        all_people = get_all_people()
        people = [p for p in all_people if args.person.lower() in p["name"].lower()]
        if not people:
            print(f"No person found matching: {args.person}")
            sys.exit(1)
    elif args.category:
        if args.category not in PEOPLE:
            print(f"Unknown category: {args.category}")
            print(f"Available: {list(PEOPLE.keys())}")
            sys.exit(1)
        people = []
        for p in PEOPLE[args.category]["people"]:
            people.append({
                **p,
                "category_id": args.category,
                "category_name": PEOPLE[args.category]["name"],
            })
    else:
        people = get_all_people()

    if args.dry_run:
        print(f"Would process {len(people)} people:")
        for p in people:
            print(f"  - {p['name']} ({p['role']})")
        return

    asyncio.run(run_pipeline(
        people=people,
        concurrency=args.concurrency,
        api_url=args.api_url,
        api_key=os.getenv("PARALLEL_API_KEY", ""),
    ))


if __name__ == "__main__":
    main()
