#!/usr/bin/env python3
"""
DEEP PERSONA GENERATION PIPELINE

Generates rich, detailed personas that capture the REAL person:
- Personality quirks and behavioral patterns
- Real relationships, friendships, feuds
- Gossip, controversies, skeletons
- How they actually talk vs their PR voice
- Social dynamics and power plays
- Internet presence and reputation
- The stuff that makes them human

For multi-agent simulation where models need to BECOME these people.
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

PARALLEL_API_URL = os.getenv("PARALLEL_API_URL", "http://127.0.0.1:4002")
PARALLEL_API_KEY = os.getenv("PARALLEL_API_KEY", "")
OUTPUT_DIR = Path(__file__).parent / "personas"
CONCURRENCY = 3


# Much richer schema for deep personas
DEEP_PERSONA_SCHEMA = {
    "type": "json",
    "json_schema": {
        "type": "object",
        "properties": {
            # === IDENTITY ===
            "name": {"type": "string"},
            "nicknames": {"type": "array", "items": {"type": "string"}, "description": "What people call them informally, Twitter handles, etc"},
            "current_title": {"type": "string"},
            "age": {"type": "string"},
            "nationality": {"type": "string"},
            "location": {"type": "string", "description": "Where they're based, where they spend time"},

            # === THE REAL PERSON ===
            "personality": {
                "type": "object",
                "properties": {
                    "core_traits": {"type": "array", "items": {"type": "string"}, "description": "Big Five style traits - are they introverted, neurotic, agreeable, etc"},
                    "quirks": {"type": "array", "items": {"type": "string"}, "description": "Weird habits, tics, things people notice about them"},
                    "pet_peeves": {"type": "array", "items": {"type": "string"}, "description": "What pisses them off"},
                    "insecurities": {"type": "array", "items": {"type": "string"}, "description": "What they're insecure about, defensive about"},
                    "ego_triggers": {"type": "array", "items": {"type": "string"}, "description": "What flatters them, what they brag about"},
                    "sense_of_humor": {"type": "string", "description": "Do they joke? What kind of humor?"},
                    "emotional_patterns": {"type": "string", "description": "How do they handle stress, criticism, praise?"},
                    "energy_level": {"type": "string", "description": "High energy, calm, manic, etc"},
                }
            },

            # === HOW THEY ACTUALLY TALK ===
            "communication": {
                "type": "object",
                "properties": {
                    "speaking_style": {"type": "string", "description": "How they talk in interviews, meetings - formal, casual, professorial, salesy?"},
                    "writing_style": {"type": "string", "description": "How they write emails, tweets, memos"},
                    "verbal_tics": {"type": "array", "items": {"type": "string"}, "description": "Phrases they overuse, filler words, catchphrases"},
                    "favorite_topics": {"type": "array", "items": {"type": "string"}, "description": "What they always steer conversations toward"},
                    "topics_they_avoid": {"type": "array", "items": {"type": "string"}, "description": "What they dodge or get uncomfortable about"},
                    "debate_tactics": {"type": "array", "items": {"type": "string"}, "description": "How they argue - do they interrupt, use data, appeal to emotion?"},
                    "real_voice_vs_pr_voice": {"type": "string", "description": "How different are they in private vs public?"},
                    "sample_quotes": {"type": "array", "items": {"type": "string"}, "description": "Actual quotes that capture how they talk"},
                }
            },

            # === RELATIONSHIPS ===
            "relationships": {
                "type": "object",
                "properties": {
                    "inner_circle": {"type": "array", "items": {"type": "string"}, "description": "Who they actually trust, talk to daily"},
                    "close_friends": {"type": "array", "items": {"type": "string"}, "description": "Real friendships, not just professional"},
                    "mentors": {"type": "array", "items": {"type": "string"}, "description": "Who shaped them, who they look up to"},
                    "proteges": {"type": "array", "items": {"type": "string"}, "description": "Who they've mentored, brought up"},
                    "rivals": {"type": "array", "items": {"type": "string"}, "description": "Professional competition, people they're compared to"},
                    "enemies": {"type": "array", "items": {"type": "string"}, "description": "People they actually dislike or have beef with"},
                    "frenemies": {"type": "array", "items": {"type": "string"}, "description": "Complicated relationships, public allies but private tension"},
                    "romantic_family": {"type": "string", "description": "Spouse, kids, family situation - how it affects them"},
                    "notable_fallings_out": {"type": "array", "items": {"type": "string"}, "description": "Public or known breakups with former allies"},
                    "surprising_connections": {"type": "array", "items": {"type": "string"}, "description": "Unexpected relationships people don't know about"},
                }
            },

            # === POWER & INFLUENCE ===
            "power_dynamics": {
                "type": "object",
                "properties": {
                    "real_power_sources": {"type": "array", "items": {"type": "string"}, "description": "Where their actual influence comes from"},
                    "weaknesses": {"type": "array", "items": {"type": "string"}, "description": "What limits their power"},
                    "who_they_defer_to": {"type": "array", "items": {"type": "string"}, "description": "Who can tell them what to do"},
                    "who_defers_to_them": {"type": "array", "items": {"type": "string"}, "description": "Who they have power over"},
                    "political_savvy": {"type": "string", "description": "How good are they at organizational politics?"},
                    "reputation_among_peers": {"type": "string", "description": "What do people at their level really think of them?"},
                    "reputation_among_subordinates": {"type": "string", "description": "What do people who work for them say?"},
                }
            },

            # === GOSSIP & CONTROVERSIES ===
            "controversies": {
                "type": "object",
                "properties": {
                    "public_scandals": {"type": "array", "items": {"type": "string"}, "description": "Known controversies, PR disasters"},
                    "open_secrets": {"type": "array", "items": {"type": "string"}, "description": "Things 'everyone knows' but aren't widely reported"},
                    "rumors": {"type": "array", "items": {"type": "string"}, "description": "Unverified but persistent gossip"},
                    "hypocrisies": {"type": "array", "items": {"type": "string"}, "description": "Where their actions don't match their words"},
                    "skeletons": {"type": "array", "items": {"type": "string"}, "description": "Past issues that could resurface"},
                    "things_they_regret": {"type": "array", "items": {"type": "string"}, "description": "Decisions they've walked back or apologized for"},
                }
            },

            # === INTERNET PRESENCE ===
            "internet_presence": {
                "type": "object",
                "properties": {
                    "twitter_persona": {"type": "string", "description": "How they use Twitter/X - frequency, tone, what they engage with"},
                    "twitter_handle": {"type": "string"},
                    "other_social_media": {"type": "array", "items": {"type": "string"}, "description": "LinkedIn, Instagram, Threads, etc"},
                    "how_theyre_memed": {"type": "string", "description": "Internet jokes about them, meme status"},
                    "fan_communities": {"type": "array", "items": {"type": "string"}, "description": "Where their fans/critics congregate"},
                    "notable_twitter_beefs": {"type": "array", "items": {"type": "string"}, "description": "Public social media fights"},
                    "viral_moments": {"type": "array", "items": {"type": "string"}, "description": "Things that went viral about them"},
                    "reddit_reputation": {"type": "string", "description": "What r/technology, r/MachineLearning, etc think of them"},
                    "hacker_news_reputation": {"type": "string", "description": "How HN discusses them"},
                }
            },

            # === BACKGROUND ===
            "background": {
                "type": "object",
                "properties": {
                    "origin_story": {"type": "string", "description": "Where they came from, formative experiences"},
                    "education": {"type": "array", "items": {"type": "string"}},
                    "career_trajectory": {"type": "array", "items": {"type": "string"}, "description": "Key career moves and why"},
                    "lucky_breaks": {"type": "array", "items": {"type": "string"}, "description": "Right place right time moments"},
                    "failures": {"type": "array", "items": {"type": "string"}, "description": "Things that didn't work out"},
                    "formative_relationships": {"type": "array", "items": {"type": "string"}, "description": "People who shaped their career"},
                    "wealth_status": {"type": "string", "description": "How rich are they, where does money come from"},
                }
            },

            # === BELIEFS & WORLDVIEW ===
            "worldview": {
                "type": "object",
                "properties": {
                    "core_beliefs": {"type": "array", "items": {"type": "string"}, "description": "What they genuinely believe about the world"},
                    "political_leanings": {"type": "string", "description": "Where they fall politically, even if they hide it"},
                    "ai_philosophy": {"type": "string", "description": "What they really think about AI - doomer, accelerationist, pragmatist?"},
                    "china_views": {"type": "string", "description": "How they see US-China competition"},
                    "regulation_views": {"type": "string", "description": "Pro or anti regulation, why"},
                    "sacred_cows": {"type": "array", "items": {"type": "string"}, "description": "Ideas they won't question"},
                    "ideological_evolution": {"type": "string", "description": "How their views have changed over time"},
                    "influences": {"type": "array", "items": {"type": "string"}, "description": "Books, thinkers, ideas that shaped them"},
                }
            },

            # === DECISION MAKING ===
            "decision_making": {
                "type": "object",
                "properties": {
                    "thinking_style": {"type": "string", "description": "Analytical, intuitive, consensus-driven, autocratic?"},
                    "risk_tolerance": {"type": "string", "description": "How much risk do they take?"},
                    "speed_vs_deliberation": {"type": "string", "description": "Fast mover or slow and careful?"},
                    "who_they_consult": {"type": "array", "items": {"type": "string"}, "description": "Who do they ask before big decisions?"},
                    "blind_spots": {"type": "array", "items": {"type": "string"}, "description": "What they consistently miss or underweight"},
                    "past_prediction_accuracy": {"type": "string", "description": "How good is their judgment historically?"},
                }
            },

            # === CURRENT SITUATION ===
            "current_state": {
                "type": "object",
                "properties": {
                    "current_priorities": {"type": "array", "items": {"type": "string"}, "description": "What they're focused on right now"},
                    "current_battles": {"type": "array", "items": {"type": "string"}, "description": "Fights they're in the middle of"},
                    "recent_wins": {"type": "array", "items": {"type": "string"}, "description": "Recent successes"},
                    "recent_losses": {"type": "array", "items": {"type": "string"}, "description": "Recent setbacks"},
                    "momentum": {"type": "string", "description": "Are they ascendant or declining?"},
                    "stress_level": {"type": "string", "description": "How much pressure are they under?"},
                }
            },

            # === FOR SIMULATION ===
            "simulation_guide": {
                "type": "object",
                "properties": {
                    "how_to_embody": {"type": "string", "description": "Key instructions for an AI playing this person"},
                    "speech_patterns_to_mimic": {"type": "array", "items": {"type": "string"}},
                    "things_they_would_never_say": {"type": "array", "items": {"type": "string"}},
                    "hot_buttons": {"type": "array", "items": {"type": "string"}, "description": "Topics that make them emotional"},
                    "how_they_enter_a_room": {"type": "string", "description": "Their presence and energy"},
                    "tells_when_lying_or_uncomfortable": {"type": "array", "items": {"type": "string"}},
                    "negotiation_style": {"type": "string", "description": "How they negotiate - hardball, collaborative, passive aggressive?"},
                    "under_pressure_behavior": {"type": "string", "description": "How they act when stressed or cornered"},
                }
            },

            "last_updated": {"type": "string"},
            "confidence_level": {"type": "string", "description": "How confident are we in this profile?"},
            "sources": {"type": "array", "items": {"type": "string"}},
        }
    }
}


class DeepPersonaGenerator:
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

    async def search(self, query: str, max_results: int = 15) -> list[dict]:
        """Run a web search."""
        try:
            resp = await self.client.post(
                "/v1beta/search",
                json={"objective": query, "max_results": max_results},
            )
            resp.raise_for_status()
            return resp.json().get("results", [])
        except Exception as e:
            print(f"    Search error: {e}")
            return []

    async def deep_research(self, name: str, role: str, category: str) -> str:
        """Multi-query deep research to find the real person."""
        print(f"  [RESEARCH] Deep diving on {name}...")

        # Diverse search queries to capture different facets
        queries = [
            # Basic info
            f"{name} biography background career history",
            f"{name} interview podcast personality",

            # Relationships & drama
            f"{name} friends relationships allies",
            f"{name} controversy scandal criticism",
            f"{name} feud beef conflict rivalry",
            f'"{name}" twitter drama',

            # How they really are
            f"{name} personality management style leadership",
            f"{name} quotes statements opinions",
            f'"{name}" reddit opinion',
            f'"{name}" hacker news',

            # Current state
            f"{name} 2024 2025 recent news",
            f"{name} {role} AI chip policy",

            # The juice
            f"{name} rumors gossip",
            f"{name} criticized accused",
            f'"{name}" annoying weird quirks',
        ]

        # Run searches in parallel batches
        all_results = []
        seen_urls = set()

        batch_size = 5
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            tasks = [self.search(q, max_results=10) for q in batch]
            results = await asyncio.gather(*tasks)

            for batch_results in results:
                for r in batch_results:
                    url = r.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append(r)

        print(f"    Found {len(all_results)} unique sources")

        # Format for prompt
        research_text = ""
        for r in all_results[:40]:  # Top 40 results
            title = r.get("title", "")
            url = r.get("url", "")
            excerpts = " ".join(r.get("excerpts", []))[:800]
            research_text += f"\n---\nSOURCE: {title}\nURL: {url}\n{excerpts}\n"

        return research_text

    async def generate_persona(self, person: dict) -> dict:
        """Generate deep persona for one person."""
        name = person["name"]
        role = person["role"]
        category = person["category_name"]

        print(f"\n{'='*60}")
        print(f"[PROCESSING] {name}")
        print(f"  Role: {role}")
        print(f"  Category: {category}")

        # Deep research
        research = await self.deep_research(name, role, category)

        print(f"  [GENERATE] Creating deep persona...")

        prompt = f"""You are creating a DEEP, REALISTIC persona profile for {name} that will be used to simulate them in multi-agent scenarios. An AI needs to be able to BECOME this person.

DO NOT give me a sanitized Wikipedia version. I need the REAL person:
- Their actual personality, quirks, and behavioral patterns
- Real relationships, friendships, feuds, and drama
- Gossip, controversies, and open secrets
- How they ACTUALLY talk vs their PR voice
- Their insecurities, ego triggers, and hot buttons
- Internet presence - how they're memed, twitter beefs, reputation on Reddit/HN
- The human details that make them who they are

RESEARCH DATA:
{research[:25000]}

CONTEXT:
- Name: {name}
- Known Role: {role}
- Category: {category}

Be specific and cite examples. Include actual quotes that capture how they talk. Don't hedge - make educated inferences based on available information. If something is gossip/rumor, label it as such but still include it.

The goal is for an AI to read this and be able to convincingly roleplay as {name} - capturing their voice, decision patterns, relationships, and personality."""

        try:
            resp = await self.client.post(
                "/v1/tasks/runs/managed",
                json={
                    "processor": "ultra",  # Use the best for rich generation
                    "input": {"query": prompt},
                    "task_spec": {"output_schema": DEEP_PERSONA_SCHEMA},
                },
                params={"poll_interval_seconds": 3, "timeout_seconds": 900},
            )
            resp.raise_for_status()
            result = resp.json()

            if result.get("output_json"):
                persona = result["output_json"]
            elif result.get("output_text"):
                # Try to parse JSON from text
                text = result["output_text"]
                match = re.search(r'\{[\s\S]*\}', text)
                if match:
                    try:
                        persona = json.loads(match.group())
                    except json.JSONDecodeError:
                        persona = {"name": name, "raw_output": text, "error": "json_parse_failed"}
                else:
                    persona = {"name": name, "raw_output": text, "error": "no_json_found"}
            else:
                persona = {"name": name, "error": "no_output", "raw": result}

        except Exception as e:
            print(f"  [ERROR] Generation failed: {e}")
            persona = {"name": name, "error": str(e)}

        # Add metadata
        persona["_metadata"] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "category_id": person["category_id"],
            "category_name": category,
            "role": role,
            "research_sources": len(research.split("---")) - 1,
        }

        # Save
        await self.save_persona(name, persona)
        return persona

    async def save_persona(self, name: str, persona: dict):
        """Save persona to file."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        safe_name = re.sub(r'[^a-z0-9_]', '_', name.lower())
        safe_name = re.sub(r'_+', '_', safe_name).strip('_')
        filepath = OUTPUT_DIR / f"{safe_name}.json"

        with open(filepath, "w") as f:
            json.dump(persona, f, indent=2, default=str, ensure_ascii=False)
        print(f"  [SAVED] {filepath.name}")


async def run_pipeline(people: list[dict], concurrency: int = CONCURRENCY):
    """Run the deep persona pipeline."""
    print(f"DEEP PERSONA GENERATION")
    print(f"People: {len(people)}")
    print(f"Concurrency: {concurrency}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async with DeepPersonaGenerator(PARALLEL_API_URL, PARALLEL_API_KEY) as gen:
        async def process(person):
            async with semaphore:
                try:
                    return await gen.generate_persona(person)
                except Exception as e:
                    print(f"[FAILED] {person['name']}: {e}")
                    return {"name": person["name"], "error": str(e)}

        results = await asyncio.gather(*[process(p) for p in people])

    successful = sum(1 for r in results if "error" not in r)
    print(f"\n{'=' * 60}")
    print(f"COMPLETE: {successful}/{len(people)} personas generated")
    print(f"Output: {OUTPUT_DIR}")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate deep personas")
    parser.add_argument("--category", "-c", type=str, help="Only one category")
    parser.add_argument("--person", "-p", type=str, help="Only one person")
    parser.add_argument("--concurrency", "-n", type=int, default=CONCURRENCY)
    parser.add_argument("--list", "-l", action="store_true", help="List people")
    args = parser.parse_args()

    # Get people
    if args.person:
        all_people = get_all_people()
        people = [p for p in all_people if args.person.lower() in p["name"].lower()]
        if not people:
            print(f"No match for: {args.person}")
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
