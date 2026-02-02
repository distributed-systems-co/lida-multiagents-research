#!/usr/bin/env python3
"""
PERSONA ENHANCER - Adds gossip, leaks, and juicy details to existing personas

Searches for:
- Leaked emails/documents
- Reddit/HN sentiment
- Glassdoor reviews
- Twitter beefs
- Anonymous sources
- Critic quotes
- Embarrassing moments
- Behind-the-scenes drama

Merges into existing profiles WITHOUT overwriting good content.
"""

import asyncio
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

# =============================================================================
# CONFIG
# =============================================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
PARALLEL_API_KEY = os.getenv("PARALLEL_API_KEY", "")
PARALLEL_API_URL = os.getenv("PARALLEL_API_URL", "https://api.parallel.ai")

FINALIZED_DIR = Path(__file__).parent / "personas" / "finalized"
ENHANCED_DIR = Path(__file__).parent / "personas" / "enhanced"

# Gossip-focused search templates
GOSSIP_SEARCHES = [
    '"{name}" leaked email internal memo',
    '"{name}" former employee says insider',
    '"{name}" reddit what people really think',
    '"{name}" hacker news criticism thread',
    '"{name}" glassdoor reviews toxic workplace',
    '"{name}" fired why real reason',
    '"{name}" feud beef drama fight',
    '"{name}" twitter beef argument ratio',
    '"{name}" hypocrite caught doing opposite',
    '"{name}" embarrassing moment gaffe video',
    '"{name}" anonymous source claims',
    '"{name}" ex-employee whistleblower',
    '"{name}" lawsuit settlement details',
    '"{name}" behind the scenes really like',
    '"{name}" enemies hate despise',
    '"{name}" betrayed former ally',
    '"{name}" secret scandal hidden',
    '"{name}" controversial statement backlash',
    '"{name}" critics say worst',
    '"{name}" intern horror story',
]

ENHANCEMENT_PROMPT = '''You are enhancing an existing persona profile with GOSSIP and JUICY DETAILS.

EXISTING PROFILE:
{existing}

NEW RESEARCH (gossip, leaks, criticism, drama):
{research}

Your task: Extract NEW juicy information from the research and add it to the appropriate sections.

Return a JSON object with ONLY the sections that have NEW information to add. Use these exact keys:

{{
  "leaked_communications": [
    {{"source": "where leaked", "date": "when", "content": "what was said", "impact": "what happened"}}
  ],
  "critic_quotes": [
    {{"critic": "name/org", "quote": "exact quote", "context": "when/why said"}}
  ],
  "reddit_sentiment": [
    {{"subreddit": "which one", "sentiment": "positive/negative/mixed", "common_criticisms": ["list"], "common_praise": ["list"]}}
  ],
  "glassdoor_intel": {{
    "rating": "X/5 if known",
    "common_complaints": ["list"],
    "management_style_reviews": ["quotes from reviews"]
  }},
  "twitter_beefs": [
    {{"opponent": "who", "date": "when", "trigger": "what started it", "outcome": "how ended"}}
  ],
  "embarrassing_moments": [
    {{"date": "when", "event": "what happened", "reaction": "how they responded", "lasting_impact": "meme status etc"}}
  ],
  "insider_accounts": [
    {{"source": "former employee/insider", "claim": "what they said", "credibility": "verified/alleged"}}
  ],
  "hypocrisy_instances": [
    {{"claim": "what they publicly say", "reality": "what they actually do", "evidence": "how we know"}}
  ],
  "hidden_controversies": [
    {{"issue": "what", "details": "specifics", "why_hidden": "why not widely known"}}
  ],
  "enemy_quotes": [
    {{"enemy": "who", "quote": "what they said", "context": "when/why"}}
  ]
}}

IMPORTANT:
- Only include sections where you found NEW information
- Be SPECIFIC: names, dates, exact quotes
- Include source credibility notes
- Don't repeat what's already in the existing profile
- If the research doesn't have good new gossip, return {{"no_new_gossip": true}}

Return ONLY valid JSON.'''


# =============================================================================
# SEARCH
# =============================================================================

async def search_gossip(name: str, max_searches: int = 20) -> str:
    """Search for gossip about a person."""
    queries = [q.format(name=name) for q in GOSSIP_SEARCHES[:max_searches]]

    research = ""
    async with httpx.AsyncClient(
        timeout=60.0,
        headers={
            "Authorization": f"Bearer {PARALLEL_API_KEY}",
            "Parallel-Beta": "search-extract-2025-10-10",
        },
    ) as client:
        for i, query in enumerate(queries):
            try:
                resp = await client.post(
                    f"{PARALLEL_API_URL}/v1beta/search",
                    json={"objective": query, "max_results": 8},
                )
                results = resp.json().get("results", [])
                for r in results:
                    title = r.get("title", "")
                    url = r.get("url", "")
                    excerpts = " ".join(r.get("excerpts", []))[:600]
                    research += f"\n[{title}]\n{url}\n{excerpts}\n"
                print(f"  Search {i+1}/{len(queries)}: {len(results)} results", end="\r")
                await asyncio.sleep(0.2)
            except Exception as e:
                pass

    print(f"  Collected {len(research):,} chars of gossip research")
    return research


# =============================================================================
# ENHANCE
# =============================================================================

async def enhance_persona(name: str, existing: dict, research: str) -> dict:
    """Use Opus 4.5 to extract gossip from research."""

    # Truncate existing for prompt
    existing_str = json.dumps(existing, indent=2)[:15000]

    prompt = ENHANCEMENT_PROMPT.format(
        existing=existing_str,
        research=research[:35000],
    )

    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "anthropic/claude-opus-4.5",
                "messages": [
                    {"role": "system", "content": "You extract gossip and juicy details from research. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 8000,
            },
        )
        data = resp.json()
        text = data["choices"][0]["message"]["content"]

    # Parse JSON
    match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
    if match:
        return json.loads(match.group(1))

    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        return json.loads(match.group())

    return {"parse_error": True, "raw": text[:2000]}


def merge_enhancements(existing: dict, enhancements: dict) -> dict:
    """Merge new gossip into existing profile without overwriting."""

    if enhancements.get("no_new_gossip") or enhancements.get("parse_error"):
        return existing

    merged = existing.copy()

    # Create or update gossip section
    if "_gossip" not in merged:
        merged["_gossip"] = {}

    gossip = merged["_gossip"]

    # Merge each gossip category
    for key in [
        "leaked_communications", "critic_quotes", "reddit_sentiment",
        "glassdoor_intel", "twitter_beefs", "embarrassing_moments",
        "insider_accounts", "hypocrisy_instances", "hidden_controversies",
        "enemy_quotes"
    ]:
        if key in enhancements and enhancements[key]:
            if key not in gossip:
                gossip[key] = []

            new_items = enhancements[key]
            if isinstance(new_items, list):
                gossip[key].extend(new_items)
            else:
                gossip[key] = new_items

    # Update metadata
    merged["_gossip"]["_enhanced_at"] = datetime.now(timezone.utc).isoformat()
    merged["_gossip"]["_model"] = "anthropic/claude-opus-4.5"

    return merged


# =============================================================================
# MAIN
# =============================================================================

async def enhance_single(name: str, input_path: Path, output_dir: Path) -> dict:
    """Enhance a single persona."""
    print(f"\n{'='*60}")
    print(f"[ENHANCE] {name}")
    print("="*60)

    # Load existing
    with open(input_path) as f:
        existing = json.load(f)

    existing_size = input_path.stat().st_size
    print(f"  Existing: {existing_size:,} bytes")

    # Search for gossip
    print("  Searching for gossip...")
    research = await search_gossip(name)

    if len(research) < 1000:
        print("  Not enough gossip found, skipping")
        return {"status": "skipped", "reason": "insufficient_research"}

    # Enhance with Opus 4.5
    print("  Extracting juicy details with Opus 4.5...")
    enhancements = await enhance_persona(name, existing, research)

    if enhancements.get("no_new_gossip"):
        print("  No new gossip found")
        # Still save to enhanced dir
        output_path = output_dir / f"{name}.json"
        with open(output_path, "w") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        return {"status": "no_new_gossip"}

    if enhancements.get("parse_error"):
        print(f"  Parse error: {enhancements.get('raw', '')[:200]}")
        return {"status": "error", "reason": "parse_error"}

    # Merge
    merged = merge_enhancements(existing, enhancements)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.json"
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    new_size = output_path.stat().st_size
    gossip_keys = [k for k in enhancements.keys() if not k.startswith("_") and k != "no_new_gossip"]

    print(f"  Added: {', '.join(gossip_keys)}")
    print(f"  Size: {existing_size:,} -> {new_size:,} bytes (+{new_size - existing_size:,})")

    return {
        "status": "enhanced",
        "added_sections": gossip_keys,
        "size_before": existing_size,
        "size_after": new_size,
    }


async def enhance_all(names: list[str] = None, max_concurrent: int = 2):
    """Enhance multiple personas."""

    if names is None:
        # Get all finalized personas
        names = [p.stem for p in FINALIZED_DIR.glob("*.json")]

    print(f"Enhancing {len(names)} personas")
    print(f"Input: {FINALIZED_DIR}")
    print(f"Output: {ENHANCED_DIR}")

    ENHANCED_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process(name):
        async with semaphore:
            input_path = FINALIZED_DIR / f"{name}.json"
            if not input_path.exists():
                return {"name": name, "status": "not_found"}
            return {"name": name, **await enhance_single(name, input_path, ENHANCED_DIR)}

    results = await asyncio.gather(*[process(n) for n in names])

    # Summary
    print(f"\n{'='*60}")
    print("ENHANCEMENT COMPLETE")
    print("="*60)

    enhanced = [r for r in results if r.get("status") == "enhanced"]
    skipped = [r for r in results if r.get("status") in ("skipped", "no_new_gossip")]
    errors = [r for r in results if r.get("status") == "error"]

    print(f"Enhanced: {len(enhanced)}")
    print(f"Skipped:  {len(skipped)}")
    print(f"Errors:   {len(errors)}")

    if enhanced:
        total_added = sum(r.get("size_after", 0) - r.get("size_before", 0) for r in enhanced)
        print(f"Total gossip added: {total_added:,} bytes")


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Enhance personas with gossip")
    parser.add_argument("-p", "--person", type=str, help="Enhance specific person")
    parser.add_argument("-n", "--concurrency", type=int, default=2, help="Max concurrent")
    parser.add_argument("-l", "--list", action="store_true", help="List available personas")
    args = parser.parse_args()

    if args.list:
        for p in sorted(FINALIZED_DIR.glob("*.json")):
            size = p.stat().st_size
            print(f"  {p.stem:40} {size:>8,} bytes")
        return

    if args.person:
        names = [n.strip() for n in args.person.split(",")]
    else:
        names = None  # All

    asyncio.run(enhance_all(names, max_concurrent=args.concurrency))


if __name__ == "__main__":
    main()
