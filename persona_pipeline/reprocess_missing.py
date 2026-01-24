#!/usr/bin/env python3
"""
Reprocess personas to fill in ONLY missing fields.
Does not overwrite existing data - only adds what's missing.
"""

import json
import os
import asyncio
import httpx
from pathlib import Path
from typing import Dict, Any, List, Optional

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = "anthropic/claude-sonnet-4.5"

# Required fields and how to generate them
REQUIRED_FIELDS = {
    "name": "Full name of the person",
    "current_role": "Their current job title and organization",
    "category": "One of: Political, Technology, Military, Intelligence, Economic, International",
}


def get_missing_fields(data: Dict[str, Any]) -> List[str]:
    """Identify which required fields are missing."""
    missing = []

    # Check name
    if not data.get("name") and not data.get("subject"):
        missing.append("name")

    # Check current_role (also check background.current_position)
    if not data.get("current_role"):
        bg = data.get("background", {})
        if not bg.get("current_position") and not bg.get("title") and not bg.get("role"):
            missing.append("current_role")

    # Check category
    if not data.get("category") and not data.get("domain"):
        missing.append("category")

    return missing


def derive_name_from_filename(filename: str) -> str:
    """Convert filename to proper name: jensen_huang -> Jensen Huang"""
    return filename.replace("_", " ").title()


async def generate_missing_fields(
    filename: str,
    existing_data: Dict[str, Any],
    missing: List[str],
) -> Dict[str, str]:
    """Use LLM to generate only the missing fields."""

    if not OPENROUTER_API_KEY:
        # Fallback: derive what we can without LLM
        result = {}
        derived_name = derive_name_from_filename(filename)

        if "name" in missing:
            result["name"] = derived_name

        if "category" in missing:
            # Guess from context
            bg = existing_data.get("background", {})
            career = str(bg.get("career_arc", "")).lower()
            if any(x in career for x in ["congress", "senator", "secretary", "president", "governor"]):
                result["category"] = "Political"
            elif any(x in career for x in ["ceo", "founder", "nvidia", "amd", "google", "meta", "openai"]):
                result["category"] = "Technology"
            elif any(x in career for x in ["general", "pentagon", "defense"]):
                result["category"] = "Military"
            elif any(x in career for x in ["cia", "nsa", "intelligence"]):
                result["category"] = "Intelligence"
            else:
                result["category"] = "Political"  # default

        if "current_role" in missing:
            result["current_role"] = ""  # Can't derive without LLM

        return result

    # Build a minimal prompt for just the missing fields
    derived_name = derive_name_from_filename(filename)

    # Extract context from existing data
    context_parts = []
    career = existing_data.get("background", {}).get("career_arc")
    if career:
        if isinstance(career, str):
            context_parts.append(f"Career: {career[:500]}")
        elif isinstance(career, list):
            context_parts.append(f"Career: {'; '.join(str(c) for c in career[:5])}")

    embody = existing_data.get("simulation_guide", {}).get("how_to_embody")
    if embody:
        if isinstance(embody, str):
            context_parts.append(f"Description: {embody[:300]}")
        elif isinstance(embody, list):
            context_parts.append(f"Description: {'; '.join(str(e) for e in embody[:3])}")

    context = "\n".join(context_parts) if context_parts else "No additional context available."

    fields_to_generate = {k: v for k, v in REQUIRED_FIELDS.items() if k in missing}

    prompt = f"""For the person "{derived_name}", provide ONLY these missing fields as JSON:

{json.dumps(fields_to_generate, indent=2)}

Context from their existing profile:
{context}

Respond with ONLY a JSON object containing these fields. No explanation."""

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 256,
                "temperature": 0.3,
            },
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]

        # Parse JSON from response
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        return json.loads(content)


async def reprocess_persona(path: Path, dry_run: bool = False) -> Optional[Dict[str, Any]]:
    """Reprocess a single persona file, filling in missing fields."""

    with open(path) as f:
        data = json.load(f)

    missing = get_missing_fields(data)

    if not missing:
        return None  # Nothing to do

    print(f"  {path.stem}: missing {missing}")

    if dry_run:
        return {"file": path.stem, "missing": missing}

    # Generate missing fields
    new_fields = await generate_missing_fields(path.stem, data, missing)

    # Merge into existing data (don't overwrite existing fields)
    updated = False
    for key, value in new_fields.items():
        if key not in data or not data[key]:
            data[key] = value
            updated = True
            print(f"    + {key}: {value[:50] if isinstance(value, str) else value}...")

    if updated:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"    ✓ Updated {path.stem}")

    return {"file": path.stem, "added": list(new_fields.keys())}


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fill in missing persona fields")
    parser.add_argument("--dry-run", action="store_true", help="Don't modify files, just report")
    parser.add_argument("--dir", default="persona_pipeline/personas/finalized", help="Persona directory")
    parser.add_argument("--persona", help="Process single persona by name")
    args = parser.parse_args()

    persona_dir = Path(args.dir)

    if args.persona:
        paths = [persona_dir / f"{args.persona}.json"]
    else:
        paths = sorted(persona_dir.glob("*.json"))

    print(f"Scanning {len(paths)} personas in {persona_dir}...")
    if args.dry_run:
        print("(DRY RUN - no files will be modified)\n")

    results = []
    for path in paths:
        if not path.exists():
            print(f"  {path.stem}: NOT FOUND")
            continue
        result = await reprocess_persona(path, dry_run=args.dry_run)
        if result:
            results.append(result)

    print(f"\n{'='*50}")
    print(f"Processed {len(results)} personas with missing fields")
    if not args.dry_run and results:
        print("Fields were added to the persona files.")


if __name__ == "__main__":
    asyncio.run(main())
