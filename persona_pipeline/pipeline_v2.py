#!/usr/bin/env python3
"""
PERSONA PIPELINE V2 - Date-stamped, non-destructive

Key improvements:
- Filenames include generation date: jensen_huang_2026-01-23.json
- NEVER overwrites if existing file is larger (preserves good versions)
- Primary: Opus + Sonnet (randomized)
- Fallback: Haiku -> GPT-4o-mini -> Gemini Flash -> DeepSeek
- Extensive research and comprehensive persona generation
"""

import asyncio
import json
import os
import re
import random
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable

import httpx

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Primary model - USE OPUS 4.5 FIRST
PRIMARY_MODELS = [
    "anthropic/claude-opus-4.5",
]

# Fallback chain - 4.5 family then others
FALLBACK_MODELS = [
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-haiku-4.5",
    "openai/gpt-4o",
    "deepseek/deepseek-chat",
]

MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0


def get_model_chain() -> list[str]:
    """Get randomized model chain: shuffled primaries + fallbacks."""
    primaries = PRIMARY_MODELS.copy()
    random.shuffle(primaries)
    return primaries + FALLBACK_MODELS


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PipelineConfig:
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    parallel_api_key: str = field(default_factory=lambda: os.getenv("PARALLEL_API_KEY", ""))
    parallel_api_url: str = field(default_factory=lambda: os.getenv("PARALLEL_API_URL", "https://api.parallel.ai"))

    output_dir: Path = field(default_factory=lambda: Path(__file__).parent / "personas")
    checkpoint_file: Path = field(default_factory=lambda: Path(__file__).parent / ".checkpoint_v2.json")

    max_concurrent: int = 2
    max_searches: int = 80
    results_per_search: int = 8
    search_batch_size: int = 8
    search_delay: float = 0.25

    # Minimum file size to consider "good" (don't overwrite if existing is larger)
    min_good_size: int = 12000


# =============================================================================
# FILE NAMING WITH DATES
# =============================================================================

def get_dated_filename(name: str, date: datetime = None) -> str:
    """
    Generate filename with date: jensen_huang_2026-01-23.json
    """
    date = date or datetime.now(timezone.utc)
    date_str = date.strftime("%Y-%m-%d")
    safe_name = re.sub(r'[^a-z0-9_]', '_', name.lower()).strip('_')
    safe_name = re.sub(r'_+', '_', safe_name)
    return f"{safe_name}_{date_str}.json"


def get_latest_persona_file(output_dir: Path, name: str) -> Optional[Path]:
    """Find the latest/largest persona file for a given name."""
    safe_name = re.sub(r'[^a-z0-9_]', '_', name.lower()).strip('_')
    safe_name = re.sub(r'_+', '_', safe_name)

    # Look for files matching pattern: name_*.json or name.json
    matches = list(output_dir.glob(f"{safe_name}*.json"))

    if not matches:
        return None

    # Return largest file (most complete)
    return max(matches, key=lambda p: p.stat().st_size)


def should_generate(output_dir: Path, name: str, min_size: int) -> tuple[bool, Optional[Path]]:
    """
    Check if we should generate a new persona.
    Returns (should_generate, existing_file)

    Rules:
    - If no existing file: generate
    - If existing file < min_size: generate (it's a fallback)
    - If existing file >= min_size: skip (good version exists)
    """
    existing = get_latest_persona_file(output_dir, name)

    if existing is None:
        return True, None

    size = existing.stat().st_size
    if size < min_size:
        return True, existing  # Existing is a fallback, regenerate

    return False, existing  # Good version exists, skip


# =============================================================================
# MODEL CALLER
# =============================================================================

class ModelCaller:
    """Calls models with fallback chain."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.stats = {"calls": 0, "failures": 0, "fallbacks": 0}

    async def call(
        self,
        messages: list[dict],
        max_tokens: int = 8000,
        json_mode: bool = False,
    ) -> tuple[str, str, dict]:
        """
        Call models in chain until success.
        Returns: (response_text, model_used, usage_stats)
        """
        model_chain = get_model_chain()
        errors = []

        for model in model_chain:
            for attempt in range(MAX_RETRIES):
                try:
                    self.stats["calls"] += 1
                    result = await self._single_call(model, messages, max_tokens, json_mode)
                    if model not in PRIMARY_MODELS:
                        self.stats["fallbacks"] += 1
                    return result
                except Exception as e:
                    errors.append(f"{model} (attempt {attempt+1}): {e}")
                    self.stats["failures"] += 1

                    if attempt < MAX_RETRIES - 1:
                        delay = RETRY_DELAY_BASE ** attempt + random.uniform(0, 1)
                        await asyncio.sleep(delay)

        raise Exception(f"All models failed:\n" + "\n".join(errors))

    async def _single_call(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        json_mode: bool,
    ) -> tuple[str, str, dict]:
        """Single model call."""
        async with httpx.AsyncClient(timeout=180.0) as client:
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}

            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})

            # Validate response
            if not text or len(text.strip()) < 100:
                raise Exception("Empty or too short response")

            # Check for refusals
            refusal_phrases = [
                "i can't help", "i cannot help", "i'm not able", "i am not able",
                "i'm unable", "i am unable", "cannot assist", "can't assist",
                "i apologize, but", "i must decline", "against my guidelines",
            ]

            text_lower = text.lower()
            for phrase in refusal_phrases:
                if phrase in text_lower[:500]:  # Only check beginning
                    raise Exception(f"Model refused: detected '{phrase}'")

            return text, model, usage


# =============================================================================
# SEARCH
# =============================================================================

class Searcher:
    """Web search via Parallel AI."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.client: Optional[httpx.AsyncClient] = None
        self.stats = {"searches": 0, "results": 0}

    async def __aenter__(self):
        self.client = httpx.AsyncClient(
            timeout=60.0,
            headers={
                "Authorization": f"Bearer {self.config.parallel_api_key}",
                "Parallel-Beta": "search-extract-2025-10-10",
            },
        )
        return self

    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()

    async def search(self, query: str) -> list[dict]:
        """Single search with retries."""
        for attempt in range(MAX_RETRIES):
            try:
                resp = await self.client.post(
                    f"{self.config.parallel_api_url}/v1beta/search",
                    json={"objective": query, "max_results": self.config.results_per_search},
                )
                resp.raise_for_status()
                self.stats["searches"] += 1
                results = resp.json().get("results", [])
                self.stats["results"] += len(results)
                return results
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY_BASE ** attempt)
        return []

    async def batch_search(self, queries: list[str], progress_cb: Callable = None) -> list[dict]:
        """Batch search with deduplication."""
        all_results = []
        seen_urls = set()

        for i in range(0, len(queries), self.config.search_batch_size):
            batch = queries[i:i + self.config.search_batch_size]
            tasks = [self.search(q) for q in batch]
            batch_results = await asyncio.gather(*tasks)

            for results in batch_results:
                for r in results:
                    url = r.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append(r)

            if progress_cb:
                progress_cb(min(i + self.config.search_batch_size, len(queries)), len(queries), len(all_results))

            await asyncio.sleep(self.config.search_delay)

        return all_results


# =============================================================================
# SEARCH TERM TEMPLATES
# =============================================================================

SEARCH_TEMPLATES = {
    "biography": [
        "{name} biography background early life",
        "{name} education university degree career history",
        "{name} origin story how started career",
    ],
    "current": [
        "{name} 2024 2025 latest news",
        "{name} current role position responsibilities",
        "{name} recent interview statements quotes",
    ],
    "personality": [
        "{name} personality traits character leadership style",
        "{name} management style how treats employees",
        "{name} temper angry outburst behavior",
        "{name} quirks habits mannerisms",
    ],
    "communication": [
        "{name} quotes famous statements speeches",
        "{name} interview transcript podcast appearance",
        '"{name}" said stated believes',
    ],
    "relationships": [
        "{name} friends allies close relationships",
        "{name} enemies rivals conflicts disputes",
        "{name} mentor influenced by mentored",
        "{name} inner circle advisors trusted",
    ],
    "controversies": [
        "{name} scandal controversy criticism",
        "{name} lawsuit sued legal problems",
        "{name} investigation allegations accused",
        "{name} mistakes failures regrets",
    ],
    "policy": [
        "{name} AI artificial intelligence views position",
        "{name} China policy stance opinion",
        "{name} regulation technology policy views",
        "{name} political views beliefs ideology",
    ],
    "social": [
        "{name} twitter X posts controversy",
        "{name} social media presence online",
        "{name} public perception reputation",
    ],
    "financial": [
        "{name} net worth salary compensation",
        "{name} investments business interests",
    ],
    "personal": [
        "{name} family spouse children",
        "{name} hobbies interests outside work",
    ],
}


def generate_search_terms(name: str, role: str) -> list[str]:
    """Generate comprehensive search terms for a person."""
    terms = []
    for category, templates in SEARCH_TEMPLATES.items():
        for t in templates:
            terms.append(t.format(name=name, role=role))

    # Add role-specific searches
    terms.append(f"{name} {role}")
    terms.append(f"{name} {role} policy decisions")
    terms.append(f"{name} {role} accomplishments achievements")

    return terms


# =============================================================================
# PERSONA SYNTHESIS
# =============================================================================

SYNTHESIS_PROMPT = """Create a comprehensive, detailed profile for {name} in JSON format.

Subject: {name}
Current Role: {role}
Category: {category}

Based on the research data below, create an EXTENSIVE profile. Be thorough and specific.
Include actual names, dates, quotes, and details from the research.

Required sections (all must have substantial content):

1. **background**:
   - origin_story (detailed early life, family background)
   - education (schools, degrees, notable experiences)
   - career_arc (chronological career progression with dates)
   - net_worth (if known)

2. **personality**:
   - core_traits (list of 5-10 personality traits with explanations)
   - quirks (specific behavioral quirks or habits)
   - triggers (what angers or frustrates them)
   - insecurities (known vulnerabilities or sensitivities)
   - strengths (what they excel at)

3. **communication**:
   - public_persona (how they present publicly)
   - private_persona (how they are described privately)
   - verbal_tics (speech patterns, favorite phrases)
   - sample_quotes (5-10 REAL quotes with context)
   - communication_style (formal/informal, verbose/concise, etc.)

4. **relationships**:
   - inner_circle (list with names and relationship context)
   - allies (professional allies with context)
   - enemies (rivals, critics with context)
   - burned_bridges (former allies now estranged)
   - key_relationships (most important relationships)

5. **controversies**:
   - scandals (with dates and details)
   - lawsuits (legal issues)
   - criticism (major criticisms received)
   - mistakes (acknowledged failures or regrets)

6. **worldview**:
   - core_beliefs (fundamental beliefs and values)
   - ai_philosophy (views on AI if relevant)
   - political_leanings (political orientation)
   - policy_positions (key policy stances)

7. **current_state**:
   - priorities (current focus areas)
   - challenges (current problems facing)
   - opportunities (current opportunities)
   - trajectory (where they seem to be heading)

8. **simulation_guide**:
   - how_to_embody (detailed instructions for roleplaying this person)
   - voice_characteristics (how they speak)
   - hot_buttons (topics that trigger strong reactions)
   - never_say (things this person would never say)
   - how_to_flatter (what pleases them)
   - how_to_provoke (what angers them)

RESEARCH DATA:
{research}

Return ONLY valid JSON. Be comprehensive - this is for research simulation purposes.
Every section should have real, specific content based on the research."""


class Synthesizer:
    """Synthesizes persona from research."""

    def __init__(self, config: PipelineConfig, caller: ModelCaller):
        self.config = config
        self.caller = caller

    async def synthesize(self, name: str, role: str, category: str, research: str) -> tuple[dict, str]:
        """
        Synthesize persona from research.
        Returns: (persona_dict, model_used)
        """
        prompt = SYNTHESIS_PROMPT.format(
            name=name,
            role=role,
            category=category,
            research=research[:50000],  # Limit research size
        )

        system = """You are a senior researcher creating detailed biographical profiles for academic simulation research.
Your profiles are used in multi-agent systems to model realistic policy debates and negotiations.
Be thorough, factual, and include both positive and challenging aspects of each subject.
Always return valid JSON."""

        text, model_used, usage = await self.caller.call(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=12000,
        )

        # Parse JSON
        persona = self._parse_json(text, name)

        return persona, model_used

    def _parse_json(self, text: str, name: str) -> dict:
        """Parse JSON from response."""
        # Try markdown code block
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding JSON object
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return {"name": name, "raw_output": text[:5000], "parse_error": True}


# =============================================================================
# CHECKPOINT
# =============================================================================

@dataclass
class Checkpoint:
    started_at: str
    completed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    results: dict = field(default_factory=dict)

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> Optional["Checkpoint"]:
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return cls(**json.load(f))
        except:
            return None


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class PersonaPipelineV2:
    """
    Improved persona pipeline with:
    - Date-stamped filenames
    - Non-destructive (won't overwrite good files)
    - Opus/Sonnet primary, Haiku fallback
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.caller = ModelCaller(config.openrouter_api_key)
        self.synthesizer = Synthesizer(config, self.caller)
        self.checkpoint: Optional[Checkpoint] = None

    async def run(self, people: list[dict], resume: bool = True) -> list[dict]:
        """Run pipeline on list of people."""

        # Load/create checkpoint
        if resume:
            self.checkpoint = Checkpoint.load(self.config.checkpoint_file)

        if not self.checkpoint:
            self.checkpoint = Checkpoint(
                started_at=datetime.now(timezone.utc).isoformat(),
            )

        self._print_header(people)

        results = []
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async with Searcher(self.config) as searcher:
            async def process(person):
                async with semaphore:
                    return await self._process_person(person, searcher)

            results = await asyncio.gather(*[process(p) for p in people])

        self._print_summary(results)
        return results

    async def _process_person(self, person: dict, searcher: Searcher) -> dict:
        """Process single person."""
        name = person["name"]
        role = person["role"]
        category = person.get("category_name", "Unknown")

        result = {"name": name, "status": "pending"}

        # Check if we should generate
        should_gen, existing = should_generate(
            self.config.output_dir,
            name,
            self.config.min_good_size
        )

        if not should_gen:
            print(f"\n[SKIP] {name} - good version exists ({existing.stat().st_size:,} bytes)")
            result["status"] = "skipped"
            result["existing_file"] = str(existing)
            self.checkpoint.skipped.append(name)
            self.checkpoint.save(self.config.checkpoint_file)
            return result

        if existing:
            print(f"\n[REGEN] {name} - existing is small ({existing.stat().st_size:,} bytes), regenerating")
        else:
            print(f"\n[NEW] {name}")

        print(f"{'='*60}")

        try:
            # Generate search terms
            print("  [1/3] Generating search terms...")
            terms = generate_search_terms(name, role)
            terms = terms[:self.config.max_searches]
            print(f"        {len(terms)} terms")

            # Search
            print("  [2/3] Searching...")
            def progress(done, total, sources):
                print(f"        {done}/{total} queries -> {sources} sources", end="\r")

            search_results = await searcher.batch_search(terms, progress)
            print(f"\n        {len(search_results)} unique sources")

            # Format research
            research = self._format_research(search_results)

            # Synthesize
            print("  [3/3] Synthesizing...")
            persona, model_used = await self.synthesizer.synthesize(name, role, category, research)

            # Add metadata
            gen_time = datetime.now(timezone.utc)
            persona["_metadata"] = {
                "generated_at": gen_time.isoformat(),
                "model": model_used,
                "category": category,
                "role": role,
                "search_terms": len(terms),
                "sources": len(search_results),
            }

            # Save with dated filename
            filename = get_dated_filename(name, gen_time)
            file_path = self.config.output_dir / filename
            self.config.output_dir.mkdir(parents=True, exist_ok=True)

            with open(file_path, "w") as f:
                json.dump(persona, f, indent=2, ensure_ascii=False)

            result["status"] = "done"
            result["file_path"] = str(file_path)
            result["file_size"] = file_path.stat().st_size
            result["model"] = model_used

            self.checkpoint.completed.append(name)
            print(f"  [DONE] {filename} ({result['file_size']:,} bytes, {model_used})")

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            self.checkpoint.failed.append(name)
            print(f"  [FAILED] {e}")

        self.checkpoint.results[name] = result
        self.checkpoint.save(self.config.checkpoint_file)

        return result

    def _format_research(self, results: list[dict]) -> str:
        """Format search results into research text."""
        text = ""
        for i, r in enumerate(results[:80]):  # Max 80 sources
            title = r.get("title", "")
            url = r.get("url", "")
            excerpts = " ".join(r.get("excerpts", []))[:600]
            text += f"\n[{i+1}] {title}\n{url}\n{excerpts}\n"
        return text

    def _print_header(self, people: list[dict]):
        print("=" * 60)
        print("PERSONA PIPELINE V2 - Date-stamped, Non-destructive")
        print("=" * 60)
        print(f"People:      {len(people)}")
        print(f"Searches:    up to {self.config.max_searches}/person")
        print(f"Min size:    {self.config.min_good_size:,} bytes (skip if larger exists)")
        print(f"Models:      {', '.join(PRIMARY_MODELS[:2])} + fallbacks")
        print(f"Output:      {self.config.output_dir}")
        print("=" * 60)

    def _print_summary(self, results: list[dict]):
        done = sum(1 for r in results if r.get("status") == "done")
        skipped = sum(1 for r in results if r.get("status") == "skipped")
        failed = sum(1 for r in results if r.get("status") == "failed")

        print(f"\n{'='*60}")
        print("COMPLETE")
        print("=" * 60)
        print(f"Generated: {done}")
        print(f"Skipped:   {skipped} (good version exists)")
        print(f"Failed:    {failed}")
        print(f"\nModel stats: {self.caller.stats}")


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    from people import get_all_people, PEOPLE

    parser = argparse.ArgumentParser(description="Persona Pipeline V2")
    parser.add_argument("-p", "--person", type=str, help="Generate for specific person")
    parser.add_argument("-c", "--category", type=str, help="Generate for category")
    parser.add_argument("-n", "--concurrency", type=int, default=2, help="Max concurrent")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoint")
    parser.add_argument("-l", "--list", action="store_true", help="List all people")
    parser.add_argument("--min-size", type=int, default=12000, help="Min file size to skip")
    args = parser.parse_args()

    if args.list:
        for cid, cat in PEOPLE.items():
            print(f"\n{cat['name']} [{cid}]:")
            for p in cat["people"]:
                print(f"  {p['name']:30} - {p['role']}")
        return

    # Get people
    if args.person:
        people = [p for p in get_all_people() if args.person.lower() in p["name"].lower()]
    elif args.category:
        cat = PEOPLE.get(args.category)
        if not cat:
            print(f"Categories: {list(PEOPLE.keys())}")
            return
        people = [{**p, "category_name": cat["name"]} for p in cat["people"]]
    else:
        people = get_all_people()

    if not people:
        print("No matching people found")
        return

    config = PipelineConfig(
        max_concurrent=args.concurrency,
        min_good_size=args.min_size,
    )

    pipeline = PersonaPipelineV2(config)
    asyncio.run(pipeline.run(people, resume=not args.no_resume))


if __name__ == "__main__":
    main()
