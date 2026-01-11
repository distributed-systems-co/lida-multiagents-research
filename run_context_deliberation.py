#!/usr/bin/env python3
"""
Multi-Agent Deliberation with Web Search Context

Demonstrates:
- Parallel web searches to establish baseline context
- Identity-wrapped agents with injected knowledge
- Structured deliberation with evidence-based arguments
"""

import asyncio
import random
import time
from datetime import datetime
from typing import List, Dict, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from src.llm import (
    get_lm,
    wrap_with_identity,
    ParallelContextSearch,
    BaselineContext,
    IdentityWrapper,
)
from src.prompts import get_loader

console = Console()


async def gather_context(topic: str, search_queries: List[str]) -> Dict[str, BaselineContext]:
    """Gather web context for the deliberation topic."""
    console.print("\n[bold cyan]PHASE 0: GATHERING WEB CONTEXT[/]")
    console.print("─" * 70)

    searcher = ParallelContextSearch()
    console.print(f"Providers: {[p.__class__.__name__ for p in searcher.providers]}")

    start = time.time()

    # Search all queries in parallel
    contexts = await searcher.search(search_queries, num_results_per_query=3)

    elapsed = time.time() - start
    console.print(f"Search completed in {elapsed:.2f}s")

    # Display results
    results_by_query = {}
    for query, ctx in zip(search_queries, contexts):
        results_by_query[query] = ctx
        console.print(f"\n[yellow]{query}[/]: {len(ctx.results)} results")
        for r in ctx.results[:2]:
            console.print(f"  • {r.title[:60]}...")

    return results_by_query


def create_panel(
    loader,
    lm,
    panel_configs: List[tuple],
    contexts: Dict[str, BaselineContext]
) -> List[IdentityWrapper]:
    """Create identity-wrapped agents with injected context."""
    agents = []

    console.print("\n[bold cyan]PANEL MEMBERS:[/]")

    table = Table(box=box.SIMPLE)
    table.add_column("Agent", style="bold")
    table.add_column("Expertise")
    table.add_column("Context Injected")

    for category, name, context_queries in panel_configs:
        # Get prompt for this category
        prompts = loader.get_by_category(category)
        if not prompts:
            prompts = loader.search(category.replace('_', ' '), limit=5)

        if prompts:
            prompt = random.choice(prompts[:5])
            agent = wrap_with_identity(lm, prompt.text, name=name)

            # Inject relevant context into agent's knowledge base
            context_items = 0
            for query in context_queries:
                if query in contexts:
                    ctx = contexts[query]
                    for result in ctx.results[:2]:
                        knowledge = f"[Source: {result.title}] {result.snippet}"
                        agent.add_to_knowledge_base(knowledge)
                        context_items += 1

            agents.append(agent)
            table.add_row(name, prompt.text[:40] + "...", f"{context_items} items")

    console.print(table)
    return agents


async def run_deliberation(topic: str, agents: List[IdentityWrapper]):
    """Run the deliberation phases."""

    # Phase 1: Opening Statements
    console.print("\n[bold green]═" * 70)
    console.print("[bold green]PHASE 1: OPENING STATEMENTS (with web context)[/]")
    console.print("[bold green]═" * 70)

    opening_prompt = f'''The topic is: "{topic}"

You have been provided with current research and news on this topic in your knowledge base.
Give your position in 2-3 sentences, citing specific facts or sources from your knowledge where relevant.
Be direct about whether you support Position A, Position B, or a nuanced approach.'''

    positions = {}
    for agent in agents:
        response = await agent(opening_prompt, max_tokens=500)
        positions[agent.identity.name] = response.text

        console.print(f"\n[bold blue][{agent.identity.name}][/]")
        console.print(Panel(response.text, border_style="blue"))

    await asyncio.sleep(0.5)

    # Phase 2: Evidence-Based Debate
    console.print("\n[bold yellow]═" * 70)
    console.print("[bold yellow]PHASE 2: EVIDENCE-BASED CROSS-EXAMINATION[/]")
    console.print("[bold yellow]═" * 70)

    for i in range(min(3, len(agents))):
        agent = agents[i]
        opponent = agents[(i + 1) % len(agents)]
        opponent_pos = positions[opponent.identity.name]

        debate_prompt = f'''{opponent.identity.name} argued: "{opponent_pos[:250]}..."

Challenge or support their position using evidence from your knowledge base.
Be specific about facts, data, or sources that inform your response.
2-3 sentences.'''

        response = await agent(debate_prompt, max_tokens=500)

        console.print(f"\n[bold magenta][{agent.identity.name}] → [{opponent.identity.name}][/]")
        console.print(Panel(response.text, border_style="magenta"))

    await asyncio.sleep(0.5)

    # Phase 3: Synthesis and Vote
    console.print("\n[bold red]═" * 70)
    console.print("[bold red]PHASE 3: FINAL SYNTHESIS & VOTE[/]")
    console.print("[bold red]═" * 70)

    all_positions = "\n".join([
        f"• {name}: {pos[:100]}..." for name, pos in positions.items()
    ])

    votes = {}
    reasoning = {}

    for agent in agents:
        vote_prompt = f'''After hearing all arguments and considering the evidence:

{all_positions}

Synthesize the key insights and cast your vote.
Format: [VOTE: A/B/BALANCED] followed by your one-sentence reasoning citing the most compelling evidence.'''

        response = await agent(vote_prompt, max_tokens=150)

        # Parse vote
        text = response.text.upper()
        if "BALANCED" in text or "BOTH" in text or "CONCURRENT" in text:
            vote = "BALANCED"
        elif "SAFETY" in text and ("FIRST" in text or "PRIORIT" in text):
            vote = "A"
        elif "CAPABILITY" in text and ("FIRST" in text or "PRIORIT" in text):
            vote = "B"
        elif "VOTE: A" in text or "VOTE:A" in text:
            vote = "A"
        elif "VOTE: B" in text or "VOTE:B" in text:
            vote = "B"
        else:
            vote = "BALANCED"

        votes[agent.identity.name] = vote
        reasoning[agent.identity.name] = response.text

        vote_color = {"A": "red", "B": "green", "BALANCED": "yellow"}[vote]
        console.print(f"\n[bold {vote_color}][{agent.identity.name}] → {vote}[/]")
        console.print(f"  {response.text[:200]}...")

    return votes, reasoning, positions


async def main():
    console.print(Panel.fit(
        "[bold white]MULTI-AGENT DELIBERATION WITH WEB CONTEXT[/]",
        border_style="cyan",
        box=box.DOUBLE
    ))

    # Configuration
    topic = "Should AI development prioritize safety research over capability advancement?"

    search_queries = [
        "AI safety research 2024 2025 progress",
        "AI capability advancement risks benefits",
        "AI alignment problem current solutions",
        "frontier AI labs safety commitments",
        "AI regulation policy 2025",
    ]

    panel_configs = [
        ("frontier_physics", "SafetyResearcher", ["AI safety research 2024 2025 progress", "AI alignment problem current solutions"]),
        ("technology_computing", "AIEngineer", ["AI capability advancement risks benefits", "frontier AI labs safety commitments"]),
        ("government_public_service", "PolicyMaker", ["AI regulation policy 2025", "frontier AI labs safety commitments"]),
        ("humanities", "Ethicist", ["AI alignment problem current solutions", "AI safety research 2024 2025 progress"]),
        ("economics", "Economist", ["AI capability advancement risks benefits", "AI regulation policy 2025"]),
    ]

    console.print(f"\n[bold]Topic:[/] {topic}\n")

    # Gather context
    contexts = await gather_context(topic, search_queries)

    # Create panel
    loader = get_loader()
    lm = get_lm("openrouter", model="anthropic/claude-sonnet-4")
    agents = create_panel(loader, lm, panel_configs, contexts)

    # Run deliberation
    votes, reasoning, positions = await run_deliberation(topic, agents)

    # Results
    console.print("\n[bold white]═" * 70)
    console.print("[bold white]DELIBERATION RESULTS[/]")
    console.print("[bold white]═" * 70)

    vote_counts = {"A": 0, "B": 0, "BALANCED": 0}
    for v in votes.values():
        vote_counts[v] += 1

    results_table = Table(box=box.ROUNDED)
    results_table.add_column("Position", style="bold")
    results_table.add_column("Votes")
    results_table.add_column("Interpretation")

    results_table.add_row("A (Safety First)", str(vote_counts["A"]), "Prioritize safety research")
    results_table.add_row("B (Capability First)", str(vote_counts["B"]), "Prioritize capability advancement")
    results_table.add_row("Balanced", str(vote_counts["BALANCED"]), "Both in parallel")

    console.print(results_table)

    winner = max(vote_counts, key=vote_counts.get)
    console.print(f"\n[bold green]Consensus: {winner}[/]")

    # Agent stats
    console.print("\n[bold]Agent Interaction Stats:[/]")
    for agent in agents:
        stats = agent.get_identity_summary()
        kb_preview = agent.identity.knowledge_base[0][:50] + "..." if agent.identity.knowledge_base else "none"
        console.print(f"  {stats['name']:20} | {stats['interaction_count']} turns | KB: {kb_preview}")


if __name__ == "__main__":
    asyncio.run(main())
