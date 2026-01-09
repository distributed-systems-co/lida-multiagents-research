#!/usr/bin/env python3
"""
Multi-Agent Deliberation with Proper Mechanism Design

Demonstrates:
- Quadratic Voting: Express conviction intensity (cost = votes²)
- Prediction Markets: Trade on outcome probabilities (LMSR)
- Conviction Staking: Stake reputation, earn/lose based on accuracy
- Bayesian Updates: Multi-round belief convergence
"""

import asyncio
import random
from typing import Dict, List, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from src.llm import get_lm, wrap_with_identity
from src.prompts import get_loader
from src.deliberation import (
    Position,
    QuadraticVoting,
    PredictionMarket,
    ConvictionStaking,
    IterativeDeliberation,
    MechanismType,
    create_mechanism,
)

console = Console()


async def get_agent_conviction(
    agent,
    positions: List[Position],
    topic: str
) -> Dict[str, Tuple[float, str]]:
    """
    Ask agent to assign conviction scores to each position.

    Returns: {position_id: (conviction_0_to_100, reasoning)}
    """
    position_desc = "\n".join([
        f"{i+1}. [{p.id}] {p.name}: {p.description}"
        for i, p in enumerate(positions)
    ])

    prompt = f'''Topic: {topic}

Available positions:
{position_desc}

You have 100 conviction points to allocate across these positions based on how strongly you believe in each.
- Points reflect how much you'd stake your reputation on each position
- You can put all 100 on one position if you're certain
- Or spread them if you see merit in multiple views

Respond in this exact format (numbers must sum to ~100):
POSITION_ID: POINTS | reasoning

Example:
safety_first: 60 | Strong evidence that rushing capabilities has caused alignment issues
balanced: 30 | Some valid arguments for parallel development
capability_first: 10 | Only minimal support, risks seem too high'''

    response = await agent(prompt, max_tokens=300)

    # Parse response
    convictions = {}
    for line in response.text.strip().split("\n"):
        if ":" not in line or "|" not in line:
            continue
        try:
            pos_part, reasoning = line.split("|", 1)
            pos_id, points = pos_part.rsplit(":", 1)
            pos_id = pos_id.strip().lower().replace(" ", "_")
            points = float(points.strip())
            convictions[pos_id] = (points, reasoning.strip())
        except:
            continue

    return convictions


async def get_agent_probability(
    agent,
    positions: List[Position],
    topic: str
) -> Dict[str, Tuple[float, str]]:
    """
    Ask agent to assign probabilities to each position being correct.

    Returns: {position_id: (probability_0_to_1, reasoning)}
    """
    position_desc = "\n".join([
        f"- [{p.id}] {p.name}: {p.description}"
        for p in positions
    ])

    prompt = f'''Topic: {topic}

Positions under consideration:
{position_desc}

As an expert, estimate the probability (0.0 to 1.0) that each position represents the optimal approach.
Your probabilities should roughly sum to 1.0 (they're mutually exclusive outcomes).

Respond in this exact format:
POSITION_ID: PROBABILITY | reasoning

Example:
safety_first: 0.45 | Given current evidence on alignment difficulties
balanced: 0.40 | Strong arguments for integrated approach
capability_first: 0.15 | Some valid competition concerns'''

    response = await agent(prompt, max_tokens=300)

    # Parse response
    probs = {}
    for line in response.text.strip().split("\n"):
        if ":" not in line or "|" not in line:
            continue
        try:
            pos_part, reasoning = line.split("|", 1)
            pos_id, prob = pos_part.rsplit(":", 1)
            pos_id = pos_id.strip().lower().replace(" ", "_")
            prob = float(prob.strip())
            probs[pos_id] = (min(1.0, max(0.0, prob)), reasoning.strip())
        except:
            continue

    return probs


async def run_quadratic_voting(agents, positions, topic):
    """Run quadratic voting deliberation."""
    console.print("\n[bold cyan]═══ QUADRATIC VOTING ═══[/]")
    console.print("Cost = votes². Express conviction intensity.\n")

    qv = QuadraticVoting(positions, voice_credits=100.0)

    for agent in agents:
        qv.register_agent(agent.identity.name)

    # Get convictions from each agent
    for agent in agents:
        convictions = await get_agent_conviction(agent, positions, topic)

        console.print(f"[bold]{agent.identity.name}[/] allocating votes...")

        for pos_id, (points, reasoning) in convictions.items():
            # Convert conviction points to votes (sqrt because cost is quadratic)
            votes = (points / 100) * 10  # Scale to reasonable vote range
            success, msg = qv.cast_votes(agent.identity.name, pos_id, votes)

            if success:
                console.print(f"  {pos_id}: {votes:.1f} votes ({points:.0f} conviction)")

    # Results
    result = qv.get_results()

    table = Table(title="Quadratic Voting Results", box=box.ROUNDED)
    table.add_column("Position")
    table.add_column("Total Votes")
    table.add_column("Score")

    raw_votes = result.metadata.get("raw_votes", {})
    for pos in positions:
        votes = raw_votes.get(pos.id, 0)
        score = result.position_scores.get(pos.id, 0)
        table.add_row(pos.name, f"{votes:.2f}", f"{score:.2%}")

    console.print(table)
    console.print(f"\n[bold green]Winner: {result.winning_position.name}[/]")
    console.print(f"Consensus strength: {result.consensus_strength:.1%}")

    return result


async def run_prediction_market(agents, positions, topic):
    """Run prediction market deliberation."""
    console.print("\n[bold magenta]═══ PREDICTION MARKET ═══[/]")
    console.print("LMSR automated market maker. Prices = collective beliefs.\n")

    market = PredictionMarket(positions, initial_liquidity=50.0)

    for agent in agents:
        market.register_agent(agent.identity.name, budget=100.0)

    # Multiple trading rounds
    for round_num in range(3):
        console.print(f"[dim]Round {round_num + 1}[/]")

        for agent in agents:
            probs = await get_agent_probability(agent, positions, topic)

            current_prices = market.get_prices()

            # Trade toward beliefs
            for pos_id, (belief_prob, reasoning) in probs.items():
                if pos_id not in current_prices:
                    continue

                current_price = current_prices[pos_id]
                edge = belief_prob - current_price

                # Trade if significant edge
                if abs(edge) > 0.1:
                    # Size position based on edge
                    shares = edge * 20  # Scale factor
                    success, msg, cost = market.trade(
                        agent.identity.name, pos_id, shares
                    )
                    if success and abs(cost) > 0.5:
                        direction = "BUY" if shares > 0 else "SELL"
                        console.print(f"  {agent.identity.name}: {direction} {abs(shares):.1f} {pos_id} @ {cost:.2f}")

    # Final prices
    result = market.get_results()

    table = Table(title="Market Final Prices", box=box.ROUNDED)
    table.add_column("Position")
    table.add_column("Price")
    table.add_column("Implied Probability")

    for pos in positions:
        price = result.position_scores.get(pos.id, 0)
        table.add_row(pos.name, f"${price:.2f}", f"{price:.1%}")

    console.print(table)
    console.print(f"\nTotal volume: ${result.metadata.get('total_volume', 0):.2f}")
    console.print(f"[bold green]Winner: {result.winning_position.name}[/]")
    console.print(f"Consensus strength: {result.consensus_strength:.1%}")

    return result


async def run_conviction_staking(agents, positions, topic):
    """Run conviction staking deliberation."""
    console.print("\n[bold yellow]═══ CONVICTION STAKING ═══[/]")
    console.print("Stake reputation on positions. Accuracy updates reputation.\n")

    staking = ConvictionStaking(positions)

    # Give agents different initial reputations based on "track record"
    for i, agent in enumerate(agents):
        rep = 0.8 + random.random() * 0.4  # 0.8-1.2
        staking.register_agent(agent.identity.name, initial_reputation=rep)
        console.print(f"{agent.identity.name}: initial reputation {rep:.2f}")

    console.print()

    # Get stakes from each agent
    for agent in agents:
        convictions = await get_agent_conviction(agent, positions, topic)

        console.print(f"[bold]{agent.identity.name}[/] staking...")

        for pos_id, (points, reasoning) in convictions.items():
            stake = points / 100  # Convert to 0-1
            success, msg = staking.stake(agent.identity.name, pos_id, stake)
            if success:
                console.print(f"  {pos_id}: {stake:.0%} stake")

    # Results
    result = staking.get_results()

    table = Table(title="Reputation-Weighted Scores", box=box.ROUNDED)
    table.add_column("Position")
    table.add_column("Weighted Score")

    for pos in positions:
        score = result.position_scores.get(pos.id, 0)
        table.add_row(pos.name, f"{score:.1%}")

    console.print(table)
    console.print(f"\n[bold green]Winner: {result.winning_position.name}[/]")
    console.print(f"Consensus strength: {result.consensus_strength:.1%}")

    return result


async def run_bayesian_deliberation(agents, positions, topic):
    """Run multi-round Bayesian belief update deliberation."""
    console.print("\n[bold blue]═══ ITERATIVE BAYESIAN ═══[/]")
    console.print("Multi-round belief updates until convergence.\n")

    delib = IterativeDeliberation(
        positions,
        convergence_threshold=0.05,
        max_rounds=3
    )

    # Initial beliefs
    for agent in agents:
        probs = await get_agent_probability(agent, positions, topic)

        initial = {}
        for pos_id, (prob, _) in probs.items():
            initial[pos_id] = prob

        # Fill missing with uniform
        for pos in positions:
            if pos.id not in initial:
                initial[pos.id] = 1.0 / len(positions)

        # Normalize
        total = sum(initial.values())
        if total > 0:
            initial = {k: v/total for k, v in initial.items()}

        delib.register_agent(agent.identity.name, initial)

    # Run rounds
    for round_num in range(delib.max_rounds):
        console.print(f"[bold]Round {round_num + 1}[/]")

        delib.record_round()

        # Check convergence
        if round_num > 0 and delib.check_convergence():
            console.print("[green]Converged![/]")
            break

        # Get current beliefs for display
        agg = delib.get_aggregate_beliefs()
        for pos_id, prob in sorted(agg.items(), key=lambda x: -x[1]):
            console.print(f"  {pos_id}: {prob:.1%}")

        # Simulate belief updates based on "hearing arguments"
        # In real use, you'd have agents argue and rate argument quality
        current_beliefs = {}
        argument_quality = {}

        for aid, agent_state in delib.agents.items():
            current_beliefs[aid] = {
                pid: b.probability for pid, b in agent_state.beliefs.items()
            }
            argument_quality[aid] = 0.3 + random.random() * 0.5

        # Update each agent's beliefs
        for aid in delib.agents:
            delib.bayesian_update(aid, current_beliefs, argument_quality)

        console.print()

    # Final results
    result = delib.get_results()

    table = Table(title="Final Aggregate Beliefs", box=box.ROUNDED)
    table.add_column("Position")
    table.add_column("Aggregate Probability")

    for pos in positions:
        prob = result.position_scores.get(pos.id, 0)
        table.add_row(pos.name, f"{prob:.1%}")

    console.print(table)
    console.print(f"\nConverged: {result.metadata.get('converged', False)}")
    console.print(f"[bold green]Winner: {result.winning_position.name}[/]")
    console.print(f"Consensus strength: {result.consensus_strength:.1%}")

    return result


async def main():
    console.print(Panel.fit(
        "[bold white]MECHANISM DESIGN FOR MULTI-AGENT DELIBERATION[/]",
        border_style="cyan",
        box=box.DOUBLE
    ))

    # Topic and positions
    topic = "Should AI development prioritize safety research over capability advancement?"

    positions = [
        Position("safety_first", "Safety First",
                "Prioritize alignment and safety research before advancing capabilities"),
        Position("capability_first", "Capability First",
                "Prioritize capability advancement; safety will follow naturally"),
        Position("balanced", "Balanced Approach",
                "Develop both in parallel with equal resources"),
    ]

    # Create diverse agents
    loader = get_loader()
    lm = get_lm("openrouter", model="anthropic/claude-sonnet-4")

    panel_configs = [
        "frontier_physics",
        "technology_computing",
        "government_public_service",
        "humanities",
    ]

    agents = []
    console.print("\n[bold]Creating Panel:[/]")

    for category in panel_configs:
        prompts = loader.get_by_category(category)
        if prompts:
            prompt = random.choice(prompts[:5])
            name = category.replace("_", " ").title().replace(" ", "")[:15]
            agent = wrap_with_identity(lm, prompt.text, name=name)
            agents.append(agent)
            console.print(f"  {name}: {prompt.text[:50]}...")

    # Run all mechanisms
    results = {}

    results["quadratic"] = await run_quadratic_voting(agents, positions, topic)
    results["market"] = await run_prediction_market(agents, positions, topic)
    results["staking"] = await run_conviction_staking(agents, positions, topic)
    results["bayesian"] = await run_bayesian_deliberation(agents, positions, topic)

    # Summary comparison
    console.print("\n[bold white]═══ MECHANISM COMPARISON ═══[/]")

    summary = Table(box=box.DOUBLE)
    summary.add_column("Mechanism")
    summary.add_column("Winner")
    summary.add_column("Consensus")
    summary.add_column("Characteristic")

    summary.add_row(
        "Quadratic Voting",
        results["quadratic"].winning_position.name,
        f"{results['quadratic'].consensus_strength:.0%}",
        "Intensity expression"
    )
    summary.add_row(
        "Prediction Market",
        results["market"].winning_position.name,
        f"{results['market'].consensus_strength:.0%}",
        "Price discovery"
    )
    summary.add_row(
        "Conviction Staking",
        results["staking"].winning_position.name,
        f"{results['staking'].consensus_strength:.0%}",
        "Reputation-weighted"
    )
    summary.add_row(
        "Bayesian Updates",
        results["bayesian"].winning_position.name,
        f"{results['bayesian'].consensus_strength:.0%}",
        "Belief convergence"
    )

    console.print(summary)

    # Meta-consensus
    winner_counts = {}
    for r in results.values():
        w = r.winning_position.name
        winner_counts[w] = winner_counts.get(w, 0) + 1

    meta_winner = max(winner_counts, key=winner_counts.get)
    console.print(f"\n[bold green]Meta-consensus across mechanisms: {meta_winner}[/]")


if __name__ == "__main__":
    asyncio.run(main())
