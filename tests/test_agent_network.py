#!/usr/bin/env python3
"""Test multi-agent conversations with Redis pub/sub and temporal graph."""

import sys
import asyncio
import logging
import warnings

# Suppress warnings
logging.getLogger("root").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

sys.path.insert(0, "/Users/arthurcolle/lida-multiagents-research")

from src.meta import (
    AgentNetwork,
    AgentRole,
    get_temporal_graph,
    reset_temporal_graph,
)


async def test_basic_conversation():
    """Test basic two-agent conversation."""
    print("\n" + "="*70)
    print("TEST 1: Basic Two-Agent Conversation (INTJ vs ENFP)")
    print("="*70 + "\n")

    network = AgentNetwork()

    # Spawn agents with personalities
    intj = network.spawn_agent("mbti_intj", agent_id="INTJ")
    enfp = network.spawn_agent("mbti_enfp", agent_id="ENFP")

    print(f"  Spawned: {intj.personality.name} ({intj.personality.archetype})")
    print(f"  Spawned: {enfp.personality.name} ({enfp.personality.archetype})\n")

    # Create conversation
    context = await network.create_conversation(
        topic="Is it better to plan everything or go with the flow?",
        agent_ids=["INTJ", "ENFP"],
    )

    print(f"  Topic: \"{context.topic}\"\n")
    print("-"*70)

    # Run conversation
    async for msg in network.run_conversation(context.conversation_id, turns=4):
        agent = network.agents[msg["agent_id"]]
        print(f"\n  [{agent.personality.name}]:")
        print(f"  {msg['content']}")

    print("\n" + "-"*70)

    # Show temporal graph stats
    temporal = get_temporal_graph()
    stats = temporal.get_stats()
    print(f"\n  Temporal Graph: {stats.get('nodes', 0)} nodes, {stats.get('edges', 0)} edges")

    await network.cleanup()


async def test_toxic_dynamic():
    """Test toxic relationship dynamic between personalities."""
    print("\n" + "="*70)
    print("TEST 2: Toxic Dynamic (Narcissist vs Anxious-Attached)")
    print("="*70 + "\n")

    network = AgentNetwork()

    narc = network.spawn_agent("dark_narcissist", agent_id="Narcissist")
    anxious = network.spawn_agent("attachment_anxious", agent_id="Anxious")

    print(f"  Spawned: {narc.personality.name} - {narc.personality.core_motivation}")
    print(f"  Spawned: {anxious.personality.name} - {anxious.personality.core_motivation}\n")

    context = await network.create_conversation(
        topic="You never seem to have time for me anymore...",
        agent_ids=["Narcissist", "Anxious"],
    )

    print(f"  Topic: \"{context.topic}\"\n")
    print("-"*70)

    async for msg in network.run_conversation(context.conversation_id, turns=5):
        agent = network.agents[msg["agent_id"]]
        print(f"\n  [{agent.personality.name}]:")
        print(f"  {msg['content']}")

    print("\n" + "-"*70)
    await network.cleanup()


async def test_parallel_conversations():
    """Test multiple conversations running in parallel."""
    print("\n" + "="*70)
    print("TEST 3: Parallel Conversations (3 concurrent dialogues)")
    print("="*70 + "\n")

    network = AgentNetwork()

    conversations = [
        {
            "personalities": ["mbti_intp", "mbti_esfj"],
            "topic": "Small talk is pointless vs. necessary for connection",
        },
        {
            "personalities": ["enneagram_8", "enneagram_2"],
            "topic": "What does real strength look like?",
        },
        {
            "personalities": ["dark_machiavellian", "attachment_secure"],
            "topic": "How do you build trust with someone new?",
        },
    ]

    print("  Running 3 conversations in parallel...\n")

    results = await network.run_parallel_conversations(conversations, turns=3)

    for i, msgs in enumerate(results):
        conv = conversations[i]
        print(f"\n  --- Conversation {i+1}: {conv['personalities'][0]} vs {conv['personalities'][1]} ---")
        print(f"  Topic: \"{conv['topic']}\"\n")
        for msg in msgs:
            print(f"    [{msg['agent_id'][:15]}]: {msg['content'][:80]}...")

    print("\n" + "-"*70)

    # Show temporal graph stats
    temporal = get_temporal_graph()
    stats = temporal.get_stats()
    print(f"\n  Temporal Graph: {stats.get('nodes', 0)} nodes")
    print(f"  Total agents spawned: {len(network.agents)}")

    await network.cleanup()


async def test_with_redis():
    """Test with Redis pub/sub if available."""
    print("\n" + "="*70)
    print("TEST 4: Redis Pub/Sub Integration")
    print("="*70 + "\n")

    network = AgentNetwork(redis_url="redis://localhost:6379")
    connected = await network.connect()

    if not connected:
        print("  Redis not available - skipping pub/sub test")
        print("  (Start Redis with: docker run -d -p 6379:6379 redis)")
        return

    print("  Connected to Redis!\n")

    # Spawn agents
    agent1 = network.spawn_agent("mbti_entj", agent_id="Commander")
    agent2 = network.spawn_agent("mbti_infp", agent_id="Mediator")

    # Connect agents to Redis
    await agent1.connect_redis(network.redis_url)
    await agent2.connect_redis(network.redis_url)

    print(f"  Agents connected to Redis pub/sub")

    # Create and run conversation
    context = await network.create_conversation(
        topic="How should a team make difficult decisions?",
        agent_ids=["Commander", "Mediator"],
    )

    print(f"  Topic: \"{context.topic}\"\n")
    print("-"*70)

    async for msg in network.run_conversation(context.conversation_id, turns=4):
        agent = network.agents[msg["agent_id"]]
        print(f"\n  [{agent.personality.name}]:")
        print(f"  {msg['content']}")

    print("\n" + "-"*70)
    await network.cleanup()


async def show_temporal_summary():
    """Show summary of temporal graph events."""
    print("\n" + "="*70)
    print("TEMPORAL GRAPH SUMMARY")
    print("="*70 + "\n")

    temporal = get_temporal_graph()
    stats = temporal.get_stats()

    print(f"  Nodes: {stats.get('nodes', 0)}")
    print(f"  Edges: {stats.get('edges', 0)}")

    # Query nodes
    nodes = temporal.query_nodes()
    print(f"\n  Node Types:")
    type_counts = {}
    for node in nodes:
        t = node.node_type
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, count in sorted(type_counts.items()):
        print(f"    {t}: {count}")

    print()


async def main():
    print("\n" + "🎭"*35)
    print("     MULTI-AGENT PERSONALITY NETWORK TEST")
    print("🎭"*35)

    # Reset temporal graph for clean test
    reset_temporal_graph()

    # Run tests (skip parallel to avoid GPU contention on Apple Silicon)
    await test_basic_conversation()
    await test_toxic_dynamic()
    # await test_parallel_conversations()  # Causes GPU contention
    await test_with_redis()
    await show_temporal_summary()

    print("\n✅ All tests complete!\n")


if __name__ == "__main__":
    asyncio.run(main())
