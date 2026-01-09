#!/usr/bin/env python3
"""
OpenRouter Multi-Agent Quorum Demo

Demonstrates:
- Multiple agents with different models (Opus 4.5, Grok-4, Sonnet 4, etc.)
- Personality-driven responses (Scholar, Pragmatist, Creative, Skeptic)
- Redis pub/sub messaging for coordination
- MCP tool integration (Jina for web search)
- Quorum deliberation on complex topics
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import List, Optional

# Rich console for pretty output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Install rich for better output: pip install rich")

from src.messaging import MessageBroker, BrokerConfig
from src.agents import OpenRouterAgent, OpenRouterAgentConfig, create_openrouter_agent
from src.llm.openrouter import MODELS

console = Console() if RICH_AVAILABLE else None


def print_header(text: str):
    if console:
        console.print(Panel(text, style="bold cyan"))
    else:
        print(f"\n{'='*60}\n{text}\n{'='*60}\n")


def print_agent(agent_id: str, model: str, personality: str, content: str):
    if console:
        title = f"[bold]{agent_id}[/bold] | {model} | {personality}"
        console.print(Panel(Markdown(content), title=title, border_style="blue"))
    else:
        print(f"\n[{agent_id}] ({model} / {personality}):")
        print(content)
        print("-" * 40)


# Agent configurations for the quorum
QUORUM_AGENTS = [
    {
        "agent_id": "opus-scholar",
        "model": "opus-4.5",
        "personality": "the_scholar",
        "description": "Deep analytical thinker using Claude Opus 4.5",
    },
    {
        "agent_id": "grok-pragmatist",
        "model": "grok-4",
        "personality": "the_pragmatist",
        "description": "Results-oriented analyst using Grok-4",
    },
    {
        "agent_id": "sonnet-creative",
        "model": "sonnet-4",
        "personality": "the_creative",
        "description": "Creative explorer using Claude Sonnet 4",
    },
    {
        "agent_id": "deepseek-skeptic",
        "model": "deepseek-r1",
        "personality": "the_skeptic",
        "description": "Critical reasoner using DeepSeek R1",
    },
]


async def run_simple_demo():
    """Simple demo without Redis - just OpenRouter calls."""
    print_header("Simple OpenRouter Demo (No Redis)")

    from src.llm.openrouter import OpenRouterClient, MODELS

    client = OpenRouterClient()

    # Test a few models
    models_to_test = ["opus-4.5", "grok-4", "sonnet-4"]
    prompt = "In one paragraph, what's the most exciting development in AI this week?"

    for model in models_to_test:
        model_id = MODELS.get(model, model)
        print(f"\n[{model}] Generating...")

        try:
            response = await client.generate(
                prompt,
                system="You are a concise AI analyst. Keep responses under 100 words.",
                model=model_id,
                max_tokens=256,
            )
            print_agent(model, model_id, "default", response.content)
        except Exception as e:
            print(f"[{model}] Error: {e}")

    await client.close()


async def run_personality_demo():
    """Demo with personalities but without Redis."""
    print_header("Personality Demo (No Redis)")

    from src.llm.openrouter import OpenRouterClient, MODELS
    from src.meta.personality import get_personality_manager, PERSONALITY_ARCHETYPES

    client = OpenRouterClient()
    pm = get_personality_manager()

    # Create personalities
    personalities = ["the_scholar", "the_pragmatist", "the_creative", "the_skeptic"]

    topic = "Should AI systems be required to explain their reasoning to users?"

    for arch in personalities:
        personality = pm.create(name=f"test-{arch}", archetype=arch)
        system_prompt = personality.generate_system_prompt()

        print(f"\n[{arch}] Generating with {personality.voice.primary_tone.value} tone...")

        try:
            response = await client.generate(
                f"Topic: {topic}\n\nShare your perspective in 2-3 sentences.",
                system=system_prompt,
                model=MODELS["sonnet-4"],
                max_tokens=256,
            )
            print_agent(
                arch,
                "sonnet-4",
                personality.voice.primary_tone.value,
                response.content
            )
        except Exception as e:
            print(f"[{arch}] Error: {e}")

    await client.close()


async def run_mcp_demo():
    """Demo with MCP tools."""
    print_header("MCP Tools Demo")

    from src.llm.mcp_client import MCPClient, MCPToolExecutor
    from src.llm.openrouter import OpenRouterClient, MODELS

    # Check for Jina API key
    if not os.getenv("JINA_API_KEY"):
        print("Set JINA_API_KEY to test MCP tools")
        return

    client = OpenRouterClient()
    jina = MCPClient.jina()

    print("Connecting to Jina MCP...")
    if await jina.connect():
        print(f"Connected! {len(jina.tools)} tools available:")
        for tool in jina.tools[:5]:
            print(f"  - {tool.name}: {tool.description[:60]}...")

        # Execute a search
        print("\nSearching for 'Grok-4 release date'...")
        result = await jina.call_tool("search_web", {"query": "Grok-4 xAI release 2025"})

        if result.success:
            print(f"Search completed in {result.duration_ms:.0f}ms")
            # Summarize with LLM
            response = await client.generate(
                f"Summarize these search results in 2-3 sentences:\n\n{result.result}",
                model=MODELS["sonnet-4"],
                max_tokens=256,
            )
            print_agent("summarizer", "sonnet-4", "default", response.content)
        else:
            print(f"Search failed: {result.error}")

        await jina.close()
    else:
        print("Failed to connect to Jina MCP")

    await client.close()


async def run_redis_quorum():
    """Full quorum demo with Redis pub/sub."""
    print_header("Redis Pub/Sub Quorum Demo")

    # Check Redis
    try:
        import redis.asyncio as redis_lib
        r = redis_lib.from_url("redis://localhost:6379")
        await r.ping()
        await r.close()
    except Exception as e:
        print(f"Redis not available: {e}")
        print("Start Redis with: docker-compose up -d redis")
        return

    # Create broker
    broker = MessageBroker(BrokerConfig(redis_url="redis://localhost:6379"))
    await broker.connect()
    await broker.start()

    agents: List[OpenRouterAgent] = []

    try:
        # Spawn agents
        print("\nSpawning agents...")
        for cfg in QUORUM_AGENTS:
            print(f"  Creating {cfg['agent_id']} ({cfg['model']} / {cfg['personality']})...")

            agent = await create_openrouter_agent(
                broker=broker,
                agent_id=cfg["agent_id"],
                model=cfg["model"],
                personality=cfg["personality"],
                mcp_servers=[],  # Add ["jina"] if you have JINA_API_KEY
            )
            agents.append(agent)

        print(f"\n{len(agents)} agents ready!")

        # Quorum topic
        topic = """
        The EU has proposed mandatory "AI explanation requirements" that would force
        all AI systems to provide human-understandable explanations for their outputs.
        This includes chatbots, recommendation systems, and automated decision-making.

        What are the implications of this proposal?
        """

        print_header("Quorum Deliberation")
        print(f"Topic: {topic.strip()}\n")

        # Gather opinions
        opinions = []
        for agent in agents:
            cfg = next(c for c in QUORUM_AGENTS if c["agent_id"] == agent.agent_id)

            print(f"[{agent.agent_id}] Deliberating...")

            response = await agent.generate(
                f"""Topic for deliberation:

{topic}

Provide your analysis in 2-3 paragraphs. Consider:
1. Technical feasibility
2. Economic impact
3. Societal implications
4. Your recommendation
""",
                model=cfg["model"],
            )

            opinions.append({
                "agent_id": agent.agent_id,
                "model": cfg["model"],
                "personality": cfg["personality"],
                "opinion": response,
            })

            print_agent(
                agent.agent_id,
                cfg["model"],
                agent.get_personality().archetype if agent.get_personality() else "default",
                response,
            )

        # Synthesize
        print_header("Synthesis")

        synthesis_prompt = f"""
You are synthesizing opinions from a multi-agent quorum on the following topic:

{topic}

The agents provided these perspectives:

"""
        for op in opinions:
            synthesis_prompt += f"\n**{op['agent_id']}** ({op['personality']}):\n{op['opinion']}\n"

        synthesis_prompt += """

Synthesize these perspectives into a coherent summary that:
1. Identifies points of agreement
2. Notes key disagreements
3. Provides a balanced recommendation

Keep it under 300 words.
"""

        # Use opus for synthesis
        synth_agent = agents[0]  # opus-scholar
        synthesis = await synth_agent.generate(synthesis_prompt)

        print_agent("synthesis", "opus-4.5", "synthesizer", synthesis)

    finally:
        # Cleanup
        print("\nShutting down agents...")
        for agent in agents:
            await agent.stop()

        await broker.stop()
        await broker.disconnect()


async def main():
    """Run demos based on available services."""
    print_header("OpenRouter Multi-Agent System")

    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: Set OPENROUTER_API_KEY environment variable")
        print("Get one at: https://openrouter.ai/keys")
        return

    print("Available models:")
    for alias, model_id in list(MODELS.items())[:15]:
        print(f"  {alias:20} -> {model_id}")

    # Menu
    print("\nSelect demo:")
    print("  1. Simple (just OpenRouter calls)")
    print("  2. Personality (with personality system)")
    print("  3. MCP Tools (with Jina search)")
    print("  4. Redis Quorum (full multi-agent)")
    print("  5. All demos")

    choice = input("\nChoice [1-5]: ").strip() or "1"

    if choice == "1":
        await run_simple_demo()
    elif choice == "2":
        await run_personality_demo()
    elif choice == "3":
        await run_mcp_demo()
    elif choice == "4":
        await run_redis_quorum()
    elif choice == "5":
        await run_simple_demo()
        await run_personality_demo()
        await run_mcp_demo()
        await run_redis_quorum()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    asyncio.run(main())
