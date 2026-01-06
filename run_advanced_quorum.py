#!/usr/bin/env python3
"""
Advanced Multi-Agent Quorum with Sophisticated Function Calling

Features:
- ReAct (Reasoning + Acting) pattern for deliberate decision-making
- Parallel tool execution with dependency resolution
- Multi-agent collaboration with shared knowledge
- Chain-of-Thought reasoning traces (visible to user)
- Confidence scoring and uncertainty quantification
- Real API integrations where possible
- Meta-reasoning for intelligent tool selection
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

from src.meta import MLXClient, MLXModelConfig, PERSONALITY_ARCHETYPES
from src.meta.personality_tools import PersonalityToolRegistry
from src.meta.advanced_agent_framework import (
    ReActAgent,
    AgentCollaborationNetwork,
    AdvancedTool,
    LiveMarketDataTool,
    NewsAnalysisTool,
)
from src.meta.industrial_intelligence import (
    IndustrialEvent,
    IndustrialEventType,
    EmotionalStance,
)


# ANSI colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    MAGENTA = '\033[35m'


def print_reasoning_trace(agent_name: str, trace: List[Any]):
    """Pretty print the reasoning trace."""
    print(f"\n{Colors.CYAN}{'─'*60}")
    print(f" {agent_name} - Reasoning Trace")
    print(f"{'─'*60}{Colors.ENDC}\n")

    for i, step in enumerate(trace, 1):
        if step.step_type == "thought":
            icon = "💭"
            color = Colors.BLUE
        elif step.step_type == "action":
            icon = "🔧"
            color = Colors.MAGENTA
        elif step.step_type == "observation":
            icon = "👁"
            color = Colors.YELLOW
        elif step.step_type == "synthesis":
            icon = "✨"
            color = Colors.GREEN
        else:
            icon = "•"
            color = ""

        print(f"{color}{icon} {step.step_type.upper()}{Colors.ENDC}")

        # Truncate long content
        content = step.content
        if len(content) > 300:
            content = content[:300] + "..."

        for line in content.split('\n'):
            if line.strip():
                print(f"  {Colors.DIM}{line}{Colors.ENDC}")

        if step.tools_involved:
            print(f"  {Colors.MAGENTA}Tools: {', '.join(step.tools_involved)}{Colors.ENDC}")

        if step.confidence:
            print(f"  {Colors.DIM}Confidence: {step.confidence:.0%}{Colors.ENDC}")

        print()


def format_stance(stance: EmotionalStance) -> str:
    """Format stance with color."""
    colors = {
        EmotionalStance.BULLISH: Colors.GREEN,
        EmotionalStance.EXCITED: Colors.GREEN + Colors.BOLD,
        EmotionalStance.BEARISH: Colors.RED,
        EmotionalStance.ALARMED: Colors.RED + Colors.BOLD,
        EmotionalStance.CAUTIOUS: Colors.YELLOW,
        EmotionalStance.SKEPTICAL: Colors.YELLOW,
        EmotionalStance.NEUTRAL: Colors.DIM,
        EmotionalStance.CONTRARIAN: Colors.BLUE,
        EmotionalStance.OPPORTUNISTIC: Colors.CYAN,
        EmotionalStance.ANALYTICAL: Colors.BLUE,
    }
    color = colors.get(stance, "")
    return f"{color}{stance.value.upper()}{Colors.ENDC}"


async def create_advanced_agent(
    role_id: str,
    role_name: str,
    personality_key: str,
    tools: List[AdvancedTool]
) -> ReActAgent:
    """Create an advanced ReAct agent with personality."""
    # Create MLX client with personality
    config = MLXModelConfig(
        model_path="mlx-community/Josiefied-Qwen3-4B-abliterated-v1-4bit",
        max_tokens=2048,
        temperature=0.7
    )
    client = MLXClient(config)

    if personality_key in PERSONALITY_ARCHETYPES:
        client.personality = PERSONALITY_ARCHETYPES[personality_key]()

    # Create ReAct agent
    agent = ReActAgent(
        name=role_name,
        llm_client=client,
        tools=tools,
        max_iterations=3,  # Limit iterations for demo
        enable_parallel=True
    )

    return agent


async def run_advanced_quorum(
    headline: str,
    event_type: IndustrialEventType,
    value_billions: float = 0.0
):
    """Run advanced quorum with ReAct agents and collaboration."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}")
    print(" ADVANCED MULTI-AGENT QUORUM")
    print(" ReAct Pattern + Parallel Execution + Agent Collaboration")
    print(f"{'='*70}{Colors.ENDC}\n")

    print(f"Model: Qwen3-4B with ReAct reasoning")
    print(f"Framework: Parallel tool execution + shared knowledge")
    print(f"Pattern: Thought → Action → Observation → Synthesis")
    print()

    # Create event
    event = IndustrialEvent(
        event_id="advanced_1",
        event_type=event_type,
        timestamp=datetime.utcnow(),
        primary_company=headline.split()[0],
        title=headline,
        value_billions=value_billions,
    )

    print(f"{Colors.CYAN}{'─'*50}")
    print(f" EVENT: {headline}")
    print(f"{'─'*50}{Colors.ENDC}\n")

    # Create collaboration network
    network = AgentCollaborationNetwork()

    # Create advanced agents with personality-specific tools
    print(f"{Colors.DIM}Creating advanced agents with ReAct reasoning...{Colors.ENDC}\n")

    # Agent 1: Market Analyst (INTJ) with market data tools
    market_tools = PersonalityToolRegistry.get_tools_for_role("market_analyst")
    market_tools.append(LiveMarketDataTool())
    market_tools.append(NewsAnalysisTool())

    # Convert to AdvancedTool (wrap existing tools)
    from src.meta.advanced_agent_framework import AdvancedTool as AT

    class WrappedTool(AT):
        def __init__(self, base_tool):
            super().__init__()
            self.base_tool = base_tool
            self.name = base_tool.name
            self.description = base_tool.description

        async def execute(self, params: Dict[str, Any]) -> Any:
            return await self.base_tool.call(params)

    advanced_market_tools = [WrappedTool(t) if not isinstance(t, AT) else t for t in market_tools]

    market_analyst = await create_advanced_agent(
        "market_analyst",
        "Market Analyst",
        "mbti_intj",
        advanced_market_tools
    )
    network.register_agent(market_analyst)

    # Agent 2: Risk Manager (Enneagram 6) with risk tools
    risk_tools = PersonalityToolRegistry.get_tools_for_role("risk_manager")
    advanced_risk_tools = [WrappedTool(t) for t in risk_tools]

    risk_manager = await create_advanced_agent(
        "risk_manager",
        "Risk Manager",
        "enneagram_6",
        advanced_risk_tools
    )
    network.register_agent(risk_manager)

    # Agent 3: Opportunity Scout (ENFP) with partnership tools
    opp_tools = PersonalityToolRegistry.get_tools_for_role("opportunity_scout")
    advanced_opp_tools = [WrappedTool(t) for t in opp_tools]

    opportunity_scout = await create_advanced_agent(
        "opportunity_scout",
        "Opportunity Scout",
        "mbti_enfp",
        advanced_opp_tools
    )
    network.register_agent(opportunity_scout)

    print(f"{Colors.GREEN}✓ Created 3 advanced agents with {len(advanced_market_tools) + len(advanced_risk_tools) + len(advanced_opp_tools)} total tools{Colors.ENDC}\n")

    # Run collaborative analysis
    print(f"{Colors.BOLD}Phase 1: Parallel ReAct Analysis{Colors.ENDC}\n")
    print(f"{Colors.DIM}Each agent will:"){Colors.ENDC}")
    print(f"{Colors.DIM}  1. THINK about what information they need{Colors.ENDC}")
    print(f"{Colors.DIM}  2. ACT by calling relevant tools (in parallel){Colors.ENDC}")
    print(f"{Colors.DIM}  3. OBSERVE the results and extract insights{Colors.ENDC}")
    print(f"{Colors.DIM}  4. SYNTHESIZE into final recommendation{Colors.ENDC}\n")

    task = f"""Analyze this corporate event and provide your expert opinion:

Event: {event.title}
Company: {event.primary_company}
Type: {event.event_type.value}
Value: ${event.value_billions}B

Use your available tools to gather data, then provide a detailed analysis with your stance (bullish/bearish/cautious/etc.) and confidence level."""

    # Collaborative solve
    result = await network.collaborative_solve(task)

    # Display individual reasoning traces
    for agent_name in result["agents_involved"]:
        agent = network.agents[agent_name]
        if agent.reasoning_trace:
            print_reasoning_trace(agent_name, agent.reasoning_trace)

    # Display tool usage
    print(f"{Colors.CYAN}{'─'*60}")
    print(f" TOOL EXECUTION SUMMARY")
    print(f"{'─'*60}{Colors.ENDC}\n")

    tools_by_agent = {}
    for tool_result in result["tool_results"]:
        agent_tools = tools_by_agent.setdefault(tool_result.tool_name, [])
        agent_tools.append(tool_result)

    total_tools_called = len(result["tool_results"])
    total_execution_time = sum(tr.execution_time_ms for tr in result["tool_results"])
    avg_confidence = sum(tr.confidence for tr in result["tool_results"]) / max(len(result["tool_results"]), 1)

    print(f"  Total tools called: {Colors.BOLD}{total_tools_called}{Colors.ENDC}")
    print(f"  Total execution time: {Colors.BOLD}{total_execution_time:.0f}ms{Colors.ENDC}")
    print(f"  Average tool confidence: {Colors.BOLD}{avg_confidence:.0%}{Colors.ENDC}\n")

    print(f"  {Colors.DIM}Tools used:{Colors.ENDC}")
    for tool_name, results in tools_by_agent.items():
        print(f"    • {tool_name}: {len(results)} calls")

    # Display individual opinions
    print(f"\n{Colors.CYAN}{'─'*60}")
    print(f" INDIVIDUAL AGENT OPINIONS")
    print(f"{'─'*60}{Colors.ENDC}\n")

    for agent_name, opinion in result["individual_opinions"].items():
        print(f"{Colors.BOLD}{agent_name}:{Colors.ENDC}")

        # Extract stance if present
        opinion_lower = opinion.lower()
        stance = None
        for s in EmotionalStance:
            if s.value in opinion_lower:
                stance = s
                break

        if stance:
            print(f"  Stance: {format_stance(stance)}")

        # Truncate long opinions
        opinion_text = opinion
        if len(opinion_text) > 300:
            opinion_text = opinion_text[:300] + "..."

        for line in opinion_text.split('\n'):
            if line.strip():
                print(f"  {Colors.DIM}{line}{Colors.ENDC}")
        print()

    # Display consensus
    print(f"{Colors.CYAN}{'─'*60}")
    print(f" COLLABORATIVE CONSENSUS")
    print(f"{'─'*60}{Colors.ENDC}\n")

    print(f"  Overall Confidence: {Colors.BOLD}{result['confidence']:.0%}{Colors.ENDC}")
    print(f"  Agents Involved: {Colors.BOLD}{len(result['agents_involved'])}{Colors.ENDC}")
    print()

    print(f"{Colors.BOLD}Consensus Analysis:{Colors.ENDC}")
    consensus_text = result["consensus"]
    if len(consensus_text) > 500:
        consensus_text = consensus_text[:500] + "..."

    for line in consensus_text.split('\n'):
        if line.strip():
            print(f"  {line}")

    print(f"\n{Colors.GREEN}{Colors.BOLD}✓ Advanced Quorum Complete{Colors.ENDC}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run advanced multi-agent quorum")
    parser.add_argument("--event", type=str, help="Event headline")
    parser.add_argument("--value", type=float, default=20.0, help="Deal value in billions")
    parser.add_argument("--demo", action="store_true", help="Run demo")

    args = parser.parse_args()

    headline = args.event or "Nvidia announces $20B acquisition of Cerebras Systems"
    event_type = IndustrialEventType.ACQUISITION_ANNOUNCED

    try:
        asyncio.run(run_advanced_quorum(headline, event_type, args.value))
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted.{Colors.ENDC}")
