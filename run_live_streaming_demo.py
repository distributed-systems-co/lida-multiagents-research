#!/usr/bin/env python3
"""
Live Streaming Demo: GDELT Feeds + Advanced Agents with Tools

Fetches real GDELT news feeds and analyzes them with streaming LLM responses,
showing the full ReAct reasoning process in real-time.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

from src.meta import MLXClient, MLXModelConfig, PERSONALITY_ARCHETYPES
from src.meta.personality_tools import (
    PersonalityToolRegistry,
    execute_tool_call,
)
from src.meta.industrial_intelligence import (
    IndustrialEvent,
    IndustrialEventType,
)
from run_live_quorum import GDELTLiveFeed, WATCHED_COMPANIES, EVENT_KEYWORDS


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
    UNDERLINE = '\033[4m'


async def stream_with_personality(
    client: MLXClient,
    prompt: str,
    agent_name: str,
    personality_key: str
) -> str:
    """Stream LLM response with personality, showing tokens in real-time."""
    # Set personality
    if personality_key in PERSONALITY_ARCHETYPES:
        client.personality = PERSONALITY_ARCHETYPES[personality_key]()

    print(f"{Colors.BOLD}{Colors.CYAN}[{agent_name}]{Colors.ENDC} ", end="", flush=True)

    full_response = ""
    async for token in client.generate_stream(prompt):
        print(token, end="", flush=True)
        full_response += token

    print()  # Newline
    return full_response


async def analyze_event_with_tools_streaming(
    event: IndustrialEvent,
    agent_name: str,
    personality_key: str,
    tools: List[Any]
):
    """Analyze an event using tools with streaming responses."""
    print(f"\n{Colors.CYAN}{'─'*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}Agent: {agent_name}{Colors.ENDC}")
    print(f"{Colors.DIM}Personality: {personality_key}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'─'*70}{Colors.ENDC}\n")

    # Create MLX client
    config = MLXModelConfig(
        model_path="mlx-community/Josiefied-Qwen3-4B-abliterated-v1-4bit",
        max_tokens=2048,
        temperature=0.7
    )
    client = MLXClient(config)

    # Build tool descriptions
    tool_descriptions = "\n".join([
        f"- {tool.name}: {tool.description}"
        for tool in tools
    ])

    # Phase 1: Initial analysis with streaming
    initial_prompt = f"""{agent_name} analyzing corporate event:

EVENT: {event.title}
Company: {event.primary_company}
Type: {event.event_type.value if event.event_type else 'general'}
{f'Value: ${event.value_billions}B' if event.value_billions else ''}

Available tools:
{tool_descriptions}

First, provide your initial thoughts on this event (2-3 sentences).
What stands out? What tools would help your analysis?"""

    print(f"{Colors.YELLOW}💭 INITIAL THOUGHT (streaming):{Colors.ENDC}\n")
    initial_thought = await stream_with_personality(
        client, initial_prompt, agent_name, personality_key
    )

    # Phase 2: Tool selection and execution
    print(f"\n{Colors.MAGENTA}🔧 TOOL SELECTION:{Colors.ENDC}\n")

    # Decide which tools to use
    tool_selection_prompt = f"""Based on this event and your initial thought:

{initial_thought}

Which 1-2 tools from your available set would be most valuable?
Available: {', '.join([t.name for t in tools])}

Respond with tool names only, comma-separated."""

    print(f"{Colors.DIM}Selecting tools...{Colors.ENDC}\n")
    tool_selection = await stream_with_personality(
        client, tool_selection_prompt, agent_name, personality_key
    )

    # Execute selected tools
    tool_results = []
    selected_tool_names = [name.strip() for name in tool_selection.split(',') if name.strip()]

    for tool_name in selected_tool_names[:2]:  # Max 2 tools
        # Find matching tool
        matching_tool = None
        for tool in tools:
            if tool.name in tool_name.lower() or tool_name.lower() in tool.name:
                matching_tool = tool
                break

        if matching_tool:
            print(f"\n{Colors.MAGENTA}⚙ Executing: {matching_tool.name}{Colors.ENDC}")

            # Generate parameters for the tool
            param_prompt = f"""Generate parameters for the {matching_tool.name} tool to analyze {event.primary_company}.

Tool description: {matching_tool.description}
Required params: {', '.join([p.name for p in matching_tool.parameters if p.required])}

Return JSON only, e.g.: {{"company": "Nvidia", "metrics": ["price", "market_cap"]}}"""

            # Get params (non-streaming for JSON)
            param_result = await client.generate(param_prompt, max_tokens=500)

            # Parse params
            import re, json
            json_match = re.search(r'\{[^}]+\}', param_result.text)
            params = {}
            if json_match:
                try:
                    params = json.loads(json_match.group(0))
                except:
                    params = {"company": event.primary_company}
            else:
                params = {"company": event.primary_company}

            print(f"{Colors.DIM}  Parameters: {params}{Colors.ENDC}")

            # Execute tool
            try:
                result = await execute_tool_call(matching_tool.name, params)
                tool_results.append({
                    "tool": matching_tool.name,
                    "result": result
                })
                print(f"{Colors.GREEN}  ✓ Result received{Colors.ENDC}")
            except Exception as e:
                print(f"{Colors.RED}  ✗ Error: {e}{Colors.ENDC}")

    # Phase 3: Synthesis with streaming
    print(f"\n{Colors.YELLOW}✨ FINAL SYNTHESIS (streaming):{Colors.ENDC}\n")

    tool_results_summary = "\n".join([
        f"- {r['tool']}: {str(r['result'])[:200]}..."
        for r in tool_results
    ])

    synthesis_prompt = f"""Based on your analysis of {event.title}:

Initial thoughts:
{initial_thought}

Tool results:
{tool_results_summary}

Provide your final assessment:
1. Your stance (bullish/bearish/cautious/neutral)
2. Key insights from the data
3. Your recommendation

Be concise (3-4 sentences) and data-driven."""

    final_analysis = await stream_with_personality(
        client, synthesis_prompt, agent_name, personality_key
    )

    print(f"\n{Colors.GREEN}✓ Analysis complete{Colors.ENDC}")

    return {
        "agent": agent_name,
        "thought": initial_thought,
        "tools_used": [r["tool"] for r in tool_results],
        "synthesis": final_analysis
    }


async def run_live_streaming_demo():
    """Main demo: Live GDELT feeds + streaming agent analysis."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}")
    print(" LIVE STREAMING DEMO")
    print(" Real GDELT Feeds + Streaming LLM Analysis with Tools")
    print(f"{'='*70}{Colors.ENDC}\n")

    print(f"{Colors.CYAN}System Status:{Colors.ENDC}")
    print(f"  • Model: {Colors.BOLD}Qwen3-4B (MLX optimized){Colors.ENDC}")
    print(f"  • Mode: {Colors.BOLD}Streaming responses{Colors.ENDC}")
    print(f"  • Tools: {Colors.BOLD}16 specialized functions{Colors.ENDC}")
    print(f"  • Data: {Colors.BOLD}Live GDELT feeds{Colors.ENDC}")
    print()

    # Phase 1: Fetch live GDELT data
    print(f"{Colors.BOLD}Phase 1: Fetching Live News Feeds{Colors.ENDC}\n")
    print(f"{Colors.DIM}Connecting to GDELT...{Colors.ENDC}")

    gdelt = GDELTLiveFeed()
    events = await gdelt.fetch_latest_events(limit=500)

    if events:
        relevant = gdelt.filter_relevant_events(events)
        print(f"{Colors.GREEN}✓ Fetched {len(events)} events, {len(relevant)} relevant{Colors.ENDC}\n")

        if not relevant:
            print(f"{Colors.YELLOW}No events match watched companies. Using sample...{Colors.ENDC}\n")
            # Create sample event
            event = IndustrialEvent(
                event_id="sample_1",
                event_type=IndustrialEventType.ACQUISITION_ANNOUNCED,
                timestamp=datetime.utcnow(),
                primary_company="Nvidia",
                title="Nvidia announces $20B acquisition of Cerebras Systems",
                value_billions=20.0
            )
            relevant = [{"matched_company": "Nvidia", "SOURCEURL": "sample"}]
            analysis_events = [event]
        else:
            # Convert GDELT events to IndustrialEvents
            analysis_events = []
            for gdelt_event in relevant[:3]:  # Top 3
                company = gdelt_event.get('matched_company', 'Unknown')

                # Classify event type
                url_lower = gdelt_event.get('SOURCEURL', '').lower()
                event_type = None
                for keyword, etype in EVENT_KEYWORDS.items():
                    if keyword in url_lower:
                        event_type = etype
                        break

                headline = f"{company}: {gdelt_event.get('Actor1Name', '')} - {gdelt_event.get('Actor2Name', '')}"

                event = IndustrialEvent(
                    event_id=gdelt_event['GLOBALEVENTID'],
                    event_type=event_type,
                    timestamp=datetime.utcnow(),
                    primary_company=company,
                    title=headline[:100],
                )
                analysis_events.append(event)
    else:
        print(f"{Colors.YELLOW}GDELT unavailable. Using sample event...{Colors.ENDC}\n")
        event = IndustrialEvent(
            event_id="sample_1",
            event_type=IndustrialEventType.ACQUISITION_ANNOUNCED,
            timestamp=datetime.utcnow(),
            primary_company="Nvidia",
            title="Nvidia announces $20B acquisition of Cerebras Systems",
            value_billions=20.0
        )
        analysis_events = [event]

    # Phase 2: Multi-agent streaming analysis
    print(f"\n{Colors.BOLD}Phase 2: Multi-Agent Streaming Analysis{Colors.ENDC}\n")

    # Select event to analyze
    event_to_analyze = analysis_events[0]

    print(f"{Colors.UNDERLINE}Event:{Colors.ENDC}")
    print(f"  {event_to_analyze.title}")
    print(f"  {Colors.DIM}Company: {event_to_analyze.primary_company}{Colors.ENDC}")
    if event_to_analyze.value_billions:
        print(f"  {Colors.DIM}Value: ${event_to_analyze.value_billions}B{Colors.ENDC}")
    print()

    # Configure agents
    agents_config = [
        {
            "name": "Market Analyst",
            "personality": "mbti_intj",
            "role": "market_analyst"
        },
        {
            "name": "Risk Manager",
            "personality": "enneagram_6",
            "role": "risk_manager"
        },
        {
            "name": "Opportunity Scout",
            "personality": "mbti_enfp",
            "role": "opportunity_scout"
        }
    ]

    # Run agents in parallel
    print(f"{Colors.BOLD}Running {len(agents_config)} agents with streaming responses...{Colors.ENDC}\n")

    tasks = []
    for agent_config in agents_config:
        tools = PersonalityToolRegistry.get_tools_for_role(agent_config["role"])
        task = analyze_event_with_tools_streaming(
            event_to_analyze,
            agent_config["name"],
            agent_config["personality"],
            tools
        )
        tasks.append(task)

    # Execute all agents in parallel
    results = await asyncio.gather(*tasks)

    # Phase 3: Consensus
    print(f"\n{Colors.BOLD}Phase 3: Consensus Synthesis{Colors.ENDC}\n")
    print(f"{Colors.CYAN}{'─'*70}{Colors.ENDC}\n")

    for result in results:
        print(f"{Colors.BOLD}{result['agent']}:{Colors.ENDC}")
        print(f"{Colors.DIM}  Tools used: {', '.join(result['tools_used']) if result['tools_used'] else 'none'}{Colors.ENDC}")

        # Show brief synthesis
        synthesis = result['synthesis']
        if len(synthesis) > 200:
            synthesis = synthesis[:200] + "..."
        print(f"  {synthesis}")
        print()

    # Summary
    print(f"{Colors.CYAN}{'─'*70}{Colors.ENDC}")
    print(f"{Colors.GREEN}{Colors.BOLD}✓ Live Streaming Demo Complete{Colors.ENDC}\n")

    print(f"{Colors.DIM}Summary:{Colors.ENDC}")
    print(f"  • Analyzed live GDELT event")
    print(f"  • {len(results)} agents with streaming responses")
    print(f"  • Total tools called: {sum(len(r['tools_used']) for r in results)}")
    print(f"  • Full reasoning traces visible")
    print()


if __name__ == "__main__":
    try:
        asyncio.run(run_live_streaming_demo())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Demo interrupted.{Colors.ENDC}")
