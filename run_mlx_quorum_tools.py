#!/usr/bin/env python3
"""
MLX-Powered Emotional Quorum with Function Calling

Enhanced version with personality-specific tools for each agent.
Each personality agent now has 2-4 specialized tools that make them more
lifelike and effective in their analysis.

Supports:
- Streaming personality responses
- Function calling during deliberation
- Tool execution with results
- Multi-turn tool-use conversations

Models tested:
- Qwen3 (best for function calling)
- Llama-3.3-Nemotron (good for agentic tasks)
- Mistral-Small (excellent tool use)
"""

import asyncio
import argparse
import sys
import re
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

# Suppress warnings
logging.getLogger("root").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

from src.meta import MLXClient, MLXModelConfig, PERSONALITY_ARCHETYPES
from src.meta.personality_tools import (
    PersonalityToolRegistry,
    execute_tool_call,
    BaseTool
)
from src.meta.industrial_intelligence import (
    IndustrialEvent,
    IndustrialEventType,
    IndustrialSector,
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


@dataclass
class AgentRole:
    """Enhanced agent role with tools."""
    role_id: str
    role_name: str
    personality_key: str
    system_prompt: str
    default_stance: EmotionalStance
    tools: List[BaseTool]  # Personality-specific tools


# Enhanced agent roles with tools
ENHANCED_AGENT_ROLES = [
    AgentRole(
        role_id="market_analyst",
        role_name="Market Analyst",
        personality_key="mbti_intj",
        system_prompt="""You are a senior market analyst with access to sophisticated analytical tools.
Use your tools to gather data, calculate valuations, analyze trends, and assess competitive dynamics.
Be strategic, data-driven, and thorough in your analysis.""",
        default_stance=EmotionalStance.ANALYTICAL,
        tools=PersonalityToolRegistry.get_tools_for_role("market_analyst")
    ),
    AgentRole(
        role_id="risk_manager",
        role_name="Risk Manager",
        personality_key="enneagram_6",
        system_prompt="""You are a chief risk officer with tools to assess risks, check compliance, and research failures.
Use your tools to identify threats, regulatory concerns, and potential failure modes.
Be cautious, thorough, and focused on protecting against downside risks.""",
        default_stance=EmotionalStance.CAUTIOUS,
        tools=PersonalityToolRegistry.get_tools_for_role("risk_manager")
    ),
    AgentRole(
        role_id="opportunity_scout",
        role_name="Opportunity Scout",
        personality_key="mbti_enfp",
        system_prompt="""You are a business development lead with tools to find partnerships, track innovations, and calculate synergies.
Use your tools to discover opportunities, identify emerging trends, and estimate value creation.
Be optimistic but grounded, enthusiastic about possibilities.""",
        default_stance=EmotionalStance.OPPORTUNISTIC,
        tools=PersonalityToolRegistry.get_tools_for_role("opportunity_scout")
    ),
    AgentRole(
        role_id="sector_specialist",
        role_name="Sector Specialist",
        personality_key="mbti_intp",
        system_prompt="""You are a sector expert with tools for technical analysis, patent research, and benchmarking.
Use your tools to dive deep into technology, assess IP strength, and compare against industry standards.
Be technically precise, thorough, and analytical.""",
        default_stance=EmotionalStance.ANALYTICAL,
        tools=PersonalityToolRegistry.get_tools_for_role("sector_specialist")
    ),
    AgentRole(
        role_id="contrarian",
        role_name="Devil's Advocate",
        personality_key="enneagram_8",
        system_prompt="""You are a contrarian analyst with tools to challenge consensus, test assumptions, and detect weaknesses.
Use your tools to find alternative perspectives, stress-test plans, and uncover hidden vulnerabilities.
Be provocative, challenging, but substantive in your counterarguments.""",
        default_stance=EmotionalStance.CONTRARIAN,
        tools=PersonalityToolRegistry.get_tools_for_role("contrarian")
    ),
]


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


def clean_response(text: str) -> str:
    """Clean up model output."""
    text = re.sub(r'<\|[^>]+\|>', '', text)
    text = re.sub(r'\|<\|[^>]+\|>', '', text)
    text = re.sub(r'</s>', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    # Get first substantial paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 20]
    if paragraphs:
        return '\n\n'.join(paragraphs[:2])[:800]  # First 2 paragraphs, max 800 chars
    return text[:800] if text else "(no response)"


def classify_stance(text: str) -> EmotionalStance:
    """Classify the stance from the response text."""
    text_lower = text.lower()

    # Check for explicit stance signals
    if any(w in text_lower for w in ["alarming", "dangerous", "threat", "serious risk", "very concerned"]):
        return EmotionalStance.ALARMED
    if any(w in text_lower for w in ["bullish", "optimistic", "strong buy", "very positive"]):
        return EmotionalStance.BULLISH
    if any(w in text_lower for w in ["exciting", "huge opportunity", "transformative", "game-changer"]):
        return EmotionalStance.EXCITED
    if any(w in text_lower for w in ["bearish", "pessimistic", "sell", "overvalued", "bubble"]):
        return EmotionalStance.BEARISH
    if any(w in text_lower for w in ["cautious", "careful", "wait and see", "uncertain"]):
        return EmotionalStance.CAUTIOUS
    if any(w in text_lower for w in ["skeptical", "doubt", "questionable", "unproven"]):
        return EmotionalStance.SKEPTICAL
    if any(w in text_lower for w in ["however", "but consider", "on the other hand", "disagree"]):
        return EmotionalStance.CONTRARIAN
    if any(w in text_lower for w in ["opportunity", "potential", "could benefit", "upside"]):
        return EmotionalStance.OPPORTUNISTIC

    return EmotionalStance.ANALYTICAL


@dataclass
class AgentOpinion:
    """An opinion from an agent with tool use."""
    agent_role: AgentRole
    raw_response: str
    cleaned_response: str
    stance: EmotionalStance
    confidence: float
    generation_time_ms: float
    tools_used: List[str]
    tool_results: Dict[str, Any]


class EnhancedMLXQuorum:
    """Emotional quorum with MLX models and function calling."""

    def __init__(
        self,
        model_path: str = "mlx-community/Qwen3-4B-Instruct-2507-gabliterated-4bit",  # Latest Qwen3 model
        max_tokens: int = 2048,  # Increased for function calling
        temperature: float = 0.7,
        max_tool_iterations: int = 2,  # Max rounds of tool calling
    ):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_tool_iterations = max_tool_iterations
        self._client: Optional[MLXClient] = None
        self._config: Optional[MLXModelConfig] = None

    def _get_client(self) -> MLXClient:
        """Lazy load the MLX client."""
        if self._client is None:
            self._config = MLXModelConfig(
                model_path=self.model_path,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            self._client = MLXClient(self._config)
        return self._client

    async def get_agent_opinion_with_tools(
        self,
        role: AgentRole,
        event: IndustrialEvent,
        context: str = "",
        stream: bool = True,
    ) -> AgentOpinion:
        """Get an opinion with tool use support."""
        client = self._get_client()

        # Set personality
        if role.personality_key in PERSONALITY_ARCHETYPES:
            client.personality = PERSONALITY_ARCHETYPES[role.personality_key]()

        # Build initial prompt with tool descriptions
        tool_descriptions = self._format_tools_for_prompt(role.tools)

        initial_prompt = f"""{role.system_prompt}

AVAILABLE TOOLS:
{tool_descriptions}

To use a tool, write: TOOL: tool_name
Then on next line write: PARAMS: {{"param": "value"}}

EVENT TO ANALYZE:
Company: {event.primary_company}
Event: {event.title}
Type: {event.event_type.value if event.event_type else 'general'}
{f'Value: ${event.value_billions}B' if event.value_billions else ''}

You may use 1-2 tools to gather data for your analysis. After using tools (or if you don't need them), provide your final assessment with your stance (bullish/bearish/cautious/etc) and reasoning."""

        # Multi-turn tool-use conversation
        tools_used = []
        tool_results = {}
        messages = [{"role": "user", "content": initial_prompt}]

        print(f"{Colors.BOLD}[{role.role_name}]{Colors.ENDC}")

        start_time = datetime.now()
        full_response = ""

        for iteration in range(self.max_tool_iterations):
            # Generate response
            if stream and iteration == self.max_tool_iterations - 1:
                # Stream final answer
                print(f"  {Colors.DIM}→{Colors.ENDC} ", end="", flush=True)
                iteration_response = ""
                async for token in client.generate_stream(messages[-1]["content"]):
                    print(token, end="", flush=True)
                    iteration_response += token
                print()  # Newline
            else:
                # Non-streaming for tool calls
                result = await client.generate(messages[-1]["content"])
                iteration_response = result.text

            full_response += iteration_response + "\n"

            # Check for tool calls
            tool_call = self._extract_tool_call(iteration_response)

            if tool_call:
                tool_name = tool_call.get("tool")
                tool_params = tool_call.get("params", {})

                print(f"  {Colors.MAGENTA}🔧 Using tool: {tool_name}{Colors.ENDC}")
                print(f"  {Colors.DIM}   Parameters: {json.dumps(tool_params, indent=None)[:100]}...{Colors.ENDC}")

                # Execute tool
                try:
                    tool_result = await execute_tool_call(tool_name, tool_params)
                    tools_used.append(tool_name)
                    tool_results[tool_name] = tool_result

                    # Format result for next turn
                    result_str = json.dumps(tool_result, indent=2)[:500]  # Limit size
                    print(f"  {Colors.GREEN}✓ Tool result received{Colors.ENDC}")

                    # Add tool result to conversation
                    messages.append({
                        "role": "assistant",
                        "content": iteration_response
                    })
                    messages.append({
                        "role": "user",
                        "content": f"Tool {tool_name} returned:\n{result_str}\n\nContinue your analysis. You can use more tools or provide your final assessment."
                    })

                except Exception as e:
                    print(f"  {Colors.RED}✗ Tool error: {e}{Colors.ENDC}")
                    break
            else:
                # No tool call, this is the final answer
                break

        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

        cleaned = clean_response(full_response)
        stance = classify_stance(cleaned)

        # Estimate confidence
        confidence = 0.7
        if any(w in cleaned.lower() for w in ["very", "strongly", "clearly", "definitely"]):
            confidence = 0.85
        elif any(w in cleaned.lower() for w in ["might", "perhaps", "possibly", "uncertain"]):
            confidence = 0.5

        # Tool use increases confidence slightly
        if tools_used:
            confidence = min(0.95, confidence + 0.05 * len(tools_used))

        return AgentOpinion(
            agent_role=role,
            raw_response=full_response,
            cleaned_response=cleaned,
            stance=stance,
            confidence=confidence,
            generation_time_ms=elapsed_ms,
            tools_used=tools_used,
            tool_results=tool_results
        )

    def _format_tools_for_prompt(self, tools: List[BaseTool]) -> str:
        """Format tools for system prompt."""
        lines = []
        for tool in tools:
            params_str = ", ".join([
                f"{p.name}({p.type})" for p in tool.parameters
            ])
            lines.append(f"- {tool.name}({params_str}): {tool.description}")
        return "\n".join(lines)

    def _extract_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract tool call from response."""
        # Look for TOOL: and PARAMS: format
        tool_match = re.search(r'TOOL:\s*(\w+)', text, re.IGNORECASE)
        if not tool_match:
            # Try JSON format as fallback
            json_pattern = r'\{[^}]*"tool"\s*:\s*"([^"]+)"[^}]*\}'
            match = re.search(json_pattern, text)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    pass
            return None

        tool_name = tool_match.group(1)

        # Look for params
        params_match = re.search(r'PARAMS:\s*(\{[^}]+\})', text, re.IGNORECASE)
        params = {}
        if params_match:
            try:
                params = json.loads(params_match.group(1))
            except:
                pass

        return {"tool": tool_name, "params": params}

    async def deliberate(
        self,
        event: IndustrialEvent,
        roles: Optional[List[AgentRole]] = None,
        context: str = "",
    ) -> List[AgentOpinion]:
        """Run full quorum deliberation with tools."""
        if roles is None:
            roles = ENHANCED_AGENT_ROLES

        opinions = []
        for role in roles:
            opinion = await self.get_agent_opinion_with_tools(role, event, context)
            opinions.append(opinion)
            print()  # Space between agents

        return opinions

    def analyze_consensus(self, opinions: List[AgentOpinion]) -> Dict[str, Any]:
        """Analyze the opinions for consensus."""
        if not opinions:
            return {"consensus": None, "strength": 0, "dissent": 1.0}

        # Count stances
        stance_counts: Dict[EmotionalStance, int] = {}
        total_confidence = 0.0
        total_tools_used = 0

        for op in opinions:
            stance_counts[op.stance] = stance_counts.get(op.stance, 0) + 1
            total_confidence += op.confidence
            total_tools_used += len(op.tools_used)

        # Find majority
        max_stance = max(stance_counts.items(), key=lambda x: x[1])
        consensus_strength = max_stance[1] / len(opinions)
        dissent = 1.0 - consensus_strength

        return {
            "consensus_stance": max_stance[0],
            "consensus_strength": consensus_strength,
            "dissent_level": dissent,
            "stance_distribution": {s.value: c for s, c in stance_counts.items()},
            "avg_confidence": total_confidence / len(opinions),
            "total_tools_used": total_tools_used,
            "avg_tools_per_agent": total_tools_used / len(opinions)
        }


async def run_enhanced_quorum(
    headline: str,
    event_type: Optional[IndustrialEventType] = None,
    value_billions: float = 0.0,
):
    """Run the enhanced quorum with tool use."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}")
    print(" MLX QUORUM WITH FUNCTION CALLING")
    print(f"{'='*70}{Colors.ENDC}\n")

    print(f"Model: mlx-community/Qwen3-4B-Instruct-2507-gabliterated-4bit (2048 tokens)")
    print(f"Agents: {len(ENHANCED_AGENT_ROLES)}")
    print(f"Tools: {sum(len(r.tools) for r in ENHANCED_AGENT_ROLES)} total (2-4 per agent)")
    print()

    # Show tool capabilities
    print(f"{Colors.CYAN}Agent Tool Capabilities:{Colors.ENDC}")
    for role in ENHANCED_AGENT_ROLES:
        tool_names = [t.name for t in role.tools]
        print(f"  {role.role_name}: {', '.join(tool_names)}")
    print()

    # Create event
    event = IndustrialEvent(
        event_id="enhanced_1",
        event_type=event_type or IndustrialEventType.ACQUISITION_ANNOUNCED,
        timestamp=datetime.utcnow(),
        primary_company=headline.split()[0],
        title=headline,
        value_billions=value_billions,
    )

    print(f"{Colors.CYAN}{'─'*50}")
    print(f" EVENT: {headline}")
    print(f"{'─'*50}{Colors.ENDC}\n")

    quorum = EnhancedMLXQuorum()

    print(f"{Colors.DIM}Agents analyzing with tools...{Colors.ENDC}\n")

    opinions = await quorum.deliberate(event)

    # Display summary
    print(f"{Colors.CYAN}{'─'*50}")
    print(" DELIBERATION SUMMARY")
    print(f"{'─'*50}{Colors.ENDC}\n")

    for op in opinions:
        stance_str = format_stance(op.stance)
        tools_badge = f"{len(op.tools_used)} tools" if op.tools_used else "no tools"
        print(f"{Colors.BOLD}{op.agent_role.role_name}{Colors.ENDC}: {stance_str} | {op.confidence:.0%} confidence | {tools_badge}")
        if op.tools_used:
            print(f"  {Colors.DIM}Tools: {', '.join(op.tools_used)}{Colors.ENDC}")

    # Analyze consensus
    analysis = quorum.analyze_consensus(opinions)

    print(f"\n{Colors.CYAN}{'─'*50}")
    print(" CONSENSUS ANALYSIS")
    print(f"{'─'*50}{Colors.ENDC}\n")

    print(f"  Consensus: {format_stance(analysis['consensus_stance'])}")
    print(f"  Strength: {analysis['consensus_strength']:.0%}")
    print(f"  Avg Confidence: {analysis['avg_confidence']:.0%}")
    print(f"  Tools Used: {analysis['total_tools_used']} ({analysis['avg_tools_per_agent']:.1f} per agent)")

    print(f"\n  Stance Distribution:")
    for stance, count in analysis['stance_distribution'].items():
        bar = "█" * count
        print(f"    {stance:15} {bar} ({count})")

    print(f"\n{Colors.GREEN}Done.{Colors.ENDC}\n")


# Sample events
SAMPLE_EVENTS = [
    ("Nvidia announces $20B acquisition of Cerebras Systems", IndustrialEventType.ACQUISITION_ANNOUNCED, 20.0),
    ("Anthropic raises $5B Series C led by Google and Salesforce", IndustrialEventType.FUNDING_ROUND, 5.0),
    ("Microsoft considering $15B acquisition of Mistral AI", IndustrialEventType.ACQUISITION_ANNOUNCED, 15.0),
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run enhanced MLX quorum with function calling")
    parser.add_argument("--event", type=str, help="Event headline to analyze")
    parser.add_argument("--value", type=float, default=0.0, help="Deal value in billions")
    parser.add_argument("--demo", action="store_true", help="Run with sample event")

    args = parser.parse_args()

    try:
        if args.demo or not args.event:
            # Default: run one sample
            headline, event_type, value = SAMPLE_EVENTS[0]
            asyncio.run(run_enhanced_quorum(headline, event_type, value))
        else:
            asyncio.run(run_enhanced_quorum(args.event, value_billions=args.value))
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted.{Colors.ENDC}")
