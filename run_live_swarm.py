#!/usr/bin/env python3
"""
Live Multi-Agent Swarm with MCP Tool Integration

A visually stunning dashboard showing AI agents deliberating with:
- Real LLM calls via OpenRouter (Claude, GPT, Grok, DeepSeek)
- MCP tool execution (Jina search, parallel tasks)
- Real-time visual effects for tool usage
- Personality-driven responses

Usage:
    python run_live_swarm.py                    # Simulation mode (no API needed)
    python run_live_swarm.py --live             # Real LLM mode (needs OPENROUTER_API_KEY)
    python run_live_swarm.py --live --tools     # LLM + MCP tools (needs JINA_API_KEY too)
    python run_live_swarm.py --agents=12        # Custom agent count
    python run_live_swarm.py --topic "Your topic here"
"""

import asyncio
import os
import random
import time
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import sys

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.columns import Columns
from rich import box
from rich.style import Style
from rich.align import Align
from rich.syntax import Syntax

from src.meta.personality import (
    get_personality_manager,
    PERSONALITY_ARCHETYPES,
    Personality,
)

console = Console(force_terminal=True, color_system="truecolor")

# ═══════════════════════════════════════════════════════════════════════════════
# VISUAL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

AGENT_PALETTES = [
    {"color": "#FF6B6B", "bg": "#2D1F1F", "name": "Coral"},
    {"color": "#4ECDC4", "bg": "#1F2D2B", "name": "Teal"},
    {"color": "#45B7D1", "bg": "#1F252D", "name": "Sky"},
    {"color": "#96CEB4", "bg": "#222D25", "name": "Sage"},
    {"color": "#DDA0DD", "bg": "#2D1F2D", "name": "Plum"},
    {"color": "#F7DC6F", "bg": "#2D2B1F", "name": "Gold"},
    {"color": "#BB8FCE", "bg": "#251F2D", "name": "Lavender"},
    {"color": "#85C1E9", "bg": "#1F252D", "name": "Azure"},
    {"color": "#F8B500", "bg": "#2D261F", "name": "Amber"},
    {"color": "#00CED1", "bg": "#1F2D2D", "name": "Cyan"},
    {"color": "#FF8C00", "bg": "#2D221F", "name": "Orange"},
    {"color": "#98FB98", "bg": "#1F2D1F", "name": "Mint"},
]

AGENT_ICONS = ["◆", "●", "▲", "■", "★", "◈", "◉", "⬟", "⬡", "◐", "◑", "▼"]

PERSONALITY_INFO = {
    "the_scholar": {"emoji": "📚", "short": "Scholar", "trait": "analytical"},
    "the_pragmatist": {"emoji": "🎯", "short": "Pragmatist", "trait": "efficient"},
    "the_creative": {"emoji": "🎨", "short": "Creative", "trait": "innovative"},
    "the_skeptic": {"emoji": "🔍", "short": "Skeptic", "trait": "critical"},
    "the_mentor": {"emoji": "🎓", "short": "Mentor", "trait": "supportive"},
    "the_synthesizer": {"emoji": "🔮", "short": "Synthesizer", "trait": "integrative"},
}

MODELS = {
    "opus": "anthropic/claude-opus-4",
    "sonnet": "anthropic/claude-sonnet-4",
    "grok": "x-ai/grok-3",
    "deepseek": "deepseek/deepseek-r1",
    "gpt4": "openai/gpt-4o",
    "llama": "meta-llama/llama-3.3-70b-instruct",
}

# Tool visualization
TOOL_ICONS = {
    "search": "🔎",
    "read_url": "📄",
    "web_search": "🌐",
    "parallel": "⚡",
    "analyze": "🧠",
    "fetch": "📥",
    "default": "🔧",
}

SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
PULSE = ["░", "▒", "▓", "█", "▓", "▒"]
TOOL_ANIM = ["⟨", "⟨⟩", "⟨⟩⟨", "⟨⟩⟨⟩", "⟩⟨⟩", "⟩⟨", "⟩"]

# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ToolCall:
    """Record of an MCP tool call."""
    timestamp: float
    agent_id: str
    tool_name: str
    arguments: Dict[str, Any]
    status: str = "pending"  # pending, running, success, error
    result: Optional[Any] = None
    duration_ms: float = 0
    error: Optional[str] = None

@dataclass
class Agent:
    id: str
    name: str
    personality_type: str
    personality: Personality
    model: str
    color: str
    bg_color: str
    icon: str
    emoji: str
    status: str = "idle"
    current_thought: str = ""
    last_response: str = ""
    messages_sent: int = 0
    position_votes: Dict[str, str] = field(default_factory=dict)
    energy: float = 1.0
    # MCP capabilities
    mcp_connected: bool = False
    available_tools: List[str] = field(default_factory=list)
    tool_calls: int = 0
    current_tool: Optional[str] = None

@dataclass
class Message:
    timestamp: float
    sender_id: str
    target: str  # agent_id or "broadcast"
    msg_type: str
    content: str
    model_used: str = ""
    tool_used: Optional[str] = None

# ═══════════════════════════════════════════════════════════════════════════════
# MCP CLIENT (LIGHTWEIGHT)
# ═══════════════════════════════════════════════════════════════════════════════

class SwarmMCPClient:
    """Lightweight MCP client for swarm agents."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.connected = False
        self.tools: Dict[str, dict] = {}
        self._session = None

    async def connect(self) -> bool:
        """Connect to MCP servers (Jina)."""
        try:
            import aiohttp
            self._session = aiohttp.ClientSession()

            # Check for Jina API key
            jina_key = os.getenv("JINA_API_KEY")
            if jina_key:
                self.tools = {
                    "web_search": {
                        "name": "web_search",
                        "description": "Search the web for information",
                        "endpoint": "https://s.jina.ai/",
                    },
                    "read_url": {
                        "name": "read_url",
                        "description": "Read content from a URL",
                        "endpoint": "https://r.jina.ai/",
                    },
                    "fact_check": {
                        "name": "fact_check",
                        "description": "Verify facts and claims",
                        "endpoint": "https://g.jina.ai/",
                    },
                }
                self.connected = True
                return True
            return False
        except Exception:
            return False

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Execute an MCP tool."""
        if not self.connected or tool_name not in self.tools:
            return {"error": f"Tool {tool_name} not available"}

        tool = self.tools[tool_name]
        jina_key = os.getenv("JINA_API_KEY")

        try:
            headers = {
                "Authorization": f"Bearer {jina_key}",
                "Accept": "application/json",
            }

            query = arguments.get("query", arguments.get("url", ""))
            url = f"{tool['endpoint']}{query}"

            async with self._session.get(url, headers=headers, timeout=10) as resp:
                if resp.status == 200:
                    try:
                        data = await resp.json()
                        return {"success": True, "result": data}
                    except:
                        text = await resp.text()
                        return {"success": True, "result": text[:500]}
                else:
                    return {"error": f"HTTP {resp.status}"}
        except asyncio.TimeoutError:
            return {"error": "Timeout"}
        except Exception as e:
            return {"error": str(e)[:100]}

    async def close(self):
        if self._session:
            await self._session.close()

# ═══════════════════════════════════════════════════════════════════════════════
# SWARM ORCHESTRATOR WITH MCP
# ═══════════════════════════════════════════════════════════════════════════════

class SwarmOrchestrator:
    """Orchestrates the multi-agent swarm with MCP tool support."""

    def __init__(self, num_agents: int = 8, live_mode: bool = False, tools_mode: bool = False):
        self.num_agents = min(num_agents, 12)
        self.live_mode = live_mode
        self.tools_mode = tools_mode
        self.agents: Dict[str, Agent] = {}
        self.messages: deque = deque(maxlen=100)
        self.tool_calls: deque = deque(maxlen=50)
        self.start_time = time.time()
        self.frame = 0

        # Deliberation state
        self.current_topic = ""
        self.phase = "initializing"
        self.consensus: Dict[str, int] = {}
        self.total_messages = 0
        self.total_tool_calls = 0

        # Active tool animation
        self.active_tools: Dict[str, ToolCall] = {}

        # LLM client
        self.llm_client = None

        # MCP clients per agent
        self.mcp_clients: Dict[str, SwarmMCPClient] = {}

        # Initialize
        self._create_agents()

    def _create_agents(self):
        """Create diverse agent swarm."""
        pm = get_personality_manager()
        archetypes = list(PERSONALITY_ARCHETYPES.keys())
        model_list = list(MODELS.keys())

        for i in range(self.num_agents):
            agent_id = f"swarm-{i:02d}"
            archetype = archetypes[i % len(archetypes)]
            palette = AGENT_PALETTES[i % len(AGENT_PALETTES)]
            model = model_list[i % len(model_list)]

            # Create personality
            personality = pm.create(name=agent_id, archetype=archetype)
            info = PERSONALITY_INFO.get(archetype, {"emoji": "🤖", "short": "Agent", "trait": "adaptive"})

            self.agents[agent_id] = Agent(
                id=agent_id,
                name=f"{info['short']}-{i:02d}",
                personality_type=archetype,
                personality=personality,
                model=model,
                color=palette["color"],
                bg_color=palette["bg"],
                icon=AGENT_ICONS[i % len(AGENT_ICONS)],
                emoji=info["emoji"],
            )

    async def init_llm(self):
        """Initialize LLM client for live mode."""
        if not self.live_mode:
            return True

        try:
            from src.llm.openrouter import OpenRouterClient
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                console.print("[yellow]Warning: OPENROUTER_API_KEY not set, using simulation mode[/]")
                self.live_mode = False
                return False
            self.llm_client = OpenRouterClient()
            return True
        except Exception as e:
            console.print(f"[yellow]LLM init failed: {e}, using simulation mode[/]")
            self.live_mode = False
            return False

    async def init_mcp(self):
        """Initialize MCP clients for all agents."""
        if not self.tools_mode:
            return 0

        connected = 0
        for agent_id in self.agents:
            client = SwarmMCPClient(agent_id)
            if await client.connect():
                self.mcp_clients[agent_id] = client
                self.agents[agent_id].mcp_connected = True
                self.agents[agent_id].available_tools = list(client.tools.keys())
                connected += 1

        return connected

    def elapsed(self) -> str:
        e = time.time() - self.start_time
        return f"{int(e//60):02d}:{int(e%60):02d}"

    def add_message(self, sender_id: str, target: str, msg_type: str, content: str,
                    model: str = "", tool: str = None):
        msg = Message(
            timestamp=time.time() - self.start_time,
            sender_id=sender_id,
            target=target,
            msg_type=msg_type,
            content=content[:100],
            model_used=model,
            tool_used=tool,
        )
        self.messages.append(msg)
        self.total_messages += 1

        if sender_id in self.agents:
            self.agents[sender_id].messages_sent += 1
            self.agents[sender_id].last_response = content[:60]
            self.agents[sender_id].energy = min(1.0, self.agents[sender_id].energy + 0.2)

    async def execute_tool(self, agent_id: str, tool_name: str, arguments: dict) -> Optional[dict]:
        """Execute an MCP tool for an agent with visual feedback."""
        if agent_id not in self.mcp_clients:
            return None

        agent = self.agents[agent_id]
        client = self.mcp_clients[agent_id]

        # Create tool call record
        call = ToolCall(
            timestamp=time.time() - self.start_time,
            agent_id=agent_id,
            tool_name=tool_name,
            arguments=arguments,
            status="running",
        )
        self.tool_calls.append(call)
        self.active_tools[f"{agent_id}:{tool_name}"] = call
        self.total_tool_calls += 1

        # Update agent state
        agent.current_tool = tool_name
        agent.status = "tool_exec"
        agent.tool_calls += 1

        start = time.time()
        try:
            result = await client.call_tool(tool_name, arguments)
            call.duration_ms = (time.time() - start) * 1000

            if "error" in result:
                call.status = "error"
                call.error = result["error"]
            else:
                call.status = "success"
                call.result = result.get("result")

            return result
        except Exception as e:
            call.status = "error"
            call.error = str(e)
            return {"error": str(e)}
        finally:
            agent.current_tool = None
            agent.status = "idle"
            # Remove from active after short delay for visual
            await asyncio.sleep(0.3)
            self.active_tools.pop(f"{agent_id}:{tool_name}", None)

    async def generate_response(self, agent: Agent, prompt: str, use_tools: bool = False) -> str:
        """Generate response using LLM or simulation."""

        # Tool-augmented generation
        if use_tools and agent.mcp_connected and self.tools_mode:
            # Decide if agent should use a tool based on personality
            should_search = random.random() < 0.4  # 40% chance
            if agent.personality_type == "the_scholar":
                should_search = random.random() < 0.7  # Scholars search more
            elif agent.personality_type == "the_skeptic":
                should_search = random.random() < 0.6  # Skeptics verify facts

            if should_search and agent.available_tools:
                tool = random.choice(agent.available_tools)
                query = self._extract_search_query(prompt, agent)

                self.add_message(agent.id, "broadcast", "tool_start",
                               f"🔧 Using {tool}: {query[:30]}...", agent.model, tool)

                result = await self.execute_tool(agent.id, tool, {"query": query})

                if result and result.get("success"):
                    # Augment prompt with tool result
                    tool_info = str(result.get("result", ""))[:200]
                    prompt = f"{prompt}\n\nRelevant information found:\n{tool_info}"
                    self.add_message(agent.id, "broadcast", "tool_result",
                                   f"✓ {tool} returned data", agent.model, tool)

        if self.live_mode and self.llm_client:
            try:
                system = agent.personality.generate_system_prompt()
                response = await self.llm_client.generate(
                    prompt,
                    system=system,
                    model=MODELS.get(agent.model, "anthropic/claude-sonnet-4"),
                    max_tokens=150,
                )
                return response.content
            except Exception as e:
                return f"[Error: {str(e)[:30]}]"
        else:
            return self._simulate_response(agent, prompt)

    def _extract_search_query(self, prompt: str, agent: Agent) -> str:
        """Extract a search query from the prompt based on personality."""
        # Simple extraction - in production would use LLM
        words = prompt.split()[:8]
        base_query = " ".join(words)

        # Personality-influenced queries
        if agent.personality_type == "the_scholar":
            return f"research {base_query}"
        elif agent.personality_type == "the_skeptic":
            return f"fact check {base_query}"
        elif agent.personality_type == "the_creative":
            return f"innovative approaches {base_query}"
        return base_query

    def _simulate_response(self, agent: Agent, prompt: str) -> str:
        """Generate simulated response based on personality."""
        responses = {
            "the_scholar": [
                "Upon careful analysis, the evidence suggests...",
                "From a theoretical standpoint, we must consider...",
                "The data indicates several key factors...",
                "Historical precedent shows us that...",
                "My research suggests three main considerations...",
            ],
            "the_pragmatist": [
                "Let's focus on what actually works here.",
                "The practical approach would be to...",
                "Bottom line: we need actionable solutions.",
                "Here's what we can implement immediately...",
                "Pragmatically, we should prioritize...",
            ],
            "the_creative": [
                "What if we approached this completely differently?",
                "I see an unexpected connection here...",
                "Imagine the possibilities if we...",
                "Here's a novel perspective on this...",
                "Let's think outside conventional bounds...",
            ],
            "the_skeptic": [
                "I need to see more evidence before agreeing.",
                "Have we considered the counterarguments?",
                "This assumption may be flawed because...",
                "Let me challenge that premise...",
                "I've verified this claim and found...",
            ],
            "the_mentor": [
                "Let me help clarify this for everyone...",
                "Building on what we've learned...",
                "Consider this perspective as well...",
                "I think we're making good progress here...",
                "This connects to larger principles of...",
            ],
            "the_synthesizer": [
                "Integrating these viewpoints, I see...",
                "There's common ground between positions...",
                "Let me connect these ideas together...",
                "The synthesis of these views suggests...",
                "Pulling these threads together reveals...",
            ],
        }
        options = responses.get(agent.personality_type, ["I have thoughts on this..."])
        return random.choice(options)

    # ═══════════════════════════════════════════════════════════════════════════
    # DELIBERATION PHASES
    # ═══════════════════════════════════════════════════════════════════════════

    async def run_deliberation(self, topic: str):
        """Run a full deliberation cycle with tool usage."""
        self.current_topic = topic
        agents = list(self.agents.values())

        # Phase 1: Analysis (with tool research)
        self.phase = "🔍 analyzing"
        for agent in agents:
            agent.status = "analyzing"
            agent.current_thought = f"Processing: {topic[:25]}..."
            await asyncio.sleep(0.1)

        # Tool-using agents do initial research
        if self.tools_mode:
            research_agents = [a for a in agents if a.mcp_connected][:3]
            for agent in research_agents:
                agent.status = "researching"
                if agent.available_tools:
                    tool = random.choice(agent.available_tools)
                    query = topic[:50]
                    self.add_message(agent.id, "broadcast", "research",
                                   f"🔎 Researching: {query[:30]}...", agent.model, tool)
                    await self.execute_tool(agent.id, tool, {"query": query})
                await asyncio.sleep(0.3)

        await asyncio.sleep(1)

        # Phase 2: Initial positions
        self.phase = "💬 discussing"
        for agent in agents:
            agent.status = "speaking"
            prompt = f"Topic: {topic}\n\nShare your initial position in 1-2 sentences."
            use_tools = random.random() < 0.3  # 30% use tools
            response = await self.generate_response(agent, prompt, use_tools=use_tools)
            agent.current_thought = response[:50]
            self.add_message(agent.id, "broadcast", "position", response, agent.model)
            await asyncio.sleep(0.25)

        await asyncio.sleep(0.8)

        # Phase 3: Cross-discussion with tool verification
        self.phase = "🔄 debating"
        for _ in range(6):
            a1 = random.choice(agents)
            a2 = random.choice([a for a in agents if a.id != a1.id])

            a1.status = "speaking"
            prompt = f"Respond to {a2.name}'s point about: {topic}"

            # Skeptics more likely to fact-check
            use_tools = a1.personality_type == "the_skeptic" and random.random() < 0.5
            response = await self.generate_response(a1, prompt, use_tools=use_tools)
            a1.current_thought = response[:50]
            self.add_message(a1.id, a2.id, "respond", response, a1.model)
            await asyncio.sleep(0.35)

        # Phase 4: Voting
        self.phase = "🗳️ voting"
        self.consensus = {"Support": 0, "Oppose": 0, "Modify": 0, "Abstain": 0}

        for agent in agents:
            agent.status = "voting"
            weights = {
                "the_scholar": [3, 1, 4, 2],
                "the_pragmatist": [4, 2, 3, 1],
                "the_creative": [3, 2, 4, 1],
                "the_skeptic": [2, 4, 3, 1],
                "the_mentor": [4, 1, 3, 2],
                "the_synthesizer": [3, 1, 5, 1],
            }
            w = weights.get(agent.personality_type, [2, 2, 2, 2])
            vote = random.choices(["Support", "Oppose", "Modify", "Abstain"], weights=w)[0]
            self.consensus[vote] += 1
            agent.position_votes[topic[:20]] = vote
            self.add_message(agent.id, "broadcast", "vote", f"Vote: {vote}", agent.model)
            await asyncio.sleep(0.2)

        # Phase 5: Synthesis
        self.phase = "🔮 synthesizing"
        synthesizers = [a for a in agents if a.personality_type == "the_synthesizer"]
        synthesizer = random.choice(synthesizers) if synthesizers else random.choice(agents)
        synthesizer.status = "synthesizing"
        synthesizer.current_thought = "Integrating all perspectives..."

        await asyncio.sleep(1.2)

        winner = max(self.consensus, key=self.consensus.get)
        self.add_message(synthesizer.id, "broadcast", "synthesis",
                        f"Consensus reached: {winner} ({self.consensus[winner]}/{len(agents)} votes)",
                        synthesizer.model)

        # Complete
        self.phase = "✅ complete"
        for agent in agents:
            agent.status = "idle"
            agent.current_thought = ""

        await asyncio.sleep(2)

    async def cleanup(self):
        """Cleanup MCP clients."""
        for client in self.mcp_clients.values():
            await client.close()


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

class DashboardRenderer:
    """Renders the live dashboard with tool visualization."""

    def __init__(self, orchestrator: SwarmOrchestrator):
        self.orch = orchestrator
        self.frame = 0

    def render(self) -> Layout:
        self.frame += 1

        # Decay agent energy
        for agent in self.orch.agents.values():
            agent.energy = max(0.1, agent.energy - 0.012)

        layout = Layout()
        layout.split_column(
            Layout(name="header", size=5),
            Layout(name="main"),
            Layout(name="footer", size=3),
        )

        layout["main"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=2),
        )

        layout["left"].split_column(
            Layout(name="agents", ratio=3),
            Layout(name="tools", ratio=2),
        )

        layout["right"].split_column(
            Layout(name="messages", ratio=2),
            Layout(name="bottom", ratio=1),
        )

        layout["bottom"].split_row(
            Layout(name="thoughts"),
            Layout(name="consensus"),
        )

        # Render components
        layout["header"].update(self._header())
        layout["agents"].update(self._agents())
        layout["tools"].update(self._tools())
        layout["messages"].update(self._messages())
        layout["thoughts"].update(self._thoughts())
        layout["consensus"].update(self._consensus())
        layout["footer"].update(self._footer())

        return layout

    def _header(self) -> Panel:
        spin = SPINNER[self.frame % len(SPINNER)]
        pulse = PULSE[self.frame % len(PULSE)]
        tool_anim = TOOL_ANIM[self.frame % len(TOOL_ANIM)]

        title = Text()
        title.append(f" {spin} ", style="bold cyan")
        title.append("LIDA", style="bold #7C3AED")
        title.append(" Swarm Intelligence ", style="bold white")
        if self.orch.tools_mode:
            title.append(f"{tool_anim}", style="#10B981")
        else:
            title.append(f"{pulse}", style="#7C3AED")

        mode_parts = []
        if self.orch.live_mode:
            mode_parts.append("[green]LLM[/]")
        else:
            mode_parts.append("[yellow]SIM[/]")
        if self.orch.tools_mode:
            mode_parts.append("[cyan]MCP[/]")
        mode = "+".join(mode_parts)

        info = Text()
        info.append(f"⏱ {self.orch.elapsed()}", style="#06B6D4")
        info.append("  │  ", style="dim")
        info.append(f"📨 {self.orch.total_messages}", style="#10B981")
        info.append("  │  ", style="dim")
        info.append(f"🔧 {self.orch.total_tool_calls}", style="#F59E0B")
        info.append("  │  ", style="dim")
        info.append(f"👥 {len(self.orch.agents)}", style="#EC4899")
        info.append("  │  ", style="dim")
        info.append(f"{self.orch.phase}", style="#8B5CF6")
        info.append("  │  ", style="dim")
        info.append(f"Mode: {mode}")

        return Panel(
            Group(Align.center(title), Align.center(info)),
            box=box.DOUBLE,
            border_style="#7C3AED",
        )

    def _agents(self) -> Panel:
        table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="bold #06B6D4",
            expand=True,
            padding=(0, 1),
        )
        table.add_column("Agent", width=12)
        table.add_column("Type", width=8)
        table.add_column("Model", width=7)
        table.add_column("⚡", width=6)
        table.add_column("🔧", width=3)
        table.add_column("📨", width=3)

        for agent in sorted(self.orch.agents.values(), key=lambda a: a.id):
            status_icon = {
                "idle": "○",
                "analyzing": "◐",
                "speaking": "●",
                "voting": "◑",
                "synthesizing": "◉",
                "tool_exec": "⚡",
                "researching": "🔎",
            }.get(agent.status, "○")

            energy_bar = "█" * int(agent.energy * 5) + "░" * (5 - int(agent.energy * 5))
            info = PERSONALITY_INFO.get(agent.personality_type, {})

            # Highlight if using tool
            name_style = f"bold {agent.color}"
            if agent.current_tool:
                name_style = f"bold reverse {agent.color}"

            tool_indicator = "✓" if agent.mcp_connected else "○"

            table.add_row(
                Text(f"{status_icon} {agent.id[-5:]}", style=name_style),
                Text(info.get("short", "Agent")[:6], style="dim"),
                Text(agent.model[:5], style="dim"),
                Text(energy_bar, style=agent.color),
                Text(tool_indicator, style="#10B981" if agent.mcp_connected else "dim"),
                str(agent.messages_sent),
            )

        return Panel(table, title="[bold]👥 Swarm[/]", border_style="#374151", box=box.ROUNDED)

    def _tools(self) -> Panel:
        """Render active and recent tool calls."""
        lines = []

        # Active tools with animation
        for key, call in list(self.orch.active_tools.items()):
            anim = SPINNER[self.frame % len(SPINNER)]
            agent = self.orch.agents.get(call.agent_id)
            color = agent.color if agent else "white"
            icon = TOOL_ICONS.get(call.tool_name, TOOL_ICONS["default"])

            line = Text()
            line.append(f"{anim} ", style="yellow")
            line.append(f"{icon} ", style=color)
            line.append(f"{call.agent_id[-5:]} ", style=f"bold {color}")
            line.append(f"{call.tool_name}", style="cyan")
            lines.append(line)

        # Recent completed tools
        recent = list(self.orch.tool_calls)[-5:]
        for call in reversed(recent):
            if call.status == "running":
                continue
            agent = self.orch.agents.get(call.agent_id)
            color = agent.color if agent else "white"
            icon = TOOL_ICONS.get(call.tool_name, TOOL_ICONS["default"])

            status_icon = "✓" if call.status == "success" else "✗"
            status_color = "#10B981" if call.status == "success" else "#EF4444"

            line = Text()
            line.append(f"{status_icon} ", style=status_color)
            line.append(f"{icon} ", style="dim")
            line.append(f"{call.agent_id[-5:]} ", style=color)
            line.append(f"{call.duration_ms:.0f}ms", style="dim")
            lines.append(line)

        if not lines:
            lines.append(Text("No tool activity yet...", style="dim italic"))

        return Panel(Group(*lines[:6]), title="[bold]🔧 MCP Tools[/]", border_style="#374151", box=box.ROUNDED)

    def _messages(self) -> Panel:
        lines = []
        for msg in list(self.orch.messages)[-12:]:
            line = Text()
            line.append(f"{msg.timestamp:5.1f}s ", style="dim")

            agent = self.orch.agents.get(msg.sender_id)
            color = agent.color if agent else "white"
            icon = agent.icon if agent else "•"

            line.append(f"{icon} ", style=color)
            line.append(f"{msg.sender_id[-5:]} ", style=f"bold {color}")

            if msg.target == "broadcast":
                line.append("→ ", style="dim")
                line.append("ALL ", style="#EC4899")
            else:
                target = self.orch.agents.get(msg.target)
                tcolor = target.color if target else "white"
                line.append("→ ", style="dim")
                line.append(f"{msg.target[-5:]} ", style=tcolor)

            # Tool indicator
            if msg.tool_used:
                tool_icon = TOOL_ICONS.get(msg.tool_used, "🔧")
                line.append(f"{tool_icon} ", style="#10B981")

            line.append(f"{msg.content[:35]}", style="dim")
            lines.append(line)

        if not lines:
            lines.append(Text("Awaiting swarm activity...", style="dim italic"))

        return Panel(Group(*lines), title="[bold]💬 Message Stream[/]", border_style="#374151", box=box.ROUNDED)

    def _thoughts(self) -> Panel:
        lines = []
        thinking = [a for a in self.orch.agents.values() if a.current_thought or a.current_tool]

        for agent in thinking[:4]:
            line = Text()
            line.append(f"{agent.emoji} ", style=agent.color)
            line.append(f"{agent.name}: ", style=f"bold {agent.color}")

            if agent.current_tool:
                tool_icon = TOOL_ICONS.get(agent.current_tool, "🔧")
                line.append(f"{tool_icon} executing...", style="italic yellow")
            else:
                line.append(agent.current_thought[:32], style="italic dim")
            lines.append(line)

        if not lines:
            spin = SPINNER[(self.frame // 2) % len(SPINNER)]
            lines.append(Text(f"{spin} Processing...", style="dim"))

        return Panel(Group(*lines), title="[bold]💭 Thoughts[/]", border_style="#374151", box=box.ROUNDED)

    def _consensus(self) -> Panel:
        lines = []

        if self.orch.current_topic:
            lines.append(Text(f"📋 {self.orch.current_topic[:40]}", style="bold"))
            lines.append(Text(""))

        total = sum(self.orch.consensus.values()) or 1
        for option, count in sorted(self.orch.consensus.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            bar_len = 15
            bar = "█" * int(pct / (100/bar_len)) + "░" * (bar_len - int(pct / (100/bar_len)))
            color = "#10B981" if pct >= 50 else "#F59E0B" if pct >= 25 else "#6B7280"

            line = Text()
            line.append(f"{option:8} ", style="bold")
            line.append(bar, style=color)
            line.append(f" {pct:4.0f}%", style=color)
            lines.append(line)

        return Panel(Group(*lines), title="[bold]🗳️ Consensus[/]", border_style="#374151", box=box.ROUNDED)

    def _footer(self) -> Panel:
        footer = Text()
        footer.append(" Press ", style="dim")
        footer.append("Ctrl+C", style="bold #F59E0B")
        footer.append(" to exit ", style="dim")
        footer.append("│", style="dim")

        # Show connected tools
        connected = sum(1 for a in self.orch.agents.values() if a.mcp_connected)
        if connected > 0:
            footer.append(f" 🔧 {connected} agents with MCP ", style="#10B981")
            footer.append("│", style="dim")

        footer.append(f" {datetime.now().strftime('%H:%M:%S')} ", style="dim")
        footer.append("│", style="dim")
        footer.append(" LIDA Multi-Agent Research ", style="dim #7C3AED")

        return Panel(footer, box=box.SIMPLE, border_style="#374151")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

TOPICS = [
    "Should autonomous AI systems be allowed to modify their own goals?",
    "How should we handle disagreement between AI agents in critical decisions?",
    "What governance structures best serve multi-agent coordination?",
    "Is emergent swarm behavior preferable to explicit orchestration?",
    "How do we establish trust in decentralized agent networks?",
    "Should AI agents specialize or maintain general capabilities?",
    "What role should human oversight play in agent-to-agent negotiations?",
    "How do we handle value alignment across diverse agent personalities?",
]

async def run_swarm(num_agents: int, live: bool, tools: bool, topic: str):
    """Main swarm runner."""
    console.print("\n[bold #7C3AED]╔═══════════════════════════════════════════════════════════╗[/]")
    console.print("[bold #7C3AED]║[/]     [bold white]LIDA Multi-Agent Swarm Dashboard[/]                  [bold #7C3AED]║[/]")
    console.print("[bold #7C3AED]╚═══════════════════════════════════════════════════════════╝[/]\n")

    # Initialize
    console.print(f"[cyan]Initializing swarm with {num_agents} agents...[/]")
    orch = SwarmOrchestrator(num_agents=num_agents, live_mode=live, tools_mode=tools)

    if live:
        success = await orch.init_llm()
        if success:
            console.print("[green]✓ LLM client connected[/]")
        else:
            console.print("[yellow]⚠ Using simulation mode[/]")

    if tools:
        connected = await orch.init_mcp()
        if connected > 0:
            console.print(f"[green]✓ MCP tools enabled for {connected} agents[/]")
        else:
            console.print("[yellow]⚠ No MCP connections (set JINA_API_KEY)[/]")
            orch.tools_mode = False

    renderer = DashboardRenderer(orch)
    console.print("[green]✓ Swarm ready[/]")
    console.print("[dim]Starting dashboard...[/]\n")
    await asyncio.sleep(1)

    # Background deliberation
    async def deliberation_loop():
        topics = [topic] if topic else TOPICS
        while True:
            t = random.choice(topics) if len(topics) > 1 else topics[0]
            await orch.run_deliberation(t)
            await asyncio.sleep(2)

    task = asyncio.create_task(deliberation_loop())

    # Dashboard
    try:
        with Live(renderer.render(), console=console, refresh_per_second=10, screen=True) as live_display:
            while True:
                live_display.update(renderer.render())
                await asyncio.sleep(0.1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await orch.cleanup()
        console.print("\n[yellow]Swarm shutdown complete.[/]")


def main():
    # Parse args
    num_agents = 8
    live_mode = False
    tools_mode = False
    topic = ""

    for arg in sys.argv[1:]:
        if arg.startswith("--agents="):
            num_agents = int(arg.split("=")[1])
        elif arg == "--live":
            live_mode = True
        elif arg == "--tools":
            tools_mode = True
        elif arg.startswith("--topic="):
            topic = arg.split("=", 1)[1]

    asyncio.run(run_swarm(num_agents, live_mode, tools_mode, topic))


if __name__ == "__main__":
    main()
