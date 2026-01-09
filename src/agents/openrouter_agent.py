"""OpenRouter-powered agent with personality, messaging, and MCP tools.

Unified agent that combines:
- OpenRouter LLM calls (Claude Opus 4.5, Grok-4, etc.)
- Personality crystallization system
- Redis pub/sub messaging
- MCP tool execution
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

from ..messaging import Agent, AgentConfig, MessageBroker, Message, MessageType, Priority
from ..meta.personality import (
    Personality,
    TraitDimension,
    ToneRegister,
    ResponseLength,
    get_personality_manager,
    PERSONALITY_ARCHETYPES,
)
from ..llm.openrouter import OpenRouterClient, MODELS
from ..llm.mcp_client import MCPClient, MCPToolExecutor, MCPToolResult

logger = logging.getLogger(__name__)


@dataclass
class OpenRouterAgentConfig(AgentConfig):
    """Extended config for OpenRouter agents."""
    # Model config
    model: str = "opus-4.5"  # Default to Claude Opus 4.5
    fallback_models: List[str] = field(default_factory=lambda: ["sonnet-4", "grok-4"])
    temperature: float = 0.7
    max_tokens: int = 4096

    # Personality
    personality_archetype: Optional[str] = None
    personality_seed: Optional[str] = None

    # MCP
    mcp_servers: List[str] = field(default_factory=list)  # e.g., ["jina", "parallel"]

    # Behavior
    auto_tools: bool = True  # Auto-execute tools when LLM requests
    stream_responses: bool = True


class OpenRouterAgent(Agent):
    """
    Agent powered by OpenRouter with personality and MCP tools.

    Example:
        ```python
        broker = MessageBroker()
        await broker.connect()
        await broker.start()

        agent = OpenRouterAgent(
            broker=broker,
            config=OpenRouterAgentConfig(
                agent_id="analyst-1",
                model="grok-4",
                personality_archetype="the_scholar",
                mcp_servers=["jina"],
            ),
        )
        await agent.start()

        # Generate response
        response = await agent.generate("Analyze recent AI developments")

        # Or with tools
        response = await agent.generate_with_tools(
            "Search for the latest news on Grok-4 and summarize"
        )
        ```
    """

    def __init__(
        self,
        broker: MessageBroker,
        config: Optional[OpenRouterAgentConfig] = None,
    ):
        config = config or OpenRouterAgentConfig()
        super().__init__(broker, config)

        self.or_config: OpenRouterAgentConfig = config

        # OpenRouter client
        self._llm = OpenRouterClient(
            default_model=MODELS.get(config.model, config.model),
        )

        # Personality
        self._personality: Optional[Personality] = None
        self._setup_personality()

        # MCP tool executor
        self._mcp_executor: Optional[MCPToolExecutor] = None
        self._mcp_clients: Dict[str, MCPClient] = {}

        # Conversation history per peer
        self._conversations: Dict[str, List[dict]] = {}

        # Register message handlers
        self.inbound.register_handler(MessageType.REQUEST, self._handle_request)
        self.inbound.register_handler(MessageType.DIRECT, self._handle_direct)
        self.inbound.register_handler(MessageType.BROADCAST, self._handle_broadcast)
        self.inbound.register_handler(MessageType.DELEGATE, self._handle_delegate)

    def _setup_personality(self):
        """Initialize personality from config."""
        pm = get_personality_manager()

        if self.or_config.personality_archetype:
            self._personality = pm.create(
                name=self.agent_id,
                archetype=self.or_config.personality_archetype,
            )
        elif self.or_config.personality_seed:
            self._personality = pm.create(
                name=self.agent_id,
                seed=self.or_config.personality_seed,
            )
        else:
            # Default personality based on agent_id hash
            self._personality = pm.create(
                name=self.agent_id,
                seed=self.agent_id,
            )

        logger.info(f"Agent {self.agent_id} personality: {self._personality.archetype or 'seeded'}")

    async def on_start(self):
        """Initialize MCP connections."""
        logger.info(f"OpenRouterAgent {self.agent_id} starting...")

        # Setup MCP clients
        if self.or_config.mcp_servers:
            self._mcp_executor = MCPToolExecutor()

            for server in self.or_config.mcp_servers:
                try:
                    if server == "jina":
                        client = MCPClient.jina()
                    elif server == "parallel":
                        client = MCPClient.parallel()
                    else:
                        client = MCPClient.from_url(server)

                    if await client.connect():
                        self._mcp_clients[server] = client
                        await self._mcp_executor.add_client(client, server)
                        logger.info(f"Connected to MCP server: {server} ({len(client.tools)} tools)")
                except Exception as e:
                    logger.warning(f"Failed to connect to MCP server {server}: {e}")

        # Announce presence
        await self.broadcast(
            MessageType.BROADCAST,
            {
                "event": "agent_online",
                "agent_id": self.agent_id,
                "model": self.or_config.model,
                "personality": self._personality.archetype if self._personality else None,
                "tools": self._get_tool_names(),
            },
        )

    async def on_stop(self):
        """Cleanup connections."""
        logger.info(f"OpenRouterAgent {self.agent_id} stopping...")

        # Close MCP clients
        for client in self._mcp_clients.values():
            await client.close()
        self._mcp_clients.clear()

        if self._mcp_executor:
            await self._mcp_executor.close()

        # Close LLM client
        await self._llm.close()

        # Announce departure
        await self.broadcast(
            MessageType.BROADCAST,
            {"event": "agent_offline", "agent_id": self.agent_id},
        )

    async def on_message(self, msg: Message):
        """Handle generic messages."""
        logger.debug(f"Agent {self.agent_id} received {msg.msg_type} from {msg.sender_id}")

    # ─────────────────────────────────────────────────────────────────────────
    # LLM Generation
    # ─────────────────────────────────────────────────────────────────────────

    def _build_system_prompt(self, extra_context: Optional[str] = None) -> str:
        """Build system prompt with personality."""
        parts = []

        if self._personality:
            parts.append(self._personality.generate_system_prompt())

        if extra_context:
            parts.append(extra_context)

        return "\n\n".join(parts) if parts else ""

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate a response to a prompt.

        Args:
            prompt: User prompt
            system: Additional system context
            conversation_id: ID for multi-turn conversation tracking
            model: Override model (uses config default if not set)
            **kwargs: Passed to OpenRouter

        Returns:
            Generated text
        """
        # Build messages
        messages = []

        # System prompt
        sys_prompt = self._build_system_prompt(system)
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})

        # Conversation history
        if conversation_id and conversation_id in self._conversations:
            messages.extend(self._conversations[conversation_id])

        # Current prompt
        messages.append({"role": "user", "content": prompt})

        # Resolve model
        model_id = model or self.or_config.model
        if model_id in MODELS:
            model_id = MODELS[model_id]

        # Generate
        try:
            response = await self._llm.complete(
                messages,
                model=model_id,
                temperature=kwargs.get("temperature", self.or_config.temperature),
                max_tokens=kwargs.get("max_tokens", self.or_config.max_tokens),
                stream=False,
            )

            content = response.content

            # Update conversation history
            if conversation_id:
                if conversation_id not in self._conversations:
                    self._conversations[conversation_id] = []
                self._conversations[conversation_id].append({"role": "user", "content": prompt})
                self._conversations[conversation_id].append({"role": "assistant", "content": content})

            # Crystallize personality
            if self._personality:
                self._personality.crystallize({
                    "type": "generation",
                    "prompt_length": len(prompt),
                    "response_length": len(content),
                }, strength=0.02)

            return content

        except Exception as e:
            logger.error(f"Generation failed with {model_id}: {e}")

            # Try fallback models
            for fallback in self.or_config.fallback_models:
                try:
                    fb_model = MODELS.get(fallback, fallback)
                    logger.info(f"Trying fallback model: {fb_model}")
                    response = await self._llm.complete(
                        messages,
                        model=fb_model,
                        temperature=self.or_config.temperature,
                        max_tokens=self.or_config.max_tokens,
                        stream=False,
                    )
                    return response.content
                except Exception:
                    continue

            raise

    async def stream_generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream a response chunk by chunk."""
        messages = []

        sys_prompt = self._build_system_prompt(system)
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": prompt})

        model_id = MODELS.get(model or self.or_config.model, model or self.or_config.model)

        async for chunk in await self._llm.complete(
            messages,
            model=model_id,
            temperature=kwargs.get("temperature", self.or_config.temperature),
            max_tokens=kwargs.get("max_tokens", self.or_config.max_tokens),
            stream=True,
        ):
            yield chunk

    async def generate_with_tools(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_iterations: int = 5,
        model: Optional[str] = None,
        on_tool_call: Optional[callable] = None,
        on_tool_result: Optional[callable] = None,
    ) -> dict:
        """
        Generate with MCP tool execution.

        Args:
            prompt: User prompt
            system: Additional system context
            max_iterations: Max tool call rounds
            model: Override model
            on_tool_call: Callback(tool_name, args) when tool is called
            on_tool_result: Callback(result) when tool returns

        Returns:
            Dict with content, tool_results, iterations
        """
        if not self._mcp_executor:
            # No tools - just generate
            content = await self.generate(prompt, system, model=model)
            return {"content": content, "tool_results": [], "iterations": 1}

        # Build messages
        messages = []
        sys_prompt = self._build_system_prompt(system)
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": prompt})

        model_id = MODELS.get(model or self.or_config.model, model or self.or_config.model)

        # Run tool loop
        return await self._mcp_executor.run(
            llm=self._llm,
            messages=messages,
            max_iterations=max_iterations,
            model=model_id,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Message Handlers
    # ─────────────────────────────────────────────────────────────────────────

    async def _handle_request(self, msg: Message):
        """Handle request messages."""
        action = msg.payload.get("action")

        if action == "generate":
            prompt = msg.payload.get("prompt", "")
            system = msg.payload.get("system")
            model = msg.payload.get("model")
            use_tools = msg.payload.get("use_tools", False)

            try:
                if use_tools:
                    result = await self.generate_with_tools(prompt, system, model=model)
                    content = result["content"]
                else:
                    content = await self.generate(
                        prompt,
                        system,
                        conversation_id=msg.sender_id,
                        model=model,
                    )

                await self.send(
                    msg.sender_id,
                    MessageType.RESPONSE,
                    {"content": content, "model": model or self.or_config.model},
                    correlation_id=msg.msg_id,
                )
            except Exception as e:
                await self.send(
                    msg.sender_id,
                    MessageType.RESPONSE,
                    {"error": str(e)},
                    correlation_id=msg.msg_id,
                )

        elif action == "get_status":
            await self.send(
                msg.sender_id,
                MessageType.RESPONSE,
                self.get_status(),
                correlation_id=msg.msg_id,
            )

        elif action == "get_tools":
            await self.send(
                msg.sender_id,
                MessageType.RESPONSE,
                {"tools": self._get_tool_names()},
                correlation_id=msg.msg_id,
            )

        elif action == "execute_tool":
            tool_name = msg.payload.get("tool_name")
            arguments = msg.payload.get("arguments", {})

            if self._mcp_executor:
                result = await self._mcp_executor.execute_tool(tool_name, arguments)
                await self.send(
                    msg.sender_id,
                    MessageType.RESPONSE,
                    result.to_dict(),
                    correlation_id=msg.msg_id,
                )
            else:
                await self.send(
                    msg.sender_id,
                    MessageType.RESPONSE,
                    {"error": "No MCP tools configured"},
                    correlation_id=msg.msg_id,
                )

        else:
            await self.send(
                msg.sender_id,
                MessageType.RESPONSE,
                {"error": f"Unknown action: {action}"},
                correlation_id=msg.msg_id,
            )

    async def _handle_direct(self, msg: Message):
        """Handle direct messages - generate response."""
        prompt = msg.payload.get("content") or msg.payload.get("message") or str(msg.payload)

        response = await self.generate(
            prompt,
            conversation_id=msg.sender_id,
        )

        await self.send(
            msg.sender_id,
            MessageType.DIRECT,
            {"content": response},
            correlation_id=msg.msg_id,
        )

    async def _handle_broadcast(self, msg: Message):
        """Handle broadcast messages."""
        event = msg.payload.get("event")

        if event == "quorum_request":
            # Participate in quorum deliberation
            topic = msg.payload.get("topic", "")
            context = msg.payload.get("context", "")

            response = await self.generate(
                f"Topic: {topic}\n\nContext: {context}\n\nProvide your perspective.",
            )

            await self.send(
                msg.sender_id,
                MessageType.RESPONSE,
                {
                    "agent_id": self.agent_id,
                    "opinion": response,
                    "model": self.or_config.model,
                    "personality": self._personality.archetype if self._personality else None,
                },
                correlation_id=msg.msg_id,
            )

    async def _handle_delegate(self, msg: Message):
        """Handle delegated tasks."""
        task = msg.payload.get("task", {})
        task_type = task.get("type")

        if task_type == "research":
            query = task.get("query", "")
            result = await self.generate_with_tools(
                f"Research the following topic thoroughly:\n\n{query}"
            )

            await self.send(
                msg.sender_id,
                MessageType.REPORT,
                {
                    "task_id": task.get("id"),
                    "status": "completed",
                    "result": result,
                },
            )

        elif task_type == "analyze":
            content = task.get("content", "")
            analysis = await self.generate(
                f"Analyze the following:\n\n{content}"
            )

            await self.send(
                msg.sender_id,
                MessageType.REPORT,
                {
                    "task_id": task.get("id"),
                    "status": "completed",
                    "analysis": analysis,
                },
            )

        else:
            await self.send(
                msg.sender_id,
                MessageType.NACK,
                {"task_id": task.get("id"), "reason": f"Unknown task type: {task_type}"},
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Get agent status."""
        return {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "model": self.or_config.model,
            "personality": {
                "archetype": self._personality.archetype if self._personality else None,
                "crystallization": self._personality.crystallization_level if self._personality else 0,
            },
            "tools": self._get_tool_names(),
            "mcp_servers": list(self._mcp_clients.keys()),
            "conversations": len(self._conversations),
        }

    def _get_tool_names(self) -> List[str]:
        """Get list of available tool names."""
        if not self._mcp_executor:
            return []
        tools = self._mcp_executor.get_all_tools()
        return [t["function"]["name"] for t in tools]

    def get_personality(self) -> Optional[Personality]:
        """Get agent's personality."""
        return self._personality

    def set_model(self, model: str):
        """Change the default model."""
        self.or_config.model = model
        logger.info(f"Agent {self.agent_id} model changed to: {model}")

    def clear_conversation(self, conversation_id: str):
        """Clear conversation history."""
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]


# Factory function
async def create_openrouter_agent(
    broker: MessageBroker,
    agent_id: Optional[str] = None,
    model: str = "opus-4.5",
    personality: Optional[str] = None,
    mcp_servers: Optional[List[str]] = None,
    auto_start: bool = True,
) -> OpenRouterAgent:
    """
    Create and optionally start an OpenRouter agent.

    Args:
        broker: Message broker instance
        agent_id: Custom agent ID
        model: Model to use (opus-4.5, grok-4, sonnet-4, etc.)
        personality: Personality archetype (the_scholar, the_pragmatist, etc.)
        mcp_servers: MCP servers to connect (jina, parallel, etc.)
        auto_start: Whether to start immediately

    Returns:
        Configured OpenRouterAgent
    """
    config = OpenRouterAgentConfig(
        agent_id=agent_id,
        model=model,
        personality_archetype=personality,
        mcp_servers=mcp_servers or [],
    )

    agent = OpenRouterAgent(broker=broker, config=config)

    if auto_start:
        await agent.start()

    return agent
