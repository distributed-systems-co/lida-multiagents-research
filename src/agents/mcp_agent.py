"""MCP-enabled agent that can execute external tools via MCP servers."""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from ..messaging import Agent, AgentConfig, MessageBroker, Message, MessageType, Priority
from ..meta.mcp_registry import (
    MCPRegistry,
    MCPServerConfig,
    MCPConnection,
    MCPTransport,
    get_mcp_registry,
)
from ..meta.templates import MCPBinding, AgentTemplate
from ..meta.structures import MetaStructure, Capability
from ..meta.timeline import (
    get_timeline_store,
    record_event,
    EventType,
    EventSeverity,
    track,
)
from ..llm.providers import (
    get_model_registry,
    get_unified_client,
    AgentModelConfig,
    ModelConfig,
)
from ..meta.personality import (
    Personality,
    get_personality_manager,
    TraitDimension,
)

logger = logging.getLogger(__name__)


class MCPExecutionResult:
    """Result from executing an MCP tool."""

    def __init__(
        self,
        tool_name: str,
        server_id: str,
        success: bool,
        result: Any = None,
        error: Optional[str] = None,
        duration: float = 0.0,
    ):
        self.tool_name = tool_name
        self.server_id = server_id
        self.success = success
        self.result = result
        self.error = error
        self.duration = duration
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "server_id": self.server_id,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "duration": self.duration,
            "timestamp": self.timestamp.isoformat(),
        }


class MCPAgent(Agent):
    """
    An agent that can execute tools from MCP servers.

    Extends the base Agent with:
    - MCP server connections
    - Tool execution routing
    - Capability-to-tool binding
    - Automatic tool invocation from messages
    """

    def __init__(
        self,
        broker: MessageBroker,
        config: Optional[AgentConfig] = None,
        structure: Optional[MetaStructure] = None,
        mcp_bindings: Optional[List[MCPBinding]] = None,
        model_config: Optional[AgentModelConfig] = None,
        personality: Optional[Personality] = None,
        personality_name: Optional[str] = None,
        auto_connect: bool = True,
    ):
        config = config or AgentConfig(agent_type="mcp_agent")
        super().__init__(broker, config)

        # Meta structure with capabilities
        self.structure = structure

        # MCP bindings: capability_name -> MCPBinding
        self._mcp_bindings: Dict[str, MCPBinding] = {}
        if mcp_bindings:
            for binding in mcp_bindings:
                self._mcp_bindings[binding.capability_name] = binding

        # Active MCP connections
        self._connections: Dict[str, MCPConnection] = {}
        self._registry = get_mcp_registry()

        # Model configuration for LLM calls
        self._model_config = model_config or AgentModelConfig(
            primary_model="anthropic/claude-sonnet-4",
            fallback_models=["openai/gpt-4.1-mini", "deepseek/deepseek-r1"],
        )
        self._llm_client = get_unified_client()

        # Personality - distinctive traits and voice
        self._personality: Optional[Personality] = personality
        if not self._personality and personality_name:
            self._personality = get_personality_manager().get(personality_name)
        self._personality_system_prompt: Optional[str] = None
        if self._personality:
            self._personality_system_prompt = self._personality.generate_system_prompt()

        # Timeline tracking
        self._timeline = get_timeline_store()

        # Execution tracking
        self._execution_history: List[MCPExecutionResult] = []
        self._pending_executions: Dict[str, asyncio.Future] = {}

        # Options
        self._auto_connect = auto_connect

        # Register message handlers
        self.inbound.register_handler(MessageType.REQUEST, self._handle_request)
        self.inbound.register_handler(MessageType.DELEGATE, self._handle_delegate)
        self.inbound.register_handler(MessageType.BROADCAST, self._handle_broadcast)
        self.inbound.register_handler(MessageType.MULTICAST, self._handle_multicast)
        self.inbound.register_handler(MessageType.RESPONSE, self._handle_response)

    async def on_start(self):
        """Initialize MCP connections."""
        logger.info(f"MCPAgent {self.agent_id} starting with {len(self._mcp_bindings)} MCP bindings...")

        # Record startup event
        await self._timeline.record(
            event_type=EventType.AGENT_STARTED,
            agent_id=self.agent_id,
            title=f"Agent started",
            description=f"MCPAgent with {len(self._mcp_bindings)} MCP bindings",
            metadata={
                "mcp_bindings": len(self._mcp_bindings),
                "model": self._model_config.primary_model,
            },
        )

        if self._auto_connect:
            await self._connect_all_servers()

        # Announce capabilities
        await self.broadcast(
            MessageType.BROADCAST,
            {
                "event": "mcp_agent_online",
                "agent_id": self.agent_id,
                "capabilities": self.list_capabilities(),
                "mcp_tools": self.list_mcp_tools(),
            },
        )

    async def on_stop(self):
        """Cleanup MCP connections."""
        logger.info(f"MCPAgent {self.agent_id} shutting down...")

        # Record shutdown event
        await self._timeline.record(
            event_type=EventType.AGENT_STOPPED,
            agent_id=self.agent_id,
            title=f"Agent stopped",
        )

        # Disconnect all MCP servers
        for server_id in list(self._connections.keys()):
            await self._disconnect_server(server_id)

        await self.broadcast(
            MessageType.BROADCAST,
            {"event": "mcp_agent_offline", "agent_id": self.agent_id},
        )

    async def on_message(self, msg: Message):
        """Handle generic messages."""
        logger.debug(f"MCPAgent {self.agent_id} received {msg.msg_type} from {msg.sender_id}")

    async def _connect_all_servers(self):
        """Connect to all MCP servers referenced in bindings."""
        server_ids = set(b.server_id for b in self._mcp_bindings.values())

        for server_id in server_ids:
            try:
                await self._connect_server(server_id)
            except Exception as e:
                logger.warning(f"Failed to connect to MCP server {server_id}: {e}")

    async def _connect_server(self, server_id: str) -> Optional[MCPConnection]:
        """Connect to a specific MCP server."""
        if server_id in self._connections:
            return self._connections[server_id]

        try:
            connection = await self._registry.connect_server(server_id)
            if connection.connected:
                self._connections[server_id] = connection
                logger.info(f"MCPAgent {self.agent_id} connected to MCP server: {server_id}")
                return connection
            else:
                logger.warning(f"Failed to connect to {server_id}: {connection.error}")
                return None
        except Exception as e:
            logger.error(f"Error connecting to MCP server {server_id}: {e}")
            return None

    async def _disconnect_server(self, server_id: str):
        """Disconnect from an MCP server."""
        if server_id in self._connections:
            await self._connections[server_id].disconnect()
            del self._connections[server_id]
            logger.info(f"MCPAgent {self.agent_id} disconnected from: {server_id}")

    # ─────────────────────────────────────────────────────────────────────────
    # MCP Tool Execution
    # ─────────────────────────────────────────────────────────────────────────

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict,
        server_id: Optional[str] = None,
    ) -> MCPExecutionResult:
        """Execute an MCP tool directly.

        Args:
            tool_name: Name of the MCP tool to execute
            arguments: Arguments to pass to the tool
            server_id: Optional server ID (auto-detected from bindings if not provided)

        Returns:
            MCPExecutionResult with the tool output
        """
        start_time = datetime.utcnow()

        # Find server from bindings if not specified
        if not server_id:
            for binding in self._mcp_bindings.values():
                if binding.tool_name == tool_name:
                    server_id = binding.server_id
                    break

        if not server_id:
            # Search all connected servers for the tool
            for sid, conn in self._connections.items():
                for tool in conn.config.tools:
                    if tool.name == tool_name:
                        server_id = sid
                        break
                if server_id:
                    break

        if not server_id:
            return MCPExecutionResult(
                tool_name=tool_name,
                server_id="unknown",
                success=False,
                error=f"No server found for tool: {tool_name}",
            )

        # Ensure connected
        connection = self._connections.get(server_id)
        if not connection:
            connection = await self._connect_server(server_id)

        if not connection or not connection.connected:
            return MCPExecutionResult(
                tool_name=tool_name,
                server_id=server_id,
                success=False,
                error=f"Not connected to server: {server_id}",
            )

        # Execute the tool with timeline tracking
        try:
            result = await connection.call_tool(tool_name, arguments)
            duration = (datetime.utcnow() - start_time).total_seconds()

            # Check for JSON-RPC error
            if "error" in result:
                exec_result = MCPExecutionResult(
                    tool_name=tool_name,
                    server_id=server_id,
                    success=False,
                    error=str(result["error"]),
                    duration=duration,
                )
                # Record error in timeline
                await self._timeline.record(
                    event_type=EventType.MCP_TOOL_ERROR,
                    agent_id=self.agent_id,
                    title=f"Tool error: {tool_name}",
                    description=str(result["error"]),
                    severity=EventSeverity.ERROR,
                    metadata={"server_id": server_id, "tool": tool_name},
                    duration_ms=duration * 1000,
                )
            else:
                exec_result = MCPExecutionResult(
                    tool_name=tool_name,
                    server_id=server_id,
                    success=True,
                    result=result.get("result", result),
                    duration=duration,
                )
                # Record success in timeline
                await self._timeline.record(
                    event_type=EventType.MCP_TOOL_RESULT,
                    agent_id=self.agent_id,
                    title=f"Tool executed: {tool_name}",
                    metadata={"server_id": server_id, "tool": tool_name},
                    duration_ms=duration * 1000,
                )
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            exec_result = MCPExecutionResult(
                tool_name=tool_name,
                server_id=server_id,
                success=False,
                error=str(e),
                duration=duration,
            )
            # Record exception in timeline
            await self._timeline.record(
                event_type=EventType.MCP_TOOL_ERROR,
                agent_id=self.agent_id,
                title=f"Tool exception: {tool_name}",
                description=str(e),
                severity=EventSeverity.ERROR,
                metadata={"server_id": server_id, "tool": tool_name, "exception": type(e).__name__},
                duration_ms=duration * 1000,
            )

        self._execution_history.append(exec_result)
        return exec_result

    async def invoke_capability(
        self,
        capability_name: str,
        inputs: dict,
    ) -> MCPExecutionResult:
        """Invoke a capability, routing to MCP tool if bound.

        Args:
            capability_name: Name of the capability to invoke
            inputs: Input values for the capability

        Returns:
            MCPExecutionResult (or simulated result for non-MCP capabilities)
        """
        binding = self._mcp_bindings.get(capability_name)

        if binding:
            # Transform inputs if specified
            arguments = inputs
            if binding.transform_input:
                arguments = binding.transform_input(inputs)

            result = await self.execute_tool(binding.tool_name, arguments, binding.server_id)

            # Transform output if specified
            if result.success and binding.transform_output:
                result.result = binding.transform_output(result.result)

            return result
        else:
            # Non-MCP capability - return placeholder
            return MCPExecutionResult(
                tool_name=capability_name,
                server_id="local",
                success=True,
                result={"message": f"Local capability {capability_name} executed", "inputs": inputs},
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Message Handlers
    # ─────────────────────────────────────────────────────────────────────────

    async def _handle_request(self, msg: Message):
        """Handle request messages - can invoke MCP tools."""
        action = msg.payload.get("action")

        if action == "execute_tool":
            # Direct MCP tool execution
            tool_name = msg.payload.get("tool_name")
            arguments = msg.payload.get("arguments", {})
            server_id = msg.payload.get("server_id")

            result = await self.execute_tool(tool_name, arguments, server_id)

            await self.send(
                msg.sender_id,
                MessageType.RESPONSE,
                result.to_dict(),
                correlation_id=msg.msg_id,
            )

        elif action == "invoke_capability":
            # Capability invocation (routes to MCP if bound)
            capability = msg.payload.get("capability")
            inputs = msg.payload.get("inputs", {})

            result = await self.invoke_capability(capability, inputs)

            await self.send(
                msg.sender_id,
                MessageType.RESPONSE,
                result.to_dict(),
                correlation_id=msg.msg_id,
            )

        elif action == "get_capabilities":
            await self.send(
                msg.sender_id,
                MessageType.RESPONSE,
                {
                    "capabilities": self.list_capabilities(),
                    "mcp_tools": self.list_mcp_tools(),
                    "connections": self.list_connections(),
                },
                correlation_id=msg.msg_id,
            )

        elif action == "get_status":
            await self.send(
                msg.sender_id,
                MessageType.RESPONSE,
                self.get_status(),
                correlation_id=msg.msg_id,
            )

        else:
            await self.send(
                msg.sender_id,
                MessageType.RESPONSE,
                {"error": f"Unknown action: {action}"},
                correlation_id=msg.msg_id,
            )

    async def _handle_delegate(self, msg: Message):
        """Handle delegated work - execute task with MCP tools."""
        task = msg.payload.get("task", msg.payload)
        task_id = task.get("id", msg.msg_id)

        # Check for MCP tool execution task
        if task.get("type") == "mcp_execute":
            tool_name = task.get("tool_name")
            arguments = task.get("arguments", {})

            result = await self.execute_tool(tool_name, arguments)

            await self.send(
                msg.sender_id,
                MessageType.REPORT,
                {
                    "task_id": task_id,
                    "status": "completed" if result.success else "failed",
                    "result": result.to_dict(),
                },
            )

        elif task.get("type") == "capability_invoke":
            capability = task.get("capability")
            inputs = task.get("inputs", {})

            result = await self.invoke_capability(capability, inputs)

            await self.send(
                msg.sender_id,
                MessageType.REPORT,
                {
                    "task_id": task_id,
                    "status": "completed" if result.success else "failed",
                    "result": result.to_dict(),
                },
            )

        else:
            # Unknown task type
            await self.send(
                msg.sender_id,
                MessageType.NACK,
                {"task_id": task_id, "reason": "unsupported_task_type"},
                correlation_id=msg.msg_id,
            )

    async def _handle_broadcast(self, msg: Message):
        """Handle broadcast messages."""
        event = msg.payload.get("event")

        if event == "mcp_tool_request":
            # Someone is looking for an agent with a specific tool
            requested_tool = msg.payload.get("tool_name")
            if self.has_tool(requested_tool):
                await self.send(
                    msg.sender_id,
                    MessageType.RESPONSE,
                    {
                        "agent_id": self.agent_id,
                        "has_tool": True,
                        "tool_name": requested_tool,
                    },
                )

    async def _handle_multicast(self, msg: Message):
        """Handle multicast messages."""
        pass

    async def _handle_response(self, msg: Message):
        """Handle response messages."""
        if msg.correlation_id and msg.correlation_id in self._pending_executions:
            future = self._pending_executions.pop(msg.correlation_id)
            future.set_result(msg.payload)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def list_capabilities(self) -> List[dict]:
        """List all capabilities."""
        if not self.structure:
            return []

        return [
            {
                "name": cap.name,
                "type": cap.capability_type.value if hasattr(cap.capability_type, 'value') else str(cap.capability_type),
                "inputs": cap.inputs,
                "outputs": cap.outputs,
                "mcp_bound": cap.name in self._mcp_bindings,
            }
            for cap in self.structure.capabilities
        ]

    def list_mcp_tools(self) -> List[dict]:
        """List all bound MCP tools."""
        return [
            {
                "capability": binding.capability_name,
                "server_id": binding.server_id,
                "tool_name": binding.tool_name,
                "auto_invoke": binding.auto_invoke,
            }
            for binding in self._mcp_bindings.values()
        ]

    def list_connections(self) -> List[dict]:
        """List active MCP connections."""
        return [
            {
                "server_id": server_id,
                "connected": conn.connected,
                "last_ping": conn.last_ping.isoformat() if conn.last_ping else None,
            }
            for server_id, conn in self._connections.items()
        ]

    def has_tool(self, tool_name: str) -> bool:
        """Check if agent has a specific MCP tool."""
        for binding in self._mcp_bindings.values():
            if binding.tool_name == tool_name:
                return True
        return False

    def has_capability(self, capability_name: str) -> bool:
        """Check if agent has a specific capability."""
        if not self.structure:
            return False
        return any(cap.name == capability_name for cap in self.structure.capabilities)

    def get_status(self) -> dict:
        """Get agent status."""
        return {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "capabilities_count": len(self.structure.capabilities) if self.structure else 0,
            "mcp_bindings_count": len(self._mcp_bindings),
            "connections": len(self._connections),
            "connected_servers": [sid for sid, c in self._connections.items() if c.connected],
            "execution_history_count": len(self._execution_history),
            "recent_executions": [e.to_dict() for e in self._execution_history[-5:]],
        }

    def get_execution_history(self, limit: int = 50) -> List[dict]:
        """Get execution history."""
        return [e.to_dict() for e in self._execution_history[-limit:]]

    # ─────────────────────────────────────────────────────────────────────────
    # LLM Methods
    # ─────────────────────────────────────────────────────────────────────────

    def get_model_config(self) -> dict:
        """Get current model configuration."""
        return self._model_config.to_dict()

    def set_model(self, model_id: str):
        """Set the primary model."""
        self._model_config.primary_model = model_id
        logger.info(f"MCPAgent {self.agent_id} switched to model: {model_id}")

    def set_model_config(self, config: AgentModelConfig):
        """Set full model configuration."""
        self._model_config = config

    async def llm_complete(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Complete a chat conversation using configured LLM.

        Args:
            messages: Chat messages [{"role": "user", "content": "..."}]
            temperature: Override temperature (uses config if not set)
            max_tokens: Override max tokens (uses config if not set)
            model_id: Override model (uses primary_model if not set)

        Returns:
            Response dict with "content", "model", "usage"
        """
        model = model_id or self._model_config.primary_model
        temp = temperature if temperature is not None else self._model_config.temperature
        tokens = max_tokens if max_tokens is not None else self._model_config.max_tokens

        start_time = datetime.utcnow()

        # Record LLM request
        await self._timeline.record(
            event_type=EventType.LLM_REQUEST,
            agent_id=self.agent_id,
            title=f"LLM request: {model}",
            metadata={
                "model": model,
                "message_count": len(messages),
                "temperature": temp,
            },
        )

        try:
            result = await self._llm_client.complete(
                messages=messages,
                model_id=model,
                temperature=temp,
                max_tokens=tokens,
                stream=False,
            )

            duration = (datetime.utcnow() - start_time).total_seconds()

            # Record LLM response
            await self._timeline.record(
                event_type=EventType.LLM_RESPONSE,
                agent_id=self.agent_id,
                title=f"LLM response: {model}",
                metadata={
                    "model": result.get("model", model),
                    "usage": result.get("usage", {}),
                },
                duration_ms=duration * 1000,
            )

            return result

        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()

            # Record error
            await self._timeline.record(
                event_type=EventType.LLM_ERROR,
                agent_id=self.agent_id,
                title=f"LLM error: {model}",
                description=str(e),
                severity=EventSeverity.ERROR,
                metadata={"model": model, "exception": type(e).__name__},
                duration_ms=duration * 1000,
            )

            # Try fallback models
            if self._model_config.retry_on_failure:
                for fallback in self._model_config.fallback_models:
                    try:
                        logger.info(f"Trying fallback model: {fallback}")
                        result = await self._llm_client.complete(
                            messages=messages,
                            model_id=fallback,
                            temperature=temp,
                            max_tokens=tokens,
                            stream=False,
                        )
                        return result
                    except Exception:
                        continue

            raise

    async def llm_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model_id: Optional[str] = None,
    ):
        """Stream a chat completion.

        Args:
            messages: Chat messages
            temperature: Override temperature
            max_tokens: Override max tokens
            model_id: Override model

        Yields:
            Content chunks as strings
        """
        model = model_id or self._model_config.primary_model
        temp = temperature if temperature is not None else self._model_config.temperature
        tokens = max_tokens if max_tokens is not None else self._model_config.max_tokens

        # Record stream start
        await self._timeline.record(
            event_type=EventType.LLM_STREAM_START,
            agent_id=self.agent_id,
            title=f"LLM stream: {model}",
            metadata={"model": model},
        )

        start_time = datetime.utcnow()
        try:
            stream = await self._llm_client.complete(
                messages=messages,
                model_id=model,
                temperature=temp,
                max_tokens=tokens,
                stream=True,
            )
            async for chunk in stream:
                yield chunk

            duration = (datetime.utcnow() - start_time).total_seconds()

            # Record stream end
            await self._timeline.record(
                event_type=EventType.LLM_STREAM_END,
                agent_id=self.agent_id,
                title=f"LLM stream complete: {model}",
                duration_ms=duration * 1000,
            )

        except Exception as e:
            await self._timeline.record(
                event_type=EventType.LLM_ERROR,
                agent_id=self.agent_id,
                title=f"LLM stream error: {model}",
                description=str(e),
                severity=EventSeverity.ERROR,
            )
            raise

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        use_personality: bool = True,
        **kwargs,
    ) -> str:
        """Simple prompt -> response generation.

        Args:
            prompt: User prompt
            system: Optional system prompt (combined with personality if both)
            use_personality: Whether to inject personality system prompt
            **kwargs: Passed to llm_complete

        Returns:
            Generated text
        """
        messages = []

        # Build system prompt combining personality and explicit system
        system_parts = []
        if use_personality and self._personality_system_prompt:
            system_parts.append(self._personality_system_prompt)
        if system:
            system_parts.append(system)

        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})

        messages.append({"role": "user", "content": prompt})

        result = await self.llm_complete(messages, **kwargs)
        content = result.get("content", "")

        # Crystallize personality based on interaction
        if use_personality and self._personality:
            self._personality.crystallize({
                "type": "generation",
                "prompt_length": len(prompt),
                "response_length": len(content),
            }, strength=0.02)

        return content

    # ─────────────────────────────────────────────────────────────────────────
    # Personality Methods
    # ─────────────────────────────────────────────────────────────────────────

    def get_personality(self) -> Optional[Personality]:
        """Get agent's personality."""
        return self._personality

    def set_personality(self, personality: Personality):
        """Set agent's personality."""
        self._personality = personality
        self._personality_system_prompt = personality.generate_system_prompt()
        logger.info(f"Agent {self.agent_id} personality set to: {personality.archetype or personality.name}")

    def set_personality_by_name(self, name: str) -> bool:
        """Set personality from the personality manager by name."""
        personality = get_personality_manager().get(name)
        if personality:
            self.set_personality(personality)
            return True
        return False

    def crystallize_personality(self, feedback: str, aspect: str = "general"):
        """Reinforce personality based on feedback."""
        if not self._personality:
            return

        from ..meta.personality import FeedbackCrystallization
        technique = FeedbackCrystallization()
        technique.apply(self._personality, {"feedback": feedback, "aspect": aspect})

        # Regenerate system prompt after crystallization
        self._personality_system_prompt = self._personality.generate_system_prompt()

    def get_personality_prompt(self) -> Optional[str]:
        """Get current personality system prompt."""
        return self._personality_system_prompt


class MCPAgentFactory:
    """Factory for creating MCPAgents from templates."""

    def __init__(self, broker: MessageBroker):
        self.broker = broker
        self._agents: Dict[str, MCPAgent] = {}

    async def create_from_template(
        self,
        template: AgentTemplate,
        params: Optional[dict] = None,
        agent_id: Optional[str] = None,
        auto_start: bool = True,
    ) -> MCPAgent:
        """Create an MCPAgent from a template.

        Args:
            template: AgentTemplate to instantiate
            params: Parameters for template instantiation
            agent_id: Optional custom agent ID
            auto_start: Whether to auto-start the agent

        Returns:
            Created MCPAgent
        """
        params = params or {}

        # Instantiate the template to get structure and bindings
        structure = template.instantiate(**params)

        # Extract MCP bindings from metadata
        bindings = []
        for binding_data in structure.metadata.get("mcp_bindings", []):
            bindings.append(MCPBinding(
                server_id=binding_data["server"],
                tool_name=binding_data["tool"],
                capability_name=binding_data["capability"],
            ))

        # Create config
        config = AgentConfig(
            agent_id=agent_id,
            agent_type=f"mcp_{template.name}",
            system_prompt=template.description,
            metadata={
                "template_id": template.template_id,
                "template_name": template.name,
            },
        )

        # Create agent
        agent = MCPAgent(
            broker=self.broker,
            config=config,
            structure=structure,
            mcp_bindings=bindings,
            auto_connect=True,
        )

        self._agents[agent.agent_id] = agent

        if auto_start:
            await agent.start()

        return agent

    async def create_research_agent(
        self,
        agent_id: Optional[str] = None,
        research_depth: int = 3,
    ) -> MCPAgent:
        """Create a research agent with Jina MCP tools."""
        from ..meta.templates import create_research_agent_template

        template = create_research_agent_template()
        return await self.create_from_template(
            template,
            params={"research_depth": research_depth},
            agent_id=agent_id,
        )

    async def create_multimodal_agent(
        self,
        agent_id: Optional[str] = None,
        research_depth: int = 5,
        include_images: bool = True,
    ) -> MCPAgent:
        """Create a multimodal research agent with full Jina capabilities."""
        from ..meta.templates import create_multimodal_research_agent_template

        template = create_multimodal_research_agent_template()
        return await self.create_from_template(
            template,
            params={"research_depth": research_depth, "include_images": include_images},
            agent_id=agent_id,
        )

    async def create_filesystem_agent(
        self,
        agent_id: Optional[str] = None,
        root_path: str = ".",
    ) -> MCPAgent:
        """Create a filesystem agent."""
        from ..meta.templates import create_filesystem_agent_template

        template = create_filesystem_agent_template()
        return await self.create_from_template(
            template,
            params={"root_path": root_path},
            agent_id=agent_id,
        )

    async def create_memory_agent(
        self,
        agent_id: Optional[str] = None,
        memory_scope: str = "session",
    ) -> MCPAgent:
        """Create a memory agent."""
        from ..meta.templates import create_memory_agent_template

        template = create_memory_agent_template()
        return await self.create_from_template(
            template,
            params={"memory_scope": memory_scope},
            agent_id=agent_id,
        )

    async def create_custom_agent(
        self,
        name: str,
        description: str,
        mcp_server_ids: List[str],
        category_filters: Optional[Dict[str, List[str]]] = None,
        agent_id: Optional[str] = None,
    ) -> MCPAgent:
        """Create a custom agent with specified MCP servers."""
        from ..meta.templates import create_composite_mcp_agent_template

        template = create_composite_mcp_agent_template(
            name=name,
            description=description,
            mcp_server_ids=mcp_server_ids,
            category_filters=category_filters,
        )
        return await self.create_from_template(template, agent_id=agent_id)

    def get_agent(self, agent_id: str) -> Optional[MCPAgent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    def list_agents(self) -> List[dict]:
        """List all created agents."""
        return [agent.get_status() for agent in self._agents.values()]

    async def terminate_agent(self, agent_id: str):
        """Terminate an agent."""
        if agent_id in self._agents:
            agent = self._agents.pop(agent_id)
            await agent.stop()

    async def terminate_all(self):
        """Terminate all agents."""
        for agent_id in list(self._agents.keys()):
            await self.terminate_agent(agent_id)
