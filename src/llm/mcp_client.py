"""MCP Client: Clean interface for Model Context Protocol servers.

Provides a unified client for connecting to MCP servers and executing tools,
with support for both SSE and stdio transports.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

import aiohttp

logger = logging.getLogger(__name__)


class MCPTransport(str, Enum):
    """Transport protocol for MCP servers."""
    SSE = "sse"
    STDIO = "stdio"
    HTTP = "http"
    WEBSOCKET = "ws"


@dataclass
class MCPTool:
    """Tool definition from an MCP server."""
    name: str
    description: str
    input_schema: dict = field(default_factory=dict)

    def to_openai_tool(self) -> dict:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema or {"type": "object", "properties": {}},
            }
        }

    def to_anthropic_tool(self) -> dict:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema or {"type": "object", "properties": {}},
        }


@dataclass
class MCPToolResult:
    """Result from executing an MCP tool."""
    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    transport: MCPTransport = MCPTransport.SSE
    url: Optional[str] = None
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    api_key_env: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0


class MCPClient:
    """
    Client for interacting with MCP (Model Context Protocol) servers.

    Supports:
    - SSE (Server-Sent Events) transport for cloud servers
    - HTTP transport for simple request/response
    - Tool discovery and execution
    - Streaming responses

    Example usage:
        ```python
        # Connect to Jina MCP
        client = MCPClient.from_url("https://mcp.jina.ai", api_key_env="JINA_API_KEY")
        await client.connect()

        # List available tools
        tools = await client.list_tools()

        # Execute a tool
        result = await client.call_tool("read_url", {"url": "https://example.com"})

        # Close connection
        await client.close()
        ```
    """

    def __init__(
        self,
        config: MCPServerConfig,
    ):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_id: Optional[str] = None
        self._connected = False
        self._tools: Dict[str, MCPTool] = {}
        self._message_endpoint: Optional[str] = None

    @classmethod
    def from_url(
        cls,
        url: str,
        name: Optional[str] = None,
        api_key_env: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
    ) -> "MCPClient":
        """Create client from a URL."""
        config = MCPServerConfig(
            name=name or url.split("//")[-1].split("/")[0],
            transport=MCPTransport.SSE,
            url=url.rstrip("/"),
            api_key_env=api_key_env,
            headers=headers or {},
            timeout=timeout,
        )
        return cls(config)

    @classmethod
    def jina(cls, api_key: Optional[str] = None) -> "MCPClient":
        """Create a Jina AI MCP client."""
        client = cls.from_url(
            "https://mcp.jina.ai",
            name="jina-mcp",
            api_key_env="JINA_API_KEY",
        )
        if api_key:
            client.config.headers["Authorization"] = f"Bearer {api_key}"
        return client

    @classmethod
    def parallel(cls, base_url: str = "http://localhost:2040") -> "MCPClient":
        """Create a Parallel AI MCP client."""
        return cls.from_url(
            base_url,
            name="parallel-mcp",
            api_key_env="PARALLEL_API_KEY",
        )

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def tools(self) -> List[MCPTool]:
        return list(self._tools.values())

    def _get_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **self.config.headers,
        }

        # Add API key from env if configured
        if self.config.api_key_env and "Authorization" not in headers:
            api_key = os.getenv(self.config.api_key_env)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

        return headers

    async def connect(self) -> bool:
        """
        Establish connection to the MCP server.

        Returns:
            True if connection successful
        """
        if self._connected:
            return True

        try:
            self._session = aiohttp.ClientSession(
                headers=self._get_headers(),
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            )

            if self.config.transport == MCPTransport.SSE:
                # Try to establish SSE session
                await self._connect_sse()
            else:
                # Simple HTTP - just verify server is reachable
                await self._connect_http()

            # Discover tools
            await self._discover_tools()

            self._connected = True
            logger.info(f"Connected to MCP server: {self.config.name} ({len(self._tools)} tools)")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to {self.config.name}: {e}")
            await self.close()
            return False

    async def _connect_sse(self):
        """Establish SSE connection and get session ID."""
        url = self.config.url

        # Try common SSE endpoints
        sse_endpoints = ["/sse", "/mcp/sse", "/v1/sse"]

        for endpoint in sse_endpoints:
            try:
                sse_url = f"{url}{endpoint}"
                async with asyncio.timeout(10):
                    async with self._session.get(
                        sse_url,
                        headers={"Accept": "text/event-stream"}
                    ) as resp:
                        if resp.status != 200:
                            continue

                        # Read initial data to get session info
                        buffer = ""
                        async for chunk in resp.content.iter_any():
                            buffer += chunk.decode()
                            if "\n" in buffer:
                                line, buffer = buffer.split("\n", 1)
                                line = line.strip()

                                if line.startswith("data:"):
                                    data = line[5:].strip()

                                    # Extract session ID from URL or JSON
                                    if "session_id=" in data:
                                        self._session_id = data.split("session_id=")[-1].split("&")[0]
                                        # Build message endpoint
                                        self._message_endpoint = f"{url}{endpoint}/messages/?session_id={self._session_id}"
                                        return

                                    try:
                                        parsed = json.loads(data)
                                        if "session_id" in parsed:
                                            self._session_id = parsed["session_id"]
                                            self._message_endpoint = f"{url}{endpoint}/messages/?session_id={self._session_id}"
                                            return
                                    except json.JSONDecodeError:
                                        pass

                            # Don't wait too long for session
                            break

                        # Fallback: use direct endpoint
                        self._message_endpoint = f"{url}/mcp"
                        return

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.debug(f"SSE endpoint {endpoint} failed: {e}")
                continue

        # Fallback to direct HTTP
        self._message_endpoint = f"{url}/mcp"

    async def _connect_http(self):
        """Simple HTTP connection verification."""
        # Try initialize request
        resp = await self._session.post(
            f"{self.config.url}/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "lida-mcp-client", "version": "1.0.0"}
                },
                "id": "init-1",
            }
        )
        if resp.status < 400:
            self._message_endpoint = f"{self.config.url}/mcp"

    async def _discover_tools(self):
        """Discover available tools from the server."""
        if not self._message_endpoint:
            return

        try:
            result = await self._rpc_call("tools/list", {})

            if "result" in result:
                tools_data = result["result"].get("tools", [])
                for t in tools_data:
                    tool = MCPTool(
                        name=t.get("name", ""),
                        description=t.get("description", ""),
                        input_schema=t.get("inputSchema", {}),
                    )
                    self._tools[tool.name] = tool

        except Exception as e:
            logger.warning(f"Failed to discover tools: {e}")

    async def _rpc_call(self, method: str, params: dict) -> dict:
        """Make a JSON-RPC call to the MCP server."""
        if not self._session or not self._message_endpoint:
            raise RuntimeError("Not connected")

        request_id = str(uuid.uuid4())[:8]

        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": request_id,
        }

        async with self._session.post(
            self._message_endpoint,
            json=payload,
        ) as resp:
            content_type = resp.headers.get("Content-Type", "")

            if resp.status >= 400:
                error_text = await resp.text()
                return {"error": {"code": resp.status, "message": error_text[:500]}}

            if "text/event-stream" in content_type:
                # Handle SSE response
                result = None
                async for line in resp.content:
                    line = line.decode().strip()
                    if line.startswith("data:"):
                        try:
                            data = json.loads(line[5:].strip())
                            if "result" in data or "error" in data:
                                return data
                        except json.JSONDecodeError:
                            continue
                return result or {"error": {"code": -1, "message": "No result in SSE stream"}}
            else:
                return await resp.json()

    async def call_tool(
        self,
        tool_name: str,
        arguments: Optional[dict] = None,
    ) -> MCPToolResult:
        """
        Execute a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            MCPToolResult with the execution result
        """
        if not self._connected:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                error="Not connected",
            )

        start = datetime.now(timezone.utc)

        try:
            result = await self._rpc_call("tools/call", {
                "name": tool_name,
                "arguments": arguments or {},
            })

            duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000

            if "error" in result:
                return MCPToolResult(
                    tool_name=tool_name,
                    success=False,
                    error=str(result["error"]),
                    duration_ms=duration,
                )

            return MCPToolResult(
                tool_name=tool_name,
                success=True,
                result=result.get("result", result),
                duration_ms=duration,
            )

        except Exception as e:
            duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                error=str(e),
                duration_ms=duration,
            )

    async def list_tools(self) -> List[MCPTool]:
        """Get list of available tools."""
        if not self._tools:
            await self._discover_tools()
        return list(self._tools.values())

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get a specific tool by name."""
        return self._tools.get(name)

    def has_tool(self, name: str) -> bool:
        """Check if a tool is available."""
        return name in self._tools

    def get_tools_for_llm(self, format: str = "openai") -> List[dict]:
        """
        Get tools in format suitable for LLM function calling.

        Args:
            format: "openai" or "anthropic"

        Returns:
            List of tool definitions
        """
        if format == "anthropic":
            return [t.to_anthropic_tool() for t in self._tools.values()]
        else:
            return [t.to_openai_tool() for t in self._tools.values()]

    async def close(self):
        """Close the connection."""
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
        self._session_id = None
        self._connected = False
        self._tools.clear()
        self._message_endpoint = None


class MCPToolExecutor:
    """
    Executor for running MCP tools with LLM integration.

    Handles the loop of:
    1. Getting LLM response with tool calls
    2. Executing tools via MCP
    3. Feeding results back to LLM

    Example:
        ```python
        executor = MCPToolExecutor()
        await executor.add_client(MCPClient.jina())

        # Execute with OpenRouter
        from src.llm.openrouter import OpenRouterClient
        llm = OpenRouterClient(default_model="anthropic/claude-opus-4.5")

        result = await executor.run(
            llm=llm,
            messages=[{"role": "user", "content": "Search for the latest AI news"}],
        )
        ```
    """

    def __init__(self):
        self._clients: Dict[str, MCPClient] = {}
        self._tool_to_client: Dict[str, str] = {}

    async def add_client(self, client: MCPClient, name: Optional[str] = None) -> bool:
        """Add an MCP client and connect."""
        if not client.connected:
            if not await client.connect():
                return False

        client_name = name or client.config.name
        self._clients[client_name] = client

        # Map tools to this client
        for tool in client.tools:
            self._tool_to_client[tool.name] = client_name

        return True

    def get_all_tools(self, format: str = "openai") -> List[dict]:
        """Get all tools from all clients."""
        tools = []
        for client in self._clients.values():
            tools.extend(client.get_tools_for_llm(format))
        return tools

    async def execute_tool(self, tool_name: str, arguments: dict) -> MCPToolResult:
        """Execute a tool, routing to the correct client."""
        client_name = self._tool_to_client.get(tool_name)
        if not client_name:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                error=f"No client found for tool: {tool_name}",
            )

        client = self._clients.get(client_name)
        if not client:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Client not found: {client_name}",
            )

        return await client.call_tool(tool_name, arguments)

    async def run(
        self,
        llm,  # OpenRouterClient or similar
        messages: List[dict],
        max_iterations: int = 5,
        model: Optional[str] = None,
        on_tool_call: Optional[Callable[[str, dict], None]] = None,
        on_tool_result: Optional[Callable[[MCPToolResult], None]] = None,
    ) -> dict:
        """
        Run an agentic loop with tool execution.

        Args:
            llm: LLM client with complete() method
            messages: Initial messages
            max_iterations: Max tool call iterations
            model: Model to use (or llm default)
            on_tool_call: Callback when tool is called
            on_tool_result: Callback when tool returns

        Returns:
            Final response dict with content and tool_calls
        """
        tools = self.get_all_tools("openai")
        conversation = list(messages)
        tool_results = []

        for i in range(max_iterations):
            # Call LLM with tools
            response = await llm.complete(
                conversation,
                model=model,
                tools=tools if tools else None,
                stream=False,
            )

            # Check for tool calls
            tool_calls = response.get("tool_calls", [])
            if not tool_calls:
                # No more tool calls - return final response
                return {
                    "content": response.get("content", ""),
                    "model": response.get("model"),
                    "tool_results": tool_results,
                    "iterations": i + 1,
                }

            # Execute tool calls
            assistant_msg = {"role": "assistant", "content": response.get("content", "")}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            conversation.append(assistant_msg)

            for tc in tool_calls:
                tool_name = tc.get("function", {}).get("name", tc.get("name", ""))
                try:
                    arguments = json.loads(tc.get("function", {}).get("arguments", tc.get("arguments", "{}")))
                except json.JSONDecodeError:
                    arguments = {}

                if on_tool_call:
                    on_tool_call(tool_name, arguments)

                # Execute
                result = await self.execute_tool(tool_name, arguments)
                tool_results.append(result)

                if on_tool_result:
                    on_tool_result(result)

                # Add result to conversation
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", str(uuid.uuid4())[:8]),
                    "content": json.dumps(result.result if result.success else {"error": result.error}),
                })

        # Max iterations reached
        return {
            "content": "Max iterations reached",
            "tool_results": tool_results,
            "iterations": max_iterations,
        }

    async def close(self):
        """Close all clients."""
        for client in self._clients.values():
            await client.close()
        self._clients.clear()
        self._tool_to_client.clear()


# Convenience functions

async def create_jina_client() -> MCPClient:
    """Create and connect to Jina MCP."""
    client = MCPClient.jina()
    await client.connect()
    return client


async def create_tool_executor(*clients: MCPClient) -> MCPToolExecutor:
    """Create executor with multiple MCP clients."""
    executor = MCPToolExecutor()
    for client in clients:
        await executor.add_client(client)
    return executor
