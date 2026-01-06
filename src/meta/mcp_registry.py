"""MCP Server Registry: Parameterizable MCP server types for agent templates.

Enables agents to be instantiated with specific MCP server capabilities,
supporting both local (stdio) and remote (SSE/WebSocket) servers.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Union
import asyncio
import aiohttp
import logging

logger = logging.getLogger(__name__)


class MCPTransport(str, Enum):
    """Transport protocol for MCP servers."""
    STDIO = "stdio"      # Local process via stdin/stdout
    SSE = "sse"          # Server-Sent Events (HTTP streaming)
    WEBSOCKET = "ws"     # WebSocket
    HTTP = "http"        # HTTP request/response


class MCPServerCategory(str, Enum):
    """Categories of MCP server capabilities."""
    SEARCH = "search"           # Web search, document search
    EMBEDDING = "embedding"     # Vector embeddings
    RERANKING = "reranking"     # Relevance reranking
    READING = "reading"         # Web/document reading
    FILESYSTEM = "filesystem"   # File operations
    DATABASE = "database"       # Database operations
    CODE = "code"              # Code execution/analysis
    VISION = "vision"          # Image analysis
    MEMORY = "memory"          # Persistent memory
    RESEARCH = "research"      # Academic/research tools
    GENERAL = "general"        # General purpose


@dataclass
class MCPTool:
    """Definition of a tool provided by an MCP server."""
    name: str
    description: str
    input_schema: dict = field(default_factory=dict)
    output_schema: dict = field(default_factory=dict)
    category: MCPServerCategory = MCPServerCategory.GENERAL

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "category": self.category.value,
        }


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    server_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    name: str = ""
    description: str = ""
    version: str = "1.0.0"

    # Connection config
    transport: MCPTransport = MCPTransport.SSE
    url: Optional[str] = None  # For SSE/WS/HTTP
    command: Optional[str] = None  # For STDIO
    args: list[str] = field(default_factory=list)

    # Authentication
    headers: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    api_key_env: Optional[str] = None  # Env var name for API key

    # Endpoints (for SSE/HTTP)
    endpoints: dict[str, str] = field(default_factory=lambda: {
        "sse": "/sse",
        "mcp": "/mcp",
    })

    # Tools provided
    tools: list[MCPTool] = field(default_factory=list)
    categories: list[MCPServerCategory] = field(default_factory=list)

    # Metadata
    source_code: Optional[str] = None
    package_name: Optional[str] = None
    auto_enable: bool = True
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "server_id": self.server_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "transport": self.transport.value,
            "url": self.url,
            "command": self.command,
            "args": self.args,
            "headers": {k: v for k, v in self.headers.items() if not k.lower().startswith("auth")},
            "endpoints": self.endpoints,
            "tools": [t.to_dict() for t in self.tools],
            "categories": [c.value for c in self.categories],
            "auto_enable": self.auto_enable,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MCPServerConfig":
        """Create config from dictionary."""
        tools = [
            MCPTool(
                name=t.get("name", ""),
                description=t.get("description", ""),
                input_schema=t.get("input_schema", {}),
                output_schema=t.get("output_schema", {}),
            )
            for t in data.get("tools", [])
        ]

        transport = data.get("transport", "sse")
        if isinstance(transport, str):
            transport = MCPTransport(transport)

        categories = [
            MCPServerCategory(c) if isinstance(c, str) else c
            for c in data.get("categories", [])
        ]

        return cls(
            server_id=data.get("server_id", str(uuid.uuid4())[:12]),
            name=data.get("name", ""),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            transport=transport,
            url=data.get("url"),
            command=data.get("command"),
            args=data.get("args", []),
            headers=data.get("headers", {}),
            env=data.get("env", {}),
            api_key_env=data.get("api_key_env"),
            endpoints=data.get("endpoints", {"sse": "/sse", "mcp": "/mcp"}),
            tools=tools,
            categories=categories,
            source_code=data.get("source_code"),
            package_name=data.get("package_name"),
            auto_enable=data.get("auto_enable", True),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MCPConnection:
    """Active connection to an MCP server."""
    config: MCPServerConfig
    connected: bool = False
    session: Optional[Any] = None  # aiohttp session or process
    session_id: Optional[str] = None  # SSE session ID for session-based servers
    last_ping: Optional[datetime] = None
    error: Optional[str] = None

    async def connect(self) -> bool:
        """Establish connection to the MCP server."""
        try:
            if self.config.transport == MCPTransport.SSE:
                self.session = aiohttp.ClientSession(headers=self.config.headers)

                # Check if server uses SSE session pattern (e.g., /mcp/sse)
                sse_endpoint = self.config.endpoints.get('sse', '/sse')
                mcp_endpoint = self.config.endpoints.get('mcp', '/mcp')

                # If SSE endpoint is different from MCP endpoint, use session-based protocol
                if '/sse' in sse_endpoint:
                    # Session-based SSE MCP (like parallel-ai)
                    sse_url = f"{self.config.url}{sse_endpoint}"
                    try:
                        async with asyncio.timeout(5):  # 5 second timeout
                            async with self.session.get(
                                sse_url,
                                headers={"Accept": "text/event-stream"}
                            ) as resp:
                                if resp.status == 200:
                                    # Read first chunk of data to get session ID
                                    buffer = ""
                                    async for chunk in resp.content.iter_any():
                                        buffer += chunk.decode()
                                        # Process complete lines
                                        while "\n" in buffer:
                                            line, buffer = buffer.split("\n", 1)
                                            line = line.strip()
                                            # Handle "data: /mcp/sse/messages/?session_id=xxx" format
                                            if line.startswith("data:"):
                                                data_content = line[5:].strip()
                                                logger.debug(f"SSE data for {self.config.server_id}: {data_content}")
                                                # Check if it's a URL with session_id
                                                if "session_id=" in data_content:
                                                    self.session_id = data_content.split("session_id=")[-1].split("&")[0]
                                                    self.connected = True
                                                    logger.info(f"Got SSE session for {self.config.server_id}: {self.session_id}")
                                                    break
                                                # Try JSON parsing
                                                try:
                                                    data = json.loads(data_content)
                                                    if "session_id" in data:
                                                        self.session_id = data["session_id"]
                                                        self.connected = True
                                                        break
                                                    elif "endpoint" in data and "session_id=" in data["endpoint"]:
                                                        self.session_id = data["endpoint"].split("session_id=")[-1].split("&")[0]
                                                        self.connected = True
                                                        break
                                                except json.JSONDecodeError:
                                                    pass
                                        if self.connected:
                                            break
                                else:
                                    logger.warning(f"SSE connect returned {resp.status} for {self.config.server_id}")
                                    self.connected = True  # Fallback
                    except asyncio.TimeoutError:
                        logger.warning(f"SSE session timeout for {self.config.server_id}, using fallback")
                        self.connected = True
                    except Exception as e:
                        logger.warning(f"SSE session init failed for {self.config.server_id}: {e}")
                        self.connected = True
                else:
                    # Simple HTTP-based MCP (like Jina cloud)
                    mcp_url = f"{self.config.url}{mcp_endpoint}"
                    try:
                        async with self.session.post(
                            mcp_url,
                            json={
                                "jsonrpc": "2.0",
                                "method": "initialize",
                                "params": {
                                    "protocolVersion": "2024-11-05",
                                    "capabilities": {},
                                    "clientInfo": {"name": "lida-mcp-agent", "version": "1.0.0"}
                                },
                                "id": "init-1",
                            },
                            headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
                        ) as resp:
                            self.connected = resp.status < 400 or resp.status == 405
                    except aiohttp.ClientError as e:
                        logger.warning(f"MCP init failed for {self.config.server_id}: {e}")
                        self.error = str(e)
                        self.connected = False

            elif self.config.transport == MCPTransport.STDIO:
                # Would spawn subprocess
                self.connected = True
            else:
                self.connected = True

            self.last_ping = datetime.now()
            return self.connected
        except Exception as e:
            self.error = str(e) if str(e) else repr(e)
            self.connected = False
            logger.error(f"Failed to connect to {self.config.server_id}: {self.error}")
            return False

    async def disconnect(self):
        """Close connection."""
        if self.session:
            await self.session.close()
            self.session = None
        self.session_id = None
        self.connected = False

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Call a tool on the MCP server."""
        if not self.connected or not self.session:
            raise RuntimeError(f"Not connected to {self.config.name}")

        if self.config.transport == MCPTransport.SSE:
            request_id = str(uuid.uuid4())

            # Determine endpoint based on session type
            if self.session_id:
                # Session-based SSE MCP
                sse_endpoint = self.config.endpoints.get('sse', '/mcp/sse')
                endpoint = f"{self.config.url}{sse_endpoint}/messages/?session_id={self.session_id}"
            else:
                # Direct HTTP MCP
                mcp_endpoint = self.config.endpoints.get('mcp', '/mcp')
                endpoint = f"{self.config.url}{mcp_endpoint}"

            try:
                async with self.session.post(
                    endpoint,
                    json={
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "params": {
                            "name": tool_name,
                            "arguments": arguments,
                        },
                        "id": request_id,
                    },
                    headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
                ) as resp:
                    content_type = resp.headers.get("Content-Type", "")

                    if resp.status >= 400:
                        error_text = await resp.text()
                        return {"error": f"HTTP {resp.status}: {error_text[:200]}"}

                    if "text/event-stream" in content_type:
                        # Handle SSE response
                        result = None
                        async for line in resp.content:
                            line = line.decode().strip()
                            if line.startswith("data:"):
                                try:
                                    data = json.loads(line[5:].strip())
                                    if "result" in data:
                                        result = data
                                        break
                                    elif "error" in data:
                                        return data
                                except json.JSONDecodeError:
                                    continue
                        return result or {"error": "No result in SSE stream"}
                    else:
                        # Regular JSON response
                        return await resp.json()
            except Exception as e:
                logger.error(f"Tool call failed: {e}")
                return {"error": str(e)}

        # Placeholder for other transports
        return {"error": f"Transport {self.config.transport} not fully implemented"}


class MCPRegistry:
    """Registry of available MCP server types.

    Provides:
    - Built-in server definitions (Jina, filesystem, etc.)
    - Loading from config files
    - Server discovery from common paths
    - Connection management
    """

    _instance: Optional["MCPRegistry"] = None

    def __init__(self):
        self.servers: dict[str, MCPServerConfig] = {}
        self.connections: dict[str, MCPConnection] = {}
        self._load_builtin_servers()

    @classmethod
    def get_instance(cls) -> "MCPRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_builtin_servers(self):
        """Load built-in MCP server definitions."""

        # Jina AI MCP Server
        jina = MCPServerConfig(
            server_id="jina-mcp",
            name="jina-mcp-server",
            description="Jina AI MCP Server with Reader, Embeddings, Reranker, and research tools",
            version="1.2.0",
            transport=MCPTransport.SSE,
            url="https://mcp.jina.ai",
            api_key_env="JINA_API_KEY",
            endpoints={"sse": "/sse", "mcp": "/v1"},
            categories=[
                MCPServerCategory.SEARCH,
                MCPServerCategory.EMBEDDING,
                MCPServerCategory.RERANKING,
                MCPServerCategory.READING,
                MCPServerCategory.RESEARCH,
            ],
            tools=[
                MCPTool("read_url", "Extract clean content from web pages", category=MCPServerCategory.READING),
                MCPTool("search_web", "Search the web for current information", category=MCPServerCategory.SEARCH),
                MCPTool("search_arxiv", "Search academic papers on arXiv", category=MCPServerCategory.RESEARCH),
                MCPTool("search_images", "Search for images across the web", category=MCPServerCategory.SEARCH),
                MCPTool("expand_query", "Expand and rewrite search queries", category=MCPServerCategory.SEARCH),
                MCPTool("parallel_read_url", "Read multiple web pages in parallel", category=MCPServerCategory.READING),
                MCPTool("parallel_search_web", "Run multiple web searches in parallel", category=MCPServerCategory.SEARCH),
                MCPTool("parallel_search_arxiv", "Run multiple arXiv searches in parallel", category=MCPServerCategory.RESEARCH),
                MCPTool("sort_by_relevance", "Rerank documents by relevance to a query", category=MCPServerCategory.RERANKING),
                MCPTool("deduplicate_strings", "Get top-k semantically unique strings", category=MCPServerCategory.EMBEDDING),
                MCPTool("generate_embeddings", "Generate embeddings for texts or images", category=MCPServerCategory.EMBEDDING),
                MCPTool("summarize_text", "Generate concise summaries", category=MCPServerCategory.READING),
                MCPTool("find_similar", "Find content similar to reference", category=MCPServerCategory.EMBEDDING),
                MCPTool("compare_texts", "Compare texts for semantic differences", category=MCPServerCategory.EMBEDDING),
                MCPTool("answer_with_evidence", "Answer questions with cited evidence", category=MCPServerCategory.RESEARCH),
                MCPTool("capture_screenshot_url", "Capture screenshots of web pages", category=MCPServerCategory.VISION),
            ],
            source_code="https://github.com/jina-ai/MCP",
            package_name="jina-mcp",
        )
        self.servers["jina-mcp"] = jina

        # Filesystem MCP Server
        filesystem = MCPServerConfig(
            server_id="filesystem-mcp",
            name="filesystem-mcp",
            description="Local filesystem operations via MCP",
            version="1.0.0",
            transport=MCPTransport.STDIO,
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem"],
            categories=[MCPServerCategory.FILESYSTEM],
            tools=[
                MCPTool("read_file", "Read file contents", category=MCPServerCategory.FILESYSTEM),
                MCPTool("write_file", "Write to a file", category=MCPServerCategory.FILESYSTEM),
                MCPTool("list_directory", "List directory contents", category=MCPServerCategory.FILESYSTEM),
                MCPTool("create_directory", "Create a directory", category=MCPServerCategory.FILESYSTEM),
                MCPTool("move_file", "Move or rename a file", category=MCPServerCategory.FILESYSTEM),
                MCPTool("delete_file", "Delete a file", category=MCPServerCategory.FILESYSTEM),
                MCPTool("get_file_info", "Get file metadata", category=MCPServerCategory.FILESYSTEM),
            ],
            package_name="@modelcontextprotocol/server-filesystem",
        )
        self.servers["filesystem-mcp"] = filesystem

        # Memory MCP Server (example)
        memory = MCPServerConfig(
            server_id="memory-mcp",
            name="memory-mcp",
            description="Persistent memory and knowledge graph via MCP",
            version="1.0.0",
            transport=MCPTransport.STDIO,
            command="npx",
            args=["-y", "@modelcontextprotocol/server-memory"],
            categories=[MCPServerCategory.MEMORY],
            tools=[
                MCPTool("store_memory", "Store a memory/fact", category=MCPServerCategory.MEMORY),
                MCPTool("recall_memory", "Recall stored memories", category=MCPServerCategory.MEMORY),
                MCPTool("search_memories", "Search through memories", category=MCPServerCategory.MEMORY),
                MCPTool("create_entity", "Create a knowledge graph entity", category=MCPServerCategory.MEMORY),
                MCPTool("create_relation", "Create a relation between entities", category=MCPServerCategory.MEMORY),
            ],
            package_name="@modelcontextprotocol/server-memory",
        )
        self.servers["memory-mcp"] = memory

        # Brave Search MCP
        brave = MCPServerConfig(
            server_id="brave-search-mcp",
            name="brave-search-mcp",
            description="Brave Search API via MCP",
            version="1.0.0",
            transport=MCPTransport.STDIO,
            command="npx",
            args=["-y", "@modelcontextprotocol/server-brave-search"],
            api_key_env="BRAVE_API_KEY",
            categories=[MCPServerCategory.SEARCH],
            tools=[
                MCPTool("brave_web_search", "Search the web via Brave", category=MCPServerCategory.SEARCH),
                MCPTool("brave_local_search", "Search local results via Brave", category=MCPServerCategory.SEARCH),
            ],
            package_name="@modelcontextprotocol/server-brave-search",
        )
        self.servers["brave-search-mcp"] = brave

        # Parallel AI MCP Server (local)
        parallel = MCPServerConfig(
            server_id="parallel-mcp",
            name="parallel-ai-mcp",
            description="Parallel AI research tools - web search, extraction, tasks, and FindAll",
            version="1.0.0",
            transport=MCPTransport.SSE,
            url="http://localhost:2040",
            api_key_env="PARALLEL_API_KEY",
            endpoints={"sse": "/mcp/sse", "mcp": "/mcp"},
            categories=[
                MCPServerCategory.SEARCH,
                MCPServerCategory.RESEARCH,
                MCPServerCategory.READING,
            ],
            tools=[
                MCPTool("search", "Web search with AI-powered excerpts", category=MCPServerCategory.SEARCH),
                MCPTool("extract", "Extract content from URLs", category=MCPServerCategory.READING),
                MCPTool("create_task_run", "Create async research task", category=MCPServerCategory.RESEARCH),
                MCPTool("manage_task_run", "Create and poll task to completion", category=MCPServerCategory.RESEARCH),
                MCPTool("get_task_run", "Get task status", category=MCPServerCategory.RESEARCH),
                MCPTool("get_task_run_result", "Get task result", category=MCPServerCategory.RESEARCH),
                MCPTool("create_findall", "FindAll entity discovery", category=MCPServerCategory.RESEARCH),
                MCPTool("get_findall_result", "Get FindAll results", category=MCPServerCategory.RESEARCH),
            ],
        )
        self.servers["parallel-mcp"] = parallel

        # Local Jina MCP (via wrangler dev)
        jina_local = MCPServerConfig(
            server_id="jina-mcp-local",
            name="jina-mcp-local",
            description="Local Jina MCP server (wrangler dev)",
            version="1.2.0",
            transport=MCPTransport.SSE,
            url="http://localhost:8787",
            api_key_env="JINA_API_KEY",
            endpoints={"sse": "/sse", "mcp": "/v1"},
            categories=[
                MCPServerCategory.SEARCH,
                MCPServerCategory.READING,
                MCPServerCategory.RESEARCH,
            ],
            tools=[
                MCPTool("read_url", "Extract content from web pages", category=MCPServerCategory.READING),
                MCPTool("search_web", "Search the web", category=MCPServerCategory.SEARCH),
                MCPTool("search_arxiv", "Search arXiv papers", category=MCPServerCategory.RESEARCH),
            ],
        )
        self.servers["jina-mcp-local"] = jina_local

    def register_server(self, config: MCPServerConfig):
        """Register an MCP server configuration."""
        self.servers[config.server_id] = config

    def get_server(self, server_id: str) -> Optional[MCPServerConfig]:
        """Get server config by ID."""
        return self.servers.get(server_id)

    def list_servers(self, category: Optional[MCPServerCategory] = None) -> list[MCPServerConfig]:
        """List all registered servers, optionally filtered by category."""
        servers = list(self.servers.values())
        if category:
            servers = [s for s in servers if category in s.categories]
        return servers

    def load_from_file(self, path: Union[str, Path]):
        """Load server configs from a JSON file."""
        path = Path(path)
        if not path.exists():
            return

        with open(path) as f:
            data = json.load(f)

        mcp_servers = data.get("mcpServers", data)
        for name, config in mcp_servers.items():
            if isinstance(config, dict):
                config["name"] = config.get("name", name)
                server = MCPServerConfig.from_dict(config)
                self.servers[server.server_id] = server

    def discover_servers(self):
        """Discover MCP servers from common config locations."""
        home = Path.home()
        search_paths = [
            home / "mcp_config.json",
            home / ".mcp" / "config.json",
            home / ".claude" / "mcp_servers.json",
            Path.cwd() / "mcp_config.json",
            Path.cwd() / ".mcp.json",
        ]

        for path in search_paths:
            if path.exists():
                self.load_from_file(path)

    async def connect_server(self, server_id: str) -> MCPConnection:
        """Connect to an MCP server."""
        config = self.get_server(server_id)
        if not config:
            raise ValueError(f"Unknown server: {server_id}")

        # Resolve API key from env
        if config.api_key_env:
            api_key = os.environ.get(config.api_key_env)
            if api_key:
                config.headers["Authorization"] = f"Bearer {api_key}"

        connection = MCPConnection(config=config)
        await connection.connect()

        if connection.connected:
            self.connections[server_id] = connection

        return connection

    async def disconnect_server(self, server_id: str):
        """Disconnect from an MCP server."""
        if server_id in self.connections:
            await self.connections[server_id].disconnect()
            del self.connections[server_id]

    def get_tools_by_category(self, category: MCPServerCategory) -> list[tuple[str, MCPTool]]:
        """Get all tools of a given category across all servers."""
        tools = []
        for server in self.servers.values():
            for tool in server.tools:
                if tool.category == category:
                    tools.append((server.server_id, tool))
        return tools

    def to_dict(self) -> dict:
        """Export registry to dictionary."""
        return {
            "servers": {sid: s.to_dict() for sid, s in self.servers.items()},
            "connections": {
                sid: {
                    "connected": c.connected,
                    "last_ping": c.last_ping.isoformat() if c.last_ping else None,
                    "error": c.error,
                }
                for sid, c in self.connections.items()
            },
        }


# Global registry access
_registry: Optional[MCPRegistry] = None


def get_mcp_registry() -> MCPRegistry:
    """Get the global MCP registry instance."""
    global _registry
    if _registry is None:
        _registry = MCPRegistry()
        _registry.discover_servers()
    return _registry


def reset_mcp_registry():
    """Reset the global registry."""
    global _registry
    _registry = None
