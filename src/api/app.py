"""FastAPI application for multi-agent system."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    SpawnAgentRequest,
    ChatMessageRequest,
    SendMessageRequest,
    CreateDatasetRequest,
    AddDataRequest,
    AgentInfo,
    StatsInfo,
)
from .websocket import ConnectionManager
from .datasets import DatasetStore

logger = logging.getLogger(__name__)


def create_app(
    orchestrator: Any = None,
    title: str = "LIDA Multi-Agent System",
) -> FastAPI:
    """Create the FastAPI application."""

    app = FastAPI(
        title=title,
        description="Real-time multi-agent orchestration and monitoring",
        version="1.0.0",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # State
    app.state.orchestrator = orchestrator
    app.state.ws_manager = ConnectionManager()
    app.state.dataset_store = DatasetStore()
    app.state.start_time = datetime.now(timezone.utc)
    app.state.stats = {
        "messages_sent": 0,
        "messages_received": 0,
        "broadcasts": 0,
        "multicasts": 0,
        "direct_messages": 0,
        "errors": 0,
    }

    # Mount static files
    static_path = Path(__file__).parent / "static"
    static_path.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

    # ─────────────────────────────────────────────────────────────────────────
    # Dashboard routes
    # ─────────────────────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        """Serve the main dashboard."""
        dashboard_path = Path(__file__).parent / "static" / "index.html"
        if dashboard_path.exists():
            return FileResponse(dashboard_path)
        return HTMLResponse(content=get_dashboard_html(), status_code=200)

    # ─────────────────────────────────────────────────────────────────────────
    # WebSocket routes
    # ─────────────────────────────────────────────────────────────────────────

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """Main WebSocket for real-time updates."""
        manager = app.state.ws_manager
        await manager.connect(websocket)

        try:
            # Send initial state
            await websocket.send_json({
                "type": "init",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "agents": get_agents_list(app),
                    "stats": get_stats(app),
                },
            })

            while True:
                data = await websocket.receive_text()
                # Handle incoming messages if needed
                logger.debug(f"WebSocket received: {data}")

        except WebSocketDisconnect:
            manager.disconnect(websocket)

    @app.websocket("/ws/chat/{agent_id}")
    async def chat_websocket(websocket: WebSocket, agent_id: str):
        """WebSocket for chatting with a specific agent."""
        manager = app.state.ws_manager
        store = app.state.dataset_store

        await manager.connect_chat(agent_id, websocket)

        try:
            # Send chat history
            history = store.get_chat_history(agent_id)
            await websocket.send_json({
                "type": "chat_history",
                "agent_id": agent_id,
                "messages": history,
            })

            while True:
                data = await websocket.receive_json()
                message = data.get("message", "")

                if message:
                    # Save user message
                    store.save_chat_message(agent_id, "user", message)
                    await manager.broadcast_chat_message(agent_id, "user", message)

                    # Get agent response
                    response = await process_chat(app, agent_id, message)

                    # Save and broadcast response
                    store.save_chat_message(agent_id, "assistant", response)
                    await manager.broadcast_chat_message(agent_id, "assistant", response)

                    await websocket.send_json({
                        "type": "chat_response",
                        "agent_id": agent_id,
                        "response": response,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

        except WebSocketDisconnect:
            manager.disconnect_chat(agent_id)

    # ─────────────────────────────────────────────────────────────────────────
    # Agent routes
    # ─────────────────────────────────────────────────────────────────────────

    @app.get("/api/agents")
    async def list_agents():
        """List all agents."""
        return {"agents": get_agents_list(app)}

    @app.get("/api/agents/{agent_id}")
    async def get_agent(agent_id: str):
        """Get a specific agent."""
        orchestrator = app.state.orchestrator
        if not orchestrator or not orchestrator.supervisor:
            raise HTTPException(404, "Orchestrator not initialized")

        agents = orchestrator.supervisor._agents
        if agent_id not in agents:
            raise HTTPException(404, f"Agent {agent_id} not found")

        state = agents[agent_id]
        return get_agent_details(state)

    @app.get("/api/agents/{agent_id}/details")
    async def get_agent_details_endpoint(agent_id: str):
        """Get detailed agent state including internal data."""
        orchestrator = app.state.orchestrator
        if not orchestrator or not orchestrator.supervisor:
            raise HTTPException(404, "Orchestrator not initialized")

        agents = orchestrator.supervisor._agents
        if agent_id not in agents:
            raise HTTPException(404, f"Agent {agent_id} not found")

        state = agents[agent_id]
        return get_agent_details(state)

    @app.post("/api/agents/spawn")
    async def spawn_agent(request: SpawnAgentRequest):
        """Spawn a new agent."""
        orchestrator = app.state.orchestrator
        if not orchestrator or not orchestrator.supervisor:
            raise HTTPException(400, "Orchestrator not initialized")

        try:
            # Map agent type to spec name
            spec_map = {
                "demiurge": "demiurge",
                "persona": f"persona_{len([a for a in orchestrator.supervisor._agents if 'persona' in a])}",
                "worker": f"worker_{len([a for a in orchestrator.supervisor._agents if 'worker' in a])}",
            }

            spec_name = spec_map.get(request.agent_type.value)
            if not spec_name:
                raise HTTPException(400, f"Unknown agent type: {request.agent_type}")

            agent_id = await orchestrator.supervisor.spawn(spec_name, request.agent_id)

            # Broadcast update
            await app.state.ws_manager.broadcast_agent_update(get_agents_list(app))

            return {"agent_id": agent_id, "status": "spawned"}

        except Exception as e:
            logger.error(f"Failed to spawn agent: {e}")
            raise HTTPException(400, str(e))

    @app.delete("/api/agents/{agent_id}")
    async def terminate_agent(agent_id: str):
        """Terminate an agent."""
        orchestrator = app.state.orchestrator
        if not orchestrator or not orchestrator.supervisor:
            raise HTTPException(400, "Orchestrator not initialized")

        try:
            await orchestrator.supervisor.terminate(agent_id, "api_request")
            await app.state.ws_manager.broadcast_agent_update(get_agents_list(app))
            return {"agent_id": agent_id, "status": "terminated"}
        except Exception as e:
            raise HTTPException(400, str(e))

    # ─────────────────────────────────────────────────────────────────────────
    # Chat routes
    # ─────────────────────────────────────────────────────────────────────────

    @app.post("/api/chat")
    async def chat(request: ChatMessageRequest):
        """Send a chat message to an agent."""
        store = app.state.dataset_store

        # Save user message
        store.save_chat_message(request.agent_id, "user", request.message)

        # Get response
        response = await process_chat(app, request.agent_id, request.message)

        # Save response
        store.save_chat_message(request.agent_id, "assistant", response)

        # Broadcast
        await app.state.ws_manager.broadcast_chat_message(
            request.agent_id, "user", request.message
        )
        await app.state.ws_manager.broadcast_chat_message(
            request.agent_id, "assistant", response
        )

        return {
            "agent_id": request.agent_id,
            "response": response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/api/chat/{agent_id}/history")
    async def get_chat_history(agent_id: str, limit: int = Query(50, ge=1, le=500)):
        """Get chat history for an agent."""
        store = app.state.dataset_store
        history = store.get_chat_history(agent_id, limit)
        return {"agent_id": agent_id, "messages": history}

    # ─────────────────────────────────────────────────────────────────────────
    # Message routes
    # ─────────────────────────────────────────────────────────────────────────

    @app.post("/api/messages/send")
    async def send_message(request: SendMessageRequest):
        """Send a message between agents."""
        orchestrator = app.state.orchestrator
        store = app.state.dataset_store

        if not orchestrator or not orchestrator.supervisor:
            raise HTTPException(400, "Orchestrator not initialized")

        agents = orchestrator.supervisor._agents
        if request.sender_id not in agents:
            raise HTTPException(404, f"Sender {request.sender_id} not found")

        sender = agents[request.sender_id].agent

        from ..messaging import MessageType
        msg_type = MessageType(request.message_type.upper())

        msg_id = await sender.send(
            request.recipient_id,
            msg_type,
            request.payload,
        )

        # Log message
        store.log_message(
            msg_id,
            request.sender_id,
            request.recipient_id,
            request.message_type,
            request.payload,
        )

        app.state.stats["messages_sent"] += 1
        app.state.stats["direct_messages"] += 1

        await app.state.ws_manager.broadcast_message_event({
            "msg_id": msg_id,
            "sender_id": request.sender_id,
            "recipient_id": request.recipient_id,
            "msg_type": request.message_type,
        })

        return {"msg_id": msg_id, "status": "sent"}

    @app.get("/api/messages/log")
    async def get_message_log(
        limit: int = Query(100, ge=1, le=1000),
        sender_id: Optional[str] = None,
        recipient_id: Optional[str] = None,
    ):
        """Get message log."""
        store = app.state.dataset_store
        messages = store.get_message_log(limit, sender_id, recipient_id)
        return {"messages": messages}

    # ─────────────────────────────────────────────────────────────────────────
    # Dataset routes
    # ─────────────────────────────────────────────────────────────────────────

    @app.get("/api/datasets")
    async def list_datasets():
        """List all datasets."""
        store = app.state.dataset_store
        return {"datasets": store.list_datasets()}

    @app.post("/api/datasets")
    async def create_dataset(request: CreateDatasetRequest):
        """Create a new dataset."""
        store = app.state.dataset_store
        try:
            dataset = store.create_dataset(
                request.name,
                request.description,
                request.schema_def,
                request.tags,
            )
            return dataset
        except Exception as e:
            raise HTTPException(400, str(e))

    @app.get("/api/datasets/{dataset_id}")
    async def get_dataset(dataset_id: str):
        """Get a dataset."""
        store = app.state.dataset_store
        dataset = store.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(404, "Dataset not found")
        return dataset

    @app.delete("/api/datasets/{dataset_id}")
    async def delete_dataset(dataset_id: str):
        """Delete a dataset."""
        store = app.state.dataset_store
        if store.delete_dataset(dataset_id):
            return {"status": "deleted"}
        raise HTTPException(404, "Dataset not found")

    @app.post("/api/datasets/{dataset_id}/records")
    async def add_records(dataset_id: str, request: AddDataRequest):
        """Add records to a dataset."""
        store = app.state.dataset_store
        dataset = store.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(404, "Dataset not found")

        added = store.add_records(dataset_id, request.records)
        return {"added": added, "total": dataset["record_count"] + added}

    @app.get("/api/datasets/{dataset_id}/records")
    async def get_records(
        dataset_id: str,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ):
        """Get records from a dataset."""
        store = app.state.dataset_store
        records = store.get_records(dataset_id, limit, offset)
        return {"records": records}

    # ─────────────────────────────────────────────────────────────────────────
    # Stats routes
    # ─────────────────────────────────────────────────────────────────────────

    @app.get("/api/stats")
    async def get_statistics():
        """Get system statistics."""
        return get_stats(app)

    # ─────────────────────────────────────────────────────────────────────────
    # Prompts API
    # ─────────────────────────────────────────────────────────────────────────

    @app.get("/api/prompts")
    async def list_prompts(
        category: Optional[str] = None,
        subcategory: Optional[str] = None,
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ):
        """List available prompts."""
        loader = orchestrator.prompt_loader if orchestrator else None
        if not loader:
            return {"prompts": [], "total": 0}

        all_prompts = loader.all_prompts()

        # Filter by category
        if category:
            from src.prompts import PromptCategory
            try:
                cat = PromptCategory(category)
                all_prompts = [p for p in all_prompts if p.category == cat]
            except ValueError:
                pass

        # Filter by subcategory
        if subcategory:
            all_prompts = [p for p in all_prompts if subcategory.lower() in p.subcategory.lower()]

        total = len(all_prompts)
        prompts = all_prompts[offset:offset + limit]

        return {
            "prompts": [
                {
                    "id": p.id,
                    "text": p.text,
                    "category": p.category.value,
                    "subcategory": p.subcategory,
                    "tags": p.tags,
                }
                for p in prompts
            ],
            "total": total,
            "offset": offset,
            "limit": limit,
        }

    @app.get("/api/prompts/categories")
    async def list_prompt_categories():
        """List prompt categories with counts."""
        loader = orchestrator.prompt_loader if orchestrator else None
        if not loader:
            return {"categories": {}, "subcategories": {}}

        return {
            "categories": loader.categories(),
            "subcategories": loader.subcategories(),
            "total": loader.count(),
        }

    @app.get("/api/prompts/{prompt_id}")
    async def get_prompt(prompt_id: int):
        """Get a specific prompt by ID."""
        loader = orchestrator.prompt_loader if orchestrator else None
        if not loader:
            raise HTTPException(status_code=404, detail="Prompt loader not available")

        prompt = loader.get(prompt_id)
        if not prompt:
            raise HTTPException(status_code=404, detail=f"Prompt {prompt_id} not found")

        return {
            "id": prompt.id,
            "text": prompt.text,
            "category": prompt.category.value,
            "subcategory": prompt.subcategory,
            "tags": prompt.tags,
        }

    @app.get("/api/prompts/search/{query}")
    async def search_prompts(query: str, limit: int = Query(20, ge=1, le=100)):
        """Search prompts by text content."""
        loader = orchestrator.prompt_loader if orchestrator else None
        if not loader:
            return {"results": []}

        results = loader.search(query, limit=limit)
        return {
            "query": query,
            "results": [
                {
                    "id": p.id,
                    "text": p.text,
                    "category": p.category.value,
                    "subcategory": p.subcategory,
                }
                for p in results
            ],
        }

    @app.get("/api/agents/{agent_id}/prompt")
    async def get_agent_prompt(agent_id: str):
        """Get the prompt assigned to an agent."""
        if not orchestrator:
            raise HTTPException(status_code=404, detail="Orchestrator not available")

        prompt_data = orchestrator.persona_prompts.get(agent_id)
        if not prompt_data:
            raise HTTPException(status_code=404, detail=f"No prompt for agent {agent_id}")

        return prompt_data

    # ─────────────────────────────────────────────────────────────────────────
    # Background tasks
    # ─────────────────────────────────────────────────────────────────────────

    @app.on_event("startup")
    async def startup():
        """Start background update loop."""
        asyncio.create_task(broadcast_updates(app))

    return app


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def get_agent_details(state) -> dict:
    """Get detailed agent information including internal state."""
    agent = state.agent
    base_info = {
        "agent_id": state.agent_id,
        "agent_type": state.agent_type,
        "status": state.status.value,
        "start_count": state.start_count,
        "last_start": state.last_start.isoformat() if state.last_start else None,
        "last_error": state.last_error,
        "inbox": agent.inbound.pending_count() if hasattr(agent, "inbound") else 0,
        "outbox": agent.outbound.pending_count() if hasattr(agent, "outbound") else 0,
        "processed": agent.inbound.stats.processed if hasattr(agent, "inbound") else 0,
        "dropped": agent.inbound.stats.dropped if hasattr(agent, "inbound") else 0,
    }

    # Add type-specific data
    if state.agent_type == "demiurge":
        base_info["world_state"] = {
            "forms": len(agent.world_state.get("S", {})),
            "laws": len(agent.world_state.get("L", {})),
            "metrics": len(agent.world_state.get("M", {})),
            "observers": len(agent.world_state.get("O", {})),
            "affordances": len(agent.world_state.get("A", {})),
            "values": len(agent.world_state.get("V", {})),
            "chronicle_entries": len(agent.world_state.get("R", [])),
        }
        base_info["subordinates"] = list(agent._subordinates.keys())
        base_info["subordinate_count"] = len(agent._subordinates)
        base_info["deliberation_pending"] = agent._deliberation_queue.qsize()
        # Recent chronicle
        chronicle = agent.world_state.get("R", [])
        base_info["recent_events"] = chronicle[-10:] if chronicle else []

    elif state.agent_type == "persona":
        base_info["prompt_id"] = getattr(agent, "prompt_id", None)
        base_info["prompt_category"] = getattr(agent, "prompt_category", None)
        base_info["prompt_subcategory"] = getattr(agent, "prompt_subcategory", None)
        base_info["persona_prompt"] = getattr(agent, "persona_prompt", "")[:200] + "..." if len(getattr(agent, "persona_prompt", "")) > 200 else getattr(agent, "persona_prompt", "")
        base_info["beliefs"] = agent._beliefs
        base_info["belief_count"] = len(agent._beliefs)
        base_info["knowledge_items"] = len(agent._knowledge_base)
        base_info["interactions"] = agent._interactions
        base_info["collaborators"] = list(agent._collaborators)
        base_info["collaborator_count"] = len(agent._collaborators)
        base_info["conversation_length"] = len(agent._conversation_history)
        base_info["capabilities"] = agent._extract_capabilities()
        # Recent conversations
        base_info["recent_conversations"] = agent._conversation_history[-6:]

    elif state.agent_type == "worker":
        stats = agent.get_stats()
        base_info.update(stats)
        base_info["work_types"] = agent.work_types
        base_info["active_task_ids"] = list(agent._active_tasks.keys())
        # Recent completed tasks
        base_info["recent_completed"] = agent._completed_tasks[-5:]
        base_info["recent_failed"] = agent._failed_tasks[-3:]

    return base_info


def get_agents_list(app: FastAPI) -> list[dict]:
    """Get list of all agents with detailed info."""
    orchestrator = app.state.orchestrator
    if not orchestrator or not orchestrator.supervisor:
        return []

    return [
        get_agent_details(state)
        for state in orchestrator.supervisor._agents.values()
    ]


def get_stats(app: FastAPI) -> dict:
    """Get system statistics."""
    orchestrator = app.state.orchestrator
    store = app.state.dataset_store
    stats = app.state.stats

    agents = get_agents_list(app)
    running = sum(1 for a in agents if a["status"] == "running")
    uptime = (datetime.now(timezone.utc) - app.state.start_time).total_seconds()

    storage_stats = store.get_stats()

    return {
        "total_agents": len(agents),
        "running_agents": running,
        "messages_sent": stats["messages_sent"],
        "messages_received": sum(a["processed"] for a in agents),
        "broadcasts": stats["broadcasts"],
        "multicasts": stats["multicasts"],
        "direct_messages": stats["direct_messages"],
        "errors": stats["errors"],
        "datasets": storage_stats["datasets"],
        "uptime_seconds": uptime,
    }


async def process_chat(app: FastAPI, agent_id: str, message: str) -> str:
    """Process a chat message with an agent."""
    orchestrator = app.state.orchestrator

    if not orchestrator or not orchestrator.supervisor:
        return "System not initialized. Please wait for agents to start."

    agents = orchestrator.supervisor._agents
    if agent_id not in agents:
        return f"Agent {agent_id} not found."

    state = agents[agent_id]
    agent = state.agent

    # Simple response based on agent type
    if state.agent_type == "demiurge":
        return f"[Demiurge] I am the orchestrator of this cosmos. Your query '{message}' has been noted in the chronicle."
    elif state.agent_type == "persona":
        return f"[Persona {agent_id}] Processing your message: '{message}'. Based on my expertise, I can assist with this query."
    elif state.agent_type == "worker":
        return f"[Worker {agent_id}] Task acknowledged: '{message}'. Ready to execute work."
    else:
        return f"[{state.agent_type}] Message received: {message}"


async def broadcast_updates(app: FastAPI):
    """Background task to broadcast periodic updates."""
    manager = app.state.ws_manager

    while True:
        await asyncio.sleep(2)  # Update every 2 seconds

        if manager.active_connections:
            try:
                await manager.broadcast_agent_update(get_agents_list(app))
                await manager.broadcast_stats(get_stats(app))
            except Exception as e:
                logger.error(f"Broadcast error: {e}")


def get_dashboard_html() -> str:
    """Return inline dashboard HTML with high data density."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LIDA Multi-Agent System</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/vue@3/dist/vue.global.js"></script>
    <style>
        :root { --bg-dark: #0a0a0f; --bg-card: #12121a; --bg-hover: #1a1a25; --accent: #6366f1; }
        body { background: var(--bg-dark); font-family: 'JetBrains Mono', 'SF Mono', monospace; }
        .card { background: var(--bg-card); border: 1px solid #1e1e2e; }
        .card:hover { border-color: #2e2e4e; }
        .metric { font-variant-numeric: tabular-nums; }
        .status-running { color: #22c55e; } .status-error { color: #ef4444; }
        .status-starting { color: #f59e0b; } .status-terminated { color: #6b7280; }
        .tag { font-size: 9px; padding: 1px 4px; border-radius: 2px; }
        .tag-demiurge { background: #7c3aed20; color: #a78bfa; border: 1px solid #7c3aed40; }
        .tag-persona { background: #0ea5e920; color: #38bdf8; border: 1px solid #0ea5e940; }
        .tag-worker { background: #f5920b20; color: #fbbf24; border: 1px solid #f59e0b40; }
        .scroll-thin::-webkit-scrollbar { width: 4px; } .scroll-thin::-webkit-scrollbar-thumb { background: #333; border-radius: 2px; }
        .belief-bar { height: 3px; background: #1e1e2e; border-radius: 1px; }
        .belief-fill { height: 100%; border-radius: 1px; transition: width 0.3s; }
        .pulse { animation: pulse 2s infinite; } @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .grid-dense { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 8px; }
        .kv { display: flex; justify-content: space-between; font-size: 11px; padding: 2px 0; border-bottom: 1px solid #1a1a25; }
        .kv:last-child { border: none; }
        .kv-key { color: #6b7280; } .kv-val { color: #e5e7eb; font-weight: 500; }
        .line-clamp-2 { display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
        .line-clamp-3 { display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
    </style>
</head>
<body class="text-gray-300 min-h-screen p-3">
<div id="app">
    <!-- Header Row -->
    <div class="flex items-center justify-between mb-3">
        <div class="flex items-center gap-3">
            <h1 class="text-lg font-bold text-white">LIDA</h1>
            <span class="text-xs text-gray-500">Multi-Agent Orchestration</span>
        </div>
        <div class="flex items-center gap-4 text-xs">
            <div class="flex gap-3">
                <span class="text-gray-500">Agents <span class="text-white metric">{{ stats.total_agents }}</span></span>
                <span class="text-gray-500">Running <span class="text-green-400 metric">{{ stats.running_agents }}</span></span>
                <span class="text-gray-500">Msgs <span class="text-blue-400 metric">{{ stats.messages_received }}</span></span>
                <span class="text-gray-500">Uptime <span class="text-yellow-400 metric">{{ formatUptime(stats.uptime_seconds) }}</span></span>
            </div>
            <div class="flex items-center gap-1">
                <div class="w-1.5 h-1.5 rounded-full" :class="connected ? 'bg-green-500 pulse' : 'bg-red-500'"></div>
                <span class="text-gray-500">{{ connected ? 'live' : 'offline' }}</span>
            </div>
        </div>
    </div>

    <!-- Main Layout -->
    <div class="flex gap-3" style="height: calc(100vh - 60px);">
        <!-- Left: Agents Grid -->
        <div class="flex-1 overflow-y-auto scroll-thin pr-1">
            <div class="grid-dense">
                <!-- Agent Cards -->
                <div v-for="agent in agents" :key="agent.agent_id"
                     class="card rounded p-3 cursor-pointer transition-all"
                     :class="{'ring-1 ring-indigo-500': selectedAgent?.agent_id === agent.agent_id}"
                     @click="selectAgent(agent)">
                    <!-- Header -->
                    <div class="flex items-center justify-between mb-2">
                        <div class="flex items-center gap-2">
                            <span class="tag" :class="'tag-' + agent.agent_type">{{ agent.agent_type.toUpperCase() }}</span>
                            <span class="text-xs font-medium text-white">{{ agent.agent_id }}</span>
                        </div>
                        <span class="text-xs" :class="'status-' + agent.status">●</span>
                    </div>

                    <!-- Metrics Row -->
                    <div class="flex gap-2 text-xs mb-2">
                        <div class="flex-1 bg-gray-900 rounded px-2 py-1">
                            <div class="text-gray-500">in</div>
                            <div class="text-white metric">{{ agent.inbox }}</div>
                        </div>
                        <div class="flex-1 bg-gray-900 rounded px-2 py-1">
                            <div class="text-gray-500">out</div>
                            <div class="text-white metric">{{ agent.outbox }}</div>
                        </div>
                        <div class="flex-1 bg-gray-900 rounded px-2 py-1">
                            <div class="text-gray-500">proc</div>
                            <div class="text-green-400 metric">{{ agent.processed }}</div>
                        </div>
                    </div>

                    <!-- Type-specific content -->
                    <!-- Demiurge -->
                    <template v-if="agent.agent_type === 'demiurge'">
                        <!-- World State Overview -->
                        <div class="mb-2 bg-gradient-to-r from-violet-900/30 to-purple-900/20 rounded p-2 border border-violet-800/30">
                            <div class="text-[10px] text-violet-300 uppercase tracking-wider mb-1">Cosmic State</div>
                            <div class="grid grid-cols-3 gap-1">
                                <div class="text-center">
                                    <div class="text-white font-bold text-sm">{{ agent.world_state?.forms || 0 }}</div>
                                    <div class="text-[9px] text-gray-500">Forms</div>
                                </div>
                                <div class="text-center">
                                    <div class="text-white font-bold text-sm">{{ agent.world_state?.laws || 0 }}</div>
                                    <div class="text-[9px] text-gray-500">Laws</div>
                                </div>
                                <div class="text-center">
                                    <div class="text-white font-bold text-sm">{{ agent.world_state?.observers || 0 }}</div>
                                    <div class="text-[9px] text-gray-500">Observers</div>
                                </div>
                            </div>
                        </div>
                        <!-- Stats Grid -->
                        <div class="grid grid-cols-2 gap-1 text-xs mb-2">
                            <div class="bg-gray-900/50 rounded px-2 py-1">
                                <div class="text-gray-500 text-[10px]">subordinates</div>
                                <div class="text-white metric">{{ agent.subordinate_count || 0 }}</div>
                            </div>
                            <div class="bg-gray-900/50 rounded px-2 py-1">
                                <div class="text-gray-500 text-[10px]">chronicle</div>
                                <div class="text-white metric">{{ agent.world_state?.chronicle_entries || 0 }}</div>
                            </div>
                            <div class="bg-gray-900/50 rounded px-2 py-1">
                                <div class="text-gray-500 text-[10px]">deliberation</div>
                                <div class="text-orange-400 metric">{{ agent.deliberation_pending || 0 }}</div>
                            </div>
                            <div class="bg-gray-900/50 rounded px-2 py-1">
                                <div class="text-gray-500 text-[10px]">affordances</div>
                                <div class="text-white metric">{{ agent.world_state?.affordances || 0 }}</div>
                            </div>
                        </div>
                        <!-- Subordinates -->
                        <div v-if="agent.subordinates?.length" class="mb-2">
                            <div class="text-[10px] text-gray-500 mb-1">Active Subordinates:</div>
                            <div class="flex flex-wrap gap-1">
                                <span v-for="sub in agent.subordinates.slice(0,6)" class="text-[9px] bg-violet-900/30 text-violet-300 px-1.5 py-0.5 rounded">{{ sub }}</span>
                                <span v-if="agent.subordinates.length > 6" class="text-[9px] text-gray-500">+{{ agent.subordinates.length - 6 }}</span>
                            </div>
                        </div>
                        <!-- Recent Events -->
                        <div v-if="agent.recent_events?.length">
                            <div class="text-[10px] text-gray-500 mb-1">Recent Chronicle:</div>
                            <div class="space-y-0.5 max-h-16 overflow-y-auto scroll-thin">
                                <div v-for="(evt, i) in agent.recent_events.slice(-3)" :key="i" class="text-[9px] text-gray-400 bg-gray-900/30 rounded px-1.5 py-0.5 truncate">{{ typeof evt === 'string' ? evt : JSON.stringify(evt).slice(0,60) }}...</div>
                            </div>
                        </div>
                    </template>

                    <!-- Persona -->
                    <template v-if="agent.agent_type === 'persona'">
                        <!-- Prompt Info -->
                        <div v-if="agent.prompt_category" class="mb-2 bg-gradient-to-r from-cyan-900/30 to-blue-900/20 rounded p-2 border border-cyan-800/30">
                            <div class="flex items-center gap-1 mb-1">
                                <span class="text-[10px] bg-cyan-500/20 text-cyan-300 px-1.5 py-0.5 rounded uppercase tracking-wider">{{ agent.prompt_category }}</span>
                                <span v-if="agent.prompt_subcategory" class="text-[10px] bg-purple-500/20 text-purple-300 px-1.5 py-0.5 rounded">{{ agent.prompt_subcategory }}</span>
                            </div>
                            <div v-if="agent.persona_prompt" class="text-[10px] text-gray-400 leading-relaxed line-clamp-2">{{ agent.persona_prompt }}</div>
                            <div v-if="agent.prompt_id" class="text-[9px] text-gray-600 mt-1">Prompt #{{ agent.prompt_id }}</div>
                        </div>
                        <!-- Stats Grid -->
                        <div class="grid grid-cols-2 gap-1 text-xs mb-2">
                            <div class="bg-gray-900/50 rounded px-2 py-1">
                                <div class="text-gray-500 text-[10px]">interactions</div>
                                <div class="text-white metric">{{ agent.interactions || 0 }}</div>
                            </div>
                            <div class="bg-gray-900/50 rounded px-2 py-1">
                                <div class="text-gray-500 text-[10px]">collaborators</div>
                                <div class="text-white metric">{{ agent.collaborator_count || 0 }}</div>
                            </div>
                            <div class="bg-gray-900/50 rounded px-2 py-1">
                                <div class="text-gray-500 text-[10px]">knowledge</div>
                                <div class="text-white metric">{{ agent.knowledge_items || 0 }}</div>
                            </div>
                            <div class="bg-gray-900/50 rounded px-2 py-1">
                                <div class="text-gray-500 text-[10px]">messages</div>
                                <div class="text-white metric">{{ agent.conversation_length || 0 }}</div>
                            </div>
                        </div>
                        <!-- Capabilities -->
                        <div v-if="agent.capabilities?.length" class="mb-2">
                            <div class="text-[10px] text-gray-500 mb-1">Capabilities:</div>
                            <div class="flex flex-wrap gap-1">
                                <span v-for="cap in agent.capabilities.slice(0,6)" class="text-[9px] bg-indigo-900/30 text-indigo-300 px-1.5 py-0.5 rounded">{{ cap }}</span>
                                <span v-if="agent.capabilities.length > 6" class="text-[9px] text-gray-500">+{{ agent.capabilities.length - 6 }}</span>
                            </div>
                        </div>
                        <!-- Beliefs -->
                        <div v-if="agent.beliefs && Object.keys(agent.beliefs).length" class="mb-2">
                            <div class="text-[10px] text-gray-500 mb-1">Beliefs:</div>
                            <div class="space-y-1">
                                <div v-for="(conf, topic) in agent.beliefs" :key="topic" class="flex items-center gap-2">
                                    <span class="text-[10px] text-gray-400 truncate flex-1">{{ topic }}</span>
                                    <div class="belief-bar w-12">
                                        <div class="belief-fill" :style="{width: (conf*100)+'%', background: conf > 0.7 ? '#22c55e' : conf > 0.4 ? '#f59e0b' : '#ef4444'}"></div>
                                    </div>
                                    <span class="text-[9px] text-gray-500 w-6 text-right">{{ (conf*100).toFixed(0) }}%</span>
                                </div>
                            </div>
                        </div>
                        <!-- Collaborators -->
                        <div v-if="agent.collaborators?.length">
                            <div class="text-[10px] text-gray-500 mb-1">Working with:</div>
                            <div class="flex flex-wrap gap-1">
                                <span v-for="c in agent.collaborators.slice(0,4)" class="text-[9px] bg-blue-900/30 text-blue-300 px-1.5 py-0.5 rounded">{{ c }}</span>
                                <span v-if="agent.collaborators.length > 4" class="text-[9px] text-gray-500">+{{ agent.collaborators.length - 4 }}</span>
                            </div>
                        </div>
                    </template>

                    <!-- Worker -->
                    <template v-if="agent.agent_type === 'worker'">
                        <div class="text-xs space-y-1">
                            <div class="kv"><span class="kv-key">Capacity</span><span class="kv-val">{{ agent.active_tasks || 0 }} / {{ agent.capacity || 5 }}</span></div>
                            <div class="kv"><span class="kv-key">Completed</span><span class="kv-val text-green-400">{{ agent.completed_tasks || 0 }}</span></div>
                            <div class="kv"><span class="kv-key">Failed</span><span class="kv-val text-red-400">{{ agent.failed_tasks || 0 }}</span></div>
                            <div class="kv"><span class="kv-key">Avg Time</span><span class="kv-val">{{ (agent.average_time || 0).toFixed(2) }}s</span></div>
                        </div>
                        <!-- Work Types -->
                        <div class="mt-2">
                            <div class="text-xs text-gray-500 mb-1">Work types:</div>
                            <div class="flex flex-wrap gap-1">
                                <span v-for="wt in agent.work_types" class="text-xs bg-yellow-900/30 text-yellow-300 px-1.5 py-0.5 rounded">{{ wt }}</span>
                            </div>
                        </div>
                        <!-- Utilization bar -->
                        <div class="mt-2">
                            <div class="flex justify-between text-xs mb-1">
                                <span class="text-gray-500">Utilization</span>
                                <span class="text-gray-400">{{ ((agent.utilization || 0) * 100).toFixed(0) }}%</span>
                            </div>
                            <div class="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                                <div class="h-full rounded-full transition-all"
                                     :style="{width: ((agent.utilization || 0) * 100) + '%'}"
                                     :class="agent.utilization > 0.8 ? 'bg-red-500' : agent.utilization > 0.5 ? 'bg-yellow-500' : 'bg-green-500'"></div>
                            </div>
                        </div>
                        <!-- Active Tasks -->
                        <div v-if="agent.active_task_ids?.length" class="mt-2">
                            <div class="text-xs text-gray-500 mb-1">Active tasks:</div>
                            <div class="space-y-0.5">
                                <div v-for="tid in agent.active_task_ids" class="text-xs text-orange-300 bg-orange-900/20 px-1.5 py-0.5 rounded truncate">{{ tid }}</div>
                            </div>
                        </div>
                    </template>
                </div>
            </div>
        </div>

        <!-- Right Panel -->
        <div class="w-80 flex flex-col gap-3">
            <!-- Chat Panel -->
            <div class="card rounded p-3 flex-1 flex flex-col min-h-0">
                <div class="flex items-center justify-between mb-2">
                    <span class="text-xs font-medium text-white">Chat</span>
                    <span v-if="selectedAgent" class="text-xs text-gray-500">{{ selectedAgent.agent_id }}</span>
                </div>
                <div class="flex-1 overflow-y-auto scroll-thin space-y-2 mb-2" ref="chatContainer">
                    <div v-for="msg in chatMessages" :key="msg.id"
                         class="text-xs p-2 rounded"
                         :class="msg.role === 'user' ? 'bg-gray-800 text-gray-300' : 'bg-indigo-900/30 text-indigo-200'">
                        <div class="text-gray-500 text-[10px] mb-0.5">{{ msg.role }}</div>
                        {{ msg.content }}
                    </div>
                    <div v-if="!selectedAgent" class="text-xs text-gray-600 text-center py-4">Select an agent to chat</div>
                </div>
                <div class="flex gap-1">
                    <input v-model="chatInput" @keyup.enter="sendChat" :disabled="!selectedAgent"
                           placeholder="Message..." class="flex-1 bg-gray-800 rounded px-2 py-1.5 text-xs focus:outline-none focus:ring-1 focus:ring-indigo-500">
                    <button @click="sendChat" :disabled="!selectedAgent || !chatInput"
                            class="bg-indigo-600 hover:bg-indigo-700 px-3 py-1.5 rounded text-xs disabled:opacity-50">Send</button>
                </div>
            </div>

            <!-- Message Log -->
            <div class="card rounded p-3 h-48 flex flex-col">
                <div class="text-xs font-medium text-white mb-2">Message Log</div>
                <div class="flex-1 overflow-y-auto scroll-thin space-y-1">
                    <div v-for="msg in messageLog" :key="msg.msg_id" class="text-[10px] bg-gray-800/50 rounded px-2 py-1">
                        <span class="text-cyan-400">{{ msg.sender_id }}</span>
                        <span class="text-gray-600"> → </span>
                        <span class="text-green-400">{{ msg.recipient_id }}</span>
                        <span class="text-gray-500 ml-1">[{{ msg.msg_type }}]</span>
                    </div>
                    <div v-if="!messageLog.length" class="text-xs text-gray-600 text-center py-2">No messages</div>
                </div>
            </div>

            <!-- Prompts Browser -->
            <div class="card rounded p-3 flex flex-col" style="max-height: 200px;">
                <div class="flex items-center justify-between mb-2">
                    <span class="text-xs font-medium text-white">Prompts Library</span>
                    <span class="text-[10px] text-gray-500">{{ promptStats.total || 880 }} total</span>
                </div>
                <!-- Category Filter -->
                <div class="flex flex-wrap gap-1 mb-2">
                    <button @click="promptFilter = ''"
                            class="text-[9px] px-1.5 py-0.5 rounded"
                            :class="promptFilter === '' ? 'bg-indigo-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'">All</button>
                    <button v-for="cat in promptCategories" :key="cat" @click="promptFilter = cat"
                            class="text-[9px] px-1.5 py-0.5 rounded"
                            :class="promptFilter === cat ? 'bg-indigo-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'">{{ cat }}</button>
                </div>
                <!-- Prompts List -->
                <div class="flex-1 overflow-y-auto scroll-thin space-y-1 min-h-0">
                    <div v-for="p in filteredPrompts.slice(0,8)" :key="p.id"
                         class="text-[10px] bg-gray-800/50 rounded px-2 py-1 cursor-pointer hover:bg-gray-700/50"
                         @click="viewPrompt(p)">
                        <div class="flex items-center gap-1 mb-0.5">
                            <span class="text-gray-500">#{{ p.id }}</span>
                            <span class="text-cyan-400">{{ p.subcategory || p.category }}</span>
                        </div>
                        <div class="text-gray-400 line-clamp-2">{{ p.text.slice(0, 80) }}...</div>
                    </div>
                </div>
            </div>

            <!-- Datasets -->
            <div class="card rounded p-3">
                <div class="text-xs font-medium text-white mb-2">Datasets ({{ datasets.length }})</div>
                <div class="space-y-1 max-h-20 overflow-y-auto scroll-thin mb-2">
                    <div v-for="ds in datasets" :key="ds.id" class="flex justify-between text-xs bg-gray-800/50 rounded px-2 py-1">
                        <span class="text-gray-300">{{ ds.name }}</span>
                        <span class="text-gray-500">{{ ds.record_count }} rec</span>
                    </div>
                </div>
                <div class="flex gap-1">
                    <input v-model="newDatasetName" placeholder="New dataset..."
                           class="flex-1 bg-gray-800 rounded px-2 py-1 text-xs focus:outline-none">
                    <button @click="createDataset" :disabled="!newDatasetName"
                            class="bg-purple-600 hover:bg-purple-700 px-2 py-1 rounded text-xs disabled:opacity-50">+</button>
                </div>
            </div>
        </div>
    </div>

    <!-- Prompt Detail Modal -->
    <div v-if="selectedPrompt" class="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4" @click.self="selectedPrompt = null">
        <div class="card rounded-lg p-4 max-w-2xl w-full max-h-[80vh] overflow-y-auto">
            <div class="flex items-center justify-between mb-3">
                <div class="flex items-center gap-2">
                    <span class="text-xs bg-cyan-500/20 text-cyan-300 px-2 py-1 rounded">{{ selectedPrompt.category }}</span>
                    <span class="text-xs bg-purple-500/20 text-purple-300 px-2 py-1 rounded">{{ selectedPrompt.subcategory }}</span>
                    <span class="text-xs text-gray-500">Prompt #{{ selectedPrompt.id }}</span>
                </div>
                <button @click="selectedPrompt = null" class="text-gray-500 hover:text-white text-lg">&times;</button>
            </div>
            <div class="text-sm text-gray-300 leading-relaxed whitespace-pre-wrap">{{ selectedPrompt.text }}</div>
            <div v-if="selectedPrompt.tags?.length" class="mt-3 flex flex-wrap gap-1">
                <span v-for="tag in selectedPrompt.tags" class="text-[10px] bg-gray-700 text-gray-300 px-2 py-0.5 rounded">{{ tag }}</span>
            </div>
        </div>
    </div>
</div>

<script>
const { createApp, ref, computed, onMounted, onUnmounted, nextTick } = Vue;
createApp({
    setup() {
        const connected = ref(false);
        const agents = ref([]);
        const stats = ref({ total_agents: 0, running_agents: 0, messages_sent: 0, messages_received: 0, datasets: 0, uptime_seconds: 0 });
        const messageLog = ref([]);
        const chatMessages = ref([]);
        const selectedAgent = ref(null);
        const chatInput = ref('');
        const chatContainer = ref(null);
        const datasets = ref([]);
        const newDatasetName = ref('');
        const prompts = ref([]);
        const promptStats = ref({ total: 0, categories: {}, subcategories: {} });
        const promptFilter = ref('');
        const selectedPrompt = ref(null);
        const promptCategories = ['humanities', 'social_sciences', 'natural_sciences', 'formal_sciences', 'underground', 'economic_extremes', 'esoteric'];
        let ws = null;

        const connect = () => {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            ws.onopen = () => { connected.value = true; };
            ws.onclose = () => { connected.value = false; setTimeout(connect, 3000); };
            ws.onmessage = (e) => handleMessage(JSON.parse(e.data));
        };

        const handleMessage = (data) => {
            if (data.type === 'init') {
                agents.value = data.data.agents || [];
                Object.assign(stats.value, data.data.stats || {});
            } else if (data.type === 'agents_update') {
                agents.value = data.data || [];
                // Update selected agent if exists
                if (selectedAgent.value) {
                    const updated = agents.value.find(a => a.agent_id === selectedAgent.value.agent_id);
                    if (updated) selectedAgent.value = updated;
                }
            } else if (data.type === 'stats_update') {
                Object.assign(stats.value, data.data || {});
            } else if (data.type === 'message_event') {
                messageLog.value.unshift(data.data);
                if (messageLog.value.length > 30) messageLog.value.pop();
            } else if (data.type === 'chat_message' && selectedAgent.value?.agent_id === data.data.agent_id) {
                chatMessages.value.push({ id: Date.now(), role: data.data.role, content: data.data.content });
                scrollChat();
            }
        };

        const selectAgent = async (agent) => {
            selectedAgent.value = agent;
            chatMessages.value = [];
            try {
                const resp = await fetch(`/api/chat/${agent.agent_id}/history`);
                const data = await resp.json();
                chatMessages.value = data.messages || [];
                scrollChat();
            } catch (e) { console.error(e); }
        };

        const sendChat = async () => {
            if (!selectedAgent.value || !chatInput.value.trim()) return;
            try {
                await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ agent_id: selectedAgent.value.agent_id, message: chatInput.value })
                });
                chatInput.value = '';
            } catch (e) { console.error(e); }
        };

        const scrollChat = () => nextTick(() => { if (chatContainer.value) chatContainer.value.scrollTop = chatContainer.value.scrollHeight; });

        const loadDatasets = async () => {
            try {
                const resp = await fetch('/api/datasets');
                datasets.value = (await resp.json()).datasets || [];
            } catch (e) { console.error(e); }
        };

        const createDataset = async () => {
            if (!newDatasetName.value.trim()) return;
            try {
                await fetch('/api/datasets', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: newDatasetName.value }) });
                newDatasetName.value = '';
                loadDatasets();
            } catch (e) { console.error(e); }
        };

        const loadPrompts = async () => {
            try {
                const [promptsResp, statsResp] = await Promise.all([
                    fetch('/api/prompts?limit=100'),
                    fetch('/api/prompts/categories')
                ]);
                const promptsData = await promptsResp.json();
                const statsData = await statsResp.json();
                prompts.value = promptsData.prompts || [];
                promptStats.value = statsData;
            } catch (e) { console.error(e); }
        };

        const filteredPrompts = computed(() => {
            if (!promptFilter.value) return prompts.value;
            return prompts.value.filter(p => p.category === promptFilter.value);
        });

        const viewPrompt = async (p) => {
            try {
                const resp = await fetch(`/api/prompts/${p.id}`);
                selectedPrompt.value = await resp.json();
            } catch (e) {
                selectedPrompt.value = p;
            }
        };

        const formatUptime = (s) => {
            if (!s) return '0s';
            const h = Math.floor(s/3600), m = Math.floor((s%3600)/60), sec = Math.floor(s%60);
            return h > 0 ? `${h}h${m}m` : m > 0 ? `${m}m${sec}s` : `${sec}s`;
        };

        onMounted(() => { connect(); loadDatasets(); loadPrompts(); });
        onUnmounted(() => { if (ws) ws.close(); });

        return { connected, agents, stats, messageLog, chatMessages, selectedAgent, chatInput, chatContainer, datasets, newDatasetName, prompts, promptStats, promptFilter, selectedPrompt, promptCategories, filteredPrompts, selectAgent, sendChat, createDataset, loadPrompts, viewPrompt, formatUptime };
    }
}).mount('#app');
</script>
</body>
</html>"""
