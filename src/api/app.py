"""FastAPI application for multi-agent system."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    SpawnAgentRequest,
    ChatMessageRequest,
    StreamingChatRequest,
    SignatureRequest,
    ExecuteSignatureRequest,
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
    # Streaming LLM routes
    # ─────────────────────────────────────────────────────────────────────────

    @app.post("/api/chat/stream")
    async def stream_chat(request: StreamingChatRequest):
        """Stream a chat response from an agent using LLM."""
        try:
            from ..llm import OpenRouterClient, SignatureBuilder, DSPyModule

            orchestrator = app.state.orchestrator
            if not orchestrator or not orchestrator.supervisor:
                raise HTTPException(400, "Orchestrator not initialized")

            agents = orchestrator.supervisor._agents
            if request.agent_id not in agents:
                raise HTTPException(404, f"Agent {request.agent_id} not found")

            state = agents[request.agent_id]
            agent = state.agent

            # Build persona-specific signature
            persona_prompt = getattr(agent, 'persona_prompt', None) or f"You are {request.agent_id}"

            sig = (
                SignatureBuilder("PersonaChat")
                .describe(f"You are embodying this persona:\n{persona_prompt[:2000]}")
                .input("message", "User message to respond to")
                .output("response", "Your response as this persona")
                .build()
            )

            client = OpenRouterClient()
            module = DSPyModule(sig, client=client, model=request.model or "anthropic/claude-sonnet-4.5")

            async def generate():
                async for chunk in module.stream(message=request.message):
                    yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        except ImportError:
            raise HTTPException(500, "LLM module not available. Install httpx.")
        except Exception as e:
            logger.error(f"Stream chat error: {e}")
            raise HTTPException(500, str(e))

    @app.post("/api/signatures")
    async def create_signature(request: SignatureRequest):
        """Create a dynamic signature for structured LLM calls."""
        try:
            from ..llm import SignatureBuilder, Field

            builder = SignatureBuilder(request.name)
            if request.description:
                builder.describe(request.description)
            if request.instructions:
                builder.instruct(request.instructions)

            for inp in request.inputs:
                builder.input(
                    name=inp.get("name", "input"),
                    description=inp.get("description", ""),
                    field_type=inp.get("type", "str"),
                    required=inp.get("required", True),
                )

            for out in request.outputs:
                builder.output(
                    name=out.get("name", "output"),
                    description=out.get("description", ""),
                    field_type=out.get("type", "str"),
                    required=out.get("required", True),
                )

            sig = builder.build()

            # Store in app state
            if not hasattr(app.state, 'signatures'):
                app.state.signatures = {}
            app.state.signatures[request.name] = sig

            return {
                "name": sig.name,
                "description": sig.description,
                "input_schema": sig.input_schema(),
                "output_schema": sig.output_schema(),
                "system_prompt": sig.to_system_prompt(),
            }

        except Exception as e:
            raise HTTPException(400, str(e))

    @app.get("/api/signatures")
    async def list_signatures():
        """List available signatures."""
        from ..llm.signatures import SIGNATURES

        predefined = list(SIGNATURES.keys())
        custom = list(getattr(app.state, 'signatures', {}).keys())

        return {
            "predefined": predefined,
            "custom": custom,
        }

    @app.post("/api/signatures/execute")
    async def execute_signature(request: ExecuteSignatureRequest):
        """Execute a signature with given inputs."""
        try:
            from ..llm import DSPyModule, get_signature

            # Try custom signature first
            custom_sigs = getattr(app.state, 'signatures', {})
            if request.signature_name in custom_sigs:
                sig = custom_sigs[request.signature_name]
            else:
                sig = get_signature(request.signature_name)

            module = DSPyModule(sig, model=request.model or "anthropic/claude-sonnet-4.5")

            if request.stream:
                async def generate():
                    async for chunk in module.stream(**request.inputs):
                        yield f"data: {chunk}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    generate(),
                    media_type="text/event-stream",
                )

            result = await module(**request.inputs)

            return {
                "signature": request.signature_name,
                "outputs": result.outputs,
                "raw_response": result.raw_response,
                "success": result.success,
                "error": result.error,
            }

        except ValueError as e:
            raise HTTPException(404, str(e))
        except Exception as e:
            logger.error(f"Signature execution error: {e}")
            raise HTTPException(500, str(e))

    @app.get("/api/llm/models")
    async def list_models():
        """List available LLM models."""
        from ..llm.openrouter import MODELS
        return {"models": MODELS}

    # ─────────────────────────────────────────────────────────────────────────
    # Cognitive Agent routes
    # ─────────────────────────────────────────────────────────────────────────

    @app.get("/api/cognitive/behaviors")
    async def list_cognitive_behaviors():
        """List available cognitive behaviors."""
        try:
            from ..llm import list_behaviors
            return {"behaviors": list_behaviors()}
        except ImportError:
            return {"behaviors": [], "error": "LLM module not available"}

    @app.post("/api/cognitive/reason")
    async def cognitive_reason(
        task: str = Query(..., description="Task to reason about"),
        model: str = Query("anthropic/claude-sonnet-4.5", description="Model to use"),
        behaviors: Optional[str] = Query(None, description="Comma-separated behaviors to use"),
    ):
        """Execute cognitive reasoning on a task."""
        try:
            from ..llm import create_cognitive_agent

            behavior_list = behaviors.split(",") if behaviors else None
            agent = create_cognitive_agent(model=model, behaviors=behavior_list)
            result = await agent.reason(task)

            return {
                "task": result["task"],
                "trace": result["trace"],
                "synthesis": result["synthesis"],
                "context_summary": result["context_summary"],
            }

        except ImportError:
            raise HTTPException(500, "LLM module not available")
        except Exception as e:
            logger.error(f"Cognitive reason error: {e}")
            raise HTTPException(500, str(e))

    @app.post("/api/cognitive/reason/stream")
    async def cognitive_reason_stream(
        task: str = Query(..., description="Task to reason about"),
        model: str = Query("anthropic/claude-sonnet-4.5", description="Model to use"),
    ):
        """Stream cognitive reasoning process."""
        try:
            from ..llm import create_cognitive_agent
            import json

            agent = create_cognitive_agent(model=model)

            async def generate():
                async for event in agent.stream_reason(task):
                    yield f"data: {json.dumps(event)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        except ImportError:
            raise HTTPException(500, "LLM module not available")
        except Exception as e:
            logger.error(f"Cognitive stream error: {e}")
            raise HTTPException(500, str(e))

    @app.post("/api/cognitive/quick")
    async def cognitive_quick(
        task: str = Query(..., description="Task to reason about"),
        model: str = Query("anthropic/claude-sonnet-4.5", description="Model to use"),
    ):
        """Quick reasoning without full behavior execution."""
        try:
            from ..llm import create_cognitive_agent

            agent = create_cognitive_agent(model=model)
            result = await agent.quick_reason(task)

            return {"task": task, "result": result}

        except ImportError:
            raise HTTPException(500, "LLM module not available")
        except Exception as e:
            logger.error(f"Quick reason error: {e}")
            raise HTTPException(500, str(e))

    @app.post("/api/cognitive/behavior/{behavior_name}")
    async def execute_behavior(
        behavior_name: str,
        model: str = Query("anthropic/claude-sonnet-4.5", description="Model to use"),
        task: str = Query(None, description="Task for context"),
        claim: str = Query(None, description="Claim to verify (for verification)"),
        observation: str = Query(None, description="Observation (for hypothesis)"),
        goal: str = Query(None, description="Goal (for backward chaining)"),
    ):
        """Execute a specific cognitive behavior."""
        try:
            from ..llm import get_behavior, AgentContext

            behavior = get_behavior(behavior_name, model=model)
            context = AgentContext(task=task or "")
            context.start_trace(task or "behavior execution")

            # Build kwargs based on behavior type
            kwargs = {}
            if behavior_name == "verification" and claim:
                kwargs["claim"] = claim
            elif behavior_name == "hypothesis_generation" and observation:
                kwargs["observation"] = observation
            elif behavior_name == "backward_chaining" and goal:
                kwargs["goal"] = goal
            elif task:
                kwargs["task"] = task

            result = await behavior.execute(context, **kwargs)

            return {
                "behavior": behavior_name,
                "result": result,
                "trace": context.current_trace.to_dict() if context.current_trace else None,
            }

        except ValueError as e:
            raise HTTPException(404, str(e))
        except ImportError:
            raise HTTPException(500, "LLM module not available")
        except Exception as e:
            logger.error(f"Behavior execution error: {e}")
            raise HTTPException(500, str(e))

    @app.get("/api/cognitive/plans")
    async def list_execution_plans(depth: int = Query(2, ge=1, le=3)):
        """List possible execution plans for cognitive reasoning."""
        try:
            from ..llm import CartesianProductPlanner

            planner = CartesianProductPlanner()
            plans = planner.generate_plans(depth)

            return {
                "plans": [
                    {
                        "name": p.name,
                        "description": p.description,
                        "steps": [s.behavior for s in p.steps],
                    }
                    for p in plans
                ],
                "conditional_plans": [
                    {
                        "name": p.name,
                        "steps": [
                            {
                                "behavior": s.behavior,
                                "condition": s.condition,
                                "fallback": s.fallback,
                            }
                            for s in p.steps
                        ],
                    }
                    for p in planner.generate_conditional_plans()
                ],
            }

        except ImportError:
            raise HTTPException(500, "LLM module not available")

    # ─────────────────────────────────────────────────────────────────────────
    # Event System routes
    # ─────────────────────────────────────────────────────────────────────────

    @app.get("/api/events/world-state")
    async def get_world_state_endpoint():
        """Get current world state."""
        from ..events import get_world_state
        state = get_world_state()
        return state.to_dict()

    @app.get("/api/events/world-state/{path:path}")
    async def get_world_state_path(path: str):
        """Get a specific path in world state."""
        from ..events import get_world_state
        state = get_world_state()
        value = state.get(path)
        return {"path": path, "value": value}

    @app.post("/api/events/world-state")
    async def set_world_state(path: str = Query(...), value: Any = None):
        """Set a value in world state."""
        from ..events import get_world_state
        state = get_world_state()
        state.set(path, value)
        return {"path": path, "value": value, "version": state._version}

    @app.post("/api/events/world-event")
    async def publish_world_event(
        title: str = Query(...),
        description: str = Query(""),
        category: str = Query("observation"),
        significance: float = Query(0.5, ge=0, le=1),
        domain: Optional[str] = Query(None),
        content: Optional[str] = Query(None),
    ):
        """Publish a world event and process effects."""
        from ..events import (
            get_world_state, get_event_bus,
            WorldEvent, WorldEventCategory
        )
        import json

        # Parse category
        try:
            cat = WorldEventCategory(category)
        except ValueError:
            cat = WorldEventCategory.OBSERVATION

        # Parse content if JSON
        parsed_content = content
        if content:
            try:
                parsed_content = json.loads(content)
            except json.JSONDecodeError:
                pass

        # Create event
        event = WorldEvent(
            category=cat,
            title=title,
            description=description,
            significance=significance,
            domain=domain,
            content=parsed_content,
        )

        # Process through world state rules
        state = get_world_state()
        effects = await state.process_event(event)

        # Also publish to event bus
        bus = get_event_bus()
        await bus.publish(event)

        return {
            "event_id": event.event_id,
            "effects_applied": len(effects),
            "effects": [
                {"type": e.effect_type.value, "path": e.path}
                for e in effects
            ],
        }

    @app.get("/api/events/rules")
    async def list_effect_rules():
        """List all effect rules."""
        from ..events import get_world_state
        state = get_world_state()
        return {
            "rules": [
                {
                    "rule_id": r.rule_id,
                    "name": r.name,
                    "description": r.description,
                    "active": r.active,
                    "priority": r.priority,
                    "triggered_count": r.triggered_count,
                    "categories": [c.value for c in r.event_categories],
                }
                for r in state.get_rules()
            ]
        }

    @app.get("/api/events/history")
    async def get_event_history(
        limit: int = Query(50, ge=1, le=500),
        event_type: Optional[str] = Query(None),
    ):
        """Get event history from the event bus."""
        from ..events import get_event_bus, EventType

        bus = get_event_bus()

        # Parse event type
        et = None
        if event_type:
            try:
                et = EventType(event_type)
            except ValueError:
                pass

        events = bus.get_history(event_type=et, limit=limit)
        return {
            "events": [e.to_dict() for e in events],
            "count": len(events),
        }

    @app.get("/api/events/stats")
    async def get_event_stats():
        """Get event bus statistics."""
        from ..events import get_event_bus, get_world_state

        bus = get_event_bus()
        state = get_world_state()

        return {
            "bus": bus.get_stats(),
            "world_state": {
                "version": state._version,
                "rules_count": len(state._rules),
                "effects_applied": len(state._effects_applied),
                "history_snapshots": len(state._history),
            },
        }

    @app.post("/api/events/agent-event")
    async def publish_agent_event(
        agent_id: str = Query(...),
        agent_name: str = Query(""),
        category: str = Query("state_change"),
        action: str = Query(""),
        details: Optional[str] = Query(None),
    ):
        """Publish an internal agent event."""
        from ..events import get_event_bus, AgentEvent, AgentEventCategory
        import json

        # Parse category
        try:
            cat = AgentEventCategory(category)
        except ValueError:
            cat = AgentEventCategory.STATE_CHANGE

        # Parse details if JSON
        parsed_details = details
        if details:
            try:
                parsed_details = json.loads(details)
            except json.JSONDecodeError:
                pass

        event = AgentEvent(
            category=cat,
            agent_id=agent_id,
            agent_name=agent_name or agent_id,
            action=action,
            details=parsed_details,
        )

        bus = get_event_bus()
        await bus.publish(event)

        return {"event_id": event.event_id, "category": cat.value}

    @app.post("/api/events/message")
    async def send_agent_message(
        sender_id: str = Query(...),
        sender_name: str = Query(""),
        recipient_id: str = Query(""),
        recipient_name: str = Query(""),
        message_type: str = Query("inform"),
        subject: str = Query(""),
        content: str = Query(...),
    ):
        """Send a message between agents via event bus."""
        from ..events import get_event_bus, Message, MessageType
        import json

        # Parse message type
        try:
            mt = MessageType(message_type)
        except ValueError:
            mt = MessageType.INFORM

        # Parse content if JSON
        parsed_content = content
        try:
            parsed_content = json.loads(content)
        except json.JSONDecodeError:
            pass

        msg = Message(
            message_type=mt,
            sender_id=sender_id,
            sender_name=sender_name or sender_id,
            recipient_id=recipient_id,
            recipient_name=recipient_name,
            subject=subject,
            content=parsed_content,
        )

        bus = get_event_bus()
        await bus.publish(msg)

        return {
            "event_id": msg.event_id,
            "message_type": mt.value,
            "is_broadcast": not recipient_id,
        }

    @app.get("/api/events/categories")
    async def list_event_categories():
        """List all available event categories."""
        from ..events import WorldEventCategory, AgentEventCategory, MessageType

        return {
            "world_events": [c.value for c in WorldEventCategory],
            "agent_events": [c.value for c in AgentEventCategory],
            "message_types": [m.value for m in MessageType],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Meta-Capability System routes
    # ─────────────────────────────────────────────────────────────────────────

    @app.get("/api/meta/structures")
    async def list_meta_structures():
        """List all meta-structures in the environment."""
        from ..meta import get_environment
        env = get_environment()
        structures = env.get_population()
        return {
            "structures": [
                {
                    "id": s.structure_id,
                    "name": s.name,
                    "type": s.structure_type.value,
                    "capability_count": len(s.capabilities),
                    "sub_structure_count": len(s.sub_structures),
                    "version": s.version,
                }
                for s in structures
            ],
            "count": len(structures),
        }

    @app.get("/api/meta/structures/{structure_id}")
    async def get_meta_structure(structure_id: str):
        """Get details of a specific meta-structure."""
        from ..meta import get_environment
        env = get_environment()
        for s in env.get_population():
            if s.structure_id == structure_id:
                return {
                    "id": s.structure_id,
                    "name": s.name,
                    "type": s.structure_type.value,
                    "capabilities": [
                        {
                            "id": c.capability_id,
                            "name": c.name,
                            "type": c.capability_type.value,
                            "inputs": c.inputs,
                            "outputs": c.outputs,
                        }
                        for c in s.capabilities
                    ],
                    "sub_structures": [sub.structure_id for sub in s.sub_structures],
                    "emerged_from": s.emerged_from,
                    "parameters": s.parameters,
                    "version": s.version,
                    "reflection": s.reflect(),
                }
        raise HTTPException(404, f"Structure {structure_id} not found")

    @app.post("/api/meta/structures")
    async def create_meta_structure(
        name: str = Query(...),
        structure_type: str = Query("composite"),
    ):
        """Create a new meta-structure."""
        from ..meta import get_environment, StructureType
        env = get_environment()

        try:
            stype = StructureType(structure_type)
        except ValueError:
            raise HTTPException(400, f"Invalid structure type: {structure_type}")

        structure = env.create_structure(name=name, structure_type=stype)
        return {
            "id": structure.structure_id,
            "name": structure.name,
            "type": structure.structure_type.value,
        }

    @app.post("/api/meta/compose")
    async def compose_structures(
        structure_ids: str = Query(..., description="Comma-separated structure IDs"),
    ):
        """Compose multiple structures into one."""
        from ..meta import get_environment
        env = get_environment()

        ids = [s.strip() for s in structure_ids.split(",")]
        structures = []
        for sid in ids:
            found = None
            for s in env.get_population():
                if s.structure_id == sid:
                    found = s
                    break
            if not found:
                raise HTTPException(404, f"Structure {sid} not found")
            structures.append(found)

        if len(structures) < 2:
            raise HTTPException(400, "Need at least 2 structures to compose")

        composed = env.compose(*structures)
        return {
            "id": composed.structure_id,
            "name": composed.name,
            "type": composed.structure_type.value,
            "capability_count": len(composed.capabilities),
            "emerged_from": composed.emerged_from,
        }

    @app.post("/api/meta/mutate/{structure_id}")
    async def mutate_structure(
        structure_id: str,
        mutation_type: Optional[str] = Query(None),
    ):
        """Apply a mutation to a structure."""
        from ..meta import get_environment, MutationType
        env = get_environment()

        # Find structure
        structure = None
        for s in env.get_population():
            if s.structure_id == structure_id:
                structure = s
                break

        if not structure:
            raise HTTPException(404, f"Structure {structure_id} not found")

        mtype = None
        if mutation_type:
            try:
                mtype = MutationType(mutation_type)
            except ValueError:
                raise HTTPException(400, f"Invalid mutation type: {mutation_type}")

        mutated = env.mutate(structure, mtype)
        return {
            "original_id": structure_id,
            "mutated_id": mutated.structure_id,
            "name": mutated.name,
            "capability_count": len(mutated.capabilities),
            "version": mutated.version,
        }

    @app.get("/api/meta/hypergraph")
    async def get_hypergraph():
        """Get the capability hypergraph."""
        from ..meta import get_capability_graph
        graph = get_capability_graph()
        return graph.to_dict()

    @app.get("/api/meta/hypergraph/stats")
    async def get_hypergraph_stats():
        """Get hypergraph statistics."""
        from ..meta import get_capability_graph
        graph = get_capability_graph()
        return graph.get_stats()

    @app.get("/api/meta/hypergraph/node/{node_id}")
    async def get_hypergraph_node(node_id: str):
        """Get a specific node from the hypergraph."""
        from ..meta import get_capability_graph
        graph = get_capability_graph()
        node = graph.get_node(node_id)
        if not node:
            raise HTTPException(404, f"Node {node_id} not found")
        return {
            "id": node.node_id,
            "type": node.node_type,
            "name": node.name,
            "signature": node.signature,
            "tags": list(node.tags),
            "access_count": node.access_count,
        }

    @app.get("/api/meta/hypergraph/related/{node_id}")
    async def get_related_nodes(
        node_id: str,
        depth: int = Query(1, ge=1, le=5),
    ):
        """Find nodes related to a given node."""
        from ..meta import get_capability_graph
        graph = get_capability_graph()

        if not graph.get_node(node_id):
            raise HTTPException(404, f"Node {node_id} not found")

        related = graph.find_related(node_id, depth=depth)
        return {
            "source": node_id,
            "depth": depth,
            "related": list(related),
            "count": len(related),
        }

    @app.get("/api/meta/hypergraph/path")
    async def find_path(
        start: str = Query(...),
        end: str = Query(...),
        max_depth: int = Query(5, ge=1, le=10),
    ):
        """Find a path between two nodes."""
        from ..meta import get_capability_graph
        graph = get_capability_graph()

        path = graph.find_path(start, end, max_depth=max_depth)
        return {
            "start": start,
            "end": end,
            "path": path,
            "found": path is not None,
            "length": len(path) if path else 0,
        }

    @app.get("/api/meta/environment/stats")
    async def get_environment_stats():
        """Get constructive environment statistics."""
        from ..meta import get_environment
        env = get_environment()
        return env.get_stats()

    @app.post("/api/meta/discover")
    async def discover_emergent_capabilities():
        """Discover emergent capabilities from combinations."""
        from ..meta import get_environment
        env = get_environment()
        discovered = env.discover_emergent_capabilities()
        return {
            "discovered": [
                {
                    "id": s.structure_id,
                    "name": s.name,
                    "type": s.structure_type.value,
                    "emergence_conditions": s.emergence_conditions,
                }
                for s in discovered
            ],
            "count": len(discovered),
        }

    @app.post("/api/meta/generate")
    async def generate_capability(
        inputs: Optional[str] = Query(None, description="Comma-separated input types"),
        outputs: Optional[str] = Query(None, description="Comma-separated output types"),
        capability_types: Optional[str] = Query(None, description="Comma-separated capability types"),
    ):
        """Generate a novel capability from requirements."""
        from ..meta import get_environment, CapabilityType
        env = get_environment()

        requirements = {}
        if inputs:
            requirements["inputs"] = [i.strip() for i in inputs.split(",")]
        if outputs:
            requirements["outputs"] = [o.strip() for o in outputs.split(",")]
        if capability_types:
            try:
                requirements["capability_types"] = [
                    CapabilityType(ct.strip()) for ct in capability_types.split(",")
                ]
            except ValueError as e:
                raise HTTPException(400, str(e))

        if not requirements:
            raise HTTPException(400, "At least one requirement must be specified")

        result = env.generate_novel_capability(requirements)
        if not result:
            return {"generated": None, "message": "No matching capability could be generated"}

        return {
            "generated": {
                "id": result.structure_id,
                "name": result.name,
                "type": result.structure_type.value,
                "capabilities": len(result.capabilities),
            },
        }

    @app.get("/api/meta/templates")
    async def list_templates():
        """List available agent templates."""
        from ..meta.templates import (
            create_observer_template,
            create_reasoner_template,
            create_actor_template,
            create_meta_agent_template,
        )

        templates = [
            create_observer_template(),
            create_reasoner_template(),
            create_actor_template(),
            create_meta_agent_template(),
        ]

        return {
            "templates": [
                {
                    "id": t.template_id,
                    "name": t.name,
                    "description": t.description,
                    "parameters": [
                        {"name": p.name, "type": str(p.param_type.__name__), "required": p.required}
                        for p in t.parameters
                    ],
                    "capability_count": len(t.capability_templates),
                }
                for t in templates
            ],
        }

    @app.post("/api/meta/instantiate/{template_name}")
    async def instantiate_template(
        template_name: str,
        params: Optional[str] = Query(None, description="JSON-encoded parameters"),
    ):
        """Instantiate an agent template."""
        import json
        from ..meta.templates import (
            create_observer_template,
            create_reasoner_template,
            create_actor_template,
            create_meta_agent_template,
        )
        from ..meta import get_environment

        template_map = {
            "observer": create_observer_template,
            "reasoner": create_reasoner_template,
            "actor": create_actor_template,
            "meta_agent": create_meta_agent_template,
        }

        if template_name not in template_map:
            raise HTTPException(404, f"Template {template_name} not found")

        template = template_map[template_name]()

        param_dict = {}
        if params:
            try:
                param_dict = json.loads(params)
            except json.JSONDecodeError:
                raise HTTPException(400, "Invalid JSON in params")

        structure = template.instantiate(**param_dict)

        # Add to environment
        env = get_environment()
        env._population[structure.structure_id] = structure
        env.graph.add_structure(structure)

        return {
            "id": structure.structure_id,
            "name": structure.name,
            "type": structure.structure_type.value,
            "capabilities": [c.name for c in structure.capabilities],
            "metadata": structure.metadata,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Temporal Dynamics Routes
    # ─────────────────────────────────────────────────────────────────────────

    @app.get("/api/temporal/graph")
    async def get_temporal_graph_state():
        """Get the current state of the temporal hypergraph."""
        from ..meta import get_temporal_graph

        tg = get_temporal_graph()
        return tg.to_temporal_dict()

    @app.post("/api/temporal/reset")
    async def reset_temporal_graph():
        """Reset the temporal hypergraph to initial state."""
        from ..meta import reset_temporal_graph

        tg = reset_temporal_graph()
        return {"status": "reset", "current_tick": tg.current_tick}

    @app.post("/api/temporal/init")
    async def initialize_temporal_from_hypergraph():
        """Initialize temporal graph from existing hypergraph structures."""
        from ..meta import get_temporal_graph, get_capability_graph
        from ..meta.hypergraph import EdgeType

        tg = get_temporal_graph()
        hg = get_capability_graph()

        # Copy nodes from capability graph
        for node_id, node in hg._nodes.items():
            if node_id not in tg._nodes:
                tg._nodes[node_id] = node
                tg._type_index[node.node_type].add(node_id)
                if node.name:
                    tg._name_index[node.name] = node_id

            # Initialize temporal state
            tg.initialize_node_state(node_id, activation=0.0)

        # Copy edges
        for edge_id, edge in hg._edges.items():
            if edge_id not in tg._edges:
                tg._edges[edge_id] = edge
                for nid in edge.nodes():
                    tg._node_to_edges[nid].add(edge_id)

            # Initialize edge state
            tg.initialize_edge_state(edge_id, activation=1.0)

        # Create default propagation rules
        tg.create_propagation_rule(
            "composition_prop",
            EdgeType.COMPOSITION,
            delay=1,
            decay=0.85,
        )
        tg.create_propagation_rule(
            "derivation_prop",
            EdgeType.DERIVATION,
            delay=2,
            decay=0.9,
        )
        tg.create_propagation_rule(
            "subsumption_prop",
            EdgeType.SUBSUMPTION,
            delay=1,
            decay=0.95,
        )
        tg.create_decay_rule("global_decay", decay_rate=0.02)

        return {
            "status": "initialized",
            "nodes": len(tg._nodes),
            "edges": len(tg._edges),
            "rules": len(tg._rules),
            "current_tick": tg.current_tick,
        }

    @app.post("/api/temporal/tick")
    async def advance_tick(ticks: int = Query(1, ge=1, le=1000)):
        """Advance the temporal simulation by specified ticks."""
        from ..meta import get_temporal_graph

        tg = get_temporal_graph()
        result = tg.run(ticks)

        return {
            "ticks_run": result["ticks_run"],
            "events_processed": result["events_processed"],
            "current_tick": result["final_tick"],
            "events": [
                {
                    "id": e.event_id,
                    "type": e.event_type.value,
                    "tick": e.tick,
                    "source": e.source_id,
                    "target": e.target_id,
                    "value": round(e.value, 4),
                }
                for e in result["events"][:50]  # Limit response size
            ],
        }

    @app.post("/api/temporal/activate/{node_id}")
    async def activate_temporal_node(
        node_id: str,
        activation: float = Query(1.0, ge=0.0, le=1.0),
        propagate: bool = Query(True),
    ):
        """Activate a node in the temporal graph."""
        from ..meta import get_temporal_graph

        tg = get_temporal_graph()

        if node_id not in tg._nodes:
            raise HTTPException(404, f"Node {node_id} not found")

        event = tg.activate_node(node_id, activation, propagate)

        return {
            "event_id": event.event_id,
            "node_id": node_id,
            "activation": activation,
            "propagate": propagate,
            "current_tick": tg.current_tick,
        }

    @app.post("/api/temporal/cascade/{node_id}")
    async def trigger_cascade(
        node_id: str,
        intensity: float = Query(1.0, ge=0.0, le=2.0),
    ):
        """Trigger a cascade event from a node."""
        from ..meta import get_temporal_graph

        tg = get_temporal_graph()

        if node_id not in tg._nodes:
            raise HTTPException(404, f"Node {node_id} not found")

        event = tg.trigger_cascade(node_id, intensity)

        return {
            "event_id": event.event_id,
            "node_id": node_id,
            "intensity": intensity,
            "current_tick": tg.current_tick,
        }

    @app.get("/api/temporal/state/{node_id}")
    async def get_node_temporal_state(
        node_id: str,
        tick: Optional[int] = Query(None),
    ):
        """Get temporal state of a node."""
        from ..meta import get_temporal_graph

        tg = get_temporal_graph()

        if node_id not in tg._nodes:
            raise HTTPException(404, f"Node {node_id} not found")

        state = tg.get_node_state(node_id, tick)

        if not state:
            return {"node_id": node_id, "state": None}

        return {
            "node_id": node_id,
            "tick": state.tick,
            "activation": round(state.activation, 4),
            "energy": round(state.energy, 4),
            "potential": round(state.potential, 4),
            "active": state.active,
            "triggered_by": state.triggered_by,
            "values": state.values,
        }

    @app.get("/api/temporal/timeline/{node_id}")
    async def get_node_timeline(
        node_id: str,
        start_tick: int = Query(0, ge=0),
        end_tick: Optional[int] = Query(None),
    ):
        """Get activation timeline for a node."""
        from ..meta import get_temporal_graph

        tg = get_temporal_graph()

        if node_id not in tg._nodes:
            raise HTTPException(404, f"Node {node_id} not found")

        timeline = tg.get_activation_timeline(node_id, start_tick, end_tick)
        peaks = tg.find_activation_peaks(node_id)

        return {
            "node_id": node_id,
            "timeline": [{"tick": t, "activation": round(a, 4)} for t, a in timeline],
            "peaks": [{"tick": t, "activation": round(a, 4)} for t, a in peaks],
            "data_points": len(timeline),
        }

    @app.get("/api/temporal/events")
    async def query_temporal_events(
        event_type: Optional[str] = Query(None),
        start_tick: Optional[int] = Query(None),
        end_tick: Optional[int] = Query(None),
        node_id: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=1000),
    ):
        """Query temporal events."""
        from ..meta import get_temporal_graph
        from ..meta.temporal import DynamicsType

        tg = get_temporal_graph()

        dtype = None
        if event_type:
            try:
                dtype = DynamicsType(event_type)
            except ValueError:
                raise HTTPException(400, f"Invalid event type: {event_type}")

        events = tg.query_events(dtype, start_tick, end_tick, node_id)

        return {
            "events": [
                {
                    "id": e.event_id,
                    "type": e.event_type.value,
                    "tick": e.tick,
                    "source": e.source_id,
                    "target": e.target_id,
                    "value": round(e.value, 4),
                    "caused_by": e.caused_by,
                }
                for e in events[:limit]
            ],
            "total": len(events),
            "returned": min(len(events), limit),
        }

    @app.get("/api/temporal/patterns")
    async def detect_temporal_patterns():
        """Detect temporal patterns in the graph dynamics."""
        from ..meta import get_temporal_graph

        tg = get_temporal_graph()
        patterns = tg.detect_patterns()

        return {
            "patterns": [
                {
                    "id": p.pattern_id,
                    "type": p.pattern_type,
                    "nodes": p.node_ids,
                    "start_tick": p.start_tick,
                    "end_tick": p.end_tick,
                    "duration": p.duration,
                    "periodicity": p.periodicity,
                    "confidence": round(p.confidence, 4),
                    "metadata": p.metadata,
                }
                for p in patterns
            ],
            "total_patterns": len(patterns),
        }

    @app.get("/api/temporal/causal/{event_id}")
    async def get_causal_chain(
        event_id: str,
        direction: str = Query("forward", regex="^(forward|backward)$"),
    ):
        """Get causal chain from/to an event."""
        from ..meta import get_temporal_graph

        tg = get_temporal_graph()

        if event_id not in tg._event_index:
            raise HTTPException(404, f"Event {event_id} not found")

        chain = tg.get_causal_chain(event_id, direction)

        return {
            "event_id": event_id,
            "direction": direction,
            "chain": [
                {
                    "id": e.event_id,
                    "type": e.event_type.value,
                    "tick": e.tick,
                    "source": e.source_id,
                    "target": e.target_id,
                    "value": round(e.value, 4),
                }
                for e in chain
            ],
            "chain_length": len(chain),
        }

    @app.get("/api/temporal/influence")
    async def compute_influence(
        source_id: str = Query(...),
        target_id: str = Query(...),
    ):
        """Compute temporal influence between nodes."""
        from ..meta import get_temporal_graph

        tg = get_temporal_graph()

        if source_id not in tg._nodes:
            raise HTTPException(404, f"Source node {source_id} not found")
        if target_id not in tg._nodes:
            raise HTTPException(404, f"Target node {target_id} not found")

        influence = tg.compute_influence(source_id, target_id)

        return {
            "source": source_id,
            "target": target_id,
            "influence": round(influence, 4),
        }

    @app.post("/api/temporal/rules")
    async def add_dynamics_rule(
        name: str = Query(...),
        rule_type: str = Query(...),
        delay: int = Query(1, ge=0),
        decay: float = Query(0.9, ge=0.0, le=1.0),
        threshold: float = Query(0.1, ge=0.0, le=1.0),
    ):
        """Add a dynamics rule to the temporal graph."""
        from ..meta import get_temporal_graph
        from ..meta.temporal import DynamicsType
        from ..meta.hypergraph import EdgeType

        tg = get_temporal_graph()

        # Determine edge type based on rule_type
        edge_map = {
            "composition": EdgeType.COMPOSITION,
            "derivation": EdgeType.DERIVATION,
            "dependency": EdgeType.DEPENDENCY,
            "emergence": EdgeType.EMERGENCE,
            "subsumption": EdgeType.SUBSUMPTION,
        }

        if rule_type not in edge_map:
            raise HTTPException(400, f"Invalid rule type: {rule_type}. Valid: {list(edge_map.keys())}")

        rule = tg.create_propagation_rule(
            name=name,
            edge_type=edge_map[rule_type],
            delay=delay,
            decay=decay,
            threshold=threshold,
        )

        return {
            "rule_id": rule.rule_id,
            "name": rule.name,
            "type": rule_type,
            "delay": delay,
            "decay": decay,
            "threshold": threshold,
        }

    @app.get("/api/temporal/stats")
    async def get_temporal_stats():
        """Get temporal dynamics statistics."""
        from ..meta import get_temporal_graph

        tg = get_temporal_graph()
        stats = tg.get_temporal_stats()

        return {
            "current_tick": tg.current_tick,
            "total_ticks": stats["total_ticks"],
            "total_events": stats["total_events"],
            "propagations": stats["propagations"],
            "cascades": stats["cascades"],
            "patterns_detected": stats["patterns_detected"],
            "active_nodes": stats["active_nodes"],
            "pending_events": stats["pending_events"],
            "rules_count": stats["rules_count"],
            "patterns_found": stats["patterns_found"],
        }

    @app.get("/api/temporal/snapshot")
    async def get_graph_snapshot(tick: Optional[int] = Query(None)):
        """Get snapshot of all node states at a specific tick."""
        from ..meta import get_temporal_graph

        tg = get_temporal_graph()
        target_tick = tick if tick is not None else tg.current_tick

        states = tg.query_at_time(target_tick)

        return {
            "tick": target_tick,
            "nodes": {
                nid: {
                    "activation": round(s.activation, 4),
                    "energy": round(s.energy, 4),
                    "potential": round(s.potential, 4),
                    "active": s.active,
                }
                for nid, s in states.items()
            },
            "node_count": len(states),
        }

    @app.post("/api/temporal/oscillator/{node_id}")
    async def add_oscillator(
        node_id: str,
        period: int = Query(10, ge=2, le=1000),
        amplitude: float = Query(0.5, ge=0.0, le=1.0),
        phase: float = Query(0.0, ge=0.0, le=6.28),
    ):
        """Add an oscillation rule to a specific node."""
        from ..meta import get_temporal_graph
        from ..meta.temporal import DynamicsRule, DynamicsType
        import math

        tg = get_temporal_graph()

        if node_id not in tg._nodes:
            raise HTTPException(404, f"Node {node_id} not found")

        def action(nid: str, state, graph):
            if nid != node_id:
                return state
            t = graph.current_tick
            oscillation = amplitude * math.sin(2 * math.pi * t / period + phase)
            new_activation = 0.5 + oscillation
            return graph.update_node_state(nid, activation=new_activation)

        rule = DynamicsRule(
            name=f"oscillator_{node_id}",
            trigger_type=DynamicsType.OSCILLATION,
            action=action,
        )
        tg.add_rule(rule)

        return {
            "rule_id": rule.rule_id,
            "node_id": node_id,
            "period": period,
            "amplitude": amplitude,
            "phase": phase,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Locality & Mailbox Routes
    # ─────────────────────────────────────────────────────────────────────────

    @app.get("/api/locality/network")
    async def get_locality_network_state():
        """Get the current state of the locality network."""
        from ..meta import get_locality_network

        net = get_locality_network()
        return net.to_dict()

    @app.post("/api/locality/reset")
    async def reset_locality_network():
        """Reset the locality network."""
        from ..meta import reset_locality_network

        net = reset_locality_network()
        return {"status": "reset", "agents": 0}

    @app.post("/api/locality/agents")
    async def register_locality_agent(
        agent_id: str = Query(...),
        x: float = Query(0.0),
        y: float = Query(0.0),
        z: Optional[float] = Query(None),
        comm_radius: float = Query(10.0),
        local_radius: float = Query(3.0),
    ):
        """Register an agent in the locality network."""
        from ..meta import get_locality_network
        from ..meta.locality import Location

        net = get_locality_network()

        coords = (x, y) if z is None else (x, y, z)
        location = Location(coordinates=coords)

        node = net.register_agent(
            agent_id=agent_id,
            location=location,
            communication_radius=comm_radius,
            local_only_radius=local_radius,
        )

        return {
            "agent_id": node.agent_id,
            "location": list(node.location.coordinates),
            "comm_radius": node.communication_radius,
            "local_radius": node.local_only_radius,
            "neighbors": list(node.neighbors),
        }

    @app.delete("/api/locality/agents/{agent_id}")
    async def unregister_locality_agent(agent_id: str):
        """Remove an agent from the locality network."""
        from ..meta import get_locality_network

        net = get_locality_network()
        net.unregister_agent(agent_id)
        return {"status": "removed", "agent_id": agent_id}

    @app.post("/api/locality/agents/{agent_id}/move")
    async def move_locality_agent(
        agent_id: str,
        x: float = Query(...),
        y: float = Query(...),
        z: Optional[float] = Query(None),
    ):
        """Move an agent to a new location."""
        from ..meta import get_locality_network
        from ..meta.locality import Location

        net = get_locality_network()

        if agent_id not in net._agents:
            raise HTTPException(404, f"Agent {agent_id} not found")

        coords = (x, y) if z is None else (x, y, z)
        new_location = Location(coordinates=coords)

        net.move_agent(agent_id, new_location)
        node = net._agents[agent_id]

        return {
            "agent_id": agent_id,
            "location": list(node.location.coordinates),
            "neighbors": list(node.neighbors),
        }

    @app.get("/api/locality/agents/{agent_id}/neighbors")
    async def get_agent_neighbors(
        agent_id: str,
        max_distance: Optional[float] = Query(None),
        local_only: bool = Query(False),
    ):
        """Get neighbors of an agent."""
        from ..meta import get_locality_network

        net = get_locality_network()

        if agent_id not in net._agents:
            raise HTTPException(404, f"Agent {agent_id} not found")

        if local_only:
            neighbors = net.get_local_neighbors(agent_id)
        else:
            neighbors = net.get_neighbors(agent_id, max_distance)

        node = net._agents[agent_id]
        return {
            "agent_id": agent_id,
            "neighbors": [
                {
                    "id": nid,
                    "distance": round(node.neighbor_distances.get(nid, 0), 4),
                }
                for nid in neighbors
            ],
            "count": len(neighbors),
        }

    @app.post("/api/locality/messages/send")
    async def send_locality_message(
        sender_id: str = Query(...),
        recipient_id: Optional[str] = Query(None),
        content: str = Query(...),
        delivery_mode: str = Query("direct"),
        priority: str = Query("normal"),
        target_group: Optional[str] = Query(None),
        target_radius: Optional[float] = Query(None),
        logged: bool = Query(True),
    ):
        """Send a message through the locality network."""
        from ..meta import get_locality_network
        from ..meta.locality import DeliveryMode, MessagePriority

        net = get_locality_network()

        if sender_id not in net._agents:
            raise HTTPException(404, f"Sender {sender_id} not found")

        try:
            mode = DeliveryMode(delivery_mode)
        except ValueError:
            raise HTTPException(400, f"Invalid delivery mode: {delivery_mode}")

        try:
            prio = MessagePriority[priority.upper()]
        except KeyError:
            raise HTTPException(400, f"Invalid priority: {priority}")

        message = net.send_message(
            sender_id=sender_id,
            recipient_id=recipient_id,
            content=content,
            delivery_mode=mode,
            priority=prio,
            target_group=target_group,
            target_radius=target_radius,
            logged=logged,
        )

        if not message:
            raise HTTPException(500, "Failed to send message")

        return {
            "message_id": message.message_id,
            "sender": message.sender_id,
            "recipient": message.recipient_id,
            "delivery_mode": message.delivery_mode.value,
            "logged": message.logged,
        }

    @app.get("/api/locality/messages/{agent_id}")
    async def get_agent_messages(
        agent_id: str,
        limit: int = Query(10, ge=1, le=100),
    ):
        """Get messages from agent's mailbox."""
        from ..meta import get_locality_network

        net = get_locality_network()

        if agent_id not in net._agents:
            raise HTTPException(404, f"Agent {agent_id} not found")

        messages = net.get_messages(agent_id, limit)

        return {
            "agent_id": agent_id,
            "messages": [
                {
                    "id": m.message_id,
                    "sender": m.sender_id,
                    "content": m.content,
                    "priority": m.priority.name,
                    "delivery_mode": m.delivery_mode.value,
                }
                for m in messages
            ],
            "count": len(messages),
        }

    @app.post("/api/locality/messages/{agent_id}/pop")
    async def pop_agent_message(agent_id: str):
        """Pop highest priority message from agent's mailbox."""
        from ..meta import get_locality_network

        net = get_locality_network()

        if agent_id not in net._agents:
            raise HTTPException(404, f"Agent {agent_id} not found")

        message = net.pop_message(agent_id)

        if not message:
            return {"agent_id": agent_id, "message": None}

        return {
            "agent_id": agent_id,
            "message": {
                "id": message.message_id,
                "sender": message.sender_id,
                "content": message.content,
                "priority": message.priority.name,
                "delivery_mode": message.delivery_mode.value,
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Groups
    # ─────────────────────────────────────────────────────────────────────────

    @app.post("/api/locality/groups")
    async def create_locality_group(
        group_id: str = Query(...),
        member_ids: Optional[str] = Query(None, description="Comma-separated member IDs"),
    ):
        """Create a group in the locality network."""
        from ..meta import get_locality_network

        net = get_locality_network()

        members = member_ids.split(",") if member_ids else []
        net.create_group(group_id, members)

        return {
            "group_id": group_id,
            "members": net.get_group_members(group_id),
        }

    @app.post("/api/locality/groups/{group_id}/join")
    async def join_locality_group(group_id: str, agent_id: str = Query(...)):
        """Add agent to a group."""
        from ..meta import get_locality_network

        net = get_locality_network()

        if agent_id not in net._agents:
            raise HTTPException(404, f"Agent {agent_id} not found")

        net.join_group(agent_id, group_id)

        return {
            "group_id": group_id,
            "agent_id": agent_id,
            "members": net.get_group_members(group_id),
        }

    @app.post("/api/locality/groups/{group_id}/leave")
    async def leave_locality_group(group_id: str, agent_id: str = Query(...)):
        """Remove agent from a group."""
        from ..meta import get_locality_network

        net = get_locality_network()
        net.leave_group(agent_id, group_id)

        return {
            "group_id": group_id,
            "agent_id": agent_id,
            "members": net.get_group_members(group_id),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Quorums
    # ─────────────────────────────────────────────────────────────────────────

    @app.post("/api/locality/quorums")
    async def create_quorum(
        name: str = Query(""),
        quorum_id: Optional[str] = Query(None),
        member_ids: Optional[str] = Query(None, description="Comma-separated member IDs"),
        consensus_threshold: float = Query(0.67, ge=0.5, le=1.0),
        min_size: int = Query(3, ge=1),
    ):
        """Create a quasi-quorum."""
        from ..meta import get_locality_network

        net = get_locality_network()

        members = member_ids.split(",") if member_ids else []

        quorum = net.create_quorum(
            quorum_id=quorum_id,
            name=name,
            members=members,
            consensus_threshold=consensus_threshold,
            min_size=min_size,
        )

        return {
            "quorum_id": quorum.quorum_id,
            "name": quorum.name,
            "state": quorum.state.value,
            "members": list(quorum.members),
            "consensus_threshold": quorum.consensus_threshold,
            "min_size": quorum.min_size,
        }

    @app.get("/api/locality/quorums/{quorum_id}")
    async def get_quorum_info(quorum_id: str):
        """Get quorum information."""
        from ..meta import get_locality_network

        net = get_locality_network()
        quorum = net.get_quorum(quorum_id)

        if not quorum:
            raise HTTPException(404, f"Quorum {quorum_id} not found")

        return {
            "quorum_id": quorum.quorum_id,
            "name": quorum.name,
            "state": quorum.state.value,
            "members": list(quorum.members),
            "member_weights": quorum.member_weights,
            "consensus_threshold": quorum.consensus_threshold,
            "active_proposals": len(quorum.active_proposals),
            "completed_proposals": len(quorum.completed_proposals),
        }

    @app.post("/api/locality/quorums/{quorum_id}/join")
    async def join_quorum(
        quorum_id: str,
        agent_id: str = Query(...),
        weight: float = Query(1.0, ge=0.0),
    ):
        """Add agent to a quorum."""
        from ..meta import get_locality_network

        net = get_locality_network()

        if quorum_id not in net._quorums:
            raise HTTPException(404, f"Quorum {quorum_id} not found")

        if agent_id not in net._agents:
            raise HTTPException(404, f"Agent {agent_id} not found")

        success = net.join_quorum(agent_id, quorum_id, weight)

        if not success:
            raise HTTPException(400, "Failed to join quorum (may be full)")

        quorum = net._quorums[quorum_id]
        return {
            "quorum_id": quorum_id,
            "agent_id": agent_id,
            "weight": weight,
            "state": quorum.state.value,
            "members": list(quorum.members),
        }

    @app.post("/api/locality/quorums/{quorum_id}/propose")
    async def create_proposal(
        quorum_id: str,
        proposer_id: str = Query(...),
        content: str = Query(...),
        proposal_type: str = Query("generic"),
        deadline_seconds: Optional[int] = Query(None),
    ):
        """Create a proposal in a quorum."""
        from ..meta import get_locality_network

        net = get_locality_network()

        proposal = net.propose(
            agent_id=proposer_id,
            quorum_id=quorum_id,
            content=content,
            proposal_type=proposal_type,
            deadline_seconds=deadline_seconds,
        )

        if not proposal:
            raise HTTPException(400, "Failed to create proposal (quorum not active or proposer not member)")

        return {
            "proposal_id": proposal.proposal_id,
            "quorum_id": quorum_id,
            "proposer": proposal.proposer_id,
            "content": proposal.content,
            "type": proposal.proposal_type,
            "threshold": proposal.threshold,
            "deadline": proposal.deadline.isoformat() if proposal.deadline else None,
        }

    @app.get("/api/locality/quorums/{quorum_id}/proposals")
    async def get_quorum_proposals(quorum_id: str):
        """Get active proposals in a quorum."""
        from ..meta import get_locality_network

        net = get_locality_network()
        proposals = net.get_active_proposals(quorum_id)

        return {
            "quorum_id": quorum_id,
            "proposals": [
                {
                    "id": p.proposal_id,
                    "proposer": p.proposer_id,
                    "content": p.content,
                    "type": p.proposal_type,
                    "votes": len(p.votes),
                    "threshold": p.threshold,
                    "deadline": p.deadline.isoformat() if p.deadline else None,
                }
                for p in proposals
            ],
        }

    @app.post("/api/locality/quorums/{quorum_id}/vote")
    async def cast_vote(
        quorum_id: str,
        voter_id: str = Query(...),
        proposal_id: str = Query(...),
        vote: bool = Query(...),
    ):
        """Cast a vote on a proposal."""
        from ..meta import get_locality_network

        net = get_locality_network()

        vote_obj = net.vote(
            agent_id=voter_id,
            quorum_id=quorum_id,
            proposal_id=proposal_id,
            vote=vote,
        )

        if not vote_obj:
            raise HTTPException(400, "Failed to cast vote (voter not member, proposal not found, or deadline passed)")

        # Check if proposal was decided
        quorum = net._quorums[quorum_id]
        decided = proposal_id in quorum.completed_proposals

        result = {
            "proposal_id": proposal_id,
            "voter": voter_id,
            "vote": vote,
            "weight": vote_obj.weight,
            "decided": decided,
        }

        if decided:
            # Find the completed proposal in log
            for pid in quorum.completed_proposals:
                if pid == proposal_id:
                    result["result"] = "passed" if True else "rejected"  # Would need to track this

        return result

    @app.get("/api/locality/stats")
    async def get_locality_stats():
        """Get locality network statistics."""
        from ..meta import get_locality_network

        net = get_locality_network()
        return net.get_stats()

    # ─────────────────────────────────────────────────────────────────────────
    # MCP Registry routes
    # ─────────────────────────────────────────────────────────────────────────

    @app.get("/api/mcp/servers")
    async def list_mcp_servers(category: Optional[str] = None):
        """List available MCP servers."""
        from ..meta import get_mcp_registry, MCPServerCategory

        registry = get_mcp_registry()
        cat = MCPServerCategory(category) if category else None
        servers = registry.list_servers(cat)

        return {
            "servers": [s.to_dict() for s in servers],
            "count": len(servers),
        }

    @app.get("/api/mcp/servers/{server_id}")
    async def get_mcp_server(server_id: str):
        """Get MCP server details."""
        from ..meta import get_mcp_registry

        registry = get_mcp_registry()
        server = registry.get_server(server_id)

        if not server:
            raise HTTPException(404, f"MCP server not found: {server_id}")

        return server.to_dict()

    @app.post("/api/mcp/servers")
    async def register_mcp_server(request: dict):
        """Register a new MCP server configuration."""
        from ..meta import get_mcp_registry, MCPServerConfig

        registry = get_mcp_registry()
        config = MCPServerConfig.from_dict(request)
        registry.register_server(config)

        return {
            "server_id": config.server_id,
            "name": config.name,
            "status": "registered",
        }

    @app.post("/api/mcp/servers/load")
    async def load_mcp_config(path: str = Body(..., embed=True)):
        """Load MCP server configs from a JSON file."""
        from ..meta import get_mcp_registry
        from pathlib import Path

        registry = get_mcp_registry()
        file_path = Path(path).expanduser()

        if not file_path.exists():
            raise HTTPException(404, f"Config file not found: {path}")

        registry.load_from_file(file_path)

        return {
            "loaded_from": str(file_path),
            "server_count": len(registry.servers),
        }

    @app.post("/api/mcp/discover")
    async def discover_mcp_servers():
        """Discover MCP servers from common config locations."""
        from ..meta import get_mcp_registry

        registry = get_mcp_registry()
        before_count = len(registry.servers)
        registry.discover_servers()
        after_count = len(registry.servers)

        return {
            "discovered": after_count - before_count,
            "total": after_count,
        }

    @app.get("/api/mcp/tools")
    async def list_mcp_tools(category: Optional[str] = None):
        """List all MCP tools, optionally filtered by category."""
        from ..meta import get_mcp_registry, MCPServerCategory

        registry = get_mcp_registry()

        if category:
            cat = MCPServerCategory(category)
            tools = registry.get_tools_by_category(cat)
            return {
                "category": category,
                "tools": [
                    {"server_id": sid, "tool": t.to_dict()}
                    for sid, t in tools
                ],
            }

        # Return all tools grouped by server
        result = {}
        for server in registry.list_servers():
            result[server.server_id] = {
                "name": server.name,
                "tools": [t.to_dict() for t in server.tools],
            }

        return {"servers": result}

    @app.get("/api/mcp/categories")
    async def list_mcp_categories():
        """List all MCP server categories."""
        from ..meta import MCPServerCategory

        return {
            "categories": [
                {"value": c.value, "name": c.name}
                for c in MCPServerCategory
            ],
        }

    @app.post("/api/mcp/connect/{server_id}")
    async def connect_mcp_server(server_id: str):
        """Connect to an MCP server."""
        from ..meta import get_mcp_registry

        registry = get_mcp_registry()
        server = registry.get_server(server_id)

        if not server:
            raise HTTPException(404, f"MCP server not found: {server_id}")

        try:
            connection = await registry.connect_server(server_id)
            return {
                "server_id": server_id,
                "connected": connection.connected,
                "error": connection.error,
            }
        except Exception as e:
            return {
                "server_id": server_id,
                "connected": False,
                "error": str(e),
            }

    @app.post("/api/mcp/disconnect/{server_id}")
    async def disconnect_mcp_server(server_id: str):
        """Disconnect from an MCP server."""
        from ..meta import get_mcp_registry

        registry = get_mcp_registry()
        await registry.disconnect_server(server_id)

        return {"server_id": server_id, "status": "disconnected"}

    @app.get("/api/mcp/connections")
    async def list_mcp_connections():
        """List active MCP server connections."""
        from ..meta import get_mcp_registry

        registry = get_mcp_registry()

        return {
            "connections": {
                sid: {
                    "connected": c.connected,
                    "last_ping": c.last_ping.isoformat() if c.last_ping else None,
                    "error": c.error,
                }
                for sid, c in registry.connections.items()
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    # MCP-enabled Template routes
    # ─────────────────────────────────────────────────────────────────────────

    @app.get("/api/templates/mcp")
    async def list_mcp_templates():
        """List available MCP-enabled agent templates."""
        from ..meta import (
            create_research_agent_template,
            create_embedding_agent_template,
            create_filesystem_agent_template,
            create_multimodal_research_agent_template,
            create_memory_agent_template,
        )

        templates = [
            create_research_agent_template(),
            create_embedding_agent_template(),
            create_filesystem_agent_template(),
            create_multimodal_research_agent_template(),
            create_memory_agent_template(),
        ]

        return {
            "templates": [
                {
                    "id": t.template_id,
                    "name": t.name,
                    "description": t.description,
                    "mcp_servers": [s.server_id for s in t.mcp_servers],
                    "parameters": [
                        {"name": p.name, "type": str(p.param_type.__name__), "required": p.required}
                        for p in t.parameters
                    ],
                    "capability_count": len(t.capability_templates),
                }
                for t in templates
            ],
        }

    @app.post("/api/templates/mcp/instantiate")
    async def instantiate_mcp_template(
        template_name: str = Body(...),
        params: dict = Body(default={}),
    ):
        """Instantiate an MCP-enabled agent template."""
        from ..meta import (
            create_research_agent_template,
            create_embedding_agent_template,
            create_filesystem_agent_template,
            create_multimodal_research_agent_template,
            create_memory_agent_template,
        )

        template_factories = {
            "research_agent": create_research_agent_template,
            "embedding_agent": create_embedding_agent_template,
            "filesystem_agent": create_filesystem_agent_template,
            "multimodal_research_agent": create_multimodal_research_agent_template,
            "memory_agent": create_memory_agent_template,
        }

        if template_name not in template_factories:
            raise HTTPException(400, f"Unknown template: {template_name}. Available: {list(template_factories.keys())}")

        template = template_factories[template_name]()
        structure = template.instantiate(**params)

        return {
            "structure_id": structure.structure_id,
            "name": structure.name,
            "capabilities": len(structure.capabilities),
            "mcp_servers": structure.metadata.get("mcp_servers", []),
            "mcp_bindings": structure.metadata.get("mcp_bindings", []),
            "metadata": structure.metadata,
        }

    @app.post("/api/templates/mcp/custom")
    async def create_custom_mcp_template(
        name: str = Body(...),
        description: str = Body(...),
        mcp_server_ids: list[str] = Body(...),
        category_filters: dict[str, list[str]] = Body(default={}),
    ):
        """Create a custom MCP-enabled agent template."""
        from ..meta import create_composite_mcp_agent_template

        template = create_composite_mcp_agent_template(
            name=name,
            description=description,
            mcp_server_ids=mcp_server_ids,
            category_filters=category_filters,
        )

        return {
            "template_id": template.template_id,
            "name": template.name,
            "description": template.description,
            "mcp_servers": [s.server_id for s in template.mcp_servers],
        }

    @app.post("/api/templates/{template_id}/add-mcp")
    async def add_mcp_to_template(
        template_id: str,
        server_id: str = Body(...),
        required: bool = Body(default=False),
        tool_filter: list[str] = Body(default=None),
        category_filter: list[str] = Body(default=None),
    ):
        """Add an MCP server to an existing template."""
        from ..meta import get_environment

        env = get_environment()
        template = env.get_template(template_id)

        if not template:
            raise HTTPException(404, f"Template not found: {template_id}")

        # Create new template with MCP server added
        new_template = template.with_mcp_server(
            server_id=server_id,
            required=required,
            tool_filter=tool_filter,
            category_filter=category_filter,
        )

        # Register the new template
        env.register_template(new_template)

        return {
            "old_template_id": template_id,
            "new_template_id": new_template.template_id,
            "mcp_servers": [s.server_id for s in new_template.mcp_servers],
        }

    @app.post("/api/agents/mcp")
    async def create_mcp_agent(
        template_name: str = Body(...),
        agent_id: Optional[str] = Body(default=None),
        params: dict = Body(default={}),
        location: Optional[list[float]] = Body(default=None),
    ):
        """Create an agent from an MCP template and register in locality network."""
        from ..meta import (
            create_research_agent_template,
            create_embedding_agent_template,
            create_filesystem_agent_template,
            create_multimodal_research_agent_template,
            create_memory_agent_template,
            get_locality_network,
            Location,
        )
        import uuid

        template_factories = {
            "research_agent": create_research_agent_template,
            "embedding_agent": create_embedding_agent_template,
            "filesystem_agent": create_filesystem_agent_template,
            "multimodal_research_agent": create_multimodal_research_agent_template,
            "memory_agent": create_memory_agent_template,
        }

        if template_name not in template_factories:
            raise HTTPException(400, f"Unknown template: {template_name}")

        template = template_factories[template_name]()
        structure = template.instantiate(**params)

        # Generate agent ID if not provided
        agent_id = agent_id or f"{template_name}_{str(uuid.uuid4())[:8]}"

        # Register in locality network
        net = get_locality_network()
        loc = Location(coordinates=tuple(location)) if location else Location(coordinates=(0.0, 0.0))

        agent_node = net.register_agent(
            agent_id=agent_id,
            location=loc,
            metadata={
                "template": template_name,
                "structure_id": structure.structure_id,
                "mcp_servers": structure.metadata.get("mcp_servers", []),
            },
        )

        return {
            "agent_id": agent_id,
            "structure_id": structure.structure_id,
            "template": template_name,
            "location": list(loc.coordinates),
            "neighbors": list(agent_node.neighbors),
            "capabilities": len(structure.capabilities),
            "mcp_bindings": structure.metadata.get("mcp_bindings", []),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # MCP Agent Runtime (Live agents with MCP execution)
    # ─────────────────────────────────────────────────────────────────────────

    # Store for MCP agent factory and standalone broker
    _mcp_agent_factory = None
    _standalone_broker = None

    async def get_mcp_factory():
        """Get or create the MCP agent factory."""
        nonlocal _mcp_agent_factory, _standalone_broker

        if _mcp_agent_factory is None:
            from ..agents import MCPAgentFactory
            from ..messaging import MessageBroker, BrokerConfig

            # Try to use orchestrator's broker first
            broker = None
            if hasattr(app.state, 'orchestrator') and app.state.orchestrator:
                if hasattr(app.state.orchestrator, 'supervisor') and app.state.orchestrator.supervisor:
                    broker = app.state.orchestrator.supervisor.broker

            # If no orchestrator broker, create a standalone one
            if broker is None:
                if _standalone_broker is None:
                    _standalone_broker = MessageBroker(BrokerConfig())
                    try:
                        await _standalone_broker.connect()
                    except Exception as e:
                        logger.warning(f"Could not connect to Redis, MCP agents will work locally only: {e}")
                broker = _standalone_broker

            _mcp_agent_factory = MCPAgentFactory(broker)

        return _mcp_agent_factory

    @app.post("/api/mcp-agents/create")
    async def create_mcp_runtime_agent(
        template_name: str = Body(...),
        agent_id: Optional[str] = Body(default=None),
        params: dict = Body(default={}),
        auto_start: bool = Body(default=True),
    ):
        """Create a live MCP agent that can execute tools.

        This creates an actual running agent (not just metadata) that can:
        - Connect to MCP servers
        - Execute MCP tools
        - Handle messages with tool invocation
        """
        factory = await get_mcp_factory()
        if not factory:
            raise HTTPException(400, "MCP factory not available")

        from ..meta.templates import (
            create_research_agent_template,
            create_embedding_agent_template,
            create_filesystem_agent_template,
            create_multimodal_research_agent_template,
            create_memory_agent_template,
        )

        template_factories = {
            "research_agent": create_research_agent_template,
            "embedding_agent": create_embedding_agent_template,
            "filesystem_agent": create_filesystem_agent_template,
            "multimodal_research_agent": create_multimodal_research_agent_template,
            "memory_agent": create_memory_agent_template,
        }

        if template_name not in template_factories:
            raise HTTPException(400, f"Unknown template: {template_name}. Available: {list(template_factories.keys())}")

        template = template_factories[template_name]()

        try:
            agent = await factory.create_from_template(
                template=template,
                params=params,
                agent_id=agent_id,
                auto_start=auto_start,
            )

            return {
                "agent_id": agent.agent_id,
                "status": agent.status.value,
                "capabilities": agent.list_capabilities(),
                "mcp_tools": agent.list_mcp_tools(),
                "connections": agent.list_connections(),
            }
        except Exception as e:
            raise HTTPException(500, f"Failed to create agent: {str(e)}")

    @app.post("/api/mcp-agents/create-custom")
    async def create_custom_mcp_agent(
        name: str = Body(...),
        description: str = Body(default="Custom MCP agent"),
        mcp_server_ids: list[str] = Body(...),
        category_filters: Optional[dict] = Body(default=None),
        agent_id: Optional[str] = Body(default=None),
    ):
        """Create a custom MCP agent with specified servers."""
        factory = await get_mcp_factory()
        if not factory:
            raise HTTPException(400, "MCP factory not available")

        try:
            agent = await factory.create_custom_agent(
                name=name,
                description=description,
                mcp_server_ids=mcp_server_ids,
                category_filters=category_filters,
                agent_id=agent_id,
            )

            return {
                "agent_id": agent.agent_id,
                "status": agent.status.value,
                "capabilities": agent.list_capabilities(),
                "mcp_tools": agent.list_mcp_tools(),
                "connections": agent.list_connections(),
            }
        except Exception as e:
            raise HTTPException(500, f"Failed to create agent: {str(e)}")

    @app.get("/api/mcp-agents")
    async def list_mcp_runtime_agents():
        """List all running MCP agents."""
        factory = await get_mcp_factory()
        if not factory:
            return {"agents": []}

        return {"agents": factory.list_agents()}

    @app.get("/api/mcp-agents/{agent_id}")
    async def get_mcp_agent_status(agent_id: str):
        """Get status of a specific MCP agent."""
        factory = await get_mcp_factory()
        if not factory:
            raise HTTPException(400, "MCP factory not available")

        agent = factory.get_agent(agent_id)
        if not agent:
            raise HTTPException(404, f"Agent not found: {agent_id}")

        return agent.get_status()

    @app.post("/api/mcp-agents/{agent_id}/execute")
    async def execute_mcp_tool(
        agent_id: str,
        tool_name: str = Body(...),
        arguments: dict = Body(default={}),
        server_id: Optional[str] = Body(default=None),
    ):
        """Execute an MCP tool on a specific agent."""
        factory = await get_mcp_factory()
        if not factory:
            raise HTTPException(400, "MCP factory not available")

        agent = factory.get_agent(agent_id)
        if not agent:
            raise HTTPException(404, f"Agent not found: {agent_id}")

        result = await agent.execute_tool(tool_name, arguments, server_id)

        return result.to_dict()

    @app.post("/api/mcp-agents/{agent_id}/invoke")
    async def invoke_mcp_capability(
        agent_id: str,
        capability: str = Body(...),
        inputs: dict = Body(default={}),
    ):
        """Invoke a capability on an MCP agent (routes to MCP tool if bound)."""
        factory = await get_mcp_factory()
        if not factory:
            raise HTTPException(400, "MCP factory not available")

        agent = factory.get_agent(agent_id)
        if not agent:
            raise HTTPException(404, f"Agent not found: {agent_id}")

        result = await agent.invoke_capability(capability, inputs)

        return result.to_dict()

    @app.get("/api/mcp-agents/{agent_id}/capabilities")
    async def get_mcp_agent_capabilities(agent_id: str):
        """Get capabilities of an MCP agent."""
        factory = await get_mcp_factory()
        if not factory:
            raise HTTPException(400, "MCP factory not available")

        agent = factory.get_agent(agent_id)
        if not agent:
            raise HTTPException(404, f"Agent not found: {agent_id}")

        return {
            "agent_id": agent_id,
            "capabilities": agent.list_capabilities(),
            "mcp_tools": agent.list_mcp_tools(),
            "connections": agent.list_connections(),
        }

    @app.get("/api/mcp-agents/{agent_id}/history")
    async def get_mcp_agent_execution_history(agent_id: str, limit: int = Query(50, ge=1, le=200)):
        """Get execution history of an MCP agent."""
        factory = await get_mcp_factory()
        if not factory:
            raise HTTPException(400, "MCP factory not available")

        agent = factory.get_agent(agent_id)
        if not agent:
            raise HTTPException(404, f"Agent not found: {agent_id}")

        return {
            "agent_id": agent_id,
            "executions": agent.get_execution_history(limit),
        }

    @app.delete("/api/mcp-agents/{agent_id}")
    async def terminate_mcp_agent(agent_id: str):
        """Terminate an MCP agent."""
        factory = await get_mcp_factory()
        if not factory:
            raise HTTPException(400, "MCP factory not available")

        agent = factory.get_agent(agent_id)
        if not agent:
            raise HTTPException(404, f"Agent not found: {agent_id}")

        await factory.terminate_agent(agent_id)

        return {"status": "terminated", "agent_id": agent_id}

    @app.post("/api/mcp-agents/{agent_id}/connect/{server_id}")
    async def connect_mcp_agent_server(agent_id: str, server_id: str):
        """Connect an MCP agent to a specific server."""
        factory = await get_mcp_factory()
        if not factory:
            raise HTTPException(400, "MCP factory not available")

        agent = factory.get_agent(agent_id)
        if not agent:
            raise HTTPException(404, f"Agent not found: {agent_id}")

        connection = await agent._connect_server(server_id)

        return {
            "agent_id": agent_id,
            "server_id": server_id,
            "connected": connection.connected if connection else False,
            "error": connection.error if connection else "Connection failed",
        }

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
    # Timeline API
    # ─────────────────────────────────────────────────────────────────────────

    @app.get("/api/timeline")
    async def get_timeline(
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        agent_id: Optional[str] = Query(None),
        event_type: Optional[str] = Query(None),
        severity: Optional[str] = Query(None),
        since: Optional[str] = Query(None, description="ISO timestamp"),
    ):
        """Get timeline events with filtering."""
        from ..meta.timeline import (
            get_timeline_store,
            TimelineQuery,
            EventType,
            EventSeverity,
        )

        store = get_timeline_store()

        # Build query
        query = TimelineQuery(
            limit=limit,
            offset=offset,
            agent_ids=[agent_id] if agent_id else None,
            order_desc=True,
        )

        if event_type:
            try:
                query.event_types = [EventType(event_type)]
            except ValueError:
                pass

        if severity:
            try:
                query.severities = [EventSeverity(severity)]
            except ValueError:
                pass

        if since:
            try:
                from datetime import datetime
                query.start_time = datetime.fromisoformat(since.replace("Z", "+00:00"))
            except ValueError:
                pass

        events = await store.query(query)
        return {
            "events": [e.to_dict() for e in events],
            "count": len(events),
            "offset": offset,
            "limit": limit,
        }

    @app.get("/api/timeline/agent/{agent_id}")
    async def get_agent_timeline_events(
        agent_id: str,
        limit: int = Query(50, ge=1, le=500),
    ):
        """Get timeline for a specific agent."""
        from ..meta.timeline import get_timeline_store

        store = get_timeline_store()
        events = await store.get_agent_timeline(agent_id, limit=limit)
        return {
            "agent_id": agent_id,
            "events": [e.to_dict() for e in events],
            "count": len(events),
        }

    @app.get("/api/timeline/recent")
    async def get_recent_timeline(limit: int = Query(50, ge=1, le=200)):
        """Get most recent timeline events."""
        from ..meta.timeline import get_timeline_store

        store = get_timeline_store()
        events = await store.get_recent(limit=limit)
        return {
            "events": [e.to_dict() for e in events],
            "count": len(events),
        }

    @app.get("/api/timeline/stats")
    async def get_timeline_stats(
        since_hours: int = Query(24, ge=1, le=168),
        agent_id: Optional[str] = Query(None),
    ):
        """Get timeline statistics."""
        from datetime import datetime, timedelta
        from ..meta.timeline import get_timeline_store

        store = get_timeline_store()
        since = datetime.now(timezone.utc) - timedelta(hours=since_hours)
        stats = await store.get_stats(since=since, agent_id=agent_id)
        return stats.to_dict()

    @app.get("/api/timeline/event/{event_id}")
    async def get_timeline_event(event_id: str):
        """Get a specific timeline event."""
        from ..meta.timeline import get_timeline_store

        store = get_timeline_store()
        event = await store.get_event(event_id)
        if not event:
            raise HTTPException(404, f"Event not found: {event_id}")
        return event.to_dict()

    @app.post("/api/timeline/record")
    async def record_timeline_event(
        event_type: str = Query(...),
        agent_id: Optional[str] = Query(None),
        title: str = Query(""),
        description: str = Query(""),
        severity: str = Query("info"),
    ):
        """Record a custom timeline event."""
        from ..meta.timeline import get_timeline_store, EventType, EventSeverity

        store = get_timeline_store()

        try:
            et = EventType(event_type)
        except ValueError:
            et = EventType.CUSTOM

        try:
            sev = EventSeverity(severity)
        except ValueError:
            sev = EventSeverity.INFO

        event = await store.record(
            event_type=et,
            agent_id=agent_id,
            title=title,
            description=description,
            severity=sev,
        )
        return event.to_dict()

    @app.get("/api/timeline/event-types")
    async def list_timeline_event_types():
        """List all available event types."""
        from ..meta.timeline import EventType, EventSeverity

        return {
            "event_types": [e.value for e in EventType],
            "severities": [s.value for s in EventSeverity],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # LLM Provider & Model Configuration API
    # ─────────────────────────────────────────────────────────────────────────

    @app.get("/api/llm/providers")
    async def list_llm_providers():
        """List all configured LLM providers."""
        from ..llm.providers import get_model_registry

        registry = get_model_registry()
        providers = registry.list_providers(enabled_only=False)
        return {
            "providers": [p.to_dict() for p in providers],
            "count": len(providers),
        }

    @app.get("/api/llm/providers/{provider_id}")
    async def get_llm_provider(provider_id: str):
        """Get details of a specific provider."""
        from ..llm.providers import get_model_registry

        registry = get_model_registry()
        provider = registry.get_provider(provider_id)
        if not provider:
            raise HTTPException(404, f"Provider not found: {provider_id}")
        return provider.to_dict()

    @app.get("/api/llm/providers/{provider_id}/models")
    async def list_provider_models(provider_id: str):
        """List all models for a provider."""
        from ..llm.providers import get_model_registry

        registry = get_model_registry()
        models = registry.get_models_by_provider(provider_id)
        return {
            "provider_id": provider_id,
            "models": [m.to_dict() for m in models],
            "count": len(models),
        }

    @app.get("/api/llm/models/all")
    async def list_all_llm_models(
        provider: Optional[str] = Query(None),
        capability: Optional[str] = Query(None),
    ):
        """List all available LLM models with optional filtering."""
        from ..llm.providers import get_model_registry, ProviderType, ModelCapability

        registry = get_model_registry()

        # Auto-initialize if needed
        if not registry._initialized:
            await registry.initialize()

        provider_filter = None
        if provider:
            try:
                provider_filter = ProviderType(provider)
            except ValueError:
                pass

        capability_filter = None
        if capability:
            try:
                capability_filter = ModelCapability(capability)
            except ValueError:
                pass

        models = registry.list_models(provider=provider_filter, capability=capability_filter)
        return {
            "models": [m.to_dict() for m in models],
            "count": len(models),
        }

    @app.post("/api/llm/models/refresh")
    async def refresh_llm_models():
        """Refresh models from OpenRouter API (fetches latest)."""
        from ..llm.providers import get_model_registry

        registry = get_model_registry()
        await registry.refresh_models()

        models = registry.list_models()
        return {
            "status": "refreshed",
            "models_count": len(models),
            "models": [m.to_dict() for m in models[:20]],  # Return first 20
        }

    @app.get("/api/llm/models/latest")
    async def get_latest_models(
        models_per_provider: int = Query(5, ge=1, le=20),
        dedupe: bool = Query(True),
    ):
        """Fetch latest models from all providers via OpenRouter."""
        from ..llm.providers import fetch_latest_models

        results = await fetch_latest_models(
            models_per_provider=models_per_provider,
            dedupe=dedupe,
        )

        return {
            "providers": {
                provider: [m.to_dict() for m in models]
                for provider, models in results.items()
            },
            "provider_count": len(results),
            "total_models": sum(len(models) for models in results.values()),
        }

    @app.get("/api/llm/models/provider/{provider_prefix}")
    async def get_provider_latest_models(
        provider_prefix: str,
        limit: int = Query(10, ge=1, le=50),
        dedupe: bool = Query(True),
    ):
        """Get latest models for a specific provider."""
        from ..llm.providers import OpenRouterModelFetcher

        fetcher = OpenRouterModelFetcher()
        models = await fetcher.get_latest_by_provider(
            provider_prefix=provider_prefix,
            limit=limit,
            dedupe=dedupe,
        )

        return {
            "provider": provider_prefix,
            "models": [m.to_dict() for m in models],
            "count": len(models),
        }

    @app.get("/api/llm/models/{model_id}")
    async def get_llm_model(model_id: str):
        """Get details of a specific model."""
        from ..llm.providers import get_model_registry

        registry = get_model_registry()
        model = registry.get_model(model_id)
        if not model:
            raise HTTPException(404, f"Model not found: {model_id}")
        return model.to_dict()

    @app.post("/api/llm/complete")
    async def llm_complete(
        model_id: str = Query("claude-sonnet-4.5"),
        temperature: float = Query(0.7, ge=0.0, le=2.0),
        max_tokens: int = Query(4096, ge=1, le=16384),
        messages: list = Body(..., description="Chat messages"),
    ):
        """Complete a chat conversation using any configured provider."""
        from ..llm.providers import get_unified_client

        client = get_unified_client()
        try:
            result = await client.complete(
                messages=messages,
                model_id=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            return result
        except Exception as e:
            raise HTTPException(500, str(e))

    @app.post("/api/llm/stream")
    async def llm_stream(
        model_id: str = Query("claude-sonnet-4.5"),
        temperature: float = Query(0.7, ge=0.0, le=2.0),
        max_tokens: int = Query(4096, ge=1, le=16384),
        messages: list = Body(..., description="Chat messages"),
    ):
        """Stream a chat completion."""
        from ..llm.providers import get_unified_client
        import json

        client = get_unified_client()

        async def generate():
            try:
                stream = await client.complete(
                    messages=messages,
                    model_id=model_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                async for chunk in stream:
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
        )

    @app.get("/api/llm/capabilities")
    async def list_model_capabilities():
        """List all model capabilities."""
        from ..llm.providers import ModelCapability

        return {
            "capabilities": [
                {"value": c.value, "name": c.name}
                for c in ModelCapability
            ],
        }

    @app.post("/api/llm/providers/register")
    async def register_custom_provider(
        provider_id: str = Query(...),
        name: str = Query(...),
        provider_type: str = Query("custom"),
        base_url: str = Query(...),
        api_key_env: str = Query(""),
        default_model: Optional[str] = Query(None),
    ):
        """Register a custom LLM provider."""
        from ..llm.providers import get_model_registry, ProviderConfig, ProviderType

        registry = get_model_registry()

        try:
            ptype = ProviderType(provider_type)
        except ValueError:
            ptype = ProviderType.CUSTOM

        config = ProviderConfig(
            provider_id=provider_id,
            provider_type=ptype,
            name=name,
            base_url=base_url,
            api_key_env=api_key_env,
            default_model=default_model,
        )

        registry.register_provider(config)
        return config.to_dict()

    @app.post("/api/llm/models/register")
    async def register_custom_model(
        model_id: str = Query(...),
        name: str = Query(...),
        provider: str = Query(...),
        provider_model_id: str = Query(...),
        context_length: int = Query(8192),
        max_output_tokens: int = Query(4096),
        capabilities: str = Query("chat,streaming", description="Comma-separated capabilities"),
    ):
        """Register a custom model."""
        from ..llm.providers import get_model_registry, ModelConfig, ProviderType, ModelCapability

        registry = get_model_registry()

        try:
            ptype = ProviderType(provider)
        except ValueError:
            ptype = ProviderType.CUSTOM

        caps = []
        for cap in capabilities.split(","):
            cap = cap.strip()
            try:
                caps.append(ModelCapability(cap))
            except ValueError:
                pass

        config = ModelConfig(
            model_id=model_id,
            name=name,
            provider=ptype,
            provider_model_id=provider_model_id,
            context_length=context_length,
            max_output_tokens=max_output_tokens,
            capabilities=caps,
        )

        registry.register_model(config)
        return config.to_dict()

    # ─────────────────────────────────────────────────────────────────────────
    # Personality API
    # ─────────────────────────────────────────────────────────────────────────

    @app.get("/api/personality/archetypes")
    async def list_personality_archetypes():
        """List available personality archetypes."""
        from ..meta.personality import get_personality_manager, PERSONALITY_ARCHETYPES
        manager = get_personality_manager()
        return {
            "archetypes": list(PERSONALITY_ARCHETYPES.keys()),
            "count": len(PERSONALITY_ARCHETYPES),
        }

    @app.get("/api/personality/list")
    async def list_personalities():
        """List all created personalities."""
        from ..meta.personality import get_personality_manager
        manager = get_personality_manager()
        personalities = []
        for name in manager.list_personalities():
            p = manager.get(name)
            if p:
                personalities.append({
                    "name": p.name,
                    "archetype": p.archetype,
                    "crystallization_level": p.crystallization_level,
                    "reinforcement_count": p.reinforcement_count,
                })
        return {"personalities": personalities}

    @app.post("/api/personality/create")
    async def create_personality(
        name: str = Query(...),
        archetype: Optional[str] = Query(None),
        seed: Optional[str] = Query(None),
        core_motivation: Optional[str] = Query(None),
    ):
        """Create a new personality."""
        from ..meta.personality import get_personality_manager
        manager = get_personality_manager()

        kwargs = {}
        if core_motivation:
            kwargs["core_motivation"] = core_motivation

        personality = manager.create(name, archetype=archetype, seed=seed, **kwargs)
        return {
            "status": "created",
            "personality": personality.to_dict(),
        }

    @app.get("/api/personality/{name}")
    async def get_personality(name: str):
        """Get personality details."""
        from ..meta.personality import get_personality_manager
        manager = get_personality_manager()
        personality = manager.get(name)
        if not personality:
            raise HTTPException(404, f"Personality not found: {name}")
        return personality.to_dict()

    @app.get("/api/personality/{name}/prompt")
    async def get_personality_prompt(name: str):
        """Get system prompt for a personality."""
        from ..meta.personality import get_personality_manager
        manager = get_personality_manager()
        prompt = manager.generate_prompt(name)
        if not prompt:
            raise HTTPException(404, f"Personality not found: {name}")
        return {"name": name, "system_prompt": prompt}

    @app.post("/api/personality/{name}/crystallize")
    async def crystallize_personality(
        name: str,
        technique: str = Query(..., description="feedback, interaction, contrast, reflection"),
        feedback: Optional[str] = Query(None),
        aspect: Optional[str] = Query(None),
        reflection: Optional[str] = Query(None),
    ):
        """Apply crystallization technique to reinforce personality."""
        from ..meta.personality import get_personality_manager
        manager = get_personality_manager()

        context = {}
        if feedback:
            context["feedback"] = feedback
        if aspect:
            context["aspect"] = aspect
        if reflection:
            context["reflection"] = reflection

        personality = manager.crystallize(name, technique, context)
        if not personality:
            raise HTTPException(404, f"Personality not found: {name}")

        return {
            "status": "crystallized",
            "crystallization_level": personality.crystallization_level,
            "reinforcement_count": personality.reinforcement_count,
        }

    @app.get("/api/personality/{name}/traits")
    async def get_personality_traits(name: str):
        """Get personality trait breakdown."""
        from ..meta.personality import get_personality_manager
        manager = get_personality_manager()
        personality = manager.get(name)
        if not personality:
            raise HTTPException(404, f"Personality not found: {name}")

        return {
            "name": name,
            "traits": personality.traits.to_dict(),
            "dominant_traits": [
                {"dimension": d.value, "value": v}
                for d, v in personality.traits.dominant_traits()
            ],
            "primary_cognitive_style": personality.traits.primary_cognitive_style().value,
            "dominant_values": [
                {"value": v.value, "strength": s}
                for v, s in personality.traits.dominant_values()
            ],
        }

    @app.get("/api/personality/{name}/voice")
    async def get_personality_voice(name: str):
        """Get personality voice pattern."""
        from ..meta.personality import get_personality_manager
        manager = get_personality_manager()
        personality = manager.get(name)
        if not personality:
            raise HTTPException(404, f"Personality not found: {name}")

        return {
            "name": name,
            "voice": personality.voice.to_dict(),
        }

    @app.post("/api/personality/{name}/adjust-trait")
    async def adjust_personality_trait(
        name: str,
        dimension: str = Query(...),
        delta: float = Query(..., ge=-1.0, le=1.0),
    ):
        """Adjust a specific trait dimension."""
        from ..meta.personality import get_personality_manager, TraitDimension
        manager = get_personality_manager()
        personality = manager.get(name)
        if not personality:
            raise HTTPException(404, f"Personality not found: {name}")

        try:
            dim = TraitDimension(dimension)
        except ValueError:
            raise HTTPException(400, f"Invalid dimension: {dimension}")

        old_value = personality.traits.get_trait(dim)
        personality.traits.adjust_trait(dim, delta)
        new_value = personality.traits.get_trait(dim)

        return {
            "dimension": dimension,
            "old_value": old_value,
            "new_value": new_value,
            "delta_applied": new_value - old_value,
        }

    @app.post("/api/personality/generate-from-seed")
    async def generate_personality_from_seed(
        name: str = Query(...),
        seed: str = Query(...),
    ):
        """Generate a deterministic personality from a seed string."""
        from ..meta.personality import get_personality_manager
        manager = get_personality_manager()
        personality = manager.create(name, seed=seed)
        return {
            "status": "generated",
            "seed": seed,
            "personality": personality.to_dict(),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Advanced Personality API
    # ─────────────────────────────────────────────────────────────────────────

    # Store advanced personalities separately
    _advanced_personalities: Dict[str, Any] = {}

    @app.post("/api/personality/advanced/create")
    async def create_advanced_personality_endpoint(
        name: str = Query(...),
        archetype: Optional[str] = Query(None),
        seed: Optional[str] = Query(None),
        origin_story: Optional[str] = Query(None),
        core_beliefs: Optional[str] = Query(None, description="Comma-separated beliefs"),
    ):
        """Create an advanced personality with full psychological depth."""
        from ..meta.personality_advanced import create_advanced_personality

        beliefs = [b.strip() for b in core_beliefs.split(",")] if core_beliefs else None
        persona = create_advanced_personality(
            name=name,
            archetype=archetype,
            seed=seed,
            origin_story=origin_story or "",
            core_beliefs=beliefs,
        )
        _advanced_personalities[name] = persona
        return {
            "status": "created",
            "personality": persona.to_dict(),
        }

    @app.get("/api/personality/advanced/{name}")
    async def get_advanced_personality(name: str):
        """Get advanced personality details."""
        if name not in _advanced_personalities:
            raise HTTPException(404, f"Advanced personality not found: {name}")
        return _advanced_personalities[name].to_dict()

    @app.get("/api/personality/advanced/{name}/prompt")
    async def get_advanced_personality_prompt(name: str):
        """Get system prompt for advanced personality."""
        if name not in _advanced_personalities:
            raise HTTPException(404, f"Advanced personality not found: {name}")
        return {
            "name": name,
            "system_prompt": _advanced_personalities[name].generate_system_prompt(),
        }

    @app.post("/api/personality/advanced/{name}/experience")
    async def process_personality_experience(
        name: str,
        content: str = Query(..., description="Experience description"),
        emotional_valence: float = Query(0.0, ge=-1.0, le=1.0),
        emotional_intensity: float = Query(0.3, ge=0.0, le=1.0),
        entity_id: Optional[str] = Query(None),
        significance: float = Query(0.5, ge=0.0, le=1.0),
    ):
        """Process an experience through the personality's psychological systems."""
        if name not in _advanced_personalities:
            raise HTTPException(404, f"Advanced personality not found: {name}")

        persona = _advanced_personalities[name]
        memory = persona.process_experience(
            content=content,
            emotional_valence=emotional_valence,
            emotional_intensity=emotional_intensity,
            entity_id=entity_id,
            significance=significance,
        )
        return {
            "memory_encoded": memory.to_dict(),
            "current_mood": persona.emotions.get_mood().value,
            "dominant_emotion": persona.emotions.get_dominant_emotion()[0].value,
            "shadow_activation": persona.shadow.activation_level,
            "developmental_stage": persona.developmental_stage,
        }

    @app.post("/api/personality/advanced/{name}/emotion")
    async def trigger_personality_emotion(
        name: str,
        emotion: str = Query(..., description="joy, trust, fear, surprise, sadness, disgust, anger, anticipation"),
        intensity: float = Query(0.5, ge=0.0, le=1.0),
        cause: str = Query(""),
    ):
        """Trigger an emotional response in the personality."""
        from ..meta.personality_advanced import EmotionalState

        if name not in _advanced_personalities:
            raise HTTPException(404, f"Advanced personality not found: {name}")

        try:
            emotion_enum = EmotionalState(emotion)
        except ValueError:
            raise HTTPException(400, f"Invalid emotion: {emotion}")

        persona = _advanced_personalities[name]
        persona.emotions.trigger(emotion_enum, intensity, cause)

        return {
            "triggered": emotion,
            "intensity": intensity,
            "resulting_mood": persona.emotions.get_mood().value,
            "emotions": persona.emotions.to_dict(),
        }

    @app.get("/api/personality/advanced/{name}/emotions")
    async def get_personality_emotions(name: str):
        """Get current emotional state of personality."""
        if name not in _advanced_personalities:
            raise HTTPException(404, f"Advanced personality not found: {name}")
        return _advanced_personalities[name].emotions.to_dict()

    @app.post("/api/personality/advanced/{name}/context")
    async def set_personality_context(
        name: str,
        context: str = Query(..., description="professional, casual, intimate, authority, subordinate, peer, stranger, conflict, celebration, crisis"),
    ):
        """Set the social context for the personality."""
        from ..meta.personality_advanced import SocialContext

        if name not in _advanced_personalities:
            raise HTTPException(404, f"Advanced personality not found: {name}")

        try:
            context_enum = SocialContext(context)
        except ValueError:
            raise HTTPException(400, f"Invalid context: {context}")

        persona = _advanced_personalities[name]
        persona.masks.set_context(context_enum)

        return {
            "context_set": context,
            "active_mask": persona.masks.get_current_mask().name if persona.masks.get_current_mask() else None,
        }

    @app.get("/api/personality/advanced/{name}/relationships")
    async def get_personality_relationships(name: str):
        """Get all relationships for the personality."""
        if name not in _advanced_personalities:
            raise HTTPException(404, f"Advanced personality not found: {name}")

        persona = _advanced_personalities[name]
        return {
            "stats": persona.relationships.get_stats(),
            "relationships": [r.to_dict() for r in persona.relationships.relationships.values()],
            "closest": [r.to_dict() for r in persona.relationships.get_closest(5)],
        }

    @app.get("/api/personality/advanced/{name}/memories")
    async def get_personality_memories(
        name: str,
        query: Optional[str] = Query(None),
        limit: int = Query(10, ge=1, le=50),
    ):
        """Recall memories from the personality's memory system."""
        if name not in _advanced_personalities:
            raise HTTPException(404, f"Advanced personality not found: {name}")

        persona = _advanced_personalities[name]
        memories = persona.memory.recall(query=query, limit=limit)
        return {
            "stats": persona.memory.get_stats(),
            "memories": [m.to_dict() for m in memories],
        }

    @app.get("/api/personality/advanced/{name}/shadow")
    async def get_personality_shadow(name: str):
        """Get shadow self state."""
        if name not in _advanced_personalities:
            raise HTTPException(404, f"Advanced personality not found: {name}")
        return _advanced_personalities[name].shadow.to_dict()

    @app.get("/api/personality/advanced/{name}/conflicts")
    async def get_personality_conflicts(name: str):
        """Get internal conflicts."""
        if name not in _advanced_personalities:
            raise HTTPException(404, f"Advanced personality not found: {name}")

        persona = _advanced_personalities[name]
        active = persona.conflicts.get_active_conflicts()
        return {
            "all_conflicts": [
                {
                    "id": c.conflict_id,
                    "pole_a": c.pole_a,
                    "pole_b": c.pole_b,
                    "tension": c.tension,
                    "dominant": c.get_dominant_pole()[0],
                    "resolved": c.is_resolved(),
                }
                for c in persona.conflicts.conflicts.values()
            ],
            "active_conflicts": [
                {
                    "id": c.conflict_id,
                    "pole_a": c.pole_a,
                    "pole_b": c.pole_b,
                    "tension": c.tension,
                }
                for c in active
            ],
        }

    @app.get("/api/personality/advanced/{name}/narrative")
    async def get_personality_narrative(name: str):
        """Get narrative identity."""
        if name not in _advanced_personalities:
            raise HTTPException(404, f"Advanced personality not found: {name}")

        persona = _advanced_personalities[name]
        return {
            "narrative": persona.narrative.to_dict(),
            "self_narrative": persona.narrative.generate_self_narrative(),
        }

    @app.post("/api/personality/advanced/{name}/narrative/moment")
    async def add_defining_moment(
        name: str,
        description: str = Query(...),
        significance: str = Query(...),
        lesson_learned: str = Query(...),
    ):
        """Add a defining moment to the personality's narrative."""
        if name not in _advanced_personalities:
            raise HTTPException(404, f"Advanced personality not found: {name}")

        persona = _advanced_personalities[name]
        persona.narrative.add_defining_moment(description, significance, lesson_learned)
        return {
            "status": "added",
            "defining_moments_count": len(persona.narrative.defining_moments),
        }

    @app.get("/api/personality/advanced/{name}/effective-traits")
    async def get_effective_traits(name: str):
        """Get effective trait values (with shadow, mask, mood applied)."""
        from ..meta.personality import TraitDimension

        if name not in _advanced_personalities:
            raise HTTPException(404, f"Advanced personality not found: {name}")

        persona = _advanced_personalities[name]
        return {
            "context": persona.masks.current_context.value,
            "shadow_activation": persona.shadow.activation_level,
            "mood": persona.emotions.get_mood().value,
            "effective_traits": {
                dim.value: persona.get_effective_trait(dim)
                for dim in TraitDimension
            },
            "base_traits": {
                dim.value: persona.traits.get_trait(dim)
                for dim in TraitDimension
            },
        }

    @app.get("/api/personality/advanced/list")
    async def list_advanced_personalities():
        """List all advanced personalities."""
        return {
            "personalities": [
                {
                    "name": name,
                    "archetype": p.archetype,
                    "stage": p.developmental_stage,
                    "mood": p.emotions.get_mood().value,
                    "growth_points": p.growth_points,
                }
                for name, p in _advanced_personalities.items()
            ]
        }

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
        base_info["persona_prompt"] = getattr(agent, "persona_prompt", "")
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
