"""
Interactive Simulation Runner

Supports:
- Long-running simulations with persistence
- Pause/resume/step execution
- Edit messages and actions mid-simulation
- Real-time state inspection
- Save/load simulation checkpoints
"""

from __future__ import annotations

import asyncio
import json
import pickle
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .driver import SimulationDriver, Character, ActionResult
from .scenarios import (
    ScenarioDefinition,
    ScenarioContext,
    ScenarioAction,
    Phase,
    ScenarioExecutor,
)


class SimulationState(Enum):
    """State of the simulation."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STEPPING = "stepping"
    WAITING_INPUT = "waiting_input"
    COMPLETED = "completed"
    ERROR = "error"


class EditMode(Enum):
    """How to handle editable content."""
    AUTO = "auto"           # Run automatically, no prompts
    CONFIRM = "confirm"     # Show content, ask to confirm or edit
    ALWAYS_EDIT = "always"  # Always prompt for edits


@dataclass
class SimulationEvent:
    """An event in the simulation timeline."""
    id: str
    timestamp: datetime
    event_type: str
    phase: str
    actor: Optional[str]
    action_type: Optional[str]
    content: Any
    editable: bool = False
    edited: bool = False
    original_content: Optional[Any] = None


@dataclass
class SimulationCheckpoint:
    """A saved state of the simulation."""
    id: str
    timestamp: datetime
    phase_name: str
    phase_index: int
    action_index: int
    context_state: Dict[str, Any]
    character_states: Dict[str, Dict[str, Any]]
    events: List[SimulationEvent]
    variables: Dict[str, Any]


@dataclass
class PendingEdit:
    """Content waiting for user edit."""
    event_id: str
    actor: str
    action_type: str
    current_content: str
    phase: str
    callback: Optional[Callable] = None


class InteractiveSimulation:
    """
    Interactive simulation runner with pause, edit, and persistence.

    Usage:
        sim = InteractiveSimulation(driver, scenario)

        # Run with auto mode
        await sim.run()

        # Run with step-by-step
        sim.set_edit_mode(EditMode.CONFIRM)
        await sim.step()  # Execute one action
        await sim.step()  # Execute next action

        # Pause and edit
        sim.pause()
        sim.edit_last_message("New content here")
        sim.resume()

        # Save/load
        sim.save_checkpoint("my_save.pkl")
        sim.load_checkpoint("my_save.pkl")
    """

    def __init__(
        self,
        driver: SimulationDriver,
        scenario: ScenarioDefinition,
        role_assignments: Dict[str, str] = None,
        edit_mode: EditMode = EditMode.AUTO,
        checkpoint_dir: str = "./checkpoints",
    ):
        self.driver = driver
        self.scenario = scenario
        self.role_assignments = role_assignments or {}
        self.edit_mode = edit_mode
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.state = SimulationState.IDLE
        self.current_phase_index = 0
        self.current_action_index = 0
        self.events: List[SimulationEvent] = []
        self.pending_edit: Optional[PendingEdit] = None

        # Callbacks
        self.on_event: Optional[Callable[[SimulationEvent], None]] = None
        self.on_state_change: Optional[Callable[[SimulationState], None]] = None
        self.on_edit_request: Optional[Callable[[PendingEdit], str]] = None

        # Context
        self.context: Optional[ScenarioContext] = None
        self.executor = ScenarioExecutor(driver)

        # Timing
        self.start_time: Optional[datetime] = None
        self.pause_time: Optional[datetime] = None
        self.total_pause_duration: float = 0

        # Session ID
        self.session_id = str(uuid.uuid4())[:8]

    def _set_state(self, new_state: SimulationState):
        """Update state and notify."""
        old_state = self.state
        self.state = new_state
        if self.on_state_change and old_state != new_state:
            self.on_state_change(new_state)

    def _add_event(
        self,
        event_type: str,
        phase: str,
        actor: Optional[str] = None,
        action_type: Optional[str] = None,
        content: Any = None,
        editable: bool = False,
    ) -> SimulationEvent:
        """Add an event to the timeline."""
        event = SimulationEvent(
            id=f"evt_{len(self.events):04d}",
            timestamp=datetime.now(),
            event_type=event_type,
            phase=phase,
            actor=actor,
            action_type=action_type,
            content=content,
            editable=editable,
        )
        self.events.append(event)
        if self.on_event:
            self.on_event(event)
        return event

    def _resolve_character(self, char_ref: str) -> Optional[str]:
        """Resolve @role to character ID."""
        if char_ref.startswith("@"):
            role = char_ref[1:]
            return self.role_assignments.get(role)
        return char_ref

    async def initialize(self):
        """Initialize the simulation context."""
        self.context = ScenarioContext(
            scenario_id=self.scenario.name,
            participants=list(self.role_assignments.values()),
            roles=dict(self.role_assignments),
        )

        # Set initial variables
        for key, value in self.scenario.initial_variables.items():
            self.context.variables[key] = value

        # Set initial beliefs
        for role, beliefs in self.scenario.initial_beliefs.items():
            char_id = self.role_assignments.get(role)
            if char_id:
                char = self.driver.get_character(char_id)
                if char:
                    for belief_key, belief_val in beliefs.items():
                        char.beliefs[belief_key] = belief_val

        self.start_time = datetime.now()
        self._add_event("simulation_start", "init", content={
            "scenario": self.scenario.name,
            "roles": self.role_assignments,
        })

        self._set_state(SimulationState.PAUSED)

    async def run(self, max_duration: float = None) -> Dict[str, Any]:
        """
        Run the simulation to completion.

        Args:
            max_duration: Maximum runtime in seconds (None = no limit)

        Returns:
            Simulation results dict
        """
        if self.state == SimulationState.IDLE:
            await self.initialize()

        self._set_state(SimulationState.RUNNING)

        start = time.time()

        while self.current_phase_index < len(self.scenario.phases):
            # Check timeout
            if max_duration and (time.time() - start) > max_duration:
                self._add_event("timeout", self._current_phase_name())
                break

            # Check if paused
            if self.state == SimulationState.PAUSED:
                await asyncio.sleep(0.1)
                continue

            # Check if waiting for input
            if self.state == SimulationState.WAITING_INPUT:
                await asyncio.sleep(0.1)
                continue

            # Execute current action
            await self._execute_current_action()

            # Allow event loop to process
            await asyncio.sleep(0)

        self._set_state(SimulationState.COMPLETED)
        return self._build_result()

    async def step(self) -> Optional[SimulationEvent]:
        """Execute a single action and pause."""
        if self.state == SimulationState.IDLE:
            await self.initialize()

        if self.current_phase_index >= len(self.scenario.phases):
            self._set_state(SimulationState.COMPLETED)
            return None

        self._set_state(SimulationState.STEPPING)
        event = await self._execute_current_action()
        self._set_state(SimulationState.PAUSED)

        return event

    async def _execute_current_action(self) -> Optional[SimulationEvent]:
        """Execute the current action in the current phase."""
        if self.current_phase_index >= len(self.scenario.phases):
            return None

        phase = self.scenario.phases[self.current_phase_index]

        # Handle phase entry
        if self.current_action_index == 0:
            self._add_event("phase_enter", phase.name)

            # Execute on_enter actions
            if phase.on_enter:
                for action in phase.on_enter:
                    await self._execute_action(action, phase.name)

        # Execute current action
        if self.current_action_index < len(phase.actions):
            action = phase.actions[self.current_action_index]
            event = await self._execute_action(action, phase.name)
            self.current_action_index += 1
            return event

        # Phase complete - handle exit and transition
        if phase.on_exit:
            for action in phase.on_exit:
                await self._execute_action(action, phase.name)

        self._add_event("phase_exit", phase.name)

        # Find next phase
        next_phase = self._find_next_phase(phase)
        if next_phase:
            self.current_phase_index = next(
                i for i, p in enumerate(self.scenario.phases)
                if p.name == next_phase
            )
            self.current_action_index = 0
        else:
            self.current_phase_index = len(self.scenario.phases)

        return None

    async def _execute_action(
        self,
        action: ScenarioAction,
        phase_name: str
    ) -> SimulationEvent:
        """Execute a single action with edit support."""
        # Resolve character reference
        char_id = None
        if action.character_id:
            char_id = self._resolve_character(action.character_id)

        # Check if this is an editable action
        is_editable = action.action_type == "character_action" and \
                      action.params.get("type") in ["say", "tell", "ask", "propose", "reply"]

        content = action.params.get("content", "")

        # Handle edit mode
        if is_editable and self.edit_mode != EditMode.AUTO:
            event = self._add_event(
                "pending_action",
                phase_name,
                actor=char_id,
                action_type=action.params.get("type"),
                content=content,
                editable=True,
            )

            if self.edit_mode == EditMode.CONFIRM or self.edit_mode == EditMode.ALWAYS_EDIT:
                # Request edit
                self.pending_edit = PendingEdit(
                    event_id=event.id,
                    actor=char_id or "unknown",
                    action_type=action.params.get("type", "action"),
                    current_content=content,
                    phase=phase_name,
                )

                if self.on_edit_request:
                    self._set_state(SimulationState.WAITING_INPUT)
                    new_content = self.on_edit_request(self.pending_edit)
                    if new_content and new_content != content:
                        event.original_content = content
                        event.content = new_content
                        event.edited = True
                        content = new_content
                    self._set_state(SimulationState.RUNNING)

                self.pending_edit = None

        # Execute the action
        result = await self.executor._execute_action(
            action,
            self.context,
            self.role_assignments
        )

        # Record event
        event = self._add_event(
            "action_executed",
            phase_name,
            actor=char_id,
            action_type=action.action_type,
            content={"params": action.params, "result": str(result) if result else None},
            editable=is_editable,
        )

        return event

    def _find_next_phase(self, current_phase: Phase) -> Optional[str]:
        """Find the next phase based on transitions."""
        if not current_phase.transitions:
            return None

        for transition in current_phase.transitions:
            if transition.condition is None:
                return transition.target

            # Evaluate condition
            if self._evaluate_condition(transition.condition):
                return transition.target

        return None

    def _evaluate_condition(self, condition) -> bool:
        """Evaluate a transition condition."""
        # Simplified condition evaluation
        if hasattr(condition, 'evaluate'):
            return condition.evaluate(self.context.variables)
        return True

    def _current_phase_name(self) -> str:
        """Get current phase name."""
        if self.current_phase_index < len(self.scenario.phases):
            return self.scenario.phases[self.current_phase_index].name
        return "completed"

    def pause(self):
        """Pause the simulation."""
        if self.state == SimulationState.RUNNING:
            self.pause_time = datetime.now()
            self._set_state(SimulationState.PAUSED)
            self._add_event("paused", self._current_phase_name())

    def resume(self):
        """Resume the simulation."""
        if self.state == SimulationState.PAUSED:
            if self.pause_time:
                self.total_pause_duration += (datetime.now() - self.pause_time).total_seconds()
                self.pause_time = None
            self._set_state(SimulationState.RUNNING)
            self._add_event("resumed", self._current_phase_name())

    def submit_edit(self, new_content: str):
        """Submit edited content for pending action."""
        if self.pending_edit and self.state == SimulationState.WAITING_INPUT:
            # Find and update the event
            for event in reversed(self.events):
                if event.id == self.pending_edit.event_id:
                    event.original_content = event.content
                    event.content = new_content
                    event.edited = True
                    break

            self.pending_edit = None
            self._set_state(SimulationState.RUNNING)

    def edit_event(self, event_id: str, new_content: Any):
        """Edit a past event (for replay scenarios)."""
        for event in self.events:
            if event.id == event_id and event.editable:
                event.original_content = event.content
                event.content = new_content
                event.edited = True
                return True
        return False

    def get_last_editable_event(self) -> Optional[SimulationEvent]:
        """Get the most recent editable event."""
        for event in reversed(self.events):
            if event.editable:
                return event
        return None

    def edit_last_message(self, new_content: str) -> bool:
        """Edit the last editable message."""
        event = self.get_last_editable_event()
        if event:
            event.original_content = event.content
            event.content = new_content
            event.edited = True
            return True
        return False

    def get_timeline(self) -> List[Dict[str, Any]]:
        """Get the event timeline."""
        return [
            {
                "id": e.id,
                "time": e.timestamp.isoformat(),
                "type": e.event_type,
                "phase": e.phase,
                "actor": e.actor,
                "action": e.action_type,
                "content": e.content,
                "edited": e.edited,
            }
            for e in self.events
        ]

    def get_character_states(self) -> Dict[str, Dict[str, Any]]:
        """Get current state of all characters."""
        states = {}
        for role, char_id in self.role_assignments.items():
            char = self.driver.get_character(char_id)
            if char:
                states[role] = {
                    "id": char_id,
                    "name": char.name,
                    "beliefs": dict(char.beliefs),
                    "action_count": len(char.action_history),
                }
        return states

    def save_checkpoint(self, name: str = None) -> str:
        """Save current state to checkpoint file."""
        if name is None:
            name = f"checkpoint_{self.session_id}_{len(self.events):04d}"

        checkpoint = SimulationCheckpoint(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            phase_name=self._current_phase_name(),
            phase_index=self.current_phase_index,
            action_index=self.current_action_index,
            context_state=self.context.variables if self.context else {},
            character_states=self.get_character_states(),
            events=self.events.copy(),
            variables=self.context.variables if self.context else {},
        )

        filepath = self.checkpoint_dir / f"{name}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(checkpoint, f)

        self._add_event("checkpoint_saved", self._current_phase_name(), content={"file": str(filepath)})
        return str(filepath)

    def load_checkpoint(self, filepath: str) -> bool:
        """Load state from checkpoint file."""
        try:
            with open(filepath, "rb") as f:
                checkpoint: SimulationCheckpoint = pickle.load(f)

            self.current_phase_index = checkpoint.phase_index
            self.current_action_index = checkpoint.action_index
            self.events = checkpoint.events

            if self.context:
                self.context.variables = checkpoint.context_state

            # Restore character states
            for role, state in checkpoint.character_states.items():
                char_id = self.role_assignments.get(role)
                if char_id:
                    char = self.driver.get_character(char_id)
                    if char:
                        char.beliefs = state.get("beliefs", {})

            self._set_state(SimulationState.PAUSED)
            self._add_event("checkpoint_loaded", self._current_phase_name(), content={"file": filepath})
            return True
        except Exception as e:
            self._add_event("error", self._current_phase_name(), content={"error": str(e)})
            return False

    def list_checkpoints(self) -> List[str]:
        """List available checkpoints."""
        return [str(f) for f in self.checkpoint_dir.glob("*.pkl")]

    def _build_result(self) -> Dict[str, Any]:
        """Build the final result dict."""
        duration = 0
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds() - self.total_pause_duration

        return {
            "session_id": self.session_id,
            "scenario_name": self.scenario.name,
            "state": self.state.value,
            "duration_seconds": duration,
            "phases_completed": self.current_phase_index,
            "total_phases": len(self.scenario.phases),
            "events": len(self.events),
            "edits_made": sum(1 for e in self.events if e.edited),
            "participants": list(self.role_assignments.values()),
            "final_state": self.get_character_states(),
            "variables": self.context.variables if self.context else {},
            "timeline": self.get_timeline(),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current simulation status."""
        return {
            "state": self.state.value,
            "session_id": self.session_id,
            "scenario": self.scenario.name,
            "current_phase": self._current_phase_name(),
            "phase_progress": f"{self.current_phase_index + 1}/{len(self.scenario.phases)}",
            "action_index": self.current_action_index,
            "events_count": len(self.events),
            "pending_edit": self.pending_edit is not None,
        }


class SimulationCLI:
    """Command-line interface for interactive simulations."""

    def __init__(self, simulation: InteractiveSimulation):
        self.sim = simulation
        self.running = False

        # Wire up callbacks
        self.sim.on_event = self._on_event
        self.sim.on_state_change = self._on_state_change
        self.sim.on_edit_request = self._on_edit_request

    def _on_event(self, event: SimulationEvent):
        """Handle simulation events."""
        if event.event_type == "action_executed":
            actor = event.actor or "System"
            char = self.sim.driver.get_character(actor) if actor != "System" else None
            name = char.name if char else actor

            if event.action_type == "character_action":
                content = event.content.get("params", {})
                action_type = content.get("type", "action")
                message = content.get("content", "")
                edited_marker = " [EDITED]" if event.edited else ""
                print(f"  [{event.phase}] {name} ({action_type}): {message}{edited_marker}")
            else:
                print(f"  [{event.phase}] {event.action_type}")

        elif event.event_type == "phase_enter":
            print(f"\n>>> Entering phase: {event.phase}")

        elif event.event_type == "phase_exit":
            print(f"<<< Exiting phase: {event.phase}")

    def _on_state_change(self, new_state: SimulationState):
        """Handle state changes."""
        print(f"[State: {new_state.value}]")

    def _on_edit_request(self, pending: PendingEdit) -> str:
        """Handle edit requests."""
        char = self.sim.driver.get_character(pending.actor)
        name = char.name if char else pending.actor

        print(f"\n--- Edit Request ---")
        print(f"Character: {name}")
        print(f"Action: {pending.action_type}")
        print(f"Current: {pending.current_content}")
        print(f"Enter new content (or press Enter to keep):")

        try:
            new_content = input("> ").strip()
            return new_content if new_content else pending.current_content
        except EOFError:
            return pending.current_content

    async def run_interactive(self):
        """Run the simulation interactively."""
        print(f"\n=== Interactive Simulation: {self.sim.scenario.name} ===")
        print("Commands: [s]tep, [r]un, [p]ause, [e]dit last, [c]heckpoint, [l]oad, [q]uit")
        print()

        await self.sim.initialize()
        self.running = True

        while self.running and self.sim.state != SimulationState.COMPLETED:
            try:
                cmd = input("\nCommand> ").strip().lower()

                if cmd in ["s", "step"]:
                    await self.sim.step()

                elif cmd in ["r", "run"]:
                    await self.sim.run(max_duration=60)

                elif cmd in ["p", "pause"]:
                    self.sim.pause()

                elif cmd in ["e", "edit"]:
                    event = self.sim.get_last_editable_event()
                    if event:
                        print(f"Current content: {event.content}")
                        new_content = input("New content: ").strip()
                        if new_content:
                            self.sim.edit_last_message(new_content)
                            print("Edited!")
                    else:
                        print("No editable events")

                elif cmd in ["c", "checkpoint", "save"]:
                    path = self.sim.save_checkpoint()
                    print(f"Saved: {path}")

                elif cmd in ["l", "load"]:
                    checkpoints = self.sim.list_checkpoints()
                    if checkpoints:
                        print("Available checkpoints:")
                        for i, cp in enumerate(checkpoints):
                            print(f"  {i}: {cp}")
                        idx = input("Load which? ").strip()
                        if idx.isdigit() and int(idx) < len(checkpoints):
                            self.sim.load_checkpoint(checkpoints[int(idx)])
                    else:
                        print("No checkpoints found")

                elif cmd in ["status", "st"]:
                    status = self.sim.get_status()
                    for k, v in status.items():
                        print(f"  {k}: {v}")

                elif cmd in ["chars", "characters"]:
                    states = self.sim.get_character_states()
                    for role, state in states.items():
                        print(f"  {role}: {state['name']} - beliefs: {state['beliefs']}")

                elif cmd in ["q", "quit", "exit"]:
                    self.running = False
                    print("Exiting...")

                elif cmd in ["h", "help", "?"]:
                    print("Commands:")
                    print("  s/step      - Execute one action")
                    print("  r/run       - Run to completion")
                    print("  p/pause     - Pause execution")
                    print("  e/edit      - Edit last message")
                    print("  c/save      - Save checkpoint")
                    print("  l/load      - Load checkpoint")
                    print("  status      - Show status")
                    print("  chars       - Show character states")
                    print("  q/quit      - Exit")

                else:
                    print("Unknown command. Type 'h' for help.")

            except KeyboardInterrupt:
                print("\nInterrupted. Type 'q' to quit or continue.")
            except EOFError:
                self.running = False

        print("\n=== Simulation Complete ===")
        result = self.sim._build_result()
        print(f"Duration: {result['duration_seconds']:.2f}s")
        print(f"Events: {result['events']}")
        print(f"Edits: {result['edits_made']}")

        return result


async def run_interactive_scenario(
    driver: SimulationDriver,
    scenario: ScenarioDefinition,
    role_assignments: Dict[str, str],
    edit_mode: EditMode = EditMode.CONFIRM,
) -> Dict[str, Any]:
    """Convenience function to run an interactive scenario."""
    sim = InteractiveSimulation(
        driver=driver,
        scenario=scenario,
        role_assignments=role_assignments,
        edit_mode=edit_mode,
    )
    cli = SimulationCLI(sim)
    return await cli.run_interactive()
