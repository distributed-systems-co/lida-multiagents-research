"""Multi-Agent Policy Simulation Orchestrator.

Integrates all simulation components:
- Simulation Engine (turn-based, continuous, negotiation)
- Coalition Dynamics (power balancing, commitments, audience costs)
- Multi-Agent Reasoning (theory of mind, strategic lookahead, belief propagation)

This is the main entry point for running sophisticated policy wargames.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from .simulation_engine import (
    Action,
    ActionType,
    AgentState,
    EscalationLevel,
    Event,
    LLMAgent,
    Persona,
    RuleBasedAgent,
    SimulationAgent,
    SimulationConfig,
    SimulationEngine,
    SimulationMode,
    WorldState,
)
from .coalition_dynamics import (
    Coalition,
    CoalitionManager,
    CoalitionType,
    CommitmentTracker,
    AudienceCostManager,
    InformationManager,
    GameTheoreticAnalyzer,
    PowerCalculator,
    PowerMetrics,
)
from .multi_agent_reasoning import (
    BeliefPropagation,
    BeliefState,
    ConsensusBuilder,
    ConversationManager,
    CoordinationProtocol,
    EnhancedReasoningAgent,
    MultiAgentCoordinator,
    ReputationSystem,
    StrategicReasoner,
    TheoryOfMind,
    VotingMethod,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Integrated Agent - Combines All Capabilities
# =============================================================================


class IntegratedAgent(SimulationAgent):
    """Agent with full suite of reasoning capabilities."""

    def __init__(
        self,
        persona: Persona,
        coordinator: MultiAgentCoordinator,
        inference_fn: Optional[Callable] = None,
        state: Optional[AgentState] = None,
    ):
        super().__init__(persona, state, inference_fn)
        self.coordinator = coordinator

        # Advanced reasoning
        self.theory_of_mind = TheoryOfMind(self.name)
        self.strategic_reasoner = StrategicReasoner(self.name, self.theory_of_mind)
        self.beliefs = BeliefState()

        # Track conversations and negotiations
        self.active_conversations: Dict[str, str] = {}  # conv_id -> role
        self.pending_proposals: List[Dict] = []

    def observe_action(self, actor: str, action: Action):
        """Process an observed action to update mental models."""
        if actor == self.name:
            return

        action_dict = {
            "type": action.action_type.value,
            "target": action.target,
            "content": action.content,
            "tick": action.tick,
        }

        # Update theory of mind
        self.theory_of_mind.update_from_action(actor, action_dict)

        # Update beliefs
        self.beliefs.update(
            f"last_action_{actor}",
            action_dict,
            confidence=0.9
        )

        # Check if action affects us
        if action.target == self.name:
            self._process_directed_action(actor, action)

        # Update emotional state based on triggers
        if self.is_triggered_by(action.content):
            self.state.stress_level = min(1.0, self.state.stress_level + 0.15)

    def _process_directed_action(self, actor: str, action: Action):
        """Process an action directed at us."""
        if action.action_type == ActionType.PROPOSAL:
            self.pending_proposals.append({
                "from": actor,
                "action": action,
                "received_at": time.time(),
            })

        elif action.action_type == ActionType.THREAT:
            # Update threat assessment
            model = self.theory_of_mind.get_model(actor)
            model.intentions.threat_level = min(1.0, model.intentions.threat_level + 0.3)

    async def decide_action(
        self,
        world: WorldState,
        visible_events: List[Event],
    ) -> Optional[Action]:
        """Decide action using full reasoning capabilities."""
        # Update beliefs from events
        for event in visible_events:
            self.beliefs.update(
                f"event_{event.event_id}",
                {"type": event.event_type, "desc": event.description[:100]},
                confidence=0.8
            )

        # Build context for strategic reasoning
        context = {
            "escalation_level": world.geopolitics.escalation_level.value,
            "ai_capability": world.ai.frontier_model_capability,
            "agi_proximity": world.ai.agi_proximity,
            "recent_events": [e.description for e in visible_events[-5:]],
            "other_agents": list(self.theory_of_mind.models.keys()),
            "stress": self.state.stress_level,
            "political_capital": self.state.political_capital,
        }

        # Get relationship context
        relationships = {}
        for agent_name in self.theory_of_mind.models.keys():
            rel = self.get_relationship_with(agent_name)
            relationships[agent_name] = rel.value
        context["relationships"] = relationships

        # Check if we have pending proposals to respond to
        if self.pending_proposals:
            return await self._respond_to_proposal(world, context)

        # Generate strategic options
        options = self.strategic_reasoner.generate_options(context)

        # Get predictions about other agents
        predictions = {}
        for other in self.theory_of_mind.models.keys():
            predictions[other] = self.theory_of_mind.predict_action(other, context)

        # If we have inference function, use LLM for final decision
        if self.inference_fn:
            return await self._llm_decide(world, context, options, predictions)

        # Otherwise use rule-based selection
        return self._rule_based_decide(world, context, options, predictions)

    async def _llm_decide(
        self,
        world: WorldState,
        context: Dict[str, Any],
        options: List,
        predictions: Dict[str, Dict],
    ) -> Optional[Action]:
        """Use LLM for decision with strategic context."""
        system_prompt = self.persona.build_system_prompt(self.state, world)

        # Add strategic reasoning context
        system_prompt += "\n\n## Strategic Analysis\n"

        for option in options[:3]:  # Top 3 options
            system_prompt += f"\n### Option: {option.description}\n"
            system_prompt += f"- Expected utility: {option.expected_utility:.2f}\n"
            system_prompt += f"- Risk: {option.risk:.2f}\n"
            if option.best_case:
                system_prompt += f"- Best case: {option.best_case}\n"
            if option.worst_case:
                system_prompt += f"- Worst case: {option.worst_case}\n"

        system_prompt += "\n## Predictions About Others\n"
        for agent, pred in predictions.items():
            system_prompt += f"- {agent}: likely to {pred.get('predicted_action', 'unknown')}\n"

        # Build user message
        user_message = """Based on the current situation and strategic analysis, decide what action to take.

Consider:
1. The strategic options presented
2. Predictions about other actors
3. Your core beliefs and red lines
4. Your current emotional state and political capital

Available action types:
- STATEMENT: Make a public statement
- PROPOSAL: Propose something to another party
- THREAT: Issue a threat (use carefully - affects credibility)
- CONCESSION: Make a concession
- ALLIANCE: Propose or strengthen alliance
- EXPORT_CONTROL: Modify export controls (if applicable)
- WAIT: Wait and observe
- ESCALATE: Deliberately escalate
- DEESCALATE: Deliberately de-escalate

Respond in JSON format:
{
    "action_type": "<type>",
    "target": "<target agent name or null>",
    "content": "<what you say or do - be specific and in character>",
    "reasoning": "<your internal reasoning - not visible to others>",
    "strategic_rationale": "<which strategic option this aligns with>",
    "escalation_intent": "<escalate|deescalate|maintain>"
}"""

        try:
            response = await self.inference_fn(
                system_prompt=system_prompt,
                user_message=user_message,
            )

            # Parse response
            response_text = response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            data = json.loads(response_text.strip())

            action_type = ActionType[data["action_type"].upper()]

            action = Action(
                action_id=f"action_{self.name}_{world.tick}_{int(time.time()*1000) % 10000}",
                agent_name=self.name,
                action_type=action_type,
                target=data.get("target"),
                content=data["content"],
                reasoning=data["reasoning"],
                tick=world.tick,
                metadata={
                    "strategic_rationale": data.get("strategic_rationale", ""),
                    "predictions_used": list(predictions.keys()),
                }
            )

            # Set escalation delta
            intent = data.get("escalation_intent", "maintain")
            if intent == "escalate":
                action.escalation_delta = 1
            elif intent == "deescalate":
                action.escalation_delta = -1

            return action

        except Exception as e:
            logger.error(f"LLM decision error for {self.name}: {e}")
            return self._rule_based_decide(world, context, options, predictions)

    def _rule_based_decide(
        self,
        world: WorldState,
        context: Dict[str, Any],
        options: List,
        predictions: Dict[str, Dict],
    ) -> Optional[Action]:
        """Rule-based decision as fallback."""
        # Select option with best expected utility adjusted for risk
        if not options:
            return Action(
                action_id=f"action_{self.name}_{world.tick}",
                agent_name=self.name,
                action_type=ActionType.WAIT,
                target=None,
                content="Observing the situation.",
                reasoning="No strategic options identified.",
                tick=world.tick,
            )

        # Adjust utility based on stress and political capital
        def adjusted_utility(opt) -> float:
            base = opt.expected_utility
            risk_adjustment = opt.risk * self.state.stress_level
            capital_factor = self.state.political_capital
            return base - risk_adjustment * capital_factor

        best_option = max(options, key=adjusted_utility)

        action_type = ActionType.STATEMENT
        if best_option.actions:
            action_str = best_option.actions[0].upper()
            try:
                action_type = ActionType[action_str]
            except KeyError:
                pass

        return Action(
            action_id=f"action_{self.name}_{world.tick}",
            agent_name=self.name,
            action_type=action_type,
            target=None,
            content=f"{self.name} pursues {best_option.description}",
            reasoning=f"Selected based on utility {best_option.expected_utility:.2f}",
            tick=world.tick,
        )

    async def _respond_to_proposal(
        self,
        world: WorldState,
        context: Dict[str, Any],
    ) -> Optional[Action]:
        """Respond to a pending proposal."""
        if not self.pending_proposals:
            return None

        proposal = self.pending_proposals.pop(0)
        proposer = proposal["from"]
        proposal_action = proposal["action"]

        # Get model of proposer
        model = self.theory_of_mind.get_model(proposer)

        # Evaluate proposal
        relationship = self.get_relationship_with(proposer)

        # Decision factors
        trust_factor = 1.0 - model.intentions.threat_level
        relationship_factor = {
            "ally": 0.8,
            "partner": 0.6,
            "neutral": 0.4,
            "rival": 0.2,
            "adversary": 0.1,
            "enemy": 0.0,
        }.get(relationship.value, 0.4)

        accept_threshold = (trust_factor + relationship_factor) / 2

        # Use LLM if available
        if self.inference_fn:
            decision, response_content = await self.respond_to_proposal(proposal_action, world)
        else:
            # Rule-based
            if accept_threshold > 0.5:
                decision = "accept"
                response_content = f"We accept the proposal from {proposer}."
            elif accept_threshold > 0.3:
                decision = "counter"
                response_content = f"We would like to discuss modifications to the proposal."
            else:
                decision = "reject"
                response_content = f"We cannot accept this proposal at this time."

        # Create response action
        action_type = {
            "accept": ActionType.CONCESSION,
            "reject": ActionType.STATEMENT,
            "counter": ActionType.PROPOSAL,
        }.get(decision, ActionType.STATEMENT)

        return Action(
            action_id=f"action_{self.name}_{world.tick}",
            agent_name=self.name,
            action_type=action_type,
            target=proposer,
            content=response_content,
            reasoning=f"Response to proposal: {decision} (threshold: {accept_threshold:.2f})",
            tick=world.tick,
            metadata={"response_to": proposal_action.action_id},
        )

    async def respond_to_proposal(
        self,
        proposal: Action,
        world: WorldState,
    ) -> Tuple[str, str]:
        """Use LLM to respond to a proposal."""
        if not self.inference_fn:
            return "reject", "Unable to process proposal."

        system_prompt = self.persona.build_system_prompt(self.state, world)

        # Add theory of mind context
        proposer_model = self.theory_of_mind.get_model(proposal.agent_name)
        system_prompt += f"\n\n## Analysis of {proposal.agent_name}\n"
        system_prompt += f"- Threat level: {proposer_model.intentions.threat_level:.2f}\n"
        system_prompt += f"- Cooperation likelihood: {proposer_model.intentions.cooperation_likelihood:.2f}\n"
        system_prompt += f"- Risk tolerance: {proposer_model.risk_tolerance:.2f}\n"
        system_prompt += f"- Aggression: {proposer_model.aggression:.2f}\n"

        user_message = f"""{proposal.agent_name} has made the following proposal to you:

"{proposal.content}"

Consider:
- Your analysis of {proposal.agent_name}'s intentions and trustworthiness
- Your core beliefs and red lines
- Whether accepting/rejecting affects your credibility
- The current strategic context

Respond in JSON format:
{{
    "decision": "<accept|reject|counter>",
    "response": "<your response - in character>",
    "reasoning": "<your internal reasoning>",
    "conditions": "<any conditions on acceptance, or null>"
}}"""

        try:
            response = await self.inference_fn(
                system_prompt=system_prompt,
                user_message=user_message,
            )

            response_text = response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            data = json.loads(response_text.strip())

            return data["decision"], data["response"]

        except Exception as e:
            logger.error(f"Proposal response error for {self.name}: {e}")
            return "reject", "Unable to process proposal at this time."


# =============================================================================
# Integrated Simulation Orchestrator
# =============================================================================


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    mode: SimulationMode = SimulationMode.TURN_BASED
    max_ticks: int = 100

    # Features to enable
    enable_theory_of_mind: bool = True
    enable_coalition_dynamics: bool = True
    enable_reputation_tracking: bool = True
    enable_belief_propagation: bool = True
    enable_strategic_reasoning: bool = True

    # Timing
    tick_duration: float = 1.0
    belief_propagation_interval: int = 5  # Propagate beliefs every N ticks

    # Logging
    log_reasoning: bool = False
    log_predictions: bool = False


class PolicySimulationOrchestrator:
    """
    Main orchestrator for policy wargaming simulations.

    Integrates:
    - SimulationEngine for execution
    - CoalitionManager for power dynamics
    - MultiAgentCoordinator for complex interactions
    - Belief propagation for information spread
    - Reputation tracking for credibility
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        inference_fn: Optional[Callable] = None,
    ):
        self.config = config or OrchestratorConfig()
        self.inference_fn = inference_fn

        # Core components
        self.world = WorldState()
        self.agents: Dict[str, IntegratedAgent] = {}

        # Multi-agent coordination
        self.coordinator = MultiAgentCoordinator()

        # Power and coalition dynamics
        self.coalition_manager: Optional[CoalitionManager] = None
        self.commitment_tracker = CommitmentTracker.__new__(CommitmentTracker)
        self.audience_manager = AudienceCostManager.__new__(AudienceCostManager)

        # Event and action history
        self.events: List[Event] = []
        self.action_history: List[Action] = []

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "on_action": [],
            "on_event": [],
            "on_escalation": [],
            "on_coalition_formed": [],
            "on_negotiation_complete": [],
        }

    def add_agent_from_persona(
        self,
        persona_path: Path,
    ) -> IntegratedAgent:
        """Load and add an agent from persona JSON."""
        persona = Persona.from_json(persona_path)

        agent = IntegratedAgent(
            persona=persona,
            coordinator=self.coordinator,
            inference_fn=self.inference_fn,
        )

        self.agents[agent.name] = agent

        # Add to belief propagation network
        for existing_name in self.agents:
            if existing_name != agent.name:
                self.coordinator.belief_propagation.add_connection(
                    agent.name, existing_name
                )

        # Initialize reputation
        self.coordinator.reputation_system.get_record(agent.name)

        return agent

    def add_agents_from_directory(
        self,
        persona_dir: Path,
        names: Optional[List[str]] = None,
    ) -> List[IntegratedAgent]:
        """Load multiple agents from directory."""
        added = []

        if names:
            for name in names:
                # Try different locations
                paths = [
                    persona_dir / f"{name}.json",
                    persona_dir / "enhanced" / f"{name}.json",
                    persona_dir / "finalized" / f"{name}.json",
                ]
                for path in paths:
                    if path.exists():
                        added.append(self.add_agent_from_persona(path))
                        break
        else:
            # Load all JSON files
            for path in persona_dir.glob("*.json"):
                try:
                    added.append(self.add_agent_from_persona(path))
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")

        return added

    def inject_event(self, event: Event):
        """Inject an event into the simulation."""
        self.events.append(event)

        # Notify all agents
        for agent in self.agents.values():
            agent.receive_event(event)

        # Apply effects to world state
        self._apply_event_effects(event)

        # Callbacks
        for callback in self._callbacks["on_event"]:
            callback(event)

    def inject_scenario(
        self,
        name: str,
        description: str,
        effects: Dict[str, Any],
    ):
        """Inject a scenario event."""
        event = Event(
            event_id=f"scenario_{name}_{self.world.tick}",
            event_type="scenario",
            source="scenario_injection",
            target=None,
            description=description,
            tick=self.world.tick,
            effects=effects,
        )
        self.inject_event(event)
        self.world.geopolitics.recent_events.append(f"[SCENARIO] {description}")

    def _apply_event_effects(self, event: Event):
        """Apply event effects to world state."""
        effects = event.effects

        if "escalation_delta" in effects:
            delta = effects["escalation_delta"]
            current = self.world.geopolitics.escalation_level.value
            new_level = max(0, min(6, current + delta))
            self.world.geopolitics.escalation_level = EscalationLevel(new_level)

            if delta > 0:
                for callback in self._callbacks["on_escalation"]:
                    callback(event, self.world.geopolitics.escalation_level)

        if "ai_capability_delta" in effects:
            self.world.ai.frontier_model_capability += effects["ai_capability_delta"]

        if "agi_proximity_delta" in effects:
            self.world.ai.agi_proximity = min(
                1.0,
                self.world.ai.agi_proximity + effects["agi_proximity_delta"]
            )

        for key, value in effects.items():
            if key.startswith("var:"):
                self.world.variables[key[4:]] = value

    async def step(self) -> List[Action]:
        """Execute one simulation step."""
        # Checkpoint
        self.world.checkpoint()

        actions = []

        # Get current visible events for each agent
        visible_events = [
            e for e in self.events[-20:]  # Last 20 events
            if e.visibility == "public"
        ]

        # Turn-based: one agent acts
        if self.config.mode == SimulationMode.TURN_BASED:
            agent_list = list(self.agents.values())
            current_agent = agent_list[self.world.tick % len(agent_list)]

            action = await current_agent.decide_action(self.world, visible_events)
            if action:
                actions.append(action)

        # Continuous: all agents may act
        elif self.config.mode == SimulationMode.CONTINUOUS:
            for agent in self.agents.values():
                # Probability-based action
                if hasattr(agent.state, 'stress_level'):
                    act_prob = 0.3 + agent.state.stress_level * 0.2
                else:
                    act_prob = 0.3

                import random
                if random.random() < act_prob:
                    action = await agent.decide_action(self.world, visible_events)
                    if action:
                        actions.append(action)

        # Process actions
        for action in actions:
            # Log
            if self.config.log_reasoning:
                logger.info(
                    f"[{action.tick}] {action.agent_name} ({action.action_type.value}): "
                    f"{action.content[:100]}... | Reasoning: {action.reasoning}"
                )

            # Broadcast to other agents
            for agent in self.agents.values():
                if agent.name != action.agent_name:
                    agent.observe_action(action.agent_name, action)

            # Convert to event
            event = action.to_event()
            self.events.append(event)

            # Track
            self.action_history.append(action)

            # Callbacks
            for callback in self._callbacks["on_action"]:
                callback(action)

        # Belief propagation
        if (
            self.config.enable_belief_propagation and
            self.world.tick % self.config.belief_propagation_interval == 0
        ):
            self.coordinator.belief_propagation.propagate_step()

        # Advance world
        self.world.advance_tick(self.config.tick_duration)

        return actions

    async def run(
        self,
        max_ticks: Optional[int] = None,
        stop_condition: Optional[Callable[[WorldState], bool]] = None,
    ) -> Dict[str, Any]:
        """Run simulation until completion or stop condition."""
        max_ticks = max_ticks or self.config.max_ticks

        for tick in range(max_ticks):
            if stop_condition and stop_condition(self.world):
                logger.info(f"Stop condition met at tick {tick}")
                break

            if self.world.geopolitics.escalation_level == EscalationLevel.TOTAL_WAR:
                logger.warning("Simulation reached TOTAL_WAR")
                break

            await self.step()

        return self.get_summary()

    async def run_negotiation(
        self,
        agent_a_name: str,
        agent_b_name: str,
        topic: str,
        initial_proposal: str,
        max_rounds: int = 10,
    ) -> Dict[str, Any]:
        """Run a bilateral negotiation."""
        agent_a = self.agents.get(agent_a_name)
        agent_b = self.agents.get(agent_b_name)

        if not agent_a or not agent_b:
            return {"error": "Agents not found"}

        # Start negotiation via coordinator
        conv_id = self.coordinator.conversation_mgr.start_negotiation(
            agent_a_name,
            agent_b_name,
            topic,
            {"proposal": initial_proposal},
        )

        results = {
            "conversation_id": conv_id,
            "topic": topic,
            "exchanges": [],
            "outcome": None,
        }

        current_proposer = agent_a
        current_responder = agent_b
        current_proposal = Action(
            action_id=f"neg_init",
            agent_name=agent_a_name,
            action_type=ActionType.PROPOSAL,
            target=agent_b_name,
            content=initial_proposal,
            reasoning="Initial proposal",
            tick=self.world.tick,
        )

        for round_num in range(max_rounds):
            # Responder evaluates and responds
            decision, response = await current_responder.respond_to_proposal(
                current_proposal, self.world
            )

            results["exchanges"].append({
                "round": round_num + 1,
                "from": current_responder.name,
                "decision": decision,
                "response": response,
            })

            if decision == "accept":
                results["outcome"] = "agreement"
                break
            elif decision == "reject":
                results["outcome"] = "breakdown"
                break
            elif decision == "counter":
                # Swap roles
                current_proposer, current_responder = current_responder, current_proposer
                current_proposal = Action(
                    action_id=f"neg_round_{round_num}",
                    agent_name=current_proposer.name,
                    action_type=ActionType.PROPOSAL,
                    target=current_responder.name,
                    content=response,
                    reasoning="Counter-proposal",
                    tick=self.world.tick,
                )

        if results["outcome"] is None:
            results["outcome"] = "timeout"

        return results

    async def run_multilateral_coordination(
        self,
        participants: List[str],
        topic: str,
        protocol: str = "voting",
    ) -> Dict[str, Any]:
        """Run multilateral coordination (voting, consensus building)."""
        valid_participants = [p for p in participants if p in self.agents]

        if protocol == "voting":
            ballot_id = self.coordinator.consensus_builder.create_ballot(
                topic=topic,
                options=["approve", "reject", "abstain"],
                voters=set(valid_participants),
                method=VotingMethod.MAJORITY,
            )

            # Each agent votes (simplified - would use LLM for real decision)
            for agent_name in valid_participants:
                agent = self.agents[agent_name]

                # Simple vote logic based on stress and cooperation tendency
                import random
                if agent.state.stress_level < 0.5:
                    vote = "approve" if random.random() < 0.6 else "reject"
                else:
                    vote = "reject" if random.random() < 0.6 else "approve"

                ballot = self.coordinator.consensus_builder.get_ballot(ballot_id)
                if ballot:
                    ballot.cast_vote(agent_name, vote)

            # Tally
            ballot = self.coordinator.consensus_builder.get_ballot(ballot_id)
            if ballot:
                result = ballot.close_and_decide()
                return {
                    "protocol": "voting",
                    "topic": topic,
                    "participants": valid_participants,
                    "tally": ballot.tally(),
                    "result": result,
                }

        return {"error": f"Unknown protocol: {protocol}"}

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive simulation summary."""
        summary = {
            "ticks": self.world.tick,
            "final_escalation": self.world.geopolitics.escalation_level.name,
            "ai_capability": self.world.ai.frontier_model_capability,
            "agi_proximity": self.world.ai.agi_proximity,
            "total_actions": len(self.action_history),

            "actions_by_agent": {
                name: len([a for a in self.action_history if a.agent_name == name])
                for name in self.agents
            },

            "actions_by_type": {},
            "escalation_timeline": [],

            # Advanced metrics
            "theory_of_mind": {},
            "reputation_scores": {},
        }

        # Action type counts
        for action in self.action_history:
            t = action.action_type.value
            summary["actions_by_type"][t] = summary["actions_by_type"].get(t, 0) + 1

        # Escalation timeline
        for event in self.events:
            if "escalation_delta" in event.effects and event.effects["escalation_delta"] != 0:
                summary["escalation_timeline"].append({
                    "tick": event.tick,
                    "source": event.source,
                    "delta": event.effects["escalation_delta"],
                })

        # Theory of mind summary
        if self.config.enable_theory_of_mind:
            for agent_name, agent in self.agents.items():
                summary["theory_of_mind"][agent_name] = {
                    other: {
                        "threat_level": model.intentions.threat_level,
                        "cooperation": model.intentions.cooperation_likelihood,
                        "aggression": model.aggression,
                        "prediction_accuracy": model.prediction_accuracy,
                    }
                    for other, model in agent.theory_of_mind.models.items()
                }

        # Reputation scores
        if self.config.enable_reputation_tracking:
            for agent_name in self.agents:
                record = self.coordinator.reputation_system.get_record(agent_name)
                summary["reputation_scores"][agent_name] = {
                    "overall": record.overall,
                    "trustworthiness": record.trustworthiness,
                    "competence": record.competence,
                    "predictability": record.predictability,
                }

        return summary

    def on(self, event_type: str, callback: Callable):
        """Register callback for events."""
        if event_type in self._callbacks:
            self._callbacks[event_type].append(callback)


# =============================================================================
# Convenience Functions
# =============================================================================


async def create_and_run_simulation(
    persona_dir: Path,
    agents: List[str],
    scenario: Optional[Tuple[str, str, Dict]] = None,
    mode: SimulationMode = SimulationMode.TURN_BASED,
    ticks: int = 20,
    inference_fn: Optional[Callable] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Create and run a simulation with minimal setup."""
    config = OrchestratorConfig(
        mode=mode,
        max_ticks=ticks,
        log_reasoning=verbose,
    )

    orchestrator = PolicySimulationOrchestrator(config, inference_fn)

    # Load agents
    orchestrator.add_agents_from_directory(persona_dir, agents)

    # Inject scenario if provided
    if scenario:
        name, desc, effects = scenario
        orchestrator.inject_scenario(name, desc, effects)

    # Add verbose logging
    if verbose:
        def log_action(action):
            print(f"[{action.tick}] {action.agent_name} ({action.action_type.value}):")
            print(f"  {action.content[:150]}")

        orchestrator.on("on_action", log_action)

    # Run
    return await orchestrator.run(max_ticks=ticks)
