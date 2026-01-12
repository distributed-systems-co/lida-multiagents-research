"""
Pre-built Scenario Templates

Ready-to-use scenarios for common multi-agent interaction patterns.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional

from .scenarios import (
    ScenarioDefinition,
    ScenarioBuilder,
    Phase,
    ScenarioAction,
    ScenarioEffect,
    Condition,
    ConditionGroup,
    ConditionOperator,
    Trigger,
    TriggerType,
)


# =============================================================================
# Helper Functions
# =============================================================================

def _action(action_type: str, **params) -> ScenarioAction:
    """Create a scenario action."""
    return ScenarioAction(action_type=action_type, params=params)


def _char_action(char_id: str, action_type: str, content: str, **kwargs) -> ScenarioAction:
    """Create a character action."""
    return ScenarioAction(
        action_type="character_action",
        character_id=char_id,
        params={"type": action_type, "content": content, **kwargs},
    )


def _condition(field: str, op: ConditionOperator, value: Any) -> Condition:
    """Create a condition."""
    return Condition(field=field, operator=op, value=value)


def _when_all(*conditions: Condition) -> ConditionGroup:
    """Create an AND condition group."""
    return ConditionGroup.all_of(*conditions)


def _when_any(*conditions: Condition) -> ConditionGroup:
    """Create an OR condition group."""
    return ConditionGroup.any_of(*conditions)


# =============================================================================
# DEBATE SCENARIOS
# =============================================================================

def create_oxford_debate(
    motion: str,
    rounds: int = 3,
) -> ScenarioDefinition:
    """Create an Oxford-style debate scenario.

    Roles required:
    - proposer: Proposes the motion
    - opposer: Opposes the motion
    - moderator: (optional) Moderates the debate

    Phases:
    1. Opening - Each side presents opening statements
    2. Rebuttal - Each side rebuts the other
    3. Cross-examination - Direct questioning
    4. Closing - Final statements
    5. Vote - Audience/judges vote
    """
    builder = ScenarioBuilder(f"Oxford Debate: {motion}")
    builder.with_description(f"Formal debate on: {motion}")
    builder.requires_role("proposer")
    builder.requires_role("opposer")
    builder.with_variable("motion", motion)
    builder.with_variable("rounds", rounds)
    builder.with_variable("proposer_points", 0)
    builder.with_variable("opposer_points", 0)

    # Opening phase
    builder.add_phase("opening", "Opening statements")
    builder.on_enter(_char_action("@moderator", "say", f"Welcome to this debate on: {motion}"))
    builder.with_action(_char_action("@proposer", "say", "I propose that ${motion} is correct because..."))
    builder.with_action(ScenarioAction(action_type="wait", params={"duration": 0.5}))
    builder.with_action(_char_action("@opposer", "say", "I oppose this motion because..."))
    builder.transition_to("rebuttal")

    # Rebuttal phase
    builder.add_phase("rebuttal", "Rebuttals")
    builder.with_action(ScenarioAction(
        action_type="loop",
        params={
            "count": rounds,
            "action": {
                "action_type": "sequence",
                "params": {
                    "actions": [
                        {"action_type": "character_action", "character_id": "@opposer",
                         "params": {"type": "say", "content": "In rebuttal to my opponent..."}},
                        {"action_type": "wait", "params": {"duration": 0.3}},
                        {"action_type": "character_action", "character_id": "@proposer",
                         "params": {"type": "say", "content": "My opponent fails to see that..."}},
                    ]
                }
            }
        }
    ))
    builder.transition_to("cross_examination")

    # Cross-examination phase
    builder.add_phase("cross_examination", "Cross-examination")
    builder.with_action(_char_action("@proposer", "ask", "How do you explain the evidence showing..."))
    builder.with_action(_char_action("@opposer", "reply", "That evidence is misleading because..."))
    builder.with_action(_char_action("@opposer", "ask", "Can you justify your position given..."))
    builder.with_action(_char_action("@proposer", "reply", "I justify it by noting that..."))
    builder.transition_to("closing")

    # Closing phase
    builder.add_phase("closing", "Closing statements")
    builder.with_action(_char_action("@proposer", "say", "In conclusion, I urge you to support the motion..."))
    builder.with_action(_char_action("@opposer", "say", "In conclusion, you must reject this motion..."))
    builder.transition_to("vote")

    # Vote phase
    builder.add_phase("vote", "Voting")
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": motion,
            "topic": "debate_motion",
            "characters": "all",
        }
    ))

    builder.max_duration(300)  # 5 minutes max

    return builder.build()


def create_panel_discussion(
    topic: str,
    questions: List[str],
) -> ScenarioDefinition:
    """Create a panel discussion scenario.

    Roles required:
    - moderator: Asks questions and manages discussion
    - panelist1, panelist2, panelist3: Panel members

    Each question becomes a phase with all panelists responding.
    """
    builder = ScenarioBuilder(f"Panel: {topic}")
    builder.with_description(f"Panel discussion on {topic}")
    builder.requires_role("moderator")
    builder.requires_role("panelist1")
    builder.requires_role("panelist2")
    builder.requires_role("panelist3")
    builder.with_variable("topic", topic)
    builder.with_variable("questions", questions)

    # Introduction
    builder.add_phase("introduction", "Panel introduction")
    builder.on_enter(_char_action("@moderator", "say", f"Welcome to our panel on {topic}"))
    builder.with_action(ScenarioAction(
        action_type="foreach",
        params={
            "characters": ["@panelist1", "@panelist2", "@panelist3"],
            "action": {
                "action_type": "character_action",
                "character_id": "$current_character",
                "params": {"type": "say", "content": "Thank you for having me. I'm looking forward to this discussion."}
            }
        }
    ))

    # Question phases
    for i, question in enumerate(questions):
        phase_name = f"question_{i+1}"
        builder.add_phase(phase_name, f"Question {i+1}")
        builder.with_variable(f"current_question", question)
        builder.on_enter(_char_action("@moderator", "ask", question))
        builder.with_action(ScenarioAction(
            action_type="dialogue_round",
            params={
                "characters": ["@panelist1", "@panelist2", "@panelist3"],
                "topic": f"question_{i+1}",
            }
        ))

        if i < len(questions) - 1:
            builder.transition_to(f"question_{i+2}")
        else:
            builder.transition_to("conclusion")

    # Conclusion
    builder.add_phase("conclusion", "Closing remarks")
    builder.with_action(_char_action("@moderator", "say", "Thank you all for this insightful discussion"))

    return builder.build()


# =============================================================================
# NEGOTIATION SCENARIOS
# =============================================================================

def create_bilateral_negotiation(
    issue: str,
    initial_positions: Dict[str, float] = None,
    max_rounds: int = 5,
) -> ScenarioDefinition:
    """Create a bilateral negotiation scenario.

    Roles required:
    - party_a: First negotiating party
    - party_b: Second negotiating party
    - mediator: (optional) Neutral mediator

    Phases:
    1. Position statement - Each party states initial position
    2. Negotiation rounds - Exchange of offers
    3. Final offer - Last chance for agreement
    4. Resolution - Accept or walk away
    """
    initial_positions = initial_positions or {"party_a": 0.8, "party_b": 0.2}

    builder = ScenarioBuilder(f"Negotiation: {issue}")
    builder.with_description(f"Bilateral negotiation on {issue}")
    builder.requires_role("party_a")
    builder.requires_role("party_b")
    builder.with_variable("issue", issue)
    builder.with_variable("round", 0)
    builder.with_variable("agreement_reached", False)
    builder.with_variable("party_a_position", initial_positions.get("party_a", 0.8))
    builder.with_variable("party_b_position", initial_positions.get("party_b", 0.2))

    builder.with_initial_beliefs("party_a", {issue: initial_positions.get("party_a", 0.8)})
    builder.with_initial_beliefs("party_b", {issue: initial_positions.get("party_b", 0.2)})

    # Position statement phase
    builder.add_phase("position_statement", "Initial positions")
    builder.with_action(_char_action("@party_a", "say", f"On {issue}, our position is {initial_positions.get('party_a', 0.8):.1f}"))
    builder.with_action(_char_action("@party_b", "say", f"Our position differs. We stand at {initial_positions.get('party_b', 0.2):.1f}"))
    builder.transition_to("negotiation")

    # Negotiation rounds
    builder.add_phase("negotiation", "Negotiation rounds")
    builder.with_action(ScenarioAction(
        action_type="loop",
        params={
            "count": max_rounds,
            "action": {
                "action_type": "negotiate",
                "params": {
                    "characters": ["@party_a", "@party_b"],
                    "topic": issue,
                    "rounds": 1,
                    "agreement_threshold": 0.15,
                }
            }
        },
        effects=[
            ScenarioEffect(effect_type="increment_variable", params={"name": "round", "amount": 1})
        ]
    ))
    builder.transition_to("final_offer", _when_any(
        _condition("variables.round", ConditionOperator.GE, max_rounds),
        _condition("variables.last_negotiation.agreement_reached", ConditionOperator.EQ, True),
    ))

    # Final offer
    builder.add_phase("final_offer", "Final offers")
    builder.with_action(_char_action("@party_a", "say", "This is our final offer on ${issue}"))
    builder.with_action(_char_action("@party_b", "say", "We accept/reject this final offer"))
    builder.transition_to("resolution")

    # Resolution
    builder.add_phase("resolution", "Resolution")
    builder.with_action(ScenarioAction(
        action_type="conditional",
        params={
            "condition": {"conditions": [{"field": "variables.last_negotiation.agreement_reached", "operator": "eq", "value": True}], "mode": "all"},
            "then": {"action_type": "character_action", "character_id": "@party_a", "params": {"type": "say", "content": "We have reached an agreement!"}},
            "else": {"action_type": "character_action", "character_id": "@party_a", "params": {"type": "say", "content": "Unfortunately, we could not reach an agreement."}}
        }
    ))

    builder.success_when(_when_all(
        _condition("variables.last_negotiation.agreement_reached", ConditionOperator.EQ, True)
    ))

    return builder.build()


def create_multi_party_negotiation(
    issue: str,
    num_parties: int = 4,
) -> ScenarioDefinition:
    """Create a multi-party negotiation with coalition formation.

    Dynamically creates roles for n parties.
    Includes coalition formation and bloc voting.
    """
    builder = ScenarioBuilder(f"Multi-Party Negotiation: {issue}")
    builder.with_description(f"{num_parties}-party negotiation on {issue}")

    for i in range(num_parties):
        builder.requires_role(f"party_{i}")
        builder.with_initial_beliefs(f"party_{i}", {issue: (i + 1) / (num_parties + 1)})

    builder.with_variable("issue", issue)
    builder.with_variable("coalitions_formed", 0)

    # Opening
    builder.add_phase("opening", "Opening statements")
    builder.with_action(ScenarioAction(
        action_type="foreach",
        params={
            "characters": [f"@party_{i}" for i in range(num_parties)],
            "action": {
                "action_type": "character_action",
                "character_id": "$current_character",
                "params": {"type": "say", "content": f"Our position on {issue} is..."}
            }
        }
    ))
    builder.transition_to("coalition_formation")

    # Coalition formation
    builder.add_phase("coalition_formation", "Coalition formation")
    builder.with_action(ScenarioAction(
        action_type="form_coalition",
        params={
            "characters": [f"@party_{i}" for i in range(num_parties)],
            "topic": issue,
            "similarity_threshold": 0.25,
            "coalition_name": "main_coalition",
        },
        effects=[ScenarioEffect(effect_type="increment_variable", params={"name": "coalitions_formed", "amount": 1})]
    ))
    builder.transition_to("bloc_negotiation")

    # Bloc negotiation
    builder.add_phase("bloc_negotiation", "Bloc negotiation")
    builder.with_action(ScenarioAction(
        action_type="negotiate",
        params={
            "characters": "coalition",
            "coalition": "main_coalition",
            "topic": issue,
            "rounds": 3,
        }
    ))
    builder.transition_to("final_vote")

    # Final vote
    builder.add_phase("final_vote", "Final vote")
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": f"Resolution on {issue}",
            "topic": issue,
            "characters": [f"@party_{i}" for i in range(num_parties)],
        }
    ))

    return builder.build()


# =============================================================================
# CRISIS SCENARIOS
# =============================================================================

def create_crisis_response(
    crisis_description: str,
    severity: float = 0.7,
    time_pressure: float = 30.0,
) -> ScenarioDefinition:
    """Create a crisis response scenario.

    Roles required:
    - leader: Decision maker
    - advisor1, advisor2: Expert advisors
    - stakeholder: Affected stakeholder

    Phases with time pressure:
    1. Crisis announcement
    2. Initial assessment
    3. Option generation
    4. Decision
    5. Implementation
    """
    builder = ScenarioBuilder(f"Crisis: {crisis_description[:50]}...")
    builder.with_description(crisis_description)
    builder.requires_role("leader")
    builder.requires_role("advisor1")
    builder.requires_role("advisor2")
    builder.requires_role("stakeholder")

    builder.with_variable("crisis", crisis_description)
    builder.with_variable("severity", severity)
    builder.with_variable("urgency", 1.0)
    builder.with_variable("options", [])
    builder.with_variable("decision_made", False)

    # Crisis announcement
    builder.add_phase("crisis_announcement", "Crisis announced")
    builder.with_timeout(time_pressure / 5)
    builder.with_action(ScenarioAction(
        action_type="broadcast",
        params={"message": f"URGENT: {crisis_description}", "topic": "crisis"}
    ))
    builder.with_action(ScenarioAction(
        action_type="foreach",
        params={
            "characters": ["@leader", "@advisor1", "@advisor2", "@stakeholder"],
            "action": {
                "action_type": "character_action",
                "character_id": "$current_character",
                "params": {"type": "react", "content": {"emotion": "concern", "event": "crisis"}}
            }
        }
    ))
    builder.transition_to("assessment")

    # Assessment phase
    builder.add_phase("assessment", "Situation assessment")
    builder.with_timeout(time_pressure / 5)
    builder.with_action(_char_action("@advisor1", "think", "analyzing the situation"))
    builder.with_action(_char_action("@advisor1", "say", "My assessment of the situation is..."))
    builder.with_action(_char_action("@advisor2", "say", "From my perspective, we need to consider..."))
    builder.with_action(_char_action("@stakeholder", "say", "This will impact us significantly because..."))
    builder.transition_to("option_generation")

    # Option generation
    builder.add_phase("option_generation", "Generating options")
    builder.with_timeout(time_pressure / 5)
    builder.with_action(_char_action("@leader", "say", "What are our options?"))
    builder.with_action(ScenarioAction(
        action_type="parallel",
        params={
            "actions": [
                {"action_type": "character_action", "character_id": "@advisor1",
                 "params": {"type": "say", "content": "Option A: We could..."}},
                {"action_type": "character_action", "character_id": "@advisor2",
                 "params": {"type": "say", "content": "Option B: Alternatively..."}},
            ]
        }
    ))
    builder.transition_to("decision")

    # Decision phase
    builder.add_phase("decision", "Making the decision")
    builder.with_timeout(time_pressure / 5)
    builder.with_action(_char_action("@leader", "think", "weighing the options"))
    builder.with_action(_char_action("@leader", "decide", "After considering all factors, I've decided..."))
    builder.with_action(ScenarioAction(
        action_type="character_action",
        character_id="@leader",
        params={"type": "say", "content": "The decision has been made."},
        effects=[ScenarioEffect(effect_type="set_variable", params={"name": "decision_made", "value": True})]
    ))
    builder.transition_to("implementation")

    # Implementation
    builder.add_phase("implementation", "Implementing decision")
    builder.with_action(_char_action("@leader", "delegate", "Execute the plan"))
    builder.with_action(ScenarioAction(
        action_type="foreach",
        params={
            "characters": ["@advisor1", "@advisor2"],
            "action": {
                "action_type": "character_action",
                "character_id": "$current_character",
                "params": {"type": "execute", "content": "Implementing assigned tasks..."}
            }
        }
    ))

    builder.max_duration(time_pressure)
    builder.success_when(_when_all(
        _condition("variables.decision_made", ConditionOperator.EQ, True)
    ))

    return builder.build()


def create_escalating_crisis(
    initial_crisis: str,
    escalation_events: List[str],
) -> ScenarioDefinition:
    """Create a crisis that escalates through multiple stages.

    Each escalation event triggers new challenges.
    """
    builder = ScenarioBuilder(f"Escalating Crisis: {initial_crisis[:30]}...")
    builder.with_description(initial_crisis)
    builder.requires_role("leader")
    builder.requires_role("team_lead")
    builder.requires_role("analyst")

    builder.with_variable("crisis_level", 1)
    builder.with_variable("escalation_count", 0)
    builder.with_variable("resolved", False)

    # Initial crisis
    builder.add_phase("initial", "Initial crisis")
    builder.with_action(ScenarioAction(
        action_type="broadcast",
        params={"message": initial_crisis, "topic": "crisis"}
    ))
    builder.with_action(ScenarioAction(
        action_type="dialogue_round",
        params={"characters": ["@leader", "@team_lead", "@analyst"], "topic": "initial_response"}
    ))
    builder.transition_to("escalation_1" if escalation_events else "resolution")

    # Escalation phases
    for i, event in enumerate(escalation_events):
        phase_name = f"escalation_{i+1}"
        builder.add_phase(phase_name, f"Escalation {i+1}")
        builder.on_enter(ScenarioAction(
            action_type="broadcast",
            params={"message": f"ESCALATION: {event}", "topic": "crisis"}
        ))
        builder.with_action(ScenarioAction(
            action_type="apply_pressure",
            params={
                "target": "@leader",
                "source": "@analyst",
                "topic": "crisis_handling",
                "intensity": 0.2 * (i + 1),
                "direction": -1,
            }
        ))
        builder.with_action(ScenarioAction(
            action_type="dialogue_round",
            params={"characters": ["@leader", "@team_lead"], "topic": f"escalation_{i+1}"}
        ))

        if i < len(escalation_events) - 1:
            builder.transition_to(f"escalation_{i+2}")
        else:
            builder.transition_to("resolution")

    # Resolution
    builder.add_phase("resolution", "Crisis resolution")
    builder.with_action(_char_action("@leader", "say", "We must take decisive action now"))
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Accept proposed resolution",
            "topic": "crisis_resolution",
            "characters": ["@leader", "@team_lead", "@analyst"],
        }
    ))

    return builder.build()


# =============================================================================
# COALITION & INFLUENCE SCENARIOS
# =============================================================================

def create_coalition_building(
    objective: str,
    factions: List[str],
) -> ScenarioDefinition:
    """Create a coalition building scenario.

    Multiple factions attempt to form coalitions around an objective.
    """
    builder = ScenarioBuilder(f"Coalition Building: {objective}")
    builder.with_description(f"Building coalitions for: {objective}")

    for faction in factions:
        builder.requires_role(faction)
        builder.with_initial_beliefs(faction, {objective: 0.3 + 0.4 * factions.index(faction) / len(factions)})

    builder.with_variable("objective", objective)
    builder.with_variable("active_coalitions", 0)

    # Opening positions
    builder.add_phase("opening_positions", "Stating positions")
    builder.with_action(ScenarioAction(
        action_type="foreach",
        params={
            "characters": [f"@{faction}" for faction in factions],
            "action": {
                "action_type": "character_action",
                "character_id": "$current_character",
                "params": {"type": "say", "content": f"Our position on {objective} is..."}
            }
        }
    ))
    builder.transition_to("bilateral_talks")

    # Bilateral talks
    builder.add_phase("bilateral_talks", "Bilateral discussions")
    for i, faction1 in enumerate(factions):
        for faction2 in factions[i+1:]:
            builder.with_action(ScenarioAction(
                action_type="negotiate",
                params={
                    "characters": [f"@{faction1}", f"@{faction2}"],
                    "topic": objective,
                    "rounds": 2,
                }
            ))
    builder.transition_to("coalition_formation")

    # Coalition formation
    builder.add_phase("coalition_formation", "Forming coalitions")
    builder.with_action(ScenarioAction(
        action_type="form_coalition",
        params={
            "characters": [f"@{faction}" for faction in factions],
            "topic": objective,
            "similarity_threshold": 0.3,
            "coalition_name": "primary_coalition",
        }
    ))
    builder.transition_to("consolidation")

    # Consolidation
    builder.add_phase("consolidation", "Coalition consolidation")
    builder.with_action(ScenarioAction(
        action_type="dialogue_round",
        params={"characters": "coalition", "coalition": "primary_coalition", "topic": "coordination"}
    ))
    builder.transition_to("final_vote")

    # Final vote
    builder.add_phase("final_vote", "Coalition vote")
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": f"Coalition support for {objective}",
            "topic": objective,
            "characters": [f"@{faction}" for faction in factions],
        }
    ))

    return builder.build()


def create_influence_campaign(
    target_role: str,
    influencer_roles: List[str],
    topic: str,
    target_belief_shift: float = 0.3,
) -> ScenarioDefinition:
    """Create an influence campaign scenario.

    Multiple influencers attempt to shift a target's beliefs.
    """
    builder = ScenarioBuilder(f"Influence Campaign: {topic}")
    builder.with_description(f"Campaign to influence {target_role} on {topic}")

    builder.requires_role(target_role)
    for role in influencer_roles:
        builder.requires_role(role)

    builder.with_variable("topic", topic)
    builder.with_variable("target_belief_shift", target_belief_shift)
    builder.with_variable("initial_belief", 0.5)
    builder.with_variable("influence_attempts", 0)

    builder.with_initial_beliefs(target_role, {topic: 0.5})

    # Assessment
    builder.add_phase("assessment", "Assessing target")
    builder.with_action(ScenarioAction(
        action_type="foreach",
        params={
            "characters": [f"@{role}" for role in influencer_roles],
            "action": {
                "action_type": "character_action",
                "character_id": "$current_character",
                "params": {"type": "observe", "content": f"@{target_role}"}
            }
        }
    ))
    builder.transition_to("soft_influence")

    # Soft influence
    builder.add_phase("soft_influence", "Soft influence attempts")
    for role in influencer_roles:
        builder.with_action(ScenarioAction(
            action_type="reveal_information",
            params={
                "revealer": f"@{role}",
                "characters": [f"@{target_role}"],
                "information": {topic: 0.7},
                "impact": 0.1,
            }
        ))
    builder.transition_to("direct_persuasion")

    # Direct persuasion
    builder.add_phase("direct_persuasion", "Direct persuasion")
    for role in influencer_roles:
        builder.with_action(ScenarioAction(
            action_type="apply_pressure",
            params={
                "target": f"@{target_role}",
                "source": f"@{role}",
                "topic": topic,
                "intensity": 0.2,
                "direction": 1,
            }
        ))
        builder.with_action(_char_action(f"@{role}", "tell",
            f"You should reconsider your position on {topic}", target=f"@{target_role}"))
    builder.transition_to("evaluation")

    # Evaluation
    builder.add_phase("evaluation", "Campaign evaluation")
    builder.with_action(_char_action(f"@{target_role}", "say", f"After all this discussion on {topic}, I believe..."))

    builder.success_when(_when_all(
        _condition(f"characters.{target_role}.beliefs.{topic}", ConditionOperator.GE, 0.5 + target_belief_shift)
    ))

    return builder.build()


# =============================================================================
# DELIBERATION SCENARIOS
# =============================================================================

def create_structured_deliberation(
    topic: str,
    deliberators: int = 5,
    rounds: int = 3,
) -> ScenarioDefinition:
    """Create a structured deliberation scenario.

    Participants go through structured rounds of deliberation
    with belief tracking and convergence measurement.
    """
    builder = ScenarioBuilder(f"Deliberation: {topic}")
    builder.with_description(f"Structured deliberation on {topic}")

    for i in range(deliberators):
        builder.requires_role(f"deliberator_{i}")
        builder.with_initial_beliefs(f"deliberator_{i}", {topic: 0.2 + 0.6 * i / (deliberators - 1)})

    builder.with_variable("topic", topic)
    builder.with_variable("round", 0)
    builder.with_variable("belief_variance", 1.0)

    # Initial positions
    builder.add_phase("initial_positions", "Initial belief elicitation")
    builder.with_action(ScenarioAction(
        action_type="foreach",
        params={
            "characters": [f"@deliberator_{i}" for i in range(deliberators)],
            "action": {
                "action_type": "character_action",
                "character_id": "$current_character",
                "params": {"type": "say", "content": f"My initial position on {topic} is..."}
            }
        }
    ))
    builder.transition_to("round_1")

    # Deliberation rounds
    for r in range(rounds):
        phase_name = f"round_{r+1}"
        builder.add_phase(phase_name, f"Deliberation round {r+1}")

        # Each deliberator speaks and listens
        builder.with_action(ScenarioAction(
            action_type="dialogue_round",
            params={
                "characters": [f"@deliberator_{i}" for i in range(deliberators)],
                "topic": f"{topic}_round_{r+1}",
            }
        ))

        # Mutual influence - everyone affects everyone slightly
        for i in range(deliberators):
            for j in range(deliberators):
                if i != j:
                    builder.with_action(ScenarioAction(
                        action_type="apply_pressure",
                        params={
                            "target": f"@deliberator_{i}",
                            "source": f"@deliberator_{j}",
                            "topic": topic,
                            "intensity": 0.05,
                            "direction": 1,
                        }
                    ))

        if r < rounds - 1:
            builder.transition_to(f"round_{r+2}")
        else:
            builder.transition_to("final_positions")

    # Final positions
    builder.add_phase("final_positions", "Final positions")
    builder.with_action(ScenarioAction(
        action_type="foreach",
        params={
            "characters": [f"@deliberator_{i}" for i in range(deliberators)],
            "action": {
                "action_type": "character_action",
                "character_id": "$current_character",
                "params": {"type": "say", "content": f"After deliberation, my position is..."}
            }
        }
    ))
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": f"Consensus position on {topic}",
            "topic": topic,
            "characters": [f"@deliberator_{i}" for i in range(deliberators)],
            "threshold": 0.5,
        }
    ))

    return builder.build()


def create_adversarial_collaboration(
    research_question: str,
) -> ScenarioDefinition:
    """Create an adversarial collaboration scenario.

    Two opposing researchers attempt to design a study
    that both would accept as definitive.
    """
    builder = ScenarioBuilder(f"Adversarial Collab: {research_question[:40]}...")
    builder.with_description(f"Adversarial collaboration on: {research_question}")

    builder.requires_role("researcher_pro")
    builder.requires_role("researcher_con")
    builder.requires_role("arbiter")

    builder.with_variable("question", research_question)
    builder.with_variable("design_agreed", False)
    builder.with_variable("predictions_logged", False)

    builder.with_initial_beliefs("researcher_pro", {"hypothesis": 0.85})
    builder.with_initial_beliefs("researcher_con", {"hypothesis": 0.15})

    # State predictions
    builder.add_phase("predictions", "Stating predictions")
    builder.with_action(_char_action("@researcher_pro", "say", "I predict the study will show support for the hypothesis"))
    builder.with_action(_char_action("@researcher_con", "say", "I predict the study will show the hypothesis is false"))
    builder.with_action(ScenarioAction(
        action_type="character_action",
        character_id="@arbiter",
        params={"type": "say", "content": "Predictions have been logged."},
        effects=[ScenarioEffect(effect_type="set_variable", params={"name": "predictions_logged", "value": True})]
    ))
    builder.transition_to("design_negotiation")

    # Design negotiation
    builder.add_phase("design_negotiation", "Negotiating study design")
    builder.with_action(ScenarioAction(
        action_type="negotiate",
        params={
            "characters": ["@researcher_pro", "@researcher_con"],
            "topic": "study_design",
            "rounds": 4,
            "agreement_threshold": 0.2,
        }
    ))
    builder.with_action(_char_action("@arbiter", "say", "Let me help find common ground"))
    builder.transition_to("design_finalization")

    # Design finalization
    builder.add_phase("design_finalization", "Finalizing design")
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Accept proposed study design",
            "topic": "study_design",
            "characters": ["@researcher_pro", "@researcher_con"],
        }
    ))
    builder.transition_to("commitment")

    # Commitment
    builder.add_phase("commitment", "Pre-commitment")
    builder.with_action(_char_action("@researcher_pro", "say", "I commit to accepting results of this design"))
    builder.with_action(_char_action("@researcher_con", "say", "I also commit to accepting the results"))
    builder.with_action(_char_action("@arbiter", "say", "Both parties have committed. The design is finalized."))

    builder.success_when(_when_all(
        _condition("variables.last_vote.passed", ConditionOperator.EQ, True)
    ))

    return builder.build()


# =============================================================================
# SCENARIO REGISTRY
# =============================================================================

SCENARIO_TEMPLATES = {
    # Debates
    "oxford_debate": create_oxford_debate,
    "panel_discussion": create_panel_discussion,

    # Negotiations
    "bilateral_negotiation": create_bilateral_negotiation,
    "multi_party_negotiation": create_multi_party_negotiation,

    # Crises
    "crisis_response": create_crisis_response,
    "escalating_crisis": create_escalating_crisis,

    # Coalition & Influence
    "coalition_building": create_coalition_building,
    "influence_campaign": create_influence_campaign,

    # Deliberation
    "structured_deliberation": create_structured_deliberation,
    "adversarial_collaboration": create_adversarial_collaboration,
}


def list_scenario_templates() -> List[str]:
    """List available scenario templates."""
    return list(SCENARIO_TEMPLATES.keys())


def create_scenario(template_name: str, **kwargs) -> ScenarioDefinition:
    """Create a scenario from a template."""
    if template_name not in SCENARIO_TEMPLATES:
        raise ValueError(f"Unknown template: {template_name}. Available: {list_scenario_templates()}")
    return SCENARIO_TEMPLATES[template_name](**kwargs)
