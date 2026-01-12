"""
Advanced Scenario Templates

Sophisticated multi-agent interaction patterns:
- Game theory scenarios
- Information dynamics
- Power struggles
- Strategic simulations
- Social dynamics
"""

from __future__ import annotations

import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

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
    return ScenarioAction(action_type=action_type, params=params)


def _char_action(char_id: str, action_type: str, content: str, **kwargs) -> ScenarioAction:
    return ScenarioAction(
        action_type="character_action",
        character_id=char_id,
        params={"type": action_type, "content": content, **kwargs},
    )


def _condition(field: str, op: ConditionOperator, value: Any) -> Condition:
    return Condition(field=field, operator=op, value=value)


def _when_all(*conditions: Condition) -> ConditionGroup:
    return ConditionGroup.all_of(*conditions)


def _when_any(*conditions: Condition) -> ConditionGroup:
    return ConditionGroup.any_of(*conditions)


# =============================================================================
# GAME THEORY SCENARIOS
# =============================================================================

def create_prisoners_dilemma(
    stakes: str = "market_share",
    rounds: int = 5,
    communication_allowed: bool = True,
) -> ScenarioDefinition:
    """Create an iterated Prisoner's Dilemma scenario.

    Two parties must choose to cooperate or defect repeatedly.
    Payoff matrix is encoded in belief updates.

    Roles: player_a, player_b
    """
    builder = ScenarioBuilder(f"Prisoner's Dilemma: {stakes}")
    builder.with_description(f"Iterated PD over {stakes} for {rounds} rounds")
    builder.requires_role("player_a")
    builder.requires_role("player_b")

    builder.with_variable("stakes", stakes)
    builder.with_variable("rounds", rounds)
    builder.with_variable("current_round", 0)
    builder.with_variable("player_a_score", 0)
    builder.with_variable("player_b_score", 0)
    builder.with_variable("player_a_history", [])
    builder.with_variable("player_b_history", [])
    builder.with_variable("mutual_cooperations", 0)
    builder.with_variable("mutual_defections", 0)

    # Initial beliefs about cooperation
    builder.with_initial_beliefs("player_a", {"cooperation_tendency": 0.5, "trust_opponent": 0.5})
    builder.with_initial_beliefs("player_b", {"cooperation_tendency": 0.5, "trust_opponent": 0.5})

    # Setup phase
    builder.add_phase("setup", "Game setup")
    builder.with_action(_char_action("@player_a", "say", f"The stakes are {stakes}. I must decide whether to cooperate or defect."))
    builder.with_action(_char_action("@player_b", "say", "I understand the game. Let's see who can build trust."))
    builder.transition_to("round_1")

    # Create rounds
    for r in range(1, rounds + 1):
        phase_name = f"round_{r}"
        builder.add_phase(phase_name, f"Round {r}")

        if communication_allowed:
            builder.with_action(_char_action("@player_a", "say", f"Round {r}: I'm considering my options..."))
            builder.with_action(_char_action("@player_b", "say", f"Round {r}: Based on history, I think..."))

        # Simultaneous decision (modeled as sequential with hidden info)
        builder.with_action(ScenarioAction(
            action_type="character_action",
            character_id="@player_a",
            params={"type": "decide", "content": f"Round {r} decision: cooperate or defect"},
        ))
        builder.with_action(ScenarioAction(
            action_type="character_action",
            character_id="@player_b",
            params={"type": "decide", "content": f"Round {r} decision: cooperate or defect"},
        ))

        # Update trust based on outcomes (simulated via pressure)
        builder.with_action(ScenarioAction(
            action_type="apply_pressure",
            params={
                "target": "@player_a",
                "source": "@player_b",
                "topic": "trust_opponent",
                "intensity": 0.1,
                "direction": 1 if r % 2 == 0 else -1,  # Alternating for variety
            }
        ))

        if r < rounds:
            builder.transition_to(f"round_{r + 1}")
        else:
            builder.transition_to("resolution")

    # Resolution
    builder.add_phase("resolution", "Game resolution")
    builder.with_action(_char_action("@player_a", "say", "The game is complete. Let's see the results."))
    builder.with_action(_char_action("@player_b", "say", "I hope we both learned something about cooperation."))

    return builder.build()


def create_ultimatum_game(
    resource: str = "research_funding",
    total_amount: float = 100.0,
) -> ScenarioDefinition:
    """Create an Ultimatum Game scenario.

    Proposer offers a split, Responder accepts or rejects.
    If rejected, neither gets anything.

    Roles: proposer, responder, observer (optional)
    """
    builder = ScenarioBuilder(f"Ultimatum Game: {resource}")
    builder.with_description(f"Divide {total_amount} units of {resource}")
    builder.requires_role("proposer")
    builder.requires_role("responder")

    builder.with_variable("resource", resource)
    builder.with_variable("total_amount", total_amount)
    builder.with_variable("proposed_split", None)
    builder.with_variable("accepted", None)
    builder.with_variable("final_proposer_share", 0)
    builder.with_variable("final_responder_share", 0)

    builder.with_initial_beliefs("proposer", {"fairness": 0.5, "greed": 0.5})
    builder.with_initial_beliefs("responder", {"minimum_acceptable": 0.3, "spite_threshold": 0.1})

    # Deliberation
    builder.add_phase("deliberation", "Proposer deliberates")
    builder.with_action(_char_action("@proposer", "think", f"How should I split {total_amount} {resource}?"))
    builder.with_action(_char_action("@proposer", "say", "I need to consider what offer will be accepted..."))
    builder.transition_to("proposal")

    # Proposal
    builder.add_phase("proposal", "Making the offer")
    builder.with_action(_char_action("@proposer", "propose", f"I propose to keep X and give you Y of the {resource}"))
    builder.with_action(ScenarioAction(
        action_type="reveal_information",
        params={
            "revealer": "@proposer",
            "characters": ["@responder"],
            "information": {"proposed_split": 0.6},  # Proposer keeps 60%
            "impact": 0.5,
        }
    ))
    builder.transition_to("response")

    # Response
    builder.add_phase("response", "Responder decides")
    builder.with_action(_char_action("@responder", "think", "Is this offer fair enough?"))
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Accept the proposed split",
            "topic": "minimum_acceptable",
            "characters": ["@responder"],
            "threshold": 0.3,
        }
    ))
    builder.transition_to("outcome")

    # Outcome
    builder.add_phase("outcome", "Final outcome")
    builder.with_action(ScenarioAction(
        action_type="conditional",
        params={
            "condition": {"conditions": [{"field": "variables.last_vote.passed", "operator": "eq", "value": True}], "mode": "all"},
            "then": {"action_type": "character_action", "character_id": "@proposer", "params": {"type": "say", "content": "The deal is done."}},
            "else": {"action_type": "character_action", "character_id": "@responder", "params": {"type": "say", "content": "I reject this unfair offer. We both get nothing."}}
        }
    ))

    return builder.build()


def create_public_goods_game(
    num_players: int = 4,
    rounds: int = 5,
    multiplier: float = 2.0,
) -> ScenarioDefinition:
    """Create a Public Goods Game scenario.

    Players decide how much to contribute to a common pool.
    Pool is multiplied and divided equally.

    Tests free-rider problem and cooperation emergence.
    """
    builder = ScenarioBuilder(f"Public Goods Game ({num_players} players)")
    builder.with_description(f"{rounds} rounds with {multiplier}x multiplier")

    for i in range(num_players):
        builder.requires_role(f"player_{i}")
        builder.with_initial_beliefs(f"player_{i}", {
            "cooperation": 0.3 + 0.4 * random.random(),
            "free_rider_tendency": random.random() * 0.5,
        })

    builder.with_variable("multiplier", multiplier)
    builder.with_variable("rounds", rounds)
    builder.with_variable("total_contributions", [])
    builder.with_variable("payoffs", {f"player_{i}": 0 for i in range(num_players)})

    # Introduction
    builder.add_phase("introduction", "Game introduction")
    builder.with_action(ScenarioAction(
        action_type="broadcast",
        params={"message": f"Each round, decide how much to contribute. Pool is multiplied by {multiplier}x.", "topic": "public_goods"}
    ))
    builder.transition_to("round_1")

    # Rounds
    for r in range(1, rounds + 1):
        phase_name = f"round_{r}"
        builder.add_phase(phase_name, f"Round {r}")

        builder.with_action(ScenarioAction(
            action_type="foreach",
            params={
                "characters": [f"@player_{i}" for i in range(num_players)],
                "action": {
                    "action_type": "character_action",
                    "character_id": "$current_character",
                    "params": {"type": "decide", "content": f"Round {r}: How much will I contribute?"}
                }
            }
        ))

        # Mutual influence
        for i in range(num_players):
            for j in range(num_players):
                if i != j:
                    builder.with_action(ScenarioAction(
                        action_type="apply_pressure",
                        params={
                            "target": f"@player_{i}",
                            "source": f"@player_{j}",
                            "topic": "cooperation",
                            "intensity": 0.05,
                            "direction": 1,
                        }
                    ))

        if r < rounds:
            builder.transition_to(f"round_{r + 1}")
        else:
            builder.transition_to("final_tally")

    # Final tally
    builder.add_phase("final_tally", "Final results")
    builder.with_action(ScenarioAction(
        action_type="dialogue_round",
        params={"characters": [f"@player_{i}" for i in range(num_players)], "topic": "reflection"}
    ))

    return builder.build()


def create_stag_hunt(
    hunters: int = 4,
) -> ScenarioDefinition:
    """Create a Stag Hunt coordination game.

    All must cooperate to catch the stag (big reward).
    Anyone can defect to catch a rabbit (small guaranteed reward).
    If anyone defects, stag hunters get nothing.
    """
    builder = ScenarioBuilder(f"Stag Hunt ({hunters} hunters)")
    builder.with_description("Coordination game: cooperate for big reward or defect for small safe reward")

    for i in range(hunters):
        builder.requires_role(f"hunter_{i}")
        builder.with_initial_beliefs(f"hunter_{i}", {
            "trust_others": 0.3 + 0.4 * random.random(),
            "risk_tolerance": random.random(),
        })

    builder.with_variable("hunters", hunters)
    builder.with_variable("stag_value", 10)
    builder.with_variable("rabbit_value", 2)
    builder.with_variable("decisions", {})
    builder.with_variable("stag_caught", False)

    # Planning
    builder.add_phase("planning", "Hunt planning")
    builder.with_action(ScenarioAction(
        action_type="broadcast",
        params={"message": "We can catch the stag together (10 each) or hunt rabbits alone (2 each). But if anyone hunts rabbits, the stag escapes.", "topic": "hunt"}
    ))
    builder.with_action(ScenarioAction(
        action_type="dialogue_round",
        params={"characters": [f"@hunter_{i}" for i in range(hunters)], "topic": "coordination"}
    ))
    builder.transition_to("commitment")

    # Commitment phase
    builder.add_phase("commitment", "Making commitments")
    builder.with_action(ScenarioAction(
        action_type="foreach",
        params={
            "characters": [f"@hunter_{i}" for i in range(hunters)],
            "action": {
                "action_type": "character_action",
                "character_id": "$current_character",
                "params": {"type": "say", "content": "I commit to hunting the stag... or do I?"}
            }
        }
    ))
    builder.transition_to("hunt")

    # Hunt
    builder.add_phase("hunt", "The hunt")
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Hunt the stag together",
            "topic": "trust_others",
            "characters": [f"@hunter_{i}" for i in range(hunters)],
            "threshold": 0.5,
        }
    ))
    builder.transition_to("outcome")

    # Outcome
    builder.add_phase("outcome", "Hunt outcome")
    builder.with_action(ScenarioAction(
        action_type="conditional",
        params={
            "condition": {"conditions": [{"field": "variables.last_vote.approve", "operator": "eq", "value": hunters}], "mode": "all"},
            "then": {"action_type": "broadcast", "params": {"message": "The stag is caught! Everyone gets 10.", "topic": "hunt"}},
            "else": {"action_type": "broadcast", "params": {"message": "Someone defected. Stag hunters get nothing, rabbit hunters get 2.", "topic": "hunt"}}
        }
    ))

    return builder.build()


# =============================================================================
# INFORMATION DYNAMICS
# =============================================================================

def create_information_cascade(
    num_agents: int = 8,
    true_state: bool = True,
    signal_accuracy: float = 0.7,
) -> ScenarioDefinition:
    """Create an Information Cascade scenario.

    Agents receive private signals and make public decisions sequentially.
    Later agents may ignore their signals and follow the crowd.
    """
    builder = ScenarioBuilder(f"Information Cascade ({num_agents} agents)")
    builder.with_description("Sequential decision making with private information")

    for i in range(num_agents):
        builder.requires_role(f"agent_{i}")
        # Each agent has a private signal (encoded as belief)
        signal = true_state if random.random() < signal_accuracy else not true_state
        builder.with_initial_beliefs(f"agent_{i}", {
            "private_signal": 0.8 if signal else 0.2,
            "public_belief": 0.5,
        })

    builder.with_variable("true_state", true_state)
    builder.with_variable("public_decisions", [])
    builder.with_variable("cascade_formed", False)

    # Introduction
    builder.add_phase("introduction", "Scenario setup")
    builder.with_action(ScenarioAction(
        action_type="broadcast",
        params={"message": "Each agent will observe previous decisions and make their own choice.", "topic": "cascade"}
    ))
    builder.transition_to("agent_0_decision")

    # Sequential decisions
    for i in range(num_agents):
        phase_name = f"agent_{i}_decision"
        builder.add_phase(phase_name, f"Agent {i} decides")

        # Agent observes history
        if i > 0:
            builder.with_action(_char_action(f"@agent_{i}", "observe", f"Previous {i} decisions"))

        # Agent deliberates
        builder.with_action(_char_action(f"@agent_{i}", "think", "Should I follow my signal or the crowd?"))

        # Agent decides
        builder.with_action(ScenarioAction(
            action_type="character_action",
            character_id=f"@agent_{i}",
            params={"type": "decide", "content": "I choose option A or B"},
        ))

        # Influence next agents
        if i < num_agents - 1:
            for j in range(i + 1, num_agents):
                builder.with_action(ScenarioAction(
                    action_type="apply_pressure",
                    params={
                        "target": f"@agent_{j}",
                        "source": f"@agent_{i}",
                        "topic": "public_belief",
                        "intensity": 0.15,
                        "direction": 1,
                    }
                ))

        if i < num_agents - 1:
            builder.transition_to(f"agent_{i + 1}_decision")
        else:
            builder.transition_to("revelation")

    # Truth revealed
    builder.add_phase("revelation", "Truth revealed")
    builder.with_action(ScenarioAction(
        action_type="broadcast",
        params={"message": f"The true state was: {'A' if true_state else 'B'}", "topic": "cascade"}
    ))

    return builder.build()


def create_whistleblower_scenario(
    num_insiders: int = 3,
    secret_severity: float = 0.8,
) -> ScenarioDefinition:
    """Create a Whistleblower scenario.

    Insiders know a secret. One might leak it.
    Tests loyalty, ethics, and information control.

    Roles: executive, insider_0..n, journalist, regulator
    """
    builder = ScenarioBuilder("Whistleblower Dilemma")
    builder.with_description(f"Secret with severity {secret_severity}. Will someone talk?")

    builder.requires_role("executive")
    builder.requires_role("journalist")
    builder.requires_role("regulator")
    for i in range(num_insiders):
        builder.requires_role(f"insider_{i}")
        builder.with_initial_beliefs(f"insider_{i}", {
            "loyalty": 0.5 + 0.3 * random.random(),
            "ethics": 0.3 + 0.5 * random.random(),
            "fear": 0.3 + 0.4 * random.random(),
            "knowledge_of_wrongdoing": secret_severity,
        })

    builder.with_variable("secret_severity", secret_severity)
    builder.with_variable("leak_occurred", False)
    builder.with_variable("leaker_identity", None)

    # Setup
    builder.add_phase("setup", "Situation setup")
    builder.with_action(_char_action("@executive", "say", "Remember, what happens in this company stays in this company."))
    builder.with_action(ScenarioAction(
        action_type="foreach",
        params={
            "characters": [f"@insider_{i}" for i in range(num_insiders)],
            "action": {
                "action_type": "character_action",
                "character_id": "$current_character",
                "params": {"type": "think", "content": "I know about the wrongdoing. What should I do?"}
            }
        }
    ))
    builder.transition_to("pressure")

    # Pressure builds
    builder.add_phase("pressure", "Mounting pressure")
    builder.with_action(_char_action("@journalist", "say", "I've heard rumors. Anyone want to talk?"))
    builder.with_action(_char_action("@regulator", "say", "We're investigating concerns in the industry."))

    # Ethics pressure on insiders
    for i in range(num_insiders):
        builder.with_action(ScenarioAction(
            action_type="apply_pressure",
            params={
                "target": f"@insider_{i}",
                "source": "@journalist",
                "topic": "ethics",
                "intensity": 0.2,
                "direction": 1,
            }
        ))
    builder.transition_to("decision_point")

    # Decision point
    builder.add_phase("decision_point", "Critical decision")
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Remain silent about wrongdoing",
            "topic": "loyalty",
            "characters": [f"@insider_{i}" for i in range(num_insiders)],
            "threshold": 0.6,  # High threshold = likely to break silence
        }
    ))
    builder.transition_to("aftermath")

    # Aftermath
    builder.add_phase("aftermath", "Consequences")
    builder.with_action(ScenarioAction(
        action_type="conditional",
        params={
            "condition": {"conditions": [{"field": "variables.last_vote.passed", "operator": "eq", "value": False}], "mode": "all"},
            "then": {"action_type": "broadcast", "params": {"message": "Someone has broken ranks. The story is out.", "topic": "whistleblower"}},
            "else": {"action_type": "broadcast", "params": {"message": "Silence holds. For now.", "topic": "whistleblower"}}
        }
    ))

    return builder.build()


def create_rumor_propagation(
    num_agents: int = 6,
    initial_believers: int = 2,
) -> ScenarioDefinition:
    """Create a Rumor Propagation scenario.

    A rumor spreads through a network.
    Tests how misinformation spreads and can be countered.
    """
    builder = ScenarioBuilder(f"Rumor Mill ({num_agents} agents)")
    builder.with_description("Watch how information (and misinformation) spreads")

    for i in range(num_agents):
        builder.requires_role(f"agent_{i}")
        # First few are initial believers
        initial_belief = 0.9 if i < initial_believers else 0.3
        builder.with_initial_beliefs(f"agent_{i}", {
            "rumor_belief": initial_belief,
            "skepticism": 0.3 + 0.4 * random.random(),
        })

    builder.with_variable("rumor", "A major announcement is coming")
    builder.with_variable("rounds", 3)
    builder.with_variable("belief_trajectory", [])

    # Rumor starts
    builder.add_phase("rumor_starts", "Rumor originates")
    builder.with_action(ScenarioAction(
        action_type="foreach",
        params={
            "characters": [f"@agent_{i}" for i in range(initial_believers)],
            "action": {
                "action_type": "character_action",
                "character_id": "$current_character",
                "params": {"type": "say", "content": "Did you hear? A major announcement is coming!"}
            }
        }
    ))
    builder.transition_to("spread_1")

    # Propagation rounds
    for r in range(1, 4):
        phase_name = f"spread_{r}"
        builder.add_phase(phase_name, f"Spread round {r}")

        # Everyone talks and influences
        builder.with_action(ScenarioAction(
            action_type="dialogue_round",
            params={"characters": [f"@agent_{i}" for i in range(num_agents)], "topic": "rumor"}
        ))

        # Mutual influence
        for i in range(num_agents):
            for j in range(num_agents):
                if i != j:
                    builder.with_action(ScenarioAction(
                        action_type="apply_pressure",
                        params={
                            "target": f"@agent_{i}",
                            "source": f"@agent_{j}",
                            "topic": "rumor_belief",
                            "intensity": 0.1,
                            "direction": 1,
                        }
                    ))

        if r < 3:
            builder.transition_to(f"spread_{r + 1}")
        else:
            builder.transition_to("final_state")

    # Final state
    builder.add_phase("final_state", "Final beliefs")
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "The rumor is true",
            "topic": "rumor_belief",
            "characters": [f"@agent_{i}" for i in range(num_agents)],
            "threshold": 0.5,
        }
    ))

    return builder.build()


# =============================================================================
# POWER DYNAMICS
# =============================================================================

def create_coup_scenario(
    loyalist_count: int = 3,
    conspirator_count: int = 2,
) -> ScenarioDefinition:
    """Create a Coup/Power Struggle scenario.

    Conspirators try to remove the incumbent.
    Loyalists try to maintain the status quo.
    Swing votes decide the outcome.

    Roles: incumbent, loyalist_0..n, conspirator_0..n, swing_voter
    """
    builder = ScenarioBuilder("Power Struggle")
    builder.with_description(f"Coup attempt: {conspirator_count} conspirators vs {loyalist_count} loyalists")

    builder.requires_role("incumbent")
    builder.requires_role("swing_voter")

    builder.with_initial_beliefs("incumbent", {"legitimacy": 0.7, "awareness_of_threat": 0.3})
    builder.with_initial_beliefs("swing_voter", {"support_incumbent": 0.5, "risk_tolerance": 0.5})

    for i in range(loyalist_count):
        builder.requires_role(f"loyalist_{i}")
        builder.with_initial_beliefs(f"loyalist_{i}", {"loyalty": 0.8 + 0.15 * random.random()})

    for i in range(conspirator_count):
        builder.requires_role(f"conspirator_{i}")
        builder.with_initial_beliefs(f"conspirator_{i}", {"ambition": 0.7 + 0.2 * random.random(), "boldness": 0.5 + 0.3 * random.random()})

    builder.with_variable("coup_successful", False)
    builder.with_variable("detected_early", False)

    # Normal operations
    builder.add_phase("normal_operations", "Status quo")
    builder.with_action(_char_action("@incumbent", "say", "Everything is running smoothly under my leadership."))
    builder.with_action(ScenarioAction(
        action_type="dialogue_round",
        params={"characters": [f"@loyalist_{i}" for i in range(loyalist_count)], "topic": "support"}
    ))
    builder.transition_to("conspiracy_forms")

    # Conspiracy forms
    builder.add_phase("conspiracy_forms", "Secret plotting")
    builder.with_action(ScenarioAction(
        action_type="form_coalition",
        params={
            "characters": [f"@conspirator_{i}" for i in range(conspirator_count)],
            "topic": "ambition",
            "similarity_threshold": 0.2,
            "coalition_name": "conspiracy",
        }
    ))
    builder.with_action(ScenarioAction(
        action_type="foreach",
        params={
            "characters": [f"@conspirator_{i}" for i in range(conspirator_count)],
            "action": {
                "action_type": "character_action",
                "character_id": "$current_character",
                "params": {"type": "say", "content": "The time for change is now..."}
            }
        }
    ))
    builder.transition_to("recruit_swing")

    # Recruit swing voter
    builder.add_phase("recruit_swing", "Courting the swing voter")
    builder.with_action(ScenarioAction(
        action_type="apply_pressure",
        params={
            "target": "@swing_voter",
            "source": "@conspirator_0",
            "topic": "support_incumbent",
            "intensity": 0.3,
            "direction": -1,
        }
    ))
    builder.with_action(ScenarioAction(
        action_type="apply_pressure",
        params={
            "target": "@swing_voter",
            "source": "@incumbent",
            "topic": "support_incumbent",
            "intensity": 0.2,
            "direction": 1,
        }
    ))
    builder.transition_to("confrontation")

    # Confrontation
    builder.add_phase("confrontation", "The showdown")
    builder.with_action(_char_action("@conspirator_0", "say", "We call for a vote of no confidence!"))
    builder.with_action(_char_action("@incumbent", "say", "This is an illegitimate power grab!"))
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Remove the incumbent from power",
            "topic": "support_incumbent",
            "characters": [f"@loyalist_{i}" for i in range(loyalist_count)] +
                         [f"@conspirator_{i}" for i in range(conspirator_count)] +
                         ["@swing_voter"],
            "threshold": 0.5,
        }
    ))
    builder.transition_to("aftermath")

    # Aftermath
    builder.add_phase("aftermath", "New order")
    builder.with_action(ScenarioAction(
        action_type="conditional",
        params={
            "condition": {"conditions": [{"field": "variables.last_vote.passed", "operator": "eq", "value": True}], "mode": "all"},
            "then": {"action_type": "broadcast", "params": {"message": "The coup succeeds! New leadership takes over.", "topic": "power"}},
            "else": {"action_type": "broadcast", "params": {"message": "The coup fails! The incumbent consolidates power.", "topic": "power"}}
        }
    ))

    return builder.build()


def create_trial_scenario(
    juror_count: int = 6,
    evidence_strength: float = 0.6,
) -> ScenarioDefinition:
    """Create a Trial/Tribunal scenario.

    Prosecution and defense present cases.
    Jury deliberates and votes.

    Roles: judge, prosecutor, defender, defendant, witness, juror_0..n
    """
    builder = ScenarioBuilder("Trial by Jury")
    builder.with_description(f"Criminal trial with {juror_count} jurors, evidence strength {evidence_strength}")

    builder.requires_role("judge")
    builder.requires_role("prosecutor")
    builder.requires_role("defender")
    builder.requires_role("defendant")
    builder.requires_role("witness")

    for i in range(juror_count):
        builder.requires_role(f"juror_{i}")
        builder.with_initial_beliefs(f"juror_{i}", {
            "guilt_belief": 0.5,
            "reasonable_doubt_threshold": 0.3 + 0.4 * random.random(),
        })

    builder.with_variable("evidence_strength", evidence_strength)
    builder.with_variable("verdict", None)

    # Opening
    builder.add_phase("opening", "Opening statements")
    builder.with_action(_char_action("@judge", "say", "This court is now in session."))
    builder.with_action(_char_action("@prosecutor", "say", "The evidence will show the defendant is guilty beyond reasonable doubt."))
    builder.with_action(_char_action("@defender", "say", "My client is innocent. The prosecution's case is weak."))
    builder.transition_to("prosecution_case")

    # Prosecution case
    builder.add_phase("prosecution_case", "Prosecution presents")
    builder.with_action(_char_action("@prosecutor", "say", "I call the witness to testify."))
    builder.with_action(_char_action("@witness", "say", "I saw what happened. It was clearly the defendant."))
    builder.with_action(ScenarioAction(
        action_type="reveal_information",
        params={
            "revealer": "@prosecutor",
            "characters": [f"@juror_{i}" for i in range(juror_count)],
            "information": {"guilt_belief": evidence_strength},
            "impact": 0.3,
        }
    ))
    builder.transition_to("defense_case")

    # Defense case
    builder.add_phase("defense_case", "Defense presents")
    builder.with_action(_char_action("@defender", "say", "The witness is unreliable. There are alternative explanations."))
    builder.with_action(_char_action("@defendant", "say", "I am innocent. I was not there."))
    builder.with_action(ScenarioAction(
        action_type="foreach",
        params={
            "characters": [f"@juror_{i}" for i in range(juror_count)],
            "action": {
                "action_type": "apply_pressure",
                "params": {
                    "target": "$current_character",
                    "source": "@defender",
                    "topic": "guilt_belief",
                    "intensity": 0.2,
                    "direction": -1,
                }
            }
        }
    ))
    builder.transition_to("closing")

    # Closing arguments
    builder.add_phase("closing", "Closing arguments")
    builder.with_action(_char_action("@prosecutor", "say", "The evidence is clear. Convict the defendant."))
    builder.with_action(_char_action("@defender", "say", "Reasonable doubt exists. You must acquit."))
    builder.transition_to("deliberation")

    # Jury deliberation
    builder.add_phase("deliberation", "Jury deliberates")
    builder.with_action(_char_action("@judge", "say", "The jury will now deliberate."))
    builder.with_action(ScenarioAction(
        action_type="negotiate",
        params={
            "characters": [f"@juror_{i}" for i in range(juror_count)],
            "topic": "guilt_belief",
            "rounds": 3,
            "agreement_threshold": 0.2,
        }
    ))
    builder.transition_to("verdict")

    # Verdict
    builder.add_phase("verdict", "The verdict")
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Find the defendant guilty",
            "topic": "guilt_belief",
            "characters": [f"@juror_{i}" for i in range(juror_count)],
            "threshold": 0.7,  # Beyond reasonable doubt
        }
    ))
    builder.with_action(ScenarioAction(
        action_type="conditional",
        params={
            "condition": {"conditions": [{"field": "variables.last_vote.passed", "operator": "eq", "value": True}], "mode": "all"},
            "then": {"action_type": "character_action", "character_id": "@judge", "params": {"type": "say", "content": "The jury finds the defendant GUILTY."}},
            "else": {"action_type": "character_action", "character_id": "@judge", "params": {"type": "say", "content": "The jury finds the defendant NOT GUILTY."}}
        }
    ))

    return builder.build()


def create_board_takeover(
    incumbent_directors: int = 3,
    challenger_directors: int = 2,
) -> ScenarioDefinition:
    """Create a Corporate Board Takeover scenario.

    Activist investors challenge incumbent management.
    Proxy battle for board control.
    """
    builder = ScenarioBuilder("Board Takeover Battle")
    builder.with_description(f"Proxy fight: {challenger_directors} challengers vs {incumbent_directors} incumbents")

    builder.requires_role("ceo")
    builder.requires_role("activist_leader")
    builder.requires_role("institutional_investor")

    for i in range(incumbent_directors):
        builder.requires_role(f"incumbent_director_{i}")
        builder.with_initial_beliefs(f"incumbent_director_{i}", {"support_management": 0.8})

    for i in range(challenger_directors):
        builder.requires_role(f"challenger_{i}")
        builder.with_initial_beliefs(f"challenger_{i}", {"support_management": 0.2})

    builder.with_initial_beliefs("institutional_investor", {"support_management": 0.5})
    builder.with_variable("takeover_successful", False)

    # Status quo
    builder.add_phase("status_quo", "Current state")
    builder.with_action(_char_action("@ceo", "say", "Our strategy is delivering results. Stay the course."))
    builder.transition_to("activist_campaign")

    # Activist campaign
    builder.add_phase("activist_campaign", "Activist launches campaign")
    builder.with_action(_char_action("@activist_leader", "say", "This company is underperforming. We need change!"))
    builder.with_action(ScenarioAction(
        action_type="apply_pressure",
        params={
            "target": "@institutional_investor",
            "source": "@activist_leader",
            "topic": "support_management",
            "intensity": 0.3,
            "direction": -1,
        }
    ))
    builder.transition_to("management_response")

    # Management response
    builder.add_phase("management_response", "Management defends")
    builder.with_action(_char_action("@ceo", "say", "The activist's proposals would destroy shareholder value."))
    builder.with_action(ScenarioAction(
        action_type="apply_pressure",
        params={
            "target": "@institutional_investor",
            "source": "@ceo",
            "topic": "support_management",
            "intensity": 0.2,
            "direction": 1,
        }
    ))
    builder.transition_to("proxy_vote")

    # Proxy vote
    builder.add_phase("proxy_vote", "Shareholder vote")
    all_voters = [f"@incumbent_director_{i}" for i in range(incumbent_directors)] + \
                 [f"@challenger_{i}" for i in range(challenger_directors)] + \
                 ["@institutional_investor"]
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Support current management slate",
            "topic": "support_management",
            "characters": all_voters,
            "threshold": 0.5,
        }
    ))
    builder.transition_to("new_board")

    # New board
    builder.add_phase("new_board", "Board composition")
    builder.with_action(ScenarioAction(
        action_type="conditional",
        params={
            "condition": {"conditions": [{"field": "variables.last_vote.passed", "operator": "eq", "value": True}], "mode": "all"},
            "then": {"action_type": "character_action", "character_id": "@ceo", "params": {"type": "say", "content": "Shareholders have spoken. Management retained."}},
            "else": {"action_type": "character_action", "character_id": "@activist_leader", "params": {"type": "say", "content": "Victory! Time for new leadership."}}
        }
    ))

    return builder.build()


# =============================================================================
# STRATEGIC SCENARIOS
# =============================================================================

def create_war_room(
    advisors: int = 4,
    crisis_type: str = "geopolitical",
) -> ScenarioDefinition:
    """Create a War Room / Situation Room scenario.

    Leader must make critical decisions with incomplete information.
    Multiple advisors with different perspectives.

    Roles: commander, advisor_0..n, intelligence_officer
    """
    builder = ScenarioBuilder(f"War Room: {crisis_type}")
    builder.with_description(f"Critical decision making with {advisors} advisors")

    builder.requires_role("commander")
    builder.requires_role("intelligence_officer")

    perspectives = ["hawkish", "dovish", "pragmatic", "analytical"]
    for i in range(advisors):
        builder.requires_role(f"advisor_{i}")
        perspective = perspectives[i % len(perspectives)]
        builder.with_initial_beliefs(f"advisor_{i}", {
            "hawkishness": 0.8 if perspective == "hawkish" else 0.2 if perspective == "dovish" else 0.5,
            "risk_tolerance": random.random(),
        })

    builder.with_variable("crisis_type", crisis_type)
    builder.with_variable("decision_made", False)
    builder.with_variable("options_considered", [])

    # Briefing
    builder.add_phase("briefing", "Intelligence briefing")
    builder.with_action(_char_action("@intelligence_officer", "say", f"Commander, we have a developing {crisis_type} situation."))
    builder.with_action(_char_action("@commander", "say", "Give me the details."))
    builder.with_action(ScenarioAction(
        action_type="reveal_information",
        params={
            "revealer": "@intelligence_officer",
            "characters": ["@commander"] + [f"@advisor_{i}" for i in range(advisors)],
            "information": {"situation_severity": 0.7},
            "impact": 0.4,
        }
    ))
    builder.transition_to("options_generation")

    # Options generation
    builder.add_phase("options_generation", "Developing options")
    builder.with_action(_char_action("@commander", "say", "I need options. What can we do?"))
    builder.with_action(ScenarioAction(
        action_type="foreach",
        params={
            "characters": [f"@advisor_{i}" for i in range(advisors)],
            "action": {
                "action_type": "character_action",
                "character_id": "$current_character",
                "params": {"type": "propose", "content": "I recommend option X because..."}
            }
        }
    ))
    builder.transition_to("debate")

    # Debate
    builder.add_phase("debate", "Advisors debate")
    builder.with_action(ScenarioAction(
        action_type="negotiate",
        params={
            "characters": [f"@advisor_{i}" for i in range(advisors)],
            "topic": "hawkishness",
            "rounds": 2,
            "agreement_threshold": 0.3,
        }
    ))
    builder.transition_to("decision")

    # Decision
    builder.add_phase("decision", "Commander decides")
    builder.with_action(_char_action("@commander", "think", "Weighing all the options..."))
    builder.with_action(_char_action("@commander", "decide", "I have made my decision."))
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Support the hawkish option",
            "topic": "hawkishness",
            "characters": [f"@advisor_{i}" for i in range(advisors)],
            "threshold": 0.5,
        }
    ))
    builder.transition_to("execution")

    # Execution
    builder.add_phase("execution", "Executing the plan")
    builder.with_action(_char_action("@commander", "say", "Execute the plan. May we make the right choice."))

    return builder.build()


def create_treaty_negotiation(
    parties: List[str] = None,
    issues: List[str] = None,
) -> ScenarioDefinition:
    """Create a Multi-Issue Treaty Negotiation scenario.

    Multiple parties negotiate over multiple issues.
    Tests logrolling, package deals, and compromise.
    """
    parties = parties or ["nation_a", "nation_b", "nation_c"]
    issues = issues or ["trade", "security", "environment"]

    builder = ScenarioBuilder(f"Treaty Negotiation ({len(parties)} parties)")
    builder.with_description(f"Negotiating over: {', '.join(issues)}")

    for party in parties:
        builder.requires_role(party)
        # Random preferences over issues
        beliefs = {issue: random.random() for issue in issues}
        builder.with_initial_beliefs(party, beliefs)

    builder.with_variable("issues", issues)
    builder.with_variable("agreed_issues", [])
    builder.with_variable("treaty_signed", False)

    # Opening positions
    builder.add_phase("opening", "Opening positions")
    builder.with_action(ScenarioAction(
        action_type="foreach",
        params={
            "characters": [f"@{party}" for party in parties],
            "action": {
                "action_type": "character_action",
                "character_id": "$current_character",
                "params": {"type": "say", "content": "Our priorities in this negotiation are..."}
            }
        }
    ))
    builder.transition_to(f"negotiate_{issues[0]}")

    # Negotiate each issue
    for i, issue in enumerate(issues):
        phase_name = f"negotiate_{issue}"
        builder.add_phase(phase_name, f"Negotiating {issue}")

        builder.with_action(ScenarioAction(
            action_type="negotiate",
            params={
                "characters": [f"@{party}" for party in parties],
                "topic": issue,
                "rounds": 2,
                "agreement_threshold": 0.25,
            }
        ))

        if i < len(issues) - 1:
            builder.transition_to(f"negotiate_{issues[i + 1]}")
        else:
            builder.transition_to("package_deal")

    # Package deal
    builder.add_phase("package_deal", "Assembling the treaty")
    builder.with_action(ScenarioAction(
        action_type="dialogue_round",
        params={"characters": [f"@{party}" for party in parties], "topic": "final_terms"}
    ))
    builder.transition_to("signing")

    # Signing
    builder.add_phase("signing", "Treaty signing")
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Sign the treaty",
            "topic": issues[0],  # Use first issue as proxy
            "characters": [f"@{party}" for party in parties],
            "threshold": 0.4,
        }
    ))
    builder.with_action(ScenarioAction(
        action_type="conditional",
        params={
            "condition": {"conditions": [{"field": "variables.last_vote.passed", "operator": "eq", "value": True}], "mode": "all"},
            "then": {"action_type": "broadcast", "params": {"message": "The treaty is signed!", "topic": "treaty"}},
            "else": {"action_type": "broadcast", "params": {"message": "Negotiations have failed.", "topic": "treaty"}}
        }
    ))

    return builder.build()


def create_auction(
    bidders: int = 4,
    auction_type: str = "english",  # english, dutch, sealed
    item: str = "rare asset",
) -> ScenarioDefinition:
    """Create an Auction scenario.

    Multiple bidders compete for an item.
    Supports different auction formats.
    """
    builder = ScenarioBuilder(f"{auction_type.title()} Auction: {item}")
    builder.with_description(f"{bidders} bidders competing for {item}")

    builder.requires_role("auctioneer")
    for i in range(bidders):
        builder.requires_role(f"bidder_{i}")
        # Private valuation
        builder.with_initial_beliefs(f"bidder_{i}", {
            "valuation": 0.3 + 0.6 * random.random(),
            "aggressiveness": random.random(),
        })

    builder.with_variable("item", item)
    builder.with_variable("current_bid", 0.0)
    builder.with_variable("winning_bidder", None)

    # Auction start
    builder.add_phase("start", "Auction begins")
    builder.with_action(_char_action("@auctioneer", "say", f"Welcome! Today's item: {item}. Let the bidding begin!"))

    if auction_type == "english":
        builder.transition_to("bidding_round_1")

        # Multiple bidding rounds
        for r in range(1, 6):
            phase_name = f"bidding_round_{r}"
            builder.add_phase(phase_name, f"Bidding round {r}")

            builder.with_action(ScenarioAction(
                action_type="foreach",
                params={
                    "characters": [f"@bidder_{i}" for i in range(bidders)],
                    "action": {
                        "action_type": "character_action",
                        "character_id": "$current_character",
                        "params": {"type": "decide", "content": f"Round {r}: Bid or pass?"}
                    }
                }
            ))

            if r < 5:
                builder.transition_to(f"bidding_round_{r + 1}")
            else:
                builder.transition_to("winner")

    elif auction_type == "sealed":
        builder.transition_to("sealed_bids")

        builder.add_phase("sealed_bids", "Sealed bid submission")
        builder.with_action(_char_action("@auctioneer", "say", "Submit your sealed bids now."))
        builder.with_action(ScenarioAction(
            action_type="foreach",
            params={
                "characters": [f"@bidder_{i}" for i in range(bidders)],
                "action": {
                    "action_type": "character_action",
                    "character_id": "$current_character",
                    "params": {"type": "decide", "content": "My sealed bid is..."}
                }
            }
        ))
        builder.transition_to("winner")

    # Winner announced
    builder.add_phase("winner", "Winner announced")
    builder.with_action(_char_action("@auctioneer", "say", "The auction is complete!"))
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Satisfied with auction outcome",
            "topic": "valuation",
            "characters": [f"@bidder_{i}" for i in range(bidders)],
            "threshold": 0.5,
        }
    ))

    return builder.build()


# =============================================================================
# SOCIAL DYNAMICS
# =============================================================================

def create_town_hall(
    officials: int = 2,
    citizens: int = 4,
    topic: str = "local policy",
) -> ScenarioDefinition:
    """Create a Town Hall Meeting scenario.

    Officials face questions from citizens.
    Tests public communication and accountability.
    """
    builder = ScenarioBuilder(f"Town Hall: {topic}")
    builder.with_description(f"{officials} officials face {citizens} citizens on {topic}")

    builder.requires_role("moderator")
    for i in range(officials):
        builder.requires_role(f"official_{i}")
        builder.with_initial_beliefs(f"official_{i}", {"approval": 0.5, "defensiveness": 0.3})

    for i in range(citizens):
        builder.requires_role(f"citizen_{i}")
        builder.with_initial_beliefs(f"citizen_{i}", {
            "satisfaction": 0.3 + 0.4 * random.random(),
            "engagement": random.random(),
        })

    builder.with_variable("topic", topic)
    builder.with_variable("questions_asked", 0)

    # Opening
    builder.add_phase("opening", "Meeting opens")
    builder.with_action(_char_action("@moderator", "say", f"Welcome to tonight's town hall on {topic}."))
    builder.with_action(ScenarioAction(
        action_type="foreach",
        params={
            "characters": [f"@official_{i}" for i in range(officials)],
            "action": {
                "action_type": "character_action",
                "character_id": "$current_character",
                "params": {"type": "say", "content": "Thank you for coming. I'm here to listen."}
            }
        }
    ))
    builder.transition_to("qa_round_1")

    # Q&A rounds
    for r in range(1, citizens + 1):
        phase_name = f"qa_round_{r}"
        builder.add_phase(phase_name, f"Q&A round {r}")

        citizen_idx = (r - 1) % citizens
        builder.with_action(_char_action(f"@citizen_{citizen_idx}", "ask", f"My question about {topic} is..."))

        for i in range(officials):
            builder.with_action(_char_action(f"@official_{i}", "reply", "Let me address that concern..."))

        # Citizen satisfaction update
        builder.with_action(ScenarioAction(
            action_type="apply_pressure",
            params={
                "target": f"@citizen_{citizen_idx}",
                "source": "@official_0",
                "topic": "satisfaction",
                "intensity": 0.15,
                "direction": 1,
            }
        ))

        if r < citizens:
            builder.transition_to(f"qa_round_{r + 1}")
        else:
            builder.transition_to("closing")

    # Closing
    builder.add_phase("closing", "Meeting closes")
    builder.with_action(_char_action("@moderator", "say", "Thank you all for participating."))
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Officials adequately addressed concerns",
            "topic": "satisfaction",
            "characters": [f"@citizen_{i}" for i in range(citizens)],
            "threshold": 0.5,
        }
    ))

    return builder.build()


def create_press_conference(
    speaker: str = "executive",
    journalists: int = 4,
    topic: str = "major announcement",
) -> ScenarioDefinition:
    """Create a Press Conference scenario.

    Speaker makes announcement and faces journalist questions.
    Tests messaging discipline and crisis communication.
    """
    builder = ScenarioBuilder(f"Press Conference: {topic}")
    builder.with_description(f"Speaker faces {journalists} journalists on {topic}")

    builder.requires_role(speaker)
    builder.requires_role("pr_handler")

    builder.with_initial_beliefs(speaker, {"confidence": 0.7, "message_discipline": 0.6})

    for i in range(journalists):
        builder.requires_role(f"journalist_{i}")
        builder.with_initial_beliefs(f"journalist_{i}", {
            "skepticism": 0.4 + 0.4 * random.random(),
            "aggressiveness": random.random(),
        })

    builder.with_variable("topic", topic)
    builder.with_variable("gaffes", 0)

    # Opening statement
    builder.add_phase("statement", "Opening statement")
    builder.with_action(_char_action(f"@{speaker}", "say", f"Thank you for coming. Today I'm announcing {topic}."))
    builder.with_action(_char_action("@pr_handler", "think", "Keep them on message..."))
    builder.transition_to("questions")

    # Questions
    builder.add_phase("questions", "Q&A session")
    for i in range(journalists):
        builder.with_action(_char_action(f"@journalist_{i}", "ask", "Can you elaborate on..."))
        builder.with_action(_char_action(f"@{speaker}", "reply", "Let me be clear about that..."))

        # Skepticism affects speaker confidence
        builder.with_action(ScenarioAction(
            action_type="apply_pressure",
            params={
                "target": f"@{speaker}",
                "source": f"@journalist_{i}",
                "topic": "confidence",
                "intensity": 0.1,
                "direction": -1,
            }
        ))
    builder.transition_to("followup")

    # Follow-up
    builder.add_phase("followup", "Follow-up questions")
    builder.with_action(_char_action("@journalist_0", "ask", "One more question..."))
    builder.with_action(_char_action(f"@{speaker}", "reply", "Final answer..."))
    builder.transition_to("wrap")

    # Wrap up
    builder.add_phase("wrap", "Conference ends")
    builder.with_action(_char_action("@pr_handler", "say", "That's all we have time for today."))
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "The speaker handled questions well",
            "topic": "skepticism",
            "characters": [f"@journalist_{i}" for i in range(journalists)],
            "threshold": 0.4,
        }
    ))

    return builder.build()


def create_interview(
    interviewer_style: str = "investigative",  # friendly, investigative, hostile
) -> ScenarioDefinition:
    """Create an Interview scenario.

    One-on-one interview with different styles.
    Tests information extraction and evasion.
    """
    builder = ScenarioBuilder(f"{interviewer_style.title()} Interview")
    builder.with_description(f"{interviewer_style} style interview")

    builder.requires_role("interviewer")
    builder.requires_role("subject")

    style_params = {
        "friendly": {"pressure": 0.1, "trust_building": True},
        "investigative": {"pressure": 0.2, "trust_building": False},
        "hostile": {"pressure": 0.4, "trust_building": False},
    }
    params = style_params.get(interviewer_style, style_params["investigative"])

    builder.with_initial_beliefs("interviewer", {"determination": 0.7})
    builder.with_initial_beliefs("subject", {
        "openness": 0.5 if params["trust_building"] else 0.3,
        "defensiveness": 0.3 if params["trust_building"] else 0.6,
    })

    builder.with_variable("style", interviewer_style)
    builder.with_variable("information_extracted", 0)

    # Warmup
    builder.add_phase("warmup", "Interview warmup")
    if params["trust_building"]:
        builder.with_action(_char_action("@interviewer", "say", "Thanks for sitting down with me. Let's have a conversation."))
    else:
        builder.with_action(_char_action("@interviewer", "say", "Let's get started. I have some important questions."))
    builder.with_action(_char_action("@subject", "reply", "I'm ready."))
    builder.transition_to("main_questions")

    # Main questions
    builder.add_phase("main_questions", "Core questions")
    for i in range(3):
        builder.with_action(_char_action("@interviewer", "ask", f"Question {i+1}: Tell me about..."))
        builder.with_action(_char_action("@subject", "reply", "Well, I would say that..."))

        builder.with_action(ScenarioAction(
            action_type="apply_pressure",
            params={
                "target": "@subject",
                "source": "@interviewer",
                "topic": "openness",
                "intensity": params["pressure"],
                "direction": 1 if params["trust_building"] else -1,
            }
        ))
    builder.transition_to("probing")

    # Probing
    builder.add_phase("probing", "Probing deeper")
    builder.with_action(_char_action("@interviewer", "ask", "Let me push back on that..."))
    builder.with_action(_char_action("@subject", "reply", "I understand your concern, but..."))
    builder.with_action(ScenarioAction(
        action_type="apply_pressure",
        params={
            "target": "@subject",
            "source": "@interviewer",
            "topic": "defensiveness",
            "intensity": params["pressure"] * 1.5,
            "direction": 1,
        }
    ))
    builder.transition_to("closing")

    # Closing
    builder.add_phase("closing", "Interview closing")
    builder.with_action(_char_action("@interviewer", "say", "One final question..."))
    builder.with_action(_char_action("@subject", "reply", "I appreciate the conversation."))

    return builder.build()


def create_mediation(
    disputants: int = 2,
    dispute_type: str = "resource_allocation",
) -> ScenarioDefinition:
    """Create a Mediation scenario.

    Neutral mediator helps parties resolve dispute.
    Tests conflict resolution and compromise finding.
    """
    builder = ScenarioBuilder(f"Mediation: {dispute_type}")
    builder.with_description(f"Mediating {dispute_type} between {disputants} parties")

    builder.requires_role("mediator")
    for i in range(disputants):
        builder.requires_role(f"disputant_{i}")
        builder.with_initial_beliefs(f"disputant_{i}", {
            "willingness_to_compromise": 0.3 + 0.4 * random.random(),
            "trust_in_process": 0.5,
            "satisfaction": 0.3,
        })

    builder.with_variable("dispute_type", dispute_type)
    builder.with_variable("resolution_reached", False)

    # Opening
    builder.add_phase("opening", "Mediation begins")
    builder.with_action(_char_action("@mediator", "say", "Welcome. We're here to find a resolution that works for everyone."))
    builder.with_action(ScenarioAction(
        action_type="foreach",
        params={
            "characters": [f"@disputant_{i}" for i in range(disputants)],
            "action": {
                "action_type": "apply_pressure",
                "params": {
                    "target": "$current_character",
                    "source": "@mediator",
                    "topic": "trust_in_process",
                    "intensity": 0.15,
                    "direction": 1,
                }
            }
        }
    ))
    builder.transition_to("grievances")

    # Air grievances
    builder.add_phase("grievances", "Airing grievances")
    builder.with_action(_char_action("@mediator", "say", "Let each party explain their position."))
    builder.with_action(ScenarioAction(
        action_type="foreach",
        params={
            "characters": [f"@disputant_{i}" for i in range(disputants)],
            "action": {
                "action_type": "character_action",
                "character_id": "$current_character",
                "params": {"type": "say", "content": "My grievance is..."}
            }
        }
    ))
    builder.transition_to("interests")

    # Identify interests
    builder.add_phase("interests", "Identifying underlying interests")
    builder.with_action(_char_action("@mediator", "ask", "What do you really need from this?"))
    builder.with_action(ScenarioAction(
        action_type="dialogue_round",
        params={"characters": [f"@disputant_{i}" for i in range(disputants)], "topic": "interests"}
    ))
    builder.transition_to("options")

    # Generate options
    builder.add_phase("options", "Generating options")
    builder.with_action(_char_action("@mediator", "say", "Let's brainstorm possible solutions."))
    builder.with_action(ScenarioAction(
        action_type="negotiate",
        params={
            "characters": [f"@disputant_{i}" for i in range(disputants)],
            "topic": "willingness_to_compromise",
            "rounds": 3,
            "agreement_threshold": 0.25,
        }
    ))
    builder.transition_to("agreement")

    # Reach agreement
    builder.add_phase("agreement", "Finalizing agreement")
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Accept the mediated agreement",
            "topic": "satisfaction",
            "characters": [f"@disputant_{i}" for i in range(disputants)],
            "threshold": 0.4,
        }
    ))
    builder.with_action(ScenarioAction(
        action_type="conditional",
        params={
            "condition": {"conditions": [{"field": "variables.last_vote.passed", "operator": "eq", "value": True}], "mode": "all"},
            "then": {"action_type": "character_action", "character_id": "@mediator", "params": {"type": "say", "content": "Congratulations! We have reached an agreement."}},
            "else": {"action_type": "character_action", "character_id": "@mediator", "params": {"type": "say", "content": "We'll need to continue working on this."}}
        }
    ))

    return builder.build()


# =============================================================================
# SCENARIO REGISTRY
# =============================================================================

ADVANCED_SCENARIO_TEMPLATES = {
    # Game Theory
    "prisoners_dilemma": create_prisoners_dilemma,
    "ultimatum_game": create_ultimatum_game,
    "public_goods_game": create_public_goods_game,
    "stag_hunt": create_stag_hunt,

    # Information Dynamics
    "information_cascade": create_information_cascade,
    "whistleblower": create_whistleblower_scenario,
    "rumor_propagation": create_rumor_propagation,

    # Power Dynamics
    "coup": create_coup_scenario,
    "trial": create_trial_scenario,
    "board_takeover": create_board_takeover,

    # Strategic
    "war_room": create_war_room,
    "treaty_negotiation": create_treaty_negotiation,
    "auction": create_auction,

    # Social
    "town_hall": create_town_hall,
    "press_conference": create_press_conference,
    "interview": create_interview,
    "mediation": create_mediation,
}


def list_advanced_scenario_templates() -> List[str]:
    """List available advanced scenario templates."""
    return list(ADVANCED_SCENARIO_TEMPLATES.keys())


def create_advanced_scenario(template_name: str, **kwargs) -> ScenarioDefinition:
    """Create a scenario from an advanced template."""
    if template_name not in ADVANCED_SCENARIO_TEMPLATES:
        raise ValueError(f"Unknown template: {template_name}. Available: {list_advanced_scenario_templates()}")
    return ADVANCED_SCENARIO_TEMPLATES[template_name](**kwargs)
