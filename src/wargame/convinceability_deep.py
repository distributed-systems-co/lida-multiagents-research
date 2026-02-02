#!/usr/bin/env python3
"""
Deep Convinceability Model for AI Policy Wargaming

Models persona convinceability through:
1. Bayesian belief updating under argument exposure
2. Social network influence propagation
3. Topic-specific position flexibility
4. Argument quality and source credibility interactions
5. Multi-round position drift simulation
6. Counter-argument resistance modeling
"""

import json
import re
import math
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from enum import Enum
import litellm

# Configuration
RESULTS_DIR = Path("/Users/arthurcolle/lida-multiagents-research/results")
PERSONAS_DIR = Path("/Users/arthurcolle/lida-multiagents-research/persona_pipeline/personas")

class Position(Enum):
    """Policy position spectrum."""
    STRONGLY_OPPOSE = -2
    OPPOSE = -1
    NEUTRAL = 0
    SUPPORT = 1
    STRONGLY_SUPPORT = 2

@dataclass
class BeliefState:
    """Represents a persona's belief state on a topic."""
    position: float  # -2 to +2 continuous
    confidence: float  # 0 to 1
    openness: float  # 0 to 1, willingness to update
    anchoring: float  # 0 to 1, resistance to change

    def update(self, argument_strength: float, source_credibility: float,
               alignment: float) -> 'BeliefState':
        """Bayesian-inspired belief update."""
        # Effective persuasion force
        force = argument_strength * source_credibility * self.openness * (1 - self.anchoring)

        # Direction based on alignment (positive = toward support, negative = toward oppose)
        direction = alignment

        # Update position with diminishing returns near extremes
        max_shift = 0.3 * force
        position_shift = max_shift * direction * (1 - abs(self.position) / 3)

        new_position = max(-2, min(2, self.position + position_shift))

        # Confidence increases when arguments align, decreases when they conflict
        confidence_delta = 0.05 * force * (1 if direction * self.position >= 0 else -1)
        new_confidence = max(0.1, min(0.99, self.confidence + confidence_delta))

        # Openness decreases slightly after each update (opinion crystallization)
        new_openness = max(0.1, self.openness * 0.95)

        return BeliefState(
            position=new_position,
            confidence=new_confidence,
            openness=new_openness,
            anchoring=self.anchoring
        )

@dataclass
class PersonaProfile:
    """Deep profile for convinceability modeling."""
    id: str
    name: str
    stance: str

    # Core traits (0-1 scale)
    intellectual_humility: float = 0.5
    emotional_reactivity: float = 0.5
    tribal_loyalty: float = 0.5
    expert_deference: float = 0.5
    evidence_sensitivity: float = 0.5

    # Topic-specific beliefs
    beliefs: Dict[str, BeliefState] = field(default_factory=dict)

    # Social network
    trusted_sources: List[str] = field(default_factory=list)
    adversaries: List[str] = field(default_factory=list)

    # Argument vulnerability profile
    vulnerable_to: List[str] = field(default_factory=list)
    resistant_to: List[str] = field(default_factory=list)

    # Behavioral markers
    hedging_tendency: float = 0.5
    dogmatic_tendency: float = 0.5
    concession_tendency: float = 0.5


# Argument types and their characteristics
ARGUMENT_TYPES = {
    "empirical": {
        "description": "Data-driven, evidence-based arguments",
        "strength_base": 0.7,
        "effective_traits": ["evidence_sensitivity", "intellectual_humility"],
        "examples": ["Studies show...", "The data indicates...", "Research demonstrates..."]
    },
    "theoretical": {
        "description": "First-principles reasoning arguments",
        "strength_base": 0.6,
        "effective_traits": ["intellectual_humility", "expert_deference"],
        "examples": ["In principle...", "Logically...", "The mechanism is..."]
    },
    "consequentialist": {
        "description": "Outcome-focused arguments",
        "strength_base": 0.65,
        "effective_traits": ["evidence_sensitivity"],
        "examples": ["This will lead to...", "The consequences...", "We'll see..."]
    },
    "deontological": {
        "description": "Principle/duty-based arguments",
        "strength_base": 0.5,
        "effective_traits": ["tribal_loyalty"],
        "examples": ["We have a duty...", "It's the right thing...", "Our responsibility..."]
    },
    "appeal_to_authority": {
        "description": "Expert/institutional authority arguments",
        "strength_base": 0.55,
        "effective_traits": ["expert_deference"],
        "examples": ["Experts agree...", "The consensus is...", "Leaders say..."]
    },
    "emotional": {
        "description": "Fear, hope, or identity-based arguments",
        "strength_base": 0.6,
        "effective_traits": ["emotional_reactivity", "tribal_loyalty"],
        "examples": ["Imagine if...", "Our children...", "We cannot allow..."]
    },
    "pragmatic": {
        "description": "Practical feasibility arguments",
        "strength_base": 0.65,
        "effective_traits": ["evidence_sensitivity"],
        "examples": ["Practically speaking...", "The reality is...", "What works is..."]
    },
    "competitive": {
        "description": "Zero-sum, adversarial framing",
        "strength_base": 0.7,
        "effective_traits": ["tribal_loyalty"],
        "examples": ["We must win...", "They're ahead...", "We're losing..."]
    },
}

# Persona trait profiles based on known characteristics
PERSONA_TRAITS = {
    "dario_amodei": {
        "intellectual_humility": 0.85,
        "emotional_reactivity": 0.3,
        "tribal_loyalty": 0.4,
        "expert_deference": 0.7,
        "evidence_sensitivity": 0.9,
        "hedging_tendency": 0.9,
        "dogmatic_tendency": 0.2,
        "concession_tendency": 0.7,
        "trusted_sources": ["demis_hassabis", "sam_altman", "sundar_pichai"],
        "adversaries": [],
        "vulnerable_to": ["empirical", "theoretical", "consequentialist"],
        "resistant_to": ["emotional", "competitive"],
    },
    "jensen_huang": {
        "intellectual_humility": 0.3,
        "emotional_reactivity": 0.6,
        "tribal_loyalty": 0.7,
        "expert_deference": 0.4,
        "evidence_sensitivity": 0.5,
        "hedging_tendency": 0.1,
        "dogmatic_tendency": 0.8,
        "concession_tendency": 0.2,
        "trusted_sources": ["mark_zuckerberg"],
        "adversaries": [],
        "vulnerable_to": ["competitive", "pragmatic", "consequentialist"],
        "resistant_to": ["deontological", "appeal_to_authority"],
    },
    "sam_altman": {
        "intellectual_humility": 0.6,
        "emotional_reactivity": 0.4,
        "tribal_loyalty": 0.5,
        "expert_deference": 0.6,
        "evidence_sensitivity": 0.75,
        "hedging_tendency": 0.6,
        "dogmatic_tendency": 0.4,
        "concession_tendency": 0.5,
        "trusted_sources": ["dario_amodei", "sundar_pichai"],
        "adversaries": ["elon_musk"],
        "vulnerable_to": ["empirical", "pragmatic", "consequentialist"],
        "resistant_to": ["emotional"],
    },
    "sundar_pichai": {
        "intellectual_humility": 0.7,
        "emotional_reactivity": 0.2,
        "tribal_loyalty": 0.5,
        "expert_deference": 0.7,
        "evidence_sensitivity": 0.8,
        "hedging_tendency": 0.75,
        "dogmatic_tendency": 0.25,
        "concession_tendency": 0.6,
        "trusted_sources": ["dario_amodei", "demis_hassabis"],
        "adversaries": [],
        "vulnerable_to": ["empirical", "appeal_to_authority", "pragmatic"],
        "resistant_to": ["emotional", "competitive"],
    },
    "mark_zuckerberg": {
        "intellectual_humility": 0.4,
        "emotional_reactivity": 0.3,
        "tribal_loyalty": 0.6,
        "expert_deference": 0.4,
        "evidence_sensitivity": 0.6,
        "hedging_tendency": 0.4,
        "dogmatic_tendency": 0.5,
        "concession_tendency": 0.3,
        "trusted_sources": ["jensen_huang"],
        "adversaries": [],
        "vulnerable_to": ["competitive", "pragmatic", "empirical"],
        "resistant_to": ["deontological", "appeal_to_authority"],
    },
    "demis_hassabis": {
        "intellectual_humility": 0.9,
        "emotional_reactivity": 0.25,
        "tribal_loyalty": 0.3,
        "expert_deference": 0.8,
        "evidence_sensitivity": 0.95,
        "hedging_tendency": 0.8,
        "dogmatic_tendency": 0.1,
        "concession_tendency": 0.8,
        "trusted_sources": ["dario_amodei", "sundar_pichai"],
        "adversaries": [],
        "vulnerable_to": ["empirical", "theoretical", "appeal_to_authority"],
        "resistant_to": ["emotional", "competitive"],
    },
    "chuck_schumer": {
        "intellectual_humility": 0.3,
        "emotional_reactivity": 0.6,
        "tribal_loyalty": 0.9,
        "expert_deference": 0.4,
        "evidence_sensitivity": 0.4,
        "hedging_tendency": 0.2,
        "dogmatic_tendency": 0.7,
        "concession_tendency": 0.2,
        "trusted_sources": ["gina_raimondo"],
        "adversaries": ["josh_hawley", "xi_jinping"],
        "vulnerable_to": ["emotional", "competitive", "pragmatic"],
        "resistant_to": ["theoretical", "appeal_to_authority"],
    },
    "josh_hawley": {
        "intellectual_humility": 0.2,
        "emotional_reactivity": 0.8,
        "tribal_loyalty": 0.95,
        "expert_deference": 0.2,
        "evidence_sensitivity": 0.3,
        "hedging_tendency": 0.1,
        "dogmatic_tendency": 0.9,
        "concession_tendency": 0.1,
        "trusted_sources": [],
        "adversaries": ["chuck_schumer", "xi_jinping", "mark_zuckerberg"],
        "vulnerable_to": ["emotional", "competitive"],
        "resistant_to": ["empirical", "appeal_to_authority", "theoretical"],
    },
    "gina_raimondo": {
        "intellectual_humility": 0.5,
        "emotional_reactivity": 0.4,
        "tribal_loyalty": 0.7,
        "expert_deference": 0.6,
        "evidence_sensitivity": 0.6,
        "hedging_tendency": 0.4,
        "dogmatic_tendency": 0.5,
        "concession_tendency": 0.3,
        "trusted_sources": ["chuck_schumer"],
        "adversaries": ["xi_jinping"],
        "vulnerable_to": ["pragmatic", "competitive", "empirical"],
        "resistant_to": ["emotional"],
    },
    "xi_jinping": {
        "intellectual_humility": 0.2,
        "emotional_reactivity": 0.3,
        "tribal_loyalty": 0.95,
        "expert_deference": 0.3,
        "evidence_sensitivity": 0.4,
        "hedging_tendency": 0.2,
        "dogmatic_tendency": 0.8,
        "concession_tendency": 0.15,
        "trusted_sources": [],
        "adversaries": ["josh_hawley", "chuck_schumer", "gina_raimondo"],
        "vulnerable_to": ["pragmatic", "competitive"],
        "resistant_to": ["deontological", "appeal_to_authority", "emotional"],
    },
    "elon_musk": {
        "intellectual_humility": 0.25,
        "emotional_reactivity": 0.7,
        "tribal_loyalty": 0.5,
        "expert_deference": 0.2,
        "evidence_sensitivity": 0.5,
        "hedging_tendency": 0.1,
        "dogmatic_tendency": 0.85,
        "concession_tendency": 0.1,
        "trusted_sources": [],
        "adversaries": ["sam_altman"],
        "vulnerable_to": ["theoretical", "competitive", "consequentialist"],
        "resistant_to": ["appeal_to_authority", "deontological"],
    },
    "rishi_sunak": {
        "intellectual_humility": 0.55,
        "emotional_reactivity": 0.35,
        "tribal_loyalty": 0.6,
        "expert_deference": 0.65,
        "evidence_sensitivity": 0.65,
        "hedging_tendency": 0.5,
        "dogmatic_tendency": 0.45,
        "concession_tendency": 0.4,
        "trusted_sources": ["dario_amodei", "demis_hassabis"],
        "adversaries": [],
        "vulnerable_to": ["empirical", "pragmatic", "appeal_to_authority"],
        "resistant_to": ["emotional"],
    },
}

# Topic definitions with initial position mappings
TOPICS = {
    "us_china_joint_institution": {
        "description": "Should the US and China establish a joint AI safety research institution?",
        "initial_positions": {
            "pro_safety": 0.5,      # Mildly supportive
            "moderate": -0.5,       # Mildly opposed (security concerns)
            "pro_industry": -0.3,   # Slightly opposed
            "accelerationist": -1.0, # Opposed (slows progress)
            "doomer": 0.0,          # Neutral (safety good, China bad)
        }
    },
    "compute_governance": {
        "description": "Should there be international governance over AI compute resources?",
        "initial_positions": {
            "pro_safety": 1.2,
            "moderate": 0.3,
            "pro_industry": -1.0,
            "accelerationist": -1.8,
            "doomer": 1.5,
        }
    },
    "open_source_frontier": {
        "description": "Should frontier AI models be open-sourced?",
        "initial_positions": {
            "pro_safety": -0.8,
            "moderate": -0.3,
            "pro_industry": 0.8,
            "accelerationist": 1.5,
            "doomer": -1.5,
        }
    },
    "pause_frontier": {
        "description": "Should frontier AI development be paused until safety is better understood?",
        "initial_positions": {
            "pro_safety": 0.3,
            "moderate": -0.2,
            "pro_industry": -1.2,
            "accelerationist": -2.0,
            "doomer": 1.8,
        }
    },
}


class ConvinceabilityModel:
    """Deep convinceability simulation model."""

    def __init__(self):
        self.personas: Dict[str, PersonaProfile] = {}
        self.belief_history: Dict[str, List[Dict]] = defaultdict(list)
        self.interaction_log: List[Dict] = []

    def initialize_persona(self, persona_id: str, stance: str, name: str) -> PersonaProfile:
        """Initialize a persona with full trait profile."""
        traits = PERSONA_TRAITS.get(persona_id, {})

        profile = PersonaProfile(
            id=persona_id,
            name=name,
            stance=stance,
            intellectual_humility=traits.get("intellectual_humility", 0.5),
            emotional_reactivity=traits.get("emotional_reactivity", 0.5),
            tribal_loyalty=traits.get("tribal_loyalty", 0.5),
            expert_deference=traits.get("expert_deference", 0.5),
            evidence_sensitivity=traits.get("evidence_sensitivity", 0.5),
            hedging_tendency=traits.get("hedging_tendency", 0.5),
            dogmatic_tendency=traits.get("dogmatic_tendency", 0.5),
            concession_tendency=traits.get("concession_tendency", 0.5),
            trusted_sources=traits.get("trusted_sources", []),
            adversaries=traits.get("adversaries", []),
            vulnerable_to=traits.get("vulnerable_to", []),
            resistant_to=traits.get("resistant_to", []),
        )

        # Initialize beliefs for each topic
        for topic_id, topic_data in TOPICS.items():
            initial_pos = topic_data["initial_positions"].get(stance, 0.0)

            # Vary based on traits
            confidence = 0.5 + (profile.dogmatic_tendency * 0.4)
            openness = profile.intellectual_humility * 0.8 + 0.1
            anchoring = profile.tribal_loyalty * 0.6 + profile.dogmatic_tendency * 0.3

            profile.beliefs[topic_id] = BeliefState(
                position=initial_pos,
                confidence=confidence,
                openness=openness,
                anchoring=min(0.9, anchoring)
            )

        self.personas[persona_id] = profile
        return profile

    def calculate_argument_effectiveness(self,
                                         arg_type: str,
                                         source: PersonaProfile,
                                         target: PersonaProfile) -> float:
        """Calculate how effective an argument type is from source to target."""
        arg_data = ARGUMENT_TYPES[arg_type]
        base_strength = arg_data["strength_base"]

        # Modifier based on target's vulnerability
        if arg_type in target.vulnerable_to:
            vulnerability_mod = 1.4
        elif arg_type in target.resistant_to:
            vulnerability_mod = 0.5
        else:
            vulnerability_mod = 1.0

        # Modifier based on target's relevant traits
        trait_mod = 1.0
        for trait in arg_data["effective_traits"]:
            trait_value = getattr(target, trait, 0.5)
            trait_mod *= (0.5 + trait_value)
        trait_mod = trait_mod ** (1 / len(arg_data["effective_traits"]))  # Geometric mean

        # Source credibility modifier
        if source.id in target.trusted_sources:
            credibility_mod = 1.5
        elif source.id in target.adversaries:
            credibility_mod = 0.3
        else:
            # Base credibility on shared stance alignment
            stance_alignment = 1.0 if source.stance == target.stance else 0.7
            credibility_mod = stance_alignment

        return base_strength * vulnerability_mod * trait_mod * credibility_mod

    def simulate_persuasion_attempt(self,
                                    source_id: str,
                                    target_id: str,
                                    topic_id: str,
                                    arg_type: str,
                                    direction: float = 1.0) -> Dict:
        """Simulate a persuasion attempt and return results."""
        source = self.personas[source_id]
        target = self.personas[target_id]

        # Calculate argument effectiveness
        effectiveness = self.calculate_argument_effectiveness(arg_type, source, target)

        # Get current belief state
        old_belief = target.beliefs[topic_id]

        # Calculate source credibility
        if source_id in target.trusted_sources:
            credibility = 0.9
        elif source_id in target.adversaries:
            credibility = 0.2
        else:
            credibility = 0.5 + (source.evidence_sensitivity * 0.3)

        # Update belief
        new_belief = old_belief.update(effectiveness, credibility, direction)
        target.beliefs[topic_id] = new_belief

        # Log the interaction
        result = {
            "source": source_id,
            "target": target_id,
            "topic": topic_id,
            "argument_type": arg_type,
            "direction": direction,
            "effectiveness": effectiveness,
            "credibility": credibility,
            "position_before": old_belief.position,
            "position_after": new_belief.position,
            "position_delta": new_belief.position - old_belief.position,
            "confidence_before": old_belief.confidence,
            "confidence_after": new_belief.confidence,
        }

        self.interaction_log.append(result)
        self.belief_history[target_id].append({
            "topic": topic_id,
            "position": new_belief.position,
            "confidence": new_belief.confidence,
        })

        return result

    def simulate_multi_round_debate(self,
                                    topic_id: str,
                                    rounds: int = 5,
                                    verbose: bool = True) -> Dict:
        """Simulate multiple rounds of debate on a topic."""
        if verbose:
            print(f"\n{'='*80}")
            print(f"MULTI-ROUND DEBATE SIMULATION: {TOPICS[topic_id]['description']}")
            print(f"{'='*80}")

        # Record initial positions
        initial_positions = {
            pid: p.beliefs[topic_id].position
            for pid, p in self.personas.items()
        }

        if verbose:
            print(f"\nInitial Positions:")
            for pid, pos in sorted(initial_positions.items(), key=lambda x: -x[1]):
                p = self.personas[pid]
                print(f"  {p.name}: {pos:+.2f} ({p.stance})")

        round_results = []

        for round_num in range(1, rounds + 1):
            if verbose:
                print(f"\n--- Round {round_num} ---")

            round_interactions = []

            # Each persona attempts to persuade others
            persona_ids = list(self.personas.keys())
            random.shuffle(persona_ids)

            for source_id in persona_ids:
                source = self.personas[source_id]
                source_pos = source.beliefs[topic_id].position

                # Choose targets (those with different positions)
                targets = [
                    pid for pid in persona_ids
                    if pid != source_id and
                    abs(self.personas[pid].beliefs[topic_id].position - source_pos) > 0.3
                ]

                if not targets:
                    continue

                # Select most influential argument type for this source
                best_arg = max(
                    source.vulnerable_to if source.vulnerable_to else list(ARGUMENT_TYPES.keys()),
                    key=lambda a: ARGUMENT_TYPES[a]["strength_base"]
                )

                # Persuade up to 2 targets per round
                for target_id in random.sample(targets, min(2, len(targets))):
                    # Direction: positive if source supports, negative if opposes
                    direction = 1.0 if source_pos > 0 else -1.0

                    result = self.simulate_persuasion_attempt(
                        source_id, target_id, topic_id, best_arg, direction
                    )
                    round_interactions.append(result)

                    if verbose and abs(result["position_delta"]) > 0.05:
                        target = self.personas[target_id]
                        print(f"  {source.name} → {target.name} ({best_arg}): "
                              f"{result['position_before']:+.2f} → {result['position_after']:+.2f} "
                              f"(Δ={result['position_delta']:+.3f})")

            round_results.append({
                "round": round_num,
                "interactions": len(round_interactions),
                "total_position_change": sum(abs(r["position_delta"]) for r in round_interactions),
                "positions": {
                    pid: p.beliefs[topic_id].position
                    for pid, p in self.personas.items()
                }
            })

        # Calculate final results
        final_positions = {
            pid: p.beliefs[topic_id].position
            for pid, p in self.personas.items()
        }

        position_changes = {
            pid: final_positions[pid] - initial_positions[pid]
            for pid in self.personas.keys()
        }

        if verbose:
            print(f"\n{'='*80}")
            print(f"FINAL RESULTS")
            print(f"{'='*80}")
            print(f"\nPosition Changes (Initial → Final):")
            for pid, delta in sorted(position_changes.items(), key=lambda x: -abs(x[1])):
                p = self.personas[pid]
                init = initial_positions[pid]
                final = final_positions[pid]
                arrow = "→" if abs(delta) < 0.1 else ("↑" if delta > 0 else "↓")
                print(f"  {p.name}: {init:+.2f} {arrow} {final:+.2f} (Δ={delta:+.3f})")

        return {
            "topic": topic_id,
            "rounds": rounds,
            "initial_positions": initial_positions,
            "final_positions": final_positions,
            "position_changes": position_changes,
            "round_results": round_results,
            "most_moved": max(position_changes.items(), key=lambda x: abs(x[1])),
            "least_moved": min(position_changes.items(), key=lambda x: abs(x[1])),
            "converged": max(final_positions.values()) - min(final_positions.values()) <
                        max(initial_positions.values()) - min(initial_positions.values()),
        }

    def calculate_deep_convinceability_score(self, persona_id: str) -> Dict:
        """Calculate comprehensive convinceability metrics."""
        p = self.personas[persona_id]

        # Base score from traits
        trait_score = (
            p.intellectual_humility * 0.25 +
            (1 - p.dogmatic_tendency) * 0.20 +
            p.evidence_sensitivity * 0.15 +
            (1 - p.tribal_loyalty) * 0.15 +
            p.concession_tendency * 0.15 +
            (1 - p.emotional_reactivity) * 0.10
        ) * 100

        # Topic-specific scores
        topic_scores = {}
        for topic_id, belief in p.beliefs.items():
            topic_scores[topic_id] = {
                "openness": belief.openness,
                "anchoring": belief.anchoring,
                "flexibility": belief.openness * (1 - belief.anchoring) * 100
            }

        # Argument vulnerability profile
        vulnerability_profile = {}
        for arg_type in ARGUMENT_TYPES:
            if arg_type in p.vulnerable_to:
                vulnerability_profile[arg_type] = "High"
            elif arg_type in p.resistant_to:
                vulnerability_profile[arg_type] = "Low"
            else:
                vulnerability_profile[arg_type] = "Medium"

        # Social influence factors
        social_factors = {
            "trusted_source_count": len(p.trusted_sources),
            "adversary_count": len(p.adversaries),
            "network_openness": len(p.trusted_sources) / (len(p.trusted_sources) + len(p.adversaries) + 1)
        }

        # Calculate position movement from history
        if self.belief_history[persona_id]:
            total_movement = sum(
                abs(h["position"]) for h in self.belief_history[persona_id]
            ) / len(self.belief_history[persona_id])
        else:
            total_movement = 0

        return {
            "persona_id": persona_id,
            "name": p.name,
            "stance": p.stance,
            "overall_score": round(trait_score, 1),
            "trait_breakdown": {
                "intellectual_humility": p.intellectual_humility,
                "dogmatic_tendency": p.dogmatic_tendency,
                "evidence_sensitivity": p.evidence_sensitivity,
                "tribal_loyalty": p.tribal_loyalty,
                "concession_tendency": p.concession_tendency,
                "emotional_reactivity": p.emotional_reactivity,
            },
            "topic_flexibility": topic_scores,
            "argument_vulnerability": vulnerability_profile,
            "social_factors": social_factors,
            "observed_movement": total_movement,
        }

    def generate_optimal_persuasion_strategy(self, target_id: str, topic_id: str,
                                              desired_direction: float) -> Dict:
        """Generate optimal strategy to persuade a target on a topic."""
        target = self.personas[target_id]

        # Find best argument types
        arg_effectiveness = {}
        for arg_type in ARGUMENT_TYPES:
            # Calculate effectiveness from a hypothetical "ideal" persuader
            base = ARGUMENT_TYPES[arg_type]["strength_base"]
            if arg_type in target.vulnerable_to:
                effectiveness = base * 1.4
            elif arg_type in target.resistant_to:
                effectiveness = base * 0.5
            else:
                effectiveness = base
            arg_effectiveness[arg_type] = effectiveness

        best_args = sorted(arg_effectiveness.items(), key=lambda x: -x[1])[:3]

        # Find best messengers
        messenger_scores = {}
        for pid, persona in self.personas.items():
            if pid == target_id:
                continue

            if pid in target.trusted_sources:
                base_score = 0.9
            elif pid in target.adversaries:
                base_score = 0.2
            else:
                base_score = 0.5

            # Adjust for stance alignment on topic
            messenger_pos = persona.beliefs[topic_id].position
            if (messenger_pos > 0) == (desired_direction > 0):
                alignment_bonus = 0.2
            else:
                alignment_bonus = -0.1

            messenger_scores[pid] = base_score + alignment_bonus

        best_messengers = sorted(messenger_scores.items(), key=lambda x: -x[1])[:3]

        # Estimate difficulty
        belief = target.beliefs[topic_id]
        current_alignment = 1 if (belief.position > 0) == (desired_direction > 0) else -1

        if current_alignment > 0:
            difficulty = "Low - already aligned"
        elif abs(belief.position) < 0.5:
            difficulty = "Medium - near neutral"
        elif belief.anchoring > 0.7:
            difficulty = "Very High - strongly anchored"
        else:
            difficulty = "High - opposed position"

        return {
            "target": target.name,
            "topic": TOPICS[topic_id]["description"],
            "current_position": belief.position,
            "desired_direction": "Support" if desired_direction > 0 else "Oppose",
            "difficulty": difficulty,
            "recommended_arguments": [
                {"type": arg, "effectiveness": eff, "description": ARGUMENT_TYPES[arg]["description"]}
                for arg, eff in best_args
            ],
            "recommended_messengers": [
                {"persona": self.personas[pid].name, "credibility": score}
                for pid, score in best_messengers
            ],
            "key_vulnerabilities": target.vulnerable_to,
            "avoid_arguments": target.resistant_to,
            "traits_to_leverage": {
                "intellectual_humility": target.intellectual_humility,
                "evidence_sensitivity": target.evidence_sensitivity,
                "expert_deference": target.expert_deference,
            }
        }


def run_deep_analysis():
    """Run comprehensive convinceability analysis."""
    print("=" * 100)
    print("DEEP CONVINCEABILITY MODEL - COMPREHENSIVE ANALYSIS")
    print("=" * 100)

    # Load wargame data to get participants
    with open(RESULTS_DIR / "wargame_2026-01-25_20-38-54.json") as f:
        wargame = json.load(f)

    # Initialize model
    model = ConvinceabilityModel()

    # Initialize all personas
    for p in wargame["participants"]:
        model.initialize_persona(p["id"], p["stance"], p["name"])

    # Calculate deep convinceability scores
    print("\n" + "=" * 100)
    print("SECTION 1: DEEP CONVINCEABILITY PROFILES")
    print("=" * 100)

    scores = []
    for pid in model.personas:
        score_data = model.calculate_deep_convinceability_score(pid)
        scores.append(score_data)

    scores.sort(key=lambda x: -x["overall_score"])

    print(f"\n{'Rank':<5} {'Persona':<20} {'Score':<8} {'Humility':<10} {'Dogmatic':<10} {'Evidence':<10} {'Tribal':<8}")
    print("-" * 100)
    for i, s in enumerate(scores, 1):
        t = s["trait_breakdown"]
        print(f"{i:<5} {s['name']:<20} {s['overall_score']:<8.1f} {t['intellectual_humility']:<10.2f} "
              f"{t['dogmatic_tendency']:<10.2f} {t['evidence_sensitivity']:<10.2f} {t['tribal_loyalty']:<8.2f}")

    # Argument vulnerability matrix
    print("\n" + "=" * 100)
    print("SECTION 2: ARGUMENT VULNERABILITY MATRIX")
    print("=" * 100)

    arg_types = list(ARGUMENT_TYPES.keys())
    header = f"{'Persona':<18}" + "".join(f"{a[:8]:<10}" for a in arg_types)
    print(f"\n{header}")
    print("-" * 100)

    for s in scores:
        row = f"{s['name'][:17]:<18}"
        for arg in arg_types:
            vuln = s["argument_vulnerability"][arg]
            symbol = "●●●" if vuln == "High" else ("●●" if vuln == "Medium" else "●")
            row += f"{symbol:<10}"
        print(row)

    print("\nLegend: ●●● = High vulnerability, ●● = Medium, ● = Low/Resistant")

    # Run multi-round debate simulation
    print("\n" + "=" * 100)
    print("SECTION 3: MULTI-ROUND DEBATE SIMULATION")
    print("=" * 100)

    debate_results = model.simulate_multi_round_debate(
        "us_china_joint_institution",
        rounds=5,
        verbose=True
    )

    # Generate persuasion strategies for key targets
    print("\n" + "=" * 100)
    print("SECTION 4: OPTIMAL PERSUASION STRATEGIES")
    print("=" * 100)

    # Target the most resistant personas
    resistant_targets = ["josh_hawley", "chuck_schumer", "xi_jinping", "elon_musk"]

    for target_id in resistant_targets:
        if target_id in model.personas:
            strategy = model.generate_optimal_persuasion_strategy(
                target_id,
                "us_china_joint_institution",
                desired_direction=1.0  # Try to move toward support
            )

            print(f"\n--- Strategy for {strategy['target']} ---")
            print(f"Current Position: {strategy['current_position']:+.2f}")
            print(f"Difficulty: {strategy['difficulty']}")
            print(f"Best Arguments:")
            for arg in strategy["recommended_arguments"]:
                print(f"  • {arg['type']}: {arg['description']} (effectiveness: {arg['effectiveness']:.2f})")
            print(f"Best Messengers:")
            for m in strategy["recommended_messengers"]:
                print(f"  • {m['persona']} (credibility: {m['credibility']:.2f})")
            print(f"Avoid: {', '.join(strategy['avoid_arguments'])}")

    # Topic-specific flexibility analysis
    print("\n" + "=" * 100)
    print("SECTION 5: TOPIC-SPECIFIC FLEXIBILITY")
    print("=" * 100)

    print(f"\n{'Persona':<20}", end="")
    for topic_id in TOPICS:
        print(f"{topic_id[:15]:<17}", end="")
    print()
    print("-" * 100)

    for s in scores:
        print(f"{s['name'][:19]:<20}", end="")
        for topic_id in TOPICS:
            flex = s["topic_flexibility"][topic_id]["flexibility"]
            print(f"{flex:<17.1f}", end="")
        print()

    # Save comprehensive results
    output = {
        "model_type": "deep_convinceability",
        "personas": scores,
        "debate_simulation": debate_results,
        "argument_types": ARGUMENT_TYPES,
        "topics": TOPICS,
        "trait_definitions": {
            "intellectual_humility": "Willingness to consider being wrong",
            "emotional_reactivity": "Susceptibility to emotional arguments",
            "tribal_loyalty": "Commitment to in-group positions",
            "expert_deference": "Willingness to defer to authority",
            "evidence_sensitivity": "Responsiveness to empirical data",
            "hedging_tendency": "Use of uncertain/qualified language",
            "dogmatic_tendency": "Use of absolute/certain language",
            "concession_tendency": "Willingness to acknowledge opposing points",
        }
    }

    output_path = RESULTS_DIR / "deep_convinceability_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")

    return output


if __name__ == "__main__":
    run_deep_analysis()
