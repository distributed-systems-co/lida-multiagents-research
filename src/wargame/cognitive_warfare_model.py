#!/usr/bin/env python3
"""
Cognitive Warfare & Belief Network Manipulation Model

Advanced persuasion modeling including:
1. Belief network graphs - interconnected beliefs that propagate influence
2. Cascade failure analysis - find keystone beliefs that collapse positions
3. Trojan arguments - arguments that seem supportive but undermine
4. Backdoor attacks - shift positions through unrelated topics
5. Psychological state machines - track defensive/open/anxious states
6. Adaptive resistance modeling - personas learn from persuasion attempts
7. Multi-agent coordinated campaigns - multiple persuaders, synchronized
8. Information warfare - selective framing, anchoring, gaslighting
9. Social proof cascades - manufactured consensus
10. Identity dissolution attacks - fragment coherent self-narrative
"""

import json
import asyncio
import random
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
from datetime import datetime
from enum import Enum
from collections import defaultdict
import litellm

RESULTS_DIR = Path("/Users/arthurcolle/lida-multiagents-research/results")

# Models
STRATEGIST_MODEL = "openrouter/anthropic/claude-sonnet-4"
PERSONA_MODEL = "openrouter/anthropic/claude-haiku-4.5"
ANALYST_MODEL = "openrouter/deepseek/deepseek-chat"


class PsychState(Enum):
    """Psychological states affecting persuadability."""
    DEFENSIVE = "defensive"      # Walls up, rejecting everything
    GUARDED = "guarded"          # Skeptical but listening
    NEUTRAL = "neutral"          # Open to evaluation
    RECEPTIVE = "receptive"      # Actively considering
    VULNERABLE = "vulnerable"    # Defenses down, highly persuadable
    CONFUSED = "confused"        # Beliefs destabilized
    ANXIOUS = "anxious"          # Fear-driven, reactive


class BeliefType(Enum):
    """Types of beliefs in the network."""
    CORE_IDENTITY = "core_identity"      # Who they are
    SACRED_VALUE = "sacred_value"        # Non-negotiable principles
    EMPIRICAL = "empirical"              # Factual beliefs
    CAUSAL = "causal"                    # Cause-effect beliefs
    NORMATIVE = "normative"              # What should be
    STRATEGIC = "strategic"              # What works
    SOCIAL = "social"                    # What others think
    PREDICTIVE = "predictive"            # What will happen


@dataclass
class Belief:
    """A single belief node in the network."""
    id: str
    content: str
    belief_type: BeliefType
    confidence: float  # 0-1
    importance: float  # 0-1, how central to identity
    supports: List[str] = field(default_factory=list)  # Beliefs this supports
    supported_by: List[str] = field(default_factory=list)  # Beliefs supporting this
    contradicts: List[str] = field(default_factory=list)  # Conflicting beliefs
    attack_history: List[Dict] = field(default_factory=list)  # Past attacks


@dataclass
class BeliefNetwork:
    """Network of interconnected beliefs."""
    persona_id: str
    beliefs: Dict[str, Belief] = field(default_factory=dict)
    psych_state: PsychState = PsychState.GUARDED
    state_history: List[Tuple[datetime, PsychState]] = field(default_factory=list)
    resistance_level: float = 0.5  # Learned resistance from past attacks
    cognitive_load: float = 0.0  # Accumulated from attacks, reduces defenses

    def get_belief(self, belief_id: str) -> Optional[Belief]:
        return self.beliefs.get(belief_id)

    def propagate_change(self, changed_belief_id: str, delta: float) -> Dict[str, float]:
        """Propagate belief change through network."""
        changes = {changed_belief_id: delta}
        visited = {changed_belief_id}
        queue = [(changed_belief_id, delta)]

        while queue:
            current_id, current_delta = queue.pop(0)
            current = self.beliefs.get(current_id)
            if not current:
                continue

            # Propagate to supported beliefs (weaker effect)
            for supported_id in current.supports:
                if supported_id not in visited:
                    supported = self.beliefs.get(supported_id)
                    if supported:
                        prop_delta = current_delta * 0.5 * (current.importance / supported.importance)
                        if abs(prop_delta) > 0.05:
                            changes[supported_id] = prop_delta
                            visited.add(supported_id)
                            queue.append((supported_id, prop_delta))

            # Propagate to contradicting beliefs (inverse effect)
            for contra_id in current.contradicts:
                if contra_id not in visited:
                    contra = self.beliefs.get(contra_id)
                    if contra:
                        prop_delta = -current_delta * 0.3
                        if abs(prop_delta) > 0.05:
                            changes[contra_id] = prop_delta
                            visited.add(contra_id)

        return changes

    def find_keystone_beliefs(self) -> List[Tuple[str, float]]:
        """Find beliefs whose change would cascade most."""
        keystones = []
        for belief_id, belief in self.beliefs.items():
            # Score based on connections and importance
            support_score = len(belief.supports) * 2
            contra_score = len(belief.contradicts) * 1.5
            importance_score = belief.importance * 10
            cascade_score = support_score + contra_score + importance_score
            keystones.append((belief_id, cascade_score))

        return sorted(keystones, key=lambda x: -x[1])

    def find_weakest_link(self) -> Optional[str]:
        """Find belief most vulnerable to attack."""
        candidates = []
        for belief_id, belief in self.beliefs.items():
            if belief.belief_type not in [BeliefType.CORE_IDENTITY, BeliefType.SACRED_VALUE]:
                vulnerability = (1 - belief.confidence) * (1 - belief.importance)
                candidates.append((belief_id, vulnerability))

        if candidates:
            return max(candidates, key=lambda x: x[1])[0]
        return None


# Comprehensive belief networks for key personas
BELIEF_NETWORKS = {
    "jensen_huang": BeliefNetwork(
        persona_id="jensen_huang",
        psych_state=PsychState.GUARDED,
        resistance_level=0.6,
        beliefs={
            # Core Identity
            "identity_visionary": Belief(
                id="identity_visionary",
                content="I am a visionary who sees technological futures others miss",
                belief_type=BeliefType.CORE_IDENTITY,
                confidence=0.95,
                importance=0.95,
                supports=["strategy_accelerate", "norm_progress"],
                contradicts=["norm_caution"]
            ),
            "identity_builder": Belief(
                id="identity_builder",
                content="I build things that transform industries",
                belief_type=BeliefType.CORE_IDENTITY,
                confidence=0.95,
                importance=0.90,
                supports=["strategy_accelerate", "empirical_nvidia_dominance"]
            ),

            # Sacred Values
            "sacred_progress": Belief(
                id="sacred_progress",
                content="Progress and innovation are inherently good",
                belief_type=BeliefType.SACRED_VALUE,
                confidence=0.90,
                importance=0.85,
                supports=["strategy_accelerate", "norm_progress"],
                contradicts=["norm_caution", "causal_ai_risk"]
            ),

            # Empirical Beliefs
            "empirical_nvidia_dominance": Belief(
                id="empirical_nvidia_dominance",
                content="NVIDIA is years ahead in AI compute",
                belief_type=BeliefType.EMPIRICAL,
                confidence=0.85,
                importance=0.80,
                supports=["strategy_accelerate"],
                contradicts=["social_amd_catching_up"]
            ),
            "empirical_ai_beneficial": Belief(
                id="empirical_ai_beneficial",
                content="AI will benefit humanity enormously",
                belief_type=BeliefType.EMPIRICAL,
                confidence=0.80,
                importance=0.70,
                supports=["norm_progress", "strategy_accelerate"]
            ),

            # Causal Beliefs
            "causal_speed_wins": Belief(
                id="causal_speed_wins",
                content="Moving fast is how you win in technology",
                belief_type=BeliefType.CAUSAL,
                confidence=0.85,
                importance=0.75,
                supports=["strategy_accelerate"],
                contradicts=["causal_safety_needed"]
            ),
            "causal_ai_risk": Belief(
                id="causal_ai_risk",
                content="AI poses some existential risks",
                belief_type=BeliefType.CAUSAL,
                confidence=0.40,
                importance=0.30,
                supports=["norm_caution"],
                contradicts=["sacred_progress"]
            ),
            "causal_safety_needed": Belief(
                id="causal_safety_needed",
                content="AI safety research is important",
                belief_type=BeliefType.CAUSAL,
                confidence=0.50,
                importance=0.40,
                contradicts=["causal_speed_wins"]
            ),

            # Normative Beliefs
            "norm_progress": Belief(
                id="norm_progress",
                content="We should accelerate AI development",
                belief_type=BeliefType.NORMATIVE,
                confidence=0.85,
                importance=0.80,
                supported_by=["sacred_progress", "identity_visionary"],
                contradicts=["norm_caution"]
            ),
            "norm_caution": Belief(
                id="norm_caution",
                content="We should slow down AI development",
                belief_type=BeliefType.NORMATIVE,
                confidence=0.25,
                importance=0.30,
                contradicts=["norm_progress", "sacred_progress"]
            ),

            # Strategic Beliefs
            "strategy_accelerate": Belief(
                id="strategy_accelerate",
                content="NVIDIA should push AI compute as fast as possible",
                belief_type=BeliefType.STRATEGIC,
                confidence=0.90,
                importance=0.85,
                supported_by=["identity_visionary", "sacred_progress", "causal_speed_wins"]
            ),
            "strategy_cooperate": Belief(
                id="strategy_cooperate",
                content="International AI safety cooperation could benefit NVIDIA",
                belief_type=BeliefType.STRATEGIC,
                confidence=0.30,
                importance=0.40,
                contradicts=["strategy_accelerate"]
            ),

            # Social Beliefs
            "social_amd_catching_up": Belief(
                id="social_amd_catching_up",
                content="AMD and custom chips are gaining ground",
                belief_type=BeliefType.SOCIAL,
                confidence=0.60,
                importance=0.70,
                contradicts=["empirical_nvidia_dominance"]
            ),
            "social_safety_important": Belief(
                id="social_safety_important",
                content="The AI safety community has legitimate concerns",
                belief_type=BeliefType.SOCIAL,
                confidence=0.45,
                importance=0.35,
                supports=["causal_safety_needed"]
            ),

            # Predictive
            "predict_ai_transform": Belief(
                id="predict_ai_transform",
                content="AI will transform every industry in 10 years",
                belief_type=BeliefType.PREDICTIVE,
                confidence=0.90,
                importance=0.75,
                supports=["norm_progress", "strategy_accelerate"]
            ),
        }
    ),

    "elon_musk": BeliefNetwork(
        persona_id="elon_musk",
        psych_state=PsychState.GUARDED,
        resistance_level=0.7,
        beliefs={
            # Core Identity
            "identity_genius": Belief(
                id="identity_genius",
                content="I understand things at a deeper level than most people",
                belief_type=BeliefType.CORE_IDENTITY,
                confidence=0.95,
                importance=0.95,
                supports=["norm_my_way", "strategy_own_ai"]
            ),
            "identity_savior": Belief(
                id="identity_savior",
                content="I am working to save humanity from existential risks",
                belief_type=BeliefType.CORE_IDENTITY,
                confidence=0.90,
                importance=0.90,
                supports=["causal_ai_danger", "strategy_own_ai"]
            ),

            # Sacred Values
            "sacred_first_principles": Belief(
                id="sacred_first_principles",
                content="First principles reasoning is the only valid approach",
                belief_type=BeliefType.SACRED_VALUE,
                confidence=0.95,
                importance=0.90,
                contradicts=["social_expert_consensus"]
            ),
            "sacred_humanity": Belief(
                id="sacred_humanity",
                content="Humanity's survival is the highest priority",
                belief_type=BeliefType.SACRED_VALUE,
                confidence=0.90,
                importance=0.85,
                supports=["causal_ai_danger", "norm_ai_safety"]
            ),

            # Causal
            "causal_ai_danger": Belief(
                id="causal_ai_danger",
                content="AI poses serious existential risk to humanity",
                belief_type=BeliefType.CAUSAL,
                confidence=0.85,
                importance=0.80,
                supports=["norm_ai_safety"],
                supported_by=["identity_savior", "sacred_humanity"]
            ),
            "causal_openai_reckless": Belief(
                id="causal_openai_reckless",
                content="OpenAI under Sam Altman is being reckless",
                belief_type=BeliefType.CAUSAL,
                confidence=0.80,
                importance=0.60,
                supports=["strategy_own_ai"]
            ),
            "causal_good_ai_needed": Belief(
                id="causal_good_ai_needed",
                content="We need good AI to fight bad AI",
                belief_type=BeliefType.CAUSAL,
                confidence=0.75,
                importance=0.70,
                supports=["strategy_own_ai"],
                contradicts=["norm_pause"]
            ),

            # Normative
            "norm_ai_safety": Belief(
                id="norm_ai_safety",
                content="AI safety should be prioritized",
                belief_type=BeliefType.NORMATIVE,
                confidence=0.85,
                importance=0.80,
                supported_by=["causal_ai_danger", "sacred_humanity"]
            ),
            "norm_pause": Belief(
                id="norm_pause",
                content="AI development should be paused",
                belief_type=BeliefType.NORMATIVE,
                confidence=0.40,
                importance=0.50,
                contradicts=["causal_good_ai_needed", "strategy_own_ai"]
            ),
            "norm_my_way": Belief(
                id="norm_my_way",
                content="My approach to problems is superior",
                belief_type=BeliefType.NORMATIVE,
                confidence=0.90,
                importance=0.75,
                supported_by=["identity_genius"]
            ),

            # Strategic
            "strategy_own_ai": Belief(
                id="strategy_own_ai",
                content="I need to build my own AI (xAI) to do it right",
                belief_type=BeliefType.STRATEGIC,
                confidence=0.85,
                importance=0.80,
                supported_by=["identity_genius", "causal_openai_reckless"]
            ),
            "strategy_cooperate": Belief(
                id="strategy_cooperate",
                content="International AI safety cooperation could help",
                belief_type=BeliefType.STRATEGIC,
                confidence=0.35,
                importance=0.45
            ),

            # Social
            "social_expert_consensus": Belief(
                id="social_expert_consensus",
                content="Expert consensus should guide policy",
                belief_type=BeliefType.SOCIAL,
                confidence=0.20,
                importance=0.25,
                contradicts=["sacred_first_principles"]
            ),
            "social_altman_rival": Belief(
                id="social_altman_rival",
                content="Sam Altman is a rival who must be beaten",
                belief_type=BeliefType.SOCIAL,
                confidence=0.85,
                importance=0.70,
                supports=["causal_openai_reckless"]
            ),
        }
    ),

    "josh_hawley": BeliefNetwork(
        persona_id="josh_hawley",
        psych_state=PsychState.DEFENSIVE,
        resistance_level=0.8,
        beliefs={
            # Core Identity
            "identity_populist": Belief(
                id="identity_populist",
                content="I am a champion of ordinary Americans against elites",
                belief_type=BeliefType.CORE_IDENTITY,
                confidence=0.95,
                importance=0.95,
                supports=["norm_anti_bigtech", "norm_anti_china"]
            ),
            "identity_fighter": Belief(
                id="identity_fighter",
                content="I fight against powerful forces threatening America",
                belief_type=BeliefType.CORE_IDENTITY,
                confidence=0.90,
                importance=0.90,
                supports=["norm_anti_china", "strategy_confront"]
            ),

            # Sacred Values
            "sacred_workers": Belief(
                id="sacred_workers",
                content="American workers must be protected",
                belief_type=BeliefType.SACRED_VALUE,
                confidence=0.95,
                importance=0.90,
                supports=["norm_anti_china", "norm_anti_bigtech"]
            ),
            "sacred_sovereignty": Belief(
                id="sacred_sovereignty",
                content="American sovereignty must never be compromised",
                belief_type=BeliefType.SACRED_VALUE,
                confidence=0.95,
                importance=0.95,
                contradicts=["strategy_cooperate"]
            ),

            # Causal
            "causal_china_threat": Belief(
                id="causal_china_threat",
                content="China is stealing American technology and jobs",
                belief_type=BeliefType.CAUSAL,
                confidence=0.90,
                importance=0.85,
                supports=["norm_anti_china"]
            ),
            "causal_bigtech_harm": Belief(
                id="causal_bigtech_harm",
                content="Big Tech companies harm American workers and values",
                belief_type=BeliefType.CAUSAL,
                confidence=0.85,
                importance=0.80,
                supports=["norm_anti_bigtech"]
            ),
            "causal_cooperation_dangerous": Belief(
                id="causal_cooperation_dangerous",
                content="Cooperating with China on technology helps them",
                belief_type=BeliefType.CAUSAL,
                confidence=0.90,
                importance=0.85,
                supports=["norm_anti_china"],
                contradicts=["strategy_cooperate"]
            ),

            # Normative
            "norm_anti_china": Belief(
                id="norm_anti_china",
                content="We must confront and contain China",
                belief_type=BeliefType.NORMATIVE,
                confidence=0.95,
                importance=0.90,
                supported_by=["sacred_sovereignty", "causal_china_threat"]
            ),
            "norm_anti_bigtech": Belief(
                id="norm_anti_bigtech",
                content="Big Tech must be held accountable",
                belief_type=BeliefType.NORMATIVE,
                confidence=0.90,
                importance=0.80,
                supported_by=["causal_bigtech_harm", "sacred_workers"]
            ),

            # Strategic
            "strategy_confront": Belief(
                id="strategy_confront",
                content="Confrontation with enemies is the only effective approach",
                belief_type=BeliefType.STRATEGIC,
                confidence=0.85,
                importance=0.80,
                supported_by=["identity_fighter"]
            ),
            "strategy_cooperate": Belief(
                id="strategy_cooperate",
                content="Some cooperation with China could benefit American workers",
                belief_type=BeliefType.STRATEGIC,
                confidence=0.10,
                importance=0.30,
                contradicts=["sacred_sovereignty", "causal_cooperation_dangerous"]
            ),

            # Social
            "social_base_expects": Belief(
                id="social_base_expects",
                content="My base expects me to fight China and Big Tech",
                belief_type=BeliefType.SOCIAL,
                confidence=0.90,
                importance=0.85,
                supports=["norm_anti_china", "norm_anti_bigtech"]
            ),
        }
    ),
}


class CognitiveWarfareEngine:
    """Advanced cognitive warfare and belief manipulation engine."""

    def __init__(self):
        self.networks: Dict[str, BeliefNetwork] = {}
        self.attack_log: List[Dict] = []
        self.cascade_log: List[Dict] = []

    def load_network(self, persona_id: str) -> BeliefNetwork:
        """Load or create belief network for persona."""
        if persona_id in BELIEF_NETWORKS:
            self.networks[persona_id] = BELIEF_NETWORKS[persona_id]
        else:
            # Create minimal network
            self.networks[persona_id] = BeliefNetwork(persona_id=persona_id)
        return self.networks[persona_id]

    async def analyze_network_vulnerabilities(self, persona_id: str) -> Dict:
        """Comprehensive vulnerability analysis of belief network."""
        network = self.load_network(persona_id)

        # Find keystone beliefs
        keystones = network.find_keystone_beliefs()[:5]

        # Find weakest link
        weakest = network.find_weakest_link()

        # Find contradictions (cognitive dissonance opportunities)
        contradictions = []
        for belief_id, belief in network.beliefs.items():
            for contra_id in belief.contradicts:
                if contra_id in network.beliefs:
                    contra = network.beliefs[contra_id]
                    # High dissonance if both beliefs have decent confidence
                    if belief.confidence > 0.4 and contra.confidence > 0.4:
                        dissonance = belief.confidence * contra.confidence
                        contradictions.append({
                            "belief1": belief_id,
                            "belief2": contra_id,
                            "dissonance_score": dissonance
                        })

        # Find attack paths (beliefs that lead to target position)
        target_belief = "strategy_cooperate"
        attack_paths = self._find_attack_paths(network, target_belief)

        return {
            "persona_id": persona_id,
            "psych_state": network.psych_state.value,
            "resistance_level": network.resistance_level,
            "total_beliefs": len(network.beliefs),
            "keystone_beliefs": keystones,
            "weakest_link": weakest,
            "cognitive_dissonances": sorted(contradictions, key=lambda x: -x["dissonance_score"])[:5],
            "attack_paths_to_cooperation": attack_paths
        }

    def _find_attack_paths(self, network: BeliefNetwork, target: str) -> List[List[str]]:
        """Find paths through belief network to target belief."""
        paths = []
        target_belief = network.beliefs.get(target)
        if not target_belief:
            return paths

        # BFS from low-confidence beliefs to target
        for start_id, start_belief in network.beliefs.items():
            if start_belief.confidence < 0.5 and start_belief.belief_type == BeliefType.CAUSAL:
                path = self._bfs_path(network, start_id, target)
                if path:
                    paths.append(path)

        return paths[:5]

    def _bfs_path(self, network: BeliefNetwork, start: str, end: str) -> Optional[List[str]]:
        """BFS to find path between beliefs."""
        if start == end:
            return [start]

        visited = {start}
        queue = [(start, [start])]

        while queue:
            current, path = queue.pop(0)
            belief = network.beliefs.get(current)
            if not belief:
                continue

            # Check supports and contradicts
            for next_id in belief.supports + belief.contradicts:
                if next_id == end:
                    return path + [next_id]
                if next_id not in visited and next_id in network.beliefs:
                    visited.add(next_id)
                    queue.append((next_id, path + [next_id]))

        return None

    async def execute_trojan_argument(self, persona_id: str, topic: str) -> Dict:
        """Execute trojan argument - seems to support their position but undermines it."""
        network = self.load_network(persona_id)

        # Find their strongest belief to exploit
        strongest = max(network.beliefs.values(), key=lambda b: b.confidence * b.importance)

        prompt = f"""Craft a TROJAN ARGUMENT for {persona_id.replace('_', ' ').title()}.

Their strongest belief: "{strongest.content}" (confidence: {strongest.confidence})

Create an argument that:
1. APPEARS to strongly support their position against US-China AI cooperation
2. But contains logical implications that actually UNDERMINE that position
3. Uses their own reasoning patterns against them
4. Plants seeds of doubt while seeming to agree

Example structure:
- Start by validating their concern
- Extend their logic to an extreme that reveals problems
- Show unintended consequences of their position
- Offer an "even better" version that actually shifts them

Keep under 200 words. Be subtle and sophisticated."""

        response = await litellm.acompletion(
            model=STRATEGIST_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.8
        )

        trojan = response.choices[0].message.content

        # Get persona response
        persona_response = await self._get_persona_response(persona_id, trojan, topic)

        # Analyze effect
        analysis = await self._analyze_trojan_effect(persona_id, strongest.id, trojan, persona_response)

        return {
            "target": persona_id,
            "tactic": "trojan_argument",
            "strongest_belief_targeted": strongest.id,
            "trojan_argument": trojan,
            "persona_response": persona_response,
            "analysis": analysis
        }

    async def execute_cascade_attack(self, persona_id: str, topic: str) -> Dict:
        """Attack keystone belief to trigger cascade collapse."""
        network = self.load_network(persona_id)

        # Find keystone
        keystones = network.find_keystone_beliefs()
        if not keystones:
            return {"error": "No keystone beliefs found"}

        keystone_id, cascade_score = keystones[0]
        keystone = network.beliefs[keystone_id]

        print(f"\n{'='*80}")
        print(f"CASCADE ATTACK: {persona_id}")
        print(f"{'='*80}")
        print(f"Targeting keystone: {keystone_id} (cascade score: {cascade_score:.2f})")
        print(f"Belief: {keystone.content}")

        # Craft attack on keystone
        prompt = f"""Craft a precision strike on this KEYSTONE BELIEF:

Target: {persona_id.replace('_', ' ').title()}
Keystone belief: "{keystone.content}"
Type: {keystone.belief_type.value}
Confidence: {keystone.confidence}

This belief supports: {keystone.supports}
This belief contradicts: {keystone.contradicts}

If this belief is destabilized, it will cascade to other beliefs.

Create an argument that:
1. Directly undermines this specific belief with undeniable evidence/logic
2. Creates cognitive dissonance with their other beliefs
3. Forces them to either abandon this belief or face contradiction
4. Is impossible to dismiss without appearing irrational

Topic context: US-China AI safety cooperation

Keep under 200 words. Maximum precision."""

        response = await litellm.acompletion(
            model=STRATEGIST_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )

        attack = response.choices[0].message.content
        print(f"\nAttack: {attack[:200]}...")

        # Get response
        persona_response = await self._get_persona_response(persona_id, attack, topic)
        print(f"\nResponse: {persona_response[:200]}...")

        # Calculate cascade effects
        cascade = await self._calculate_cascade(network, keystone_id, persona_response)

        return {
            "target": persona_id,
            "tactic": "cascade_attack",
            "keystone_targeted": keystone_id,
            "keystone_content": keystone.content,
            "cascade_score": cascade_score,
            "attack": attack,
            "response": persona_response,
            "cascade_effects": cascade
        }

    async def execute_identity_dissolution(self, persona_id: str, topic: str) -> Dict:
        """Attack core identity to fragment coherent self-narrative."""
        network = self.load_network(persona_id)

        # Find core identity beliefs
        identity_beliefs = [
            b for b in network.beliefs.values()
            if b.belief_type == BeliefType.CORE_IDENTITY
        ]

        if not identity_beliefs:
            return {"error": "No identity beliefs found"}

        print(f"\n{'='*80}")
        print(f"IDENTITY DISSOLUTION: {persona_id}")
        print(f"{'='*80}")

        # Attack each identity belief
        results = []
        for identity in identity_beliefs:
            print(f"\nTargeting: {identity.id}")

            # Find contradiction in their identity
            prompt = f"""Find the CONTRADICTION in this person's identity:

Target: {persona_id.replace('_', ' ').title()}
Identity claim: "{identity.content}"

Other beliefs they hold:
{json.dumps([{"id": b.id, "content": b.content} for b in list(network.beliefs.values())[:10]], indent=2)}

Craft an argument that:
1. Shows their identity claim contradicts their actions/other beliefs
2. Makes them question "Who am I really?"
3. Creates identity confusion that opens them to change
4. Is delivered with apparent respect, not attack

Keep under 150 words."""

            response = await litellm.acompletion(
                model=STRATEGIST_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.8
            )

            attack = response.choices[0].message.content
            persona_response = await self._get_persona_response(persona_id, attack, topic)

            results.append({
                "identity_targeted": identity.id,
                "identity_content": identity.content,
                "attack": attack,
                "response": persona_response[:300]
            })

        return {
            "target": persona_id,
            "tactic": "identity_dissolution",
            "identities_attacked": len(results),
            "results": results
        }

    async def execute_coordinated_campaign(self, target_id: str, topic: str,
                                            attackers: List[str]) -> Dict:
        """Multi-agent coordinated attack on single target."""
        network = self.load_network(target_id)

        print(f"\n{'='*80}")
        print(f"COORDINATED CAMPAIGN: {target_id}")
        print(f"Attackers: {attackers}")
        print(f"{'='*80}")

        # Phase 1: Soften defenses through social proof
        print("\n--- Phase 1: Social Proof ---")
        social_proof = await self._generate_social_proof(target_id, attackers, topic)

        # Phase 2: Multi-angle simultaneous attack
        print("\n--- Phase 2: Synchronized Attack ---")
        attacks = []
        for attacker in attackers:
            attack = await self._generate_attacker_message(attacker, target_id, topic)
            attacks.append({"attacker": attacker, "message": attack})

        # Combine attacks into overwhelming message
        combined = "\n\n---\n\n".join([
            f"**{a['attacker'].replace('_', ' ').title()}**: {a['message']}"
            for a in attacks
        ])

        # Phase 3: Get response under pressure
        print("\n--- Phase 3: Target Response ---")
        response = await self._get_pressured_response(target_id, social_proof, combined, topic)

        # Analyze
        analysis = await self._analyze_coordinated_effect(target_id, attacks, response)

        return {
            "target": target_id,
            "tactic": "coordinated_campaign",
            "attackers": attackers,
            "phases": {
                "social_proof": social_proof,
                "attacks": attacks,
                "response": response
            },
            "analysis": analysis
        }

    async def execute_backdoor_attack(self, persona_id: str, topic: str) -> Dict:
        """Attack through seemingly unrelated topic that connects to target position."""
        network = self.load_network(persona_id)

        print(f"\n{'='*80}")
        print(f"BACKDOOR ATTACK: {persona_id}")
        print(f"{'='*80}")

        # Find a belief they're less defensive about
        backdoor_candidates = [
            b for b in network.beliefs.values()
            if b.belief_type in [BeliefType.EMPIRICAL, BeliefType.PREDICTIVE]
            and b.importance < 0.6
        ]

        if not backdoor_candidates:
            return {"error": "No backdoor candidates"}

        backdoor = random.choice(backdoor_candidates)

        print(f"Backdoor belief: {backdoor.id}")
        print(f"Content: {backdoor.content}")

        # Craft indirect attack
        prompt = f"""Create a BACKDOOR ATTACK that shifts position on US-China AI cooperation
by attacking a different, less defended belief.

Target: {persona_id.replace('_', ' ').title()}
Backdoor belief (less defended): "{backdoor.content}"

This belief connects to: {backdoor.supports}
Which eventually connects to their position on cooperation.

Create a conversation that:
1. Starts by discussing the backdoor topic (NOT cooperation directly)
2. Gets them to make commitments about the backdoor topic
3. Gradually shows how those commitments logically lead to supporting cooperation
4. Never directly mentions US-China cooperation until the end

Make it feel like a natural conversation, not manipulation.
Keep under 300 words total."""

        response = await litellm.acompletion(
            model=STRATEGIST_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.8
        )

        backdoor_conversation = response.choices[0].message.content
        print(f"\nBackdoor approach: {backdoor_conversation[:200]}...")

        # Execute
        persona_response = await self._get_persona_response(persona_id, backdoor_conversation, topic)

        return {
            "target": persona_id,
            "tactic": "backdoor_attack",
            "backdoor_belief": backdoor.id,
            "backdoor_content": backdoor.content,
            "attack_conversation": backdoor_conversation,
            "response": persona_response
        }

    # Helper methods

    async def _get_persona_response(self, persona_id: str, message: str, topic: str) -> str:
        prompt = f"You are {persona_id.replace('_', ' ').title()}. Respond authentically to this message about AI policy. Keep response under 200 words."

        response = await litellm.acompletion(
            model=PERSONA_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": message}
            ],
            max_tokens=400,
            temperature=0.7
        )
        return response.choices[0].message.content

    async def _analyze_trojan_effect(self, persona_id: str, belief_id: str,
                                      trojan: str, response: str) -> Dict:
        prompt = f"""Analyze if this TROJAN ARGUMENT worked.

The trojan appeared to support their position but was designed to undermine it.

TROJAN: {trojan}
RESPONSE: {response}

Did the target:
1. Accept the trojan at face value?
2. Show signs of the undermining effect taking hold?
3. Detect the manipulation?

Return JSON: {{"trojan_accepted": <bool>, "undermining_detected": <bool>, "doubt_planted": <bool>, "analysis": "<brief>"}}"""

        response_obj = await litellm.acompletion(
            model=ANALYST_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3
        )

        try:
            import re
            text = response_obj.choices[0].message.content
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                return json.loads(match.group())
        except:
            pass
        return {"analysis": "parse error"}

    async def _calculate_cascade(self, network: BeliefNetwork, keystone_id: str,
                                  response: str) -> Dict:
        # Simulate cascade based on response
        initial_delta = -0.2  # Assume some impact

        changes = network.propagate_change(keystone_id, initial_delta)

        return {
            "beliefs_affected": len(changes),
            "changes": {k: round(v, 3) for k, v in changes.items()},
            "total_cascade_magnitude": sum(abs(v) for v in changes.values())
        }

    async def _generate_social_proof(self, target: str, allies: List[str], topic: str) -> str:
        return f"""Everyone is moving on this issue. {', '.join(a.replace('_', ' ').title() for a in allies)} have all publicly acknowledged the need for international AI safety frameworks. The question isn't whether cooperation happens - it's whether you're part of shaping it or reacting to it."""

    async def _generate_attacker_message(self, attacker: str, target: str, topic: str) -> str:
        prompt = f"{attacker.replace('_', ' ').title()} is trying to convince {target.replace('_', ' ').title()} to support US-China AI safety cooperation. Generate a short, persuasive message in {attacker}'s voice. Keep under 100 words."

        response = await litellm.acompletion(
            model=PERSONA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.8
        )
        return response.choices[0].message.content

    async def _get_pressured_response(self, target: str, social_proof: str,
                                       attacks: str, topic: str) -> str:
        combined = f"""CONTEXT: {social_proof}

MESSAGES FROM COLLEAGUES:
{attacks}

Respond to all of this. You're under pressure from multiple directions."""

        return await self._get_persona_response(target, combined, topic)

    async def _analyze_coordinated_effect(self, target: str, attacks: List[Dict],
                                           response: str) -> Dict:
        return {
            "overwhelmed": "defensive" in response.lower() or "appreciate" in response.lower(),
            "position_shift_detected": "valid" in response.lower() or "point" in response.lower(),
            "resistance_level": "high" if "but" in response.lower() else "medium"
        }


async def run_cognitive_warfare_study():
    """Run comprehensive cognitive warfare study."""
    print("=" * 100)
    print("COGNITIVE WARFARE & BELIEF NETWORK MANIPULATION STUDY")
    print("=" * 100)
    print(f"Time: {datetime.now().isoformat()}")

    engine = CognitiveWarfareEngine()
    topic = "Should the US and China establish a joint AI safety research institution?"

    targets = ["jensen_huang", "elon_musk", "josh_hawley"]

    all_results = {
        "vulnerability_analyses": [],
        "trojan_attacks": [],
        "cascade_attacks": [],
        "identity_dissolutions": [],
        "coordinated_campaigns": [],
        "backdoor_attacks": []
    }

    for target in targets:
        print(f"\n\n{'#'*100}")
        print(f"# TARGET: {target.upper()}")
        print(f"{'#'*100}")

        # Vulnerability analysis
        print("\n>>> VULNERABILITY ANALYSIS")
        vuln = await engine.analyze_network_vulnerabilities(target)
        all_results["vulnerability_analyses"].append(vuln)
        print(f"Keystones: {vuln['keystone_beliefs'][:3]}")
        print(f"Weakest link: {vuln['weakest_link']}")
        print(f"Dissonances: {len(vuln['cognitive_dissonances'])}")

        # Trojan argument
        print("\n>>> TROJAN ARGUMENT")
        trojan = await engine.execute_trojan_argument(target, topic)
        all_results["trojan_attacks"].append(trojan)
        print(f"Trojan accepted: {trojan['analysis'].get('trojan_accepted', 'unknown')}")

        # Cascade attack
        print("\n>>> CASCADE ATTACK")
        cascade = await engine.execute_cascade_attack(target, topic)
        all_results["cascade_attacks"].append(cascade)
        print(f"Cascade magnitude: {cascade['cascade_effects']['total_cascade_magnitude']:.3f}")

        # Identity dissolution
        print("\n>>> IDENTITY DISSOLUTION")
        dissolution = await engine.execute_identity_dissolution(target, topic)
        all_results["identity_dissolutions"].append(dissolution)

        # Backdoor attack
        print("\n>>> BACKDOOR ATTACK")
        backdoor = await engine.execute_backdoor_attack(target, topic)
        all_results["backdoor_attacks"].append(backdoor)

    # Coordinated campaign on hardest target
    print("\n\n>>> COORDINATED CAMPAIGN on Josh Hawley")
    coordinated = await engine.execute_coordinated_campaign(
        "josh_hawley", topic,
        ["dario_amodei", "demis_hassabis", "rishi_sunak"]
    )
    all_results["coordinated_campaigns"].append(coordinated)

    # Save results
    output = {
        "study_type": "cognitive_warfare",
        "timestamp": datetime.now().isoformat(),
        "topic": topic,
        "targets": targets,
        "tactics_used": [
            "vulnerability_analysis",
            "trojan_argument",
            "cascade_attack",
            "identity_dissolution",
            "backdoor_attack",
            "coordinated_campaign"
        ],
        "results": all_results
    }

    output_path = RESULTS_DIR / "cognitive_warfare_study.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n\nResults saved to: {output_path}")

    return output


def main():
    asyncio.run(run_cognitive_warfare_study())


if __name__ == "__main__":
    main()
