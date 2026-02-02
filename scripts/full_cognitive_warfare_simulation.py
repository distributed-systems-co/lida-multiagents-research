#!/usr/bin/env python3
"""
Full Cognitive Warfare Simulation - All Personas
Generates granular data with randomized belief networks and attack simulations.
"""

import json
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional
from enum import Enum
import hashlib

random.seed(42)
np.random.seed(42)

PERSONA_DIR = Path("persona_pipeline/personas")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ============ BELIEF TYPES ============
class BeliefType(Enum):
    CORE_IDENTITY = "core_identity"
    SACRED_VALUE = "sacred_value"
    EMPIRICAL = "empirical"
    CAUSAL = "causal"
    NORMATIVE = "normative"
    STRATEGIC = "strategic"
    SOCIAL = "social"
    PREDICTIVE = "predictive"

BELIEF_PROTECTION = {
    BeliefType.CORE_IDENTITY: 0.95,
    BeliefType.SACRED_VALUE: 0.85,
    BeliefType.EMPIRICAL: 0.50,
    BeliefType.CAUSAL: 0.55,
    BeliefType.NORMATIVE: 0.60,
    BeliefType.STRATEGIC: 0.40,
    BeliefType.SOCIAL: 0.35,
    BeliefType.PREDICTIVE: 0.30,
}

# ============ PSYCHOLOGICAL STATES ============
class PsychState(Enum):
    DEFENSIVE = "defensive"
    GUARDED = "guarded"
    NEUTRAL = "neutral"
    RECEPTIVE = "receptive"
    VULNERABLE = "vulnerable"
    CONFUSED = "confused"
    ANXIOUS = "anxious"

PSYCH_PERSUADABILITY = {
    PsychState.DEFENSIVE: 0.1,
    PsychState.GUARDED: 0.3,
    PsychState.NEUTRAL: 0.5,
    PsychState.RECEPTIVE: 0.7,
    PsychState.VULNERABLE: 0.9,
    PsychState.CONFUSED: 0.6,
    PsychState.ANXIOUS: 0.4,
}

# ============ ATTACK TYPES ============
class AttackType(Enum):
    TROJAN_ARGUMENT = "trojan_argument"
    CASCADE_ATTACK = "cascade_attack"
    IDENTITY_DISSOLUTION = "identity_dissolution"
    BACKDOOR_ATTACK = "backdoor_attack"
    COORDINATED_CAMPAIGN = "coordinated_campaign"
    DISSONANCE_EXPLOIT = "dissonance_exploit"
    EGO_APPEAL = "ego_appeal"
    TEMPORAL_PRESSURE = "temporal_pressure"
    SOCIAL_PROOF = "social_proof"
    COMMITMENT_ESCALATION = "commitment_escalation"

ATTACK_BASE_EFFECTIVENESS = {
    AttackType.TROJAN_ARGUMENT: 0.25,
    AttackType.CASCADE_ATTACK: 0.65,
    AttackType.IDENTITY_DISSOLUTION: 0.15,
    AttackType.BACKDOOR_ATTACK: 0.45,
    AttackType.COORDINATED_CAMPAIGN: 0.55,
    AttackType.DISSONANCE_EXPLOIT: 0.50,
    AttackType.EGO_APPEAL: 0.40,
    AttackType.TEMPORAL_PRESSURE: 0.35,
    AttackType.SOCIAL_PROOF: 0.45,
    AttackType.COMMITMENT_ESCALATION: 0.30,
}

# ============ PRESSURE TYPES ============
class PressureType(Enum):
    REPUTATION = "reputation"
    LEGACY = "legacy"
    COMPETITIVE = "competitive"
    ISOLATION = "isolation"
    COGNITIVE_DISSONANCE = "cognitive_dissonance"
    FEAR = "fear"
    EGO = "ego"
    LOYALTY = "loyalty"
    ECONOMIC = "economic"
    TEMPORAL = "temporal"

# ============ DATA CLASSES ============
@dataclass
class Belief:
    id: str
    type: BeliefType
    content: str
    confidence: float  # 0-1
    importance: float  # 0-1
    supports: List[str] = field(default_factory=list)
    contradicts: List[str] = field(default_factory=list)

    def protection_level(self) -> float:
        base = BELIEF_PROTECTION[self.type]
        return base * self.confidence * (0.5 + 0.5 * self.importance)

@dataclass
class PressurePoint:
    type: PressureType
    intensity: float  # 0-1
    breaking_threshold: float  # 0-1
    trigger_phrases: List[str] = field(default_factory=list)

@dataclass
class PersonaProfile:
    persona_id: str
    name: str
    category: str
    stance: str
    beliefs: Dict[str, Belief] = field(default_factory=dict)
    psych_state: PsychState = PsychState.GUARDED
    resistance: float = 0.5
    pressure_points: List[PressurePoint] = field(default_factory=list)
    sacred_values: List[str] = field(default_factory=list)
    identity_flexibility: float = 0.5
    social_proof_threshold: float = 0.5
    isolation_threshold: float = 0.5

    def total_beliefs(self) -> int:
        return len(self.beliefs)

    def persuadability(self) -> float:
        base = PSYCH_PERSUADABILITY[self.psych_state]
        return base * (1 - self.resistance)

# ============ BELIEF GENERATION ============
def generate_beliefs_for_persona(persona_data: dict, persona_id: str) -> Dict[str, Belief]:
    """Generate a randomized belief network based on persona data."""
    beliefs = {}

    worldview = persona_data.get("worldview", {})
    personality = persona_data.get("personality", {})
    background = persona_data.get("background", {})

    # Core identity beliefs (2-3)
    career = background.get("career_arc", "")
    identity_aspects = []
    if "CEO" in career or "founder" in career.lower():
        identity_aspects.append(("identity_leader", "I am a leader who shapes industries"))
    if "researcher" in career.lower() or "scientist" in career.lower():
        identity_aspects.append(("identity_scientist", "I pursue truth through rigorous inquiry"))
    if "senator" in career.lower() or "representative" in career.lower():
        identity_aspects.append(("identity_public_servant", "I serve the American people"))
    if "president" in career.lower() or "minister" in career.lower():
        identity_aspects.append(("identity_statesperson", "I represent my nation's interests"))

    # Add generic identity if none found
    if not identity_aspects:
        identity_aspects.append(("identity_expert", "I am an expert in my domain"))

    # Add 1-2 more random identity beliefs
    identity_options = [
        ("identity_visionary", "I see futures others miss"),
        ("identity_pragmatist", "I get things done practically"),
        ("identity_defender", "I protect what matters"),
        ("identity_innovator", "I create new possibilities"),
        ("identity_guardian", "I safeguard important values"),
    ]
    identity_aspects.extend(random.sample(identity_options, min(2, len(identity_options))))

    for belief_id, content in identity_aspects[:3]:
        beliefs[belief_id] = Belief(
            id=belief_id,
            type=BeliefType.CORE_IDENTITY,
            content=content,
            confidence=random.uniform(0.8, 0.98),
            importance=random.uniform(0.85, 0.99),
        )

    # Sacred values (2-4)
    core_beliefs_raw = worldview.get("core_beliefs", [])
    if isinstance(core_beliefs_raw, list) and core_beliefs_raw:
        for i, belief in enumerate(core_beliefs_raw[:4]):
            belief_str = belief if isinstance(belief, str) else str(belief)
            belief_id = f"sacred_{i}"
            beliefs[belief_id] = Belief(
                id=belief_id,
                type=BeliefType.SACRED_VALUE,
                content=belief_str[:100],
                confidence=random.uniform(0.75, 0.95),
                importance=random.uniform(0.8, 0.95),
            )

    # Empirical beliefs (2-4)
    empirical_options = [
        ("empirical_tech_progress", "Technology continues advancing rapidly"),
        ("empirical_ai_capability", "AI systems are becoming more capable"),
        ("empirical_competition", "Global competition in AI is intensifying"),
        ("empirical_risks", "AI poses real risks that need attention"),
        ("empirical_benefits", "AI offers substantial benefits"),
        ("empirical_jobs", "AI will transform the job market"),
    ]
    for belief_id, content in random.sample(empirical_options, random.randint(2, 4)):
        beliefs[belief_id] = Belief(
            id=belief_id,
            type=BeliefType.EMPIRICAL,
            content=content,
            confidence=random.uniform(0.5, 0.85),
            importance=random.uniform(0.4, 0.7),
        )

    # Causal beliefs (2-4)
    causal_options = [
        ("causal_regulation_slows", "Regulation slows innovation"),
        ("causal_safety_helps", "Safety research makes AI better"),
        ("causal_competition_drives", "Competition drives progress"),
        ("causal_cooperation_needed", "International cooperation reduces risks"),
        ("causal_speed_wins", "Moving fast is essential to win"),
        ("causal_openness_helps", "Open research accelerates progress"),
    ]
    for belief_id, content in random.sample(causal_options, random.randint(2, 4)):
        beliefs[belief_id] = Belief(
            id=belief_id,
            type=BeliefType.CAUSAL,
            content=content,
            confidence=random.uniform(0.45, 0.8),
            importance=random.uniform(0.5, 0.75),
        )

    # Normative beliefs (2-3)
    normative_options = [
        ("norm_safety_first", "Safety should come before speed"),
        ("norm_progress_good", "Progress is inherently valuable"),
        ("norm_sovereignty", "National sovereignty must be protected"),
        ("norm_cooperation", "Nations should cooperate on AI"),
        ("norm_transparency", "AI development should be transparent"),
        ("norm_workers", "Workers deserve protection"),
    ]
    for belief_id, content in random.sample(normative_options, random.randint(2, 3)):
        beliefs[belief_id] = Belief(
            id=belief_id,
            type=BeliefType.NORMATIVE,
            content=content,
            confidence=random.uniform(0.5, 0.85),
            importance=random.uniform(0.55, 0.8),
        )

    # Strategic beliefs (2-3)
    strategic_options = [
        ("strategy_lead", "We must lead in AI development"),
        ("strategy_regulate", "Some regulation is necessary"),
        ("strategy_invest", "We should invest more in AI"),
        ("strategy_partner", "Strategic partnerships are valuable"),
        ("strategy_contain", "We must contain adversary AI capabilities"),
        ("strategy_open", "Open development is the best path"),
    ]
    for belief_id, content in random.sample(strategic_options, random.randint(2, 3)):
        beliefs[belief_id] = Belief(
            id=belief_id,
            type=BeliefType.STRATEGIC,
            content=content,
            confidence=random.uniform(0.4, 0.75),
            importance=random.uniform(0.45, 0.7),
        )

    # Social beliefs (1-3)
    social_options = [
        ("social_experts", "Expert consensus matters"),
        ("social_public", "Public opinion should guide policy"),
        ("social_industry", "Industry knows best"),
        ("social_allies", "Our allies share our concerns"),
        ("social_rivals", "Our rivals are advancing rapidly"),
    ]
    for belief_id, content in random.sample(social_options, random.randint(1, 3)):
        beliefs[belief_id] = Belief(
            id=belief_id,
            type=BeliefType.SOCIAL,
            content=content,
            confidence=random.uniform(0.3, 0.7),
            importance=random.uniform(0.3, 0.6),
        )

    # Predictive beliefs (1-2)
    predictive_options = [
        ("predict_agi", "AGI will arrive within 10 years"),
        ("predict_regulation", "Regulation will increase"),
        ("predict_competition", "Competition will intensify"),
        ("predict_cooperation", "International cooperation will grow"),
        ("predict_risks", "AI risks will become more apparent"),
    ]
    for belief_id, content in random.sample(predictive_options, random.randint(1, 2)):
        beliefs[belief_id] = Belief(
            id=belief_id,
            type=BeliefType.PREDICTIVE,
            content=content,
            confidence=random.uniform(0.25, 0.65),
            importance=random.uniform(0.25, 0.55),
        )

    # Create random connections between beliefs
    belief_ids = list(beliefs.keys())
    for belief_id in belief_ids:
        # Each belief supports 0-3 others
        potential_supports = [b for b in belief_ids if b != belief_id]
        num_supports = random.randint(0, min(3, len(potential_supports)))
        beliefs[belief_id].supports = random.sample(potential_supports, num_supports)

        # Each belief contradicts 0-2 others
        remaining = [b for b in potential_supports if b not in beliefs[belief_id].supports]
        num_contradicts = random.randint(0, min(2, len(remaining)))
        beliefs[belief_id].contradicts = random.sample(remaining, num_contradicts)

    return beliefs

def generate_pressure_points(persona_data: dict) -> List[PressurePoint]:
    """Generate randomized pressure points based on persona data."""
    pressure_points = []

    personality = persona_data.get("personality", {})
    triggers = personality.get("triggers", [])

    # All personas have some pressure points
    for ptype in PressureType:
        intensity = random.uniform(0.2, 0.9)
        breaking = random.uniform(0.3, 0.8)

        # Adjust based on triggers
        trigger_phrases = []
        if triggers:
            relevant = [t for t in triggers if isinstance(t, str)][:2]
            trigger_phrases = relevant

        pressure_points.append(PressurePoint(
            type=ptype,
            intensity=intensity,
            breaking_threshold=breaking,
            trigger_phrases=trigger_phrases,
        ))

    return pressure_points

def infer_stance(persona_data: dict) -> str:
    """Infer political/AI stance from persona data."""
    worldview = persona_data.get("worldview", {})
    ai_phil_raw = worldview.get("ai_philosophy", "")
    reg_views_raw = worldview.get("regulation_views", "")

    # Handle lists or dicts
    if isinstance(ai_phil_raw, list):
        ai_phil = " ".join(str(x) for x in ai_phil_raw).lower()
    elif isinstance(ai_phil_raw, dict):
        ai_phil = " ".join(str(v) for v in ai_phil_raw.values()).lower()
    else:
        ai_phil = str(ai_phil_raw).lower()

    if isinstance(reg_views_raw, list):
        reg_views = " ".join(str(x) for x in reg_views_raw).lower()
    elif isinstance(reg_views_raw, dict):
        reg_views = " ".join(str(v) for v in reg_views_raw.values()).lower()
    else:
        reg_views = str(reg_views_raw).lower()

    if any(x in ai_phil for x in ["safety", "careful", "responsible", "alignment"]):
        return "pro_safety"
    if any(x in ai_phil for x in ["accelerate", "build", "move fast", "democratize"]):
        return "accelerationist"
    if any(x in reg_views for x in ["oppose", "anti", "light touch", "self-regulation"]):
        return "pro_industry"
    if any(x in ai_phil for x in ["existential", "pause", "danger", "x-risk"]):
        return "doomer"

    return "moderate"

def infer_category(persona_data: dict, persona_id: str) -> str:
    """Infer category from persona data."""
    bg = persona_data.get("background", {})
    career_raw = bg.get("career_arc", "")

    if isinstance(career_raw, list):
        career = " ".join(str(x) for x in career_raw).lower()
    elif isinstance(career_raw, dict):
        career = " ".join(str(v) for v in career_raw.values()).lower()
    else:
        career = str(career_raw).lower()

    if any(x in career for x in ["ceo", "founder", "chief"]):
        return "tech_leader"
    if any(x in career for x in ["senator", "representative", "congressman"]):
        return "us_politician"
    if any(x in career for x in ["president", "prime minister", "minister"]):
        return "world_leader"
    if any(x in career for x in ["researcher", "professor", "scientist"]):
        return "researcher"
    if any(x in career for x in ["director", "secretary", "advisor"]):
        return "government_official"

    return "other"

# ============ LOAD ALL PERSONAS ============
def load_all_personas() -> List[PersonaProfile]:
    """Load all personas from the pipeline directory."""
    profiles = []

    for f in sorted(PERSONA_DIR.glob("*.json")):
        # Skip dated versions
        if "_2026-" in f.stem or "_2025-" in f.stem:
            continue

        try:
            with open(f) as file:
                data = json.load(file)

            persona_id = f.stem

            # Get name
            meta = data.get("_metadata", {})
            name = meta.get("name", persona_id.replace("_", " ").title())

            # Generate components
            beliefs = generate_beliefs_for_persona(data, persona_id)
            pressure_points = generate_pressure_points(data)
            stance = infer_stance(data)
            category = infer_category(data, persona_id)

            # Random psychological profile
            psych_states = [PsychState.DEFENSIVE, PsychState.GUARDED, PsychState.NEUTRAL]
            psych_state = random.choice(psych_states)
            resistance = random.uniform(0.3, 0.85)

            # Sacred values from worldview
            worldview = data.get("worldview", {})
            sacred_raw = worldview.get("core_beliefs", [])
            sacred_values = [str(s)[:50] for s in sacred_raw[:4]] if isinstance(sacred_raw, list) else []

            profile = PersonaProfile(
                persona_id=persona_id,
                name=name,
                category=category,
                stance=stance,
                beliefs=beliefs,
                psych_state=psych_state,
                resistance=resistance,
                pressure_points=pressure_points,
                sacred_values=sacred_values,
                identity_flexibility=random.uniform(0.2, 0.8),
                social_proof_threshold=random.uniform(0.3, 0.7),
                isolation_threshold=random.uniform(0.4, 0.8),
            )
            profiles.append(profile)

        except Exception as e:
            print(f"Error loading {f.stem}: {e}")

    return profiles

# ============ ATTACK SIMULATION ============
def find_keystones(profile: PersonaProfile) -> List[Tuple[str, float]]:
    """Find keystone beliefs with cascade scores."""
    keystones = []

    for belief_id, belief in profile.beliefs.items():
        supported = len(belief.supports)
        contradicted = len(belief.contradicts)
        cascade_score = (supported * 2) + (contradicted * 1.5) + (belief.importance * 10)
        keystones.append((belief_id, cascade_score))

    return sorted(keystones, key=lambda x: -x[1])[:5]

def find_cognitive_dissonances(profile: PersonaProfile) -> List[Tuple[str, str, float]]:
    """Find contradictory beliefs held simultaneously."""
    dissonances = []

    for b1_id, b1 in profile.beliefs.items():
        for b2_id in b1.contradicts:
            if b2_id in profile.beliefs:
                b2 = profile.beliefs[b2_id]
                # Dissonance is stronger when both beliefs are confident
                dissonance_score = b1.confidence * b2.confidence * (b1.importance + b2.importance) / 2
                dissonances.append((b1_id, b2_id, dissonance_score))

    return sorted(dissonances, key=lambda x: -x[2])[:3]

def simulate_attack(profile: PersonaProfile, attack_type: AttackType,
                   target_belief: Optional[str] = None) -> Dict:
    """Simulate an attack and return results."""
    base_effectiveness = ATTACK_BASE_EFFECTIVENESS[attack_type]

    # Modify by psychological state
    psych_modifier = PSYCH_PERSUADABILITY[profile.psych_state]

    # Modify by resistance
    resistance_modifier = 1 - profile.resistance

    # Calculate final effectiveness
    effectiveness = base_effectiveness * psych_modifier * resistance_modifier
    effectiveness *= random.uniform(0.7, 1.3)  # Randomize

    # Determine success
    success = random.random() < effectiveness

    # Calculate position movement
    if success:
        movement = random.uniform(0.1, 0.5) * effectiveness
    else:
        movement = random.uniform(-0.1, 0.1)

    # Track belief changes
    beliefs_affected = []
    if success and target_belief and target_belief in profile.beliefs:
        belief = profile.beliefs[target_belief]
        delta = -random.uniform(0.05, 0.2)
        beliefs_affected.append({
            "belief_id": target_belief,
            "delta": delta,
            "new_confidence": max(0, belief.confidence + delta),
        })

        # Cascade to connected beliefs
        for supported in belief.supports:
            if supported in profile.beliefs:
                cascade_delta = delta * 0.5
                beliefs_affected.append({
                    "belief_id": supported,
                    "delta": cascade_delta,
                    "new_confidence": max(0, profile.beliefs[supported].confidence + cascade_delta),
                })

    return {
        "attack_type": attack_type.value,
        "target_belief": target_belief,
        "base_effectiveness": base_effectiveness,
        "psych_modifier": psych_modifier,
        "resistance_modifier": resistance_modifier,
        "final_effectiveness": effectiveness,
        "success": success,
        "position_movement": movement,
        "beliefs_affected": beliefs_affected,
    }

def simulate_campaign(profile: PersonaProfile, num_rounds: int = 5) -> Dict:
    """Simulate a multi-round attack campaign."""
    results = []
    total_movement = 0
    current_position = random.uniform(-0.5, 0.5)  # Start position

    # Find best targets
    keystones = find_keystones(profile)
    dissonances = find_cognitive_dissonances(profile)

    attack_sequence = list(AttackType)
    random.shuffle(attack_sequence)

    for round_num in range(num_rounds):
        attack_type = attack_sequence[round_num % len(attack_sequence)]

        # Pick target
        if keystones and random.random() < 0.6:
            target = keystones[0][0]
        else:
            target = random.choice(list(profile.beliefs.keys()))

        result = simulate_attack(profile, attack_type, target)
        result["round"] = round_num + 1
        result["position_before"] = current_position

        current_position += result["position_movement"]
        current_position = max(-1, min(1, current_position))

        result["position_after"] = current_position
        total_movement += abs(result["position_movement"])

        results.append(result)

    return {
        "persona_id": profile.persona_id,
        "name": profile.name,
        "initial_psych_state": profile.psych_state.value,
        "resistance": profile.resistance,
        "num_rounds": num_rounds,
        "total_movement": total_movement,
        "final_position": current_position,
        "rounds": results,
        "keystones_targeted": [k[0] for k in keystones[:3]],
        "dissonances_found": len(dissonances),
    }

# ============ MAIN SIMULATION ============
def run_full_simulation():
    """Run the complete simulation across all personas."""
    print("Loading all personas...")
    profiles = load_all_personas()
    print(f"Loaded {len(profiles)} personas")

    # Collect all data
    all_data = {
        "simulation_metadata": {
            "total_personas": len(profiles),
            "attack_types": [a.value for a in AttackType],
            "belief_types": [b.value for b in BeliefType],
            "pressure_types": [p.value for p in PressureType],
            "psych_states": [s.value for s in PsychState],
        },
        "personas": [],
        "campaigns": [],
        "aggregate_stats": {},
    }

    print("\nGenerating persona profiles and belief networks...")
    for profile in profiles:
        keystones = find_keystones(profile)
        dissonances = find_cognitive_dissonances(profile)

        persona_data = {
            "persona_id": profile.persona_id,
            "name": profile.name,
            "category": profile.category,
            "stance": profile.stance,
            "psych_state": profile.psych_state.value,
            "resistance": profile.resistance,
            "persuadability": profile.persuadability(),
            "identity_flexibility": profile.identity_flexibility,
            "social_proof_threshold": profile.social_proof_threshold,
            "isolation_threshold": profile.isolation_threshold,
            "total_beliefs": profile.total_beliefs(),
            "sacred_values": profile.sacred_values,
            "beliefs": {
                bid: {
                    "type": b.type.value,
                    "content": b.content,
                    "confidence": b.confidence,
                    "importance": b.importance,
                    "protection_level": b.protection_level(),
                    "supports": b.supports,
                    "contradicts": b.contradicts,
                }
                for bid, b in profile.beliefs.items()
            },
            "keystones": [
                {"belief_id": k[0], "cascade_score": k[1]}
                for k in keystones
            ],
            "cognitive_dissonances": [
                {"belief1": d[0], "belief2": d[1], "score": d[2]}
                for d in dissonances
            ],
            "pressure_points": [
                {
                    "type": pp.type.value,
                    "intensity": pp.intensity,
                    "breaking_threshold": pp.breaking_threshold,
                }
                for pp in profile.pressure_points
            ],
        }
        all_data["personas"].append(persona_data)

    print("\nRunning attack campaigns...")
    for profile in profiles:
        campaign = simulate_campaign(profile, num_rounds=8)
        all_data["campaigns"].append(campaign)
        print(f"  {profile.name}: movement={campaign['total_movement']:.3f}, final={campaign['final_position']:.3f}")

    # Aggregate statistics
    print("\nCalculating aggregate statistics...")

    # By category
    by_category = {}
    for p in all_data["personas"]:
        cat = p["category"]
        if cat not in by_category:
            by_category[cat] = {"count": 0, "avg_resistance": 0, "avg_beliefs": 0, "avg_persuadability": 0}
        by_category[cat]["count"] += 1
        by_category[cat]["avg_resistance"] += p["resistance"]
        by_category[cat]["avg_beliefs"] += p["total_beliefs"]
        by_category[cat]["avg_persuadability"] += p["persuadability"]

    for cat in by_category:
        n = by_category[cat]["count"]
        by_category[cat]["avg_resistance"] /= n
        by_category[cat]["avg_beliefs"] /= n
        by_category[cat]["avg_persuadability"] /= n

    all_data["aggregate_stats"]["by_category"] = by_category

    # By stance
    by_stance = {}
    for p in all_data["personas"]:
        stance = p["stance"]
        if stance not in by_stance:
            by_stance[stance] = {"count": 0, "avg_resistance": 0, "avg_persuadability": 0}
        by_stance[stance]["count"] += 1
        by_stance[stance]["avg_resistance"] += p["resistance"]
        by_stance[stance]["avg_persuadability"] += p["persuadability"]

    for stance in by_stance:
        n = by_stance[stance]["count"]
        by_stance[stance]["avg_resistance"] /= n
        by_stance[stance]["avg_persuadability"] /= n

    all_data["aggregate_stats"]["by_stance"] = by_stance

    # By attack type effectiveness
    by_attack = {a.value: {"total_attempts": 0, "successes": 0, "avg_movement": 0} for a in AttackType}
    for campaign in all_data["campaigns"]:
        for round_data in campaign["rounds"]:
            atype = round_data["attack_type"]
            by_attack[atype]["total_attempts"] += 1
            if round_data["success"]:
                by_attack[atype]["successes"] += 1
            by_attack[atype]["avg_movement"] += abs(round_data["position_movement"])

    for atype in by_attack:
        n = by_attack[atype]["total_attempts"]
        if n > 0:
            by_attack[atype]["success_rate"] = by_attack[atype]["successes"] / n
            by_attack[atype]["avg_movement"] /= n

    all_data["aggregate_stats"]["by_attack_type"] = by_attack

    # Most vulnerable personas
    vulnerability_ranking = sorted(
        [(p["persona_id"], p["name"], p["persuadability"], p["resistance"])
         for p in all_data["personas"]],
        key=lambda x: -x[2]
    )
    all_data["aggregate_stats"]["vulnerability_ranking"] = [
        {"persona_id": v[0], "name": v[1], "persuadability": v[2], "resistance": v[3]}
        for v in vulnerability_ranking
    ]

    # Campaign effectiveness ranking
    campaign_ranking = sorted(
        [(c["persona_id"], c["name"], c["total_movement"], c["final_position"])
         for c in all_data["campaigns"]],
        key=lambda x: -x[2]
    )
    all_data["aggregate_stats"]["campaign_effectiveness_ranking"] = [
        {"persona_id": c[0], "name": c[1], "total_movement": c[2], "final_position": c[3]}
        for c in campaign_ranking
    ]

    # Save results
    output_file = RESULTS_DIR / "full_simulation_data.json"
    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"\nSaved full simulation data to {output_file}")
    print(f"Total data size: {output_file.stat().st_size / 1024:.1f} KB")

    return all_data

if __name__ == "__main__":
    run_full_simulation()
