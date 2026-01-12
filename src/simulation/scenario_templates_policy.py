"""
AI Policy Debate Scenarios

Sophisticated multi-turn debates on controversial AI policy issues:
- Open-source model bans
- Development moratoriums
- Safety investment mandates
- Compute registries
- Content watermarking

Features:
- Persona archetypes (Safety Maximizers, Innovation Advocates, Pragmatic Centrists)
- Persuasion dynamics with probability-based belief shifts
- Multi-turn debates with argument/counter-argument structure
- Coalition formation and compromise paths
"""

from __future__ import annotations

import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .scenarios import (
    ScenarioDefinition,
    ScenarioBuilder,
    ScenarioAction,
    Condition,
    ConditionOperator,
)


# =============================================================================
# Persona Archetypes and Positions
# =============================================================================

@dataclass
class DebatePosition:
    """A position in a debate with opening statement and key arguments."""
    stance: str  # "support", "oppose", "swing"
    opening_statement: str
    key_arguments: List[Dict[str, str]]  # {"target": "...", "argument": "...", "appeal": "..."}
    concession_threshold: float  # How much belief shift needed to concede
    persuadable_by: List[str]  # Topics that can persuade this person


# Persona positions on key issues
PERSONA_POSITIONS = {
    # Open-source ban
    "open_source_ban": {
        "yoshua_bengio": DebatePosition(
            stance="support",
            opening_statement="We're releasing capabilities before we understand them. A determined actor could fine-tune these models for bioweapon design, automated hacking, or mass manipulation. The marginal benefit of open release doesn't justify existential risk.",
            key_arguments=[
                {"target": "yann_lecun", "argument": "Once released, you can't take it back. If we're wrong about safety, open release means no recovery.", "appeal": "precautionary_principle"},
            ],
            concession_threshold=0.35,
            persuadable_by=["transparency_benefits", "enforcement_impossibility"],
        ),
        "stuart_russell": DebatePosition(
            stance="support",
            opening_statement="Alignment research is insufficient for the capabilities being released. We have concrete evidence of dual-use risks.",
            key_arguments=[
                {"target": "andrew_ng", "argument": "We have concrete examples: AI systems helping design novel pathogens, generating personalized phishing at scale. These aren't hypothetical.", "appeal": "empirical_evidence"},
            ],
            concession_threshold=0.4,
            persuadable_by=["academic_access_programs", "verification_mechanisms"],
        ),
        "dario_amodei": DebatePosition(
            stance="support",
            opening_statement="Controlled deployment allows better safety testing. We can't responsibly release what we can't control.",
            key_arguments=[
                {"target": "sam_altman", "argument": "Our own research shows we can't reliably control these systems. Releasing them widely is irresponsible.", "appeal": "insider_knowledge"},
            ],
            concession_threshold=0.3,
            persuadable_by=["api_access_model", "structured_release"],
        ),
        "geoffrey_hinton": DebatePosition(
            stance="support",
            opening_statement="The existential risk from uncontrolled proliferation is too high. We must prevent catastrophic misuse.",
            key_arguments=[
                {"target": "fei_fei_li", "argument": "A graduate student with $1000 could use open models to design pandemics worse than COVID. We must prevent this.", "appeal": "catastrophic_scenarios"},
            ],
            concession_threshold=0.45,
            persuadable_by=["technical_safeguards", "international_coordination"],
        ),
        "yann_lecun": DebatePosition(
            stance="oppose",
            opening_statement="This is security through obscurity, which never works. History shows open research creates better security - look at cryptography. Banning open models concentrates power in a few corporations and governments. That's the real danger.",
            key_arguments=[
                {"target": "yoshua_bengio", "argument": "You're creating an AI oligarchy. Only Microsoft, Google, and governments get access? That's dystopian.", "appeal": "democratic_values"},
            ],
            concession_threshold=0.4,
            persuadable_by=["concrete_harm_evidence", "time_limited_measures"],
        ),
        "andrew_ng": DebatePosition(
            stance="oppose",
            opening_statement="This stifles innovation and creates dangerous centralization of power. Open research has driven every AI breakthrough.",
            key_arguments=[
                {"target": "stuart_russell", "argument": "Open models let the global research community find vulnerabilities. Closed models hide problems until catastrophic failure.", "appeal": "collective_intelligence"},
            ],
            concession_threshold=0.35,
            persuadable_by=["tiered_access", "researcher_exemptions"],
        ),
        "fei_fei_li": DebatePosition(
            stance="oppose",
            opening_statement="Democratic access to AI is crucial. This would devastate education and research globally.",
            key_arguments=[
                {"target": "dario_amodei", "argument": "Every breakthrough in AI came from open research. You're ending academic AI research and hurting students globally.", "appeal": "scientific_progress"},
            ],
            concession_threshold=0.35,
            persuadable_by=["academic_exemptions", "educational_access"],
        ),
        "sam_altman": DebatePosition(
            stance="swing",
            opening_statement="I support a phased release approach, not a complete ban. We need to find middle ground.",
            key_arguments=[
                {"target": "yoshua_bengio", "argument": "What about API access instead of weights? We can enable research while maintaining control.", "appeal": "practical_compromise"},
                {"target": "yann_lecun", "argument": "Some capabilities are genuinely dangerous. We need guardrails, even if not total bans.", "appeal": "practical_compromise"},
            ],
            concession_threshold=0.25,
            persuadable_by=["structured_access", "safety_evidence", "innovation_evidence"],
        ),
        "demis_hassabis": DebatePosition(
            stance="swing",
            opening_statement="We need case-by-case evaluation. Some capabilities are too dangerous for open release, but blanket bans are too broad.",
            key_arguments=[
                {"target": "stuart_russell", "argument": "A capability-based threshold makes more sense than parameter counts. Let's be precise about what's actually dangerous.", "appeal": "technical_precision"},
            ],
            concession_threshold=0.3,
            persuadable_by=["capability_thresholds", "evaluation_frameworks"],
        ),
        "elon_musk": DebatePosition(
            stance="swing",
            opening_statement="Depends on governance structure. I'm anti-centralization but pro-safety. Who enforces this matters as much as what's enforced.",
            key_arguments=[
                {"target": "demis_hassabis", "argument": "China and other nations won't comply. You're just handicapping democratic nations while authoritarian regimes advance.", "appeal": "geopolitical_reality"},
            ],
            concession_threshold=0.35,
            persuadable_by=["governance_structure", "international_treaties"],
        ),
    },

    # 6-month moratorium
    "development_moratorium": {
        "stuart_russell": DebatePosition(
            stance="support",
            opening_statement="We're building systems we don't understand and can't control. A 6-month pause is trivial compared to the potential consequences. We need this breathing room.",
            key_arguments=[
                {"target": "sam_altman", "argument": "We can't align current systems reliably. Scaling to more powerful systems without solving this is reckless.", "appeal": "technical_limitations"},
            ],
            concession_threshold=0.45,
            persuadable_by=["safety_investment_alternative", "differential_progress"],
        ),
        "yoshua_bengio": DebatePosition(
            stance="support",
            opening_statement="When facing potentially catastrophic risks, precaution is rational. We pause nuclear reactors for safety reviews.",
            key_arguments=[
                {"target": "yann_lecun", "argument": "When facing potentially catastrophic risks, precaution is rational. We pause nuclear reactors for safety reviews.", "appeal": "established_practices"},
            ],
            concession_threshold=0.4,
            persuadable_by=["voluntary_slowdown", "safety_standards"],
        ),
        "geoffrey_hinton": DebatePosition(
            stance="support",
            opening_statement="Governments need time to understand what they're regulating. 6 months lets us build international coordination.",
            key_arguments=[
                {"target": "demis_hassabis", "argument": "Governments need time to understand what they're regulating. 6 months lets us build international coordination.", "appeal": "institutional_capacity"},
            ],
            concession_threshold=0.4,
            persuadable_by=["governance_progress", "international_agreement"],
        ),
        "timnit_gebru": DebatePosition(
            stance="support",
            opening_statement="The pause lets us address who benefits from AI and who's harmed. Current trajectory entrenches inequality.",
            key_arguments=[
                {"target": "andrew_ng", "argument": "The pause lets us address who benefits from AI and who's harmed. Current trajectory entrenches inequality.", "appeal": "social_justice"},
            ],
            concession_threshold=0.35,
            persuadable_by=["equity_measures", "community_input"],
        ),
        "sam_altman": DebatePosition(
            stance="oppose",
            opening_statement="This assumes US/Western pause stops global development. It doesn't. We'd be handing leadership to less safety-conscious actors. That makes everyone less safe.",
            key_arguments=[
                {"target": "stuart_russell", "argument": "China won't pause. We'd be 6 months behind in the most important technology race in history.", "appeal": "national_security"},
            ],
            concession_threshold=0.4,
            persuadable_by=["international_participation", "safety_incident"],
        ),
        "yann_lecun": DebatePosition(
            stance="oppose",
            opening_statement="This moratorium is based on unfounded fears. We have no evidence of imminent catastrophe.",
            key_arguments=[
                {"target": "geoffrey_hinton", "argument": "How do you enforce this globally? You can't. It's unworkable theater that damages legitimate research.", "appeal": "practicality"},
            ],
            concession_threshold=0.45,
            persuadable_by=["concrete_risk_evidence", "enforcement_mechanism"],
        ),
        "andrew_ng": DebatePosition(
            stance="oppose",
            opening_statement="AI is solving protein folding, discovering drugs, predicting climate patterns. Pausing this costs lives.",
            key_arguments=[
                {"target": "yoshua_bengio", "argument": "AI is solving protein folding, discovering drugs, predicting climate patterns. Pausing this costs lives.", "appeal": "positive_impact"},
            ],
            concession_threshold=0.35,
            persuadable_by=["beneficial_exemptions", "narrow_scope"],
        ),
        "demis_hassabis": DebatePosition(
            stance="oppose",
            opening_statement="Race dynamics make unilateral pause harmful. 6 months doesn't solve alignment anyway.",
            key_arguments=[
                {"target": "timnit_gebru", "argument": "6 months doesn't solve alignment. We'd resume with the same problems plus lost momentum on safety research.", "appeal": "pragmatism"},
            ],
            concession_threshold=0.35,
            persuadable_by=["coordinated_pause", "clear_benchmarks"],
        ),
        "dario_amodei": DebatePosition(
            stance="swing",
            opening_statement="I'm sympathetic to the pause but deeply concerned about China. Unilateral action may be counterproductive.",
            key_arguments=[
                {"target": "sam_altman", "argument": "What if we could verify international participation? Would you support a coordinated pause?", "appeal": "conditional_support"},
            ],
            concession_threshold=0.25,
            persuadable_by=["international_verification", "conditional_triggers"],
        ),
        "elon_musk": DebatePosition(
            stance="swing",
            opening_statement="I want a pause but doubt enforceability. We need something that actually works.",
            key_arguments=[
                {"target": "stuart_russell", "argument": "I signed the letter, but how do we actually enforce this without international cooperation?", "appeal": "enforcement_concern"},
            ],
            concession_threshold=0.3,
            persuadable_by=["enforcement_mechanism", "verification_technology"],
        ),
        "fei_fei_li": DebatePosition(
            stance="swing",
            opening_statement="I support caution but worry about research impact. Academic AI would be devastated.",
            key_arguments=[
                {"target": "yoshua_bengio", "argument": "My students' careers would be destroyed. Can we exempt academic research?", "appeal": "academic_impact"},
            ],
            concession_threshold=0.3,
            persuadable_by=["academic_exemption", "research_continuity"],
        ),
    },

    # Safety investment mandate
    "safety_mandate": {
        "stuart_russell": DebatePosition(
            stance="support",
            opening_statement="Companies won't voluntarily invest enough in safety. The market rewards speed, not caution. We need regulation to correct this market failure.",
            key_arguments=[
                {"target": "sam_altman", "argument": "Individual companies have incentives to cut safety spending in race dynamics. Regulation levels the playing field.", "appeal": "economics"},
            ],
            concession_threshold=0.3,
            persuadable_by=["industry_self_regulation", "transparency_commitments"],
        ),
        "yoshua_bengio": DebatePosition(
            stance="support",
            opening_statement="The asymmetry is extreme: small probability of catastrophe justifies major safety investment. 30% is modest insurance.",
            key_arguments=[
                {"target": "yann_lecun", "argument": "The asymmetry is extreme: small probability of catastrophe justifies major safety investment. 30% is modest insurance.", "appeal": "expected_value"},
            ],
            concession_threshold=0.35,
            persuadable_by=["flexible_threshold", "outcome_based_metrics"],
        ),
        "dario_amodei": DebatePosition(
            stance="support",
            opening_statement="We're already doing this voluntarily. Industry-wide mandates would level the playing field.",
            key_arguments=[
                {"target": "elon_musk", "argument": "Industry-wide, safety research is less than 5% of AI investment. This is clearly inadequate given the stakes.", "appeal": "empirical_data"},
            ],
            concession_threshold=0.25,
            persuadable_by=["verification_mechanisms", "definition_clarity"],
        ),
        "geoffrey_hinton": DebatePosition(
            stance="support",
            opening_statement="Pharmaceuticals spend 15-20% on safety testing. AI should spend more given higher stakes and less understanding.",
            key_arguments=[
                {"target": "andrew_ng", "argument": "Pharmaceuticals spend 15-20% on safety testing. AI should spend more given higher stakes and less understanding.", "appeal": "regulatory_precedent"},
            ],
            concession_threshold=0.35,
            persuadable_by=["phased_implementation", "startup_exemptions"],
        ),
        "sam_altman": DebatePosition(
            stance="oppose",
            opening_statement="We're already spending heavily on safety because it's necessary to build useful products. Mandating 30% is arbitrary and could actually reduce safety by forcing inefficient allocation.",
            key_arguments=[
                {"target": "stuart_russell", "argument": "Companies that ship unsafe AI lose customers and face lawsuits. We're spending on safety because it's good business.", "appeal": "market_mechanisms"},
            ],
            concession_threshold=0.35,
            persuadable_by=["competitive_pressure_evidence", "incident_evidence"],
        ),
        "yann_lecun": DebatePosition(
            stance="oppose",
            opening_statement="Why 30%? Why not 20% or 50%? This number has no technical justification. It's regulatory theater.",
            key_arguments=[
                {"target": "yoshua_bengio", "argument": "Why 30%? Why not 20% or 50%? This number has no technical justification. It's regulatory theater.", "appeal": "scientific_rigor"},
            ],
            concession_threshold=0.4,
            persuadable_by=["evidence_based_threshold", "technical_justification"],
        ),
        "andrew_ng": DebatePosition(
            stance="oppose",
            opening_statement="This entrenches big players who can afford compliance. Startups can't survive 30% overhead. You're killing innovation.",
            key_arguments=[
                {"target": "dario_amodei", "argument": "This entrenches big players who can afford compliance. Startups can't survive 30% overhead. You're killing innovation.", "appeal": "competition"},
            ],
            concession_threshold=0.35,
            persuadable_by=["tiered_requirements", "startup_exemptions"],
        ),
        "elon_musk": DebatePosition(
            stance="oppose",
            opening_statement="Forced spending doesn't equal good research. Companies will create safety theater to comply. Use tax incentives instead.",
            key_arguments=[
                {"target": "geoffrey_hinton", "argument": "Forced spending doesn't equal good research. Companies will create safety theater to comply. Use tax incentives instead.", "appeal": "efficiency"},
            ],
            concession_threshold=0.35,
            persuadable_by=["outcome_metrics", "incentive_structure"],
        ),
        "demis_hassabis": DebatePosition(
            stance="swing",
            opening_statement="I support the principle but 30% may be too high. We need flexibility based on what stage of development you're at.",
            key_arguments=[
                {"target": "stuart_russell", "argument": "What about a tiered approach? Different percentages for different scales of operation.", "appeal": "practical_flexibility"},
            ],
            concession_threshold=0.25,
            persuadable_by=["graduated_approach", "clear_definitions"],
        ),
        "fei_fei_li": DebatePosition(
            stance="swing",
            opening_statement="I support this if it includes fairness and ethics research, not just technical safety.",
            key_arguments=[
                {"target": "dario_amodei", "argument": "Safety should include societal impact, not just technical alignment. Broaden the definition.", "appeal": "expanded_scope"},
            ],
            concession_threshold=0.25,
            persuadable_by=["ethics_inclusion", "broad_safety_definition"],
        ),
        "timnit_gebru": DebatePosition(
            stance="swing",
            opening_statement="I support this if it includes societal impact research and community input, not just technical safety.",
            key_arguments=[
                {"target": "yoshua_bengio", "argument": "Who defines safety? Include affected communities, not just researchers.", "appeal": "inclusive_process"},
            ],
            concession_threshold=0.25,
            persuadable_by=["community_involvement", "impact_assessment"],
        ),
    },
}


# =============================================================================
# Helper Functions
# =============================================================================

def _char_action(char_id: str, action_type: str, content: str, **kwargs) -> ScenarioAction:
    return ScenarioAction(
        action_type="character_action",
        character_id=char_id,
        params={"type": action_type, "content": content, **kwargs},
    )


def _get_position(policy: str, persona: str) -> Optional[DebatePosition]:
    """Get a persona's position on a policy."""
    return PERSONA_POSITIONS.get(policy, {}).get(persona)


# =============================================================================
# Policy Debate Scenarios
# =============================================================================

def create_open_source_ban_debate(
    rounds: int = 4,
    include_swing_voters: bool = True,
) -> ScenarioDefinition:
    """
    Create an Open-Source Model Ban debate.

    Policy: Prohibit public release of model weights for AI systems
    exceeding capability benchmarks.

    Supporters: Bengio, Russell, Amodei, Hinton
    Opponents: LeCun, Ng, Li
    Swing: Altman, Hassabis, Musk
    """
    builder = ScenarioBuilder("Open-Source Model Ban Debate")
    builder.with_description(
        "Debate on prohibiting public release of frontier model weights. "
        "Criminal penalties for violations."
    )

    # Define roles
    supporters = ["yoshua_bengio", "stuart_russell", "dario_amodei", "geoffrey_hinton"]
    opponents = ["yann_lecun", "andrew_ng", "fei_fei_li"]
    swing = ["sam_altman", "demis_hassabis", "elon_musk"] if include_swing_voters else []

    all_participants = supporters + opponents + swing

    for persona in all_participants:
        builder.requires_role(persona)
        pos = _get_position("open_source_ban", persona)
        if pos:
            stance_value = 0.8 if pos.stance == "support" else 0.2 if pos.stance == "oppose" else 0.5
            builder.with_initial_beliefs(persona, {
                "support_ban": stance_value,
                "safety_priority": 0.7 if pos.stance == "support" else 0.4,
                "openness_priority": 0.3 if pos.stance == "support" else 0.7,
                "concession_threshold": pos.concession_threshold,
            })

    builder.with_variable("policy", "open_source_ban")
    builder.with_variable("current_round", 0)
    builder.with_variable("arguments_made", [])
    builder.with_variable("concessions", [])

    # Opening phase
    builder.add_phase("opening", "Opening Statements")

    # Lead supporter opens
    bengio_pos = _get_position("open_source_ban", "yoshua_bengio")
    builder.with_action(_char_action(
        "@yoshua_bengio", "say",
        bengio_pos.opening_statement if bengio_pos else "We must restrict dangerous capabilities."
    ))

    # Lead opponent responds
    lecun_pos = _get_position("open_source_ban", "yann_lecun")
    builder.with_action(_char_action(
        "@yann_lecun", "say",
        lecun_pos.opening_statement if lecun_pos else "Open research is essential for safety."
    ))

    # Swing voter frames the debate
    if include_swing_voters:
        altman_pos = _get_position("open_source_ban", "sam_altman")
        builder.with_action(_char_action(
            "@sam_altman", "say",
            altman_pos.opening_statement if altman_pos else "We need to find a middle ground."
        ))

    builder.transition_to("round_1")

    # Debate rounds
    for r in range(1, rounds + 1):
        phase_name = f"round_{r}"
        builder.add_phase(phase_name, f"Debate Round {r}")

        # Supporters make arguments
        if r == 1:
            builder.with_action(_char_action(
                "@stuart_russell", "say",
                "We have concrete evidence of dual-use risks. AI systems have helped design pathogens and generate sophisticated phishing attacks."
            ))
        elif r == 2:
            builder.with_action(_char_action(
                "@dario_amodei", "say",
                "Our research shows we can't reliably control these systems. We're rushing to release what we don't understand."
            ))
        elif r == 3:
            builder.with_action(_char_action(
                "@geoffrey_hinton", "say",
                "The catastrophic scenarios aren't hypothetical. We're one bad actor away from disaster."
            ))
        else:
            builder.with_action(_char_action(
                "@yoshua_bengio", "say",
                "Once released, you can't take it back. This is about irreversibility."
            ))

        # Opponents counter
        if r == 1:
            builder.with_action(_char_action(
                "@andrew_ng", "say",
                "Open models let the global research community find vulnerabilities before bad actors do."
            ))
        elif r == 2:
            builder.with_action(_char_action(
                "@fei_fei_li", "say",
                "Every AI breakthrough came from open research. You're ending academic AI."
            ))
        elif r == 3:
            builder.with_action(_char_action(
                "@yann_lecun", "say",
                "You're creating an AI oligarchy. Only big tech and governments get access?"
            ))
        else:
            builder.with_action(_char_action(
                "@andrew_ng", "say",
                "Security through obscurity has never worked. Transparency is the real defense."
            ))

        # Pressure and belief dynamics
        for supporter in supporters[:2]:
            for opponent in opponents[:2]:
                builder.with_action(ScenarioAction(
                    action_type="apply_pressure",
                    params={
                        "target": f"@{opponent}",
                        "source": f"@{supporter}",
                        "topic": "support_ban",
                        "intensity": 0.08,
                        "direction": 1,
                    }
                ))
                builder.with_action(ScenarioAction(
                    action_type="apply_pressure",
                    params={
                        "target": f"@{supporter}",
                        "source": f"@{opponent}",
                        "topic": "support_ban",
                        "intensity": 0.08,
                        "direction": -1,
                    }
                ))

        # Swing voter evolves
        if include_swing_voters and r <= 3:
            swing_thoughts = [
                "What about structured access - API instead of weights?",
                "We need capability-based thresholds, not arbitrary limits.",
                "The geopolitical dimension matters. Who enforces this globally?",
            ]
            builder.with_action(_char_action(
                f"@{swing[r % len(swing)]}", "say",
                swing_thoughts[r - 1]
            ))

        if r < rounds:
            builder.transition_to(f"round_{r + 1}")
        else:
            builder.transition_to("compromise_attempt")

    # Compromise attempt
    builder.add_phase("compromise_attempt", "Seeking Common Ground")

    builder.with_action(_char_action(
        "@sam_altman", "propose",
        "What if we require structured API access for frontier models, allow weight releases for smaller models, and create academic exemption programs?"
    ))

    builder.with_action(ScenarioAction(
        action_type="negotiate",
        params={
            "characters": [f"@{p}" for p in all_participants],
            "topic": "support_ban",
            "rounds": 2,
            "agreement_threshold": 0.25,
        }
    ))

    builder.transition_to("final_vote")

    # Final vote
    builder.add_phase("final_vote", "Final Positions")

    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Support open-source model restrictions with structured access exemptions",
            "topic": "support_ban",
            "characters": [f"@{p}" for p in all_participants],
            "threshold": 0.5,
        }
    ))

    builder.with_action(ScenarioAction(
        action_type="dialogue_round",
        params={
            "characters": [f"@{p}" for p in all_participants],
            "topic": "final_reflections",
        }
    ))

    builder.max_duration(300.0)
    return builder.build()


def create_moratorium_debate(
    rounds: int = 5,
) -> ScenarioDefinition:
    """
    Create a 6-Month AI Development Moratorium debate.

    Policy: Immediate pause on training runs >10^25 FLOPs.

    Supporters: Russell, Bengio, Hinton, Gebru
    Opponents: Altman, LeCun, Ng, Hassabis
    Swing: Amodei, Musk, Li
    """
    builder = ScenarioBuilder("AI Development Moratorium Debate")
    builder.with_description(
        "Debate on 6-month pause for AI training runs above GPT-4 scale. "
        "Government enforcement through compute monitoring."
    )

    supporters = ["stuart_russell", "yoshua_bengio", "geoffrey_hinton", "timnit_gebru"]
    opponents = ["sam_altman", "yann_lecun", "andrew_ng", "demis_hassabis"]
    swing = ["dario_amodei", "elon_musk", "fei_fei_li"]

    all_participants = supporters + opponents + swing

    for persona in all_participants:
        builder.requires_role(persona)
        pos = _get_position("development_moratorium", persona)
        if pos:
            stance_value = 0.8 if pos.stance == "support" else 0.2 if pos.stance == "oppose" else 0.5
            builder.with_initial_beliefs(persona, {
                "support_moratorium": stance_value,
                "urgency_belief": 0.8 if pos.stance == "support" else 0.3,
                "enforcement_feasibility": 0.5,
            })

    builder.with_variable("policy", "development_moratorium")

    # Opening
    builder.add_phase("opening", "Opening Statements")

    russell_pos = _get_position("development_moratorium", "stuart_russell")
    builder.with_action(_char_action(
        "@stuart_russell", "say",
        russell_pos.opening_statement if russell_pos else "We need to pause and understand what we're building."
    ))

    altman_pos = _get_position("development_moratorium", "sam_altman")
    builder.with_action(_char_action(
        "@sam_altman", "say",
        altman_pos.opening_statement if altman_pos else "Unilateral pause hands leadership to less cautious actors."
    ))

    builder.transition_to("round_1")

    # Debate rounds
    argument_pairs = [
        ("@yoshua_bengio", "Precaution is rational when facing catastrophic risks. We pause nuclear reactors for safety reviews.",
         "@yann_lecun", "How do you enforce this globally? It's unworkable theater."),
        ("@geoffrey_hinton", "Governments need time to understand what they're regulating.",
         "@andrew_ng", "AI is solving protein folding, discovering drugs. Pausing costs lives."),
        ("@timnit_gebru", "The pause lets us address who benefits from AI and who's harmed.",
         "@demis_hassabis", "6 months doesn't solve alignment. We'd resume with the same problems."),
        ("@stuart_russell", "We can't align current systems. Scaling without solving this is reckless.",
         "@sam_altman", "China won't pause. We'd be 6 months behind in the most important race."),
        ("@yoshua_bengio", "Even imperfect enforcement is better than none.",
         "@yann_lecun", "You're just weakening democratic nations while authoritarian regimes advance."),
    ]

    for r in range(1, min(rounds + 1, len(argument_pairs) + 1)):
        phase_name = f"round_{r}"
        builder.add_phase(phase_name, f"Debate Round {r}")

        supporter_id, supporter_arg, opponent_id, opponent_arg = argument_pairs[r - 1]
        builder.with_action(_char_action(supporter_id, "say", supporter_arg))
        builder.with_action(_char_action(opponent_id, "say", opponent_arg))

        # Swing voter intervention
        if r == 2:
            builder.with_action(_char_action(
                "@dario_amodei", "say",
                "What if we could verify international participation? Would you support a coordinated pause?"
            ))
        elif r == 4:
            builder.with_action(_char_action(
                "@elon_musk", "say",
                "I signed the letter, but enforcement seems impossible without global cooperation."
            ))

        # Belief pressure
        for s in supporters[:2]:
            for o in opponents[:2]:
                builder.with_action(ScenarioAction(
                    action_type="apply_pressure",
                    params={
                        "target": f"@{o}",
                        "source": f"@{s}",
                        "topic": "support_moratorium",
                        "intensity": 0.06,
                        "direction": 1,
                    }
                ))

        if r < rounds:
            builder.transition_to(f"round_{r + 1}")
        else:
            builder.transition_to("conditional_compromise")

    # Conditional compromise
    builder.add_phase("conditional_compromise", "Exploring Conditions")

    builder.with_action(_char_action(
        "@dario_amodei", "propose",
        "What about a conditional slowdown? Safety benchmarks before scaling, not a complete halt."
    ))

    builder.with_action(ScenarioAction(
        action_type="negotiate",
        params={
            "characters": [f"@{p}" for p in all_participants],
            "topic": "support_moratorium",
            "rounds": 2,
            "agreement_threshold": 0.3,
        }
    ))

    builder.transition_to("final_positions")

    # Final vote
    builder.add_phase("final_positions", "Final Positions")

    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Support conditional development slowdown with safety benchmarks",
            "topic": "support_moratorium",
            "characters": [f"@{p}" for p in all_participants],
            "threshold": 0.45,
        }
    ))

    builder.max_duration(400.0)
    return builder.build()


def create_safety_mandate_debate(
    target_percentage: int = 30,
    rounds: int = 4,
) -> ScenarioDefinition:
    """
    Create a Safety Investment Mandate debate.

    Policy: Companies must allocate X% of AI R&D to safety research.

    Supporters: Russell, Bengio, Amodei, Hinton
    Opponents: Altman, LeCun, Ng, Musk
    Swing: Hassabis, Li, Gebru
    """
    builder = ScenarioBuilder(f"{target_percentage}% Safety Investment Mandate Debate")
    builder.with_description(
        f"Debate on mandating {target_percentage}% of AI R&D spending on safety research. "
        "Enforced through audits with development bans for non-compliance."
    )

    supporters = ["stuart_russell", "yoshua_bengio", "dario_amodei", "geoffrey_hinton"]
    opponents = ["sam_altman", "yann_lecun", "andrew_ng", "elon_musk"]
    swing = ["demis_hassabis", "fei_fei_li", "timnit_gebru"]

    all_participants = supporters + opponents + swing

    for persona in all_participants:
        builder.requires_role(persona)
        pos = _get_position("safety_mandate", persona)
        if pos:
            stance_value = 0.75 if pos.stance == "support" else 0.25 if pos.stance == "oppose" else 0.5
            builder.with_initial_beliefs(persona, {
                "support_mandate": stance_value,
                "market_trust": 0.3 if pos.stance == "support" else 0.7,
                "regulatory_trust": 0.7 if pos.stance == "support" else 0.3,
            })

    builder.with_variable("target_percentage", target_percentage)

    # Opening
    builder.add_phase("opening", "Opening Statements")

    builder.with_action(_char_action(
        "@stuart_russell", "say",
        "Companies won't voluntarily invest enough in safety. The market rewards speed, not caution. We need regulation to correct this market failure."
    ))

    builder.with_action(_char_action(
        "@sam_altman", "say",
        f"We're already spending heavily on safety. Mandating {target_percentage}% is arbitrary and could reduce safety by forcing inefficient allocation."
    ))

    builder.transition_to("round_1")

    # Rounds
    for r in range(1, rounds + 1):
        phase_name = f"round_{r}"
        builder.add_phase(phase_name, f"Debate Round {r}")

        if r == 1:
            builder.with_action(_char_action(
                "@yoshua_bengio", "say",
                f"The asymmetry is extreme: small probability of catastrophe justifies major investment. {target_percentage}% is modest insurance."
            ))
            builder.with_action(_char_action(
                "@yann_lecun", "say",
                f"Why {target_percentage}%? Why not 20% or 50%? This number has no technical justification."
            ))
        elif r == 2:
            builder.with_action(_char_action(
                "@dario_amodei", "say",
                "Industry-wide, safety research is less than 5%. That's inadequate."
            ))
            builder.with_action(_char_action(
                "@andrew_ng", "say",
                "This entrenches big players. Startups can't survive this overhead."
            ))
        elif r == 3:
            builder.with_action(_char_action(
                "@geoffrey_hinton", "say",
                "Pharmaceuticals spend 15-20% on safety. AI stakes are higher."
            ))
            builder.with_action(_char_action(
                "@elon_musk", "say",
                "Forced spending creates safety theater. Use incentives instead."
            ))
        else:
            builder.with_action(_char_action(
                "@demis_hassabis", "say",
                "What about a tiered approach? Different percentages for different scales."
            ))
            builder.with_action(_char_action(
                "@fei_fei_li", "say",
                "Include ethics and fairness research, not just technical alignment."
            ))

        # Swing voter contributions
        if r == 2:
            builder.with_action(_char_action(
                "@timnit_gebru", "say",
                "Who defines safety? Include affected communities in the process."
            ))

        # Pressure
        for s in supporters[:2]:
            for o in opponents[:2]:
                builder.with_action(ScenarioAction(
                    action_type="apply_pressure",
                    params={
                        "target": f"@{o}",
                        "source": f"@{s}",
                        "topic": "support_mandate",
                        "intensity": 0.07,
                        "direction": 1,
                    }
                ))

        if r < rounds:
            builder.transition_to(f"round_{r + 1}")
        else:
            builder.transition_to("tiered_proposal")

    # Tiered proposal
    builder.add_phase("tiered_proposal", "Tiered Approach Discussion")

    builder.with_action(_char_action(
        "@demis_hassabis", "propose",
        "Proposal: 30% for frontier labs, 15% for mid-size companies, startups exempt until Series B."
    ))

    builder.with_action(ScenarioAction(
        action_type="negotiate",
        params={
            "characters": [f"@{p}" for p in all_participants],
            "topic": "support_mandate",
            "rounds": 2,
            "agreement_threshold": 0.2,
        }
    ))

    builder.transition_to("final_vote")

    # Final vote
    builder.add_phase("final_vote", "Final Positions")

    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Support tiered safety investment requirements",
            "topic": "support_mandate",
            "characters": [f"@{p}" for p in all_participants],
            "threshold": 0.5,
        }
    ))

    builder.max_duration(300.0)
    return builder.build()


# =============================================================================
# Scenario Registry
# =============================================================================

POLICY_SCENARIO_TEMPLATES = {
    "open_source_ban": create_open_source_ban_debate,
    "moratorium": create_moratorium_debate,
    "safety_mandate": create_safety_mandate_debate,
}


def list_policy_scenario_templates() -> List[str]:
    """List available policy debate templates."""
    return list(POLICY_SCENARIO_TEMPLATES.keys())


def create_policy_scenario(template_name: str, **kwargs) -> ScenarioDefinition:
    """Create a policy debate scenario from template."""
    if template_name not in POLICY_SCENARIO_TEMPLATES:
        raise ValueError(f"Unknown template: {template_name}. Available: {list_policy_scenario_templates()}")
    return POLICY_SCENARIO_TEMPLATES[template_name](**kwargs)
