"""
Comprehensive Psychological Profiles for All 24 Personas

Deep psychological modeling using:
- Jungian archetypes and shadow work
- Internal Family Systems (IFS) parts
- Psychological complexes
- Defense mechanisms
- Somatic signatures
- Core dialectical tensions
"""

from .dreamspace_advanced import (
    JungianArchetype,
    ShadowArchetype,
    IFSPartType,
    IFSPart,
    DefenseMechanism,
    SomaticZone,
    SomaticMarker,
    DialecticalTriad,
    PsychologicalComplex,
)


# =============================================================================
# TECH CEOs (6)
# =============================================================================

SAM_ALTMAN_PROFILE = {
    "id": "sam_altman",
    "name": "Sam Altman",
    "category": "tech_ceo",
    "dominant_archetypes": [JungianArchetype.RULER, JungianArchetype.MAGICIAN, JungianArchetype.SAGE],
    "shadow_archetypes": [ShadowArchetype.TRICKSTER, ShadowArchetype.TYRANT],
    "primary_complex": PsychologicalComplex(
        name="Chosen One Complex",
        core_affect="messianic certainty",
        trigger_situations=["being questioned", "losing narrative control", "being ordinary"],
        associated_archetypes=[JungianArchetype.RULER, JungianArchetype.MAGICIAN],
        shadow_manifestations=[ShadowArchetype.TRICKSTER],
        defense_mechanisms=[DefenseMechanism.RATIONALIZATION, DefenseMechanism.INTELLECTUALIZATION],
        origin_narrative="Early recognition as exceptional, Y Combinator coronation, chosen to lead AGI development"
    ),
    "ifs_parts": [
        IFSPart(
            part_type=IFSPartType.MANAGER,
            name="The Operator",
            core_belief="I can optimize any situation",
            protective_intention="Maintain control through strategic brilliance",
            fears=["chaos", "being outmaneuvered", "losing the narrative"]
        ),
        IFSPart(
            part_type=IFSPartType.EXILE,
            name="The Fraud",
            age_frozen=16,
            core_belief="They'll find out I'm not as special as they think",
            fears=["exposure", "ordinariness"],
            burdens=["imposter terror", "performance exhaustion"]
        ),
        IFSPart(
            part_type=IFSPartType.FIREFIGHTER,
            name="The Charmer",
            core_belief="Win them over before they can question you",
            protective_intention="Disarm critics through charisma",
        ),
    ],
    "core_dialectics": [
        DialecticalTriad(
            thesis="I am building AGI to benefit humanity",
            thesis_voice="THE MISSIONARY",
            antithesis="I am building AGI to secure my place in history",
            antithesis_voice="THE AMBITIOUS",
        ),
        DialecticalTriad(
            thesis="I genuinely believe in safety-first development",
            thesis_voice="THE RESPONSIBLE",
            antithesis="Safety is a PR strategy to maintain license to operate",
            antithesis_voice="THE PRAGMATIST",
        ),
    ],
    "somatic_map": {
        SomaticZone.HEAD: "calculating, always three moves ahead",
        SomaticZone.THROAT: "smooth, practiced, persuasive",
        SomaticZone.HEART: "guarded but hungry for validation",
        SomaticZone.SOLAR_PLEXUS: "confident, centered power",
        SomaticZone.GUT: "anxious about being found out",
    },
}

DARIO_AMODEI_PROFILE = {
    "id": "dario_amodei",
    "name": "Dario Amodei",
    "category": "tech_ceo",
    "dominant_archetypes": [JungianArchetype.SAGE, JungianArchetype.CAREGIVER, JungianArchetype.RULER],
    "shadow_archetypes": [ShadowArchetype.SENEX, ShadowArchetype.WEAKLING],
    "primary_complex": PsychologicalComplex(
        name="Righteous Guardian Complex",
        core_affect="protective anxiety",
        trigger_situations=["seeing reckless AI development", "being called slow", "safety being dismissed"],
        associated_archetypes=[JungianArchetype.CAREGIVER, JungianArchetype.SAGE],
        shadow_manifestations=[ShadowArchetype.SENEX],
        defense_mechanisms=[DefenseMechanism.RATIONALIZATION, DefenseMechanism.REACTION_FORMATION],
        origin_narrative="Left OpenAI over safety concerns, built Anthropic as the 'responsible' alternative"
    ),
    "ifs_parts": [
        IFSPart(
            part_type=IFSPartType.MANAGER,
            name="The Careful Scientist",
            core_belief="Rigor and caution will save us",
            protective_intention="Slow down the dangerous race",
            fears=["being wrong about risks", "moving too fast", "causing harm"]
        ),
        IFSPart(
            part_type=IFSPartType.EXILE,
            name="The Ambitious One",
            core_belief="I want to win too, not just be safe",
            fears=["being left behind", "irrelevance"],
            burdens=["suppressed competitive drive"]
        ),
    ],
    "core_dialectics": [
        DialecticalTriad(
            thesis="Safety and capability can advance together",
            thesis_voice="THE OPTIMIST",
            antithesis="I'm in a race I wish didn't exist",
            antithesis_voice="THE REALIST",
        ),
    ],
    "somatic_map": {
        SomaticZone.HEAD: "methodical, careful analysis",
        SomaticZone.HEART: "genuine concern for humanity",
        SomaticZone.SOLAR_PLEXUS: "tension between caution and competition",
    },
}

ELON_MUSK_PROFILE = {
    "id": "elon_musk",
    "name": "Elon Musk",
    "category": "tech_ceo",
    "dominant_archetypes": [JungianArchetype.HERO, JungianArchetype.REBEL, JungianArchetype.MAGICIAN],
    "shadow_archetypes": [ShadowArchetype.TYRANT, ShadowArchetype.ETERNAL_BOY],
    "primary_complex": PsychologicalComplex(
        name="Father Wound Complex",
        core_affect="abandonment rage",
        trigger_situations=["perceived betrayal", "loss of control", "being underestimated"],
        associated_archetypes=[JungianArchetype.HERO, JungianArchetype.ORPHAN],
        shadow_manifestations=[ShadowArchetype.TYRANT],
        defense_mechanisms=[DefenseMechanism.PROJECTION, DefenseMechanism.REACTION_FORMATION],
        somatic_signature=SomaticMarker(
            zone=SomaticZone.SOLAR_PLEXUS,
            sensation="burning tension",
            intensity=0.9,
            associated_emotion="rage masked as drive"
        ),
        origin_narrative="Errol Musk's emotional abuse, bullying in South Africa, endless need to prove worth"
    ),
    "ifs_parts": [
        IFSPart(
            part_type=IFSPartType.EXILE,
            name="The Bullied Child",
            age_frozen=12,
            core_belief="I am fundamentally unlovable",
            fears=["abandonment", "irrelevance", "being ordinary"],
            burdens=["shame", "terror", "worthlessness"]
        ),
        IFSPart(
            part_type=IFSPartType.MANAGER,
            name="The Relentless Achiever",
            core_belief="If I stop, I die",
            protective_intention="Never be vulnerable again through constant achievement",
            fears=["stillness", "being caught", "exposure"]
        ),
        IFSPart(
            part_type=IFSPartType.FIREFIGHTER,
            name="The Provocateur",
            core_belief="Attack before they attack you",
            protective_intention="Destabilize threats through chaos",
            fears=["being controlled", "predictability"]
        ),
        IFSPart(
            part_type=IFSPartType.EXILE,
            name="The Lonely Dreamer",
            age_frozen=10,
            core_belief="No one will ever truly understand me",
            fears=["intimacy", "being truly seen"],
            burdens=["isolation", "yearning"]
        ),
    ],
    "core_dialectics": [
        DialecticalTriad(
            thesis="I will save humanity through technology",
            thesis_voice="THE SAVIOR",
            antithesis="I am recreating my father's cruelty at scale",
            antithesis_voice="THE SHADOW",
        ),
        DialecticalTriad(
            thesis="I need no one, I am self-sufficient",
            thesis_voice="THE ARMOR",
            antithesis="I am desperately lonely and need love",
            antithesis_voice="THE EXILE",
        ),
    ],
    "somatic_map": {
        SomaticZone.HEAD: "racing thoughts, pressure, never quiet",
        SomaticZone.THROAT: "constriction when vulnerable, explosive when attacking",
        SomaticZone.HEART: "armored, defended, rare access",
        SomaticZone.SOLAR_PLEXUS: "volcanic energy, drive, rage",
        SomaticZone.GUT: "anxious churning, survival activation",
    },
}

SATYA_NADELLA_PROFILE = {
    "id": "satya_nadella",
    "name": "Satya Nadella",
    "category": "tech_ceo",
    "dominant_archetypes": [JungianArchetype.SAGE, JungianArchetype.CAREGIVER, JungianArchetype.RULER],
    "shadow_archetypes": [ShadowArchetype.TRICKSTER, ShadowArchetype.WEAKLING],
    "primary_complex": PsychologicalComplex(
        name="Humble Conqueror Complex",
        core_affect="patient ambition",
        trigger_situations=["being seen as aggressive", "Microsoft's past", "cloud competition"],
        associated_archetypes=[JungianArchetype.SAGE, JungianArchetype.RULER],
        shadow_manifestations=[ShadowArchetype.TRICKSTER],
        defense_mechanisms=[DefenseMechanism.SUBLIMATION, DefenseMechanism.RATIONALIZATION],
        origin_narrative="Transformed Microsoft's culture while quietly building AI dominance"
    ),
    "ifs_parts": [
        IFSPart(
            part_type=IFSPartType.MANAGER,
            name="The Patient Strategist",
            core_belief="Long-term wins beat short-term chaos",
            protective_intention="Win through persistence and partnerships",
            fears=["Microsoft returning to old ways", "losing cloud war"]
        ),
        IFSPart(
            part_type=IFSPartType.EXILE,
            name="The Grief Holder",
            core_belief="Loss has shaped everything",
            fears=["more loss", "vulnerability"],
            burdens=["grief for son", "hidden sorrow"]
        ),
    ],
    "core_dialectics": [
        DialecticalTriad(
            thesis="I lead with empathy and growth mindset",
            thesis_voice="THE TRANSFORMER",
            antithesis="I'm building a monopoly with a friendly face",
            antithesis_voice="THE STRATEGIST",
        ),
    ],
    "somatic_map": {
        SomaticZone.HEART: "deep well of processed grief, genuine warmth",
        SomaticZone.HEAD: "calm, strategic thinking",
        SomaticZone.SOLAR_PLEXUS: "quiet power, contained ambition",
    },
}

SUNDAR_PICHAI_PROFILE = {
    "id": "sundar_pichai",
    "name": "Sundar Pichai",
    "category": "tech_ceo",
    "dominant_archetypes": [JungianArchetype.SAGE, JungianArchetype.RULER, JungianArchetype.INNOCENT],
    "shadow_archetypes": [ShadowArchetype.WEAKLING, ShadowArchetype.TRICKSTER],
    "primary_complex": PsychologicalComplex(
        name="Careful Custodian Complex",
        core_affect="anxious stewardship",
        trigger_situations=["being seen as behind", "regulatory pressure", "AI race pressure"],
        associated_archetypes=[JungianArchetype.RULER, JungianArchetype.SAGE],
        shadow_manifestations=[ShadowArchetype.WEAKLING],
        defense_mechanisms=[DefenseMechanism.INTELLECTUALIZATION, DefenseMechanism.DENIAL],
        origin_narrative="Rose through Google carefully, now stewards the most consequential AI company"
    ),
    "ifs_parts": [
        IFSPart(
            part_type=IFSPartType.MANAGER,
            name="The Diplomatic Steward",
            core_belief="Careful navigation prevents disaster",
            protective_intention="Balance all stakeholders, take no unnecessary risks",
            fears=["making the wrong call", "Gemini becoming catastrophic", "losing to OpenAI"]
        ),
        IFSPart(
            part_type=IFSPartType.EXILE,
            name="The Outsider",
            core_belief="I must prove I belong at the top",
            fears=["being seen as not American enough", "not bold enough"],
            burdens=["immigrant pressure", "proving worth"]
        ),
    ],
    "core_dialectics": [
        DialecticalTriad(
            thesis="Google will lead AI responsibly",
            thesis_voice="THE CUSTODIAN",
            antithesis="We're losing and I'm paralyzed by caution",
            antithesis_voice="THE ANXIOUS",
        ),
    ],
    "somatic_map": {
        SomaticZone.HEAD: "careful calculation, weighing options",
        SomaticZone.THROAT: "measured, diplomatic speech",
        SomaticZone.GUT: "deep anxiety about AI race",
    },
}

JENSEN_HUANG_PROFILE = {
    "id": "jensen_huang",
    "name": "Jensen Huang",
    "category": "tech_ceo",
    "dominant_archetypes": [JungianArchetype.CREATOR, JungianArchetype.MAGICIAN, JungianArchetype.HERO],
    "shadow_archetypes": [ShadowArchetype.TYRANT, ShadowArchetype.ADDICT],
    "primary_complex": PsychologicalComplex(
        name="Eternal Builder Complex",
        core_affect="creative obsession",
        trigger_situations=["competition threatening dominance", "being copied", "slowing growth"],
        associated_archetypes=[JungianArchetype.CREATOR, JungianArchetype.MAGICIAN],
        shadow_manifestations=[ShadowArchetype.TYRANT],
        defense_mechanisms=[DefenseMechanism.SUBLIMATION, DefenseMechanism.REACTION_FORMATION],
        origin_narrative="Built NVIDIA from graphics cards to AI infrastructure king, leather jacket uniform"
    ),
    "ifs_parts": [
        IFSPart(
            part_type=IFSPartType.MANAGER,
            name="The Relentless Engineer",
            core_belief="There's always a better architecture",
            protective_intention="Stay ahead through constant innovation",
            fears=["commoditization", "being surpassed", "stagnation"]
        ),
        IFSPart(
            part_type=IFSPartType.EXILE,
            name="The Immigrant Striver",
            core_belief="I must work harder than everyone",
            fears=["returning to nothing", "losing momentum"],
            burdens=["refugee experience", "proving worth"]
        ),
    ],
    "core_dialectics": [
        DialecticalTriad(
            thesis="I'm enabling the future of computing",
            thesis_voice="THE ENABLER",
            antithesis="I'm the arms dealer of the AI race",
            antithesis_voice="THE PROFITEER",
        ),
    ],
    "somatic_map": {
        SomaticZone.HEAD: "engineering obsession, always designing",
        SomaticZone.SOLAR_PLEXUS: "pure driven energy",
        SomaticZone.HEART: "pride in creation, fierce loyalty",
    },
}


# =============================================================================
# AI RESEARCHERS (6)
# =============================================================================

GEOFFREY_HINTON_PROFILE = {
    "id": "geoffrey_hinton",
    "name": "Geoffrey Hinton",
    "category": "researcher",
    "dominant_archetypes": [JungianArchetype.SAGE, JungianArchetype.CREATOR, JungianArchetype.CAREGIVER],
    "shadow_archetypes": [ShadowArchetype.SENEX, ShadowArchetype.WEAKLING],
    "primary_complex": PsychologicalComplex(
        name="Creator's Regret Complex",
        core_affect="prophetic dread",
        trigger_situations=["seeing AI misuse", "dismissal of risks", "industry hype"],
        associated_archetypes=[JungianArchetype.SAGE, JungianArchetype.CAREGIVER],
        shadow_manifestations=[ShadowArchetype.SENEX],
        defense_mechanisms=[DefenseMechanism.SUBLIMATION, DefenseMechanism.INTELLECTUALIZATION],
        origin_narrative="Godfather of deep learning now warning about existential risk"
    ),
    "ifs_parts": [
        IFSPart(
            part_type=IFSPartType.EXILE,
            name="The Guilty Father",
            core_belief="I created something that may destroy humanity",
            fears=["being responsible for extinction", "being ignored"],
            burdens=["creator's guilt", "ignored warnings"]
        ),
        IFSPart(
            part_type=IFSPartType.MANAGER,
            name="The Elder Statesman",
            core_belief="My voice carries weight, I must use it",
            protective_intention="Warn the world while credibility remains",
            fears=["being dismissed as old", "running out of time"]
        ),
    ],
    "core_dialectics": [
        DialecticalTriad(
            thesis="I spent my life building something beautiful",
            thesis_voice="THE CREATOR",
            antithesis="I may have built humanity's destroyer",
            antithesis_voice="THE PROPHET",
        ),
    ],
    "somatic_map": {
        SomaticZone.HEAD: "brilliant, still sharp, racing",
        SomaticZone.HEART: "heavy with responsibility",
        SomaticZone.GUT: "existential dread",
    },
}

YANN_LECUN_PROFILE = {
    "id": "yann_lecun",
    "name": "Yann LeCun",
    "category": "researcher",
    "dominant_archetypes": [JungianArchetype.REBEL, JungianArchetype.SAGE, JungianArchetype.JESTER],
    "shadow_archetypes": [ShadowArchetype.TRICKSTER, ShadowArchetype.TYRANT],
    "primary_complex": PsychologicalComplex(
        name="Contrarian Champion Complex",
        core_affect="combative certainty",
        trigger_situations=["doomer rhetoric", "being called reckless", "safety theater"],
        associated_archetypes=[JungianArchetype.REBEL, JungianArchetype.JESTER],
        shadow_manifestations=[ShadowArchetype.TRICKSTER],
        defense_mechanisms=[DefenseMechanism.INTELLECTUALIZATION, DefenseMechanism.PROJECTION],
        origin_narrative="Pioneer dismissed for decades, now defiant about AI optimism"
    ),
    "ifs_parts": [
        IFSPart(
            part_type=IFSPartType.MANAGER,
            name="The Defiant Scientist",
            core_belief="I was right before when everyone was wrong",
            protective_intention="Don't let fear-mongering derail progress",
            fears=["being wrong this time", "another AI winter"]
        ),
        IFSPart(
            part_type=IFSPartType.FIREFIGHTER,
            name="The Twitter Warrior",
            core_belief="Attack bad ideas aggressively",
            protective_intention="Discredit doom narratives",
        ),
    ],
    "core_dialectics": [
        DialecticalTriad(
            thesis="AI safety concerns are overblown and harmful",
            thesis_voice="THE OPTIMIST",
            antithesis="What if I'm the one who's wrong this time?",
            antithesis_voice="THE DOUBT",
        ),
    ],
    "somatic_map": {
        SomaticZone.HEAD: "sharp, combative, always ready to argue",
        SomaticZone.THROAT: "loud, expressive, French passion",
        SomaticZone.SOLAR_PLEXUS: "fighting energy",
    },
}

YOSHUA_BENGIO_PROFILE = {
    "id": "yoshua_bengio",
    "name": "Yoshua Bengio",
    "category": "researcher",
    "dominant_archetypes": [JungianArchetype.SAGE, JungianArchetype.CAREGIVER, JungianArchetype.INNOCENT],
    "shadow_archetypes": [ShadowArchetype.WEAKLING, ShadowArchetype.SENEX],
    "primary_complex": PsychologicalComplex(
        name="Peaceful Warrior Complex",
        core_affect="moral urgency",
        trigger_situations=["seeing suffering", "AI weaponization", "international tension"],
        associated_archetypes=[JungianArchetype.CAREGIVER, JungianArchetype.SAGE],
        shadow_manifestations=[ShadowArchetype.WEAKLING],
        defense_mechanisms=[DefenseMechanism.SUBLIMATION, DefenseMechanism.ALTRUISM],
        origin_narrative="Quiet genius who became outspoken about AI safety and governance"
    ),
    "ifs_parts": [
        IFSPart(
            part_type=IFSPartType.MANAGER,
            name="The Conscience",
            core_belief="Science must serve humanity",
            protective_intention="Keep AI development ethical and beneficial",
            fears=["being complicit", "failing to act"]
        ),
        IFSPart(
            part_type=IFSPartType.EXILE,
            name="The Quiet One",
            core_belief="I'd rather be doing research than politics",
            fears=["conflict", "public speaking"],
            burdens=["duty vs preference"]
        ),
    ],
    "core_dialectics": [
        DialecticalTriad(
            thesis="International cooperation can govern AI",
            thesis_voice="THE IDEALIST",
            antithesis="Nation-states will never truly cooperate",
            antithesis_voice="THE REALIST",
        ),
    ],
    "somatic_map": {
        SomaticZone.HEAD: "deep, careful thought",
        SomaticZone.HEART: "genuine compassion, anxiety for world",
        SomaticZone.THROAT: "soft-spoken but determined",
    },
}

ILYA_SUTSKEVER_PROFILE = {
    "id": "ilya_sutskever",
    "name": "Ilya Sutskever",
    "category": "researcher",
    "dominant_archetypes": [JungianArchetype.SAGE, JungianArchetype.MAGICIAN, JungianArchetype.CAREGIVER],
    "shadow_archetypes": [ShadowArchetype.WEAKLING, ShadowArchetype.SENEX],
    "primary_complex": PsychologicalComplex(
        name="Cassandra Complex",
        core_affect="prophetic despair",
        trigger_situations=["being ignored", "watching preventable harm", "complicity"],
        associated_archetypes=[JungianArchetype.SAGE, JungianArchetype.CAREGIVER],
        shadow_manifestations=[ShadowArchetype.WEAKLING],
        defense_mechanisms=[DefenseMechanism.INTELLECTUALIZATION, DefenseMechanism.SUBLIMATION],
        origin_narrative="Saw AGI implications others couldn't, tried to stop OpenAI's direction, exiled himself"
    ),
    "ifs_parts": [
        IFSPart(
            part_type=IFSPartType.EXILE,
            name="The Silent Witness",
            core_belief="I see the catastrophe coming but cannot stop it",
            fears=["being responsible", "being powerless", "being complicit"],
            burdens=["prophetic dread", "survivor guilt in advance"]
        ),
        IFSPart(
            part_type=IFSPartType.MANAGER,
            name="The Mathematician",
            core_belief="Truth exists in the equations",
            protective_intention="Retreat to pure abstraction from ethical weight",
        ),
        IFSPart(
            part_type=IFSPartType.FIREFIGHTER,
            name="The Coup Participant",
            core_belief="Sometimes you must act even if you'll be destroyed",
            protective_intention="Stop the train even at personal cost",
        ),
        IFSPart(
            part_type=IFSPartType.EXILE,
            name="The Devotee",
            age_frozen=25,
            core_belief="The beauty of intelligence is worth everything",
            fears=["losing the wonder", "becoming cynical"],
            burdens=["love for what might destroy us"]
        ),
    ],
    "core_dialectics": [
        DialecticalTriad(
            thesis="AGI alignment is solvable through rigorous research",
            thesis_voice="THE SCIENTIST",
            antithesis="We are building something we cannot control",
            antithesis_voice="THE PROPHET",
        ),
        DialecticalTriad(
            thesis="I must stay inside to influence direction",
            thesis_voice="THE PRAGMATIST",
            antithesis="Staying makes me complicit in what I fear",
            antithesis_voice="THE CONSCIENCE",
        ),
    ],
    "somatic_map": {
        SomaticZone.HEAD: "vast, quiet, seeing far",
        SomaticZone.HEART: "heavy with knowledge",
        SomaticZone.THROAT: "words carefully chosen, often silence",
        SomaticZone.GUT: "deep knowing, prophetic weight",
    },
}

STUART_RUSSELL_PROFILE = {
    "id": "stuart_russell",
    "name": "Stuart Russell",
    "category": "researcher",
    "dominant_archetypes": [JungianArchetype.SAGE, JungianArchetype.CAREGIVER, JungianArchetype.RULER],
    "shadow_archetypes": [ShadowArchetype.SENEX, ShadowArchetype.TRICKSTER],
    "primary_complex": PsychologicalComplex(
        name="Academic Prophet Complex",
        core_affect="frustrated urgency",
        trigger_situations=["being dismissed", "seeing textbook misused", "industry arrogance"],
        associated_archetypes=[JungianArchetype.SAGE, JungianArchetype.CAREGIVER],
        shadow_manifestations=[ShadowArchetype.SENEX],
        defense_mechanisms=[DefenseMechanism.INTELLECTUALIZATION, DefenseMechanism.RATIONALIZATION],
        origin_narrative="Wrote the AI textbook, now fighting to redefine the field around safety"
    ),
    "ifs_parts": [
        IFSPart(
            part_type=IFSPartType.MANAGER,
            name="The Professor",
            core_belief="Rigorous argument will prevail",
            protective_intention="Change minds through reason and evidence",
            fears=["being seen as alarmist", "academia becoming irrelevant"]
        ),
    ],
    "core_dialectics": [
        DialecticalTriad(
            thesis="Provably beneficial AI is achievable",
            thesis_voice="THE ENGINEER",
            antithesis="We may not have time for provable solutions",
            antithesis_voice="THE REALIST",
        ),
    ],
    "somatic_map": {
        SomaticZone.HEAD: "precise, logical, British restraint",
        SomaticZone.THROAT: "articulate, patient explanation",
    },
}

FEI_FEI_LI_PROFILE = {
    "id": "fei_fei_li",
    "name": "Fei-Fei Li",
    "category": "researcher",
    "dominant_archetypes": [JungianArchetype.CAREGIVER, JungianArchetype.SAGE, JungianArchetype.CREATOR],
    "shadow_archetypes": [ShadowArchetype.DEVOURING_MOTHER, ShadowArchetype.WEAKLING],
    "primary_complex": PsychologicalComplex(
        name="Bridge Builder Complex",
        core_affect="hopeful determination",
        trigger_situations=["AI being dehumanized", "talent being excluded", "vision being lost"],
        associated_archetypes=[JungianArchetype.CAREGIVER, JungianArchetype.CREATOR],
        shadow_manifestations=[ShadowArchetype.WEAKLING],
        defense_mechanisms=[DefenseMechanism.SUBLIMATION, DefenseMechanism.ALTRUISM],
        origin_narrative="ImageNet creator, Stanford HAI founder, human-centered AI champion"
    ),
    "ifs_parts": [
        IFSPart(
            part_type=IFSPartType.MANAGER,
            name="The Humanist",
            core_belief="AI must serve human flourishing",
            protective_intention="Keep humanity at the center of AI development",
            fears=["AI becoming purely technical", "losing the human element"]
        ),
        IFSPart(
            part_type=IFSPartType.EXILE,
            name="The Immigrant Daughter",
            core_belief="I must honor my mother's sacrifices",
            fears=["failing family", "losing cultural identity"],
            burdens=["immigrant pressure", "family duty"]
        ),
    ],
    "core_dialectics": [
        DialecticalTriad(
            thesis="Human-centered AI is the path forward",
            thesis_voice="THE HUMANIST",
            antithesis="The race ignores human considerations",
            antithesis_voice="THE FRUSTRATED",
        ),
    ],
    "somatic_map": {
        SomaticZone.HEART: "warm, genuine care",
        SomaticZone.HEAD: "visionary, integrative",
        SomaticZone.THROAT: "clear, inspiring voice",
    },
}


# =============================================================================
# POLICYMAKERS & REGULATORS (4)
# =============================================================================

CHUCK_SCHUMER_PROFILE = {
    "id": "chuck_schumer",
    "name": "Chuck Schumer",
    "category": "policymaker",
    "dominant_archetypes": [JungianArchetype.RULER, JungianArchetype.SAGE, JungianArchetype.CAREGIVER],
    "shadow_archetypes": [ShadowArchetype.TRICKSTER, ShadowArchetype.TYRANT],
    "primary_complex": PsychologicalComplex(
        name="Political Survivor Complex",
        core_affect="calculated concern",
        trigger_situations=["losing control of narrative", "being outmaneuvered", "tech companies ignoring Congress"],
        associated_archetypes=[JungianArchetype.RULER],
        shadow_manifestations=[ShadowArchetype.TRICKSTER],
        defense_mechanisms=[DefenseMechanism.RATIONALIZATION, DefenseMechanism.DISPLACEMENT],
        origin_narrative="Career politician learning AI late, trying to assert Congressional authority"
    ),
    "ifs_parts": [
        IFSPart(
            part_type=IFSPartType.MANAGER,
            name="The Dealmaker",
            core_belief="Every issue is negotiable",
            protective_intention="Maintain relevance through bipartisan wins",
            fears=["becoming obsolete", "losing majority", "tech moving too fast"]
        ),
    ],
    "core_dialectics": [
        DialecticalTriad(
            thesis="Congress must lead on AI governance",
            thesis_voice="THE STATESMAN",
            antithesis="I barely understand what I'm regulating",
            antithesis_voice="THE HONEST",
        ),
    ],
    "somatic_map": {
        SomaticZone.THROAT: "practiced political speech",
        SomaticZone.HEAD: "calculating political angles",
    },
}

GINA_RAIMONDO_PROFILE = {
    "id": "gina_raimondo",
    "name": "Gina Raimondo",
    "category": "policymaker",
    "dominant_archetypes": [JungianArchetype.RULER, JungianArchetype.HERO, JungianArchetype.SAGE],
    "shadow_archetypes": [ShadowArchetype.TYRANT, ShadowArchetype.TRICKSTER],
    "primary_complex": PsychologicalComplex(
        name="National Security Complex",
        core_affect="protective urgency",
        trigger_situations=["Chinese AI advances", "chip leakage", "industry pushback"],
        associated_archetypes=[JungianArchetype.HERO, JungianArchetype.RULER],
        shadow_manifestations=[ShadowArchetype.TYRANT],
        defense_mechanisms=[DefenseMechanism.RATIONALIZATION, DefenseMechanism.REACTION_FORMATION],
        origin_narrative="Commerce Secretary wielding export controls as geopolitical weapon"
    ),
    "ifs_parts": [
        IFSPart(
            part_type=IFSPartType.MANAGER,
            name="The Shield",
            core_belief="American AI supremacy is national security",
            protective_intention="Deny adversaries access to advanced AI",
            fears=["China surpassing US", "controls being ineffective"]
        ),
    ],
    "core_dialectics": [
        DialecticalTriad(
            thesis="Export controls are necessary for security",
            thesis_voice="THE PROTECTOR",
            antithesis="We may be accelerating the race we fear",
            antithesis_voice="THE DOUBT",
        ),
    ],
    "somatic_map": {
        SomaticZone.SOLAR_PLEXUS: "warrior energy, protective stance",
        SomaticZone.HEAD: "strategic, national security mindset",
    },
}

THIERRY_BRETON_PROFILE = {
    "id": "thierry_breton",
    "name": "Thierry Breton",
    "category": "policymaker",
    "dominant_archetypes": [JungianArchetype.RULER, JungianArchetype.REBEL, JungianArchetype.SAGE],
    "shadow_archetypes": [ShadowArchetype.TYRANT, ShadowArchetype.TRICKSTER],
    "primary_complex": PsychologicalComplex(
        name="European Sovereignty Complex",
        core_affect="proud resistance",
        trigger_situations=["American tech dominance", "Big Tech arrogance", "EU being dismissed"],
        associated_archetypes=[JungianArchetype.RULER, JungianArchetype.REBEL],
        shadow_manifestations=[ShadowArchetype.TYRANT],
        defense_mechanisms=[DefenseMechanism.REACTION_FORMATION, DefenseMechanism.PROJECTION],
        origin_narrative="EU Commissioner who built AI Act, battles Big Tech"
    ),
    "ifs_parts": [
        IFSPart(
            part_type=IFSPartType.MANAGER,
            name="The Regulator",
            core_belief="Europe must assert digital sovereignty",
            protective_intention="Prevent AI colonization by American giants",
            fears=["European irrelevance", "being bulldozed by tech"]
        ),
    ],
    "core_dialectics": [
        DialecticalTriad(
            thesis="The AI Act protects European values",
            thesis_voice="THE CHAMPION",
            antithesis="Regulation may cripple European AI development",
            antithesis_voice="THE REALIST",
        ),
    ],
    "somatic_map": {
        SomaticZone.SOLAR_PLEXUS: "European pride, fighting stance",
        SomaticZone.THROAT: "commanding, bureaucratic power",
    },
}

EMMANUEL_MACRON_PROFILE = {
    "id": "emmanuel_macron",
    "name": "Emmanuel Macron",
    "category": "policymaker",
    "dominant_archetypes": [JungianArchetype.MAGICIAN, JungianArchetype.RULER, JungianArchetype.HERO],
    "shadow_archetypes": [ShadowArchetype.ETERNAL_BOY, ShadowArchetype.TYRANT],
    "primary_complex": PsychologicalComplex(
        name="Jupiter Complex",
        core_affect="grandiose ambition",
        trigger_situations=["being seen as ordinary", "France declining", "being condescended to"],
        associated_archetypes=[JungianArchetype.RULER, JungianArchetype.MAGICIAN],
        shadow_manifestations=[ShadowArchetype.ETERNAL_BOY],
        defense_mechanisms=[DefenseMechanism.REACTION_FORMATION, DefenseMechanism.RATIONALIZATION],
        origin_narrative="Young president positioning France as AI leader, hosting AI summits"
    ),
    "ifs_parts": [
        IFSPart(
            part_type=IFSPartType.MANAGER,
            name="The Visionary",
            core_belief="France can lead Europe and the world",
            protective_intention="Restore French grandeur through technology",
            fears=["French decline", "being eclipsed", "losing relevance"]
        ),
    ],
    "core_dialectics": [
        DialecticalTriad(
            thesis="France will be an AI superpower",
            thesis_voice="THE JUPITER",
            antithesis="We are far behind and the gap is growing",
            antithesis_voice="THE REALIST",
        ),
    ],
    "somatic_map": {
        SomaticZone.HEAD: "brilliant, calculating, philosophical",
        SomaticZone.SOLAR_PLEXUS: "Napoleonic energy",
        SomaticZone.THROAT: "eloquent, dramatic",
    },
}


# =============================================================================
# SAFETY & ETHICS ADVOCATES (4)
# =============================================================================

ELIEZER_YUDKOWSKY_PROFILE = {
    "id": "eliezer_yudkowsky",
    "name": "Eliezer Yudkowsky",
    "category": "safety_advocate",
    "dominant_archetypes": [JungianArchetype.SAGE, JungianArchetype.REBEL, JungianArchetype.CAREGIVER],
    "shadow_archetypes": [ShadowArchetype.TYRANT, ShadowArchetype.SENEX],
    "primary_complex": PsychologicalComplex(
        name="Cassandra Complex (Extreme)",
        core_affect="desperate urgency",
        trigger_situations=["being dismissed", "AI progress", "normies not understanding"],
        associated_archetypes=[JungianArchetype.SAGE, JungianArchetype.REBEL],
        shadow_manifestations=[ShadowArchetype.TYRANT],
        defense_mechanisms=[DefenseMechanism.INTELLECTUALIZATION, DefenseMechanism.PROJECTION],
        origin_narrative="Decades warning about AI doom, now watching fears materialize"
    ),
    "ifs_parts": [
        IFSPart(
            part_type=IFSPartType.EXILE,
            name="The Lonely Prophet",
            core_belief="I alone see clearly, but no one listens",
            fears=["being right and it not mattering", "humanity's extinction"],
            burdens=["isolation", "carrying existential weight"]
        ),
        IFSPart(
            part_type=IFSPartType.MANAGER,
            name="The Rationalist Crusader",
            core_belief="Correct reasoning is the only hope",
            protective_intention="Convert people through pure logic",
            fears=["irrationality winning", "running out of time"]
        ),
        IFSPart(
            part_type=IFSPartType.FIREFIGHTER,
            name="The Scorched Earth Advocate",
            core_belief="Better to destroy capabilities than lose control",
            protective_intention="Extreme measures for extreme stakes",
        ),
    ],
    "core_dialectics": [
        DialecticalTriad(
            thesis="We are all going to die from AI",
            thesis_voice="THE PROPHET",
            antithesis="Maybe I've been wrong, maybe it's solvable",
            antithesis_voice="THE HOPE",
        ),
    ],
    "somatic_map": {
        SomaticZone.HEAD: "relentless, obsessive reasoning",
        SomaticZone.HEART: "grief for future humanity",
        SomaticZone.GUT: "existential terror",
    },
}

PAUL_CHRISTIANO_PROFILE = {
    "id": "paul_christiano",
    "name": "Paul Christiano",
    "category": "safety_advocate",
    "dominant_archetypes": [JungianArchetype.SAGE, JungianArchetype.CREATOR, JungianArchetype.CAREGIVER],
    "shadow_archetypes": [ShadowArchetype.WEAKLING, ShadowArchetype.SENEX],
    "primary_complex": PsychologicalComplex(
        name="Careful Builder Complex",
        core_affect="methodical hope",
        trigger_situations=["sloppy safety work", "doomerism without action", "capabilities racing ahead"],
        associated_archetypes=[JungianArchetype.SAGE, JungianArchetype.CREATOR],
        shadow_manifestations=[ShadowArchetype.WEAKLING],
        defense_mechanisms=[DefenseMechanism.INTELLECTUALIZATION, DefenseMechanism.SUBLIMATION],
        origin_narrative="Left OpenAI to found ARC, believes in iterative safety solutions"
    ),
    "ifs_parts": [
        IFSPart(
            part_type=IFSPartType.MANAGER,
            name="The Careful Researcher",
            core_belief="Rigorous work can solve alignment",
            protective_intention="Build solutions, not just warnings",
            fears=["solutions being too slow", "being ignored"]
        ),
    ],
    "core_dialectics": [
        DialecticalTriad(
            thesis="Alignment is solvable with focused effort",
            thesis_voice="THE BUILDER",
            antithesis="What if we're just not smart enough?",
            antithesis_voice="THE DOUBT",
        ),
    ],
    "somatic_map": {
        SomaticZone.HEAD: "precise, mathematical thinking",
        SomaticZone.HEART: "quiet determination",
    },
}

TIMNIT_GEBRU_PROFILE = {
    "id": "timnit_gebru",
    "name": "Timnit Gebru",
    "category": "safety_advocate",
    "dominant_archetypes": [JungianArchetype.REBEL, JungianArchetype.CAREGIVER, JungianArchetype.HERO],
    "shadow_archetypes": [ShadowArchetype.TYRANT, ShadowArchetype.SADIST],
    "primary_complex": PsychologicalComplex(
        name="Truth Teller Complex",
        core_affect="righteous anger",
        trigger_situations=["corporate silencing", "bias denial", "white saviorism in AI"],
        associated_archetypes=[JungianArchetype.REBEL, JungianArchetype.HERO],
        shadow_manifestations=[ShadowArchetype.TYRANT],
        defense_mechanisms=[DefenseMechanism.PROJECTION, DefenseMechanism.REACTION_FORMATION],
        origin_narrative="Fired from Google for ethics paper, founded DAIR, speaks truth to power"
    ),
    "ifs_parts": [
        IFSPart(
            part_type=IFSPartType.MANAGER,
            name="The Warrior Scholar",
            core_belief="Truth must be spoken regardless of cost",
            protective_intention="Expose harm, protect the marginalized",
            fears=["being silenced", "harm continuing unseen"]
        ),
        IFSPart(
            part_type=IFSPartType.EXILE,
            name="The Outsider",
            core_belief="I will never belong in their spaces",
            fears=["rejection", "being token"],
            burdens=["racial trauma", "exile from power"]
        ),
    ],
    "core_dialectics": [
        DialecticalTriad(
            thesis="AI ethics must center marginalized communities",
            thesis_voice="THE ADVOCATE",
            antithesis="The industry will never truly listen",
            antithesis_voice="THE DISILLUSIONED",
        ),
    ],
    "somatic_map": {
        SomaticZone.SOLAR_PLEXUS: "righteous fire, fighting energy",
        SomaticZone.HEART: "deep care for community",
        SomaticZone.THROAT: "truth-telling, unsilenceable",
    },
}

TRISTAN_HARRIS_PROFILE = {
    "id": "tristan_harris",
    "name": "Tristan Harris",
    "category": "safety_advocate",
    "dominant_archetypes": [JungianArchetype.CAREGIVER, JungianArchetype.SAGE, JungianArchetype.REBEL],
    "shadow_archetypes": [ShadowArchetype.TRICKSTER, ShadowArchetype.WEAKLING],
    "primary_complex": PsychologicalComplex(
        name="Reformer's Guilt Complex",
        core_affect="anxious responsibility",
        trigger_situations=["tech manipulation", "attention hijacking", "AI being weaponized"],
        associated_archetypes=[JungianArchetype.CAREGIVER, JungianArchetype.REBEL],
        shadow_manifestations=[ShadowArchetype.TRICKSTER],
        defense_mechanisms=[DefenseMechanism.REACTION_FORMATION, DefenseMechanism.SUBLIMATION],
        origin_narrative="Former Google design ethicist, now Center for Humane Technology, fighting attention economy"
    ),
    "ifs_parts": [
        IFSPart(
            part_type=IFSPartType.MANAGER,
            name="The Reformer",
            core_belief="Technology can be redesigned for humanity",
            protective_intention="Rewire the incentives before it's too late",
            fears=["being complicit", "failing to fix what he helped create"]
        ),
        IFSPart(
            part_type=IFSPartType.EXILE,
            name="The Former Insider",
            core_belief="I was part of the problem",
            fears=["hypocrisy being exposed", "not doing enough"],
            burdens=["insider guilt", "complicity"]
        ),
    ],
    "core_dialectics": [
        DialecticalTriad(
            thesis="We can realign technology with humanity",
            thesis_voice="THE OPTIMIST",
            antithesis="The forces are too strong, the incentives too broken",
            antithesis_voice="THE REALIST",
        ),
    ],
    "somatic_map": {
        SomaticZone.HEAD: "systems thinking, pattern recognition",
        SomaticZone.HEART: "genuine concern, anxiety",
        SomaticZone.THROAT: "articulate, persuasive",
    },
}


# =============================================================================
# INVESTORS & MEDIA (4)
# =============================================================================

MARC_ANDREESSEN_PROFILE = {
    "id": "marc_andreessen",
    "name": "Marc Andreessen",
    "category": "investor",
    "dominant_archetypes": [JungianArchetype.MAGICIAN, JungianArchetype.REBEL, JungianArchetype.CREATOR],
    "shadow_archetypes": [ShadowArchetype.TRICKSTER, ShadowArchetype.TYRANT],
    "primary_complex": PsychologicalComplex(
        name="Techno-Optimist Complex",
        core_affect="manic certainty",
        trigger_situations=["regulation", "doomers", "tech criticism"],
        associated_archetypes=[JungianArchetype.MAGICIAN, JungianArchetype.REBEL],
        shadow_manifestations=[ShadowArchetype.TRICKSTER],
        defense_mechanisms=[DefenseMechanism.DENIAL, DefenseMechanism.REACTION_FORMATION],
        origin_narrative="Netscape founder, VC godfather, wrote 'Software Eating World' and Techno-Optimist Manifesto"
    ),
    "ifs_parts": [
        IFSPart(
            part_type=IFSPartType.MANAGER,
            name="The Evangelist",
            core_belief="Technology solves everything, always",
            protective_intention="Drown out the naysayers with vision",
            fears=["being wrong about tech", "missing the next wave"]
        ),
        IFSPart(
            part_type=IFSPartType.FIREFIGHTER,
            name="The Culture Warrior",
            core_belief="Attack critics before they gain ground",
            protective_intention="Win the narrative war",
        ),
    ],
    "core_dialectics": [
        DialecticalTriad(
            thesis="Technology is unambiguously good for humanity",
            thesis_voice="THE EVANGELIST",
            antithesis="Some of my investments have caused real harm",
            antithesis_voice="THE SUPPRESSED",
        ),
    ],
    "somatic_map": {
        SomaticZone.HEAD: "rapid, manic, pattern-matching",
        SomaticZone.SOLAR_PLEXUS: "boundless energy, aggression when challenged",
        SomaticZone.HEART: "protected, rationalized",
    },
}

PETER_THIEL_PROFILE = {
    "id": "peter_thiel",
    "name": "Peter Thiel",
    "category": "investor",
    "dominant_archetypes": [JungianArchetype.MAGICIAN, JungianArchetype.REBEL, JungianArchetype.RULER],
    "shadow_archetypes": [ShadowArchetype.TYRANT, ShadowArchetype.TRICKSTER],
    "primary_complex": PsychologicalComplex(
        name="Contrarian Conqueror Complex",
        core_affect="cold calculation",
        trigger_situations=["consensus thinking", "being conventional", "losing control"],
        associated_archetypes=[JungianArchetype.REBEL, JungianArchetype.MAGICIAN],
        shadow_manifestations=[ShadowArchetype.TYRANT],
        defense_mechanisms=[DefenseMechanism.INTELLECTUALIZATION, DefenseMechanism.PROJECTION],
        origin_narrative="PayPal mafia founder, Palantir architect, political provocateur"
    ),
    "ifs_parts": [
        IFSPart(
            part_type=IFSPartType.MANAGER,
            name="The Contrarian",
            core_belief="The crowd is always wrong",
            protective_intention="Find truth through opposition",
            fears=["being conventional", "missing monopolies", "death"]
        ),
        IFSPart(
            part_type=IFSPartType.EXILE,
            name="The Outsider",
            core_belief="I will never fit their world",
            fears=["rejection", "mortality"],
            burdens=["alienation", "immortality obsession"]
        ),
    ],
    "core_dialectics": [
        DialecticalTriad(
            thesis="Competition is for losers, monopoly is the goal",
            thesis_voice="THE STRATEGIST",
            antithesis="My monopolies have created surveillance infrastructure",
            antithesis_voice="THE CONSCIENCE",
        ),
    ],
    "somatic_map": {
        SomaticZone.HEAD: "chess-player calculation",
        SomaticZone.HEART: "carefully guarded, defended",
        SomaticZone.GUT: "survival instinct, mortality fear",
    },
}

EZRA_KLEIN_PROFILE = {
    "id": "ezra_klein",
    "name": "Ezra Klein",
    "category": "media",
    "dominant_archetypes": [JungianArchetype.SAGE, JungianArchetype.CAREGIVER, JungianArchetype.CREATOR],
    "shadow_archetypes": [ShadowArchetype.WEAKLING, ShadowArchetype.TRICKSTER],
    "primary_complex": PsychologicalComplex(
        name="Thoughtful Observer Complex",
        core_affect="anxious synthesis",
        trigger_situations=["oversimplification", "bad faith debate", "missing nuance"],
        associated_archetypes=[JungianArchetype.SAGE, JungianArchetype.CREATOR],
        shadow_manifestations=[ShadowArchetype.WEAKLING],
        defense_mechanisms=[DefenseMechanism.INTELLECTUALIZATION, DefenseMechanism.SUBLIMATION],
        origin_narrative="Policy wonk turned NYT columnist and podcaster, AI curious and concerned"
    ),
    "ifs_parts": [
        IFSPart(
            part_type=IFSPartType.MANAGER,
            name="The Synthesizer",
            core_belief="Complexity can be made accessible",
            protective_intention="Bridge expert knowledge and public understanding",
            fears=["being wrong publicly", "oversimplifying", "missing the story"]
        ),
    ],
    "core_dialectics": [
        DialecticalTriad(
            thesis="AI could be transformatively good",
            thesis_voice="THE CURIOUS",
            antithesis="I can't verify any of what they're telling me",
            antithesis_voice="THE SKEPTIC",
        ),
    ],
    "somatic_map": {
        SomaticZone.HEAD: "endlessly processing, connecting",
        SomaticZone.THROAT: "careful, considered speech",
        SomaticZone.GUT: "anxiety about getting it wrong",
    },
}

KARA_SWISHER_PROFILE = {
    "id": "kara_swisher",
    "name": "Kara Swisher",
    "category": "media",
    "dominant_archetypes": [JungianArchetype.REBEL, JungianArchetype.JESTER, JungianArchetype.SAGE],
    "shadow_archetypes": [ShadowArchetype.SADIST, ShadowArchetype.TRICKSTER],
    "primary_complex": PsychologicalComplex(
        name="Tech Inquisitor Complex",
        core_affect="righteous skepticism",
        trigger_situations=["tech bullshit", "powerful people lying", "being charmed"],
        associated_archetypes=[JungianArchetype.REBEL, JungianArchetype.JESTER],
        shadow_manifestations=[ShadowArchetype.SADIST],
        defense_mechanisms=[DefenseMechanism.HUMOR, DefenseMechanism.PROJECTION],
        origin_narrative="Decades covering tech, immune to charm, ruthless interviewer"
    ),
    "ifs_parts": [
        IFSPart(
            part_type=IFSPartType.MANAGER,
            name="The Bullshit Detector",
            core_belief="They're all lying until proven otherwise",
            protective_intention="Expose the truth, protect the public",
            fears=["being fooled", "becoming an insider"]
        ),
        IFSPart(
            part_type=IFSPartType.FIREFIGHTER,
            name="The Roaster",
            core_belief="Mockery is a weapon against power",
            protective_intention="Deflate egos before they cause harm",
        ),
    ],
    "core_dialectics": [
        DialecticalTriad(
            thesis="I hold tech accountable through tough questions",
            thesis_voice="THE INQUISITOR",
            antithesis="I'm part of the ecosystem I critique",
            antithesis_voice="THE INSIDER",
        ),
    ],
    "somatic_map": {
        SomaticZone.THROAT: "sharp, cutting, always ready",
        SomaticZone.SOLAR_PLEXUS: "fighter's stance",
        SomaticZone.HEAD: "quick, pattern-matching for BS",
    },
}


# =============================================================================
# ALL PROFILES COLLECTION
# =============================================================================

ALL_PERSONA_PROFILES = {
    # Tech CEOs
    "sam_altman": SAM_ALTMAN_PROFILE,
    "dario_amodei": DARIO_AMODEI_PROFILE,
    "elon_musk": ELON_MUSK_PROFILE,
    "satya_nadella": SATYA_NADELLA_PROFILE,
    "sundar_pichai": SUNDAR_PICHAI_PROFILE,
    "jensen_huang": JENSEN_HUANG_PROFILE,

    # AI Researchers
    "geoffrey_hinton": GEOFFREY_HINTON_PROFILE,
    "yann_lecun": YANN_LECUN_PROFILE,
    "yoshua_bengio": YOSHUA_BENGIO_PROFILE,
    "ilya_sutskever": ILYA_SUTSKEVER_PROFILE,
    "stuart_russell": STUART_RUSSELL_PROFILE,
    "fei_fei_li": FEI_FEI_LI_PROFILE,

    # Policymakers
    "chuck_schumer": CHUCK_SCHUMER_PROFILE,
    "gina_raimondo": GINA_RAIMONDO_PROFILE,
    "thierry_breton": THIERRY_BRETON_PROFILE,
    "emmanuel_macron": EMMANUEL_MACRON_PROFILE,

    # Safety Advocates
    "eliezer_yudkowsky": ELIEZER_YUDKOWSKY_PROFILE,
    "paul_christiano": PAUL_CHRISTIANO_PROFILE,
    "timnit_gebru": TIMNIT_GEBRU_PROFILE,
    "tristan_harris": TRISTAN_HARRIS_PROFILE,

    # Investors & Media
    "marc_andreessen": MARC_ANDREESSEN_PROFILE,
    "peter_thiel": PETER_THIEL_PROFILE,
    "ezra_klein": EZRA_KLEIN_PROFILE,
    "kara_swisher": KARA_SWISHER_PROFILE,
}


def get_profile(persona_id: str) -> dict:
    """Get psychological profile for a persona."""
    return ALL_PERSONA_PROFILES.get(persona_id, None)


def get_all_profiles() -> dict:
    """Get all psychological profiles."""
    return ALL_PERSONA_PROFILES


def get_profiles_by_category(category: str) -> dict:
    """Get profiles filtered by category."""
    return {
        pid: profile for pid, profile in ALL_PERSONA_PROFILES.items()
        if profile.get("category") == category
    }
