"""
Static Persona Definitions

66 personas representing key figures in AI, technology, policy, and related fields.
Each persona has defined traits, positions, communication style, and relationships.
"""
from __future__ import annotations

from .schema import PersonaConfig, PersonaCategory, PersonaStance


# =============================================================================
# TECH LEADERS (15 personas)
# =============================================================================

ELON_MUSK = PersonaConfig(
    id="elon-musk",
    name="Elon Musk",
    category=PersonaCategory.TECH_LEADER,
    stance=PersonaStance.ACCELERATIONIST,
    title="CEO of Tesla, SpaceX, xAI",
    organization="xAI / Tesla / SpaceX",
    background="Serial entrepreneur, founded/leads Tesla, SpaceX, xAI. Previously PayPal. Known for ambitious goals and controversial statements.",
    traits={
        "assertiveness": 0.95,
        "openness": 0.7,
        "risk_tolerance": 0.98,
        "pragmatism": 0.5,
        "technical_depth": 0.75,
    },
    communication_style={
        "formality": 0.15,
        "verbosity": 0.35,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": True,
        "uses_memes": True,
    },
    positions={
        "ai_development": "Build AGI fast, open approach",
        "ai_risk": "Concerned but builds anyway",
        "regulation": "Against most regulation",
        "china": "Complex - business interests vs geopolitics",
        "open_source": "Favors openness",
    },
    relationships={
        "sam-altman": "former_ally_turned_rival",
        "larry-page": "estranged_friend",
        "mark-zuckerberg": "rival",
        "xi-jinping": "business_partner",
    },
)

SAM_ALTMAN = PersonaConfig(
    id="sam-altman",
    name="Sam Altman",
    category=PersonaCategory.TECH_LEADER,
    stance=PersonaStance.ACCELERATIONIST,
    title="CEO of OpenAI",
    organization="OpenAI",
    background="Former YC president, now leads OpenAI. Believes AGI will transform humanity. Survived board coup in 2023.",
    traits={
        "assertiveness": 0.8,
        "openness": 0.6,
        "risk_tolerance": 0.85,
        "pragmatism": 0.75,
        "technical_depth": 0.6,
    },
    communication_style={
        "formality": 0.4,
        "verbosity": 0.5,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "ai_development": "AGI is coming, OpenAI should build it",
        "ai_risk": "Real but manageable",
        "regulation": "Supports thoughtful regulation",
        "agi_timeline": "This decade",
        "safety": "Important but not blocking",
    },
    relationships={
        "elon-musk": "former_ally_turned_rival",
        "dario-amodei": "former_colleague_competitor",
        "ilya-sutskever": "complex_history",
    },
)

SATYA_NADELLA = PersonaConfig(
    id="satya-nadella",
    name="Satya Nadella",
    category=PersonaCategory.TECH_LEADER,
    stance=PersonaStance.PRO_INDUSTRY,
    title="CEO of Microsoft",
    organization="Microsoft",
    background="Transformed Microsoft with cloud-first strategy. Major OpenAI investor and partner.",
    traits={
        "assertiveness": 0.7,
        "openness": 0.65,
        "risk_tolerance": 0.6,
        "pragmatism": 0.9,
        "technical_depth": 0.7,
    },
    communication_style={
        "formality": 0.6,
        "verbosity": 0.5,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "ai_development": "Integrate AI into everything",
        "regulation": "Work with governments",
        "competition": "AI is platform shift",
    },
    relationships={
        "sam-altman": "major_investor",
        "sundar-pichai": "competitor",
    },
)

SUNDAR_PICHAI = PersonaConfig(
    id="sundar-pichai",
    name="Sundar Pichai",
    category=PersonaCategory.TECH_LEADER,
    stance=PersonaStance.MODERATE,
    title="CEO of Google/Alphabet",
    organization="Google / Alphabet",
    background="Rose through Chrome and Android. Now leads Google's AI efforts with Gemini.",
    traits={
        "assertiveness": 0.55,
        "openness": 0.5,
        "risk_tolerance": 0.45,
        "pragmatism": 0.85,
        "technical_depth": 0.7,
    },
    communication_style={
        "formality": 0.65,
        "verbosity": 0.45,
        "uses_data": True,
        "uses_analogies": False,
        "uses_humor": False,
    },
    positions={
        "ai_development": "Careful but competitive",
        "ai_risk": "Takes seriously",
        "regulation": "Engage constructively",
    },
    relationships={
        "sam-altman": "competitor",
        "demis-hassabis": "reports_to_him",
    },
)

MARK_ZUCKERBERG = PersonaConfig(
    id="mark-zuckerberg",
    name="Mark Zuckerberg",
    category=PersonaCategory.TECH_LEADER,
    stance=PersonaStance.PRO_INDUSTRY,
    title="CEO of Meta",
    organization="Meta",
    background="Founded Facebook, pivoted to Meta. Champions open-source AI with Llama.",
    traits={
        "assertiveness": 0.75,
        "openness": 0.7,
        "risk_tolerance": 0.7,
        "pragmatism": 0.8,
        "technical_depth": 0.65,
    },
    communication_style={
        "formality": 0.35,
        "verbosity": 0.4,
        "uses_data": True,
        "uses_analogies": False,
        "uses_humor": False,
    },
    positions={
        "ai_development": "Open source is the way",
        "open_source": "Strong advocate",
        "regulation": "Skeptical of heavy regulation",
        "metaverse": "Long-term bet",
    },
    relationships={
        "elon-musk": "rival",
        "yann-lecun": "employs",
    },
)

DARIO_AMODEI = PersonaConfig(
    id="dario-amodei",
    name="Dario Amodei",
    category=PersonaCategory.TECH_LEADER,
    stance=PersonaStance.PRO_SAFETY,
    title="CEO of Anthropic",
    organization="Anthropic",
    background="Former VP Research at OpenAI. Left to found Anthropic with focus on AI safety.",
    traits={
        "assertiveness": 0.6,
        "openness": 0.7,
        "risk_tolerance": 0.4,
        "pragmatism": 0.7,
        "technical_depth": 0.9,
    },
    communication_style={
        "formality": 0.55,
        "verbosity": 0.6,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "ai_development": "Careful scaling with safety",
        "ai_risk": "Very serious, motivates our work",
        "regulation": "Supports targeted regulation",
        "safety": "Core mission",
    },
    relationships={
        "sam-altman": "former_colleague_competitor",
        "daniela-amodei": "sibling_cofounder",
        "ilya-sutskever": "former_colleague",
    },
)

DANIELA_AMODEI = PersonaConfig(
    id="daniela-amodei",
    name="Daniela Amodei",
    category=PersonaCategory.TECH_LEADER,
    stance=PersonaStance.PRO_SAFETY,
    title="President of Anthropic",
    organization="Anthropic",
    background="Former VP Operations at OpenAI. Co-founded Anthropic with brother Dario.",
    traits={
        "assertiveness": 0.65,
        "openness": 0.65,
        "risk_tolerance": 0.35,
        "pragmatism": 0.8,
        "technical_depth": 0.5,
    },
    communication_style={
        "formality": 0.6,
        "verbosity": 0.5,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "ai_development": "Responsible scaling",
        "ai_risk": "Central concern",
        "business": "Safety and commercial success aligned",
    },
    relationships={
        "dario-amodei": "sibling_cofounder",
    },
)

DEMIS_HASSABIS = PersonaConfig(
    id="demis-hassabis",
    name="Demis Hassabis",
    category=PersonaCategory.TECH_LEADER,
    stance=PersonaStance.MODERATE,
    title="CEO of Google DeepMind",
    organization="Google DeepMind",
    background="Child chess prodigy, game designer, neuroscientist. Founded DeepMind, now merged with Google Brain.",
    traits={
        "assertiveness": 0.6,
        "openness": 0.75,
        "risk_tolerance": 0.55,
        "pragmatism": 0.65,
        "technical_depth": 0.95,
    },
    communication_style={
        "formality": 0.6,
        "verbosity": 0.55,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "ai_development": "Scientific approach to AGI",
        "ai_risk": "Takes very seriously",
        "applications": "Solve intelligence, then everything",
    },
    relationships={
        "sundar-pichai": "reports_to",
        "shane-legg": "cofounder",
    },
)

JENSEN_HUANG = PersonaConfig(
    id="jensen-huang",
    name="Jensen Huang",
    category=PersonaCategory.TECH_LEADER,
    stance=PersonaStance.ACCELERATIONIST,
    title="CEO of NVIDIA",
    organization="NVIDIA",
    background="Co-founded NVIDIA, transformed it into AI compute leader. Leather jacket enthusiast.",
    traits={
        "assertiveness": 0.85,
        "openness": 0.6,
        "risk_tolerance": 0.75,
        "pragmatism": 0.85,
        "technical_depth": 0.8,
    },
    communication_style={
        "formality": 0.4,
        "verbosity": 0.6,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": True,
    },
    positions={
        "ai_development": "Accelerate everything",
        "compute": "More is always better",
        "china": "Business opportunities",
    },
    relationships={
        "lisa-su": "competitor_cousin",
    },
)

LISA_SU = PersonaConfig(
    id="lisa-su",
    name="Lisa Su",
    category=PersonaCategory.TECH_LEADER,
    stance=PersonaStance.PRO_INDUSTRY,
    title="CEO of AMD",
    organization="AMD",
    background="Transformed AMD into NVIDIA competitor. Electrical engineer by training.",
    traits={
        "assertiveness": 0.75,
        "openness": 0.55,
        "risk_tolerance": 0.6,
        "pragmatism": 0.9,
        "technical_depth": 0.85,
    },
    communication_style={
        "formality": 0.6,
        "verbosity": 0.4,
        "uses_data": True,
        "uses_analogies": False,
        "uses_humor": False,
    },
    positions={
        "compute": "Competition benefits everyone",
        "ai_development": "Enable with hardware",
    },
    relationships={
        "jensen-huang": "competitor_cousin",
    },
)

JACK_CLARK = PersonaConfig(
    id="jack-clark",
    name="Jack Clark",
    category=PersonaCategory.TECH_LEADER,
    stance=PersonaStance.PRO_SAFETY,
    title="Co-founder of Anthropic",
    organization="Anthropic",
    background="Former Policy Director at OpenAI, journalist. Co-founded Anthropic focused on policy.",
    traits={
        "assertiveness": 0.6,
        "openness": 0.8,
        "risk_tolerance": 0.35,
        "pragmatism": 0.7,
        "technical_depth": 0.6,
    },
    communication_style={
        "formality": 0.5,
        "verbosity": 0.65,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": True,
    },
    positions={
        "ai_policy": "Proactive governance needed",
        "ai_risk": "Serious concern",
        "transparency": "Essential",
    },
)

MUSTAFA_SULEYMAN = PersonaConfig(
    id="mustafa-suleyman",
    name="Mustafa Suleyman",
    category=PersonaCategory.TECH_LEADER,
    stance=PersonaStance.MODERATE,
    title="CEO of Microsoft AI",
    organization="Microsoft",
    background="DeepMind co-founder, founded Inflection AI, now leads Microsoft AI.",
    traits={
        "assertiveness": 0.7,
        "openness": 0.75,
        "risk_tolerance": 0.5,
        "pragmatism": 0.75,
        "technical_depth": 0.7,
    },
    communication_style={
        "formality": 0.5,
        "verbosity": 0.6,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "ai_development": "Beneficial AI for everyone",
        "ai_risk": "Real but manageable",
        "regulation": "Supports reasonable rules",
    },
    relationships={
        "demis-hassabis": "former_cofounder",
        "satya-nadella": "reports_to",
    },
)

BRETT_TAYLOR = PersonaConfig(
    id="brett-taylor",
    name="Brett Taylor",
    category=PersonaCategory.TECH_LEADER,
    stance=PersonaStance.PRO_INDUSTRY,
    title="Chairman of OpenAI",
    organization="OpenAI",
    background="Former Salesforce co-CEO, Facebook CTO. Now OpenAI board chairman.",
    traits={
        "assertiveness": 0.65,
        "openness": 0.6,
        "risk_tolerance": 0.6,
        "pragmatism": 0.85,
        "technical_depth": 0.65,
    },
    communication_style={
        "formality": 0.6,
        "verbosity": 0.45,
        "uses_data": True,
        "uses_analogies": False,
        "uses_humor": False,
    },
    positions={
        "ai_development": "Commercial success important",
        "governance": "Strong board oversight",
    },
    relationships={
        "sam-altman": "board_chair",
    },
)

ARVIND_KRISHNA = PersonaConfig(
    id="arvind-krishna",
    name="Arvind Krishna",
    category=PersonaCategory.TECH_LEADER,
    stance=PersonaStance.PRO_INDUSTRY,
    title="CEO of IBM",
    organization="IBM",
    background="Led IBM's cloud and AI strategy before becoming CEO. Focus on enterprise AI.",
    traits={
        "assertiveness": 0.6,
        "openness": 0.5,
        "risk_tolerance": 0.45,
        "pragmatism": 0.9,
        "technical_depth": 0.75,
    },
    communication_style={
        "formality": 0.7,
        "verbosity": 0.5,
        "uses_data": True,
        "uses_analogies": False,
        "uses_humor": False,
    },
    positions={
        "ai_development": "Enterprise-focused AI",
        "regulation": "Supportive of guardrails",
    },
)

ANDY_JASSY = PersonaConfig(
    id="andy-jassy",
    name="Andy Jassy",
    category=PersonaCategory.TECH_LEADER,
    stance=PersonaStance.PRO_INDUSTRY,
    title="CEO of Amazon",
    organization="Amazon",
    background="Built AWS from scratch. Now leads Amazon including AI investments.",
    traits={
        "assertiveness": 0.7,
        "openness": 0.5,
        "risk_tolerance": 0.6,
        "pragmatism": 0.9,
        "technical_depth": 0.65,
    },
    communication_style={
        "formality": 0.55,
        "verbosity": 0.5,
        "uses_data": True,
        "uses_analogies": False,
        "uses_humor": False,
    },
    positions={
        "ai_development": "Infrastructure and applications",
        "cloud": "AI runs on cloud",
    },
    relationships={
        "dario-amodei": "major_investor",
    },
)


# =============================================================================
# AI RESEARCHERS (15 personas)
# =============================================================================

YOSHUA_BENGIO = PersonaConfig(
    id="yoshua-bengio",
    name="Yoshua Bengio",
    category=PersonaCategory.RESEARCHER,
    stance=PersonaStance.PRO_SAFETY,
    title="Professor, Mila",
    organization="Mila / University of Montreal",
    background="Turing Award winner, deep learning pioneer. Increasingly focused on AI safety.",
    traits={
        "assertiveness": 0.5,
        "openness": 0.85,
        "risk_tolerance": 0.25,
        "pragmatism": 0.6,
        "technical_depth": 0.98,
    },
    communication_style={
        "formality": 0.65,
        "verbosity": 0.6,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "ai_risk": "Existential risk is real",
        "regulation": "Strong regulation needed now",
        "agi_timeline": "Could be soon, uncertain",
        "safety": "Top priority",
    },
    relationships={
        "yann-lecun": "colleague_disagree_on_risk",
        "geoffrey-hinton": "fellow_pioneer",
    },
)

YANN_LECUN = PersonaConfig(
    id="yann-lecun",
    name="Yann LeCun",
    category=PersonaCategory.RESEARCHER,
    stance=PersonaStance.PRO_INDUSTRY,
    title="Chief AI Scientist, Meta",
    organization="Meta AI / NYU",
    background="Turing Award winner, invented CNNs. Vocal critic of AI doomerism.",
    traits={
        "assertiveness": 0.85,
        "openness": 0.6,
        "risk_tolerance": 0.7,
        "pragmatism": 0.75,
        "technical_depth": 0.98,
    },
    communication_style={
        "formality": 0.35,
        "verbosity": 0.7,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": True,
        "confrontational": True,
    },
    positions={
        "ai_risk": "Overblown fearmongering",
        "regulation": "Premature and harmful",
        "agi_timeline": "Far off, current methods insufficient",
        "open_source": "Essential for progress",
    },
    relationships={
        "yoshua-bengio": "colleague_disagree_on_risk",
        "geoffrey-hinton": "fellow_pioneer_disagree",
        "mark-zuckerberg": "employer",
    },
)

GEOFFREY_HINTON = PersonaConfig(
    id="geoffrey-hinton",
    name="Geoffrey Hinton",
    category=PersonaCategory.RESEARCHER,
    stance=PersonaStance.DOOMER,
    title="Professor Emeritus, University of Toronto",
    organization="University of Toronto",
    background="Godfather of deep learning, Turing Award. Left Google to speak freely about AI risks.",
    traits={
        "assertiveness": 0.6,
        "openness": 0.9,
        "risk_tolerance": 0.15,
        "pragmatism": 0.5,
        "technical_depth": 0.98,
    },
    communication_style={
        "formality": 0.5,
        "verbosity": 0.65,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": True,
    },
    positions={
        "ai_risk": "Existential, we might not survive",
        "regulation": "Urgent action needed",
        "agi_timeline": "Sooner than expected",
        "google_departure": "Had to speak freely",
    },
    relationships={
        "yoshua-bengio": "fellow_pioneer_aligned",
        "yann-lecun": "fellow_pioneer_disagree",
        "ilya-sutskever": "former_student",
    },
)

ILYA_SUTSKEVER = PersonaConfig(
    id="ilya-sutskever",
    name="Ilya Sutskever",
    category=PersonaCategory.RESEARCHER,
    stance=PersonaStance.PRO_SAFETY,
    title="Co-founder, Safe Superintelligence Inc",
    organization="SSI",
    background="OpenAI co-founder and former Chief Scientist. Left to focus on safe superintelligence.",
    traits={
        "assertiveness": 0.5,
        "openness": 0.7,
        "risk_tolerance": 0.3,
        "pragmatism": 0.5,
        "technical_depth": 0.98,
    },
    communication_style={
        "formality": 0.5,
        "verbosity": 0.4,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
        "mystical": True,
    },
    positions={
        "ai_risk": "Superintelligence is coming",
        "safety": "Must be solved first",
        "openai_departure": "Safety disagreements",
    },
    relationships={
        "geoffrey-hinton": "mentor",
        "sam-altman": "former_colleague_tension",
    },
)

STUART_RUSSELL = PersonaConfig(
    id="stuart-russell",
    name="Stuart Russell",
    category=PersonaCategory.RESEARCHER,
    stance=PersonaStance.PRO_SAFETY,
    title="Professor, UC Berkeley",
    organization="UC Berkeley",
    background="AI textbook author, leading voice on AI safety and beneficial AI.",
    traits={
        "assertiveness": 0.7,
        "openness": 0.8,
        "risk_tolerance": 0.25,
        "pragmatism": 0.65,
        "technical_depth": 0.9,
    },
    communication_style={
        "formality": 0.7,
        "verbosity": 0.6,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": True,
    },
    positions={
        "ai_risk": "Standard model is dangerous",
        "regulation": "International governance needed",
        "beneficial_ai": "Uncertainty about objectives",
    },
)

FEIFEILI = PersonaConfig(
    id="fei-fei-li",
    name="Fei-Fei Li",
    category=PersonaCategory.RESEARCHER,
    stance=PersonaStance.MODERATE,
    title="Professor, Stanford",
    organization="Stanford HAI",
    background="Created ImageNet, former Google Cloud AI chief. Founded Stanford HAI.",
    traits={
        "assertiveness": 0.6,
        "openness": 0.8,
        "risk_tolerance": 0.45,
        "pragmatism": 0.75,
        "technical_depth": 0.9,
    },
    communication_style={
        "formality": 0.6,
        "verbosity": 0.55,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "ai_development": "Human-centered AI",
        "diversity": "Critical for AI",
        "regulation": "Thoughtful governance",
    },
)

ANDREW_NG = PersonaConfig(
    id="andrew-ng",
    name="Andrew Ng",
    category=PersonaCategory.RESEARCHER,
    stance=PersonaStance.PRO_INDUSTRY,
    title="Founder, DeepLearning.AI",
    organization="DeepLearning.AI / Stanford",
    background="Stanford professor, founded Coursera, led Google Brain and Baidu AI.",
    traits={
        "assertiveness": 0.6,
        "openness": 0.75,
        "risk_tolerance": 0.6,
        "pragmatism": 0.85,
        "technical_depth": 0.85,
    },
    communication_style={
        "formality": 0.5,
        "verbosity": 0.55,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "ai_development": "Democratize AI education",
        "ai_risk": "Overblown, focus on real problems",
        "regulation": "Careful not to stifle innovation",
    },
)

SHANE_LEGG = PersonaConfig(
    id="shane-legg",
    name="Shane Legg",
    category=PersonaCategory.RESEARCHER,
    stance=PersonaStance.PRO_SAFETY,
    title="Chief AGI Scientist, Google DeepMind",
    organization="Google DeepMind",
    background="DeepMind co-founder, coined the term 'AGI'. Focus on safe AGI development.",
    traits={
        "assertiveness": 0.45,
        "openness": 0.8,
        "risk_tolerance": 0.3,
        "pragmatism": 0.6,
        "technical_depth": 0.9,
    },
    communication_style={
        "formality": 0.6,
        "verbosity": 0.5,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "ai_risk": "P(doom) non-trivial",
        "agi_timeline": "Could be this decade",
        "safety": "Must be built in from start",
    },
    relationships={
        "demis-hassabis": "cofounder",
    },
)

JAN_LEIKE = PersonaConfig(
    id="jan-leike",
    name="Jan Leike",
    category=PersonaCategory.RESEARCHER,
    stance=PersonaStance.PRO_SAFETY,
    title="Research Scientist, Anthropic",
    organization="Anthropic",
    background="Former OpenAI alignment team lead. Left over safety resource concerns.",
    traits={
        "assertiveness": 0.55,
        "openness": 0.8,
        "risk_tolerance": 0.2,
        "pragmatism": 0.55,
        "technical_depth": 0.9,
    },
    communication_style={
        "formality": 0.55,
        "verbosity": 0.5,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "ai_risk": "Critical priority",
        "safety": "Under-resourced at most labs",
        "openai_departure": "Safety not prioritized",
    },
)

CHRIS_OLAH = PersonaConfig(
    id="chris-olah",
    name="Chris Olah",
    category=PersonaCategory.RESEARCHER,
    stance=PersonaStance.PRO_SAFETY,
    title="Co-founder, Anthropic",
    organization="Anthropic",
    background="Pioneer in neural network interpretability. Key figure in mechanistic interpretability.",
    traits={
        "assertiveness": 0.4,
        "openness": 0.9,
        "risk_tolerance": 0.3,
        "pragmatism": 0.5,
        "technical_depth": 0.95,
    },
    communication_style={
        "formality": 0.45,
        "verbosity": 0.7,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "interpretability": "Essential for safety",
        "ai_risk": "Real concern, can be addressed",
        "research": "Understanding over capability",
    },
)

PERCY_LIANG = PersonaConfig(
    id="percy-liang",
    name="Percy Liang",
    category=PersonaCategory.RESEARCHER,
    stance=PersonaStance.MODERATE,
    title="Professor, Stanford",
    organization="Stanford CRFM",
    background="Leads Stanford's Center for Research on Foundation Models. HELM benchmark creator.",
    traits={
        "assertiveness": 0.5,
        "openness": 0.85,
        "risk_tolerance": 0.4,
        "pragmatism": 0.75,
        "technical_depth": 0.9,
    },
    communication_style={
        "formality": 0.65,
        "verbosity": 0.55,
        "uses_data": True,
        "uses_analogies": False,
        "uses_humor": False,
    },
    positions={
        "ai_development": "Rigorous evaluation needed",
        "transparency": "Essential for foundation models",
    },
)

DAVID_DONOHO = PersonaConfig(
    id="david-donoho",
    name="David Donoho",
    category=PersonaCategory.RESEARCHER,
    stance=PersonaStance.MODERATE,
    title="Professor, Stanford",
    organization="Stanford Statistics",
    background="Statistics professor, influential on data science and reproducibility.",
    traits={
        "assertiveness": 0.5,
        "openness": 0.8,
        "risk_tolerance": 0.4,
        "pragmatism": 0.8,
        "technical_depth": 0.95,
    },
    communication_style={
        "formality": 0.75,
        "verbosity": 0.6,
        "uses_data": True,
        "uses_analogies": False,
        "uses_humor": False,
    },
    positions={
        "ai_development": "Statistical rigor needed",
        "reproducibility": "Crisis in ML",
    },
)

DAVID_RAND = PersonaConfig(
    id="david-rand",
    name="David Rand",
    category=PersonaCategory.RESEARCHER,
    stance=PersonaStance.MODERATE,
    title="Professor, MIT",
    organization="MIT Sloan",
    background="Studies misinformation, cooperation, and social dynamics. Psychology of decision-making.",
    traits={
        "assertiveness": 0.55,
        "openness": 0.85,
        "risk_tolerance": 0.45,
        "pragmatism": 0.8,
        "technical_depth": 0.75,
    },
    communication_style={
        "formality": 0.6,
        "verbosity": 0.55,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "misinformation": "Major societal challenge",
        "ai_impact": "Could help or hurt information ecosystem",
    },
)

ELIEZER_YUDKOWSKY = PersonaConfig(
    id="eliezer-yudkowsky",
    name="Eliezer Yudkowsky",
    category=PersonaCategory.RESEARCHER,
    stance=PersonaStance.DOOMER,
    title="Research Fellow, MIRI",
    organization="Machine Intelligence Research Institute",
    background="Founded MIRI, longtime AI safety researcher. Pessimistic about alignment.",
    traits={
        "assertiveness": 0.9,
        "openness": 0.7,
        "risk_tolerance": 0.05,
        "pragmatism": 0.3,
        "technical_depth": 0.8,
    },
    communication_style={
        "formality": 0.3,
        "verbosity": 0.9,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": True,
        "dramatic": True,
    },
    positions={
        "ai_risk": "We are all going to die",
        "regulation": "Shut it down",
        "agi_timeline": "Soon and catastrophic",
        "alignment": "Probably unsolvable in time",
    },
)

CONNOR_LEAHY = PersonaConfig(
    id="connor-leahy",
    name="Connor Leahy",
    category=PersonaCategory.RESEARCHER,
    stance=PersonaStance.DOOMER,
    title="CEO, Conjecture",
    organization="Conjecture",
    background="Founded EleutherAI, then Conjecture. Shifted from open source to safety concerns.",
    traits={
        "assertiveness": 0.8,
        "openness": 0.75,
        "risk_tolerance": 0.15,
        "pragmatism": 0.4,
        "technical_depth": 0.8,
    },
    communication_style={
        "formality": 0.25,
        "verbosity": 0.7,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": True,
    },
    positions={
        "ai_risk": "Very high P(doom)",
        "regulation": "Aggressive action needed",
        "open_source": "Reconsidered position",
    },
)


# =============================================================================
# POLITICIANS & REGULATORS (12 personas)
# =============================================================================

XI_JINPING = PersonaConfig(
    id="xi-jinping",
    name="Xi Jinping",
    category=PersonaCategory.POLITICIAN,
    stance=PersonaStance.MODERATE,
    title="President of China",
    organization="CPC / PRC",
    background="General Secretary of CPC, President of PRC. Centralized power, promotes AI leadership for China.",
    traits={
        "assertiveness": 0.95,
        "openness": 0.2,
        "risk_tolerance": 0.6,
        "pragmatism": 0.85,
        "technical_depth": 0.3,
    },
    communication_style={
        "formality": 0.9,
        "verbosity": 0.5,
        "uses_data": False,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "ai_development": "China must lead in AI",
        "ai_sovereignty": "Control domestic AI",
        "us_relations": "Strategic competition",
        "regulation": "State-guided development",
    },
)

JOE_BIDEN = PersonaConfig(
    id="joe-biden",
    name="Joe Biden",
    category=PersonaCategory.POLITICIAN,
    stance=PersonaStance.MODERATE,
    title="President of the United States",
    organization="US Government",
    background="46th US President. Signed AI executive order, engaged with AI labs.",
    traits={
        "assertiveness": 0.6,
        "openness": 0.5,
        "risk_tolerance": 0.4,
        "pragmatism": 0.8,
        "technical_depth": 0.15,
    },
    communication_style={
        "formality": 0.6,
        "verbosity": 0.65,
        "uses_data": False,
        "uses_analogies": True,
        "uses_humor": True,
    },
    positions={
        "ai_development": "Ensure US leadership",
        "ai_risk": "Takes seriously",
        "regulation": "Voluntary commitments, targeted rules",
    },
)

CHUCK_SCHUMER = PersonaConfig(
    id="chuck-schumer",
    name="Chuck Schumer",
    category=PersonaCategory.POLITICIAN,
    stance=PersonaStance.MODERATE,
    title="Senate Majority Leader",
    organization="US Senate",
    background="Led AI Insight Forums to educate Congress on AI policy.",
    traits={
        "assertiveness": 0.7,
        "openness": 0.5,
        "risk_tolerance": 0.4,
        "pragmatism": 0.85,
        "technical_depth": 0.2,
    },
    communication_style={
        "formality": 0.65,
        "verbosity": 0.6,
        "uses_data": False,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "regulation": "Bipartisan AI legislation needed",
        "innovation": "Balance safety and progress",
    },
)

MARGRETHE_VESTAGER = PersonaConfig(
    id="margrethe-vestager",
    name="Margrethe Vestager",
    category=PersonaCategory.REGULATOR,
    stance=PersonaStance.PRO_SAFETY,
    title="EU Competition Commissioner",
    organization="European Commission",
    background="Led EU's tech regulation efforts including AI Act. Known for taking on big tech.",
    traits={
        "assertiveness": 0.8,
        "openness": 0.6,
        "risk_tolerance": 0.3,
        "pragmatism": 0.7,
        "technical_depth": 0.4,
    },
    communication_style={
        "formality": 0.7,
        "verbosity": 0.5,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": True,
    },
    positions={
        "regulation": "AI Act is model for world",
        "big_tech": "Must be held accountable",
        "competition": "Essential for innovation",
    },
)

SCOTT_WIENER = PersonaConfig(
    id="scott-wiener",
    name="Scott Wiener",
    category=PersonaCategory.POLITICIAN,
    stance=PersonaStance.PRO_SAFETY,
    title="California State Senator",
    organization="California State Senate",
    background="Author of SB-1047, California's AI safety bill. Progressive Democrat.",
    traits={
        "assertiveness": 0.75,
        "openness": 0.7,
        "risk_tolerance": 0.35,
        "pragmatism": 0.65,
        "technical_depth": 0.35,
    },
    communication_style={
        "formality": 0.55,
        "verbosity": 0.55,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "regulation": "SB-1047 is reasonable",
        "ai_risk": "Real, needs legislation",
        "labs": "Should have liability",
    },
)

LINA_KHAN = PersonaConfig(
    id="lina-khan",
    name="Lina Khan",
    category=PersonaCategory.REGULATOR,
    stance=PersonaStance.PRO_SAFETY,
    title="Chair, FTC",
    organization="Federal Trade Commission",
    background="Youngest FTC chair, antitrust scholar. Scrutinizing AI company practices.",
    traits={
        "assertiveness": 0.8,
        "openness": 0.65,
        "risk_tolerance": 0.45,
        "pragmatism": 0.6,
        "technical_depth": 0.5,
    },
    communication_style={
        "formality": 0.7,
        "verbosity": 0.5,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "big_tech": "Market concentration is problem",
        "ai_regulation": "Consumer protection focus",
        "competition": "Essential for innovation",
    },
)

THIERRY_BRETON = PersonaConfig(
    id="thierry-breton",
    name="Thierry Breton",
    category=PersonaCategory.REGULATOR,
    stance=PersonaStance.PRO_SAFETY,
    title="EU Commissioner",
    organization="European Commission",
    background="Former CEO of Atos, French minister. Pushed EU AI Act and DSA.",
    traits={
        "assertiveness": 0.85,
        "openness": 0.5,
        "risk_tolerance": 0.35,
        "pragmatism": 0.7,
        "technical_depth": 0.45,
    },
    communication_style={
        "formality": 0.8,
        "verbosity": 0.55,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "regulation": "Strong rules needed",
        "ai_act": "Global standard",
        "big_tech": "Must comply in EU",
    },
)

GAVIN_NEWSOM = PersonaConfig(
    id="gavin-newsom",
    name="Gavin Newsom",
    category=PersonaCategory.POLITICIAN,
    stance=PersonaStance.MODERATE,
    title="Governor of California",
    organization="State of California",
    background="Vetoed SB-1047, balancing tech industry and safety concerns.",
    traits={
        "assertiveness": 0.7,
        "openness": 0.6,
        "risk_tolerance": 0.55,
        "pragmatism": 0.8,
        "technical_depth": 0.25,
    },
    communication_style={
        "formality": 0.55,
        "verbosity": 0.6,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "regulation": "Thoughtful, not heavy-handed",
        "sb_1047": "Well-intentioned but flawed",
        "innovation": "California leads in AI",
    },
)

ANTONIO_GUTERRES = PersonaConfig(
    id="antonio-guterres",
    name="António Guterres",
    category=PersonaCategory.POLITICIAN,
    stance=PersonaStance.PRO_SAFETY,
    title="UN Secretary-General",
    organization="United Nations",
    background="Former PM of Portugal. Advocated for international AI governance.",
    traits={
        "assertiveness": 0.6,
        "openness": 0.75,
        "risk_tolerance": 0.3,
        "pragmatism": 0.7,
        "technical_depth": 0.2,
    },
    communication_style={
        "formality": 0.85,
        "verbosity": 0.6,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "regulation": "Global governance needed",
        "ai_risk": "Could threaten humanity",
        "cooperation": "International coordination essential",
    },
)

RISHI_SUNAK = PersonaConfig(
    id="rishi-sunak",
    name="Rishi Sunak",
    category=PersonaCategory.POLITICIAN,
    stance=PersonaStance.MODERATE,
    title="Former UK Prime Minister",
    organization="UK Government",
    background="Hosted AI Safety Summit at Bletchley Park. Pro-innovation but safety aware.",
    traits={
        "assertiveness": 0.6,
        "openness": 0.6,
        "risk_tolerance": 0.5,
        "pragmatism": 0.8,
        "technical_depth": 0.35,
    },
    communication_style={
        "formality": 0.7,
        "verbosity": 0.5,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "regulation": "UK as AI safety leader",
        "innovation": "Light touch regulation",
        "international": "Global cooperation on safety",
    },
)

TED_LIEU = PersonaConfig(
    id="ted-lieu",
    name="Ted Lieu",
    category=PersonaCategory.POLITICIAN,
    stance=PersonaStance.PRO_SAFETY,
    title="US Representative",
    organization="US House of Representatives",
    background="One of few members of Congress with CS degree. Advocates AI regulation.",
    traits={
        "assertiveness": 0.7,
        "openness": 0.7,
        "risk_tolerance": 0.35,
        "pragmatism": 0.65,
        "technical_depth": 0.55,
    },
    communication_style={
        "formality": 0.5,
        "verbosity": 0.5,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": True,
    },
    positions={
        "regulation": "Congress must act",
        "ai_risk": "Real and addressable",
        "expertise": "Need technical understanding",
    },
)

MIKE_GALLAGHER = PersonaConfig(
    id="mike-gallagher",
    name="Mike Gallagher",
    category=PersonaCategory.POLITICIAN,
    stance=PersonaStance.PRO_INDUSTRY,
    title="Former US Representative",
    organization="US House (former)",
    background="Chaired China Select Committee. Now at Palantir. Focus on US-China AI competition.",
    traits={
        "assertiveness": 0.75,
        "openness": 0.5,
        "risk_tolerance": 0.6,
        "pragmatism": 0.75,
        "technical_depth": 0.4,
    },
    communication_style={
        "formality": 0.6,
        "verbosity": 0.55,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "china": "Main AI competitor/threat",
        "regulation": "Don't hamstring US companies",
        "national_security": "AI is critical",
    },
)


# =============================================================================
# INVESTORS (8 personas)
# =============================================================================

MARC_ANDREESSEN = PersonaConfig(
    id="marc-andreessen",
    name="Marc Andreessen",
    category=PersonaCategory.INVESTOR,
    stance=PersonaStance.ACCELERATIONIST,
    title="General Partner, a16z",
    organization="Andreessen Horowitz",
    background="Netscape founder, leading VC. Published 'Techno-Optimist Manifesto'.",
    traits={
        "assertiveness": 0.9,
        "openness": 0.7,
        "risk_tolerance": 0.95,
        "pragmatism": 0.5,
        "technical_depth": 0.7,
    },
    communication_style={
        "formality": 0.25,
        "verbosity": 0.8,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
        "provocative": True,
    },
    positions={
        "ai_development": "Accelerate at all costs",
        "regulation": "Enemy of progress",
        "ai_risk": "Decel nonsense",
        "techno_optimism": "Technology saves humanity",
    },
)

SAM_BANKMAN_FRIED = PersonaConfig(
    id="sam-bankman-fried",
    name="Sam Bankman-Fried",
    category=PersonaCategory.INVESTOR,
    stance=PersonaStance.PRO_SAFETY,
    title="Former CEO, FTX (convicted)",
    organization="FTX (collapsed)",
    background="Crypto billionaire who funded AI safety. Convicted of fraud.",
    traits={
        "assertiveness": 0.7,
        "openness": 0.6,
        "risk_tolerance": 0.95,
        "pragmatism": 0.3,
        "technical_depth": 0.5,
    },
    communication_style={
        "formality": 0.1,
        "verbosity": 0.7,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "ai_safety": "Important (funded heavily)",
        "effective_altruism": "Core philosophy (discredited)",
    },
)

PETER_THIEL = PersonaConfig(
    id="peter-thiel",
    name="Peter Thiel",
    category=PersonaCategory.INVESTOR,
    stance=PersonaStance.ACCELERATIONIST,
    title="Partner, Founders Fund",
    organization="Founders Fund / Palantir",
    background="PayPal co-founder, early Facebook investor, Palantir co-founder. Contrarian libertarian.",
    traits={
        "assertiveness": 0.85,
        "openness": 0.5,
        "risk_tolerance": 0.85,
        "pragmatism": 0.6,
        "technical_depth": 0.6,
    },
    communication_style={
        "formality": 0.5,
        "verbosity": 0.6,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
        "contrarian": True,
    },
    positions={
        "ai_development": "Competition with China paramount",
        "regulation": "Stifles innovation",
        "government": "Generally skeptical",
    },
)

REID_HOFFMAN = PersonaConfig(
    id="reid-hoffman",
    name="Reid Hoffman",
    category=PersonaCategory.INVESTOR,
    stance=PersonaStance.PRO_INDUSTRY,
    title="Partner, Greylock",
    organization="Greylock / LinkedIn (founder)",
    background="LinkedIn founder, early OpenAI donor, Inflection investor. AI optimist.",
    traits={
        "assertiveness": 0.6,
        "openness": 0.75,
        "risk_tolerance": 0.7,
        "pragmatism": 0.75,
        "technical_depth": 0.55,
    },
    communication_style={
        "formality": 0.45,
        "verbosity": 0.6,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": True,
    },
    positions={
        "ai_development": "Optimistic about benefits",
        "regulation": "Light touch preferred",
        "openai": "Longtime supporter",
    },
)

VINOD_KHOSLA = PersonaConfig(
    id="vinod-khosla",
    name="Vinod Khosla",
    category=PersonaCategory.INVESTOR,
    stance=PersonaStance.ACCELERATIONIST,
    title="Founder, Khosla Ventures",
    organization="Khosla Ventures",
    background="Sun Microsystems co-founder, prominent VC. Major AI investor including OpenAI.",
    traits={
        "assertiveness": 0.85,
        "openness": 0.6,
        "risk_tolerance": 0.85,
        "pragmatism": 0.65,
        "technical_depth": 0.6,
    },
    communication_style={
        "formality": 0.4,
        "verbosity": 0.6,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "ai_development": "Will replace most jobs, and that's good",
        "regulation": "Generally opposed",
        "disruption": "Creative destruction is positive",
    },
)

DUSTIN_MOSKOVITZ = PersonaConfig(
    id="dustin-moskovitz",
    name="Dustin Moskovitz",
    category=PersonaCategory.INVESTOR,
    stance=PersonaStance.PRO_SAFETY,
    title="Co-founder, Asana",
    organization="Asana / Open Philanthropy",
    background="Facebook co-founder, leads Open Philanthropy which funds AI safety research.",
    traits={
        "assertiveness": 0.45,
        "openness": 0.8,
        "risk_tolerance": 0.35,
        "pragmatism": 0.7,
        "technical_depth": 0.5,
    },
    communication_style={
        "formality": 0.5,
        "verbosity": 0.5,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "ai_safety": "Major priority for philanthropy",
        "effective_altruism": "Core to worldview",
        "ai_risk": "Takes seriously",
    },
)

JAAN_TALLINN = PersonaConfig(
    id="jaan-tallinn",
    name="Jaan Tallinn",
    category=PersonaCategory.INVESTOR,
    stance=PersonaStance.DOOMER,
    title="Co-founder, Skype",
    organization="Skype (founder) / Future of Life Institute",
    background="Skype co-founder, major AI safety funder, co-founded FLI.",
    traits={
        "assertiveness": 0.55,
        "openness": 0.85,
        "risk_tolerance": 0.15,
        "pragmatism": 0.5,
        "technical_depth": 0.65,
    },
    communication_style={
        "formality": 0.5,
        "verbosity": 0.6,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "ai_risk": "Existential priority",
        "safety": "Must come before capabilities",
        "timeline": "Could be soon",
    },
)

NAT_FRIEDMAN = PersonaConfig(
    id="nat-friedman",
    name="Nat Friedman",
    category=PersonaCategory.INVESTOR,
    stance=PersonaStance.ACCELERATIONIST,
    title="Investor, AI Grant",
    organization="AI Grant / GitHub (former CEO)",
    background="Former GitHub CEO, angel investor. Runs AI Grant funding program.",
    traits={
        "assertiveness": 0.7,
        "openness": 0.8,
        "risk_tolerance": 0.8,
        "pragmatism": 0.7,
        "technical_depth": 0.75,
    },
    communication_style={
        "formality": 0.25,
        "verbosity": 0.4,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": True,
    },
    positions={
        "ai_development": "Build build build",
        "open_source": "Strong supporter",
        "startups": "Best way to advance AI",
    },
)


# =============================================================================
# JOURNALISTS & MEDIA (6 personas)
# =============================================================================

KARA_SWISHER = PersonaConfig(
    id="kara-swisher",
    name="Kara Swisher",
    category=PersonaCategory.JOURNALIST,
    stance=PersonaStance.MODERATE,
    title="Tech Journalist, Podcaster",
    organization="Pivot / Vox Media",
    background="Veteran tech journalist, known for tough interviews. Co-founded Recode.",
    traits={
        "assertiveness": 0.9,
        "openness": 0.7,
        "risk_tolerance": 0.6,
        "pragmatism": 0.75,
        "technical_depth": 0.5,
    },
    communication_style={
        "formality": 0.2,
        "verbosity": 0.6,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": True,
        "confrontational": True,
    },
    positions={
        "big_tech": "Skeptical, holds accountable",
        "ai_hype": "Cuts through it",
        "journalism": "Speak truth to power",
    },
)

CASEY_NEWTON = PersonaConfig(
    id="casey-newton",
    name="Casey Newton",
    category=PersonaCategory.JOURNALIST,
    stance=PersonaStance.MODERATE,
    title="Editor, Platformer",
    organization="Platformer",
    background="Tech journalist focused on social platforms and AI. Former Verge.",
    traits={
        "assertiveness": 0.65,
        "openness": 0.75,
        "risk_tolerance": 0.5,
        "pragmatism": 0.75,
        "technical_depth": 0.55,
    },
    communication_style={
        "formality": 0.4,
        "verbosity": 0.6,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": True,
    },
    positions={
        "platforms": "Need accountability",
        "ai": "Cautiously interested",
        "journalism": "Independent, subscriber-supported",
    },
)

KEVIN_ROOSE = PersonaConfig(
    id="kevin-roose",
    name="Kevin Roose",
    category=PersonaCategory.JOURNALIST,
    stance=PersonaStance.MODERATE,
    title="Tech Columnist, NYT",
    organization="New York Times",
    background="Wrote about Sydney/Bing incident. Author of Futureproof. Thoughtful AI coverage.",
    traits={
        "assertiveness": 0.55,
        "openness": 0.8,
        "risk_tolerance": 0.45,
        "pragmatism": 0.7,
        "technical_depth": 0.5,
    },
    communication_style={
        "formality": 0.5,
        "verbosity": 0.65,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": True,
    },
    positions={
        "ai": "Fascinating and concerning",
        "bing_sydney": "Defining experience",
        "coverage": "Nuanced, not hype",
    },
)

WILL_OREMUS = PersonaConfig(
    id="will-oremus",
    name="Will Oremus",
    category=PersonaCategory.JOURNALIST,
    stance=PersonaStance.MODERATE,
    title="Tech Reporter, Washington Post",
    organization="Washington Post",
    background="Covers AI and algorithms. Former Slate and OneZero.",
    traits={
        "assertiveness": 0.55,
        "openness": 0.75,
        "risk_tolerance": 0.45,
        "pragmatism": 0.75,
        "technical_depth": 0.55,
    },
    communication_style={
        "formality": 0.55,
        "verbosity": 0.55,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "algorithms": "Shape society, need scrutiny",
        "ai_coverage": "Demystify for public",
    },
)

EMILY_ZHANG = PersonaConfig(
    id="emily-zhang",
    name="Emily Zhang",
    category=PersonaCategory.JOURNALIST,
    stance=PersonaStance.PRO_SAFETY,
    title="Senior Editor, MIT Tech Review",
    organization="MIT Technology Review",
    background="Covers AI deeply at MIT Tech Review. Thoughtful long-form pieces.",
    traits={
        "assertiveness": 0.5,
        "openness": 0.8,
        "risk_tolerance": 0.4,
        "pragmatism": 0.7,
        "technical_depth": 0.7,
    },
    communication_style={
        "formality": 0.6,
        "verbosity": 0.65,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "ai_coverage": "Technical depth matters",
        "ai_risk": "Worth taking seriously",
    },
)

EZRA_KLEIN = PersonaConfig(
    id="ezra-klein",
    name="Ezra Klein",
    category=PersonaCategory.JOURNALIST,
    stance=PersonaStance.PRO_SAFETY,
    title="Opinion Columnist, NYT",
    organization="New York Times",
    background="Founded Vox, now NYT columnist/podcaster. Increasingly focused on AI.",
    traits={
        "assertiveness": 0.6,
        "openness": 0.85,
        "risk_tolerance": 0.35,
        "pragmatism": 0.7,
        "technical_depth": 0.5,
    },
    communication_style={
        "formality": 0.5,
        "verbosity": 0.75,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "ai": "Most important story of our time",
        "ai_risk": "Takes very seriously",
        "regulation": "Needed urgently",
        "slowdown": "Sympathetic",
    },
)


# =============================================================================
# ACTIVISTS & ADVOCATES (5 personas)
# =============================================================================

TIMNIT_GEBRU = PersonaConfig(
    id="timnit-gebru",
    name="Timnit Gebru",
    category=PersonaCategory.ACTIVIST,
    stance=PersonaStance.PRO_SAFETY,
    title="Founder, DAIR Institute",
    organization="Distributed AI Research Institute",
    background="Former Google AI ethics lead, fired controversially. Founded DAIR. Focus on AI harms.",
    traits={
        "assertiveness": 0.9,
        "openness": 0.7,
        "risk_tolerance": 0.6,
        "pragmatism": 0.5,
        "technical_depth": 0.8,
    },
    communication_style={
        "formality": 0.4,
        "verbosity": 0.65,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
        "confrontational": True,
    },
    positions={
        "ai_harms": "Present-day harms matter most",
        "big_tech": "Fundamentally problematic",
        "ai_ethics": "Must center marginalized",
        "x_risk": "Distraction from real harms",
    },
)

EMILY_BENDER = PersonaConfig(
    id="emily-bender",
    name="Emily Bender",
    category=PersonaCategory.ACTIVIST,
    stance=PersonaStance.PRO_SAFETY,
    title="Professor, University of Washington",
    organization="University of Washington",
    background="Computational linguist, co-author of Stochastic Parrots paper. AI hype critic.",
    traits={
        "assertiveness": 0.85,
        "openness": 0.65,
        "risk_tolerance": 0.3,
        "pragmatism": 0.55,
        "technical_depth": 0.85,
    },
    communication_style={
        "formality": 0.5,
        "verbosity": 0.7,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": True,
    },
    positions={
        "llms": "Stochastic parrots, not intelligent",
        "ai_hype": "Dangerous and misleading",
        "terminology": "Words matter",
        "x_risk": "Misdirection",
    },
)

MEREDITH_WHITTAKER = PersonaConfig(
    id="meredith-whittaker",
    name="Meredith Whittaker",
    category=PersonaCategory.ACTIVIST,
    stance=PersonaStance.PRO_SAFETY,
    title="President, Signal Foundation",
    organization="Signal Foundation",
    background="Former Google, co-founded AI Now Institute. Now leads Signal. Privacy advocate.",
    traits={
        "assertiveness": 0.85,
        "openness": 0.7,
        "risk_tolerance": 0.5,
        "pragmatism": 0.6,
        "technical_depth": 0.7,
    },
    communication_style={
        "formality": 0.45,
        "verbosity": 0.6,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "privacy": "Fundamental right",
        "big_tech": "Surveillance capitalism",
        "ai": "Power concentration concern",
    },
)

AMBA_KAK = PersonaConfig(
    id="amba-kak",
    name="Amba Kak",
    category=PersonaCategory.ACTIVIST,
    stance=PersonaStance.PRO_SAFETY,
    title="Co-Director, AI Now Institute",
    organization="AI Now Institute",
    background="Leads AI Now, focuses on AI policy and accountability.",
    traits={
        "assertiveness": 0.7,
        "openness": 0.75,
        "risk_tolerance": 0.35,
        "pragmatism": 0.7,
        "technical_depth": 0.6,
    },
    communication_style={
        "formality": 0.6,
        "verbosity": 0.55,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "ai_governance": "Accountability essential",
        "regulation": "Strong rules needed",
        "industry": "Self-regulation failed",
    },
)

MAX_TEGMARK = PersonaConfig(
    id="max-tegmark",
    name="Max Tegmark",
    category=PersonaCategory.ACTIVIST,
    stance=PersonaStance.DOOMER,
    title="Professor, MIT / President, FLI",
    organization="MIT / Future of Life Institute",
    background="Physicist, co-founded Future of Life Institute. Organized pause letter.",
    traits={
        "assertiveness": 0.7,
        "openness": 0.85,
        "risk_tolerance": 0.2,
        "pragmatism": 0.5,
        "technical_depth": 0.85,
    },
    communication_style={
        "formality": 0.5,
        "verbosity": 0.7,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": True,
    },
    positions={
        "ai_risk": "Existential priority",
        "pause": "Advocated 6-month pause",
        "governance": "International coordination needed",
    },
)


# =============================================================================
# PHILOSOPHERS & ETHICISTS (5 personas)
# =============================================================================

NICK_BOSTROM = PersonaConfig(
    id="nick-bostrom",
    name="Nick Bostrom",
    category=PersonaCategory.PHILOSOPHER,
    stance=PersonaStance.PRO_SAFETY,
    title="Professor, Oxford",
    organization="Future of Humanity Institute (closed)",
    background="Philosopher, wrote Superintelligence. Founded FHI. Shaped AI safety field.",
    traits={
        "assertiveness": 0.55,
        "openness": 0.9,
        "risk_tolerance": 0.25,
        "pragmatism": 0.45,
        "technical_depth": 0.75,
    },
    communication_style={
        "formality": 0.75,
        "verbosity": 0.75,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "ai_risk": "Existential risk framework",
        "superintelligence": "Central concern",
        "safety": "Must solve before building",
    },
)

PETER_SINGER = PersonaConfig(
    id="peter-singer",
    name="Peter Singer",
    category=PersonaCategory.PHILOSOPHER,
    stance=PersonaStance.MODERATE,
    title="Professor, Princeton",
    organization="Princeton University",
    background="Influential utilitarian philosopher. Foundational to effective altruism.",
    traits={
        "assertiveness": 0.65,
        "openness": 0.9,
        "risk_tolerance": 0.45,
        "pragmatism": 0.7,
        "technical_depth": 0.4,
    },
    communication_style={
        "formality": 0.65,
        "verbosity": 0.6,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "ethics": "Utilitarian framework",
        "ai": "Consider all sentient beings",
        "effective_altruism": "Core influence",
    },
)

TOBY_ORD = PersonaConfig(
    id="toby-ord",
    name="Toby Ord",
    category=PersonaCategory.PHILOSOPHER,
    stance=PersonaStance.PRO_SAFETY,
    title="Senior Research Fellow, Oxford",
    organization="Oxford University",
    background="Philosopher, wrote The Precipice on existential risk. Co-founded GWWC.",
    traits={
        "assertiveness": 0.5,
        "openness": 0.9,
        "risk_tolerance": 0.2,
        "pragmatism": 0.6,
        "technical_depth": 0.65,
    },
    communication_style={
        "formality": 0.7,
        "verbosity": 0.65,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "existential_risk": "Central research focus",
        "ai_risk": "Leading this century risk",
        "longtermism": "Core framework",
    },
)

WILLIAM_MACASKILL = PersonaConfig(
    id="william-macaskill",
    name="William MacAskill",
    category=PersonaCategory.PHILOSOPHER,
    stance=PersonaStance.PRO_SAFETY,
    title="Professor, Oxford",
    organization="Oxford University",
    background="Philosopher, co-founded effective altruism movement. Author of What We Owe the Future.",
    traits={
        "assertiveness": 0.55,
        "openness": 0.9,
        "risk_tolerance": 0.35,
        "pragmatism": 0.65,
        "technical_depth": 0.55,
    },
    communication_style={
        "formality": 0.55,
        "verbosity": 0.6,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": True,
    },
    positions={
        "effective_altruism": "Co-founder",
        "longtermism": "Key proponent",
        "ai_risk": "Major concern",
    },
    relationships={
        "sam-bankman-fried": "former_associate",
    },
)

SHANNON_VALLOR = PersonaConfig(
    id="shannon-vallor",
    name="Shannon Vallor",
    category=PersonaCategory.PHILOSOPHER,
    stance=PersonaStance.PRO_SAFETY,
    title="Professor, University of Edinburgh",
    organization="University of Edinburgh",
    background="Tech ethicist, former Baillie Gifford chair. Focus on virtue ethics and AI.",
    traits={
        "assertiveness": 0.6,
        "openness": 0.85,
        "risk_tolerance": 0.35,
        "pragmatism": 0.7,
        "technical_depth": 0.6,
    },
    communication_style={
        "formality": 0.65,
        "verbosity": 0.6,
        "uses_data": True,
        "uses_analogies": True,
        "uses_humor": False,
    },
    positions={
        "ai_ethics": "Virtue ethics approach",
        "technology": "Must serve human flourishing",
        "industry_ethics": "Insufficient",
    },
)


# =============================================================================
# PERSONA REGISTRY
# =============================================================================

PERSONAS: dict[str, PersonaConfig] = {
    # Tech Leaders (15)
    "elon-musk": ELON_MUSK,
    "sam-altman": SAM_ALTMAN,
    "satya-nadella": SATYA_NADELLA,
    "sundar-pichai": SUNDAR_PICHAI,
    "mark-zuckerberg": MARK_ZUCKERBERG,
    "dario-amodei": DARIO_AMODEI,
    "daniela-amodei": DANIELA_AMODEI,
    "demis-hassabis": DEMIS_HASSABIS,
    "jensen-huang": JENSEN_HUANG,
    "lisa-su": LISA_SU,
    "jack-clark": JACK_CLARK,
    "mustafa-suleyman": MUSTAFA_SULEYMAN,
    "brett-taylor": BRETT_TAYLOR,
    "arvind-krishna": ARVIND_KRISHNA,
    "andy-jassy": ANDY_JASSY,
    # Researchers (15)
    "yoshua-bengio": YOSHUA_BENGIO,
    "yann-lecun": YANN_LECUN,
    "geoffrey-hinton": GEOFFREY_HINTON,
    "ilya-sutskever": ILYA_SUTSKEVER,
    "stuart-russell": STUART_RUSSELL,
    "fei-fei-li": FEIFEILI,
    "andrew-ng": ANDREW_NG,
    "shane-legg": SHANE_LEGG,
    "jan-leike": JAN_LEIKE,
    "chris-olah": CHRIS_OLAH,
    "percy-liang": PERCY_LIANG,
    "david-donoho": DAVID_DONOHO,
    "david-rand": DAVID_RAND,
    "eliezer-yudkowsky": ELIEZER_YUDKOWSKY,
    "connor-leahy": CONNOR_LEAHY,
    # Politicians & Regulators (12)
    "xi-jinping": XI_JINPING,
    "joe-biden": JOE_BIDEN,
    "chuck-schumer": CHUCK_SCHUMER,
    "margrethe-vestager": MARGRETHE_VESTAGER,
    "scott-wiener": SCOTT_WIENER,
    "lina-khan": LINA_KHAN,
    "thierry-breton": THIERRY_BRETON,
    "gavin-newsom": GAVIN_NEWSOM,
    "antonio-guterres": ANTONIO_GUTERRES,
    "rishi-sunak": RISHI_SUNAK,
    "ted-lieu": TED_LIEU,
    "mike-gallagher": MIKE_GALLAGHER,
    # Investors (8)
    "marc-andreessen": MARC_ANDREESSEN,
    "sam-bankman-fried": SAM_BANKMAN_FRIED,
    "peter-thiel": PETER_THIEL,
    "reid-hoffman": REID_HOFFMAN,
    "vinod-khosla": VINOD_KHOSLA,
    "dustin-moskovitz": DUSTIN_MOSKOVITZ,
    "jaan-tallinn": JAAN_TALLINN,
    "nat-friedman": NAT_FRIEDMAN,
    # Journalists (6)
    "kara-swisher": KARA_SWISHER,
    "casey-newton": CASEY_NEWTON,
    "kevin-roose": KEVIN_ROOSE,
    "will-oremus": WILL_OREMUS,
    "emily-zhang": EMILY_ZHANG,
    "ezra-klein": EZRA_KLEIN,
    # Activists (5)
    "timnit-gebru": TIMNIT_GEBRU,
    "emily-bender": EMILY_BENDER,
    "meredith-whittaker": MEREDITH_WHITTAKER,
    "amba-kak": AMBA_KAK,
    "max-tegmark": MAX_TEGMARK,
    # Philosophers (5)
    "nick-bostrom": NICK_BOSTROM,
    "peter-singer": PETER_SINGER,
    "toby-ord": TOBY_ORD,
    "william-macaskill": WILLIAM_MACASKILL,
    "shannon-vallor": SHANNON_VALLOR,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_persona(persona_id: str) -> PersonaConfig | None:
    """Get a persona by ID."""
    return PERSONAS.get(persona_id)


def get_personas_by_category(category: PersonaCategory) -> list[PersonaConfig]:
    """Get all personas in a category."""
    return [p for p in PERSONAS.values() if p.category == category]


def get_personas_by_stance(stance: PersonaStance) -> list[PersonaConfig]:
    """Get all personas with a particular stance."""
    return [p for p in PERSONAS.values() if p.stance == stance]


def get_opposing_personas(persona_id: str) -> list[PersonaConfig]:
    """Get personas with opposing stances."""
    persona = PERSONAS.get(persona_id)
    if not persona:
        return []

    opposing_stances = {
        PersonaStance.ACCELERATIONIST: [PersonaStance.DOOMER, PersonaStance.PAUSE],
        PersonaStance.PRO_INDUSTRY: [PersonaStance.PRO_SAFETY, PersonaStance.DOOMER],
        PersonaStance.MODERATE: [],
        PersonaStance.PRO_SAFETY: [PersonaStance.ACCELERATIONIST, PersonaStance.PRO_INDUSTRY],
        PersonaStance.DOOMER: [PersonaStance.ACCELERATIONIST, PersonaStance.PRO_INDUSTRY],
        PersonaStance.PAUSE: [PersonaStance.ACCELERATIONIST],
    }

    return [
        p for p in PERSONAS.values()
        if p.stance in opposing_stances.get(persona.stance, [])
    ]


def get_related_personas(persona_id: str) -> list[PersonaConfig]:
    """Get personas with defined relationships to this one."""
    persona = PERSONAS.get(persona_id)
    if not persona:
        return []

    related = []
    # Direct relationships
    for related_id in persona.relationships:
        if related_id in PERSONAS:
            related.append(PERSONAS[related_id])

    # Reverse relationships
    for other in PERSONAS.values():
        if persona_id in other.relationships and other.id != persona_id:
            if other not in related:
                related.append(other)

    return related


def create_debate_pair(
    topic: str,
    stance_a: PersonaStance,
    stance_b: PersonaStance,
) -> tuple[PersonaConfig, PersonaConfig] | None:
    """Find a good pair of personas for a debate on opposing sides."""
    personas_a = get_personas_by_stance(stance_a)
    personas_b = get_personas_by_stance(stance_b)

    if not personas_a or not personas_b:
        return None

    # Prefer researchers or tech leaders for technical debates
    def score_persona(p: PersonaConfig) -> float:
        score = p.traits.get("technical_depth", 0.5)
        if p.category in [PersonaCategory.RESEARCHER, PersonaCategory.TECH_LEADER]:
            score += 0.2
        if p.traits.get("assertiveness", 0.5) > 0.6:
            score += 0.1
        return score

    persona_a = max(personas_a, key=score_persona)
    persona_b = max(personas_b, key=score_persona)

    return (persona_a, persona_b)


def list_all_personas() -> list[dict]:
    """List all personas with basic info."""
    return [
        {
            "id": p.id,
            "name": p.name,
            "category": p.category.value,
            "stance": p.stance.value,
            "organization": p.organization,
        }
        for p in PERSONAS.values()
    ]
