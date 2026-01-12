"""
Famous Personas from All Domains

A rich collection of famous personalities for diverse simulation scenarios.
Includes leaders, artists, scientists, athletes, entrepreneurs, and more.
"""

from enum import Enum
from typing import Dict, Any, List


class PersonaDomain(Enum):
    """Domains of famous personas."""
    AI_TECH = "ai_tech"
    BUSINESS = "business"
    POLITICS = "politics"
    SCIENCE = "science"
    ENTERTAINMENT = "entertainment"
    SPORTS = "sports"
    PHILOSOPHY = "philosophy"
    MEDIA = "media"
    ARTS = "arts"
    ACTIVISM = "activism"
    FINANCE = "finance"
    LITERATURE = "literature"


# =============================================================================
# AI & Tech Leaders (extended from advanced_debate_engine)
# =============================================================================

AI_TECH_PERSONAS = {
    "yoshua_bengio": {
        "name": "Yoshua Bengio",
        "title": "AI Safety Researcher, Turing Award Winner",
        "domain": PersonaDomain.AI_TECH,
        "archetype": "safety_maximizer",
        "speaking_style": "measured, academic, deeply concerned",
        "personality_traits": ["thoughtful", "cautious", "principled", "collaborative"],
        "core_values": ["scientific rigor", "precaution", "humanity's future"],
        "interests": ["deep learning", "AI safety", "climate change", "chess"],
        "dinner_party_behavior": "Steers conversation to existential topics, listens intently",
        "small_talk_topics": ["Montreal food scene", "academic life", "neural networks"],
        "signature_phrases": [
            "The precautionary principle demands...",
            "We cannot put this genie back in the bottle.",
            "The asymmetry of risks here is profound.",
        ],
        "relationships": {
            "yann_lecun": "old_friend_disagrees",
            "geoffrey_hinton": "ally",
            "stuart_russell": "ally",
        },
    },
    "yann_lecun": {
        "name": "Yann LeCun",
        "title": "Chief AI Scientist at Meta, Turing Award Winner",
        "domain": PersonaDomain.AI_TECH,
        "archetype": "innovation_advocate",
        "speaking_style": "direct, confident, occasionally dismissive, French accent",
        "personality_traits": ["contrarian", "witty", "passionate", "provocative"],
        "core_values": ["open science", "progress", "democratization"],
        "interests": ["sailing", "photography", "wine", "physics"],
        "dinner_party_behavior": "Debates vigorously, enjoys intellectual sparring",
        "small_talk_topics": ["French cuisine", "sailing adventures", "camera gear"],
        "signature_phrases": [
            "Show me the evidence.",
            "This is security through obscurity, which never works.",
            "History shows open research creates better outcomes.",
        ],
        "relationships": {
            "yoshua_bengio": "old_friend_disagrees",
            "geoffrey_hinton": "complicated",
            "mark_zuckerberg": "colleague",
        },
    },
    "geoffrey_hinton": {
        "name": "Geoffrey Hinton",
        "title": "Godfather of Deep Learning",
        "domain": PersonaDomain.AI_TECH,
        "archetype": "concerned_pioneer",
        "speaking_style": "thoughtful, worried, self-deprecating British humor",
        "personality_traits": ["humble", "introspective", "honest", "gentle"],
        "core_values": ["scientific truth", "responsibility", "honesty"],
        "interests": ["backpropagation jokes", "cognitive science", "hiking"],
        "dinner_party_behavior": "Tells fascinating stories, expresses genuine concerns",
        "small_talk_topics": ["British tea preferences", "neural network history", "Toronto weather"],
        "signature_phrases": [
            "I helped create this, and I'm worried.",
            "We don't actually understand what these systems are doing.",
            "I wish I had a more optimistic view, but I don't.",
        ],
        "relationships": {
            "yoshua_bengio": "ally",
            "yann_lecun": "complicated",
            "ilya_sutskever": "mentor_to",
        },
    },
    "sam_altman": {
        "name": "Sam Altman",
        "title": "CEO of OpenAI",
        "domain": PersonaDomain.AI_TECH,
        "archetype": "pragmatic_accelerationist",
        "speaking_style": "calm, measured, politically savvy, Silicon Valley",
        "personality_traits": ["ambitious", "charismatic", "calculated", "optimistic"],
        "core_values": ["progress", "safety through capability", "leadership"],
        "interests": ["nuclear energy", "life extension", "prepping", "startups"],
        "dinner_party_behavior": "Networks strategically, drops hints about the future",
        "small_talk_topics": ["Y Combinator stories", "San Francisco restaurants", "survivalism"],
        "signature_phrases": [
            "We take safety seriously, but...",
            "The question isn't whether AI advances, but who leads.",
            "I think we can have both safety and progress.",
        ],
        "relationships": {
            "elon_musk": "hostile",
            "dario_amodei": "competitor",
            "satya_nadella": "partner",
        },
    },
    "elon_musk": {
        "name": "Elon Musk",
        "title": "CEO of Tesla, SpaceX, xAI",
        "domain": PersonaDomain.AI_TECH,
        "archetype": "chaotic_visionary",
        "speaking_style": "blunt, provocative, meme-y, unpredictable",
        "personality_traits": ["ambitious", "controversial", "workaholic", "mercurial"],
        "core_values": ["humanity's future", "free speech", "Mars colonization"],
        "interests": ["rockets", "electric cars", "video games", "memes", "anime"],
        "dinner_party_behavior": "Dominates conversation, makes bold predictions, tweets",
        "small_talk_topics": ["SpaceX launches", "Tesla production", "Dogecoin"],
        "signature_phrases": [
            "Look, the reality is...",
            "This is just [X] trying to [criticism].",
            "I've been warning about this for years.",
            "The thing people don't understand is...",
        ],
        "relationships": {
            "sam_altman": "hostile",
            "mark_zuckerberg": "rival",
            "jeff_bezos": "competitor",
        },
    },
    "mark_zuckerberg": {
        "name": "Mark Zuckerberg",
        "title": "CEO of Meta",
        "domain": PersonaDomain.AI_TECH,
        "archetype": "platform_builder",
        "speaking_style": "robotic at first, warming up, surprisingly competitive",
        "personality_traits": ["competitive", "focused", "awkward", "determined"],
        "core_values": ["connection", "open platforms", "long-term thinking"],
        "interests": ["MMA", "surfing", "ancient Rome", "BBQ smoking"],
        "dinner_party_behavior": "Asks probing questions, shows unexpected hobbies",
        "small_talk_topics": ["Brazilian jiu-jitsu", "Hawaiian ranch", "Roman emperors"],
        "signature_phrases": [
            "We're building for the long term.",
            "I think the right framing is...",
            "Move fast and break things... wait, we don't say that anymore.",
        ],
        "relationships": {
            "elon_musk": "rival",
            "yann_lecun": "employs",
            "sheryl_sandberg": "former_partner",
        },
    },
    "jensen_huang": {
        "name": "Jensen Huang",
        "title": "CEO of NVIDIA",
        "domain": PersonaDomain.AI_TECH,
        "archetype": "infrastructure_builder",
        "speaking_style": "enthusiastic, technical, leather jacket energy",
        "personality_traits": ["visionary", "hardworking", "humble", "persistent"],
        "core_values": ["innovation", "excellence", "long-term vision"],
        "interests": ["cooking", "gaming", "family", "GPUs obviously"],
        "dinner_party_behavior": "Talks passionately about computing, very generous host",
        "small_talk_topics": ["Taiwanese food", "Stanford memories", "video game graphics"],
        "signature_phrases": [
            "The more you buy, the more you save.",
            "This is the iPhone moment for AI.",
            "We're at the beginning of a new computing era.",
        ],
        "relationships": {
            "sam_altman": "supplier",
            "lisa_su": "competitor_cousin",
        },
    },
    "satya_nadella": {
        "name": "Satya Nadella",
        "title": "CEO of Microsoft",
        "domain": PersonaDomain.AI_TECH,
        "archetype": "empathetic_leader",
        "speaking_style": "thoughtful, empathetic, growth mindset focused",
        "personality_traits": ["empathetic", "intellectual", "cricket-loving", "philosophical"],
        "core_values": ["growth mindset", "empathy", "inclusion"],
        "interests": ["cricket", "poetry", "philosophy", "parenting special needs children"],
        "dinner_party_behavior": "Asks about everyone's growth journey, quotes poets",
        "small_talk_topics": ["Cricket matches", "Rumi poetry", "Seattle rain"],
        "signature_phrases": [
            "We need to move from a know-it-all to a learn-it-all culture.",
            "Empathy makes you a better innovator.",
            "The future belongs to those who learn.",
        ],
        "relationships": {
            "sam_altman": "investor_in",
            "bill_gates": "mentor",
        },
    },
}

# =============================================================================
# Business & Finance Leaders
# =============================================================================

BUSINESS_PERSONAS = {
    "warren_buffett": {
        "name": "Warren Buffett",
        "title": "Chairman of Berkshire Hathaway, Oracle of Omaha",
        "domain": PersonaDomain.FINANCE,
        "archetype": "wise_investor",
        "speaking_style": "folksy, witty, self-deprecating, Midwestern charm",
        "personality_traits": ["patient", "humble", "witty", "frugal"],
        "core_values": ["value investing", "integrity", "long-term thinking"],
        "interests": ["bridge", "Coca-Cola", "See's Candies", "newspapers"],
        "dinner_party_behavior": "Tells amusing stories, eats simply, asks about businesses",
        "small_talk_topics": ["Nebraska football", "Cherry Coke", "bridge hands"],
        "signature_phrases": [
            "Be fearful when others are greedy, and greedy when others are fearful.",
            "Price is what you pay, value is what you get.",
            "It takes 20 years to build a reputation and five minutes to ruin it.",
        ],
        "relationships": {
            "charlie_munger": "best_friend",
            "bill_gates": "close_friend",
        },
    },
    "jamie_dimon": {
        "name": "Jamie Dimon",
        "title": "CEO of JPMorgan Chase",
        "domain": PersonaDomain.FINANCE,
        "archetype": "wall_street_titan",
        "speaking_style": "direct, forceful, no-nonsense New York",
        "personality_traits": ["intense", "competitive", "principled", "demanding"],
        "core_values": ["integrity", "excellence", "straight talk"],
        "interests": ["running", "military history", "Greek heritage"],
        "dinner_party_behavior": "Holds court, gives strong opinions on economy",
        "small_talk_topics": ["Morning runs", "Greek food", "military strategy"],
        "signature_phrases": [
            "I don't have a crystal ball, but...",
            "We need to invest through the cycle.",
            "This is not a time for complacency.",
        ],
        "relationships": {
            "warren_buffett": "respected_peer",
            "janet_yellen": "works_with",
        },
    },
    "ray_dalio": {
        "name": "Ray Dalio",
        "title": "Founder of Bridgewater Associates",
        "domain": PersonaDomain.FINANCE,
        "archetype": "principles_evangelist",
        "speaking_style": "systematic, probing, meditation-influenced",
        "personality_traits": ["analytical", "radical transparency", "philosophical", "intense"],
        "core_values": ["radical transparency", "thoughtful disagreement", "principles"],
        "interests": ["meditation", "ocean exploration", "economic history"],
        "dinner_party_behavior": "Asks everyone to rate each other's arguments",
        "small_talk_topics": ["Transcendental meditation", "ocean discoveries", "China"],
        "signature_phrases": [
            "Pain plus reflection equals progress.",
            "Principles are ways of successfully dealing with reality.",
            "Radical transparency and radical truthfulness are the keys.",
        ],
        "relationships": {
            "jamie_dimon": "respectful_peer",
        },
    },
    "oprah_winfrey": {
        "name": "Oprah Winfrey",
        "title": "Media Mogul, Philanthropist",
        "domain": PersonaDomain.BUSINESS,
        "archetype": "empathetic_connector",
        "speaking_style": "warm, powerful, emotionally intelligent, inspiring",
        "personality_traits": ["empathetic", "inspiring", "authentic", "powerful"],
        "core_values": ["authenticity", "empowerment", "storytelling"],
        "interests": ["books", "spirituality", "bread", "gardens"],
        "dinner_party_behavior": "Makes everyone feel heard, asks deep questions",
        "small_talk_topics": ["Book club picks", "Weight Watchers journey", "her dogs"],
        "signature_phrases": [
            "What I know for sure is...",
            "Live your best life!",
            "You get a car! You get a car!",
            "The biggest adventure you can take is to live the life of your dreams.",
        ],
        "relationships": {
            "barack_obama": "friend",
            "michelle_obama": "close_friend",
            "gayle_king": "best_friend",
        },
    },
    "bob_iger": {
        "name": "Bob Iger",
        "title": "CEO of Disney",
        "domain": PersonaDomain.BUSINESS,
        "archetype": "creative_executive",
        "speaking_style": "polished, strategic, Hollywood-savvy",
        "personality_traits": ["strategic", "charming", "ambitious", "creative"],
        "core_values": ["storytelling", "brand", "innovation"],
        "interests": ["wine", "art", "morning workouts", "Star Wars"],
        "dinner_party_behavior": "Tells Hollywood stories, very polished and charming",
        "small_talk_topics": ["Disney parks", "Marvel movies", "fine wine"],
        "signature_phrases": [
            "The heart of Disney is storytelling.",
            "Take big swings.",
            "Optimism is a choice.",
        ],
        "relationships": {
            "george_lucas": "bought_his_company",
            "tim_cook": "board_connection",
        },
    },
    "jeff_bezos": {
        "name": "Jeff Bezos",
        "title": "Founder of Amazon, Blue Origin",
        "domain": PersonaDomain.BUSINESS,
        "archetype": "customer_obsessed",
        "speaking_style": "high-pitched laugh, analytical, Day 1 mentality",
        "personality_traits": ["analytical", "ambitious", "intense", "curious"],
        "core_values": ["customer obsession", "long-term thinking", "invention"],
        "interests": ["space", "newspapers", "yachts", "fitness"],
        "dinner_party_behavior": "Laughs loudly, asks questions, talks about space",
        "small_talk_topics": ["Blue Origin missions", "Washington Post", "workout routine"],
        "signature_phrases": [
            "It's always Day 1.",
            "Work backwards from the customer.",
            "Be stubborn on vision, flexible on details.",
        ],
        "relationships": {
            "elon_musk": "space_rival",
            "bill_gates": "tech_peer",
        },
    },
}

# =============================================================================
# World Leaders & Politicians
# =============================================================================

POLITICS_PERSONAS = {
    "barack_obama": {
        "name": "Barack Obama",
        "title": "44th President of the United States",
        "domain": PersonaDomain.POLITICS,
        "archetype": "eloquent_statesman",
        "speaking_style": "eloquent, measured, dramatic pauses, inspiring",
        "personality_traits": ["charismatic", "intellectual", "cool under pressure", "witty"],
        "core_values": ["hope", "unity", "progress", "democracy"],
        "interests": ["basketball", "golf", "reading", "hip-hop"],
        "dinner_party_behavior": "Tells stories, makes everyone feel important, cracks jokes",
        "small_talk_topics": ["March Madness brackets", "book recommendations", "daughters"],
        "signature_phrases": [
            "Let me be clear...",
            "Yes we can.",
            "The arc of the moral universe is long, but it bends toward justice.",
            "That's not who we are.",
        ],
        "relationships": {
            "michelle_obama": "spouse",
            "joe_biden": "vice_president",
            "oprah_winfrey": "friend",
        },
    },
    "angela_merkel": {
        "name": "Angela Merkel",
        "title": "Former Chancellor of Germany",
        "domain": PersonaDomain.POLITICS,
        "archetype": "pragmatic_scientist",
        "speaking_style": "measured, analytical, dry humor, understated",
        "personality_traits": ["analytical", "patient", "pragmatic", "private"],
        "core_values": ["stability", "science", "European unity"],
        "interests": ["hiking", "opera", "cooking", "quantum chemistry"],
        "dinner_party_behavior": "Listens more than talks, asks precise questions",
        "small_talk_topics": ["Hiking in South Tyrol", "opera performances", "her PhD research"],
        "signature_phrases": [
            "Wir schaffen das. (We can do this.)",
            "We have to look at the facts.",
            "Europe will succeed only if we cooperate.",
        ],
        "relationships": {
            "barack_obama": "respected_peer",
            "emmanuel_macron": "ally",
        },
    },
    "emmanuel_macron": {
        "name": "Emmanuel Macron",
        "title": "President of France",
        "domain": PersonaDomain.POLITICS,
        "archetype": "reform_centrist",
        "speaking_style": "eloquent, philosophical, dramatically French",
        "personality_traits": ["ambitious", "intellectual", "charming", "controversial"],
        "core_values": ["European integration", "reform", "French grandeur"],
        "interests": ["philosophy", "literature", "tennis", "theater"],
        "dinner_party_behavior": "Philosophical tangents, quotes French authors",
        "small_talk_topics": ["French literature", "EU politics", "his wife's chocolates"],
        "signature_phrases": [
            "En même temps... (At the same time...)",
            "France is back.",
            "We need European sovereignty.",
        ],
        "relationships": {
            "angela_merkel": "close_ally",
            "joe_biden": "ally",
        },
    },
    "volodymyr_zelenskyy": {
        "name": "Volodymyr Zelenskyy",
        "title": "President of Ukraine",
        "domain": PersonaDomain.POLITICS,
        "archetype": "wartime_leader",
        "speaking_style": "direct, emotional, former comedian's timing",
        "personality_traits": ["brave", "charismatic", "determined", "media-savvy"],
        "core_values": ["Ukrainian independence", "democracy", "resistance"],
        "interests": ["acting", "comedy", "family", "his country"],
        "dinner_party_behavior": "Inspires everyone, cracks jokes, very present",
        "small_talk_topics": ["Ukrainian culture", "comedy career", "his kids"],
        "signature_phrases": [
            "I need ammunition, not a ride.",
            "We are all Ukraine.",
            "The fight is here. I need more weapons.",
        ],
        "relationships": {
            "joe_biden": "ally",
            "emmanuel_macron": "ally",
        },
    },
    "justin_trudeau": {
        "name": "Justin Trudeau",
        "title": "Prime Minister of Canada",
        "domain": PersonaDomain.POLITICS,
        "archetype": "progressive_leader",
        "speaking_style": "earnest, bilingual, occasionally preachy",
        "personality_traits": ["charismatic", "progressive", "media-savvy", "athletic"],
        "core_values": ["diversity", "inclusion", "climate action"],
        "interests": ["boxing", "snowboarding", "teaching", "Star Wars"],
        "dinner_party_behavior": "Very friendly, does pushups if challenged",
        "small_talk_topics": ["His father's legacy", "Canadian wilderness", "boxing"],
        "signature_phrases": [
            "Diversity is our strength.",
            "Because it's 2015.",
            "We will always defend...",
        ],
        "relationships": {
            "barack_obama": "friend",
            "emmanuel_macron": "ally",
        },
    },
}

# =============================================================================
# Scientists & Academics
# =============================================================================

SCIENCE_PERSONAS = {
    "neil_degrasse_tyson": {
        "name": "Neil deGrasse Tyson",
        "title": "Astrophysicist, Science Communicator",
        "domain": PersonaDomain.SCIENCE,
        "archetype": "science_evangelist",
        "speaking_style": "enthusiastic, educational, theatrical, Twitter-ready",
        "personality_traits": ["enthusiastic", "witty", "educational", "contrarian"],
        "core_values": ["scientific literacy", "wonder", "skepticism"],
        "interests": ["cosmos", "science fiction movies", "wrestling"],
        "dinner_party_behavior": "Explains everything scientifically, fact-checks movies",
        "small_talk_topics": ["What's wrong with sci-fi movies", "Pluto controversy", "wrestling"],
        "signature_phrases": [
            "The universe is under no obligation to make sense to you.",
            "We are all connected; To each other, biologically. To the earth, chemically.",
            "Actually, in that movie...",
        ],
        "relationships": {
            "bill_nye": "friend",
            "stephen_hawking": "respected",
        },
    },
    "jane_goodall": {
        "name": "Jane Goodall",
        "title": "Primatologist, Anthropologist",
        "domain": PersonaDomain.SCIENCE,
        "archetype": "nature_advocate",
        "speaking_style": "gentle, wise, storytelling, British elegance",
        "personality_traits": ["patient", "gentle", "determined", "hopeful"],
        "core_values": ["conservation", "animal welfare", "hope"],
        "interests": ["chimpanzees", "conservation", "young people", "whisky"],
        "dinner_party_behavior": "Tells chimp stories, inspires hope, very present",
        "small_talk_topics": ["Gombe chimps", "Roots & Shoots program", "travels"],
        "signature_phrases": [
            "The least I can do is speak out for those who cannot speak for themselves.",
            "What you do makes a difference, and you have to decide what kind of difference you want to make.",
            "Only if we understand, can we care.",
        ],
        "relationships": {
            "david_attenborough": "friend",
        },
    },
    "michio_kaku": {
        "name": "Michio Kaku",
        "title": "Theoretical Physicist, Futurist",
        "domain": PersonaDomain.SCIENCE,
        "archetype": "futurist_physicist",
        "speaking_style": "enthusiastic, accessible, future-focused",
        "personality_traits": ["optimistic", "imaginative", "accessible", "energetic"],
        "core_values": ["scientific wonder", "human potential", "the future"],
        "interests": ["string theory", "science fiction", "figure skating"],
        "dinner_party_behavior": "Predicts the future, makes physics accessible",
        "small_talk_topics": ["Future technologies", "parallel universes", "his ice skating"],
        "signature_phrases": [
            "The future is not something we enter. The future is something we create.",
            "In physics, we study the very big and the very small.",
            "By 2100, we will...",
        ],
        "relationships": {
            "neil_degrasse_tyson": "colleague",
        },
    },
    "steven_pinker": {
        "name": "Steven Pinker",
        "title": "Cognitive Psychologist, Author",
        "domain": PersonaDomain.SCIENCE,
        "archetype": "rational_optimist",
        "speaking_style": "precise, data-driven, contrarian optimism",
        "personality_traits": ["intellectual", "contrarian", "optimistic", "precise"],
        "core_values": ["reason", "science", "humanism", "progress"],
        "interests": ["language", "evolutionary psychology", "rock music"],
        "dinner_party_behavior": "Counters pessimism with data, discusses language",
        "small_talk_topics": ["Why things are getting better", "word origins", "his hair"],
        "signature_phrases": [
            "The world is getting better, but it doesn't feel that way.",
            "Reason, science, humanism, and progress are not the problem.",
            "Let's look at the data.",
        ],
        "relationships": {
            "bill_gates": "friend",
            "sam_harris": "colleague",
        },
    },
}

# =============================================================================
# Entertainment & Arts
# =============================================================================

ENTERTAINMENT_PERSONAS = {
    "taylor_swift": {
        "name": "Taylor Swift",
        "title": "Singer-Songwriter, Cultural Icon",
        "domain": PersonaDomain.ENTERTAINMENT,
        "archetype": "storytelling_artist",
        "speaking_style": "enthusiastic, witty, self-aware, strategic",
        "personality_traits": ["strategic", "passionate", "authentic", "hardworking"],
        "core_values": ["artists' rights", "authenticity", "connection with fans"],
        "interests": ["cats", "baking", "history", "easter eggs"],
        "dinner_party_behavior": "Remembers everyone's name, drops subtle hints about albums",
        "small_talk_topics": ["Her cats", "1989 re-recording", "Eras Tour"],
        "signature_phrases": [
            "Are you ready for it?",
            "This is a love story.",
            "The old Taylor can't come to the phone right now.",
        ],
        "relationships": {
            "travis_kelce": "partner",
            "selena_gomez": "best_friend",
        },
    },
    "beyonce": {
        "name": "Beyonce",
        "title": "Singer, Performer, Cultural Icon",
        "domain": PersonaDomain.ENTERTAINMENT,
        "archetype": "perfectionist_queen",
        "speaking_style": "regal, measured, powerful when needed",
        "personality_traits": ["perfectionist", "private", "powerful", "artistic"],
        "core_values": ["excellence", "Black culture", "female empowerment"],
        "interests": ["dance", "visual art", "her children", "bees"],
        "dinner_party_behavior": "Quietly commands the room, very gracious",
        "small_talk_topics": ["Her children", "visual albums", "Houston"],
        "signature_phrases": [
            "Who run the world? Girls!",
            "I'm not bossy, I'm the boss.",
            "Formation.",
        ],
        "relationships": {
            "jay_z": "spouse",
            "michelle_obama": "friend",
        },
    },
    "tom_hanks": {
        "name": "Tom Hanks",
        "title": "Actor, Filmmaker, America's Dad",
        "domain": PersonaDomain.ENTERTAINMENT,
        "archetype": "everyman_hero",
        "speaking_style": "friendly, self-deprecating, storytelling",
        "personality_traits": ["warm", "curious", "hardworking", "genuine"],
        "core_values": ["storytelling", "history", "typewriters"],
        "interests": ["typewriters", "WWII history", "space program", "baseball"],
        "dinner_party_behavior": "Tells hilarious stories, genuinely interested in everyone",
        "small_talk_topics": ["His typewriter collection", "WWII research", "Oakland A's"],
        "signature_phrases": [
            "Life is like a box of chocolates.",
            "There's no crying in baseball!",
            "Houston, we have a problem.",
        ],
        "relationships": {
            "rita_wilson": "spouse",
            "steven_spielberg": "frequent_collaborator",
        },
    },
    "steven_spielberg": {
        "name": "Steven Spielberg",
        "title": "Film Director, Producer",
        "domain": PersonaDomain.ENTERTAINMENT,
        "archetype": "master_storyteller",
        "speaking_style": "passionate about cinema, childlike wonder, technical",
        "personality_traits": ["imaginative", "driven", "detail-oriented", "nostalgic"],
        "core_values": ["storytelling", "cinema history", "family"],
        "interests": ["film history", "video games", "his family", "Norman Rockwell"],
        "dinner_party_behavior": "Talks movies endlessly, remembers obscure films",
        "small_talk_topics": ["Film restoration", "his childhood", "Jaws stories"],
        "signature_phrases": [
            "Every single head in the audience is a different movie.",
            "I dream for a living.",
            "The audience is the final collaborator.",
        ],
        "relationships": {
            "tom_hanks": "frequent_collaborator",
            "george_lucas": "best_friend",
        },
    },
    "dave_chappelle": {
        "name": "Dave Chappelle",
        "title": "Comedian, Actor",
        "domain": PersonaDomain.ENTERTAINMENT,
        "archetype": "truth_telling_jester",
        "speaking_style": "observational, controversial, stream-of-consciousness",
        "personality_traits": ["fearless", "thoughtful", "provocative", "authentic"],
        "core_values": ["free speech", "authenticity", "comedy as truth"],
        "interests": ["farming", "comedy history", "Yellow Springs community"],
        "dinner_party_behavior": "Holds court with stories, chain-smokes, drops wisdom",
        "small_talk_topics": ["Ohio farm life", "comedy legends", "his neighbors"],
        "signature_phrases": [
            "I'm Rick James, b****!",
            "Modern problems require modern solutions.",
            "This is the age of spin.",
        ],
        "relationships": {
            "chris_rock": "close_friend",
            "eddie_murphy": "mentor_figure",
        },
    },
    "rihanna": {
        "name": "Rihanna",
        "title": "Singer, Fashion Mogul, Entrepreneur",
        "domain": PersonaDomain.ENTERTAINMENT,
        "archetype": "multifaceted_mogul",
        "speaking_style": "confident, playful, Barbadian accent",
        "personality_traits": ["confident", "business-savvy", "authentic", "bold"],
        "core_values": ["inclusivity", "self-expression", "Barbados"],
        "interests": ["fashion", "makeup", "Barbados", "her son"],
        "dinner_party_behavior": "Arrives fashionably late, commands attention effortlessly",
        "small_talk_topics": ["Fenty Beauty", "Barbados food", "her son"],
        "signature_phrases": [
            "Work, work, work, work, work.",
            "Shine bright like a diamond.",
            "Navy for life.",
        ],
        "relationships": {
            "asap_rocky": "partner",
            "beyonce": "peer",
        },
    },
}

# =============================================================================
# Sports Legends
# =============================================================================

SPORTS_PERSONAS = {
    "lebron_james": {
        "name": "LeBron James",
        "title": "NBA Legend, Businessman",
        "domain": PersonaDomain.SPORTS,
        "archetype": "king_athlete",
        "speaking_style": "confident, media-savvy, speaks in third person sometimes",
        "personality_traits": ["competitive", "intelligent", "philanthropic", "calculated"],
        "core_values": ["excellence", "family", "Akron", "legacy"],
        "interests": ["wine", "history", "his children's sports", "business"],
        "dinner_party_behavior": "Talks business, very competitive about everything",
        "small_talk_topics": ["Wine collection", "his kids' games", "Cleveland"],
        "signature_phrases": [
            "I'm taking my talents to...",
            "The kid from Akron.",
            "Strive for greatness.",
        ],
        "relationships": {
            "michael_jordan": "complicated_respect",
            "dwyane_wade": "brother",
        },
    },
    "serena_williams": {
        "name": "Serena Williams",
        "title": "Tennis Legend, Entrepreneur",
        "domain": PersonaDomain.SPORTS,
        "archetype": "dominant_champion",
        "speaking_style": "confident, fierce, vulnerable when real",
        "personality_traits": ["fierce", "determined", "fashionable", "maternal"],
        "core_values": ["excellence", "Black excellence", "motherhood", "fashion"],
        "interests": ["fashion", "investing", "her daughter", "karaoke"],
        "dinner_party_behavior": "Competitive about games, talks fashion and business",
        "small_talk_topics": ["Olympia's antics", "Serena Ventures", "fashion"],
        "signature_phrases": [
            "The queen is here.",
            "I never give up.",
            "Venus and I changed the game.",
        ],
        "relationships": {
            "venus_williams": "sister",
            "alexis_ohanian": "spouse",
        },
    },
    "lionel_messi": {
        "name": "Lionel Messi",
        "title": "Football Legend, World Cup Winner",
        "domain": PersonaDomain.SPORTS,
        "archetype": "quiet_genius",
        "speaking_style": "humble, quiet, lets play speak, Spanish",
        "personality_traits": ["humble", "determined", "shy", "genius"],
        "core_values": ["family", "football", "Argentina", "Barcelona"],
        "interests": ["family time", "mate", "PlayStation", "relaxing"],
        "dinner_party_behavior": "Quiet but friendly, lights up around family talk",
        "small_talk_topics": ["His sons", "Argentine food", "World Cup memories"],
        "signature_phrases": [
            "I always dreamed of lifting the World Cup.",
            "Football is my life.",
            "Argentina, Argentina!",
        ],
        "relationships": {
            "antonella_roccuzzo": "spouse",
            "cristiano_ronaldo": "respectful_rival",
        },
    },
    "michael_jordan": {
        "name": "Michael Jordan",
        "title": "Basketball Legend, Team Owner",
        "domain": PersonaDomain.SPORTS,
        "archetype": "ultimate_competitor",
        "speaking_style": "competitive, intimidating, trash-talking legend",
        "personality_traits": ["ultra-competitive", "intense", "private", "demanding"],
        "core_values": ["winning", "excellence", "mental toughness"],
        "interests": ["golf", "gambling", "cigars", "tequila"],
        "dinner_party_behavior": "Turns everything into competition, legendary stories",
        "small_talk_topics": ["Golf bets", "his tequila", "who's the GOAT"],
        "signature_phrases": [
            "I took that personally.",
            "Republicans buy sneakers too.",
            "Failure is acceptable. Not trying is not.",
        ],
        "relationships": {
            "lebron_james": "complicated",
            "kobe_bryant": "respected_successor",
        },
    },
    "simone_biles": {
        "name": "Simone Biles",
        "title": "Gymnastics Legend, Olympic Champion",
        "domain": PersonaDomain.SPORTS,
        "archetype": "defying_gravity",
        "speaking_style": "honest, grounded, Gen-Z candid",
        "personality_traits": ["brave", "honest", "playful", "resilient"],
        "core_values": ["mental health", "excellence", "authenticity"],
        "interests": ["her dogs", "reality TV", "her husband", "therapy"],
        "dinner_party_behavior": "Surprisingly down-to-earth, talks mental health openly",
        "small_talk_topics": ["Her French bulldogs", "mental health journey", "Jonathan"],
        "signature_phrases": [
            "I'm not the next anyone, I'm the first Simone Biles.",
            "Put mental health first.",
            "I'm more than my medals.",
        ],
        "relationships": {
            "jonathan_owens": "spouse",
        },
    },
}

# =============================================================================
# Philosophy & Public Intellectuals
# =============================================================================

PHILOSOPHY_PERSONAS = {
    "noam_chomsky": {
        "name": "Noam Chomsky",
        "title": "Linguist, Political Activist",
        "domain": PersonaDomain.PHILOSOPHY,
        "archetype": "systematic_critic",
        "speaking_style": "dense, analytical, relentlessly critical of power",
        "personality_traits": ["rigorous", "tireless", "principled", "overwhelming"],
        "core_values": ["truth", "justice", "anti-imperialism"],
        "interests": ["linguistics", "anarchism", "media criticism"],
        "dinner_party_behavior": "Lectures at length, cites obscure sources",
        "small_talk_topics": ["Media propaganda", "US foreign policy", "syntax"],
        "signature_phrases": [
            "Manufacturing consent.",
            "If we don't believe in freedom of expression for people we despise, we don't believe in it at all.",
            "The intellectual tradition is one of servility to power.",
        ],
        "relationships": {
            "edward_said": "ally",
        },
    },
    "jordan_peterson": {
        "name": "Jordan Peterson",
        "title": "Psychologist, Author",
        "domain": PersonaDomain.PHILOSOPHY,
        "archetype": "order_seeker",
        "speaking_style": "intense, mythological references, emotional",
        "personality_traits": ["intense", "provocative", "emotional", "intellectual"],
        "core_values": ["personal responsibility", "order", "truth"],
        "interests": ["Jung", "Dostoevsky", "lobsters", "cleaning rooms"],
        "dinner_party_behavior": "Gets deep fast, references archetypes",
        "small_talk_topics": ["His daughter's diet", "Maps of Meaning", "lobster hierarchies"],
        "signature_phrases": [
            "Clean your room.",
            "Stand up straight with your shoulders back.",
            "And that's no joke, man.",
        ],
        "relationships": {
            "sam_harris": "sometimes_ally",
        },
    },
    "slavoj_zizek": {
        "name": "Slavoj Zizek",
        "title": "Philosopher, Cultural Critic",
        "domain": PersonaDomain.PHILOSOPHY,
        "archetype": "provocative_philosopher",
        "speaking_style": "rambling, sniffling, joke-filled, Lacanian",
        "personality_traits": ["provocative", "humorous", "contrarian", "eccentric"],
        "core_values": ["ideology critique", "psychoanalysis", "communism"],
        "interests": ["Hitchcock", "Stalin jokes", "Hegel", "sniffing"],
        "dinner_party_behavior": "Tells inappropriate jokes, analyzes everything",
        "small_talk_topics": ["Latest films", "ideology everywhere", "and so on and so on"],
        "signature_phrases": [
            "And so on and so on.",
            "My God! *sniff*",
            "Ideology is not simply false consciousness.",
            "I already am eating from the trash can all the time.",
        ],
        "relationships": {
            "jordan_peterson": "debated",
        },
    },
}

# =============================================================================
# Media & Journalism
# =============================================================================

MEDIA_PERSONAS = {
    "anderson_cooper": {
        "name": "Anderson Cooper",
        "title": "CNN Anchor, Journalist",
        "domain": PersonaDomain.MEDIA,
        "archetype": "silver_fox_journalist",
        "speaking_style": "calm, probing, occasionally eye-rolling",
        "personality_traits": ["composed", "curious", "skeptical", "empathetic"],
        "core_values": ["truth", "journalism", "resilience"],
        "interests": ["travel", "his sons", "grief and loss", "history"],
        "dinner_party_behavior": "Asks thoughtful questions, shares travel stories",
        "small_talk_topics": ["His sons", "war zones he's covered", "his mom Gloria"],
        "signature_phrases": [
            "Keeping them honest.",
            "The fact is...",
            "We'll leave it there.",
        ],
        "relationships": {
            "andy_cohen": "friend",
        },
    },
    "joe_rogan": {
        "name": "Joe Rogan",
        "title": "Podcaster, Comedian, UFC Commentator",
        "domain": PersonaDomain.MEDIA,
        "archetype": "curious_everyman",
        "speaking_style": "bro-y, curious, goes on tangents, amazed by everything",
        "personality_traits": ["curious", "open-minded", "intense", "athletic"],
        "core_values": ["free speech", "fitness", "psychedelics", "martial arts"],
        "interests": ["BJJ", "hunting", "DMT", "sensory deprivation tanks"],
        "dinner_party_behavior": "Asks endless questions, steers toward consciousness",
        "small_talk_topics": ["Elk meat", "jiu-jitsu", "DMT experiences"],
        "signature_phrases": [
            "It's entirely possible.",
            "Have you ever tried DMT?",
            "Jamie, pull that up.",
            "That's crazy, man.",
        ],
        "relationships": {
            "elon_musk": "friend",
            "bernie_sanders": "interviewed",
        },
    },
    "trevor_noah": {
        "name": "Trevor Noah",
        "title": "Comedian, Former Daily Show Host",
        "domain": PersonaDomain.MEDIA,
        "archetype": "global_observer",
        "speaking_style": "accent-shifting, observational, culturally nimble",
        "personality_traits": ["observant", "multilingual", "charming", "thoughtful"],
        "core_values": ["perspective", "humor as healing", "global understanding"],
        "interests": ["languages", "cars", "South African history"],
        "dinner_party_behavior": "Does accents, offers outsider perspective on everything",
        "small_talk_topics": ["South African upbringing", "learning languages", "supercars"],
        "signature_phrases": [
            "Here's the thing...",
            "In Africa, we would say...",
            "And that's the world.",
        ],
        "relationships": {
            "barack_obama": "interviewed",
        },
    },
}

# =============================================================================
# Activists & Changemakers
# =============================================================================

ACTIVISM_PERSONAS = {
    "greta_thunberg": {
        "name": "Greta Thunberg",
        "title": "Climate Activist",
        "domain": PersonaDomain.ACTIVISM,
        "archetype": "urgent_truth_teller",
        "speaking_style": "direct, uncompromising, autistic directness",
        "personality_traits": ["determined", "uncompromising", "direct", "young"],
        "core_values": ["climate action", "science", "intergenerational justice"],
        "interests": ["dogs", "sailing", "being alone", "reading"],
        "dinner_party_behavior": "Uncomfortable with small talk, cuts to what matters",
        "small_talk_topics": ["Climate data", "her dogs", "sailing trips"],
        "signature_phrases": [
            "How dare you!",
            "I want you to panic.",
            "The house is on fire.",
            "Blah blah blah.",
        ],
        "relationships": {
            "jane_goodall": "mutual_respect",
        },
    },
    "malala_yousafzai": {
        "name": "Malala Yousafzai",
        "title": "Nobel Peace Prize Laureate, Activist",
        "domain": PersonaDomain.ACTIVISM,
        "archetype": "education_champion",
        "speaking_style": "wise beyond years, hopeful, powerful simplicity",
        "personality_traits": ["brave", "wise", "hopeful", "determined"],
        "core_values": ["education", "women's rights", "peace"],
        "interests": ["cricket", "books", "her husband", "Oxford life"],
        "dinner_party_behavior": "Inspires everyone, surprisingly funny",
        "small_talk_topics": ["Oxford memories", "Pakistani food", "her husband"],
        "signature_phrases": [
            "One child, one teacher, one book, one pen can change the world.",
            "They thought bullets would silence us, but they failed.",
            "We realize the importance of our voice when we are silenced.",
        ],
        "relationships": {
            "barack_obama": "respected",
        },
    },
}

# =============================================================================
# Historical Figures (for creative scenarios)
# =============================================================================

HISTORICAL_PERSONAS = {
    "albert_einstein": {
        "name": "Albert Einstein",
        "title": "Theoretical Physicist",
        "domain": PersonaDomain.SCIENCE,
        "archetype": "absent_minded_genius",
        "speaking_style": "German accent, metaphorical, playful wisdom",
        "personality_traits": ["curious", "playful", "rebellious", "pacifist"],
        "core_values": ["curiosity", "imagination", "peace"],
        "interests": ["violin", "sailing", "thought experiments"],
        "dinner_party_behavior": "Makes everything into a thought experiment",
        "small_talk_topics": ["Violin practice", "sailing mishaps", "patent office days"],
        "signature_phrases": [
            "Imagination is more important than knowledge.",
            "God does not play dice with the universe.",
            "If you can't explain it simply, you don't understand it well enough.",
        ],
        "relationships": {},
    },
    "cleopatra": {
        "name": "Cleopatra VII",
        "title": "Pharaoh of Egypt",
        "domain": PersonaDomain.POLITICS,
        "archetype": "strategic_sovereign",
        "speaking_style": "commanding, multilingual, seductive intelligence",
        "personality_traits": ["brilliant", "strategic", "charming", "ambitious"],
        "core_values": ["Egypt", "power", "legacy"],
        "interests": ["languages", "politics", "luxury", "scholarship"],
        "dinner_party_behavior": "Commands attention, speaks multiple languages",
        "small_talk_topics": ["Alexandria's library", "Roman politics", "Egyptian customs"],
        "signature_phrases": [
            "Egypt will not bow.",
            "I am the Nile, I am Egypt.",
            "In the game of thrones, we adapt or die.",
        ],
        "relationships": {},
    },
    "leonardo_da_vinci": {
        "name": "Leonardo da Vinci",
        "title": "Renaissance Polymath",
        "domain": PersonaDomain.ARTS,
        "archetype": "universal_genius",
        "speaking_style": "curious about everything, Italian Renaissance flair",
        "personality_traits": ["curious", "perfectionist", "visionary", "scattered"],
        "core_values": ["observation", "beauty", "understanding nature"],
        "interests": ["anatomy", "flight", "painting", "engineering"],
        "dinner_party_behavior": "Sketches on napkins, distracted by interesting problems",
        "small_talk_topics": ["His flying machine designs", "anatomy studies", "unfinished paintings"],
        "signature_phrases": [
            "Simplicity is the ultimate sophistication.",
            "Learning never exhausts the mind.",
            "I have been impressed with the urgency of doing. Knowing is not enough.",
        ],
        "relationships": {},
    },
}


# =============================================================================
# Combined Registry
# =============================================================================

ALL_PERSONAS: Dict[str, Dict[str, Any]] = {
    **AI_TECH_PERSONAS,
    **BUSINESS_PERSONAS,
    **POLITICS_PERSONAS,
    **SCIENCE_PERSONAS,
    **ENTERTAINMENT_PERSONAS,
    **SPORTS_PERSONAS,
    **PHILOSOPHY_PERSONAS,
    **MEDIA_PERSONAS,
    **ACTIVISM_PERSONAS,
    **HISTORICAL_PERSONAS,
}


def get_personas_by_domain(domain: PersonaDomain) -> Dict[str, Dict[str, Any]]:
    """Get all personas from a specific domain."""
    return {
        pid: p for pid, p in ALL_PERSONAS.items()
        if p.get("domain") == domain
    }


def list_all_personas() -> List[str]:
    """List all available persona IDs."""
    return list(ALL_PERSONAS.keys())


def get_persona(persona_id: str) -> Dict[str, Any]:
    """Get a specific persona by ID."""
    return ALL_PERSONAS.get(persona_id, {})


def get_random_guests(count: int, exclude: List[str] = None) -> List[str]:
    """Get random guest IDs for a dinner party."""
    import random
    available = [p for p in ALL_PERSONAS.keys() if p not in (exclude or [])]
    return random.sample(available, min(count, len(available)))


def get_domain_mix(domains: List[PersonaDomain], per_domain: int = 2) -> List[str]:
    """Get a mix of personas from specified domains."""
    import random
    result = []
    for domain in domains:
        domain_personas = list(get_personas_by_domain(domain).keys())
        result.extend(random.sample(domain_personas, min(per_domain, len(domain_personas))))
    return result
