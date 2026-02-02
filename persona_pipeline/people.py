"""
100 Most Influential People in Chip, GPU & AI-Hardware Policy
For LIDA multi-agent simulation personas
"""

PEOPLE = {
    "regime_political": {
        "name": "Regime-Level Political Actors",
        "people": [
            {"name": "Joe Biden", "role": "sets export control & CHIPS Act regime"},
            {"name": "Donald Trump", "role": "regime switch, hard decoupling risk"},
            {"name": "Xi Jinping", "role": "chip self-sufficiency doctrine"},
            {"name": "Li Qiang", "role": "industrial execution power"},
            {"name": "Ursula von der Leyen", "role": "EU Chips Act + export alignment"},
            {"name": "Emmanuel Macron", "role": "EU industrial sovereignty"},
            {"name": "Olaf Scholz", "role": "fabs + export enforcement"},
            {"name": "Rishi Sunak", "role": "AI safety + compute governance"},
            {"name": "Fumio Kishida", "role": "lithography & alliance control"},
            {"name": "Lee Hsien Loong", "role": "hub for chip logistics & capital"},
        ]
    },
    "us_national_security": {
        "name": "US National Security & Export Control",
        "people": [
            {"name": "Jake Sullivan", "role": "AI export doctrine"},
            {"name": "Gina Raimondo", "role": "GPU export rules"},
            {"name": "Alan Estevez", "role": "enforcement authority"},
            {"name": "Antony Blinken", "role": "allied coordination"},
            {"name": "Lloyd Austin", "role": "defense-AI demand"},
            {"name": "Avril Haines", "role": "threat framing"},
            {"name": "Kathleen Hicks", "role": "military compute"},
            {"name": "Arati Prabhakar", "role": "compute thresholds"},
            {"name": "Tarun Chhabra", "role": "diffusion governance"},
            {"name": "Ann Neuberger", "role": "chip security"},
            {"name": "Lisa Monaco", "role": "enforcement risk"},
            {"name": "Lael Brainard", "role": "industrial coordination"},
            {"name": "Katherine Tai", "role": "trade retaliation"},
            {"name": "Chris Inglis", "role": "cyber-hardware risk"},
            {"name": "Elizabeth Sherwood-Randall", "role": "infrastructure risk"},
        ]
    },
    "us_congress": {
        "name": "US Congress",
        "people": [
            {"name": "Chuck Schumer", "role": "CHIPS Act architect"},
            {"name": "Mike Gallagher", "role": "hardline policy"},
            {"name": "Maria Cantwell", "role": "chip regulation"},
            {"name": "Mark Warner", "role": "AI security"},
            {"name": "Marco Rubio", "role": "China hawk"},
            {"name": "Todd Young", "role": "industrial policy"},
            {"name": "Josh Hawley", "role": "populist regulation"},
            {"name": "Ron Wyden", "role": "privacy & tech law"},
            {"name": "Michael McCaul", "role": "export alignment"},
            {"name": "Raja Krishnamoorthi", "role": "enforcement pressure"},
            {"name": "Jim Himes", "role": "oversight"},
            {"name": "Adam Schiff", "role": "intelligence framing"},
            {"name": "Cathy McMorris Rodgers", "role": "regulation"},
            {"name": "Jason Smith", "role": "trade leverage"},
            {"name": "Young Kim", "role": "alliance policy"},
        ]
    },
    "frontier_ai_gpu": {
        "name": "Frontier AI & GPU Controllers",
        "people": [
            {"name": "Jensen Huang", "role": "GPU choke point"},
            {"name": "Sam Altman", "role": "frontier training demand"},
            {"name": "Dario Amodei", "role": "safety-oriented scaling"},
            {"name": "Demis Hassabis", "role": "UK/EU leverage"},
            {"name": "Satya Nadella", "role": "cloud gatekeeper"},
            {"name": "Sundar Pichai", "role": "TPU + cloud"},
            {"name": "Mark Zuckerberg", "role": "open-weights diffusion"},
            {"name": "Elon Musk", "role": "compute + narrative"},
            {"name": "Lisa Su", "role": "GPU competition"},
            {"name": "Pat Gelsinger", "role": "national fabs"},
            {"name": "Masayoshi Son", "role": "capital allocator"},
            {"name": "Jack Clark", "role": "policy translation"},
            {"name": "Ilya Sutskever", "role": "alignment authority"},
            {"name": "Alexandr Wang", "role": "data + defense"},
            {"name": "Arthur Mensch", "role": "EU compute strategy"},
        ]
    },
    "semiconductor_supply_chain": {
        "name": "Semiconductor Supply-Chain Chokepoints",
        "people": [
            {"name": "C.C. Wei", "role": "advanced node monopoly"},
            {"name": "Christophe Fouquet", "role": "EUV lithography"},
            {"name": "Peter Wennink", "role": "export diplomacy"},
            {"name": "Sanjay Mehrotra", "role": "memory bottleneck"},
            {"name": "Chey Tae-won", "role": "HBM supply"},
            {"name": "Masayuki Omoto", "role": "fab tooling"},
            {"name": "Lip-Bu Tan", "role": "chip design software"},
            {"name": "Aart de Geus", "role": "EDA control"},
            {"name": "Mark Liu", "role": "geopolitical signaling"},
            {"name": "Victor Peng", "role": "FPGA influence"},
            {"name": "Hock Tan", "role": "interconnects"},
            {"name": "Ren Zhengfei", "role": "China self-reliance"},
            {"name": "Liang Mong Song", "role": "node leapfrogging"},
            {"name": "Robin Li", "role": "China LLM compute"},
            {"name": "Pony Ma", "role": "cloud diffusion"},
        ]
    },
    "capital_think_tanks": {
        "name": "Capital, Think Tanks & Governance",
        "people": [
            {"name": "Eric Schmidt", "role": "national strategy"},
            {"name": "Jason Matheny", "role": "diffusion frameworks"},
            {"name": "Yoshua Bengio", "role": "legitimacy anchor"},
            {"name": "Stuart Russell", "role": "control narrative"},
            {"name": "Max Tegmark", "role": "public pressure"},
            {"name": "Allan Dafoe", "role": "compute governance"},
            {"name": "Helen Toner", "role": "policy bridge"},
            {"name": "Ben Garfinkel", "role": "thresholds"},
            {"name": "James Manyika", "role": "AI strategy"},
            {"name": "Holden Karnofsky", "role": "funding leverage"},
            {"name": "Chris Meserole", "role": "diplomacy framing"},
            {"name": "Daniel Castro", "role": "industry advocacy"},
            {"name": "Adam Thierer", "role": "deregulatory voice"},
            {"name": "Marc Andreessen", "role": "anti-regulation"},
            {"name": "Peter Thiel", "role": "defense-AI capital"},
            {"name": "Reid Hoffman", "role": "soft power"},
            {"name": "Larry Fink", "role": "infrastructure capital"},
            {"name": "Ray Dalio", "role": "US-China narrative"},
            {"name": "Naval Ravikant", "role": "ideology diffusion"},
            {"name": "David Sacks", "role": "media leverage"},
            {"name": "Koen De Backer", "role": "global standards"},
            {"name": "Dan Huttenlocher", "role": "academic authority"},
            {"name": "Lynn Parker", "role": "funding direction"},
            {"name": "Fei-Fei Li", "role": "ethics legitimacy"},
            {"name": "Alex Rives", "role": "compute spillover"},
            {"name": "Jason Clinton", "role": "hardware risk"},
            {"name": "Leopold Aschenbrenner", "role": "catastrophic risk"},
            {"name": "Daniel Kokotajlo", "role": "timelines influence"},
            {"name": "Yann LeCun", "role": "open science pressure"},
            {"name": "Mira Murati", "role": "deployment control"},
        ]
    },
}


def get_all_people():
    """Get flat list of all people with their category."""
    all_people = []
    for category_id, category in PEOPLE.items():
        for person in category["people"]:
            all_people.append({
                "name": person["name"],
                "role": person["role"],
                "category_id": category_id,
                "category_name": category["name"],
            })
    return all_people


def get_people_by_category(category_id: str):
    """Get people in a specific category."""
    if category_id not in PEOPLE:
        raise ValueError(f"Unknown category: {category_id}")
    return PEOPLE[category_id]["people"]


if __name__ == "__main__":
    all_people = get_all_people()
    print(f"Total people: {len(all_people)}")
    for cat_id, cat in PEOPLE.items():
        print(f"  {cat['name']}: {len(cat['people'])}")
