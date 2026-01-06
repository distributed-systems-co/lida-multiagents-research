"""
Real-world datasets for global simulation.

Contains:
- World leaders and heads of state
- Major corporations (Fortune Global 500, tech giants)
- International organizations (UN, NATO, EU, etc.)
- Central banks and financial institutions
- Think tanks and research institutions
- Major media organizations
- Top universities
- Religious organizations
- Military alliances
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any
from enum import Enum
from datetime import date


# ═══════════════════════════════════════════════════════════════════════════════
# WORLD LEADERS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WorldLeader:
    """A world leader or head of state/government."""
    name: str
    country_code: str
    title: str
    role: str  # head_of_state, head_of_government, both
    party: str = ""
    in_office_since: int = 2020
    ideology: str = ""
    tags: Set[str] = field(default_factory=set)


# Current world leaders (as of late 2024 - simplified set)
WORLD_LEADERS: Dict[str, WorldLeader] = {
    # G7 + Major Powers
    "US_POTUS": WorldLeader("Joe Biden", "US", "President", "both", "Democratic", 2021, "liberal"),
    "CN_PRESIDENT": WorldLeader("Xi Jinping", "CN", "President", "both", "CPC", 2013, "nationalist"),
    "RU_PRESIDENT": WorldLeader("Vladimir Putin", "RU", "President", "both", "United Russia", 2000, "nationalist"),
    "IN_PM": WorldLeader("Narendra Modi", "IN", "Prime Minister", "head_of_government", "BJP", 2014, "nationalist"),
    "JP_PM": WorldLeader("Fumio Kishida", "JP", "Prime Minister", "head_of_government", "LDP", 2021, "conservative"),
    "DE_CHANCELLOR": WorldLeader("Olaf Scholz", "DE", "Chancellor", "head_of_government", "SPD", 2021, "progressive"),
    "GB_PM": WorldLeader("Rishi Sunak", "GB", "Prime Minister", "head_of_government", "Conservative", 2022, "conservative"),
    "FR_PRESIDENT": WorldLeader("Emmanuel Macron", "FR", "President", "head_of_state", "LREM", 2017, "centrist"),

    # Other G20
    "BR_PRESIDENT": WorldLeader("Lula da Silva", "BR", "President", "both", "PT", 2023, "progressive"),
    "MX_PRESIDENT": WorldLeader("Andrés Manuel López Obrador", "MX", "President", "both", "MORENA", 2018, "populist"),
    "AU_PM": WorldLeader("Anthony Albanese", "AU", "Prime Minister", "head_of_government", "Labor", 2022, "progressive"),
    "KR_PRESIDENT": WorldLeader("Yoon Suk-yeol", "KR", "President", "both", "PPP", 2022, "conservative"),
    "ID_PRESIDENT": WorldLeader("Joko Widodo", "ID", "President", "both", "PDI-P", 2014, "centrist"),
    "SA_CROWN_PRINCE": WorldLeader("Mohammed bin Salman", "SA", "Crown Prince", "head_of_government", "", 2017, "reformist"),
    "TR_PRESIDENT": WorldLeader("Recep Tayyip Erdoğan", "TR", "President", "both", "AKP", 2014, "islamist"),
    "ZA_PRESIDENT": WorldLeader("Cyril Ramaphosa", "ZA", "President", "both", "ANC", 2018, "progressive"),

    # Europe
    "IT_PM": WorldLeader("Giorgia Meloni", "IT", "Prime Minister", "head_of_government", "FdI", 2022, "nationalist"),
    "ES_PM": WorldLeader("Pedro Sánchez", "ES", "Prime Minister", "head_of_government", "PSOE", 2018, "progressive"),
    "PL_PM": WorldLeader("Donald Tusk", "PL", "Prime Minister", "head_of_government", "PO", 2023, "liberal"),
    "NL_PM": WorldLeader("Mark Rutte", "NL", "Prime Minister", "head_of_government", "VVD", 2010, "liberal"),

    # Middle East
    "IL_PM": WorldLeader("Benjamin Netanyahu", "IL", "Prime Minister", "head_of_government", "Likud", 2022, "conservative"),
    "IR_PRESIDENT": WorldLeader("Ebrahim Raisi", "IR", "President", "head_of_government", "", 2021, "conservative"),
    "EG_PRESIDENT": WorldLeader("Abdel Fattah el-Sisi", "EG", "President", "both", "", 2014, "authoritarian"),
    "AE_PRESIDENT": WorldLeader("Mohamed bin Zayed Al Nahyan", "AE", "President", "both", "", 2022, "reformist"),

    # Asia
    "PK_PM": WorldLeader("Shehbaz Sharif", "PK", "Prime Minister", "head_of_government", "PML-N", 2022, "centrist"),
    "TH_PM": WorldLeader("Srettha Thavisin", "TH", "Prime Minister", "head_of_government", "Pheu Thai", 2023, "populist"),
    "VN_PRESIDENT": WorldLeader("Vo Van Thuong", "VN", "President", "head_of_state", "CPV", 2023, "communist"),
    "PH_PRESIDENT": WorldLeader("Bongbong Marcos", "PH", "President", "both", "PFP", 2022, "conservative"),
    "SG_PM": WorldLeader("Lee Hsien Loong", "SG", "Prime Minister", "head_of_government", "PAP", 2004, "technocratic"),

    # Africa
    "NG_PRESIDENT": WorldLeader("Bola Tinubu", "NG", "President", "both", "APC", 2023, "conservative"),
    "KE_PRESIDENT": WorldLeader("William Ruto", "KE", "President", "both", "UDA", 2022, "conservative"),
    "ET_PM": WorldLeader("Abiy Ahmed", "ET", "Prime Minister", "head_of_government", "PP", 2018, "reformist"),
}


# ═══════════════════════════════════════════════════════════════════════════════
# MAJOR CORPORATIONS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Corporation:
    """A major corporation."""
    name: str
    ticker: str
    headquarters_country: str
    industry: str
    market_cap_billions: float = 0.0
    employees: int = 0
    revenue_billions: float = 0.0
    ceo: str = ""
    founded: int = 1900
    tags: Set[str] = field(default_factory=set)


# Top global corporations by market cap and influence
MAJOR_CORPORATIONS: Dict[str, Corporation] = {
    # Tech Giants
    "AAPL": Corporation("Apple", "AAPL", "US", "technology", 3000, 164000, 383, "Tim Cook", 1976, {"big_tech", "hardware", "consumer"}),
    "MSFT": Corporation("Microsoft", "MSFT", "US", "technology", 2800, 221000, 211, "Satya Nadella", 1975, {"big_tech", "software", "cloud"}),
    "GOOGL": Corporation("Alphabet", "GOOGL", "US", "technology", 1700, 190000, 307, "Sundar Pichai", 1998, {"big_tech", "advertising", "ai"}),
    "AMZN": Corporation("Amazon", "AMZN", "US", "technology", 1500, 1541000, 574, "Andy Jassy", 1994, {"big_tech", "ecommerce", "cloud"}),
    "NVDA": Corporation("NVIDIA", "NVDA", "US", "technology", 1200, 26000, 61, "Jensen Huang", 1993, {"ai", "semiconductors", "gpu"}),
    "META": Corporation("Meta Platforms", "META", "US", "technology", 800, 86000, 134, "Mark Zuckerberg", 2004, {"big_tech", "social_media", "vr"}),
    "TSLA": Corporation("Tesla", "TSLA", "US", "automotive", 700, 140000, 97, "Elon Musk", 2003, {"ev", "energy", "ai"}),

    # Chinese Tech
    "TCEHY": Corporation("Tencent", "TCEHY", "CN", "technology", 400, 108000, 86, "Ma Huateng", 1998, {"social_media", "gaming", "fintech"}),
    "BABA": Corporation("Alibaba", "BABA", "CN", "technology", 200, 235000, 126, "Eddie Wu", 1999, {"ecommerce", "cloud", "fintech"}),
    "9988": Corporation("ByteDance", "PRIVATE", "CN", "technology", 220, 150000, 80, "Liang Rubo", 2012, {"social_media", "ai", "tiktok"}),

    # Finance
    "JPM": Corporation("JPMorgan Chase", "JPM", "US", "finance", 450, 293000, 128, "Jamie Dimon", 2000, {"banking", "investment"}),
    "V": Corporation("Visa", "V", "US", "finance", 500, 26500, 32, "Ryan McInerney", 1958, {"payments", "fintech"}),
    "MA": Corporation("Mastercard", "MA", "US", "finance", 400, 29900, 25, "Michael Miebach", 1966, {"payments", "fintech"}),
    "GS": Corporation("Goldman Sachs", "GS", "US", "finance", 120, 49000, 47, "David Solomon", 1869, {"banking", "investment"}),
    "BRK": Corporation("Berkshire Hathaway", "BRK.A", "US", "finance", 780, 396000, 302, "Warren Buffett", 1839, {"conglomerate", "insurance"}),
    "HSBC": Corporation("HSBC", "HSBA", "GB", "finance", 150, 219000, 52, "Noel Quinn", 1865, {"banking", "global"}),

    # Healthcare/Pharma
    "JNJ": Corporation("Johnson & Johnson", "JNJ", "US", "healthcare", 400, 152000, 98, "Joaquin Duato", 1886, {"pharma", "consumer"}),
    "UNH": Corporation("UnitedHealth", "UNH", "US", "healthcare", 480, 400000, 324, "Andrew Witty", 1977, {"insurance", "healthcare"}),
    "PFE": Corporation("Pfizer", "PFE", "US", "healthcare", 160, 88000, 100, "Albert Bourla", 1849, {"pharma", "biotech"}),
    "NVS": Corporation("Novartis", "NVS", "CH", "healthcare", 200, 105000, 51, "Vas Narasimhan", 1996, {"pharma", "biotech"}),
    "ROG": Corporation("Roche", "ROG", "CH", "healthcare", 220, 103000, 67, "Thomas Schinecker", 1896, {"pharma", "diagnostics"}),

    # Energy
    "XOM": Corporation("ExxonMobil", "XOM", "US", "energy", 450, 62000, 413, "Darren Woods", 1999, {"oil", "gas"}),
    "CVX": Corporation("Chevron", "CVX", "US", "energy", 300, 43000, 246, "Mike Wirth", 1879, {"oil", "gas"}),
    "SHEL": Corporation("Shell", "SHEL", "GB", "energy", 200, 86000, 386, "Wael Sawan", 1907, {"oil", "gas", "renewables"}),
    "TTE": Corporation("TotalEnergies", "TTE", "FR", "energy", 150, 101000, 263, "Patrick Pouyanné", 1924, {"oil", "gas", "renewables"}),
    "2222": Corporation("Saudi Aramco", "2222.SR", "SA", "energy", 2000, 70000, 604, "Amin H. Nasser", 1933, {"oil", "state_owned"}),

    # Consumer
    "WMT": Corporation("Walmart", "WMT", "US", "retail", 430, 2100000, 648, "Doug McMillon", 1962, {"retail", "ecommerce"}),
    "PG": Corporation("Procter & Gamble", "PG", "US", "consumer", 350, 107000, 82, "Jon Moeller", 1837, {"consumer", "fmcg"}),
    "KO": Corporation("Coca-Cola", "KO", "US", "consumer", 260, 82500, 45, "James Quincey", 1892, {"beverages", "consumer"}),
    "NKE": Corporation("Nike", "NKE", "US", "consumer", 160, 83700, 51, "John Donahoe", 1964, {"apparel", "sports"}),
    "MCD": Corporation("McDonald's", "MCD", "US", "consumer", 200, 200000, 23, "Chris Kempczinski", 1940, {"fast_food", "franchise"}),
    "LVMUY": Corporation("LVMH", "MC", "FR", "consumer", 400, 196000, 86, "Bernard Arnault", 1987, {"luxury", "fashion"}),

    # Industrial
    "BA": Corporation("Boeing", "BA", "US", "aerospace", 130, 171000, 78, "Dave Calhoun", 1916, {"aerospace", "defense"}),
    "LMT": Corporation("Lockheed Martin", "LMT", "US", "defense", 120, 116000, 67, "Jim Taiclet", 1926, {"defense", "aerospace"}),
    "GE": Corporation("General Electric", "GE", "US", "industrial", 120, 172000, 64, "Larry Culp", 1892, {"industrial", "aviation"}),
    "CAT": Corporation("Caterpillar", "CAT", "US", "industrial", 150, 113000, 67, "Jim Umpleby", 1925, {"machinery", "construction"}),
    "TM": Corporation("Toyota", "TM", "JP", "automotive", 230, 375000, 275, "Koji Sato", 1937, {"automotive", "ev"}),
    "VWAGY": Corporation("Volkswagen", "VOW3", "DE", "automotive", 70, 675000, 295, "Oliver Blume", 1937, {"automotive", "ev"}),

    # Telecom
    "T": Corporation("AT&T", "T", "US", "telecom", 110, 160000, 121, "John Stankey", 1885, {"telecom", "media"}),
    "VZ": Corporation("Verizon", "VZ", "US", "telecom", 160, 117000, 137, "Hans Vestberg", 2000, {"telecom", "5g"}),
    "005930": Corporation("Samsung Electronics", "005930.KS", "KR", "technology", 300, 270000, 234, "Lee Jae-yong", 1969, {"semiconductors", "consumer"}),
}


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNATIONAL ORGANIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class InternationalOrg:
    """An international organization."""
    name: str
    abbreviation: str
    headquarters: str
    founded: int
    member_count: int
    org_type: str  # global, regional, economic, military, humanitarian
    head: str = ""
    head_title: str = ""
    budget_billions: float = 0.0
    tags: Set[str] = field(default_factory=set)


INTERNATIONAL_ORGS: Dict[str, InternationalOrg] = {
    # Global Governance
    "UN": InternationalOrg("United Nations", "UN", "US", 1945, 193, "global",
                           "António Guterres", "Secretary-General", 3.2, {"governance", "humanitarian", "peacekeeping"}),
    "WTO": InternationalOrg("World Trade Organization", "WTO", "CH", 1995, 164, "economic",
                            "Ngozi Okonjo-Iweala", "Director-General", 0.2, {"trade", "economic"}),
    "WHO": InternationalOrg("World Health Organization", "WHO", "CH", 1948, 194, "global",
                            "Tedros Adhanom", "Director-General", 6.7, {"health", "humanitarian"}),
    "IAEA": InternationalOrg("Int'l Atomic Energy Agency", "IAEA", "AT", 1957, 178, "global",
                             "Rafael Grossi", "Director General", 0.6, {"nuclear", "energy", "security"}),
    "INTERPOL": InternationalOrg("Int'l Criminal Police Org", "INTERPOL", "FR", 1923, 195, "global",
                                  "Jürgen Stock", "Secretary General", 0.15, {"security", "law_enforcement"}),

    # Economic/Financial
    "IMF": InternationalOrg("International Monetary Fund", "IMF", "US", 1944, 190, "economic",
                            "Kristalina Georgieva", "Managing Director", 1.0, {"finance", "development"}),
    "WB": InternationalOrg("World Bank", "WB", "US", 1944, 189, "economic",
                           "Ajay Banga", "President", 30.0, {"development", "finance"}),
    "BIS": InternationalOrg("Bank for Int'l Settlements", "BIS", "CH", 1930, 63, "economic",
                            "Agustín Carstens", "General Manager", 0.3, {"central_banks", "finance"}),
    "OECD": InternationalOrg("Org for Economic Cooperation", "OECD", "FR", 1961, 38, "economic",
                             "Mathias Cormann", "Secretary-General", 0.4, {"economic", "policy"}),

    # Regional - Europe
    "EU": InternationalOrg("European Union", "EU", "BE", 1993, 27, "regional",
                           "Ursula von der Leyen", "Commission President", 186.0, {"political", "economic", "europe"}),
    "ECB": InternationalOrg("European Central Bank", "ECB", "DE", 1998, 20, "economic",
                            "Christine Lagarde", "President", 0.5, {"central_bank", "euro"}),
    "OSCE": InternationalOrg("Org for Security & Cooperation", "OSCE", "AT", 1975, 57, "regional",
                             "Helga Schmid", "Secretary General", 0.14, {"security", "europe"}),

    # Regional - Americas
    "OAS": InternationalOrg("Organization of American States", "OAS", "US", 1948, 35, "regional",
                            "Luis Almagro", "Secretary General", 0.14, {"americas", "democracy"}),

    # Regional - Asia-Pacific
    "ASEAN": InternationalOrg("Assoc of Southeast Asian Nations", "ASEAN", "ID", 1967, 10, "regional",
                              "Kao Kim Hourn", "Secretary-General", 0.03, {"asia", "economic"}),
    "APEC": InternationalOrg("Asia-Pacific Economic Cooperation", "APEC", "SG", 1989, 21, "economic",
                             "", "Rotating Host", 0.01, {"trade", "asia_pacific"}),
    "SCO": InternationalOrg("Shanghai Cooperation Org", "SCO", "CN", 2001, 9, "regional",
                            "Zhang Ming", "Secretary-General", 0.01, {"security", "eurasia"}),

    # Regional - Africa/Middle East
    "AU": InternationalOrg("African Union", "AU", "ET", 2002, 55, "regional",
                           "Moussa Faki", "Commission Chairperson", 0.7, {"africa", "development"}),
    "GCC": InternationalOrg("Gulf Cooperation Council", "GCC", "SA", 1981, 6, "regional",
                            "Jasem Albudaiwi", "Secretary-General", 0.02, {"gulf", "economic"}),
    "ARAB_LEAGUE": InternationalOrg("Arab League", "LAS", "EG", 1945, 22, "regional",
                                     "Ahmed Aboul Gheit", "Secretary-General", 0.05, {"arab", "political"}),

    # Military Alliances
    "NATO": InternationalOrg("North Atlantic Treaty Org", "NATO", "BE", 1949, 31, "military",
                             "Jens Stoltenberg", "Secretary General", 3.0, {"military", "defense", "western"}),
    "CSTO": InternationalOrg("Collective Security Treaty Org", "CSTO", "RU", 1992, 6, "military",
                             "Imangali Tasmagambetov", "Secretary General", 0.01, {"military", "russia"}),
    "AUKUS": InternationalOrg("Australia-UK-US Alliance", "AUKUS", "US", 2021, 3, "military",
                              "", "", 0.0, {"military", "indo_pacific", "submarines"}),
    "QUAD": InternationalOrg("Quadrilateral Security Dialogue", "QUAD", "US", 2017, 4, "military",
                             "", "", 0.0, {"security", "indo_pacific"}),

    # Economic Blocs
    "BRICS": InternationalOrg("BRICS", "BRICS", "CN", 2009, 10, "economic",
                              "", "Rotating Chair", 0.0, {"emerging", "global_south"}),
    "G7": InternationalOrg("Group of Seven", "G7", "IT", 1975, 7, "economic",
                           "", "Rotating Chair", 0.0, {"advanced_economies", "western"}),
    "G20": InternationalOrg("Group of Twenty", "G20", "BR", 1999, 20, "economic",
                            "", "Rotating Chair", 0.0, {"global_economy"}),

    # Trade Agreements
    "USMCA": InternationalOrg("US-Mexico-Canada Agreement", "USMCA", "US", 2020, 3, "economic",
                              "", "", 0.0, {"trade", "north_america"}),
    "RCEP": InternationalOrg("Regional Comprehensive Economic Partnership", "RCEP", "CN", 2020, 15, "economic",
                             "", "", 0.0, {"trade", "asia_pacific"}),
    "CPTPP": InternationalOrg("Trans-Pacific Partnership", "CPTPP", "JP", 2018, 11, "economic",
                              "", "", 0.0, {"trade", "pacific"}),
    "MERCOSUR": InternationalOrg("Southern Common Market", "MERCOSUR", "BR", 1991, 4, "economic",
                                 "", "", 0.01, {"trade", "south_america"}),
}


# ═══════════════════════════════════════════════════════════════════════════════
# CENTRAL BANKS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CentralBank:
    """A central bank."""
    name: str
    country_code: str
    abbreviation: str
    governor: str
    founded: int
    currency: str
    reserves_billions: float = 0.0
    tags: Set[str] = field(default_factory=set)


CENTRAL_BANKS: Dict[str, CentralBank] = {
    "FED": CentralBank("Federal Reserve", "US", "Fed", "Jerome Powell", 1913, "USD", 8000, {"reserve_currency"}),
    "ECB": CentralBank("European Central Bank", "EU", "ECB", "Christine Lagarde", 1998, "EUR", 5000, {"reserve_currency"}),
    "PBOC": CentralBank("People's Bank of China", "CN", "PBOC", "Pan Gongsheng", 1948, "CNY", 3200, {"reserve_currency"}),
    "BOJ": CentralBank("Bank of Japan", "JP", "BoJ", "Kazuo Ueda", 1882, "JPY", 1200, {"reserve_currency"}),
    "BOE": CentralBank("Bank of England", "GB", "BoE", "Andrew Bailey", 1694, "GBP", 600, {"reserve_currency"}),
    "SNB": CentralBank("Swiss National Bank", "CH", "SNB", "Thomas Jordan", 1907, "CHF", 800, {"safe_haven"}),
    "RBI": CentralBank("Reserve Bank of India", "IN", "RBI", "Shaktikanta Das", 1935, "INR", 600, {"emerging"}),
    "BCB": CentralBank("Central Bank of Brazil", "BR", "BCB", "Roberto Campos Neto", 1964, "BRL", 350, {"emerging"}),
    "CBR": CentralBank("Central Bank of Russia", "RU", "CBR", "Elvira Nabiullina", 1990, "RUB", 580, {"sanctioned"}),
    "SAMA": CentralBank("Saudi Arabian Monetary Authority", "SA", "SAMA", "Fahad Al-Mubarak", 1952, "SAR", 430, {"oil"}),
    "BOK": CentralBank("Bank of Korea", "KR", "BoK", "Rhee Chang-yong", 1950, "KRW", 420, {"developed"}),
    "RBA": CentralBank("Reserve Bank of Australia", "AU", "RBA", "Michele Bullock", 1960, "AUD", 50, {"commodity"}),
    "RBNZ": CentralBank("Reserve Bank of New Zealand", "NZ", "RBNZ", "Adrian Orr", 1934, "NZD", 30, {"developed"}),
    "BANXICO": CentralBank("Bank of Mexico", "MX", "Banxico", "Victoria Rodríguez", 1925, "MXN", 200, {"emerging"}),
}


# ═══════════════════════════════════════════════════════════════════════════════
# THINK TANKS & RESEARCH
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ThinkTank:
    """A think tank or research institution."""
    name: str
    abbreviation: str
    headquarters_country: str
    founded: int
    focus: List[str]
    ideology: str = ""  # liberal, conservative, centrist, etc.
    tags: Set[str] = field(default_factory=set)


THINK_TANKS: Dict[str, ThinkTank] = {
    # US - Liberal/Centrist
    "BROOKINGS": ThinkTank("Brookings Institution", "Brookings", "US", 1916,
                           ["economics", "foreign_policy", "governance"], "centrist", {"influential"}),
    "CFR": ThinkTank("Council on Foreign Relations", "CFR", "US", 1921,
                     ["foreign_policy", "international_relations"], "centrist", {"influential", "establishment"}),
    "RAND": ThinkTank("RAND Corporation", "RAND", "US", 1948,
                      ["defense", "technology", "policy"], "centrist", {"defense", "influential"}),
    "CAP": ThinkTank("Center for American Progress", "CAP", "US", 2003,
                     ["domestic_policy", "economics"], "liberal", {"progressive"}),

    # US - Conservative
    "HERITAGE": ThinkTank("Heritage Foundation", "Heritage", "US", 1973,
                          ["conservative_policy", "economics"], "conservative", {"influential"}),
    "AEI": ThinkTank("American Enterprise Institute", "AEI", "US", 1938,
                     ["economics", "foreign_policy"], "conservative", {"influential"}),
    "CATO": ThinkTank("Cato Institute", "Cato", "US", 1977,
                      ["libertarian_policy", "economics"], "libertarian", {"free_market"}),
    "HOOVER": ThinkTank("Hoover Institution", "Hoover", "US", 1919,
                        ["economics", "foreign_policy"], "conservative", {"stanford"}),

    # Europe
    "CHATHAM": ThinkTank("Chatham House", "RIIA", "GB", 1920,
                         ["foreign_policy", "international_affairs"], "centrist", {"influential"}),
    "IISS": ThinkTank("Int'l Institute for Strategic Studies", "IISS", "GB", 1958,
                      ["security", "defense", "geopolitics"], "centrist", {"influential"}),
    "ECFR": ThinkTank("European Council on Foreign Relations", "ECFR", "GB", 2007,
                      ["european_policy", "foreign_affairs"], "liberal", {"european"}),
    "BRUEGEL": ThinkTank("Bruegel", "Bruegel", "BE", 2005,
                         ["economics", "european_policy"], "centrist", {"european", "economics"}),
    "SWP": ThinkTank("German Institute for Int'l and Security Affairs", "SWP", "DE", 1962,
                     ["security", "foreign_policy"], "centrist", {"german"}),

    # Asia
    "CSIS_ID": ThinkTank("Centre for Strategic and Int'l Studies", "CSIS-ID", "ID", 1971,
                         ["southeast_asia", "foreign_policy"], "centrist", {"asean"}),
    "JETRO": ThinkTank("Japan External Trade Organization", "JETRO", "JP", 1958,
                       ["trade", "economics"], "centrist", {"trade"}),
    "CICIR": ThinkTank("China Institutes of Contemporary Int'l Relations", "CICIR", "CN", 1965,
                       ["international_relations", "security"], "nationalist", {"chinese"}),

    # Global
    "WEF": ThinkTank("World Economic Forum", "WEF", "CH", 1971,
                     ["economics", "global_governance", "sustainability"], "globalist", {"davos", "influential"}),
    "CLUB_ROME": ThinkTank("Club of Rome", "CoR", "CH", 1968,
                           ["sustainability", "limits_to_growth"], "progressive", {"environment"}),
}


# ═══════════════════════════════════════════════════════════════════════════════
# MEDIA ORGANIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MediaOrg:
    """A major media organization."""
    name: str
    headquarters_country: str
    media_type: str  # news_agency, broadcaster, newspaper, digital
    parent_company: str = ""
    reach: str = "national"  # national, regional, global
    ideology: str = ""
    language: str = "en"
    tags: Set[str] = field(default_factory=set)


MEDIA_ORGANIZATIONS: Dict[str, MediaOrg] = {
    # News Agencies
    "AP": MediaOrg("Associated Press", "US", "news_agency", "", "global", "centrist", "en", {"wire_service"}),
    "REUTERS": MediaOrg("Reuters", "GB", "news_agency", "Thomson Reuters", "global", "centrist", "en", {"wire_service"}),
    "AFP": MediaOrg("Agence France-Presse", "FR", "news_agency", "", "global", "centrist", "fr", {"wire_service"}),
    "XINHUA": MediaOrg("Xinhua News Agency", "CN", "news_agency", "Chinese Gov", "global", "state", "zh", {"state_media"}),
    "TASS": MediaOrg("TASS", "RU", "news_agency", "Russian Gov", "global", "state", "ru", {"state_media"}),

    # US Broadcasters
    "CNN": MediaOrg("CNN", "US", "broadcaster", "Warner Bros Discovery", "global", "liberal", "en", {"cable_news"}),
    "FOX": MediaOrg("Fox News", "US", "broadcaster", "Fox Corporation", "national", "conservative", "en", {"cable_news"}),
    "MSNBC": MediaOrg("MSNBC", "US", "broadcaster", "NBCUniversal", "national", "liberal", "en", {"cable_news"}),
    "PBS": MediaOrg("PBS", "US", "broadcaster", "", "national", "centrist", "en", {"public"}),
    "NPR": MediaOrg("NPR", "US", "broadcaster", "", "national", "liberal", "en", {"public", "radio"}),

    # International Broadcasters
    "BBC": MediaOrg("BBC", "GB", "broadcaster", "", "global", "centrist", "en", {"public", "influential"}),
    "DW": MediaOrg("Deutsche Welle", "DE", "broadcaster", "German Gov", "global", "centrist", "de", {"public"}),
    "AL_JAZEERA": MediaOrg("Al Jazeera", "QA", "broadcaster", "Qatar Gov", "global", "centrist", "ar", {"middle_east"}),
    "CGTN": MediaOrg("CGTN", "CN", "broadcaster", "Chinese Gov", "global", "state", "zh", {"state_media"}),
    "RT": MediaOrg("RT", "RU", "broadcaster", "Russian Gov", "global", "state", "ru", {"state_media"}),
    "NHK": MediaOrg("NHK", "JP", "broadcaster", "", "national", "centrist", "ja", {"public"}),
    "FRANCE24": MediaOrg("France 24", "FR", "broadcaster", "France Médias Monde", "global", "centrist", "fr", {"public"}),

    # Newspapers
    "NYT": MediaOrg("New York Times", "US", "newspaper", "NYT Company", "global", "liberal", "en", {"prestige"}),
    "WAPO": MediaOrg("Washington Post", "US", "newspaper", "Nash Holdings", "national", "liberal", "en", {"prestige"}),
    "WSJ": MediaOrg("Wall Street Journal", "US", "newspaper", "News Corp", "global", "conservative", "en", {"business"}),
    "FT": MediaOrg("Financial Times", "GB", "newspaper", "Nikkei", "global", "centrist", "en", {"business"}),
    "ECONOMIST": MediaOrg("The Economist", "GB", "newspaper", "", "global", "liberal", "en", {"influential"}),
    "GUARDIAN": MediaOrg("The Guardian", "GB", "newspaper", "Scott Trust", "global", "liberal", "en", {"progressive"}),
    "SPIEGEL": MediaOrg("Der Spiegel", "DE", "newspaper", "", "national", "liberal", "de", {"investigative"}),
    "LE_MONDE": MediaOrg("Le Monde", "FR", "newspaper", "", "national", "centrist", "fr", {"prestige"}),

    # Digital/Tech
    "BLOOMBERG": MediaOrg("Bloomberg", "US", "digital", "Bloomberg LP", "global", "centrist", "en", {"business", "data"}),
    "POLITICO": MediaOrg("Politico", "US", "digital", "Axel Springer", "national", "centrist", "en", {"political"}),
}


# ═══════════════════════════════════════════════════════════════════════════════
# TOP UNIVERSITIES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class University:
    """A major university."""
    name: str
    country_code: str
    city: str
    founded: int
    students: int = 0
    endowment_billions: float = 0.0
    ranking_type: str = ""  # ivy, russell, c9, etc.
    tags: Set[str] = field(default_factory=set)


TOP_UNIVERSITIES: Dict[str, University] = {
    # US - Ivy League
    "HARVARD": University("Harvard University", "US", "Cambridge", 1636, 23000, 50, "ivy", {"elite", "research"}),
    "YALE": University("Yale University", "US", "New Haven", 1701, 14000, 41, "ivy", {"elite", "research"}),
    "PRINCETON": University("Princeton University", "US", "Princeton", 1746, 8500, 35, "ivy", {"elite", "research"}),
    "COLUMBIA": University("Columbia University", "US", "New York", 1754, 33000, 14, "ivy", {"elite", "research"}),
    "UPENN": University("University of Pennsylvania", "US", "Philadelphia", 1740, 28000, 21, "ivy", {"elite", "research"}),
    "BROWN": University("Brown University", "US", "Providence", 1764, 10000, 6, "ivy", {"liberal_arts"}),
    "DARTMOUTH": University("Dartmouth College", "US", "Hanover", 1769, 6700, 8, "ivy", {"liberal_arts"}),
    "CORNELL": University("Cornell University", "US", "Ithaca", 1865, 25000, 10, "ivy", {"research"}),

    # US - Other Elite
    "MIT": University("MIT", "US", "Cambridge", 1861, 11500, 27, "elite", {"tech", "research"}),
    "STANFORD": University("Stanford University", "US", "Stanford", 1885, 17000, 38, "elite", {"tech", "research"}),
    "CALTECH": University("Caltech", "US", "Pasadena", 1891, 2400, 16, "elite", {"tech", "research"}),
    "UCHICAGO": University("University of Chicago", "US", "Chicago", 1890, 18000, 11, "elite", {"research"}),
    "DUKE": University("Duke University", "US", "Durham", 1838, 17000, 12, "elite", {"research"}),
    "BERKELEY": University("UC Berkeley", "US", "Berkeley", 1868, 45000, 6, "public", {"research", "public"}),
    "UCLA": University("UCLA", "US", "Los Angeles", 1919, 46000, 7, "public", {"research", "public"}),

    # UK - Oxbridge
    "OXFORD": University("University of Oxford", "GB", "Oxford", 1096, 26000, 8, "russell", {"elite", "ancient"}),
    "CAMBRIDGE": University("University of Cambridge", "GB", "Cambridge", 1209, 24000, 9, "russell", {"elite", "ancient"}),
    "IMPERIAL": University("Imperial College London", "GB", "London", 1907, 20000, 1, "russell", {"tech", "research"}),
    "LSE": University("London School of Economics", "GB", "London", 1895, 12000, 0.4, "russell", {"economics", "social_science"}),
    "UCL": University("UCL", "GB", "London", 1826, 50000, 0.2, "russell", {"research"}),

    # Europe
    "ETH": University("ETH Zürich", "CH", "Zürich", 1855, 24000, 1, "european", {"tech", "research"}),
    "EPFL": University("EPFL", "CH", "Lausanne", 1853, 12000, 0.5, "european", {"tech", "research"}),
    "SORBONNE": University("Sorbonne University", "FR", "Paris", 1257, 55000, 0.2, "european", {"ancient", "research"}),
    "TUM": University("Technical University of Munich", "DE", "Munich", 1868, 50000, 0.1, "european", {"tech"}),
    "KU_LEUVEN": University("KU Leuven", "BE", "Leuven", 1425, 60000, 0.3, "european", {"ancient", "research"}),

    # Asia
    "TSINGHUA": University("Tsinghua University", "CN", "Beijing", 1911, 53000, 5, "c9", {"elite", "tech"}),
    "PEKING": University("Peking University", "CN", "Beijing", 1898, 42000, 3, "c9", {"elite", "research"}),
    "TOKYO": University("University of Tokyo", "JP", "Tokyo", 1877, 28000, 1, "imperial", {"elite", "research"}),
    "KYOTO": University("Kyoto University", "JP", "Kyoto", 1897, 22000, 0.5, "imperial", {"research"}),
    "NUS": University("National University of Singapore", "SG", "Singapore", 1905, 40000, 5, "asian", {"elite", "research"}),
    "NTU": University("Nanyang Technological University", "SG", "Singapore", 1991, 33000, 2, "asian", {"tech"}),
    "SEOUL": University("Seoul National University", "KR", "Seoul", 1946, 27000, 1, "korean", {"elite", "research"}),
    "IIT_B": University("IIT Bombay", "IN", "Mumbai", 1958, 11000, 0.1, "iit", {"tech", "elite"}),
    "IIT_D": University("IIT Delhi", "IN", "Delhi", 1961, 9000, 0.1, "iit", {"tech", "elite"}),

    # Australia
    "UMELB": University("University of Melbourne", "AU", "Melbourne", 1853, 65000, 4, "group_of_8", {"research"}),
    "USYD": University("University of Sydney", "AU", "Sydney", 1850, 73000, 2, "group_of_8", {"research"}),
    "ANU": University("Australian National University", "AU", "Canberra", 1946, 25000, 2, "group_of_8", {"research"}),
}


# ═══════════════════════════════════════════════════════════════════════════════
# RELIGIOUS ORGANIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ReligiousOrg:
    """A major religious organization."""
    name: str
    religion: str
    headquarters: str
    leader: str
    leader_title: str
    followers_millions: float = 0.0
    founded: int = 0
    tags: Set[str] = field(default_factory=set)


RELIGIOUS_ORGANIZATIONS: Dict[str, ReligiousOrg] = {
    "VATICAN": ReligiousOrg("Holy See", "Catholic", "VA", "Pope Francis", "Pope", 1300, 33, {"global", "influential"}),
    "ECUMENICAL_PATRIARCHATE": ReligiousOrg("Ecumenical Patriarchate", "Orthodox", "TR", "Bartholomew I", "Patriarch", 260, 330, {"orthodox"}),
    "MOSCOW_PATRIARCHATE": ReligiousOrg("Russian Orthodox Church", "Orthodox", "RU", "Kirill I", "Patriarch", 100, 1448, {"orthodox", "russian"}),
    "ANGLICAN_COMMUNION": ReligiousOrg("Anglican Communion", "Anglican", "GB", "Justin Welby", "Archbishop", 85, 1534, {"protestant"}),
    "WORLD_COUNCIL_CHURCHES": ReligiousOrg("World Council of Churches", "Ecumenical", "CH", "Jerry Pillay", "General Secretary", 580, 1948, {"ecumenical"}),
    "BAPTIST_WORLD": ReligiousOrg("Baptist World Alliance", "Baptist", "US", "Tomás Mackey", "President", 47, 1905, {"protestant"}),
    "METHODIST_WORLD": ReligiousOrg("World Methodist Council", "Methodist", "US", "", "President", 80, 1881, {"protestant"}),
    "LUTHERAN_WORLD": ReligiousOrg("Lutheran World Federation", "Lutheran", "CH", "Anne Burghardt", "General Secretary", 77, 1947, {"protestant"}),
    "OIC": ReligiousOrg("Organisation of Islamic Cooperation", "Islam", "SA", "Hissein Brahim Taha", "Secretary-General", 1800, 1969, {"islamic", "intergovernmental"}),
    "AZHAR": ReligiousOrg("Al-Azhar", "Sunni Islam", "EG", "Ahmed el-Tayeb", "Grand Imam", 1000, 970, {"sunni", "influential"}),
    "JEWISH_AGENCY": ReligiousOrg("Jewish Agency", "Judaism", "IL", "Doron Almog", "Chairman", 15, 1929, {"jewish", "zionist"}),
    "DALAI_LAMA": ReligiousOrg("Tibetan Government in Exile", "Buddhism", "IN", "Dalai Lama", "Spiritual Leader", 500, 1959, {"buddhist", "tibetan"}),
}


# ═══════════════════════════════════════════════════════════════════════════════
# INDUSTRIAL & AI COMPANIES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IndustrialCompanyProfile:
    """An industrial or AI company profile."""
    name: str
    sector: str  # ai_foundation, ai_infrastructure, ai_chips, ai_robotics, automation, defense, etc.
    headquarters_country: str
    valuation_billions: float = 0.0
    is_public: bool = False
    ticker: str = ""
    ceo: str = ""
    founded: int = 2000
    arr_millions: float = 0.0
    employees: int = 0
    lead_investors: List[str] = field(default_factory=list)
    is_acquired: bool = False
    acquirer: str = ""
    acquisition_price_billions: float = 0.0
    tags: Set[str] = field(default_factory=set)


INDUSTRIAL_AI_COMPANIES: Dict[str, IndustrialCompanyProfile] = {
    # AI Foundation Models
    "OPENAI": IndustrialCompanyProfile("OpenAI", "ai_foundation", "US", 300.0, False, "", "Sam Altman", 2015, 0, 3000, ["SoftBank", "Microsoft"], tags={"ai", "foundation_model", "chatgpt"}),
    "ANTHROPIC": IndustrialCompanyProfile("Anthropic", "ai_foundation", "US", 183.0, False, "", "Dario Amodei", 2021, 0, 1000, ["Nvidia", "Google", "Amazon"], tags={"ai", "foundation_model", "safety"}),
    "XAI": IndustrialCompanyProfile("xAI", "ai_foundation", "US", 50.0, False, "", "Elon Musk", 2023, 0, 100, [], tags={"ai", "foundation_model", "grok"}),
    "SSI": IndustrialCompanyProfile("Safe Superintelligence", "ai_foundation", "US", 32.0, False, "", "Ilya Sutskever", 2024, 0, 20, [], tags={"ai", "foundation_model", "safety"}),
    "MISTRAL": IndustrialCompanyProfile("Mistral AI", "ai_foundation", "FR", 6.0, False, "", "Arthur Mensch", 2023, 0, 60, ["a16z", "Microsoft"], tags={"ai", "foundation_model", "open_source"}),
    "COHERE": IndustrialCompanyProfile("Cohere", "ai_foundation", "CA", 7.0, False, "", "Aidan Gomez", 2019, 0, 450, ["Oracle", "Salesforce"], tags={"ai", "foundation_model", "enterprise"}),
    "THINKING_MACHINES": IndustrialCompanyProfile("Thinking Machines Lab", "ai_foundation", "US", 12.0, False, "", "Mira Murati", 2024, 0, 30, [], tags={"ai", "foundation_model", "ex_openai"}),
    "REFLECTION": IndustrialCompanyProfile("Reflection AI", "ai_foundation", "US", 8.0, False, "", "", 2024, 0, 50, ["Nvidia", "Sequoia"], tags={"ai", "foundation_model"}),

    # AI Infrastructure
    "COREWEAVE": IndustrialCompanyProfile("CoreWeave", "ai_infrastructure", "US", 15.0, True, "CRWV", "Michael Intrator", 2017, 0, 500, [], tags={"ai", "gpu_cloud", "infrastructure"}),
    "LAMBDA_LABS": IndustrialCompanyProfile("Lambda Labs", "ai_infrastructure", "US", 12.5, False, "", "Stephen Balaban", 2012, 505, 400, [], tags={"ai", "gpu_cloud", "ipo_2026"}),
    "TOGETHER_AI": IndustrialCompanyProfile("Together AI", "ai_infrastructure", "US", 3.3, False, "", "Vipul Prakash", 2022, 300, 150, [], tags={"ai", "inference", "training"}),
    "BASETEN": IndustrialCompanyProfile("Baseten", "ai_infrastructure", "US", 2.15, False, "", "", 2019, 0, 100, [], tags={"ai", "inference", "ml_ops"}),
    "MODAL": IndustrialCompanyProfile("Modal", "ai_infrastructure", "US", 1.1, False, "", "", 2021, 0, 50, [], tags={"ai", "serverless", "ml_ops"}),

    # AI Chips
    "CEREBRAS": IndustrialCompanyProfile("Cerebras", "ai_chips", "US", 15.0, False, "", "Andrew Feldman", 2016, 0, 400, [], tags={"ai", "chips", "wafer_scale", "ipo_2026"}),
    "GROQ": IndustrialCompanyProfile("Groq", "ai_chips", "US", 20.0, False, "", "Jonathan Ross", 2016, 0, 300, ["Nvidia"], True, "Nvidia", 20.0, {"ai", "chips", "inference", "acquired"}),
    "TENSTORRENT": IndustrialCompanyProfile("Tenstorrent", "ai_chips", "CA", 3.2, False, "", "Jim Keller", 2016, 0, 400, [], tags={"ai", "chips", "risc_v"}),
    "D_MATRIX": IndustrialCompanyProfile("d-Matrix", "ai_chips", "US", 2.0, False, "", "", 2019, 0, 150, [], tags={"ai", "chips", "in_memory"}),
    "SAMBANOVA": IndustrialCompanyProfile("SambaNova", "ai_chips", "US", 1.6, False, "", "", 2017, 0, 500, ["Intel"], True, "Intel", 1.6, {"ai", "chips", "acquired"}),

    # AI Coding Tools
    "CURSOR": IndustrialCompanyProfile("Cursor", "ai_coding", "US", 29.3, False, "", "Michael Truell", 2022, 1000, 100, ["a16z", "Thrive"], tags={"ai", "coding", "fastest_saas"}),
    "COGNITION": IndustrialCompanyProfile("Cognition (Devin)", "ai_coding", "US", 10.2, False, "", "Scott Wu", 2023, 73, 50, ["Founders Fund"], tags={"ai", "coding", "agents"}),
    "POOLSIDE": IndustrialCompanyProfile("Poolside", "ai_coding", "US", 12.0, False, "", "", 2023, 50, 80, [], tags={"ai", "coding", "code_generation"}),

    # AI Enterprise
    "HARVEY": IndustrialCompanyProfile("Harvey", "ai_enterprise", "US", 8.0, False, "", "Winston Weinberg", 2022, 150, 200, ["Sequoia"], tags={"ai", "legal", "enterprise"}),
    "GLEAN": IndustrialCompanyProfile("Glean", "ai_enterprise", "US", 7.2, False, "", "Arvind Jain", 2019, 200, 500, ["Sequoia"], tags={"ai", "search", "enterprise"}),
    "WRITER": IndustrialCompanyProfile("Writer", "ai_enterprise", "US", 1.9, False, "", "", 2020, 47, 200, [], tags={"ai", "content", "enterprise"}),
    "IRONCLAD": IndustrialCompanyProfile("Ironclad", "ai_enterprise", "US", 3.2, False, "", "", 2014, 150, 500, [], tags={"ai", "contracts", "legal"}),

    # AI Video/Audio
    "ELEVENLABS": IndustrialCompanyProfile("ElevenLabs", "ai_media", "GB", 6.6, False, "", "", 2022, 200, 100, [], tags={"ai", "voice", "synthesis"}),
    "SYNTHESIA": IndustrialCompanyProfile("Synthesia", "ai_media", "GB", 4.0, False, "", "", 2017, 146, 400, [], tags={"ai", "video", "avatars"}),
    "SUNO": IndustrialCompanyProfile("Suno", "ai_media", "US", 2.45, False, "", "", 2022, 200, 50, [], tags={"ai", "music", "generation"}),
    "RUNWAY": IndustrialCompanyProfile("Runway", "ai_media", "US", 4.0, False, "", "", 2018, 0, 100, [], tags={"ai", "video", "generation", "acquisition_target"}),
    "LUMA_AI": IndustrialCompanyProfile("Luma AI", "ai_media", "US", 3.2, False, "", "", 2021, 0, 50, [], tags={"ai", "3d", "video"}),
    "PIKA_LABS": IndustrialCompanyProfile("Pika Labs", "ai_media", "US", 0.5, False, "", "", 2023, 0, 30, [], tags={"ai", "video", "generation"}),

    # AI Robotics
    "FIGURE_AI": IndustrialCompanyProfile("Figure AI", "ai_robotics", "US", 39.0, False, "", "Brett Adcock", 2022, 0, 400, ["BMW", "Intel"], tags={"ai", "humanoid", "robotics"}),
    "PHYSICAL_INTELLIGENCE": IndustrialCompanyProfile("Physical Intelligence", "ai_robotics", "US", 5.6, False, "", "Karol Hausman", 2024, 0, 50, [], tags={"ai", "robotics", "foundation_model"}),
    "AGILITY_ROBOTICS": IndustrialCompanyProfile("Agility Robotics", "ai_robotics", "US", 2.1, False, "", "Damion Shelton", 2015, 0, 200, ["Amazon"], tags={"ai", "humanoid", "digit"}),
    "1X_TECHNOLOGIES": IndustrialCompanyProfile("1X Technologies", "ai_robotics", "NO", 0.5, False, "", "", 2014, 0, 100, [], tags={"ai", "humanoid", "neo"}),

    # Autonomous Vehicles
    "WAYMO": IndustrialCompanyProfile("Waymo", "autonomous_vehicles", "US", 100.0, False, "", "Tekedra Mawakana", 2016, 0, 2500, ["Alphabet"], tags={"av", "robotaxi", "leader"}),
    "AURORA": IndustrialCompanyProfile("Aurora Innovation", "autonomous_vehicles", "US", 5.0, True, "AUR", "Chris Urmson", 2017, 0, 1600, [], tags={"av", "trucking"}),
    "PONY_AI": IndustrialCompanyProfile("Pony.ai", "autonomous_vehicles", "CN", 8.5, False, "", "", 2016, 0, 1000, [], tags={"av", "robotaxi", "china"}),

    # Warehouse/Logistics Automation
    "SYMBOTIC": IndustrialCompanyProfile("Symbotic", "warehouse_automation", "US", 8.0, True, "SYM", "Rick Cohen", 2007, 0, 2000, [], tags={"automation", "warehouse", "ai"}),
    "LOCUS_ROBOTICS": IndustrialCompanyProfile("Locus Robotics", "warehouse_automation", "US", 2.0, False, "", "Rick Faulk", 2014, 0, 600, ["Tiger Global"], tags={"automation", "amr", "warehouse"}),
    "AMAZON_ROBOTICS": IndustrialCompanyProfile("Amazon Robotics", "warehouse_automation", "US", 0.0, False, "", "", 2012, 0, 5000, ["Amazon"], tags={"automation", "warehouse", "amazon"}),

    # Defense/Aerospace
    "ANDURIL": IndustrialCompanyProfile("Anduril Industries", "defense", "US", 30.5, False, "", "Palmer Luckey", 2017, 0, 2500, ["a16z"], tags={"defense", "ai", "drones"}),
    "SHIELD_AI": IndustrialCompanyProfile("Shield AI", "defense", "US", 5.3, False, "", "Ryan Tseng", 2015, 0, 700, [], tags={"defense", "ai", "autonomous"}),
    "SPACEX": IndustrialCompanyProfile("SpaceX", "defense", "US", 350.0, False, "", "Elon Musk", 2002, 0, 13000, [], tags={"space", "starlink", "defense"}),

    # Energy/Nuclear
    "X_ENERGY": IndustrialCompanyProfile("X-energy", "energy", "US", 2.5, False, "", "J. Clay Sell", 2009, 0, 500, [], tags={"nuclear", "smr", "energy"}),
    "NUSCALE": IndustrialCompanyProfile("NuScale Power", "energy", "US", 0.5, True, "SMR", "", 2007, 0, 500, [], tags={"nuclear", "smr", "energy"}),
    "KAIROS_POWER": IndustrialCompanyProfile("Kairos Power", "energy", "US", 1.0, False, "", "", 2016, 0, 300, ["Google"], tags={"nuclear", "molten_salt", "energy"}),
    "FORM_ENERGY": IndustrialCompanyProfile("Form Energy", "energy", "US", 1.2, False, "", "", 2017, 0, 600, [], tags={"battery", "grid", "storage"}),

    # Healthcare/Biotech AI
    "XAIRA": IndustrialCompanyProfile("Xaira Therapeutics", "healthcare_ai", "US", 1.0, False, "", "Marc Tessier-Lavigne", 2024, 0, 50, [], tags={"ai", "drug_discovery", "biotech"}),
    "RECURSION": IndustrialCompanyProfile("Recursion", "healthcare_ai", "US", 3.0, True, "RXRX", "", 2013, 0, 700, [], tags={"ai", "drug_discovery", "biotech"}),
    "ISOMORPHIC_LABS": IndustrialCompanyProfile("Isomorphic Labs", "healthcare_ai", "GB", 0.0, False, "", "", 2021, 0, 100, ["Alphabet"], tags={"ai", "drug_discovery", "deepmind"}),

    # Manufacturing/Industrial Automation
    "SIEMENS_AI": IndustrialCompanyProfile("Siemens Digital Industries", "industrial_automation", "DE", 0.0, False, "", "", 1847, 0, 50000, [], tags={"automation", "digital_twin", "manufacturing"}),
    "ROCKWELL": IndustrialCompanyProfile("Rockwell Automation", "industrial_automation", "US", 30.0, True, "ROK", "", 1903, 0, 29000, [], tags={"automation", "plc", "manufacturing"}),
    "FANUC": IndustrialCompanyProfile("Fanuc", "industrial_automation", "JP", 35.0, True, "6954.T", "", 1972, 0, 8500, [], tags={"robotics", "cnc", "manufacturing"}),

    # Semiconductors (Additional)
    "ASML": IndustrialCompanyProfile("ASML", "semiconductors", "NL", 350.0, True, "ASML", "Peter Wennink", 1984, 0, 42000, [], tags={"chips", "euv", "monopoly"}),
    "TSMC": IndustrialCompanyProfile("TSMC", "semiconductors", "TW", 900.0, True, "TSM", "C.C. Wei", 1987, 0, 73000, [], tags={"chips", "foundry", "dominant"}),
}


# ═══════════════════════════════════════════════════════════════════════════════
# AGGREGATED DATA ACCESS
# ═══════════════════════════════════════════════════════════════════════════════

def get_all_entities() -> Dict[str, Dict]:
    """Get all entities as a combined dictionary."""
    return {
        "world_leaders": {k: v.__dict__ for k, v in WORLD_LEADERS.items()},
        "corporations": {k: v.__dict__ for k, v in MAJOR_CORPORATIONS.items()},
        "international_orgs": {k: v.__dict__ for k, v in INTERNATIONAL_ORGS.items()},
        "central_banks": {k: v.__dict__ for k, v in CENTRAL_BANKS.items()},
        "think_tanks": {k: v.__dict__ for k, v in THINK_TANKS.items()},
        "media_orgs": {k: v.__dict__ for k, v in MEDIA_ORGANIZATIONS.items()},
        "universities": {k: v.__dict__ for k, v in TOP_UNIVERSITIES.items()},
        "religious_orgs": {k: v.__dict__ for k, v in RELIGIOUS_ORGANIZATIONS.items()},
        "industrial_ai_companies": {k: v.__dict__ for k, v in INDUSTRIAL_AI_COMPANIES.items()},
    }


def get_entities_by_country(country_code: str) -> Dict[str, List]:
    """Get all entities for a specific country."""
    result = {
        "leaders": [],
        "corporations": [],
        "central_banks": [],
        "think_tanks": [],
        "media_orgs": [],
        "universities": [],
    }

    for k, v in WORLD_LEADERS.items():
        if v.country_code == country_code:
            result["leaders"].append(v)

    for k, v in MAJOR_CORPORATIONS.items():
        if v.headquarters_country == country_code:
            result["corporations"].append(v)

    for k, v in CENTRAL_BANKS.items():
        if v.country_code == country_code:
            result["central_banks"].append(v)

    for k, v in THINK_TANKS.items():
        if v.headquarters_country == country_code:
            result["think_tanks"].append(v)

    for k, v in MEDIA_ORGANIZATIONS.items():
        if v.headquarters_country == country_code:
            result["media_orgs"].append(v)

    for k, v in TOP_UNIVERSITIES.items():
        if v.country_code == country_code:
            result["universities"].append(v)

    return result


def get_entities_by_tag(tag: str) -> Dict[str, List]:
    """Get all entities with a specific tag."""
    result = {
        "leaders": [],
        "corporations": [],
        "international_orgs": [],
        "think_tanks": [],
        "media_orgs": [],
        "universities": [],
        "religious_orgs": [],
    }

    for k, v in WORLD_LEADERS.items():
        if tag in v.tags:
            result["leaders"].append(v)

    for k, v in MAJOR_CORPORATIONS.items():
        if tag in v.tags:
            result["corporations"].append(v)

    for k, v in INTERNATIONAL_ORGS.items():
        if tag in v.tags:
            result["international_orgs"].append(v)

    for k, v in THINK_TANKS.items():
        if tag in v.tags:
            result["think_tanks"].append(v)

    for k, v in MEDIA_ORGANIZATIONS.items():
        if tag in v.tags:
            result["media_orgs"].append(v)

    for k, v in TOP_UNIVERSITIES.items():
        if tag in v.tags:
            result["universities"].append(v)

    for k, v in RELIGIOUS_ORGANIZATIONS.items():
        if tag in v.tags:
            result["religious_orgs"].append(v)

    return result


def get_dataset_stats() -> Dict[str, int]:
    """Get counts of all dataset entries."""
    return {
        "world_leaders": len(WORLD_LEADERS),
        "corporations": len(MAJOR_CORPORATIONS),
        "international_orgs": len(INTERNATIONAL_ORGS),
        "central_banks": len(CENTRAL_BANKS),
        "think_tanks": len(THINK_TANKS),
        "media_orgs": len(MEDIA_ORGANIZATIONS),
        "universities": len(TOP_UNIVERSITIES),
        "religious_orgs": len(RELIGIOUS_ORGANIZATIONS),
        "industrial_ai_companies": len(INDUSTRIAL_AI_COMPANIES),
        "total": (len(WORLD_LEADERS) + len(MAJOR_CORPORATIONS) + len(INTERNATIONAL_ORGS) +
                  len(CENTRAL_BANKS) + len(THINK_TANKS) + len(MEDIA_ORGANIZATIONS) +
                  len(TOP_UNIVERSITIES) + len(RELIGIOUS_ORGANIZATIONS) + len(INDUSTRIAL_AI_COMPANIES)),
    }


def get_industrial_companies_by_sector(sector: str) -> List[IndustrialCompanyProfile]:
    """Get industrial/AI companies by sector."""
    return [c for c in INDUSTRIAL_AI_COMPANIES.values() if c.sector == sector]


def get_acquisition_targets() -> List[IndustrialCompanyProfile]:
    """Get companies likely to be acquisition targets."""
    return [
        c for c in INDUSTRIAL_AI_COMPANIES.values()
        if not c.is_acquired and not c.is_public and "acquisition_target" in c.tags
    ]


def get_acquired_companies() -> List[IndustrialCompanyProfile]:
    """Get companies that have been acquired."""
    return [c for c in INDUSTRIAL_AI_COMPANIES.values() if c.is_acquired]
