"""
Global social structures for simulating world-scale hierarchies.

Models thousands of interconnected hierarchies including:
- Governments and political structures
- Corporations and economic entities
- NGOs and civil society
- Religious organizations
- Military and security
- Academic and research institutions
- Media and information networks
- Cultural and social groups

With country-based groupings and tag-based meta-facets.
"""

from __future__ import annotations
import uuid
import random
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Set, Optional, Tuple, Any, Callable
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════════════
# GEOGRAPHIC HIERARCHIES
# ═══════════════════════════════════════════════════════════════════════════════

class Region(str, Enum):
    """Major world regions."""
    NORTH_AMERICA = "north_america"
    SOUTH_AMERICA = "south_america"
    EUROPE = "europe"
    AFRICA = "africa"
    MIDDLE_EAST = "middle_east"
    CENTRAL_ASIA = "central_asia"
    SOUTH_ASIA = "south_asia"
    EAST_ASIA = "east_asia"
    SOUTHEAST_ASIA = "southeast_asia"
    OCEANIA = "oceania"
    CARIBBEAN = "caribbean"
    ARCTIC = "arctic"
    ANTARCTIC = "antarctic"


class SubRegion(str, Enum):
    """Sub-regions for finer geographic grouping."""
    # North America
    US_NORTHEAST = "us_northeast"
    US_SOUTHEAST = "us_southeast"
    US_MIDWEST = "us_midwest"
    US_SOUTHWEST = "us_southwest"
    US_WEST = "us_west"
    CANADA_EAST = "canada_east"
    CANADA_WEST = "canada_west"
    MEXICO_NORTH = "mexico_north"
    MEXICO_SOUTH = "mexico_south"

    # Europe
    WESTERN_EUROPE = "western_europe"
    EASTERN_EUROPE = "eastern_europe"
    NORTHERN_EUROPE = "northern_europe"
    SOUTHERN_EUROPE = "southern_europe"
    CENTRAL_EUROPE = "central_europe"
    BALKANS = "balkans"
    BALTIC = "baltic"

    # Asia
    CHINA_COASTAL = "china_coastal"
    CHINA_INTERIOR = "china_interior"
    JAPAN_KOREA = "japan_korea"
    INDIA_NORTH = "india_north"
    INDIA_SOUTH = "india_south"
    GULF_STATES = "gulf_states"
    LEVANT = "levant"

    # Africa
    NORTH_AFRICA = "north_africa"
    WEST_AFRICA = "west_africa"
    EAST_AFRICA = "east_africa"
    CENTRAL_AFRICA = "central_africa"
    SOUTHERN_AFRICA = "southern_africa"

    # Others
    PACIFIC_ISLANDS = "pacific_islands"
    ANDEAN = "andean"
    SOUTHERN_CONE = "southern_cone"


@dataclass
class Country:
    """Country with metadata for grouping."""
    code: str  # ISO 3166-1 alpha-2
    name: str
    region: Region
    sub_region: Optional[SubRegion] = None
    population: int = 0
    gdp_billions: float = 0.0
    languages: List[str] = field(default_factory=list)
    alliances: Set[str] = field(default_factory=set)  # NATO, EU, ASEAN, etc.
    tags: Set[str] = field(default_factory=set)
    coordinates: Optional[Tuple[float, float]] = None  # (latitude, longitude) of capital

    def get_coordinates(self) -> Optional[Tuple[float, float]]:
        """Get coordinates, falling back to lookup if not set."""
        if self.coordinates:
            return self.coordinates
        # Lazy import to avoid circular dependency
        try:
            from .geospatial import COUNTRY_COORDINATES
            return COUNTRY_COORDINATES.get(self.code)
        except ImportError:
            return None

    def distance_to(self, other: "Country") -> Optional[float]:
        """Calculate distance to another country in kilometers."""
        coords1 = self.get_coordinates()
        coords2 = other.get_coordinates()
        if not coords1 or not coords2:
            return None

        import math
        R = 6371  # Earth's radius in km
        lat1, lon1 = math.radians(coords1[0]), math.radians(coords1[1])
        lat2, lon2 = math.radians(coords2[0]), math.radians(coords2[1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))

        return R * c


# Comprehensive world countries database (195 UN member states + territories)
COUNTRIES: Dict[str, Country] = {
    # ═══════════════════════════════════════════════════════════════════════════════
    # NORTH AMERICA (3 countries)
    # ═══════════════════════════════════════════════════════════════════════════════
    "US": Country("US", "United States", Region.NORTH_AMERICA, SubRegion.US_NORTHEAST,
                  331000000, 25000, ["en"], {"NATO", "G7", "G20", "OECD", "Five_Eyes", "USMCA"}),
    "CA": Country("CA", "Canada", Region.NORTH_AMERICA, SubRegion.CANADA_EAST,
                  38000000, 2000, ["en", "fr"], {"NATO", "G7", "G20", "OECD", "Five_Eyes", "USMCA"}),
    "MX": Country("MX", "Mexico", Region.NORTH_AMERICA, SubRegion.MEXICO_NORTH,
                  130000000, 1300, ["es"], {"G20", "OECD", "USMCA"}),

    # ═══════════════════════════════════════════════════════════════════════════════
    # CENTRAL AMERICA & CARIBBEAN (20 countries)
    # ═══════════════════════════════════════════════════════════════════════════════
    "GT": Country("GT", "Guatemala", Region.CARIBBEAN, None, 18000000, 86, ["es"], {"CAFTA"}),
    "BZ": Country("BZ", "Belize", Region.CARIBBEAN, None, 420000, 2, ["en", "es"], {"CARICOM"}),
    "HN": Country("HN", "Honduras", Region.CARIBBEAN, None, 10000000, 28, ["es"], {"CAFTA"}),
    "SV": Country("SV", "El Salvador", Region.CARIBBEAN, None, 6500000, 32, ["es"], {"CAFTA"}),
    "NI": Country("NI", "Nicaragua", Region.CARIBBEAN, None, 6700000, 14, ["es"], set()),
    "CR": Country("CR", "Costa Rica", Region.CARIBBEAN, None, 5100000, 62, ["es"], {"CAFTA", "OECD"}),
    "PA": Country("PA", "Panama", Region.CARIBBEAN, None, 4400000, 77, ["es"], set()),
    "CU": Country("CU", "Cuba", Region.CARIBBEAN, None, 11300000, 100, ["es"], set()),
    "JM": Country("JM", "Jamaica", Region.CARIBBEAN, None, 2970000, 15, ["en"], {"CARICOM"}),
    "HT": Country("HT", "Haiti", Region.CARIBBEAN, None, 11400000, 14, ["fr", "ht"], {"CARICOM"}),
    "DO": Country("DO", "Dominican Republic", Region.CARIBBEAN, None, 10900000, 95, ["es"], {"CAFTA"}),
    "TT": Country("TT", "Trinidad and Tobago", Region.CARIBBEAN, None, 1400000, 24, ["en"], {"CARICOM"}),
    "BS": Country("BS", "Bahamas", Region.CARIBBEAN, None, 400000, 12, ["en"], {"CARICOM"}),
    "BB": Country("BB", "Barbados", Region.CARIBBEAN, None, 290000, 5, ["en"], {"CARICOM"}),
    "LC": Country("LC", "Saint Lucia", Region.CARIBBEAN, None, 180000, 2, ["en"], {"CARICOM"}),
    "GD": Country("GD", "Grenada", Region.CARIBBEAN, None, 113000, 1, ["en"], {"CARICOM"}),
    "VC": Country("VC", "Saint Vincent", Region.CARIBBEAN, None, 111000, 1, ["en"], {"CARICOM"}),
    "AG": Country("AG", "Antigua and Barbuda", Region.CARIBBEAN, None, 98000, 2, ["en"], {"CARICOM"}),
    "DM": Country("DM", "Dominica", Region.CARIBBEAN, None, 72000, 1, ["en"], {"CARICOM"}),
    "KN": Country("KN", "Saint Kitts and Nevis", Region.CARIBBEAN, None, 53000, 1, ["en"], {"CARICOM"}),

    # ═══════════════════════════════════════════════════════════════════════════════
    # SOUTH AMERICA (12 countries)
    # ═══════════════════════════════════════════════════════════════════════════════
    "BR": Country("BR", "Brazil", Region.SOUTH_AMERICA, SubRegion.SOUTHERN_CONE,
                  213000000, 1900, ["pt"], {"G20", "BRICS", "Mercosur"}),
    "AR": Country("AR", "Argentina", Region.SOUTH_AMERICA, SubRegion.SOUTHERN_CONE,
                  45000000, 490, ["es"], {"G20", "Mercosur"}),
    "CO": Country("CO", "Colombia", Region.SOUTH_AMERICA, SubRegion.ANDEAN,
                  51000000, 340, ["es"], {"Pacific_Alliance", "OECD"}),
    "CL": Country("CL", "Chile", Region.SOUTH_AMERICA, SubRegion.ANDEAN,
                  19000000, 320, ["es"], {"Pacific_Alliance", "OECD", "APEC"}),
    "PE": Country("PE", "Peru", Region.SOUTH_AMERICA, SubRegion.ANDEAN,
                  33000000, 240, ["es", "qu"], {"Pacific_Alliance", "APEC"}),
    "VE": Country("VE", "Venezuela", Region.SOUTH_AMERICA, SubRegion.ANDEAN,
                  28000000, 100, ["es"], {"OPEC"}),
    "EC": Country("EC", "Ecuador", Region.SOUTH_AMERICA, SubRegion.ANDEAN,
                  18000000, 110, ["es"], {"OPEC"}),
    "BO": Country("BO", "Bolivia", Region.SOUTH_AMERICA, SubRegion.ANDEAN,
                  12000000, 43, ["es", "qu", "ay"], set()),
    "PY": Country("PY", "Paraguay", Region.SOUTH_AMERICA, SubRegion.SOUTHERN_CONE,
                  7100000, 41, ["es", "gn"], {"Mercosur"}),
    "UY": Country("UY", "Uruguay", Region.SOUTH_AMERICA, SubRegion.SOUTHERN_CONE,
                  3500000, 62, ["es"], {"Mercosur"}),
    "GY": Country("GY", "Guyana", Region.SOUTH_AMERICA, None, 790000, 15, ["en"], {"CARICOM"}),
    "SR": Country("SR", "Suriname", Region.SOUTH_AMERICA, None, 590000, 4, ["nl"], {"CARICOM"}),

    # ═══════════════════════════════════════════════════════════════════════════════
    # WESTERN EUROPE (12 countries)
    # ═══════════════════════════════════════════════════════════════════════════════
    "GB": Country("GB", "United Kingdom", Region.EUROPE, SubRegion.WESTERN_EUROPE,
                  67000000, 3100, ["en"], {"NATO", "G7", "G20", "OECD", "Five_Eyes", "AUKUS"}),
    "FR": Country("FR", "France", Region.EUROPE, SubRegion.WESTERN_EUROPE,
                  67000000, 2900, ["fr"], {"NATO", "EU", "G7", "G20", "OECD"}),
    "DE": Country("DE", "Germany", Region.EUROPE, SubRegion.CENTRAL_EUROPE,
                  83000000, 4200, ["de"], {"NATO", "EU", "G7", "G20", "OECD"}),
    "IT": Country("IT", "Italy", Region.EUROPE, SubRegion.SOUTHERN_EUROPE,
                  60000000, 2100, ["it"], {"NATO", "EU", "G7", "G20", "OECD"}),
    "ES": Country("ES", "Spain", Region.EUROPE, SubRegion.SOUTHERN_EUROPE,
                  47000000, 1400, ["es", "ca", "eu", "gl"], {"NATO", "EU", "OECD"}),
    "NL": Country("NL", "Netherlands", Region.EUROPE, SubRegion.WESTERN_EUROPE,
                  17500000, 1000, ["nl"], {"NATO", "EU", "OECD"}),
    "BE": Country("BE", "Belgium", Region.EUROPE, SubRegion.WESTERN_EUROPE,
                  11600000, 580, ["nl", "fr", "de"], {"NATO", "EU", "OECD"}),
    "PT": Country("PT", "Portugal", Region.EUROPE, SubRegion.SOUTHERN_EUROPE,
                  10300000, 250, ["pt"], {"NATO", "EU", "OECD"}),
    "IE": Country("IE", "Ireland", Region.EUROPE, SubRegion.WESTERN_EUROPE,
                  5000000, 500, ["en", "ga"], {"EU", "OECD"}),
    "AT": Country("AT", "Austria", Region.EUROPE, SubRegion.CENTRAL_EUROPE,
                  9000000, 470, ["de"], {"EU", "OECD"}),
    "CH": Country("CH", "Switzerland", Region.EUROPE, SubRegion.CENTRAL_EUROPE,
                  8700000, 800, ["de", "fr", "it", "rm"], {"OECD"}),
    "LU": Country("LU", "Luxembourg", Region.EUROPE, SubRegion.WESTERN_EUROPE,
                  640000, 85, ["fr", "de", "lb"], {"NATO", "EU", "OECD"}),

    # ═══════════════════════════════════════════════════════════════════════════════
    # NORTHERN EUROPE (10 countries)
    # ═══════════════════════════════════════════════════════════════════════════════
    "SE": Country("SE", "Sweden", Region.EUROPE, SubRegion.NORTHERN_EUROPE,
                  10400000, 590, ["sv"], {"NATO", "EU", "OECD"}),
    "NO": Country("NO", "Norway", Region.EUROPE, SubRegion.NORTHERN_EUROPE,
                  5400000, 480, ["no"], {"NATO", "OECD"}),
    "DK": Country("DK", "Denmark", Region.EUROPE, SubRegion.NORTHERN_EUROPE,
                  5800000, 400, ["da"], {"NATO", "EU", "OECD"}),
    "FI": Country("FI", "Finland", Region.EUROPE, SubRegion.NORTHERN_EUROPE,
                  5500000, 300, ["fi", "sv"], {"NATO", "EU", "OECD"}),
    "IS": Country("IS", "Iceland", Region.EUROPE, SubRegion.NORTHERN_EUROPE,
                  370000, 25, ["is"], {"NATO", "OECD"}),
    "EE": Country("EE", "Estonia", Region.EUROPE, SubRegion.BALTIC,
                  1330000, 37, ["et"], {"NATO", "EU", "OECD"}),
    "LV": Country("LV", "Latvia", Region.EUROPE, SubRegion.BALTIC,
                  1880000, 38, ["lv"], {"NATO", "EU", "OECD"}),
    "LT": Country("LT", "Lithuania", Region.EUROPE, SubRegion.BALTIC,
                  2800000, 65, ["lt"], {"NATO", "EU", "OECD"}),
    "GL": Country("GL", "Greenland", Region.ARCTIC, None, 57000, 3, ["kl", "da"], set()),
    "FO": Country("FO", "Faroe Islands", Region.EUROPE, SubRegion.NORTHERN_EUROPE, 54000, 3, ["fo", "da"], set()),

    # ═══════════════════════════════════════════════════════════════════════════════
    # CENTRAL EUROPE (7 countries)
    # ═══════════════════════════════════════════════════════════════════════════════
    "PL": Country("PL", "Poland", Region.EUROPE, SubRegion.CENTRAL_EUROPE,
                  38000000, 680, ["pl"], {"NATO", "EU", "OECD"}),
    "CZ": Country("CZ", "Czech Republic", Region.EUROPE, SubRegion.CENTRAL_EUROPE,
                  10700000, 280, ["cs"], {"NATO", "EU", "OECD"}),
    "SK": Country("SK", "Slovakia", Region.EUROPE, SubRegion.CENTRAL_EUROPE,
                  5500000, 115, ["sk"], {"NATO", "EU", "OECD"}),
    "HU": Country("HU", "Hungary", Region.EUROPE, SubRegion.CENTRAL_EUROPE,
                  9700000, 180, ["hu"], {"NATO", "EU", "OECD"}),
    "SI": Country("SI", "Slovenia", Region.EUROPE, SubRegion.CENTRAL_EUROPE,
                  2100000, 60, ["sl"], {"NATO", "EU", "OECD"}),
    "LI": Country("LI", "Liechtenstein", Region.EUROPE, SubRegion.CENTRAL_EUROPE,
                  39000, 7, ["de"], set()),
    "MC": Country("MC", "Monaco", Region.EUROPE, SubRegion.WESTERN_EUROPE,
                  39000, 8, ["fr"], set()),

    # ═══════════════════════════════════════════════════════════════════════════════
    # EASTERN EUROPE (10 countries)
    # ═══════════════════════════════════════════════════════════════════════════════
    "RU": Country("RU", "Russia", Region.EUROPE, SubRegion.EASTERN_EUROPE,
                  144000000, 1800, ["ru"], {"G20", "BRICS", "SCO", "CIS"}),
    "UA": Country("UA", "Ukraine", Region.EUROPE, SubRegion.EASTERN_EUROPE,
                  41000000, 160, ["uk"], set()),
    "BY": Country("BY", "Belarus", Region.EUROPE, SubRegion.EASTERN_EUROPE,
                  9300000, 60, ["be", "ru"], {"CIS", "CSTO"}),
    "MD": Country("MD", "Moldova", Region.EUROPE, SubRegion.EASTERN_EUROPE,
                  2600000, 14, ["ro"], set()),
    "RO": Country("RO", "Romania", Region.EUROPE, SubRegion.EASTERN_EUROPE,
                  19000000, 290, ["ro"], {"NATO", "EU"}),
    "BG": Country("BG", "Bulgaria", Region.EUROPE, SubRegion.EASTERN_EUROPE,
                  6900000, 80, ["bg"], {"NATO", "EU"}),
    "GE": Country("GE", "Georgia", Region.EUROPE, SubRegion.EASTERN_EUROPE,
                  3700000, 19, ["ka"], set()),
    "AM": Country("AM", "Armenia", Region.EUROPE, SubRegion.EASTERN_EUROPE,
                  3000000, 14, ["hy"], {"CIS", "CSTO"}),
    "AZ": Country("AZ", "Azerbaijan", Region.EUROPE, SubRegion.EASTERN_EUROPE,
                  10100000, 55, ["az"], {"CIS"}),
    "KZ": Country("KZ", "Kazakhstan", Region.CENTRAL_ASIA, None,
                  19000000, 190, ["kk", "ru"], {"CIS", "SCO", "CSTO"}),

    # ═══════════════════════════════════════════════════════════════════════════════
    # BALKANS (10 countries)
    # ═══════════════════════════════════════════════════════════════════════════════
    "HR": Country("HR", "Croatia", Region.EUROPE, SubRegion.BALKANS,
                  4000000, 68, ["hr"], {"NATO", "EU"}),
    "RS": Country("RS", "Serbia", Region.EUROPE, SubRegion.BALKANS,
                  6900000, 60, ["sr"], set()),
    "BA": Country("BA", "Bosnia and Herzegovina", Region.EUROPE, SubRegion.BALKANS,
                  3270000, 22, ["bs", "hr", "sr"], set()),
    "ME": Country("ME", "Montenegro", Region.EUROPE, SubRegion.BALKANS,
                  620000, 6, ["sr"], {"NATO"}),
    "MK": Country("MK", "North Macedonia", Region.EUROPE, SubRegion.BALKANS,
                  2080000, 13, ["mk", "sq"], {"NATO"}),
    "AL": Country("AL", "Albania", Region.EUROPE, SubRegion.BALKANS,
                  2880000, 18, ["sq"], {"NATO"}),
    "XK": Country("XK", "Kosovo", Region.EUROPE, SubRegion.BALKANS,
                  1800000, 9, ["sq", "sr"], set()),
    "GR": Country("GR", "Greece", Region.EUROPE, SubRegion.SOUTHERN_EUROPE,
                  10400000, 220, ["el"], {"NATO", "EU", "OECD"}),
    "CY": Country("CY", "Cyprus", Region.EUROPE, SubRegion.SOUTHERN_EUROPE,
                  1210000, 27, ["el", "tr"], {"EU"}),
    "MT": Country("MT", "Malta", Region.EUROPE, SubRegion.SOUTHERN_EUROPE,
                  520000, 17, ["mt", "en"], {"EU"}),

    # ═══════════════════════════════════════════════════════════════════════════════
    # MIDDLE EAST (17 countries)
    # ═══════════════════════════════════════════════════════════════════════════════
    "TR": Country("TR", "Turkey", Region.MIDDLE_EAST, SubRegion.LEVANT,
                  85000000, 800, ["tr"], {"NATO", "G20", "OECD"}),
    "IL": Country("IL", "Israel", Region.MIDDLE_EAST, SubRegion.LEVANT,
                  9400000, 520, ["he", "ar"], {"OECD"}),
    "SA": Country("SA", "Saudi Arabia", Region.MIDDLE_EAST, SubRegion.GULF_STATES,
                  35000000, 800, ["ar"], {"G20", "OPEC", "GCC"}),
    "AE": Country("AE", "United Arab Emirates", Region.MIDDLE_EAST, SubRegion.GULF_STATES,
                  10000000, 420, ["ar"], {"OPEC", "GCC"}),
    "QA": Country("QA", "Qatar", Region.MIDDLE_EAST, SubRegion.GULF_STATES,
                  2900000, 180, ["ar"], {"OPEC", "GCC"}),
    "KW": Country("KW", "Kuwait", Region.MIDDLE_EAST, SubRegion.GULF_STATES,
                  4300000, 135, ["ar"], {"OPEC", "GCC"}),
    "BH": Country("BH", "Bahrain", Region.MIDDLE_EAST, SubRegion.GULF_STATES,
                  1500000, 40, ["ar"], {"GCC"}),
    "OM": Country("OM", "Oman", Region.MIDDLE_EAST, SubRegion.GULF_STATES,
                  5100000, 85, ["ar"], {"GCC"}),
    "YE": Country("YE", "Yemen", Region.MIDDLE_EAST, SubRegion.GULF_STATES,
                  30500000, 20, ["ar"], set()),
    "IQ": Country("IQ", "Iraq", Region.MIDDLE_EAST, SubRegion.LEVANT,
                  41200000, 200, ["ar", "ku"], {"OPEC"}),
    "IR": Country("IR", "Iran", Region.MIDDLE_EAST, None,
                  85000000, 350, ["fa"], {"OPEC", "SCO"}),
    "SY": Country("SY", "Syria", Region.MIDDLE_EAST, SubRegion.LEVANT,
                  18300000, 10, ["ar"], set()),
    "LB": Country("LB", "Lebanon", Region.MIDDLE_EAST, SubRegion.LEVANT,
                  6800000, 20, ["ar", "fr"], set()),
    "JO": Country("JO", "Jordan", Region.MIDDLE_EAST, SubRegion.LEVANT,
                  10300000, 45, ["ar"], set()),
    "PS": Country("PS", "Palestine", Region.MIDDLE_EAST, SubRegion.LEVANT,
                  5100000, 5, ["ar"], set()),
    "AF": Country("AF", "Afghanistan", Region.CENTRAL_ASIA, None,
                  40000000, 20, ["ps", "fa"], set()),
    "PK": Country("PK", "Pakistan", Region.SOUTH_ASIA, None,
                  225000000, 350, ["ur", "en"], {"SCO", "OIC"}),

    # ═══════════════════════════════════════════════════════════════════════════════
    # CENTRAL ASIA (5 countries)
    # ═══════════════════════════════════════════════════════════════════════════════
    "UZ": Country("UZ", "Uzbekistan", Region.CENTRAL_ASIA, None,
                  34000000, 70, ["uz"], {"CIS", "SCO"}),
    "TM": Country("TM", "Turkmenistan", Region.CENTRAL_ASIA, None,
                  6100000, 45, ["tk"], {"CIS"}),
    "TJ": Country("TJ", "Tajikistan", Region.CENTRAL_ASIA, None,
                  9700000, 10, ["tg"], {"CIS", "SCO", "CSTO"}),
    "KG": Country("KG", "Kyrgyzstan", Region.CENTRAL_ASIA, None,
                  6600000, 9, ["ky", "ru"], {"CIS", "SCO", "CSTO"}),
    "MN": Country("MN", "Mongolia", Region.EAST_ASIA, None,
                  3300000, 15, ["mn"], set()),

    # ═══════════════════════════════════════════════════════════════════════════════
    # SOUTH ASIA (8 countries)
    # ═══════════════════════════════════════════════════════════════════════════════
    "IN": Country("IN", "India", Region.SOUTH_ASIA, SubRegion.INDIA_NORTH,
                  1400000000, 3500, ["hi", "en"], {"G20", "BRICS", "QUAD", "SCO"}),
    "BD": Country("BD", "Bangladesh", Region.SOUTH_ASIA, None,
                  166000000, 420, ["bn"], set()),
    "LK": Country("LK", "Sri Lanka", Region.SOUTH_ASIA, None,
                  22000000, 75, ["si", "ta"], set()),
    "NP": Country("NP", "Nepal", Region.SOUTH_ASIA, None,
                  30000000, 37, ["ne"], set()),
    "BT": Country("BT", "Bhutan", Region.SOUTH_ASIA, None,
                  780000, 2.5, ["dz"], set()),
    "MV": Country("MV", "Maldives", Region.SOUTH_ASIA, None,
                  540000, 5, ["dv"], set()),
    "MM": Country("MM", "Myanmar", Region.SOUTHEAST_ASIA, None,
                  54000000, 70, ["my"], {"ASEAN"}),

    # ═══════════════════════════════════════════════════════════════════════════════
    # EAST ASIA (6 countries/territories)
    # ═══════════════════════════════════════════════════════════════════════════════
    "CN": Country("CN", "China", Region.EAST_ASIA, SubRegion.CHINA_COASTAL,
                  1400000000, 18000, ["zh"], {"G20", "BRICS", "SCO", "APEC"}),
    "JP": Country("JP", "Japan", Region.EAST_ASIA, SubRegion.JAPAN_KOREA,
                  125000000, 4200, ["ja"], {"G7", "G20", "OECD", "QUAD", "APEC"}),
    "KR": Country("KR", "South Korea", Region.EAST_ASIA, SubRegion.JAPAN_KOREA,
                  52000000, 1800, ["ko"], {"G20", "OECD", "APEC"}),
    "KP": Country("KP", "North Korea", Region.EAST_ASIA, SubRegion.JAPAN_KOREA,
                  26000000, 30, ["ko"], set()),
    "TW": Country("TW", "Taiwan", Region.EAST_ASIA, SubRegion.CHINA_COASTAL,
                  24000000, 750, ["zh"], {"APEC"}),
    "HK": Country("HK", "Hong Kong", Region.EAST_ASIA, SubRegion.CHINA_COASTAL,
                  7500000, 360, ["zh", "en"], {"APEC"}),

    # ═══════════════════════════════════════════════════════════════════════════════
    # SOUTHEAST ASIA (11 countries)
    # ═══════════════════════════════════════════════════════════════════════════════
    "ID": Country("ID", "Indonesia", Region.SOUTHEAST_ASIA, None,
                  274000000, 1200, ["id"], {"G20", "ASEAN", "APEC"}),
    "TH": Country("TH", "Thailand", Region.SOUTHEAST_ASIA, None,
                  70000000, 500, ["th"], {"ASEAN", "APEC"}),
    "VN": Country("VN", "Vietnam", Region.SOUTHEAST_ASIA, None,
                  98000000, 400, ["vi"], {"ASEAN", "APEC"}),
    "PH": Country("PH", "Philippines", Region.SOUTHEAST_ASIA, None,
                  111000000, 400, ["tl", "en"], {"ASEAN", "APEC"}),
    "MY": Country("MY", "Malaysia", Region.SOUTHEAST_ASIA, None,
                  33000000, 370, ["ms"], {"ASEAN", "APEC"}),
    "SG": Country("SG", "Singapore", Region.SOUTHEAST_ASIA, None,
                  5900000, 400, ["en", "zh", "ms", "ta"], {"ASEAN", "APEC"}),
    "KH": Country("KH", "Cambodia", Region.SOUTHEAST_ASIA, None,
                  16900000, 27, ["km"], {"ASEAN"}),
    "LA": Country("LA", "Laos", Region.SOUTHEAST_ASIA, None,
                  7400000, 19, ["lo"], {"ASEAN"}),
    "BN": Country("BN", "Brunei", Region.SOUTHEAST_ASIA, None,
                  440000, 14, ["ms"], {"ASEAN", "APEC"}),
    "TL": Country("TL", "Timor-Leste", Region.SOUTHEAST_ASIA, None,
                  1340000, 2, ["pt", "tet"], {"ASEAN"}),

    # ═══════════════════════════════════════════════════════════════════════════════
    # OCEANIA (14 countries)
    # ═══════════════════════════════════════════════════════════════════════════════
    "AU": Country("AU", "Australia", Region.OCEANIA, SubRegion.PACIFIC_ISLANDS,
                  26000000, 1700, ["en"], {"G20", "OECD", "QUAD", "Five_Eyes", "AUKUS", "APEC"}),
    "NZ": Country("NZ", "New Zealand", Region.OCEANIA, SubRegion.PACIFIC_ISLANDS,
                  5100000, 250, ["en", "mi"], {"OECD", "Five_Eyes", "APEC"}),
    "PG": Country("PG", "Papua New Guinea", Region.OCEANIA, SubRegion.PACIFIC_ISLANDS,
                  9100000, 26, ["en", "tpi", "ho"], {"APEC"}),
    "FJ": Country("FJ", "Fiji", Region.OCEANIA, SubRegion.PACIFIC_ISLANDS,
                  900000, 5, ["en", "fj", "hi"], set()),
    "SB": Country("SB", "Solomon Islands", Region.OCEANIA, SubRegion.PACIFIC_ISLANDS,
                  700000, 1.5, ["en"], set()),
    "VU": Country("VU", "Vanuatu", Region.OCEANIA, SubRegion.PACIFIC_ISLANDS,
                  310000, 0.9, ["en", "fr", "bi"], set()),
    "WS": Country("WS", "Samoa", Region.OCEANIA, SubRegion.PACIFIC_ISLANDS,
                  200000, 0.8, ["sm", "en"], set()),
    "TO": Country("TO", "Tonga", Region.OCEANIA, SubRegion.PACIFIC_ISLANDS,
                  106000, 0.5, ["to", "en"], set()),
    "FM": Country("FM", "Micronesia", Region.OCEANIA, SubRegion.PACIFIC_ISLANDS,
                  115000, 0.4, ["en"], set()),
    "KI": Country("KI", "Kiribati", Region.OCEANIA, SubRegion.PACIFIC_ISLANDS,
                  120000, 0.2, ["en", "gil"], set()),
    "MH": Country("MH", "Marshall Islands", Region.OCEANIA, SubRegion.PACIFIC_ISLANDS,
                  60000, 0.2, ["en", "mh"], set()),
    "PW": Country("PW", "Palau", Region.OCEANIA, SubRegion.PACIFIC_ISLANDS,
                  18000, 0.3, ["en", "pau"], set()),
    "NR": Country("NR", "Nauru", Region.OCEANIA, SubRegion.PACIFIC_ISLANDS,
                  11000, 0.1, ["en", "na"], set()),
    "TV": Country("TV", "Tuvalu", Region.OCEANIA, SubRegion.PACIFIC_ISLANDS,
                  12000, 0.05, ["en", "tvl"], set()),

    # ═══════════════════════════════════════════════════════════════════════════════
    # NORTH AFRICA (6 countries)
    # ═══════════════════════════════════════════════════════════════════════════════
    "EG": Country("EG", "Egypt", Region.AFRICA, SubRegion.NORTH_AFRICA,
                  104000000, 400, ["ar"], {"AU", "Arab_League"}),
    "LY": Country("LY", "Libya", Region.AFRICA, SubRegion.NORTH_AFRICA,
                  7000000, 40, ["ar"], {"AU", "OPEC", "Arab_League"}),
    "TN": Country("TN", "Tunisia", Region.AFRICA, SubRegion.NORTH_AFRICA,
                  12000000, 45, ["ar", "fr"], {"AU", "Arab_League"}),
    "DZ": Country("DZ", "Algeria", Region.AFRICA, SubRegion.NORTH_AFRICA,
                  44000000, 170, ["ar", "fr"], {"AU", "OPEC", "Arab_League"}),
    "MA": Country("MA", "Morocco", Region.AFRICA, SubRegion.NORTH_AFRICA,
                  37000000, 130, ["ar", "fr"], {"AU", "Arab_League"}),
    "SD": Country("SD", "Sudan", Region.AFRICA, SubRegion.NORTH_AFRICA,
                  44000000, 30, ["ar"], {"AU", "Arab_League"}),

    # ═══════════════════════════════════════════════════════════════════════════════
    # WEST AFRICA (16 countries)
    # ═══════════════════════════════════════════════════════════════════════════════
    "NG": Country("NG", "Nigeria", Region.AFRICA, SubRegion.WEST_AFRICA,
                  211000000, 450, ["en"], {"AU", "ECOWAS", "OPEC"}),
    "GH": Country("GH", "Ghana", Region.AFRICA, SubRegion.WEST_AFRICA,
                  32000000, 75, ["en"], {"AU", "ECOWAS"}),
    "CI": Country("CI", "Côte d'Ivoire", Region.AFRICA, SubRegion.WEST_AFRICA,
                  27000000, 70, ["fr"], {"AU", "ECOWAS"}),
    "SN": Country("SN", "Senegal", Region.AFRICA, SubRegion.WEST_AFRICA,
                  17000000, 27, ["fr"], {"AU", "ECOWAS"}),
    "ML": Country("ML", "Mali", Region.AFRICA, SubRegion.WEST_AFRICA,
                  21000000, 18, ["fr"], {"AU"}),
    "BF": Country("BF", "Burkina Faso", Region.AFRICA, SubRegion.WEST_AFRICA,
                  21000000, 17, ["fr"], {"AU"}),
    "NE": Country("NE", "Niger", Region.AFRICA, SubRegion.WEST_AFRICA,
                  25000000, 14, ["fr"], {"AU"}),
    "GN": Country("GN", "Guinea", Region.AFRICA, SubRegion.WEST_AFRICA,
                  13000000, 16, ["fr"], {"AU"}),
    "BJ": Country("BJ", "Benin", Region.AFRICA, SubRegion.WEST_AFRICA,
                  12000000, 17, ["fr"], {"AU", "ECOWAS"}),
    "TG": Country("TG", "Togo", Region.AFRICA, SubRegion.WEST_AFRICA,
                  8400000, 8, ["fr"], {"AU", "ECOWAS"}),
    "SL": Country("SL", "Sierra Leone", Region.AFRICA, SubRegion.WEST_AFRICA,
                  8100000, 4, ["en"], {"AU", "ECOWAS"}),
    "LR": Country("LR", "Liberia", Region.AFRICA, SubRegion.WEST_AFRICA,
                  5200000, 3.5, ["en"], {"AU", "ECOWAS"}),
    "MR": Country("MR", "Mauritania", Region.AFRICA, SubRegion.WEST_AFRICA,
                  4800000, 8, ["ar", "fr"], {"AU", "Arab_League"}),
    "GM": Country("GM", "Gambia", Region.AFRICA, SubRegion.WEST_AFRICA,
                  2500000, 2, ["en"], {"AU", "ECOWAS"}),
    "GW": Country("GW", "Guinea-Bissau", Region.AFRICA, SubRegion.WEST_AFRICA,
                  2000000, 1.5, ["pt"], {"AU", "ECOWAS"}),
    "CV": Country("CV", "Cape Verde", Region.AFRICA, SubRegion.WEST_AFRICA,
                  560000, 2, ["pt"], {"AU", "ECOWAS"}),

    # ═══════════════════════════════════════════════════════════════════════════════
    # CENTRAL AFRICA (9 countries)
    # ═══════════════════════════════════════════════════════════════════════════════
    "CD": Country("CD", "DR Congo", Region.AFRICA, SubRegion.CENTRAL_AFRICA,
                  92000000, 55, ["fr"], {"AU"}),
    "AO": Country("AO", "Angola", Region.AFRICA, SubRegion.CENTRAL_AFRICA,
                  33000000, 75, ["pt"], {"AU", "OPEC"}),
    "CM": Country("CM", "Cameroon", Region.AFRICA, SubRegion.CENTRAL_AFRICA,
                  27000000, 45, ["fr", "en"], {"AU"}),
    "TD": Country("TD", "Chad", Region.AFRICA, SubRegion.CENTRAL_AFRICA,
                  17000000, 11, ["fr", "ar"], {"AU"}),
    "CG": Country("CG", "Congo", Region.AFRICA, SubRegion.CENTRAL_AFRICA,
                  5700000, 15, ["fr"], {"AU", "OPEC"}),
    "GA": Country("GA", "Gabon", Region.AFRICA, SubRegion.CENTRAL_AFRICA,
                  2300000, 18, ["fr"], {"AU", "OPEC"}),
    "GQ": Country("GQ", "Equatorial Guinea", Region.AFRICA, SubRegion.CENTRAL_AFRICA,
                  1500000, 12, ["es", "fr", "pt"], {"AU", "OPEC"}),
    "CF": Country("CF", "Central African Republic", Region.AFRICA, SubRegion.CENTRAL_AFRICA,
                  4900000, 2, ["fr", "sg"], {"AU"}),
    "ST": Country("ST", "São Tomé and Príncipe", Region.AFRICA, SubRegion.CENTRAL_AFRICA,
                  220000, 0.5, ["pt"], {"AU"}),

    # ═══════════════════════════════════════════════════════════════════════════════
    # EAST AFRICA (14 countries)
    # ═══════════════════════════════════════════════════════════════════════════════
    "ET": Country("ET", "Ethiopia", Region.AFRICA, SubRegion.EAST_AFRICA,
                  118000000, 110, ["am"], {"AU", "BRICS"}),
    "KE": Country("KE", "Kenya", Region.AFRICA, SubRegion.EAST_AFRICA,
                  54000000, 110, ["sw", "en"], {"AU"}),
    "TZ": Country("TZ", "Tanzania", Region.AFRICA, SubRegion.EAST_AFRICA,
                  62000000, 75, ["sw", "en"], {"AU"}),
    "UG": Country("UG", "Uganda", Region.AFRICA, SubRegion.EAST_AFRICA,
                  47000000, 45, ["en", "sw"], {"AU"}),
    "RW": Country("RW", "Rwanda", Region.AFRICA, SubRegion.EAST_AFRICA,
                  13000000, 12, ["rw", "en", "fr"], {"AU"}),
    "BI": Country("BI", "Burundi", Region.AFRICA, SubRegion.EAST_AFRICA,
                  12000000, 3, ["rn", "fr"], {"AU"}),
    "SS": Country("SS", "South Sudan", Region.AFRICA, SubRegion.EAST_AFRICA,
                  11000000, 3, ["en"], {"AU"}),
    "SO": Country("SO", "Somalia", Region.AFRICA, SubRegion.EAST_AFRICA,
                  16000000, 7, ["so", "ar"], {"AU", "Arab_League"}),
    "ER": Country("ER", "Eritrea", Region.AFRICA, SubRegion.EAST_AFRICA,
                  3600000, 2, ["ti", "ar"], {"AU"}),
    "DJ": Country("DJ", "Djibouti", Region.AFRICA, SubRegion.EAST_AFRICA,
                  990000, 3.5, ["fr", "ar"], {"AU", "Arab_League"}),
    "MU": Country("MU", "Mauritius", Region.AFRICA, SubRegion.EAST_AFRICA,
                  1270000, 14, ["en", "fr"], {"AU"}),
    "SC": Country("SC", "Seychelles", Region.AFRICA, SubRegion.EAST_AFRICA,
                  99000, 2, ["en", "fr", "crs"], {"AU"}),
    "KM": Country("KM", "Comoros", Region.AFRICA, SubRegion.EAST_AFRICA,
                  890000, 1, ["ar", "fr", "sw"], {"AU", "Arab_League"}),
    "MG": Country("MG", "Madagascar", Region.AFRICA, SubRegion.EAST_AFRICA,
                  28000000, 14, ["mg", "fr"], {"AU"}),

    # ═══════════════════════════════════════════════════════════════════════════════
    # SOUTHERN AFRICA (10 countries)
    # ═══════════════════════════════════════════════════════════════════════════════
    "ZA": Country("ZA", "South Africa", Region.AFRICA, SubRegion.SOUTHERN_AFRICA,
                  60000000, 400, ["en", "af", "zu", "xh"], {"G20", "BRICS", "AU"}),
    "ZW": Country("ZW", "Zimbabwe", Region.AFRICA, SubRegion.SOUTHERN_AFRICA,
                  15000000, 20, ["en", "sn", "nd"], {"AU"}),
    "ZM": Country("ZM", "Zambia", Region.AFRICA, SubRegion.SOUTHERN_AFRICA,
                  19000000, 22, ["en"], {"AU"}),
    "MW": Country("MW", "Malawi", Region.AFRICA, SubRegion.SOUTHERN_AFRICA,
                  19000000, 12, ["en", "ny"], {"AU"}),
    "MZ": Country("MZ", "Mozambique", Region.AFRICA, SubRegion.SOUTHERN_AFRICA,
                  32000000, 17, ["pt"], {"AU"}),
    "BW": Country("BW", "Botswana", Region.AFRICA, SubRegion.SOUTHERN_AFRICA,
                  2400000, 18, ["en", "tn"], {"AU"}),
    "NA": Country("NA", "Namibia", Region.AFRICA, SubRegion.SOUTHERN_AFRICA,
                  2500000, 13, ["en"], {"AU"}),
    "SZ": Country("SZ", "Eswatini", Region.AFRICA, SubRegion.SOUTHERN_AFRICA,
                  1200000, 5, ["en", "ss"], {"AU"}),
    "LS": Country("LS", "Lesotho", Region.AFRICA, SubRegion.SOUTHERN_AFRICA,
                  2200000, 2.5, ["en", "st"], {"AU"}),
    "AQ": Country("AQ", "Antarctica", Region.ANTARCTIC, None,
                  1000, 0, [], {"Antarctic_Treaty"}),
}


# ═══════════════════════════════════════════════════════════════════════════════
# ORGANIZATIONAL ARCHETYPES
# ═══════════════════════════════════════════════════════════════════════════════

class OrgType(str, Enum):
    """Types of organizations."""
    # Government
    GOVERNMENT_EXECUTIVE = "government_executive"
    GOVERNMENT_LEGISLATIVE = "government_legislative"
    GOVERNMENT_JUDICIAL = "government_judicial"
    GOVERNMENT_AGENCY = "government_agency"
    GOVERNMENT_LOCAL = "government_local"
    GOVERNMENT_STATE = "government_state"

    # Intergovernmental
    INTERGOVERNMENTAL_GLOBAL = "intergovernmental_global"  # UN, WTO
    INTERGOVERNMENTAL_REGIONAL = "intergovernmental_regional"  # EU, ASEAN
    INTERGOVERNMENTAL_MILITARY = "intergovernmental_military"  # NATO
    INTERGOVERNMENTAL_ECONOMIC = "intergovernmental_economic"  # IMF, World Bank

    # Corporate
    CORPORATION_MULTINATIONAL = "corporation_multinational"
    CORPORATION_NATIONAL = "corporation_national"
    CORPORATION_SME = "corporation_sme"
    CORPORATION_STARTUP = "corporation_startup"
    CORPORATION_STATE_OWNED = "corporation_state_owned"

    # Financial
    BANK_CENTRAL = "bank_central"
    BANK_COMMERCIAL = "bank_commercial"
    BANK_INVESTMENT = "bank_investment"
    HEDGE_FUND = "hedge_fund"
    SOVEREIGN_WEALTH = "sovereign_wealth"
    INSURANCE = "insurance"

    # NGO/Civil Society
    NGO_INTERNATIONAL = "ngo_international"
    NGO_NATIONAL = "ngo_national"
    NGO_LOCAL = "ngo_local"
    FOUNDATION = "foundation"
    CHARITY = "charity"
    ADVOCACY_GROUP = "advocacy_group"

    # Religious
    RELIGIOUS_GLOBAL = "religious_global"  # Vatican, OIC
    RELIGIOUS_NATIONAL = "religious_national"
    RELIGIOUS_LOCAL = "religious_local"
    RELIGIOUS_SECT = "religious_sect"

    # Military/Security
    MILITARY_COMMAND = "military_command"
    MILITARY_BRANCH = "military_branch"
    INTELLIGENCE_AGENCY = "intelligence_agency"
    PRIVATE_MILITARY = "private_military"
    POLICE = "police"

    # Academic/Research
    UNIVERSITY = "university"
    RESEARCH_INSTITUTE = "research_institute"
    THINK_TANK = "think_tank"
    ACADEMY = "academy"

    # Media
    MEDIA_CONGLOMERATE = "media_conglomerate"
    NEWS_ORGANIZATION = "news_organization"
    SOCIAL_PLATFORM = "social_platform"
    PUBLISHER = "publisher"

    # Cultural
    CULTURAL_INSTITUTION = "cultural_institution"
    SPORTS_ORGANIZATION = "sports_organization"
    ENTERTAINMENT = "entertainment"

    # Labor/Professional
    LABOR_UNION = "labor_union"
    PROFESSIONAL_ASSOCIATION = "professional_association"
    TRADE_ASSOCIATION = "trade_association"

    # Criminal/Informal
    CRIMINAL_ORGANIZATION = "criminal_organization"
    INFORMAL_NETWORK = "informal_network"
    DIASPORA_COMMUNITY = "diaspora_community"


class HierarchyType(str, Enum):
    """Types of hierarchical structures."""
    STRICT = "strict"                    # Military, traditional corps
    MATRIX = "matrix"                    # Cross-functional
    FLAT = "flat"                        # Startups, collectives
    NETWORK = "network"                  # Distributed, decentralized
    FEDERATED = "federated"              # Autonomous units
    THEOCRATIC = "theocratic"            # Religious authority
    DEMOCRATIC = "democratic"            # Elected leadership
    OLIGARCHIC = "oligarchic"            # Small ruling group
    CONSENSUS = "consensus"              # Collective decision
    HYBRID = "hybrid"                    # Mixed structures


# ═══════════════════════════════════════════════════════════════════════════════
# TAG-BASED META-FACETS
# ═══════════════════════════════════════════════════════════════════════════════

class IndustryTag(str, Enum):
    """Industry classifications."""
    TECHNOLOGY = "technology"
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    ENERGY = "energy"
    MANUFACTURING = "manufacturing"
    RETAIL = "retail"
    AGRICULTURE = "agriculture"
    DEFENSE = "defense"
    AEROSPACE = "aerospace"
    AUTOMOTIVE = "automotive"
    TELECOMMUNICATIONS = "telecommunications"
    MEDIA_ENTERTAINMENT = "media_entertainment"
    REAL_ESTATE = "real_estate"
    HOSPITALITY = "hospitality"
    TRANSPORTATION = "transportation"
    EDUCATION = "education"
    PHARMA_BIOTECH = "pharma_biotech"
    MINING = "mining"
    CONSTRUCTION = "construction"
    LEGAL = "legal"
    CONSULTING = "consulting"


class IdeologyTag(str, Enum):
    """Political/ideological leanings."""
    LIBERAL = "liberal"
    CONSERVATIVE = "conservative"
    PROGRESSIVE = "progressive"
    LIBERTARIAN = "libertarian"
    SOCIALIST = "socialist"
    NATIONALIST = "nationalist"
    GLOBALIST = "globalist"
    ENVIRONMENTALIST = "environmentalist"
    RELIGIOUS_CONSERVATIVE = "religious_conservative"
    SECULAR = "secular"
    CENTRIST = "centrist"
    POPULIST = "populist"
    TECHNOCRATIC = "technocratic"


class InfluenceTag(str, Enum):
    """Spheres of influence."""
    ECONOMIC = "economic"
    POLITICAL = "political"
    MILITARY = "military"
    CULTURAL = "cultural"
    TECHNOLOGICAL = "technological"
    INFORMATIONAL = "informational"
    RELIGIOUS = "religious"
    DIPLOMATIC = "diplomatic"
    SCIENTIFIC = "scientific"
    SOCIAL = "social"


class SizeTag(str, Enum):
    """Organization size categories."""
    MICRO = "micro"           # < 10 members
    SMALL = "small"           # 10-50
    MEDIUM = "medium"         # 50-250
    LARGE = "large"           # 250-1000
    ENTERPRISE = "enterprise" # 1000-10000
    MEGA = "mega"             # > 10000


class ReachTag(str, Enum):
    """Geographic reach."""
    LOCAL = "local"
    CITY = "city"
    STATE = "state"
    NATIONAL = "national"
    REGIONAL = "regional"
    CONTINENTAL = "continental"
    GLOBAL = "global"


class PowerTag(str, Enum):
    """Power/influence level."""
    MARGINAL = "marginal"
    LOCAL_POWER = "local_power"
    REGIONAL_POWER = "regional_power"
    MAJOR_PLAYER = "major_player"
    SUPERPOWER = "superpower"
    HEGEMON = "hegemon"


class AlignmentTag(str, Enum):
    """Geopolitical alignment."""
    WESTERN = "western"
    EASTERN = "eastern"
    GLOBAL_SOUTH = "global_south"
    NON_ALIGNED = "non_aligned"
    NEUTRAL = "neutral"
    CONTESTED = "contested"


# ═══════════════════════════════════════════════════════════════════════════════
# ROLE TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RoleTemplate:
    """Template for a role within an organization."""
    name: str
    level: int  # 0 = top, higher = lower in hierarchy
    reports_to: Optional[str] = None
    direct_reports: int = 0  # Expected number
    tags: Set[str] = field(default_factory=set)


# Standard corporate hierarchy
CORPORATE_ROLES = [
    RoleTemplate("CEO", 0, None, 5, {"executive", "leadership"}),
    RoleTemplate("CFO", 1, "CEO", 3, {"executive", "finance"}),
    RoleTemplate("COO", 1, "CEO", 5, {"executive", "operations"}),
    RoleTemplate("CTO", 1, "CEO", 4, {"executive", "technology"}),
    RoleTemplate("CMO", 1, "CEO", 3, {"executive", "marketing"}),
    RoleTemplate("VP_Sales", 2, "COO", 5, {"management", "sales"}),
    RoleTemplate("VP_Engineering", 2, "CTO", 8, {"management", "engineering"}),
    RoleTemplate("VP_Product", 2, "CTO", 4, {"management", "product"}),
    RoleTemplate("Director", 3, None, 4, {"management"}),
    RoleTemplate("Manager", 4, "Director", 6, {"management"}),
    RoleTemplate("Team_Lead", 5, "Manager", 5, {"leadership"}),
    RoleTemplate("Senior", 6, "Team_Lead", 0, {"individual_contributor"}),
    RoleTemplate("Mid", 7, "Senior", 0, {"individual_contributor"}),
    RoleTemplate("Junior", 8, "Mid", 0, {"individual_contributor"}),
]

# Government hierarchy
GOVERNMENT_ROLES = [
    RoleTemplate("Head_of_State", 0, None, 3, {"executive", "leadership"}),
    RoleTemplate("Head_of_Government", 0, "Head_of_State", 10, {"executive", "leadership"}),
    RoleTemplate("Cabinet_Minister", 1, "Head_of_Government", 5, {"executive", "policy"}),
    RoleTemplate("Deputy_Minister", 2, "Cabinet_Minister", 4, {"executive"}),
    RoleTemplate("Director_General", 3, "Deputy_Minister", 6, {"senior_official"}),
    RoleTemplate("Director", 4, "Director_General", 5, {"management"}),
    RoleTemplate("Department_Head", 5, "Director", 8, {"management"}),
    RoleTemplate("Section_Chief", 6, "Department_Head", 6, {"supervision"}),
    RoleTemplate("Senior_Officer", 7, "Section_Chief", 0, {"official"}),
    RoleTemplate("Officer", 8, "Senior_Officer", 0, {"official"}),
    RoleTemplate("Clerk", 9, "Officer", 0, {"support"}),
]

# Military hierarchy
MILITARY_ROLES = [
    RoleTemplate("Commander_in_Chief", 0, None, 4, {"command", "strategic"}),
    RoleTemplate("General", 1, "Commander_in_Chief", 4, {"command", "strategic"}),
    RoleTemplate("Lieutenant_General", 2, "General", 3, {"command"}),
    RoleTemplate("Major_General", 3, "Lieutenant_General", 4, {"command"}),
    RoleTemplate("Brigadier", 4, "Major_General", 5, {"command"}),
    RoleTemplate("Colonel", 5, "Brigadier", 4, {"command", "tactical"}),
    RoleTemplate("Lieutenant_Colonel", 6, "Colonel", 3, {"command"}),
    RoleTemplate("Major", 7, "Lieutenant_Colonel", 4, {"officer"}),
    RoleTemplate("Captain", 8, "Major", 3, {"officer"}),
    RoleTemplate("Lieutenant", 9, "Captain", 2, {"officer"}),
    RoleTemplate("Sergeant_Major", 10, "Lieutenant", 5, {"NCO"}),
    RoleTemplate("Sergeant", 11, "Sergeant_Major", 8, {"NCO"}),
    RoleTemplate("Corporal", 12, "Sergeant", 4, {"NCO"}),
    RoleTemplate("Private", 13, "Corporal", 0, {"enlisted"}),
]

# Religious hierarchy (Catholic-style)
RELIGIOUS_ROLES = [
    RoleTemplate("Pope", 0, None, 10, {"spiritual_leader"}),
    RoleTemplate("Cardinal", 1, "Pope", 5, {"senior_clergy"}),
    RoleTemplate("Archbishop", 2, "Cardinal", 8, {"clergy"}),
    RoleTemplate("Bishop", 3, "Archbishop", 12, {"clergy"}),
    RoleTemplate("Monsignor", 4, "Bishop", 0, {"clergy"}),
    RoleTemplate("Parish_Priest", 5, "Bishop", 3, {"clergy"}),
    RoleTemplate("Deacon", 6, "Parish_Priest", 0, {"clergy"}),
    RoleTemplate("Lay_Leader", 7, "Parish_Priest", 10, {"laity"}),
]

# Academic hierarchy
ACADEMIC_ROLES = [
    RoleTemplate("Chancellor", 0, None, 4, {"leadership"}),
    RoleTemplate("Provost", 1, "Chancellor", 6, {"academic_leadership"}),
    RoleTemplate("Dean", 2, "Provost", 8, {"academic_leadership"}),
    RoleTemplate("Department_Chair", 3, "Dean", 15, {"faculty_leadership"}),
    RoleTemplate("Full_Professor", 4, "Department_Chair", 0, {"faculty", "tenured"}),
    RoleTemplate("Associate_Professor", 5, "Department_Chair", 0, {"faculty", "tenured"}),
    RoleTemplate("Assistant_Professor", 6, "Department_Chair", 0, {"faculty"}),
    RoleTemplate("Lecturer", 7, "Department_Chair", 0, {"faculty"}),
    RoleTemplate("Postdoc", 8, None, 0, {"researcher"}),
    RoleTemplate("PhD_Student", 9, "Full_Professor", 0, {"student", "researcher"}),
    RoleTemplate("Masters_Student", 10, None, 0, {"student"}),
]

# NGO hierarchy
NGO_ROLES = [
    RoleTemplate("Executive_Director", 0, None, 5, {"leadership"}),
    RoleTemplate("Board_Chair", 0, None, 0, {"governance"}),
    RoleTemplate("Deputy_Director", 1, "Executive_Director", 4, {"leadership"}),
    RoleTemplate("Program_Director", 2, "Deputy_Director", 6, {"management"}),
    RoleTemplate("Country_Director", 2, "Deputy_Director", 10, {"field_leadership"}),
    RoleTemplate("Program_Manager", 3, "Program_Director", 5, {"management"}),
    RoleTemplate("Project_Coordinator", 4, "Program_Manager", 4, {"coordination"}),
    RoleTemplate("Field_Officer", 5, "Project_Coordinator", 0, {"field"}),
    RoleTemplate("Volunteer_Coordinator", 4, "Program_Manager", 20, {"coordination"}),
    RoleTemplate("Volunteer", 6, "Volunteer_Coordinator", 0, {"volunteer"}),
]


# ═══════════════════════════════════════════════════════════════════════════════
# ORGANIZATION ENTITY
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Organization:
    """An organization in the global network."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    org_type: OrgType = OrgType.CORPORATION_NATIONAL
    hierarchy_type: HierarchyType = HierarchyType.STRICT

    # Geographic
    headquarters_country: str = ""  # ISO code
    operating_countries: Set[str] = field(default_factory=set)
    regions: Set[Region] = field(default_factory=set)

    # Tags
    industries: Set[IndustryTag] = field(default_factory=set)
    ideologies: Set[IdeologyTag] = field(default_factory=set)
    influences: Set[InfluenceTag] = field(default_factory=set)
    size: SizeTag = SizeTag.MEDIUM
    reach: ReachTag = ReachTag.NATIONAL
    power: PowerTag = PowerTag.LOCAL_POWER
    alignment: AlignmentTag = AlignmentTag.NON_ALIGNED

    # Custom tags
    custom_tags: Set[str] = field(default_factory=set)

    # Relationships
    parent_org: Optional[str] = None
    subsidiary_orgs: Set[str] = field(default_factory=set)
    allied_orgs: Set[str] = field(default_factory=set)
    rival_orgs: Set[str] = field(default_factory=set)

    # Members
    member_count: int = 0
    roles: List[str] = field(default_factory=list)  # Role IDs

    # Metadata
    founded: int = 2000
    active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def all_tags(self) -> Set[str]:
        """Get all tags as strings."""
        tags = set(self.custom_tags)
        tags.update(i.value for i in self.industries)
        tags.update(i.value for i in self.ideologies)
        tags.update(i.value for i in self.influences)
        tags.add(self.size.value)
        tags.add(self.reach.value)
        tags.add(self.power.value)
        tags.add(self.alignment.value)
        tags.add(self.org_type.value)
        tags.add(self.hierarchy_type.value)
        return tags


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL NETWORK
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GlobalAgent:
    """An agent within the global network."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    name: str = ""
    organization_id: str = ""
    role: str = ""
    level: int = 0

    # Geographic
    country: str = ""
    region: Optional[Region] = None

    # Tags
    tags: Set[str] = field(default_factory=set)

    # Network
    reports_to: Optional[str] = None
    direct_reports: Set[str] = field(default_factory=set)
    cross_org_connections: Set[str] = field(default_factory=set)

    # Metadata
    influence_score: float = 0.0
    active: bool = True


class GlobalNetwork:
    """
    A global-scale network of organizations and agents.
    Supports thousands of hierarchies with geographic and tag-based facets.
    """

    def __init__(self):
        self.organizations: Dict[str, Organization] = {}
        self.agents: Dict[str, GlobalAgent] = {}
        self.countries: Dict[str, Country] = dict(COUNTRIES)

        # Indexes for fast lookup
        self._by_country: Dict[str, Set[str]] = defaultdict(set)  # country -> org_ids
        self._by_region: Dict[Region, Set[str]] = defaultdict(set)  # region -> org_ids
        self._by_type: Dict[OrgType, Set[str]] = defaultdict(set)  # type -> org_ids
        self._by_tag: Dict[str, Set[str]] = defaultdict(set)  # tag -> org_ids
        self._by_industry: Dict[IndustryTag, Set[str]] = defaultdict(set)
        self._by_alignment: Dict[AlignmentTag, Set[str]] = defaultdict(set)

        # Agent indexes
        self._agents_by_org: Dict[str, Set[str]] = defaultdict(set)
        self._agents_by_country: Dict[str, Set[str]] = defaultdict(set)
        self._agents_by_tag: Dict[str, Set[str]] = defaultdict(set)

        # Inter-org relationships
        self._alliances: Dict[str, Set[str]] = defaultdict(set)  # org_id -> allied_org_ids
        self._rivalries: Dict[str, Set[str]] = defaultdict(set)

    # ─────────────────────────────────────────────────────────────────────────
    # Organization Management
    # ─────────────────────────────────────────────────────────────────────────

    def add_organization(self, org: Organization) -> str:
        """Add an organization to the network."""
        self.organizations[org.id] = org

        # Update indexes
        self._by_country[org.headquarters_country].add(org.id)
        for country in org.operating_countries:
            self._by_country[country].add(org.id)
        for region in org.regions:
            self._by_region[region].add(org.id)
        self._by_type[org.org_type].add(org.id)
        self._by_alignment[org.alignment].add(org.id)

        for industry in org.industries:
            self._by_industry[industry].add(org.id)

        for tag in org.all_tags():
            self._by_tag[tag].add(org.id)

        return org.id

    def create_organization(self, name: str, org_type: OrgType,
                           headquarters: str, **kwargs) -> Organization:
        """Create and add an organization."""
        country = self.countries.get(headquarters)
        regions = {country.region} if country else set()

        org = Organization(
            name=name,
            org_type=org_type,
            headquarters_country=headquarters,
            regions=regions,
            **kwargs
        )
        self.add_organization(org)
        return org

    def get_orgs_by_country(self, country_code: str) -> List[Organization]:
        """Get all organizations in a country."""
        return [self.organizations[oid] for oid in self._by_country.get(country_code, set())]

    def get_orgs_by_region(self, region: Region) -> List[Organization]:
        """Get all organizations in a region."""
        return [self.organizations[oid] for oid in self._by_region.get(region, set())]

    def get_orgs_by_type(self, org_type: OrgType) -> List[Organization]:
        """Get all organizations of a type."""
        return [self.organizations[oid] for oid in self._by_type.get(org_type, set())]

    def get_orgs_by_tag(self, tag: str) -> List[Organization]:
        """Get all organizations with a tag."""
        return [self.organizations[oid] for oid in self._by_tag.get(tag, set())]

    def get_orgs_by_tags(self, tags: Set[str], match_all: bool = False) -> List[Organization]:
        """Get organizations matching tags (any or all)."""
        if not tags:
            return list(self.organizations.values())

        if match_all:
            result_ids = None
            for tag in tags:
                tag_orgs = self._by_tag.get(tag, set())
                if result_ids is None:
                    result_ids = set(tag_orgs)
                else:
                    result_ids &= tag_orgs
            return [self.organizations[oid] for oid in (result_ids or set())]
        else:
            result_ids = set()
            for tag in tags:
                result_ids |= self._by_tag.get(tag, set())
            return [self.organizations[oid] for oid in result_ids]

    # ─────────────────────────────────────────────────────────────────────────
    # Agent Management
    # ─────────────────────────────────────────────────────────────────────────

    def add_agent(self, agent: GlobalAgent) -> str:
        """Add an agent to the network."""
        self.agents[agent.id] = agent

        self._agents_by_org[agent.organization_id].add(agent.id)
        self._agents_by_country[agent.country].add(agent.id)

        for tag in agent.tags:
            self._agents_by_tag[tag].add(agent.id)

        return agent.id

    def create_agent(self, name: str, org_id: str, role: str,
                    level: int = 0, country: str = None, **kwargs) -> GlobalAgent:
        """Create and add an agent."""
        org = self.organizations.get(org_id)
        if not country and org:
            country = org.headquarters_country

        region = None
        if country and country in self.countries:
            region = self.countries[country].region

        agent = GlobalAgent(
            name=name,
            organization_id=org_id,
            role=role,
            level=level,
            country=country or "",
            region=region,
            **kwargs
        )
        self.add_agent(agent)
        return agent

    def populate_organization(self, org_id: str, roles: List[RoleTemplate],
                              scale: float = 1.0) -> List[GlobalAgent]:
        """Populate an organization with agents based on role templates."""
        org = self.organizations.get(org_id)
        if not org:
            return []

        agents = []
        role_agents: Dict[str, List[GlobalAgent]] = defaultdict(list)

        for role in sorted(roles, key=lambda r: r.level):
            count = max(1, int(role.direct_reports * scale)) if role.direct_reports else 1
            if role.level == 0:
                count = 1  # Only one top role

            for i in range(count):
                name = f"{role.name}_{i+1}" if count > 1 else role.name
                agent = self.create_agent(
                    name=f"{org.name}_{name}",
                    org_id=org_id,
                    role=role.name,
                    level=role.level,
                    country=org.headquarters_country,
                    tags=set(role.tags),
                )

                # Set reporting relationship
                if role.reports_to and role.reports_to in role_agents:
                    potential_managers = role_agents[role.reports_to]
                    if potential_managers:
                        manager = random.choice(potential_managers)
                        agent.reports_to = manager.id
                        manager.direct_reports.add(agent.id)

                agents.append(agent)
                role_agents[role.name].append(agent)

        org.member_count = len(agents)
        return agents

    def get_agents_by_org(self, org_id: str) -> List[GlobalAgent]:
        """Get all agents in an organization."""
        return [self.agents[aid] for aid in self._agents_by_org.get(org_id, set())]

    def get_agents_by_country(self, country_code: str) -> List[GlobalAgent]:
        """Get all agents in a country."""
        return [self.agents[aid] for aid in self._agents_by_country.get(country_code, set())]

    def get_agents_by_tag(self, tag: str) -> List[GlobalAgent]:
        """Get all agents with a tag."""
        return [self.agents[aid] for aid in self._agents_by_tag.get(tag, set())]

    # ─────────────────────────────────────────────────────────────────────────
    # Relationship Management
    # ─────────────────────────────────────────────────────────────────────────

    def add_alliance(self, org_id1: str, org_id2: str):
        """Add alliance between organizations."""
        self._alliances[org_id1].add(org_id2)
        self._alliances[org_id2].add(org_id1)

        if org_id1 in self.organizations:
            self.organizations[org_id1].allied_orgs.add(org_id2)
        if org_id2 in self.organizations:
            self.organizations[org_id2].allied_orgs.add(org_id1)

    def add_rivalry(self, org_id1: str, org_id2: str):
        """Add rivalry between organizations."""
        self._rivalries[org_id1].add(org_id2)
        self._rivalries[org_id2].add(org_id1)

        if org_id1 in self.organizations:
            self.organizations[org_id1].rival_orgs.add(org_id2)
        if org_id2 in self.organizations:
            self.organizations[org_id2].rival_orgs.add(org_id1)

    def get_allied_orgs(self, org_id: str) -> List[Organization]:
        """Get allied organizations."""
        return [self.organizations[oid] for oid in self._alliances.get(org_id, set())
                if oid in self.organizations]

    def get_rival_orgs(self, org_id: str) -> List[Organization]:
        """Get rival organizations."""
        return [self.organizations[oid] for oid in self._rivalries.get(org_id, set())
                if oid in self.organizations]

    # ─────────────────────────────────────────────────────────────────────────
    # Bulk Generation
    # ─────────────────────────────────────────────────────────────────────────

    def generate_country_government(self, country_code: str) -> List[Organization]:
        """Generate government structure for a country."""
        country = self.countries.get(country_code)
        if not country:
            return []

        orgs = []

        # Executive branch
        exec_org = self.create_organization(
            name=f"{country.name}_Executive",
            org_type=OrgType.GOVERNMENT_EXECUTIVE,
            headquarters=country_code,
            size=SizeTag.LARGE,
            reach=ReachTag.NATIONAL,
            power=PowerTag.MAJOR_PLAYER,
            influences={InfluenceTag.POLITICAL, InfluenceTag.ECONOMIC},
        )
        orgs.append(exec_org)

        # Ministries/Agencies
        ministries = ["Finance", "Defense", "Foreign", "Interior", "Health", "Education"]
        for ministry in ministries:
            min_org = self.create_organization(
                name=f"{country.name}_{ministry}",
                org_type=OrgType.GOVERNMENT_AGENCY,
                headquarters=country_code,
                size=SizeTag.LARGE,
                reach=ReachTag.NATIONAL,
                parent_org=exec_org.id,
            )
            exec_org.subsidiary_orgs.add(min_org.id)
            orgs.append(min_org)

        # Central bank
        cb = self.create_organization(
            name=f"{country.name}_Central_Bank",
            org_type=OrgType.BANK_CENTRAL,
            headquarters=country_code,
            industries={IndustryTag.FINANCE},
            size=SizeTag.MEDIUM,
            reach=ReachTag.NATIONAL,
            influences={InfluenceTag.ECONOMIC},
        )
        orgs.append(cb)

        # Military
        mil = self.create_organization(
            name=f"{country.name}_Armed_Forces",
            org_type=OrgType.MILITARY_COMMAND,
            headquarters=country_code,
            industries={IndustryTag.DEFENSE},
            size=SizeTag.MEGA,
            reach=ReachTag.NATIONAL,
            influences={InfluenceTag.MILITARY},
        )
        orgs.append(mil)

        # Intelligence
        intel = self.create_organization(
            name=f"{country.name}_Intelligence",
            org_type=OrgType.INTELLIGENCE_AGENCY,
            headquarters=country_code,
            industries={IndustryTag.DEFENSE},
            size=SizeTag.MEDIUM,
            reach=ReachTag.GLOBAL,
            influences={InfluenceTag.INFORMATIONAL},
        )
        orgs.append(intel)

        return orgs

    def generate_global_corporations(self, count: int = 50) -> List[Organization]:
        """Generate global corporations across industries."""
        orgs = []

        industries = list(IndustryTag)
        countries = list(self.countries.keys())

        for i in range(count):
            industry = industries[i % len(industries)]
            hq = random.choice(countries)

            org = self.create_organization(
                name=f"GlobalCorp_{industry.value}_{i+1}",
                org_type=OrgType.CORPORATION_MULTINATIONAL,
                headquarters=hq,
                operating_countries=set(random.sample(countries, min(10, len(countries)))),
                industries={industry},
                size=random.choice([SizeTag.ENTERPRISE, SizeTag.MEGA]),
                reach=ReachTag.GLOBAL,
                power=random.choice([PowerTag.MAJOR_PLAYER, PowerTag.REGIONAL_POWER]),
            )
            orgs.append(org)

        return orgs

    def generate_ngos(self, count: int = 30) -> List[Organization]:
        """Generate international NGOs."""
        orgs = []

        focus_areas = ["Human_Rights", "Environment", "Health", "Education",
                       "Poverty", "Democracy", "Refugees", "Children", "Women"]

        for i in range(count):
            focus = focus_areas[i % len(focus_areas)]
            hq = random.choice(["US", "GB", "CH", "FR", "DE"])

            org = self.create_organization(
                name=f"NGO_{focus}_{i+1}",
                org_type=OrgType.NGO_INTERNATIONAL,
                headquarters=hq,
                size=random.choice([SizeTag.MEDIUM, SizeTag.LARGE]),
                reach=random.choice([ReachTag.REGIONAL, ReachTag.GLOBAL]),
                influences={InfluenceTag.SOCIAL, InfluenceTag.CULTURAL},
                ideologies={IdeologyTag.PROGRESSIVE, IdeologyTag.GLOBALIST},
            )
            orgs.append(org)

        return orgs

    def generate_universities(self, count: int = 40) -> List[Organization]:
        """Generate universities across countries."""
        orgs = []

        countries = ["US", "GB", "DE", "FR", "JP", "CN", "AU", "IN", "SG", "CH"]

        for i in range(count):
            hq = countries[i % len(countries)]

            org = self.create_organization(
                name=f"University_{hq}_{i+1}",
                org_type=OrgType.UNIVERSITY,
                headquarters=hq,
                industries={IndustryTag.EDUCATION},
                size=random.choice([SizeTag.LARGE, SizeTag.ENTERPRISE]),
                reach=ReachTag.NATIONAL,
                influences={InfluenceTag.SCIENTIFIC, InfluenceTag.CULTURAL},
            )
            orgs.append(org)

        return orgs

    def generate_full_global_network(self,
                                     govt_countries: List[str] = None,
                                     corps: int = 50,
                                     ngos: int = 30,
                                     universities: int = 40) -> Dict[str, int]:
        """Generate a complete global network."""
        stats = {"governments": 0, "corporations": 0, "ngos": 0,
                 "universities": 0, "total_orgs": 0, "total_agents": 0}

        # Governments
        govt_countries = govt_countries or ["US", "CN", "DE", "GB", "FR", "JP", "IN", "BR", "RU", "AU"]
        for country in govt_countries:
            orgs = self.generate_country_government(country)
            stats["governments"] += len(orgs)

            # Populate key orgs
            for org in orgs[:3]:  # Top 3 per country
                agents = self.populate_organization(org.id, GOVERNMENT_ROLES, scale=0.3)
                stats["total_agents"] += len(agents)

        # Corporations
        corp_orgs = self.generate_global_corporations(corps)
        stats["corporations"] = len(corp_orgs)
        for org in corp_orgs[:20]:  # Populate top 20
            agents = self.populate_organization(org.id, CORPORATE_ROLES, scale=0.5)
            stats["total_agents"] += len(agents)

        # NGOs
        ngo_orgs = self.generate_ngos(ngos)
        stats["ngos"] = len(ngo_orgs)
        for org in ngo_orgs[:10]:  # Populate top 10
            agents = self.populate_organization(org.id, NGO_ROLES, scale=0.4)
            stats["total_agents"] += len(agents)

        # Universities
        uni_orgs = self.generate_universities(universities)
        stats["universities"] = len(uni_orgs)
        for org in uni_orgs[:15]:  # Populate top 15
            agents = self.populate_organization(org.id, ACADEMIC_ROLES, scale=0.3)
            stats["total_agents"] += len(agents)

        stats["total_orgs"] = len(self.organizations)

        return stats

    # ─────────────────────────────────────────────────────────────────────────
    # Analysis & Export
    # ─────────────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Get network statistics."""
        return {
            "total_organizations": len(self.organizations),
            "total_agents": len(self.agents),
            "countries_covered": len(self._by_country),
            "regions_covered": len(self._by_region),
            "org_types": {t.value: len(ids) for t, ids in self._by_type.items()},
            "industries": {i.value: len(ids) for i, ids in self._by_industry.items()},
            "alignments": {a.value: len(ids) for a, ids in self._by_alignment.items()},
            "alliances": sum(len(allies) for allies in self._alliances.values()) // 2,
            "rivalries": sum(len(rivals) for rivals in self._rivalries.values()) // 2,
        }

    def export_network(self) -> Dict[str, Any]:
        """Export the network to a dictionary."""
        return {
            "organizations": {
                oid: {
                    "id": org.id,
                    "name": org.name,
                    "type": org.org_type.value,
                    "headquarters": org.headquarters_country,
                    "tags": list(org.all_tags()),
                    "member_count": org.member_count,
                    "allied_orgs": list(org.allied_orgs),
                    "rival_orgs": list(org.rival_orgs),
                }
                for oid, org in self.organizations.items()
            },
            "agents": {
                aid: {
                    "id": agent.id,
                    "name": agent.name,
                    "org_id": agent.organization_id,
                    "role": agent.role,
                    "level": agent.level,
                    "country": agent.country,
                    "tags": list(agent.tags),
                }
                for aid, agent in self.agents.items()
            },
            "stats": self.get_stats(),
        }
