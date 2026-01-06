"""
Geospatial coordinate systems and Military Grid Reference System (MGRS).

Provides:
- Latitude/Longitude coordinates with precision
- MGRS (Military Grid Reference System) conversion
- UTM zone calculations
- Geospatial indexing and queries
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Set, Optional, Tuple, Any


# ═══════════════════════════════════════════════════════════════════════════════
# COORDINATE SYSTEMS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Coordinates:
    """Geographic coordinates with metadata."""
    latitude: float  # -90 to 90
    longitude: float  # -180 to 180
    altitude_m: float = 0.0  # meters above sea level
    precision_m: float = 1000.0  # coordinate precision in meters
    datum: str = "WGS84"  # coordinate reference system

    def __post_init__(self):
        """Validate coordinates."""
        if not -90 <= self.latitude <= 90:
            raise ValueError(f"Latitude must be -90 to 90, got {self.latitude}")
        if not -180 <= self.longitude <= 180:
            raise ValueError(f"Longitude must be -180 to 180, got {self.longitude}")

    def to_dms(self) -> Tuple[str, str]:
        """Convert to degrees-minutes-seconds format."""
        def decimal_to_dms(decimal: float, is_lat: bool) -> str:
            direction = ""
            if is_lat:
                direction = "N" if decimal >= 0 else "S"
            else:
                direction = "E" if decimal >= 0 else "W"

            decimal = abs(decimal)
            degrees = int(decimal)
            minutes = int((decimal - degrees) * 60)
            seconds = ((decimal - degrees) * 60 - minutes) * 60

            return f"{degrees}°{minutes}'{seconds:.2f}\"{direction}"

        return (decimal_to_dms(self.latitude, True),
                decimal_to_dms(self.longitude, False))

    def distance_to(self, other: "Coordinates") -> float:
        """Calculate distance to another point in kilometers (Haversine formula)."""
        R = 6371  # Earth's radius in km

        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))

        return R * c

    def bearing_to(self, other: "Coordinates") -> float:
        """Calculate bearing to another point in degrees."""
        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)

        dlon = lon2 - lon1

        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

        bearing = math.atan2(x, y)
        return (math.degrees(bearing) + 360) % 360


# ═══════════════════════════════════════════════════════════════════════════════
# UTM ZONES
# ═══════════════════════════════════════════════════════════════════════════════

# UTM zone letters (C-X, excluding I and O)
UTM_ZONE_LETTERS = "CDEFGHJKLMNPQRSTUVWX"

def get_utm_zone(lat: float, lon: float) -> Tuple[int, str]:
    """Get UTM zone number and letter for coordinates."""
    # Zone number (1-60)
    zone_number = int((lon + 180) / 6) + 1

    # Special cases for Norway and Svalbard
    if 56 <= lat < 64 and 3 <= lon < 12:
        zone_number = 32
    elif 72 <= lat < 84:
        if 0 <= lon < 9:
            zone_number = 31
        elif 9 <= lon < 21:
            zone_number = 33
        elif 21 <= lon < 33:
            zone_number = 35
        elif 33 <= lon < 42:
            zone_number = 37

    # Zone letter
    if -80 <= lat < 84:
        letter_idx = int((lat + 80) / 8)
        if letter_idx >= len(UTM_ZONE_LETTERS):
            letter_idx = len(UTM_ZONE_LETTERS) - 1
        zone_letter = UTM_ZONE_LETTERS[letter_idx]
    else:
        zone_letter = "Z" if lat >= 84 else "A"

    return zone_number, zone_letter


def lat_lon_to_utm(lat: float, lon: float) -> Tuple[int, str, float, float]:
    """Convert lat/lon to UTM coordinates."""
    zone_number, zone_letter = get_utm_zone(lat, lon)

    # WGS84 ellipsoid parameters
    a = 6378137.0  # semi-major axis
    f = 1 / 298.257223563  # flattening
    k0 = 0.9996  # scale factor

    e = math.sqrt(2*f - f**2)  # eccentricity
    e2 = e**2 / (1 - e**2)  # second eccentricity squared

    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    # Central meridian
    lon0 = math.radians((zone_number - 1) * 6 - 180 + 3)

    N = a / math.sqrt(1 - e**2 * math.sin(lat_rad)**2)
    T = math.tan(lat_rad)**2
    C = e2 * math.cos(lat_rad)**2
    A = (lon_rad - lon0) * math.cos(lat_rad)

    # Meridional arc
    M = a * ((1 - e**2/4 - 3*e**4/64 - 5*e**6/256) * lat_rad
             - (3*e**2/8 + 3*e**4/32 + 45*e**6/1024) * math.sin(2*lat_rad)
             + (15*e**4/256 + 45*e**6/1024) * math.sin(4*lat_rad)
             - (35*e**6/3072) * math.sin(6*lat_rad))

    # Easting
    easting = k0 * N * (A + (1-T+C)*A**3/6
                        + (5-18*T+T**2+72*C-58*e2)*A**5/120) + 500000

    # Northing
    northing = k0 * (M + N*math.tan(lat_rad)*(A**2/2
                     + (5-T+9*C+4*C**2)*A**4/24
                     + (61-58*T+T**2+600*C-330*e2)*A**6/720))

    if lat < 0:
        northing += 10000000  # Southern hemisphere offset

    return zone_number, zone_letter, easting, northing


# ═══════════════════════════════════════════════════════════════════════════════
# MILITARY GRID REFERENCE SYSTEM (MGRS)
# ═══════════════════════════════════════════════════════════════════════════════

# 100km grid square letters
MGRS_COL_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # No I or O
MGRS_ROW_LETTERS = "ABCDEFGHJKLMNPQRSTUV"  # 20 letters, no I or O

def get_mgrs_letters(zone_number: int, easting: float, northing: float) -> Tuple[str, str]:
    """Get MGRS 100km grid square letters."""
    # Column letter (based on easting)
    col_idx = int(easting / 100000) - 1
    # Adjust for zone set (zones are in sets of 6)
    set_number = (zone_number - 1) % 6
    col_idx = (col_idx + set_number * 8) % 24
    col_letter = MGRS_COL_LETTERS[col_idx]

    # Row letter (based on northing)
    row_idx = int(northing / 100000) % 20
    # Odd zones use different starting row
    if zone_number % 2 == 0:
        row_idx = (row_idx + 5) % 20
    row_letter = MGRS_ROW_LETTERS[row_idx]

    return col_letter, row_letter


@dataclass
class MGRSCoordinate:
    """Military Grid Reference System coordinate."""
    zone_number: int  # 1-60
    zone_letter: str  # C-X
    column_letter: str  # A-Z (no I, O)
    row_letter: str  # A-V (no I, O)
    easting: int  # meters within 100km square
    northing: int  # meters within 100km square
    precision: int = 1  # 1=10km, 2=1km, 3=100m, 4=10m, 5=1m

    def __str__(self) -> str:
        """Format as standard MGRS string."""
        # Determine digits based on precision
        digits = self.precision
        e_str = str(self.easting).zfill(5)[:digits]
        n_str = str(self.northing).zfill(5)[:digits]

        return f"{self.zone_number}{self.zone_letter}{self.column_letter}{self.row_letter}{e_str}{n_str}"

    def to_short(self) -> str:
        """Format as abbreviated MGRS (grid zone + square only)."""
        return f"{self.zone_number}{self.zone_letter}{self.column_letter}{self.row_letter}"

    @classmethod
    def from_coordinates(cls, coords: Coordinates, precision: int = 5) -> "MGRSCoordinate":
        """Create MGRS from lat/lon coordinates."""
        zone_num, zone_let, easting, northing = lat_lon_to_utm(
            coords.latitude, coords.longitude
        )
        col_let, row_let = get_mgrs_letters(zone_num, easting, northing)

        # Get easting/northing within 100km square
        e_100k = int(easting) % 100000
        n_100k = int(northing) % 100000

        return cls(
            zone_number=zone_num,
            zone_letter=zone_let,
            column_letter=col_let,
            row_letter=row_let,
            easting=e_100k,
            northing=n_100k,
            precision=precision
        )


def coords_to_mgrs(lat: float, lon: float, precision: int = 5) -> str:
    """Convert lat/lon to MGRS string."""
    coords = Coordinates(lat, lon)
    mgrs = MGRSCoordinate.from_coordinates(coords, precision)
    return str(mgrs)


# ═══════════════════════════════════════════════════════════════════════════════
# COUNTRY COORDINATES DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

# Capital city coordinates for all countries (lat, lon)
COUNTRY_COORDINATES: Dict[str, Tuple[float, float]] = {
    # North America
    "US": (38.8951, -77.0364),  # Washington, D.C.
    "CA": (45.4215, -75.6972),  # Ottawa
    "MX": (19.4326, -99.1332),  # Mexico City

    # Central America & Caribbean
    "GT": (14.6349, -90.5069),  # Guatemala City
    "BZ": (17.2510, -88.7590),  # Belmopan
    "HN": (14.0723, -87.1921),  # Tegucigalpa
    "SV": (13.6929, -89.2182),  # San Salvador
    "NI": (12.1150, -86.2362),  # Managua
    "CR": (9.9281, -84.0907),   # San José
    "PA": (8.9824, -79.5199),   # Panama City
    "CU": (23.1136, -82.3666),  # Havana
    "JM": (17.9714, -76.7936),  # Kingston
    "HT": (18.5944, -72.3074),  # Port-au-Prince
    "DO": (18.4861, -69.9312),  # Santo Domingo
    "TT": (10.6596, -61.5086),  # Port of Spain
    "BS": (25.0480, -77.3554),  # Nassau
    "BB": (13.1132, -59.5988),  # Bridgetown
    "LC": (14.0101, -60.9870),  # Castries
    "GD": (12.0561, -61.7488),  # St. George's
    "VC": (13.1600, -61.2248),  # Kingstown
    "AG": (17.1175, -61.8456),  # St. John's
    "DM": (15.3092, -61.3794),  # Roseau
    "KN": (17.3026, -62.7177),  # Basseterre

    # South America
    "BR": (-15.7975, -47.8919),  # Brasília
    "AR": (-34.6037, -58.3816),  # Buenos Aires
    "CO": (4.7110, -74.0721),    # Bogotá
    "CL": (-33.4489, -70.6693),  # Santiago
    "PE": (-12.0464, -77.0428),  # Lima
    "VE": (10.4806, -66.9036),   # Caracas
    "EC": (-0.1807, -78.4678),   # Quito
    "BO": (-16.4897, -68.1193),  # La Paz
    "PY": (-25.2637, -57.5759),  # Asunción
    "UY": (-34.9011, -56.1645),  # Montevideo
    "GY": (6.8013, -58.1551),    # Georgetown
    "SR": (5.8520, -55.2038),    # Paramaribo

    # Western Europe
    "GB": (51.5074, -0.1278),    # London
    "FR": (48.8566, 2.3522),     # Paris
    "DE": (52.5200, 13.4050),    # Berlin
    "IT": (41.9028, 12.4964),    # Rome
    "ES": (40.4168, -3.7038),    # Madrid
    "NL": (52.3676, 4.9041),     # Amsterdam
    "BE": (50.8503, 4.3517),     # Brussels
    "PT": (38.7223, -9.1393),    # Lisbon
    "IE": (53.3498, -6.2603),    # Dublin
    "AT": (48.2082, 16.3738),    # Vienna
    "CH": (46.9480, 7.4474),     # Bern
    "LU": (49.6116, 6.1319),     # Luxembourg City

    # Northern Europe
    "SE": (59.3293, 18.0686),    # Stockholm
    "NO": (59.9139, 10.7522),    # Oslo
    "DK": (55.6761, 12.5683),    # Copenhagen
    "FI": (60.1699, 24.9384),    # Helsinki
    "IS": (64.1466, -21.9426),   # Reykjavík
    "EE": (59.4370, 24.7536),    # Tallinn
    "LV": (56.9496, 24.1052),    # Riga
    "LT": (54.6872, 25.2797),    # Vilnius
    "GL": (64.1814, -51.6941),   # Nuuk
    "FO": (62.0079, -6.7904),    # Tórshavn

    # Central Europe
    "PL": (52.2297, 21.0122),    # Warsaw
    "CZ": (50.0755, 14.4378),    # Prague
    "SK": (48.1486, 17.1077),    # Bratislava
    "HU": (47.4979, 19.0402),    # Budapest
    "SI": (46.0569, 14.5058),    # Ljubljana
    "LI": (47.1410, 9.5209),     # Vaduz
    "MC": (43.7384, 7.4246),     # Monaco

    # Eastern Europe
    "RU": (55.7558, 37.6173),    # Moscow
    "UA": (50.4501, 30.5234),    # Kyiv
    "BY": (53.9045, 27.5615),    # Minsk
    "MD": (47.0105, 28.8638),    # Chișinău
    "RO": (44.4268, 26.1025),    # Bucharest
    "BG": (42.6977, 23.3219),    # Sofia
    "GE": (41.7151, 44.8271),    # Tbilisi
    "AM": (40.1872, 44.5152),    # Yerevan
    "AZ": (40.4093, 49.8671),    # Baku
    "KZ": (51.1605, 71.4704),    # Astana

    # Balkans
    "HR": (45.8150, 15.9819),    # Zagreb
    "RS": (44.7866, 20.4489),    # Belgrade
    "BA": (43.8563, 18.4131),    # Sarajevo
    "ME": (42.4304, 19.2594),    # Podgorica
    "MK": (41.9973, 21.4280),    # Skopje
    "AL": (41.3275, 19.8187),    # Tirana
    "XK": (42.6629, 21.1655),    # Pristina
    "GR": (37.9838, 23.7275),    # Athens
    "CY": (35.1856, 33.3823),    # Nicosia
    "MT": (35.8989, 14.5146),    # Valletta

    # Middle East
    "TR": (39.9334, 32.8597),    # Ankara
    "IL": (31.7683, 35.2137),    # Jerusalem
    "SA": (24.7136, 46.6753),    # Riyadh
    "AE": (24.4539, 54.3773),    # Abu Dhabi
    "QA": (25.2854, 51.5310),    # Doha
    "KW": (29.3759, 47.9774),    # Kuwait City
    "BH": (26.2285, 50.5860),    # Manama
    "OM": (23.5880, 58.3829),    # Muscat
    "YE": (15.3694, 44.1910),    # Sana'a
    "IQ": (33.3152, 44.3661),    # Baghdad
    "IR": (35.6892, 51.3890),    # Tehran
    "SY": (33.5138, 36.2765),    # Damascus
    "LB": (33.8938, 35.5018),    # Beirut
    "JO": (31.9454, 35.9284),    # Amman
    "PS": (31.9028, 35.2064),    # Ramallah
    "AF": (34.5553, 69.2075),    # Kabul
    "PK": (33.6844, 73.0479),    # Islamabad

    # Central Asia
    "UZ": (41.2995, 69.2401),    # Tashkent
    "TM": (37.9601, 58.3261),    # Ashgabat
    "TJ": (38.5598, 68.7739),    # Dushanbe
    "KG": (42.8746, 74.5698),    # Bishkek
    "MN": (47.8864, 106.9057),   # Ulaanbaatar

    # South Asia
    "IN": (28.6139, 77.2090),    # New Delhi
    "BD": (23.8103, 90.4125),    # Dhaka
    "LK": (6.9271, 79.8612),     # Colombo
    "NP": (27.7172, 85.3240),    # Kathmandu
    "BT": (27.4728, 89.6390),    # Thimphu
    "MV": (4.1755, 73.5093),     # Malé
    "MM": (19.7633, 96.0785),    # Naypyidaw

    # East Asia
    "CN": (39.9042, 116.4074),   # Beijing
    "JP": (35.6762, 139.6503),   # Tokyo
    "KR": (37.5665, 126.9780),   # Seoul
    "KP": (39.0392, 125.7625),   # Pyongyang
    "TW": (25.0330, 121.5654),   # Taipei
    "HK": (22.3193, 114.1694),   # Hong Kong

    # Southeast Asia
    "ID": (-6.2088, 106.8456),   # Jakarta
    "TH": (13.7563, 100.5018),   # Bangkok
    "VN": (21.0278, 105.8342),   # Hanoi
    "PH": (14.5995, 120.9842),   # Manila
    "MY": (3.1390, 101.6869),    # Kuala Lumpur
    "SG": (1.3521, 103.8198),    # Singapore
    "KH": (11.5564, 104.9282),   # Phnom Penh
    "LA": (17.9757, 102.6331),   # Vientiane
    "BN": (4.9031, 114.9398),    # Bandar Seri Begawan
    "TL": (-8.5569, 125.5603),   # Dili

    # Oceania
    "AU": (-35.2809, 149.1300),  # Canberra
    "NZ": (-41.2865, 174.7762),  # Wellington
    "PG": (-9.4438, 147.1803),   # Port Moresby
    "FJ": (-18.1416, 178.4419),  # Suva
    "SB": (-9.4456, 159.9729),   # Honiara
    "VU": (-17.7334, 168.3220),  # Port Vila
    "WS": (-13.8333, -171.7500), # Apia
    "TO": (-21.2114, -175.1998), # Nuku'alofa
    "FM": (6.9248, 158.1610),    # Palikir
    "KI": (1.3382, 172.9784),    # Tarawa
    "MH": (7.1164, 171.1858),    # Majuro
    "PW": (7.5150, 134.5825),    # Ngerulmud
    "NR": (-0.5228, 166.9315),   # Yaren
    "TV": (-8.5167, 179.2167),   # Funafuti

    # North Africa
    "EG": (30.0444, 31.2357),    # Cairo
    "LY": (32.8872, 13.1913),    # Tripoli
    "TN": (36.8065, 10.1815),    # Tunis
    "DZ": (36.7538, 3.0588),     # Algiers
    "MA": (34.0209, -6.8416),    # Rabat
    "SD": (15.5007, 32.5599),    # Khartoum

    # West Africa
    "NG": (9.0765, 7.3986),      # Abuja
    "GH": (5.6037, -0.1870),     # Accra
    "CI": (6.8276, -5.2893),     # Yamoussoukro
    "SN": (14.7167, -17.4677),   # Dakar
    "ML": (12.6392, -8.0029),    # Bamako
    "BF": (12.3714, -1.5197),    # Ouagadougou
    "NE": (13.5137, 2.1098),     # Niamey
    "GN": (9.6412, -13.5784),    # Conakry
    "BJ": (6.4969, 2.6289),      # Porto-Novo
    "TG": (6.1256, 1.2254),      # Lomé
    "SL": (8.4657, -13.2317),    # Freetown
    "LR": (6.2907, -10.7605),    # Monrovia
    "MR": (18.0735, -15.9582),   # Nouakchott
    "GM": (13.4549, -16.5790),   # Banjul
    "GW": (11.8636, -15.5977),   # Bissau
    "CV": (14.9315, -23.5087),   # Praia

    # Central Africa
    "CD": (-4.4419, 15.2663),    # Kinshasa
    "AO": (-8.8390, 13.2894),    # Luanda
    "CM": (3.8480, 11.5021),     # Yaoundé
    "TD": (12.1348, 15.0557),    # N'Djamena
    "CG": (-4.2634, 15.2429),    # Brazzaville
    "GA": (0.4162, 9.4673),      # Libreville
    "GQ": (3.7523, 8.7742),      # Malabo
    "CF": (4.3947, 18.5582),     # Bangui
    "ST": (0.3365, 6.7273),      # São Tomé

    # East Africa
    "ET": (9.0320, 38.7469),     # Addis Ababa
    "KE": (-1.2921, 36.8219),    # Nairobi
    "TZ": (-6.7924, 39.2083),    # Dodoma
    "UG": (0.3476, 32.5825),     # Kampala
    "RW": (-1.9403, 29.8739),    # Kigali
    "BI": (-3.3731, 29.3644),    # Gitega
    "SS": (4.8594, 31.5713),     # Juba
    "SO": (2.0469, 45.3182),     # Mogadishu
    "ER": (15.3229, 38.9251),    # Asmara
    "DJ": (11.5721, 43.1456),    # Djibouti City
    "MU": (-20.1609, 57.5012),   # Port Louis
    "SC": (-4.6191, 55.4513),    # Victoria
    "KM": (-11.7172, 43.2473),   # Moroni
    "MG": (-18.8792, 47.5079),   # Antananarivo

    # Southern Africa
    "ZA": (-25.7479, 28.2293),   # Pretoria
    "ZW": (-17.8252, 31.0335),   # Harare
    "ZM": (-15.3875, 28.3228),   # Lusaka
    "MW": (-13.9626, 33.7741),   # Lilongwe
    "MZ": (-25.9692, 32.5732),   # Maputo
    "BW": (-24.6282, 25.9231),   # Gaborone
    "NA": (-22.5609, 17.0658),   # Windhoek
    "SZ": (-26.3054, 31.1367),   # Mbabane
    "LS": (-29.3151, 27.4869),   # Maseru

    # Antarctica
    "AQ": (-90.0000, 0.0000),    # South Pole
}


def get_country_coordinates(country_code: str) -> Optional[Coordinates]:
    """Get coordinates for a country's capital."""
    if country_code in COUNTRY_COORDINATES:
        lat, lon = COUNTRY_COORDINATES[country_code]
        return Coordinates(lat, lon, precision_m=100.0)
    return None


def get_country_mgrs(country_code: str, precision: int = 3) -> Optional[str]:
    """Get MGRS reference for a country's capital."""
    coords = get_country_coordinates(country_code)
    if coords:
        return coords_to_mgrs(coords.latitude, coords.longitude, precision)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# GEOSPATIAL BOUNDING BOXES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BoundingBox:
    """Geographic bounding box."""
    north: float
    south: float
    east: float
    west: float

    def contains(self, lat: float, lon: float) -> bool:
        """Check if point is within bounding box."""
        return (self.south <= lat <= self.north and
                self.west <= lon <= self.east)

    def overlaps(self, other: "BoundingBox") -> bool:
        """Check if bounding boxes overlap."""
        return not (self.east < other.west or
                   self.west > other.east or
                   self.north < other.south or
                   self.south > other.north)

    def center(self) -> Coordinates:
        """Get center point of bounding box."""
        return Coordinates(
            (self.north + self.south) / 2,
            (self.east + self.west) / 2
        )

    @classmethod
    def from_center_radius(cls, center: Coordinates, radius_km: float) -> "BoundingBox":
        """Create bounding box from center point and radius."""
        # Approximate degrees per km
        lat_per_km = 1 / 111.0
        lon_per_km = 1 / (111.0 * math.cos(math.radians(center.latitude)))

        dlat = radius_km * lat_per_km
        dlon = radius_km * lon_per_km

        return cls(
            north=center.latitude + dlat,
            south=center.latitude - dlat,
            east=center.longitude + dlon,
            west=center.longitude - dlon
        )


# Country approximate bounding boxes
COUNTRY_BOUNDS: Dict[str, BoundingBox] = {
    "US": BoundingBox(49.38, 24.52, -66.95, -124.77),
    "CN": BoundingBox(53.56, 18.16, 134.77, 73.50),
    "RU": BoundingBox(81.86, 41.19, 180.0, 19.64),
    "CA": BoundingBox(83.11, 41.68, -52.62, -141.00),
    "BR": BoundingBox(5.27, -33.75, -34.79, -73.99),
    "AU": BoundingBox(-10.06, -43.64, 153.64, 113.16),
    "IN": BoundingBox(35.50, 6.75, 97.40, 68.11),
    "DE": BoundingBox(55.06, 47.27, 15.04, 5.87),
    "FR": BoundingBox(51.09, 41.36, 9.56, -5.14),
    "GB": BoundingBox(60.86, 49.87, 1.76, -8.65),
    "JP": BoundingBox(45.52, 24.25, 145.82, 122.93),
}
