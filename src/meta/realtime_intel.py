"""
Real-Time Intelligence System.

Full-spectrum 15-minute GDELT snapshots with per-country intelligence aggregation.
Provides real-time situational awareness and query capabilities for any country.
"""

from __future__ import annotations
import asyncio
import aiohttp
import csv
import io
import json
import zipfile
import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Set, Optional, Tuple, Any, Callable
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# INTELLIGENCE EVENT
# ═══════════════════════════════════════════════════════════════════════════════

class ThreatLevel(str, Enum):
    """Threat assessment levels."""
    CRITICAL = "critical"      # Immediate threat, mass violence
    HIGH = "high"              # Significant conflict, military action
    ELEVATED = "elevated"      # Tensions, protests, coercion
    GUARDED = "guarded"        # Minor disputes, verbal conflicts
    LOW = "low"                # Normal diplomatic activity


class EventCategory(str, Enum):
    """High-level event categories."""
    COOPERATION = "cooperation"
    DIPLOMACY = "diplomacy"
    VERBAL_CONFLICT = "verbal_conflict"
    MATERIAL_CONFLICT = "material_conflict"
    HUMANITARIAN = "humanitarian"
    MILITARY = "military"
    ECONOMIC = "economic"
    PROTEST = "protest"


@dataclass
class IntelEvent:
    """Processed intelligence event."""
    event_id: str
    timestamp: datetime

    # Actors
    actor1_code: str
    actor1_name: str
    actor1_country: str
    actor1_type: str

    actor2_code: str
    actor2_name: str
    actor2_country: str
    actor2_type: str

    # Event details
    event_code: str
    event_root: str
    event_description: str
    category: EventCategory

    # Metrics
    quad_class: int
    goldstein_scale: float
    num_mentions: int
    num_sources: int
    num_articles: int
    avg_tone: float

    # Location
    location_name: str
    location_country: str
    latitude: float
    longitude: float
    mgrs: str

    # Assessment
    confidence_score: float
    admiralty_rating: str
    threat_contribution: float

    # Source
    source_url: str

    def is_conflict(self) -> bool:
        return self.quad_class in (3, 4)

    def is_material(self) -> bool:
        return self.quad_class in (2, 4)

    def is_severe(self) -> bool:
        return self.goldstein_scale < -5


# ═══════════════════════════════════════════════════════════════════════════════
# COUNTRY INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CountryIntelligence:
    """Aggregated intelligence for a single country."""
    country_code: str
    country_name: str
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Event counts
    total_events: int = 0
    events_as_actor1: int = 0
    events_as_actor2: int = 0

    # Classification
    cooperative_events: int = 0
    conflictual_events: int = 0
    material_actions: int = 0
    verbal_actions: int = 0

    # Threat metrics
    threat_level: ThreatLevel = ThreatLevel.LOW
    threat_score: float = 0.0  # 0-100
    conflict_intensity: float = 0.0

    # Sentiment
    avg_tone: float = 0.0
    tone_trend: str = "stable"  # improving, stable, deteriorating

    # Goldstein aggregate
    avg_goldstein: float = 0.0
    min_goldstein: float = 0.0
    max_goldstein: float = 0.0

    # Key relationships
    top_partners: List[Tuple[str, int, float]] = field(default_factory=list)  # (country, events, avg_tone)
    top_adversaries: List[Tuple[str, int, float]] = field(default_factory=list)

    # Event type breakdown
    event_types: Dict[str, int] = field(default_factory=dict)

    # Geographic hotspots (MGRS zones with high activity)
    hotspot_zones: List[Tuple[str, int]] = field(default_factory=list)

    # Recent significant events
    significant_events: List[IntelEvent] = field(default_factory=list)

    # Confidence
    avg_confidence: float = 0.0
    source_diversity: int = 0  # Number of unique sources

    # Time series (hourly aggregates)
    hourly_event_counts: List[Tuple[datetime, int]] = field(default_factory=list)
    hourly_tone: List[Tuple[datetime, float]] = field(default_factory=list)

    def to_summary(self) -> str:
        """Generate text summary."""
        lines = [
            f"═══ {self.country_name} ({self.country_code}) Intelligence Summary ═══",
            f"Last Updated: {self.last_updated.strftime('%Y-%m-%d %H:%M UTC')}",
            f"",
            f"THREAT ASSESSMENT: {self.threat_level.value.upper()} (Score: {self.threat_score:.1f}/100)",
            f"",
            f"EVENT STATISTICS:",
            f"  Total Events: {self.total_events:,}",
            f"  As Primary Actor: {self.events_as_actor1:,}",
            f"  As Secondary Actor: {self.events_as_actor2:,}",
            f"  Cooperative: {self.cooperative_events:,} ({100*self.cooperative_events/max(1,self.total_events):.1f}%)",
            f"  Conflictual: {self.conflictual_events:,} ({100*self.conflictual_events/max(1,self.total_events):.1f}%)",
            f"",
            f"SENTIMENT:",
            f"  Average Tone: {self.avg_tone:+.2f} ({self.tone_trend})",
            f"  Goldstein Range: {self.min_goldstein:+.1f} to {self.max_goldstein:+.1f}",
            f"",
        ]

        if self.top_partners:
            lines.append("TOP COOPERATIVE PARTNERS:")
            for country, events, tone in self.top_partners[:5]:
                lines.append(f"  {country}: {events} events (tone: {tone:+.1f})")
            lines.append("")

        if self.top_adversaries:
            lines.append("TOP CONFLICT RELATIONSHIPS:")
            for country, events, tone in self.top_adversaries[:5]:
                lines.append(f"  {country}: {events} events (tone: {tone:+.1f})")
            lines.append("")

        if self.event_types:
            lines.append("EVENT TYPES:")
            for etype, count in sorted(self.event_types.items(), key=lambda x: -x[1])[:8]:
                lines.append(f"  {etype}: {count}")
            lines.append("")

        if self.significant_events:
            lines.append("RECENT SIGNIFICANT EVENTS:")
            for event in self.significant_events[:5]:
                lines.append(f"  • {event.actor1_name} → {event.actor2_name}: {event.event_description}")
                lines.append(f"    [{event.admiralty_rating}] {event.num_mentions} mentions, tone: {event.avg_tone:+.1f}")
            lines.append("")

        lines.append(f"Data Confidence: {self.avg_confidence:.2f} | Sources: {self.source_diversity}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# GDELT SNAPSHOT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GDELTSnapshot:
    """A single 15-minute GDELT snapshot."""
    timestamp: datetime
    events: List[IntelEvent] = field(default_factory=list)
    fetch_duration_ms: int = 0
    raw_event_count: int = 0

    @property
    def event_count(self) -> int:
        return len(self.events)


# ═══════════════════════════════════════════════════════════════════════════════
# REAL-TIME INTELLIGENCE SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

# CAMEO event code descriptions
EVENT_DESCRIPTIONS = {
    "01": "Public Statement", "010": "Make Statement", "011": "Decline Comment",
    "012": "Make Pessimistic Comment", "013": "Make Optimistic Comment",
    "014": "Consider Policy Option", "015": "Acknowledge Responsibility",
    "016": "Deny Responsibility", "017": "Engage in Negotiation",
    "018": "Consult", "019": "Mediate",
    "02": "Appeal", "020": "Make Appeal", "021": "Appeal for Cooperation",
    "022": "Appeal for Diplomatic Cooperation", "023": "Appeal for Aid",
    "024": "Appeal for Political Reform", "025": "Appeal for Rights",
    "026": "Appeal for Policy Change", "027": "Appeal for Ceasefire",
    "028": "Appeal for Peace",
    "03": "Express Intent to Cooperate", "030": "Express Intent to Cooperate",
    "031": "Express Intent to Engage", "032": "Express Intent to Provide Aid",
    "033": "Express Intent to Meet", "034": "Express Intent to Settle",
    "035": "Express Intent to Collaborate", "036": "Express Intent to Institute Reform",
    "04": "Consult", "040": "Consult", "041": "Discuss", "042": "Make Visit",
    "043": "Host Visit", "044": "Meet at Third Location", "045": "Summit",
    "046": "Mediate/Negotiate",
    "05": "Diplomatic Cooperation", "050": "Engage in Diplomatic Cooperation",
    "051": "Praise", "052": "Rally Support", "053": "Grant Access",
    "054": "Sign Agreement", "055": "Apologize", "056": "Forgive",
    "057": "Call for Cooperation",
    "06": "Material Cooperation", "060": "Provide Material Cooperation",
    "061": "Cooperate Economically", "062": "Provide Military Aid",
    "063": "Provide Humanitarian Aid", "064": "Provide Economic Aid",
    "07": "Provide Aid", "070": "Provide Aid", "071": "Provide Economic Aid",
    "072": "Provide Military Aid", "073": "Provide Humanitarian Aid",
    "074": "Provide Refuge", "075": "Grant Political Status",
    "08": "Yield", "080": "Yield", "081": "Yield Position", "082": "Ease Restrictions",
    "083": "Release Prisoners", "084": "Yield Territory", "085": "Yield Authority",
    "086": "Allow Inspection",
    "09": "Investigate", "090": "Investigate", "091": "Investigate Crime",
    "092": "Investigate Human Rights", "093": "Investigate Corruption",
    "10": "Demand", "100": "Demand", "101": "Demand Information",
    "102": "Demand Policy Change", "103": "Demand Rights", "104": "Demand Ceasefire",
    "105": "Demand Access", "106": "Demand Release", "107": "Demand Apology",
    "11": "Disapprove", "110": "Disapprove", "111": "Criticize", "112": "Denounce",
    "113": "Accuse", "114": "Rally Opposition", "115": "File Lawsuit",
    "116": "Find Guilty",
    "12": "Reject", "120": "Reject", "121": "Reject Cooperation",
    "122": "Reject Request", "123": "Reject Material Aid", "124": "Reject Proposal",
    "125": "Refuse to Release", "126": "Prevent Participation", "127": "Expel",
    "128": "Ban Institution",
    "13": "Threaten", "130": "Threaten", "131": "Threaten Non-Force",
    "132": "Threaten Economic", "133": "Threaten Political",
    "134": "Threaten Military", "135": "Threaten Attack", "136": "Threaten War",
    "137": "Threaten Nuclear", "138": "Threaten Terrorism",
    "14": "Protest", "140": "Protest", "141": "Demonstrate", "142": "Hunger Strike",
    "143": "Strike", "144": "Boycott", "145": "Civil Disobedience",
    "15": "Exhibit Force", "150": "Exhibit Force", "151": "Increase Force Posture",
    "152": "Mobilize", "153": "Alert", "154": "Show of Force",
    "155": "Military Exercise",
    "16": "Reduce Relations", "160": "Reduce Relations", "161": "Reduce Diplomatic",
    "162": "Reduce Economic", "163": "Impose Sanctions", "164": "Halt Aid",
    "165": "Halt Negotiations", "166": "Expel Personnel",
    "17": "Coerce", "170": "Coerce", "171": "Seize Property", "172": "Confiscate",
    "173": "Kidnap", "174": "Arrest", "175": "Assassinate",
    "18": "Assault", "180": "Assault", "181": "Physical Assault",
    "182": "Sexual Assault", "183": "Torture", "184": "Kill",
    "185": "Mass Killing",
    "19": "Fight", "190": "Fight", "191": "Clash", "192": "Military Engagement",
    "193": "Armed Clash", "194": "Use Artillery",
    "195": "Use Heavy Weapons", "196": "Bombing",
    "20": "Mass Violence", "200": "Mass Violence", "201": "Mass Killing",
    "202": "Genocide", "203": "Ethnic Cleansing",
}


class RealTimeIntelligence:
    """
    Full-spectrum real-time intelligence system.

    Maintains 15-minute GDELT snapshots and provides per-country intelligence.
    """

    def __init__(self, data_dir: str = ".intel_data", max_history_hours: int = 240):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.max_history_hours = max_history_hours  # 10 days default

        # Snapshot storage
        self.snapshots: Dict[str, GDELTSnapshot] = {}  # key = timestamp string

        # Country intelligence cache
        self.country_intel: Dict[str, CountryIntelligence] = {}

        # Global aggregates
        self.total_events: int = 0
        self.global_avg_tone: float = 0.0
        self.global_threat_level: ThreatLevel = ThreatLevel.LOW

        # Fetch tracking
        self.last_fetch: Optional[datetime] = None
        self.fetch_errors: List[Tuple[datetime, str]] = []

        # Country name mapping
        self._load_country_names()

    def _load_country_names(self):
        """Load country code to name mapping."""
        try:
            from .global_structures import COUNTRIES
            self.country_names = {code: c.name for code, c in COUNTRIES.items()}
        except ImportError:
            self.country_names = {}

    def _get_country_name(self, code: str) -> str:
        """Get country name from code."""
        return self.country_names.get(code, code)

    def _parse_event(self, row: List[str], timestamp: datetime) -> Optional[IntelEvent]:
        """Parse a GDELT CSV row into an IntelEvent."""
        try:
            if len(row) < 58:
                return None

            # Safe value extraction
            def safe_int(val: str, default: int = 0) -> int:
                try:
                    return int(val) if val and val.strip().lstrip('-').isdigit() else default
                except:
                    return default

            def safe_float(val: str, default: float = 0.0) -> float:
                try:
                    return float(val) if val and val.strip() else default
                except:
                    return default

            event_code = row[26] if len(row) > 26 else ""
            event_root = row[28] if len(row) > 28 else (event_code[:2] if event_code else "")
            quad_class = safe_int(row[29]) if len(row) > 29 else 0
            goldstein = safe_float(row[30])

            # Determine category
            if quad_class == 1:
                category = EventCategory.COOPERATION
            elif quad_class == 2:
                category = EventCategory.HUMANITARIAN if event_root in ("07", "08") else EventCategory.ECONOMIC
            elif quad_class == 3:
                category = EventCategory.VERBAL_CONFLICT
            elif quad_class == 4:
                if event_root in ("18", "19", "20"):
                    category = EventCategory.MILITARY
                elif event_root == "14":
                    category = EventCategory.PROTEST
                else:
                    category = EventCategory.MATERIAL_CONFLICT
            else:
                category = EventCategory.DIPLOMACY

            # Get description
            event_desc = EVENT_DESCRIPTIONS.get(event_code,
                         EVENT_DESCRIPTIONS.get(event_root, f"Event {event_code}"))

            # Calculate threat contribution
            threat_contrib = 0.0
            if goldstein < -5:
                threat_contrib = min(10, abs(goldstein))
            elif goldstein < 0:
                threat_contrib = abs(goldstein) * 0.3

            # Parse coordinates
            lat = safe_float(row[56]) if len(row) > 56 else 0.0
            lon = safe_float(row[57]) if len(row) > 57 else 0.0

            # Generate MGRS if we have coordinates
            mgrs = ""
            if lat != 0.0 or lon != 0.0:
                try:
                    from .geospatial import coords_to_mgrs
                    mgrs = coords_to_mgrs(lat, lon, precision=4)
                except:
                    pass

            # Calculate confidence
            num_mentions = safe_int(row[31])
            num_sources = safe_int(row[32])
            confidence = min(1.0, 0.3 + (num_sources * 0.05) + (num_mentions * 0.01))

            # Admiralty rating
            if num_sources >= 5 and num_mentions >= 20:
                admiralty = "B2"
            elif num_sources >= 3 or num_mentions >= 10:
                admiralty = "C2"
            elif num_sources >= 2:
                admiralty = "C3"
            else:
                admiralty = "F3"

            return IntelEvent(
                event_id=row[0],
                timestamp=timestamp,
                actor1_code=row[5] if len(row) > 5 else "",
                actor1_name=row[6] if len(row) > 6 else "",
                actor1_country=row[7] if len(row) > 7 else "",
                actor1_type=row[12] if len(row) > 12 else "",
                actor2_code=row[15] if len(row) > 15 else "",
                actor2_name=row[16] if len(row) > 16 else "",
                actor2_country=row[17] if len(row) > 17 else "",
                actor2_type=row[22] if len(row) > 22 else "",
                event_code=event_code,
                event_root=event_root,
                event_description=event_desc,
                category=category,
                quad_class=quad_class,
                goldstein_scale=goldstein,
                num_mentions=num_mentions,
                num_sources=num_sources,
                num_articles=safe_int(row[33]),
                avg_tone=safe_float(row[34]),
                location_name=row[53] if len(row) > 53 else "",
                location_country=row[55] if len(row) > 55 else "",
                latitude=lat,
                longitude=lon,
                mgrs=mgrs,
                confidence_score=confidence,
                admiralty_rating=admiralty,
                threat_contribution=threat_contrib,
                source_url=row[60] if len(row) > 60 else "",
            )
        except Exception as e:
            logger.debug(f"Failed to parse event: {e}")
            return None

    async def fetch_snapshot(self, timestamp: datetime) -> Optional[GDELTSnapshot]:
        """Fetch a single 15-minute GDELT snapshot."""
        ts_str = timestamp.strftime("%Y%m%d%H%M%S")

        # Check cache
        if ts_str in self.snapshots:
            return self.snapshots[ts_str]

        url = f"http://data.gdeltproject.org/gdeltv2/{ts_str}.export.CSV.zip"

        start_time = datetime.now()
        events = []
        raw_count = 0

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        return None

                    data = await resp.read()

                    with zipfile.ZipFile(io.BytesIO(data)) as zf:
                        for name in zf.namelist():
                            if name.endswith('.CSV'):
                                with zf.open(name) as f:
                                    content = f.read().decode('utf-8', errors='ignore')
                                    reader = csv.reader(io.StringIO(content), delimiter='\t')
                                    for row in reader:
                                        raw_count += 1
                                        event = self._parse_event(row, timestamp)
                                        if event:
                                            events.append(event)

            duration = int((datetime.now() - start_time).total_seconds() * 1000)
            snapshot = GDELTSnapshot(
                timestamp=timestamp,
                events=events,
                fetch_duration_ms=duration,
                raw_event_count=raw_count,
            )

            self.snapshots[ts_str] = snapshot
            return snapshot

        except Exception as e:
            self.fetch_errors.append((datetime.now(timezone.utc), str(e)))
            logger.warning(f"Failed to fetch {ts_str}: {e}")
            return None

    async def fetch_full_spectrum(self, hours: int = 240,
                                   progress_callback: Optional[Callable] = None) -> int:
        """
        Fetch all 15-minute snapshots for the specified period.

        Args:
            hours: Number of hours of history to fetch (default 240 = 10 days)
            progress_callback: Optional callback(current, total, events) for progress

        Returns:
            Total number of events fetched
        """
        now = datetime.now(timezone.utc)

        # Generate all 15-minute timestamps
        timestamps = []
        for minutes_ago in range(0, hours * 60, 15):
            ts = now - timedelta(minutes=minutes_ago)
            ts = ts.replace(minute=(ts.minute // 15) * 15, second=0, microsecond=0)
            timestamps.append(ts)

        total_cycles = len(timestamps)
        total_events = 0
        successful = 0

        print(f"Fetching {total_cycles} snapshots ({hours} hours)...")
        print(f"Period: {timestamps[-1].strftime('%Y-%m-%d %H:%M')} to {timestamps[0].strftime('%Y-%m-%d %H:%M')} UTC")

        # Fetch in batches to avoid overwhelming
        batch_size = 10
        for i in range(0, len(timestamps), batch_size):
            batch = timestamps[i:i+batch_size]
            tasks = [self.fetch_snapshot(ts) for ts in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, GDELTSnapshot) and result:
                    successful += 1
                    total_events += result.event_count

            if progress_callback:
                progress_callback(min(i + batch_size, total_cycles), total_cycles, total_events)

            # Brief pause between batches
            if i + batch_size < len(timestamps):
                await asyncio.sleep(0.5)

        self.last_fetch = datetime.now(timezone.utc)
        self.total_events = total_events

        # Rebuild country intelligence
        self._rebuild_country_intel()

        print(f"\nCompleted: {successful}/{total_cycles} snapshots, {total_events:,} events")
        return total_events

    def _rebuild_country_intel(self):
        """Rebuild country intelligence from all snapshots."""
        # Reset
        self.country_intel = {}

        # Aggregate all events
        all_events = []
        for snapshot in self.snapshots.values():
            all_events.extend(snapshot.events)

        if not all_events:
            return

        # Group events by country
        country_events: Dict[str, List[IntelEvent]] = defaultdict(list)
        for event in all_events:
            if event.actor1_country:
                country_events[event.actor1_country].append(event)
            if event.actor2_country and event.actor2_country != event.actor1_country:
                country_events[event.actor2_country].append(event)

        # Build intelligence for each country
        for country_code, events in country_events.items():
            self._build_country_intel(country_code, events, all_events)

        # Calculate global metrics
        tones = [e.avg_tone for e in all_events if e.avg_tone != 0]
        self.global_avg_tone = sum(tones) / len(tones) if tones else 0

        # Determine global threat level
        conflict_events = sum(1 for e in all_events if e.is_conflict())
        conflict_ratio = conflict_events / len(all_events) if all_events else 0
        severe_events = sum(1 for e in all_events if e.is_severe())

        if severe_events > len(all_events) * 0.1:
            self.global_threat_level = ThreatLevel.CRITICAL
        elif conflict_ratio > 0.4:
            self.global_threat_level = ThreatLevel.HIGH
        elif conflict_ratio > 0.25:
            self.global_threat_level = ThreatLevel.ELEVATED
        elif conflict_ratio > 0.15:
            self.global_threat_level = ThreatLevel.GUARDED
        else:
            self.global_threat_level = ThreatLevel.LOW

    def _build_country_intel(self, country_code: str, events: List[IntelEvent],
                             all_events: List[IntelEvent]):
        """Build intelligence for a single country."""
        intel = CountryIntelligence(
            country_code=country_code,
            country_name=self._get_country_name(country_code),
            last_updated=datetime.now(timezone.utc),
        )

        intel.total_events = len(events)
        intel.events_as_actor1 = sum(1 for e in events if e.actor1_country == country_code)
        intel.events_as_actor2 = sum(1 for e in events if e.actor2_country == country_code)

        # Classification
        intel.cooperative_events = sum(1 for e in events if e.quad_class in (1, 2))
        intel.conflictual_events = sum(1 for e in events if e.quad_class in (3, 4))
        intel.material_actions = sum(1 for e in events if e.quad_class in (2, 4))
        intel.verbal_actions = sum(1 for e in events if e.quad_class in (1, 3))

        # Sentiment
        tones = [e.avg_tone for e in events if e.avg_tone != 0]
        intel.avg_tone = sum(tones) / len(tones) if tones else 0

        # Goldstein
        goldsteins = [e.goldstein_scale for e in events]
        intel.avg_goldstein = sum(goldsteins) / len(goldsteins) if goldsteins else 0
        intel.min_goldstein = min(goldsteins) if goldsteins else 0
        intel.max_goldstein = max(goldsteins) if goldsteins else 0

        # Threat calculation
        threat_score = 0
        for e in events:
            threat_score += e.threat_contribution
        threat_score = min(100, threat_score / max(1, len(events)) * 10)
        intel.threat_score = threat_score

        if threat_score >= 75:
            intel.threat_level = ThreatLevel.CRITICAL
        elif threat_score >= 50:
            intel.threat_level = ThreatLevel.HIGH
        elif threat_score >= 30:
            intel.threat_level = ThreatLevel.ELEVATED
        elif threat_score >= 15:
            intel.threat_level = ThreatLevel.GUARDED
        else:
            intel.threat_level = ThreatLevel.LOW

        # Relationships
        partner_events: Dict[str, List[IntelEvent]] = defaultdict(list)
        adversary_events: Dict[str, List[IntelEvent]] = defaultdict(list)

        for e in events:
            other = e.actor2_country if e.actor1_country == country_code else e.actor1_country
            if other and other != country_code:
                if e.quad_class in (1, 2):
                    partner_events[other].append(e)
                elif e.quad_class in (3, 4):
                    adversary_events[other].append(e)

        # Top partners
        partner_stats = []
        for other, evts in partner_events.items():
            avg_t = sum(e.avg_tone for e in evts) / len(evts)
            partner_stats.append((other, len(evts), avg_t))
        intel.top_partners = sorted(partner_stats, key=lambda x: -x[1])[:10]

        # Top adversaries
        adversary_stats = []
        for other, evts in adversary_events.items():
            avg_t = sum(e.avg_tone for e in evts) / len(evts)
            adversary_stats.append((other, len(evts), avg_t))
        intel.top_adversaries = sorted(adversary_stats, key=lambda x: -x[1])[:10]

        # Event types
        for e in events:
            desc = EVENT_DESCRIPTIONS.get(e.event_root, e.event_code)
            intel.event_types[desc] = intel.event_types.get(desc, 0) + 1

        # MGRS hotspots
        mgrs_counts: Dict[str, int] = defaultdict(int)
        for e in events:
            if e.mgrs and len(e.mgrs) >= 5:
                zone = e.mgrs[:5]
                mgrs_counts[zone] += 1
        intel.hotspot_zones = sorted(mgrs_counts.items(), key=lambda x: -x[1])[:10]

        # Significant events (high coverage or severe)
        significant = sorted(
            [e for e in events if e.num_sources >= 3 or e.is_severe()],
            key=lambda x: -(x.num_mentions + x.threat_contribution * 10)
        )[:20]
        intel.significant_events = significant

        # Confidence
        intel.avg_confidence = sum(e.confidence_score for e in events) / len(events) if events else 0
        intel.source_diversity = len(set(e.source_url.split('/')[2] if '/' in e.source_url else ''
                                         for e in events if e.source_url))

        # Tone trend (compare recent vs older)
        if len(events) >= 20:
            sorted_events = sorted(events, key=lambda x: x.timestamp)
            recent = sorted_events[-len(sorted_events)//4:]
            older = sorted_events[:len(sorted_events)//4]
            recent_tone = sum(e.avg_tone for e in recent) / len(recent) if recent else 0
            older_tone = sum(e.avg_tone for e in older) / len(older) if older else 0
            if recent_tone > older_tone + 0.5:
                intel.tone_trend = "improving"
            elif recent_tone < older_tone - 0.5:
                intel.tone_trend = "deteriorating"
            else:
                intel.tone_trend = "stable"

        self.country_intel[country_code] = intel

    def get_country_intel(self, country_code: str) -> Optional[CountryIntelligence]:
        """Get intelligence for a specific country."""
        return self.country_intel.get(country_code.upper())

    def query_country(self, country_code: str) -> str:
        """Query intelligence for a country and return formatted report."""
        intel = self.get_country_intel(country_code.upper())
        if not intel:
            return f"No intelligence available for country code: {country_code}"
        return intel.to_summary()

    def get_global_summary(self) -> str:
        """Get global intelligence summary."""
        lines = [
            "═══════════════════════════════════════════════════════════════",
            "GLOBAL INTELLIGENCE SUMMARY",
            "═══════════════════════════════════════════════════════════════",
            f"Last Updated: {self.last_fetch.strftime('%Y-%m-%d %H:%M UTC') if self.last_fetch else 'Never'}",
            f"Total Events: {self.total_events:,}",
            f"Countries Tracked: {len(self.country_intel)}",
            f"Snapshots: {len(self.snapshots)}",
            f"",
            f"GLOBAL THREAT LEVEL: {self.global_threat_level.value.upper()}",
            f"Global Sentiment: {self.global_avg_tone:+.2f}",
            "",
            "TOP 15 COUNTRIES BY ACTIVITY:",
        ]

        sorted_countries = sorted(
            self.country_intel.values(),
            key=lambda x: -x.total_events
        )[:15]

        for i, intel in enumerate(sorted_countries, 1):
            threat_indicator = {
                ThreatLevel.CRITICAL: "🔴",
                ThreatLevel.HIGH: "🟠",
                ThreatLevel.ELEVATED: "🟡",
                ThreatLevel.GUARDED: "🟢",
                ThreatLevel.LOW: "⚪",
            }.get(intel.threat_level, "⚪")

            lines.append(
                f"  {i:2d}. {intel.country_code} {intel.country_name[:25]:25s} "
                f"{intel.total_events:6,} events {threat_indicator} {intel.threat_level.value}"
            )

        # Hotspots
        lines.extend([
            "",
            "CONFLICT HOTSPOTS (Highest Threat):",
        ])

        hotspots = sorted(
            [c for c in self.country_intel.values() if c.threat_level in (ThreatLevel.CRITICAL, ThreatLevel.HIGH)],
            key=lambda x: -x.threat_score
        )[:10]

        for intel in hotspots:
            lines.append(f"  • {intel.country_name}: {intel.threat_level.value} ({intel.threat_score:.0f}/100)")

        return "\n".join(lines)

    def get_threat_briefing(self) -> str:
        """Get a threat-focused briefing."""
        lines = [
            "═══════════════════════════════════════════════════════════════",
            "THREAT BRIEFING",
            "═══════════════════════════════════════════════════════════════",
            "",
        ]

        # Critical threats
        critical = [c for c in self.country_intel.values() if c.threat_level == ThreatLevel.CRITICAL]
        if critical:
            lines.append("CRITICAL THREATS:")
            for intel in sorted(critical, key=lambda x: -x.threat_score):
                lines.append(f"\n  {intel.country_name} ({intel.country_code})")
                lines.append(f"    Threat Score: {intel.threat_score:.0f}/100")
                lines.append(f"    Conflict Events: {intel.conflictual_events}")
                lines.append(f"    Avg Tone: {intel.avg_tone:+.2f}")
                if intel.top_adversaries:
                    adv = intel.top_adversaries[0]
                    lines.append(f"    Primary Adversary: {adv[0]} ({adv[1]} events)")
                if intel.significant_events:
                    lines.append(f"    Recent: {intel.significant_events[0].event_description}")

        # High threats
        high = [c for c in self.country_intel.values() if c.threat_level == ThreatLevel.HIGH]
        if high:
            lines.append("\n\nHIGH THREATS:")
            for intel in sorted(high, key=lambda x: -x.threat_score)[:10]:
                lines.append(f"  • {intel.country_name}: score {intel.threat_score:.0f}, "
                           f"{intel.conflictual_events} conflicts, tone {intel.avg_tone:+.1f}")

        # Elevated
        elevated = [c for c in self.country_intel.values() if c.threat_level == ThreatLevel.ELEVATED]
        if elevated:
            lines.append(f"\n\nELEVATED ({len(elevated)} countries):")
            for intel in sorted(elevated, key=lambda x: -x.threat_score)[:15]:
                lines.append(f"  • {intel.country_code}: {intel.country_name}")

        return "\n".join(lines)

    def save_state(self, filepath: str = None):
        """Save current state to disk."""
        if filepath is None:
            filepath = self.data_dir / "intel_state.json"

        state = {
            "last_fetch": self.last_fetch.isoformat() if self.last_fetch else None,
            "total_events": self.total_events,
            "snapshot_count": len(self.snapshots),
            "country_count": len(self.country_intel),
            "global_threat": self.global_threat_level.value,
            "global_tone": self.global_avg_tone,
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def clear_old_data(self, hours: int = None):
        """Remove snapshots older than specified hours."""
        if hours is None:
            hours = self.max_history_hours

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        old_keys = [k for k, v in self.snapshots.items() if v.timestamp < cutoff]
        for k in old_keys:
            del self.snapshots[k]


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON ACCESS
# ═══════════════════════════════════════════════════════════════════════════════

_intel_system: Optional[RealTimeIntelligence] = None


def get_intel_system() -> RealTimeIntelligence:
    """Get the global intelligence system instance."""
    global _intel_system
    if _intel_system is None:
        _intel_system = RealTimeIntelligence()
    return _intel_system


def reset_intel_system():
    """Reset the global intelligence system."""
    global _intel_system
    _intel_system = None
