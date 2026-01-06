"""
GDELT (Global Database of Events, Language, and Tone) Integration.

Fetches and processes real-time global event data from GDELT:
- Events: Political events, conflicts, cooperation
- Actors: Countries, organizations, ethnic groups
- GKG: Global Knowledge Graph with entities, themes, tone
- Mentions: News article mentions and sentiment

GDELT updates every 15 minutes with global news coverage.
"""

from __future__ import annotations
import os
import io
import gzip
import csv
import json
import asyncio
import aiohttp
import hashlib
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Optional, Any, Callable, Tuple
from pathlib import Path
import zipfile
from collections import defaultdict
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# GDELT DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class GDELTEventCode(str, Enum):
    """CAMEO event codes - top level categories."""
    MAKE_PUBLIC_STATEMENT = "01"
    APPEAL = "02"
    EXPRESS_INTENT_COOPERATE = "03"
    CONSULT = "04"
    DIPLOMATIC_COOPERATION = "05"
    MATERIAL_COOPERATION = "06"
    PROVIDE_AID = "07"
    YIELD = "08"
    INVESTIGATE = "09"
    DEMAND = "10"
    DISAPPROVE = "11"
    REJECT = "12"
    THREATEN = "13"
    PROTEST = "14"
    EXHIBIT_FORCE = "15"
    REDUCE_RELATIONS = "16"
    COERCE = "17"
    ASSAULT = "18"
    FIGHT = "19"
    USE_UNCONVENTIONAL_MASS_VIOLENCE = "20"


class GDELTActorType(str, Enum):
    """Actor type codes."""
    GOVERNMENT = "GOV"
    MILITARY = "MIL"
    REBEL = "REB"
    OPPOSITION = "OPP"
    POLITICAL_PARTY = "PTY"
    ETHNIC = "ETH"
    RELIGIOUS = "REL"
    EDUCATION = "EDU"
    BUSINESS = "BUS"
    MEDIA = "MED"
    REFUGEE = "REF"
    LABOR = "LAB"
    NGO = "NGO"
    INTERNATIONAL_ORG = "IGO"
    CRIMINAL = "CRM"
    CIVILIAN = "CVL"
    HEALTHCARE = "HLH"
    LEGAL = "LEG"
    SPORTS = "SPO"
    ENVIRONMENT = "ENV"


@dataclass
class GDELTActor:
    """An actor in a GDELT event."""
    code: str = ""
    name: str = ""
    country_code: str = ""
    known_group_code: str = ""
    ethnic_code: str = ""
    religion_codes: List[str] = field(default_factory=list)
    type_codes: List[str] = field(default_factory=list)
    geo_type: int = 0
    geo_fullname: str = ""
    geo_country_code: str = ""
    geo_lat: float = 0.0
    geo_lon: float = 0.0


@dataclass
class GDELTEvent:
    """A GDELT event record."""
    global_event_id: str = ""
    day: int = 0  # YYYYMMDD
    month_year: int = 0
    year: int = 0
    fraction_date: float = 0.0

    # Actors
    actor1: GDELTActor = field(default_factory=GDELTActor)
    actor2: GDELTActor = field(default_factory=GDELTActor)

    # Event details
    is_root_event: bool = False
    event_code: str = ""
    event_base_code: str = ""
    event_root_code: str = ""
    quad_class: int = 0  # 1=verbal coop, 2=material coop, 3=verbal conflict, 4=material conflict

    # Goldstein scale (-10 to +10, conflict to cooperation)
    goldstein_scale: float = 0.0
    num_mentions: int = 0
    num_sources: int = 0
    num_articles: int = 0
    avg_tone: float = 0.0

    # Location
    action_geo_type: int = 0
    action_geo_fullname: str = ""
    action_geo_country_code: str = ""
    action_geo_lat: float = 0.0
    action_geo_lon: float = 0.0

    # Source
    date_added: str = ""
    source_url: str = ""

    # Enhanced fields (populated by assessment pipeline)
    mgrs_coordinate: str = ""  # Military Grid Reference System
    confidence_score: float = 0.0  # 0-1 confidence rating
    admiralty_rating: str = ""  # e.g., "B2"
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    source_reliability: str = ""  # A-F rating
    assessed_at: Optional[datetime] = None

    def is_cooperative(self) -> bool:
        """Check if event is cooperative."""
        return self.quad_class in (1, 2)

    def is_conflictual(self) -> bool:
        """Check if event is conflictual."""
        return self.quad_class in (3, 4)

    def is_material(self) -> bool:
        """Check if event involves material action."""
        return self.quad_class in (2, 4)

    def has_geo(self) -> bool:
        """Check if event has geographic coordinates."""
        return self.action_geo_lat != 0.0 or self.action_geo_lon != 0.0

    def is_assessed(self) -> bool:
        """Check if event has been assessed for confidence."""
        return self.assessed_at is not None

    def to_assessment_dict(self) -> Dict[str, Any]:
        """Convert to dict format for confidence assessment."""
        return {
            "avg_tone": self.avg_tone,
            "goldstein_scale": self.goldstein_scale,
            "num_mentions": self.num_mentions,
            "num_sources": self.num_sources,
            "num_articles": self.num_articles,
            "geo_lat": self.action_geo_lat,
            "geo_lon": self.action_geo_lon,
            "actor1": self.actor1.name or self.actor1.country_code,
            "actor2": self.actor2.name or self.actor2.country_code,
            "event_code": self.event_code,
        }


@dataclass
class GDELTMention:
    """A mention of an event in news."""
    global_event_id: str = ""
    event_time: str = ""
    mention_time: str = ""
    mention_type: int = 0  # 1=web, 2=citationonly, 3=core, 4=dtic, 5=jstor, 6=nontextualsource
    mention_source_name: str = ""
    mention_identifier: str = ""
    sentence_id: int = 0
    actor1_char_offset: int = 0
    actor2_char_offset: int = 0
    action_char_offset: int = 0
    in_raw_text: bool = False
    confidence: int = 0
    mention_doc_len: int = 0
    mention_doc_tone: float = 0.0


@dataclass
class GKGEntity:
    """An entity from the Global Knowledge Graph."""
    name: str = ""
    entity_type: str = ""  # PERSON, ORGANIZATION, LOCATION
    offset: int = 0
    length: int = 0
    confidence: float = 0.0


@dataclass
class GKGRecord:
    """A Global Knowledge Graph record."""
    gkg_record_id: str = ""
    date: str = ""
    source_collection_id: int = 0
    source_common_name: str = ""
    document_identifier: str = ""

    # Themes
    themes: List[str] = field(default_factory=list)

    # Entities
    persons: List[str] = field(default_factory=list)
    organizations: List[str] = field(default_factory=list)
    locations: List[Tuple[str, str, float, float]] = field(default_factory=list)  # name, country, lat, lon

    # Tone
    tone: float = 0.0
    positive_score: float = 0.0
    negative_score: float = 0.0
    polarity: float = 0.0
    activity_density: float = 0.0
    word_count: int = 0

    # GCAM (Global Content Analysis Measures)
    gcam_scores: Dict[str, float] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# GDELT DATA FETCHER
# ═══════════════════════════════════════════════════════════════════════════════

class GDELTDataSource(str, Enum):
    """GDELT data sources."""
    EVENTS = "events"
    MENTIONS = "mentions"
    GKG = "gkg"


GDELT_BASE_URLS = {
    "events": "http://data.gdeltproject.org/gdeltv2/",
    "gkg": "http://data.gdeltproject.org/gdeltv2/",
    "masterfile_events": "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt",
    "masterfile_translation": "http://data.gdeltproject.org/gdeltv2/masterfilelist-translation.txt",
    "lastupdate": "http://data.gdeltproject.org/gdeltv2/lastupdate.txt",
}


class GDELTFetcher:
    """
    Fetches and parses GDELT data files.

    GDELT updates every 15 minutes. Files are named:
    - Events: YYYYMMDDHHMMSS.export.CSV.zip
    - Mentions: YYYYMMDDHHMMSS.mentions.CSV.zip
    - GKG: YYYYMMDDHHMMSS.gkg.csv.zip
    """

    def __init__(self, cache_dir: str = ".gdelt_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _get_session(self) -> aiohttp.ClientSession:
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session

    async def get_latest_files(self) -> Dict[str, str]:
        """Get URLs of the latest GDELT files."""
        session = self._get_session()
        async with session.get(GDELT_BASE_URLS["lastupdate"]) as resp:
            text = await resp.text()

        files = {}
        for line in text.strip().split("\n"):
            parts = line.split()
            if len(parts) >= 3:
                url = parts[2]
                if ".export." in url:
                    files["events"] = url
                elif ".mentions." in url:
                    files["mentions"] = url
                elif ".gkg." in url:
                    files["gkg"] = url

        return files

    async def fetch_file(self, url: str, use_cache: bool = True) -> bytes:
        """Fetch a GDELT file (with optional caching)."""
        cache_key = hashlib.md5(url.encode()).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.gz"

        if use_cache and cache_path.exists():
            # Check if cache is less than 15 minutes old
            mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
            if datetime.now() - mtime < timedelta(minutes=15):
                return cache_path.read_bytes()

        session = self._get_session()
        async with session.get(url) as resp:
            data = await resp.read()

        if use_cache:
            cache_path.write_bytes(data)

        return data

    def _decompress(self, data: bytes, url: str) -> str:
        """Decompress GDELT file data."""
        if url.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                # Get the first file in the archive
                name = zf.namelist()[0]
                return zf.read(name).decode("utf-8", errors="replace")
        elif url.endswith(".gz"):
            return gzip.decompress(data).decode("utf-8", errors="replace")
        else:
            return data.decode("utf-8", errors="replace")

    async def fetch_events(self, url: str = None) -> List[GDELTEvent]:
        """Fetch and parse GDELT events."""
        if not url:
            files = await self.get_latest_files()
            url = files.get("events")

        if not url:
            return []

        data = await self.fetch_file(url)
        text = self._decompress(data, url)

        events = []
        reader = csv.reader(io.StringIO(text), delimiter="\t")

        for row in reader:
            if len(row) < 58:
                continue

            try:
                event = GDELTEvent(
                    global_event_id=row[0],
                    day=int(row[1]) if row[1] else 0,
                    month_year=int(row[2]) if row[2] else 0,
                    year=int(row[3]) if row[3] else 0,
                    fraction_date=float(row[4]) if row[4] else 0.0,

                    actor1=GDELTActor(
                        code=row[5],
                        name=row[6],
                        country_code=row[7],
                        known_group_code=row[8],
                        ethnic_code=row[9],
                        religion_codes=row[10].split(",") if row[10] else [],
                        type_codes=row[11].split(",") if row[11] else [],
                        geo_type=int(row[35]) if row[35] else 0,
                        geo_fullname=row[36],
                        geo_country_code=row[37],
                        geo_lat=float(row[39]) if row[39] else 0.0,
                        geo_lon=float(row[40]) if row[40] else 0.0,
                    ),

                    actor2=GDELTActor(
                        code=row[15],
                        name=row[16],
                        country_code=row[17],
                        known_group_code=row[18],
                        ethnic_code=row[19],
                        religion_codes=row[20].split(",") if row[20] else [],
                        type_codes=row[21].split(",") if row[21] else [],
                        geo_type=int(row[42]) if row[42] else 0,
                        geo_fullname=row[43],
                        geo_country_code=row[44],
                        geo_lat=float(row[46]) if row[46] else 0.0,
                        geo_lon=float(row[47]) if row[47] else 0.0,
                    ),

                    is_root_event=row[25] == "1",
                    event_code=row[26],
                    event_base_code=row[27],
                    event_root_code=row[28],
                    quad_class=int(row[29]) if row[29] else 0,
                    goldstein_scale=float(row[30]) if row[30] else 0.0,
                    num_mentions=int(row[31]) if row[31] else 0,
                    num_sources=int(row[32]) if row[32] else 0,
                    num_articles=int(row[33]) if row[33] else 0,
                    avg_tone=float(row[34]) if row[34] else 0.0,

                    action_geo_type=int(row[49]) if row[49] else 0,
                    action_geo_fullname=row[50],
                    action_geo_country_code=row[51],
                    action_geo_lat=float(row[53]) if row[53] else 0.0,
                    action_geo_lon=float(row[54]) if row[54] else 0.0,

                    date_added=row[57] if len(row) > 57 else "",
                    source_url=row[58] if len(row) > 58 else "",
                )
                events.append(event)
            except (ValueError, IndexError) as e:
                continue

        return events

    async def fetch_mentions(self, url: str = None) -> List[GDELTMention]:
        """Fetch and parse GDELT mentions."""
        if not url:
            files = await self.get_latest_files()
            url = files.get("mentions")

        if not url:
            return []

        data = await self.fetch_file(url)
        text = self._decompress(data, url)

        mentions = []
        reader = csv.reader(io.StringIO(text), delimiter="\t")

        for row in reader:
            if len(row) < 14:
                continue

            try:
                mention = GDELTMention(
                    global_event_id=row[0],
                    event_time=row[1],
                    mention_time=row[2],
                    mention_type=int(row[3]) if row[3] else 0,
                    mention_source_name=row[4],
                    mention_identifier=row[5],
                    sentence_id=int(row[6]) if row[6] else 0,
                    actor1_char_offset=int(row[7]) if row[7] else 0,
                    actor2_char_offset=int(row[8]) if row[8] else 0,
                    action_char_offset=int(row[9]) if row[9] else 0,
                    in_raw_text=row[10] == "1",
                    confidence=int(row[11]) if row[11] else 0,
                    mention_doc_len=int(row[12]) if row[12] else 0,
                    mention_doc_tone=float(row[13]) if row[13] else 0.0,
                )
                mentions.append(mention)
            except (ValueError, IndexError):
                continue

        return mentions

    async def fetch_gkg(self, url: str = None) -> List[GKGRecord]:
        """Fetch and parse Global Knowledge Graph records."""
        if not url:
            files = await self.get_latest_files()
            url = files.get("gkg")

        if not url:
            return []

        data = await self.fetch_file(url)
        text = self._decompress(data, url)

        records = []
        reader = csv.reader(io.StringIO(text), delimiter="\t")

        for row in reader:
            if len(row) < 15:
                continue

            try:
                # Parse themes
                themes = row[7].split(";") if row[7] else []

                # Parse persons
                persons = row[9].split(";") if row[9] else []

                # Parse organizations
                organizations = row[10].split(";") if row[10] else []

                # Parse locations
                locations = []
                if row[8]:
                    for loc in row[8].split(";"):
                        parts = loc.split("#")
                        if len(parts) >= 7:
                            locations.append((
                                parts[1],  # name
                                parts[3],  # country code
                                float(parts[5]) if parts[5] else 0.0,  # lat
                                float(parts[6]) if parts[6] else 0.0,  # lon
                            ))

                # Parse tone
                tone_parts = row[14].split(",") if row[14] else []

                record = GKGRecord(
                    gkg_record_id=row[0],
                    date=row[1],
                    source_collection_id=int(row[2]) if row[2] else 0,
                    source_common_name=row[3],
                    document_identifier=row[4],
                    themes=themes,
                    persons=persons,
                    organizations=organizations,
                    locations=locations,
                    tone=float(tone_parts[0]) if len(tone_parts) > 0 and tone_parts[0] else 0.0,
                    positive_score=float(tone_parts[1]) if len(tone_parts) > 1 and tone_parts[1] else 0.0,
                    negative_score=float(tone_parts[2]) if len(tone_parts) > 2 and tone_parts[2] else 0.0,
                    polarity=float(tone_parts[3]) if len(tone_parts) > 3 and tone_parts[3] else 0.0,
                    activity_density=float(tone_parts[4]) if len(tone_parts) > 4 and tone_parts[4] else 0.0,
                    word_count=int(float(tone_parts[6])) if len(tone_parts) > 6 and tone_parts[6] else 0,
                )
                records.append(record)
            except (ValueError, IndexError):
                continue

        return records


# ═══════════════════════════════════════════════════════════════════════════════
# CRON JOB SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CronJob:
    """A scheduled job."""
    name: str
    schedule: str  # cron expression or interval
    func: Callable
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None


class CronScheduler:
    """
    Simple cron-style scheduler for GDELT data fetching.

    Supports:
    - Interval-based scheduling (every N minutes/hours)
    - Time-based scheduling (at specific times)
    - One-shot jobs
    """

    def __init__(self):
        self.jobs: Dict[str, CronJob] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def add_job(self, name: str, schedule: str, func: Callable,
                *args, **kwargs) -> CronJob:
        """
        Add a job to the scheduler.

        Schedule formats:
        - "every 15m" - every 15 minutes
        - "every 1h" - every hour
        - "every 1d" - every day
        - "at HH:MM" - at specific time daily
        - "once" - run once immediately
        """
        job = CronJob(
            name=name,
            schedule=schedule,
            func=func,
            args=args,
            kwargs=kwargs,
        )
        job.next_run = self._calculate_next_run(schedule)
        self.jobs[name] = job
        return job

    def remove_job(self, name: str):
        """Remove a job from the scheduler."""
        self.jobs.pop(name, None)

    def enable_job(self, name: str):
        """Enable a job."""
        if name in self.jobs:
            self.jobs[name].enabled = True

    def disable_job(self, name: str):
        """Disable a job."""
        if name in self.jobs:
            self.jobs[name].enabled = False

    def _calculate_next_run(self, schedule: str, from_time: datetime = None) -> datetime:
        """Calculate the next run time for a schedule."""
        now = from_time or datetime.now()

        if schedule == "once":
            return now

        if schedule.startswith("every "):
            interval = schedule[6:]
            if interval.endswith("m"):
                minutes = int(interval[:-1])
                return now + timedelta(minutes=minutes)
            elif interval.endswith("h"):
                hours = int(interval[:-1])
                return now + timedelta(hours=hours)
            elif interval.endswith("d"):
                days = int(interval[:-1])
                return now + timedelta(days=days)
            elif interval.endswith("s"):
                seconds = int(interval[:-1])
                return now + timedelta(seconds=seconds)

        if schedule.startswith("at "):
            time_str = schedule[3:]
            target_time = datetime.strptime(time_str, "%H:%M").time()
            next_run = datetime.combine(now.date(), target_time)
            if next_run <= now:
                next_run += timedelta(days=1)
            return next_run

        # Default to 15 minutes (GDELT update interval)
        return now + timedelta(minutes=15)

    async def _run_job(self, job: CronJob):
        """Run a single job."""
        try:
            job.last_run = datetime.now()
            if asyncio.iscoroutinefunction(job.func):
                await job.func(*job.args, **job.kwargs)
            else:
                job.func(*job.args, **job.kwargs)
            job.run_count += 1
            job.last_error = None
        except Exception as e:
            job.error_count += 1
            job.last_error = str(e)
            logger.error(f"Job {job.name} failed: {e}")

        # Calculate next run
        if job.schedule != "once":
            job.next_run = self._calculate_next_run(job.schedule)
        else:
            job.enabled = False

    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self._running:
            now = datetime.now()

            for job in self.jobs.values():
                if job.enabled and job.next_run and job.next_run <= now:
                    await self._run_job(job)

            # Sleep for a short interval
            await asyncio.sleep(1)

    def start(self):
        """Start the scheduler."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop())
        logger.info("Scheduler started")

    def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
        logger.info("Scheduler stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        return {
            "running": self._running,
            "jobs": {
                name: {
                    "enabled": job.enabled,
                    "schedule": job.schedule,
                    "last_run": job.last_run.isoformat() if job.last_run else None,
                    "next_run": job.next_run.isoformat() if job.next_run else None,
                    "run_count": job.run_count,
                    "error_count": job.error_count,
                    "last_error": job.last_error,
                }
                for name, job in self.jobs.items()
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# GDELT DATA MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class GDELTDataManager:
    """
    Manages GDELT data fetching, storage, and analysis.

    Provides:
    - Automatic data fetching on schedule
    - Event aggregation and statistics
    - Actor/entity extraction
    - Trend detection
    - Confidence assessment pipeline
    - MGRS geospatial indexing
    """

    def __init__(self, cache_dir: str = ".gdelt_cache", data_dir: str = ".gdelt_data",
                 enable_assessment: bool = True):
        self.cache_dir = Path(cache_dir)
        self.data_dir = Path(data_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)

        self.fetcher = GDELTFetcher(cache_dir)
        self.scheduler = CronScheduler()
        self.enable_assessment = enable_assessment

        # In-memory data stores
        self.events: List[GDELTEvent] = []
        self.mentions: List[GDELTMention] = []
        self.gkg_records: List[GKGRecord] = []

        # Aggregated data
        self.event_counts_by_country: Dict[str, int] = defaultdict(int)
        self.event_counts_by_type: Dict[str, int] = defaultdict(int)
        self.actor_mention_counts: Dict[str, int] = defaultdict(int)
        self.theme_counts: Dict[str, int] = defaultdict(int)
        self.entity_counts: Dict[str, int] = defaultdict(int)

        # Confidence tracking
        self.events_by_confidence: Dict[str, List[GDELTEvent]] = {
            "high": [],      # > 0.8
            "medium": [],    # 0.5 - 0.8
            "low": [],       # < 0.5
        }
        self.avg_confidence_timeline: List[Tuple[datetime, float]] = []

        # MGRS spatial index
        self.events_by_mgrs_zone: Dict[str, List[GDELTEvent]] = defaultdict(list)

        # Time series
        self.event_timeline: List[Tuple[datetime, int]] = []
        self.tone_timeline: List[Tuple[datetime, float]] = []

        # Callbacks
        self.on_events_updated: List[Callable] = []
        self.on_gkg_updated: List[Callable] = []

    def _assess_event(self, event: GDELTEvent) -> GDELTEvent:
        """Apply confidence assessment and MGRS conversion to an event."""
        try:
            # Import here to avoid circular dependency
            from .intel_classification import assess_gdelt_event, get_source_profile
            from .geospatial import coords_to_mgrs

            # Get source profile if we can identify the source
            source_profile = None
            if event.source_url:
                # Extract domain from URL
                import re
                domain_match = re.search(r'://([^/]+)', event.source_url)
                if domain_match:
                    domain = domain_match.group(1).lower()
                    # Map common domains to source profiles
                    domain_map = {
                        "apnews.com": "AP",
                        "reuters.com": "REUTERS",
                        "bbc.com": "BBC",
                        "bbc.co.uk": "BBC",
                        "cnn.com": "CNN",
                        "nytimes.com": "NYT",
                        "washingtonpost.com": "WAPO",
                        "theguardian.com": "GUARDIAN",
                        "wsj.com": "WSJ",
                        "aljazeera.com": "ALJAZEERA",
                    }
                    for key, source_id in domain_map.items():
                        if key in domain:
                            source_profile = get_source_profile(source_id)
                            break

            # Perform confidence assessment
            assessment = assess_gdelt_event(event.to_assessment_dict(), source_profile)

            event.confidence_score = assessment.confidence_score
            event.admiralty_rating = str(assessment.admiralty_rating)
            event.confidence_interval = assessment.confidence_interval
            event.source_reliability = assessment.admiralty_rating.source_reliability.value
            event.assessed_at = datetime.now()

            # Add MGRS coordinate if geo data present
            if event.has_geo():
                try:
                    event.mgrs_coordinate = coords_to_mgrs(
                        event.action_geo_lat,
                        event.action_geo_lon,
                        precision=4
                    )
                except Exception:
                    pass  # Invalid coordinates

        except ImportError:
            # Assessment modules not available
            pass
        except Exception as e:
            logger.warning(f"Failed to assess event {event.global_event_id}: {e}")

        return event

    def _categorize_by_confidence(self, event: GDELTEvent):
        """Add event to confidence-based categories."""
        if event.confidence_score > 0.8:
            self.events_by_confidence["high"].append(event)
        elif event.confidence_score >= 0.5:
            self.events_by_confidence["medium"].append(event)
        else:
            self.events_by_confidence["low"].append(event)

        # Add to MGRS index
        if event.mgrs_coordinate:
            # Use first 5 chars as zone key (e.g., "18SUJ")
            zone_key = event.mgrs_coordinate[:5]
            self.events_by_mgrs_zone[zone_key].append(event)

    def get_high_confidence_events(self, min_confidence: float = 0.8) -> List[GDELTEvent]:
        """Get events with high confidence scores."""
        return [e for e in self.events if e.confidence_score >= min_confidence]

    def get_events_in_mgrs_zone(self, zone_prefix: str) -> List[GDELTEvent]:
        """Get events in a specific MGRS zone."""
        return self.events_by_mgrs_zone.get(zone_prefix, [])

    def get_confidence_stats(self) -> Dict[str, Any]:
        """Get confidence statistics for current events."""
        if not self.events:
            return {"total": 0, "assessed": 0, "avg_confidence": 0.0}

        assessed = [e for e in self.events if e.is_assessed()]
        if not assessed:
            return {"total": len(self.events), "assessed": 0, "avg_confidence": 0.0}

        return {
            "total": len(self.events),
            "assessed": len(assessed),
            "avg_confidence": sum(e.confidence_score for e in assessed) / len(assessed),
            "high_confidence": len([e for e in assessed if e.confidence_score > 0.8]),
            "medium_confidence": len([e for e in assessed if 0.5 <= e.confidence_score <= 0.8]),
            "low_confidence": len([e for e in assessed if e.confidence_score < 0.5]),
            "with_geo": len([e for e in assessed if e.has_geo()]),
            "with_mgrs": len([e for e in assessed if e.mgrs_coordinate]),
        }

    async def start(self):
        """Start the data manager with scheduled fetching."""
        # Set up scheduled jobs
        self.scheduler.add_job(
            "fetch_events",
            "every 15m",
            self._fetch_and_process_events
        )
        self.scheduler.add_job(
            "fetch_gkg",
            "every 15m",
            self._fetch_and_process_gkg
        )
        self.scheduler.add_job(
            "aggregate_stats",
            "every 1h",
            self._aggregate_statistics
        )

        # Initial fetch
        await self._fetch_and_process_events()
        await self._fetch_and_process_gkg()

        self.scheduler.start()

    def stop(self):
        """Stop the data manager."""
        self.scheduler.stop()

    async def _fetch_and_process_events(self):
        """Fetch and process latest events."""
        async with GDELTFetcher(str(self.cache_dir)) as fetcher:
            events = await fetcher.fetch_events()

        if events:
            # Apply confidence assessment and MGRS conversion
            if self.enable_assessment:
                events = [self._assess_event(e) for e in events]

            self.events.extend(events)
            # Keep only last 24 hours
            cutoff = datetime.now() - timedelta(hours=24)
            self.events = [e for e in self.events
                          if self._parse_gdelt_date(e.day) > cutoff]

            # Update counts and categorization
            for event in events:
                if event.actor1.country_code:
                    self.event_counts_by_country[event.actor1.country_code] += 1
                if event.actor2.country_code:
                    self.event_counts_by_country[event.actor2.country_code] += 1
                self.event_counts_by_type[event.event_root_code] += 1

                # Categorize by confidence
                if self.enable_assessment and event.is_assessed():
                    self._categorize_by_confidence(event)

            # Notify callbacks
            for callback in self.on_events_updated:
                if asyncio.iscoroutinefunction(callback):
                    await callback(events)
                else:
                    callback(events)

        logger.info(f"Processed {len(events)} events, total: {len(self.events)}")

    async def _fetch_and_process_gkg(self):
        """Fetch and process latest GKG records."""
        async with GDELTFetcher(str(self.cache_dir)) as fetcher:
            records = await fetcher.fetch_gkg()

        if records:
            self.gkg_records.extend(records)
            # Keep only last 24 hours worth
            if len(self.gkg_records) > 100000:
                self.gkg_records = self.gkg_records[-100000:]

            # Update counts
            for record in records:
                for theme in record.themes:
                    self.theme_counts[theme] += 1
                for person in record.persons:
                    self.entity_counts[person] += 1
                for org in record.organizations:
                    self.entity_counts[org] += 1

            # Notify callbacks
            for callback in self.on_gkg_updated:
                if asyncio.iscoroutinefunction(callback):
                    await callback(records)
                else:
                    callback(records)

        logger.info(f"Processed {len(records)} GKG records, total: {len(self.gkg_records)}")

    async def _aggregate_statistics(self):
        """Aggregate statistics periodically."""
        now = datetime.now()

        # Event count over time
        recent_events = [e for e in self.events
                        if self._parse_gdelt_date(e.day) > now - timedelta(hours=1)]
        self.event_timeline.append((now, len(recent_events)))

        # Average tone over time
        if recent_events:
            avg_tone = sum(e.avg_tone for e in recent_events) / len(recent_events)
            self.tone_timeline.append((now, avg_tone))

        # Keep only last 7 days of timeline
        cutoff = now - timedelta(days=7)
        self.event_timeline = [(t, c) for t, c in self.event_timeline if t > cutoff]
        self.tone_timeline = [(t, tone) for t, tone in self.tone_timeline if t > cutoff]

    def _parse_gdelt_date(self, day: int) -> datetime:
        """Parse GDELT day format (YYYYMMDD) to datetime."""
        try:
            return datetime.strptime(str(day), "%Y%m%d")
        except:
            return datetime.now()

    # ─────────────────────────────────────────────────────────────────────────
    # Query Methods
    # ─────────────────────────────────────────────────────────────────────────

    def get_events_by_country(self, country_code: str) -> List[GDELTEvent]:
        """Get events involving a country."""
        return [e for e in self.events
                if e.actor1.country_code == country_code
                or e.actor2.country_code == country_code
                or e.action_geo_country_code == country_code]

    def get_events_by_type(self, event_code: str) -> List[GDELTEvent]:
        """Get events by CAMEO code."""
        return [e for e in self.events if e.event_root_code == event_code]

    def get_cooperative_events(self) -> List[GDELTEvent]:
        """Get cooperative events."""
        return [e for e in self.events if e.is_cooperative()]

    def get_conflictual_events(self) -> List[GDELTEvent]:
        """Get conflictual events."""
        return [e for e in self.events if e.is_conflictual()]

    def get_top_actors(self, n: int = 20) -> List[Tuple[str, int]]:
        """Get top actors by mention count."""
        return sorted(self.actor_mention_counts.items(), key=lambda x: -x[1])[:n]

    def get_top_themes(self, n: int = 20) -> List[Tuple[str, int]]:
        """Get top themes."""
        return sorted(self.theme_counts.items(), key=lambda x: -x[1])[:n]

    def get_top_entities(self, n: int = 20) -> List[Tuple[str, int]]:
        """Get top entities (people and organizations)."""
        return sorted(self.entity_counts.items(), key=lambda x: -x[1])[:n]

    def get_country_relations(self, country1: str, country2: str) -> Dict[str, Any]:
        """Get relations between two countries."""
        events = [e for e in self.events
                  if (e.actor1.country_code == country1 and e.actor2.country_code == country2)
                  or (e.actor1.country_code == country2 and e.actor2.country_code == country1)]

        cooperative = sum(1 for e in events if e.is_cooperative())
        conflictual = sum(1 for e in events if e.is_conflictual())
        avg_goldstein = sum(e.goldstein_scale for e in events) / len(events) if events else 0
        avg_tone = sum(e.avg_tone for e in events) / len(events) if events else 0

        return {
            "total_events": len(events),
            "cooperative": cooperative,
            "conflictual": conflictual,
            "avg_goldstein": avg_goldstein,
            "avg_tone": avg_tone,
            "relationship_score": avg_goldstein,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics."""
        return {
            "total_events": len(self.events),
            "total_gkg_records": len(self.gkg_records),
            "countries_covered": len(self.event_counts_by_country),
            "themes_tracked": len(self.theme_counts),
            "entities_tracked": len(self.entity_counts),
            "scheduler": self.scheduler.get_status(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_gdelt_manager: Optional[GDELTDataManager] = None


def get_gdelt_manager() -> GDELTDataManager:
    """Get the global GDELT data manager."""
    global _gdelt_manager
    if _gdelt_manager is None:
        _gdelt_manager = GDELTDataManager()
    return _gdelt_manager


def reset_gdelt_manager():
    """Reset the global GDELT data manager."""
    global _gdelt_manager
    if _gdelt_manager:
        _gdelt_manager.stop()
    _gdelt_manager = None
