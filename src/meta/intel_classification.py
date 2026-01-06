"""
Intelligence Classification and Confidence Assessment Framework.

Provides:
- Classification levels (UNCLASSIFIED through TOP SECRET)
- Source reliability ratings (A-F)
- Information credibility ratings (1-6)
- Admiralty/NATO confidence system
- GDELT/media source evaluation
- Temporal decay functions for confidence
- Multi-source corroboration scoring
"""

from __future__ import annotations
import math
import hashlib
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Dict, List, Set, Optional, Tuple, Any, Callable
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION LEVELS
# ═══════════════════════════════════════════════════════════════════════════════

class ClassificationLevel(IntEnum):
    """Security classification levels (US-style)."""
    UNCLASSIFIED = 0
    CONTROLLED_UNCLASSIFIED = 1  # CUI / FOUO
    CONFIDENTIAL = 2
    SECRET = 3
    TOP_SECRET = 4
    TOP_SECRET_SCI = 5  # Sensitive Compartmented Information


class DisseminationControl(str, Enum):
    """Dissemination control markings."""
    UNRESTRICTED = "UNRESTRICTED"
    NOFORN = "NOFORN"           # No Foreign Nationals
    RELTO = "REL TO"           # Releasable To (specific countries)
    ORCON = "ORCON"            # Originator Controlled
    PROPIN = "PROPIN"          # Proprietary Information
    IMCON = "IMCON"            # Intelligence Methods
    FISA = "FISA"              # FISA derived
    WAIVED = "WAIVED"          # Pre-approved for release
    EYES_ONLY = "EYES ONLY"    # Named recipients only


class SCI_Compartment(str, Enum):
    """SCI compartment codes (fictional/generic)."""
    HUMINT = "HCS"       # Human Intelligence Control System
    SIGINT = "SI"        # Special Intelligence (SIGINT)
    COMINT = "TK"        # TALENT KEYHOLE (imagery)
    GEOINT = "G"         # Geospatial Intelligence
    MASINT = "MR"        # Measurement and Signature
    OSINT = "OS"         # Open Source (typically UNCLASS)
    FININT = "FI"        # Financial Intelligence
    CYBERINT = "CY"      # Cyber Intelligence


@dataclass
class ClassificationMarking:
    """Complete classification marking for a piece of information."""
    level: ClassificationLevel = ClassificationLevel.UNCLASSIFIED
    compartments: Set[SCI_Compartment] = field(default_factory=set)
    dissemination: Set[DisseminationControl] = field(default_factory=set)
    releasable_to: Set[str] = field(default_factory=set)  # Country codes
    declassify_on: Optional[datetime] = None
    classified_by: str = ""
    reason: str = ""

    def __str__(self) -> str:
        """Format as standard classification banner."""
        parts = [self.level.name.replace("_", " ")]

        if self.compartments:
            parts.append("//")
            parts.append("/".join(c.value for c in sorted(self.compartments)))

        if DisseminationControl.NOFORN in self.dissemination:
            parts.append("//NOFORN")
        elif DisseminationControl.RELTO in self.dissemination and self.releasable_to:
            parts.append(f"//REL TO {', '.join(sorted(self.releasable_to))}")

        return "".join(parts)

    @classmethod
    def unclassified(cls) -> "ClassificationMarking":
        """Create UNCLASSIFIED marking."""
        return cls(level=ClassificationLevel.UNCLASSIFIED)

    @classmethod
    def from_osint(cls) -> "ClassificationMarking":
        """Create marking for open source intelligence."""
        return cls(
            level=ClassificationLevel.UNCLASSIFIED,
            compartments={SCI_Compartment.OSINT}
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ADMIRALTY/NATO CONFIDENCE SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class SourceReliability(str, Enum):
    """
    Admiralty System - Source Reliability Rating (A-F).
    Evaluates the source itself, not the specific information.
    """
    A = "A"  # Completely Reliable - No doubt of authenticity, trustworthiness
    B = "B"  # Usually Reliable - Minor doubt, used successfully in past
    C = "C"  # Fairly Reliable - Doubt of reliability, provided valid info before
    D = "D"  # Not Usually Reliable - Significant doubt, little valid info before
    E = "E"  # Unreliable - Lacking authenticity, untrustworthy
    F = "F"  # Reliability Cannot Be Judged - No basis for evaluation


class InformationCredibility(IntEnum):
    """
    Admiralty System - Information Credibility Rating (1-6).
    Evaluates the specific piece of information, not the source.
    """
    CONFIRMED = 1          # Confirmed by other independent sources
    PROBABLY_TRUE = 2      # Likely to be true based on logical analysis
    POSSIBLY_TRUE = 3      # Possibly true, not yet confirmed or denied
    DOUBTFUL = 4          # Doubtful, inconsistent with other info
    IMPROBABLE = 5        # Improbable, contradicted by other info
    CANNOT_BE_JUDGED = 6  # Truth cannot be judged


@dataclass
class AdmiraltyRating:
    """
    Combined Admiralty/NATO confidence rating.
    Example: "B2" = Usually Reliable source, Probably True information
    """
    source_reliability: SourceReliability
    info_credibility: InformationCredibility

    def __str__(self) -> str:
        return f"{self.source_reliability.value}{self.info_credibility.value}"

    @property
    def confidence_score(self) -> float:
        """Convert to 0-1 confidence score."""
        # Source reliability scores
        source_scores = {"A": 1.0, "B": 0.8, "C": 0.6, "D": 0.4, "E": 0.2, "F": 0.5}
        # Information credibility scores
        info_scores = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.2, 6: 0.5}

        source_score = source_scores.get(self.source_reliability.value, 0.5)
        info_score = info_scores.get(self.info_credibility.value, 0.5)

        # Combined score (geometric mean)
        return math.sqrt(source_score * info_score)

    @classmethod
    def from_string(cls, rating: str) -> "AdmiraltyRating":
        """Parse rating string like 'B2' or 'A1'."""
        if len(rating) != 2:
            raise ValueError(f"Invalid Admiralty rating: {rating}")

        source = SourceReliability(rating[0].upper())
        info = InformationCredibility(int(rating[1]))

        return cls(source, info)


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

class SourceType(str, Enum):
    """Types of intelligence sources."""
    # Traditional INT types
    HUMINT = "HUMINT"      # Human Intelligence
    SIGINT = "SIGINT"      # Signals Intelligence
    IMINT = "IMINT"        # Imagery Intelligence
    MASINT = "MASINT"      # Measurement & Signature
    GEOINT = "GEOINT"      # Geospatial Intelligence
    OSINT = "OSINT"        # Open Source Intelligence
    TECHINT = "TECHINT"    # Technical Intelligence
    FININT = "FININT"      # Financial Intelligence
    CYBERINT = "CYBERINT"  # Cyber Intelligence

    # OSINT sub-types
    NEWS_WIRE = "NEWS_WIRE"          # AP, Reuters, AFP
    NEWS_BROADCAST = "NEWS_BROADCAST" # CNN, BBC, etc.
    NEWS_PRINT = "NEWS_PRINT"        # NYT, WSJ, etc.
    SOCIAL_MEDIA = "SOCIAL_MEDIA"    # Twitter/X, etc.
    GOVERNMENT = "GOVERNMENT"        # Official gov sources
    ACADEMIC = "ACADEMIC"            # Research papers
    NGO = "NGO"                      # NGO reports
    CORPORATE = "CORPORATE"          # Company filings/PR
    GDELT = "GDELT"                  # GDELT processed data
    AGGREGATED = "AGGREGATED"        # Multiple sources


class MediaBias(str, Enum):
    """Media bias classification."""
    FAR_LEFT = "far_left"
    LEFT = "left"
    CENTER_LEFT = "center_left"
    CENTER = "center"
    CENTER_RIGHT = "center_right"
    RIGHT = "right"
    FAR_RIGHT = "far_right"
    STATE_CONTROLLED = "state_controlled"
    UNKNOWN = "unknown"


@dataclass
class SourceProfile:
    """Profile of an information source."""
    source_id: str
    name: str
    source_type: SourceType
    country: str = ""
    bias: MediaBias = MediaBias.UNKNOWN

    # Reliability metrics (0-1)
    accuracy_score: float = 0.5      # Historical accuracy
    timeliness_score: float = 0.5    # How quickly they report
    depth_score: float = 0.5         # Depth of coverage
    verification_score: float = 0.5   # Fact-checking practices

    # Track record
    total_reports: int = 0
    confirmed_accurate: int = 0
    confirmed_inaccurate: int = 0
    retractions: int = 0

    # Metadata
    established_year: int = 2000
    reach: str = "national"  # local, national, international
    language: str = "en"
    tags: Set[str] = field(default_factory=set)

    @property
    def reliability_rating(self) -> SourceReliability:
        """Calculate source reliability rating."""
        if self.total_reports == 0:
            return SourceReliability.F

        accuracy = self.confirmed_accurate / max(1, self.confirmed_accurate + self.confirmed_inaccurate)
        combined = (self.accuracy_score * 0.4 + accuracy * 0.4 +
                   self.verification_score * 0.2)

        if combined >= 0.9:
            return SourceReliability.A
        elif combined >= 0.75:
            return SourceReliability.B
        elif combined >= 0.55:
            return SourceReliability.C
        elif combined >= 0.35:
            return SourceReliability.D
        else:
            return SourceReliability.E


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-DEFINED SOURCE PROFILES
# ═══════════════════════════════════════════════════════════════════════════════

# Major wire services (generally most reliable for breaking news)
WIRE_SERVICES: Dict[str, SourceProfile] = {
    "AP": SourceProfile("AP", "Associated Press", SourceType.NEWS_WIRE, "US",
                        MediaBias.CENTER, 0.9, 0.95, 0.7, 0.9, reach="international"),
    "REUTERS": SourceProfile("REUTERS", "Reuters", SourceType.NEWS_WIRE, "GB",
                             MediaBias.CENTER, 0.9, 0.95, 0.75, 0.9, reach="international"),
    "AFP": SourceProfile("AFP", "Agence France-Presse", SourceType.NEWS_WIRE, "FR",
                         MediaBias.CENTER, 0.85, 0.9, 0.7, 0.85, reach="international"),
    "XINHUA": SourceProfile("XINHUA", "Xinhua News Agency", SourceType.NEWS_WIRE, "CN",
                            MediaBias.STATE_CONTROLLED, 0.6, 0.85, 0.7, 0.4, reach="international"),
    "TASS": SourceProfile("TASS", "TASS", SourceType.NEWS_WIRE, "RU",
                          MediaBias.STATE_CONTROLLED, 0.5, 0.8, 0.6, 0.3, reach="international"),
}

# Major broadcasters
BROADCASTERS: Dict[str, SourceProfile] = {
    "BBC": SourceProfile("BBC", "BBC News", SourceType.NEWS_BROADCAST, "GB",
                         MediaBias.CENTER_LEFT, 0.85, 0.85, 0.8, 0.85, reach="international"),
    "CNN": SourceProfile("CNN", "CNN", SourceType.NEWS_BROADCAST, "US",
                         MediaBias.CENTER_LEFT, 0.75, 0.9, 0.7, 0.7, reach="international"),
    "FOX": SourceProfile("FOX", "Fox News", SourceType.NEWS_BROADCAST, "US",
                         MediaBias.RIGHT, 0.65, 0.85, 0.6, 0.5, reach="national"),
    "ALJAZEERA": SourceProfile("ALJAZEERA", "Al Jazeera", SourceType.NEWS_BROADCAST, "QA",
                               MediaBias.CENTER, 0.75, 0.8, 0.8, 0.7, reach="international"),
    "RT": SourceProfile("RT", "RT", SourceType.NEWS_BROADCAST, "RU",
                        MediaBias.STATE_CONTROLLED, 0.4, 0.75, 0.6, 0.2, reach="international"),
    "CGTN": SourceProfile("CGTN", "CGTN", SourceType.NEWS_BROADCAST, "CN",
                          MediaBias.STATE_CONTROLLED, 0.5, 0.7, 0.6, 0.3, reach="international"),
}

# Major newspapers
NEWSPAPERS: Dict[str, SourceProfile] = {
    "NYT": SourceProfile("NYT", "New York Times", SourceType.NEWS_PRINT, "US",
                         MediaBias.CENTER_LEFT, 0.85, 0.7, 0.9, 0.85, reach="international"),
    "WSJ": SourceProfile("WSJ", "Wall Street Journal", SourceType.NEWS_PRINT, "US",
                         MediaBias.CENTER_RIGHT, 0.85, 0.75, 0.85, 0.85, reach="international"),
    "WAPO": SourceProfile("WAPO", "Washington Post", SourceType.NEWS_PRINT, "US",
                          MediaBias.CENTER_LEFT, 0.8, 0.75, 0.85, 0.8, reach="international"),
    "FT": SourceProfile("FT", "Financial Times", SourceType.NEWS_PRINT, "GB",
                        MediaBias.CENTER, 0.9, 0.7, 0.9, 0.9, reach="international"),
    "GUARDIAN": SourceProfile("GUARDIAN", "The Guardian", SourceType.NEWS_PRINT, "GB",
                              MediaBias.LEFT, 0.75, 0.75, 0.8, 0.75, reach="international"),
    "ECONOMIST": SourceProfile("ECONOMIST", "The Economist", SourceType.NEWS_PRINT, "GB",
                               MediaBias.CENTER, 0.85, 0.5, 0.95, 0.85, reach="international"),
}

# Government/Official sources
GOVERNMENT_SOURCES: Dict[str, SourceProfile] = {
    "WHITEHOUSE": SourceProfile("WHITEHOUSE", "White House", SourceType.GOVERNMENT, "US",
                                MediaBias.UNKNOWN, 0.8, 0.9, 0.6, 0.7, reach="international"),
    "STATE_DEPT": SourceProfile("STATE_DEPT", "US State Department", SourceType.GOVERNMENT, "US",
                                MediaBias.UNKNOWN, 0.8, 0.8, 0.7, 0.7, reach="international"),
    "PENTAGON": SourceProfile("PENTAGON", "US DoD/Pentagon", SourceType.GOVERNMENT, "US",
                              MediaBias.UNKNOWN, 0.75, 0.8, 0.6, 0.6, reach="international"),
    "KREMLIN": SourceProfile("KREMLIN", "Kremlin", SourceType.GOVERNMENT, "RU",
                             MediaBias.STATE_CONTROLLED, 0.4, 0.7, 0.5, 0.3, reach="international"),
    "FOREIGN_MINISTRY_CN": SourceProfile("FMPRC", "Chinese Foreign Ministry", SourceType.GOVERNMENT, "CN",
                                         MediaBias.STATE_CONTROLLED, 0.5, 0.75, 0.5, 0.3, reach="international"),
}


def get_source_profile(source_id: str) -> Optional[SourceProfile]:
    """Look up source profile by ID."""
    all_sources = {**WIRE_SERVICES, **BROADCASTERS, **NEWSPAPERS, **GOVERNMENT_SOURCES}
    return all_sources.get(source_id.upper())


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIDENCE ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConfidenceFactors:
    """Factors contributing to confidence assessment."""
    source_reliability: float = 0.5     # 0-1, from source profile
    corroboration_count: int = 0        # Number of independent sources
    recency_hours: float = 0            # Hours since report
    geographic_proximity: float = 0.5   # 0-1, reporter proximity to event
    specificity: float = 0.5            # 0-1, level of detail
    internal_consistency: float = 1.0   # 0-1, self-contradictions
    expert_consensus: float = 0.5       # 0-1, expert agreement
    historical_pattern: float = 0.5     # 0-1, fits historical patterns


@dataclass
class ConfidenceAssessment:
    """
    Complete confidence assessment for a piece of intelligence.
    """
    # Core ratings
    admiralty_rating: AdmiraltyRating
    confidence_score: float  # 0-1 overall confidence

    # Contributing factors
    factors: ConfidenceFactors

    # Metadata
    assessed_at: datetime = field(default_factory=datetime.utcnow)
    assessed_by: str = "SYSTEM"
    assessment_notes: str = ""

    # Source tracking
    primary_source_id: str = ""
    corroborating_sources: List[str] = field(default_factory=list)

    # Uncertainty
    confidence_interval: Tuple[float, float] = (0.0, 1.0)  # Low, high bounds

    def decay(self, hours_elapsed: float, half_life_hours: float = 24.0) -> float:
        """
        Apply temporal decay to confidence.
        Information becomes less certain over time without confirmation.
        """
        decay_factor = 0.5 ** (hours_elapsed / half_life_hours)
        return self.confidence_score * decay_factor

    @classmethod
    def calculate(cls, factors: ConfidenceFactors,
                  primary_source: Optional[SourceProfile] = None) -> "ConfidenceAssessment":
        """Calculate confidence assessment from factors."""

        # Determine source reliability rating
        if primary_source:
            source_rel = primary_source.reliability_rating
            factors.source_reliability = {"A": 1.0, "B": 0.8, "C": 0.6,
                                         "D": 0.4, "E": 0.2, "F": 0.5}[source_rel.value]
        else:
            source_rel = SourceReliability.F

        # Calculate corroboration bonus
        corroboration_bonus = min(0.3, factors.corroboration_count * 0.1)

        # Calculate recency penalty (half-life of 48 hours)
        recency_factor = 0.5 ** (factors.recency_hours / 48.0)

        # Weighted combination
        base_score = (
            factors.source_reliability * 0.25 +
            factors.specificity * 0.15 +
            factors.internal_consistency * 0.15 +
            factors.geographic_proximity * 0.1 +
            factors.expert_consensus * 0.15 +
            factors.historical_pattern * 0.1 +
            corroboration_bonus
        )

        # Apply recency factor
        final_score = base_score * recency_factor

        # Determine information credibility rating
        if final_score >= 0.85:
            info_cred = InformationCredibility.CONFIRMED
        elif final_score >= 0.7:
            info_cred = InformationCredibility.PROBABLY_TRUE
        elif final_score >= 0.5:
            info_cred = InformationCredibility.POSSIBLY_TRUE
        elif final_score >= 0.35:
            info_cred = InformationCredibility.DOUBTFUL
        elif final_score >= 0.2:
            info_cred = InformationCredibility.IMPROBABLE
        else:
            info_cred = InformationCredibility.CANNOT_BE_JUDGED

        # Calculate confidence interval
        uncertainty = 0.15 * (1 - final_score) + 0.1
        ci_low = max(0, final_score - uncertainty)
        ci_high = min(1, final_score + uncertainty)

        return cls(
            admiralty_rating=AdmiraltyRating(source_rel, info_cred),
            confidence_score=final_score,
            factors=factors,
            primary_source_id=primary_source.source_id if primary_source else "",
            confidence_interval=(ci_low, ci_high)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# GDELT-SPECIFIC CONFIDENCE MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

def assess_gdelt_event(event_data: Dict[str, Any],
                       source_profile: Optional[SourceProfile] = None) -> ConfidenceAssessment:
    """
    Calculate confidence assessment for a GDELT event.

    Uses GDELT's built-in metrics plus our framework.
    """
    factors = ConfidenceFactors()

    # Extract GDELT confidence signals
    avg_tone = event_data.get("avg_tone", 0)
    num_mentions = event_data.get("num_mentions", 1)
    num_sources = event_data.get("num_sources", 1)
    num_articles = event_data.get("num_articles", 1)

    # Source reliability from profile or default
    if source_profile:
        factors.source_reliability = source_profile.accuracy_score
    else:
        factors.source_reliability = 0.5

    # Corroboration from GDELT mention count
    factors.corroboration_count = min(num_sources, 10)

    # Recency (GDELT updates every 15 minutes)
    event_date = event_data.get("date_added")
    if event_date:
        if isinstance(event_date, datetime):
            delta = datetime.now(timezone.utc) - event_date
            factors.recency_hours = delta.total_seconds() / 3600
        else:
            factors.recency_hours = 1.0  # Assume recent
    else:
        factors.recency_hours = 1.0

    # Specificity based on data completeness
    required_fields = ["actor1", "actor2", "event_code", "geo_lat", "geo_lon"]
    present_fields = sum(1 for f in required_fields if event_data.get(f))
    factors.specificity = present_fields / len(required_fields)

    # Internal consistency (tone extremity can indicate bias)
    tone_extremity = abs(avg_tone) / 20.0  # GDELT tone is roughly -20 to +20
    factors.internal_consistency = 1 - min(0.5, tone_extremity)

    # Calculate assessment
    return ConfidenceAssessment.calculate(factors, source_profile)


def assess_news_article(title: str, content: str, source_id: str,
                       published_at: datetime,
                       corroborating_urls: List[str] = None) -> ConfidenceAssessment:
    """Calculate confidence assessment for a news article."""
    factors = ConfidenceFactors()

    # Get source profile
    source_profile = get_source_profile(source_id)
    if source_profile:
        factors.source_reliability = source_profile.accuracy_score
    else:
        factors.source_reliability = 0.5

    # Corroboration
    factors.corroboration_count = len(corroborating_urls or [])

    # Recency
    delta = datetime.now(timezone.utc) - published_at
    factors.recency_hours = delta.total_seconds() / 3600

    # Specificity heuristics (presence of specific details)
    specificity_indicators = [
        any(c.isdigit() for c in content[:500]),  # Contains numbers
        "said" in content.lower()[:500],           # Has quotes
        "according to" in content.lower(),         # Attribution
        len(content) > 500,                        # Sufficient length
    ]
    factors.specificity = sum(specificity_indicators) / len(specificity_indicators)

    # Default other factors
    factors.internal_consistency = 0.8
    factors.expert_consensus = 0.5
    factors.historical_pattern = 0.5

    return ConfidenceAssessment.calculate(factors, source_profile)


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFIED INTELLIGENCE REPORT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IntelligenceReport:
    """
    A classified intelligence report with full markings and confidence.
    """
    report_id: str
    title: str
    summary: str

    # Classification
    classification: ClassificationMarking

    # Confidence
    confidence: ConfidenceAssessment

    # Content
    body: str = ""
    key_entities: List[str] = field(default_factory=list)
    locations: List[Tuple[float, float]] = field(default_factory=list)  # lat, lon
    mgrs_references: List[str] = field(default_factory=list)

    # Temporal
    event_time: Optional[datetime] = None
    report_time: datetime = field(default_factory=datetime.utcnow)
    valid_until: Optional[datetime] = None

    # Source tracking
    sources: List[str] = field(default_factory=list)
    collection_method: SourceType = SourceType.OSINT

    # Metadata
    tags: Set[str] = field(default_factory=set)
    related_reports: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Format report header."""
        return f"""
{self.classification}
REPORT ID: {self.report_id}
DATE: {self.report_time.strftime('%Y-%m-%d %H:%M:%S')}Z
CONFIDENCE: {self.confidence.admiralty_rating} ({self.confidence.confidence_score:.0%})

TITLE: {self.title}

SUMMARY: {self.summary}
"""

    @classmethod
    def from_gdelt_event(cls, event_data: Dict[str, Any],
                         report_id: Optional[str] = None) -> "IntelligenceReport":
        """Create intelligence report from GDELT event."""
        # Generate report ID
        if not report_id:
            event_hash = hashlib.md5(str(event_data).encode()).hexdigest()[:12]
            report_id = f"GDELT-{event_hash}"

        # Get confidence assessment
        confidence = assess_gdelt_event(event_data)

        # Extract entities
        entities = []
        if event_data.get("actor1_name"):
            entities.append(event_data["actor1_name"])
        if event_data.get("actor2_name"):
            entities.append(event_data["actor2_name"])

        # Extract location
        locations = []
        mgrs_refs = []
        if event_data.get("geo_lat") and event_data.get("geo_lon"):
            lat, lon = event_data["geo_lat"], event_data["geo_lon"]
            locations.append((lat, lon))
            # Import here to avoid circular dependency
            try:
                from .geospatial import coords_to_mgrs
                mgrs_refs.append(coords_to_mgrs(lat, lon, precision=3))
            except ImportError:
                pass

        return cls(
            report_id=report_id,
            title=f"Event: {event_data.get('event_code', 'Unknown')}",
            summary=f"{entities[0] if entities else 'Unknown'} -> {entities[1] if len(entities) > 1 else 'Unknown'}",
            classification=ClassificationMarking.from_osint(),
            confidence=confidence,
            key_entities=entities,
            locations=locations,
            mgrs_references=mgrs_refs,
            collection_method=SourceType.GDELT,
            tags={"gdelt", "auto-generated"}
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIDENCE AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

class ConfidenceAggregator:
    """Aggregate confidence assessments from multiple sources."""

    def __init__(self):
        self.assessments: List[ConfidenceAssessment] = []
        self.source_weights: Dict[str, float] = {}

    def add_assessment(self, assessment: ConfidenceAssessment, weight: float = 1.0):
        """Add an assessment with optional weight."""
        self.assessments.append(assessment)
        if assessment.primary_source_id:
            self.source_weights[assessment.primary_source_id] = weight

    def aggregate(self) -> ConfidenceAssessment:
        """Calculate aggregate confidence from all assessments."""
        if not self.assessments:
            return ConfidenceAssessment(
                admiralty_rating=AdmiraltyRating(SourceReliability.F,
                                                 InformationCredibility.CANNOT_BE_JUDGED),
                confidence_score=0.0,
                factors=ConfidenceFactors()
            )

        # Weighted average of confidence scores
        total_weight = sum(self.source_weights.get(a.primary_source_id, 1.0)
                          for a in self.assessments)

        weighted_score = sum(
            a.confidence_score * self.source_weights.get(a.primary_source_id, 1.0)
            for a in self.assessments
        ) / total_weight

        # Corroboration bonus (diminishing returns)
        unique_sources = len(set(a.primary_source_id for a in self.assessments
                                if a.primary_source_id))
        corroboration_bonus = min(0.2, unique_sources * 0.05)
        final_score = min(1.0, weighted_score + corroboration_bonus)

        # Aggregate factors
        agg_factors = ConfidenceFactors(
            source_reliability=sum(a.factors.source_reliability for a in self.assessments) / len(self.assessments),
            corroboration_count=unique_sources,
            recency_hours=min(a.factors.recency_hours for a in self.assessments),
            specificity=max(a.factors.specificity for a in self.assessments),
            internal_consistency=sum(a.factors.internal_consistency for a in self.assessments) / len(self.assessments),
        )

        # Determine ratings
        if final_score >= 0.85:
            source_rel = SourceReliability.A
            info_cred = InformationCredibility.CONFIRMED
        elif final_score >= 0.7:
            source_rel = SourceReliability.B
            info_cred = InformationCredibility.PROBABLY_TRUE
        elif final_score >= 0.5:
            source_rel = SourceReliability.C
            info_cred = InformationCredibility.POSSIBLY_TRUE
        elif final_score >= 0.35:
            source_rel = SourceReliability.D
            info_cred = InformationCredibility.DOUBTFUL
        else:
            source_rel = SourceReliability.E
            info_cred = InformationCredibility.IMPROBABLE

        return ConfidenceAssessment(
            admiralty_rating=AdmiraltyRating(source_rel, info_cred),
            confidence_score=final_score,
            factors=agg_factors,
            corroborating_sources=[a.primary_source_id for a in self.assessments if a.primary_source_id],
            assessment_notes=f"Aggregated from {len(self.assessments)} sources"
        )
