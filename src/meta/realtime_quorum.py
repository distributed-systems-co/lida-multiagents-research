"""Real-time Prediction Engine with Emotional Quorum Integration.

Connects GDELT events to the industrial intelligence system,
generates predictions, and runs agent deliberations in real-time.

Provides:
- Real-time GDELT event processing
- Predictive modeling for acquisitions, market moves
- Live emotional quorum deliberations
- Signal aggregation and alerting
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .industrial_intelligence import (
    AcquisitionSignal,
    EmotionalQuorum,
    EmotionalStance,
    GDELTIndustrialProcessor,
    IndustrialCompany,
    IndustrialEvent,
    IndustrialEventType,
    IndustrialRegistry,
    IndustrialSector,
    QuorumDeliberation,
    get_industrial_registry,
)
from .timeline import (
    EventSeverity,
    EventType,
    TimelineEvent,
    get_timeline_store,
    record_event,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class PredictionType(str, Enum):
    """Types of predictions the system can make."""
    ACQUISITION = "acquisition"
    IPO = "ipo"
    FUNDING_ROUND = "funding_round"
    VALUATION_CHANGE = "valuation_change"
    MARKET_SHIFT = "market_shift"
    LEADERSHIP_CHANGE = "leadership_change"
    PARTNERSHIP = "partnership"
    REGULATORY_ACTION = "regulatory_action"
    SECTOR_CONSOLIDATION = "sector_consolidation"
    BANKRUPTCY_RISK = "bankruptcy_risk"


class PredictionConfidence(str, Enum):
    """Confidence levels for predictions."""
    VERY_HIGH = "very_high"    # 80-100%
    HIGH = "high"              # 60-80%
    MEDIUM = "medium"          # 40-60%
    LOW = "low"                # 20-40%
    SPECULATIVE = "speculative"  # <20%


@dataclass
class Prediction:
    """A prediction about future events."""
    prediction_id: str
    prediction_type: PredictionType
    timestamp: datetime

    # Target
    target_company: str
    target_sector: Optional[IndustrialSector] = None

    # Prediction details
    description: str = ""
    predicted_outcome: str = ""
    confidence: PredictionConfidence = PredictionConfidence.MEDIUM
    probability: float = 0.5

    # Timing
    time_horizon_days: int = 90
    earliest_date: Optional[datetime] = None
    latest_date: Optional[datetime] = None

    # Supporting evidence
    signals: List[AcquisitionSignal] = field(default_factory=list)
    supporting_events: List[str] = field(default_factory=list)  # event IDs
    quorum_deliberation_id: Optional[str] = None

    # If acquisition
    likely_acquirers: List[str] = field(default_factory=list)
    estimated_price_billions: float = 0.0
    price_range: Tuple[float, float] = (0.0, 0.0)

    # Status
    is_resolved: bool = False
    actual_outcome: str = ""
    resolution_date: Optional[datetime] = None
    was_correct: Optional[bool] = None

    # Metadata
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "prediction_id": self.prediction_id,
            "prediction_type": self.prediction_type.value,
            "timestamp": self.timestamp.isoformat(),
            "target_company": self.target_company,
            "target_sector": self.target_sector.value if self.target_sector else None,
            "description": self.description,
            "predicted_outcome": self.predicted_outcome,
            "confidence": self.confidence.value,
            "probability": self.probability,
            "time_horizon_days": self.time_horizon_days,
            "signals": [s.value for s in self.signals],
            "likely_acquirers": self.likely_acquirers,
            "estimated_price_billions": self.estimated_price_billions,
            "is_resolved": self.is_resolved,
            "was_correct": self.was_correct,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# REAL-TIME EVENT STREAM
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RealTimeEvent:
    """A real-time event from various sources."""
    event_id: str
    source: str  # gdelt, news_api, sec_filing, twitter, etc.
    timestamp: datetime

    # Content
    headline: str = ""
    summary: str = ""
    full_text: str = ""

    # Entities
    companies_mentioned: List[str] = field(default_factory=list)
    people_mentioned: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)

    # Classification
    event_type: Optional[IndustrialEventType] = None
    sector: Optional[IndustrialSector] = None
    sentiment: float = 0.0  # -1 to 1
    importance: float = 0.5  # 0 to 1

    # Source metadata
    source_url: str = ""
    source_credibility: float = 0.8

    # Processing status
    processed: bool = False
    quorum_deliberation_id: Optional[str] = None
    predictions_generated: List[str] = field(default_factory=list)


class EventStream:
    """Manages real-time event ingestion and processing."""

    def __init__(self, gdelt_data_path: Optional[Path] = None):
        self.gdelt_data_path = gdelt_data_path or Path.home() / "lida-multiagents-research" / ".gdelt_data"
        self._events: List[RealTimeEvent] = []
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._subscribers: List[Callable[[RealTimeEvent], Any]] = []
        self._running = False
        self._lock = asyncio.Lock()

    async def start(self):
        """Start the event stream."""
        self._running = True
        logger.info("Event stream started")

    async def stop(self):
        """Stop the event stream."""
        self._running = False
        logger.info("Event stream stopped")

    def subscribe(self, callback: Callable[[RealTimeEvent], Any]):
        """Subscribe to event stream."""
        self._subscribers.append(callback)

    async def publish(self, event: RealTimeEvent):
        """Publish an event to subscribers."""
        async with self._lock:
            self._events.append(event)

        for subscriber in self._subscribers:
            try:
                result = subscriber(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Error in event subscriber: {e}")

    async def ingest_gdelt_event(self, gdelt_data: dict) -> Optional[RealTimeEvent]:
        """Convert GDELT data to RealTimeEvent."""
        try:
            event = RealTimeEvent(
                event_id=str(gdelt_data.get("GLOBALEVENTID", uuid.uuid4())),
                source="gdelt",
                timestamp=datetime.utcnow(),
                headline=f"{gdelt_data.get('Actor1Name', 'Unknown')} - {gdelt_data.get('Actor2Name', 'Unknown')}",
                companies_mentioned=[
                    gdelt_data.get("Actor1Name", ""),
                    gdelt_data.get("Actor2Name", ""),
                ],
                source_url=gdelt_data.get("SOURCEURL", ""),
                sentiment=gdelt_data.get("AvgTone", 0) / 10,  # Normalize
                importance=min(1.0, gdelt_data.get("NumMentions", 1) / 100),
            )

            await self.publish(event)
            return event

        except Exception as e:
            logger.error(f"Error ingesting GDELT event: {e}")
            return None

    async def get_recent_events(self, limit: int = 100) -> List[RealTimeEvent]:
        """Get recent events."""
        async with self._lock:
            return sorted(
                self._events,
                key=lambda e: e.timestamp,
                reverse=True
            )[:limit]


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class PredictionEngine:
    """Generates predictions based on events and patterns."""

    # Known acquisition patterns from historical data
    ACQUISITION_PATTERNS = {
        "ai_foundation": {
            "typical_acquirers": ["Microsoft", "Google", "Amazon", "Meta", "Apple", "Nvidia"],
            "typical_multiple": 25,  # Revenue multiple
            "avg_time_to_exit_years": 4,
        },
        "ai_chips": {
            "typical_acquirers": ["Nvidia", "Intel", "AMD", "Qualcomm", "Broadcom"],
            "typical_multiple": 15,
            "avg_time_to_exit_years": 5,
        },
        "ai_enterprise": {
            "typical_acquirers": ["Salesforce", "Microsoft", "ServiceNow", "Oracle", "SAP"],
            "typical_multiple": 12,
            "avg_time_to_exit_years": 5,
        },
        "ai_coding": {
            "typical_acquirers": ["Microsoft", "Google", "Atlassian", "JetBrains"],
            "typical_multiple": 20,
            "avg_time_to_exit_years": 3,
        },
        "ai_media": {
            "typical_acquirers": ["Adobe", "Spotify", "Apple", "Amazon", "Netflix"],
            "typical_multiple": 15,
            "avg_time_to_exit_years": 4,
        },
        "defense": {
            "typical_acquirers": ["Lockheed Martin", "Raytheon", "Northrop Grumman", "L3Harris"],
            "typical_multiple": 8,
            "avg_time_to_exit_years": 6,
        },
    }

    # Signal weights for acquisition probability
    SIGNAL_WEIGHTS = {
        AcquisitionSignal.GROWTH_STALLING: 0.15,
        AcquisitionSignal.COMPETITIVE_PRESSURE: 0.12,
        AcquisitionSignal.INBOUND_INTEREST: 0.25,
        AcquisitionSignal.STRATEGIC_REVIEW: 0.20,
        AcquisitionSignal.FOUNDER_FATIGUE: 0.10,
        AcquisitionSignal.KEY_TALENT_LEAVING: 0.08,
        AcquisitionSignal.CASH_RUNWAY_LOW: 0.18,
        AcquisitionSignal.MARKET_PEAK: 0.05,
        AcquisitionSignal.REGULATORY_TAILWIND: 0.07,
        AcquisitionSignal.STRATEGIC_FIT: 0.15,
        AcquisitionSignal.BIDDING_WAR: 0.20,
    }

    def __init__(self, registry: IndustrialRegistry):
        self.registry = registry
        self._predictions: Dict[str, Prediction] = {}
        self._lock = asyncio.Lock()

    async def analyze_company(
        self,
        company: IndustrialCompany,
        recent_events: List[IndustrialEvent],
    ) -> List[Prediction]:
        """Analyze a company and generate predictions."""
        predictions = []

        # Calculate acquisition probability
        acq_probability = await self._calculate_acquisition_probability(company, recent_events)

        if acq_probability > 0.3:
            pred = await self._generate_acquisition_prediction(company, acq_probability)
            predictions.append(pred)

        # Check for IPO signals
        if await self._check_ipo_signals(company):
            pred = await self._generate_ipo_prediction(company)
            predictions.append(pred)

        # Check for funding signals
        funding_pred = await self._check_funding_signals(company, recent_events)
        if funding_pred:
            predictions.append(funding_pred)

        # Store predictions
        async with self._lock:
            for pred in predictions:
                self._predictions[pred.prediction_id] = pred

        return predictions

    async def _calculate_acquisition_probability(
        self,
        company: IndustrialCompany,
        recent_events: List[IndustrialEvent],
    ) -> float:
        """Calculate probability of acquisition."""
        base_probability = 0.1

        # Add signal weights
        for signal in company.acquisition_signals:
            base_probability += self.SIGNAL_WEIGHTS.get(signal, 0.05)

        # Adjust based on company characteristics
        if company.is_public:
            base_probability *= 0.7  # Harder to acquire public companies

        if company.valuation_billions > 50:
            base_probability *= 0.5  # Very expensive targets less likely

        # Check recent events for acquisition signals
        for event in recent_events:
            if event.primary_company == company.name:
                if event.event_type == IndustrialEventType.LAYOFFS:
                    base_probability += 0.1
                elif event.event_type == IndustrialEventType.KEY_DEPARTURE:
                    base_probability += 0.08
                elif event.event_type == IndustrialEventType.PARTNERSHIP_ANNOUNCED:
                    base_probability += 0.05  # Could be acquirer relationship building

        return min(0.95, base_probability)

    async def _generate_acquisition_prediction(
        self,
        company: IndustrialCompany,
        probability: float,
    ) -> Prediction:
        """Generate an acquisition prediction."""
        sector_patterns = self.ACQUISITION_PATTERNS.get(company.sector.value, {})
        typical_acquirers = sector_patterns.get("typical_acquirers", [])
        typical_multiple = sector_patterns.get("typical_multiple", 10)

        # Estimate price
        if company.arr_millions > 0:
            estimated_price = company.arr_millions * typical_multiple / 1000
        else:
            estimated_price = company.valuation_billions * 1.3  # 30% premium

        # Determine confidence
        if probability > 0.7:
            confidence = PredictionConfidence.HIGH
        elif probability > 0.5:
            confidence = PredictionConfidence.MEDIUM
        else:
            confidence = PredictionConfidence.LOW

        # Rank likely acquirers
        likely_acquirers = []
        for acquirer in typical_acquirers:
            if acquirer in company.partners:
                likely_acquirers.insert(0, acquirer)  # Partners more likely
            else:
                likely_acquirers.append(acquirer)

        return Prediction(
            prediction_id=str(uuid.uuid4()),
            prediction_type=PredictionType.ACQUISITION,
            timestamp=datetime.utcnow(),
            target_company=company.name,
            target_sector=company.sector,
            description=f"{company.name} likely to be acquired",
            predicted_outcome=f"Acquisition by {likely_acquirers[0] if likely_acquirers else 'strategic buyer'}",
            confidence=confidence,
            probability=probability,
            time_horizon_days=365,
            signals=list(company.acquisition_signals),
            likely_acquirers=likely_acquirers[:5],
            estimated_price_billions=estimated_price,
            price_range=(estimated_price * 0.7, estimated_price * 1.5),
        )

    async def _check_ipo_signals(self, company: IndustrialCompany) -> bool:
        """Check if company shows IPO signals."""
        if company.is_public or company.is_acquired:
            return False

        # IPO criteria
        if company.valuation_billions >= 5 and company.arr_millions >= 200:
            return True
        if company.valuation_billions >= 10:
            return True
        if "ipo_2026" in company.tags:
            return True

        return False

    async def _generate_ipo_prediction(self, company: IndustrialCompany) -> Prediction:
        """Generate IPO prediction."""
        return Prediction(
            prediction_id=str(uuid.uuid4()),
            prediction_type=PredictionType.IPO,
            timestamp=datetime.utcnow(),
            target_company=company.name,
            target_sector=company.sector,
            description=f"{company.name} likely to IPO",
            predicted_outcome="Public listing",
            confidence=PredictionConfidence.MEDIUM,
            probability=0.6,
            time_horizon_days=365,
            estimated_price_billions=company.valuation_billions * 1.2,
        )

    async def _check_funding_signals(
        self,
        company: IndustrialCompany,
        recent_events: List[IndustrialEvent],
    ) -> Optional[Prediction]:
        """Check for upcoming funding round signals."""
        if company.is_public:
            return None

        # Check if company is due for funding
        if company.last_funding_date:
            months_since_funding = (datetime.utcnow() - company.last_funding_date).days / 30
            if months_since_funding > 18:
                return Prediction(
                    prediction_id=str(uuid.uuid4()),
                    prediction_type=PredictionType.FUNDING_ROUND,
                    timestamp=datetime.utcnow(),
                    target_company=company.name,
                    target_sector=company.sector,
                    description=f"{company.name} due for new funding round",
                    predicted_outcome="Series raise",
                    confidence=PredictionConfidence.MEDIUM,
                    probability=0.5,
                    time_horizon_days=180,
                )

        return None

    async def get_predictions(
        self,
        prediction_type: Optional[PredictionType] = None,
        sector: Optional[IndustrialSector] = None,
        min_probability: float = 0.0,
    ) -> List[Prediction]:
        """Get predictions with optional filters."""
        async with self._lock:
            predictions = list(self._predictions.values())

        if prediction_type:
            predictions = [p for p in predictions if p.prediction_type == prediction_type]

        if sector:
            predictions = [p for p in predictions if p.target_sector == sector]

        predictions = [p for p in predictions if p.probability >= min_probability]

        return sorted(predictions, key=lambda p: p.probability, reverse=True)

    async def resolve_prediction(
        self,
        prediction_id: str,
        actual_outcome: str,
        was_correct: bool,
    ):
        """Mark a prediction as resolved."""
        async with self._lock:
            if prediction_id in self._predictions:
                pred = self._predictions[prediction_id]
                pred.is_resolved = True
                pred.actual_outcome = actual_outcome
                pred.was_correct = was_correct
                pred.resolution_date = datetime.utcnow()


# ═══════════════════════════════════════════════════════════════════════════════
# REAL-TIME QUORUM ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class QuorumOrchestrator:
    """Orchestrates real-time quorum deliberations on events."""

    def __init__(
        self,
        registry: IndustrialRegistry,
        prediction_engine: PredictionEngine,
        event_stream: EventStream,
    ):
        self.registry = registry
        self.prediction_engine = prediction_engine
        self.event_stream = event_stream
        self.quorum = registry.quorum

        self._deliberation_history: List[QuorumDeliberation] = []
        self._alert_callbacks: List[Callable[[QuorumDeliberation], Any]] = []
        self._running = False
        self._lock = asyncio.Lock()

        # Statistics
        self._stats = {
            "events_processed": 0,
            "deliberations_completed": 0,
            "predictions_generated": 0,
            "alerts_triggered": 0,
        }

    async def start(self):
        """Start the orchestrator."""
        self._running = True

        # Subscribe to event stream
        self.event_stream.subscribe(self._on_event)

        logger.info("Quorum orchestrator started")

    async def stop(self):
        """Stop the orchestrator."""
        self._running = False
        logger.info("Quorum orchestrator stopped")

    def on_alert(self, callback: Callable[[QuorumDeliberation], Any]):
        """Register alert callback."""
        self._alert_callbacks.append(callback)

    async def _on_event(self, event: RealTimeEvent):
        """Handle incoming event."""
        if not self._running:
            return

        self._stats["events_processed"] += 1

        # Convert to industrial event if relevant
        industrial_event = await self._classify_event(event)

        if industrial_event:
            # Run quorum deliberation
            deliberation = await self._deliberate(industrial_event)

            # Generate predictions based on deliberation
            predictions = await self._generate_predictions(industrial_event, deliberation)

            # Check for alerts
            await self._check_alerts(deliberation)

            # Record in timeline
            await self._record_timeline(industrial_event, deliberation)

    async def _classify_event(self, event: RealTimeEvent) -> Optional[IndustrialEvent]:
        """Classify a real-time event as an industrial event."""
        if not event.companies_mentioned:
            return None

        # Simple classification based on keywords
        headline_lower = event.headline.lower()

        event_type = None
        if any(word in headline_lower for word in ["acquire", "acquisition", "buys", "bought"]):
            event_type = IndustrialEventType.ACQUISITION_ANNOUNCED
        elif any(word in headline_lower for word in ["merger", "merge"]):
            event_type = IndustrialEventType.MERGER_ANNOUNCED
        elif any(word in headline_lower for word in ["ipo", "goes public", "public offering"]):
            event_type = IndustrialEventType.IPO_FILED
        elif any(word in headline_lower for word in ["raises", "funding", "series", "investment"]):
            event_type = IndustrialEventType.FUNDING_ROUND
        elif any(word in headline_lower for word in ["layoff", "cuts", "restructur"]):
            event_type = IndustrialEventType.LAYOFFS
        elif any(word in headline_lower for word in ["bankrupt", "chapter 11"]):
            event_type = IndustrialEventType.BANKRUPTCY
        elif any(word in headline_lower for word in ["partnership", "partner", "collaborat"]):
            event_type = IndustrialEventType.PARTNERSHIP_ANNOUNCED
        elif any(word in headline_lower for word in ["ceo", "appoint", "executive"]):
            event_type = IndustrialEventType.CEO_CHANGE

        if not event_type:
            return None

        # Determine market impact from sentiment and importance
        if event.importance > 0.7 or abs(event.sentiment) > 0.5:
            market_impact = "high"
        elif event.importance > 0.4:
            market_impact = "medium"
        else:
            market_impact = "low"

        return IndustrialEvent(
            event_id=event.event_id,
            event_type=event_type,
            timestamp=event.timestamp,
            primary_company=event.companies_mentioned[0] if event.companies_mentioned else "Unknown",
            secondary_company=event.companies_mentioned[1] if len(event.companies_mentioned) > 1 else None,
            title=event.headline,
            description=event.summary,
            source=event.source,
            source_url=event.source_url,
            market_impact=market_impact,
        )

    async def _deliberate(self, event: IndustrialEvent) -> QuorumDeliberation:
        """Run quorum deliberation on event."""
        deliberation = await self.quorum.deliberate(event)

        async with self._lock:
            self._deliberation_history.append(deliberation)
            # Keep last 1000 deliberations
            if len(self._deliberation_history) > 1000:
                self._deliberation_history = self._deliberation_history[-1000:]

        self._stats["deliberations_completed"] += 1

        return deliberation

    async def _generate_predictions(
        self,
        event: IndustrialEvent,
        deliberation: QuorumDeliberation,
    ) -> List[Prediction]:
        """Generate predictions based on event and deliberation."""
        predictions = []

        # Get company if tracked
        company = await self.registry.get_company(event.primary_company.lower().replace(" ", "_"))

        if company:
            predictions = await self.prediction_engine.analyze_company(
                company,
                [event],
            )

        self._stats["predictions_generated"] += len(predictions)

        return predictions

    async def _check_alerts(self, deliberation: QuorumDeliberation):
        """Check if deliberation warrants an alert."""
        should_alert = False

        # Alert on high urgency
        if deliberation.urgency_level in ["high", "critical"]:
            should_alert = True

        # Alert on strong consensus with alarmed stance
        if (deliberation.consensus_stance == EmotionalStance.ALARMED and
            deliberation.consensus_strength > 0.6):
            should_alert = True

        # Alert on excited stance with high consensus (opportunity)
        if (deliberation.consensus_stance == EmotionalStance.EXCITED and
            deliberation.consensus_strength > 0.7):
            should_alert = True

        if should_alert:
            self._stats["alerts_triggered"] += 1

            for callback in self._alert_callbacks:
                try:
                    result = callback(deliberation)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")

    async def _record_timeline(
        self,
        event: IndustrialEvent,
        deliberation: QuorumDeliberation,
    ):
        """Record event and deliberation in timeline."""
        timeline = get_timeline_store()

        # Record the industrial event
        await timeline.record(
            event_type=EventType.INDUSTRIAL_ACQUISITION if event.event_type == IndustrialEventType.ACQUISITION_ANNOUNCED else EventType.CUSTOM,
            title=event.title,
            description=event.description,
            severity=EventSeverity.WARNING if deliberation.urgency_level in ["high", "critical"] else EventSeverity.INFO,
            metadata={
                "industrial_event_id": event.event_id,
                "event_type": event.event_type.value,
                "primary_company": event.primary_company,
                "market_impact": event.market_impact,
            },
            tags={"industrial", event.event_type.value},
        )

        # Record the deliberation
        await timeline.record(
            event_type=EventType.QUORUM_DELIBERATION_END,
            title=f"Quorum: {deliberation.consensus_stance.value if deliberation.consensus_stance else 'mixed'}",
            description="; ".join(deliberation.key_insights[:3]),
            severity=EventSeverity.WARNING if deliberation.urgency_level in ["high", "critical"] else EventSeverity.INFO,
            metadata={
                "deliberation_id": deliberation.deliberation_id,
                "consensus_strength": deliberation.consensus_strength,
                "dissent_level": deliberation.dissent_level,
                "urgency_level": deliberation.urgency_level,
            },
            tags={"quorum", deliberation.urgency_level},
        )

    async def get_stats(self) -> dict:
        """Get orchestrator statistics."""
        return {
            **self._stats,
            "deliberation_history_size": len(self._deliberation_history),
        }

    async def get_recent_deliberations(self, limit: int = 20) -> List[QuorumDeliberation]:
        """Get recent deliberations."""
        async with self._lock:
            return sorted(
                self._deliberation_history,
                key=lambda d: d.timestamp,
                reverse=True
            )[:limit]


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION MODE
# ═══════════════════════════════════════════════════════════════════════════════

class EventSimulator:
    """Simulates real-time events for testing and demonstration."""

    SAMPLE_EVENTS = [
        {
            "headline": "Nvidia in talks to acquire AI chip startup Cerebras",
            "companies": ["Nvidia", "Cerebras"],
            "event_type": IndustrialEventType.ACQUISITION_ANNOUNCED,
            "sector": IndustrialSector.SEMICONDUCTOR_CHIPS,
            "importance": 0.9,
            "sentiment": 0.3,
        },
        {
            "headline": "Anthropic raises $10B in new funding round led by Amazon",
            "companies": ["Anthropic", "Amazon"],
            "event_type": IndustrialEventType.FUNDING_ROUND,
            "sector": IndustrialSector.AI_FOUNDATION,
            "importance": 0.85,
            "sentiment": 0.6,
        },
        {
            "headline": "Cursor reaches $2B ARR, fastest SaaS company ever",
            "companies": ["Cursor"],
            "event_type": IndustrialEventType.VALUATION_CHANGE,
            "sector": IndustrialSector.AI_FOUNDATION,
            "importance": 0.7,
            "sentiment": 0.8,
        },
        {
            "headline": "Adobe acquires Runway for $6B in video AI push",
            "companies": ["Adobe", "Runway"],
            "event_type": IndustrialEventType.ACQUISITION_COMPLETED,
            "sector": IndustrialSector.AI_FOUNDATION,
            "importance": 0.95,
            "sentiment": 0.4,
        },
        {
            "headline": "Figure AI lays off 20% of workforce amid restructuring",
            "companies": ["Figure AI"],
            "event_type": IndustrialEventType.LAYOFFS,
            "sector": IndustrialSector.AI_ROBOTICS,
            "importance": 0.6,
            "sentiment": -0.5,
        },
        {
            "headline": "Thomson Reuters in advanced talks to acquire Harvey AI",
            "companies": ["Thomson Reuters", "Harvey"],
            "event_type": IndustrialEventType.ACQUISITION_ANNOUNCED,
            "sector": IndustrialSector.AI_ENTERPRISE,
            "importance": 0.8,
            "sentiment": 0.3,
        },
        {
            "headline": "Lambda Labs files S-1 for IPO targeting $15B valuation",
            "companies": ["Lambda Labs"],
            "event_type": IndustrialEventType.IPO_FILED,
            "sector": IndustrialSector.AI_INFRASTRUCTURE,
            "importance": 0.75,
            "sentiment": 0.5,
        },
        {
            "headline": "Waymo expands to 10 new cities, announces $5B investment",
            "companies": ["Waymo", "Alphabet"],
            "event_type": IndustrialEventType.CAPACITY_EXPANSION,
            "sector": IndustrialSector.AUTONOMOUS_VEHICLES,
            "importance": 0.7,
            "sentiment": 0.6,
        },
        {
            "headline": "Anduril wins $2B Pentagon contract for autonomous drones",
            "companies": ["Anduril"],
            "event_type": IndustrialEventType.CONTRACT_WON,
            "sector": IndustrialSector.DEFENSE_AEROSPACE,
            "importance": 0.8,
            "sentiment": 0.5,
        },
        {
            "headline": "Vertical farming startup Plenty declares bankruptcy",
            "companies": ["Plenty"],
            "event_type": IndustrialEventType.BANKRUPTCY,
            "sector": IndustrialSector.AGRICULTURE_FOOD,
            "importance": 0.6,
            "sentiment": -0.7,
        },
    ]

    def __init__(self, event_stream: EventStream):
        self.event_stream = event_stream
        self._running = False

    async def start(self, interval_seconds: float = 5.0):
        """Start simulating events."""
        self._running = True

        while self._running:
            # Pick random event
            event_data = random.choice(self.SAMPLE_EVENTS)

            # Create event with some randomization
            event = RealTimeEvent(
                event_id=str(uuid.uuid4()),
                source="simulator",
                timestamp=datetime.utcnow(),
                headline=event_data["headline"],
                companies_mentioned=event_data["companies"],
                event_type=event_data["event_type"],
                sector=event_data["sector"],
                importance=event_data["importance"] * random.uniform(0.8, 1.2),
                sentiment=event_data["sentiment"] * random.uniform(0.8, 1.2),
            )

            await self.event_stream.publish(event)
            logger.info(f"Simulated event: {event.headline}")

            await asyncio.sleep(interval_seconds)

    async def stop(self):
        """Stop simulation."""
        self._running = False


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

class RealTimeQuorumSystem:
    """Main interface for the real-time quorum system."""

    def __init__(self):
        self.registry = get_industrial_registry()
        self.event_stream = EventStream()
        self.prediction_engine = PredictionEngine(self.registry)
        self.orchestrator = QuorumOrchestrator(
            self.registry,
            self.prediction_engine,
            self.event_stream,
        )
        self.simulator = EventSimulator(self.event_stream)

        self._alert_handler: Optional[Callable] = None

    async def start(self, simulate: bool = False, interval: float = 5.0):
        """Start the real-time system."""
        await self.event_stream.start()
        await self.orchestrator.start()

        if simulate:
            asyncio.create_task(self.simulator.start(interval))

        logger.info(f"Real-time quorum system started (simulate={simulate})")

    async def stop(self):
        """Stop the system."""
        await self.simulator.stop()
        await self.orchestrator.stop()
        await self.event_stream.stop()

    def set_alert_handler(self, handler: Callable[[QuorumDeliberation], Any]):
        """Set handler for alerts."""
        self.orchestrator.on_alert(handler)

    async def inject_event(
        self,
        headline: str,
        companies: List[str],
        event_type: Optional[IndustrialEventType] = None,
        importance: float = 0.5,
        sentiment: float = 0.0,
    ) -> Tuple[RealTimeEvent, Optional[QuorumDeliberation]]:
        """Manually inject an event and see quorum response."""
        event = RealTimeEvent(
            event_id=str(uuid.uuid4()),
            source="manual",
            timestamp=datetime.utcnow(),
            headline=headline,
            companies_mentioned=companies,
            event_type=event_type,
            importance=importance,
            sentiment=sentiment,
        )

        await self.event_stream.publish(event)

        # Wait briefly for processing
        await asyncio.sleep(0.1)

        # Get latest deliberation
        deliberations = await self.orchestrator.get_recent_deliberations(limit=1)

        return event, deliberations[0] if deliberations else None

    async def get_status(self) -> dict:
        """Get system status."""
        stats = await self.orchestrator.get_stats()
        predictions = await self.prediction_engine.get_predictions()

        return {
            "status": "running",
            "stats": stats,
            "active_predictions": len(predictions),
            "high_confidence_predictions": len([
                p for p in predictions
                if p.confidence in [PredictionConfidence.HIGH, PredictionConfidence.VERY_HIGH]
            ]),
        }

    async def get_predictions(
        self,
        prediction_type: Optional[PredictionType] = None,
        min_probability: float = 0.3,
    ) -> List[Prediction]:
        """Get current predictions."""
        return await self.prediction_engine.get_predictions(
            prediction_type=prediction_type,
            min_probability=min_probability,
        )

    async def get_recent_deliberations(self, limit: int = 20) -> List[QuorumDeliberation]:
        """Get recent quorum deliberations."""
        return await self.orchestrator.get_recent_deliberations(limit)


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO / CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

async def run_demo():
    """Run a demonstration of the real-time quorum system."""
    print("\n" + "="*80)
    print("REAL-TIME QUORUM SYSTEM - DEMO")
    print("="*80 + "\n")

    system = RealTimeQuorumSystem()

    # Set up alert handler
    def handle_alert(deliberation: QuorumDeliberation):
        print(f"\n{'!'*60}")
        print(f"ALERT: {deliberation.event.title}")
        print(f"Urgency: {deliberation.urgency_level}")
        print(f"Consensus: {deliberation.consensus_stance.value if deliberation.consensus_stance else 'mixed'}")
        print(f"Strength: {deliberation.consensus_strength:.2f}")
        print(f"Recommended Actions:")
        for action in deliberation.recommended_actions[:3]:
            print(f"  - {action}")
        print(f"{'!'*60}\n")

    system.set_alert_handler(handle_alert)

    # Start with simulation
    await system.start(simulate=True, interval=3.0)

    print("System started. Simulating events...\n")
    print("-"*60)

    # Run for a bit
    for i in range(10):
        await asyncio.sleep(3)

        # Print status
        status = await system.get_status()
        print(f"\n[Tick {i+1}] Events: {status['stats']['events_processed']}, "
              f"Deliberations: {status['stats']['deliberations_completed']}, "
              f"Predictions: {status['active_predictions']}")

        # Print recent deliberation
        deliberations = await system.get_recent_deliberations(limit=1)
        if deliberations:
            d = deliberations[0]
            print(f"  Latest: {d.event.title[:50]}...")
            print(f"  Consensus: {d.consensus_stance.value if d.consensus_stance else 'mixed'} "
                  f"({d.consensus_strength:.0%} agreement)")
            if d.key_insights:
                print(f"  Insight: {d.key_insights[0]}")

    # Show predictions
    print("\n" + "="*60)
    print("PREDICTIONS")
    print("="*60)

    predictions = await system.get_predictions(min_probability=0.3)
    for pred in predictions[:5]:
        print(f"\n{pred.prediction_type.value.upper()}: {pred.target_company}")
        print(f"  Probability: {pred.probability:.0%}")
        print(f"  Confidence: {pred.confidence.value}")
        if pred.likely_acquirers:
            print(f"  Likely acquirers: {', '.join(pred.likely_acquirers[:3])}")
        if pred.estimated_price_billions:
            print(f"  Est. price: ${pred.estimated_price_billions:.1f}B")

    await system.stop()
    print("\nDemo complete.")


# Global instance
_realtime_system: Optional[RealTimeQuorumSystem] = None


def get_realtime_system() -> RealTimeQuorumSystem:
    """Get or create the global real-time system."""
    global _realtime_system
    if _realtime_system is None:
        _realtime_system = RealTimeQuorumSystem()
    return _realtime_system


if __name__ == "__main__":
    asyncio.run(run_demo())
