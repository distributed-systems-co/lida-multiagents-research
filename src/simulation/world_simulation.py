"""
World Simulation - Physical Environment, Causality, and Temporal Dynamics

This module provides a rich simulated world environment including:
- Physical spaces and locations
- Objects with properties and affordances
- Causal reasoning and event chains
- Temporal dynamics (day/night, seasons, aging)
- Resource systems and economics
- Weather and environmental effects
- Event scheduling and narrative time

Based on:
- Situation calculus
- Qualitative physics
- Narrative simulation
- Agent-based modeling
"""

import asyncio
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import heapq


# ============================================================================
# TIME SYSTEM
# ============================================================================

class TimeScale(Enum):
    """Time scales for simulation"""
    REAL_TIME = "real_time"
    FAST_FORWARD = "fast_forward"  # 1 second = 1 hour
    NARRATIVE = "narrative"  # Time jumps between events
    PAUSED = "paused"


@dataclass
class SimulationTime:
    """Tracks simulation time with calendar and cycles"""
    year: int = 1
    month: int = 1
    day: int = 1
    hour: int = 8  # Start at 8 AM
    minute: int = 0
    tick: int = 0

    @property
    def total_minutes(self) -> int:
        return (
            self.year * 365 * 24 * 60 +
            self.month * 30 * 24 * 60 +
            self.day * 24 * 60 +
            self.hour * 60 +
            self.minute
        )

    @property
    def time_of_day(self) -> str:
        if 5 <= self.hour < 12:
            return "morning"
        elif 12 <= self.hour < 17:
            return "afternoon"
        elif 17 <= self.hour < 21:
            return "evening"
        else:
            return "night"

    @property
    def season(self) -> str:
        if self.month in [3, 4, 5]:
            return "spring"
        elif self.month in [6, 7, 8]:
            return "summer"
        elif self.month in [9, 10, 11]:
            return "autumn"
        else:
            return "winter"

    def advance(self, minutes: int = 1):
        """Advance time by minutes"""
        self.minute += minutes
        while self.minute >= 60:
            self.minute -= 60
            self.hour += 1
        while self.hour >= 24:
            self.hour -= 24
            self.day += 1
        while self.day > 30:
            self.day -= 30
            self.month += 1
        while self.month > 12:
            self.month -= 12
            self.year += 1
        self.tick += 1

    def to_string(self) -> str:
        return f"Year {self.year}, Month {self.month}, Day {self.day}, {self.hour:02d}:{self.minute:02d}"


# ============================================================================
# SPATIAL SYSTEM
# ============================================================================

class LocationType(Enum):
    """Types of locations"""
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    UNDERGROUND = "underground"
    UNDERWATER = "underwater"
    AERIAL = "aerial"
    VIRTUAL = "virtual"


@dataclass
class Location:
    """A location in the world"""
    id: str
    name: str
    location_type: LocationType = LocationType.INDOOR
    description: str = ""
    coordinates: Tuple[float, float, float] = (0, 0, 0)
    size: float = 10.0  # Radius or area
    capacity: int = 10
    accessibility: float = 1.0  # 0-1, how easy to access

    # Connections to other locations
    connections: Dict[str, float] = field(default_factory=dict)  # Location ID -> distance

    # Location properties
    properties: Dict[str, Any] = field(default_factory=dict)
    contained_objects: Set[str] = field(default_factory=set)
    contained_agents: Set[str] = field(default_factory=set)

    # Environmental conditions (inherited from region if not set)
    local_weather: Optional[str] = None
    local_temperature: Optional[float] = None
    light_level: float = 1.0  # 0-1

    def can_enter(self, agent_id: str) -> bool:
        """Check if an agent can enter this location"""
        return len(self.contained_agents) < self.capacity


@dataclass
class Region:
    """A larger area containing multiple locations"""
    id: str
    name: str
    description: str
    locations: Set[str] = field(default_factory=set)
    weather: str = "clear"
    temperature: float = 20.0  # Celsius
    terrain: str = "plains"


class SpatialGraph:
    """Graph of connected locations"""

    def __init__(self):
        self.locations: Dict[str, Location] = {}
        self.regions: Dict[str, Region] = {}
        self.location_to_region: Dict[str, str] = {}

    def add_location(self, location: Location, region_id: str = None):
        """Add a location to the world"""
        self.locations[location.id] = location
        if region_id and region_id in self.regions:
            self.regions[region_id].locations.add(location.id)
            self.location_to_region[location.id] = region_id

    def add_region(self, region: Region):
        """Add a region to the world"""
        self.regions[region.id] = region

    def connect_locations(self, loc1_id: str, loc2_id: str, distance: float,
                         bidirectional: bool = True):
        """Connect two locations"""
        if loc1_id in self.locations and loc2_id in self.locations:
            self.locations[loc1_id].connections[loc2_id] = distance
            if bidirectional:
                self.locations[loc2_id].connections[loc1_id] = distance

    def find_path(self, start_id: str, end_id: str) -> List[str]:
        """Find shortest path between locations using A*"""
        if start_id not in self.locations or end_id not in self.locations:
            return []

        start = self.locations[start_id]
        end = self.locations[end_id]

        # A* search
        open_set = [(0, start_id, [start_id])]
        closed = set()

        while open_set:
            cost, current_id, path = heapq.heappop(open_set)

            if current_id == end_id:
                return path

            if current_id in closed:
                continue
            closed.add(current_id)

            current = self.locations[current_id]
            for neighbor_id, distance in current.connections.items():
                if neighbor_id in closed:
                    continue

                new_cost = cost + distance
                # Heuristic: straight line distance
                neighbor = self.locations.get(neighbor_id)
                if neighbor:
                    heuristic = math.sqrt(sum(
                        (a - b) ** 2
                        for a, b in zip(neighbor.coordinates, end.coordinates)
                    ))
                    heapq.heappush(
                        open_set,
                        (new_cost + heuristic, neighbor_id, path + [neighbor_id])
                    )

        return []  # No path found

    def get_nearby_locations(self, location_id: str, max_distance: float) -> List[Tuple[str, float]]:
        """Get locations within distance"""
        if location_id not in self.locations:
            return []

        start = self.locations[location_id]
        nearby = []

        for loc_id, loc in self.locations.items():
            if loc_id == location_id:
                continue
            dist = math.sqrt(sum(
                (a - b) ** 2
                for a, b in zip(start.coordinates, loc.coordinates)
            ))
            if dist <= max_distance:
                nearby.append((loc_id, dist))

        return sorted(nearby, key=lambda x: x[1])


# ============================================================================
# OBJECT SYSTEM
# ============================================================================

class ObjectCategory(Enum):
    """Categories of objects"""
    TOOL = "tool"
    CONTAINER = "container"
    CONSUMABLE = "consumable"
    FURNITURE = "furniture"
    VEHICLE = "vehicle"
    WEAPON = "weapon"
    DOCUMENT = "document"
    ELECTRONIC = "electronic"
    NATURAL = "natural"
    ABSTRACT = "abstract"


@dataclass
class Affordance:
    """What can be done with an object"""
    action: str
    requirements: Dict[str, Any] = field(default_factory=dict)
    effects: Dict[str, Any] = field(default_factory=dict)
    duration: int = 1  # Time units


@dataclass
class WorldObject:
    """An object in the world"""
    id: str
    name: str
    category: ObjectCategory
    description: str = ""

    # Physical properties
    location_id: Optional[str] = None
    container_id: Optional[str] = None  # If inside another object
    owner_id: Optional[str] = None

    # Object properties
    properties: Dict[str, Any] = field(default_factory=dict)
    affordances: List[Affordance] = field(default_factory=list)

    # State
    condition: float = 1.0  # 0-1, degradation
    quantity: int = 1
    is_portable: bool = True
    is_visible: bool = True

    def can_afford(self, action: str, agent_properties: Dict[str, Any]) -> bool:
        """Check if an action is possible given agent properties"""
        for affordance in self.affordances:
            if affordance.action == action:
                for req, value in affordance.requirements.items():
                    if agent_properties.get(req, 0) < value:
                        return False
                return True
        return False


class ObjectRegistry:
    """Manages all objects in the world"""

    def __init__(self):
        self.objects: Dict[str, WorldObject] = {}
        self.location_index: Dict[str, Set[str]] = defaultdict(set)
        self.owner_index: Dict[str, Set[str]] = defaultdict(set)
        self.category_index: Dict[ObjectCategory, Set[str]] = defaultdict(set)

    def register(self, obj: WorldObject):
        """Register an object"""
        self.objects[obj.id] = obj
        if obj.location_id:
            self.location_index[obj.location_id].add(obj.id)
        if obj.owner_id:
            self.owner_index[obj.owner_id].add(obj.id)
        self.category_index[obj.category].add(obj.id)

    def move_object(self, obj_id: str, new_location_id: str):
        """Move object to new location"""
        if obj_id not in self.objects:
            return

        obj = self.objects[obj_id]
        if obj.location_id:
            self.location_index[obj.location_id].discard(obj_id)
        obj.location_id = new_location_id
        self.location_index[new_location_id].add(obj_id)

    def transfer_ownership(self, obj_id: str, new_owner_id: str):
        """Transfer object ownership"""
        if obj_id not in self.objects:
            return

        obj = self.objects[obj_id]
        if obj.owner_id:
            self.owner_index[obj.owner_id].discard(obj_id)
        obj.owner_id = new_owner_id
        self.owner_index[new_owner_id].add(obj_id)

    def get_by_location(self, location_id: str) -> List[WorldObject]:
        """Get all objects at a location"""
        return [
            self.objects[oid]
            for oid in self.location_index.get(location_id, set())
        ]


# ============================================================================
# CAUSAL SYSTEM
# ============================================================================

class CausalRelation(Enum):
    """Types of causal relationships"""
    CAUSES = "causes"  # A causes B
    ENABLES = "enables"  # A enables B (necessary but not sufficient)
    PREVENTS = "prevents"  # A prevents B
    MODIFIES = "modifies"  # A modifies B's effect
    TRIGGERS = "triggers"  # A triggers B immediately


@dataclass
class CausalRule:
    """A rule describing causal relationships"""
    id: str
    condition: Dict[str, Any]  # What must be true
    effect: Dict[str, Any]  # What happens
    relation: CausalRelation = CausalRelation.CAUSES
    probability: float = 1.0  # Probability of effect given condition
    delay: int = 0  # Time delay before effect
    description: str = ""


@dataclass
class Event:
    """An event that has occurred or is scheduled"""
    id: str
    event_type: str
    timestamp: SimulationTime
    location_id: Optional[str] = None
    agent_id: Optional[str] = None
    object_id: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    caused_by: Optional[str] = None  # ID of causing event
    effects: List[str] = field(default_factory=list)  # IDs of caused events


class CausalEngine:
    """
    Engine for causal reasoning and event propagation.
    """

    def __init__(self):
        self.rules: Dict[str, CausalRule] = {}
        self.event_history: List[Event] = []
        self.pending_effects: List[Tuple[int, Event]] = []  # (time, event)
        self.causal_chains: Dict[str, List[str]] = {}  # event_id -> chain of effects

    def add_rule(self, rule: CausalRule):
        """Add a causal rule"""
        self.rules[rule.id] = rule

    def check_condition(self, condition: Dict[str, Any],
                       world_state: Dict[str, Any]) -> bool:
        """Check if a condition is met"""
        for key, required in condition.items():
            actual = world_state.get(key)
            if isinstance(required, dict):
                # Complex condition
                op = required.get("op", "eq")
                val = required.get("value")
                if op == "eq" and actual != val:
                    return False
                elif op == "gt" and not (actual is not None and actual > val):
                    return False
                elif op == "lt" and not (actual is not None and actual < val):
                    return False
                elif op == "contains" and val not in (actual or []):
                    return False
            else:
                if actual != required:
                    return False
        return True

    def process_event(self, event: Event, world_state: Dict[str, Any]) -> List[Event]:
        """Process an event and generate consequent events"""
        self.event_history.append(event)
        caused_events = []

        for rule in self.rules.values():
            # Check if rule condition is met
            combined_state = {**world_state, **event.properties, "event_type": event.event_type}
            if self.check_condition(rule.condition, combined_state):
                # Rule fires
                if random.random() < rule.probability:
                    effect_event = Event(
                        id=f"effect_{len(self.event_history)}_{rule.id}",
                        event_type=rule.effect.get("event_type", "effect"),
                        timestamp=event.timestamp,  # Will be adjusted by delay
                        location_id=event.location_id,
                        agent_id=rule.effect.get("agent_id", event.agent_id),
                        properties=rule.effect,
                        caused_by=event.id
                    )

                    if rule.delay > 0:
                        # Schedule for later
                        self.pending_effects.append((
                            event.timestamp.total_minutes + rule.delay,
                            effect_event
                        ))
                    else:
                        caused_events.append(effect_event)
                        event.effects.append(effect_event.id)

        # Build causal chain
        if event.caused_by:
            if event.caused_by in self.causal_chains:
                self.causal_chains[event.id] = self.causal_chains[event.caused_by] + [event.id]
            else:
                self.causal_chains[event.id] = [event.caused_by, event.id]
        else:
            self.causal_chains[event.id] = [event.id]

        return caused_events

    def get_pending_effects(self, current_time: int) -> List[Event]:
        """Get effects that should fire at current time"""
        ready = []
        still_pending = []

        for time, event in self.pending_effects:
            if time <= current_time:
                ready.append(event)
            else:
                still_pending.append((time, event))

        self.pending_effects = still_pending
        return ready

    def trace_causation(self, event_id: str) -> List[str]:
        """Trace the causal chain leading to an event"""
        return self.causal_chains.get(event_id, [event_id])


# ============================================================================
# RESOURCE SYSTEM
# ============================================================================

class ResourceType(Enum):
    """Types of resources"""
    MATERIAL = "material"  # Physical materials
    ENERGY = "energy"
    CURRENCY = "currency"
    INFORMATION = "information"
    INFLUENCE = "influence"
    TIME = "time"


@dataclass
class Resource:
    """A resource in the world"""
    id: str
    name: str
    resource_type: ResourceType
    quantity: float = 0.0
    max_quantity: float = float('inf')
    regeneration_rate: float = 0.0
    decay_rate: float = 0.0


@dataclass
class ResourceNode:
    """A source or sink of resources"""
    id: str
    location_id: str
    resources: Dict[str, float] = field(default_factory=dict)  # Resource ID -> quantity
    production_rates: Dict[str, float] = field(default_factory=dict)
    consumption_rates: Dict[str, float] = field(default_factory=dict)


class ResourceSystem:
    """Manages resources and economic flows"""

    def __init__(self):
        self.resource_types: Dict[str, Resource] = {}
        self.nodes: Dict[str, ResourceNode] = {}
        self.agent_resources: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.transaction_log: List[Dict] = []

    def define_resource(self, resource: Resource):
        """Define a new resource type"""
        self.resource_types[resource.id] = resource

    def add_node(self, node: ResourceNode):
        """Add a resource node"""
        self.nodes[node.id] = node

    def transfer(self, from_agent: str, to_agent: str,
                resource_id: str, amount: float) -> bool:
        """Transfer resources between agents"""
        if self.agent_resources[from_agent].get(resource_id, 0) < amount:
            return False

        self.agent_resources[from_agent][resource_id] -= amount
        self.agent_resources[to_agent][resource_id] = \
            self.agent_resources[to_agent].get(resource_id, 0) + amount

        self.transaction_log.append({
            "from": from_agent,
            "to": to_agent,
            "resource": resource_id,
            "amount": amount,
            "timestamp": datetime.now()
        })

        return True

    def harvest(self, agent_id: str, node_id: str,
               resource_id: str, amount: float) -> float:
        """Harvest resources from a node"""
        if node_id not in self.nodes:
            return 0.0

        node = self.nodes[node_id]
        available = node.resources.get(resource_id, 0)
        harvested = min(available, amount)

        node.resources[resource_id] = available - harvested
        self.agent_resources[agent_id][resource_id] = \
            self.agent_resources[agent_id].get(resource_id, 0) + harvested

        return harvested

    def tick(self):
        """Update resources for one time unit"""
        # Node production/consumption
        for node in self.nodes.values():
            for res_id, rate in node.production_rates.items():
                current = node.resources.get(res_id, 0)
                resource = self.resource_types.get(res_id)
                max_qty = resource.max_quantity if resource else float('inf')
                node.resources[res_id] = min(max_qty, current + rate)

            for res_id, rate in node.consumption_rates.items():
                current = node.resources.get(res_id, 0)
                node.resources[res_id] = max(0, current - rate)


# ============================================================================
# WEATHER SYSTEM
# ============================================================================

class WeatherType(Enum):
    """Weather conditions"""
    CLEAR = "clear"
    CLOUDY = "cloudy"
    RAIN = "rain"
    STORM = "storm"
    SNOW = "snow"
    FOG = "fog"
    WIND = "wind"
    EXTREME_HEAT = "extreme_heat"
    EXTREME_COLD = "extreme_cold"


@dataclass
class WeatherState:
    """Current weather state"""
    weather_type: WeatherType = WeatherType.CLEAR
    temperature: float = 20.0
    humidity: float = 0.5
    wind_speed: float = 5.0
    visibility: float = 1.0
    precipitation: float = 0.0


class WeatherSystem:
    """Simulates weather patterns"""

    def __init__(self):
        self.current_weather: Dict[str, WeatherState] = {}  # Region -> weather
        self.forecast: List[Tuple[int, str, WeatherState]] = []  # (time, region, weather)
        self.transition_matrix = {
            WeatherType.CLEAR: {WeatherType.CLEAR: 0.7, WeatherType.CLOUDY: 0.2, WeatherType.WIND: 0.1},
            WeatherType.CLOUDY: {WeatherType.CLOUDY: 0.5, WeatherType.RAIN: 0.3, WeatherType.CLEAR: 0.2},
            WeatherType.RAIN: {WeatherType.RAIN: 0.4, WeatherType.STORM: 0.2, WeatherType.CLOUDY: 0.3, WeatherType.CLEAR: 0.1},
            WeatherType.STORM: {WeatherType.STORM: 0.3, WeatherType.RAIN: 0.5, WeatherType.CLOUDY: 0.2},
            WeatherType.SNOW: {WeatherType.SNOW: 0.6, WeatherType.CLOUDY: 0.3, WeatherType.CLEAR: 0.1},
            WeatherType.FOG: {WeatherType.FOG: 0.4, WeatherType.CLOUDY: 0.3, WeatherType.CLEAR: 0.3},
            WeatherType.WIND: {WeatherType.WIND: 0.3, WeatherType.STORM: 0.2, WeatherType.CLEAR: 0.3, WeatherType.CLOUDY: 0.2},
        }

    def initialize_region(self, region_id: str, initial_weather: WeatherType = None):
        """Initialize weather for a region"""
        if initial_weather is None:
            initial_weather = random.choice(list(WeatherType))

        self.current_weather[region_id] = WeatherState(weather_type=initial_weather)

    def update(self, region_id: str, current_time: SimulationTime):
        """Update weather for a region"""
        if region_id not in self.current_weather:
            self.initialize_region(region_id)

        state = self.current_weather[region_id]

        # Transition weather type
        transitions = self.transition_matrix.get(state.weather_type, {})
        if transitions:
            r = random.random()
            cumulative = 0
            for weather_type, prob in transitions.items():
                cumulative += prob
                if r < cumulative:
                    state.weather_type = weather_type
                    break

        # Adjust temperature based on time of day and season
        base_temp = {"spring": 15, "summer": 25, "autumn": 12, "winter": 0}[current_time.season]
        time_modifier = {"morning": -2, "afternoon": 5, "evening": 0, "night": -5}[current_time.time_of_day]
        state.temperature = base_temp + time_modifier + random.gauss(0, 2)

        # Update other parameters based on weather type
        if state.weather_type == WeatherType.RAIN:
            state.precipitation = random.uniform(0.3, 0.8)
            state.humidity = random.uniform(0.7, 1.0)
            state.visibility = random.uniform(0.5, 0.8)
        elif state.weather_type == WeatherType.STORM:
            state.precipitation = random.uniform(0.7, 1.0)
            state.humidity = 1.0
            state.visibility = random.uniform(0.2, 0.5)
            state.wind_speed = random.uniform(30, 60)
        elif state.weather_type == WeatherType.FOG:
            state.visibility = random.uniform(0.1, 0.3)
            state.humidity = random.uniform(0.9, 1.0)
        elif state.weather_type == WeatherType.CLEAR:
            state.precipitation = 0
            state.visibility = 1.0
            state.humidity = random.uniform(0.3, 0.6)

    def get_weather(self, region_id: str) -> WeatherState:
        """Get current weather for a region"""
        if region_id not in self.current_weather:
            self.initialize_region(region_id)
        return self.current_weather[region_id]


# ============================================================================
# EVENT SCHEDULER
# ============================================================================

@dataclass
class ScheduledEvent:
    """An event scheduled to occur"""
    id: str
    event_type: str
    scheduled_time: int  # Total minutes from start
    location_id: Optional[str] = None
    participants: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    recurring: bool = False
    recurrence_interval: int = 0  # Minutes between occurrences
    condition: Optional[Dict[str, Any]] = None


class EventScheduler:
    """Schedules and triggers events"""

    def __init__(self):
        self.scheduled: List[ScheduledEvent] = []
        self.recurring: List[ScheduledEvent] = []
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)

    def schedule(self, event: ScheduledEvent):
        """Schedule an event"""
        if event.recurring:
            self.recurring.append(event)
        heapq.heappush(self.scheduled, (event.scheduled_time, event.id, event))

    def register_handler(self, event_type: str, handler: Callable):
        """Register a handler for an event type"""
        self.event_handlers[event_type].append(handler)

    def get_due_events(self, current_time: int,
                      world_state: Dict[str, Any] = None) -> List[ScheduledEvent]:
        """Get events that should fire at current time"""
        due = []

        while self.scheduled and self.scheduled[0][0] <= current_time:
            _, _, event = heapq.heappop(self.scheduled)

            # Check condition if present
            if event.condition and world_state:
                if not all(world_state.get(k) == v for k, v in event.condition.items()):
                    continue

            due.append(event)

        # Handle recurring events
        for event in self.recurring:
            if current_time >= event.scheduled_time:
                if (current_time - event.scheduled_time) % event.recurrence_interval < 1:
                    due.append(event)

        return due

    def trigger_event(self, event: ScheduledEvent, world_state: Dict[str, Any] = None):
        """Trigger an event and call handlers"""
        for handler in self.event_handlers.get(event.event_type, []):
            handler(event, world_state)


# ============================================================================
# WORLD SIMULATION
# ============================================================================

class WorldSimulation:
    """
    Complete world simulation integrating all systems.
    """

    def __init__(self, name: str = "Simulated World"):
        self.name = name

        # Time
        self.time = SimulationTime()
        self.time_scale = TimeScale.REAL_TIME
        self.tick_duration = 1  # Minutes per tick

        # Space
        self.spatial = SpatialGraph()

        # Objects
        self.objects = ObjectRegistry()

        # Causality
        self.causal = CausalEngine()

        # Resources
        self.resources = ResourceSystem()

        # Weather
        self.weather = WeatherSystem()

        # Events
        self.scheduler = EventScheduler()

        # Agent positions
        self.agent_locations: Dict[str, str] = {}

        # World state for causal reasoning
        self.state: Dict[str, Any] = {}

        # History
        self.history: List[Dict] = []

        self.running = False

    def create_location(self, loc_id: str, name: str,
                       location_type: LocationType = LocationType.INDOOR,
                       region_id: str = None, **kwargs) -> Location:
        """Create a new location"""
        location = Location(
            id=loc_id,
            name=name,
            location_type=location_type,
            **kwargs
        )
        self.spatial.add_location(location, region_id)
        return location

    def create_region(self, region_id: str, name: str, **kwargs) -> Region:
        """Create a new region"""
        region = Region(id=region_id, name=name, **kwargs)
        self.spatial.add_region(region)
        self.weather.initialize_region(region_id)
        return region

    def create_object(self, obj_id: str, name: str, category: ObjectCategory,
                     location_id: str = None, **kwargs) -> WorldObject:
        """Create a new object"""
        obj = WorldObject(
            id=obj_id,
            name=name,
            category=category,
            location_id=location_id,
            **kwargs
        )
        self.objects.register(obj)
        if location_id and location_id in self.spatial.locations:
            self.spatial.locations[location_id].contained_objects.add(obj_id)
        return obj

    def place_agent(self, agent_id: str, location_id: str) -> bool:
        """Place an agent at a location"""
        if location_id not in self.spatial.locations:
            return False

        location = self.spatial.locations[location_id]
        if not location.can_enter(agent_id):
            return False

        # Remove from old location
        old_loc_id = self.agent_locations.get(agent_id)
        if old_loc_id and old_loc_id in self.spatial.locations:
            self.spatial.locations[old_loc_id].contained_agents.discard(agent_id)

        # Add to new location
        location.contained_agents.add(agent_id)
        self.agent_locations[agent_id] = location_id

        return True

    def move_agent(self, agent_id: str, destination_id: str) -> Tuple[bool, List[str]]:
        """Move an agent to a destination, returning path taken"""
        current = self.agent_locations.get(agent_id)
        if not current:
            return False, []

        path = self.spatial.find_path(current, destination_id)
        if not path:
            return False, []

        # Move through path
        for loc_id in path[1:]:  # Skip starting location
            if not self.place_agent(agent_id, loc_id):
                return False, path[:path.index(loc_id)]

        return True, path

    def trigger_event(self, event_type: str, location_id: str = None,
                     agent_id: str = None, **properties) -> Event:
        """Trigger an event in the world"""
        event = Event(
            id=f"event_{len(self.causal.event_history)}",
            event_type=event_type,
            timestamp=SimulationTime(
                year=self.time.year, month=self.time.month,
                day=self.time.day, hour=self.time.hour, minute=self.time.minute
            ),
            location_id=location_id,
            agent_id=agent_id,
            properties=properties
        )

        # Process through causal engine
        world_state = self.get_state()
        caused_events = self.causal.process_event(event, world_state)

        # Process caused events recursively
        for caused in caused_events:
            self.causal.process_event(caused, world_state)

        return event

    def get_state(self) -> Dict[str, Any]:
        """Get current world state"""
        return {
            "time": self.time.to_string(),
            "time_of_day": self.time.time_of_day,
            "season": self.time.season,
            "agent_count": len(self.agent_locations),
            "object_count": len(self.objects.objects),
            **self.state
        }

    def tick(self):
        """Advance world by one tick"""
        # Advance time
        self.time.advance(self.tick_duration)

        # Update weather for all regions
        for region_id in self.spatial.regions:
            self.weather.update(region_id, self.time)

        # Update resources
        self.resources.tick()

        # Process pending causal effects
        pending = self.causal.get_pending_effects(self.time.total_minutes)
        world_state = self.get_state()
        for event in pending:
            self.causal.process_event(event, world_state)

        # Process scheduled events
        due_events = self.scheduler.get_due_events(self.time.total_minutes, world_state)
        for event in due_events:
            self.scheduler.trigger_event(event, world_state)

        # Record history
        self.history.append({
            "tick": self.time.tick,
            "time": self.time.to_string(),
            "event_count": len(self.causal.event_history),
            "agent_locations": dict(self.agent_locations)
        })

        if len(self.history) > 1000:
            self.history.pop(0)

    async def run(self, ticks: int = 100, delay: float = 0.1):
        """Run the simulation"""
        self.running = True

        for _ in range(ticks):
            if not self.running:
                break
            self.tick()
            await asyncio.sleep(delay)

        self.running = False

    def stop(self):
        """Stop the simulation"""
        self.running = False

    def get_summary(self) -> Dict[str, Any]:
        """Get simulation summary"""
        return {
            "name": self.name,
            "time": self.time.to_string(),
            "tick": self.time.tick,
            "locations": len(self.spatial.locations),
            "regions": len(self.spatial.regions),
            "objects": len(self.objects.objects),
            "agents": len(self.agent_locations),
            "events_processed": len(self.causal.event_history),
            "weather": {
                region_id: state.weather_type.value
                for region_id, state in self.weather.current_weather.items()
            }
        }


# Convenience functions
def create_world(name: str = "Default World") -> WorldSimulation:
    """Create a new world simulation"""
    return WorldSimulation(name)


def demo_world():
    """Demonstrate world simulation"""
    world = create_world("Demo World")

    # Create a region
    city = world.create_region("city", "Downtown City", terrain="urban")

    # Create locations
    office = world.create_location("office", "Office Building",
                                   LocationType.INDOOR, "city",
                                   capacity=50,
                                   coordinates=(0, 0, 0))
    park = world.create_location("park", "Central Park",
                                LocationType.OUTDOOR, "city",
                                capacity=200,
                                coordinates=(100, 0, 0))
    cafe = world.create_location("cafe", "Coffee Shop",
                                LocationType.INDOOR, "city",
                                capacity=20,
                                coordinates=(50, 30, 0))

    # Connect locations
    world.spatial.connect_locations("office", "park", 100)
    world.spatial.connect_locations("office", "cafe", 50)
    world.spatial.connect_locations("park", "cafe", 60)

    # Create objects
    world.create_object("computer1", "Desktop Computer",
                       ObjectCategory.ELECTRONIC, "office")
    world.create_object("bench1", "Park Bench",
                       ObjectCategory.FURNITURE, "park",
                       is_portable=False)

    # Place agents
    world.place_agent("alice", "office")
    world.place_agent("bob", "park")

    # Add causal rule
    world.causal.add_rule(CausalRule(
        id="rain_reduces_park_visitors",
        condition={"event_type": "weather_change", "weather": "rain"},
        effect={"event_type": "location_emptying", "location_id": "park"},
        relation=CausalRelation.CAUSES,
        probability=0.7
    ))

    print("=== World Demo ===")
    print(f"World: {world.name}")
    print(f"Locations: {len(world.spatial.locations)}")

    # Run for a few ticks
    for _ in range(10):
        world.tick()

    print(f"\nAfter 10 ticks:")
    print(f"  Time: {world.time.to_string()}")
    print(f"  Weather: {world.weather.get_weather('city').weather_type.value}")
    print(f"  Summary: {world.get_summary()}")

    # Move an agent
    success, path = world.move_agent("alice", "cafe")
    print(f"\nAlice moved to cafe: {success}, path: {path}")

    return world


if __name__ == "__main__":
    demo_world()
