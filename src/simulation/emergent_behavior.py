"""
Emergent Behavior Engine - Swarm Intelligence and Self-Organization

This module implements sophisticated emergence and self-organization patterns:
- Swarm Intelligence (boid-like behaviors, ant colony optimization)
- Self-Organizing Maps (SOMs)
- Cellular Automata
- Complex Adaptive Systems
- Phase Transitions and Critical Points
- Stigmergy (indirect communication through environment)
- Morphogenesis (pattern formation)

Based on:
- Complex Systems Theory
- Swarm Intelligence Research
- Self-Organization in Nature
- Emergence in Artificial Life
"""

import asyncio
import random
import math
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import heapq


# ============================================================================
# SPATIAL REPRESENTATION
# ============================================================================

@dataclass
class Vector3D:
    """3D vector for spatial calculations"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vector3D") -> "Vector3D":
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3D") -> "Vector3D":
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vector3D":
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar: float) -> "Vector3D":
        if scalar == 0:
            return Vector3D()
        return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)

    @property
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> "Vector3D":
        mag = self.magnitude
        if mag == 0:
            return Vector3D()
        return self / mag

    def distance_to(self, other: "Vector3D") -> float:
        return (self - other).magnitude

    def dot(self, other: "Vector3D") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def limit(self, max_val: float) -> "Vector3D":
        if self.magnitude > max_val:
            return self.normalize() * max_val
        return Vector3D(self.x, self.y, self.z)


# ============================================================================
# SWARM INTELLIGENCE
# ============================================================================

class SwarmBehavior(Enum):
    """Types of swarm behaviors"""
    SEPARATION = "separation"  # Avoid crowding neighbors
    ALIGNMENT = "alignment"  # Steer towards average heading
    COHESION = "cohesion"  # Steer towards center of mass
    SEEKING = "seeking"  # Move towards target
    FLEEING = "fleeing"  # Move away from threat
    WANDERING = "wandering"  # Random exploration
    FLOCKING = "flocking"  # Combined boid behavior
    FORAGING = "foraging"  # Search for resources


@dataclass
class SwarmAgent:
    """An agent in the swarm"""
    id: str
    position: Vector3D = field(default_factory=Vector3D)
    velocity: Vector3D = field(default_factory=Vector3D)
    acceleration: Vector3D = field(default_factory=Vector3D)
    max_speed: float = 2.0
    max_force: float = 0.1
    perception_radius: float = 5.0
    personal_space: float = 1.0
    state: Dict[str, Any] = field(default_factory=dict)

    def apply_force(self, force: Vector3D):
        """Apply a steering force"""
        self.acceleration = self.acceleration + force

    def update(self, dt: float = 1.0):
        """Update position and velocity"""
        self.velocity = (self.velocity + self.acceleration).limit(self.max_speed)
        self.position = self.position + self.velocity * dt
        self.acceleration = Vector3D()  # Reset

    def steer_towards(self, target: Vector3D) -> Vector3D:
        """Calculate steering force towards target"""
        desired = (target - self.position).normalize() * self.max_speed
        steer = desired - self.velocity
        return steer.limit(self.max_force)


class SwarmSimulation:
    """
    Simulates swarm behavior with emergent patterns.
    Implements Reynolds' Boids algorithm and extensions.
    """

    def __init__(self, bounds: Tuple[float, float, float] = (100, 100, 100)):
        self.agents: Dict[str, SwarmAgent] = {}
        self.bounds = bounds
        self.obstacles: List[Vector3D] = []
        self.attractors: List[Tuple[Vector3D, float]] = []  # Position, strength
        self.repellors: List[Tuple[Vector3D, float]] = []

        # Behavior weights
        self.weights = {
            SwarmBehavior.SEPARATION: 1.5,
            SwarmBehavior.ALIGNMENT: 1.0,
            SwarmBehavior.COHESION: 1.0,
            SwarmBehavior.SEEKING: 2.0,
            SwarmBehavior.FLEEING: 3.0,
            SwarmBehavior.WANDERING: 0.3
        }

        # Stigmergy - pheromone trails
        self.pheromone_grid: Dict[Tuple[int, int, int], Dict[str, float]] = defaultdict(dict)
        self.pheromone_decay = 0.99

        self.time_step = 0

    def add_agent(self, agent_id: str, position: Vector3D = None) -> SwarmAgent:
        """Add an agent to the swarm"""
        if position is None:
            position = Vector3D(
                random.uniform(0, self.bounds[0]),
                random.uniform(0, self.bounds[1]),
                random.uniform(0, self.bounds[2])
            )

        agent = SwarmAgent(id=agent_id, position=position)
        self.agents[agent_id] = agent
        return agent

    def get_neighbors(self, agent: SwarmAgent, radius: float = None) -> List[SwarmAgent]:
        """Get neighbors within perception radius"""
        if radius is None:
            radius = agent.perception_radius

        neighbors = []
        for other_id, other in self.agents.items():
            if other_id != agent.id:
                dist = agent.position.distance_to(other.position)
                if dist < radius:
                    neighbors.append(other)
        return neighbors

    def calculate_separation(self, agent: SwarmAgent, neighbors: List[SwarmAgent]) -> Vector3D:
        """Calculate separation steering force"""
        steer = Vector3D()
        count = 0

        for neighbor in neighbors:
            dist = agent.position.distance_to(neighbor.position)
            if 0 < dist < agent.personal_space:
                # Weight by inverse distance
                diff = (agent.position - neighbor.position).normalize() / dist
                steer = steer + diff
                count += 1

        if count > 0:
            steer = steer / count
            steer = steer.normalize() * agent.max_speed - agent.velocity
            steer = steer.limit(agent.max_force)

        return steer

    def calculate_alignment(self, agent: SwarmAgent, neighbors: List[SwarmAgent]) -> Vector3D:
        """Calculate alignment steering force"""
        if not neighbors:
            return Vector3D()

        avg_velocity = Vector3D()
        for neighbor in neighbors:
            avg_velocity = avg_velocity + neighbor.velocity

        avg_velocity = avg_velocity / len(neighbors)
        avg_velocity = avg_velocity.normalize() * agent.max_speed

        steer = avg_velocity - agent.velocity
        return steer.limit(agent.max_force)

    def calculate_cohesion(self, agent: SwarmAgent, neighbors: List[SwarmAgent]) -> Vector3D:
        """Calculate cohesion steering force"""
        if not neighbors:
            return Vector3D()

        center = Vector3D()
        for neighbor in neighbors:
            center = center + neighbor.position

        center = center / len(neighbors)
        return agent.steer_towards(center)

    def calculate_seeking(self, agent: SwarmAgent, target: Vector3D) -> Vector3D:
        """Calculate seeking steering force"""
        return agent.steer_towards(target)

    def calculate_fleeing(self, agent: SwarmAgent, threat: Vector3D,
                         panic_radius: float = 10.0) -> Vector3D:
        """Calculate fleeing steering force"""
        dist = agent.position.distance_to(threat)
        if dist > panic_radius:
            return Vector3D()

        flee = agent.position - threat
        flee = flee.normalize() * agent.max_speed * (1 - dist / panic_radius)
        steer = flee - agent.velocity
        return steer.limit(agent.max_force * 2)  # More urgent

    def calculate_wandering(self, agent: SwarmAgent) -> Vector3D:
        """Calculate random wandering force"""
        return Vector3D(
            random.gauss(0, 0.1),
            random.gauss(0, 0.1),
            random.gauss(0, 0.1)
        )

    def apply_boid_rules(self, agent: SwarmAgent):
        """Apply Reynolds' boid rules"""
        neighbors = self.get_neighbors(agent)

        separation = self.calculate_separation(agent, neighbors) * self.weights[SwarmBehavior.SEPARATION]
        alignment = self.calculate_alignment(agent, neighbors) * self.weights[SwarmBehavior.ALIGNMENT]
        cohesion = self.calculate_cohesion(agent, neighbors) * self.weights[SwarmBehavior.COHESION]
        wandering = self.calculate_wandering(agent) * self.weights[SwarmBehavior.WANDERING]

        agent.apply_force(separation)
        agent.apply_force(alignment)
        agent.apply_force(cohesion)
        agent.apply_force(wandering)

        # Apply attractor/repellor forces
        for attractor, strength in self.attractors:
            seeking = self.calculate_seeking(agent, attractor) * strength
            agent.apply_force(seeking)

        for repellor, strength in self.repellors:
            fleeing = self.calculate_fleeing(agent, repellor) * strength
            agent.apply_force(fleeing)

    def enforce_bounds(self, agent: SwarmAgent):
        """Keep agent within bounds with soft boundaries"""
        margin = 10.0
        turn_factor = 0.5

        force = Vector3D()

        if agent.position.x < margin:
            force.x = turn_factor
        elif agent.position.x > self.bounds[0] - margin:
            force.x = -turn_factor

        if agent.position.y < margin:
            force.y = turn_factor
        elif agent.position.y > self.bounds[1] - margin:
            force.y = -turn_factor

        if agent.position.z < margin:
            force.z = turn_factor
        elif agent.position.z > self.bounds[2] - margin:
            force.z = -turn_factor

        agent.apply_force(force)

    def deposit_pheromone(self, position: Vector3D, pheromone_type: str,
                         amount: float = 1.0):
        """Deposit pheromone at position"""
        cell = (
            int(position.x),
            int(position.y),
            int(position.z)
        )
        current = self.pheromone_grid[cell].get(pheromone_type, 0)
        self.pheromone_grid[cell][pheromone_type] = min(10.0, current + amount)

    def sense_pheromone(self, position: Vector3D, pheromone_type: str,
                       radius: float = 3.0) -> float:
        """Sense pheromone concentration around position"""
        total = 0.0
        count = 0

        for dx in range(-int(radius), int(radius) + 1):
            for dy in range(-int(radius), int(radius) + 1):
                for dz in range(-int(radius), int(radius) + 1):
                    cell = (
                        int(position.x) + dx,
                        int(position.y) + dy,
                        int(position.z) + dz
                    )
                    if cell in self.pheromone_grid:
                        total += self.pheromone_grid[cell].get(pheromone_type, 0)
                        count += 1

        return total / max(1, count)

    def decay_pheromones(self):
        """Decay all pheromones"""
        empty_cells = []
        for cell, pheromones in self.pheromone_grid.items():
            for ptype in list(pheromones.keys()):
                pheromones[ptype] *= self.pheromone_decay
                if pheromones[ptype] < 0.01:
                    del pheromones[ptype]
            if not pheromones:
                empty_cells.append(cell)

        for cell in empty_cells:
            del self.pheromone_grid[cell]

    def step(self, dt: float = 1.0):
        """Advance simulation by one step"""
        self.time_step += 1

        # Apply forces
        for agent in self.agents.values():
            self.apply_boid_rules(agent)
            self.enforce_bounds(agent)

        # Update positions
        for agent in self.agents.values():
            agent.update(dt)

        # Decay pheromones
        self.decay_pheromones()

    def get_swarm_metrics(self) -> Dict[str, Any]:
        """Get metrics about swarm behavior"""
        if not self.agents:
            return {}

        agents = list(self.agents.values())

        # Center of mass
        center = Vector3D()
        for agent in agents:
            center = center + agent.position
        center = center / len(agents)

        # Spread (standard deviation from center)
        spread = sum(agent.position.distance_to(center) for agent in agents) / len(agents)

        # Average velocity
        avg_velocity = Vector3D()
        for agent in agents:
            avg_velocity = avg_velocity + agent.velocity
        avg_velocity = avg_velocity / len(agents)

        # Velocity alignment (how aligned are velocities)
        alignment = 0.0
        for agent in agents:
            if agent.velocity.magnitude > 0 and avg_velocity.magnitude > 0:
                alignment += agent.velocity.normalize().dot(avg_velocity.normalize())
        alignment /= len(agents)

        return {
            "center": (center.x, center.y, center.z),
            "spread": spread,
            "avg_speed": avg_velocity.magnitude,
            "alignment": alignment,
            "agent_count": len(agents),
            "pheromone_cells": len(self.pheromone_grid)
        }


# ============================================================================
# CELLULAR AUTOMATA
# ============================================================================

class CellState(Enum):
    """Cell states for various automata"""
    DEAD = 0
    ALIVE = 1
    EXCITED = 2
    REFRACTORY = 3


@dataclass
class Cell:
    """A cell in the automaton"""
    position: Tuple[int, int]
    state: CellState = CellState.DEAD
    age: int = 0
    properties: Dict[str, Any] = field(default_factory=dict)


class CellularAutomaton:
    """
    Generic cellular automaton with customizable rules.
    Supports Conway's Game of Life, Wolfram rules, and custom rules.
    """

    def __init__(self, width: int = 50, height: int = 50,
                 neighborhood: str = "moore"):
        self.width = width
        self.height = height
        self.neighborhood = neighborhood  # "moore" or "von_neumann"
        self.grid: Dict[Tuple[int, int], Cell] = {}
        self.rules: Callable[[Cell, List[Cell]], CellState] = self._game_of_life_rule
        self.generation = 0
        self.history: List[Dict[Tuple[int, int], CellState]] = []

        # Initialize grid
        for x in range(width):
            for y in range(height):
                self.grid[(x, y)] = Cell(position=(x, y))

    def get_neighbors(self, x: int, y: int) -> List[Cell]:
        """Get neighboring cells"""
        neighbors = []

        if self.neighborhood == "moore":
            offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        else:  # von_neumann
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dx, dy in offsets:
            nx, ny = (x + dx) % self.width, (y + dy) % self.height
            if (nx, ny) in self.grid:
                neighbors.append(self.grid[(nx, ny)])

        return neighbors

    def _game_of_life_rule(self, cell: Cell, neighbors: List[Cell]) -> CellState:
        """Conway's Game of Life rule"""
        alive_count = sum(1 for n in neighbors if n.state == CellState.ALIVE)

        if cell.state == CellState.ALIVE:
            if alive_count in (2, 3):
                return CellState.ALIVE
            return CellState.DEAD
        else:
            if alive_count == 3:
                return CellState.ALIVE
            return CellState.DEAD

    def set_rule(self, rule_func: Callable[[Cell, List[Cell]], CellState]):
        """Set custom rule function"""
        self.rules = rule_func

    def randomize(self, density: float = 0.3):
        """Randomly initialize grid"""
        for cell in self.grid.values():
            if random.random() < density:
                cell.state = CellState.ALIVE
            else:
                cell.state = CellState.DEAD

    def set_pattern(self, pattern: List[Tuple[int, int]], offset: Tuple[int, int] = (0, 0)):
        """Set a pattern on the grid"""
        for px, py in pattern:
            x, y = (px + offset[0]) % self.width, (py + offset[1]) % self.height
            if (x, y) in self.grid:
                self.grid[(x, y)].state = CellState.ALIVE

    def step(self):
        """Advance one generation"""
        new_states = {}

        for pos, cell in self.grid.items():
            neighbors = self.get_neighbors(*pos)
            new_states[pos] = self.rules(cell, neighbors)

        # Record history
        self.history.append({
            pos: cell.state for pos, cell in self.grid.items()
        })
        if len(self.history) > 100:
            self.history.pop(0)

        # Apply new states
        for pos, new_state in new_states.items():
            old_state = self.grid[pos].state
            self.grid[pos].state = new_state
            if new_state == old_state:
                self.grid[pos].age += 1
            else:
                self.grid[pos].age = 0

        self.generation += 1

    def count_alive(self) -> int:
        """Count alive cells"""
        return sum(1 for cell in self.grid.values() if cell.state == CellState.ALIVE)

    def detect_patterns(self) -> List[Dict[str, Any]]:
        """Detect common patterns (still lifes, oscillators, spaceships)"""
        patterns = []

        # Check for period-2 oscillators
        if len(self.history) >= 2:
            if self.history[-1] == self.history[-2]:
                patterns.append({"type": "still_life", "period": 1})
            elif len(self.history) >= 3 and self.history[-1] == self.history[-3]:
                patterns.append({"type": "oscillator", "period": 2})

        return patterns

    def get_entropy(self) -> float:
        """Calculate spatial entropy of the grid"""
        alive = self.count_alive()
        total = self.width * self.height

        if alive == 0 or alive == total:
            return 0.0

        p = alive / total
        return -p * math.log2(p) - (1-p) * math.log2(1-p)


# ============================================================================
# SELF-ORGANIZING SYSTEMS
# ============================================================================

@dataclass
class SOMNode:
    """Node in Self-Organizing Map"""
    position: Tuple[int, int]
    weights: List[float]
    activation: float = 0.0


class SelfOrganizingMap:
    """
    Kohonen Self-Organizing Map for unsupervised clustering
    and pattern formation.
    """

    def __init__(self, width: int = 10, height: int = 10, input_dim: int = 3):
        self.width = width
        self.height = height
        self.input_dim = input_dim
        self.nodes: Dict[Tuple[int, int], SOMNode] = {}

        # Initialize with random weights
        for x in range(width):
            for y in range(height):
                weights = [random.random() for _ in range(input_dim)]
                self.nodes[(x, y)] = SOMNode(
                    position=(x, y),
                    weights=weights
                )

        self.learning_rate = 0.5
        self.sigma = max(width, height) / 2  # Neighborhood radius
        self.iteration = 0

    def find_bmu(self, input_vec: List[float]) -> SOMNode:
        """Find Best Matching Unit"""
        best_node = None
        best_dist = float('inf')

        for node in self.nodes.values():
            dist = sum((i - w) ** 2 for i, w in zip(input_vec, node.weights))
            if dist < best_dist:
                best_dist = dist
                best_node = node

        return best_node

    def neighborhood_function(self, bmu_pos: Tuple[int, int],
                             node_pos: Tuple[int, int]) -> float:
        """Calculate neighborhood influence"""
        dist = math.sqrt(
            (bmu_pos[0] - node_pos[0])**2 +
            (bmu_pos[1] - node_pos[1])**2
        )
        return math.exp(-dist**2 / (2 * self.sigma**2))

    def train_step(self, input_vec: List[float]):
        """Train on one input vector"""
        bmu = self.find_bmu(input_vec)

        # Update all nodes
        for node in self.nodes.values():
            influence = self.neighborhood_function(bmu.position, node.position)

            for i in range(len(node.weights)):
                node.weights[i] += self.learning_rate * influence * (
                    input_vec[i] - node.weights[i]
                )

        # Decay learning parameters
        self.iteration += 1
        self.learning_rate *= 0.99
        self.sigma *= 0.99

    def get_activations(self, input_vec: List[float]) -> Dict[Tuple[int, int], float]:
        """Get activation levels for all nodes given input"""
        activations = {}
        for pos, node in self.nodes.items():
            dist = sum((i - w) ** 2 for i, w in zip(input_vec, node.weights))
            activations[pos] = math.exp(-dist)
        return activations


# ============================================================================
# EMERGENT PATTERN DETECTION
# ============================================================================

class EmergenceType(Enum):
    """Types of emergent phenomena"""
    CLUSTER_FORMATION = "cluster"
    PHASE_TRANSITION = "phase_transition"
    SYNCHRONIZATION = "synchronization"
    OSCILLATION = "oscillation"
    STRATIFICATION = "stratification"
    POLARIZATION = "polarization"
    CASCADE = "cascade"
    EQUILIBRIUM = "equilibrium"


@dataclass
class EmergentPattern:
    """A detected emergent pattern"""
    pattern_type: EmergenceType
    strength: float  # 0-1 how strong the pattern
    description: str
    involved_elements: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmergenceDetector:
    """
    Detects emergent patterns in multi-agent systems.
    """

    def __init__(self):
        self.history: List[Dict[str, Any]] = []
        self.detected_patterns: List[EmergentPattern] = []
        self.thresholds = {
            EmergenceType.CLUSTER_FORMATION: 0.6,
            EmergenceType.PHASE_TRANSITION: 0.7,
            EmergenceType.SYNCHRONIZATION: 0.8,
            EmergenceType.POLARIZATION: 0.7
        }

    def record_state(self, state: Dict[str, Any]):
        """Record a system state for analysis"""
        self.history.append({
            "state": state,
            "timestamp": datetime.now()
        })

        if len(self.history) > 1000:
            self.history.pop(0)

    def detect_clustering(self, positions: List[Tuple[float, float]],
                         labels: List[str] = None) -> Optional[EmergentPattern]:
        """Detect spatial clustering"""
        if len(positions) < 3:
            return None

        # Simple clustering detection via nearest neighbor analysis
        total_nn_dist = 0
        for i, pos1 in enumerate(positions):
            min_dist = float('inf')
            for j, pos2 in enumerate(positions):
                if i != j:
                    dist = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    min_dist = min(min_dist, dist)
            total_nn_dist += min_dist

        avg_nn_dist = total_nn_dist / len(positions)

        # Compare to random expectation
        area = max(p[0] for p in positions) * max(p[1] for p in positions)
        expected_nn = 0.5 * math.sqrt(area / len(positions)) if len(positions) > 0 else 0

        if expected_nn > 0:
            clustering_ratio = 1 - (avg_nn_dist / expected_nn)
        else:
            clustering_ratio = 0

        if clustering_ratio > self.thresholds[EmergenceType.CLUSTER_FORMATION]:
            pattern = EmergentPattern(
                pattern_type=EmergenceType.CLUSTER_FORMATION,
                strength=clustering_ratio,
                description=f"Spatial clustering detected with ratio {clustering_ratio:.2f}",
                involved_elements=labels or []
            )
            self.detected_patterns.append(pattern)
            return pattern

        return None

    def detect_synchronization(self, phases: List[float]) -> Optional[EmergentPattern]:
        """Detect phase synchronization (all values becoming similar)"""
        if len(phases) < 2:
            return None

        mean_phase = sum(phases) / len(phases)
        variance = sum((p - mean_phase)**2 for p in phases) / len(phases)
        sync_score = 1 - min(1, variance)

        if sync_score > self.thresholds[EmergenceType.SYNCHRONIZATION]:
            pattern = EmergentPattern(
                pattern_type=EmergenceType.SYNCHRONIZATION,
                strength=sync_score,
                description=f"Synchronization detected with score {sync_score:.2f}",
                involved_elements=[]
            )
            self.detected_patterns.append(pattern)
            return pattern

        return None

    def detect_phase_transition(self, order_parameter: float,
                               history: List[float] = None) -> Optional[EmergentPattern]:
        """Detect phase transitions (sudden changes in order parameter)"""
        if not history or len(history) < 3:
            return None

        # Calculate rate of change
        recent = history[-10:]
        if len(recent) >= 2:
            avg_change = sum(abs(recent[i] - recent[i-1]) for i in range(1, len(recent))) / (len(recent) - 1)

            # Check for sudden change
            if len(history) >= 2:
                current_change = abs(history[-1] - history[-2])
                if avg_change > 0 and current_change > avg_change * 3:
                    pattern = EmergentPattern(
                        pattern_type=EmergenceType.PHASE_TRANSITION,
                        strength=min(1.0, current_change / avg_change / 3),
                        description=f"Phase transition detected - order parameter changed rapidly",
                        involved_elements=[],
                        metadata={"order_parameter": order_parameter, "change_ratio": current_change / avg_change if avg_change > 0 else 0}
                    )
                    self.detected_patterns.append(pattern)
                    return pattern

        return None

    def detect_polarization(self, opinions: List[float]) -> Optional[EmergentPattern]:
        """Detect opinion polarization (bimodal distribution)"""
        if len(opinions) < 4:
            return None

        # Check for bimodal distribution
        low = sum(1 for o in opinions if o < 0.3)
        high = sum(1 for o in opinions if o > 0.7)
        middle = len(opinions) - low - high

        polarization = (low + high - middle) / len(opinions)

        if polarization > self.thresholds[EmergenceType.POLARIZATION]:
            pattern = EmergentPattern(
                pattern_type=EmergenceType.POLARIZATION,
                strength=polarization,
                description=f"Polarization detected: {low} low, {high} high, {middle} middle",
                involved_elements=[],
                metadata={"low_count": low, "high_count": high, "middle_count": middle}
            )
            self.detected_patterns.append(pattern)
            return pattern

        return None

    def get_emergence_summary(self) -> Dict[str, Any]:
        """Get summary of detected emergence"""
        pattern_counts = defaultdict(int)
        for pattern in self.detected_patterns:
            pattern_counts[pattern.pattern_type.value] += 1

        return {
            "total_patterns": len(self.detected_patterns),
            "pattern_counts": dict(pattern_counts),
            "recent_patterns": [
                {
                    "type": p.pattern_type.value,
                    "strength": p.strength,
                    "description": p.description
                }
                for p in self.detected_patterns[-10:]
            ]
        }


# ============================================================================
# COMPLEX ADAPTIVE SYSTEM
# ============================================================================

@dataclass
class AdaptiveAgent:
    """An agent that adapts based on feedback"""
    id: str
    strategy: List[float]  # Strategy parameters
    fitness: float = 0.0
    age: int = 0
    lineage: List[str] = field(default_factory=list)


class ComplexAdaptiveSystem:
    """
    A complex adaptive system with evolutionary dynamics.
    Agents adapt, reproduce, and compete.
    """

    def __init__(self, strategy_dim: int = 5, max_population: int = 100):
        self.strategy_dim = strategy_dim
        self.max_population = max_population
        self.agents: Dict[str, AdaptiveAgent] = {}
        self.generation = 0
        self.fitness_history: List[Dict[str, float]] = []
        self.mutation_rate = 0.1
        self.crossover_rate = 0.3

        # Environment can change
        self.environment: Dict[str, float] = {}

    def spawn_agent(self, agent_id: str = None,
                   strategy: List[float] = None) -> AdaptiveAgent:
        """Spawn a new agent"""
        if agent_id is None:
            agent_id = f"agent_{len(self.agents)}_{self.generation}"

        if strategy is None:
            strategy = [random.random() for _ in range(self.strategy_dim)]

        agent = AdaptiveAgent(id=agent_id, strategy=strategy)
        self.agents[agent_id] = agent
        return agent

    def evaluate_fitness(self, agent: AdaptiveAgent,
                        fitness_func: Callable[[List[float], Dict], float] = None) -> float:
        """Evaluate agent fitness"""
        if fitness_func:
            agent.fitness = fitness_func(agent.strategy, self.environment)
        else:
            # Default: match environment
            agent.fitness = 1.0 - sum(
                abs(agent.strategy[i % len(agent.strategy)] - v)
                for i, (k, v) in enumerate(sorted(self.environment.items()))
            ) / max(1, len(self.environment))

        return agent.fitness

    def mutate(self, strategy: List[float]) -> List[float]:
        """Mutate a strategy"""
        new_strategy = []
        for s in strategy:
            if random.random() < self.mutation_rate:
                new_strategy.append(max(0, min(1, s + random.gauss(0, 0.1))))
            else:
                new_strategy.append(s)
        return new_strategy

    def crossover(self, parent1: AdaptiveAgent, parent2: AdaptiveAgent) -> List[float]:
        """Crossover two parent strategies"""
        child_strategy = []
        for i in range(self.strategy_dim):
            if random.random() < 0.5:
                child_strategy.append(parent1.strategy[i])
            else:
                child_strategy.append(parent2.strategy[i])
        return child_strategy

    def selection(self) -> List[AdaptiveAgent]:
        """Select agents for reproduction (tournament selection)"""
        selected = []
        agents = list(self.agents.values())

        while len(selected) < len(agents) // 2:
            tournament = random.sample(agents, min(3, len(agents)))
            winner = max(tournament, key=lambda a: a.fitness)
            selected.append(winner)

        return selected

    def evolve(self, fitness_func: Callable = None):
        """Run one generation of evolution"""
        # Evaluate fitness
        for agent in self.agents.values():
            self.evaluate_fitness(agent, fitness_func)
            agent.age += 1

        # Record fitness
        self.fitness_history.append({
            agent_id: agent.fitness
            for agent_id, agent in self.agents.items()
        })

        # Selection
        parents = self.selection()

        # Create next generation
        new_agents = {}

        # Keep top performers (elitism)
        sorted_agents = sorted(self.agents.values(), key=lambda a: -a.fitness)
        for elite in sorted_agents[:max(1, len(sorted_agents) // 10)]:
            new_agents[elite.id] = elite

        # Reproduce
        while len(new_agents) < min(self.max_population, len(self.agents)):
            if random.random() < self.crossover_rate and len(parents) >= 2:
                p1, p2 = random.sample(parents, 2)
                child_strategy = self.crossover(p1, p2)
            else:
                parent = random.choice(parents)
                child_strategy = parent.strategy.copy()

            # Mutate
            child_strategy = self.mutate(child_strategy)

            child = self.spawn_agent(strategy=child_strategy)
            child.lineage = [p.id for p in random.sample(parents, min(2, len(parents)))]
            new_agents[child.id] = child

        self.agents = new_agents
        self.generation += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get population statistics"""
        if not self.agents:
            return {}

        fitnesses = [a.fitness for a in self.agents.values()]

        return {
            "generation": self.generation,
            "population_size": len(self.agents),
            "avg_fitness": sum(fitnesses) / len(fitnesses),
            "max_fitness": max(fitnesses),
            "min_fitness": min(fitnesses),
            "fitness_variance": sum((f - sum(fitnesses)/len(fitnesses))**2 for f in fitnesses) / len(fitnesses)
        }


# ============================================================================
# INTEGRATED ENGINE
# ============================================================================

class EmergentBehaviorEngine:
    """
    Master engine integrating all emergence and self-organization systems.
    """

    def __init__(self):
        self.swarm = SwarmSimulation()
        self.automaton = CellularAutomaton()
        self.som = SelfOrganizingMap()
        self.cas = ComplexAdaptiveSystem()
        self.detector = EmergenceDetector()

        self.time_step = 0
        self.event_log: List[Dict] = []

    def step(self):
        """Advance all systems by one step"""
        self.time_step += 1

        # Step each system
        self.swarm.step()
        self.automaton.step()

        # Detect emergence
        swarm_metrics = self.swarm.get_swarm_metrics()
        if "alignment" in swarm_metrics:
            self.detector.detect_synchronization(
                [a.velocity.magnitude for a in self.swarm.agents.values()]
            )

        ca_entropy = self.automaton.get_entropy()
        if self.time_step > 1:
            self.detector.detect_phase_transition(
                ca_entropy,
                [h.get("entropy", 0) for h in self.event_log[-10:]]
            )

        # Log state
        self.event_log.append({
            "time_step": self.time_step,
            "swarm_metrics": swarm_metrics,
            "ca_alive": self.automaton.count_alive(),
            "entropy": ca_entropy,
            "cas_stats": self.cas.get_statistics()
        })

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all systems"""
        return {
            "time_step": self.time_step,
            "swarm": self.swarm.get_swarm_metrics(),
            "automaton": {
                "generation": self.automaton.generation,
                "alive_cells": self.automaton.count_alive(),
                "entropy": self.automaton.get_entropy()
            },
            "cas": self.cas.get_statistics(),
            "emergence": self.detector.get_emergence_summary()
        }


# Convenience functions
def create_emergence_engine() -> EmergentBehaviorEngine:
    """Create a new emergence engine"""
    return EmergentBehaviorEngine()


def demo_emergence():
    """Demonstrate emergence systems"""
    engine = create_emergence_engine()

    # Initialize swarm
    for i in range(30):
        engine.swarm.add_agent(f"boid_{i}")

    # Initialize CA
    engine.automaton.randomize(0.3)

    # Initialize CAS
    for i in range(20):
        engine.cas.spawn_agent()
    engine.cas.environment = {"target_0": 0.8, "target_1": 0.3, "target_2": 0.5}

    # Run
    print("=== Emergence Demo ===")
    for _ in range(50):
        engine.step()
        if engine.time_step % 10 == 0:
            summary = engine.get_summary()
            print(f"\nStep {engine.time_step}:")
            print(f"  Swarm alignment: {summary['swarm'].get('alignment', 0):.2f}")
            print(f"  CA entropy: {summary['automaton']['entropy']:.2f}")
            print(f"  CAS avg fitness: {summary['cas'].get('avg_fitness', 0):.2f}")
            print(f"  Emergent patterns: {summary['emergence']['total_patterns']}")

        # Evolve CAS every 10 steps
        if engine.time_step % 10 == 0:
            engine.cas.evolve()

    return engine


if __name__ == "__main__":
    demo_emergence()
