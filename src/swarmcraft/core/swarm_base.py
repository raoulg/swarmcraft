"""
Base class for swarm optimization algorithms.

This module provides the foundational structure for implementing various
swarm intelligence algorithms like PSO, ABC, etc. It enforces the common
interface while allowing algorithm-specific implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional, Callable
import numpy as np
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum
from swarmcraft.models.session import Participant
from loguru import logger


class ParticleState(Enum):
    """Represents the current behavioral state of a particle."""

    EXPLORING = "exploring"  # Random exploration phase
    EXPLOITING = "exploiting"  # Following swarm/personal best
    SHARING = "sharing"  # Broadcasting information to swarm


class Particle(BaseModel):
    """Represents a single particle in the swarm."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    position: List[float]
    velocity: List[float]
    fitness: float
    personal_best_position: List[float]
    personal_best_fitness: float
    state: ParticleState = ParticleState.EXPLOITING

    @field_validator("position", "velocity", "personal_best_position")
    @classmethod
    def validate_arrays(cls, v):
        """Ensure positions and velocities are valid lists."""
        if not isinstance(v, (list, np.ndarray)):
            raise ValueError("Position and velocity must be lists or numpy arrays")
        return list(v) if isinstance(v, np.ndarray) else v

    @property
    def position_array(self) -> np.ndarray:
        """Get position as numpy array."""
        return np.array(self.position)

    @property
    def velocity_array(self) -> np.ndarray:
        """Get velocity as numpy array."""
        return np.array(self.velocity)

    @property
    def personal_best_array(self) -> np.ndarray:
        """Get personal best position as numpy array."""
        return np.array(self.personal_best_position)

    def update_position(self, new_position: np.ndarray) -> None:
        """Update position from numpy array."""
        self.position = new_position.tolist()

    def update_velocity(self, new_velocity: np.ndarray) -> None:
        """Update velocity from numpy array."""
        self.velocity = new_velocity.tolist()

    def update_personal_best(self, position: np.ndarray, fitness: float) -> None:
        """Update personal best position and fitness."""
        self.personal_best_position = position.tolist()
        self.personal_best_fitness = fitness


class SwarmState(BaseModel):
    """Represents the current state of the entire swarm."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    particles: List[Particle]
    global_best_position: List[float]
    global_best_fitness: float
    global_best_particle_id: str
    iteration: int
    phase: str = Field(..., description="Current optimization phase")

    @field_validator("global_best_position")
    @classmethod
    def validate_global_best(cls, v):
        """Ensure global best position is a valid list."""
        if not isinstance(v, (list, np.ndarray)):
            raise ValueError("Global best position must be a list or numpy array")
        return list(v) if isinstance(v, np.ndarray) else v

    @field_validator("phase")
    @classmethod
    def validate_phase(cls, v):
        """Validate phase is one of the expected values."""
        valid_phases = ["initialization", "optimization", "convergence"]
        if v not in valid_phases:
            raise ValueError(f"Phase must be one of: {valid_phases}")
        return v

    @property
    def global_best_array(self) -> np.ndarray:
        """Get global best position as numpy array."""
        return np.array(self.global_best_position)

    def update_global_best(
        self, position: np.ndarray, fitness: float, particle_id: str
    ) -> None:
        """Update global best position, fitness, and particle ID."""
        self.global_best_position = position.tolist()
        self.global_best_fitness = fitness
        self.global_best_particle_id = particle_id


class SwarmOptimizer(ABC):
    """
    Abstract base class for swarm optimization algorithms.

    This class defines the common interface and shared functionality
    for swarm-based optimization algorithms. Subclasses must implement
    the algorithm-specific update rules.
    """

    def __init__(
        self,
        dimensions: int,
        bounds: List[Tuple[float, float]],
        loss_function: Callable[[np.ndarray], float],
        initial_positions: Optional[List[np.ndarray]] = None,
        population_size: int = 20,
        max_iterations: int = 100,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the swarm optimizer.

        Args:
            dimensions: Number of dimensions in the search space
            bounds: List of (min, max) tuples for each dimension
            loss_function: Function to minimize f(position) -> fitness_score
            population_size: Number of particles in the swarm
            max_iterations: Maximum number of optimization iterations
            random_seed: Seed for reproducible results
        """
        self.dimensions = dimensions
        self.bounds = np.array(bounds)
        self.loss_function = loss_function
        self.population_size = population_size
        self.max_iterations = max_iterations

        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialize swarm state
        self.history: List[SwarmState] = []

        # Initialize swarm by default
        self.swarm_state: SwarmState = self.initialize_swarm(initial_positions)

    def initialize_swarm(
        self, initial_positions: Optional[List[np.ndarray]] = None
    ) -> SwarmState:
        """
        Initialize the swarm with random or provided positions.

        Args:
            initial_positions: Optional list of initial positions for particles

        Returns:
            Initial swarm state
        """
        particles = []
        logger.info(f"init swarm with bounds: {self.bounds}")

        for i in range(self.population_size):
            # Use provided position or generate random one
            if initial_positions and i < len(initial_positions):
                position = np.array(initial_positions[i])
            else:
                position = self._generate_random_position()

            # Initialize particle
            fitness = self.loss_function(position)
            particle = Particle(
                id=f"particle_{i}",
                position=position.tolist(),
                velocity=[0.0] * self.dimensions,
                fitness=fitness,
                personal_best_position=position.tolist(),
                personal_best_fitness=fitness,
            )
            particles.append(particle)

        # Find global best
        best_particle = min(particles, key=lambda p: p.fitness)

        self.swarm_state = SwarmState(
            particles=particles,
            global_best_position=best_particle.position,
            global_best_fitness=best_particle.fitness,
            global_best_particle_id=best_particle.id,
            iteration=0,
            phase="initialization",
        )

        self.history.append(self.swarm_state.model_copy(deep=True))
        return self.swarm_state

    def step(self) -> SwarmState:
        """
        Perform one iteration of the optimization algorithm.

        Returns:
            Updated swarm state
        """
        # Update particles using algorithm-specific rules
        self._update_particles()

        # Evaluate fitness and update personal/global bests
        self._evaluate_and_update_bests()

        # Update iteration counter and phase
        self.swarm_state.iteration += 1
        self.swarm_state.phase = self._determine_phase()

        # Store history
        self.history.append(self.swarm_state.model_copy(deep=True))

        return self.swarm_state

    @abstractmethod
    def _update_particles(self) -> None:
        """
        Update particle positions and velocities.

        This method must be implemented by each specific algorithm
        (PSO, ABC, etc.) with their own update rules.
        """
        pass

    def _evaluate_and_update_bests(self) -> bool:
        """Evaluate fitness and update personal and global bests."""
        global_best_updated = False

        for particle in self.swarm_state.particles:
            # Ensure particle is within bounds
            position_array = self._enforce_bounds(particle.position_array)
            particle.update_position(position_array)

            # Evaluate fitness
            particle.fitness = self.loss_function(particle.position_array)

            # Update personal best
            if particle.fitness < particle.personal_best_fitness:
                particle.update_personal_best(particle.position_array, particle.fitness)

            # Check for global best update
            if particle.fitness < self.swarm_state.global_best_fitness:
                self.swarm_state.update_global_best(
                    particle.position_array, particle.fitness, particle.id
                )
                global_best_updated = True

        return global_best_updated

    def _generate_random_position(self) -> np.ndarray:
        """Generate a random position within bounds."""
        return np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], size=self.dimensions
        )

    def _enforce_bounds(self, position: np.ndarray) -> np.ndarray:
        """Enforce boundary constraints on position."""
        return np.clip(position, self.bounds[:, 0], self.bounds[:, 1])

    def _determine_phase(self) -> str:
        """Determine current optimization phase based on progress."""
        progress = self.swarm_state.iteration / self.max_iterations

        if progress < 0.1:
            return "initialization"
        elif progress < 0.8:
            return "optimization"
        else:
            return "convergence"

    def _copy_swarm_state(self, state: SwarmState) -> SwarmState:
        """Create a deep copy of swarm state for history."""
        # Pydantic V2 handles deep copying automatically
        return state.model_copy(deep=True)

    def get_particle_by_id(self, particle_id: str) -> Optional[Particle]:
        """Get particle by ID."""
        for particle in self.swarm_state.particles:
            if particle.id == particle_id:
                return particle
        return None

    def get_swarm_statistics(self) -> Dict[str, Any]:
        """Get current swarm statistics."""
        fitnesses = [p.fitness for p in self.swarm_state.particles]

        return {
            "iteration": self.swarm_state.iteration,
            "phase": self.swarm_state.phase,
            "global_best_fitness": self.swarm_state.global_best_fitness,
            "global_best_position": self.swarm_state.global_best_position,
            "global_best_particle": self.swarm_state.global_best_particle_id,
            "fitness_stats": {
                "mean": float(np.mean(fitnesses)),
                "std": float(np.std(fitnesses)),
                "min": float(np.min(fitnesses)),
                "max": float(np.max(fitnesses)),
            },
            "diversity": self._calculate_diversity(),
            "particles_exploring": len(
                [
                    p
                    for p in self.swarm_state.particles
                    if p.state == ParticleState.EXPLORING
                ]
            ),
        }

    def _calculate_diversity(self) -> float:
        """Calculate swarm diversity (average distance between particles)."""
        if len(self.swarm_state.particles) < 2:
            return 0.0

        positions = np.array([p.position for p in self.swarm_state.particles])
        distances = []

        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distances.append(np.linalg.norm(positions[i] - positions[j]))

        return float(np.mean(distances)) if distances else 0.0

    def sync_participants_from_swarm(
        self,
        participants: List[Participant],
        grid_size: int,
    ) -> List[Participant]:
        """
        Updates a list of Participant objects with the current state from the swarm
        and returns the updated list.
        """
        if len(participants) != len(self.swarm_state.particles):
            print("Warning: Participant count and particle count do not match.")
            return participants

        bounds = self.bounds

        for i, participant in enumerate(participants):
            particle = self.swarm_state.particles[i]
            continuous_pos = particle.position

            x, y = continuous_pos[0], continuous_pos[1]
            col = int((x - bounds[0][0]) / (bounds[0][1] - bounds[0][0]) * grid_size)
            row = int((y - bounds[1][0]) / (bounds[1][1] - bounds[1][0]) * grid_size)
            grid_pos = [
                max(0, min(grid_size - 1, col)),
                max(0, min(grid_size - 1, row)),
            ]

            # Calculate normalized position [0..1]
            norm_x = (x - bounds[0][0]) / (bounds[0][1] - bounds[0][0])
            norm_y = (y - bounds[1][0]) / (bounds[1][1] - bounds[1][0])

            participant.position = grid_pos
            participant.continuous_position = [float(x), float(y)]
            participant.normalized_position = [float(norm_x), float(norm_y)]
            participant.fitness = particle.fitness
            participant.velocity_magnitude = float(
                np.linalg.norm(particle.velocity_array)
            )

        return participants
