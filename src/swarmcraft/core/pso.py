"""
Particle Swarm Optimization (PSO) implementation.

This module implements the classic PSO algorithm with enhancements for
human interaction and experiential learning. It includes mechanisms for
random exploration behavior and social dynamics that mirror real
collaborative intelligence.
"""

import numpy as np
from typing import List, Optional, Callable, Tuple
import random

from swarmcraft.core.swarm_base import (
    SwarmOptimizer,
    Particle,
    ParticleState,
    SwarmState,
)


class PSO(SwarmOptimizer):
    """
    Particle Swarm Optimization implementation.

    This implementation includes the classic PSO algorithm with additional
    features for human interaction:
    - Random exploration assignments for "weird ant" behavior
    - Adaptive inertia for dynamic optimization phases
    - Social influence parameters that can be adjusted for group dynamics
    """

    def __init__(
        self,
        dimensions: int,
        bounds: List[Tuple[float, float]],
        loss_function: Callable[[np.ndarray], float],
        population_size: int = 20,
        max_iterations: int = 100,
        # PSO-specific parameters
        inertia_weight: float = 0.9,
        cognitive_coefficient: float = 2.0,
        social_coefficient: float = 2.0,
        # Human interaction parameters
        exploration_probability: float = 0.1,
        min_exploration_probability: Optional[float] = None,
        min_inertia: float = 0.4,
        max_velocity_factor: float = 0.2,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize PSO optimizer.

        Args:
            dimensions: Number of dimensions in search space
            bounds: List of (min, max) bounds for each dimension
            loss_function: Function to minimize
            population_size: Number of particles
            max_iterations: Maximum optimization iterations
            inertia_weight: Starting inertia weight (momentum)
            cognitive_coefficient: Personal best influence (c1)
            social_coefficient: Global best influence (c2)
            exploration_probability: Chance a particle becomes explorer each round
            min_exploration_probability: The minimum exploration probability to decay to. If None, it remains constant.
            min_inertia: Minimum inertia weight for adaptive decay
            max_velocity_factor: Maximum velocity as fraction of search space
            random_seed: Random seed for reproducibility
        """
        super().__init__(
            dimensions=dimensions,
            bounds=bounds,
            loss_function=loss_function,
            population_size=population_size,
            max_iterations=max_iterations,
            random_seed=random_seed,
        )

        # PSO parameters
        self.inertia_weight = inertia_weight
        self.initial_inertia = inertia_weight
        self.cognitive_coefficient = cognitive_coefficient
        self.social_coefficient = social_coefficient

        # Human interaction parameters
        self.initial_exploration_probability = exploration_probability
        self.current_exploration_probability = exploration_probability
        self.min_exploration_probability = min_exploration_probability
        self.min_inertia = min_inertia

        # Calculate maximum velocity based on search space
        search_range = self.bounds[:, 1] - self.bounds[:, 0]
        self.max_velocity = max_velocity_factor * search_range

    def step(self) -> SwarmState:
        """
        Perform one iteration of the optimization algorithm.

        Returns:
            Updated swarm state
        """
        # Increment iteration at the beginning of the step for correct parameter calculation
        self.swarm_state.iteration += 1

        # Update particles using algorithm-specific rules
        self._update_particles()

        # Evaluate fitness and update personal/global bests
        self._evaluate_and_update_bests()

        # Update phase based on the new iteration count
        self.swarm_state.phase = self._determine_phase()

        # Store history
        self.history.append(self.swarm_state.model_copy(deep=True))

        return self.swarm_state

    def _update_adaptive_parameters(self) -> None:
        """Update adaptive parameters using linear decay based on progress."""
        # Avoid division by zero if max_iterations is 0 or 1
        if self.max_iterations <= 1:
            return

        # Clamp progress between 0.0 and 1.0 to prevent strange behavior at the end
        progress = min(1.0, self.swarm_state.iteration / self.max_iterations)

        # 1. Update inertia weight
        self.inertia_weight = (
            self.initial_inertia - (self.initial_inertia - self.min_inertia) * progress
        )

        # 2. Update exploration probability (if decay is configured)
        if (
            self.min_exploration_probability is not None
            and self.min_exploration_probability < self.initial_exploration_probability
        ):
            self.current_exploration_probability = (
                self.initial_exploration_probability
                - (
                    (
                        self.initial_exploration_probability
                        - self.min_exploration_probability
                    )
                    * progress
                )
            )

    def _update_particles(self) -> None:
        """
        Update particle positions and velocities using PSO rules.

        This method implements the core PSO algorithm with enhancements:
        1. Adaptive inertia weight that decreases over time
        2. Random exploration assignments for some particles
        3. Velocity clamping to prevent runaway behavior
        """
        # Update adaptive parameters
        self._update_adaptive_parameters()

        # Assign exploration roles randomly
        self._assign_exploration_roles()

        # Update each particle
        for particle in self.swarm_state.particles:
            if particle.state == ParticleState.EXPLORING:
                self._update_exploring_particle(particle)
            else:
                self._update_standard_particle(particle)

            # Clamp velocity and ensure bounds
            self._clamp_velocity(particle)

    def _assign_exploration_roles(self) -> None:
        """
        Randomly assign exploration roles to particles.

        This creates the "weird ant" behavior where some particles
        break from the swarm to explore unknown regions.
        """
        for particle in self.swarm_state.particles:
            if random.random() < self.current_exploration_probability:
                particle.state = ParticleState.EXPLORING
            else:
                particle.state = ParticleState.EXPLOITING

    def _update_standard_particle(self, particle: Particle) -> None:
        """
        Update particle using standard PSO velocity equation.

        v(t+1) = w*v(t) + c1*r1*(pbest - x(t)) + c2*r2*(gbest - x(t))
        x(t+1) = x(t) + v(t+1)
        """
        # Get current values as numpy arrays
        position = particle.position_array
        velocity = particle.velocity_array
        personal_best = particle.personal_best_array
        global_best = self.swarm_state.global_best_array

        # Generate random coefficients
        r1 = np.random.random(self.dimensions)
        r2 = np.random.random(self.dimensions)

        # Calculate velocity components
        inertia_component = self.inertia_weight * velocity
        cognitive_component = (
            self.cognitive_coefficient * r1 * (personal_best - position)
        )
        social_component = self.social_coefficient * r2 * (global_best - position)

        # Update velocity and position
        new_velocity = inertia_component + cognitive_component + social_component
        new_position = position + new_velocity

        # Update particle
        particle.update_velocity(new_velocity)
        particle.update_position(new_position)

    def _update_exploring_particle(self, particle: Particle) -> None:
        """
        Update exploring particle with more random behavior.

        Exploring particles have reduced social influence and added
        random velocity components to encourage discovery of new regions.
        """
        # Get current values
        position = particle.position_array
        velocity = particle.velocity_array
        personal_best = particle.personal_best_array
        global_best = self.swarm_state.global_best_array

        # Generate random coefficients (higher randomness)
        r1 = np.random.random(self.dimensions)
        r2 = np.random.random(self.dimensions) * 0.5  # Reduced social influence
        r3 = np.random.random(self.dimensions)  # Random exploration

        # Calculate velocity components
        inertia_component = self.inertia_weight * velocity
        cognitive_component = (
            self.cognitive_coefficient * r1 * (personal_best - position)
        )
        social_component = self.social_coefficient * r2 * (global_best - position)

        # Add random exploration component
        search_range = self.bounds[:, 1] - self.bounds[:, 0]
        exploration_component = (
            0.3 * r3 * search_range * (2 * np.random.random(self.dimensions) - 1)
        )

        # Update velocity and position
        new_velocity = (
            inertia_component
            + cognitive_component
            + social_component
            + exploration_component
        )
        new_position = position + new_velocity

        # Update particle
        particle.update_velocity(new_velocity)
        particle.update_position(new_position)

    def _clamp_velocity(self, particle: Particle) -> None:
        """Clamp particle velocity to prevent runaway behavior."""
        velocity = particle.velocity_array
        velocity = np.clip(velocity, -self.max_velocity, self.max_velocity)
        particle.update_velocity(velocity)

    def get_pso_statistics(self) -> dict:
        """Get PSO-specific statistics."""
        base_stats = self.get_swarm_statistics()

        if not self.swarm_state:
            return base_stats

        # Calculate velocity statistics
        velocities = [
            np.linalg.norm(p.velocity_array) for p in self.swarm_state.particles
        ]

        pso_stats = {
            "current_inertia": self.inertia_weight,
            "current_exploration_probability": self.current_exploration_probability,
            "velocity_stats": {
                "mean_speed": float(np.mean(velocities)),
                "max_speed": float(np.max(velocities)),
                "min_speed": float(np.min(velocities)),
            },
            "exploration_stats": {
                "explorers": len(
                    [
                        p
                        for p in self.swarm_state.particles
                        if p.state == ParticleState.EXPLORING
                    ]
                ),
                "exploiters": len(
                    [
                        p
                        for p in self.swarm_state.particles
                        if p.state == ParticleState.EXPLOITING
                    ]
                ),
            },
        }

        # Merge with base statistics
        return {**base_stats, **pso_stats}

    def get_particle_info(self, particle_id: str) -> Optional[dict]:
        """
        Get detailed information about a specific particle.

        This is useful for providing personalized feedback to human participants.
        """
        particle = self.get_particle_by_id(particle_id)
        if not particle:
            return None

        return {
            "id": particle.id,
            "position": particle.position,
            "fitness": particle.fitness,
            "personal_best_fitness": particle.personal_best_fitness,
            "velocity_magnitude": float(np.linalg.norm(particle.velocity_array)),
            "state": particle.state.value,
            "distance_to_global_best": float(
                np.linalg.norm(
                    particle.position_array - self.swarm_state.global_best_array
                )
            ),
            "improvement_over_personal_best": (
                particle.personal_best_fitness - particle.fitness
                if particle.fitness < particle.personal_best_fitness
                else 0.0
            ),
        }

    def suggest_next_position(self, particle_id: str) -> Optional[List[float]]:
        """
        Suggest next position for a particle without updating the swarm.

        This can be used to give participants hints about where to move
        without forcing the movement.
        """
        particle = self.get_particle_by_id(particle_id)
        if not particle:
            return None

        # Temporarily update this particle to get suggestion
        if particle.state == ParticleState.EXPLORING:
            # For explorers, suggest a more random position
            search_range = self.bounds[:, 1] - self.bounds[:, 0]
            random_offset = (
                0.1 * search_range * (2 * np.random.random(self.dimensions) - 1)
            )
            suggested_position = particle.position_array + random_offset
        else:
            # For normal particles, use standard PSO prediction
            personal_best = particle.personal_best_array
            global_best = self.swarm_state.global_best_array
            position = particle.position_array

            # Simple suggestion: move toward weighted average of personal and global best
            # relative to current position
            alpha = 0.3  # Weight for personal best direction
            beta = 0.7  # Weight for global best direction

            personal_direction = personal_best - position
            global_direction = global_best - position

            # Suggest position based on weighted directions from current position
            movement = alpha * personal_direction + beta * global_direction
            suggested_position = position + 0.5 * movement  # Take half step

        # Ensure within bounds
        suggested_position = self._enforce_bounds(suggested_position)
        return suggested_position.tolist()

    def force_exploration_phase(self) -> None:
        """
        Force all particles into exploration mode.

        This can be used during the experience to demonstrate
        the value of exploration vs exploitation.
        """
        if self.swarm_state:
            for particle in self.swarm_state.particles:
                particle.state = ParticleState.EXPLORING

    def force_exploitation_phase(self) -> None:
        """Force all particles into exploitation mode."""
        if self.swarm_state:
            for particle in self.swarm_state.particles:
                particle.state = ParticleState.EXPLOITING
