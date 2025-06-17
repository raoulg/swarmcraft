"""
Artificial Bee Colony (ABC) optimization algorithm implementation.

Minimal implementation that fits the existing SwarmOptimizer structure.
Uses existing ParticleState enum without adding new fields yet.
"""

import numpy as np
from typing import List, Optional, Callable, Tuple
import random

from swarmcraft.core.swarm_base import (
    SwarmOptimizer,
    ParticleState,
    SwarmState,
)


class ABC(SwarmOptimizer):
    """
    Artificial Bee Colony optimization algorithm.

    Minimal implementation using existing structure:
    - Employed bees: ParticleState.EXPLOITING (working known sources)
    - Onlooker bees: ParticleState.SHARING (choosing based on others)
    - Scout bees: ParticleState.EXPLORING (random search)
    """

    def __init__(
        self,
        dimensions: int,
        bounds: List[Tuple[float, float]],
        loss_function: Callable[[np.ndarray], float],
        population_size: int = 20,
        max_iterations: int = 100,
        # ABC-specific parameters
        limit: int = 10,  # Abandonment limit for food sources
        employed_ratio: float = 0.5,  # Fraction that are employed bees
        random_seed: Optional[int] = None,
    ):
        """Initialize ABC optimizer with minimal changes to existing structure."""
        super().__init__(
            dimensions=dimensions,
            bounds=bounds,
            loss_function=loss_function,
            population_size=population_size,
            max_iterations=max_iterations,
            random_seed=random_seed,
        )

        self.limit = limit
        self.employed_ratio = employed_ratio

        # Calculate bee counts
        self.employed_count = int(population_size * employed_ratio)
        self.onlooker_count = population_size - self.employed_count

        # Track food source abandonment (indexed by employed bee)
        self.trial_counter = np.zeros(self.employed_count)

        # Assign initial states using existing ParticleState enum
        self._assign_initial_states()

    def _assign_initial_states(self):
        """Assign initial states to particles using existing enum."""
        for i, particle in enumerate(self.swarm_state.particles):
            if i < self.employed_count:
                particle.state = ParticleState.EXPLOITING  # Employed bees
            else:
                particle.state = ParticleState.SHARING  # Onlooker bees

    def _update_particles(self) -> None:
        """
        Update particles using ABC algorithm.

        This implements the abstract method from SwarmOptimizer.
        """
        # 1. Employed bee phase
        self._employed_bee_phase()

        # 2. Onlooker bee phase
        self._onlooker_bee_phase()

        # 3. Scout bee phase
        self._scout_bee_phase()

    def step(self) -> SwarmState:
        """Perform one iteration of ABC algorithm."""
        self.swarm_state.iteration += 1

        # Update particles using algorithm-specific rules
        self._update_particles()

        # Evaluate fitness and update personal/global bests
        self._evaluate_and_update_bests()

        # Update phase
        self.swarm_state.phase = self._determine_phase()

        # Store history
        self.history.append(self.swarm_state.model_copy(deep=True))

        return self.swarm_state

    def _employed_bee_phase(self):
        """Employed bees exploit their food sources."""
        for i in range(self.employed_count):
            particle = self.swarm_state.particles[i]

            # Generate candidate solution
            candidate_position = self._generate_candidate_solution(
                particle.position_array, i
            )
            candidate_fitness = self.loss_function(candidate_position)

            # Greedy selection
            if candidate_fitness < particle.fitness:
                particle.update_position(candidate_position)
                particle.fitness = candidate_fitness
                self.trial_counter[i] = 0  # Reset trial counter
            else:
                self.trial_counter[i] += 1  # Increment trial counter

    def _onlooker_bee_phase(self):
        """Onlooker bees choose food sources based on probability."""
        # Calculate selection probabilities based on fitness
        employed_fitnesses = [
            self.swarm_state.particles[i].fitness for i in range(self.employed_count)
        ]

        # Convert fitness to probability (lower fitness = higher probability)
        max_fitness = max(employed_fitnesses) if employed_fitnesses else 1.0
        probabilities = [(max_fitness - f + 1e-10) for f in employed_fitnesses]
        total_prob = sum(probabilities)

        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        else:
            probabilities = [1.0 / len(employed_fitnesses)] * len(employed_fitnesses)

        # Onlooker bees select food sources
        for i in range(self.employed_count, self.population_size):
            onlooker = self.swarm_state.particles[i]

            # Select food source based on probability
            selected_source_idx = np.random.choice(self.employed_count, p=probabilities)
            selected_source = self.swarm_state.particles[selected_source_idx]

            # Generate candidate solution near selected source
            candidate_position = self._generate_candidate_solution(
                selected_source.position_array, selected_source_idx
            )
            candidate_fitness = self.loss_function(candidate_position)

            # Update onlooker if candidate is better
            if candidate_fitness < onlooker.fitness:
                onlooker.update_position(candidate_position)
                onlooker.fitness = candidate_fitness

    def _scout_bee_phase(self):
        """Convert abandoned food sources to scout bees."""
        for i in range(self.employed_count):
            if self.trial_counter[i] >= self.limit:
                # Convert to scout bee - random search
                particle = self.swarm_state.particles[i]
                new_position = self._generate_random_position()
                particle.update_position(new_position)
                particle.fitness = self.loss_function(new_position)
                self.trial_counter[i] = 0

                # Temporarily change state to exploring
                particle.state = ParticleState.EXPLORING
            else:
                # Make sure employed bees stay in exploiting state
                if i < self.employed_count:
                    self.swarm_state.particles[i].state = ParticleState.EXPLOITING

    def _generate_candidate_solution(
        self, current_position: np.ndarray, source_idx: int
    ) -> np.ndarray:
        """Generate candidate solution by modifying current position."""
        # Select random dimension to modify
        j = random.randint(0, self.dimensions - 1)

        # Select random partner (different from current source)
        partner_idx = random.randint(0, self.employed_count - 1)
        while partner_idx == source_idx and self.employed_count > 1:
            partner_idx = random.randint(0, self.employed_count - 1)

        partner_position = self.swarm_state.particles[partner_idx].position_array

        # Generate candidate position
        candidate = current_position.copy()
        phi = random.uniform(-1, 1)  # Random number between -1 and 1
        candidate[j] = current_position[j] + phi * (
            current_position[j] - partner_position[j]
        )

        # Ensure bounds
        candidate = self._enforce_bounds(candidate)

        return candidate

    def get_abc_statistics(self) -> dict:
        """Get ABC-specific statistics (compatible with existing PSO structure)."""
        base_stats = self.get_swarm_statistics()

        # Count particles by state (using existing ParticleState)
        state_counts = {
            "employed_bees": len(
                [
                    p
                    for p in self.swarm_state.particles
                    if p.state == ParticleState.EXPLOITING
                ]
            ),
            "onlooker_bees": len(
                [
                    p
                    for p in self.swarm_state.particles
                    if p.state == ParticleState.SHARING
                ]
            ),
            "scout_bees": len(
                [
                    p
                    for p in self.swarm_state.particles
                    if p.state == ParticleState.EXPLORING
                ]
            ),
        }

        abc_stats = {
            "bee_distribution": state_counts,
            "abandoned_sources": int(np.sum(self.trial_counter >= self.limit)),
            "average_trials": float(np.mean(self.trial_counter)),
            "limit": self.limit,
        }

        return {**base_stats, **abc_stats}

    # Make this compatible with existing PSO interface
    def get_pso_statistics(self) -> dict:
        """Alias for get_abc_statistics to maintain compatibility."""
        return self.get_abc_statistics()

    def suggest_next_position(self, particle_id: str) -> Optional[List[float]]:
        """
        Suggest next position for a particle without updating the swarm.

        For ABC, this simulates the employed bee or onlooker bee behavior
        depending on the particle's current role.
        """
        particle = self.get_particle_by_id(particle_id)
        if not particle:
            return None

        # Find particle index to determine role
        particle_index = None
        for i, p in enumerate(self.swarm_state.particles):
            if p.id == particle_id:
                particle_index = i
                break

        if particle_index is None:
            return None

        if particle_index < self.employed_count:
            # Employed bee: suggest position based on current food source exploration
            candidate = self._generate_candidate_solution(
                particle.position_array, particle_index
            )
            return candidate.tolist()
        else:
            # Onlooker bee: suggest position based on best employed bee sources
            employed_fitnesses = [
                self.swarm_state.particles[i].fitness
                for i in range(self.employed_count)
            ]

            if employed_fitnesses:
                # Choose best employed bee to follow
                best_employed_idx = min(
                    range(len(employed_fitnesses)), key=lambda i: employed_fitnesses[i]
                )
                best_employed = self.swarm_state.particles[best_employed_idx]

                # Suggest position near the best employed bee
                candidate = self._generate_candidate_solution(
                    best_employed.position_array, best_employed_idx
                )
                return candidate.tolist()
            else:
                # Fallback: random position
                return self._generate_random_position().tolist()
