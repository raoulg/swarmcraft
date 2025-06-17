"""
Additional test class for ABC algorithm.
Add this to your existing tests/test_core.py file.
"""

import pytest
import numpy as np
from swarmcraft.core.abc import ABC
from swarmcraft.core.swarm_base import ParticleState
from swarmcraft.core.loss_functions import RastriginLandscape, EcologicalLandscape


class TestABC:
    """Test suite for ABC implementation."""

    def test_abc_initialization(self):
        """Test ABC initialization with automatic swarm creation."""

        def simple_loss(x):
            return np.sum(x**2)

        abc = ABC(
            dimensions=2,
            bounds=[(-5, 5), (-5, 5)],
            loss_function=simple_loss,
            population_size=10,
            limit=5,
            employed_ratio=0.5,
            random_seed=42,
        )

        # Should be automatically initialized
        assert abc.swarm_state is not None
        assert len(abc.swarm_state.particles) == 10
        assert abc.swarm_state.iteration == 0
        assert abc.swarm_state.phase == "initialization"

        # Check ABC-specific parameters
        assert abc.limit == 5
        assert abc.employed_ratio == 0.5
        assert abc.employed_count == 5
        assert abc.onlooker_count == 5

    def test_abc_initial_role_assignment(self):
        """Test that initial roles are assigned correctly using existing ParticleState."""

        def simple_loss(x):
            return np.sum(x**2)

        abc = ABC(
            dimensions=2,
            bounds=[(-5, 5), (-5, 5)],
            loss_function=simple_loss,
            population_size=10,
            employed_ratio=0.6,  # 6 employed, 4 onlooker
            random_seed=42,
        )

        # Check role assignment
        employed_count = sum(
            1 for p in abc.swarm_state.particles if p.state == ParticleState.EXPLOITING
        )
        onlooker_count = sum(
            1 for p in abc.swarm_state.particles if p.state == ParticleState.SHARING
        )

        assert employed_count == 6  # First 6 should be employed
        assert onlooker_count == 4  # Last 4 should be onlookers

    def test_abc_step_execution(self):
        """Test ABC step execution."""

        def simple_loss(x):
            return np.sum(x**2)

        abc = ABC(
            dimensions=2,
            bounds=[(-5, 5), (-5, 5)],
            loss_function=simple_loss,
            population_size=8,
            limit=3,
            random_seed=42,
        )

        initial_iteration = abc.swarm_state.iteration

        # Execute step
        new_state = abc.step()

        assert new_state.iteration == initial_iteration + 1
        assert len(abc.history) > 1  # History should be recorded

        # Fitness should be re-evaluated (might be same or different)
        assert isinstance(new_state.global_best_fitness, float)

    def test_abc_optimization_progress(self):
        """Test that ABC makes optimization progress."""

        def simple_loss(x):
            return np.sum(x**2)  # Simple quadratic with minimum at origin

        abc = ABC(
            dimensions=2,
            bounds=[(-5, 5), (-5, 5)],
            loss_function=simple_loss,
            population_size=20,
            max_iterations=50,
            limit=5,
            random_seed=42,
        )

        initial_fitness = abc.swarm_state.global_best_fitness

        # Run optimization
        for _ in range(20):
            abc.step()

        final_fitness = abc.swarm_state.global_best_fitness

        # Should improve (lower fitness is better)
        assert final_fitness <= initial_fitness

    def test_abc_scout_bee_creation(self):
        """Test that scout bees are created when sources are abandoned."""

        def simple_loss(x):
            return np.sum(x**2)

        abc = ABC(
            dimensions=2,
            bounds=[(-5, 5), (-5, 5)],
            loss_function=simple_loss,
            population_size=10,
            limit=2,  # Low limit to trigger abandonment quickly
            employed_ratio=0.5,
            random_seed=42,
        )

        # Force high trial counters to trigger scout behavior
        abc.trial_counter.fill(abc.limit)  # All employed sources are "abandoned"

        # Run a step to trigger scout bee phase
        abc.step()

        # Should have some scout bees (EXPLORING state)
        scout_count = sum(
            1 for p in abc.swarm_state.particles if p.state == ParticleState.EXPLORING
        )

        # At least some scouts should be created
        assert scout_count > 0

    def test_abc_with_rastrigin(self):
        """Test ABC on Rastrigin landscape."""
        landscape = RastriginLandscape(A=10.0, dimensions=2)

        abc = ABC(
            dimensions=2,
            bounds=landscape.metadata.recommended_bounds,
            loss_function=landscape.evaluate,
            population_size=20,
            limit=10,
            random_seed=42,
        )

        # Run optimization
        for _ in range(30):
            abc.step()

        # Should find reasonably good solution
        final_fitness = abc.swarm_state.global_best_fitness
        assert final_fitness < 50  # Should find something better than random

    def test_abc_statistics(self):
        """Test ABC statistics generation."""

        def simple_loss(x):
            return np.sum(x**2)

        abc = ABC(
            dimensions=2,
            bounds=[(-5, 5), (-5, 5)],
            loss_function=simple_loss,
            population_size=10,
            random_seed=42,
        )

        abc.step()  # Run one step to populate stats
        stats = abc.get_abc_statistics()

        # Check required fields
        assert "bee_distribution" in stats
        assert "abandoned_sources" in stats
        assert "average_trials" in stats
        assert "limit" in stats
        assert "global_best_fitness" in stats
        assert "diversity" in stats

        # Check bee distribution structure
        bee_dist = stats["bee_distribution"]
        assert "employed_bees" in bee_dist
        assert "onlooker_bees" in bee_dist
        assert "scout_bees" in bee_dist

        # Total should equal population size
        total_bees = sum(bee_dist.values())
        assert total_bees == 10

    def test_abc_pso_compatibility(self):
        """Test that ABC maintains compatibility with PSO interface."""

        def simple_loss(x):
            return np.sum(x**2)

        abc = ABC(
            dimensions=2,
            bounds=[(-5, 5), (-5, 5)],
            loss_function=simple_loss,
            population_size=10,
            random_seed=42,
        )

        # Should have PSO-compatible method
        stats = abc.get_pso_statistics()
        assert isinstance(stats, dict)
        assert "global_best_fitness" in stats

    def test_abc_candidate_solution_generation(self):
        """Test candidate solution generation mechanism."""

        def simple_loss(x):
            return np.sum(x**2)

        abc = ABC(
            dimensions=2,
            bounds=[(-5, 5), (-5, 5)],
            loss_function=simple_loss,
            population_size=10,
            employed_ratio=0.5,
            random_seed=42,
        )

        # Test candidate generation for first employed bee
        current_pos = abc.swarm_state.particles[0].position_array
        candidate = abc._generate_candidate_solution(current_pos, 0)

        # Should be within bounds
        assert -5 <= candidate[0] <= 5
        assert -5 <= candidate[1] <= 5

        # Should be different from original (very likely with random component)
        assert not np.array_equal(candidate, current_pos)

    def test_abc_employed_bee_phase(self):
        """Test employed bee phase behavior."""

        def simple_loss(x):
            return np.sum(x**2)

        abc = ABC(
            dimensions=2,
            bounds=[(-5, 5), (-5, 5)],
            loss_function=simple_loss,
            population_size=10,
            employed_ratio=0.5,
            random_seed=42,
        )

        # Run employed bee phase
        abc._employed_bee_phase()

        # Positions might have changed (depending on if candidates were better)
        # At minimum, trial counters should be updated
        assert len(abc.trial_counter) == 5
        assert all(counter >= 0 for counter in abc.trial_counter)

    def test_abc_onlooker_bee_phase(self):
        """Test onlooker bee phase behavior."""

        def simple_loss(x):
            return np.sum(x**2)

        abc = ABC(
            dimensions=2,
            bounds=[(-5, 5), (-5, 5)],
            loss_function=simple_loss,
            population_size=10,
            employed_ratio=0.5,
            random_seed=42,
        )

        # Run onlooker bee phase
        abc._onlooker_bee_phase()

        # Onlooker positions might have changed
        # This tests that the phase runs without error
        assert len(abc.swarm_state.particles[5:]) == 5

    def test_abc_with_ecological_landscape(self):
        """Test ABC optimization on ecological landscape."""
        landscape = EcologicalLandscape()

        abc = ABC(
            dimensions=2,
            bounds=landscape.metadata.recommended_bounds,
            loss_function=landscape.evaluate,
            population_size=20,
            max_iterations=100,
            limit=15,
            random_seed=42,
        )

        # Run optimization
        for _ in range(25):
            abc.step()

        # Should find reasonable solution
        final_fitness = abc.swarm_state.global_best_fitness
        final_position = abc.swarm_state.global_best_position

        # Should be better than random
        assert final_fitness < 100

        # Position should be within bounds
        bounds = landscape.metadata.recommended_bounds
        assert bounds[0][0] <= final_position[0] <= bounds[0][1]
        assert bounds[1][0] <= final_position[1] <= bounds[1][1]

    def test_abc_different_employed_ratios(self):
        """Test ABC with different employed bee ratios."""

        def simple_loss(x):
            return np.sum(x**2)

        for ratio in [0.3, 0.5, 0.7]:
            abc = ABC(
                dimensions=2,
                bounds=[(-5, 5), (-5, 5)],
                loss_function=simple_loss,
                population_size=10,
                employed_ratio=ratio,
                random_seed=42,
            )

            expected_employed = int(10 * ratio)
            expected_onlooker = 10 - expected_employed

            assert abc.employed_count == expected_employed
            assert abc.onlooker_count == expected_onlooker

            # Should still work
            abc.step()
            assert abc.swarm_state.iteration == 1

    def test_abc_position_suggestion(self):
        """Test ABC position suggestion functionality (API compatibility)."""

        def simple_loss(x):
            return np.sum(x**2)

        abc = ABC(
            dimensions=2,
            bounds=[(-5, 5), (-5, 5)],
            loss_function=simple_loss,
            population_size=6,
            employed_ratio=0.5,  # 3 employed, 3 onlookers
            random_seed=42,
        )

        # Test employed bee suggestion (first 3 particles)
        employed_particle_id = abc.swarm_state.particles[0].id
        employed_suggestion = abc.suggest_next_position(employed_particle_id)

        assert employed_suggestion is not None
        assert len(employed_suggestion) == 2
        assert isinstance(employed_suggestion, list)

        # Should be within bounds
        assert -5 <= employed_suggestion[0] <= 5
        assert -5 <= employed_suggestion[1] <= 5

        # Test onlooker bee suggestion (last 3 particles)
        onlooker_particle_id = abc.swarm_state.particles[4].id
        onlooker_suggestion = abc.suggest_next_position(onlooker_particle_id)

        assert onlooker_suggestion is not None
        assert len(onlooker_suggestion) == 2
        assert isinstance(onlooker_suggestion, list)

        # Should be within bounds
        assert -5 <= onlooker_suggestion[0] <= 5
        assert -5 <= onlooker_suggestion[1] <= 5

        # Test non-existent particle
        assert abc.suggest_next_position("nonexistent") is None

    def test_abc_edge_case_single_participant(self):
        """Test ABC with single participant (edge case that caused division by zero)."""

        def simple_loss(x):
            return np.sum(x**2)

        # This used to cause division by zero: 1 participant with 0.6 ratio = 0 employed bees
        abc = ABC(
            dimensions=2,
            bounds=[(-5, 5), (-5, 5)],
            loss_function=simple_loss,
            population_size=1,
            employed_ratio=0.6,  # Would give 0 employed bees without the fix
            random_seed=42,
        )

        # Should have at least 1 employed bee
        assert abc.employed_count >= 1
        assert abc.onlooker_count >= 0
        assert abc.employed_count + abc.onlooker_count == 1

        # Should be able to perform a step without errors
        abc.step()
        assert abc.swarm_state.iteration == 1

    def test_abc_minimum_employed_bees(self):
        """Test that ABC always has at least 1 employed bee."""

        def simple_loss(x):
            return np.sum(x**2)

        # Test various small population sizes
        for pop_size in [1, 2, 3]:
            for ratio in [0.1, 0.3, 0.5]:
                abc = ABC(
                    dimensions=2,
                    bounds=[(-5, 5), (-5, 5)],
                    loss_function=simple_loss,
                    population_size=pop_size,
                    employed_ratio=ratio,
                    random_seed=42,
                )

                # Should always have at least 1 employed bee
                assert abc.employed_count >= 1
                assert abc.employed_count <= pop_size
                assert abc.onlooker_count >= 0
                assert abc.employed_count + abc.onlooker_count == pop_size

                # Should work without errors
                abc.step()
                assert abc.swarm_state.iteration == 1


# Additional fixture for ABC testing
@pytest.fixture
def test_abc(simple_loss_function):
    """ABC instance for testing."""
    return ABC(
        dimensions=2,
        bounds=[(-5, 5), (-5, 5)],
        loss_function=simple_loss_function,
        population_size=10,
        limit=5,
        employed_ratio=0.5,
        random_seed=42,
    )
