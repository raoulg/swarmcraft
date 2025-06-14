"""
Loss functions for swarm optimization with human interaction.

This module provides various optimization landscapes designed for experiential
learning about swarm intelligence, local minima, and collective optimization.
Each function is designed to create meaningful human experiences while
maintaining mathematical rigor.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class LandscapeType(Enum):
    """Types of optimization landscapes."""

    MATHEMATICAL = "mathematical"  # Pure mathematical functions
    ECOLOGICAL = "ecological"  # Environmental/sustainability themes
    SOCIAL = "social"  # Social coordination problems
    ECONOMIC = "economic"  # Resource allocation themes
    PSYCHOLOGICAL = "psychological"  # Personal growth, meaning, human-AI collaboration


class LandscapeMetadata(BaseModel):
    """Metadata describing an optimization landscape."""

    name: str
    description: str
    landscape_type: LandscapeType
    dimensions: int
    recommended_bounds: List[Tuple[float, float]]
    global_minimum: List[float]
    global_minimum_value: float
    local_minima_count: int
    difficulty_level: int = Field(ge=1, le=5, description="1=easy, 5=very hard")
    story_context: Optional[str] = None
    axis_labels: Optional[List[str]] = None


class OptimizationLandscape(ABC):
    """
    Abstract base class for optimization landscapes.

    This provides a common interface for different loss functions
    while allowing rich metadata for human interaction and storytelling.
    """

    def __init__(self):
        self._metadata = self._create_metadata()

    @abstractmethod
    def _create_metadata(self) -> LandscapeMetadata:
        """Create metadata describing this landscape."""
        pass

    @abstractmethod
    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluate the loss function at given position.

        Args:
            position: Position to evaluate (numpy array)

        Returns:
            Loss value (lower is better)
        """
        pass

    @property
    def metadata(self) -> LandscapeMetadata:
        """Get landscape metadata."""
        return self._metadata

    def get_fitness_color(self, fitness: float) -> str:
        """
        Convert fitness value to color for human feedback.

        Returns hex color string for mobile display.
        """
        # Normalize fitness to [0, 1] range for coloring
        # This is a default implementation - subclasses can override
        min_expected = self._metadata.global_minimum_value
        max_expected = min_expected + 50  # Rough estimate

        normalized = max(
            0, min(1, (fitness - min_expected) / (max_expected - min_expected))
        )

        # Green (good) to Red (bad) gradient
        if normalized < 0.5:
            # Green to Yellow
            r = int(255 * normalized * 2)
            g = 255
            b = 0
        else:
            # Yellow to Red
            r = 255
            g = int(255 * (1 - (normalized - 0.5) * 2))
            b = 0

        return f"#{r:02x}{g:02x}{b:02x}"

    def get_fitness_audio_frequency(self, fitness: float) -> float:
        """
        Convert fitness to audio frequency for soundscape.

        Returns frequency in Hz (higher fitness = higher frequency = more dissonant)
        """
        min_expected = self._metadata.global_minimum_value
        max_expected = min_expected + 50

        normalized = max(
            0, min(1, (fitness - min_expected) / (max_expected - min_expected))
        )

        # Map to frequency range: 200 Hz (good) to 800 Hz (bad)
        return 200 + normalized * 600

    def describe_position(self, position: np.ndarray) -> str:
        """
        Provide human-readable description of position quality.

        This can be overridden by subclasses to provide domain-specific descriptions.
        """
        fitness = self.evaluate(position)

        if fitness <= self._metadata.global_minimum_value + 1:
            return "Excellent! You're at the global optimum."
        elif fitness <= self._metadata.global_minimum_value + 5:
            return "Very good! You're near the optimal solution."
        elif fitness <= self._metadata.global_minimum_value + 15:
            return "Good progress, but there's room for improvement."
        elif fitness <= self._metadata.global_minimum_value + 30:
            return "You're in a decent area, but far from optimal."
        else:
            return "This area has high cost - consider exploring elsewhere."


class RastriginLandscape(OptimizationLandscape):
    """
    Rastrigin function - classic multimodal optimization problem.

    The Rastrigin function has many local minima but one global minimum,
    making it perfect for demonstrating the exploration vs exploitation dilemma.
    """

    def __init__(self, A: float = 10.0, dimensions: int = 2):
        """
        Initialize Rastrigin landscape.

        Args:
            A: Amplitude parameter (higher = more local minima)
            dimensions: Number of dimensions
        """
        self.A = A
        self.dimensions = dimensions
        super().__init__()

    def _create_metadata(self) -> LandscapeMetadata:
        bounds = [(-5.12, 5.12)] * self.dimensions
        return LandscapeMetadata(
            name="Rastrigin Function",
            description="A highly multimodal function with many local minima and one global minimum at origin",
            landscape_type=LandscapeType.MATHEMATICAL,
            dimensions=self.dimensions,
            recommended_bounds=bounds,
            global_minimum=[0.0] * self.dimensions,
            global_minimum_value=0.0,
            local_minima_count=int(5**self.dimensions),  # Approximate
            difficulty_level=4,
            story_context="Navigate through a landscape of mathematical peaks and valleys to find the single point of perfect harmony.",
            axis_labels=["X Coordinate", "Y Coordinate"]
            if self.dimensions == 2
            else None,
        )

    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluate Rastrigin function.

        f(x) = A*n + Σ[x_i² - A*cos(2π*x_i)]
        """
        n = len(position)
        return self.A * n + np.sum(position**2 - self.A * np.cos(2 * np.pi * position))

    def describe_position(self, position: np.ndarray) -> str:
        """Describe position in terms of the mathematical landscape."""
        fitness = self.evaluate(position)
        distance_from_origin = np.linalg.norm(position)

        if fitness < 1:
            return f"Perfect! You've found the global minimum (fitness: {fitness:.2f})"
        elif fitness < 10:
            return f"Excellent! Very close to optimal (fitness: {fitness:.2f})"
        elif distance_from_origin < 1:
            return (
                f"Good! Near the center but in a local valley (fitness: {fitness:.2f})"
            )
        elif fitness < 50:
            return f"Decent position, but you might be stuck in a local minimum (fitness: {fitness:.2f})"
        else:
            return f"High cost area - try exploring toward the center (fitness: {fitness:.2f})"


class EcologicalLandscape(OptimizationLandscape):
    """
    Ecological sustainability optimization landscape.

    Represents the challenge of balancing economic development with
    environmental protection - a perfect example of the Moloch problem
    where local incentives lead to globally suboptimal outcomes.
    """

    def __init__(self):
        """Initialize ecological landscape."""
        super().__init__()

    def _create_metadata(self) -> LandscapeMetadata:
        return LandscapeMetadata(
            name="Sustainable Development Challenge",
            description="Balance economic growth with environmental protection to maximize long-term societal wellbeing",
            landscape_type=LandscapeType.ECOLOGICAL,
            dimensions=2,
            recommended_bounds=[(0, 10), (0, 10)],
            global_minimum=[6.5, 7.0],  # Moderate development, strong regulation
            global_minimum_value=5.0,
            local_minima_count=3,
            difficulty_level=3,
            story_context="""You are leaders making policy decisions for your civilization. 
            The X-axis represents Economic Development intensity.
            The Y-axis represents Environmental Regulation strength.
            Your goal is to maximize long-term societal wellbeing by finding the optimal balance.""",
            axis_labels=["Economic Development", "Environmental Regulation"],
        )

    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluate ecological landscape.

        This function creates:
        - Local minimum at high development, low regulation (short-term thinking)
        - Local minimum at low development, high regulation (economic stagnation)
        - Global minimum at moderate-high development with strong regulation (sustainability)
        """
        development, regulation = position[0], position[1]

        # Economic benefit (higher development is better, but with diminishing returns)
        economic_benefit = 8 * np.log(development + 1) - 0.05 * development**2

        # Environmental cost (pollution increases with development, decreases with regulation)
        pollution_cost = (development**2) / (regulation + 1)

        # Regulation cost (strong regulation has implementation costs)
        regulation_cost = 0.3 * regulation**1.5

        # Social instability (extreme positions create instability)
        if development < 2 and regulation > 8:  # Over-regulation trap
            instability_cost = 15
        elif development > 8 and regulation < 3:  # Pollution trap
            instability_cost = 20
        else:
            instability_cost = 0

        # Innovation bonus (moderate development + strong regulation drives innovation)
        if 4 <= development <= 8 and regulation >= 6:
            innovation_bonus = -8
        else:
            innovation_bonus = 0

        # Total cost (minimize for better outcomes)
        total_cost = (
            pollution_cost
            + regulation_cost
            + instability_cost
            - economic_benefit
            + innovation_bonus
            + 25
        )  # Offset to keep positive

        return max(0, total_cost)  # Ensure non-negative

    def describe_position(self, position: np.ndarray) -> str:
        """Describe position in terms of policy outcomes."""
        development, regulation = position[0], position[1]
        fitness = self.evaluate(position)

        if fitness < 8:
            return f"Sustainable success! Your policies create prosperity while protecting the environment. (Cost: {fitness:.1f})"
        elif development > 8 and regulation < 3:
            return f"Pollution trap! High growth but environmental collapse looms. (Cost: {fitness:.1f})"
        elif development < 2 and regulation > 8:
            return f"Stagnation trap! Clean but economically struggling society. (Cost: {fitness:.1f})"
        elif fitness < 20:
            return (
                f"Reasonable balance, but optimization possible. (Cost: {fitness:.1f})"
            )
        else:
            return f"Challenging situation - major policy changes needed. (Cost: {fitness:.1f})"

    def get_fitness_color(self, fitness: float) -> str:
        """Custom coloring for ecological theme."""
        # Green for sustainable, brown/red for unsustainable
        if fitness < 8:
            return "#00AA00"  # Bright green - sustainable
        elif fitness < 15:
            return "#88AA00"  # Yellow-green - decent
        elif fitness < 25:
            return "#AAAA00"  # Yellow - concerning
        elif fitness < 35:
            return "#AA5500"  # Orange - problematic
        else:
            return "#AA0000"  # Red - crisis


# Factory function for easy landscape creation
def create_landscape(landscape_name: str, **kwargs) -> OptimizationLandscape:
    """
    Factory function to create optimization landscapes.

    Args:
        landscape_name: Name of landscape to create
        **kwargs: Parameters for landscape constructor

    Returns:
        OptimizationLandscape instance
    """
    landscapes = {
        "rastrigin": RastriginLandscape,
        "ecological": EcologicalLandscape,
    }

    if landscape_name not in landscapes:
        raise ValueError(
            f"Unknown landscape: {landscape_name}. Available: {list(landscapes.keys())}"
        )

    return landscapes[landscape_name](**kwargs)


# Utility functions for landscape analysis
def analyze_landscape(
    landscape: OptimizationLandscape,
    bounds: List[Tuple[float, float]],
    resolution: int = 50,
) -> Dict[str, Any]:
    """
    Analyze a landscape to understand its structure.

    This is useful for debugging and understanding landscape properties.
    """
    if landscape.metadata.dimensions != 2:
        raise ValueError("Analysis currently only supports 2D landscapes")

    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = landscape.evaluate(np.array([X[i, j], Y[i, j]]))

    return {
        "X": X,
        "Y": Y,
        "Z": Z,
        "min_value": float(np.min(Z)),
        "max_value": float(np.max(Z)),
        "mean_value": float(np.mean(Z)),
        "std_value": float(np.std(Z)),
    }
