"""
Loss functions for swarm optimization with human interaction.

j́his module provides various optimization landscapes designed for experiential
learning about swarm intelligence, local minima, and collective optimization.
Each function is designed to create meaningful human experiences while
maintaining mathematical rigor.
"""

import numpy as np
import colorsys
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
from loguru import logger


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
    def evaluate(self, position: np.ndarray | list[float]) -> float:
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

    def get_fitness_color(
        self, fitness: float, velocity_magnitude: Optional[float] = None
    ) -> str:
        """
        Convert fitness value to color. Higher velocity increases saturation.
        """
        min_expected = self._metadata.global_minimum_value
        max_expected = min_expected + 50

        normalized_fitness = max(
            0, min(1, (fitness - min_expected) / (max_expected - min_expected))
        )

        if normalized_fitness < 0.5:
            r = int(255 * normalized_fitness * 2)
            g = 255
            b = 0
        else:
            r = 255
            g = int(255 * (1 - (normalized_fitness - 0.5) * 2))
            b = 0

        if velocity_magnitude is not None:
            max_expected_velocity = 5.0
            norm_velocity = min(1.0, velocity_magnitude / max_expected_velocity)

            hue, lightness, _ = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)

            new_saturation = 0.4 + (norm_velocity * 0.6)

            r_new, g_new, b_new = colorsys.hls_to_rgb(hue, lightness, new_saturation)
            r, g, b = int(r_new * 255), int(g_new * 255), int(b_new * 255)

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


class QuadraticLandscape(OptimizationLandscape):
    """
    Simple quadratic function - a smooth bowl perfect for testing convergence.
    """

    def __init__(self, grid_size: int, dimensions: int = 2):
        self.dimensions = dimensions
        self.grid_size = grid_size
        # Center is at the middle of the grid
        self.center = np.array([grid_size / 2] * dimensions)
        super().__init__()

    def _create_metadata(self) -> LandscapeMetadata:
        # Bounds now match the grid size
        bounds = [(0.0, float(self.grid_size))] * self.dimensions

        global_min = [self.grid_size / 2] * self.dimensions  # Center of grid
        logger.info(f"Using bounds: {bounds}, global_min: {global_min}")

        return LandscapeMetadata(
            name="Quadratic Bowl",
            description="A simple, smooth, convex function with a single global minimum at the center.",
            landscape_type=LandscapeType.MATHEMATICAL,
            dimensions=self.dimensions,
            recommended_bounds=bounds,
            global_minimum=global_min,
            global_minimum_value=0.0,
            local_minima_count=0,
            difficulty_level=1,
            story_context="This is a simple test landscape. Find the bottom of the bowl at the center!",
            axis_labels=["X", "Y"] if self.dimensions == 2 else None,
        )

    def evaluate(self, position: np.ndarray | list[float]) -> float:
        """Evaluates f(x) = Σ((x_i - center_i)^2)"""
        if isinstance(position, list):
            position = np.array(position)
        # Shift so minimum is at center
        shifted = position - self.center
        result = np.sum(shifted**2)
        logger.debug(
            f"using position: {position}, shifted: {shifted}, result: {result}"
        )
        return result


class RastriginLandscape(OptimizationLandscape):
    """
    Rastrigin function - classic multimodal optimization problem.

    The Rastrigin function has many local minima but one global minimum,
    making it perfect for demonstrating the exploration vs exploitation dilemma.
    """

    def __init__(self, grid_size: int, A: float = 10.0, dimensions: int = 2):
        """
        Initialize Rastrigin landscape.

        Args:
            grid_size: Size of the grid (bounds will be [0, grid_size])
            A: Amplitude parameter (higher = more local minima)
            dimensions: Number of dimensions
        """
        self.A = A
        self.dimensions = dimensions
        self.grid_size = grid_size
        # Center is at the middle of the grid
        self.center = np.array([grid_size / 2] * dimensions)
        # Scale factor to map grid coordinates to standard Rastrigin range
        # Standard Rastrigin uses [-5.12, 5.12], so we scale to that range
        self.scale = 10.24 / grid_size  # 10.24 = 5.12 * 2
        super().__init__()

    def _create_metadata(self) -> LandscapeMetadata:
        # Bounds now match the grid size
        bounds = [(0.0, float(self.grid_size))] * self.dimensions
        global_min = [self.grid_size / 2] * self.dimensions  # Center of grid
        return LandscapeMetadata(
            name="Rastrigin Function",
            description="A highly multimodal function with many local minima and one global minimum at center",
            landscape_type=LandscapeType.MATHEMATICAL,
            dimensions=self.dimensions,
            recommended_bounds=bounds,
            global_minimum=global_min,
            global_minimum_value=0.0,
            local_minima_count=int(5**self.dimensions),  # Approximate
            difficulty_level=4,
            story_context="Navigate through a landscape of mathematical peaks and valleys to find the single point of perfect harmony at the center.",
            axis_labels=["X Coordinate", "Y Coordinate"]
            if self.dimensions == 2
            else None,
        )

    def evaluate(self, position: np.ndarray | list[float]) -> float:
        """
        Evaluate Rastrigin function.

        f(x) = A*n + Σ[(x_i - center_i)² - A*cos(2π*(x_i - center_i)*scale)]
        """
        if isinstance(position, list):
            position = np.array(position)
        # Shift to center and scale to standard Rastrigin range
        shifted = (position - self.center) * self.scale
        n = len(position)
        return self.A * n + np.sum(shifted**2 - self.A * np.cos(2 * np.pi * shifted))

    def describe_position(self, position: np.ndarray) -> str:
        """Describe position in terms of the mathematical landscape."""
        fitness = self.evaluate(position)
        distance_from_center = np.linalg.norm(position - self.center)

        if fitness < 1:
            return f"Perfect! You've found the global minimum (fitness: {fitness:.2f})"
        elif fitness < 10:
            return f"Excellent! Very close to optimal (fitness: {fitness:.2f})"
        elif distance_from_center < 1:
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

    def __init__(self, grid_size: int):
        """Initialize ecological landscape."""
        self.grid_size = grid_size
        # Scale factor to map grid to [0, 10] range for the function
        self.scale = 10.0 / grid_size
        # Optimal position in grid coordinates (scaled from [6.5, 7.0])
        self.optimal_development = 6.5 / 10.0 * grid_size  # ~65% across
        self.optimal_regulation = 7.0 / 10.0 * grid_size  # ~70% across
        super().__init__()

    def _create_metadata(self) -> LandscapeMetadata:
        return LandscapeMetadata(
            name="Sustainable Development Challenge",
            description="Balance economic growth with environmental protection to maximize long-term societal wellbeing",
            landscape_type=LandscapeType.ECOLOGICAL,
            dimensions=2,
            recommended_bounds=[(0, float(self.grid_size)), (0, float(self.grid_size))],
            global_minimum=[self.optimal_development, self.optimal_regulation],
            global_minimum_value=5.0,
            local_minima_count=3,
            difficulty_level=3,
            story_context="""You are leaders making policy decisions for your civilization.
            The X-axis represents Economic Development intensity.
            The Y-axis represents Environmental Regulation strength.
            Your goal is to maximize long-term societal wellbeing by finding the optimal balance.""",
            axis_labels=["Economic Development", "Environmental Regulation"],
        )

    def evaluate(self, position: np.ndarray | list[float]) -> float:
        """
        Evaluate ecological landscape.

        This function creates:
        - Local minimum at high development, low regulation (short-term thinking)
        - Local minimum at low development, high regulation (economic stagnation)
        - Global minimum at moderate-high development with strong regulation (sustainability)
        """
        if isinstance(position, list):
            position = np.array(position)

        # Scale grid coordinates to [0, 10] range for function evaluation
        development = position[0] * self.scale
        regulation = position[1] * self.scale

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

    def get_fitness_color(
        self, fitness: float, velocity_magnitude: Optional[float] = None
    ) -> str:
        """Custom coloring for ecological theme, with velocity affecting saturation."""
        # First, determine the base hex color based on fitness
        if fitness < 8:
            hex_color = "#00AA00"  # Bright green
        elif fitness < 15:
            hex_color = "#88AA00"  # Yellow-green
        elif fitness < 25:
            hex_color = "#AAAA00"  # Yellow
        elif fitness < 35:
            hex_color = "#AA5500"  # Orange
        else:
            hex_color = "#AA0000"  # Red

        # If velocity is provided, adjust the saturation of the chosen color
        if velocity_magnitude is not None:
            max_expected_velocity = 5.0
            norm_velocity = min(1.0, velocity_magnitude / max_expected_velocity)

            # Convert hex to RGB floats (0-1)
            hex_color = hex_color.lstrip("#")
            r, g, b = tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

            # Convert RGB to HLS
            hue, lightness, _ = colorsys.rgb_to_hls(r, g, b)
            new_saturation = 0.4 + (norm_velocity * 0.6)

            # Convert back to RGB and then to a new hex string
            r_new, g_new, b_new = colorsys.hls_to_rgb(hue, lightness, new_saturation)
            return (
                f"#{int(r_new * 255):02x}{int(g_new * 255):02x}{int(b_new * 255):02x}"
            )

        return hex_color


# Factory function for easy landscape creation
def create_landscape(
    landscape_name: str, grid_size: int, **kwargs
) -> OptimizationLandscape:
    """
    Factory function to create optimization landscapes.

    Args:
        landscape_name: Name of landscape to create
        grid_size: Size of the grid (default: 25)
        **kwargs: Additional parameters for landscape constructor

    Returns:
        OptimizationLandscape instance
    """
    landscapes = {
        "rastrigin": RastriginLandscape,
        "ecological": EcologicalLandscape,
        "quadratic": QuadraticLandscape,
    }

    if landscape_name not in landscapes:
        raise ValueError(
            f"Unknown landscape: {landscape_name}. Available: {list(landscapes.keys())}"
        )

    # Add grid_size to kwargs
    kwargs["grid_size"] = grid_size
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
