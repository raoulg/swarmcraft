"""
Landscape visualizer for optimization landscapes using Plotly.

This module provides interactive visualization tools for optimization landscapes,
perfect for debugging, analysis, and understanding. Interactive plots help
developers quickly spot issues with loss functions and understand their structure
before using them in human experiences.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Tuple, Optional, Dict, Any, Callable

from swarmcraft.core.loss_functions import OptimizationLandscape


class LandscapeVisualizer:
    """
    Interactive visualizer for optimization landscapes using Plotly.

    Provides multiple interactive visualization methods to understand landscape
    structure, identify local minima, and debug loss function implementations.
    """

    def __init__(self, theme: str = "plotly_dark"):
        """
        Initialize visualizer.

        Args:
            theme: Plotly theme ('plotly_dark', 'plotly_white', 'ggplot2', etc.)
        """
        self.theme = theme

    def plot_2d_landscape(
        self,
        landscape: OptimizationLandscape,
        bounds: Optional[List[Tuple[float, float]]] = None,
        resolution: int = 100,
        show_global_minimum: bool = True,
        contour_levels: int = 20,
        show_heatmap: bool = True,
        # New grid parameters
        show_grid: bool = False,
        grid_size: Optional[int] = None,
        show_grid_values: bool = False,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Create interactive 2D contour/heatmap plot of landscape.

        Args:
            landscape: Landscape to visualize
            bounds: Custom bounds, or None to use landscape metadata
            resolution: Grid resolution for evaluation
            show_global_minimum: Whether to mark global minimum
            contour_levels: Number of contour levels
            show_heatmap: Whether to show heatmap background
            show_grid: Whether to overlay discrete grid
            grid_size: Size of discrete grid (e.g., 25 for 25x25)
            show_grid_values: Whether to show loss values at grid centers
            save_path: Optional path to save HTML file

        Returns:
            Plotly Figure object
        """
        if landscape.metadata.dimensions != 2:
            raise ValueError("2D plotting only supports 2-dimensional landscapes")

        # Use provided bounds or landscape defaults
        if bounds is None:
            bounds = landscape.metadata.recommended_bounds

        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]

        # Create evaluation grid
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x, y)

        # Evaluate landscape
        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = landscape.evaluate(np.array([X[i, j], Y[i, j]]))

        # Create figure
        fig = go.Figure()

        # Add heatmap background if requested
        if show_heatmap:
            fig.add_trace(
                go.Heatmap(
                    x=x,
                    y=y,
                    z=Z,
                    colorscale="Viridis",
                    opacity=0.7,
                    name="Loss Landscape",
                    hovertemplate="<b>%{fullData.name}</b><br>"
                    + f"{landscape.metadata.axis_labels[0] if landscape.metadata.axis_labels else 'X'}: %{{x}}<br>"
                    + f"{landscape.metadata.axis_labels[1] if landscape.metadata.axis_labels else 'Y'}: %{{y}}<br>"
                    + "Loss: %{z:.3f}<extra></extra>",
                )
            )

        # Add contour lines
        fig.add_trace(
            go.Contour(
                x=x,
                y=y,
                z=Z,
                showscale=not show_heatmap,  # Only show colorbar if no heatmap
                contours=dict(
                    coloring="lines",
                    showlabels=True,
                    labelfont=dict(size=10, color="white"),
                ),
                line=dict(width=1),
                ncontours=contour_levels,
                name="Contours",
                hovertemplate="<b>%{fullData.name}</b><br>"
                + f"{landscape.metadata.axis_labels[0] if landscape.metadata.axis_labels else 'X'}: %{{x}}<br>"
                + f"{landscape.metadata.axis_labels[1] if landscape.metadata.axis_labels else 'Y'}: %{{y}}<br>"
                + "Loss: %{z:.3f}<extra></extra>",
            )
        )

        # Mark global minimum
        if show_global_minimum:
            gm = landscape.metadata.global_minimum
            gm_value = landscape.evaluate(np.array(gm))
            fig.add_trace(
                go.Scatter(
                    x=[gm[0]],
                    y=[gm[1]],
                    mode="markers",
                    marker=dict(
                        symbol="star",
                        size=20,
                        color="red",
                        line=dict(width=2, color="white"),
                    ),
                    name="Global Minimum",
                    hovertemplate="<b>Global Minimum</b><br>"
                    + f"{landscape.metadata.axis_labels[0] if landscape.metadata.axis_labels else 'X'}: {gm[0]:.3f}<br>"
                    + f"{landscape.metadata.axis_labels[1] if landscape.metadata.axis_labels else 'Y'}: {gm[1]:.3f}<br>"
                    + f"Loss: {gm_value:.3f}<extra></extra>",
                )
            )

        # Add discrete grid overlay if requested
        if show_grid and grid_size:
            self._add_grid_overlay(fig, bounds, grid_size, landscape, show_grid_values)

        # Update layout
        fig.update_layout(
            title=f"{landscape.metadata.name} - Interactive Landscape",
            xaxis_title=landscape.metadata.axis_labels[0]
            if landscape.metadata.axis_labels
            else "X",
            yaxis_title=landscape.metadata.axis_labels[1]
            if landscape.metadata.axis_labels
            else "Y",
            template=self.theme,
            width=800,
            height=600,
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_3d_landscape(
        self,
        landscape: OptimizationLandscape,
        bounds: Optional[List[Tuple[float, float]]] = None,
        resolution: int = 50,
        show_global_minimum: bool = True,
        opacity: float = 0.8,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Create interactive 3D surface plot of landscape.

        Args:
            landscape: Landscape to visualize
            bounds: Custom bounds, or None to use landscape metadata
            resolution: Grid resolution (lower for 3D to avoid performance issues)
            show_global_minimum: Whether to mark global minimum
            opacity: Surface opacity
            save_path: Optional path to save HTML file

        Returns:
            Plotly Figure object
        """
        if landscape.metadata.dimensions != 2:
            raise ValueError("3D plotting only supports 2-dimensional landscapes")

        # Use provided bounds or landscape defaults
        if bounds is None:
            bounds = landscape.metadata.recommended_bounds

        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]

        # Create evaluation grid
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x, y)

        # Evaluate landscape
        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = landscape.evaluate(np.array([X[i, j], Y[i, j]]))

        # Create 3D surface
        fig = go.Figure()

        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=Z,
                colorscale="Viridis",
                opacity=opacity,
                name="Loss Surface",
                hovertemplate="<b>Loss Surface</b><br>"
                + f"{landscape.metadata.axis_labels[0] if landscape.metadata.axis_labels else 'X'}: %{{x}}<br>"
                + f"{landscape.metadata.axis_labels[1] if landscape.metadata.axis_labels else 'Y'}: %{{y}}<br>"
                + "Loss: %{z:.3f}<extra></extra>",
            )
        )

        # Mark global minimum
        if show_global_minimum:
            gm = landscape.metadata.global_minimum
            gm_value = landscape.evaluate(np.array(gm))
            fig.add_trace(
                go.Scatter3d(
                    x=[gm[0]],
                    y=[gm[1]],
                    z=[gm_value],
                    mode="markers",
                    marker=dict(
                        symbol="diamond",
                        size=10,
                        color="red",
                        line=dict(width=2, color="white"),
                    ),
                    name="Global Minimum",
                    hovertemplate="<b>Global Minimum</b><br>"
                    + f"{landscape.metadata.axis_labels[0] if landscape.metadata.axis_labels else 'X'}: {gm[0]:.3f}<br>"
                    + f"{landscape.metadata.axis_labels[1] if landscape.metadata.axis_labels else 'Y'}: {gm[1]:.3f}<br>"
                    + f"Loss: {gm_value:.3f}<extra></extra>",
                )
            )

        # Update layout
        fig.update_layout(
            title=f"{landscape.metadata.name} - 3D Interactive Surface",
            scene=dict(
                xaxis_title=landscape.metadata.axis_labels[0]
                if landscape.metadata.axis_labels
                else "X",
                yaxis_title=landscape.metadata.axis_labels[1]
                if landscape.metadata.axis_labels
                else "Y",
                zaxis_title="Loss Value",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            ),
            template=self.theme,
            width=800,
            height=600,
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_cross_sections(
        self,
        landscape: OptimizationLandscape,
        bounds: Optional[List[Tuple[float, float]]] = None,
        resolution: int = 200,
        num_sections: int = 5,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Plot interactive cross-sections of the landscape.

        Args:
            landscape: Landscape to visualize
            bounds: Custom bounds, or None to use landscape metadata
            resolution: Number of points per cross-section
            num_sections: Number of cross-sections to plot
            save_path: Optional path to save HTML file

        Returns:
            Plotly Figure object
        """
        if landscape.metadata.dimensions != 2:
            raise ValueError(
                "Cross-section plotting only supports 2-dimensional landscapes"
            )

        # Use provided bounds or landscape defaults
        if bounds is None:
            bounds = landscape.metadata.recommended_bounds

        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]

        # Create subplots
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                f"Cross-sections along {landscape.metadata.axis_labels[0] if landscape.metadata.axis_labels else 'X'}-axis",
                f"Cross-sections along {landscape.metadata.axis_labels[1] if landscape.metadata.axis_labels else 'Y'}-axis",
            ),
        )

        # X cross-sections (varying x, fixed y)
        y_values = np.linspace(y_min, y_max, num_sections)
        x_range = np.linspace(x_min, x_max, resolution)

        colors = px.colors.qualitative.Set1[:num_sections]

        for i, y_val in enumerate(y_values):
            z_values = [landscape.evaluate(np.array([x, y_val])) for x in x_range]
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=z_values,
                    mode="lines",
                    name=f"Y = {y_val:.1f}",
                    line=dict(color=colors[i % len(colors)]),
                    hovertemplate=f"<b>Y = {y_val:.1f}</b><br>"
                    + f"{landscape.metadata.axis_labels[0] if landscape.metadata.axis_labels else 'X'}: %{{x}}<br>"
                    + "Loss: %{y:.3f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        # Y cross-sections (varying y, fixed x)
        x_values = np.linspace(x_min, x_max, num_sections)
        y_range = np.linspace(y_min, y_max, resolution)

        for i, x_val in enumerate(x_values):
            z_values = [landscape.evaluate(np.array([x_val, y])) for y in y_range]
            fig.add_trace(
                go.Scatter(
                    x=y_range,
                    y=z_values,
                    mode="lines",
                    name=f"X = {x_val:.1f}",
                    line=dict(color=colors[i % len(colors)]),
                    showlegend=False,  # Avoid duplicate legend entries
                    hovertemplate=f"<b>X = {x_val:.1f}</b><br>"
                    + f"{landscape.metadata.axis_labels[1] if landscape.metadata.axis_labels else 'Y'}: %{{x}}<br>"
                    + "Loss: %{y:.3f}<extra></extra>",
                ),
                row=1,
                col=2,
            )

        # Update layout
        fig.update_xaxes(
            title_text=landscape.metadata.axis_labels[0]
            if landscape.metadata.axis_labels
            else "X",
            row=1,
            col=1,
        )
        fig.update_xaxes(
            title_text=landscape.metadata.axis_labels[1]
            if landscape.metadata.axis_labels
            else "Y",
            row=1,
            col=2,
        )
        fig.update_yaxes(title_text="Loss Value", row=1, col=1)
        fig.update_yaxes(title_text="Loss Value", row=1, col=2)

        fig.update_layout(
            title=f"{landscape.metadata.name} - Cross-sections",
            template=self.theme,
            width=1200,
            height=500,
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def analyze_landscape_statistics(
        self,
        landscape: OptimizationLandscape,
        bounds: Optional[List[Tuple[float, float]]] = None,
        resolution: int = 100,
    ) -> Dict[str, Any]:
        """
        Analyze statistical properties of the landscape.

        Args:
            landscape: Landscape to analyze
            bounds: Custom bounds, or None to use landscape metadata
            resolution: Grid resolution for sampling

        Returns:
            Dictionary of statistics
        """
        if landscape.metadata.dimensions != 2:
            raise ValueError(
                "Analysis currently only supports 2-dimensional landscapes"
            )

        # Use provided bounds or landscape defaults
        if bounds is None:
            bounds = landscape.metadata.recommended_bounds

        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]

        # Sample landscape
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x, y)

        values = []
        for i in range(resolution):
            for j in range(resolution):
                values.append(landscape.evaluate(np.array([X[i, j], Y[i, j]])))

        values = np.array(values)

        # Calculate statistics
        stats = {
            "min_value": float(np.min(values)),
            "max_value": float(np.max(values)),
            "mean_value": float(np.mean(values)),
            "median_value": float(np.median(values)),
            "std_value": float(np.std(values)),
            "range": float(np.max(values) - np.min(values)),
            "global_minimum_actual": landscape.metadata.global_minimum,
            "global_minimum_expected": landscape.metadata.global_minimum_value,
            "landscape_type": landscape.metadata.landscape_type.value,
            "difficulty_level": landscape.metadata.difficulty_level,
        }

        return stats

    def plot_swarm_on_landscape(
        self,
        landscape: OptimizationLandscape,
        particle_positions: List[List[float]],
        particle_fitnesses: Optional[List[float]] = None,
        particle_ids: Optional[List[str]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
        resolution: int = 100,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Plot swarm particles overlaid on interactive landscape.

        Args:
            landscape: Landscape to visualize
            particle_positions: List of [x, y] positions
            particle_fitnesses: Optional list of fitness values for coloring
            particle_ids: Optional list of particle IDs
            bounds: Custom bounds, or None to use landscape metadata
            resolution: Grid resolution for background landscape
            save_path: Optional path to save HTML file

        Returns:
            Plotly Figure object
        """
        if landscape.metadata.dimensions != 2:
            raise ValueError("Swarm plotting only supports 2-dimensional landscapes")

        # Create landscape background
        fig = self.plot_2d_landscape(
            landscape, bounds, resolution, show_global_minimum=True, show_heatmap=True
        )

        # Extract particle coordinates
        x_coords = [pos[0] for pos in particle_positions]
        y_coords = [pos[1] for pos in particle_positions]

        # Create hover text
        if particle_ids:
            hover_text = [
                f"<b>Particle {pid}</b><br>"
                + f"{landscape.metadata.axis_labels[0] if landscape.metadata.axis_labels else 'X'}: {x:.3f}<br>"
                + f"{landscape.metadata.axis_labels[1] if landscape.metadata.axis_labels else 'Y'}: {y:.3f}<br>"
                + (f"Fitness: {fit:.3f}" if particle_fitnesses else "")
                for x, y, pid, fit in zip(
                    x_coords,
                    y_coords,
                    particle_ids,
                    particle_fitnesses or [None] * len(x_coords),
                )
            ]
        else:
            hover_text = [
                "<b>Particle</b><br>"
                + f"{landscape.metadata.axis_labels[0] if landscape.metadata.axis_labels else 'X'}: {x:.3f}<br>"
                + f"{landscape.metadata.axis_labels[1] if landscape.metadata.axis_labels else 'Y'}: {y:.3f}<br>"
                + (f"Fitness: {fit:.3f}" if particle_fitnesses else "")
                for x, y, fit in zip(
                    x_coords, y_coords, particle_fitnesses or [None] * len(x_coords)
                )
            ]

        # Add particles
        if particle_fitnesses:
            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="markers",
                    marker=dict(
                        size=12,
                        color=particle_fitnesses,
                        colorscale="Plasma",
                        colorbar=dict(title="Particle Fitness"),
                        line=dict(width=2, color="white"),
                    ),
                    name="Swarm Particles",
                    hovertemplate="%{text}<extra></extra>",
                    text=hover_text,
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="markers",
                    marker=dict(
                        size=12, color="white", line=dict(width=2, color="black")
                    ),
                    name="Swarm Particles",
                    hovertemplate="%{text}<extra></extra>",
                    text=hover_text,
                )
            )

        fig.update_layout(title=f"{landscape.metadata.name} - Swarm Positions")

        if save_path:
            fig.write_html(save_path)

        return fig

    def create_dashboard(
        self,
        landscape: OptimizationLandscape,
        bounds: Optional[List[Tuple[float, float]]] = None,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Create comprehensive dashboard with multiple views.

        Args:
            landscape: Landscape to analyze
            bounds: Custom bounds, or None to use landscape metadata
            save_path: Optional path to save HTML file

        Returns:
            Plotly Figure with subplots
        """
        if landscape.metadata.dimensions != 2:
            raise ValueError("Dashboard only supports 2-dimensional landscapes")

        # Create 2x2 subplot layout
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Contour Plot",
                "3D Surface",
                "Statistics",
                "Cross-sections",
            ),
            specs=[
                [{"type": "xy"}, {"type": "scene"}],
                [{"type": "xy"}, {"type": "xy"}],
            ],
        )

        # Use provided bounds or landscape defaults
        if bounds is None:
            bounds = landscape.metadata.recommended_bounds

        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]

        # Create evaluation grid
        resolution = 50  # Lower resolution for dashboard
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x, y)

        # Evaluate landscape
        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = landscape.evaluate(np.array([X[i, j], Y[i, j]]))

        # 1. Contour plot
        fig.add_trace(
            go.Contour(x=x, y=y, z=Z, colorscale="Viridis", showscale=False),
            row=1,
            col=1,
        )

        # 2. 3D Surface
        fig.add_trace(
            go.Surface(x=x, y=y, z=Z, colorscale="Viridis", showscale=False),
            row=1,
            col=2,
        )

        # 3. Statistics (text summary)
        stats = self.analyze_landscape_statistics(landscape, bounds)
        stats_text = f"""
        <b>{landscape.metadata.name}</b><br>
        Type: {stats["landscape_type"]}<br>
        Difficulty: {stats["difficulty_level"]}/5<br>
        <br>
        Min Value: {stats["min_value"]:.3f}<br>
        Max Value: {stats["max_value"]:.3f}<br>
        Mean: {stats["mean_value"]:.3f}<br>
        Std Dev: {stats["std_value"]:.3f}<br>
        Range: {stats["range"]:.3f}<br>
        <br>
        Global Min Expected: {stats["global_minimum_expected"]:.3f}<br>
        Global Min Position: ({stats["global_minimum_actual"][0]:.2f}, {stats["global_minimum_actual"][1]:.2f})
        """

        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode="text",
                text=[stats_text],
                textposition="middle center",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # 4. Cross-section
        x_mid = (x_min + x_max) / 2
        z_values = [landscape.evaluate(np.array([xi, x_mid])) for xi in x]
        fig.add_trace(
            go.Scatter(x=x, y=z_values, mode="lines", name="Mid Cross-section"),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(
            title=f"{landscape.metadata.name} - Analysis Dashboard",
            template=self.theme,
            width=1200,
            height=800,
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_discrete_landscape(
        self,
        landscape: OptimizationLandscape,
        bounds: Optional[List[Tuple[float, float]]] = None,
        grid_size: int = 25,
        show_continuous: bool = True,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Plot landscape showing both continuous and discrete versions.

        Args:
            landscape: Landscape to visualize
            bounds: Custom bounds, or None to use landscape metadata
            grid_size: Size of discrete grid
            show_continuous: Whether to show continuous background
            save_path: Optional path to save HTML file

        Returns:
            Plotly Figure object
        """
        # Create base continuous plot if requested
        if show_continuous:
            fig = self.plot_2d_landscape(
                landscape,
                bounds,
                resolution=100,
                show_heatmap=True,
                contour_levels=15,
                show_grid=True,
                grid_size=grid_size,
                show_grid_values=True,
            )
        else:
            fig = go.Figure()

            # Add grid overlay without continuous background
            if bounds is None:
                bounds = landscape.metadata.recommended_bounds

            self._add_grid_overlay(fig, bounds, grid_size, landscape, show_values=True)

            # Set up axes
            x_min, x_max = bounds[0]
            y_min, y_max = bounds[1]
            fig.update_layout(
                xaxis=dict(range=[x_min, x_max]), yaxis=dict(range=[y_min, y_max])
            )

        fig.update_layout(
            title=f"{landscape.metadata.name} - Discrete Grid ({grid_size}x{grid_size})",
            xaxis_title=landscape.metadata.axis_labels[0]
            if landscape.metadata.axis_labels
            else "X",
            yaxis_title=landscape.metadata.axis_labels[1]
            if landscape.metadata.axis_labels
            else "Y",
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def _add_grid_overlay(
        self,
        fig: go.Figure,
        bounds: List[Tuple[float, float]],
        grid_size: int,
        landscape: OptimizationLandscape,
        show_values: bool = False,
    ) -> None:
        """
        Add discrete grid overlay to existing figure.

        Args:
            fig: Plotly figure to modify
            bounds: Coordinate bounds
            grid_size: Size of grid (e.g., 25 for 25x25)
            landscape: Landscape for evaluating grid center values
            show_values: Whether to show loss values at grid centers
        """
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]

        # Create grid lines
        x_grid = np.linspace(x_min, x_max, grid_size + 1)
        y_grid = np.linspace(y_min, y_max, grid_size + 1)

        # Add vertical grid lines
        for x in x_grid:
            fig.add_trace(
                go.Scatter(
                    x=[x, x],
                    y=[y_min, y_max],
                    mode="lines",
                    line=dict(color="white", width=1, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Add horizontal grid lines
        for y in y_grid:
            fig.add_trace(
                go.Scatter(
                    x=[x_min, x_max],
                    y=[y, y],
                    mode="lines",
                    line=dict(color="white", width=1, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Add grid center values if requested
        if show_values:
            # Calculate grid cell centers
            x_centers = (x_grid[:-1] + x_grid[1:]) / 2
            y_centers = (y_grid[:-1] + y_grid[1:]) / 2

            # Evaluate landscape at grid centers
            grid_x, grid_y = np.meshgrid(x_centers, y_centers)

            x_coords = []
            y_coords = []
            values = []
            hover_texts = []

            for i in range(grid_size):
                for j in range(grid_size):
                    x_coord = grid_x[i, j]
                    y_coord = grid_y[i, j]
                    value = landscape.evaluate(np.array([x_coord, y_coord]))

                    x_coords.append(x_coord)
                    y_coords.append(y_coord)
                    values.append(value)
                    hover_texts.append(
                        f"<b>Grid Cell ({j}, {grid_size - 1 - i})</b><br>"
                        + f"Center: ({x_coord:.2f}, {y_coord:.2f})<br>"
                        + f"Loss: {value:.3f}<extra></extra>"
                    )

            # Add grid center points with values
            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="markers+text",
                    marker=dict(
                        size=8,
                        color=values,
                        colorscale="Viridis",
                        showscale=False,
                        line=dict(width=1, color="white"),
                    ),
                    text=[f"{v:.1f}" for v in values],
                    textposition="middle center",
                    textfont=dict(size=8, color="white"),
                    name="Grid Values",
                    hovertemplate="%{hovertext}",
                    hovertext=hover_texts,
                )
            )

    def create_discrete_landscape_function(
        self,
        landscape: OptimizationLandscape,
        bounds: Optional[List[Tuple[float, float]]] = None,
        grid_size: int = 25,
    ) -> Callable:
        """
        Create a discrete version of a continuous landscape function.

        This function maps continuous coordinates to discrete grid cells
        and evaluates the landscape at the grid cell centers.

        Args:
            landscape: Continuous landscape to discretize
            bounds: Coordinate bounds, or None to use landscape metadata
            grid_size: Size of discrete grid

        Returns:
            Function that takes grid coordinates (i, j) and returns loss value
        """
        if bounds is None:
            bounds = landscape.metadata.recommended_bounds

        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]

        # Pre-compute grid cell centers
        x_centers = np.linspace(x_min, x_max, grid_size, endpoint=False) + (
            x_max - x_min
        ) / (2 * grid_size)
        y_centers = np.linspace(y_min, y_max, grid_size, endpoint=False) + (
            y_max - y_min
        ) / (2 * grid_size)

        def discrete_loss_function(grid_coords: List[int]) -> float:
            """
            Evaluate loss at discrete grid coordinates.

            Args:
                grid_coords: [i, j] grid coordinates (0-indexed)

            Returns:
                Loss value at grid cell center
            """
            i, j = grid_coords

            # Clamp to valid grid range
            i = max(0, min(grid_size - 1, i))
            j = max(0, min(grid_size - 1, j))

            # Convert to continuous coordinates (cell center)
            x = x_centers[j]  # j maps to x-axis
            y = y_centers[
                grid_size - 1 - i
            ]  # i maps to y-axis, flipped for matrix indexing

            return landscape.evaluate(np.array([x, y]))

        return discrete_loss_function


# Convenience function for quick visualization
def quick_visualize(
    landscape_name: str,
    plot_type: str = "2d",
    show_grid: bool = False,
    grid_size: Optional[int] = None,
    **landscape_kwargs,
) -> go.Figure:
    """
    Quickly visualize a landscape by name.

    Args:
        landscape_name: Name of landscape to create and visualize
        plot_type: Type of plot ('2d', '3d', 'cross', 'dashboard', 'discrete')
        show_grid: Whether to show discrete grid overlay
        grid_size: Size of discrete grid (e.g., 25 for 25x25)
        **landscape_kwargs: Arguments for landscape creation

    Returns:
        Plotly Figure object
    """
    from swarmcraft.core.loss_functions import create_landscape

    landscape = create_landscape(landscape_name, **landscape_kwargs)
    visualizer = LandscapeVisualizer()

    if landscape.metadata.dimensions != 2:
        raise ValueError(
            f"Quick visualization only supports 2D landscapes. "
            f"Landscape '{landscape_name}' has {landscape.metadata.dimensions} dimensions."
        )

    if plot_type == "2d":
        return visualizer.plot_2d_landscape(
            landscape, show_grid=show_grid, grid_size=grid_size
        )
    elif plot_type == "3d":
        return visualizer.plot_3d_landscape(landscape)
    elif plot_type == "cross":
        return visualizer.plot_cross_sections(landscape)
    elif plot_type == "dashboard":
        return visualizer.create_dashboard(landscape)
    elif plot_type == "discrete":
        return visualizer.plot_discrete_landscape(landscape, grid_size=grid_size or 25)
    else:
        raise ValueError(
            f"Unknown plot type: {plot_type}. Use '2d', '3d', 'cross', 'dashboard', or 'discrete'."
        )
