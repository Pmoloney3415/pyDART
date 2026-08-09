"""Assembly of the solid-sphere illumination forward model."""

from __future__ import annotations

from pydart.illumination.deposition import DepositionResult, calculate_deposition
from pydart.simulation.simulation import Simulation


def simulate_illumination(simulation: Simulation) -> DepositionResult:
    """Run inverse-projection deposition for an initialized simulation."""
    return calculate_deposition(
        simulation.target,
        simulation.beams,
        simulation_index=simulation.simulation_index,
        l_max=simulation.l_max,
        surface_quadrature_order=simulation.surface_quadrature_order,
        visibility_smoothing_epsilon=(simulation.visibility_smoothing_epsilon),
    )
