"""Inverse-projection power deposition on target surface cells."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from pydart.geometry.projection import project_to_beam_planes
from pydart.geometry.spherical_mesh import create_spherical_cell_quadrature
from pydart.illumination.profiles import supergaussian_intensity
from pydart.illumination.visibility import projected_cosines_from_normals
from pydart.model.beams import Beams
from pydart.model.target import Target


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DepositionResult:
    """Smoothed cell maps and unsmoothed intercepted powers in watts."""

    per_beam: Array
    total: Array
    unsmoothed_deposited_power_per_beam: Array
    target: Target
    beams: Beams
    incident_power: Array
    simulation_index: int = 0
    l_max: int = 20
    surface_quadrature_order: int = 1
    visibility_smoothing_epsilon: float = 0.05

    def tree_flatten(self):
        children = (
            self.per_beam,
            self.total,
            self.unsmoothed_deposited_power_per_beam,
            self.target,
            self.beams,
            self.incident_power,
        )
        return children, (
            self.simulation_index,
            self.l_max,
            self.surface_quadrature_order,
            self.visibility_smoothing_epsilon,
        )

    @classmethod
    def tree_unflatten(cls, auxiliary_data, children):
        (
            simulation_index,
            l_max,
            surface_quadrature_order,
            visibility_smoothing_epsilon,
        ) = auxiliary_data
        return cls(
            *children,
            simulation_index=simulation_index,
            l_max=l_max,
            surface_quadrature_order=surface_quadrature_order,
            visibility_smoothing_epsilon=visibility_smoothing_epsilon,
        )

    def get_metrics(self):
        """Calculate JAX-compatible global and spherical-harmonic metrics."""
        from pydart.metrics.global_metrics import calculate_metrics

        return calculate_metrics(
            deposition=self.total,
            target=self.target,
            incident_power=self.incident_power,
            unsmoothed_deposited_power=jnp.sum(
                self.unsmoothed_deposited_power_per_beam
            ),
            l_max=self.l_max,
            simulation_index=self.simulation_index,
        )

    def save_deposition_data(self, output_directory) -> None:
        """Save deposition and target arrays into the simulation HDF5 file."""
        from pydart.io.results import save_deposition_result

        save_deposition_result(self, output_directory)


def calculate_deposition(
    target: Target,
    beams: Beams,
    simulation_index: int = 0,
    l_max: int = 20,
    surface_quadrature_order: int = 1,
    visibility_smoothing_epsilon: float = 0.05,
) -> DepositionResult:
    """Calculate cell-integrated power using equal-area subcell quadrature."""
    sample_points, sample_normals, sample_areas = create_spherical_cell_quadrature(
        target, surface_quadrature_order
    )
    _, local_coordinates = project_to_beam_planes(
        sample_points,
        beams,
    )
    intensities = supergaussian_intensity(local_coordinates, beams)
    weighted_intensities = intensities * sample_areas[..., None]
    smoothed_projection = projected_cosines_from_normals(
        sample_normals,
        beams,
        smoothing_epsilon=visibility_smoothing_epsilon,
    )
    exact_projection = projected_cosines_from_normals(
        sample_normals,
        beams,
        smoothing_epsilon=0.0,
    )
    per_beam = jnp.sum(
        weighted_intensities * smoothed_projection,
        axis=(-3, -2),
    )
    unsmoothed_deposited_power_per_beam = jnp.sum(
        weighted_intensities * exact_projection,
        axis=tuple(range(weighted_intensities.ndim - 1)),
    )
    return DepositionResult(
        per_beam=per_beam,
        total=jnp.sum(per_beam, axis=-1),
        unsmoothed_deposited_power_per_beam=(
            unsmoothed_deposited_power_per_beam
        ),
        target=target,
        beams=beams,
        incident_power=jnp.sum(beams.powers),
        simulation_index=simulation_index,
        l_max=l_max,
        surface_quadrature_order=surface_quadrature_order,
        visibility_smoothing_epsilon=visibility_smoothing_epsilon,
    )
