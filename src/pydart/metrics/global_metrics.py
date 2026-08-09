"""Global and spherical-harmonic illumination metrics."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from pydart.metrics.spherical_harmonics import (
    spherical_harmonic_coefficients,
)
from pydart.model.target import Target


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MetricsResult:
    """JAX-compatible diagnostics for a surface deposition map."""

    harmonic_coefficients: Array
    power_by_l: Array
    normalized_power_by_l: Array
    ell: Array
    deposited_power: Array
    incident_power: Array
    deposited_fraction: Array
    mean_power_density: Array
    rms_nonuniformity: Array
    simulation_index: int = 0
    l_max: int = 20

    def tree_flatten(self):
        children = (
            self.harmonic_coefficients,
            self.power_by_l,
            self.normalized_power_by_l,
            self.ell,
            self.deposited_power,
            self.incident_power,
            self.deposited_fraction,
            self.mean_power_density,
            self.rms_nonuniformity,
        )
        return children, (self.simulation_index, self.l_max)

    @classmethod
    def tree_unflatten(cls, auxiliary_data, children):
        simulation_index, l_max = auxiliary_data
        return cls(
            *children,
            simulation_index=simulation_index,
            l_max=l_max,
        )

    def save_to_directory(self, output_directory) -> None:
        """Save harmonic arrays to HDF5 and scalar metrics to JSON."""
        from pydart.io.results import save_metrics_result

        save_metrics_result(self, output_directory)


@partial(jax.jit, static_argnames=("l_max", "simulation_index"))
def calculate_metrics(
    deposition: Array,
    target: Target,
    incident_power: Array,
    l_max: int = 20,
    simulation_index: int = 0,
) -> MetricsResult:
    """Calculate power-density harmonics and global illumination metrics."""
    deposition = deposition.astype(jnp.float64)
    cell_areas = target.cell_areas.astype(jnp.float64)
    radius = target.radius.astype(jnp.float64)
    spherical_coordinates = target.spherical_coordinates.astype(jnp.float64)
    incident_power = incident_power.astype(jnp.float64)

    power_density = deposition / cell_areas
    solid_angles = cell_areas / radius**2
    coefficients = spherical_harmonic_coefficients(
        power_density,
        spherical_coordinates,
        solid_angles,
        l_max,
    )
    power_by_l = jnp.sum(jnp.abs(coefficients) ** 2, axis=1)
    normalized_power_by_l = power_by_l / power_by_l[0]

    deposited_power = jnp.sum(deposition)
    total_area = jnp.sum(cell_areas)
    mean_power_density = deposited_power / total_area
    variance = (
        jnp.sum((power_density - mean_power_density) ** 2 * cell_areas) / total_area
    )

    return MetricsResult(
        harmonic_coefficients=coefficients,
        power_by_l=power_by_l,
        normalized_power_by_l=normalized_power_by_l,
        ell=jnp.arange(l_max + 1, dtype=jnp.int32),
        deposited_power=deposited_power,
        incident_power=incident_power,
        deposited_fraction=deposited_power / incident_power,
        mean_power_density=mean_power_density,
        rms_nonuniformity=jnp.sqrt(variance) / mean_power_density,
        simulation_index=simulation_index,
        l_max=l_max,
    )
