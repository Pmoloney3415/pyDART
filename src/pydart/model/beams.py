"""Batched, JAX-compatible laser beam data structures."""

from __future__ import annotations

from dataclasses import dataclass

import jax
from jax import Array


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Beams:
    """Numerical data for an ensemble of beams.

    Every array has a leading ``n_beams`` dimension. Names are static metadata
    and are therefore excluded from transformations such as ``jax.jit``.
    """

    origins: Array
    physical_origins: Array
    pointing_locations: Array
    directions: Array
    basis_x: Array
    basis_y: Array
    powers: Array
    maximum_power_fractions: Array
    power_fractions_of_maximum: Array
    facility_power: Array
    frequencies: Array
    spot_widths: Array
    supergaussian_indices: Array
    spot_rotations: Array
    spot_shape_codes: Array
    numerical_domain_radius: Array
    names: tuple[str, ...]

    def tree_flatten(self):
        children = (
            self.origins,
            self.physical_origins,
            self.pointing_locations,
            self.directions,
            self.basis_x,
            self.basis_y,
            self.powers,
            self.maximum_power_fractions,
            self.power_fractions_of_maximum,
            self.facility_power,
            self.frequencies,
            self.spot_widths,
            self.supergaussian_indices,
            self.spot_rotations,
            self.spot_shape_codes,
            self.numerical_domain_radius,
        )
        return children, self.names

    @classmethod
    def tree_unflatten(cls, names, children):
        return cls(*children, names=names)

    @property
    def n_beams(self) -> int:
        return self.origins.shape[0]
