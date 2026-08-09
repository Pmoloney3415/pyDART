"""JAX-compatible target surface data structures."""

from __future__ import annotations

from dataclasses import dataclass

import jax
from jax import Array


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Target:
    """Cell-centred discretisation of a spherical target surface."""

    radius: Array
    spherical_coordinates: Array
    cartesian_coordinates: Array
    surface_normals: Array
    cell_areas: Array

    def tree_flatten(self):
        return (
            self.radius,
            self.spherical_coordinates,
            self.cartesian_coordinates,
            self.surface_normals,
            self.cell_areas,
        ), None

    @classmethod
    def tree_unflatten(cls, auxiliary_data, children):
        del auxiliary_data
        return cls(*children)
