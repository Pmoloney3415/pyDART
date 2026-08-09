"""Differentiable transformations between beam parameters and beam state."""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
from jax import Array

from pydart.model.beams import Beams


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BeamParameters:
    """Continuously variable beam parameters used by optimization."""

    physical_origins: Array
    pointing_locations: Array
    power_fractions_of_maximum: Array
    spot_widths: Array
    supergaussian_indices: Array
    spot_rotations: Array

    def tree_flatten(self):
        return (
            self.physical_origins,
            self.pointing_locations,
            self.power_fractions_of_maximum,
            self.spot_widths,
            self.supergaussian_indices,
            self.spot_rotations,
        ), None

    @classmethod
    def tree_unflatten(cls, auxiliary_data, children):
        del auxiliary_data
        return cls(*children)


def parameters_from_beams(beams: Beams) -> BeamParameters:
    """Extract the continuously variable leaves from an initialized beam set."""
    return BeamParameters(
        physical_origins=beams.physical_origins,
        pointing_locations=beams.pointing_locations,
        power_fractions_of_maximum=beams.power_fractions_of_maximum,
        spot_widths=beams.spot_widths,
        supergaussian_indices=beams.supergaussian_indices,
        spot_rotations=beams.spot_rotations,
    )


def apply_beam_parameters(
    beams: Beams,
    parameters: BeamParameters,
) -> Beams:
    """Rebuild all derived beam geometry using only JAX operations.

    Facility power and per-beam maximum capacities remain fixed. The hard
    domain intersection and local-frame fallback make this transformation
    piecewise differentiable at their geometric boundaries.
    """
    direction_vectors = parameters.pointing_locations - parameters.physical_origins
    directions = direction_vectors / jnp.linalg.norm(
        direction_vectors,
        axis=-1,
        keepdims=True,
    )
    origins = _numerical_origins(
        parameters.physical_origins,
        directions,
        beams.numerical_domain_radius,
    )
    basis_x, basis_y = _local_beam_frames(
        directions,
        parameters.spot_rotations,
    )
    powers = (
        beams.facility_power
        * beams.maximum_power_fractions
        * parameters.power_fractions_of_maximum
    )
    return replace(
        beams,
        physical_origins=parameters.physical_origins,
        origins=origins,
        pointing_locations=parameters.pointing_locations,
        directions=directions,
        basis_x=basis_x,
        basis_y=basis_y,
        powers=powers,
        power_fractions_of_maximum=parameters.power_fractions_of_maximum,
        spot_widths=parameters.spot_widths,
        supergaussian_indices=parameters.supergaussian_indices,
        spot_rotations=parameters.spot_rotations,
    )


def _numerical_origins(
    physical_origins: Array,
    directions: Array,
    domain_radius: Array,
) -> Array:
    origin_radius = jnp.linalg.norm(physical_origins, axis=-1)
    outside = origin_radius > domain_radius
    along_axis = jnp.einsum("bi,bi->b", physical_origins, directions)
    discriminant = along_axis**2 - (origin_radius**2 - domain_radius**2)
    entry_distance = -along_axis - jnp.sqrt(jnp.maximum(discriminant, 0.0))
    intersections = physical_origins + entry_distance[:, None] * directions
    return jnp.where(outside[:, None], intersections, physical_origins)


def _local_beam_frames(
    directions: Array,
    rotations: Array,
) -> tuple[Array, Array]:
    global_z = jnp.asarray([0.0, 0.0, 1.0], dtype=directions.dtype)
    global_y = jnp.asarray([0.0, 1.0, 0.0], dtype=directions.dtype)
    projected_z = global_z - directions[:, 2:3] * directions
    projected_y = global_y - directions[:, 1:2] * directions
    use_y = jnp.linalg.norm(projected_z, axis=-1, keepdims=True) < 1.0e-12
    projected = jnp.where(use_y, projected_y, projected_z)
    unrotated_y = projected / jnp.linalg.norm(
        projected,
        axis=-1,
        keepdims=True,
    )
    unrotated_x = jnp.cross(unrotated_y, directions)
    cosine = jnp.cos(rotations)[:, None]
    sine = jnp.sin(rotations)[:, None]
    basis_x = cosine * unrotated_x + sine * unrotated_y
    basis_y = -sine * unrotated_x + cosine * unrotated_y
    return basis_x, basis_y
