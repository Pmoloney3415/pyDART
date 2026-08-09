"""Beam visibility on outward-facing target surfaces."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from pydart.model.beams import Beams
from pydart.model.target import Target


def projected_cosines(target: Target, beams: Beams) -> Array:
    """Return positive projected-area factors for all cells and beams."""
    return projected_cosines_from_normals(target.surface_normals, beams)


def projected_cosines_from_normals(
    surface_normals: Array,
    beams: Beams,
    smoothing_epsilon: float = 0.0,
) -> Array:
    """Return exact or smoothly regularized projected-area factors."""
    normal_dot_direction = jnp.einsum(
        "...i,bi->...b",
        surface_normals,
        beams.directions,
    )
    incidence_cosine = -normal_dot_direction
    epsilon = jnp.asarray(smoothing_epsilon, dtype=incidence_cosine.dtype)
    return 0.5 * (incidence_cosine + jnp.sqrt(incidence_cosine**2 + epsilon**2))


def visible_beams(target: Target, beams: Beams) -> Array:
    """Return whether each beam is incident on each surface cell."""
    return projected_cosines(target, beams) > 0.0
