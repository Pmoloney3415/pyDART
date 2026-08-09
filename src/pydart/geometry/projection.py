"""Inverse projection of surface locations onto transverse beam planes."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from pydart.model.beams import Beams


def project_to_beam_planes(
    surface_points: Array,
    beams: Beams,
) -> tuple[Array, Array]:
    """Back-project surface points onto every beam's numerical port plane.

    Returns projected Cartesian points with shape ``(..., n_beams, 3)`` and
    local ``(x', y')`` coordinates with shape ``(..., n_beams, 2)``.
    """
    relative = surface_points[..., None, :] - beams.origins
    axial_distance = jnp.einsum("...bi,bi->...b", relative, beams.directions)
    projected_points = (
        surface_points[..., None, :] - axial_distance[..., None] * beams.directions
    )
    transverse = projected_points - beams.origins
    local_x = jnp.einsum("...bi,bi->...b", transverse, beams.basis_x)
    local_y = jnp.einsum("...bi,bi->...b", transverse, beams.basis_y)
    return projected_points, jnp.stack((local_x, local_y), axis=-1)
