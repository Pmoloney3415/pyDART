"""Coordinate conversions used by the numerical core.

Spherical coordinates are ordered ``(r, phi, theta)``. ``phi`` is the
azimuthal angle in ``[-pi, pi)`` and ``theta`` is the polar angle in
``[0, pi]`` measured down from the positive z-axis.
"""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
from jax import Array

CoordinateSystem = Literal["cartesian", "spherical"]


def spherical_to_cartesian(coordinates: Array) -> Array:
    """Convert one or more ``(..., 3)`` spherical locations to Cartesian."""
    coordinates = jnp.asarray(coordinates)
    radius, phi, theta = jnp.moveaxis(coordinates, -1, 0)
    radial_xy = radius * jnp.sin(theta)
    return jnp.stack(
        (
            radial_xy * jnp.cos(phi),
            radial_xy * jnp.sin(phi),
            radius * jnp.cos(theta),
        ),
        axis=-1,
    )


def cartesian_to_spherical(coordinates: Array) -> Array:
    """Convert one or more ``(..., 3)`` Cartesian locations to spherical."""
    coordinates = jnp.asarray(coordinates)
    x, y, z = jnp.moveaxis(coordinates, -1, 0)
    radius = jnp.linalg.norm(coordinates, axis=-1)
    safe_radius = jnp.where(radius > 0, radius, 1)
    theta = jnp.where(
        radius > 0,
        jnp.arccos(jnp.clip(z / safe_radius, -1, 1)),
        0,
    )
    phi = jnp.arctan2(y, x)
    return jnp.stack((radius, phi, theta), axis=-1)


def to_cartesian(
    coordinates: Array,
    coordinate_system: CoordinateSystem,
) -> Array:
    """Return Cartesian locations from either supported coordinate system."""
    if coordinate_system == "cartesian":
        return jnp.asarray(coordinates)
    if coordinate_system == "spherical":
        return spherical_to_cartesian(coordinates)
    raise ValueError(f"Unsupported coordinate system '{coordinate_system}'.")
