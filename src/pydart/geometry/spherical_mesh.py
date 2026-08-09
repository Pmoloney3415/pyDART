"""Construction of cell-centred spherical surface meshes."""

from __future__ import annotations

import jax.numpy as jnp

from pydart.geometry.coordinates import spherical_to_cartesian
from pydart.model.target import Target


def create_spherical_target(
    radius: float,
    n_azimuthal: int,
    n_polar: int,
) -> Target:
    """Create a uniform angular mesh with exact finite-cell surface areas."""
    phi_edges = jnp.linspace(-jnp.pi, jnp.pi, n_azimuthal + 1)
    theta_edges = jnp.linspace(0.0, jnp.pi, n_polar + 1)
    phi = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    theta = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    phi_grid, theta_grid = jnp.meshgrid(phi, theta, indexing="ij")

    radius_array = jnp.asarray(radius)
    radii = jnp.full_like(phi_grid, radius_array)
    spherical_coordinates = jnp.stack(
        (radii, phi_grid, theta_grid),
        axis=-1,
    )
    cartesian_coordinates = spherical_to_cartesian(spherical_coordinates)
    surface_normals = cartesian_coordinates / radius_array

    delta_phi = phi_edges[1:] - phi_edges[:-1]
    polar_integrals = jnp.cos(theta_edges[:-1]) - jnp.cos(theta_edges[1:])
    cell_areas = radius_array**2 * delta_phi[:, None] * polar_integrals[None, :]

    return Target(
        radius=radius_array,
        spherical_coordinates=spherical_coordinates,
        cartesian_coordinates=cartesian_coordinates,
        surface_normals=surface_normals,
        cell_areas=cell_areas,
    )


def create_spherical_cell_quadrature(
    target: Target,
    order: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return equal-area midpoint samples within every spherical cell.

    The quadrature is uniform in azimuth and ``cos(theta)``, the natural
    surface-area coordinates on a sphere. Returned arrays have leading shape
    ``(n_azimuthal, n_polar, order, order)``. The final dimension of the point
    and normal arrays contains Cartesian components.
    """
    if order <= 0:
        raise ValueError("Spherical cell quadrature order must be positive.")

    n_azimuthal, n_polar = target.cell_areas.shape
    phi_edges = jnp.linspace(-jnp.pi, jnp.pi, n_azimuthal + 1)
    theta_edges = jnp.linspace(0.0, jnp.pi, n_polar + 1)
    fractions = (jnp.arange(order, dtype=target.radius.dtype) + 0.5) / order

    phi = (
        phi_edges[:-1, None]
        + (phi_edges[1:] - phi_edges[:-1])[:, None] * fractions[None, :]
    )
    cosine_edges = jnp.cos(theta_edges)
    cosine_theta = (
        cosine_edges[:-1, None]
        + (cosine_edges[1:] - cosine_edges[:-1])[:, None] * fractions[None, :]
    )
    theta = jnp.arccos(jnp.clip(cosine_theta, -1.0, 1.0))

    phi_grid = jnp.broadcast_to(
        phi[:, None, :, None],
        (n_azimuthal, n_polar, order, order),
    )
    theta_grid = jnp.broadcast_to(
        theta[None, :, None, :],
        (n_azimuthal, n_polar, order, order),
    )
    sine_theta = jnp.sin(theta_grid)
    normals = jnp.stack(
        (
            sine_theta * jnp.cos(phi_grid),
            sine_theta * jnp.sin(phi_grid),
            jnp.cos(theta_grid),
        ),
        axis=-1,
    )
    points = target.radius * normals
    areas = jnp.broadcast_to(
        target.cell_areas[..., None, None] / order**2,
        (n_azimuthal, n_polar, order, order),
    )
    return points, normals, areas
