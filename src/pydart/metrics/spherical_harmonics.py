"""JAX-compatible complex spherical-harmonic decomposition."""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax import Array


def spherical_harmonic_coefficients(
    power_density: Array,
    spherical_coordinates: Array,
    solid_angles: Array,
    l_max: int,
) -> Array:
    r"""Return coefficients ``a_lm = integral(q Y_lm* dOmega)``.

    The output shape is ``(l_max + 1, 2*l_max + 1)`` and the second index is
    ``m + l_max``. Entries for which ``abs(m) > l`` are zero. The harmonics
    are complex, orthonormal, and include the Condon--Shortley phase.
    """
    phi = spherical_coordinates[..., 1]
    theta = spherical_coordinates[..., 2]
    cosine_theta = jnp.cos(theta)
    sine_theta = jnp.sqrt(jnp.maximum(1.0 - cosine_theta**2, 0.0))
    weighted_density = power_density * solid_angles
    coefficients = jnp.zeros(
        (l_max + 1, 2 * l_max + 1),
        dtype=jnp.complex128,
    )

    p_mm = jnp.ones_like(cosine_theta)
    for m in range(l_max + 1):
        coefficients = _store_coefficient(
            coefficients,
            weighted_density,
            phi,
            p_mm,
            l=m,
            m=m,
            l_max=l_max,
        )

        if m < l_max:
            p_l_minus_two = p_mm
            p_l_minus_one = (2 * m + 1) * cosine_theta * p_mm
            coefficients = _store_coefficient(
                coefficients,
                weighted_density,
                phi,
                p_l_minus_one,
                l=m + 1,
                m=m,
                l_max=l_max,
            )

            for ell in range(m + 2, l_max + 1):
                p_lm = (
                    (2 * ell - 1) * cosine_theta * p_l_minus_one
                    - (ell + m - 1) * p_l_minus_two
                ) / (ell - m)
                coefficients = _store_coefficient(
                    coefficients,
                    weighted_density,
                    phi,
                    p_lm,
                    l=ell,
                    m=m,
                    l_max=l_max,
                )
                p_l_minus_two, p_l_minus_one = p_l_minus_one, p_lm

        p_mm = -(2 * m + 1) * sine_theta * p_mm

    return coefficients


def _store_coefficient(
    coefficients: Array,
    weighted_density: Array,
    phi: Array,
    associated_legendre: Array,
    l: int,
    m: int,
    l_max: int,
) -> Array:
    normalization = math.sqrt(
        (2 * l + 1)
        / (4.0 * math.pi)
        * math.exp(math.lgamma(l - m + 1) - math.lgamma(l + m + 1))
    )
    harmonic = normalization * associated_legendre * jnp.exp(1j * m * phi)
    coefficient = jnp.sum(weighted_density * jnp.conj(harmonic))
    coefficients = coefficients.at[l, l_max + m].set(coefficient)
    if m > 0:
        negative_coefficient = ((-1) ** m) * jnp.conj(coefficient)
        coefficients = coefficients.at[l, l_max - m].set(negative_coefficient)
    return coefficients
