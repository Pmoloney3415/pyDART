"""Normalized transverse laser intensity profiles."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.scipy.special import gamma

from pydart.model.beams import Beams


def supergaussian_intensity(local_coordinates: Array, beams: Beams) -> Array:
    r"""Evaluate normalized elliptical super-Gaussian beam intensities.

    The convention is

    ``I = I0 exp(-[((x/wx)^2 + (y/wy)^2)^(m/2)])``.

    ``I0`` is selected so that integration over the complete transverse plane
    equals each beam's incident power.
    """
    scaled = local_coordinates / beams.spot_widths
    elliptical_radius_squared = jnp.sum(scaled**2, axis=-1)
    indices = beams.supergaussian_indices
    exponent = elliptical_radius_squared ** (indices / 2.0)
    normalization_area = (
        2.0
        * jnp.pi
        * beams.spot_widths[:, 0]
        * beams.spot_widths[:, 1]
        * gamma(2.0 / indices)
        / indices
    )
    peak_intensities = beams.powers / normalization_area
    return peak_intensities * jnp.exp(-exponent)
