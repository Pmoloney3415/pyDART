"""Semantic validation for simulation configurations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pydart.config.simulation_config import BeamConfig, PyDARTConfig


def validate_simulation_config(config: PyDARTConfig) -> None:
    """Validate a fully constructed simulation configuration."""
    if config.simulation.plot_dpi <= 0:
        raise ValueError("simulation.plot_dpi must be positive.")
    if config.simulation.surface_quadrature_order <= 0:
        raise ValueError("simulation.surface_quadrature_order must be positive.")
    if not 0.0 <= config.simulation.visibility_smoothing_epsilon <= 1.0:
        raise ValueError("simulation.visibility_smoothing_epsilon must be in [0, 1].")
    if config.target.radius <= 0.0:
        raise ValueError("Target radius must be positive.")
    if config.target.n_polar <= 0:
        raise ValueError("n_polar must be positive.")
    if config.target.n_azimuthal <= 0:
        raise ValueError("n_azimuthal must be positive.")
    if config.target.numerical_origin_radius_factor <= 1.0:
        raise ValueError("numerical_origin_radius_factor must be greater than 1.")
    if config.metrics.l_max < 0:
        raise ValueError("metrics.l_max cannot be negative.")
    if config.metrics.l_max >= config.target.n_polar:
        raise ValueError("metrics.l_max must be smaller than target.n_polar.")
    if 2 * config.metrics.l_max >= config.target.n_azimuthal:
        raise ValueError("target.n_azimuthal must be greater than 2 * metrics.l_max.")
    if config.laser.total_incident_power <= 0.0:
        raise ValueError("Total incident power must be positive.")
    if config.laser.n_beams != len(config.beams):
        raise ValueError(
            f"n_beams = {config.laser.n_beams}, "
            f"but {len(config.beams)} beam definitions were supplied."
        )

    maximum_power_fraction_sum = sum(
        beam.maximum_power_fraction for beam in config.beams
    )
    if not np.isclose(maximum_power_fraction_sum, 1.0, rtol=1.0e-4, atol=1.0e-4):
        raise ValueError(
            "Beam maximum_power_fraction values must sum to 1. "
            f"Current sum = {maximum_power_fraction_sum}."
        )
    for beam in config.beams:
        _validate_beam(beam)


def _validate_beam(beam: BeamConfig) -> None:
    supported_coordinate_systems = {"cartesian", "spherical"}
    if beam.origin_coordinate_system not in supported_coordinate_systems:
        raise ValueError(
            f"Unsupported origin coordinate system '{beam.origin_coordinate_system}'."
        )
    if beam.pointing_coordinate_system not in supported_coordinate_systems:
        raise ValueError(
            "Unsupported pointing coordinate system "
            f"'{beam.pointing_coordinate_system}'."
        )
    if beam.origin.shape != (3,):
        raise ValueError(f"Beam '{beam.name}' origin must contain 3 values.")
    if beam.pointing.shape != (3,):
        raise ValueError(f"Beam '{beam.name}' pointing must contain 3 values.")

    _validate_location(beam.name, "origin", beam.origin, beam.origin_coordinate_system)
    _validate_location(
        beam.name, "pointing", beam.pointing, beam.pointing_coordinate_system
    )
    if beam.maximum_power_fraction <= 0.0:
        raise ValueError(f"Beam '{beam.name}' maximum power fraction must be positive.")
    if not 0.0 <= beam.power_fraction_of_maximum <= 1.0:
        raise ValueError(
            f"Beam '{beam.name}' power fraction of maximum must be in [0, 1]."
        )
    if beam.frequency < 0.0:
        raise ValueError(f"Beam '{beam.name}' frequency cannot be negative.")
    if beam.spot.shape not in {"circular", "elliptical"}:
        raise ValueError(f"Unknown spot shape '{beam.spot.shape}'.")
    if beam.spot.width_x <= 0.0 or beam.spot.width_y <= 0.0:
        raise ValueError(f"Beam '{beam.name}' spot widths must be positive.")
    if beam.spot.supergaussian_index <= 0.0:
        raise ValueError(f"Beam '{beam.name}' super-Gaussian index must be positive.")
    if beam.spot.rotation < 0.0 or beam.spot.rotation >= 360.0:
        raise ValueError(
            f"Beam '{beam.name}' spot rotation must be in [0, 360) degrees."
        )
    if beam.spot.shape == "circular" and not np.isclose(
        beam.spot.width_x, beam.spot.width_y
    ):
        raise ValueError(f"Circular beam '{beam.name}' must have width_x == width_y.")


def _validate_location(
    beam_name: str,
    location_name: str,
    location: np.ndarray,
    coordinate_system: str,
) -> None:
    if not np.all(np.isfinite(location)):
        raise ValueError(
            f"Beam '{beam_name}' {location_name} must contain finite values."
        )
    if coordinate_system != "spherical":
        return

    radius, phi, theta = location
    if radius < 0.0:
        raise ValueError(
            f"Beam '{beam_name}' {location_name} radius cannot be negative."
        )
    if phi < -np.pi or phi >= np.pi:
        raise ValueError(
            f"Beam '{beam_name}' {location_name} azimuth must be in [-pi, pi)."
        )
    if theta < 0.0 or theta > np.pi:
        raise ValueError(
            f"Beam '{beam_name}' {location_name} polar angle must be in [0, pi]."
        )
