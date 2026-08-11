"""Semantic validation for optimisation configurations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydart.config.optimisation_config import (
        OptimisationConfig,
        SurfaceVariableConfig,
    )


def validate_optimisation_config(config: OptimisationConfig) -> None:
    """Validate a fully constructed optimisation configuration."""
    if config.run.solver not in {"scipy_lbfgsb", "jaxopt_lbfgsb"}:
        raise ValueError(
            "optimisation.solver must be 'scipy_lbfgsb' or 'jaxopt_lbfgsb'."
        )
    if config.run.maximum_iterations <= 0:
        raise ValueError("maximum_iterations must be positive.")
    if config.run.objective_relative_tolerance <= 0.0:
        raise ValueError("objective_relative_tolerance must be positive.")
    if config.run.projected_gradient_tolerance <= 0.0:
        raise ValueError("projected_gradient_tolerance must be positive.")
    if config.run.maximum_wall_time_seconds <= 0.0:
        raise ValueError("maximum_wall_time_seconds must be positive.")
    if config.run.checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive.")
    if config.run.history_plot_interval <= 0:
        raise ValueError("history_plot_interval must be positive.")
    if config.restarts.number <= 0:
        raise ValueError("optimisation.restarts.number must be positive.")

    beam_names = {beam.name for beam in config.simulation.beams}
    unknown_frozen_beams = set(config.variables.frozen_beams) - beam_names
    if unknown_frozen_beams:
        raise ValueError(f"Unknown frozen beam names: {sorted(unknown_frozen_beams)}.")
    if set(config.variables.frozen_beams) == beam_names:
        raise ValueError("At least one beam must remain unfrozen.")

    power = config.variables.power
    if not 0.0 <= power.minimum_fraction_of_maximum:
        raise ValueError("Minimum power fraction cannot be negative.")
    if power.maximum_fraction_of_maximum > 1.0:
        raise ValueError("Maximum power fraction cannot exceed 1.")
    if power.minimum_fraction_of_maximum > power.maximum_fraction_of_maximum:
        raise ValueError("Power fraction bounds are reversed.")

    _validate_surface_variables("origin", config.variables.origin)
    _validate_surface_variables("pointing", config.variables.pointing)
    spot = config.variables.spot
    _validate_bounds("width_x", spot.minimum_width_x, spot.maximum_width_x, 0.0)
    _validate_bounds("width_y", spot.minimum_width_y, spot.maximum_width_y, 0.0)
    _validate_bounds(
        "supergaussian_index",
        spot.minimum_supergaussian_index,
        spot.maximum_supergaussian_index,
        0.0,
    )
    if not 0.0 <= spot.minimum_rotation_degrees < 180.0:
        raise ValueError("Minimum rotation must be in [0, 180).")
    if not 0.0 < spot.maximum_rotation_degrees <= 180.0:
        raise ValueError("Maximum rotation must be in (0, 180].")
    if spot.minimum_rotation_degrees >= spot.maximum_rotation_degrees:
        raise ValueError("Rotation bounds are reversed.")
    if spot.force_circular and spot.rotation_enabled:
        raise ValueError("Rotation cannot be optimized for forced-circular spots.")
    if not any(
        (
            power.enabled,
            config.variables.origin.enabled,
            config.variables.pointing.enabled,
            spot.width_enabled,
            spot.rotation_enabled,
            spot.supergaussian_index_enabled,
        )
    ):
        raise ValueError("At least one variable type must be enabled.")

    for degree, weight in config.objective.mode_weights:
        if degree < 0 or degree > config.simulation.metrics.l_max:
            raise ValueError(f"Objective mode degree {degree} is unavailable.")
        if weight < 0.0:
            raise ValueError("Objective weights cannot be negative.")
    if config.objective.rms_weight < 0.0:
        raise ValueError("Objective weights cannot be negative.")
    if config.objective.deposited_power_weight < 0.0:
        raise ValueError("Objective weights cannot be negative.")
    if (
        config.objective.l1_mode_weight is not None
        and config.objective.l1_mode_weight < 0.0
    ):
        raise ValueError("The l=1 mode weight cannot be negative.")
    if (
        config.objective.mode_decrease_power is not None
        and config.objective.mode_decrease_power < 0.0
    ):
        raise ValueError("The mode decrease power cannot be negative.")


def _validate_surface_variables(name: str, variables: SurfaceVariableConfig) -> None:
    if variables.constraint not in {"bounded", "unconstrained"}:
        raise ValueError(f"{name}.constraint must be 'bounded' or 'unconstrained'.")
    if variables.constraint == "bounded":
        angle = variables.maximum_angular_displacement_degrees
        if angle is None or not 0.0 < angle <= 180.0:
            raise ValueError(
                f"{name} bounded angular displacement must be in (0, 180]."
            )


def _validate_bounds(name: str, minimum: float, maximum: float, floor: float) -> None:
    if minimum <= floor:
        raise ValueError(f"Minimum {name} must be greater than {floor}.")
    if minimum >= maximum:
        raise ValueError(f"{name} bounds are reversed or equal.")
