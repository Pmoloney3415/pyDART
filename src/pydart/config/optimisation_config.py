"""Configuration schema for beam-layout optimization problems."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from pydart.config.simulation_config import PyDARTConfig, load_config


@dataclass(frozen=True)
class OptimisationRunConfig:
    index: int
    solver: str
    simulation_config: Path
    output_directory: Path
    maximum_iterations: int
    device_iteration_chunk_size: int
    objective_relative_tolerance: float
    projected_gradient_tolerance: float
    maximum_wall_time_seconds: float
    checkpoint_interval: int
    history_plot_interval: int
    save_best_simulation: bool
    save_simulation_plots: bool
    archive_previous_best_simulations: bool


@dataclass(frozen=True)
class RestartConfig:
    number: int
    include_base_design: bool
    random_seed: int


@dataclass(frozen=True)
class PowerVariableConfig:
    enabled: bool
    minimum_fraction_of_maximum: float
    maximum_fraction_of_maximum: float


@dataclass(frozen=True)
class SurfaceVariableConfig:
    enabled: bool
    constraint: str
    maximum_angular_displacement_degrees: float | None


@dataclass(frozen=True)
class SpotVariableConfig:
    width_enabled: bool
    force_circular: bool
    minimum_width_x: float
    maximum_width_x: float
    minimum_width_y: float
    maximum_width_y: float
    rotation_enabled: bool
    minimum_rotation_degrees: float
    maximum_rotation_degrees: float
    supergaussian_index_enabled: bool
    minimum_supergaussian_index: float
    maximum_supergaussian_index: float


@dataclass(frozen=True)
class VariableConfig:
    frozen_beams: tuple[str, ...]
    power: PowerVariableConfig
    origin: SurfaceVariableConfig
    pointing: SurfaceVariableConfig
    spot: SpotVariableConfig


@dataclass(frozen=True)
class ObjectiveConfig:
    deposition_log_weight: float
    deposition_log_epsilon: float
    acceptable_rms_nonuniformity: float
    rms_power: float


@dataclass(frozen=True)
class OptimisationConfig:
    run: OptimisationRunConfig
    variables: VariableConfig
    objective: ObjectiveConfig
    simulation: PyDARTConfig
    restarts: RestartConfig
    source_path: Path


def load_optimisation_config(filename: str | Path) -> OptimisationConfig:
    """Load an optimization deck and its referenced base simulation deck."""
    filename = Path(filename).resolve()
    with filename.open("rb") as stream:
        data = tomllib.load(stream)
    base_directory = filename.parent
    run_data = data["optimisation"]
    simulation_path = (base_directory / run_data["simulation_config"]).resolve()
    simulation = load_config(simulation_path)
    run = OptimisationRunConfig(
        index=int(run_data["index"]),
        solver=str(run_data.get("solver", "scipy_lbfgsb")),
        simulation_config=simulation_path,
        output_directory=(base_directory / run_data["output_directory"]).resolve(),
        maximum_iterations=int(run_data["maximum_iterations"]),
        device_iteration_chunk_size=int(
            run_data.get("device_iteration_chunk_size", 10)
        ),
        objective_relative_tolerance=float(
            run_data.get("objective_relative_tolerance", 1.0e-9)
        ),
        projected_gradient_tolerance=float(
            run_data.get("projected_gradient_tolerance", 1.0e-6)
        ),
        maximum_wall_time_seconds=float(
            run_data.get("maximum_wall_time_seconds", 3600.0)
        ),
        checkpoint_interval=int(run_data.get("checkpoint_interval", 10)),
        history_plot_interval=int(run_data.get("history_plot_interval", 10)),
        save_best_simulation=bool(run_data.get("save_best_simulation", True)),
        save_simulation_plots=bool(run_data.get("save_simulation_plots", True)),
        archive_previous_best_simulations=bool(
            run_data.get("archive_previous_best_simulations", False)
        ),
    )
    restart_data = run_data.get("restarts", {})
    restarts = RestartConfig(
        number=int(restart_data.get("number", 1)),
        include_base_design=bool(restart_data.get("include_base_design", True)),
        random_seed=int(restart_data.get("random_seed", 0)),
    )
    variable_data = data["variables"]
    variables = VariableConfig(
        frozen_beams=tuple(str(name) for name in variable_data["frozen_beams"]),
        power=_read_power_variables(variable_data["power"]),
        origin=_read_surface_variables(variable_data["origin"]),
        pointing=_read_surface_variables(variable_data["pointing"]),
        spot=_read_spot_variables(variable_data["spot"]),
    )
    objective_data = data.get("objective", {})
    obsolete_keys = {
        "deposited_power_weight",
        "deposition_exponential_weight",
        "symmetry_log_epsilon",
        "deposition_shortfall_weight",
        "deposition_huber_width",
        "objective_log_epsilon",
        "rms_weight",
        "mode_weight_option",
        "mode_weights",
        "l1_mode_weight",
        "mode_decrease_power",
    } & objective_data.keys()
    if obsolete_keys:
        raise ValueError(
            f"Obsolete objective options {sorted(obsolete_keys)} have been replaced "
            "by deposition_log_weight, deposition_log_epsilon, "
            "acceptable_rms_nonuniformity, and rms_power."
        )
    objective = ObjectiveConfig(
        deposition_log_weight=float(
            objective_data.get("deposition_log_weight", 2.0)
        ),
        deposition_log_epsilon=float(
            objective_data.get("deposition_log_epsilon", 1.0e-8)
        ),
        acceptable_rms_nonuniformity=float(
            objective_data.get("acceptable_rms_nonuniformity", 0.01)
        ),
        rms_power=float(objective_data.get("rms_power", 2.0)),
    )
    config = OptimisationConfig(
        run=run,
        variables=variables,
        objective=objective,
        simulation=simulation,
        restarts=restarts,
        source_path=filename,
    )
    from pydart.config.optimisation_validation import validate_optimisation_config

    validate_optimisation_config(config)
    return config


def _read_power_variables(data: dict) -> PowerVariableConfig:
    return PowerVariableConfig(
        enabled=bool(data["enabled"]),
        minimum_fraction_of_maximum=float(data["minimum_fraction_of_maximum"]),
        maximum_fraction_of_maximum=float(data["maximum_fraction_of_maximum"]),
    )


def _read_surface_variables(data: dict) -> SurfaceVariableConfig:
    maximum_displacement = data.get("maximum_angular_displacement_degrees")
    return SurfaceVariableConfig(
        enabled=bool(data["enabled"]),
        constraint=str(data["constraint"]),
        maximum_angular_displacement_degrees=(
            None if maximum_displacement is None else float(maximum_displacement)
        ),
    )


def _read_spot_variables(data: dict) -> SpotVariableConfig:
    return SpotVariableConfig(
        width_enabled=bool(data["width_enabled"]),
        force_circular=bool(data["force_circular"]),
        minimum_width_x=float(data["minimum_width_x"]),
        maximum_width_x=float(data["maximum_width_x"]),
        minimum_width_y=float(data["minimum_width_y"]),
        maximum_width_y=float(data["maximum_width_y"]),
        rotation_enabled=bool(data["rotation_enabled"]),
        minimum_rotation_degrees=float(data["minimum_rotation_degrees"]),
        maximum_rotation_degrees=float(data["maximum_rotation_degrees"]),
        supergaussian_index_enabled=bool(data["supergaussian_index_enabled"]),
        minimum_supergaussian_index=float(data["minimum_supergaussian_index"]),
        maximum_supergaussian_index=float(data["maximum_supergaussian_index"]),
    )
