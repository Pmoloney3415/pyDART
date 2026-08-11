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
    rms_weight: float
    deposited_power_weight: float
    mode_weight_option: str
    l1_mode_weight: float | None
    mode_decrease_power: float | None
    mode_weights: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class OptimisationConfig:
    run: OptimisationRunConfig
    variables: VariableConfig
    objective: ObjectiveConfig
    simulation: PyDARTConfig
    restarts: RestartConfig
    source_path: Path


def decreasing_mode_weights(
    l_max: int,
    l1_mode_weight: float,
    mode_decrease_power: float,
) -> tuple[tuple[int, float], ...]:
    """Return weights ``w_l = w_1 / l**p`` through the simulation ``l_max``."""
    return tuple(
        (
            degree,
            l1_mode_weight / degree**mode_decrease_power,
        )
        for degree in range(1, l_max + 1)
    )


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
    mode_weight_option = str(
        objective_data.get(
            "mode_weight_option",
            "explicit" if "mode_weights" in objective_data else "decreasing",
        )
    )
    l1_mode_weight = None
    mode_decrease_power = None
    if mode_weight_option == "explicit":
        if "mode_weights" not in objective_data:
            raise ValueError(
                "Explicit mode weighting requires [objective.mode_weights]."
            )
        mode_weights = tuple(
            sorted(
                (int(degree), float(weight))
                for degree, weight in objective_data["mode_weights"].items()
            )
        )
    elif mode_weight_option == "decreasing":
        if "mode_weights" in objective_data:
            raise ValueError(
                "Do not provide explicit mode_weights with decreasing weighting."
            )
        l1_mode_weight = float(objective_data.get("l1_mode_weight", 1.0))
        mode_decrease_power = float(objective_data.get("mode_decrease_power", 2.0))
        mode_weights = decreasing_mode_weights(
            simulation.metrics.l_max,
            l1_mode_weight,
            mode_decrease_power,
        )
    else:
        raise ValueError("mode_weight_option must be 'explicit' or 'decreasing'.")
    objective = ObjectiveConfig(
        rms_weight=float(objective_data.get("rms_weight", 1.0)),
        deposited_power_weight=float(
            objective_data.get("deposited_power_weight", 0.25)
        ),
        mode_weight_option=mode_weight_option,
        l1_mode_weight=l1_mode_weight,
        mode_decrease_power=mode_decrease_power,
        mode_weights=mode_weights,
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
