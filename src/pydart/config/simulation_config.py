# src/pydart/config.py

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    index: int
    output_directory: Path
    save_deposition_data: bool
    save_metrics: bool
    plot_data: bool = False
    plot_dpi: int = 200
    surface_quadrature_order: int = 1
    visibility_smoothing_epsilon: float = 0.05


@dataclass(frozen=True)
class TargetConfig:
    radius: float
    n_polar: int
    n_azimuthal: int
    numerical_origin_radius_factor: float = 10.0


@dataclass(frozen=True)
class LaserConfig:
    n_beams: int
    total_incident_power: float


@dataclass(frozen=True)
class SpotConfig:
    shape: str
    width_x: float
    width_y: float
    supergaussian_index: float
    rotation: float


@dataclass(frozen=True)
class BeamConfig:
    name: str
    origin_coordinate_system: str
    pointing_coordinate_system: str
    origin: np.ndarray
    pointing: np.ndarray
    maximum_power_fraction: float
    power_fraction_of_maximum: float
    frequency: float
    spot: SpotConfig


@dataclass(frozen=True)
class MetricsConfig:
    l_max: int = 20


@dataclass(frozen=True)
class PyDARTConfig:
    simulation: SimulationConfig
    target: TargetConfig
    laser: LaserConfig
    beams: tuple[BeamConfig, ...]
    metrics: MetricsConfig = MetricsConfig()


def load_config(filename: str | Path) -> PyDARTConfig:
    """
    Read and validate a pyDART TOML input deck.
    """

    filename = Path(filename)

    with filename.open("rb") as f:
        data = tomllib.load(f)

    simulation = _read_simulation_config(data["simulation"])
    target = _read_target_config(data["target"])
    laser = _read_laser_config(data["laser"])
    metrics = _read_metrics_config(data.get("metrics", {}))

    beams = tuple(_read_beam_config(entry) for entry in data["beam"])

    config = PyDARTConfig(
        simulation=simulation,
        target=target,
        laser=laser,
        beams=beams,
        metrics=metrics,
    )

    from pydart.config.simulation_validation import validate_simulation_config

    validate_simulation_config(config)

    return config


def _read_simulation_config(data: dict) -> SimulationConfig:
    return SimulationConfig(
        index=int(data["index"]),
        output_directory=Path(data["output_directory"]),
        save_deposition_data=bool(data["save_deposition_data"]),
        save_metrics=bool(data["save_metrics"]),
        plot_data=bool(data.get("plot_data", False)),
        plot_dpi=int(data.get("plot_dpi", 200)),
        surface_quadrature_order=int(data.get("surface_quadrature_order", 1)),
        visibility_smoothing_epsilon=float(
            data.get("visibility_smoothing_epsilon", 0.05)
        ),
    )


def _read_target_config(data: dict) -> TargetConfig:
    return TargetConfig(
        radius=float(data["radius"]),
        n_polar=int(data["n_polar"]),
        n_azimuthal=int(data["n_azimuthal"]),
        numerical_origin_radius_factor=float(
            data.get("numerical_origin_radius_factor", 10.0)
        ),
    )


def _read_laser_config(data: dict) -> LaserConfig:
    return LaserConfig(
        n_beams=int(data["n_beams"]),
        total_incident_power=float(data["total_incident_power"]),
    )


def _read_metrics_config(data: dict) -> MetricsConfig:
    return MetricsConfig(l_max=int(data.get("l_max", 20)))


def _read_beam_config(data: dict) -> BeamConfig:

    spot_data = data["spot"]

    spot = SpotConfig(
        shape=str(spot_data["shape"]),
        width_x=float(spot_data["width_x"]),
        width_y=float(spot_data["width_y"]),
        supergaussian_index=float(spot_data["supergaussian_index"]),
        rotation=float(spot_data["rotation"]),
    )

    return BeamConfig(
        name=str(data["name"]),
        origin_coordinate_system=str(data["origin_coordinate_system"]),
        pointing_coordinate_system=str(data["pointing_coordinate_system"]),
        origin=np.asarray(data["origin"], dtype=float),
        pointing=np.asarray(data["pointing"], dtype=float),
        maximum_power_fraction=float(data["maximum_power_fraction"]),
        power_fraction_of_maximum=float(data["power_fraction_of_maximum"]),
        frequency=float(data["frequency"]),
        spot=spot,
    )
