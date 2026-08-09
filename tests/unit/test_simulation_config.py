from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pydart.config.simulation_config import (
    BeamConfig,
    LaserConfig,
    MetricsConfig,
    PyDARTConfig,
    SimulationConfig,
    SpotConfig,
    TargetConfig,
    load_config,
)
from pydart.config.simulation_validation import validate_simulation_config

CONFIG_DIRECTORY = Path(__file__).parents[2] / "configs" / "simulations"


@pytest.mark.parametrize("filename", ["OMEGA60_500um.toml", "six_beam_500um.toml"])
def test_bundled_simulation_configs_load(filename: str) -> None:
    config = load_config(CONFIG_DIRECTORY / filename)

    assert len(config.beams) == config.laser.n_beams
    assert config.beams
    assert all(beam.origin.shape == (3,) for beam in config.beams)
    assert all(beam.pointing.shape == (3,) for beam in config.beams)
    assert all(np.all(np.isfinite(beam.origin)) for beam in config.beams)
    assert sum(beam.maximum_power_fraction for beam in config.beams) == pytest.approx(
        1.0
    )
    assert config.metrics.l_max < config.target.n_polar
    assert 2 * config.metrics.l_max < config.target.n_azimuthal


def _valid_config() -> PyDARTConfig:
    beam = BeamConfig(
        name="beam",
        origin_coordinate_system="cartesian",
        pointing_coordinate_system="cartesian",
        origin=np.asarray([2.0, 0.0, 0.0]),
        pointing=np.zeros(3),
        maximum_power_fraction=1.0,
        power_fraction_of_maximum=1.0,
        frequency=1.0,
        spot=SpotConfig("circular", 0.1, 0.1, 2.0, 0.0),
    )
    return PyDARTConfig(
        simulation=SimulationConfig(0, Path("results"), False, False),
        target=TargetConfig(1.0, 8, 16),
        laser=LaserConfig(1, 10.0),
        beams=(beam,),
        metrics=MetricsConfig(3),
    )


@pytest.mark.parametrize(
    ("invalid_config", "message"),
    [
        (
            lambda config: replace(config, target=replace(config.target, radius=0.0)),
            "radius",
        ),
        (
            lambda config: replace(config, laser=replace(config.laser, n_beams=2)),
            "n_beams",
        ),
        (
            lambda config: replace(
                config,
                beams=(replace(config.beams[0], maximum_power_fraction=0.5),),
            ),
            "sum to 1",
        ),
        (
            lambda config: replace(
                config,
                beams=(
                    replace(
                        config.beams[0],
                        spot=replace(config.beams[0].spot, width_x=0.0),
                    ),
                ),
            ),
            "spot widths",
        ),
        (
            lambda config: replace(
                config, metrics=replace(config.metrics, l_max=config.target.n_polar)
            ),
            "l_max",
        ),
    ],
)
def test_simulation_validation_rejects_invalid_configs(
    invalid_config, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_simulation_config(invalid_config(_valid_config()))
