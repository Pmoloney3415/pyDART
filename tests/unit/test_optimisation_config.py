from dataclasses import replace
from pathlib import Path

import pytest

import pydart.config.optimisation_config as optimisation_config_module
from pydart.config.optimisation_config import (
    decreasing_mode_weights,
    load_optimisation_config,
)
from pydart.config.optimisation_validation import validate_optimisation_config
from pydart.config.simulation_config import load_config

CONFIG_DIRECTORY = Path(__file__).parents[2] / "configs" / "optimisations"
SIMULATION_DIRECTORY = Path(__file__).parents[2] / "configs" / "simulations"


@pytest.mark.parametrize(
    "filename",
    [
        "OMEGA60_optimisation.toml",
        "generic_60_beam_design.toml",
        "six_beam_design.toml",
    ],
)
def test_bundled_optimisation_configs_load(filename: str) -> None:
    config = load_optimisation_config(CONFIG_DIRECTORY / filename)

    beam_names = {beam.name for beam in config.simulation.beams}
    assert set(config.variables.frozen_beams) <= beam_names
    assert config.run.simulation_config.is_file()
    assert config.restarts.number > 0
    assert config.objective.mode_weights
    assert all(
        0 <= degree <= config.simulation.metrics.l_max
        for degree, _ in config.objective.mode_weights
    )


def test_decreasing_mode_weights_use_configured_scale_and_power() -> None:
    weights = decreasing_mode_weights(4, l1_mode_weight=2.0, mode_decrease_power=1.0)

    assert weights == ((1, 2.0), (2, 1.0), (3, 2.0 / 3.0), (4, 0.5))


def test_explicit_mode_weights_are_parsed_verbatim(tmp_path: Path, monkeypatch) -> None:
    simulation = load_config(SIMULATION_DIRECTORY / "six_beam_500um.toml")
    data = {
        "optimisation": {
            "index": 1,
            "simulation_config": "simulation.toml",
            "output_directory": "results",
            "maximum_iterations": 10,
        },
        "variables": {
            "frozen_beams": [],
            "power": {
                "enabled": True,
                "minimum_fraction_of_maximum": 0.0,
                "maximum_fraction_of_maximum": 1.0,
            },
            "origin": {"enabled": False, "constraint": "unconstrained"},
            "pointing": {"enabled": False, "constraint": "unconstrained"},
            "spot": {
                "width_enabled": False,
                "force_circular": False,
                "minimum_width_x": 1.0e-4,
                "maximum_width_x": 2.0e-4,
                "minimum_width_y": 1.0e-4,
                "maximum_width_y": 2.0e-4,
                "rotation_enabled": False,
                "minimum_rotation_degrees": 0.0,
                "maximum_rotation_degrees": 180.0,
                "supergaussian_index_enabled": False,
                "minimum_supergaussian_index": 1.0,
                "maximum_supergaussian_index": 2.0,
            },
        },
        "objective": {
            "mode_weight_option": "explicit",
            "mode_weights": {"2": 0.75, "4": 0.125},
        },
    }
    path = tmp_path / "optimisation.toml"
    path.touch()
    monkeypatch.setattr(optimisation_config_module.tomllib, "load", lambda stream: data)
    monkeypatch.setattr(
        optimisation_config_module, "load_config", lambda path: simulation
    )

    config = load_optimisation_config(path)

    assert config.objective.mode_weights == ((2, 0.75), (4, 0.125))


@pytest.mark.parametrize("case", ["unknown_beam", "reversed_power", "disabled"])
def test_optimisation_validation_rejects_invalid_configs(case: str) -> None:
    config = load_optimisation_config(CONFIG_DIRECTORY / "six_beam_design.toml")
    if case == "unknown_beam":
        config = replace(
            config,
            variables=replace(config.variables, frozen_beams=("missing",)),
        )
    elif case == "reversed_power":
        config = replace(
            config,
            variables=replace(
                config.variables,
                power=replace(
                    config.variables.power,
                    minimum_fraction_of_maximum=0.8,
                    maximum_fraction_of_maximum=0.2,
                ),
            ),
        )
    else:
        config = replace(
            config,
            variables=replace(
                config.variables,
                power=replace(config.variables.power, enabled=False),
                origin=replace(config.variables.origin, enabled=False),
                pointing=replace(config.variables.pointing, enabled=False),
                spot=replace(
                    config.variables.spot,
                    width_enabled=False,
                    rotation_enabled=False,
                    supergaussian_index_enabled=False,
                ),
            ),
        )

    with pytest.raises(ValueError):
        validate_optimisation_config(config)
