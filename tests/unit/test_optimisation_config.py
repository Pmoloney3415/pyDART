from dataclasses import replace
from pathlib import Path

import pytest

from pydart.config.optimisation_config import load_optimisation_config
from pydart.config.optimisation_validation import validate_optimisation_config

CONFIG_DIRECTORY = Path(__file__).parents[2] / "configs" / "optimisations"


@pytest.mark.parametrize(
    "filename",
    [
        "OMEGA60_optimisation.toml",
        "generic_60_beam_design.toml",
        "six_beam_design_scipy.toml",
    ],
)
def test_bundled_optimisation_configs_load(filename: str) -> None:
    config = load_optimisation_config(CONFIG_DIRECTORY / filename)

    beam_names = {beam.name for beam in config.simulation.beams}
    assert set(config.variables.frozen_beams) <= beam_names
    assert config.run.simulation_config.is_file()
    assert config.restarts.number > 0
    assert config.run.solver == "scipy_lbfgsb"
    assert config.run.device_iteration_chunk_size == 10
    assert config.objective.deposition_log_weight > 0.0
    assert config.objective.deposition_log_epsilon == 1.0e-8
    assert config.objective.acceptable_rms_nonuniformity == 0.01
    assert config.objective.rms_power == 2.0


@pytest.mark.parametrize(
    ("filename", "solver", "index", "chunk_size", "output_interval"),
    [
        (
            "generic_48_beam_design.toml",
            "jaxopt_lbfgsb",
            48,
            1000,
            5000,
        ),
        (
            "generic_48_beam_design_cpu.toml",
            "scipy_lbfgsb",
            148,
            10,
            500,
        ),
    ],
)
def test_48_beam_optimisation_configs(
    filename: str,
    solver: str,
    index: int,
    chunk_size: int,
    output_interval: int,
) -> None:
    config = load_optimisation_config(CONFIG_DIRECTORY / filename)

    assert config.run.solver == solver
    assert config.run.index == index
    assert config.run.device_iteration_chunk_size == chunk_size
    assert config.run.checkpoint_interval == output_interval
    assert config.run.history_plot_interval == output_interval
    assert config.simulation.laser.n_beams == 48
    assert config.variables.power.enabled
    assert config.variables.origin.enabled
    assert config.variables.pointing.enabled
    assert config.variables.spot.width_enabled
    assert config.variables.spot.rotation_enabled
    assert config.variables.spot.supergaussian_index_enabled
    assert not config.variables.frozen_beams


@pytest.mark.parametrize(
    "case",
    [
        "unknown_beam",
        "reversed_power",
        "disabled",
        "unknown_solver",
        "invalid_chunk_size",
    ],
)
def test_optimisation_validation_rejects_invalid_configs(case: str) -> None:
    config = load_optimisation_config(CONFIG_DIRECTORY / "six_beam_design_scipy.toml")
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
    elif case == "disabled":
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
    elif case == "unknown_solver":
        config = replace(config, run=replace(config.run, solver="not_a_solver"))
    else:
        config = replace(
            config,
            run=replace(config.run, device_iteration_chunk_size=0),
        )

    with pytest.raises(ValueError):
        validate_optimisation_config(config)
