from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pydart.config.simulation_config import (
    BeamConfig,
    LaserConfig,
    MetricsConfig,
    PyDARTConfig,
    SimulationConfig,
    SpotConfig,
    TargetConfig,
)
from pydart.simulation.simulation import initialise_simulation


def _two_beam_simulation():
    beams = (
        BeamConfig(
            name="beam_a",
            origin_coordinate_system="cartesian",
            pointing_coordinate_system="cartesian",
            origin=np.asarray([3.0, 0.4, 0.7]),
            pointing=np.asarray([0.02, -0.03, 0.01]),
            maximum_power_fraction=0.6,
            power_fraction_of_maximum=0.9,
            frequency=1.0,
            spot=SpotConfig("elliptical", 0.30, 0.22, 2.7, 17.0),
        ),
        BeamConfig(
            name="beam_b",
            origin_coordinate_system="cartesian",
            pointing_coordinate_system="cartesian",
            origin=np.asarray([-0.6, 2.8, -0.5]),
            pointing=np.asarray([-0.04, 0.02, -0.015]),
            maximum_power_fraction=0.4,
            power_fraction_of_maximum=0.8,
            frequency=1.0,
            spot=SpotConfig("elliptical", 0.26, 0.24, 3.4, 61.0),
        ),
    )
    config = PyDARTConfig(
        simulation=SimulationConfig(0, Path("results/test"), False, False),
        target=TargetConfig(
            radius=0.5,
            n_polar=32,
            n_azimuthal=64,
            numerical_origin_radius_factor=5.0,
        ),
        laser=LaserConfig(n_beams=2, total_incident_power=20.0),
        beams=beams,
        metrics=MetricsConfig(l_max=4),
    )
    return initialise_simulation(config)


def _parameter_vector(simulation):
    parameters = simulation.beam_parameters()
    return jnp.asarray(
        [
            parameters.power_fractions_of_maximum[0],
            parameters.pointing_locations[0, 1],
            parameters.spot_widths[0, 0],
            parameters.spot_widths[1, 1],
            parameters.spot_rotations[0],
            parameters.supergaussian_indices[1],
        ]
    )


def _metric_function(simulation):
    baseline = simulation.beam_parameters()

    def metrics_from_vector(values):
        parameters = replace(
            baseline,
            power_fractions_of_maximum=(
                baseline.power_fractions_of_maximum.at[0].set(values[0])
            ),
            pointing_locations=baseline.pointing_locations.at[0, 1].set(values[1]),
            spot_widths=(
                baseline.spot_widths.at[0, 0].set(values[2]).at[1, 1].set(values[3])
            ),
            spot_rotations=baseline.spot_rotations.at[0].set(values[4]),
            supergaussian_indices=(baseline.supergaussian_indices.at[1].set(values[5])),
        )
        metrics = simulation.with_beam_parameters(parameters).run().get_metrics()
        return jnp.stack(
            (
                metrics.deposited_fraction,
                metrics.rms_nonuniformity,
                metrics.normalized_power_by_l[1],
                metrics.normalized_power_by_l[2],
                metrics.normalized_power_by_l[4],
            )
        )

    return jax.jit(metrics_from_vector)


def _central_difference(function, values, steps):
    columns = []
    for index in range(values.size):
        offset = jnp.zeros_like(values).at[index].set(steps[index])
        columns.append(
            (function(values + offset) - function(values - offset))
            / (2.0 * steps[index])
        )
    return jnp.stack(columns, axis=1)


def test_key_metric_jacobian_matches_central_finite_differences() -> None:
    simulation = _two_beam_simulation()
    values = _parameter_vector(simulation)
    function = _metric_function(simulation)
    automatic = jax.jacrev(function)(values)
    base_steps = 1.0e-4 * jnp.maximum(jnp.abs(values), 1.0)

    finite_difference_jacobians = [
        _central_difference(function, values, scale * base_steps)
        for scale in (1.0, 0.5, 0.25)
    ]

    assert automatic.shape == (5, 6)
    assert bool(jnp.all(jnp.isfinite(automatic)))
    assert bool(jnp.all(jnp.max(jnp.abs(automatic), axis=0) > 1.0e-6))
    assert bool(jnp.all(jnp.max(jnp.abs(automatic), axis=1) > 1.0e-6))
    for finite_difference in finite_difference_jacobians:
        assert bool(jnp.all(jnp.isfinite(finite_difference)))
        np.testing.assert_allclose(
            automatic,
            finite_difference,
            rtol=3.0e-3,
            atol=2.0e-7,
        )
