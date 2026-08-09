from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pydart.config.simulation_config import (
    BeamConfig,
    LaserConfig,
    PyDARTConfig,
    SimulationConfig,
    SpotConfig,
    TargetConfig,
)
from pydart.geometry.coordinates import (
    cartesian_to_spherical,
    spherical_to_cartesian,
)
from pydart.simulation.simulation import initialise_simulation


def _config() -> PyDARTConfig:
    spot = SpotConfig(
        shape="circular",
        width_x=1.0e-4,
        width_y=1.0e-4,
        supergaussian_index=4.0,
        rotation=90.0,
    )
    beams = (
        BeamConfig(
            name="spherical",
            origin_coordinate_system="spherical",
            pointing_coordinate_system="cartesian",
            origin=np.asarray([2.0, 0.0, np.pi / 2]),
            pointing=np.zeros(3),
            maximum_power_fraction=0.25,
            power_fraction_of_maximum=1.0,
            frequency=1.0,
            spot=spot,
        ),
        BeamConfig(
            name="cartesian",
            origin_coordinate_system="cartesian",
            pointing_coordinate_system="spherical",
            origin=np.asarray([0.0, 2.0, 0.0]),
            pointing=np.asarray([0.0, 0.0, 0.0]),
            maximum_power_fraction=0.75,
            power_fraction_of_maximum=1.0,
            frequency=2.0,
            spot=spot,
        ),
    )
    return PyDARTConfig(
        simulation=SimulationConfig(0, Path("results/test"), False, False),
        target=TargetConfig(radius=0.5, n_polar=4, n_azimuthal=8),
        laser=LaserConfig(n_beams=2, total_incident_power=20.0),
        beams=beams,
    )


def test_coordinate_round_trip_uses_r_phi_theta_order() -> None:
    spherical = jnp.asarray([2.0, -jnp.pi / 2, jnp.pi / 2])
    cartesian = spherical_to_cartesian(spherical)

    np.testing.assert_allclose(cartesian, [0.0, -2.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(cartesian_to_spherical(cartesian), spherical, atol=1e-6)


def test_initialise_simulation_builds_batched_jax_state() -> None:
    simulation = initialise_simulation(_config())

    assert simulation.beams.origins.shape == (2, 3)
    assert simulation.beams.pointing_locations.shape == (2, 3)
    assert simulation.beams.directions.shape == (2, 3)
    assert simulation.beams.basis_x.shape == (2, 3)
    assert simulation.beams.basis_y.shape == (2, 3)
    assert simulation.target.spherical_coordinates.shape == (8, 4, 3)
    assert simulation.target.cartesian_coordinates.shape == (8, 4, 3)
    assert simulation.target.cell_areas.shape == (8, 4)
    assert simulation.surface_quadrature_order == 1
    assert simulation.visibility_smoothing_epsilon == 0.05
    assert all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(simulation))

    np.testing.assert_allclose(
        simulation.beams.directions,
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        atol=1e-6,
    )
    np.testing.assert_allclose(simulation.beams.powers, [5.0, 15.0])
    np.testing.assert_allclose(
        jnp.sum(simulation.beams.basis_x * simulation.beams.directions, axis=-1),
        0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        jnp.sum(simulation.beams.basis_y * simulation.beams.directions, axis=-1),
        0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        jnp.sum(simulation.target.cell_areas),
        4 * jnp.pi * 0.5**2,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        jnp.linalg.norm(simulation.target.surface_normals, axis=-1),
        1.0,
        atol=1e-6,
    )


def test_names_are_static_pytree_metadata() -> None:
    simulation = initialise_simulation(_config())

    assert simulation.beams.names == ("spherical", "cartesian")
    assert all(not isinstance(leaf, str) for leaf in jax.tree.leaves(simulation))


def test_far_origins_are_moved_to_configured_numerical_domain() -> None:
    config = _config()
    far_beam = BeamConfig(
        name="far",
        origin_coordinate_system="cartesian",
        pointing_coordinate_system="cartesian",
        origin=np.asarray([100.0, 0.0, 0.0]),
        pointing=np.zeros(3),
        maximum_power_fraction=1.0,
        power_fraction_of_maximum=1.0,
        frequency=1.0,
        spot=config.beams[0].spot,
    )
    config = PyDARTConfig(
        simulation=config.simulation,
        target=TargetConfig(
            radius=0.5,
            n_polar=4,
            n_azimuthal=8,
            numerical_origin_radius_factor=10.0,
        ),
        laser=LaserConfig(n_beams=1, total_incident_power=20.0),
        beams=(far_beam,),
    )

    simulation = initialise_simulation(config)

    np.testing.assert_allclose(simulation.beams.origins, [[5.0, 0.0, 0.0]])


@pytest.mark.parametrize(
    ("origin", "pointing", "message"),
    [
        ([2.0, 0.0, 0.0], [2.0, 0.0, 0.0], "distinct"),
        ([100.0, 0.0, 0.0], [100.0, 1.0, 0.0], "do not intersect"),
        ([100.0, 0.0, 0.0], [101.0, 0.0, 0.0], "point away"),
    ],
)
def test_invalid_beam_geometry_is_rejected(origin, pointing, message: str) -> None:
    config = _config()
    beam = replace(
        config.beams[0],
        origin_coordinate_system="cartesian",
        pointing_coordinate_system="cartesian",
        origin=np.asarray(origin),
        pointing=np.asarray(pointing),
        maximum_power_fraction=1.0,
    )
    config = replace(
        config,
        laser=replace(config.laser, n_beams=1),
        beams=(beam,),
    )

    with pytest.raises(ValueError, match=message):
        initialise_simulation(config)


def test_reapplying_extracted_parameters_preserves_beam_state() -> None:
    simulation = initialise_simulation(_config())

    rebuilt = simulation.with_beam_parameters(simulation.beam_parameters())

    for original, updated in zip(
        jax.tree.leaves(simulation.beams),
        jax.tree.leaves(rebuilt.beams),
        strict=True,
    ):
        np.testing.assert_allclose(updated, original, atol=1e-6)
