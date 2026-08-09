from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pydart.config.simulation_config import (
    BeamConfig,
    LaserConfig,
    PyDARTConfig,
    SimulationConfig,
    SpotConfig,
    TargetConfig,
)
from pydart.geometry.projection import project_to_beam_planes
from pydart.geometry.spherical_mesh import create_spherical_cell_quadrature
from pydart.illumination.deposition import calculate_deposition
from pydart.illumination.profiles import supergaussian_intensity
from pydart.illumination.visibility import (
    projected_cosines,
    projected_cosines_from_normals,
)
from pydart.simulation.simulation import initialise_simulation


def _single_beam_config(
    *,
    rotation: float = 0.0,
    n_azimuthal: int = 128,
    n_polar: int = 64,
) -> PyDARTConfig:
    beam = BeamConfig(
        name="beam",
        origin_coordinate_system="cartesian",
        pointing_coordinate_system="cartesian",
        origin=np.asarray([100.0, 0.0, 0.0]),
        pointing=np.zeros(3),
        maximum_power_fraction=1.0,
        power_fraction_of_maximum=1.0,
        frequency=1.0,
        spot=SpotConfig(
            shape="elliptical",
            width_x=0.1,
            width_y=0.2,
            supergaussian_index=2.0,
            rotation=rotation,
        ),
    )
    return PyDARTConfig(
        simulation=SimulationConfig(0, Path("results/test"), False, False),
        target=TargetConfig(
            radius=1.0,
            n_polar=n_polar,
            n_azimuthal=n_azimuthal,
            numerical_origin_radius_factor=10.0,
        ),
        laser=LaserConfig(n_beams=1, total_incident_power=10.0),
        beams=(beam,),
    )


def test_supergaussian_uses_requested_e_to_minus_rho_power_m_form() -> None:
    simulation = initialise_simulation(_single_beam_config())
    beam = simulation.beams
    peak = beam.powers[0] / (jnp.pi * 0.1 * 0.2)

    values = supergaussian_intensity(jnp.asarray([[[[0.0, 0.0]]]]), beam)
    edge = supergaussian_intensity(
        jnp.asarray([[[[0.1, 0.0]]]]),
        beam,
    )

    np.testing.assert_allclose(values.squeeze(), peak, rtol=1e-6)
    np.testing.assert_allclose(edge.squeeze(), peak / np.e, rtol=1e-6)


def test_projection_uses_z_aligned_zero_rotation_axis() -> None:
    simulation = initialise_simulation(_single_beam_config(rotation=0.0))
    point = jnp.asarray([[0.0, 0.0, 0.25]])

    projected, local = project_to_beam_planes(point, simulation.beams)

    np.testing.assert_allclose(projected, [[[10.0, 0.0, 0.25]]], atol=1e-6)
    np.testing.assert_allclose(local, [[[0.0, 0.25]]], atol=1e-6)


def test_visibility_is_positive_only_on_incident_hemisphere() -> None:
    simulation = initialise_simulation(_single_beam_config())
    cosines = projected_cosines(simulation.target, simulation.beams)[..., 0]
    x = simulation.target.cartesian_coordinates[..., 0]

    assert bool(jnp.all(cosines[x > 0] > 0))
    assert bool(jnp.all(cosines[x < 0] == 0))


def test_smooth_visibility_regularizes_the_grazing_incidence_boundary() -> None:
    simulation = initialise_simulation(_single_beam_config())
    normals = jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])

    factors = projected_cosines_from_normals(
        normals,
        simulation.beams,
        smoothing_epsilon=0.05,
    )[:, 0]

    expected = 0.5 * (
        jnp.asarray([1.0, 0.0, -1.0])
        + jnp.sqrt(jnp.asarray([1.0, 0.0, -1.0]) ** 2 + 0.05**2)
    )
    np.testing.assert_allclose(factors, expected, rtol=1e-6)
    np.testing.assert_allclose(factors[1], 0.025, rtol=1e-6)
    assert bool(factors[2] > 0.0)


def test_deposition_has_expected_shapes_and_is_jittable() -> None:
    simulation = initialise_simulation(_single_beam_config())

    result = jax.jit(calculate_deposition)(simulation.target, simulation.beams)

    assert result.per_beam.shape == (128, 64, 1)
    assert result.total.shape == (128, 64)
    np.testing.assert_allclose(result.total, result.per_beam[..., 0])
    assert bool(jnp.all(result.total >= 0.0))


def test_total_deposition_is_sum_of_per_beam_contributions() -> None:
    config = _single_beam_config(n_azimuthal=16, n_polar=8)
    first = replace(config.beams[0], maximum_power_fraction=0.5)
    second = replace(
        first,
        name="beam_2",
        origin=np.asarray([-100.0, 0.0, 0.0]),
    )
    config = replace(
        config,
        laser=replace(config.laser, n_beams=2),
        beams=(first, second),
    )

    result = initialise_simulation(config).run()

    assert result.per_beam.shape[-1] == 2
    np.testing.assert_allclose(result.total, jnp.sum(result.per_beam, axis=-1))


def test_subcell_quadrature_has_equal_exact_area_weights() -> None:
    target = initialise_simulation(_single_beam_config(n_azimuthal=8, n_polar=4)).target

    points, normals, areas = create_spherical_cell_quadrature(target, order=3)

    assert points.shape == (8, 4, 3, 3, 3)
    assert normals.shape == points.shape
    assert areas.shape == (8, 4, 3, 3)
    np.testing.assert_allclose(
        jnp.sum(areas, axis=(-2, -1)),
        target.cell_areas,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        jnp.linalg.norm(normals, axis=-1),
        1.0,
        atol=1e-6,
    )


def test_deposition_uses_configured_subcell_quadrature_order() -> None:
    config = _single_beam_config(n_azimuthal=7, n_polar=5)
    config = replace(
        config,
        simulation=replace(config.simulation, surface_quadrature_order=3),
    )
    simulation = initialise_simulation(config)

    centre_sampled = calculate_deposition(
        simulation.target,
        simulation.beams,
        surface_quadrature_order=1,
    )
    subcell_sampled = simulation.run()

    assert subcell_sampled.surface_quadrature_order == 3
    assert not bool(jnp.allclose(subcell_sampled.total, centre_sampled.total))


def test_large_sphere_intercepts_normalized_narrow_beam_power() -> None:
    simulation = initialise_simulation(
        _single_beam_config(n_azimuthal=512, n_polar=256)
    )

    result = calculate_deposition(simulation.target, simulation.beams)

    np.testing.assert_allclose(jnp.sum(result.total), 10.0, rtol=3e-3)
