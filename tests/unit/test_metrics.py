from __future__ import annotations

import json

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from pydart.geometry.spherical_mesh import create_spherical_target
from pydart.illumination.deposition import DepositionResult
from pydart.metrics.global_metrics import calculate_metrics


def _axisymmetric_deposition(epsilon: float = 0.0):
    target = create_spherical_target(
        radius=1.0,
        n_azimuthal=128,
        n_polar=64,
    )
    cosine_theta = jnp.cos(target.spherical_coordinates[..., 2])
    power_density = 1.0 + epsilon * cosine_theta
    return target, power_density * target.cell_areas


def test_uniform_deposition_has_only_monopole() -> None:
    target, deposition = _axisymmetric_deposition()

    metrics = calculate_metrics(
        deposition,
        target,
        incident_power=jnp.asarray(4.0 * jnp.pi),
        l_max=4,
    )

    np.testing.assert_allclose(
        metrics.harmonic_coefficients[0, 4],
        np.sqrt(4.0 * np.pi),
        rtol=1e-7,
    )
    np.testing.assert_allclose(metrics.normalized_power_by_l[0], 1.0)
    np.testing.assert_allclose(
        metrics.normalized_power_by_l[1:],
        0.0,
        atol=2e-7,
    )
    np.testing.assert_allclose(metrics.rms_nonuniformity, 0.0, atol=1e-12)


def test_dipole_field_has_expected_l1_coefficient_and_power() -> None:
    epsilon = 0.3
    target, deposition = _axisymmetric_deposition(epsilon)

    metrics = calculate_metrics(
        deposition,
        target,
        incident_power=jnp.asarray(4.0 * jnp.pi),
        l_max=3,
    )

    np.testing.assert_allclose(
        metrics.harmonic_coefficients[1, 3],
        epsilon * np.sqrt(4.0 * np.pi / 3.0),
        rtol=3e-4,
    )
    np.testing.assert_allclose(
        metrics.normalized_power_by_l[1],
        epsilon**2 / 3.0,
        rtol=6e-4,
    )
    np.testing.assert_allclose(
        metrics.rms_nonuniformity,
        epsilon / np.sqrt(3.0),
        rtol=3e-4,
    )


def test_metrics_are_jax_arrays_and_finite() -> None:
    target, deposition = _axisymmetric_deposition(0.1)

    metrics = calculate_metrics(
        deposition,
        target,
        incident_power=jnp.asarray(4.0 * jnp.pi),
        l_max=3,
        simulation_index=7,
    )

    assert all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(metrics))
    assert bool(jnp.all(jnp.isfinite(metrics.harmonic_coefficients)))
    assert bool(jnp.all(jnp.isfinite(metrics.power_by_l)))
    assert metrics.simulation_index == 7


def test_hdf5_and_json_persistence_use_indexed_paths(tmp_path) -> None:
    target, deposition = _axisymmetric_deposition(0.1)
    result = DepositionResult(
        per_beam=deposition[..., None],
        total=deposition,
        unsmoothed_deposited_power_per_beam=jnp.asarray([3.0 * jnp.pi]),
        target=target,
        beams=_plot_test_beams(),
        incident_power=jnp.asarray(4.0 * jnp.pi),
        simulation_index=7,
        l_max=3,
    )
    metrics = result.get_metrics()

    result.save_deposition_data(tmp_path)
    metrics.save_to_directory(tmp_path)

    simulation_directory = tmp_path / "simulation_7"
    h5_path = simulation_directory / "simulation_results_7.h5"
    json_path = simulation_directory / "summary_7.json"
    assert h5_path.is_file()
    assert json_path.is_file()

    with h5py.File(h5_path, "r") as handle:
        assert handle.attrs["surface_quadrature_order"] == 1
        assert handle.attrs["visibility_smoothing_epsilon"] == 0.05
        assert handle["deposition/total_cell_power"].shape == (128, 64)
        assert handle["deposition/per_beam_cell_power"].shape == (128, 64, 1)
        assert handle[
            "deposition/unsmoothed_deposited_power_per_beam"
        ].shape == (1,)
        assert handle["harmonics/coefficients"].shape == (4, 7)
        assert np.iscomplexobj(handle["harmonics/coefficients"][:])
        assert handle["target/spherical_coordinates"].shape == (128, 64, 3)

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert summary["simulation_index"] == 7
    assert summary["l_max"] == 3
    np.testing.assert_allclose(summary["deposited_fraction"], 0.75)
    np.testing.assert_allclose(
        summary["smoothed_deposited_fraction"],
        1.0,
        rtol=2e-4,
    )


def _plot_test_beams():
    from pydart.model.beams import Beams

    vector = jnp.asarray([[1.0, 0.0, 0.0]])
    return Beams(
        origins=vector,
        physical_origins=vector,
        pointing_locations=jnp.zeros((1, 3)),
        directions=-vector,
        basis_x=jnp.asarray([[0.0, -1.0, 0.0]]),
        basis_y=jnp.asarray([[0.0, 0.0, 1.0]]),
        powers=jnp.asarray([4.0 * jnp.pi]),
        maximum_power_fractions=jnp.ones(1),
        power_fractions_of_maximum=jnp.ones(1),
        facility_power=jnp.asarray(4.0 * jnp.pi),
        frequencies=jnp.ones(1),
        spot_widths=jnp.ones((1, 2)),
        supergaussian_indices=jnp.asarray([2.0]),
        spot_rotations=jnp.zeros(1),
        spot_shape_codes=jnp.zeros(1, dtype=jnp.int32),
        numerical_domain_radius=jnp.asarray(10.0),
        names=("beam",),
    )
