from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pydart.config.optimisation_config import load_optimisation_config
from pydart.optimisation import OptimisationProblem
from pydart.optimisation.problem import _bounded_surface_offsets

CONFIG_DIRECTORY = Path(__file__).parents[2] / "configs" / "optimisations"


def _six_beam_problem(*, coarse: bool = False) -> OptimisationProblem:
    config = load_optimisation_config(CONFIG_DIRECTORY / "six_beam_design_scipy.toml")
    if coarse:
        simulation = replace(
            config.simulation,
            target=replace(
                config.simulation.target,
                n_polar=12,
                n_azimuthal=24,
            ),
            metrics=replace(config.simulation.metrics, l_max=4),
        )
        config = replace(config, simulation=simulation)
    return OptimisationProblem(config)


def test_layout_is_normalized_and_initial_design_preserves_baseline() -> None:
    problem = _six_beam_problem()

    assert problem.n_parameters > 0
    assert problem.initial_parameters.shape == (problem.n_parameters,)
    assert bool(jnp.all(problem.initial_parameters >= problem.lower_bounds))
    assert bool(jnp.all(problem.initial_parameters <= problem.upper_bounds))
    assert len(problem.parameter_names) == problem.n_parameters

    decoded = problem.beam_parameters(problem.initial_parameters)
    baseline = problem.base_simulation.beams
    np.testing.assert_allclose(
        decoded.physical_origins, baseline.physical_origins, atol=1e-6
    )
    np.testing.assert_allclose(
        decoded.power_fractions_of_maximum,
        baseline.power_fractions_of_maximum,
    )
    np.testing.assert_allclose(decoded.spot_widths, baseline.spot_widths)
    rebuilt = problem.simulation(problem.initial_parameters).beams
    np.testing.assert_allclose(rebuilt.directions, baseline.directions, atol=2e-6)


def test_bounds_decode_to_physical_scalar_bounds() -> None:
    problem = _six_beam_problem()
    low = problem.beam_parameters(problem.lower_bounds)
    high = problem.beam_parameters(problem.upper_bounds)
    variables = problem.config.variables

    np.testing.assert_allclose(
        low.power_fractions_of_maximum,
        variables.power.minimum_fraction_of_maximum,
    )
    np.testing.assert_allclose(
        high.power_fractions_of_maximum,
        variables.power.maximum_fraction_of_maximum,
    )
    np.testing.assert_allclose(
        low.spot_widths,
        np.broadcast_to(
            [variables.spot.minimum_width_x, variables.spot.minimum_width_y],
            low.spot_widths.shape,
        ),
    )
    np.testing.assert_allclose(
        high.spot_widths,
        np.broadcast_to(
            [variables.spot.maximum_width_x, variables.spot.maximum_width_y],
            high.spot_widths.shape,
        ),
    )
    np.testing.assert_allclose(
        low.spot_rotations, np.deg2rad(variables.spot.minimum_rotation_degrees)
    )
    np.testing.assert_allclose(
        high.spot_rotations, np.deg2rad(variables.spot.maximum_rotation_degrees)
    )
    np.testing.assert_allclose(
        low.supergaussian_indices,
        variables.spot.minimum_supergaussian_index,
    )
    np.testing.assert_allclose(
        high.supergaussian_indices,
        variables.spot.maximum_supergaussian_index,
    )


def test_origin_motion_transports_unadjusted_pointing() -> None:
    problem = _six_beam_problem()
    design = problem.initial_parameters
    origin = problem.parameter_blocks[0]
    moved_direction = jnp.asarray([0.2, -0.3, 0.9327379])
    moved_direction = moved_direction / jnp.linalg.norm(moved_direction)
    moved_values = (moved_direction + 1.0) / 2.0
    design = design.at[origin.start : origin.start + 3].set(moved_values)

    decoded = problem.beam_parameters(design)
    origin_direction = decoded.physical_origins[0] / jnp.linalg.norm(
        decoded.physical_origins[0]
    )
    pointing_direction = decoded.pointing_locations[0] / jnp.linalg.norm(
        decoded.pointing_locations[0]
    )

    np.testing.assert_allclose(origin_direction, moved_direction, atol=1e-6)
    np.testing.assert_allclose(pointing_direction, moved_direction, atol=1e-6)


def test_bounded_pointing_uses_smooth_square_to_disk_map() -> None:
    maximum_angle = jnp.deg2rad(45.0)
    reference = jnp.asarray([[1.0, 0.0, 0.0]])

    def angular_offset(values: Sequence[float]) -> float:
        direction = _bounded_surface_offsets(
            reference, jnp.asarray([values]), maximum_angle
        )[0]
        return float(jnp.arccos(jnp.clip(jnp.dot(reference[0], direction), -1.0, 1.0)))

    center = angular_offset([0.5, 0.5])
    edge = angular_offset([1.0, 0.5])
    corner = angular_offset([1.0, 1.0])
    formerly_clipped_interior = angular_offset([0.9, 0.9])

    np.testing.assert_allclose(center, 0.0, atol=1e-7)
    np.testing.assert_allclose(edge, maximum_angle, atol=1e-6)
    np.testing.assert_allclose(corner, maximum_angle, atol=1e-6)
    assert center < formerly_clipped_interior < corner


def test_bounded_pointing_retains_radial_gradient_outside_unit_circle() -> None:
    maximum_angle = jnp.deg2rad(45.0)
    reference = jnp.asarray([[1.0, 0.0, 0.0]])
    values = jnp.asarray([0.9, 0.9])

    jacobian = jax.jacrev(
        lambda point: _bounded_surface_offsets(
            reference, point[None, :], maximum_angle
        )[0]
    )(values)
    radial_change = jacobian @ jnp.asarray([1.0, 1.0])

    assert bool(jnp.all(jnp.isfinite(jacobian)))
    assert float(jnp.linalg.norm(radial_change)) > 1.0e-3


def test_frozen_beam_is_absent_from_design_and_unchanged() -> None:
    config = load_optimisation_config(CONFIG_DIRECTORY / "six_beam_design_scipy.toml")
    unfrozen_problem = OptimisationProblem(config)
    variables = replace(config.variables, frozen_beams=("beam_1",))
    problem = OptimisationProblem(replace(config, variables=variables))

    decoded = problem.beam_parameters(problem.lower_bounds)
    baseline = problem.base_simulation.beams

    assert problem.n_parameters < unfrozen_problem.n_parameters
    assert not any(name.startswith("beam_1.") for name in problem.parameter_names)
    np.testing.assert_allclose(
        decoded.physical_origins[0], baseline.physical_origins[0]
    )
    np.testing.assert_allclose(decoded.spot_widths[0], baseline.spot_widths[0])
    np.testing.assert_allclose(
        decoded.power_fractions_of_maximum[0],
        baseline.power_fractions_of_maximum[0],
    )


def test_objective_components_sum_to_finite_value_with_gradient() -> None:
    problem = _six_beam_problem(coarse=True)
    metrics = problem.metrics(problem.initial_parameters)

    (value, terms), gradient = jax.value_and_grad(
        problem.objective_with_aux, has_aux=True
    )(problem.initial_parameters)

    np.testing.assert_allclose(
        value,
        terms.symmetry_contribution + terms.deposition_contribution,
    )
    expected_rms_ratio_power = (
        terms.rms_nonuniformity / problem.config.objective.acceptable_rms_nonuniformity
    ) ** problem.config.objective.rms_power
    np.testing.assert_allclose(terms.rms_ratio_power, expected_rms_ratio_power)
    np.testing.assert_allclose(
        terms.symmetry_contribution,
        jnp.log1p(expected_rms_ratio_power),
    )
    safe_deposition = (
        terms.deposited_capacity_fraction
        + problem.config.objective.deposition_log_epsilon
    )
    np.testing.assert_allclose(
        terms.deposition_contribution,
        -problem.config.objective.deposition_log_weight * jnp.log(safe_deposition),
    )
    np.testing.assert_allclose(terms.rms_nonuniformity, metrics.rms_nonuniformity)
    np.testing.assert_allclose(
        terms.deposited_capacity_fraction,
        metrics.deposited_power / problem.base_simulation.beams.facility_power,
    )
    assert bool(jnp.isfinite(value))
    assert gradient.shape == (problem.n_parameters,)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0.0


def test_high_harmonic_resolution_is_excluded_from_objective_jit() -> None:
    problem = _six_beam_problem(coarse=True)
    simulation = replace(
        problem.config.simulation,
        metrics=replace(problem.config.simulation.metrics, l_max=20),
    )
    problem = OptimisationProblem(replace(problem.config, simulation=simulation))
    compiled_value_and_gradient = jax.jit(
        jax.value_and_grad(problem.objective_with_aux, has_aux=True)
    )

    (value, terms), gradient = compiled_value_and_gradient(problem.initial_parameters)

    assert bool(jnp.isfinite(value))
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert terms.rms_nonuniformity.shape == ()
