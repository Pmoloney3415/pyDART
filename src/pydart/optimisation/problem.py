"""Differentiable parameterization of a configured illumination design."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from pydart.config.optimisation_config import OptimisationConfig
from pydart.model.parameters import BeamParameters
from pydart.simulation.simulation import Simulation, initialise_simulation


@dataclass(frozen=True)
class ParameterBlock:
    """Description of one contiguous block in the normalized design vector."""

    name: str
    start: int
    stop: int
    beam_indices: tuple[int, ...]
    components_per_beam: int


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ObjectiveTerms:
    """JAX-compatible objective components and physical diagnostics."""

    symmetry_contribution: Array
    rms_ratio_power: Array
    deposition_contribution: Array
    rms_nonuniformity: Array
    deposited_capacity_fraction: Array

    def tree_flatten(self):
        return (
            self.symmetry_contribution,
            self.rms_ratio_power,
            self.deposition_contribution,
            self.rms_nonuniformity,
            self.deposited_capacity_fraction,
        ), None

    @classmethod
    def tree_unflatten(cls, auxiliary_data, children):
        del auxiliary_data
        return cls(*children)


class OptimisationProblem:
    """Map a bounded, normalized vector onto a differentiable simulation.

    All exposed design variables live in ``[0, 1]``. Physical scalar bounds
    are applied linearly. Surface locations use fixed-radius directions:
    bounded motion uses a two-component tangent displacement, while
    unrestricted motion uses a redundant normalized Cartesian vector.
    """

    def __init__(self, config: OptimisationConfig):
        self.config = config
        self.base_simulation = initialise_simulation(config.simulation)
        self._baseline = self.base_simulation.beam_parameters()
        self._active_indices = tuple(
            index
            for index, name in enumerate(self.base_simulation.beams.names)
            if name not in config.variables.frozen_beams
        )
        self._origin_radii = jnp.linalg.norm(self._baseline.physical_origins, axis=-1)
        self._origin_directions = (
            self._baseline.physical_origins / self._origin_radii[:, None]
        )
        self._nominal_intersections = _ray_sphere_intersections(
            self._baseline.physical_origins,
            self.base_simulation.beams.directions,
            self.base_simulation.target.radius,
        )
        self._nominal_intersection_directions = (
            self._nominal_intersections / self.base_simulation.target.radius
        )
        self._blocks, initial = self._build_layout()
        self.initial_parameters = jnp.asarray(initial)
        if bool(
            jnp.any((self.initial_parameters < 0.0) | (self.initial_parameters > 1.0))
        ):
            raise ValueError(
                "The base simulation contains an enabled variable outside its "
                "optimization bounds."
            )
        self.lower_bounds = jnp.zeros_like(self.initial_parameters)
        self.upper_bounds = jnp.ones_like(self.initial_parameters)

    @property
    def n_parameters(self) -> int:
        return int(self.initial_parameters.size)

    @property
    def parameter_blocks(self) -> tuple[ParameterBlock, ...]:
        return self._blocks

    @property
    def parameter_names(self) -> tuple[str, ...]:
        names: list[str] = []
        component_names = {
            "origin": ("x", "y", "z"),
            "origin_offset": ("tangent_x", "tangent_y"),
            "pointing": ("x", "y", "z"),
            "pointing_offset": ("tangent_x", "tangent_y"),
            "spot_width": ("width",),
            "spot_width_xy": ("width_x", "width_y"),
            "power": ("fraction_of_maximum",),
            "rotation": ("rotation",),
            "supergaussian_index": ("index",),
        }
        beam_names = self.base_simulation.beams.names
        for block in self._blocks:
            components = component_names[block.name]
            for beam_index in block.beam_indices:
                names.extend(
                    f"{beam_names[beam_index]}.{component}" for component in components
                )
        return tuple(names)

    def beam_parameters(self, design: Array) -> BeamParameters:
        """Decode normalized design variables into physical beam parameters."""
        design = jnp.asarray(design)
        baseline = self._baseline
        origins = baseline.physical_origins

        origin_block = self._find_block("origin", "origin_offset")
        if origin_block is not None:
            values = self._block_values(design, origin_block)
            indices = jnp.asarray(origin_block.beam_indices)
            reference = self._origin_directions[indices]
            if origin_block.name == "origin":
                directions = _unit_vectors_from_cube(values, reference)
            else:
                angle = jnp.deg2rad(
                    self.config.variables.origin.maximum_angular_displacement_degrees
                )
                directions = _bounded_surface_offsets(reference, values, angle)
            origins = origins.at[indices].set(
                self._origin_radii[indices, None] * directions
            )

        transported = _transport_directions(
            self._origin_directions,
            origins / self._origin_radii[:, None],
            self._nominal_intersection_directions,
        )
        pointing_directions = transported
        pointing_block = self._find_block("pointing", "pointing_offset")
        if pointing_block is not None:
            values = self._block_values(design, pointing_block)
            indices = jnp.asarray(pointing_block.beam_indices)
            if pointing_block.name == "pointing":
                absolute = _unit_vectors_from_cube(
                    values,
                    self._nominal_intersection_directions[indices],
                )
                adjusted = _transport_directions(
                    self._nominal_intersection_directions[indices],
                    absolute,
                    transported[indices],
                )
            else:
                angle = jnp.deg2rad(
                    self.config.variables.pointing.maximum_angular_displacement_degrees
                )
                adjusted = _bounded_surface_offsets(transported[indices], values, angle)
            pointing_directions = pointing_directions.at[indices].set(adjusted)
        pointing = self.base_simulation.target.radius * pointing_directions

        power = baseline.power_fractions_of_maximum
        widths = baseline.spot_widths
        rotations = baseline.spot_rotations
        indices_sg = baseline.supergaussian_indices
        active = jnp.asarray(self._active_indices)
        variables = self.config.variables

        block = self._find_block("power")
        if block is not None:
            values = self._block_values(design, block)[:, 0]
            power = power.at[active].set(
                _linear(
                    values,
                    variables.power.minimum_fraction_of_maximum,
                    variables.power.maximum_fraction_of_maximum,
                )
            )
        block = self._find_block("spot_width", "spot_width_xy")
        if block is not None:
            values = self._block_values(design, block)
            spot = variables.spot
            width_x = _linear(values[:, 0], spot.minimum_width_x, spot.maximum_width_x)
            if spot.force_circular:
                selected_widths = jnp.stack((width_x, width_x), axis=-1)
            else:
                width_y = _linear(
                    values[:, 1], spot.minimum_width_y, spot.maximum_width_y
                )
                selected_widths = jnp.stack((width_x, width_y), axis=-1)
            widths = widths.at[active].set(selected_widths)
        block = self._find_block("rotation")
        if block is not None:
            values = self._block_values(design, block)[:, 0]
            spot = variables.spot
            rotations = rotations.at[active].set(
                jnp.deg2rad(
                    _linear(
                        values,
                        spot.minimum_rotation_degrees,
                        spot.maximum_rotation_degrees,
                    )
                )
            )
        block = self._find_block("supergaussian_index")
        if block is not None:
            values = self._block_values(design, block)[:, 0]
            spot = variables.spot
            indices_sg = indices_sg.at[active].set(
                _linear(
                    values,
                    spot.minimum_supergaussian_index,
                    spot.maximum_supergaussian_index,
                )
            )

        return BeamParameters(
            physical_origins=origins,
            pointing_locations=pointing,
            power_fractions_of_maximum=power,
            spot_widths=widths,
            supergaussian_indices=indices_sg,
            spot_rotations=rotations,
        )

    def simulation(self, design: Array) -> Simulation:
        """Build the forward simulation represented by ``design``."""
        return self.base_simulation.with_beam_parameters(self.beam_parameters(design))

    def metrics(self, design: Array):
        """Run the forward model and return differentiable metrics."""
        return self.simulation(design).run().get_metrics()

    def objective(self, design: Array) -> Array:
        """Return the configured dimensionless scalar loss."""
        return self.objective_with_aux(design)[0]

    def objective_with_aux(self, design: Array) -> tuple[Array, ObjectiveTerms]:
        r"""Return ``-w log(D) + log(1 + (R/R0)**p)`` and diagnostics."""
        deposition = self.simulation(design).run()
        cell_areas = deposition.target.cell_areas
        total_area = jnp.sum(cell_areas)
        smoothed_deposited_power = jnp.sum(deposition.total)
        mean_power_density = smoothed_deposited_power / total_area
        power_density = deposition.total / cell_areas
        variance = (
            jnp.sum((power_density - mean_power_density) ** 2 * cell_areas) / total_area
        )
        rms_nonuniformity = jnp.sqrt(variance) / mean_power_density
        deposited_capacity_fraction = (
            jnp.sum(deposition.unsmoothed_deposited_power_per_beam)
            / self.base_simulation.beams.facility_power
        )
        objective = self.config.objective
        rms_ratio_power = (
            rms_nonuniformity / objective.acceptable_rms_nonuniformity
        ) ** objective.rms_power
        symmetry_contribution = jnp.log1p(rms_ratio_power)
        safe_deposited_fraction = (
            deposited_capacity_fraction + objective.deposition_log_epsilon
        )
        deposition_contribution = -objective.deposition_log_weight * jnp.log(
            safe_deposited_fraction
        )
        loss = symmetry_contribution + deposition_contribution
        return loss, ObjectiveTerms(
            symmetry_contribution=symmetry_contribution,
            rms_ratio_power=rms_ratio_power,
            deposition_contribution=deposition_contribution,
            rms_nonuniformity=rms_nonuniformity,
            deposited_capacity_fraction=deposited_capacity_fraction,
        )

    def value_and_gradient(self, design: Array) -> tuple[Array, Array]:
        """Evaluate the objective and its reverse-mode JAX gradient."""
        return jax.value_and_grad(self.objective)(design)

    def clipped(self, design: Array) -> Array:
        """Project a candidate vector onto the explicit box constraints."""
        return jnp.clip(design, self.lower_bounds, self.upper_bounds)

    def _find_block(self, *names: str) -> ParameterBlock | None:
        return next((block for block in self._blocks if block.name in names), None)

    @staticmethod
    def _block_values(design: Array, block: ParameterBlock) -> Array:
        return design[block.start : block.stop].reshape(
            len(block.beam_indices), block.components_per_beam
        )

    def _build_layout(self) -> tuple[tuple[ParameterBlock, ...], list[float]]:
        blocks: list[ParameterBlock] = []
        initial: list[float] = []
        active = self._active_indices

        def add(name: str, values: Array) -> None:
            flat = jnp.asarray(values).reshape(-1).tolist()
            start = len(initial)
            initial.extend(float(value) for value in flat)
            blocks.append(
                ParameterBlock(
                    name,
                    start,
                    len(initial),
                    active,
                    len(flat) // len(active),
                )
            )

        variables = self.config.variables
        active_array = jnp.asarray(active)
        if variables.origin.enabled:
            if variables.origin.constraint == "unconstrained":
                add("origin", (self._origin_directions[active_array] + 1.0) / 2.0)
            else:
                add("origin_offset", jnp.full((len(active), 2), 0.5))
        if variables.pointing.enabled:
            if variables.pointing.constraint == "unconstrained":
                add(
                    "pointing",
                    (self._nominal_intersection_directions[active_array] + 1.0) / 2.0,
                )
            else:
                add("pointing_offset", jnp.full((len(active), 2), 0.5))
        if variables.power.enabled:
            power = variables.power
            add(
                "power",
                _inverse_linear(
                    self._baseline.power_fractions_of_maximum[active_array],
                    power.minimum_fraction_of_maximum,
                    power.maximum_fraction_of_maximum,
                )[:, None],
            )
        spot = variables.spot
        if spot.width_enabled:
            widths = self._baseline.spot_widths[active_array]
            x = _inverse_linear(
                widths[:, 0], spot.minimum_width_x, spot.maximum_width_x
            )
            if spot.force_circular:
                add("spot_width", x[:, None])
            else:
                y = _inverse_linear(
                    widths[:, 1], spot.minimum_width_y, spot.maximum_width_y
                )
                add("spot_width_xy", jnp.stack((x, y), axis=-1))
        if spot.rotation_enabled:
            degrees = jnp.rad2deg(self._baseline.spot_rotations[active_array])
            add(
                "rotation",
                _inverse_linear(
                    degrees,
                    spot.minimum_rotation_degrees,
                    spot.maximum_rotation_degrees,
                )[:, None],
            )
        if spot.supergaussian_index_enabled:
            add(
                "supergaussian_index",
                _inverse_linear(
                    self._baseline.supergaussian_indices[active_array],
                    spot.minimum_supergaussian_index,
                    spot.maximum_supergaussian_index,
                )[:, None],
            )
        return tuple(blocks), initial


def _linear(values: Array, minimum: float, maximum: float) -> Array:
    return minimum + values * (maximum - minimum)


def _inverse_linear(values: Array, minimum: float, maximum: float) -> Array:
    return (values - minimum) / (maximum - minimum)


def _unit_vectors_from_cube(values: Array, fallback: Array) -> Array:
    vectors = 2.0 * values - 1.0
    norms = jnp.linalg.norm(vectors, axis=-1, keepdims=True)
    normalized = vectors / jnp.maximum(norms, 1.0e-12)
    return jnp.where(norms > 1.0e-12, normalized, fallback)


def _surface_basis(directions: Array) -> tuple[Array, Array]:
    z = jnp.asarray([0.0, 0.0, 1.0], dtype=directions.dtype)
    y = jnp.asarray([0.0, 1.0, 0.0], dtype=directions.dtype)
    projected_z = z - directions[:, 2:3] * directions
    projected_y = y - directions[:, 1:2] * directions
    use_y = jnp.linalg.norm(projected_z, axis=-1, keepdims=True) < 1.0e-8
    first = jnp.where(use_y, projected_y, projected_z)
    first = first / jnp.linalg.norm(first, axis=-1, keepdims=True)
    second = jnp.cross(directions, first)
    return first, second


def _bounded_surface_offsets(
    reference: Array, values: Array, maximum_angle: Array
) -> Array:
    first, second = _surface_basis(reference)
    square = 2.0 * values - 1.0
    x = square[:, 0]
    y = square[:, 1]
    disk = jnp.stack(
        (
            x * jnp.sqrt(1.0 - 0.5 * y**2),
            y * jnp.sqrt(1.0 - 0.5 * x**2),
        ),
        axis=-1,
    )
    tangent_components = disk * maximum_angle
    angle = jnp.sqrt(jnp.sum(tangent_components**2, axis=-1) + 1.0e-30)
    tangent = tangent_components[:, 0:1] * first + tangent_components[:, 1:2] * second
    return (
        jnp.cos(angle)[:, None] * reference
        + jnp.sinc(angle / jnp.pi)[:, None] * tangent
    )


def _transport_directions(source: Array, destination: Array, vectors: Array) -> Array:
    cross = jnp.cross(source, destination)
    cosine = jnp.sum(source * destination, axis=-1, keepdims=True)
    cross_vector = jnp.cross(cross, vectors)
    twice_crossed = jnp.cross(cross, cross_vector)
    regular = (
        vectors + cross_vector + twice_crossed / jnp.maximum(1.0 + cosine, 1.0e-12)
    )
    first, _ = _surface_basis(source)
    antipodal = 2.0 * jnp.sum(first * vectors, axis=-1, keepdims=True) * first - vectors
    return jnp.where(cosine < -1.0 + 1.0e-7, antipodal, regular)


def _ray_sphere_intersections(
    origins: Array, directions: Array, radius: Array
) -> Array:
    along = jnp.sum(origins * directions, axis=-1)
    discriminant = along**2 - (jnp.sum(origins**2, axis=-1) - radius**2)
    distance = -along - jnp.sqrt(jnp.maximum(discriminant, 0.0))
    return origins + distance[:, None] * directions
