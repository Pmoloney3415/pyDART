"""Simulation state construction and execution."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from pydart.config.simulation_config import PyDARTConfig
from pydart.geometry.spherical_mesh import create_spherical_target
from pydart.model.beams import Beams
from pydart.model.target import Target


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Simulation:
    """Initialised numerical state ready for the forward model."""

    target: Target
    beams: Beams
    simulation_index: int = 0
    l_max: int = 20
    surface_quadrature_order: int = 1
    visibility_smoothing_epsilon: float = 0.05

    def tree_flatten(self):
        return (self.target, self.beams), (
            self.simulation_index,
            self.l_max,
            self.surface_quadrature_order,
            self.visibility_smoothing_epsilon,
        )

    @classmethod
    def tree_unflatten(cls, auxiliary_data, children):
        (
            simulation_index,
            l_max,
            surface_quadrature_order,
            visibility_smoothing_epsilon,
        ) = auxiliary_data
        return cls(
            *children,
            simulation_index=simulation_index,
            l_max=l_max,
            surface_quadrature_order=surface_quadrature_order,
            visibility_smoothing_epsilon=visibility_smoothing_epsilon,
        )

    def run(self):
        """Run the solid-sphere illumination model."""
        from pydart.illumination.simulation import simulate_illumination

        return simulate_illumination(self)

    def beam_parameters(self):
        """Return the continuously variable beam-parameter PyTree."""
        from pydart.model.parameters import parameters_from_beams

        return parameters_from_beams(self.beams)

    def with_beam_parameters(self, parameters):
        """Return a new simulation with differentiably rebuilt beam state."""
        from dataclasses import replace

        from pydart.model.parameters import apply_beam_parameters

        return replace(
            self,
            beams=apply_beam_parameters(self.beams, parameters),
        )


def initialise_simulation(config: PyDARTConfig) -> Simulation:
    """Convert a validated configuration into batched numerical structures."""
    target = create_spherical_target(
        radius=config.target.radius,
        n_azimuthal=config.target.n_azimuthal,
        n_polar=config.target.n_polar,
    )
    beams = _initialise_beams(config)
    return Simulation(
        target=target,
        beams=beams,
        simulation_index=config.simulation.index,
        l_max=config.metrics.l_max,
        surface_quadrature_order=config.simulation.surface_quadrature_order,
        visibility_smoothing_epsilon=(config.simulation.visibility_smoothing_epsilon),
    )


def _initialise_beams(config: PyDARTConfig) -> Beams:
    physical_origins = np.stack(
        [
            _to_cartesian(beam.origin, beam.origin_coordinate_system)
            for beam in config.beams
        ]
    )
    pointing_locations_np = np.stack(
        [
            _to_cartesian(beam.pointing, beam.pointing_coordinate_system)
            for beam in config.beams
        ]
    )
    direction_vectors = pointing_locations_np - physical_origins
    direction_norms = np.linalg.norm(direction_vectors, axis=-1, keepdims=True)
    if np.any(direction_norms == 0):
        raise ValueError("Beam origin and pointing location must be distinct.")
    directions = direction_vectors / direction_norms
    origins = _move_origins_to_numerical_domain(
        physical_origins,
        directions,
        config.target.radius * config.target.numerical_origin_radius_factor,
    )
    basis_x, basis_y = _create_local_beam_frames(
        directions,
        np.deg2rad([beam.spot.rotation for beam in config.beams]),
    )

    maximum_power_fractions = jnp.asarray(
        [beam.maximum_power_fraction for beam in config.beams]
    )
    power_fractions_of_maximum = jnp.asarray(
        [beam.power_fraction_of_maximum for beam in config.beams]
    )
    powers = (
        config.laser.total_incident_power
        * maximum_power_fractions
        * power_fractions_of_maximum
    )
    spot_shape_codes = jnp.asarray(
        [0 if beam.spot.shape == "circular" else 1 for beam in config.beams],
        dtype=jnp.int32,
    )

    return Beams(
        origins=jnp.asarray(origins),
        physical_origins=jnp.asarray(physical_origins),
        pointing_locations=jnp.asarray(pointing_locations_np),
        directions=jnp.asarray(directions),
        basis_x=jnp.asarray(basis_x),
        basis_y=jnp.asarray(basis_y),
        powers=powers,
        maximum_power_fractions=maximum_power_fractions,
        power_fractions_of_maximum=power_fractions_of_maximum,
        facility_power=jnp.asarray(config.laser.total_incident_power),
        frequencies=jnp.asarray([beam.frequency for beam in config.beams]),
        spot_widths=jnp.asarray(
            [[beam.spot.width_x, beam.spot.width_y] for beam in config.beams]
        ),
        supergaussian_indices=jnp.asarray(
            [beam.spot.supergaussian_index for beam in config.beams]
        ),
        spot_rotations=jnp.deg2rad(
            jnp.asarray([beam.spot.rotation for beam in config.beams])
        ),
        spot_shape_codes=spot_shape_codes,
        numerical_domain_radius=jnp.asarray(
            config.target.radius * config.target.numerical_origin_radius_factor
        ),
        names=tuple(beam.name for beam in config.beams),
    )


def _to_cartesian(coordinates: np.ndarray, coordinate_system: str) -> np.ndarray:
    """Initialization-only float64 coordinate conversion."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinate_system == "cartesian":
        return coordinates
    radius, phi, theta = coordinates
    radial_xy = radius * np.sin(theta)
    return np.asarray(
        [radial_xy * np.cos(phi), radial_xy * np.sin(phi), radius * np.cos(theta)]
    )


def _move_origins_to_numerical_domain(
    origins: np.ndarray,
    directions: np.ndarray,
    domain_radius: float,
) -> np.ndarray:
    """Move upstream beam origins to their entry into a bounding sphere."""
    numerical_origins = origins.copy()
    outside = np.linalg.norm(origins, axis=-1) > domain_radius
    along_axis = np.einsum("bi,bi->b", origins, directions)
    discriminant = along_axis**2 - (
        np.einsum("bi,bi->b", origins, origins) - domain_radius**2
    )

    misses = outside & (discriminant < 0.0)
    if np.any(misses):
        indices = np.flatnonzero(misses).tolist()
        raise ValueError(
            "Beam centroid axes do not intersect the numerical domain for "
            f"beam indices {indices}. Increase numerical_origin_radius_factor."
        )

    entry_distance = -along_axis - np.sqrt(np.maximum(discriminant, 0.0))
    points_away = outside & (entry_distance < 0.0)
    if np.any(points_away):
        indices = np.flatnonzero(points_away).tolist()
        raise ValueError(
            f"Beam directions point away from the numerical domain: {indices}."
        )

    numerical_origins[outside] += entry_distance[outside, None] * directions[outside]
    return numerical_origins


def _create_local_beam_frames(
    directions: np.ndarray,
    rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct rotated transverse frames with y aligned to projected +z."""
    global_z = np.asarray([0.0, 0.0, 1.0])
    global_y = np.asarray([0.0, 1.0, 0.0])
    reference = np.broadcast_to(global_z, directions.shape).copy()
    projected = (
        reference - np.einsum("bi,bi->b", reference, directions)[:, None] * directions
    )
    near_pole = np.linalg.norm(projected, axis=-1) < 1.0e-12
    reference[near_pole] = global_y
    projected = (
        reference - np.einsum("bi,bi->b", reference, directions)[:, None] * directions
    )
    unrotated_y = projected / np.linalg.norm(projected, axis=-1, keepdims=True)
    unrotated_x = np.cross(unrotated_y, directions)

    cosine = np.cos(rotations)[:, None]
    sine = np.sin(rotations)[:, None]
    basis_x = cosine * unrotated_x + sine * unrotated_y
    basis_y = -sine * unrotated_x + cosine * unrotated_y
    return basis_x, basis_y
