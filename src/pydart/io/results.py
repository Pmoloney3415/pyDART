"""Portable HDF5 and JSON persistence for simulation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import numpy as np

if TYPE_CHECKING:
    from pydart.illumination.deposition import DepositionResult
    from pydart.metrics.global_metrics import MetricsResult


def simulation_output_paths(
    output_directory: str | Path,
    simulation_index: int,
    *,
    simulation_label: str | None = None,
) -> tuple[Path, Path]:
    """Return the indexed HDF5 and JSON result paths."""
    label = str(simulation_index) if simulation_label is None else simulation_label
    simulation_directory = Path(output_directory) / f"simulation_{label}"
    return (
        simulation_directory / f"simulation_results_{label}.h5",
        simulation_directory / f"summary_{label}.json",
    )


def save_deposition_result(
    result: DepositionResult,
    output_directory: str | Path,
    *,
    simulation_label: str | None = None,
    metadata: dict[str, int | float | str] | None = None,
) -> Path:
    """Write deposition and target arrays to the indexed HDF5 file."""
    h5py = _import_h5py()
    h5_path, _ = simulation_output_paths(
        output_directory,
        result.simulation_index,
        simulation_label=simulation_label,
    )
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "a") as handle:
        handle.attrs["format"] = "pyDART simulation results"
        handle.attrs["simulation_index"] = result.simulation_index
        handle.attrs["surface_quadrature_order"] = result.surface_quadrature_order
        handle.attrs["visibility_smoothing_epsilon"] = (
            result.visibility_smoothing_epsilon
        )
        _write_metadata_attributes(handle, metadata)
        deposition = _replace_group(handle, "deposition")
        _create_dataset(
            deposition,
            "total_cell_power",
            result.total,
            units="W",
        )
        _create_dataset(
            deposition,
            "per_beam_cell_power",
            result.per_beam,
            units="W",
        )
        _create_dataset(
            deposition,
            "unsmoothed_deposited_power_per_beam",
            result.unsmoothed_deposited_power_per_beam,
            units="W",
        )
        deposition.attrs["axis_order"] = "azimuthal, polar, beam"
        deposition.attrs["cell_power_visibility"] = "smoothed"

        target = _replace_group(handle, "target")
        _create_dataset(
            target,
            "spherical_coordinates",
            result.target.spherical_coordinates,
            units="m, rad, rad",
        )
        _create_dataset(
            target,
            "cartesian_coordinates",
            result.target.cartesian_coordinates,
            units="m",
        )
        _create_dataset(
            target,
            "surface_normals",
            result.target.surface_normals,
            units="1",
        )
        _create_dataset(
            target,
            "cell_areas",
            result.target.cell_areas,
            units="m^2",
        )
        target.attrs["spherical_coordinate_order"] = "r, phi, theta"
        target.attrs["azimuth_range"] = "[-pi, pi)"
        target.attrs["radius_m"] = float(_as_numpy(result.target.radius))

        beams = _replace_group(handle, "beams")
        _create_dataset(
            beams,
            "physical_origins",
            result.beams.physical_origins,
            units="m",
        )
        _create_dataset(
            beams,
            "numerical_origins",
            result.beams.origins,
            units="m",
        )
        _create_dataset(
            beams,
            "pointing_locations",
            result.beams.pointing_locations,
            units="m",
        )
        _create_dataset(
            beams,
            "directions",
            result.beams.directions,
            units="1",
        )
        _create_dataset(
            beams,
            "maximum_power_fractions",
            result.beams.maximum_power_fractions,
            units="1",
        )
        _create_dataset(
            beams,
            "power_fractions_of_maximum",
            result.beams.power_fractions_of_maximum,
            units="1",
        )
        _create_dataset(beams, "incident_powers", result.beams.powers, units="W")
        beams.create_dataset(
            "names",
            data=np.asarray(result.beams.names, dtype="S"),
        )
        handle.attrs["incident_power_W"] = float(_as_numpy(result.incident_power))

    return h5_path


def save_metrics_result(
    metrics: MetricsResult,
    output_directory: str | Path,
    *,
    simulation_label: str | None = None,
    metadata: dict[str, int | float | str] | None = None,
) -> tuple[Path, Path]:
    """Write harmonic arrays to HDF5 and scalar diagnostics to JSON."""
    h5py = _import_h5py()
    h5_path, json_path = simulation_output_paths(
        output_directory,
        metrics.simulation_index,
        simulation_label=simulation_label,
    )
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "a") as handle:
        handle.attrs["format"] = "pyDART simulation results"
        handle.attrs["simulation_index"] = metrics.simulation_index
        _write_metadata_attributes(handle, metadata)
        harmonics = _replace_group(handle, "harmonics")
        _create_dataset(
            harmonics,
            "coefficients",
            metrics.harmonic_coefficients,
            units="W m^-2",
        )
        _create_dataset(
            harmonics,
            "power_by_l",
            metrics.power_by_l,
            units="W^2 m^-4",
        )
        _create_dataset(
            harmonics,
            "normalized_power_by_l",
            metrics.normalized_power_by_l,
            units="1",
        )
        _create_dataset(harmonics, "ell", metrics.ell, units="1")
        _create_dataset(
            harmonics,
            "m",
            np.arange(-metrics.l_max, metrics.l_max + 1),
            units="1",
        )
        harmonics.attrs["coefficient_indexing"] = "coefficients[l, m + l_max]"
        harmonics.attrs["normalization"] = "complex orthonormal"
        harmonics.attrs["condon_shortley_phase"] = True

        summary = _replace_group(handle, "summary")
        scalar_values = _summary_values(metrics)
        for name, value in scalar_values.items():
            summary.create_dataset(name, data=value)

    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "simulation_index": metrics.simulation_index,
                "l_max": metrics.l_max,
                **(metadata or {}),
                **_summary_values(metrics),
            },
            stream,
            indent=2,
        )
        stream.write("\n")

    return h5_path, json_path


def _summary_values(metrics: MetricsResult) -> dict[str, float]:
    return {
        "deposited_power_W": float(_as_numpy(metrics.deposited_power)),
        "incident_power_W": float(_as_numpy(metrics.incident_power)),
        "deposited_fraction": float(_as_numpy(metrics.deposited_fraction)),
        "smoothed_deposited_power_W": float(
            _as_numpy(metrics.smoothed_deposited_power)
        ),
        "smoothed_deposited_fraction": float(
            _as_numpy(metrics.smoothed_deposited_fraction)
        ),
        "mean_power_density_W_m2": float(_as_numpy(metrics.mean_power_density)),
        "rms_nonuniformity": float(_as_numpy(metrics.rms_nonuniformity)),
    }


def _replace_group(handle, name: str):
    if name in handle:
        del handle[name]
    return handle.create_group(name)


def _write_metadata_attributes(handle, metadata) -> None:
    for name, value in (metadata or {}).items():
        handle.attrs[name] = value


def _create_dataset(group, name: str, values, units: str) -> None:
    dataset = group.create_dataset(
        name,
        data=_as_numpy(values),
        compression="gzip" if np.ndim(values) > 0 else None,
        shuffle=np.ndim(values) > 0,
    )
    dataset.attrs["units"] = units


def _as_numpy(values) -> np.ndarray:
    return np.asarray(jax.device_get(values))


def _import_h5py():
    try:
        import h5py
    except ImportError as error:
        raise ImportError(
            "Saving pyDART results requires the optional 'h5py' dependency."
        ) from error
    return h5py
