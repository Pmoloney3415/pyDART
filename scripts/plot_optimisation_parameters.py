"""Plot final optimized beam parameters against beam index."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the final beam parameters from an optimization checkpoint."
    )
    parser.add_argument(
        "result",
        type=Path,
        help="An optimisation_N directory or optimisation checkpoint HDF5 file.",
    )
    parser.add_argument("--output", type=Path, help="Output PNG path.")
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args(argv)


def find_checkpoint(result: Path) -> Path:
    result = result.resolve()
    if result.is_file():
        return result
    checkpoints = sorted(result.glob("optimisation_checkpoint_*.h5"))
    if len(checkpoints) != 1:
        raise ValueError(
            f"Expected one optimisation checkpoint in '{result}', "
            f"found {len(checkpoints)}."
        )
    return checkpoints[0]


def load_best_parameters(checkpoint: Path) -> tuple[dict[str, np.ndarray], float]:
    with h5py.File(checkpoint, "r") as handle:
        group = handle["global_best"]
        parameters = {
            name: np.asarray(group[name])
            for name in (
                "physical_origins",
                "pointing_locations",
                "power_fractions_of_maximum",
                "spot_widths",
                "spot_rotations",
                "supergaussian_indices",
            )
        }
        objective = float(group.attrs["objective"])
    return parameters, objective


def spherical_angles(cartesian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radius = np.linalg.norm(cartesian, axis=1)
    azimuth = np.rad2deg(np.arctan2(cartesian[:, 1], cartesian[:, 0]))
    polar = np.rad2deg(np.arccos(np.clip(cartesian[:, 2] / radius, -1.0, 1.0)))
    return azimuth, polar


def plot_parameter_summary(parameters: dict[str, np.ndarray], objective: float):
    beam_index = np.arange(1, len(parameters["power_fractions_of_maximum"]) + 1)
    origin_azimuth, origin_polar = spherical_angles(parameters["physical_origins"])
    pointing_azimuth, pointing_polar = spherical_angles(
        parameters["pointing_locations"]
    )

    figure, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    marker = {"marker": ".", "linewidth": 1.0}

    axes[0, 0].plot(beam_index, parameters["power_fractions_of_maximum"], **marker)
    axes[0, 0].set_ylabel("Fraction of maximum")
    axes[0, 0].set_title("Beam power")

    widths = parameters["spot_widths"] * 1.0e6
    axes[0, 1].plot(beam_index, widths[:, 0], label=r"$w_x$", **marker)
    axes[0, 1].plot(beam_index, widths[:, 1], label=r"$w_y$", **marker)
    axes[0, 1].set_ylabel("Width (µm)")
    axes[0, 1].set_title("Spot widths")
    axes[0, 1].legend()

    axes[1, 0].plot(beam_index, parameters["supergaussian_indices"], **marker)
    axes[1, 0].set_ylabel("Index")
    axes[1, 0].set_title("Super-Gaussian index")

    rotations = np.rad2deg(parameters["spot_rotations"])
    axes[1, 1].plot(beam_index, rotations, **marker)
    axes[1, 1].set_ylabel("Rotation (degrees)")
    axes[1, 1].set_title("Spot rotation")

    axes[2, 0].plot(beam_index, origin_polar, label="origin", **marker)
    axes[2, 0].plot(beam_index, pointing_polar, label="pointing", **marker)
    axes[2, 0].set_ylabel("Angle (degrees)")
    axes[2, 0].set_title("Polar angle")
    axes[2, 0].legend()

    axes[2, 1].plot(beam_index, origin_azimuth, label="origin", **marker)
    axes[2, 1].plot(beam_index, pointing_azimuth, label="pointing", **marker)
    axes[2, 1].set_ylabel("Angle (degrees)")
    axes[2, 1].set_title("Azimuthal angle")
    axes[2, 1].legend()

    for axis in axes.flat:
        axis.set_xlabel("Beam index")
        axis.set_xlim(0.5, beam_index[-1] + 0.5)
        axis.grid(alpha=0.25)
    figure.suptitle(f"Best optimization design — objective {objective:.6e}")
    figure.tight_layout()
    return figure


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    checkpoint = find_checkpoint(arguments.result)
    parameters, objective = load_best_parameters(checkpoint)
    figure = plot_parameter_summary(parameters, objective)
    output = (
        arguments.output.resolve()
        if arguments.output is not None
        else checkpoint.parent / "optimisation_parameter_summary.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=arguments.dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
