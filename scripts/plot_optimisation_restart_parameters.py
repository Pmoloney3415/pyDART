"""Plot the best beam parameters from each optimization restart."""

from __future__ import annotations

import argparse
from pathlib import Path

from plot_optimisation_parameters import (
    find_checkpoint,
    load_restart_parameters,
    save_parameter_plot,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a separate parameter summary for each completed restart."
    )
    parser.add_argument(
        "result",
        type=Path,
        help="An optimisation_N directory or optimisation checkpoint HDF5 file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Output directory (default: "
            "RESULT/restart_parameter_plots)."
        ),
    )
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    checkpoint = find_checkpoint(arguments.result)
    output_directory = (
        arguments.output_dir.resolve()
        if arguments.output_dir is not None
        else checkpoint.parent / "restart_parameter_plots"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    for restart_number, parameters, objective in load_restart_parameters(checkpoint):
        output = output_directory / f"restart_{restart_number}_parameters.png"
        save_parameter_plot(
            (parameters, objective),
            output,
            arguments.dpi,
            design_label=f"Restart {restart_number} best design",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
