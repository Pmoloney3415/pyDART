"""Render an animation-style result frame for each optimization restart best."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from plot_optimisation_animation import (
    PostprocessingError,
    discover_restart_snapshots,
    load_history,
    render_frame,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one animation-style PNG for each restart-best design."
    )
    parser.add_argument("run_directory", type=Path, help="optimisation_N directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: RUN_DIRECTORY/restart_result_plots).",
    )
    parser.add_argument("--dpi", type=int, default=120)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        run_directory = arguments.run_directory.resolve()
        if not run_directory.is_dir():
            raise PostprocessingError(
                f"Run directory '{run_directory}' does not exist."
            )
        history = load_history(run_directory)
        snapshots = discover_restart_snapshots(run_directory)
        output_directory = (
            arguments.output_dir.resolve()
            if arguments.output_dir is not None
            else run_directory / "restart_result_plots"
        )
        for snapshot in snapshots:
            output = output_directory / (
                f"restart_{snapshot.restart_index}_result.png"
            )
            render_frame(
                history,
                snapshot,
                snapshot.history_index,
                output,
                dpi=arguments.dpi,
                design_label=f"Restart {snapshot.restart_index} best design",
            )
            print(f"Saved {output}")
    except PostprocessingError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
