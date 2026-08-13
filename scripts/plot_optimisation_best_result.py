"""Render the animation-style result frame for an optimization's overall best."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from plot_optimisation_animation import (
    PostprocessingError,
    load_history,
    load_saved_snapshot,
    render_frame,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one animation-style PNG for the overall-best design."
    )
    parser.add_argument("run_directory", type=Path, help="optimisation_N directory")
    parser.add_argument("--output", type=Path, help="Output PNG path.")
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
        snapshot = load_saved_snapshot(run_directory / "best_simulation")
        output = (
            arguments.output.resolve()
            if arguments.output is not None
            else run_directory / "optimisation_best_result.png"
        )
        render_frame(
            history,
            snapshot,
            int(history.history_index[-1]),
            output,
            dpi=arguments.dpi,
            design_label=f"Best overall design from restart {snapshot.restart_index}",
        )
    except PostprocessingError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
