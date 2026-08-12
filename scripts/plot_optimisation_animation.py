"""Render an optimization's archived best designs as PNG frames and an MP4."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pydart.plotting import (
    plot_beam_geometry_mollweide,
    plot_deposition_mollweide,
    plot_mode_power_by_l,
)


class PostprocessingError(RuntimeError):
    """A cleanly reportable error caused by incomplete optimization outputs."""


@dataclass(frozen=True)
class History:
    history_index: np.ndarray
    restart_index: np.ndarray
    iteration: np.ndarray
    objective: np.ndarray
    deposited_capacity_fraction: np.ndarray
    rms_nonuniformity: np.ndarray


@dataclass(frozen=True)
class Snapshot:
    history_index: int
    restart_index: int
    iteration: int
    h5_path: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a six-panel PNG-frame and MP4 history from an optimization "
            "checkpoint and its archived best simulations."
        )
    )
    parser.add_argument("run_directory", type=Path, help="optimisation_N directory")
    parser.add_argument(
        "--step", type=_positive_int, default=10, help="history indices per frame"
    )
    parser.add_argument("--fps", type=_positive_float, default=5.0)
    parser.add_argument("--dpi", type=_positive_int, default=120)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="MP4 path (default: RUN_DIRECTORY/optimisation_history.mp4)",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="PNG directory (default: RUN_DIRECTORY/animation_frames)",
    )
    return parser.parse_args(argv)


def load_history(run_directory: Path) -> History:
    checkpoints = sorted(run_directory.glob("optimisation_checkpoint_*.h5"))
    if len(checkpoints) != 1:
        raise PostprocessingError(
            f"Expected exactly one optimisation_checkpoint_*.h5 in "
            f"'{run_directory}', found {len(checkpoints)}."
        )
    with h5py.File(checkpoints[0], "r") as handle:
        if "history" not in handle:
            raise PostprocessingError(f"Checkpoint '{checkpoints[0]}' has no history.")
        group = handle["history"]
        required = (
            "history_index",
            "restart_index",
            "iteration",
            "objective",
            "deposited_capacity_fraction",
            "rms_nonuniformity",
        )
        missing = [name for name in required if name not in group]
        if missing:
            raise PostprocessingError(
                f"Checkpoint history is missing: {', '.join(missing)}."
            )
        values = {name: np.asarray(group[name]) for name in required}
    if values["history_index"].size == 0:
        raise PostprocessingError("The optimization checkpoint contains no history.")
    order = np.argsort(values["history_index"])
    return History(**{name: values[name][order] for name in required})


def discover_snapshots(run_directory: Path) -> tuple[Snapshot, ...]:
    archive = run_directory / "previous_best_simulations"
    if not archive.is_dir():
        raise PostprocessingError(
            f"Archived best simulations were not found at '{archive}'. Run the "
            "optimization with archive_previous_best_simulations = true."
        )
    snapshots: list[Snapshot] = []
    for metadata_path in sorted(
        archive.glob("simulation_*/optimisation_snapshot.json")
    ):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            h5_files = list(metadata_path.parent.glob("simulation_results_*.h5"))
            if len(h5_files) != 1:
                raise ValueError(f"found {len(h5_files)} result files")
            snapshots.append(
                Snapshot(
                    history_index=int(metadata["history_index"]),
                    restart_index=int(metadata["restart_index"]),
                    iteration=int(metadata["iteration"]),
                    h5_path=h5_files[0],
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise PostprocessingError(
                f"Invalid archived snapshot at '{metadata_path.parent}': {error}."
            ) from error
    if not snapshots:
        raise PostprocessingError(
            f"No archived best snapshots were found in '{archive}'."
        )
    snapshots.sort(key=lambda item: item.history_index)
    return tuple(snapshots)


def load_saved_snapshot(directory: Path) -> Snapshot:
    """Load the single optimization snapshot stored beneath a stable directory."""
    metadata_paths = list(directory.glob("simulation_*/optimisation_snapshot.json"))
    if len(metadata_paths) != 1:
        raise PostprocessingError(
            f"Expected one saved simulation beneath '{directory}', "
            f"found {len(metadata_paths)}."
        )
    metadata_path = metadata_paths[0]
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        h5_files = list(metadata_path.parent.glob("simulation_results_*.h5"))
        if len(h5_files) != 1:
            raise ValueError(f"found {len(h5_files)} result files")
        return Snapshot(
            history_index=int(metadata["history_index"]),
            restart_index=int(metadata["restart_index"]),
            iteration=int(metadata["iteration"]),
            h5_path=h5_files[0],
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise PostprocessingError(
            f"Invalid saved snapshot at '{metadata_path.parent}': {error}."
        ) from error


def discover_restart_snapshots(run_directory: Path) -> tuple[Snapshot, ...]:
    """Return one saved best snapshot for each completed restart."""
    root = run_directory / "restart_best_simulations"
    if not root.is_dir():
        raise PostprocessingError(
            f"Restart-best simulations were not found at '{root}'."
        )
    snapshots = tuple(
        load_saved_snapshot(directory)
        for directory in sorted(
            root.glob("restart_*"),
            key=lambda path: int(path.name.removeprefix("restart_")),
        )
        if directory.is_dir()
    )
    if not snapshots:
        raise PostprocessingError(f"No restart-best simulations were found in '{root}'.")
    return snapshots


def load_snapshot_data(snapshot: Snapshot):
    with h5py.File(snapshot.h5_path, "r") as handle:
        try:
            deposition = handle["deposition"]
            target_group = handle["target"]
            beams_group = handle["beams"]
            harmonics = handle["harmonics"]
            target = SimpleNamespace(
                radius=float(target_group.attrs["radius_m"]),
                cartesian_coordinates=np.asarray(target_group["cartesian_coordinates"]),
                cell_areas=np.asarray(target_group["cell_areas"]),
            )
            beams = SimpleNamespace(
                physical_origins=np.asarray(beams_group["physical_origins"]),
                origins=np.asarray(beams_group["numerical_origins"]),
                directions=np.asarray(beams_group["directions"]),
            )
            result = SimpleNamespace(
                total=np.asarray(deposition["total_cell_power"]),
                target=target,
                beams=beams,
            )
            metrics = SimpleNamespace(
                ell=np.asarray(harmonics["ell"]),
                normalized_power_by_l=np.asarray(harmonics["normalized_power_by_l"]),
            )
        except KeyError as error:
            raise PostprocessingError(
                f"Snapshot '{snapshot.h5_path}' is missing HDF5 field {error}."
            ) from error
    return result, metrics


def frame_indices(history: History, step: int) -> tuple[int, ...]:
    first = int(history.history_index[0])
    last = int(history.history_index[-1])
    indices = list(range(first, last + 1, step))
    if indices[-1] != last:
        indices.append(last)
    return tuple(indices)


def snapshot_at(snapshots: tuple[Snapshot, ...], history_index: int) -> Snapshot:
    available = [item for item in snapshots if item.history_index <= history_index]
    if not available:
        raise PostprocessingError(
            f"No archived best design exists at or before history index {history_index}."
        )
    return available[-1]


def render_frame(
    history: History,
    snapshot: Snapshot,
    history_index: int,
    output_path: Path,
    *,
    dpi: int,
    design_label: str | None = None,
) -> None:
    result, metrics = load_snapshot_data(snapshot)
    figure = plt.figure(figsize=(16, 14), constrained_layout=True)
    grid = figure.add_gridspec(3, 2, width_ratios=(1.0, 1.15))
    objective_ax = figure.add_subplot(grid[0, 0])
    fraction_ax = figure.add_subplot(grid[1, 0])
    rms_ax = figure.add_subplot(grid[2, 0])
    pointing_ax = figure.add_subplot(grid[0, 1], projection="mollweide")
    deposition_ax = figure.add_subplot(grid[1, 1], projection="mollweide")
    mode_ax = figure.add_subplot(grid[2, 1])

    visible = history.history_index <= history_index
    restart_indices = np.unique(history.restart_index)
    colors = matplotlib.colormaps["jet"](np.linspace(0.0, 1.0, restart_indices.size))
    for restart_index, color in zip(restart_indices, colors, strict=True):
        selected = visible & (history.restart_index == restart_index)
        if not np.any(selected):
            continue
        iteration = history.iteration[selected] + 1
        label = f"Restart {int(restart_index) + 1}"
        objective_ax.semilogy(
            iteration,
            _positive(history.objective[selected]),
            color=color,
            label=label,
        )
        fraction_ax.plot(
            iteration,
            history.deposited_capacity_fraction[selected],
            color=color,
            label=label,
        )
        rms_ax.semilogy(
            iteration,
            _positive(history.rms_nonuniformity[selected]),
            color=color,
            label=label,
        )
    _format_history_axis(objective_ax, "Objective", "Loss")
    _format_history_axis(
        fraction_ax, "Deposited facility capacity", "Deposited / facility maximum"
    )
    _format_history_axis(
        rms_ax, "Illumination non-uniformity", "RMS / mean", xlabel=True
    )
    for axis in (objective_ax, fraction_ax, rms_ax):
        axis.set_xscale("log")
        axis.set_xlim(1, max(2, int(np.max(history.iteration)) + 1))
    objective_ax.legend(fontsize=8, ncol=2)

    plot_beam_geometry_mollweide(result, pointing_ax)
    plot_deposition_mollweide(result, deposition_ax)
    plot_mode_power_by_l(metrics, mode_ax)
    if design_label is None:
        design_label = f"Global best from restart {snapshot.restart_index}"
    figure.suptitle(
        f"Optimization history index {history_index} - {design_label}, "
        f"iteration {snapshot.iteration}",
        fontsize=16,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def write_mp4(frame_paths: list[Path], output_path: Path, fps: float) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as error:
        raise PostprocessingError(
            "MP4 support is not installed. Install it with "
            "'python -m pip install -e .[postprocessing]'."
        ) from error
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with imageio.get_writer(
            output_path, fps=fps, codec="libx264", macro_block_size=2
        ) as writer:
            for frame_path in frame_paths:
                writer.append_data(imageio.imread(frame_path))
    except (OSError, RuntimeError, ValueError) as error:
        raise PostprocessingError(
            f"Could not create MP4 '{output_path}': {error}"
        ) from error


def create_animation(args: argparse.Namespace) -> tuple[list[Path], Path]:
    run_directory = args.run_directory.resolve()
    if not run_directory.is_dir():
        raise PostprocessingError(f"Run directory '{run_directory}' does not exist.")
    history = load_history(run_directory)
    snapshots = discover_snapshots(run_directory)
    frames_directory = (
        args.frames_dir.resolve()
        if args.frames_dir is not None
        else run_directory / "animation_frames"
    )
    output_path = (
        args.output.resolve()
        if args.output is not None
        else run_directory / "optimisation_history.mp4"
    )
    paths = []
    for number, index in enumerate(frame_indices(history, args.step)):
        path = frames_directory / f"frame_{number:05d}_index_{index}.png"
        render_frame(history, snapshot_at(snapshots, index), index, path, dpi=args.dpi)
        paths.append(path)
        print(f"Saved {path}")
    write_mp4(paths, output_path, args.fps)
    return paths, output_path


def _format_history_axis(axis, title: str, ylabel: str, *, xlabel: bool = False):
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    if xlabel:
        axis.set_xlabel("Accepted history index")
    axis.grid(alpha=0.25)


def _positive(values: np.ndarray) -> np.ndarray:
    positive = values[values > 0.0]
    floor = max(float(positive.min()) * 0.1, 1.0e-16) if positive.size else 1.0e-16
    return np.maximum(values, floor)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        frames, output = create_animation(args)
    except PostprocessingError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Saved {len(frames)} frames and MP4 animation to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
