"""Progress plots for multi-start illumination optimization."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pydart.optimisation.optimise import IterationRecord


def plot_optimisation_history(history: Sequence[IterationRecord]):
    """Create a six-panel diagnostic summary from accepted iterates."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    restart_indices = sorted({record.restart_index for record in history})
    for restart_index in restart_indices:
        records = [
            record for record in history if record.restart_index == restart_index
        ]
        iteration = np.asarray([record.iteration for record in records]) + 1
        label = f"restart {restart_index + 1}"
        axes[0, 0].loglog(
            iteration,
            _positive_for_log([record.objective for record in records]),
            marker=".",
            alpha=0.75,
            label=label,
        )
        axes[0, 2].loglog(
            iteration,
            _positive_for_log([record.rms_nonuniformity for record in records]),
            marker=".",
            alpha=0.75,
            label=label,
        )
        axes[1, 0].loglog(
            iteration,
            _positive_for_log(
                [record.deposited_capacity_fraction for record in records]
            ),
            marker=".",
            alpha=0.75,
            label=label,
        )
        axes[1, 1].loglog(
            iteration,
            _positive_for_log([record.projected_gradient_norm for record in records]),
            alpha=0.75,
            label=label,
        )

    best_restart_index = min(
        restart_indices,
        key=lambda index: min(
            record.objective for record in history if record.restart_index == index
        ),
    )
    best_records = [
        record for record in history if record.restart_index == best_restart_index
    ]
    best_iteration = np.asarray([record.iteration for record in best_records]) + 1
    axes[0, 1].loglog(
        best_iteration,
        _positive_for_log([record.symmetry_contribution for record in best_records]),
        label="symmetry",
    )
    axes[0, 1].loglog(
        best_iteration,
        _positive_for_log([record.deposition_contribution for record in best_records]),
        linestyle=":",
        label="deposition",
    )

    best_by_restart = []
    for restart_index in restart_indices:
        values = [
            record.objective
            for record in history
            if record.restart_index == restart_index
        ]
        best_by_restart.append(min(values))
    display_restart_indices = np.asarray(restart_indices) + 1
    axes[1, 2].scatter(
        display_restart_indices,
        best_by_restart,
        color="dodgerblue",
    )
    axes[1, 2].set_xticks(display_restart_indices)

    axes[0, 0].set(title="Objective", xlabel="Accepted iteration + 1", ylabel="Loss")
    axes[0, 1].set(
        title=f"Objective components (best: restart {best_restart_index + 1})",
        xlabel="Accepted iteration + 1",
        ylabel="Contribution",
    )
    axes[0, 2].set(
        title="Illumination nonuniformity",
        xlabel="Accepted iteration + 1",
        ylabel="RMS / mean",
    )
    axes[1, 0].set(
        title="Deposited facility capacity",
        xlabel="Accepted iteration + 1",
        ylabel="Deposited / facility maximum",
    )
    axes[1, 1].set(
        title="Gradient convergence",
        xlabel="Accepted iteration + 1",
        ylabel="L2 norm",
    )
    axes[1, 2].set(
        title="Best result per restart",
        xlabel="Restart",
        ylabel="Best objective",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=7, ncol=2)
    axes[1, 1].legend(fontsize=7, ncol=2)
    figure.suptitle("pyDART optimization history", fontsize=16)
    return figure


def _positive_for_log(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    positive = values[values > 0.0]
    floor = max(float(positive.min()) * 0.1, 1.0e-16) if positive.size else 1.0e-16
    return np.maximum(values, floor)


def save_optimisation_history_plot(
    history: Sequence[IterationRecord],
    output_path: str | Path,
    *,
    dpi: int = 160,
) -> Path:
    """Overwrite the current optimization-history PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = plot_optimisation_history(history)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path
