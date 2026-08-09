"""Composition and persistence of the key simulation plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pydart.plotting.geometry import plot_beam_geometry_mollweide
from pydart.plotting.harmonics import (
    plot_lm_amplitude_heatmap,
    plot_mode_power_by_l,
)
from pydart.plotting.illumination import (
    plot_deposition_mollweide,
    plot_deposition_sphere,
    plot_latitude_lineout,
    plot_longitude_lineout,
    plot_per_beam_deposited_fractions,
    power_density,
)


def plot_key_data(result, metrics, *, cmap="inferno"):
    """Create the three-row summary dashboard for one simulation."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(19, 16), constrained_layout=True)
    grid = figure.add_gridspec(3, 3, height_ratios=(1.05, 1.0, 0.72))
    beam_ax = figure.add_subplot(grid[0, 0], projection="mollweide")
    mollweide_ax = figure.add_subplot(grid[0, 1], projection="mollweide")
    sphere_ax = figure.add_subplot(grid[0, 2], projection="3d")
    mode_ax = figure.add_subplot(grid[1, 0])
    heatmap_ax = figure.add_subplot(grid[1, 1])
    fraction_ax = figure.add_subplot(grid[1, 2])
    lineout_grid = grid[2, :].subgridspec(1, 2)
    longitude_ax = figure.add_subplot(lineout_grid[0, 0])
    latitude_ax = figure.add_subplot(lineout_grid[0, 1])

    density = power_density(result)
    norm = mpl.colors.Normalize(
        vmin=float(np.nanmin(density)),
        vmax=float(np.nanmax(density)),
    )
    plot_beam_geometry_mollweide(result, beam_ax)
    plot_deposition_mollweide(result, mollweide_ax, cmap=cmap, norm=norm)
    plot_deposition_sphere(result, sphere_ax, cmap=cmap, norm=norm)
    plot_mode_power_by_l(metrics, mode_ax)
    plot_lm_amplitude_heatmap(metrics, heatmap_ax)
    plot_per_beam_deposited_fractions(result, fraction_ax)
    plot_longitude_lineout(result, longitude_ax)
    plot_latitude_lineout(result, latitude_ax)
    figure.suptitle(
        f"pyDART simulation {result.simulation_index}",
        fontsize=18,
    )
    return figure


def save_key_plots(
    result,
    metrics,
    output_directory: str | Path,
    *,
    dpi: int = 200,
    cmap="inferno",
    simulation_label: str | None = None,
) -> Path:
    """Create and save the indexed PNG summary, then close the figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label = (
        str(result.simulation_index) if simulation_label is None else simulation_label
    )
    simulation_directory = Path(output_directory) / f"simulation_{label}"
    simulation_directory.mkdir(parents=True, exist_ok=True)
    output_path = simulation_directory / f"key_plots_{label}.png"
    figure = plot_key_data(result, metrics, cmap=cmap)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path
