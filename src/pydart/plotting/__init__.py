"""Reusable plots and summary dashboards for pyDART results."""

from pydart.plotting.geometry import plot_beam_geometry_mollweide
from pydart.plotting.harmonics import (
    plot_lm_amplitude_heatmap,
    plot_mode_power_by_l,
)
from pydart.plotting.illumination import (
    plot_deposition_lineouts,
    plot_deposition_mollweide,
    plot_deposition_sphere,
    plot_deposition_square,
    plot_latitude_lineout,
    plot_longitude_lineout,
    plot_per_beam_deposited_fractions,
)
from pydart.plotting.optimisation import (
    plot_optimisation_history,
    save_optimisation_history_plot,
)
from pydart.plotting.summary import plot_key_data, save_key_plots

__all__ = [
    "plot_beam_geometry_mollweide",
    "plot_deposition_lineouts",
    "plot_deposition_mollweide",
    "plot_deposition_sphere",
    "plot_deposition_square",
    "plot_key_data",
    "plot_latitude_lineout",
    "plot_lm_amplitude_heatmap",
    "plot_longitude_lineout",
    "plot_mode_power_by_l",
    "plot_optimisation_history",
    "plot_per_beam_deposited_fractions",
    "save_key_plots",
    "save_optimisation_history_plot",
]
