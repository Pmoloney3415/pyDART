from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pydart.config.simulation_config import (
    BeamConfig,
    LaserConfig,
    PyDARTConfig,
    SimulationConfig,
    SpotConfig,
    TargetConfig,
)
from pydart.optimisation import IterationRecord
from pydart.plotting import (
    plot_beam_geometry_mollweide,
    plot_deposition_lineouts,
    plot_deposition_mollweide,
    plot_deposition_sphere,
    plot_deposition_square,
    plot_latitude_lineout,
    plot_lm_amplitude_heatmap,
    plot_longitude_lineout,
    plot_mode_power_by_l,
    plot_optimisation_history,
    plot_per_beam_deposited_fractions,
    save_key_plots,
)
from pydart.simulation.simulation import initialise_simulation


@pytest.fixture(scope="module")
def results_and_metrics():
    beam = BeamConfig(
        name="beam",
        origin_coordinate_system="cartesian",
        pointing_coordinate_system="cartesian",
        origin=np.asarray([100.0, 0.0, 0.0]),
        pointing=np.zeros(3),
        maximum_power_fraction=1.0,
        power_fraction_of_maximum=1.0,
        frequency=1.0,
        spot=SpotConfig("circular", 0.3, 0.3, 2.0, 0.0),
    )
    config = PyDARTConfig(
        simulation=SimulationConfig(3, Path("results"), False, False),
        target=TargetConfig(1.0, 24, 48, 10.0),
        laser=LaserConfig(1, 10.0),
        beams=(beam,),
    )
    result = initialise_simulation(config).run()
    return result, result.get_metrics()


def test_public_plot_functions_create_figures(results_and_metrics) -> None:
    result, metrics = results_and_metrics
    functions = (
        lambda: plot_deposition_square(result),
        lambda: plot_deposition_mollweide(result),
        lambda: plot_deposition_sphere(result),
        lambda: plot_beam_geometry_mollweide(result),
        lambda: plot_mode_power_by_l(metrics),
        lambda: plot_lm_amplitude_heatmap(metrics),
        lambda: plot_per_beam_deposited_fractions(result),
        lambda: plot_longitude_lineout(result),
        lambda: plot_latitude_lineout(result),
    )

    for make_plot in functions:
        figure, axes = make_plot()
        assert figure is axes.figure
        plt.close(figure)

    figure, axes = plot_deposition_lineouts(result)
    assert len(axes) == 2
    plt.close(figure)


def test_mode_power_omits_monopole(results_and_metrics) -> None:
    _, metrics = results_and_metrics

    figure, axis = plot_mode_power_by_l(metrics)
    plotted_ell = axis.lines[0].get_xdata()

    assert plotted_ell[0] == 1
    assert 0 not in plotted_ell
    plt.close(figure)


def test_optimisation_history_creates_figure() -> None:
    def record(restart, iteration, objective, deposited_fraction):
        return IterationRecord(
            restart_index=restart,
            iteration=iteration,
            history_index=restart * 2 + iteration,
            function_evaluations=iteration + 1,
            elapsed_seconds=float(iteration),
            objective=objective,
            symmetry_contribution=objective * 0.9,
            rms_ratio_power=objective * 0.6,
            deposition_contribution=objective * 0.1,
            rms_nonuniformity=objective * 0.5,
            deposited_capacity_fraction=deposited_fraction,
            gradient_norm=objective * 2.0,
            projected_gradient_norm=objective,
            design=np.zeros(2),
        )

    history = (
        record(0, 0, 1.0, 0.9),
        record(0, 1, 0.2, 0.99),
        record(1, 0, 2.0, 0.8),
        record(1, 1, 0.5, 1.0),
    )
    figure = plot_optimisation_history(history)

    assert figure.axes
    shortfall_axis = figure.axes[3]
    assert shortfall_axis.get_yscale() == "log"
    assert "1-p" in shortfall_axis.get_ylabel()
    np.testing.assert_allclose(shortfall_axis.lines[0].get_ydata(), [0.1, 0.01])
    assert np.all(shortfall_axis.lines[1].get_ydata() > 0.0)
    plt.close(figure)


def test_key_plot_saves_to_indexed_png(tmp_path, results_and_metrics) -> None:
    result, metrics = results_and_metrics

    path = save_key_plots(result, metrics, tmp_path, dpi=50)

    assert path == tmp_path / "simulation_3" / "key_plots_3.png"
    assert path.is_file()
    assert path.stat().st_size > 0
