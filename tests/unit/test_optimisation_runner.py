from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np

from pydart.config.optimisation_config import load_optimisation_config
from pydart.optimisation import OptimisationProblem, OptimisationRunner
from pydart.optimisation.persistence import load_optimisation_checkpoint

matplotlib.use("Agg")

CONFIG_DIRECTORY = Path(__file__).parents[2] / "configs" / "optimisations"


def test_six_beam_smoke_optimisation_writes_recoverable_outputs(
    tmp_path: Path,
) -> None:
    config = load_optimisation_config(CONFIG_DIRECTORY / "six_beam_design.toml")
    simulation = replace(
        config.simulation,
        simulation=replace(config.simulation.simulation, plot_dpi=40),
        target=replace(
            config.simulation.target,
            n_polar=8,
            n_azimuthal=16,
        ),
        metrics=replace(config.simulation.metrics, l_max=4),
    )
    run = replace(
        config.run,
        output_directory=tmp_path,
        maximum_iterations=1,
        checkpoint_interval=1,
        history_plot_interval=1,
        maximum_wall_time_seconds=120.0,
        save_best_simulation=False,
        archive_previous_best_simulations=False,
    )
    restarts = replace(config.restarts, number=1)
    objective = replace(
        config.objective,
        mode_weights=config.objective.mode_weights[:4],
    )
    problem = OptimisationProblem(
        replace(
            config,
            simulation=simulation,
            run=run,
            restarts=restarts,
            objective=objective,
        )
    )

    result = OptimisationRunner(problem).run()

    optimisation_index = problem.config.run.index
    output = tmp_path / f"optimisation_{optimisation_index}"
    assert np.isfinite(result.best_objective)
    assert len(result.restart_results) == 1
    assert len(result.history) >= 1
    checkpoint_path = output / f"optimisation_checkpoint_{optimisation_index}.h5"
    summary_path = output / f"optimisation_summary_{optimisation_index}.json"
    assert checkpoint_path.is_file()
    assert summary_path.is_file()

    history, restarts, best, elapsed = load_optimisation_checkpoint(checkpoint_path)
    assert len(history) == len(result.history)
    assert len(restarts) == 1
    assert best.design.shape == (problem.n_parameters,)
    assert best.objective == result.best_objective
    assert elapsed > 0.0
