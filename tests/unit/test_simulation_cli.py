from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from pydart import plotting
from pydart.cli import simulate


def test_simulation_cli_runs_configured_simulation(monkeypatch, capsys) -> None:
    metrics = SimpleNamespace(deposited_fraction=0.75, rms_nonuniformity=0.125)
    result = SimpleNamespace(get_metrics=lambda: metrics)
    simulation = SimpleNamespace(run=lambda: result)
    config = SimpleNamespace(
        simulation=SimpleNamespace(
            output_directory=Path("results/test"),
            save_deposition_data=False,
            save_metrics=False,
            plot_data=False,
            plot_dpi=200,
        )
    )
    loaded_paths = []

    monkeypatch.setattr(
        simulate,
        "load_config",
        lambda path: loaded_paths.append(path) or config,
    )
    monkeypatch.setattr(simulate, "initialise_simulation", lambda loaded: simulation)

    assert simulate.main(["simulation.toml"]) == 0
    assert loaded_paths == ["simulation.toml"]
    output = capsys.readouterr().out
    assert "Deposited fraction" in output
    assert "RMS nonuniformity" in output


def test_simulation_cli_honours_configured_outputs(monkeypatch) -> None:
    output_directory = Path("results/test")
    metrics = SimpleNamespace(
        deposited_fraction=0.75,
        rms_nonuniformity=0.125,
        save_to_directory=Mock(),
    )
    result = SimpleNamespace(
        get_metrics=lambda: metrics,
        save_deposition_data=Mock(),
    )
    simulation = SimpleNamespace(run=lambda: result)
    config = SimpleNamespace(
        simulation=SimpleNamespace(
            output_directory=output_directory,
            save_deposition_data=True,
            save_metrics=True,
            plot_data=True,
            plot_dpi=150,
        )
    )
    save_key_plots = Mock()
    monkeypatch.setattr(simulate, "load_config", lambda path: config)
    monkeypatch.setattr(simulate, "initialise_simulation", lambda loaded: simulation)
    monkeypatch.setattr(plotting, "save_key_plots", save_key_plots)

    assert simulate.main(["simulation.toml"]) == 0
    result.save_deposition_data.assert_called_once_with(output_directory)
    metrics.save_to_directory.assert_called_once_with(output_directory)
    save_key_plots.assert_called_once_with(
        result,
        metrics,
        output_directory,
        dpi=150,
    )
