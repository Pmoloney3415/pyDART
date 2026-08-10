from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from pydart import plotting
from pydart.cli import simulate


def test_simulation_cli_runs_configured_simulation(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    config_path = tmp_path / "simulation.toml"
    config_path.write_text("# test configuration\n", encoding="utf-8")
    metrics = SimpleNamespace(deposited_fraction=0.75, rms_nonuniformity=0.125)
    result = SimpleNamespace(get_metrics=lambda: metrics)
    simulation = SimpleNamespace(run=lambda: result)
    config = SimpleNamespace(
        simulation=SimpleNamespace(
            index=3,
            output_directory=tmp_path / "results",
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
    monkeypatch.setattr(simulate.jax, "block_until_ready", lambda value: value)

    assert simulate.main([str(config_path)]) == 0
    assert loaded_paths == [str(config_path)]
    output = capsys.readouterr().out
    assert "Deposited fraction" in output
    assert "RMS nonuniformity" in output
    assert "Timing summary" in output
    simulation_directory = tmp_path / "results" / "simulation_3"
    assert (simulation_directory / "simulation_timing_3.json").is_file()
    assert (simulation_directory / "used_configs" / "simulation.toml").is_file()


def test_simulation_cli_honours_configured_outputs(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "simulation.toml"
    config_path.write_text("# test configuration\n", encoding="utf-8")
    output_directory = tmp_path / "results"
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
            index=4,
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
    monkeypatch.setattr(simulate.jax, "block_until_ready", lambda value: value)

    assert simulate.main([str(config_path)]) == 0
    result.save_deposition_data.assert_called_once_with(output_directory)
    metrics.save_to_directory.assert_called_once_with(output_directory)
    save_key_plots.assert_called_once_with(
        result,
        metrics,
        output_directory,
        dpi=150,
    )
