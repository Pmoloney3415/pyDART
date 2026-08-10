"""Command-line entry point for pyDART simulation runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax

from pydart.config import load_config
from pydart.io.results import simulation_output_paths
from pydart.io.run_artifacts import (
    Timer,
    copy_used_config,
    format_timing_summary,
    save_timing_summary,
)
from pydart.simulation import initialise_simulation


def main(argv: list[str] | None = None) -> int:
    """Run a configured simulation and its requested output actions."""
    parser = argparse.ArgumentParser(description="Run a pyDART simulation.")
    parser.add_argument("config", help="Path to the simulation TOML deck.")
    arguments = parser.parse_args(argv)

    timer = Timer()
    config = load_config(arguments.config)
    output_directory = config.simulation.output_directory
    simulation_index = config.simulation.index
    result_paths = simulation_output_paths(output_directory, simulation_index)
    simulation_directory = result_paths[0].parent

    timer.start("io")
    copy_used_config(
        arguments.config,
        simulation_directory / "used_configs" / Path(arguments.config).name,
    )
    timer.stop("io")

    timer.start("simulation_compute")
    result = initialise_simulation(config).run()
    metrics = result.get_metrics()
    jax.block_until_ready((result, metrics))
    timer.stop("simulation_compute")

    if config.simulation.save_deposition_data:
        timer.start("io")
        result.save_deposition_data(output_directory)
        timer.stop("io")
    if config.simulation.save_metrics:
        timer.start("io")
        metrics.save_to_directory(output_directory)
        timer.stop("io")
    if config.simulation.plot_data:
        from pydart.plotting import save_key_plots

        timer.start("plotting")
        save_key_plots(
            result,
            metrics,
            output_directory,
            dpi=config.simulation.plot_dpi,
        )
        timer.stop("plotting")

    timing = timer.summary()
    save_timing_summary(
        timing,
        simulation_directory / f"simulation_timing_{simulation_index}.json",
        metadata={"run_type": "simulation", "simulation_index": simulation_index},
    )

    print(f"Deposited fraction: {float(metrics.deposited_fraction):.6f}")
    print(f"RMS nonuniformity: {float(metrics.rms_nonuniformity):.6e}")
    print(format_timing_summary(timing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
