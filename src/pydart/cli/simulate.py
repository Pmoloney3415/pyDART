"""Command-line entry point for pyDART simulation runs."""

from __future__ import annotations

import argparse

from pydart.config import load_config
from pydart.simulation import initialise_simulation


def main(argv: list[str] | None = None) -> int:
    """Run a configured simulation and its requested output actions."""
    parser = argparse.ArgumentParser(description="Run a pyDART simulation.")
    parser.add_argument("config", help="Path to the simulation TOML deck.")
    arguments = parser.parse_args(argv)

    config = load_config(arguments.config)
    result = initialise_simulation(config).run()
    metrics = result.get_metrics()
    output_directory = config.simulation.output_directory

    if config.simulation.save_deposition_data:
        result.save_deposition_data(output_directory)
    if config.simulation.save_metrics:
        metrics.save_to_directory(output_directory)
    if config.simulation.plot_data:
        from pydart.plotting import save_key_plots

        save_key_plots(
            result,
            metrics,
            output_directory,
            dpi=config.simulation.plot_dpi,
        )

    print(f"Deposited fraction: {float(metrics.deposited_fraction):.6f}")
    print(f"RMS nonuniformity: {float(metrics.rms_nonuniformity):.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
