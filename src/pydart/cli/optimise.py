"""Command-line entry point for pyDART optimization runs."""

from __future__ import annotations

import argparse

from pydart.config import load_optimisation_config
from pydart.io.run_artifacts import format_timing_summary
from pydart.optimisation import OptimisationProblem, OptimisationRunner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optimize a pyDART beam design.")
    parser.add_argument("config", help="Path to the optimization TOML deck.")
    parser.add_argument(
        "--resume",
        help=(
            "Resume approximately from a checkpoint's best design. SciPy's "
            "internal L-BFGS history is restarted."
        ),
    )
    arguments = parser.parse_args(argv)

    config = load_optimisation_config(arguments.config)
    result = OptimisationRunner(
        OptimisationProblem(config),
        resume_checkpoint=arguments.resume,
    ).run()
    print(f"Best objective: {result.best_objective:.8e}")
    print(f"Best RMS nonuniformity: {result.best_record.rms_nonuniformity:.8e}")
    print(
        "Best deposited facility fraction: "
        f"{result.best_record.deposited_capacity_fraction:.8e}"
    )
    print(result.message)
    if result.timing is not None:
        print(format_timing_summary(result.timing))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
