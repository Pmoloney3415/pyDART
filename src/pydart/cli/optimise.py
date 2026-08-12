"""Command-line entry point for pyDART optimization runs."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence

from pydart.config import load_optimisation_config
from pydart.io.run_artifacts import format_timing_summary
from pydart.optimisation import OptimisationProblem, OptimisationRunner, RestartResult

_NEAR_OPTIMUM_RELATIVE_TOLERANCE = 0.01
_NEAR_OPTIMUM_ABSOLUTE_TOLERANCE = 1.0e-8


def format_restart_summary(
    restart_results: Sequence[RestartResult], best_objective: float
) -> str:
    """Summarise convergence and agreement between completed restarts."""
    completed = len(restart_results)
    converged = sum(restart.success for restart in restart_results)
    near_optimum = sum(
        math.isclose(
            restart.best_objective,
            best_objective,
            rel_tol=_NEAR_OPTIMUM_RELATIVE_TOLERANCE,
            abs_tol=_NEAR_OPTIMUM_ABSOLUTE_TOLERANCE,
        )
        for restart in restart_results
    )
    return "\n".join(
        (
            (
                f"Restart convergence: {converged}/{completed} converged; "
                f"{completed - converged}/{completed} stopped without convergence."
            ),
            (
                f"Near-best restarts: {near_optimum}/{completed} finished within "
                f"{100.0 * _NEAR_OPTIMUM_RELATIVE_TOLERANCE:.1f}% of the best "
                "objective."
            ),
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optimize a pyDART beam design.")
    parser.add_argument("config", help="Path to the optimization TOML deck.")
    parser.add_argument(
        "--resume",
        help=(
            "Resume approximately from a checkpoint's best design. The "
            "optimizer's internal L-BFGS history is restarted."
        ),
    )
    arguments = parser.parse_args(argv)

    config = load_optimisation_config(arguments.config)
    problem = OptimisationProblem(config)
    print(
        f"Starting {config.run.solver} optimisation with "
        f"{config.restarts.number} restarts and {problem.n_parameters} parameters...",
        flush=True,
    )
    result = OptimisationRunner(
        problem,
        resume_checkpoint=arguments.resume,
    ).run()
    print(f"Best objective: {result.best_objective:.8e}")
    print(f"Best RMS nonuniformity: {result.best_record.rms_nonuniformity:.8e}")
    print(
        "Best deposited facility fraction: "
        f"{result.best_record.deposited_capacity_fraction:.8e}"
    )
    print(format_restart_summary(result.restart_results, result.best_objective))
    print(result.message)
    if result.timing is not None:
        print(format_timing_summary(result.timing))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
