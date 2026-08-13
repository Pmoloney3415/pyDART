from __future__ import annotations

import numpy as np

from pydart.cli.optimise import format_restart_summary
from pydart.optimisation import RestartResult


def _restart(*, success: bool, best_objective: float) -> RestartResult:
    return RestartResult(
        restart_index=0,
        success=success,
        status=0 if success else 1,
        message="Finished.",
        iterations=10,
        function_evaluations=20,
        best_objective=best_objective,
        best_design=np.zeros(1),
    )


def test_restart_summary_counts_convergence_and_near_best_independently() -> None:
    restarts = (
        _restart(success=True, best_objective=10.0),
        _restart(success=False, best_objective=10.05),
        _restart(success=False, best_objective=10.2),
    )

    summary = format_restart_summary(restarts, best_objective=10.0)

    assert "1/3 converged" in summary
    assert "2/3 stopped without convergence" in summary
    assert "2/3 finished within 1.0%" in summary


def test_restart_summary_uses_absolute_tolerance_near_zero() -> None:
    restarts = (
        _restart(success=True, best_objective=0.0),
        _restart(success=True, best_objective=5.0e-9),
        _restart(success=True, best_objective=2.0e-8),
    )

    summary = format_restart_summary(restarts, best_objective=0.0)

    assert "2/3 finished within 1.0%" in summary
