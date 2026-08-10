"""Multi-start, checkpointed L-BFGS-B optimization runner."""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pydart.io.run_artifacts import (
    Timer,
    copy_used_config,
    save_timing_summary,
)
from pydart.optimisation.persistence import (
    load_optimisation_checkpoint,
    optimisation_output_directory,
    save_optimisation_checkpoint,
    save_optimisation_summary,
    save_simulation_snapshot,
)
from pydart.optimisation.problem import ObjectiveTerms, OptimisationProblem
from pydart.plotting.optimisation import save_optimisation_history_plot


@dataclass(frozen=True)
class IterationRecord:
    """Host-side diagnostics for one initial or accepted design."""

    restart_index: int
    iteration: int
    history_index: int
    function_evaluations: int
    elapsed_seconds: float
    objective: float
    rms_contribution: float
    mode_contribution: float
    deposition_contribution: float
    rms_nonuniformity: float
    deposited_capacity_fraction: float
    gradient_norm: float
    projected_gradient_norm: float
    design: np.ndarray
    normalized_power_by_l: np.ndarray


@dataclass(frozen=True)
class RestartResult:
    """Terminal information from one L-BFGS-B start."""

    restart_index: int
    success: bool
    status: int
    message: str
    iterations: int
    function_evaluations: int
    best_objective: float
    best_design: np.ndarray


@dataclass(frozen=True)
class OptimisationResult:
    """Best design and complete accepted-iteration history."""

    problem: OptimisationProblem
    success: bool
    message: str
    elapsed_seconds: float
    history: tuple[IterationRecord, ...]
    restart_results: tuple[RestartResult, ...]
    best_record: IterationRecord
    timing: dict | None = None

    @property
    def best_design(self) -> np.ndarray:
        return self.best_record.design

    @property
    def best_objective(self) -> float:
        return self.best_record.objective


class OptimisationRunner:
    """Run configured SciPy L-BFGS-B starts using JIT-compiled JAX gradients."""

    def __init__(
        self,
        problem: OptimisationProblem,
        resume_checkpoint: str | Path | None = None,
    ):
        self.problem = problem
        self._history: list[IterationRecord] = []
        self._restart_results: list[RestartResult] = []
        self._best_record: IterationRecord | None = None
        self._best_changed_since_snapshot = False
        self._last_archived_history_index: int | None = None
        self._start_time = 0.0
        self._prior_elapsed = 0.0
        self._resume_design: np.ndarray | None = None
        self._restart_index_offset = 0
        self._timer = Timer()
        if resume_checkpoint is not None:
            history, restart_results, best, elapsed = load_optimisation_checkpoint(
                resume_checkpoint
            )
            if best.design.shape != (problem.n_parameters,):
                raise ValueError(
                    "Checkpoint design size does not match this optimization problem."
                )
            if best.normalized_power_by_l.shape != (
                problem.config.simulation.metrics.l_max + 1,
            ):
                raise ValueError(
                    "Checkpoint harmonic resolution does not match this problem."
                )
            self._history.extend(history)
            self._restart_results.extend(restart_results)
            self._best_record = best
            self._prior_elapsed = elapsed
            self._resume_design = best.design.copy()
            self._restart_index_offset = (
                max(record.restart_index for record in history) + 1
            )
            self._last_archived_history_index = best.history_index

    def run(self) -> OptimisationResult:
        """Execute all starts, checkpoint progress, and return the global best."""
        try:
            from scipy.optimize import minimize
        except ImportError as error:
            raise ImportError(
                "Running pyDART optimization requires the optional 'scipy' dependency."
            ) from error

        self._timer = Timer()
        self._start_time = time.monotonic()
        output_directory = optimisation_output_directory(self.problem)
        used_configs = output_directory / "used_configs"
        self._timer.start("io")
        copy_used_config(
            self.problem.config.source_path,
            used_configs / "optimisation.toml",
        )
        copy_used_config(
            self.problem.config.run.simulation_config,
            used_configs / "simulation.toml",
        )
        self._timer.stop("io")
        evaluator = _JaxObjectiveEvaluator(self.problem, self._timer)
        starts = self._starting_designs()
        stopped_for_time = False

        for local_restart_index, start in enumerate(starts):
            restart_index = self._restart_index_offset + local_restart_index
            if self._elapsed() >= self.problem.config.run.maximum_wall_time_seconds:
                stopped_for_time = True
                break
            evaluator.reset_restart_count()
            restart_best = self._record(
                evaluator,
                start,
                restart_index=restart_index,
                iteration=0,
            )
            if len(self._history) == 1:
                self._persist(checkpoint=True, history_plot=True, snapshot=True)

            accepted_iterations = 0

            def callback(design, restart_index=restart_index):
                nonlocal accepted_iterations, restart_best
                accepted_iterations += 1
                record = self._record(
                    evaluator,
                    design,
                    restart_index=restart_index,
                    iteration=accepted_iterations,
                )
                if record.objective < restart_best.objective:
                    restart_best = record
                checkpoint = (
                    accepted_iterations % self.problem.config.run.checkpoint_interval
                    == 0
                )
                history_plot = (
                    accepted_iterations % self.problem.config.run.history_plot_interval
                    == 0
                )
                if checkpoint or history_plot:
                    self._persist(
                        checkpoint=checkpoint,
                        history_plot=history_plot,
                        snapshot=checkpoint,
                    )
                if self._elapsed() >= self.problem.config.run.maximum_wall_time_seconds:
                    raise _WallTimeExceeded

            try:
                scipy_result = minimize(
                    evaluator.scipy_value_and_gradient,
                    np.asarray(start, dtype=np.float64),
                    method="L-BFGS-B",
                    jac=True,
                    bounds=[(0.0, 1.0)] * self.problem.n_parameters,
                    callback=callback,
                    options={
                        "maxiter": self.problem.config.run.maximum_iterations,
                        "ftol": (self.problem.config.run.objective_relative_tolerance),
                        "gtol": (self.problem.config.run.projected_gradient_tolerance),
                        "maxls": 20,
                    },
                )
                restart_result = RestartResult(
                    restart_index=restart_index,
                    success=bool(scipy_result.success),
                    status=int(scipy_result.status),
                    message=str(scipy_result.message),
                    iterations=int(scipy_result.nit),
                    function_evaluations=int(scipy_result.nfev),
                    best_objective=restart_best.objective,
                    best_design=restart_best.design.copy(),
                )
            except _WallTimeExceeded:
                stopped_for_time = True
                restart_result = RestartResult(
                    restart_index=restart_index,
                    success=False,
                    status=2,
                    message="Maximum wall time reached.",
                    iterations=accepted_iterations,
                    function_evaluations=evaluator.restart_evaluations,
                    best_objective=restart_best.objective,
                    best_design=restart_best.design.copy(),
                )
            self._restart_results.append(restart_result)
            self._persist(checkpoint=True, history_plot=True, snapshot=True)
            if stopped_for_time:
                break

        if self._best_record is None:
            raise RuntimeError("Optimization stopped before evaluating any design.")
        success = any(result.success for result in self._restart_results)
        if stopped_for_time:
            message = "Maximum wall time reached; returning the best saved design."
        elif success:
            message = "At least one L-BFGS-B restart converged."
        else:
            message = (
                "All restarts stopped without convergence; returning the best design."
            )
        result = OptimisationResult(
            problem=self.problem,
            success=success,
            message=message,
            elapsed_seconds=self._elapsed(),
            history=tuple(self._history),
            restart_results=tuple(self._restart_results),
            best_record=self._best_record,
        )
        self._persist(checkpoint=True, history_plot=True, snapshot=True)
        self._timer.start("io")
        save_optimisation_summary(result)
        self._timer.stop("io")
        timing = self._timer.summary()
        result = replace(result, timing=timing)
        save_timing_summary(
            timing,
            output_directory
            / f"optimisation_timing_{self.problem.config.run.index}.json",
            metadata={
                "run_type": "optimisation",
                "optimisation_index": self.problem.config.run.index,
            },
        )
        return result

    def _record(
        self,
        evaluator: _JaxObjectiveEvaluator,
        design,
        *,
        restart_index: int,
        iteration: int,
    ) -> IterationRecord:
        value, gradient, terms = evaluator.evaluate(design)
        design = np.asarray(design, dtype=np.float64).copy()
        projected_gradient = _projected_gradient(design, gradient)
        record = IterationRecord(
            restart_index=restart_index,
            iteration=iteration,
            history_index=len(self._history),
            function_evaluations=evaluator.restart_evaluations,
            elapsed_seconds=self._elapsed(),
            objective=value,
            rms_contribution=float(terms.rms_contribution),
            mode_contribution=float(terms.mode_contribution),
            deposition_contribution=float(terms.deposition_contribution),
            rms_nonuniformity=float(terms.rms_nonuniformity),
            deposited_capacity_fraction=float(terms.deposited_capacity_fraction),
            gradient_norm=float(np.linalg.norm(gradient)),
            projected_gradient_norm=float(np.linalg.norm(projected_gradient)),
            design=design,
            normalized_power_by_l=np.asarray(
                terms.normalized_power_by_l, dtype=np.float64
            ).copy(),
        )
        self._history.append(record)
        if self._best_record is None or record.objective < self._best_record.objective:
            self._best_record = record
            self._best_changed_since_snapshot = True
        return record

    def _persist(
        self,
        *,
        checkpoint: bool,
        history_plot: bool,
        snapshot: bool,
    ) -> None:
        if self._best_record is None:
            return
        output_directory = optimisation_output_directory(self.problem)
        if checkpoint:
            self._timer.start("io")
            save_optimisation_checkpoint(
                self.problem,
                self._history,
                self._best_record,
                self._restart_results,
                self._elapsed(),
            )
            self._timer.stop("io")
        if history_plot:
            path = output_directory / (
                f"optimisation_history_{self.problem.config.run.index}.png"
            )
            self._timer.start("plotting")
            save_optimisation_history_plot(
                self._history,
                path,
                dpi=self.problem.config.simulation.simulation.plot_dpi,
            )
            self._timer.stop("plotting")
        if (
            snapshot
            and self._best_changed_since_snapshot
            and self.problem.config.run.save_best_simulation
        ):
            best_directory = output_directory / "best_simulation"
            if best_directory.exists():
                self._timer.start("io")
                shutil.rmtree(best_directory)
                self._timer.stop("io")
            self._timer.start("best_simulation_output")
            best_snapshot_directory = save_simulation_snapshot(
                self.problem,
                self._best_record,
                best_directory,
                save_plots=self.problem.config.run.save_simulation_plots,
            )
            self._timer.stop("best_simulation_output")
            if (
                self.problem.config.run.archive_previous_best_simulations
                and self._last_archived_history_index != self._best_record.history_index
            ):
                archive_root = output_directory / "previous_best_simulations"
                archive_directory = archive_root / best_snapshot_directory.name
                self._timer.start("io")
                archive_root.mkdir(parents=True, exist_ok=True)
                shutil.copytree(
                    best_snapshot_directory,
                    archive_directory,
                    dirs_exist_ok=True,
                )
                self._timer.stop("io")
                self._last_archived_history_index = self._best_record.history_index
            self._best_changed_since_snapshot = False

    def _starting_designs(self) -> tuple[np.ndarray, ...]:
        restarts = self.problem.config.restarts
        random = np.random.default_rng(restarts.random_seed)
        starts: list[np.ndarray] = []
        number = restarts.number
        if self._resume_design is not None:
            completed_random_starts = max(
                0,
                len(self._restart_results) - int(restarts.include_base_design),
            )
            if completed_random_starts:
                random.uniform(
                    0.0,
                    1.0,
                    (completed_random_starts, self.problem.n_parameters),
                )
            starts.append(self._resume_design)
            number = max(1, restarts.number - len(self._restart_results))
        elif restarts.include_base_design:
            starts.append(np.asarray(self.problem.initial_parameters, dtype=np.float64))
        while len(starts) < number:
            starts.append(random.uniform(0.0, 1.0, self.problem.n_parameters))
        return tuple(starts[:number])

    def _elapsed(self) -> float:
        return self._prior_elapsed + time.monotonic() - self._start_time


class _JaxObjectiveEvaluator:
    """Cache the most recent device evaluation for SciPy and callbacks."""

    def __init__(self, problem: OptimisationProblem, timer: Timer):
        self._function = jax.jit(
            jax.value_and_grad(problem.objective_with_aux, has_aux=True)
        )
        self._cached_design: np.ndarray | None = None
        self._cached_result: tuple[float, np.ndarray, ObjectiveTerms] | None = None
        self.total_evaluations = 0
        self.restart_evaluations = 0
        self._timer = timer

    def reset_restart_count(self) -> None:
        self.restart_evaluations = 0

    def evaluate(self, design) -> tuple[float, np.ndarray, ObjectiveTerms]:
        design = np.asarray(design, dtype=np.float64)
        if self._cached_design is not None and np.array_equal(
            design, self._cached_design
        ):
            assert self._cached_result is not None
            return self._cached_result
        self._timer.start("optimisation_compute")
        (value, terms), gradient = self._function(jnp.asarray(design))
        value, gradient, terms = jax.device_get((value, gradient, terms))
        self._timer.stop("optimisation_compute")
        result = (
            float(value),
            np.asarray(gradient, dtype=np.float64),
            terms,
        )
        self._cached_design = design.copy()
        self._cached_result = result
        self.total_evaluations += 1
        self.restart_evaluations += 1
        return result

    def scipy_value_and_gradient(self, design) -> tuple[float, np.ndarray]:
        value, gradient, _ = self.evaluate(design)
        return value, gradient


class _WallTimeExceeded(Exception):
    pass


def _projected_gradient(design: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    tolerance = 1.0e-12
    blocked_lower = (design <= tolerance) & (gradient > 0.0)
    blocked_upper = (design >= 1.0 - tolerance) & (gradient < 0.0)
    return np.where(blocked_lower | blocked_upper, 0.0, gradient)
