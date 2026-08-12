"""Multi-start, checkpointed L-BFGS-B optimization runner."""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NamedTuple

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
    symmetry_contribution: float
    rms_ratio_power: float
    deposition_contribution: float
    rms_nonuniformity: float
    deposited_capacity_fraction: float
    gradient_norm: float
    projected_gradient_norm: float
    design: np.ndarray


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


@dataclass(frozen=True)
class _JaxoptHostValues:
    design: np.ndarray
    value: float
    gradient: np.ndarray
    terms: ObjectiveTerms
    error: float
    failed_linesearch: bool
    function_evaluations: int


class _JaxoptLoopState(NamedTuple):
    parameters: Any
    solver_state: Any
    previous_objective: Any
    function_evaluation_offset: Any
    accepted_iterations: Any
    status: Any


class _JaxoptIterationValues(NamedTuple):
    design: Any
    value: Any
    gradient: Any
    terms: ObjectiveTerms
    error: Any
    failed_linesearch: Any
    function_evaluations: Any
    valid: Any


@dataclass(frozen=True)
class _JaxoptBackend:
    solver: Any
    bounds: tuple[Any, Any]
    run_chunk: Any


class OptimisationRunner:
    """Run configured SciPy or JAXopt L-BFGS-B starts."""

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
        self._jaxopt_backend: _JaxoptBackend | None = None
        if resume_checkpoint is not None:
            history, restart_results, best, elapsed = load_optimisation_checkpoint(
                resume_checkpoint
            )
            if best.design.shape != (problem.n_parameters,):
                raise ValueError(
                    "Checkpoint design size does not match this optimization problem."
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
        self._timer = Timer()
        self._jaxopt_backend = None
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
            if self.problem.config.run.solver == "scipy_lbfgsb":
                restart_result, restart_stopped_for_time = self._run_scipy_restart(
                    evaluator, start, restart_index
                )
            else:
                restart_result, restart_stopped_for_time = self._run_jaxopt_restart(
                    evaluator, start, restart_index
                )
            stopped_for_time = stopped_for_time or restart_stopped_for_time
            self._restart_results.append(restart_result)
            self._save_restart_best(restart_result.restart_index)
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
                "solver": self.problem.config.run.solver,
            },
        )
        return result

    def _run_scipy_restart(
        self,
        evaluator: _JaxObjectiveEvaluator,
        start: np.ndarray,
        restart_index: int,
    ) -> tuple[RestartResult, bool]:
        from scipy.optimize import minimize

        evaluator.reset_restart_count()
        restart_best = self._record(
            evaluator,
            start,
            restart_index=restart_index,
            iteration=0,
        )
        self._persist_first_record()
        accepted_iterations = 0

        def callback(design):
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
            self._after_accepted_iteration(accepted_iterations)

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
                    "ftol": self.problem.config.run.objective_relative_tolerance,
                    "gtol": self.problem.config.run.projected_gradient_tolerance,
                    "maxls": 20,
                },
            )
            return (
                RestartResult(
                    restart_index=restart_index,
                    success=bool(scipy_result.success),
                    status=int(scipy_result.status),
                    message=str(scipy_result.message),
                    iterations=int(scipy_result.nit),
                    function_evaluations=int(scipy_result.nfev),
                    best_objective=restart_best.objective,
                    best_design=restart_best.design.copy(),
                ),
                False,
            )
        except _WallTimeExceeded:
            return (
                self._wall_time_restart_result(
                    restart_index,
                    accepted_iterations,
                    evaluator.restart_evaluations,
                    restart_best,
                ),
                True,
            )

    def _run_jaxopt_restart(
        self,
        evaluator: _JaxObjectiveEvaluator,
        start: np.ndarray,
        restart_index: int,
    ) -> tuple[RestartResult, bool]:
        run = self.problem.config.run
        evaluator.reset_restart_count()
        parameters = jnp.asarray(start, dtype=jnp.float64)
        backend = self._get_jaxopt_backend(evaluator, parameters)
        self._timer.start("optimisation_compute")
        solver_state = backend.solver.init_state(parameters, backend.bounds)
        host = _jaxopt_values_on_host(
            parameters, solver_state, function_evaluation_offset=0
        )
        self._timer.stop("optimisation_compute")
        evaluator.set_jaxopt_restart_evaluations(host.function_evaluations)
        restart_best = self._record_values(
            host.design,
            host.value,
            host.gradient,
            host.terms,
            function_evaluations=host.function_evaluations,
            restart_index=restart_index,
            iteration=0,
        )
        self._persist_first_record()
        initial_status = (
            _JAXOPT_PROJECTED_GRADIENT_CONVERGED
            if host.error <= run.projected_gradient_tolerance
            else _JAXOPT_RUNNING
        )
        loop_state = _JaxoptLoopState(
            parameters=parameters,
            solver_state=solver_state,
            previous_objective=solver_state.value,
            function_evaluation_offset=jnp.asarray(0, dtype=jnp.int32),
            accepted_iterations=jnp.asarray(0, dtype=jnp.int32),
            status=jnp.asarray(initial_status, dtype=jnp.int32),
        )

        try:
            while int(loop_state.status) == _JAXOPT_RUNNING:
                accepted_iterations = int(loop_state.accepted_iterations)
                steps_to_run = self._jaxopt_steps_to_next_host_event(
                    accepted_iterations
                )
                self._timer.start("optimisation_compute")
                loop_state, device_history = backend.run_chunk(
                    loop_state, jnp.asarray(steps_to_run, dtype=jnp.int32)
                )
                loop_state, device_history = jax.device_get(
                    (loop_state, device_history)
                )
                self._timer.stop("optimisation_compute")
                function_evaluations = _jaxopt_function_evaluations(loop_state)
                evaluator.set_jaxopt_restart_evaluations(function_evaluations)
                for index, valid in enumerate(np.asarray(device_history.valid)):
                    if not valid:
                        continue
                    accepted_iterations += 1
                    history_value = _jaxopt_history_value(device_history, index)
                    record = self._record_values(
                        history_value.design,
                        history_value.value,
                        history_value.gradient,
                        history_value.terms,
                        function_evaluations=history_value.function_evaluations,
                        restart_index=restart_index,
                        iteration=accepted_iterations,
                    )
                    if record.objective < restart_best.objective:
                        restart_best = record
                self._after_jaxopt_chunk(accepted_iterations)
        except _WallTimeExceeded:
            return (
                self._wall_time_restart_result(
                    restart_index,
                    int(loop_state.accepted_iterations),
                    _jaxopt_function_evaluations(loop_state),
                    restart_best,
                ),
                True,
            )

        success, status, message = _jaxopt_terminal_result(int(loop_state.status))
        return (
            _jaxopt_restart_result(
                restart_index,
                success,
                status,
                message,
                int(loop_state.accepted_iterations),
                _jaxopt_function_evaluations(loop_state),
                restart_best,
            ),
            False,
        )

    def _get_jaxopt_backend(
        self, evaluator: _JaxObjectiveEvaluator, parameters
    ) -> _JaxoptBackend:
        if self._jaxopt_backend is not None:
            return self._jaxopt_backend

        from jaxopt import LBFGSB

        run = self.problem.config.run
        solver = LBFGSB(
            evaluator.jax_value_and_gradient,
            value_and_grad=True,
            has_aux=True,
            maxiter=run.maximum_iterations,
            tol=run.projected_gradient_tolerance,
            maxls=20,
            stop_if_linesearch_fails=True,
            implicit_diff=False,
            jit=True,
        )
        bounds = (jnp.zeros_like(parameters), jnp.ones_like(parameters))
        self._jaxopt_backend = _JaxoptBackend(
            solver=solver,
            bounds=bounds,
            run_chunk=_build_jaxopt_chunk(
                solver,
                bounds,
                chunk_size=run.device_iteration_chunk_size,
                maximum_iterations=run.maximum_iterations,
                objective_relative_tolerance=run.objective_relative_tolerance,
                projected_gradient_tolerance=run.projected_gradient_tolerance,
            ),
        )
        return self._jaxopt_backend

    def _jaxopt_steps_to_next_host_event(self, accepted_iterations: int) -> int:
        run = self.problem.config.run
        steps_remaining = run.maximum_iterations - accepted_iterations
        steps_to_checkpoint = run.checkpoint_interval - (
            accepted_iterations % run.checkpoint_interval
        )
        steps_to_plot = run.history_plot_interval - (
            accepted_iterations % run.history_plot_interval
        )
        return min(
            run.device_iteration_chunk_size,
            steps_remaining,
            steps_to_checkpoint,
            steps_to_plot,
        )

    def _after_jaxopt_chunk(self, accepted_iterations: int) -> None:
        run = self.problem.config.run
        checkpoint = accepted_iterations % run.checkpoint_interval == 0
        history_plot = accepted_iterations % run.history_plot_interval == 0
        if checkpoint or history_plot:
            self._persist(
                checkpoint=checkpoint,
                history_plot=history_plot,
                snapshot=checkpoint,
            )
        if self._elapsed() >= run.maximum_wall_time_seconds:
            raise _WallTimeExceeded

    def _persist_first_record(self) -> None:
        if len(self._history) == 1:
            self._persist(checkpoint=True, history_plot=True, snapshot=True)

    def _after_accepted_iteration(self, accepted_iterations: int) -> None:
        checkpoint = (
            accepted_iterations % self.problem.config.run.checkpoint_interval == 0
        )
        history_plot = (
            accepted_iterations % self.problem.config.run.history_plot_interval == 0
        )
        if checkpoint or history_plot:
            self._persist(
                checkpoint=checkpoint,
                history_plot=history_plot,
                snapshot=checkpoint,
            )
        if self._elapsed() >= self.problem.config.run.maximum_wall_time_seconds:
            raise _WallTimeExceeded

    @staticmethod
    def _wall_time_restart_result(
        restart_index: int,
        accepted_iterations: int,
        function_evaluations: int,
        restart_best: IterationRecord,
    ) -> RestartResult:
        return RestartResult(
            restart_index=restart_index,
            success=False,
            status=2,
            message="Maximum wall time reached.",
            iterations=accepted_iterations,
            function_evaluations=function_evaluations,
            best_objective=restart_best.objective,
            best_design=restart_best.design.copy(),
        )

    def _record(
        self,
        evaluator: _JaxObjectiveEvaluator,
        design,
        *,
        restart_index: int,
        iteration: int,
    ) -> IterationRecord:
        value, gradient, terms = evaluator.evaluate(design)
        return self._record_values(
            design,
            value,
            gradient,
            terms,
            function_evaluations=evaluator.restart_evaluations,
            restart_index=restart_index,
            iteration=iteration,
        )

    def _record_values(
        self,
        design,
        value: float,
        gradient,
        terms: ObjectiveTerms,
        *,
        function_evaluations: int,
        restart_index: int,
        iteration: int,
    ) -> IterationRecord:
        design = np.asarray(design, dtype=np.float64).copy()
        gradient = np.asarray(gradient, dtype=np.float64)
        projected_gradient = _projected_gradient(design, gradient)
        record = IterationRecord(
            restart_index=restart_index,
            iteration=iteration,
            history_index=len(self._history),
            function_evaluations=function_evaluations,
            elapsed_seconds=self._elapsed(),
            objective=value,
            symmetry_contribution=float(terms.symmetry_contribution),
            rms_ratio_power=float(terms.rms_ratio_power),
            deposition_contribution=float(terms.deposition_contribution),
            rms_nonuniformity=float(terms.rms_nonuniformity),
            deposited_capacity_fraction=float(terms.deposited_capacity_fraction),
            gradient_norm=float(np.linalg.norm(gradient)),
            projected_gradient_norm=float(np.linalg.norm(projected_gradient)),
            design=design,
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

    def _save_restart_best(self, restart_index: int) -> None:
        """Write a stable full snapshot of one completed restart's best design."""
        if not self.problem.config.run.save_best_simulation:
            return
        restart_best = min(
            (
                record
                for record in self._history
                if record.restart_index == restart_index
            ),
            key=lambda record: record.objective,
        )
        output_directory = optimisation_output_directory(self.problem)
        restart_directory = (
            output_directory
            / "restart_best_simulations"
            / f"restart_{restart_index + 1}"
        )
        if restart_directory.exists():
            self._timer.start("io")
            shutil.rmtree(restart_directory)
            self._timer.stop("io")
        self._timer.start("best_simulation_output")
        save_simulation_snapshot(
            self.problem,
            restart_best,
            restart_directory,
            save_plots=self.problem.config.run.save_simulation_plots,
        )
        self._timer.stop("best_simulation_output")

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
    """Compile the shared objective and cache evaluations made for SciPy."""

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

    def set_jaxopt_restart_evaluations(self, count: int) -> None:
        delta = count - self.restart_evaluations
        self.total_evaluations += max(0, delta)
        self.restart_evaluations = count

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

    def jax_value_and_gradient(self, design):
        return self._function(design)


class _WallTimeExceeded(Exception):
    pass


_JAXOPT_RUNNING = 0
_JAXOPT_PROJECTED_GRADIENT_CONVERGED = 1
_JAXOPT_RELATIVE_OBJECTIVE_CONVERGED = 2
_JAXOPT_LINESEARCH_FAILED = 3
_JAXOPT_MAXIMUM_ITERATIONS = 4


def _projected_gradient(design: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    tolerance = 1.0e-12
    blocked_lower = (design <= tolerance) & (gradient > 0.0)
    blocked_upper = (design >= 1.0 - tolerance) & (gradient < 0.0)
    return np.where(blocked_lower | blocked_upper, 0.0, gradient)


def _build_jaxopt_chunk(
    solver,
    bounds,
    *,
    chunk_size: int,
    maximum_iterations: int,
    objective_relative_tolerance: float,
    projected_gradient_tolerance: float,
):
    def run_chunk(loop_state: _JaxoptLoopState, steps_to_run):
        def scan_step(state: _JaxoptLoopState, chunk_index):
            should_update = (
                (state.status == _JAXOPT_RUNNING)
                & (chunk_index < steps_to_run)
                & (state.accepted_iterations < maximum_iterations)
            )

            def update(_):
                step = solver.update(
                    state.parameters,
                    state.solver_state,
                    bounds,
                )
                raw_parameters, raw_solver_state = step.params, step.state
                clipped_parameters = jnp.clip(
                    raw_parameters,
                    bounds[0],
                    bounds[1],
                )
                bound_violation = jnp.maximum(
                    jnp.max(bounds[0] - raw_parameters),
                    jnp.max(raw_parameters - bounds[1]),
                )
                raw_function_evaluations = (
                    state.function_evaluation_offset + raw_solver_state.num_fun_eval
                )

                def reinitialise(_):
                    return (
                        solver.init_state(clipped_parameters, bounds),
                        raw_function_evaluations,
                    )

                def retain_history(_):
                    return raw_solver_state, state.function_evaluation_offset

                corrected_solver_state, function_evaluation_offset = jax.lax.cond(
                    bound_violation > 1.0e-12,
                    reinitialise,
                    retain_history,
                    operand=None,
                )
                accepted_iterations = state.accepted_iterations + 1
                scale = jnp.maximum(
                    1.0,
                    jnp.maximum(
                        jnp.abs(state.previous_objective),
                        jnp.abs(corrected_solver_state.value),
                    ),
                )
                relative_reduction = (
                    state.previous_objective - corrected_solver_state.value
                ) / scale
                relative_converged = (relative_reduction >= 0.0) & (
                    relative_reduction <= objective_relative_tolerance
                )
                projected_converged = (
                    corrected_solver_state.error <= projected_gradient_tolerance
                )
                status = jnp.asarray(_JAXOPT_RUNNING, dtype=jnp.int32)
                status = jnp.where(
                    relative_converged,
                    _JAXOPT_RELATIVE_OBJECTIVE_CONVERGED,
                    status,
                )
                status = jnp.where(
                    projected_converged,
                    _JAXOPT_PROJECTED_GRADIENT_CONVERGED,
                    status,
                )
                status = jnp.where(
                    raw_solver_state.failed_linesearch,
                    _JAXOPT_LINESEARCH_FAILED,
                    status,
                )
                status = jnp.where(
                    (status == _JAXOPT_RUNNING)
                    & (accepted_iterations >= maximum_iterations),
                    _JAXOPT_MAXIMUM_ITERATIONS,
                    status,
                )
                function_evaluations = (
                    function_evaluation_offset + corrected_solver_state.num_fun_eval
                )
                new_state = _JaxoptLoopState(
                    parameters=clipped_parameters,
                    solver_state=corrected_solver_state,
                    previous_objective=corrected_solver_state.value,
                    function_evaluation_offset=function_evaluation_offset,
                    accepted_iterations=accepted_iterations,
                    status=status,
                )
                values = _JaxoptIterationValues(
                    design=clipped_parameters,
                    value=corrected_solver_state.value,
                    gradient=corrected_solver_state.grad,
                    terms=corrected_solver_state.aux,
                    error=corrected_solver_state.error,
                    failed_linesearch=raw_solver_state.failed_linesearch,
                    function_evaluations=function_evaluations,
                    valid=jnp.asarray(True),
                )
                return new_state, values

            def skip(_):
                function_evaluations = (
                    state.function_evaluation_offset + state.solver_state.num_fun_eval
                )
                values = _JaxoptIterationValues(
                    design=state.parameters,
                    value=state.solver_state.value,
                    gradient=state.solver_state.grad,
                    terms=state.solver_state.aux,
                    error=state.solver_state.error,
                    failed_linesearch=jnp.asarray(False),
                    function_evaluations=function_evaluations,
                    valid=jnp.asarray(False),
                )
                return state, values

            return jax.lax.cond(should_update, update, skip, operand=None)

        return jax.lax.scan(
            scan_step,
            loop_state,
            jnp.arange(chunk_size, dtype=jnp.int32),
        )

    return jax.jit(run_chunk)


def _jaxopt_values_on_host(
    parameters, state, *, function_evaluation_offset: int
) -> _JaxoptHostValues:
    (
        design,
        value,
        gradient,
        terms,
        error,
        failed_linesearch,
        function_evaluations,
    ) = jax.device_get(
        (
            parameters,
            state.value,
            state.grad,
            state.aux,
            state.error,
            state.failed_linesearch,
            state.num_fun_eval,
        )
    )
    return _JaxoptHostValues(
        design=np.asarray(design, dtype=np.float64),
        value=float(value),
        gradient=np.asarray(gradient, dtype=np.float64),
        terms=terms,
        error=float(error),
        failed_linesearch=bool(failed_linesearch),
        function_evaluations=(function_evaluation_offset + int(function_evaluations)),
    )


def _jaxopt_history_value(history, index: int) -> _JaxoptHostValues:
    return _JaxoptHostValues(
        design=np.asarray(history.design[index], dtype=np.float64),
        value=float(history.value[index]),
        gradient=np.asarray(history.gradient[index], dtype=np.float64),
        terms=jax.tree.map(lambda values: values[index], history.terms),
        error=float(history.error[index]),
        failed_linesearch=bool(history.failed_linesearch[index]),
        function_evaluations=int(history.function_evaluations[index]),
    )


def _jaxopt_function_evaluations(loop_state: _JaxoptLoopState) -> int:
    return int(loop_state.function_evaluation_offset) + int(
        loop_state.solver_state.num_fun_eval
    )


def _jaxopt_terminal_result(status: int) -> tuple[bool, int, str]:
    if status == _JAXOPT_PROJECTED_GRADIENT_CONVERGED:
        return True, 0, "Projected gradient tolerance reached."
    if status == _JAXOPT_RELATIVE_OBJECTIVE_CONVERGED:
        return True, 0, "Relative objective tolerance reached."
    if status == _JAXOPT_LINESEARCH_FAILED:
        return False, 2, "JAXopt line search failed."
    if status == _JAXOPT_MAXIMUM_ITERATIONS:
        return False, 1, "Maximum iterations reached."
    raise RuntimeError(f"Unknown JAXopt terminal status: {status}.")


def _jaxopt_restart_result(
    restart_index: int,
    success: bool,
    status: int,
    message: str,
    iterations: int,
    function_evaluations: int,
    restart_best: IterationRecord,
) -> RestartResult:
    return RestartResult(
        restart_index=restart_index,
        success=success,
        status=status,
        message=message,
        iterations=iterations,
        function_evaluations=function_evaluations,
        best_objective=restart_best.objective,
        best_design=restart_best.design.copy(),
    )
