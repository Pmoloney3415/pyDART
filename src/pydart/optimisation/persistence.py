"""Checkpoint and snapshot persistence for optimization runs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import numpy as np

if TYPE_CHECKING:
    from pydart.optimisation.optimise import (
        IterationRecord,
        OptimisationResult,
        RestartResult,
    )
    from pydart.optimisation.problem import OptimisationProblem


def load_optimisation_checkpoint(path: str | Path):
    """Load accepted history and restart summaries for approximate resumption."""
    try:
        import h5py
    except ImportError as error:
        raise ImportError("Optimization checkpoints require 'h5py'.") from error

    from pydart.optimisation.optimise import IterationRecord, RestartResult

    path = Path(path)
    with h5py.File(path, "r") as handle:
        history = handle["history"]
        count = history["objective"].shape[0]
        records = tuple(
            IterationRecord(
                restart_index=int(history["restart_index"][index]),
                iteration=int(history["iteration"][index]),
                history_index=int(history["history_index"][index]),
                function_evaluations=int(history["function_evaluations"][index]),
                elapsed_seconds=float(history["elapsed_seconds"][index]),
                objective=float(history["objective"][index]),
                rms_contribution=float(history["rms_contribution"][index]),
                mode_contribution=float(history["mode_contribution"][index]),
                deposition_contribution=float(
                    history["deposition_contribution"][index]
                ),
                rms_nonuniformity=float(history["rms_nonuniformity"][index]),
                deposited_capacity_fraction=float(
                    history["deposited_capacity_fraction"][index]
                ),
                gradient_norm=float(history["gradient_norm"][index]),
                projected_gradient_norm=float(
                    history["projected_gradient_norm"][index]
                ),
                design=np.asarray(history["design"][index]),
                normalized_power_by_l=np.asarray(
                    history["normalized_power_by_l"][index]
                ),
            )
            for index in range(count)
        )
        restart_results = []
        for name in sorted(handle["restarts"], key=int):
            group = handle["restarts"][name]
            restart_results.append(
                RestartResult(
                    restart_index=int(name),
                    success=bool(group.attrs["success"]),
                    status=int(group.attrs["status"]),
                    message=str(group.attrs["message"]),
                    iterations=int(group.attrs["iterations"]),
                    function_evaluations=int(group.attrs["function_evaluations"]),
                    best_objective=float(group.attrs["best_objective"]),
                    best_design=np.asarray(group["best_design"]),
                )
            )
        elapsed_seconds = float(handle.attrs["elapsed_seconds"])
    if not records:
        raise ValueError("The checkpoint contains no evaluated designs.")
    best_record = min(records, key=lambda record: record.objective)
    return records, tuple(restart_results), best_record, elapsed_seconds


def optimisation_output_directory(problem: OptimisationProblem) -> Path:
    """Return the indexed output directory for one optimization problem."""
    run = problem.config.run
    return Path(run.output_directory) / f"optimisation_{run.index}"


def save_optimisation_checkpoint(
    problem: OptimisationProblem,
    history: Sequence[IterationRecord],
    best_record: IterationRecord,
    restart_results: Sequence[RestartResult],
    elapsed_seconds: float,
) -> Path:
    """Atomically rewrite the recoverable HDF5 checkpoint."""
    try:
        import h5py
    except ImportError as error:
        raise ImportError("Optimization checkpoints require 'h5py'.") from error

    output_directory = optimisation_output_directory(problem)
    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / f"optimisation_checkpoint_{problem.config.run.index}.h5"
    temporary_path = path.with_suffix(".h5.tmp")
    records = list(history)

    with h5py.File(temporary_path, "w") as handle:
        handle.attrs["format"] = "pyDART optimization checkpoint"
        handle.attrs["optimisation_index"] = problem.config.run.index
        handle.attrs["elapsed_seconds"] = elapsed_seconds
        handle.attrs["resume_semantics"] = (
            "Restart L-BFGS-B from saved best design; internal Hessian history "
            "is not serialized."
        )
        configuration = handle.create_group("configuration")
        configuration.create_dataset(
            "optimisation_toml",
            data=problem.config.source_path.read_text(encoding="utf-8"),
        )
        configuration.create_dataset(
            "simulation_toml",
            data=problem.config.run.simulation_config.read_text(encoding="utf-8"),
        )
        configuration.create_dataset(
            "parameter_names",
            data=np.asarray(problem.parameter_names, dtype="S"),
        )

        history_group = handle.create_group("history")
        scalar_fields = (
            "restart_index",
            "iteration",
            "history_index",
            "function_evaluations",
            "elapsed_seconds",
            "objective",
            "rms_contribution",
            "mode_contribution",
            "deposition_contribution",
            "rms_nonuniformity",
            "deposited_capacity_fraction",
            "gradient_norm",
            "projected_gradient_norm",
        )
        for name in scalar_fields:
            history_group.create_dataset(
                name,
                data=np.asarray([getattr(record, name) for record in records]),
            )
        history_group.create_dataset(
            "design",
            data=np.stack([record.design for record in records]),
            compression="gzip",
            shuffle=True,
        )
        history_group.create_dataset(
            "normalized_power_by_l",
            data=np.stack([record.normalized_power_by_l for record in records]),
            compression="gzip",
            shuffle=True,
        )

        _write_design_state(handle.create_group("current"), problem, records[-1])
        _write_design_state(handle.create_group("global_best"), problem, best_record)
        restarts = handle.create_group("restarts")
        for result in restart_results:
            group = restarts.create_group(str(result.restart_index))
            group.attrs["success"] = result.success
            group.attrs["status"] = result.status
            group.attrs["message"] = result.message
            group.attrs["iterations"] = result.iterations
            group.attrs["function_evaluations"] = result.function_evaluations
            group.attrs["best_objective"] = result.best_objective
            group.create_dataset("best_design", data=result.best_design)

    temporary_path.replace(path)
    return path


def save_optimisation_summary(result: OptimisationResult) -> Path:
    """Write a compact JSON summary of the completed optimization."""
    output_directory = optimisation_output_directory(result.problem)
    output_directory.mkdir(parents=True, exist_ok=True)
    path = (
        output_directory
        / f"optimisation_summary_{result.problem.config.run.index}.json"
    )
    best = result.best_record
    data = {
        "optimisation_index": result.problem.config.run.index,
        "success": result.success,
        "message": result.message,
        "elapsed_seconds": result.elapsed_seconds,
        "completed_restarts": len(result.restart_results),
        "best_restart": best.restart_index + 1,
        "best_iteration": best.iteration,
        "best_objective": best.objective,
        "best_rms_nonuniformity": best.rms_nonuniformity,
        "best_deposited_capacity_fraction": best.deposited_capacity_fraction,
        "best_design": best.design.tolist(),
        "parameter_names": list(result.problem.parameter_names),
        "restart_results": [
            {
                "restart_index": restart.restart_index + 1,
                "success": restart.success,
                "status": restart.status,
                "message": restart.message,
                "iterations": restart.iterations,
                "function_evaluations": restart.function_evaluations,
                "best_objective": restart.best_objective,
            }
            for restart in result.restart_results
        ],
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def save_simulation_snapshot(
    problem: OptimisationProblem,
    record: IterationRecord,
    output_directory: str | Path,
    *,
    save_plots: bool = True,
) -> Path:
    """Save full simulation data and metrics, optionally including key plots."""
    from pydart.io.results import save_deposition_result, save_metrics_result

    output_directory = Path(output_directory)
    restart_number = record.restart_index + 1
    label = f"restart{restart_number}_index_{record.iteration}"
    simulation = replace(
        problem.simulation(record.design),
        simulation_index=record.iteration,
    )
    result = simulation.run()
    metrics = result.get_metrics()
    metadata = {
        "optimisation_restart": restart_number,
        "accepted_iteration": record.iteration,
        "global_history_index": record.history_index,
        "function_evaluations": record.function_evaluations,
    }
    save_deposition_result(
        result,
        output_directory,
        simulation_label=label,
        metadata=metadata,
    )
    save_metrics_result(
        metrics,
        output_directory,
        simulation_label=label,
        metadata=metadata,
    )
    if save_plots:
        from pydart.plotting import save_key_plots

        save_key_plots(
            result,
            metrics,
            output_directory,
            dpi=problem.config.simulation.simulation.plot_dpi,
            simulation_label=label,
        )
    simulation_directory = output_directory / f"simulation_{label}"
    metadata_path = simulation_directory / "optimisation_snapshot.json"
    metadata_path.write_text(
        json.dumps(
            {
                "restart_index": restart_number,
                "iteration": record.iteration,
                "history_index": record.history_index,
                "objective": record.objective,
                "rms_nonuniformity": record.rms_nonuniformity,
                "deposited_capacity_fraction": (record.deposited_capacity_fraction),
                "design": record.design.tolist(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return simulation_directory


def _write_design_state(group, problem, record) -> None:
    parameters = problem.beam_parameters(record.design)
    group.attrs["objective"] = record.objective
    group.attrs["restart_index"] = record.restart_index
    group.attrs["iteration"] = record.iteration
    group.create_dataset("design", data=record.design)
    group.create_dataset(
        "physical_origins", data=np.asarray(jax.device_get(parameters.physical_origins))
    )
    group.create_dataset(
        "pointing_locations",
        data=np.asarray(jax.device_get(parameters.pointing_locations)),
    )
    group.create_dataset(
        "power_fractions_of_maximum",
        data=np.asarray(jax.device_get(parameters.power_fractions_of_maximum)),
    )
    group.create_dataset(
        "spot_widths", data=np.asarray(jax.device_get(parameters.spot_widths))
    )
    group.create_dataset(
        "spot_rotations",
        data=np.asarray(jax.device_get(parameters.spot_rotations)),
    )
    group.create_dataset(
        "supergaussian_indices",
        data=np.asarray(jax.device_get(parameters.supergaussian_indices)),
    )
