"""Run the configurable six-beam facility design optimization."""

from pydart.config import load_optimisation_config
from pydart.optimisation import OptimisationProblem, OptimisationRunner

config = load_optimisation_config("configs/optimisations/six_beam_design.toml")
problem = OptimisationProblem(config)
result = OptimisationRunner(problem).run()

print(f"Best objective: {result.best_objective:.8e}")
print(f"Completed restarts: {len(result.restart_results)}")
print(result.message)
