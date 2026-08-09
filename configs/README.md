# pyDART

## Configs

The configs directory is to store simulation config files.

There is a separate folder both for:
- Initialising indiviudal simulations (for a given beam layout and target, computing the deposition map and modes).
- Initialising optimisations (Finding the optimal configuration, given bounds on input parameters).

Optimisation decks reference a base simulation deck using a path relative to
the optimisation file. They select independently variable quantities, their
bounds, optional frozen beam names, and objective weights. Physical port
origins remain at their configured radius while their angular positions may
be bounded or unconstrained on the facility sphere.

Optimization run controls live in `[optimisation]`. L-BFGS-B uses the
normalized `[0, 1]` design bounds, with configurable objective and projected
gradient tolerances, wall time, checkpoint intervals, and history-plot
intervals. `[optimisation.restarts]` controls the total number of starts,
whether the base design is included, and the reproducible random seed.

When `archive_previous_best_simulations = true`, a complete snapshot is saved
under `previous_best_simulations/` whenever a checkpoint observes a new global
best. This is useful for small design studies; larger production runs will
usually leave it disabled. `best_simulation/` is always the stable location of
the most recently checkpointed global best when `save_best_simulation = true`.
An interrupted run can be restarted from its best saved design with
`pydart-optimise CONFIG --resume CHECKPOINT`. This preserves the recorded
history, but starts a fresh L-BFGS approximation because SciPy does not expose
its internal inverse-Hessian memory for serialization.
