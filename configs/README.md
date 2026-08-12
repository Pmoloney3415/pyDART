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

`visibility_smoothing_epsilon` regularizes the surface deposition map used for
RMS non-uniformity and spherical harmonics. Deposited power and deposited
fraction are calculated separately with exact hard visibility, so smoothing
does not create artificial intercepted power in those metrics or objectives.

Optimization run controls live in `[optimisation]`. Set `solver` to
`"scipy_lbfgsb"` (the default) or `"jaxopt_lbfgsb"`. Both L-BFGS-B backends
use the normalized `[0, 1]` design bounds, with configurable objective and
projected-gradient tolerances, wall time, checkpoint intervals, and
history-plot intervals. `[optimisation.restarts]` controls the total number of
starts, whether the base design is included, and the reproducible random seed.
For JAXopt, `device_iteration_chunk_size` controls the maximum number of
accepted iterations compiled and dispatched together before diagnostics return
to the host. It defaults to `10`; checkpoint and plot boundaries can shorten a
chunk without triggering a separate compilation.

When `archive_previous_best_simulations = true`, a complete snapshot is saved
under `previous_best_simulations/` whenever a checkpoint observes a new global
best. This is useful for small design studies; larger production runs will
usually leave it disabled. `best_simulation/` is always the stable location of
the most recently checkpointed global best when `save_best_simulation = true`.
Each completed restart also writes its best full snapshot beneath
`restart_best_simulations/restart_N/`. The globally best design therefore
appears both there and in the stable `best_simulation/` location.
Snapshot HDF5 and JSON data are always written when best-simulation saving is
enabled. Set `save_simulation_plots = false` to omit the key-plot PNGs while
retaining that data for later plotting. The setting defaults to `true`.
An interrupted run can be restarted from its best saved design with
`pydart-optimise CONFIG --resume CHECKPOINT`. This preserves the recorded
history, but starts a fresh L-BFGS approximation because optimizer state is not
currently serialized.
