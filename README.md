# pyDART

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![CI](https://github.com/Pmoloney3415/pyDART/actions/workflows/ci.yml/badge.svg)](https://github.com/Pmoloney3415/pyDART/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/Pmoloney3415/pyDART/branch/main/graph/badge.svg)](https://codecov.io/gh/Pmoloney3415/pyDART)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green)](LICENSE)

pyDART (*Python Differentially Accelerated Ray Tracer*) is a differentiable
illumination ray tracer for studying and optimising high-power laser beam
configurations for inertial-confinement fusion.

The package uses JAX for automatic differentiation and accelerated numerical
execution. It currently models illumination of a solid spherical target and
provides tools for beam-layout optimisation, spherical-harmonic analysis,
result persistence, and plotting.

## Requirements

- Python 3.11, 3.12, or 3.13
- [uv](https://docs.astral.sh/uv/) for the recommended setup

The locked default environment installs CPU-compatible JAX. Accelerator users
should follow the [official JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
for their platform and CUDA or ROCm version.

## Install from a GitHub checkout

Clone the repository and reproduce the locked development environment:

```shell
git clone https://github.com/Pmoloney3415/pyDART.git
cd pyDART
uv sync --locked
```

Run commands through uv without activating the environment:

```shell
uv run pydart-simulate --help
uv run pydart-optimise --help
```

An ordinary editable installation is also supported as a fallback:

```shell
python -m venv .venv
python -m pip install -e .
```

Activate that environment before invoking the `pydart-simulate` or
`pydart-optimise` commands.

## Run a simulation

Simulation behaviour and output are controlled by a TOML input deck. From the
repository root, run the bundled six-beam example with:

```shell
uv run pydart-simulate configs/simulations/six_beam_500um.toml
```

Results are written beneath the output directory selected in the deck. The
`configs/optimisations` directory contains corresponding optimisation decks;
these are research-scale examples and may run for substantially longer.
Every command-line run writes a JSON timing summary and preserves its input
TOML deck under the run's `used_configs/` directory. Optimisation runs preserve
both the optimisation deck and its referenced simulation deck.

The Python API exposes the same workflow:

```python
from pydart import initialise_simulation, load_config

config = load_config("configs/simulations/six_beam_500um.toml")
simulation = initialise_simulation(config)
result = simulation.run()
metrics = result.get_metrics()

print(float(metrics.deposited_fraction))
print(float(metrics.rms_nonuniformity))
```

## Run an optimisation

```shell
uv run pydart-optimise configs/optimisations/six_beam_design_scipy.toml
```

## Animate an optimization

Install the optional post-processing dependencies and render an optimization
whose previous best simulations were archived:

```shell
uv sync --extra postprocessing
python scripts/plot_optimisation_animation.py \
    results/optimisations/optimisation_5 --step 10 --fps 5
```

For a regular pip installation, use
`python -m pip install -e ".[postprocessing]"` instead.

The script saves six-panel PNG frames under `animation_frames/` and combines
them into `optimisation_history.mp4`. It requires the run to contain one
optimization checkpoint and snapshots produced with
`archive_previous_best_simulations = true`.

## Plot optimization results

```shell
# Overall-best beam parameters
python scripts/plot_optimisation_parameters.py results/optimisations/optimisation_80/

# Per-restart beam parameters
python scripts/plot_optimisation_restart_parameters.py results/optimisations/optimisation_80/

# Animation-style frame for the overall-best result
python scripts/plot_optimisation_best_result.py results/optimisations/optimisation_80/

# Animation-style frames for every restart best
python scripts/plot_optimisation_restart_results.py results/optimisations/optimisation_80/
```

Per-restart parameter images are written to `restart_parameter_plots/`, while
per-restart result frames are written to `restart_result_plots/`.

## Development checks

The uv lockfile is committed so contributors and CI use the same resolved
dependencies. After changing `pyproject.toml`, update it deliberately with
`uv lock`.

```shell
uv run pre-commit install
uv lock --check
uv run ruff format --check src tests examples
uv run ruff check src tests examples
uv run pytest --cov=pydart --cov-report=term-missing --cov-fail-under=80
uv build --no-sources
```

## Project layout

```text
src/         Installable Python package
tests/       Unit and differentiation tests
configs/     Simulation and optimisation input decks
examples/    Small Python usage examples
scripts/     Post-processing utilities
results/     Generated output (not version-controlled)
```

## License

pyDART is distributed under the [BSD 3-Clause License](LICENSE).
