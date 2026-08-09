"""Differentiable illumination ray tracing and beam optimisation."""

from importlib.metadata import PackageNotFoundError, version

# pyDART's metric calculations intentionally use float64 and complex128.  JAX
# disables 64-bit values by default, and this process-wide option must be set
# before any pyDART functions are traced or arrays are created.
from jax import config as jax_config

try:
    jax_config.update("jax_enable_x64", True)
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "pyDART requires JAX 64-bit mode. Set JAX_ENABLE_X64=1 before importing JAX/pyDART."
    ) from exc

from pydart.config import PyDARTConfig, load_config
from pydart.simulation import Simulation, initialise_simulation

try:
    __version__ = version("pydart")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "PyDARTConfig",
    "Simulation",
    "__version__",
    "initialise_simulation",
    "load_config",
]
