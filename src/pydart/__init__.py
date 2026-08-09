"""Differentiable illumination ray tracing and beam optimisation."""

from importlib.metadata import PackageNotFoundError, version

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
