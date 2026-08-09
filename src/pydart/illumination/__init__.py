"""Solid-sphere illumination forward model."""

from pydart.illumination.deposition import DepositionResult, calculate_deposition
from pydart.illumination.simulation import simulate_illumination

__all__ = ["DepositionResult", "calculate_deposition", "simulate_illumination"]
