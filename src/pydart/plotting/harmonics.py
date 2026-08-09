"""Plots of spherical-harmonic mode content."""

from __future__ import annotations

import numpy as np


def plot_mode_power_by_l(metrics, ax=None):
    """Plot mode power as a percentage of the monopole power."""
    import matplotlib.pyplot as plt

    if ax is None:
        figure, ax = plt.subplots(figsize=(7, 4.5))
    else:
        figure = ax.figure
    ell = np.asarray(metrics.ell)[1:]
    percentage = 100.0 * np.asarray(metrics.normalized_power_by_l)[1:]
    maximum = float(np.max(percentage))
    if maximum > 0.0:
        y_max = 10.0 ** np.ceil(np.log10(maximum))
    else:
        y_max = 1.0
    y_min = y_max * 1.0e-4
    ax.semilogy(
        ell,
        np.maximum(percentage, y_min),
        marker="o",
        color="dodgerblue",
        markeredgecolor="black",
    )
    ax.set_xlabel(r"Spherical-harmonic degree $\ell$")
    ax.set_ylabel(r"Mode power, $100 P_\ell/P_0$ [%]")
    ax.set_title("Power by spherical-harmonic degree")
    ax.set_xticks(ell)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, which="both", alpha=0.3)
    return figure, ax


def plot_lm_amplitude_heatmap(metrics, ax=None, *, cmap="viridis"):
    """Plot coefficient magnitude as a percentage of the monopole amplitude."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if ax is None:
        figure, ax = plt.subplots(figsize=(8, 5))
    else:
        figure = ax.figure
    coefficients = np.asarray(metrics.harmonic_coefficients)
    monopole = abs(coefficients[0, metrics.l_max])
    percentage = 100.0 * np.abs(coefficients) / monopole
    ell = np.arange(metrics.l_max + 1)[:, None]
    m = np.arange(-metrics.l_max, metrics.l_max + 1)[None, :]
    percentage = np.ma.masked_where(np.abs(m) > ell, percentage)
    positive = percentage.compressed()
    positive = positive[positive > 0]
    maximum = float(positive.max())
    minimum = max(float(positive.min()), maximum * 1.0e-8)
    image = ax.imshow(
        percentage,
        origin="lower",
        aspect="auto",
        extent=(
            -metrics.l_max - 0.5,
            metrics.l_max + 0.5,
            -0.5,
            metrics.l_max + 0.5,
        ),
        cmap=cmap,
        norm=mpl.colors.LogNorm(vmin=minimum, vmax=maximum),
        interpolation="nearest",
    )
    ax.set_xlabel(r"Azimuthal order $m$")
    ax.set_ylabel(r"Degree $\ell$")
    ax.set_title(r"Harmonic amplitude, $100|a_{\ell m}|/|a_{00}|$")
    figure.colorbar(image, ax=ax, label="Amplitude relative to monopole [%]")
    return figure, ax
