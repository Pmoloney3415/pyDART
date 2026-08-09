"""Plots of beam geometry on the spherical target."""

from __future__ import annotations

import numpy as np


def plot_beam_geometry_mollweide(result, ax=None):
    """Plot physical beam origins and centroid intersections on a Mollweide map."""
    import matplotlib.pyplot as plt

    if ax is None:
        figure, ax = plt.subplots(
            figsize=(8, 5),
            subplot_kw={"projection": "mollweide"},
        )
    else:
        figure = ax.figure

    origins = np.asarray(result.beams.physical_origins)
    directions = np.asarray(result.beams.directions)
    numerical_origins = np.asarray(result.beams.origins)
    radius = float(np.asarray(result.target.radius))
    intersections, valid = _target_intersections(
        numerical_origins,
        directions,
        radius,
    )
    origin_vectors = origins / np.linalg.norm(origins, axis=-1, keepdims=True)
    intersection_vectors = intersections / radius

    for origin, intersection, is_valid in zip(
        origin_vectors,
        intersection_vectors,
        valid,
        strict=True,
    ):
        if not is_valid:
            continue
        longitude, latitude = _great_circle(origin, intersection)
        longitude, latitude = _split_at_seam(longitude, latitude)
        ax.plot(
            longitude,
            latitude,
            linestyle="--",
            linewidth=0.7,
            color="0.35",
            alpha=0.7,
            zorder=1,
        )

    origin_phi, origin_latitude = _angles(origin_vectors)
    point_phi, point_latitude = _angles(intersection_vectors[valid])
    ax.scatter(
        origin_phi,
        origin_latitude,
        s=70,
        marker="o",
        facecolors="white",
        edgecolors="black",
        linewidths=0.8,
        label="Port origin",
        zorder=2,
    )
    ax.scatter(
        point_phi,
        point_latitude,
        s=38,
        marker="X",
        facecolors="dodgerblue",
        edgecolors="black",
        linewidths=0.6,
        label="Centroid at target",
        zorder=3,
    )
    ax.set_title("Beam origins and target intersections")
    ax.tick_params(axis="x", labelbottom=False)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower center", fontsize="small")
    return figure, ax


def _target_intersections(origins, directions, radius):
    along_axis = np.einsum("bi,bi->b", origins, directions)
    discriminant = along_axis**2 - (np.einsum("bi,bi->b", origins, origins) - radius**2)
    valid = discriminant >= 0.0
    distance = -along_axis - np.sqrt(np.maximum(discriminant, 0.0))
    valid &= distance >= 0.0
    return origins + distance[:, None] * directions, valid


def _angles(unit_vectors):
    phi = np.arctan2(unit_vectors[:, 1], unit_vectors[:, 0])
    latitude = np.arcsin(np.clip(unit_vectors[:, 2], -1.0, 1.0))
    return phi, latitude


def _great_circle(start, end, samples=48):
    cosine = np.clip(np.dot(start, end), -1.0, 1.0)
    angle = np.arccos(cosine)
    fraction = np.linspace(0.0, 1.0, samples)
    if angle < 1.0e-10:
        vectors = np.broadcast_to(start, (samples, 3))
    else:
        vectors = (
            np.sin((1.0 - fraction) * angle)[:, None] * start
            + np.sin(fraction * angle)[:, None] * end
        ) / np.sin(angle)
    return _angles(vectors)


def _split_at_seam(longitude, latitude):
    seam = np.flatnonzero(np.abs(np.diff(longitude)) > np.pi) + 1
    longitude = longitude.astype(float, copy=True)
    latitude = latitude.astype(float, copy=True)
    longitude[seam] = np.nan
    latitude[seam] = np.nan
    return longitude, latitude
