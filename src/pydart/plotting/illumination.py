"""Surface-deposition plots for square, Mollweide, and 3D views."""

from __future__ import annotations

import numpy as np


def power_density(result):
    """Return deposited power per surface area in W m^-2 as a NumPy array."""
    return np.asarray(result.total / result.target.cell_areas)


def plot_deposition_square(
    result,
    ax=None,
    *,
    cmap="inferno",
    norm=None,
    add_colorbar=True,
):
    """Plot power density on rectangular azimuth/latitude axes."""
    import matplotlib.pyplot as plt

    if ax is None:
        figure, ax = plt.subplots(figsize=(8, 4.5))
    else:
        figure = ax.figure
    mesh = _angular_mesh(ax, result, cmap, norm)
    ax.set_xlabel("Azimuth, $\\phi$ [rad]")
    ax.set_ylabel("Latitude, $\\pi/2-\\theta$ [rad]")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi / 2, np.pi / 2)
    ax.set_title("Surface power density (rectangular)")
    if add_colorbar:
        figure.colorbar(mesh, ax=ax, label=r"Power density [W m$^{-2}$]")
    return figure, ax


def plot_deposition_mollweide(
    result,
    ax=None,
    *,
    cmap="inferno",
    norm=None,
    add_colorbar=True,
):
    """Plot power density using a Mollweide projection."""
    import matplotlib.pyplot as plt

    if ax is None:
        figure, ax = plt.subplots(
            figsize=(8, 5),
            subplot_kw={"projection": "mollweide"},
        )
    else:
        figure = ax.figure
    mesh = _angular_mesh(ax, result, cmap, norm)
    ax.tick_params(axis="x", labelbottom=False)
    ax.grid(True, color="white", alpha=0.25)
    ax.set_title("Surface power density (Mollweide)")
    if add_colorbar:
        figure.colorbar(mesh, ax=ax, label=r"Power density [W m$^{-2}$]")
    return figure, ax


def plot_deposition_sphere(
    result,
    ax=None,
    *,
    cmap="inferno",
    norm=None,
    add_colorbar=True,
):
    """Plot the spherical target with power density mapped onto its surface."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if ax is None:
        figure = plt.figure(figsize=(7, 6))
        ax = figure.add_subplot(111, projection="3d")
    else:
        figure = ax.figure
    density = power_density(result)
    norm = norm or mpl.colors.Normalize(
        vmin=float(np.nanmin(density)),
        vmax=float(np.nanmax(density)),
    )
    colormap = mpl.colormaps[cmap]
    coordinates = np.asarray(result.target.cartesian_coordinates)
    # Preserve the complete simulation mesh. Matplotlib assigns one color to
    # each surface polygon, so retaining every cell is the highest-fidelity
    # native rendering available without interpolating new physical data.
    coordinates = np.concatenate((coordinates, coordinates[:1]), axis=0)
    density = np.concatenate((density, density[:1]), axis=0)
    ax.plot_surface(
        coordinates[..., 0],
        coordinates[..., 1],
        coordinates[..., 2],
        facecolors=colormap(norm(density)),
        linewidth=0,
        antialiased=True,
        shade=False,
        rcount=coordinates.shape[0],
        ccount=coordinates.shape[1],
    )
    radius = float(np.asarray(result.target.radius))
    ax.set(xlim=(-radius, radius), ylim=(-radius, radius), zlim=(-radius, radius))
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Deposited power on target")
    if add_colorbar:
        scalar_map = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
        figure.colorbar(
            scalar_map,
            ax=ax,
            shrink=0.7,
            label=r"Power density [W m$^{-2}$]",
        )
    return figure, ax


def plot_per_beam_deposited_fractions(result, ax=None):
    """Plot intercepted power divided by incident power for every beam."""
    import matplotlib.pyplot as plt

    if ax is None:
        figure, ax = plt.subplots(figsize=(8, 4.5))
    else:
        figure = ax.figure
    deposited = np.asarray(result.unsmoothed_deposited_power_per_beam)
    incident = np.asarray(result.beams.powers)
    fractions = deposited / incident
    beam_number = np.arange(result.beams.n_beams)
    ax.scatter(
        beam_number,
        100.0 * fractions,
        s=24,
        marker="o",
        facecolors="dodgerblue",
        edgecolors="black",
        linewidths=0.5,
    )
    ax.set_xlabel("Beam array index")
    ax.set_ylabel("Deposited incident power [%]")
    ax.set_title("Per-beam intercepted power")
    ax.set_ylim(0.0, max(100.0, 105.0 * float(fractions.max())))
    return figure, ax


def plot_longitude_lineout(result, ax=None):
    """Plot equatorial and polar-angle-averaged power density versus phi."""
    import matplotlib.pyplot as plt

    if ax is None:
        figure, ax = plt.subplots(figsize=(7, 4.5))
    else:
        figure = ax.figure
    density = power_density(result)
    areas = np.asarray(result.target.cell_areas)
    coordinates = np.asarray(result.target.spherical_coordinates)
    mean_density = np.average(density, weights=areas)
    longitude_profile = np.sum(density * areas, axis=1) / np.sum(areas, axis=1)
    phi = coordinates[:, 0, 1]
    theta = coordinates[0, :, 2]
    equator_index = int(np.argmin(np.abs(theta - np.pi / 2.0)))
    equator_theta = theta[equator_index]
    ax.plot(
        np.rad2deg(phi),
        longitude_profile / mean_density,
        color="dodgerblue",
        linewidth=2.0,
        label=r"Average over $\theta$",
    )
    ax.plot(
        np.rad2deg(phi),
        density[:, equator_index] / mean_density,
        color="darkorange",
        linewidth=1.4,
        linestyle="--",
        label=rf"Equatorial cell ($\theta={np.rad2deg(equator_theta):.1f}^\circ$)",
    )
    _format_lineout_axes(
        ax,
        xlabel=r"Azimuth, $\phi$ [degrees]",
        title=r"Deposition versus $\phi$ (averaged over $\theta$)",
    )
    ax.set_xlim(-180.0, 180.0)
    ax.legend(fontsize="small")
    return figure, ax


def plot_latitude_lineout(result, ax=None):
    """Plot prime-meridian and azimuth-averaged density versus latitude."""
    import matplotlib.pyplot as plt

    if ax is None:
        figure, ax = plt.subplots(figsize=(7, 4.5))
    else:
        figure = ax.figure
    density = power_density(result)
    areas = np.asarray(result.target.cell_areas)
    coordinates = np.asarray(result.target.spherical_coordinates)
    mean_density = np.average(density, weights=areas)
    latitude_profile = np.sum(density * areas, axis=0) / np.sum(areas, axis=0)
    latitude = np.pi / 2.0 - coordinates[0, :, 2]
    phi = coordinates[:, 0, 1]
    meridian_index = int(np.argmin(np.abs(phi)))
    meridian_phi = phi[meridian_index]
    ax.plot(
        np.rad2deg(latitude[::-1]),
        (latitude_profile / mean_density)[::-1],
        color="dodgerblue",
        linewidth=2.0,
        label=r"Average over $\phi$",
    )
    ax.plot(
        np.rad2deg(latitude[::-1]),
        (density[meridian_index, :] / mean_density)[::-1],
        color="darkorange",
        linewidth=1.4,
        linestyle="--",
        label=rf"Prime-meridian cell ($\phi={np.rad2deg(meridian_phi):.1f}^\circ$)",
    )
    _format_lineout_axes(
        ax,
        xlabel=r"Latitude, $\pi/2-\theta$ [degrees]",
        title=r"Deposition versus latitude (averaged over $\phi$)",
    )
    ax.set_xlim(-90.0, 90.0)
    ax.legend(fontsize="small")
    return figure, ax


def plot_deposition_lineouts(result, axes=None):
    """Plot the complementary phi and latitude averages side by side."""
    import matplotlib.pyplot as plt

    if axes is None:
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    else:
        figure = axes[0].figure
    plot_longitude_lineout(result, axes[0])
    plot_latitude_lineout(result, axes[1])
    return figure, axes


def _format_lineout_axes(ax, *, xlabel, title):
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Average power density / global mean")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def _angular_mesh(ax, result, cmap, norm):
    density = power_density(result)
    n_phi, n_theta = density.shape
    phi_edges = np.linspace(-np.pi, np.pi, n_phi + 1)
    latitude_edges = np.linspace(-np.pi / 2, np.pi / 2, n_theta + 1)
    return ax.pcolormesh(
        phi_edges,
        latitude_edges,
        density[:, ::-1].T,
        shading="flat",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
