"""Ease plotting production and fluid flow information."""
from __future__ import annotations

from typing import Union

import matplotlib.pyplot as plt
import matplotlib.scale as mscale
import matplotlib.ticker as ticker
import matplotlib.transforms as mtransforms
import numpy as np

from bluebonnet.flow import (
    IdealReservoir,
    MultiPhaseReservoir,
    SinglePhaseReservoir,
    TwoPhaseReservoir,
)

Reservoir = Union[
    IdealReservoir, SinglePhaseReservoir, TwoPhaseReservoir, MultiPhaseReservoir
]


class SquareRootScale(mscale.ScaleBase):
    """ScaleBase class for generating square root scale."""

    name = "squareroot"

    def __init__(self, axis, **kwargs):
        """Initialize for axis."""
        mscale.ScaleBase.__init__(self, axis, **kwargs)

    def set_default_locators_and_formatters(self, axis):
        """Set major and minor locators and formatters."""
        axis.set_major_locator(ticker.AutoLocator())
        axis.set_major_formatter(ticker.ScalarFormatter())
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):  # noqa: ARG
        """Do not allow negative values."""
        return max(0.0, vmin), vmax

    class SquareRootTransform(mtransforms.Transform):
        """Transform from linear to square root position."""

        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a):
            """Actual transform."""
            return np.array(a) ** 0.5

        def inverted(self):
            """Inverse transform."""
            return SquareRootScale.InvertedSquareRootTransform()

    class InvertedSquareRootTransform(mtransforms.Transform):
        """Inverted square-root transform."""

        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform(self, a):
            """Square everything."""
            return np.array(a) ** 2

        def inverted(self):
            """Square root it. (Inverse of inverse)."""
            return SquareRootScale.SquareRootTransform()

    def get_transform(self):
        """Get square root transform."""
        return self.SquareRootTransform()


mscale.register_scale(SquareRootScale)


def plot_pseudopressure(
    reservoir: Reservoir,
    every: int = 200,
    rescale: bool = False,
    ax: plt.Axes = None,
    x_max: float = 1,
    y_max: float | None = None,
) -> plt.Axes:
    """Plot pseudopressure versus distance over time.

    Parameters
    ----------
    reservoir : Reservoir
        simulated reservoir
    every : int, optional
        timesteps between a plotted pseudopressure profile, by default 200
    rescale : bool, optional
        if true, rescale pseudopressure by initial value, by default False
    ax : plt.Axes, optional
        axes to plot on, by default None
    x_max : float, optional
        maximum distance to plot, by default 1
    y_max : float | None, optional
        maximum pseudopressure to plot, by default None

    Returns
    -------
    plt.Axes
        pseudopressure plotted
    """
    if ax is None:
        _, ax = plt.subplots()
    x = np.linspace(1 / reservoir.nx, 1, reservoir.nx)
    pinit = reservoir.pseudopressure[0, -1]
    for i, p in enumerate(reservoir.pseudopressure):
        if i % every == 0:
            if rescale:
                pscale = (p - p[0]) / (pinit - p[0])
                ax.plot(x, pscale, color="steelblue")
            else:
                ax.plot(x, p, color="steelblue")
    ax.set(xlabel="x", ylabel="Pseudopressure", xlim=(0, x_max), ylim=(0, y_max))
    return ax


def plot_recovery_rate(
    reservoir: Reservoir, ax: plt.Axes | None = None, change_ticks: bool = False
) -> plt.Axes:
    """Plot recovery rate over time.

    Parameters
    ----------
    reservoir : Reservoir
        simulated reservoir
    ax : plt.Axes | None, optional
        axes to plot on, by default None
    change_ticks : bool, optional
        if true, set xticks, by default False

    Returns
    -------
    plt.Axes
        recovery rate, plotted
    """
    if ax is None:
        _, ax = plt.subplots()

    cumulative = reservoir.recovery_factor()
    rate = np.gradient(cumulative, reservoir.time)
    ax.plot(reservoir.time, rate)
    ax.set(
        xscale="log",
        yscale="log",
        ylim=(1.0e-4, None),
        xlim=(1e-6, max(reservoir.time)),
        xlabel="Scaled time",
        ylabel="Recovery rate",
    )
    if change_ticks:
        tick_locs = (np.logspace(-7, 0, 7) * np.sqrt(max(reservoir.time))) ** 2
        ax.set_xticks(tick_locs)
    return ax


def plot_recovery_factor(
    reservoir: Reservoir, ax: plt.Axes | None = None, change_ticks: bool = False
) -> plt.Axes:
    """Plot cumulatie recovery over time.

    Parameters
    ----------
    reservoir : Reservoir
        simulated reservoir
    ax : plt.Axes | None, optional
        axes to plot on, by default None
    change_ticks : bool, optional
        if true, set xticks, by default False

    Returns
    -------
    plt.Axes
        recovery rate, plotted
    """
    if ax is None:
        _, ax = plt.subplots()
    rf = reservoir.recovery_factor()
    time = reservoir.time
    ax.plot(time, rf)
    ax.set(
        xscale="squareroot",
        ylim=(0, None),
        xlim=(0, max(time)),
        xlabel="Scaled time",
        ylabel="Recovery factor",
    )
    if change_ticks:
        tick_locs = np.round((np.linspace(0, 1, 7) * np.sqrt(max(time))) ** 2, 1)
        ax.set_xticks(tick_locs)
    return ax
