"""Forecast when bottomhole/fracface pressure is known and varying."""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from lmfit import Minimizer, Parameters
from numpy.typing import NDArray

from bluebonnet.flow import FlowProperties, SinglePhaseReservoir


def _obj_function(
    params: Parameters,
    days: NDArray,
    production: NDArray,
    pvt_table: pd.DataFrame,
    pressure_fracface: NDArray,
) -> NDArray:
    """Calculate mismatch between scaling solution and production.

    Parameters
    ----------
    params : Parameters
        Fitting parameters
    days : NDArray
        days well is on production
    production : NDArray
        observed cumulative production over time
    pvt_table : pd.DataFrame
        pressure-volume-temperature data table
    pressure_fracface : NDArray
        pressure at the fracture face over time

    Returns
    -------
    NDArray
        mismatch between model and cumulative production
    """
    tau = params["tau"].value
    resource_in_place = params["M"].value
    pressure_initial = params["p_initial"].value
    # pressure_fracface = pressure_initial
    # print(
    #     " tau is {:7.5g}, pressure_initial is {:7.5g} and M is {:7.5g}".format(
    #         tau, pressure_initial, resource_in_place
    #     )
    # )
    t = days / tau
    flow_propertiesM = FlowProperties(pvt_table, pressure_initial)
    res_realgasM = SinglePhaseReservoir(
        80, pressure_initial, pressure_initial, flow_propertiesM
    )
    res_realgasM.simulate(t, pressure_fracface=pressure_fracface)
    recovery_factor = res_realgasM.recovery_factor()
    return resource_in_place * recovery_factor - production


def fit_production_pressure(
    prod_data: pd.DataFrame,
    pvt_table: pd.DataFrame,
    pressure_initial: float,
    filter_window_size: int | None = None,
    pressure_imax: float = 15000,
    inplace_max: float = 100000,
    filter_zero_prod_days: bool = True,
    n_iter: int = 100,
    params: Parameters | None = None,
) -> Parameters:
    """Fit cumulative production given fracface pressure.

    Parameters
    ----------
    prod_data : pd.DataFrame
        contains columns 'Days', 'Gas', and 'Pressure'
    pvt_table : pd.DataFrame
        information on equation of state, for example from build_pvt_gas
    pressure_initial : float
        guess for initial reservoir pressure
    filter_window_size : int or None
        If not None, boxcar filter size to average pressure data
    pressure_imax : float, Optional
        maximum allowed initial reservoir pressure. pvt had better include this pressure
    inplace_max : float
        Maximum allowed resource in place
    filter_zero_prod_days : bool
        Filter out days without gas production or pressure value. Also
        shortens days on production to only include productive days.
    n_iter : integer, default to 100
        number of times to iterate until stabilizes
    params : Parameters
        Initial guesses for tau, M, and p_initial. You can pass in results from previous fit.

    Returns
    -------
    params : Parameters
        Best fits for tau, M, and p_initial
    """
    if filter_zero_prod_days:
        prod_data = prod_data[
            (prod_data["Gas"] > 0) & (pd.notna(prod_data["Pressure"]))
        ][["Days", "Gas", "Pressure"]]
    else:
        prod_data = prod_data[["Days", "Gas", "Pressure"]]

    time = np.arange(0, len(prod_data["Days"]))
    pressure_fracface = np.array(prod_data["Pressure"])

    # with noisy data, sometimes a boxcar filter is beneficial
    if filter_window_size is not None:
        pressure_fracface = sp.ndimage.uniform_filter1d(
            pressure_fracface, size=filter_window_size
        )
    cumulative_prod = np.cumsum(np.array(prod_data["Gas"]))

    if params is None:
        params = Parameters()
        params.add(
            "tau", value=1000.0, min=30.0, max=time[len(time) - 1] * 2
        )  # units: days
        params.add(
            "M",
            value=cumulative_prod[-1],
            min=cumulative_prod[len(cumulative_prod) - 2],
            max=inplace_max,
        )
        params.add(
            "p_initial",
            value=pressure_initial,
            min=max(pressure_fracface),
            max=pressure_imax,
        )  # units: psi
    mini = Minimizer(
        _obj_function,
        params,
        fcn_args=(time, cumulative_prod, pvt_table, pressure_fracface),
    )
    result = mini.minimize(method="Nelder", max_nfev=n_iter)

    return result


def plot_production_comparison(
    prod_data: pd.DataFrame,
    pvt_table: pd.DataFrame,
    params: Parameters,
    filter_window_size: int | None = None,
    filter_zero_prod_days: bool = True,
    well_name: str = "Well Name",
) -> Any:
    """Compare production to match.

    Parameters
    ----------
    prod_data : pd.DataFrame
        contains columns 'Days', 'Gas', and 'Pressure'
    pvt_table : pd.DataFrame
        information on equation of state, for example from build_pvt_gas
    params : Parameters
        fit result parameters
    filter_window_size : int or None
        If not None, boxcar filter size to average pressure data
    filter_zero_prod_days : bool
        Filter out days without gas production or pressure value. Also
        shortens days on production to only include productive days.
    params : Parameters
        Best fits for tau, M, and p_initial
    well_name : str
        name to label production with

    Returns
    -------
    fig, (ax1, ax2): Any
        matplotlib figure and tuple of axes with cumulative production and
        pressure over time (scaled by tau)
    """
    if filter_zero_prod_days:
        prod_data = prod_data[
            (prod_data["Gas"] > 0) & (pd.notna(prod_data["Pressure"]))
        ][["Days", "Gas", "Pressure"]]
        time = np.arange(len(prod_data["Days"]))
    else:
        prod_data = prod_data[["Days", "Gas", "Pressure"]]
        time = prod_data["Days"]

    pressure_fracface = np.array(prod_data["Pressure"])
    #
    if filter_window_size is not None:
        pressure_fracface = sp.ndimage.uniform_filter1d(
            pressure_fracface, size=filter_window_size
        )
    #
    cumulative_prod = np.cumsum(np.array(prod_data["Gas"]))

    resource_in_place = params["M"].value
    tau = params["tau"].value
    pressure_initial = params["p_initial"].value
    # pressure_fracface = pressure_initial

    flow_propertiesM = FlowProperties(pvt_table, pressure_initial)
    res_realgasM = SinglePhaseReservoir(
        80, pressure_fracface, pressure_initial, flow_propertiesM
    )
    res_realgasM.simulate(time / tau, pressure_fracface=pressure_fracface)

    rf2M = res_realgasM.recovery_factor()
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.set_size_inches(5, 6)
    ax1.plot(
        time / tau,
        rf2M,
        "--",
        label=f"Production; tau={tau:7.5g}, M={resource_in_place:7.5g}",
    )
    ax1.plot(time / tau, cumulative_prod / resource_in_place, label=well_name)
    ax1.legend()
    ax1.set(
        xlabel="Time",
        ylabel="Cumulative Production",
        ylim=(0, None),
        xscale="squareroot",
        xlim=(0, None),
    )

    ax2.plot(time / tau, pressure_fracface, label="Pressure (psi)")
    ax2.legend()
    ax2.set(
        xlabel="Time",
        ylabel="Pressure (psi)",
        ylim=(0, None),
        xscale="squareroot",
        xlim=(0, None),
    )

    return fig, (ax1, ax2)
