from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from bluebonnet.flow import FlowProperties
from bluebonnet.plotting import (
    SinglePhaseReservoir,
    plot_pseudopressure,
    plot_recovery_factor,
    plot_recovery_rate,
)

nx = 30
nt = 1000
pf = 100.0
pi = 2e3
columns_renamer_gas = {
    "P": "pressure",
    "Z-Factor": "z-factor",
    "Cg": "compressibility",
    "Viscosity": "viscosity",
    "Density": "density",
}
pvt_gas = pd.read_csv("tests/data/pvt_gas.csv").rename(columns=columns_renamer_gas)
fluid = FlowProperties(pvt_gas, pi)


@pytest.fixture()
def reservoir():
    reservoir = SinglePhaseReservoir(
        nx, pressure_fracface=pf, pressure_initial=pi, fluid=fluid
    )
    t_end = 11
    time_scaled = np.linspace(0, np.sqrt(t_end), nt) ** 2
    reservoir.simulate(time_scaled)
    return reservoir


@pytest.mark.mpl_image_compare()
@pytest.mark.parametrize("rescale", [True, False])
def test_plot_pseudopressure(reservoir, rescale):
    fig, ax = plt.subplots()
    plot_pseudopressure(reservoir, every=50, ax=ax, rescale=rescale)
    return fig


@pytest.mark.mpl_image_compare()
@pytest.mark.parametrize("change_ticks", [True, False])
def test_plot_recovery_rate(reservoir, change_ticks):
    fig, ax = plt.subplots()
    plot_recovery_rate(reservoir, ax, change_ticks=change_ticks)
    return fig


@pytest.mark.mpl_image_compare()
@pytest.mark.parametrize("change_ticks", [True, False])
def test_plot_recovery_factor(reservoir, change_ticks):
    fig, ax = plt.subplots()
    plot_recovery_factor(reservoir, ax, change_ticks=change_ticks)
    return fig
