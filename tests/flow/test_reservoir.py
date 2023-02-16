"""Define a suite a tests for the reservoir module."""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
import pytest
from bluebonnet.flow import (
    FlowProperties,
    IdealReservoir,
    MultiPhaseReservoir,
    SinglePhaseReservoir,
)
from scipy.optimize import curve_fit

nx = (30,)
nt = (1200,)
pf = (100,)
pr = (8_000,)
columns_renamer_gas = {
    "P": "pressure",
    "Z-Factor": "z-factor",
    "Cg": "compressibility",
    "Viscosity": "viscosity",
    "Density": "density",
}
columns_renamer_oil = {
    "P": "pressure",
    "Z-Factor": "z-factor",
    "Co": "compressibility",
    "Oil_Viscosity": "viscosity",
    "Oil_Density": "density",
}
pvt_gas = pd.read_csv("tests/data/pvt_gas.csv").rename(columns=columns_renamer_gas)
pvt_oil = pd.read_csv("tests/data/pvt_oil.csv").rename(columns=columns_renamer_oil)
fluid = (FlowProperties(pvt_gas, pr[-1]),)  # FlowProperties(pvt_oil, pr[-1]))
sim_props = list(product(nx, pf, pr, fluid))
So, Sg, Sw = (0.7, 0.2, 0.1)
reservoirs = (IdealReservoir, SinglePhaseReservoir)  # TODO:, MultiPhaseReservoir)


@pytest.mark.parametrize("Reservoir", reservoirs)
@pytest.mark.parametrize(("nx", "pf", "pi", "fluid"), sim_props)
def reservoir_start(nx, pf, pi, fluid, Reservoir):
    if Reservoir == MultiPhaseReservoir:
        reservoir = Reservoir(nx, pf, pi, fluid, So, Sg, Sw)
        assert reservoir.So == So
        assert reservoir.Sg == Sg
        assert reservoir.Sw == Sw
    else:
        reservoir = Reservoir(nx, pf, pi, fluid)

    assert reservoir.nx == nx
    assert reservoir.pressure_fracface == pf
    assert reservoir.pressure_initial == pi
    assert reservoir.fluid == fluid


@pytest.mark.parametrize("nt", nt)
@pytest.mark.parametrize("Reservoir", reservoirs)
@pytest.mark.parametrize("nx,pf,pi,fluid", sim_props)
class TestRun:
    def test_rf_fails_early(self, nx, pf, pi, fluid, Reservoir, nt):  # noqa: ARG002
        if Reservoir == MultiPhaseReservoir:
            reservoir = Reservoir(nx, pf, pi, fluid, So, Sg, Sw)
        else:
            reservoir = Reservoir(nx, pf, pi, fluid)
        with pytest.raises(RuntimeError):
            # make sure you can't run recovery_factor before simulate
            reservoir.recovery_factor()

    def test_rf_early_asymptotes(self, nx, pf, pi, fluid, Reservoir, nt):
        if Reservoir == MultiPhaseReservoir:
            reservoir = Reservoir(nx, pf, pi, fluid, So, Sg, Sw)
        else:
            reservoir = Reservoir(nx, pf, pi, fluid)
        end_t = 9.0
        time = np.linspace(0, np.sqrt(end_t), nt) ** 2
        reservoir.simulate(time)

        def logslope(time, rf, mask=None):
            if mask is None:
                mask = (time > 0.02) & (time < 0.3)
            time_log = np.log(time[mask])
            rf_log = np.log(rf[mask])

            def curve(time, intercept, slope):
                return intercept + time * slope

            (intercept, slope), _ = curve_fit(curve, time_log, rf_log, p0=[0, 1])
            return slope

        # rf goes as sqrt-time in early production
        for density in (True, False):
            rf = reservoir.recovery_factor(density=density)
            slope = logslope(time, rf, (time > 0.1) & (time < 0.3))
            assert (
                np.abs(slope - 0.5) < 0.05
            ), "Early recovery factor goes as the square root"

    def test_rf_late_asymptotes(self, nx, pf, pi, fluid, Reservoir, nt):
        if Reservoir == MultiPhaseReservoir:
            reservoir = Reservoir(nx, pf, pi, fluid, So, Sg, Sw)
        else:
            reservoir = Reservoir(nx, pf, pi, fluid)
        end_t = 100
        time = np.linspace(0, np.sqrt(end_t), nt) ** 2
        reservoir.simulate(time)
        if False:
            import matplotlib.pyplot as plt
            from bluebonnet.plotting import plot_pseudopressure, plot_recovery_factor

            fig, ax = plt.subplots()
            ax = plot_recovery_factor(reservoir, ax)
            fig.savefig("rf.png")
            fig, ax = plt.subplots()
            ax = plot_pseudopressure(reservoir, 100, ax=ax)
            fig.savefig("pseudopressure.png")

        def logslope(time, rf, mask=None):
            if mask is None:
                mask = (time > 0.02) & (time < 0.2)
            time_log = np.log(time[mask])
            rf_log = np.log(rf[mask])

            def curve(time, intercept, slope):
                return intercept + time * slope

            (_, slope), _ = curve_fit(curve, time_log, rf_log, p0=[0, 1])
            return slope

        # rf stops increasing at late time
        for density in (True, False):
            rf = reservoir.recovery_factor(density=density)
            slope = logslope(time, rf, time > 10)
            assert np.abs(slope) < 0.03, "Late recovery factor is slow"
            assert slope > -1e-10, "Late recovery never trends negative"
