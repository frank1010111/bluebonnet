"""
Define a suite a tests for the reservoir module
"""
import pytest
from itertools import product
import numpy as np
from scipy.optimize import curve_fit
from bluebonnet.fluids.fluid import Fluid
from bluebonnet.flow.reservoir import (
    FlowProperties,
    FlowPropertiesMultiPhase,
    IdealReservoir,
    SinglePhaseReservoir,
    MultiPhaseReservoir,
    relative_permeabilities,
)

nx = (20, 50)
nt = (100, 10_000)
pf = (100, 1000)
pr = (2000, 10_000)
fluid = (
    FlowProperties(
        {"pseudopressure": np.linspace(0, 1, 50), "alpha": np.ones(50)}, 1.0
    ),
)
sim_props = list(product(nx, pf, pr, fluid))
end_t = 9.0
So, Sg, Sw = (0.7, 0.2, 0.1)
reservoirs = (IdealReservoir, SinglePhaseReservoir, MultiPhaseReservoir)


@pytest.mark.parametrize(
    "Reservoir", (IdealReservoir, SinglePhaseReservoir, MultiPhaseReservoir)
)
@pytest.mark.parametrize("nx,pf,pi,fluid", sim_props)
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
    def test_rf_fails_early(self, nx, pf, pi, fluid, Reservoir, nt):
        if Reservoir == MultiPhaseReservoir:
            reservoir = Reservoir(nx, pf, pi, fluid, So, Sg, Sw)
        else:
            reservoir = Reservoir(nx, pf, pi, fluid)
        time = np.linspace(0, np.sqrt(end_t), nt) ** 2
        with pytest.raises(RuntimeError):
            # make sure you can't run recovery_factor before simulate
            reservoir.recovery_factor()

    def test_rf_early_asymptotes(self, nx, pf, pi, fluid, Reservoir, nt):
        if Reservoir == MultiPhaseReservoir:
            reservoir = Reservoir(nx, pf, pi, fluid, So, Sg, Sw)
        else:
            reservoir = Reservoir(nx, pf, pi, fluid)
        time = np.linspace(0, np.sqrt(end_t), nt) ** 2
        reservoir.simulate(time)
        rf = reservoir.recovery_factor()

        def logslope(time, rf, mask=None):
            if mask is None:
                mask = (time > 0.02) & (time < 0.2)
            time_log = np.log(time[mask])
            rf_log = np.log(rf[mask])
            curve = lambda t, intercept, slope: intercept + t * slope
            (intercept, slope), _ = curve_fit(curve, time_log, rf_log, p0=[0, 1])
            return slope

        # rf goes as sqrt-time in early production
        slope = logslope(time, rf, (time > 0.02) & (time < 0.2))
        assert (
            np.abs(slope - 0.5) < 0.05
        ), "Early recovery factor goes as the square root"

    def test_rf_late_asymptotes(self, nx, pf, pi, fluid, Reservoir, nt):
        if Reservoir == MultiPhaseReservoir:
            reservoir = Reservoir(nx, pf, pi, fluid, So, Sg, Sw)
        else:
            reservoir = Reservoir(nx, pf, pi, fluid)
        time = np.linspace(0, np.sqrt(end_t), nt) ** 2
        reservoir.simulate(time)
        rf = reservoir.recovery_factor()

        def logslope(time, rf, mask=None):
            if mask is None:
                mask = (time > 0.02) & (time < 0.2)
            time_log = np.log(time[mask])
            rf_log = np.log(rf[mask])
            curve = lambda t, intercept, slope: intercept + t * slope
            (intercept, slope), _ = curve_fit(curve, time_log, rf_log, p0=[0, 1])
            return slope

        # rf stops increasing at late time
        slope = logslope(time, rf, time > 4)
        assert np.abs(slope) < 0.01, "Late recovery factor is slow"
        assert slope > 0, "Late recovery never trends negative"
