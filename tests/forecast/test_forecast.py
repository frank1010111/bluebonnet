"""Test fitting and forecasting production."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bluebonnet.flow import FlowProperties, IdealReservoir, SinglePhaseReservoir
from bluebonnet.forecast import (
    Bounds,
    ForecasterOnePhase,
    fit_production_pressure,
    plot_production_comparison,
)

t_end = 6.0
nx = 40
nt = 2_000
pf = 500.0
pi = 5_000.0
time_scaled = np.linspace(0, np.sqrt(t_end), nt) ** 2


@pytest.fixture()
def rf_curve():
    """Return function for recovery factor from an ideal reservoir."""
    reservoir = IdealReservoir(nx, pf, pi, None)
    reservoir.simulate(time_scaled)
    reservoir.recovery_factor()
    return reservoir.recovery_factor_interpolator()


def test_bounds():
    """Make sure bounds uses post_init properly."""
    correct_bounds = Bounds(M=(0, 1), tau=(2, 3))
    assert correct_bounds.fit_bounds() == ((0, 2), (1, 3))
    with pytest.raises(ValueError):
        Bounds(M=(1, 2, 3), tau=(0, 1))
    with pytest.raises(ValueError):
        Bounds(M=(1, 2), tau=(1,))
    with pytest.raises(ValueError):
        Bounds(M=(1, 0), tau=(0, 1))
    with pytest.raises(ValueError):
        Bounds((0, 1), (20, 10))


def test_ForecasterOnePhase(rf_curve):
    """Test forecaster."""
    tau_test = 3.0
    M_test = 300.0
    cum_production = M_test * rf_curve(time_scaled / tau_test)

    # Fit the test production
    scaling_curve = ForecasterOnePhase(rf_curve)
    scaling_curve.fit(time_scaled, cum_production)
    M_fit = scaling_curve.M_
    tau_fit = scaling_curve.tau_
    assert M_fit == pytest.approx(M_test)
    assert tau_fit == pytest.approx(tau_test)

    # Test prediction
    cum_fit = scaling_curve.forecast_cum(time_scaled)
    assert np.allclose(cum_production, cum_fit)


@pytest.mark.parametrize("res_type", ["ideal reservoir", "real reservoir"])
def test_fit_production_pressure(res_type):
    """Test fitting for production and pressure."""
    pressure_v_time = np.full(nt, pf)
    pressure_v_time[nt // 4 : nt // 2] /= 2.0  # noqa: E203
    pressure_v_time[nt // 2 :] /= 4.0  # noqa: E203
    if res_type.startswith("ideal"):
        pvt_table = None
        reservoir = IdealReservoir(nx, pf, pi, None)
        reservoir.simulate(time_scaled, pressure_v_time)
        rf = reservoir.recovery_factor()
    else:
        pvt_table = pd.read_csv("tests/data/pvt_gas_HAYNESVILLE SHALE_20.csv")
        flow_props = FlowProperties(pvt_table, pi)
        reservoir = SinglePhaseReservoir(nx, pf, pi, flow_props)
        reservoir.simulate(time_scaled, pressure_v_time)
        rf = reservoir.recovery_factor()
    tau_in = 180.0
    prod = pd.DataFrame(
        {"Days": time_scaled * tau_in, "Gas": rf, "Pressure": pressure_v_time}
    )
    result = fit_production_pressure(prod, pvt_table, pi, n_iter=4)
    assert result.params["tau"].value < 1000, "is tau moving in the right direction?"
    assert result.params["M"].value == pytest.approx(1)
