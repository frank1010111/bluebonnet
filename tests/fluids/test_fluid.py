"""Define a suite a tests for the fluid module."""
from __future__ import annotations

import numpy as np
import pytest
from bluebonnet.fluids import Fluid


@pytest.fixture()
def fluid_instance(oil_properties):
    fluid = Fluid(
        oil_properties.temperature,
        oil_properties.api_gravity,
        oil_properties.gas_specific_gravity,
        oil_properties.solution_gor_initial,
        salinity=15,
    )
    return fluid


def test_bw(water_properties):
    real_fvf = pytest.approx(1.0312698)
    fluid_instance = Fluid(
        temperature=water_properties.temperature,
        api_gravity=0,
        gas_specific_gravity=0,
        solution_gor_initial=0,
        salinity=water_properties.salinity,
    )
    water_fvf = fluid_instance.water_FVF(np.full(2, water_properties.pressure))
    assert water_fvf[0] == real_fvf


def test_mu_w(water_properties):
    real_mu = pytest.approx(0.548030320898)
    fluid_instance = Fluid(
        temperature=water_properties.temperature,
        api_gravity=0,
        gas_specific_gravity=0,
        solution_gor_initial=0,
        salinity=water_properties.salinity,
    )
    water_mu = fluid_instance.water_viscosity(np.full(2, water_properties.pressure))
    assert water_mu[0] == real_mu


def test_bo(oil_properties):
    if np.abs(oil_properties.pressure - 3000) < 1:
        real_fvf = pytest.approx(1.381015, rel=1e-3)
    elif np.abs(oil_properties.pressure - 2000) < 1:
        real_fvf = pytest.approx(1.292999, rel=1e-3)
    else:
        msg = "haven't studied this pressure yet"
        raise ValueError(msg)
    fluid_instance = Fluid(
        temperature=oil_properties.temperature,
        api_gravity=oil_properties.api_gravity,
        gas_specific_gravity=oil_properties.gas_specific_gravity,
        solution_gor_initial=oil_properties.solution_gor_initial,
        salinity=0,
    )
    assert fluid_instance.oil_FVF(oil_properties.pressure) == real_fvf
    oil_fvf = fluid_instance.oil_FVF(np.full(2, oil_properties.pressure))
    assert oil_fvf[0] == real_fvf


def test_mu_o(oil_properties):
    if np.abs(oil_properties.pressure - 3000) < 1:
        real_mu = pytest.approx(0.511367423, rel=1e-3)
    elif np.abs(oil_properties.pressure - 2000) < 1:
        real_mu = pytest.approx(0.581137969, rel=1e-3)
    else:
        msg = "haven't studied this pressure yet"
        raise ValueError(msg)
    fluid_instance = Fluid(
        temperature=oil_properties.temperature,
        api_gravity=oil_properties.api_gravity,
        gas_specific_gravity=oil_properties.gas_specific_gravity,
        solution_gor_initial=oil_properties.solution_gor_initial,
        salinity=0,
    )
    oil_mu = fluid_instance.oil_viscosity(np.full(2, oil_properties.pressure))
    assert oil_mu[0] == real_mu


def test_bg(gas_properties):
    if gas_properties.fluid == "dry gas":
        real_fvf = pytest.approx(0.04317427, rel=1e-3)
    elif gas_properties.fluid == "wet gas":
        real_fvf = pytest.approx(0.000813928, rel=1e-3)
    else:
        ValueError("gas_properties.fluid must be one of 'wet gas' or 'dry gas'")
    fluid_instance = Fluid(
        temperature=gas_properties.temperature,
        api_gravity=10,
        gas_specific_gravity=gas_properties.specific_gravity,
        solution_gor_initial=1e3,
        salinity=0,
    )
    gas_fvf = fluid_instance.gas_FVF(
        np.full(2, gas_properties.pressure),
        gas_properties.temperature_pc,
        gas_properties.pressure_pc,
    )
    assert gas_fvf[0] == real_fvf


def test_mu_g(gas_properties):
    if gas_properties.fluid == "dry gas":
        real_mu = pytest.approx(0.016528333, rel=1e-3)
    elif gas_properties.fluid == "wet gas":
        real_mu = pytest.approx(0.023553396, rel=1e-3)
    else:
        ValueError("gas_properties.fluid must be one of 'wet gas' or 'dry gas'")
    fluid_instance = Fluid(
        temperature=gas_properties.temperature,
        api_gravity=10,
        gas_specific_gravity=gas_properties.specific_gravity,
        solution_gor_initial=1e3,
        salinity=0,
    )
    gas_mu = fluid_instance.gas_viscosity(
        np.full(2, gas_properties.pressure),
        gas_properties.temperature_pc,
        gas_properties.pressure_pc,
    )
    assert gas_mu[0] == real_mu
