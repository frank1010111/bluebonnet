"""Define a suite a tests for the gas module."""
from __future__ import annotations

import numpy as np
import pytest
from bluebonnet.fluids import gas

TEMPERATURE_STANDARD = 60.0
PRESSURE_STANDARD = 14.70


def test_make_nonhydrocarbon_properties():
    """Test creation of non-hydrocarbon molecular properties."""
    nitrogen = 0.03
    hydrogen_sulfide = 0.012
    co2 = 0.018
    dtypes = [
        ("name", "U20"),
        ("fraction", "f8"),
        ("molecular weight", "f8"),
        ("critical temperature", "f8"),
        ("critical pressure", "f8"),
    ]
    correct_array = np.array(
        [
            ("Nitrogen", nitrogen, 28.01, 226.98, 492.26),
            ("Hydrogen sulfide", hydrogen_sulfide, 34.08, 672.35, 1299.97),
            ("CO2", co2, 44.01, 547.54, 1070.67),
        ],
        dtype=dtypes,
    )

    test_array = gas.make_nonhydrocarbon_properties(nitrogen, hydrogen_sulfide, co2)
    for col in (
        "fraction",
        "molecular weight",
        "critical temperature",
        "critical pressure",
    ):
        np.testing.assert_allclose(test_array[col], correct_array[col])

    helium = 0.01
    others = [("Helium", helium, 4.0, 226.8, 493.1)]

    gas.make_nonhydrocarbon_properties(nitrogen, hydrogen_sulfide, co2, *others)


def test_pseudocritical_points_Sutton(gas_properties):
    """Test Sutton pseudocritical pressure and temperature."""
    fluid = gas_properties.fluid
    specific_gravity = gas_properties.specific_gravity
    temperature_pc_true = pytest.approx(gas_properties.temperature_pc)
    pressure_pc_true = pytest.approx(gas_properties.pressure_pc)
    if fluid == "dry gas":
        fractions = (0.03, 0.012, 0.018)
    elif fluid == "wet gas":
        fractions = (0.05, 0.01, 0.04)
    else:
        raise NotImplementedError
    non_hydrocarbon_properties = gas.make_nonhydrocarbon_properties(*fractions)
    temperature_pc, pressure_pc = gas.pseudocritical_point_Sutton(
        specific_gravity, non_hydrocarbon_properties, fluid
    )
    assert pressure_pc == pressure_pc_true
    assert temperature_pc == temperature_pc_true


def test_zfactor_DAK(gas_properties):
    """Test Dranchuk's z-factor calculation."""
    (
        temperature,
        pressure,
        temperature_pc,
        pressure_pc,
        _,
        fluid,
    ) = gas_properties
    if fluid == "dry gas":
        real_z = pytest.approx(0.996901, rel=1e-3)
    else:
        real_z = pytest.approx(1.066513, rel=1e-3)
    z_factor_DAK = gas.z_factor_DAK(temperature, pressure, temperature_pc, pressure_pc)
    assert z_factor_DAK == real_z


def test_b_factor_DAK(gas_properties):
    """Test Dranchuk's b_g (depends on z-factor)."""
    (
        temperature,
        pressure,
        temperature_pc,
        pressure_pc,
        _,
        fluid,
    ) = gas_properties
    if fluid == "dry gas":
        real_b_g = pytest.approx(0.04317415921420302, rel=1e-3)
    else:
        real_b_g = pytest.approx(8.139289641971048e-4, rel=1e-3)
    b_factor = gas.b_factor_DAK(
        temperature,
        pressure,
        temperature_pc,
        pressure_pc,
        TEMPERATURE_STANDARD,
        PRESSURE_STANDARD,
    )
    assert b_factor == real_b_g


def test_density_DAK(gas_properties):
    """Test Dranchuk's density calculation (depends on z-factor)."""
    (
        temperature,
        pressure,
        temperature_pc,
        pressure_pc,
        gas_specific_gravity,
        fluid,
    ) = gas_properties
    if fluid == "dry gas":
        real_density = pytest.approx(0.204702, rel=1e-3)
    else:
        real_density = pytest.approx(12.79783, rel=1e-3)
    density = gas.density_DAK(
        temperature, pressure, temperature_pc, pressure_pc, gas_specific_gravity
    )
    assert density == real_density


def test_compressibility_DAK(gas_properties):
    """Test Dranchuk's compressibility (depends on z-factor)."""
    (
        temperature,
        pressure,
        temperature_pc,
        pressure_pc,
        gas_specific_gravity,
        fluid,
    ) = gas_properties
    if fluid == "dry gas":
        real_compressibility = pytest.approx(0.01002548578259275, rel=1e-3)
    else:
        real_compressibility = pytest.approx(1.4644387216173005e-4, rel=1e-3)
    compressibility = gas.compressibility_DAK(
        temperature, pressure, temperature_pc, pressure_pc
    )
    assert compressibility == real_compressibility


def test_viscosity_Sutton(gas_properties):
    """Test Sutton's viscosity (depends on density, which depends on z-factor)."""
    (
        temperature,
        pressure,
        temperature_pc,
        pressure_pc,
        gas_specific_gravity,
        fluid,
    ) = gas_properties
    if fluid == "dry gas":
        real_viscosity = pytest.approx(1.6528333290904862e-2, rel=1e-3)
    else:
        real_viscosity = pytest.approx(2.3553396039641815e-2, rel=1e-3)
    viscosity = gas.viscosity_Sutton(
        temperature, pressure, temperature_pc, pressure_pc, gas_specific_gravity
    )
    assert viscosity == real_viscosity


def test_pseudopressure_Hussainy(gas_properties):
    """Test Al Hussainy's pseudopressure calculation (depends on viscosity, density, z-factor)."""
    (
        temperature,
        pressure,
        temperature_pc,
        pressure_pc,
        gas_specific_gravity,
        fluid,
    ) = gas_properties
    if fluid == "dry gas":
        real_m = pytest.approx(593363, rel=1e-3)
    else:
        real_m = pytest.approx(1280589747, rel=1e-3)
    m = gas.pseudopressure_Hussainy(
        temperature,
        pressure,
        temperature_pc,
        pressure_pc,
        gas_specific_gravity,
        PRESSURE_STANDARD,
    )
    assert m == real_m
