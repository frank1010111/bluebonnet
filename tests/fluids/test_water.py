"""Define a suite a tests for the gas module."""
from __future__ import annotations

import pytest
from bluebonnet.fluids import water

TEMPERATURE_STANDARD = 60.0
PRESSURE_STANDARD = 14.70


def test_b_water_McCain(water_properties):
    temperature = water_properties.temperature
    pressure = water_properties.pressure
    b_water_true = pytest.approx(1.0312698)
    assert water.b_water_McCain(temperature, pressure) == b_water_true


def test_b_water_McCain_dp(water_properties):
    temperature = water_properties.temperature
    pressure = water_properties.pressure
    b_water_dp_true = pytest.approx(-2.9382699e-06)
    assert water.b_water_McCain_dp(temperature, pressure) == b_water_dp_true


def test_compressibility_McCain(water_properties):
    temperature = water_properties.temperature
    pressure = water_properties.pressure
    salinity = water_properties.salinity
    co_water_true = pytest.approx(3.0860375940019587e-06)
    assert (
        water.compressibility_water_McCain(temperature, pressure, salinity)
        == co_water_true
    )


def test_density_McCain(water_properties):
    density_water_true = pytest.approx(67.205706)
    assert (
        water.density_water_McCain(
            water_properties.temperature,
            water_properties.pressure,
            water_properties.salinity,
        )
        == density_water_true
    )


def test_viscosity_McCain(water_properties):
    viscosity_water_true = pytest.approx(0.548030320898)
    assert (
        water.viscosity_water_McCain(
            water_properties.temperature,
            water_properties.pressure,
            water_properties.salinity,
        )
        == viscosity_water_true
    )
