"""
Define a suite a tests for the gas module
"""
from __future__ import annotations

from collections import namedtuple

import pytest

from bluebonnet.fluids import water

TEMPERATURE_STANDARD = 60.0
PRESSURE_STANDARD = 14.70

WaterParams = namedtuple(
    "WaterParams", "temperature, pressure, salinity, gas_saturated, gas_properties"
)


@pytest.fixture(params=[True, False])
def water_properties(request):
    temperature = 200
    pressure = 4000
    salinity = 15
    gas_saturated = request.param
    gas_properties = None
    return WaterParams(temperature, pressure, salinity, gas_saturated, gas_properties)


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
