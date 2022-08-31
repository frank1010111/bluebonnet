"""Fixtures for testing fluids modules."""

from __future__ import annotations

from typing import Any, NamedTuple

import pytest


class WaterParams(NamedTuple):
    temperature: float
    pressure: float
    salinity: float
    gas_saturated: float
    gas_properties: Any


class OilParams(NamedTuple):
    temperature: float
    pressure: float
    api_gravity: float
    gas_specific_gravity: float
    solution_gor_initial: float
    fluid: str


class GasParams(NamedTuple):
    temperature: float
    pressure: float
    temperature_pc: float
    pressure_pc: float
    specific_gravity: float
    fluid: str


@pytest.fixture(params=[True, False])
def water_properties(request):
    temperature = 200
    pressure = 4000
    salinity = 15
    gas_saturated = request.param
    gas_properties = None
    return WaterParams(temperature, pressure, salinity, gas_saturated, gas_properties)


@pytest.fixture(params=[(3000.0, "black oil"), (2000.0, "black oil")])
def oil_properties(request):
    pressure, fluid = request.param
    if fluid == "black oil":
        out = OilParams(
            temperature=200.0,
            pressure=pressure,
            api_gravity=35.0,
            gas_specific_gravity=0.8,
            solution_gor_initial=650.0,
            fluid="black oil",
        )
    return out


@pytest.fixture(params=["dry gas", "wet gas"])
def gas_properties(request):
    if request.param == "dry gas":
        out = GasParams(
            temperature=400,
            pressure=100,
            temperature_pc=-102.21827232417752,
            pressure_pc=648.510797253794,
            specific_gravity=0.65,
            fluid=request.param,
        )
    else:
        out = GasParams(
            temperature=300,
            pressure=5014.7,
            temperature_pc=-80.95111110103215,
            pressure_pc=656.7949325583305,
            specific_gravity=0.7661054354004884,
            fluid=request.param,
        )
    return out
