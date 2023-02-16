"""Calculate fluid PVT properties."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy as sp
from numpy.typing import NDArray
from scipy.integrate import cumtrapz

from bluebonnet.fluids.gas import (
    b_factor_DAK,
    compressibility_DAK,
    density_DAK,
    make_nonhydrocarbon_properties,
    pseudocritical_point_Sutton,
    viscosity_Sutton,
    z_factor_DAK,
)
from bluebonnet.fluids.oil import (
    b_o_Standing,
    pressure_bubblepoint_Standing,
    viscosity_beggs_robinson,
)
from bluebonnet.fluids.water import b_water_McCain, viscosity_water_McCain

PRESSURE_STANDARD = 14.7
"""STP pressure (14.7 psi)"""
TEMPERATURE_STANDARD = 60
"""STP Temperature (60 degrees F)"""


def build_pvt_gas(
    gas_values: Mapping, gas_dryness: str, maximum_pressure: float = 14_000
) -> pd.DataFrame:
    """Build a table of PVT properties for use in the flow module.

    Parameters
    -----------
    gas_values : dictionary
        keys include 'N2','H2S','CO2', 'Gas Specific Gravity',
        'Reservoir Temperature (deg F)'
    gas_dryness : str
        One of 'wet gas' or 'dry gas'
    maximum_pressure : float
        initial reservoir pressure to calculate up to.

    Returns
    -------
    pd.DataFrame
        pvt_table
    """
    non_hydrocarbon_properties = make_nonhydrocarbon_properties(
        gas_values["N2"], gas_values["H2S"], gas_values["CO2"]
    )
    temperature_pc, pressure_pc = pseudocritical_point_Sutton(
        gas_values["Gas Specific Gravity"],
        non_hydrocarbon_properties,
        gas_dryness,
    )
    pressure = np.arange(10.0, maximum_pressure, 10.0)
    temperature = gas_values["Reservoir Temperature (deg F)"]
    z_factor = np.array(
        [z_factor_DAK(temperature, p, temperature_pc, pressure_pc) for p in pressure]
    )
    density = np.array(
        [
            density_DAK(
                temperature,
                p,
                temperature_pc,
                pressure_pc,
                float(gas_values["Gas Specific Gravity"]),
            )
            for p in pressure
        ]
    )
    viscosity = np.array(
        [
            viscosity_Sutton(
                temperature,
                p,
                temperature_pc,
                pressure_pc,
                float(gas_values["Gas Specific Gravity"]),
            )
            for p in pressure
        ]
    )
    compressibility = np.array(
        [
            compressibility_DAK(temperature, p, temperature_pc, pressure_pc)
            for p in pressure
        ]
    )
    pvt_gas = pd.DataFrame(
        data={
            "temperature": np.full_like(pressure, temperature),
            "pressure": pressure,
            "Density": density,
            "z-factor": z_factor,
            "compressibility": compressibility,
            "viscosity": viscosity,
        }
    )
    pseudopressure = 2 * cumtrapz(
        pvt_gas["pressure"] / (pvt_gas["viscosity"] * pvt_gas["z-factor"]),
        pvt_gas["pressure"],
        initial=0.0,
    )
    pvt_gas["pseudopressure"] = pseudopressure
    return pvt_gas


@dataclass
class Fluid:
    r"""Fluid PVT properties.

    Parameters
    ----------
    temperature: float
        reservoir temperature (in F)
    api_gravity: float
        oil gravity in degrees API
    gas_specific_gravity: float
        Gas gravity relative to air
    solution_gor_initial: float
        initial reservoir gas-oil ratio in scf/bbl
    salinity: float
        water salinity in weight percent total dissolved solids
    water_saturation_initial: float
        initial water saturation (V/V)
    """

    temperature: float
    """Temperature in F"""
    api_gravity: float
    """Oil API gravity"""
    gas_specific_gravity: float
    """Gas specific gravity"""
    solution_gor_initial: float
    """Initial gas-oil ratio"""
    salinity: float = 0.0
    """Salinity (total dissolved solids)"""
    water_saturation_initial: float = 0.0
    """Initial water saturation"""

    def water_FVF(
        self, pressure: NDArray[np.float] | float
    ) -> NDArray[np.float] | float:
        """Water formation volume factor (B-factor) from McCain.

        Parameters
        ----------
        pressure: np.ndarray
            water pressure in psia
        Returns
        ----------
        b_w: ndarray
            b-factor in reservoir bbl / standard bbl
        """
        b_w = np.array([b_water_McCain(self.temperature, p) for p in pressure])
        return b_w

    def water_viscosity(self, pressure: NDArray | float):
        """Water viscosity from McCain (1991).

        Parameters
        ----------
        pressure: ndarray
            water pressure in psia

        Returns
        -------
        mu_w : float
            viscosity in centipoise
        """
        return viscosity_water_McCain(self.temperature, pressure, self.salinity)

    def gas_FVF(
        self,
        pressure: NDArray | float,
        temperature_pseudocritical: float,
        pressure_pseudocritical: float,
    ) -> NDArray | float:
        """Gas formation volume factor (Bg) from Dranchuk and Abou-Kassem (1975).

        Parameters
        ----------
        pressure : ndarray
            reservoir pressure in psia
        temperature_pseudocritical : float
            pseudocritical temperature in Fahrenheit.
        pressure_pseudocritical : float
            pseudocritical pressure in psia

        Returns
        -------
        b_g: ndarray
            b-factor (reservoir barrels / scf)
        """
        b_g = np.array(
            [
                b_factor_DAK(
                    self.temperature,
                    p,
                    temperature_pseudocritical,
                    pressure_pseudocritical,
                )
                for p in pressure
            ]
        )
        return b_g

    def gas_viscosity(
        self,
        pressure: NDArray | float,
        temperature_pseudocritical: float,
        pressure_pseudocritical: float,
    ) -> NDArray | float:
        """Calculate the viscosity for gas using Sutton's Fudamental PVT Calculations (2007).

        Parameters
        ----------
        pressure : float
            reservoir pressure in psia
        temperature_pseudocritical : float
            pseudocritical temperature in Fahrenheit.
        pressure_pseudocritical : float
            pseudocritical pressure in psia

        Returns
        -------
        mu_g : ndarray
            viscosity_gas (centipoise)

        Examples
        -------
        >>> Fluid(400, 35, 0.65, 0).viscosity_gas(100, -102, 649)
        0.01652719692109309
        """
        viscosity = np.array(
            [
                viscosity_Sutton(
                    self.temperature,
                    p,
                    temperature_pseudocritical,
                    pressure_pseudocritical,
                    self.gas_specific_gravity,
                )
                for p in pressure
            ]
        )
        return viscosity

    def oil_FVF(self, pressure: NDArray | float) -> NDArray | float:
        """Calculate the oil formation volume factor (Bo) using Standing.

        Parameters
        ----------
        pressure : float
            reservoir pressure (psia)

        Returns
        -------
        b_o : ndarray
            :math:`b_o`, the formation volume factor (V/V)
        """
        fvf_oil = b_o_Standing(
            self.temperature,
            pressure,
            self.api_gravity,
            self.gas_specific_gravity,
            self.solution_gor_initial,
        )
        return fvf_oil

    def oil_viscosity(self, pressure: NDArray | float) -> NDArray | float:
        r"""Calculate the oil viscosity using Beggs-Robinson.

        Returns
        -------
        mu_o: ndarray
            :math:`\mu_o`, the oil viscosity
        """
        visc_br = np.vectorize(viscosity_beggs_robinson)
        mu_o = visc_br(
            self.temperature,
            pressure,
            self.api_gravity,
            self.gas_specific_gravity,
            self.solution_gor_initial,
        )
        return mu_o

    def pressure_bubblepoint(
        self,
    ) -> float:
        """Calculate the bubble point pressure using Standing.

        Returns
        -------
        p_bubble : float
            Bubble point pressure in psia.

        Examples
        -------
        >>> Fluid(200, 35, 0.8, 650).pressure_bubblepoint()
        2627.2017021875276
        """
        p_bubble = pressure_bubblepoint_Standing(
            self.temperature,
            self.api_gravity,
            self.gas_specific_gravity,
            self.solution_gor_initial,
        )
        return p_bubble


def pseudopressure(
    pressure: NDArray[np.float64],
    viscosity: NDArray[np.float64],
    z_factor: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Calculate the pseudopressure using Al-Hussainy's relation.

    Parameters
    ----------
    pressure: NDArray, psi
    viscosity: NDArray, cp
    z_factor: NDArray, dimensionless

    Returns
    -------
    m : NDArray
        pseudopressure, psia^2 / cp
    """
    pp = 2 * pressure / (viscosity * z_factor)
    return sp.integrate.cumulative_trapezoid(pp, pressure, initial=0.0)
