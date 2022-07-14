from dataclasses import dataclass
import numpy as np
from numpy import ndarray
from typing import Union
import numpy.typing as npt
import scipy as sp
import pandas as pd
from collections import namedtuple
from scipy.integrate import cumtrapz

from .gas import (
    make_nonhydrocarbon_properties,
    z_factor_DAK,
    b_factor_DAK,
    density_DAK,
    compressibility_DAK,
    pseudocritical_point_Sutton,
    viscosity_Sutton,
    pseudopressure_Hussainy,
)
from .oil import b_o_Standing, pressure_bubblepoint_Standing, viscosity_beggs_robinson
from .water import b_water_McCain, density_water_McCain, viscosity_water_McCain

PRESSURE_STANDARD = 14.7
TEMPERATURE_STANDARD = 60
PseudocriticalPoint = namedtuple(
    "PseudocriticalPoint", ["temperature_pseudocritical", "pressure_pseudocritical"]
)

def BuildPVT(FieldValues,GasDryness,maximum_pressure=14000):
    """
    FieldValues must contain 'N2','H2S','CO2', 'Gas Specific Gravity','Reservoir Temperature (deg F)'
    """

    non_hydrocarbon_properties=make_nonhydrocarbon_properties(
        float(FieldValues['N2']),
        float(FieldValues['H2S']),
        float(FieldValues['CO2']))

    (Tc,Pc)=pseudocritical_point_Sutton(
        float(FieldValues['Gas Specific Gravity']),
        non_hydrocarbon_properties,
        GasDryness)

    Pressure=np.arange(0,maximum_pressure,10.0)
    Pressure[0]=0.001

    T=np.zeros(len(Pressure))+float(FieldValues['Reservoir Temperature (deg F)'])

    Z=np.array( [
        z_factor_DAK(
            float(FieldValues['Reservoir Temperature (deg F)']),
            p,
            Tc,
            Pc)
        for p in Pressure])

    density=np.array([density_DAK(
        float(FieldValues['Reservoir Temperature (deg F)']),
        p,
        Tc,
        Pc,
        float(FieldValues['Gas Specific Gravity']))
    for p in Pressure])
  
    viscosity = np.array([
        viscosity_Sutton(
                    float(FieldValues['Reservoir Temperature (deg F)']),
                    p,
                    Tc,
                    Pc,
                    float(FieldValues['Gas Specific Gravity']))
        for p in Pressure ])

    compressibility = np.array([
        compressibility_DAK(
            float(FieldValues['Reservoir Temperature (deg F)']),
            p,
            Tc,
            Pc)
        for p in Pressure])
    pvt_gas=pd.DataFrame(data={'T':T,'P':Pressure,'Density':density,'Z-Factor':Z,'Cg':compressibility,'Viscosity':viscosity})
    ms=2*cumtrapz(pvt_gas.P/(pvt_gas.Viscosity*pvt_gas["Z-Factor"]),pvt_gas.P)
    ms=np.concatenate(([0],ms))
    pvt_gas['pseudopressure']=ms
    
    return(pvt_gas)

@dataclass
class Fluid:
    """
    Fluid properties

    Parameters
    ----------
    temperature: reservoir temperature (in F)
    api_gravity: oil gravity in degrees API
    gas_specific_gravity: Gas gravity relative to air
    solution_gor_initial: initial reservoir gas-oil ratio in scf/bbl
    salinity: water salinity in weight percent total dissolved solids
    water_saturation_initial: initial water saturation (V/V)

    Methods
    ---------
    water_FVF: Bw
    water_viscosity: $\\mu_w$
    gas_FVF: Bg
    gas_viscosity: $\\mu_g$
    oil_FVF: Bo
    oil_viscosity: $\\mu_o$
    pressure_bubblepoint
    """

    temperature: float
    api_gravity: float
    gas_specific_gravity: float
    solution_gor_initial: float
    salinity: float = 0.0
    water_saturation_initial = 0.0

    def water_FVF(self, pressure: Union[ndarray, float]) -> Union[ndarray, float]:
        """
        Water formation volume factor (B-factor) from McCain

        Parameters
        ----------
        pressure: np.ndarray
            water pressure in psia
        Returns
        ----------
        ndarray
            b-factor in reservoir bbl/ standard bbl
        """
        b_w = np.array([b_water_McCain(self.temperature, p) for p in pressure])
        return b_w

    def water_viscosity(self, pressure: Union[ndarray, float]):
        """
        Water viscosity from McCain (1991)

        Parameters
        ----------
        pressure: ndarray
            water pressure in psia

        Returns
        -------
        float
            viscosity in centipoise
        """
        return viscosity_water_McCain(self.temperature, pressure, self.salinity)

    def gas_FVF(
        self,
        pressure: Union[ndarray, float],
        temperature_pseudocritical: float,
        pressure_pseudocritical: float,
    ) -> Union[ndarray, float]:
        """
        Gas formation volume factor (Bg) from Dranchuk and Abou-Kassem (1975)

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
        ndarray
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
        pressure: Union[ndarray, float],
        temperature_pseudocritical: float,
        pressure_pseudocritical: float,
    ) -> Union[ndarray, float]:
        """
        Calculates the viscosity for gas using Sutton's Fudamental PVT Calculations (2007)

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
        ndarray
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

    def oil_FVF(self, pressure: Union[ndarray, float]) -> Union[ndarray, float]:
        """
        Calculates the oil formation volume factor (Bo) using Standing.

        Parameters
        ----------
        pressure : float
            reservoir pressure (psia)

        Returns
        -------
        ndarray
            Bo, the formation volume factor (V/V)
        """
        fvf_oil = b_o_Standing(
            self.temperature,
            pressure,
            self.api_gravity,
            self.gas_specific_gravity,
            self.solution_gor_initial,
        )
        return fvf_oil

    def oil_viscosity(self, pressure: Union[ndarray, float]) -> Union[ndarray, float]:
        """
        Calcualtes the oil viscosity using Beggs-Robinson

        Returns
        -------
        ndarray
            $\\mu_o$, the oil viscosity
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
        """
        Calculates the bubble point pressure using Standing.

        Returns
        -------
        float
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
    pressure: ndarray, viscosity: ndarray, z_factor: ndarray
) -> ndarray:
    """
    Calculates the pseudopressure using Al-Hussainy's relation

    Parameters
    ----------
    pressure: NDArray, psi
    viscosity: NDArray, cp
    z_factor: NDArray, dimensionless

    Returns
    -------
    NDArray
        pseudopressure, psia^2 / cp
    """
    pp = 2 * pressure / (viscosity * z_factor)
    return sp.integrate.cumulative_trapezoid(pp, pressure, initial=0.0)
