from dataclasses import dataclass
import numpy as np
import math
from typing import List, Set, Dict, Tuple, Optional
import numpy.typing as npt
from collections import namedtuple
from scipy.optimize import minimize
from scipy.integrate import quad

from .gas import (
    make_nonhydrocarbon_properties,
    z_factor_DAK,
    b_factor_gas_DAK,
    density_gas_DAK,
    compressibility_gas_DAK,
    viscosity_gas_Sutton,
    pseudopressure_Hussainy,
)

PRESSURE_STANDARD = 14.7
TEMPERATURE_STANDARD = 60
PseudocriticalPoint = namedtuple('PseudocriticalPoint',['temperature_pseudocritical','pressure_pseudocritical'])

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
    
    Main methods
    ---------
    water_FVF: Bw
    gas_FVF_DAK: Bg
    gas_viscosity: $\mu_g$
    oil_FVF_standing: Bo
    oil_viscosity: $\mu_o$
    """
    temperature: float
    api_gravity: float
    gas_specific_gravity: float
    solution_gor_initial: float
    
    def water_FVF(self, pressure: npt.ArrayLike) -> npt.ArrayLike:
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
        dV_dp = (
            -1.95301e-9 * pressure * temperature - 1.72834e-13 * pressure ** 2 * temperature
            - 3.58922e-7 * pressure - 2.25341e-10 * pressure ** 2
        )
        dV_dt = -1.0001e-2 + 1.33391e-4 * temperature + 5.50654e-7 * temperature ** 2
        return (1 + dV_dp) * (1 + dV_dt)


    def gas_FVF_DAK(self, pressure: npt.ArrayLike, temperature_pseudocritical: float, pressure_pseudocritical: float) -> npt.ArrayLike:
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
        z_factor = np.array([
            z_factor_DAK(self.temperature, p, temperature_pseudocritical, pressure_pseudocritical) for p in pressure
        ])
        b_factor = (
            z_factor
            * (temperature + 459.67)
            * PRESSURE_STANDARD
            / ((TEMPERATURE_STANDARD + 459.67) * pressure)
        )
        return b_factor / 5.615
    
    
        
    def oil_FVF_standing(self, pressure:npt.ArrayLike) -> npt.ArrayLike:
        """
        Calculates the oil formation volume factor (Bo) using Standing.

        Parameters
        ----------
        pressure : float
            reservoir pressure (psia)
        
        Returns
        -------
        float
            Bo, the formation volume factor (V/V)
        """
        fvf_bubblepoint = self._bo_bubblepoint_standing(self.solution_gor_initial)
        pressure_bubblepoint = self._p_bubblepoint_standing()
        compressibility_undersat = self.oil_compressibility_undersat_spivey(pressure[pressure >= pressure_bubblepoint])
        solution_gor = self.solution_GOR_standing(pressure)
        fvf_oil = np.empty_like(pressure)
        fvf_oil[pressure>=pressure_bubblepoint] = fvf_bubblepoint * np.exp(compressibility_undersat * (pressure_bubblepoint - pressure[pressure >= pressure_bubblepoint]))
        fvf_oil[pressure<pressure_bubblepoint] = self._bo_bubblepoint_standing(solution_gor[pressure<pressure_bubblepoint])
        return fvf_oil
        
    def _bo_bubblepoint_standing(self, solution_gor: npt.ArrayLike) -> npt.ArrayLike:
        """
        Calculates the oil formation volume factor (Bo) at the bubble point using Standing.

        Parameters
        ----------
        solution_gor : ndarray
            Initial gas:oil ratio in scf/bbl.

        Returns
        -------
        ndarray
            B_o (rb/stb)
        """
        oil_specific_gravity = 141.5 / (131.5 + self.api_gravity)
        fvf_bubblepoint = (
            0.9759
            + 0.00012
            * (
                solution_gor
                * np.sqrt(self.gas_specific_gravity / oil_specific_gravity)
                + 1.25 * temperature
            )
            ** 1.2
        )
        return fvf_bubblepoint

        
    def _p_bubblepoint_standing(self, ) -> float:
        """
        Calculates the bubble point pressure using Standing.  

        Returns
        -------
        float
            Bubble point pressure in psia.

        Examples
        -------
        >>> Fluid(200, 35, 0.8, 650)._p_bubblepoint_standing()
        2627.2017021875276
        """
        p_bubble = (
            18.2
            * (
                (solution_gor_initial / gas_specific_gravity) ** 0.83
                * 10 ** (0.00091 * temperature - 0.0125 * api_gravity)
            - 1.4
            )
        )
        return p_bubble
        
        
    def oil_compressibility_undersat_standing(self, pressure: npt.ArrayLike) -> npt.ArrayLike:
        """
        Calculates the oil compressibility (c_o) using Standing.

        Parameters
        ----------
        pressure : ndarray
            reservoir pressure (psia)

        Returns
        -------
        ndarray
            oil_compressibility in 1/psi

        Example
        >>> Fluid(200, 35, 0.8, 650).oil_compressibility_undersat_standing(3_000)
        1.478788544207282e-05
        """
        oil_specific_gravity = 141.5 / (131.5 + self.api_gravity)
        pressure_bubblepoint = self._p_bubblepoint_standing()
        b_o_bubblepoint = self._bo_bubblepoint_standing()
        density_bubblepoint = (
            62.37 * oil_specific_gravity
            + 0.0136 * self.gas_specific_gravity * self.solution_gor_initial
        ) / b_o_bubblepoint
        oil_compressibility_undersat = 1e-6 * np.exp(
            (density_bubblepoint + 0.004347 * (pressure - pressure_bubblepoint) - 79.1)
            / (7.141e-4 * (pressure - pressure_bubblepoint) - 12.938)
        )
        return oil_compressibility
        
        
    def oil_compressibility_undersat_spivey(self, pressure: npt.ArrayLike) -> npt.ArrayLike:
        """
        Calculates the oil compressibility (c_o) using Spivey.

        Parameters
        ----------
        pressure : ndarray
            reservoir pressure (psia)

        Returns
        -------
        ndarray
            oil_compressibility in 1/psi

        Example
        >>> Fluid(200, 35, 0.8, 650).oil_compressibility_undersat_Spivey(3_000)
        9.177554347361075e-06
        """
        pressure_bubblepoint = self._p_bubblepoint_standing()
        reduced_pressure = pressure / pressure_bubblepoint
#         C_labels = [
#             'api_gravity', 'gas_specific_gravity', 'pressure_bubblepoint', 'reduced_pressure', 'solution_gor_initial','temperature'
#         ] 
        C0 = np.array([3.011, -0.835, 3.51, 0.327, -1.918, 2.52])
        C1 = np.array([-2.6254, -0.259, -0.0289, -0.608, -0.642, -2.73])
        C2 = np.array([0.497, 0.382, -0.0584, 0.0911, 0.154, 0.429])
        
        X = np.array(
            [[
                np.log(x)
                for x in (
                    api_gravity,
                    gas_specific_gravity,
                    pressure_bubblepoint,
                    rp,
                    solution_gor_initial,
                    temperature,
                )
            ] for rp in reduced_pressure
            ]
        )
        z = np.sum(C0) + X @ C1  + (X ** 2) @ C2
        compressibility_bubblepoint = np.exp(2.434 + 0.475 * z + 0.048 * z ** 2)
        compressibility_derivative = (-0.608 + 0.1822 * np.log(reduced_pressure)) / pressure
        compressibility_2nd_derivative = (
            compressibility_bubblepoint * (0.475 + 0.096 * z) * compressibility_derivative
        )
        oil_compressibility = 1e-6 * (
            compressibility_bubblepoint
            + (pressure - pressure_bubblepoint) * compressibility_2nd_derivative
        )
        return oil_compressibility
        
        
        
    def solution_GOR_standing(self, pressure: npt.ArrayLike) -> npt.ArrayLike:
        """
        Calculates the solution GOR partition using Standing.

        Parameters
        ----------
        pressure : ndarray
            reservoir pressure (psia)

        Returns
        -------
        ndarray
            Solution GOR in scf/stb
        """
        pressure_bubblepoint = self._p_bubblepoint_standing()
        solution_GOR = np.full_like(pressure, self.solution_gor_initial)
        solution_GOR[pressure< pressure_bubblepoint] = (
            self.gas_specific_gravity * ((pressure[pressure < pressure_bubblepoint] / 18.2 + 1.4) 
               * 10 ** (0.0125 * self.api_gravity - 0.00091 * self.temperature)
              ) ** (1 / 0.83)
        )
        return solution_GOR
    
    def gas_viscosity(self, pressure: npt.ArrayLike, temperature_pseudocritical:float, pressure_pseudocritical: float) -> npt.ArrayLike:
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
        viscosity = np.array([
            viscosity_gas_Sutton(self.temperature, p, temperature_pseudocritical, pressure_pseudocritical, self.gas_specific_gravity) for p in pressure
        ])
        return viscosity
    
