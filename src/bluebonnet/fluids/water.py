"""PVT and viscosity for water from the correlations provided by McCain."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def b_water_McCain(
    temperature: float, pressure: float | NDArray[np.float64]
) -> float | NDArray[np.float64]:
    """Calculate the b-factor for water.

    Parameters
    ----------
    temperature : float
        water temperature in Fahrenheit
    pressure: np.ndarray
        water pressure in psia

    Returns
    -------
    b_w : float
        b-factor (reservoir bbl / standard bbl)

    Examples
    --------
    >>> b_water_McCain(400, 3000)

    """
    dV_dp = (
        -1.95301e-9 * pressure * temperature
        - 1.72834e-13 * pressure**2 * temperature
        - 3.58922e-7 * pressure
        - 2.25341e-10 * pressure**2
    )
    dV_dt = -1.0001e-2 + 1.33391e-4 * temperature + 5.50654e-7 * temperature**2
    return (1 + dV_dp) * (1 + dV_dt)


def b_water_McCain_dp(
    temperature: float, pressure: float | NDArray[np.float64]
) -> float | NDArray[np.float64]:
    """Calculate the derivative of the b-factor for water with respect to pressure.

    Parameters
    ----------
    temperature : float
        water temperature in Fahrenheit
    pressure: float | NDArray
        water pressure in psia

    Returns
    -------
    b_w_dp : float
        derivative of b-factor (reservoir bbl / standard bbl) / psi

    Examples
    --------
    >>> b_water_McCain_dp(400, 3000)

    """
    d2V_dp2 = (
        -1.95301e-9 * temperature
        - 2 * 1.72834e-13 * pressure * temperature
        - 3.58922e-7
        - 2 * 2.25341e-10 * pressure
    )
    dV_dt = -1.0001e-2 + 1.33391e-4 * temperature + 5.50654e-7 * temperature**2
    return d2V_dp2 * (1 + dV_dt)


def compressibility_water_McCain(
    temperature: float, pressure: float | NDArray[np.float64], salinity: float
) -> float | NDArray[np.float64]:
    """Calculate the compressibility for water.

    Parameters
    ----------
    temperature : float
        water temperature in Fahrenheit
    pressure: float | NDArray
        water pressure in psia
    salinity: float
        salinity in weight percent total dissolved solids

    Returns
    -------
    c_w : float
        compressibility of water in 1/psi

    Examples
    --------
    >>> density_water_McCain(400, 3000, 15)

    """
    c_w = 1 / (7.033 * pressure + 0.5415 * salinity - 537 * temperature + 403300.0)
    return c_w


def density_water_McCain(
    temperature: float, pressure: float | NDArray[np.float64], salinity: float
) -> float | NDArray[np.float64]:
    r"""Calculate the density for water.

    Parameters
    ----------
    temperature : float
        water temperature in Fahrenheit
    pressure: float | NDArray
        water pressure in psia
    salinity: float
        salinity in weight percent total dissolved solids

    Returns
    -------
    rho_w : float
        density in lb-mass / cu ft, :math:`\rho_w`

    Examples
    --------
    >>> density_water_McCain(400, 3000, 15)

    """
    b_water = b_water_McCain(temperature, pressure)
    density_stp = 62.368 + 0.438603 * salinity + 1.60074e-3 * salinity**2
    return density_stp / b_water


def viscosity_water_McCain(
    temperature: float, pressure: float | NDArray[np.float64], salinity: float
) -> float | NDArray[np.float64]:
    r"""Calculate the viscosity for water, Using McCain (1991).

    Parameters
    ----------
    temperature : float
        water temperature in Fahrenheit
    pressure: float
        water pressure in psia
    salinity: float
        salinity in weight percent total dissolved solids

    Returns
    -------
    mu_w : float
        viscosity in centipoise, :math:`\mu_w`

    Examples
    --------
    >>> viscosity_water_McCain(400, 3000, 15)
    0.2627774655403418
    """
    A = (
        109.574
        - 8.40564 * salinity
        + 0.313314 * salinity**2
        + 8.72213e-3 * salinity**3
    )
    B = (
        1.12166
        - 2.63951e-2 * salinity
        + 6.79461e-4 * salinity**2
        + 5.47119e-5 * salinity**3
        - 1.55586e-6 * salinity**4
    )
    mu_water = (
        A
        * temperature**-B
        * (0.9994 + 4.0295e-5 * pressure + 3.1062e-9 * pressure**2)
    )
    return mu_water
