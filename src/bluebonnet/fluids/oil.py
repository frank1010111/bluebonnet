"""Oil PVT properties."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from bluebonnet.fluids.gas import b_factor_DAK


def b_o_Standing(
    temperature: float,
    pressure: NDArray | float,
    api_gravity: float,
    gas_specific_gravity: float,
    solution_gor_initial: float,
) -> NDArray | float:
    """Calculate the oil formation volume factor (Bo) using Standing.

    Parameters
    ----------
    temperature : float
        reservoir temperature in Fahrenheit.
    pressure : float
        reservoir pressure (psia)
    api_gravity : float
        Oil gravity in degrees API.
    gas_specific_gravity : float
        Gas gravity relative to air.
    solution_gor_initial : float
        Initial gas:oil ratio in scf/bbl.

    Returns
    -------
    float
        Bo, the formation volume factor (V/V)

    Examples
    -------
    >>> b_o_Standing(200, 3_000, 35, 0.8, 650)

    >>> b_o_Standing(200, 2_000, 35, 0.8, 650)
    1.2820114682225974
    """
    pressure_bubblepoint = pressure_bubblepoint_Standing(
        temperature, api_gravity, gas_specific_gravity, solution_gor_initial
    )
    fvf_bubblepoint = b_o_bubblepoint_Standing(
        temperature, api_gravity, gas_specific_gravity, solution_gor_initial
    )
    solution_gor = solution_gor_Standing(
        temperature, pressure, api_gravity, gas_specific_gravity, solution_gor_initial
    )
    if np.ndim(pressure) != 0:  # check for non-scalar
        compressibility_undersat = oil_compressibility_undersat_Spivey(
            temperature,
            pressure[pressure >= pressure_bubblepoint],
            api_gravity,
            gas_specific_gravity,
            solution_gor_initial,
        )
        fvf_oil = np.empty_like(pressure)
        fvf_oil[pressure >= pressure_bubblepoint] = fvf_bubblepoint * np.exp(
            compressibility_undersat
            * (pressure_bubblepoint - pressure[pressure >= pressure_bubblepoint])
        )
        fvf_oil[pressure < pressure_bubblepoint] = b_o_bubblepoint_Standing(
            temperature,
            api_gravity,
            gas_specific_gravity,
            solution_gor[pressure < pressure_bubblepoint],
        )
    else:
        if pressure >= pressure_bubblepoint:
            compressibility_undersat = oil_compressibility_undersat_Spivey(
                temperature,
                pressure,
                api_gravity,
                gas_specific_gravity,
                solution_gor_initial,
            )
            fvf_oil = fvf_bubblepoint * np.exp(
                compressibility_undersat * (pressure_bubblepoint - pressure)
            )
        else:
            fvf_oil = b_o_bubblepoint_Standing(
                temperature, api_gravity, gas_specific_gravity, solution_gor
            )
    return fvf_oil


def pressure_bubblepoint_Standing(
    temperature: float,
    api_gravity: float,
    gas_specific_gravity: float,
    solution_gor_initial: float,
) -> float:
    """Calculate the bubble point pressure using Standing.

    Parameters
    ----------
    temperature : float
        reservoir temperature in Fahrenheit.
    api_gravity : float
        Oil gravity in degrees API.
    gas_specific_gravity : float
        Gas gravity relative to air.
    solution_gor_initial : float
        Gas: oil ratio in scf/bbl.

    Returns
    -------
    p_b : float
        Bubble point pressure in psia.

    Examples
    -------
    >>> pressure_bubblepoint_Standing(200, 35, 0.8, 650)
    2627.2017021875276
    """
    pressure_bubble = 18.2 * (
        (solution_gor_initial / gas_specific_gravity) ** 0.83
        * 10 ** (0.00091 * temperature - 0.0125 * api_gravity)
        - 1.4
    )
    return pressure_bubble


def b_o_bubblepoint_Standing(
    temperature: float,
    api_gravity: float,
    gas_specific_gravity: float,
    solution_gor_initial: NDArray | float,
) -> NDArray | float:
    """Calculate the oil formation volume factor (Bo) at the bubble point using Standing.

    Parameters
    ----------
    temperature : float
        reservoir temperature in Fahrenheit.
    api_gravity : float
        Oil gravity in degrees API.
    gas_specific_gravity : float
        Gas gravity relative to air.
    solution_gor_initial : NDArray
        Initial gas:oil ratio in scf/bbl.

    Returns
    -------
    b_o : NDArray
        Oil formation volume factor (rb/stb)

    Example
    -------
    >>> b_o_bubblepoint_Standing(200, 35, 0.8, 650)
    1.3860514623492897
    """
    oil_specific_gravity = 141.5 / (131.5 + api_gravity)
    fvf_bubblepoint = (
        0.9759
        + 0.00012
        * (
            solution_gor_initial * np.sqrt(gas_specific_gravity / oil_specific_gravity)
            + 1.25 * temperature
        )
        ** 1.2
    )
    return fvf_bubblepoint


def db_o_dgor_Standing(
    temperature: float,
    api_gravity: float,
    gas_specific_gravity: float,
    solution_gor_initial: float,
) -> float:
    """Calculate the derivative of Bo per solution GOR at the bubble point.

    Parameters
    ----------
    temperature : float
        reservoir temperature in Fahrenheit.
    api_gravity : float
        Oil gravity in degrees API.
    gas_specific_gravity : float
        Gas gravity relative to air.
    solution_gor_initial : NDArray
        Initial gas:oil ratio in scf/bbl.

    Returns
    -------
    dbo_dgor:  NDArray
        Change in Oil formation volume factor per unit GOR (rb/stb / (scf/bbl))

    Example
    -------
    >>> db_o_dgor_Standing(200, 35, 0.8, 650)
    """
    oil_specific_gravity = 141.5 / (131.5 + api_gravity)
    sqrt_gravity_ratio = np.sqrt(gas_specific_gravity / oil_specific_gravity)
    dB_o_dgor = (
        1.2
        * 1.2e-4
        * (solution_gor_initial * sqrt_gravity_ratio + 1.25 * temperature) ** 0.2
        * sqrt_gravity_ratio
    )
    return dB_o_dgor


def solution_gor_Standing(
    temperature: float,
    pressure: NDArray | float,
    api_gravity: float,
    gas_specific_gravity: float,
    solution_gor_initial: float,
) -> NDArray | float:
    """Calculate the solution GOR partition using Standing.

    Parameters
    ----------
    temperature : float
        reservoir temperature in Fahrenheit.
    pressure : float or array
        reservoir pressure (psia)
    api_gravity : float
        Oil gravity in degrees API.
    gas_specific_gravity : float
        Gas gravity relative to air.
    solution_gor_initial : float
        Gas: oil ratio in scf/bbl.

    Returns
    -------
    gor : float
        Solution GOR in scf/stb

    Examples
    -------
    >>> solution_gor_Standing(200, 3_000, 35, 0.8, 650)
    650
    >>> solution_gor_Standing(200, 2_000, 35, 0.8, 650)
    453.6099792270382
    """
    pressure_bubblepoint = pressure_bubblepoint_Standing(
        temperature, api_gravity, gas_specific_gravity, solution_gor_initial
    )

    def gor_belowbubble(pressure: NDArray | float) -> NDArray | float:
        gor = gas_specific_gravity * (
            (pressure / 18.2 + 1.4)
            * 10 ** (0.0125 * api_gravity - 0.00091 * temperature)
        ) ** (1 / 0.83)
        return gor

    if np.ndim(pressure) != 0:
        solution_gor = np.full_like(pressure, solution_gor_initial)
        solution_gor[pressure < pressure_bubblepoint] = gor_belowbubble(
            pressure[pressure < pressure_bubblepoint]
        )
    else:
        if pressure >= pressure_bubblepoint:
            solution_gor = solution_gor_initial
        else:
            solution_gor = gor_belowbubble(pressure)
    return solution_gor


def dgor_dpressure_Standing(
    temperature: float,
    pressure: float,
    api_gravity: float,
    gas_specific_gravity: float,
    solution_gor_initial: float,
) -> float:
    r"""Calculate the instantaneous change in GOR with pressure using Standing.

    Parameters
    ----------
    temperature : float
        reservoir temperature in Fahrenheit.
    pressure : float
        reservoir pressure (psia)
    api_gravity : float
        Oil gravity in degrees API.
    gas_specific_gravity : float
        Gas gravity relative to air.
    solution_gor_initial : float
        Initial gas:oil ratio in scf/bbl.

    Returns
    -------
    dgor : float
        Change in GOR per pressure ( scf / scf / psi). :math:`\partial R_s/\partial p`

    Examples
    --------
    >>> dgor_dpressure_Standing(200, 3_000, 35, 0.8, 650)
    0
    >>> dgor_dpressure_Standing(200, 2_000, 35, 0.8, 650)
    0.2824395998221715
    """
    pressure_bubblepoint = pressure_bubblepoint_Standing(
        temperature, api_gravity, gas_specific_gravity, solution_gor_initial
    )
    if pressure >= pressure_bubblepoint:
        d_gor = 0.0
    else:
        d_gor = (
            gas_specific_gravity
            / (0.83 * 18.2)
            * (pressure / 18.2 + 1.4) ** (1 / 0.83 - 1)
            * (10 ** ((0.0125 * api_gravity - 0.00091 * temperature) / 0.83))
        )
    return d_gor


def oil_compressibility_undersat_Standing(
    temperature: float,
    pressure: float,
    api_gravity: float,
    gas_specific_gravity: float,
    solution_gor_initial: float,
) -> float:
    """Calculate the oil compressibility (c_o) using Standing.

    Parameters
    ----------
    temperature : float
        reservoir temperature in Fahrenheit.
    pressure : float
        reservoir pressure (psia)
    api_gravity : float
        Oil gravity in degrees API.
    gas_specific_gravity : float
        Gas gravity relative to air.
    solution_gor_initial : float
        Initial gas:oil ratio in scf/bbl.

    Returns
    -------
    c_o : float
        oil compressibility in 1/psi

    Example
    -------
    >>> oil_compressibility_undersat_Standing(200, 3_000, 35, 0.8, 650)
    1.478788544207282e-05
    """
    oil_specific_gravity = 141.5 / (131.5 + api_gravity)
    pressure_bubblepoint = pressure_bubblepoint_Standing(
        temperature, api_gravity, gas_specific_gravity, solution_gor_initial
    )
    b_o_bubblepoint = b_o_bubblepoint_Standing(
        temperature, api_gravity, gas_specific_gravity, solution_gor_initial
    )
    density_bubblepoint = (
        62.37 * oil_specific_gravity
        + 0.0136 * gas_specific_gravity * solution_gor_initial
    ) / b_o_bubblepoint
    oil_compressibility = 1e-6 * math.exp(
        (density_bubblepoint + 0.004347 * (pressure - pressure_bubblepoint) - 79.1)
        / (7.141e-4 * (pressure - pressure_bubblepoint) - 12.938)
    )
    return oil_compressibility


def oil_compressibility_undersat_Spivey(
    temperature: float,
    pressure: NDArray | float,
    api_gravity: float,
    gas_specific_gravity: float,
    solution_gor_initial: float,
) -> NDArray | float:
    """Calculate the oil compressibility (c_o) using Spivey.

    Parameters
    ----------
    temperature : float
        reservoir temperature in Fahrenheit.
    pressure : float or array
        reservoir pressure (psia)
    api_gravity : float
        Oil gravity in degrees API.
    gas_specific_gravity : float
        Gas gravity relative to air.
    solution_gor_initial : float
        Initial gas:oil ratio in scf/bbl.

    Returns
    -------
    c_o : float
        oil compressibility in 1/psi

    Examples
    --------
    >>> oil_compressibility_undersat_Spivey(200, 3_000, 35, 0.8, 650)
    9.177554347361075e-06
    """
    pressure_bubblepoint = pressure_bubblepoint_Standing(
        temperature, api_gravity, gas_specific_gravity, solution_gor_initial
    )
    # sometimes this is passed an empty pressure array when pressures are all
    # above the bubblepoint
    if np.size(pressure) == 0:
        return np.array([], dtype=np.float64)
    reduced_pressure = pressure / pressure_bubblepoint
    # set up constants for quadratic formula
    # C_labels = [
    #     'api_gravity', 'gas_specific_gravity', 'pressure_bubblepoint',
    #     'reduced_pressure', 'solution_gor_initial','temperature'
    # ]
    C0 = np.array([3.011, -0.835, 3.51, 0.327, -1.918, 2.52])
    C1 = np.array([-2.6254, -0.259, -0.0289, -0.608, -0.642, -2.73])
    C2 = np.array([0.497, 0.382, -0.0584, 0.0911, 0.154, 0.429])
    if np.ndim(reduced_pressure) == 0:
        X = np.log(
            [
                api_gravity,
                gas_specific_gravity,
                pressure_bubblepoint,
                reduced_pressure,
                solution_gor_initial,
                temperature,
            ],
            dtype="f8",
        )
    else:
        X = np.log(
            [
                [
                    api_gravity,
                    gas_specific_gravity,
                    pressure_bubblepoint,
                    rp,
                    solution_gor_initial,
                    temperature,
                ]
                for rp in reduced_pressure
            ]
        )
    z = np.sum(C0) + X @ C1 + (X**2) @ C2
    compressibility_bubblepoint = np.exp(2.434 + 0.475 * z + 0.048 * z**2)
    compressibility_derivative = (-0.608 + 0.1822 * np.log(reduced_pressure)) / pressure
    compressibility_2nd_derivative = (
        compressibility_bubblepoint * (0.475 + 0.096 * z) * compressibility_derivative
    )
    oil_compressibility = 1e-6 * (
        compressibility_bubblepoint
        + (pressure - pressure_bubblepoint) * compressibility_2nd_derivative
    )
    return oil_compressibility


def oil_compressibility_Standing(
    temperature: float,
    pressure: float,
    api_gravity: float,
    gas_specific_gravity: float,
    solution_gor_initial: float,
    temperature_pseudocritical: float,
    pressure_pseudocritical: float,
    temperature_standard: float = 60,
    pressure_standard: float = 14.7,
) -> float:
    """Calculate the oil formation volume factor (Bo) using Standing.

    Parameters
    ----------
    temperature : float
        reservoir temperature in Fahrenheit.
    pressure : float
        reservoir pressure (psia)
    api_gravity : float
        Oil gravity in degrees API.
    gas_specific_gravity : float
        Gas gravity relative to air.
    solution_gor_initial : float
        Initial gas:oil ratio in scf/bbl.
    pressure_pseudocritical : float
        Gas pseudocritical pressure (use gas module) in psia
    temperature_pseudocritical : float
        Gas pseudocritical temperature (use gas module) in F
    temperature_standard : float
        standard temperature in F (default: 60)
    pressure_standard : float
        standard pressure in psia (default: 14.70)

    Returns
    -------
    c_o : float
        oil compressibility in 1/psi

    Examples
    ------
    >>> oil_compressibility_Standing(200, 3000, 35, 0.8, 650, -72.2, 653)

    >>> oil_compressibility_Standing(200, 2000, 35, 0.8, 650, -72.2, 653)

    """
    pressure_bp = pressure_bubblepoint_Standing(
        temperature, api_gravity, gas_specific_gravity, solution_gor_initial
    )
    if pressure >= pressure_bp:
        compressibility = oil_compressibility_undersat_Spivey(
            temperature,
            pressure,
            api_gravity,
            gas_specific_gravity,
            solution_gor_initial,
        )
    else:
        b_g = b_factor_DAK(
            temperature,
            pressure,
            temperature_pseudocritical,
            pressure_pseudocritical,
            temperature_standard,
            pressure_standard,
        )
        solution_gor = solution_gor_Standing(
            temperature,
            pressure,
            api_gravity,
            gas_specific_gravity,
            solution_gor_initial,
        )
        dGOR_dpressure = (
            gas_specific_gravity
            / 0.83
            * (pressure / 18.2 + 1.4) ** (1 / 0.83 - 1)
            / 18.2
            * 10 ** ((0.0125 * api_gravity - 0.00091 * temperature) / 0.83)
        )
        b_o_bubblepoint = b_o_bubblepoint_Standing(
            temperature, api_gravity, gas_specific_gravity, solution_gor_initial
        )
        dBo_dGOR = db_o_dgor_Standing(
            temperature, api_gravity, gas_specific_gravity, solution_gor
        )
        compressibility = (b_g - dBo_dGOR) * dGOR_dpressure / b_o_bubblepoint
    return compressibility


def density_Standing(
    temperature: float,
    pressure: float,
    api_gravity: float,
    gas_specific_gravity: float,
    solution_gor_initial: float,
) -> float:
    """Calculate the oil density using Standing.

    Parameters
    ----------
    temperature : float
        reservoir temperature in Fahrenheit.
    pressure : float
        reservoir pressure (psia)
    api_gravity : float
        Oil gravity in degrees API.
    gas_specific_gravity : float
        Gas gravity relative to air.
    solution_gor_initial : float
        Initial gas:oil ratio in scf/bbl.

    Returns
    -------
    rho_o : float
        oil density (lb/ft^3)

    Examples
    ------
    >>> density_Standing(200, 2000, 35, 0.8, 650)
    44.989365953905136
    """
    oil_specific_gravity = 141.5 / (131.5 + api_gravity)
    solution_gor = solution_gor_Standing(
        temperature, pressure, api_gravity, gas_specific_gravity, solution_gor_initial
    )
    b_o = b_o_Standing(
        temperature, pressure, api_gravity, gas_specific_gravity, solution_gor_initial
    )
    density = (
        62.37 * oil_specific_gravity + 0.0136 * gas_specific_gravity * solution_gor
    ) / b_o
    return density


def viscosity_beggs_robinson(
    temperature: float,
    pressure: float,
    api_gravity: float,
    gas_specific_gravity: float,
    solution_gor_initial: float,
) -> float:
    r"""Calculate the oil viscosity using Beggs-Robinson.

    Parameters
    ----------
    temperature : float
        reservoir temperature in Fahrenheit.
    pressure : float
        reservoir pressure (psia)
    api_gravity : float
        Oil gravity in degrees API.
    gas_specific_gravity : float
        Gas gravity relative to air.
    solution_gor_initial : float
        Initial gas:oil ratio in scf/bbl.

    Returns
    -------
    mu_o : float
        :math:`\mu_o`, the oil viscosity
    """
    pressure_bubblepoint = pressure_bubblepoint_Standing(
        temperature, api_gravity, gas_specific_gravity, solution_gor_initial
    )
    solution_gor = solution_gor_Standing(
        temperature, pressure, api_gravity, gas_specific_gravity, solution_gor_initial
    )
    if pressure >= pressure_bubblepoint:
        mu_o_dead = (
            10 ** (10 ** (3.0324 - 0.02023 * api_gravity) * temperature**-1.163) - 1
        )
        mu_o_live = _mu_dead_to_live_br(mu_o_dead, solution_gor_initial)
        mu_o = mu_o_live * (pressure / pressure_bubblepoint) ** (
            2.6 * pressure**1.187 * np.exp(-11.513 - 8.98e-5 * pressure)
        )
    else:
        mu_o_dead = (
            10 ** (10 ** (3.0324 - 0.02023 * api_gravity) * temperature**-1.163) - 1
        )
        mu_o = _mu_dead_to_live_br(mu_o_dead, solution_gor)
    return mu_o


def _mu_dead_to_live_br(
    mu_dead: NDArray | float, solution_gor_initial: NDArray | float
) -> NDArray | float:
    mu_live = (
        10.715
        * (solution_gor_initial + 100) ** -0.515
        * mu_dead ** (5.44 * (solution_gor_initial + 150) ** -0.338)
    )
    return mu_live
