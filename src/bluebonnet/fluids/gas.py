"""Gas pvt properties, using Dranchuk and Abou-Kassem's correlations."""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.optimize import minimize


# Gas property calculations
def make_nonhydrocarbon_properties(
    nitrogen: float, hydrogen_sulfide: float, co2: float, *others: dict[str, float]
) -> NDArray:
    """Create an array of the nonhydrocarbon molecules present.

    Parameters
    ----------
    nitrogen : float
        compositional fraction of N2
    hydrogen_sulfide : float
        compositional fraction of H2S
    co2 : float
        compositional fraction of CO2
    *others : list(tuple)
       list of tuples of (name, compositional fraction, molecular weight,
                          critical temperature (in R), critical pressure (psia))
       for other non-hydrocarbon molecules

    Returns
    --------
    non_hydrocrabon_properties : NDArray
        structured array of non-hydrocarbon fluid properties

    Examples
    --------
    >>> make_nonhydrocarbon_properties(0.03, 0.012, 0.018)
    """
    non_hydrocarbon_properties = np.array(
        [
            ("Nitrogen", nitrogen, 28.01, 226.98, 492.26),
            ("Hydrogen sulfide", hydrogen_sulfide, 34.08, 672.35, 1299.97),
            ("CO2", co2, 44.01, 547.54, 1070.67),
            *others,
        ],
        dtype=[
            ("name", "U20"),
            ("fraction", "f8"),
            ("molecular weight", "f8"),
            ("critical temperature", "f8"),
            ("critical pressure", "f8"),
        ],
    )
    return non_hydrocarbon_properties


def z_factor_DAK(
    temperature: float,
    pressure: float,
    temperature_pseudocritical: float,
    pressure_pseudocritical: float,
) -> float:
    """Calculate the z-factor for gas using Dranchuk and Abou-Kassem (1975).

    Parameters
    ----------
    temperature : float
        reservoir temperature in Fahrenheit.
    pressure : float
        reservoir pressure in psia
    temperature_pseudocritical : float
        pseudocritical temperature in Fahrenheit.
    pressure_pseudocritical : float
        pseudocritical pressure in psia

    Returns
    -------
    z_fact: float
        z_factor (dimensionless)

    Examples
    ------
    >>> z_factor_DAK(400, 100, -102, 649)
    0.9969013621293381
    """
    A = np.array(
        [
            0.3265,
            -1.07,
            -0.5339,
            0.01569,
            -0.05165,
            0.5475,
            -0.7361,
            0.1844,
            0.1056,
            0.6134,
            0.721,
        ]
    )
    temp_reduced = (temperature + 459.67) / (temperature_pseudocritical + 459.67)
    pressure_reduced = pressure / pressure_pseudocritical
    C = np.zeros(5)  # Taylor series expansion
    C[0] = (
        A[0] * A[1] / temp_reduced
        + A[2] / (temp_reduced**3)
        + A[3] / (temp_reduced**4)
        + A[4] / (temp_reduced**5)
    )
    C[1] = A[5] + A[6] / temp_reduced + A[7] / (temp_reduced**2)
    C[2] = -A[8] * (A[6] / temp_reduced + A[7] / (temp_reduced**2))
    C[3] = A[9] / (temp_reduced**3)
    C[4] = A[9] * A[10] / (temp_reduced**3)

    def calculate_error_fraction(rho: NDArray[np.float64]):
        rho = rho[0]
        B = math.exp(-A[10] * rho**2)
        F_rho = (
            0.27 * pressure_reduced / (temp_reduced * rho)
            - 1
            - C[0] * rho
            - C[1] * rho**2
            - C[2] * rho**5
            - C[3] * rho**2 * B
            - C[4] * rho**4 * B
        )
        DF_rho = (
            -0.27 * pressure_reduced / (temp_reduced * rho**2)
            - C[0]
            - 2 * C[1] * rho
            - 5 * C[2] * rho**4
            - 2 * C[3] * rho * B
            + 2 * A[10] * rho * B * C[3] * rho**2
            - 4 * C[4] * rho**3 * B
            + 2 * A[10] * rho * B * C[4] * rho**4
        )
        return math.fabs(F_rho / DF_rho)

    rho_guess = 0.27 * pressure_reduced / temp_reduced
    bounds = (
        (rho_guess / 5, rho_guess * 20),
    )  # bounds go from a z-factor of 0.05 to 5
    result = minimize(calculate_error_fraction, rho_guess, bounds=bounds)
    rho = result.x[0]
    Z_factor = 0.27 * pressure_reduced / (rho * temp_reduced)
    return Z_factor


def z_factor_hallyarbrough(pressure: float, temperature: float) -> float:
    """Get Z-factor for gas from Hall-Yarbrough's iterative approach.

    Parameters
    ----------
    p : float | NDArray[np.float64]
        pressure (psi)
    t : float
        temperature (Rankine)

    Returns
    -------
    zfact: float:
        z-factor

    References
    ----------
    `Hall-Yarbrough estimation <https://wiki.whitson.com/eos/eos_models/zfactor/index.html#hall-yarbrough-estimation-of-gas-z-factor>`_
    """  # noqa: E501
    t = 1 / temperature
    y = 0.001
    fdum = 1
    while np.abs(fdum) > 0.001:
        fdum = (
            -0.06125 * pressure * t * np.exp(-1.2 * (1 - t) ** 2)
            + (y + y**2 + y**3 - y**4) / (1 - y) ** 3
            - (14.76 * t - 9.76 * t**2 + 4.58 * t**3) * y**2
            + (90.7 * t - 242.2 * t**2 + 42.4 * t**3) * y ** (2.18 + 2.82 * t)
        )
        dfdy = (
            (1 + 4 * y + 4 * y**2 - 4 * y**3 + y**4) / (1 - y) ** 4
            - (29.52 * t - 19.52 * t**2 + 9.16 * t**3) * y
            + (2.18 + 2.82 * t)
            * (90.7 * t - 242.2 * t**2 + 42.4 * t**3)
            * y ** (1.18 + 2.82 * t)
        )
        y = y - fdum / dfdy
    zfact = 0.06125 * pressure * t * np.exp(-1.2 * (1 - t) ** 2) / y
    return zfact


def b_factor_DAK(
    temperature: float,
    pressure: float,
    temperature_pseudocritical: float,
    pressure_pseudocritical: float,
    temperature_standard: float = 60,
    pressure_standard: float = 14.70,
) -> float:
    """Calculate the b-factor for gas using Dranchuk and Abou-Kassem (1975).

    Parameters
    ----------
    temperature : float
        reservoir temperature in Fahrenheit.
    pressure : float
        reservoir pressure in psia
    temperature_pseudocritical : float
        pseudocritical temperature in Fahrenheit.
    pressure_pseudocritical : float
        pseudocritical pressure in psia

    Returns
    -------
    b_g : float
        b-factor (reservoir barrels / scf)

    Examples
    -------
    >>> b_factor_DAK(400, 100, -102, 649, 60, 14.7)
    0.04317415921420302
    """
    z_factor = z_factor_DAK(
        temperature, pressure, temperature_pseudocritical, pressure_pseudocritical
    )
    b_factor = (
        z_factor
        * (temperature + 459.67)
        * pressure_standard
        / ((temperature_standard + 459.67) * pressure)
    )
    return b_factor / 5.615


def density_DAK(
    temperature: float,
    pressure: float,
    temperature_pseudocritical: float,
    pressure_pseudocritical: float,
    specific_gravity: float,
) -> float:
    """Calculate the density for gas using Dranchuk and Abou-Kassem (1975).

    Parameters
    ----------
    temperature : float
        reservoir temperature in Fahrenheit.
    pressure : float
        reservoir pressure in psia
    temperature_pseudocritical : float
        pseudocritical temperature in Fahrenheit.
    pressure_pseudocritical : float
        pseudocritical pressure in psia
    specific_gravity : float
        specific gravity relative to air (molecular weight/ molecular weight)

    Returns
    -------
    rho_g : float
        density_gas (lb / cubic ft)

    Examples
    -------
    >>> density_DAK(400, 100, -102, 649, 0.65) # returns 0.2143
    """
    MOLECULAR_WEIGHT_AIR = 28.964
    R = 10.73159
    molecular_weight = MOLECULAR_WEIGHT_AIR * specific_gravity
    z_factor = z_factor_DAK(
        temperature, pressure, temperature_pseudocritical, pressure_pseudocritical
    )
    density_gas = pressure * molecular_weight / (z_factor * R * (temperature + 459.67))
    return density_gas


def compressibility_DAK(
    temperature: float,
    pressure: float,
    temperature_pseudocritical: float,
    pressure_pseudocritical: float,
) -> float:
    """Calculate the compressibility for gas using Dranchuk and Abou-Kassem (1975).

    Parameters
    ----------
    temperature : float
        reservoir temperature in Fahrenheit.
    pressure : float
        reservoir pressure in psia
    temperature_pseudocritical : float
        pseudocritical temperature in Fahrenheit.
    pressure_pseudocritical : float
        pseudocritical pressure in psia

    Returns
    -------
    c_g: float
        compressibility (1 / psi)

    Examples
    ------
    >>> compressibility_DAK(400, 104.7, -102, 649)
    0.009576560643021937
    """
    A = np.array(
        [
            0.3265,
            -1.07,
            -0.5339,
            0.01569,
            -0.05165,
            0.5475,
            -0.7361,
            0.1844,
            0.1056,
            0.6134,
            0.721,
        ]
    )
    temp_reduced = (temperature + 459.67) / (temperature_pseudocritical + 459.67)
    pressure_reduced = pressure / pressure_pseudocritical
    z_factor = z_factor_DAK(
        temperature, pressure, temperature_pseudocritical, pressure_pseudocritical
    )
    rho = 0.27 * pressure_reduced / (temp_reduced * z_factor)
    dz_drho = (
        A[0]
        + A[1] / temp_reduced
        + A[2] / (temp_reduced**3)
        + A[3] / (temp_reduced**4)
        + A[4] / (temp_reduced**5)
    )
    dz_drho += 2 * (A[5] + A[6] / temp_reduced + A[7] / (temp_reduced**2)) * rho
    dz_drho += -5 * A[8] * (A[6] / temp_reduced + A[7] / (temp_reduced**2)) * rho**4
    dz_drho += (
        2 * A[9] * rho / (temp_reduced**3)
        + 2 * A[9] * A[10] * rho**3 / (temp_reduced**3)
        - 2 * A[9] * A[10] ** 2 * rho**5 / (temp_reduced**3)
    ) * math.exp(-A[10] * rho**2)
    compressibility_reduced = 1.0 / pressure_reduced - 0.27 / (
        z_factor**2 * temp_reduced
    ) * (dz_drho / (1 + rho * dz_drho / z_factor))
    compressibility = compressibility_reduced / pressure_pseudocritical
    return compressibility


def viscosity_Sutton(
    temperature: float,
    pressure: float,
    temperature_pseudocritical: float,
    pressure_pseudocritical: float,
    specific_gravity: float,
) -> float:
    """Calculate the viscosity for gas using Sutton's Fudamental PVT Calculations (2007).

    Parameters
    ----------
    temperature : float
        reservoir temperature in Fahrenheit.
    pressure : float
        reservoir pressure in psia
    temperature_pseudocritical : float
        pseudocritical temperature in Fahrenheit.
    pressure_pseudocritical : float
        pseudocritical pressure in psia
    specific_gravity : float
        specific gravity relative to air (density/density)

    Returns
    -------
    mu_g : float
        viscosity_gas (centipoise)

    Examples
    -------
    >>> viscosity_Sutton(400, 100, -102, 649, 0.65)
    0.01652719692109309
    """
    MOLECULAR_WEIGHT_AIR = 28.964
    temp_reduced = (temperature + 459.67) / (temperature_pseudocritical + 459.67)
    molecular_weight = specific_gravity * MOLECULAR_WEIGHT_AIR
    rho = density_DAK(
        temperature,
        pressure,
        temperature_pseudocritical,
        pressure_pseudocritical,
        specific_gravity,
    )
    rho *= 16.018463 / 1e3  # convert to grams / cc
    xi = 0.949 * (
        (temperature_pseudocritical + 459.67)
        / (molecular_weight**3 * pressure_pseudocritical**4)
    ) ** (1.0 / 6.0)
    viscosity_lowpressure = (
        1e-5
        * (
            8.07 * temp_reduced**0.618
            - 3.57 * math.exp(-0.449 * temp_reduced)
            + 3.40 * math.exp(-4.058 * temp_reduced)
            + 0.18
        )
        / xi
    )
    X = 3.47 + 1588.0 / (temperature + 459.67) + 9e-4 * molecular_weight
    Y = 1.66378 - 4.679e-3 * X
    viscosity = viscosity_lowpressure * math.exp(X * rho**Y)
    return viscosity


def pseudocritical_point_Sutton(
    specific_gravity: float,
    non_hydrocarbon_properties: NDArray,
    fluid: str = "wet gas",
) -> tuple[float, float]:
    """Calculate the pseudocritical pressure and temperature from Sutton (2007).

    Parameters
    ----------
    specific_gravity : float
        specific gravity relative to air (molecular weight / molecular weight)
    non_hydrocarbon_properties: np.NDArray
        record array of non-hydrocarbon fluid properties
        **MUST HAVE H2S as second row, CO2 as third row**
    fluid : string
        whether the gas is 'dry gas' or 'wet gas'

    Returns
    -------
    pseudocritical_temp : float
        temperature_pseudocritical (F)
    pseudocritical_p : float
        pressure_pseudocritical (psia)

    Examples
    ------
    >>> non_hydrocarbon_properties = make_nonhydrocarbon_properties(0.03, 0.012, 0.018)
    >>> points_pseudocritical_Sutton(0.65, non_hydrocarbon_properties, "dry gas")
    (-102.21827232417752, 648.510797253794)
    >>> non_hydrocarbon_properties = make_nonhydrocarbon_properties(0.05, 0.01, 0.04)
    >>> points_pseudocritical_Sutton(0.8, non_hydrocarbon_properties, "wet gas")
    (-72.20351526841193, 653.2582064200534)
    """
    if fluid not in ("dry gas", "wet gas"):
        msg = f"fluid must be one of ('dry gas','wet gas'), not {fluid}"
        raise ValueError(msg)
    MOLECULAR_WEIGHT_AIR = 28.964
    fraction = non_hydrocarbon_properties["fraction"]
    molecular_weight = non_hydrocarbon_properties["molecular weight"]
    temp_critical_nonhc = non_hydrocarbon_properties["critical temperature"]
    pressure_critical_nonhc = non_hydrocarbon_properties["critical pressure"]
    fraction_hydrocarbon = 1 - non_hydrocarbon_properties["fraction"].sum()
    specific_gravity_hydrocarbon = (
        specific_gravity - sum(fraction * molecular_weight) / MOLECULAR_WEIGHT_AIR
    ) / fraction_hydrocarbon
    if fluid == "dry gas":
        temp_critical_hydrocarbon = (
            120.1
            + 429 * specific_gravity_hydrocarbon
            - 62.9 * specific_gravity_hydrocarbon**2
        )
        pressure_critical_hydrocarbon = (
            671.1
            - 14 * specific_gravity_hydrocarbon
            - 34.3 * specific_gravity_hydrocarbon**2
        )
    else:
        temp_critical_hydrocarbon = (
            164.3
            + 357.7 * specific_gravity_hydrocarbon
            - 67.7 * specific_gravity_hydrocarbon**2
        )
        pressure_critical_hydrocarbon = (
            744
            - 125.4 * specific_gravity_hydrocarbon
            + 5.9 * specific_gravity_hydrocarbon**2
        )
    temperature_star = fraction_hydrocarbon * temp_critical_hydrocarbon + sum(
        fraction * temp_critical_nonhc
    )
    pressure_star = fraction_hydrocarbon * pressure_critical_hydrocarbon + sum(
        fraction * pressure_critical_nonhc
    )
    # epsilon is the Wichert and Aziz (1970) correction
    epsilon = (
        120 * (fraction[2] + fraction[1]) ** 0.9
        - 120 * (fraction[2] + fraction[1]) ** 1.6
        + 15 * (math.sqrt(fraction[1]) - fraction[1] ** 4)
    )
    temperature_pseudocritical = temperature_star - epsilon
    pressure_pseudocritical = (
        pressure_star
        * (temperature_star - epsilon)
        / (temperature_star + fraction[1] * (1 - fraction[1]) * epsilon)
    )
    return temperature_pseudocritical - 459.67, pressure_pseudocritical


def pseudopressure_Hussainy(
    temperature: float,
    pressure: float,
    temperature_pseudocritical: float,
    pressure_pseudocritical: float,
    specific_gravity: float,
    pressure_standard: float = 14.70,
) -> float:
    """Calculate the pseudopressure for gas using Al Hussainy (1966).

    Parameters
    ----------
    temperature : float
        reservoir temperature in Fahrenheit.
    pressure : float
        reservoir pressure in psia
    temperature_pseudocritical : float
        pseudocritical temperature in Fahrenheit.
    pressure_pseudocritical : float
        pseudocritical pressure in psia
    specific_gravity : float
        specific gravity relative to air (density/density)

    Returns
    -------
    m : float
        pseudopressure (psi^2 / centipoise)

    Examples
    -------
    >>> pseudopressure_Hussainy(400, 100, -102, 649, 0.65)
    593363.7626437937
    """

    def integrand(pressure: float):
        viscosity = viscosity_Sutton(
            temperature,
            pressure,
            temperature_pseudocritical,
            pressure_pseudocritical,
            specific_gravity,
        )
        z_factor = z_factor_DAK(
            temperature, pressure, temperature_pseudocritical, pressure_pseudocritical
        )
        return 2 * pressure / (viscosity * z_factor)

    # use scipy's quadrature (wraps QUADPACK) for integration
    pseudopressure, err = quad(integrand, pressure_standard, pressure, limit=100)
    return pseudopressure
