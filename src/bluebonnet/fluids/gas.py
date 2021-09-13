# gas.py
# Gas property calculations
def make_nonhydrocarbon_properties(nitrogen:float, hydrogen_sulfide:float, co2:float, *others) -> npt.ArrayLike:
    """
    Creates an array of the nonhydrocarbon molecules present

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
    np.ndarray
        structured array of non-hydrocarbon fluid properties

    Examples
    --------
    make_nonhydrocarbon_properties(0.03, 0.012, 0.018)
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
    """
    Calculates the z-factor for gas using Dranchuk and Abou-Kassem (1975)

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
    float
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
        + A[2] / (temp_reduced ** 3)
        + A[3] / (temp_reduced ** 4)
        + A[4] / (temp_reduced ** 5)
    )
    C[1] = A[5] + A[6] / temp_reduced + A[7] / (temp_reduced ** 2)
    C[2] = -A[8] * (A[6] / temp_reduced + A[7] / (temp_reduced ** 2))
    C[3] = A[9] / (temp_reduced ** 3)
    C[4] = A[9] * A[10] / (temp_reduced ** 3)

    def calculate_error_fraction(rho):
        rho = rho[0]
        B = math.exp(-A[10] * rho ** 2)
        F_rho = (
            0.27 * pressure_reduced / (temp_reduced * rho)
            - 1
            - C[0] * rho
            - C[1] * rho ** 2
            - C[2] * rho ** 5
            - C[3] * rho ** 2 * B
            - C[4] * rho ** 4 * B
        )
        DF_rho = (
            -0.27 * pressure_reduced / (temp_reduced * rho ** 2)
            - C[0]
            - 2 * C[1] * rho
            - 5 * C[2] * rho ** 4
            - 2 * C[3] * rho * B
            + 2 * A[10] * rho * B * C[3] * rho ** 2
            - 4 * C[4] * rho ** 3 * B
            + 2 * A[10] * rho * B * C[4] * rho ** 4
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


def b_factor_gas_DAK(
    temperature: float,
    pressure: float,
    temperature_pseudocritical: float,
    pressure_pseudocritical: float,
    temperature_standard: float = 60,
    pressure_standard: float = 14.70,
) -> float:
    """
    Calculates the b-factor for gas using Dranchuk and Abou-Kassem (1975)

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
    float
        b-factor (reservoir barrels / scf)
    
    Examples
    -------
    >>> b_factor_gas_DAK(400, 100, -102, 649, 60, 14.7)
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


def density_gas_DAK(
    temperature: float,
    pressure: float,
    temperature_pseudocritical: float,
    pressure_pseudocritical: float,
    specific_gravity: float,
) -> float:
    """
    Calculates the density for gas using Dranchuk and Abou-Kassem (1975)

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
    float
        density_gas (lb / cubic ft)
    
    Examples
    -------
    density_gas_DAK(400, 100, -102, 649, 0.65) # returns 0.2143
    """
    MOLECULAR_WEIGHT_AIR = 28.964
    R = 10.73159
    molecular_weight = MOLECULAR_WEIGHT_AIR * specific_gravity
    z_factor = z_factor_DAK(
        temperature, pressure, temperature_pseudocritical, pressure_pseudocritical
    )
    density_gas = pressure * molecular_weight / (z_factor * R * (temperature + 459.67))
    return density_gas


def compressibility_gas_DAK(
    temperature: float,
    pressure: float,
    temperature_pseudocritical: float,
    pressure_pseudocritical: float,
) -> float:
    """
    Calculates the compressibility for gas using Dranchuk and Abou-Kassem (1975)

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
    float
        compressibility (1 / psi)

    Examples
    ------
    >>> compressibility_gas_DAK(400, 104.7, -102, 649) 
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
        + A[2] / (temp_reduced ** 3)
        + A[3] / (temp_reduced ** 4)
        + A[4] / (temp_reduced ** 5)
    )
    dz_drho += 2 * (A[5] + A[6] / temp_reduced + A[7] / (temp_reduced ** 2)) * rho
    dz_drho += -5 * A[8] * (A[6] / temp_reduced + A[7] / (temp_reduced ** 2)) * rho ** 4
    dz_drho += (
        2 * A[9] * rho / (temp_reduced ** 3)
        + 2 * A[9] * A[10] * rho ** 3 / (temp_reduced ** 3)
        - 2 * A[9] * A[10] ** 2 * rho ** 5 / (temp_reduced ** 3)
    ) * math.exp(-A[10] * rho ** 2)
    compressibility_reduced = 1.0 / pressure_reduced - 0.27 / (
        z_factor ** 2 * temp_reduced
    ) * (dz_drho / (1 + rho * dz_drho / z_factor))
    compressibility = compressibility_reduced / pressure_pseudocritical
    return compressibility


def viscosity_gas_Sutton(
    temperature: float,
    pressure: float,
    temperature_pseudocritical: float,
    pressure_pseudocritical: float,
    specific_gravity: float,
) -> float:
    """
    Calculates the viscosity for gas using Sutton's Fudamental PVT Calculations (2007)

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
    float
        viscosity_gas (centipoise)
    
    Examples
    -------
    >>> viscosity_gas_Sutton(400, 100, -102, 649, 0.65)
    0.01652719692109309
    """
    MOLECULAR_WEIGHT_AIR = 28.964
    temp_reduced = (temperature + 459.67) / (temperature_pseudocritical + 459.67)
    molecular_weight = specific_gravity * MOLECULAR_WEIGHT_AIR
    rho = density_gas_DAK(
        temperature,
        pressure,
        temperature_pseudocritical,
        pressure_pseudocritical,
        specific_gravity,
    )
    rho *= 16.018463 / 1e3  # convert to grams / cc
    xi = 0.949 * (
        (temperature_pseudocritical + 459.67)
        / (molecular_weight ** 3 * pressure_pseudocritical ** 4)
    ) ** (1.0 / 6.0)
    viscosity_lowpressure = (
        1e-5
        * (
            8.07 * temp_reduced ** 0.618
            - 3.57 * math.exp(-0.449 * temp_reduced)
            + 3.40 * math.exp(-4.058 * temp_reduced)
            + 0.18
        )
        / xi
    )
    X = 3.47 + 1588.0 / (temperature + 459.67) + 9e-4 * molecular_weight
    Y = 1.66378 - 4.679e-3 * X
    viscosity = viscosity_lowpressure * math.exp(X * rho ** Y)
    return viscosity


def pseudopressure_Hussainy(
    temperature: float,
    pressure: float,
    temperature_pseudocritical: float,
    pressure_pseudocritical: float,
    specific_gravity: float,
    pressure_standard: float = 14.70,
) -> float:
    """
    Calculates the pseudopressure for gas using Al Hussainy (1966)

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
    float
        pseudopressure (psi^2 / centipoise)
    
    Examples
    -------
    >>> pseudopressure_Hussainy(400, 100, -102, 649, 0.65)
    593363.7626437937
    """

    def integrand(pressure):
        viscosity = viscosity_gas_Sutton(
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
