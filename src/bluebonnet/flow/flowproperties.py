"""Flow properties for reservoir fluids.

Pressure (or pseudo-pressure) changes affect all sorts of things. This
module includes structures for storing these pressure-dependent properties
in order to aid the reservoir simulators in the `reservoir` module.
"""
from __future__ import annotations

import copy
import warnings
from collections import namedtuple
from typing import Mapping

import numpy as np
from numpy import ndarray
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import LinearNDInterpolator, interp1d


class FlowProperties:
    r"""
    Flow properties for the system.

    This is used to translate from scaled pseudopressure to diffusivity and to capture
    the effect of expansion

    Attributes
    ----------
    pvt_props : Mapping

        has accessors for pseudopressure, alpha, compressibility, viscosity, z-factor

        pseudopressure: pseudopressure in psi^2/centipoise

        pressure: pore pressure

        alpha: hydraulic diffusivity

        compressibility: total compressibility

        viscosity: dynamic viscosity :math:`\mu`

        z-factor: compressibility factor, :math:`Z = p/\rho R T`

        m-scaled: pseudopressure scaled by initial pseudopressure

    m_i : float
        pseudopressure that corresponds to initial reservoir pressure
    m_scaled_func: function
        calculates scaled pseudopressure from pressure
    alpha_func : function
        calculated hydraulic diffusivity from scaled pseudopressure
    """

    def __init__(self, pvt_props: Mapping[str, ndarray], p_i: float):
        """Wrap table of flow properties with useful methods.

        If alpha is not in the table, calculates hydraulic diffusivity as a function of
        compressibility and viscosity.

        Parameters
        -----------
        pvt_props : Mapping
            has accessors for pseudopressure, alpha (optional), compressibility, viscosity,
            z-factor

        p_i : float
            Initial reservoir pressure
        """
        pvt_props = copy.copy(pvt_props)
        need_cols_long = {
            "pseudopressure",
            "compressibility",
            "pressure",
            "viscosity",
            "z-factor",
        }
        need_cols_short = {"pressure", "pseudopressure", "alpha"}
        if (
            need_cols_long.intersection(pvt_props) != need_cols_long
            and need_cols_short.intersection(pvt_props) != need_cols_short
        ):
            raise ValueError("Need pvt_props to have: " + ", ".join(need_cols_short))
        if "alpha" in pvt_props:
            m_scale_func = interp1d(
                pvt_props["pressure"], 1 / pvt_props["pseudopressure"]
            )
            warnings.warn(
                "warning: scaling pseudopressure, using user's hydraulic diffusivity"
            )
            m_scaling_factor = m_scale_func(p_i)
        else:
            pseudopressure_scaling = (
                1
                / 2
                * pvt_props["compressibility"]
                * pvt_props["pressure"]
                * pvt_props["viscosity"]
                * pvt_props["z-factor"]
                / pvt_props["pressure"] ** 2
            )
            m_scale_func = interp1d(pvt_props["pressure"], pseudopressure_scaling)
            m_scaling_factor = m_scale_func(p_i)
            pvt_props["alpha"] = 1 / (  # mypy: ignore
                pvt_props["compressibility"] * pvt_props["viscosity"]
            )
        pvt_props["m-scaled"] = pvt_props["pseudopressure"] * m_scaling_factor
        self.m_scaled_func = interp1d(pvt_props["pressure"], pvt_props["m-scaled"])
        self.m_i = self.m_scaled_func(p_i)
        self.alpha = interp1d(
            pvt_props["m-scaled"],
            pvt_props["alpha"],
            fill_value=(min(pvt_props["alpha"]), max(pvt_props["alpha"])),
            bounds_error=False,
        )
        self.pvt_props = pvt_props

    def __repr__(self):
        """Representation."""
        return self.pvt_props.__repr__()


FlowPropertiesOnePhase = FlowProperties


class FlowPropertiesTwoPhase(FlowProperties):
    """
    Flow properties for the system.

    This is used to translate from scaled pseudopressure to diffusivity and to capture
    the effect of expansion

    References
    ----------
    Ruiz Maraggi, L.M., Lake, L.W. and Walsh, M.P., 2020. "A Two-Phase Non-Linear One-
        Dimensional Flow Model for Reserves Estimation in Tight Oil and Gas
        Condensate Reservoirs Using Scaling Principles." In SPE Latin American and
        Caribbean Petroleum Engineering Conference. OnePetro.
        https://doi.org/10.2118/199032-MS
    """

    @classmethod
    def from_table(
        cls,
        pvt_props: Mapping,
        kr_props: Mapping,
        reference_densities: dict,
        phi: float,
        Sw: float,
        p_i: float,
    ):
        """Create FlowProperties from tables.

        Using fluid properties from a PVT table and a relative permeability table,
        builds an interpolator for alpha from pseudopressure

        Parameters
        ----------
        df_pvt : Mapping
            table of PVT properties by pressure with co-varying So
            columns: pseudopressure, pressure, Bo, Bg, Bw, Rs, Rv, mu_o, mu_g, mu_w, So
        df_kr : Mapping
            table of relative permeabilities
            columns: So, Sg, Sw, kro, krg, krw
        reference_densities: dictionary of strings pointing to floats
            density at STP for oil, gas, and water
            rho_o0: oil, rho_g0: gas, rho_w0: water
        phi : float
            formation porosity (varies from 0 to 1)
        Sw : float
            water saturation (assumes Sw below or equal to Swirr)
        fvf_scale : float
            formation volume factor scale for recovery factor
        """
        need_cols_pvt = {
            "pseudopressure",
            "pressure",
            "Bo",
            "Bg",
            "Bw",
            "Rs",
            "Rv",
            "mu_o",
            "mu_g",
            "mu_w",
            "So",
        }
        need_cols_kr = {"So", "Sg", "Sw", "kro", "krg", "krw"}
        if need_cols_pvt.intersection(pvt_props) != need_cols_pvt:
            raise ValueError(f"df_pvt needs all of {need_cols_pvt}")
        if need_cols_kr.intersection(kr_props) != need_cols_kr:
            raise ValueError(f"df_kr needs all of {need_cols_kr}")
        pvt = {
            prop: interp1d(
                pvt_props["pressure"], pvt_props[prop], fill_value="extrapolate"
            )
            for prop in need_cols_pvt
        }
        pvt.update(reference_densities)
        # pvt.update({"rho_o0": rho_o0, "rho_g0": rho_g0, "rho_w0": rho_w0})
        kr = {
            fluid: interp1d(kr_props["So"], kr_props[fluid])
            for fluid in ("kro", "krg", "krw")
        }
        alpha_calc = alpha_multiphase(
            pvt_props["pressure"], pvt_props["So"], phi, Sw, pvt, kr
        )
        pseudopressure = pseudopressure_threephase(
            pvt_props["pressure"], pvt_props["So"], Sw, pvt, kr
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            object = cls(
                {
                    "pressure": pvt_props["pressure"],
                    "pseudopressure": pseudopressure,
                    "alpha": alpha_calc,
                },
                p_i,
            )
        object.pvt = pvt
        object.kr = kr
        return object


def alpha_multiphase(
    pressure: ndarray, So: ndarray, phi: float, Sw: float, pvt: dict, kr: dict
) -> ndarray:
    """Calculate hydraulic diffusivity for a multiphase (two or three phase) system.

    Args:
        pressure (ndarray): absolute pressure for the cells
        So (ndarray): Oil saturation for the cells
        phi (float): porosity (assumed constant), in range from 0 to 1
        Sw (float): Water saturation (assumed constant)
        pvt (dict): PVT properties including
            `rho_x0`: density at reference pressure
            `mu_x`: viscosity (function of pressure)
            `Bx`: formation volume factor (function of pressure)
            for the x-components o (oil), g (gas), w (water)
            `Rv`, `Rs`: dissolved and free gas (functions of pressure)
        kr (dict): relative permeability for x-phases (o, g, w)
            `krx`: relative permeability (function of So)

    Returns:
        ndarray: total hydraulic diffusivity
    """
    lambda_combined = lambda_combined_func(pressure, So, pvt, kr)
    compressibility_combined = compressibility_combined_func(pressure, So, phi, Sw, pvt)
    return lambda_combined / compressibility_combined


def lambda_combined_func(
    pressure: ndarray, So: ndarray, pvt: dict, kr: dict
) -> ndarray:
    """Calculate mobility for three phase system.

    Args:
        pressure (ndarray): reservoir pressure for the cells
        So (ndarray): oil saturation (varies from 0-1) for the cells
        pvt (dict): PVT properties including
            `rho_x0`: density at reference pressure
            `mu_x`: viscosity (function of pressure)
            `Bx`: formation volume factor (function of pressure)
            for the x-components o (oil), g (gas), w (water)
            `Rv`, `Rs`: dissolved and free gas (functions of pressure)
        kr (dict): relative permeability for x-phases (o, g, w)
            `krx`: relative permeability (function of So)

    Returns:
        ndarray: mobility for the cells
    """
    lambda_oil = pvt["rho_o0"] * (
        pvt["Rv"](pressure)
        * kr["krg"](So)
        / (pvt["mu_g"](pressure) * pvt["Bg"](pressure))
        + kr["kro"](So) / (pvt["mu_o"](pressure) * pvt["Bo"](pressure))
    )
    lambda_gas = pvt["rho_g0"] * (
        pvt["Rs"](pressure)
        * kr["kro"](So)
        / (pvt["mu_o"](pressure) * pvt["Bo"](pressure))
        + kr["krg"](So) / (pvt["mu_g"](pressure) * pvt["Bg"](pressure))
    )
    lambda_water = pvt["rho_w0"] * (
        kr["krw"](So) / (pvt["mu_w"](pressure) * pvt["Bw"](pressure))
    )  # + pvt["Rsw"](pressure) * kr["k_g"](So) / (pvt["mu_g"](pressure) * pvt["Bg"](pressure))
    return lambda_oil + lambda_gas + lambda_water


def compressibility_combined_func(
    pressure: ndarray, So: ndarray, phi: float, Sw: float | ndarray, pvt: dict
) -> ndarray:
    """Calculate three-phase compressibility.

    Args:
        pressure (ndarray): reservoir pressure for the cells
        So (ndarray): oil saturation (varies from 0-1) for the cells
        pvt (dict): PVT properties including
            `rho_x0`: density at reference pressure
            `mu_x`: viscosity (function of pressure)
            `Bx`: formation volume factor (function of pressure)
            for the x-components o (oil), g (gas), w (water)
            `Rv`, `Rs`: dissolved and free gas (functions of pressure)
        kr (dict): relative permeability for x-phases (o, g, w)
            `krx`: relative permeability (function of So)


    Returns:
        ndarray: Total fluid compressibility
    """
    Sg = 1 - So - Sw
    oil_cp = (
        phi
        * pvt["rho_o0"]
        * (
            pvt["Rv"](pressure + 0.5) * Sg / pvt["Bg"](pressure + 0.5)
            + So / pvt["Bo"](pressure + 0.5)
            - pvt["Rv"](pressure - 0.5) * Sg / pvt["Bg"](pressure - 0.5)
            + So / pvt["Bo"](pressure - 0.5)
        )
    )
    gas_cp = (
        phi
        * pvt["rho_g0"]
        * (
            pvt["Rs"](pressure + 0.5) * So / pvt["Bo"](pressure + 0.5)
            + Sg / pvt["Bg"](pressure + 0.5)
            - pvt["Rs"](pressure - 0.5) * So / pvt["Bo"](pressure - 0.5)
            + Sg / pvt["Bg"](pressure - 0.5)
        )
    )
    water_cp = (
        phi
        * pvt["rho_w0"]
        * (Sw / pvt["Bw"](pressure + 0.5) - Sw / pvt["Bw"](pressure - 0.5))
    )
    return oil_cp + gas_cp + water_cp


def pseudopressure_threephase(
    pressure: ndarray, So: ndarray, Sw: float | ndarray, pvt: dict, kr: dict
) -> ndarray:
    """Calculate pseudopressure over pressure for three-phase fluid.

    Args:
        pressure (ndarray): reservoir pressure for the cells
        So (ndarray): oil saturation (varies from 0-1) for the cells
        pvt (dict): PVT properties including
            `rho_x0`: density at reference pressure
            `mu_x`: viscosity (function of pressure)
            `Bx`: formation volume factor (function of pressure)
            for the x-components o (oil), g (gas), w (water)
            `Rv`, `Rs`: dissolved and free gas (functions of pressure)
        kr (dict): relative permeability for x-phases (o, g, w)
            `krx`: relative permeability (function of So)

    Returns:
    -------
        ndarray: pseudopressure
    """
    lambda_oil = pvt["rho_o0"] * (
        pvt["Rv"](pressure)
        * kr["krg"](So)
        / (pvt["mu_g"](pressure) * pvt["Bg"](pressure))
        + kr["kro"](So) / (pvt["mu_o"](pressure) * pvt["Bo"](pressure))
    )
    lambda_gas = pvt["rho_g0"] * (
        pvt["Rs"](pressure)
        * kr["kro"](So)
        / (pvt["mu_o"](pressure) * pvt["Bo"](pressure))
        + kr["krg"](So) / (pvt["mu_g"](pressure) * pvt["Bg"](pressure))
    )
    lambda_water = pvt["rho_w0"] * (
        kr["krw"](So) / (pvt["mu_w"](pressure) * pvt["Bw"](pressure))
    )
    integrand = lambda_oil + lambda_gas + lambda_water
    pseudopressure = cumulative_trapezoid(pressure, integrand, initial=0)
    return pseudopressure


class FlowPropertiesMultiPhase(FlowProperties):
    """
    Flow properties for a multiphase system.

    This is used to translate from scaled pseudopressure and saturations to diffusivity
    and to capture the effect of expansion

    Parameters
    ----------
    df : Mapping
        has columns for pseudopressure, alpha, So, Sg, Sw
        pseudopressure: pseudopressure scaled from 0 for frac face, 1 for initial
            reservoir conditions
        alpha: hydraulic diffusivity (needn't be scaled)
        So: oil saturation
        Sg: gas saturation
        Sw: water saturation
    fvf_scale: dict
        includes Bo,Bg,Bw at initial conditions divided by FVF at the frac face
    """

    def __init__(self, df: Mapping[str, ndarray]):
        """Wrap table of pressure-dependent flow properties.

        Args:
            df (Mapping[str, ndarray]): table, including pseudopressure, So, Sg, Sw

        Raises:
            ValueError: Table columns are missing
        """
        need_cols = {"pseudopressure", "alpha", "So", "Sg", "Sw"}
        if need_cols.intersection(df.columns) != need_cols:
            raise ValueError(
                "Need input dataframe to have 'pseudopressure', 'compressibility',"
                " and 'alpha' columns"
            )
        self.df = df
        x = df["pseudopressure", "So", "Sg", "Sw"]
        self.alpha = LinearNDInterpolator(x, df["alpha"])


RelPermParams = namedtuple(
    "RelPermParams", "n_o n_g n_w S_or S_wc S_gc k_ro_max k_rw_max k_rg_max"
)


def relative_permeabilities(
    saturations: ndarray,
    params: RelPermParams,
) -> ndarray:
    """Brooks-Corey power-law relative permeability.

    Parameters
    ----------
    saturations: ndarray, specifically a record array
        records include So, Sg, Sw
    params: RelPermParams
        Includes Corey exponents, residual saturations, and max relative permeabilities

    Returns
    -------
    k_rel: numpy record array
        records include k_o, k_w, k_g (aka oil, water gas)
    """
    if np.any(np.abs([sum(v) - 1 for v in saturations]) > 1e-3):
        raise ValueError("Saturations must sum to 1")
    if max(params.n_o, params.n_g, params.n_w) > 6:
        raise ValueError("Exponents must be less than 6")
    if min(params.n_o, params.n_g, params.n_w) < 1:
        raise ValueError("Exponents must be at least 1")
    if min(params.S_or, params.S_wc, params.S_gc) < 0:
        raise ValueError("Critical saturations must be at least 0")
    if max(params.S_or, params.S_wc, params.S_gc) > 1:
        raise ValueError("Critical saturations must be less than 1")
    if min(params.k_ro_max, params.k_rw_max, params.k_rg_max) < 0:
        raise ValueError("Max relative permeability must be at least 0")
    if max(params.k_ro_max, params.k_rw_max, params.k_rg_max) > 1:
        raise ValueError("Max relative permeability must be less than 1")

    denominator = 1 - params.S_or - params.S_wc - params.S_gc
    kro = (
        params.k_ro_max
        * ((saturations["So"] - params.S_or) / denominator) ** params.n_o
    )
    krw = (
        params.k_rw_max
        * ((saturations["Sw"] - params.S_wc) / denominator) ** params.n_w
    )
    krg = (
        params.k_rg_max
        * ((saturations["Sg"] - params.S_gc) / denominator) ** params.n_g
    )
    k_rel = np.array(
        list(zip(kro, krw, krg)), dtype=[(i, np.float64) for i in ("kro", "krw", "krg")]
    )
    for i in ("kro", "krw", "krg"):
        k_rel[i][k_rel[i] < 0] = 0  # negative permeability seems bad
    return k_rel
