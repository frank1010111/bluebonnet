import numpy as np
from scipy.interpolate import interp1d, LinearNDInterpolator
from scipy import integrate
import numpy.typing as npt
from numpy import ndarray
from typing import Any, Union, Iterable, Mapping, Optional, Callable
from collections import namedtuple

# from bluebonnet.fluids.gas import pseudopressure_Hussainy


class FlowProperties:
    """
    Flow properties for the system.

    This is used to translate from scaled pseudopressure to diffusivity and to capture
    the effect of expansion

    Parameters
    ----------
    df : Mapping
        has accessors for pseudopressure, alpha
        pseudopressure: pseudopressure scaled from 0 for frac face, 1 for initial
            reservoir conditions
        alpha: hydraulic diffusivity (needn't be scaled)
    fvf_scale : float
        formation volume factor scale for recovery factor
    """

    def __init__(self, df: Mapping[str, ndarray], fvf_scale: float = 1):
        need_cols = {"pseudopressure", "alpha"}
        if need_cols.intersection(df) != need_cols:
            raise ValueError("Need input df to have 'pseudopressure' and 'alpha'")
        # if abs(min(df["pseudopressure"])) > 1e-6:
        #     raise ValueError("Minimum pseudopressure should be 0 (did you rescale?)")
        # if abs(max(df["pseudopressure"]) - 1.0) > 1e-6:
        #     raise ValueError("Maximum pseudopressure should be 1 (did you rescale?)")
        self.df = df
        x = df["pseudopressure"]
        self.alpha = interp1d(x, df["alpha"])
        self.fvf_scale = fvf_scale

    def __repr__(self):
        return self.df.__repr__()
    
class FlowPropertiesMarder(FlowProperties):
    """
    Flow properties for the system.

    This is used to translate from scaled pseudopressure to diffusivity and to capture
    the effect of expansion

    Parameters
    ----------
    df : Mapping
        has accessors for pseudopressure, alpha
        pseudopressure: pseudopressure in psi^2/centipoise: NOT SCALED
        alpha: hydraulic diffusivity: NOT SCALED
    """

    def __init__(self, df: Mapping[str, ndarray],Pi):
        need_cols = {"pseudopressure","Cg","P","Viscosity","Z-Factor"}
        if need_cols.intersection(df) != need_cols:
            raise ValueError("Need input df to have 'pseudopressure','Cg','P','Viscosity' and 'Z-Factor'")
        df=df.assign(m_scale=1/2*df.Cg*df.P*df.Viscosity*df['Z-Factor']/df.P**2)
        #df=df.assign(m_initial_scaled=df.pseudopressure*df.m_scale)
        df=df.assign(alpha= 1/(df.Cg * df.Viscosity))
        #m_initial_scaled_func=interp1d(df.P,df.m_initial_scaled)
        self.m_scale_func=interp1d(df.P,df.m_scale)
        self.ms=self.m_scale_func(Pi)
        
        df=df.assign(m_scaled=df["pseudopressure"]*self.ms)
        self.m_scaled_func=interp1d(df.P,df.m_scaled)
        self.mi=self.m_scaled_func(Pi)
        self.alpha_func = interp1d(df.m_scaled, df["alpha"])
        self.df = df
        


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
        df_pvt: Mapping,
        df_kr: Mapping,
        reference_densities: dict,
        phi: float,
        Sw: float,
        fvf_scale: float,
    ):
        """Create FlowProperties from tables

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
        if need_cols_pvt.intersection(df_pvt) != need_cols_pvt:
            raise ValueError(f"df_pvt needs all of {need_cols_pvt}")
        if need_cols_kr.intersection(df_kr) != need_cols_kr:
            raise ValueError(f"df_kr needs all of {need_cols_kr}")
        pvt = {
            prop: interp1d(df_pvt["pressure"], df_pvt[prop], fill_value="extrapolate")
            for prop in need_cols_pvt
        }
        pvt.update(reference_densities)
        # pvt.update({"rho_o0": rho_o0, "rho_g0": rho_g0, "rho_w0": rho_w0})
        kr = {
            fluid: interp1d(df_kr["So"], df_kr[fluid])
            for fluid in ("kro", "krg", "krw")
        }
        alpha_calc = alpha_multiphase(
            df_pvt["pressure"], df_pvt["So"], phi, Sw, pvt, kr
        )
        object = cls(
            {"pseudopressure": df_pvt["pseudopressure"], "alpha": alpha_calc}, fvf_scale
        )
        object.pvt = pvt
        object.kr = kr
        return object


def alpha_multiphase(
    pressure: ndarray, So: ndarray, phi: float, Sw: float, pvt: dict, kr: dict
):
    lambda_combined = lambda_combined_func(pressure, So, pvt, kr)
    compressibility_combined = compressibility_combined_func(pressure, So, phi, Sw, pvt)
    return lambda_combined / compressibility_combined


def lambda_combined_func(pressure: ndarray, So: ndarray, pvt: dict, kr: dict):
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
    pressure: ndarray, So: ndarray, phi: float, Sw: Union[float, ndarray], pvt: dict
):
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


class FlowPropertiesMultiPhase(FlowProperties):
    """
    Flow properties for a multiphase system

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

    def __init__(self: Mapping[str, ndarray], fvf_scale: float):
        need_cols = {"pseudopressure", "alpha", "So", "Sg", "Sw"}
        if need_cols.intersection(df.columns) != need_cols:
            raise ValueError(
                "Need input dataframe to have 'pseudopressure', 'compressibility',"
                " and 'alpha' columns"
            )
        self.df = df
        x = df["pseudopressure", "So", "Sg", "Sw"]
        self.alpha = interpolate.LinearNDInterpolator(x, df.alpha)


RelPermParams = namedtuple(
    "RelPermParams", "n_o n_g n_w S_or S_wc S_gc k_ro_max k_rw_max k_rg_max"
)


def relative_permeabilities(
    saturations: ndarray,
    params: RelPermParams,
) -> ndarray:
    """
    Brooks-Corey power-law relative permeability

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
