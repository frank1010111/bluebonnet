from dataclasses import dataclass
import numpy as np
from scipy import interpolate, sparse, integrate
import numpy.typing as npt
from collections import namedtuple
from bluebonnet.fluids.fluid import Fluid
from bluebonnet.fluids.gas import pseudopressure_Hussainy


class FlowProperties:
    """
    Flow properties for the system.

    This is used to translate from scaled pseudopressure to diffusivity and to capture
    the effect of expansion

    Parameters
    ----------
    df: DataFrame with columns for pseudopressure, alpha, So, Sg, Sw
        pseudopressure: pseudopressure scaled from 0 for frac face, 1 for initial
            reservoir conditions
        alpha: hydraulic diffusivity (needn't be scaled)
    fvf_scale: float formation volume factor at initial conditions divided by FVF at the
         frac face
    """

    def __init__(self, df, fvf_scale):
        need_cols = set(["pseudopressure", "alpha"])
        if need_cols.intersection(df.columns) != need_cols:
            raise ValueError(
                "Need input dataframe to have 'pseudopressure' and 'alpha' columns"
            )
        if abs(min(df.pseudopressure)) > 1e-6:
            raise ValueError("Minimum pseudopressure should be 0 (did you rescale?)")
        if abs(max(df.pseudopressure) - 1.0) > 1e-6:
            raise ValueError("Maximum pseudopressure should be 1 (did you rescale?)")
        self.df = df
        x = df.pseudopressure
        self.alpha = interpolate.interp1d(x, df.alpha)
        self.fvf_scale = fvf_scale

    def __repr__(self):
        return self.df.__repr__()


FlowPropertiesOnePhase = FlowProperties


class FlowPropertiesMultiPhase(FlowProperties):
    """
    Flow properties for a multiphase system

    This is used to translate from scaled pseudopressure and saturations to diffusivity
    and to capture the effect of expansion

    Parameters
    ----------
    df: DataFrame with columns for pseudopressure, alpha, So, Sg, Sw
        pseudopressure: pseudopressure scaled from 0 for frac face, 1 for initial
            reservoir conditions
        alpha: hydraulic diffusivity (needn't be scaled)
        So: oil saturation
        Sg: gas saturation
        Sw: water saturation
    fvf_scale: dict for the Bo,Bg,Bw at initial conditions divided by FVF at the frac
        face
    """

    def __init__(self, df, fvf_scale):
        need_cols = set(["pseudopressure", "alpha", "So", "Sg", "Sw"])
        if need_cols.intersection(df.columns) != need_cols:
            raise ValueError(
                "Need input dataframe to have 'pseudopressure', 'compressibility',"
                " and 'alpha' columns"
            )
        self.df = df
        x = df["pseudopressure", "So", "Sg", "Sw"]
        self.alpha = interpolate.LinearNDInterpolator(x, df.alpha)


@dataclass
class IdealReservoir:
    """
    Class for building scaling solutions of production from hydrofractured wells

    Parameters
    ----------
    nx: number of spatial nodes
    pressure_fracface: drawdown pressure at x=0 (psi)
    pressure_initial: reservoir pressure before production (psi)
    fluid: reservoir fluid PVT/flow properties

    Methods
    ----------
    simulate: calculate pressure over time
    recovery_factor:
    """

    nx: int
    pressure_fracface: float
    pressure_initial: float
    fluid: FlowProperties

    def __post_init___(self):
        "Last initialization steps"

    def simulate(self, time: npt.NDArray[np.float64]):
        """
        Calculate simulation pressure over time

        Parameters
        ----------
        time: array of times to solve for pressure
        """
        self.time = time
        x = np.linspace(0, 1, self.nx)
        dx_squared = (x[1] - x[0]) ** 2
        pseudopressure = np.empty((len(time), self.nx))
        pseudopressure[0, :] = 1
        pseudopressure[0, 0] = 0
        for i in range(time.shape[0] - 1):
            b = pseudopressure[i]
            mesh_ratio = (time[i + 1] - time[i]) / dx_squared
            alpha_scaled = self.alpha_scaled(b)
            kt_h2 = mesh_ratio * alpha_scaled
            a_matrix = _build_matrix(kt_h2)
            pseudopressure[i + 1], info = sparse.linalg.bicgstab(a_matrix, b)
        self.pseudopressure = pseudopressure

    def recovery_factor(self):
        assert hasattr(self, "time") and hasattr(self, "pseudopressure"), (
            "Need to run simulate before getting recovery factor",
        )
        h_inv = self.nx - 1.0
        pp = self.pseudopressure[:, :3]
        dp_dx = (-pp[:, 2] + 4 * pp[:, 1] - 3 * pp[:, 0]) * h_inv * 0.5
        cumulative = integrate.cumulative_trapezoid(dp_dx, self.time, initial=0)
        self.recovery = cumulative * self.fvf_scale()
        return self.recovery

    def alpha_scaled(
        self, pseudopressure: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        "Calculate scaled diffusivity"
        return np.ones_like(pseudopressure)

    def fvf_scale(self):
        return 1 - self.pressure_fracface / self.pressure_initial


class SinglePhaseReservoir(IdealReservoir):
    def alpha_scaled(
        self, pseudopressure: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        "Calculate scaled diffusivity"
        alpha = self.fluid.alpha
        return alpha(pseudopressure) / alpha(1)

    def fvf_scale(self):
        return self.fluid.fvf_scale


@dataclass
class MultiPhaseReservoir(SinglePhaseReservoir):
    So_init: float
    Sg_init: float
    Sw_init: float

    def simulate(self, time: npt.NDArray[np.float64]):
        """
        Calculate simulation pressure over time

        Parameters
        ----------
        time: array of times to solve for pressure
        """
        raise NotImplementedError  # TODO: saturation changes
        self.time = time
        x = np.linspace(0, 1, self.nx)
        dx_squared = (x[1] - x[0]) ** 2
        pseudopressure = np.empty((len(time), self.nx))
        pseudopressure[0, :] = 1
        pseudopressure[0, 0] = 0
        for i in range(time.shape[0] - 1):
            mesh_ratio = (time[i + 1] - time[i]) / dx_squared
            b = pseudopressure[i]
            a_matrix = self._build_matrix(b, mesh_ratio)
            pseudopressure[i + 1], info = sparse.linalg.bicgstab(a_matrix, b)
        self.pseudopressure = pseudopressure

    def alpha_scaled(
        self,
        pseudopressure: npt.NDArray[np.float64],
        So: npt.NDArray[np.float64],
        Sg: npt.NDArray[np.float64],
        Sw: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        "Calculate scaled diffusivity"
        alpha = self.fluid.alpha
        return alpha(pseudopressure, So, Sg, Sw) / alpha(1)


RelPermParams = namedtuple(
    "RelPermParams", "n_o n_g n_w S_or S_wc S_gc k_ro_max k_rw_max k_rg_max"
)


def relative_permeabilities(
    saturations: npt.NDArray,
    params: RelPermParams,
) -> npt.NDArray:
    """
    Brooks-Corey power-law relative permeability

    Parameters
    ----------
    saturations: numpy record array with columns for So, Sg, Sw
    params: RelPermParams with Corey exponents, residual saturations, and max relative permeabilities

    Returns
    -------
    k_rel: numpy record array with k_o, k_w, k_g (aka oil, water gas)
    """
    assert (
        np.abs(sum(v for v in saturations.values()) - 1) < 1e-3
    ), "Saturations must sum to 1"
    assert max(params.n_o, params.n_g, params.n_w) <= 6, "Exponents must be less than 6"
    assert min(params.n_o, params.n_g, params.n_w) >= 1, "Exponents must be at least 1"
    assert (
        min(params.S_or, params.S_wc, params.S_gc) >= 0
    ), "Critical saturations must be at least 0"
    assert (
        max(params.S_or, params.S_wc, params.S_gc) <= 1
    ), "Critical saturations must be less than 1"
    assert (
        min(params.k_ro_max, params.k_rw_max, params.k_rg_max) >= 0
    ), "Max relative permeability must be at least 0"
    assert (
        max(params.k_ro_max, params.k_rw_max, params.k_rg_max) <= 1
    ), "Max relative permeability must be less than 1"

    denominator = 1 - params.S_or - params.S_wc - params.S_gc
    k_o = (
        params.k_ro_max
        * ((saturations["So"] - params.S_or) / denominator) ** params.n_o
    )
    k_w = (
        params.k_rw_max
        * ((saturations["Sw"] - params.S_wc) / denominator) ** params.n_w
    )
    k_g = (
        params.k_rg_max
        * ((saturations["Sg"] - params.S_gr) / denominator) ** params.n_g
    )
    k_rel = np.array(
        [k_o, k_w, k_g], dtype=[(i, np.float64) for i in ("k_o", "k_w", "k_g")]
    )
    k_rel[k_rel < 0] = 0  # negative permeability seems bad
    return k_rel


def _build_matrix(kt_h2: npt.NDArray[np.float64]):
    """
    Set up A matrix for timestepping

    Parameters
    ----------
    kt_h2: ndarray, diffusivity * dt / dx^2
    """
    diagonal_long = 1.0 + 2 * kt_h2
    diagonal_long[0] = -1.0
    diagonal_low = np.concatenate([[0], -kt_h2[2:-1], [-2 * kt_h2[-1]]])
    diagonal_upper = np.concatenate([[0, -kt_h2[1]], -kt_h2[2:-1]])
    a_matrix = sparse.diags(
        [diagonal_low, diagonal_long, diagonal_upper], [-1, 0, 1], format="csr"
    )
    return a_matrix
