from dataclasses import dataclass
import numpy as np
import scipy as sp
import numpy.typing as npt
from bluebonnet.fluids.fluid import Fluid
from bluebonnet.fluids.gas import pseudopressure_Hussainy


class FlowProperties:
    def __init__(self, df):
        need_cols = set(["pseudopressure", "alpha"])
        if need_cols.intersection(df.columns) != need_cols:
            raise ValueError(
                "Need input dataframe to have 'pseudopressure' and 'alpha' columns"
            )
        self.df = df
        x = df.pseudopressure
        self.alpha = sp.interpolate.interp1d(x, df.alpha)

    def __repr__(self):
        return df.__repr__()


FlowPropertiesOnePhase = FlowProperties


class FlowPropertiesMultiPhase(FlowProperties):
    def __init__(self, df):
        need_cols = set(["pseudopressure", "alpha", "So", "Sg", "Sw"])
        if need_cols.intersection(df.columns) != need_cols:
            raise ValueError(
                "Need input dataframe to have 'pseudopressure', 'compressibility',"
                " and 'alpha' columns"
            )
        self.df = df
        x = df["pseudopressure", "So", "Sg", "Sw"]
        self.alpha = sp.interpolate.LinearNDInterpolator(x, df.alpha)


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
            mesh_ratio = (time[i + 1] - time[i]) / dx_squared
            b = pseudopressure[i]
            a_matrix = self._build_matrix(b, mesh_ratio)
            pseudopressure[i + 1], info = sp.sparse.linalg.bicgstab(a_matrix, b)
        self.pseudopressure = pseudopressure

    def recovery_factor(self):
        assert hasattr(self, "time") and hasattr(self, "pseudopressure"), (
            "Need to run simulate before getting recovery factor",
        )
        h_inv = self.nx - 1.0
        pp = self.pseudopressure[:, :3]
        dp_dx = (-pp[:, 2] + 4 * pp[:, 1] - 3 * pp[:, 0]) * h_inv * 0.5
        cumulative = sp.integrate.cumulative_trapezoid(dp_dx, self.time, initial=0)
        self.recovery = cumulative * (
            1 - self.pressure_fracface / self.pressure_initial
        )
        return self.recovery

    def _build_matrix(self, pseudopressure: npt.NDArray[np.float64], mesh_ratio: float):
        """
        Set up A matrix for timestepping

        Parameters
        ----------
        pseudopressure: ndarray
        mesh_ratio: float, dt/dx^2
        """
        alpha_scaled = self.alpha_scaled(pseudopressure)
        changeability = mesh_ratio * alpha_scaled
        diagonal_long = 1.0 + 2 * changeability
        diagonal_long[0] = -1.0
        diagonal_low = np.concatenate(
            [[0], -changeability[2:-1], [-2 * changeability[-1]]]
        )
        diagonal_upper = np.concatenate([[0, -changeability[1]], -changeability[2:-1]])
        a_matrix = sp.sparse.diags(
            [diagonal_low, diagonal_long, diagonal_upper], [-1, 0, 1], format="csr"
        )
        return a_matrix

    def alpha_scaled(
        self, pseudopressure: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        "Calculate scaled diffusivity"
        return np.ones_like(pseudopressure)
