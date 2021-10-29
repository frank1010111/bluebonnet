from dataclasses import dataclass
import numpy as np
from scipy import interpolate, sparse, integrate
import numpy.typing as npt
from numpy import ndarray
from typing import Any, Union, Iterable, Mapping, Optional, Callable
from collections import namedtuple
from bluebonnet.fluids.fluid import Fluid
from .flowproperties import FlowProperties


@dataclass
class IdealReservoir:
    """
    Class for building scaling solutions of production from hydrofractured wells

    Parameters
    ----------
    nx : int
        number of spatial nodes
    pressure_fracface : float
        drawdown pressure at x=0 (psi)
    pressure_initial : float
        reservoir pressure before production (psi)
    fluid : FlowProperties
        reservoir fluid PVT/flow properties

    Methods
    ----------
    simulate : calculate pressure over time
    recovery_factor : calculate recovery factor over time
    """

    nx: int
    pressure_fracface: float
    pressure_initial: float
    fluid: FlowProperties

    def __post_init___(self):
        "Last initialization steps"

    def simulate(self, time: ndarray):
        """
        Calculate simulation pressure over time

        Parameters
        ----------
        time : ndarray
            times to solve for pressure
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

    def recovery_factor(self, time: Optional[ndarray] = None) -> ndarray:
        """Calculate recovery factor over time

        If time has is not specified, requires that `simulate` has been run

        Parameters
        ----------
        time : ndarray, Optional
            times to calculate recovery factor at

        Returns
        -------
        recovery : ndarray
            recovery factor over time
        """
        if time is None:
            try:
                time = self.time
            except AttributeError:
                raise RuntimeError(
                    "Need to run simulate before calculating recovery factor",
                )
        h_inv = self.nx - 1.0
        pp = self.pseudopressure[:, :3]
        dp_dx = (-pp[:, 2] + 4 * pp[:, 1] - 3 * pp[:, 0]) * h_inv * 0.5
        cumulative = integrate.cumulative_trapezoid(dp_dx, self.time, initial=0)
        self.recovery = cumulative * self.fvf_scale()
        return self.recovery

    def recovery_factor_interpolator(self) -> Callable:
        """Generates a function to get recovery factor from time

        Requires that `recovery_factor` has been run

        Returns
        -------
        scipy interpolator object that takes in time and spits out recovery factor
        """
        try:
            time = self.time
            recovery = self.recovery
        except AttributeError:
            raise RuntimeError(
                "Need to run recovery_factor",
            )
        interpolator = interpolate.interp1d(
            time, recovery, bounds_error=False, fill_value=(0, recovery[-1])
        )
        return interpolator

    def alpha_scaled(self, pseudopressure: ndarray) -> ndarray:
        "Calculate scaled diffusivity"
        return np.ones_like(pseudopressure)

    def fvf_scale(self):
        return 1 - self.pressure_fracface / self.pressure_initial


class SinglePhaseReservoir(IdealReservoir):
    def alpha_scaled(self, pseudopressure: ndarray) -> ndarray:
        "Calculate scaled diffusivity"
        alpha = self.fluid.alpha
        return alpha(pseudopressure) / alpha(1)

    def fvf_scale(self):
        return self.fluid.fvf_scale


@dataclass
class TwoPhaseReservoir(SinglePhaseReservoir):
    """Oil-gas reservoir simulation

    References
    ----------
    Ruiz Maraggi, L.M., Lake, L.W. and Walsh, M.P., 2020. "A Two-Phase Non-Linear One-
        Dimensional Flow Model for Reserves Estimation in Tight Oil and Gas
        Condensate Reservoirs Using Scaling Principles." In SPE Latin American and
        Caribbean Petroleum Engineering Conference. OnePetro.
        https://doi.org/10.2118/199032-MS
    """

    Sw_init: float

    def simulate(self, time: ndarray):
        """Calculate simulation pressure over time

        Parameters
        ----------
        time : ndarray
            times to solve for pressure
        """
        super().simulate()


@dataclass
class MultiPhaseReservoir(SinglePhaseReservoir):
    So_init: float
    Sg_init: float
    Sw_init: float

    def simulate(self, time: ndarray):
        """Calculate simulation pressure over time

        Parameters
        ----------
        time : ndarray
            times to solve for pressure
        """
        raise NotImplementedError  # TODO: saturation changes

        x = np.linspace(0, 1, self.nx)
        dx_squared = (x[1] - x[0]) ** 2
        pseudopressure = np.empty((len(time), self.nx))
        pseudopressure[0, :] = 1
        pseudopressure[0, 0] = 0
        sat_names = ("So", "Sg", "Sw")
        saturations = np.empty(
            (len(time), self.nx),
            dtype=[(s, np.float64) for s in sat_names],
        )
        for i in range(time.shape[0] - 1):
            b = pseudopressure[i]
            sat = saturations[i]
            dt = time[i + 1] - time[i]
            mesh_ratio = dt / dx_squared
            alpha_scaled = self.alpha_scaled(b, sat)
            kt_h2 = mesh_ratio * alpha_scaled
            a_matrix = _build_matrix(kt_h2)
            pseudopressure[i + 1], info = sparse.linalg.bicgstab(a_matrix, b)
            saturations[i + 1] = self._step_saturation(sat, b, pseudopressure[i + 1])
        self.time = time
        self.pseudopressure = pseudopressure

    def alpha_scaled(
        self,
        pseudopressure: ndarray,
        saturation: ndarray,
    ) -> ndarray:
        """
        Calculate scaled diffusivity given pseudopressure and saturations

        Parameters
        ----------
        pseudopressure : ndarray
            scaled pseudopressure
        saturaion : ndarray
            record array with So, Sg, Sw records

        Returns
        -------
        alpha : ndarray
            scaled diffusivity
        """
        alpha = self.fluid.alpha
        s = saturation
        return alpha(pseudopressure, s["So"], s["Sg"], s["Sw"]) / alpha(1)

    def _step_saturation(
        self, saturation: ndarray, ppressure_old: ndarray, ppressure_new: ndarray
    ) -> ndarray:
        "Step to new saturation"
        return NotImplementedError


def _build_matrix(kt_h2: ndarray) -> sparse.spmatrix:
    """
    Set up A matrix for timestepping

    Follows :math: `A x = b` -> :math: `x = A \ b`

    Parameters
    ----------
    kt_h2: ndarray
        diffusivity * dt / dx^2

    Returns
    -------
    a_matrix: sp.sparse.matrix
        The A matrix
    """
    diagonal_long = 1.0 + 2 * kt_h2
    diagonal_long[0] = -1.0
    diagonal_low = np.concatenate([[0], -kt_h2[2:-1], [-2 * kt_h2[-1]]])
    diagonal_upper = np.concatenate([[0, -kt_h2[1]], -kt_h2[2:-1]])
    a_matrix = sparse.diags(
        [diagonal_low, diagonal_long, diagonal_upper], [-1, 0, 1], format="csr"
    )
    return a_matrix
