"""reservoir: scaling solutions for hydrofractured wells.

This module calculates the scaling curves, using flow properties and finite
difference methods.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numpy import ndarray
from scipy import integrate, interpolate, sparse

from bluebonnet.flow.flowproperties import FlowProperties

_ATOL = 1e-12


@dataclass
class IdealReservoir:
    """
    Class for building scaling solutions of production from hydrofractured wells.

    This maintains simulation parameters and fluid properties.

    Parameters
    ----------
    nx : int
        number of spatial nodes
    pressure_fracface : float | NDArray
        drawdown pressure at x=0 (psi)
    pressure_initial : float
        reservoir pressure before production (psi)
    fluid : FlowProperties
        reservoir fluid PVT/flow properties
    """

    nx: int
    """number of spatial nodes"""
    pressure_fracface: float | npt.NDArray
    """drawdown pressure at :math:`x=0` (psi)"""
    pressure_initial: float
    """reservoir pressure before production (psi)"""
    fluid: FlowProperties
    """reservoir fluid PVT/flow properties"""

    def __post_init__(self):
        """Last initialization steps."""

    def simulate(self, time: ndarray):
        """
        Calculate simulation pressure over time.

        Parameters
        ----------
        time : ndarray
            times to solve for pressure
        """
        self.time = time
        x = np.linspace(0, 1, self.nx)
        dx_squared = (x[1] - x[0]) ** 2
        pseudopressure = np.empty((len(time), self.nx))
        pseudopressure[0, :] = 1.0
        for i in range(time.shape[0] - 1):
            b = pseudopressure[i]
            mesh_ratio = (time[i + 1] - time[i]) / dx_squared
            alpha_scaled = self.alpha_scaled(b)
            kt_h2 = mesh_ratio * alpha_scaled
            a_matrix = _build_matrix(kt_h2)
            pseudopressure[i + 1], _ = sparse.linalg.bicgstab(a_matrix, b, atol=_ATOL)
        self.pseudopressure = pseudopressure

    def recovery_factor(self, time: ndarray | None = None, density=False) -> ndarray:
        """Calculate recovery factor over time.

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
            except AttributeError as e:
                msg = "Need to run simulate before calculating recovery factor"
                raise RuntimeError(msg) from e
        h_inv = self.nx - 1.0
        if density:
            pseudopressure_to_mass = interpolate.interp1d(
                self.fluid.pvt_props["m-scaled"],
                self.fluid.pvt_props["density"],
                fill_value="extrapolate",
            )
            mass = pseudopressure_to_mass(self.pseudopressure)
            mass_over_time = np.sum(mass, 1)
            cumulative = 1.0 - mass_over_time / mass_over_time[0]
        else:
            pp = self.pseudopressure[:, :3]
            rate = (-pp[:, 2] + 4 * pp[:, 1] - 3 * pp[:, 0]) * h_inv * 0.5  # dp_dx
            cumulative = integrate.cumulative_trapezoid(rate, self.time, initial=0)
        self.recovery = cumulative * self.fvf_scale()
        return self.recovery

    def recovery_factor_interpolator(self) -> Callable:
        """Generate a function to get recovery factor from time.

        Requires that `recovery_factor` has been run

        Returns
        -------
        scipy interpolator object that takes in time and spits out recovery factor
        """
        try:
            time = self.time
        except AttributeError as e:
            msg = "Need to run simulate"
            raise RuntimeError(msg) from e
        try:
            recovery = self.recovery
        except AttributeError:
            recovery = self.recovery_factor(time)

        interpolator = interpolate.interp1d(
            time, recovery, bounds_error=False, fill_value=(0, recovery[-1])
        )
        return interpolator

    def alpha_scaled(self, pseudopressure: ndarray) -> ndarray:
        """Calculate scaled diffusivity."""
        return np.ones_like(pseudopressure)

    def fvf_scale(self) -> float:
        """Scaling for formation volume factor.

        Returns
        -------
        FVF : float
        """
        return 1 - self.pressure_fracface / self.pressure_initial


class SinglePhaseReservoir(IdealReservoir):
    """Single-phase real fluid reservoir."""

    def alpha_scaled(self, pseudopressure: ndarray) -> ndarray:
        """Calculate scaled diffusivity."""
        alpha = self.fluid.alpha
        return alpha(pseudopressure) / alpha(self.fluid.m_i)

    def fvf_scale(self) -> float:
        """Scaling for formation volume factor.

        Returns
        -------
        FVF : float
        """
        return 1

    def simulate(
        self, time: ndarray[float], pressure_fracface: ndarray[float] | None = None
    ):
        """Calculate simulation pressure over time.

        Args
        ----
        time : ndarray
            times to solve for pressure
        pressure_fracface : Iterable[float] | None, optional
            pressure at frac-face over time. Defaults to None, which is no change

        Raises
        ------
        ValueError: wrong length changing pressure at frac-face
        """
        self.time = time
        dx_squared = (1 / self.nx) ** 2
        pseudopressure = np.empty((len(time), self.nx))
        if pressure_fracface is None:
            pressure_fracface = np.full(len(time), self.pressure_fracface)
        else:
            if len(pressure_fracface) != len(time):
                msg = (
                    "Pressure time series does not match time variable:"
                    f" {len(pressure_fracface)} versus {len(time)}"
                )
                raise ValueError(msg)
            self.pressure_fracface = pressure_fracface
        m_i = self.fluid.m_i
        m_f = self.fluid.m_scaled_func(pressure_fracface)
        pseudopressure_initial = np.full(self.nx, m_i)
        pseudopressure_initial[0] = m_f[0]
        pseudopressure[0, :] = pseudopressure_initial

        for i in range(len(time) - 1):
            mesh_ratio = (time[i + 1] - time[i]) / dx_squared
            b = np.minimum(pseudopressure[i].copy(), m_i)
            # Enforce the boundary condition at x=0
            b[0] = m_f[i] + self.alpha_scaled(m_f[i]) * m_f[i] * mesh_ratio
            try:
                alpha_scaled = self.alpha_scaled(b)
            except ValueError as e:
                msg = f"scaling failed where m_initial={m_i}  m_fracface={m_f}"
                raise ValueError(msg) from e
            kt_h2 = mesh_ratio * alpha_scaled
            a_matrix = _build_matrix(kt_h2)
            pseudopressure[i + 1], _ = sparse.linalg.bicgstab(a_matrix, b, atol=_ATOL)
        self.pseudopressure = pseudopressure


@dataclass
class TwoPhaseReservoir(SinglePhaseReservoir):
    """Oil-gas reservoir simulation.

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
        """Calculate simulation pressure over time.

        Parameters
        ----------
        time : ndarray
            times to solve for pressure
        """
        super().simulate(time)


@dataclass
class MultiPhaseReservoir(SinglePhaseReservoir):
    """Reservoir with three phases: oil, gas, and water all flowing.

    Parameters
    ----------
    nx : int
        number of spatial nodes
    pressure_fracface : float | NDArray
        drawdown pressure at x=0 (psi)
    pressure_initial : float
        reservoir pressure before production (psi)
    fluid : FlowProperties
        reservoir fluid PVT/flow properties
    So_init : float
        oil saturation at the beginning
    Sw_init : float
        water saturation
    Sg_init : float
        gas saturation
    """

    So_init: float
    """Initial oil saturation."""
    Sw_init: float
    """Initial water saturation."""
    Sg_init: float
    """Initial gas saturation."""

    def simulate(self, time: ndarray):
        """Calculate simulation pressure over time.

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
            pseudopressure[i + 1], info = sparse.linalg.bicgstab(
                a_matrix, b, atol=_ATOL
            )
            saturations[i + 1] = self._step_saturation(sat, b, pseudopressure[i + 1])
        self.time = time
        self.pseudopressure = pseudopressure

    def alpha_scaled(
        self,
        pseudopressure: ndarray,
        saturation: ndarray,
    ) -> ndarray:
        """
        Calculate scaled diffusivity given pseudopressure and saturations.

        Parameters
        ----------
        pseudopressure : ndarray
            scaled pseudopressure
        saturation : ndarray
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
        self,
        saturation: ndarray,  # noqa: ARG002
        ppressure_old: ndarray,  # noqa: ARG002
        ppressure_new: ndarray,  # noqa: ARG002
    ) -> ndarray:
        """Calculate new saturation.

        Args
        ----
        saturation : ndarray
            old saturation
        ppressure_old : ndarray
            old pressure
        ppressure_new : ndarray
            new pressure

        Returns
        -------
        new saturation: ndarray
        """
        return NotImplementedError


def _build_matrix(kt_h2: ndarray) -> sparse.spmatrix:
    r"""
    Set up A matrix for timestepping.

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
    # diagonal_long[0] = -1.0
    # diagonal_low = np.concatenate([[0], -kt_h2[2:-1], [-2 * kt_h2[-1]]])
    # diagonal_upper = np.concatenate([[0, -kt_h2[1]], -kt_h2[2:-1]])
    diagonal_long[-1] = 1.0 + kt_h2[-1]
    diagonal_low = -kt_h2[1:]
    diagonal_upper = -kt_h2[0:-1]
    a_matrix = sparse.diags(
        [diagonal_low, diagonal_long, diagonal_upper], [-1, 0, 1], format="csr"
    )
    return a_matrix
