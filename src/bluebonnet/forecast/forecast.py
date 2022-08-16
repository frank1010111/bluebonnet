"""Fit and forecast production from hydrofractured reservoirs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt
from numpy import ndarray
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class Bounds:
    """Set the upper and lower limits for the fitting curve.

    Parameters
    ----------
    M: tuple of floats
        (min, max) for resource in place
    tau: tuple of floats
        (min, max) for time-to-BDF
    """

    M: tuple[float, float]
    tau: tuple[float, float]

    def __post_init___(self):
        """Validate bounds."""
        if len(self.M) != 2:
            raise ValueError("M must be two elements")
        if len(self.tau) != 2:
            raise ValueError("tau must be two elements")
        if self.M[0] >= self.M[1]:
            raise ValueError(f"{self.M[0]=} must be greater than {self.M[1]=}")
        if self.tau[0] >= self.tau[1]:
            raise ValueError(f"{self.tau[0]=} must be greater than {self.tau[1]=}")

    def fit_bounds(self):
        """Return bounds for forecaster instance fits."""
        return ((self.M[0], self.tau[0]), (self.M[1], self.tau[1]))


_default_bounds = Bounds(M=(0, np.inf), tau=(1e-10, np.inf))


@dataclass
class ForecasterOnePhase:
    """Forecaster for production decline models.

    Parameters
    ----------
    rf_curve: the recovery factor over scaled time, as a callable
              (interpolators are best)
    bounds: the minimum and maximum values that the rf_curve accepts
              (watch the limits for the rf_curve function)

    Methods
    ----------
    forecast_cum: forecast the cumulative production over time
    fit: fit M and tau for the scaling curve
    """

    rf_curve: Callable
    bounds: Bounds = _default_bounds

    def forecast_cum(
        self,
        time_on_production: npt.NDArray[np.float64],
        M: float | None = None,
        tau: float | None = None,
    ):
        """Forecast cumulative production.

        Parameters
        ----------
        time_on_production: time since the well started producing
                            (days or years, ideally, since months are uneven), ndarray
        M: the resource in place (Mstb, Mscf, or other standard units), float
        tau: the time until BDF/depletion flow
             (days or years, same units as time on prod), float
        """
        if M is None:
            M = self.M_
        if tau is None:
            tau = self.tau_
        return _forecast_cum_onephase(self.rf_curve, time_on_production, M, tau)

    def fit(
        self,
        time_on_production: ndarray,
        cum_production: ndarray,
        tau: float | None = None,
    ):
        """Fit well production to the physics-based scaling model.

        Parameters
        ----------
        time_on_production: time since the well started producing
                            (days or years, ideally, since months are uneven), ndarray
        cum_production: the cumulative production over time
                        (Mstb, Mscf, or other standard units), ndarray
        tau: [Optional] the time-to-depletion (BDF), same units as time on prod, float
        """
        if tau is None:
            p0 = (cum_production[-1] * 2, time_on_production[-1] * 5)
            bounds = self.bounds.fit_bounds()
            forecast = lambda top, M, tau: _forecast_cum_onephase(  # noqa: E731
                self.rf_curve, top, M, tau
            )
        else:
            p0 = (cum_production[-1] * 2,)
            bounds = self.bounds.M
            forecast = lambda top, M: _forecast_cum_onephase(  # noqa: BLK,E731
                self.rf_curve, top, M, tau
            )

        fit, covariance = curve_fit(
            forecast,
            time_on_production,
            cum_production,
            p0,
            bounds=bounds,
        )
        self.time_on_production = time_on_production
        self.cum_production = cum_production
        if tau is None:
            self.M_, self.tau_ = fit
        else:
            self.M_ = fit[0]
            self.tau_ = tau


def _forecast_cum_onephase(
    rf_curve: Callable, time_on_production: ndarray, M: float, tau: float
) -> ndarray:
    time_scaled = time_on_production / tau
    rf = M * rf_curve(time_scaled)
    return rf
