"""Fit and forecast production.

For fitting one-phase or pseudo-one-phase production, use ForecasterOnePhase.

To fit production when fracface pressure is known, use fit_production_pressure.
"""

# read version from installed package
from __future__ import annotations

from importlib.metadata import version

__version__ = version("bluebonnet")

from bluebonnet.forecast.forecast import Bounds, ForecasterOnePhase
from bluebonnet.forecast.forecast_pressure import (
    fit_production_pressure,
    plot_production_comparison,
)

__all__ = [
    "Bounds",
    "ForecasterOnePhase",
    "fit_production_pressure",
    "plot_production_comparison",
]
