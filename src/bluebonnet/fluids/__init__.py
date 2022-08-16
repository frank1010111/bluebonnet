"""
Fluid pressure and temperature-dependent properties.

Uses common empirical correlations to get properties. These include density,
saturation pressure, viscosity, and the pseudopressure transform.
"""
from __future__ import annotations

from bluebonnet.fluids.fluid import Fluid, build_pvt_gas, pseudopressure

__all__ = [
    "build_pvt_gas",
    "Fluid",
    "pseudopressure",
]
