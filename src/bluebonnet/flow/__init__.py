from importlib.metadata import version

__version__ = version("bluebonnet")
from .reservoir import (
    IdealReservoir,
    FlowProperties,
    SinglePhaseReservoir,
    MultiPhaseReservoir,
    RelPermParams,
)
