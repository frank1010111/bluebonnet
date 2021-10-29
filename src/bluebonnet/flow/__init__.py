from importlib.metadata import version

__version__ = version("bluebonnet")
from .reservoir import (
    IdealReservoir,
    SinglePhaseReservoir,
    MultiPhaseReservoir,
)

from .flowproperties import (
    FlowProperties,
    FlowPropertiesOnePhase,
    FlowPropertiesTwoPhase,
    FlowPropertiesMultiPhase,
    RelPermParams,
    relative_permeabilities,
)
