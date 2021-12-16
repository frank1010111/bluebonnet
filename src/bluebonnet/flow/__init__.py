#from importlib.metadata import version

#__version__ = version("bluebonnet")
from .reservoir import (
    IdealReservoir,
    SinglePhaseReservoir,
    SinglePhaseReservoirMarder,
    MultiPhaseReservoir,
)

from .flowproperties import (
    FlowProperties,
    FlowPropertiesMarder,
    FlowPropertiesOnePhase,
    FlowPropertiesTwoPhase,
    FlowPropertiesMultiPhase,
    RelPermParams,
    relative_permeabilities,
)
