"""Flow, pressure changes, and production from hydraulically fractured reservoirs."""
from __future__ import annotations

from bluebonnet.flow.flowproperties import (
    FlowProperties,
    FlowPropertiesMultiPhase,
    FlowPropertiesOnePhase,
    FlowPropertiesTwoPhase,
    RelPermParams,
    relative_permeabilities,
    relative_permeabilities_twophase,
    rescale_pseudopressure,
)
from bluebonnet.flow.reservoir import (
    IdealReservoir,
    MultiPhaseReservoir,
    SinglePhaseReservoir,
    TwoPhaseReservoir,
)

__all__ = [
    "FlowProperties",
    "FlowPropertiesMultiPhase",
    "FlowPropertiesOnePhase",
    "FlowPropertiesTwoPhase",
    "RelPermParams",
    "relative_permeabilities",
    "IdealReservoir",
    "MultiPhaseReservoir",
    "SinglePhaseReservoir",
    "TwoPhaseReservoir",
    "relative_permeabilities_twophase",
    "rescale_pseudopressure",
]
