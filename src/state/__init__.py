"""State modeling package for the HVAC optimization system.

This package provides the foundational Pydantic data models that
represent the complete building state at any simulation timestep.
Every downstream component receives a BuildingState instance as
its primary input.
"""

from src.state.building import BuildingState
from src.state.equipment import EquipmentState, EquipmentStatus
from src.state.weather import WeatherState
from src.state.zone import (
    CO2_DEGRADED_PPM,
    CO2_HAZARDOUS_PPM,
    PMV_ACCEPT_HIGH,
    PMV_ACCEPT_LOW,
    TEMP_SAFETY_MAX,
    TEMP_SAFETY_MIN,
    ZoneState,
)

__all__ = [
    "BuildingState",
    "EquipmentState",
    "EquipmentStatus",
    "WeatherState",
    "ZoneState",
    "PMV_ACCEPT_LOW",
    "PMV_ACCEPT_HIGH",
    "TEMP_SAFETY_MIN",
    "TEMP_SAFETY_MAX",
    "CO2_DEGRADED_PPM",
    "CO2_HAZARDOUS_PPM",
]
