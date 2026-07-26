"""Building state model — top-level aggregate for the pipeline.

BuildingState is the single object that flows through every
pipeline stage (Feature Extractor → Score Engine → Event
Generator → Agents → LLM Orchestrator → Action Engine →
Feedback Loop). Each stage receives it read-only and produces
its own typed output; no stage mutates the state directly.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict

from pydantic import BaseModel, Field, model_validator

from src.state.equipment import EquipmentState
from src.state.weather import WeatherState
from src.state.zone import ZoneState

# Grid carbon intensity threshold above which the Carbon Agent
# activates load-shifting recommendations (gCO2e/kWh).
CARBON_HIGH_THRESHOLD: float = 400.0


class BuildingState(BaseModel):
    """Complete snapshot of building state at one simulation step.

    This is the canonical input contract for every pipeline stage.
    All fields are immutable once constructed; downstream stages
    must produce new objects (score vectors, event lists, etc.)
    rather than updating state in-place.

    Attributes:
        timestamp: Wall-clock time of this simulation step in UTC.
        weather: Ambient weather conditions for this timestep.
        zones: Mapping from zone_id to the corresponding
            ZoneState. At minimum one zone must be present.
        equipment: Mapping from equipment_id to the corresponding
            EquipmentState. May be empty in unit-test scenarios
            that do not exercise the Action Engine.
        energy_price: Current electricity tariff in $/kWh.
        carbon_intensity: Current grid carbon intensity in
            gCO₂e/kWh, used by the Carbon Agent and Score Engine.
    """

    timestamp: datetime = Field(
        ...,
        description="UTC timestamp of this simulation step.",
    )
    weather: WeatherState = Field(
        ...,
        description="Ambient weather conditions.",
    )
    zones: Dict[str, ZoneState] = Field(
        ...,
        description="Zone ID → ZoneState mapping.",
        min_length=1,
    )
    equipment: Dict[str, EquipmentState] = Field(
        default_factory=dict,
        description="Equipment ID → EquipmentState mapping.",
    )
    energy_price: float = Field(
        ...,
        description="Current electricity tariff ($/kWh).",
        ge=0.0,
    )
    carbon_intensity: float = Field(
        ...,
        description="Grid carbon intensity (gCO₂e/kWh).",
        ge=0.0,
    )

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def validate_zone_ids_consistent(self) -> "BuildingState":
        """Ensure every ZoneState's zone_id matches its dict key.

        Returns:
            The validated BuildingState instance.

        Raises:
            ValueError: If any dict key does not match the
                embedded ``zone_id`` field.
        """
        for key, zone in self.zones.items():
            if zone.zone_id != key:
                raise ValueError(
                    f"Zone key '{key}' does not match "
                    f"ZoneState.zone_id '{zone.zone_id}'."
                )
        return self

    @model_validator(mode="after")
    def validate_equipment_ids_consistent(
        self,
    ) -> "BuildingState":
        """Ensure every EquipmentState's equipment_id matches its
        dict key.

        Returns:
            The validated BuildingState instance.

        Raises:
            ValueError: If any dict key does not match the
                embedded ``equipment_id`` field.
        """
        for key, equip in self.equipment.items():
            if equip.equipment_id != key:
                raise ValueError(
                    f"Equipment key '{key}' does not match "
                    f"EquipmentState.equipment_id "
                    f"'{equip.equipment_id}'."
                )
        return self

    # ----------------------------------------------------------
    # Convenience properties used across pipeline stages
    # ----------------------------------------------------------

    @property
    def occupied_zones(self) -> Dict[str, ZoneState]:
        """Return only zones with at least one occupant.

        Returns:
            Filtered dict of occupied ZoneState instances.
        """
        return {
            zid: z for zid, z in self.zones.items() if z.is_occupied
        }

    @property
    def high_density_zones(self) -> Dict[str, ZoneState]:
        """Return zones whose occupancy triggers the High-Density
        Occupancy Floor policy (occupancy > 10).

        Returns:
            Filtered dict of high-density ZoneState instances.
        """
        return {
            zid: z
            for zid, z in self.zones.items()
            if z.is_high_density
        }

    @property
    def is_high_carbon_window(self) -> bool:
        """Return True if current grid carbon intensity exceeds
        the activation threshold for the Carbon Agent.

        Returns:
            True when ``carbon_intensity >= CARBON_HIGH_THRESHOLD``.
        """
        return self.carbon_intensity >= CARBON_HIGH_THRESHOLD

    @property
    def commandable_equipment(self) -> Dict[str, EquipmentState]:
        """Return only equipment units that can receive commands
        (i.e. those not in FAULT status).

        Returns:
            Filtered dict of commandable EquipmentState instances.
        """
        return {
            eid: e
            for eid, e in self.equipment.items()
            if e.is_commandable
        }
