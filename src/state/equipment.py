"""Equipment state model for the HVAC optimization system.

Represents the real-time operational status of a single HVAC
equipment unit (AHU, VAV box, chiller, heat pump, etc.).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class EquipmentStatus(str, Enum):
    """Operational status of an HVAC equipment unit.

    Attributes:
        ON: Unit is actively running within normal parameters.
        OFF: Unit is intentionally off / in standby.
        FAULT: Unit has reported a fault condition and requires
            inspection before setpoint commands can be issued.
    """

    ON = "ON"
    OFF = "OFF"
    FAULT = "FAULT"


class EquipmentState(BaseModel):
    """Real-time operational state of a single HVAC equipment unit.

    The Action Engine reads this model to determine whether a unit
    is eligible to receive a new setpoint command (e.g. a FAULT
    unit is excluded from all action proposals).

    Attributes:
        equipment_id: Unique identifier matching the BMS asset
            registry (e.g. ``"AHU_01"``, ``"VAV_B2_04"``).
        status: Current operational status of the unit.
        current_power_kw: Real-time active power draw in kW.
        fan_speed_pct: Fan speed as a percentage of maximum
            rated speed (0–100).
        damper_position_pct: Outdoor-air damper position as a
            percentage of fully open (0 = closed, 100 = fully
            open).
    """

    equipment_id: str = Field(
        ...,
        description="Unique BMS asset identifier.",
        min_length=1,
    )
    status: EquipmentStatus = Field(
        ...,
        description="Operational status of the equipment unit.",
    )
    current_power_kw: float = Field(
        ...,
        description="Real-time active power draw (kW).",
        ge=0.0,
    )
    fan_speed_pct: float = Field(
        ...,
        description="Fan speed percentage (0–100).",
        ge=0.0,
        le=100.0,
    )
    damper_position_pct: float = Field(
        ...,
        description="Outdoor-air damper position (0–100).",
        ge=0.0,
        le=100.0,
    )

    model_config = {"frozen": True}

    @property
    def is_commandable(self) -> bool:
        """Return True if the unit can receive setpoint commands.

        A unit is commandable only when its status is ON or OFF;
        FAULT units must be excluded from all action proposals
        by the Action Engine.

        Returns:
            True if the unit status is not FAULT.
        """
        return self.status != EquipmentStatus.FAULT
