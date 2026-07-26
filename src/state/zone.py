"""Zone state model for the HVAC optimization system.

Represents the current environmental and occupancy conditions
inside a single physical zone at one simulation timestep.
ZoneState is the primary input to the Comfort Agent and the
Air Quality Agent.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

# ASHRAE-55 thermal comfort acceptability band.
PMV_ACCEPT_LOW: float = -0.5
PMV_ACCEPT_HIGH: float = 0.5

# Safety temperature hard limits (°C) as per architecture spec.
TEMP_SAFETY_MIN: float = 18.0
TEMP_SAFETY_MAX: float = 26.0

# CO₂ threshold above which air quality degrades (PPM).
CO2_DEGRADED_PPM: float = 1000.0
CO2_HAZARDOUS_PPM: float = 1200.0


class ZoneState(BaseModel):
    """Environmental and occupancy state of a single building zone.

    Every scalar value is a real-time measurement or EnergyPlus
    output at a single simulation timestep. No historical context
    is stored here; the Feedback Loop (Phase 7) is responsible
    for maintaining rolling statistics.

    Attributes:
        zone_id: Unique identifier for the zone (e.g.
            ``"conf_room_exec"``, ``"open_office_201"``).
        temperature: Dry-bulb air temperature inside the zone
            in degrees Celsius.
        relative_humidity: Relative humidity inside the zone
            as a percentage (0–100).
        pmv: Predicted Mean Vote index calculated by EnergyPlus
            or a pre-processor. Range: -3.0 (very cold) to
            +3.0 (very hot). ASHRAE-55 acceptable band is
            ±0.5.
        co2_ppm: Indoor CO₂ concentration in PPM.
        occupancy_count: Number of occupants present in the
            zone at this timestep, as reported by the
            occupancy sensor / schedule.
        cooling_setpoint: Active cooling setpoint temperature
            in degrees Celsius.
        heating_setpoint: Active heating setpoint temperature
            in degrees Celsius.
    """

    zone_id: str = Field(
        ...,
        description="Unique zone identifier.",
        min_length=1,
    )
    temperature: float = Field(
        ...,
        description="Zone dry-bulb air temperature (°C).",
        ge=-10.0,
        le=60.0,
    )
    relative_humidity: float = Field(
        ...,
        description="Zone relative humidity (%).",
        ge=0.0,
        le=100.0,
    )
    pmv: float = Field(
        ...,
        description="Predicted Mean Vote (-3.0 to +3.0).",
        ge=-3.0,
        le=3.0,
    )
    co2_ppm: float = Field(
        ...,
        description="Indoor CO₂ concentration (PPM).",
        ge=300.0,
        le=5000.0,
    )
    occupancy_count: int = Field(
        ...,
        description="Number of occupants in the zone.",
        ge=0,
    )
    cooling_setpoint: float = Field(
        ...,
        description="Active cooling setpoint (°C).",
        ge=TEMP_SAFETY_MIN,
        le=TEMP_SAFETY_MAX,
    )
    heating_setpoint: float = Field(
        ...,
        description="Active heating setpoint (°C).",
        ge=TEMP_SAFETY_MIN,
        le=TEMP_SAFETY_MAX,
    )

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def validate_setpoint_band(self) -> "ZoneState":
        """Ensure heating setpoint does not exceed cooling setpoint.

        A deadband of at least 1 °C between heating and cooling
        setpoints is required to avoid HVAC short-cycling.

        Returns:
            The validated ZoneState instance.

        Raises:
            ValueError: If ``heating_setpoint >= cooling_setpoint``.
        """
        if self.heating_setpoint >= self.cooling_setpoint:
            raise ValueError(
                f"Zone '{self.zone_id}': heating_setpoint "
                f"({self.heating_setpoint}°C) must be strictly "
                f"less than cooling_setpoint "
                f"({self.cooling_setpoint}°C)."
            )
        return self

    @property
    def is_occupied(self) -> bool:
        """Return True if at least one occupant is present.

        Returns:
            True when ``occupancy_count > 0``.
        """
        return self.occupancy_count > 0

    @property
    def is_high_density(self) -> bool:
        """Return True when occupancy triggers the High-Density
        Occupancy Floor policy (occupancy > 10).

        Per the Decision Policy spec, comfort and air-quality
        constraints become non-negotiable when this flag is True
        regardless of peak tariff conditions.

        Returns:
            True when ``occupancy_count > 10``.
        """
        return self.occupancy_count > 10

    @property
    def comfort_in_band(self) -> bool:
        """Return True if PMV falls within the ASHRAE-55
        acceptable thermal comfort band (±0.5).

        Returns:
            True when ``PMV_ACCEPT_LOW <= pmv <= PMV_ACCEPT_HIGH``.
        """
        return PMV_ACCEPT_LOW <= self.pmv <= PMV_ACCEPT_HIGH

    @property
    def air_quality_degraded(self) -> bool:
        """Return True if CO₂ exceeds the degraded threshold.

        Returns:
            True when ``co2_ppm >= CO2_DEGRADED_PPM`` (1000 PPM).
        """
        return self.co2_ppm >= CO2_DEGRADED_PPM
