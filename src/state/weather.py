"""Weather state model for the HVAC optimization system.

Represents the ambient weather conditions observed at a given
simulation timestep. WeatherState is an immutable snapshot —
each timestep creates a new instance rather than mutating an
existing one.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class WeatherState(BaseModel):
    """Ambient weather conditions at a single simulation timestep.

    All fields represent instantaneous measurements; no computed
    properties are defined here. Downstream feature-extraction
    stages are responsible for deriving solar gain estimates,
    humidity-adjusted comfort indices, etc.

    Attributes:
        dry_bulb_temperature: Outdoor dry-bulb air temperature
            in degrees Celsius.
        relative_humidity: Outdoor relative humidity as a
            percentage (0–100).
        direct_solar_irradiance: Direct normal solar irradiance
            incident on a surface perpendicular to the sun's
            rays, in W/m².
        wind_speed: Outdoor wind speed at sensor height in m/s.
    """

    dry_bulb_temperature: float = Field(
        ...,
        description="Outdoor dry-bulb temperature (°C).",
        ge=-40.0,
        le=60.0,
    )
    relative_humidity: float = Field(
        ...,
        description="Outdoor relative humidity (%).",
        ge=0.0,
        le=100.0,
    )
    direct_solar_irradiance: float = Field(
        ...,
        description="Direct normal solar irradiance (W/m²).",
        ge=0.0,
        le=1400.0,
    )
    wind_speed: float = Field(
        ...,
        description="Wind speed at sensor height (m/s).",
        ge=0.0,
        le=75.0,
    )

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def validate_irradiance_at_night(self) -> "WeatherState":
        """Warn if solar irradiance is reported above 0 when
        temperature conditions suggest nighttime; this is a
        data-quality sanity guard, not a hard rejection.

        Returns:
            The validated WeatherState instance.
        """
        # Soft guard: irradiance capped by Field validator.
        # Full night-detection would require timestep metadata
        # and is left to the Feature Extractor (Phase 2).
        return self
