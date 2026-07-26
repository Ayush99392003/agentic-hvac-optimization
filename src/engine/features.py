"""Feature extraction for the HVAC optimization pipeline.

Converts raw numeric metrics from ZoneState and BuildingState
into qualitative, human-readable feature labels. These labels
are consumed directly by the Event Generator (Phase 3) and are
included in decision audit traces.

All logic is deterministic: same inputs always produce the same
feature labels. No ML inference occurs here.

Architecture reference: Section 2 (Decision Flow), Section 7
(Score thresholds drive feature boundaries).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from src.state.building import CARBON_HIGH_THRESHOLD, BuildingState
from src.state.zone import (
    CO2_DEGRADED_PPM,
    CO2_HAZARDOUS_PPM,
    PMV_ACCEPT_HIGH,
    PMV_ACCEPT_LOW,
    ZoneState,
)

# ---------------------------------------------------------------------------
# Feature label enumerations
# ---------------------------------------------------------------------------


class ComfortFeature(str, Enum):
    """Qualitative thermal comfort label derived from PMV.

    Boundaries align with the ASHRAE-55 comfort formula used in
    the Score Engine (architecture Section 7.1).

    Attributes:
        COMFORTABLE: PMV within the acceptable band (±0.5).
        SLIGHTLY_WARM: PMV in (0.5, 1.0] — mild heat stress.
        HOT: PMV > 1.0 — significant heat stress.
        SLIGHTLY_COOL: PMV in [-1.0, -0.5) — mild cold stress.
        COLD: PMV < -1.0 — significant cold stress.
    """

    COMFORTABLE = "COMFORTABLE"
    SLIGHTLY_WARM = "SLIGHTLY_WARM"
    HOT = "HOT"
    SLIGHTLY_COOL = "SLIGHTLY_COOL"
    COLD = "COLD"


class AirQualityFeature(str, Enum):
    """Qualitative indoor air quality label derived from CO₂ PPM.

    Boundaries align with the air quality scoring formula used in
    the Score Engine (architecture Section 7.2).

    Attributes:
        GOOD: CO₂ <= 800 PPM — fresh, high-quality air.
        MODERATE: CO₂ in (800, 1000] PPM — acceptable, monitor.
        DEGRADED: CO₂ in (1000, 1200] PPM — action recommended.
        HAZARDOUS: CO₂ > 1200 PPM — immediate ventilation needed.
    """

    GOOD = "GOOD"
    MODERATE = "MODERATE"
    DEGRADED = "DEGRADED"
    HAZARDOUS = "HAZARDOUS"


class EnergyFeature(str, Enum):
    """Qualitative energy consumption label.

    Derived from the ratio of current power draw to zone baseline,
    modulated by the current electricity tariff.

    Attributes:
        EFFICIENT: Power at or below baseline.
        ELEVATED: Power 0–35 % above baseline.
        HIGH: Power 35–70 % above baseline.
        CRITICAL: Power > 70 % above baseline.
    """

    EFFICIENT = "EFFICIENT"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class CarbonFeature(str, Enum):
    """Qualitative grid carbon intensity label.

    Derived from the normalized carbon score formula
    (architecture Section 7.4).

    Attributes:
        CLEAN: Grid is running on predominantly low-carbon
            sources — good time to consume energy.
        MODERATE: Intermediate carbon intensity.
        DIRTY: High-carbon grid window — defer non-essential
            HVAC loads if possible.
    """

    CLEAN = "CLEAN"
    MODERATE = "MODERATE"
    DIRTY = "DIRTY"


# ---------------------------------------------------------------------------
# Feature vector models
# ---------------------------------------------------------------------------


class ZoneFeatureVector(BaseModel):
    """Typed feature labels for a single zone at one timestep.

    This is the output of the Feature Extractor for a single zone.
    It is passed alongside the ZoneState to the Score Engine and
    Event Generator.

    Attributes:
        zone_id: The zone this vector describes.
        comfort: Qualitative thermal comfort label.
        air_quality: Qualitative air quality label.
        is_occupied: True if at least one occupant is present.
        is_high_density: True when occupancy > 10 (High-Density
            Occupancy Floor policy is active).
        occupancy_count: Raw occupant count for reference.
    """

    zone_id: str = Field(..., description="Zone identifier.")
    comfort: ComfortFeature = Field(
        ..., description="Thermal comfort label."
    )
    air_quality: AirQualityFeature = Field(
        ..., description="Indoor air quality label."
    )
    is_occupied: bool = Field(
        ..., description="True if zone has >= 1 occupant."
    )
    is_high_density: bool = Field(
        ...,
        description=(
            "True if occupancy > 10 (High-Density Floor active)."
        ),
    )
    occupancy_count: int = Field(
        ..., description="Number of occupants."
    )

    model_config = {"frozen": True}


class BuildingFeatureReport(BaseModel):
    """Feature labels for the entire building at one timestep.

    Aggregates per-zone feature vectors and adds building-level
    energy and carbon labels that span all zones / equipment.

    Attributes:
        zone_features: Mapping from zone_id to ZoneFeatureVector.
        energy: Building-level energy consumption label.
        carbon: Grid carbon intensity label.
        avg_tariff_ratio: Current price / historical average,
            used downstream for the energy score calculation.
    """

    zone_features: dict[str, ZoneFeatureVector] = Field(
        ..., description="Zone ID -> ZoneFeatureVector mapping."
    )
    energy: EnergyFeature = Field(
        ..., description="Building-level energy label."
    )
    carbon: CarbonFeature = Field(
        ..., description="Grid carbon intensity label."
    )
    avg_tariff_ratio: float = Field(
        ...,
        description=(
            "Price_current / Price_avg ratio used by Score Engine."
        ),
        ge=0.0,
    )

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Feature extraction functions
# ---------------------------------------------------------------------------


def _comfort_feature(pmv: float) -> ComfortFeature:
    """Map a PMV value to a qualitative ComfortFeature label.

    Args:
        pmv: Predicted Mean Vote value in [-3.0, +3.0].

    Returns:
        The corresponding ComfortFeature enum member.
    """
    if PMV_ACCEPT_LOW <= pmv <= PMV_ACCEPT_HIGH:
        return ComfortFeature.COMFORTABLE
    if pmv > PMV_ACCEPT_HIGH:
        return (
            ComfortFeature.SLIGHTLY_WARM
            if pmv <= 1.0
            else ComfortFeature.HOT
        )
    # pmv < PMV_ACCEPT_LOW
    return (
        ComfortFeature.SLIGHTLY_COOL
        if pmv >= -1.0
        else ComfortFeature.COLD
    )


def _air_quality_feature(co2_ppm: float) -> AirQualityFeature:
    """Map a CO₂ concentration to a qualitative air quality label.

    Args:
        co2_ppm: Indoor CO₂ concentration in PPM.

    Returns:
        The corresponding AirQualityFeature enum member.
    """
    if co2_ppm <= 800.0:
        return AirQualityFeature.GOOD
    if co2_ppm <= CO2_DEGRADED_PPM:
        return AirQualityFeature.MODERATE
    if co2_ppm <= CO2_HAZARDOUS_PPM:
        return AirQualityFeature.DEGRADED
    return AirQualityFeature.HAZARDOUS


def _energy_feature(power_ratio: float) -> EnergyFeature:
    """Map a power ratio (current / baseline) to an EnergyFeature.

    Args:
        power_ratio: P_current / P_baseline. Values <= 1.0 map
            to EFFICIENT.

    Returns:
        The corresponding EnergyFeature enum member.
    """
    excess = max(0.0, power_ratio - 1.0)
    if excess <= 0.0:
        return EnergyFeature.EFFICIENT
    if excess <= 0.35:
        return EnergyFeature.ELEVATED
    if excess <= 0.70:
        return EnergyFeature.HIGH
    return EnergyFeature.CRITICAL


def _carbon_feature(
    carbon_intensity: float,
    carbon_min: float,
    carbon_max: float,
) -> CarbonFeature:
    """Map grid carbon intensity to a qualitative carbon label.

    Uses the same normalization window as the Carbon Score formula
    (architecture Section 7.4).

    Args:
        carbon_intensity: Current gCO₂e/kWh.
        carbon_min: Day's minimum gCO₂e/kWh (cleanest window).
        carbon_max: Day's maximum gCO₂e/kWh (dirtiest window).

    Returns:
        The corresponding CarbonFeature enum member.
    """
    window = max(carbon_max - carbon_min, 1.0)  # avoid div/0
    normalised = (carbon_intensity - carbon_min) / window
    if normalised <= 0.33:
        return CarbonFeature.CLEAN
    if normalised <= 0.66:
        return CarbonFeature.MODERATE
    return CarbonFeature.DIRTY


def extract_zone_features(zone: ZoneState) -> ZoneFeatureVector:
    """Extract qualitative feature labels from a single ZoneState.

    Args:
        zone: The zone snapshot to extract features from.

    Returns:
        A frozen ZoneFeatureVector for the zone.
    """
    return ZoneFeatureVector(
        zone_id=zone.zone_id,
        comfort=_comfort_feature(zone.pmv),
        air_quality=_air_quality_feature(zone.co2_ppm),
        is_occupied=zone.is_occupied,
        is_high_density=zone.is_high_density,
        occupancy_count=zone.occupancy_count,
    )


def extract_building_features(
    state: BuildingState,
    *,
    avg_energy_price: float,
    total_power_kw: float,
    baseline_power_kw: float,
    carbon_min: float,
    carbon_max: float,
) -> BuildingFeatureReport:
    """Extract the full building feature report from a BuildingState.

    Args:
        state: The full building state snapshot.
        avg_energy_price: Historical average electricity price
            ($/kWh) used to compute the tariff ratio.
        total_power_kw: Aggregate kW draw across all equipment
            at this timestep.
        baseline_power_kw: Expected nominal kW draw for the
            building under normal operating conditions.
        carbon_min: Day-ahead forecast minimum grid carbon
            intensity (gCO₂e/kWh).
        carbon_max: Day-ahead forecast maximum grid carbon
            intensity (gCO₂e/kWh).

    Returns:
        A frozen BuildingFeatureReport.
    """
    zone_features = {
        zid: extract_zone_features(zone)
        for zid, zone in state.zones.items()
    }

    power_ratio = total_power_kw / max(baseline_power_kw, 0.001)
    tariff_ratio = state.energy_price / max(
        avg_energy_price, 0.001
    )

    return BuildingFeatureReport(
        zone_features=zone_features,
        energy=_energy_feature(power_ratio),
        carbon=_carbon_feature(
            state.carbon_intensity, carbon_min, carbon_max
        ),
        avg_tariff_ratio=tariff_ratio,
    )
