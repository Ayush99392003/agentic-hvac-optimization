"""Event Generator — deterministic transformation of scores
into high-level semantic events.

Each detection rule takes the BuildingState, BuildingScoreReport,
and BuildingFeatureReport as inputs and emits zero or more Event
objects. Rules are fully deterministic: same inputs always produce
the same events.

Architecture reference: Section 3 (Event Generator node),
Section 1.2 (Ontological Relationships — Metric → Event mapping).

Detection Threshold Reference
------------------------------
ZONE_OVERHEATING    : comfort_score < 75  AND zone is occupied
POOR_AIR_QUALITY    : air_quality_score < 75
ZONE_UNDERUTILIZED  : zone is NOT occupied AND power draw detected
PEAK_DEMAND_RISK    : building energy_score < 50
HIGH_CARBON_WINDOW  : carbon_score < 35
COOLING_INEFFICIENT : comfort_score < 55 AND energy_score < 55
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.engine.features import BuildingFeatureReport
from src.engine.scores import BuildingScoreReport
from src.events.models import (
    CRITICAL_SCORE,
    HIGH_SCORE,
    MEDIUM_SCORE,
    Event,
    EventList,
    EventSeverity,
    EventType,
    _severity_from_score,
)
from src.state.building import BuildingState

# ---------------------------------------------------------------------------
# Detection thresholds (all scores are 0-100, higher = better)
# ---------------------------------------------------------------------------

# Comfort score below which ZONE_OVERHEATING fires for an occupied zone.
OVERHEATING_COMFORT_THRESHOLD: float = 75.0

# Air quality score below which POOR_AIR_QUALITY fires.
AIR_QUALITY_THRESHOLD: float = 75.0

# Building energy score below which PEAK_DEMAND_RISK fires.
PEAK_DEMAND_ENERGY_THRESHOLD: float = 50.0

# Carbon score below which HIGH_CARBON_WINDOW fires.
HIGH_CARBON_SCORE_THRESHOLD: float = 35.0

# Both thresholds must be breached for COOLING_INEFFICIENT.
COOLING_INEFFICIENT_COMFORT_THRESHOLD: float = 55.0
COOLING_INEFFICIENT_ENERGY_THRESHOLD: float = 55.0


# ---------------------------------------------------------------------------
# Individual detection rules
# ---------------------------------------------------------------------------


def _detect_overheating(
    state: BuildingState,
    scores: BuildingScoreReport,
) -> list[Event]:
    """Detect ZONE_OVERHEATING for all occupied zones.

    Fires when comfort_score < OVERHEATING_COMFORT_THRESHOLD and
    the zone has at least one occupant.

    Args:
        state: Current building state.
        scores: Score report for this timestep.

    Returns:
        List of ZONE_OVERHEATING Event instances.
    """
    events: list[Event] = []
    for zid, zone_score in scores.zone_scores.items():
        zone = state.zones[zid]
        if (
            zone.is_occupied
            and zone_score.comfort < OVERHEATING_COMFORT_THRESHOLD
        ):
            sev = _severity_from_score(zone_score.comfort)
            direction = "warm" if zone.pmv > 0 else "cool"
            events.append(
                Event(
                    event_type=EventType.ZONE_OVERHEATING,
                    severity=sev,
                    zone_id=zid,
                    trigger_score=zone_score.comfort,
                    trigger_metric="comfort_score",
                    trigger_value=zone.pmv,
                    description=(
                        f"Zone '{zid}' is thermally uncomfortable "
                        f"(PMV={zone.pmv:+.2f}, too {direction}) "
                        f"with {zone.occupancy_count} occupants. "
                        f"Comfort score: {zone_score.comfort:.1f}."
                    ),
                    timestamp=state.timestamp,
                )
            )
    return events


def _detect_poor_air_quality(
    state: BuildingState,
    scores: BuildingScoreReport,
) -> list[Event]:
    """Detect POOR_AIR_QUALITY for all zones above CO₂ threshold.

    Fires when air_quality_score < AIR_QUALITY_THRESHOLD.
    Triggers regardless of occupancy — unoccupied zones may still
    require ventilation before next occupancy.

    Args:
        state: Current building state.
        scores: Score report for this timestep.

    Returns:
        List of POOR_AIR_QUALITY Event instances.
    """
    events: list[Event] = []
    for zid, zone_score in scores.zone_scores.items():
        if zone_score.air_quality < AIR_QUALITY_THRESHOLD:
            zone = state.zones[zid]
            sev = _severity_from_score(zone_score.air_quality)
            events.append(
                Event(
                    event_type=EventType.POOR_AIR_QUALITY,
                    severity=sev,
                    zone_id=zid,
                    trigger_score=zone_score.air_quality,
                    trigger_metric="air_quality_score",
                    trigger_value=zone.co2_ppm,
                    description=(
                        f"Zone '{zid}' CO2 is {zone.co2_ppm:.0f} PPM "
                        f"(air quality score: "
                        f"{zone_score.air_quality:.1f}). "
                        f"Ventilation increase recommended."
                    ),
                    timestamp=state.timestamp,
                )
            )
    return events


def _detect_underutilized_zones(
    state: BuildingState,
    features: BuildingFeatureReport,
) -> list[Event]:
    """Detect ZONE_UNDERUTILIZED for unoccupied but conditioned zones.

    Fires when a zone has zero occupants but the building's
    total power draw is elevated, suggesting that HVAC is
    conditioning an empty space unnecessarily.

    Note: The power signal is building-level in Phase 2. Per-zone
    power metering is a Phase 7 / production enhancement.

    Args:
        state: Current building state.
        features: Feature report for this timestep.

    Returns:
        List of ZONE_UNDERUTILIZED Event instances.
    """
    from src.engine.features import EnergyFeature

    events: list[Event] = []
    # Only raise underutilization events when building is drawing
    # elevated or higher power (ELEVATED, HIGH, CRITICAL).
    non_efficient = features.energy in (
        EnergyFeature.ELEVATED,
        EnergyFeature.HIGH,
        EnergyFeature.CRITICAL,
    )
    if not non_efficient:
        return events

    for zid, zone_fv in features.zone_features.items():
        if not zone_fv.is_occupied:
            events.append(
                Event(
                    event_type=EventType.ZONE_UNDERUTILIZED,
                    severity=EventSeverity.MEDIUM,
                    zone_id=zid,
                    trigger_score=50.0,  # proxy — zone-level power TBD
                    trigger_metric="occupancy_count",
                    trigger_value=float(
                        state.zones[zid].occupancy_count
                    ),
                    description=(
                        f"Zone '{zid}' is unoccupied but building "
                        f"energy draw is {features.energy.value}. "
                        f"HVAC setpoints may be unnecessarily active."
                    ),
                    timestamp=state.timestamp,
                )
            )
    return events


def _detect_peak_demand_risk(
    state: BuildingState,
    scores: BuildingScoreReport,
) -> list[Event]:
    """Detect PEAK_DEMAND_RISK at building level.

    Fires when building energy_score < PEAK_DEMAND_ENERGY_THRESHOLD.

    Args:
        state: Current building state.
        scores: Score report for this timestep.

    Returns:
        List containing zero or one PEAK_DEMAND_RISK Event.
    """
    if scores.energy >= PEAK_DEMAND_ENERGY_THRESHOLD:
        return []

    sev = _severity_from_score(scores.energy)
    return [
        Event(
            event_type=EventType.PEAK_DEMAND_RISK,
            severity=sev,
            zone_id=None,
            trigger_score=scores.energy,
            trigger_metric="energy_score",
            trigger_value=state.energy_price,
            description=(
                f"Building energy score is {scores.energy:.1f} "
                f"(tariff: ${state.energy_price:.3f}/kWh). "
                f"Peak demand shedding recommended."
            ),
            timestamp=state.timestamp,
        )
    ]


def _detect_high_carbon_window(
    state: BuildingState,
    scores: BuildingScoreReport,
) -> list[Event]:
    """Detect HIGH_CARBON_WINDOW at building level.

    Fires when carbon_score < HIGH_CARBON_SCORE_THRESHOLD.

    Args:
        state: Current building state.
        scores: Score report for this timestep.

    Returns:
        List containing zero or one HIGH_CARBON_WINDOW Event.
    """
    if scores.carbon >= HIGH_CARBON_SCORE_THRESHOLD:
        return []

    sev = _severity_from_score(scores.carbon)
    return [
        Event(
            event_type=EventType.HIGH_CARBON_WINDOW,
            severity=sev,
            zone_id=None,
            trigger_score=scores.carbon,
            trigger_metric="carbon_score",
            trigger_value=state.carbon_intensity,
            description=(
                f"Grid carbon intensity is "
                f"{state.carbon_intensity:.0f} gCO2e/kWh "
                f"(carbon score: {scores.carbon:.1f}). "
                f"Shift non-essential loads to cleaner window."
            ),
            timestamp=state.timestamp,
        )
    ]


def _detect_cooling_inefficiency(
    state: BuildingState,
    scores: BuildingScoreReport,
) -> list[Event]:
    """Detect COOLING_INEFFICIENT for zones with dual comfort and
    energy penalty.

    Fires when both comfort_score AND energy_score are below their
    respective thresholds, indicating that HVAC is consuming excess
    energy without achieving thermal comfort.

    Args:
        state: Current building state.
        scores: Score report for this timestep.

    Returns:
        List of COOLING_INEFFICIENT Event instances.
    """
    events: list[Event] = []
    if scores.energy >= COOLING_INEFFICIENT_ENERGY_THRESHOLD:
        return events

    for zid, zone_score in scores.zone_scores.items():
        zone = state.zones[zid]
        if (
            zone.is_occupied
            and zone_score.comfort
            < COOLING_INEFFICIENT_COMFORT_THRESHOLD
        ):
            events.append(
                Event(
                    event_type=EventType.COOLING_INEFFICIENT,
                    severity=EventSeverity.HIGH,
                    zone_id=zid,
                    trigger_score=zone_score.comfort,
                    trigger_metric="comfort_score+energy_score",
                    trigger_value=zone.pmv,
                    description=(
                        f"Zone '{zid}' is uncomfortable "
                        f"(comfort={zone_score.comfort:.1f}) while "
                        f"building energy draw is high "
                        f"(energy={scores.energy:.1f}). "
                        f"Cooling is inefficient — review HVAC "
                        f"operation."
                    ),
                    timestamp=state.timestamp,
                )
            )
    return events


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_events(
    state: BuildingState,
    scores: BuildingScoreReport,
    features: BuildingFeatureReport,
) -> EventList:
    """Run all detection rules and return the full EventList.

    Detection rules are applied in a fixed order. Each rule is
    independent — a single zone can appear in multiple events
    simultaneously (e.g. ZONE_OVERHEATING + POOR_AIR_QUALITY).

    Args:
        state: Full building state snapshot.
        scores: Score report produced by the Score Engine.
        features: Feature report produced by the Feature Extractor.

    Returns:
        A frozen EventList containing all detected events for
        this simulation timestep.
    """
    detected: list[Event] = []
    detected += _detect_overheating(state, scores)
    detected += _detect_poor_air_quality(state, scores)
    detected += _detect_underutilized_zones(state, features)
    detected += _detect_peak_demand_risk(state, scores)
    detected += _detect_high_carbon_window(state, scores)
    detected += _detect_cooling_inefficiency(state, scores)

    return EventList(
        events=detected,
        timestamp=datetime.now(tz=timezone.utc),
    )
