"""Score Engine — deterministic 0-100 scoring of building state.

Implements the four score formulas from architecture Section 7:

    S_comfort  — ASHRAE-55 piecewise PMV formula (§7.1)
    S_air      — CO₂-based air quality score     (§7.2)
    S_energy   — Baseline-deviation energy score  (§7.3)
    S_carbon   — Grid carbon window normalization  (§7.4)

All formulas are deterministic. The same inputs always produce
the same scores. No thresholds are tuned at runtime.

Score convention: 100 = optimal, 0 = worst possible.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.state.building import BuildingState
from src.state.zone import PMV_ACCEPT_HIGH, PMV_ACCEPT_LOW


# ---------------------------------------------------------------------------
# Score vector models
# ---------------------------------------------------------------------------


class ZoneScoreVector(BaseModel):
    """Numeric 0-100 scores for a single zone at one timestep.

    Every score is independent; they are not combined into a
    single weighted sum (architecture §6.2 decision: keep scores
    separate so the LLM Orchestrator can reason about trade-offs
    between objectives explicitly).

    Attributes:
        zone_id: Zone these scores describe.
        comfort: Thermal comfort score (ASHRAE-55 PMV formula).
        air_quality: Indoor air quality score (CO₂ formula).
    """

    zone_id: str = Field(..., description="Zone identifier.")
    comfort: float = Field(
        ...,
        description="Comfort score [0, 100].",
        ge=0.0,
        le=100.0,
    )
    air_quality: float = Field(
        ...,
        description="Air quality score [0, 100].",
        ge=0.0,
        le=100.0,
    )

    model_config = {"frozen": True}


class BuildingScoreReport(BaseModel):
    """Score report for the full building at one simulation step.

    Contains per-zone scores plus building-level energy and
    carbon scores that are computed across all equipment/grid data.

    Attributes:
        zone_scores: Mapping from zone_id to ZoneScoreVector.
        energy: Building-level energy efficiency score [0, 100].
        carbon: Grid carbon cleanliness score [0, 100].
    """

    zone_scores: dict[str, ZoneScoreVector] = Field(
        ..., description="Zone ID -> ZoneScoreVector mapping."
    )
    energy: float = Field(
        ...,
        description="Building energy score [0, 100].",
        ge=0.0,
        le=100.0,
    )
    carbon: float = Field(
        ...,
        description="Grid carbon score [0, 100].",
        ge=0.0,
        le=100.0,
    )

    model_config = {"frozen": True}

    def worst_comfort_zone(self) -> str | None:
        """Return the zone_id with the lowest comfort score.

        Returns:
            The zone_id string, or None if no zones are present.
        """
        if not self.zone_scores:
            return None
        return min(
            self.zone_scores,
            key=lambda zid: self.zone_scores[zid].comfort,
        )

    def worst_air_quality_zone(self) -> str | None:
        """Return the zone_id with the lowest air quality score.

        Returns:
            The zone_id string, or None if no zones are present.
        """
        if not self.zone_scores:
            return None
        return min(
            self.zone_scores,
            key=lambda zid: self.zone_scores[zid].air_quality,
        )


# ---------------------------------------------------------------------------
# Score formula implementations
# ---------------------------------------------------------------------------


def _comfort_score(pmv: float) -> float:
    """Compute the ASHRAE-55 piecewise comfort score.

    Architecture §7.1:
        S_comfort = 100 - 20 * |PMV|          if |PMV| <= 0.5
        S_comfort = max(0, 90 - 60*(|PMV|-0.5)) if |PMV| >  0.5

    Continuity check at |PMV| = 0.5:
        branch-1 → 100 - 10 = 90
        branch-2 → 90 - 0   = 90  ✓

    Args:
        pmv: Predicted Mean Vote in [-3.0, +3.0].

    Returns:
        Comfort score in [0.0, 100.0].
    """
    abs_pmv = abs(pmv)
    if abs_pmv <= PMV_ACCEPT_HIGH:  # PMV_ACCEPT_HIGH == 0.5
        return 100.0 - 20.0 * abs_pmv
    return max(0.0, 90.0 - 60.0 * (abs_pmv - PMV_ACCEPT_HIGH))


def _air_quality_score(co2_ppm: float) -> float:
    """Compute the CO₂-based air quality score.

    Architecture §7.2:
        S_air = 100                          if CO2 <= 600
        S_air = max(0, 100 - (CO2-600)/10)  if CO2 >  600

    Args:
        co2_ppm: Indoor CO₂ concentration in PPM.

    Returns:
        Air quality score in [0.0, 100.0].
    """
    if co2_ppm <= 600.0:
        return 100.0
    return max(0.0, 100.0 - (co2_ppm - 600.0) / 10.0)


def _energy_score(
    power_ratio: float,
    tariff_ratio: float,
) -> float:
    """Compute the energy efficiency score.

    Architecture §7.3:
        S_energy = max(0, 100 - 100 * max(0, ratio-1) * price_r)

    Sanity checks:
        ratio=1.0, price_r=1.0 → S = 100 (normal baseline) ✓
        ratio=1.35, price_r=2.0 → S = 100 - 70 = 30        ✓

    Args:
        power_ratio: P_current / P_baseline.
        tariff_ratio: Price_current / Price_avg.

    Returns:
        Energy score in [0.0, 100.0].
    """
    excess = max(0.0, power_ratio - 1.0)
    return max(0.0, 100.0 - 100.0 * excess * tariff_ratio)


def _carbon_score(
    carbon_intensity: float,
    carbon_min: float,
    carbon_max: float,
) -> float:
    """Compute the grid carbon cleanliness score.

    Architecture §7.4:
        S_carbon = clamp(100 - 100*(C-C_min)/(C_max-C_min), 0, 100)

    Args:
        carbon_intensity: Current grid carbon intensity
            (gCO₂e/kWh).
        carbon_min: Day-ahead forecast minimum (cleanest window).
        carbon_max: Day-ahead forecast maximum (dirtiest window).

    Returns:
        Carbon score in [0.0, 100.0].

    Raises:
        ValueError: If carbon_max < carbon_min.
    """
    if carbon_max < carbon_min:
        raise ValueError(
            f"carbon_max ({carbon_max}) must be >= "
            f"carbon_min ({carbon_min})."
        )
    window = max(carbon_max - carbon_min, 1.0)  # guard div/0
    raw = 100.0 - 100.0 * (carbon_intensity - carbon_min) / window
    return max(0.0, min(100.0, raw))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_zone(zone_id: str, pmv: float, co2_ppm: float) -> ZoneScoreVector:
    """Compute comfort and air-quality scores for a single zone.

    Args:
        zone_id: Unique identifier of the zone.
        pmv: Predicted Mean Vote for the zone.
        co2_ppm: Indoor CO₂ concentration in PPM.

    Returns:
        A frozen ZoneScoreVector.
    """
    return ZoneScoreVector(
        zone_id=zone_id,
        comfort=_comfort_score(pmv),
        air_quality=_air_quality_score(co2_ppm),
    )


def score_building(
    state: BuildingState,
    *,
    total_power_kw: float,
    baseline_power_kw: float,
    avg_energy_price: float,
    carbon_min: float,
    carbon_max: float,
) -> BuildingScoreReport:
    """Compute the full BuildingScoreReport for one simulation step.

    Args:
        state: Full building state snapshot.
        total_power_kw: Aggregate active power draw across all
            equipment at this timestep.
        baseline_power_kw: Expected nominal power draw under
            normal operating conditions.
        avg_energy_price: Historical average electricity tariff
            ($/kWh) used to scale the energy penalty.
        carbon_min: Day-ahead forecast minimum carbon intensity
            (gCO₂e/kWh).
        carbon_max: Day-ahead forecast maximum carbon intensity
            (gCO₂e/kWh).

    Returns:
        A frozen BuildingScoreReport containing per-zone scores
        plus building-level energy and carbon scores.
    """
    zone_scores = {
        zid: score_zone(
            zone_id=zid,
            pmv=zone.pmv,
            co2_ppm=zone.co2_ppm,
        )
        for zid, zone in state.zones.items()
    }

    power_ratio = total_power_kw / max(baseline_power_kw, 0.001)
    tariff_ratio = state.energy_price / max(avg_energy_price, 0.001)

    return BuildingScoreReport(
        zone_scores=zone_scores,
        energy=_energy_score(power_ratio, tariff_ratio),
        carbon=_carbon_score(
            state.carbon_intensity, carbon_min, carbon_max
        ),
    )
