"""Feedback Engine — expected-vs-actual comparison and audit trail.

Compares what the pipeline intended to achieve (from the
OrchestratorPlan and pre-action BuildingState) against what the
simulator actually observed (SimStepResult). Produces a
FeedbackReport for each cycle, forming the audit trail required
by architecture §2.2.

The FeedbackEngine also advances the BuildingState for the next
cycle by folding the simulation result back into a new
BuildingState snapshot — closing the feedback loop.

Architecture reference: Section 7 (Simulation & Feedback Loop),
Section 2.2 (Audit Trail — expected vs. actual).
"""

from __future__ import annotations

import logging
from datetime import datetime

from src.engine.scores import score_building
from src.orchestrator.models import OrchestratorPlan
from src.simulation.models import (
    FeedbackReport,
    SimStepResult,
    ZoneDelta,
)
from src.state.building import BuildingState
from src.state.zone import ZoneState

logger = logging.getLogger(__name__)


def _compute_expected_temp(
    zone_id: str,
    state: BuildingState,
    plan: OrchestratorPlan,
) -> float:
    """Compute the temperature the pipeline intended to achieve.

    Uses the same first-order lag as ReplayedSimulator so
    'expected' and 'actual' use a consistent model.

    Args:
        zone_id: Zone to compute expected temperature for.
        state: Pre-action building state.
        plan: Orchestrator plan with setpoint deltas.

    Returns:
        Expected zone temperature after one step (°C).
    """
    zone = state.zones[zone_id]
    zplan = plan.zone_plans.get(zone_id)

    if zplan and zplan.cooling_setpoint_delta is not None:
        intended_sp = (
            zone.cooling_setpoint + zplan.cooling_setpoint_delta
        )
    else:
        intended_sp = zone.cooling_setpoint

    # First-order lag (same as ReplayedSimulator._THERMAL_RESPONSE)
    from src.simulation.simulator import _THERMAL_RESPONSE

    gap = zone.temperature - intended_sp
    return zone.temperature - _THERMAL_RESPONSE * gap


def _compute_expected_co2(
    zone_id: str,
    state: BuildingState,
    plan: OrchestratorPlan,
) -> float:
    """Compute the CO₂ the pipeline expected after one step.

    Args:
        zone_id: Zone to compute expected CO₂ for.
        state: Pre-action building state.
        plan: Orchestrator plan with damper deltas.

    Returns:
        Expected CO₂ level (PPM) after one step.
    """
    from src.simulation.simulator import (
        _CO2_DECAY_WITH_VENT,
        _CO2_RISE_PER_OCCUPANT,
        _MIN_EFFECTIVE_DAMPER,
        _MIN_EFFECTIVE_FAN,
    )

    zone = state.zones[zone_id]
    zplan = plan.zone_plans.get(zone_id)
    equip = next((e for e in state.equipment.values()), None)

    current_damper = equip.damper_position_pct if equip else 50.0
    current_fan = equip.fan_speed_pct if equip else 60.0

    eff_damper = (
        current_damper + zplan.damper_delta_pct
        if zplan and zplan.damper_delta_pct is not None
        else current_damper
    )
    eff_fan = (
        current_fan + zplan.fan_speed_delta_pct
        if zplan and zplan.fan_speed_delta_pct is not None
        else current_fan
    )

    vent_active = (
        eff_damper >= _MIN_EFFECTIVE_DAMPER
        and eff_fan >= _MIN_EFFECTIVE_FAN
    )
    if vent_active:
        co2_change = (
            -_CO2_DECAY_WITH_VENT
            + zone.occupancy_count * _CO2_RISE_PER_OCCUPANT
        )
    else:
        co2_change = zone.occupancy_count * _CO2_RISE_PER_OCCUPANT * 2

    return max(380.0, zone.co2_ppm + co2_change)


def compute_feedback(
    cycle_index: int,
    state: BuildingState,
    plan: OrchestratorPlan,
    sim_result: SimStepResult,
    avg_energy_price: float,
    baseline_power_kw: float,
    carbon_min: float,
    carbon_max: float,
) -> FeedbackReport:
    """Produce a FeedbackReport for one completed pipeline cycle.

    Compares expected outcomes (derived from the plan and pre-action
    state) against actual outcomes (from the simulator). Calculates
    energy savings and comfort tracking errors.

    Args:
        cycle_index: Sequential cycle number (0-indexed).
        state: Pre-action building state for this cycle.
        plan: The OrchestratorPlan executed this cycle.
        sim_result: Observed post-action state from the simulator.
        avg_energy_price: Average tariff for score computation.
        baseline_power_kw: Expected baseline building power (kW).
        carbon_min: Grid carbon forecast minimum (gCO2e/kWh).
        carbon_max: Grid carbon forecast maximum (gCO2e/kWh).

    Returns:
        A FeedbackReport for this cycle.
    """
    zone_deltas: dict[str, ZoneDelta] = {}

    pre_scores = score_building(
        state,
        total_power_kw=sum(
            e.current_power_kw for e in state.equipment.values()
        ) or baseline_power_kw,
        baseline_power_kw=baseline_power_kw,
        avg_energy_price=avg_energy_price,
        carbon_min=carbon_min,
        carbon_max=carbon_max,
    )

    for zid in state.zones:
        sim_zone = sim_result.zone_results.get(zid)
        if sim_zone is None:
            continue

        expected_temp = _compute_expected_temp(zid, state, plan)
        expected_co2 = _compute_expected_co2(zid, state, plan)

        # Setpoint tracking error: commanded vs. achieved.
        zplan = plan.zone_plans.get(zid)
        zone = state.zones[zid]
        commanded_sp = (
            zone.cooling_setpoint + zplan.cooling_setpoint_delta
            if zplan and zplan.cooling_setpoint_delta is not None
            else zone.cooling_setpoint
        )
        tracking_err = (
            sim_zone.cooling_setpoint_achieved - commanded_sp
        )

        # Comfort score change: compare pre and post PMV.
        pre_pmv = state.zones[zid].pmv
        post_pmv = sim_zone.pmv
        pre_comfort = pre_scores.zone_scores[zid].comfort
        # Approximate post-comfort using same formula.
        if abs(post_pmv) <= 0.5:
            post_comfort = 100.0 - 20.0 * abs(post_pmv)
        else:
            post_comfort = max(
                0.0, 90.0 - 60.0 * (abs(post_pmv) - 0.5)
            )
        comfort_delta = post_comfort - pre_comfort

        zone_deltas[zid] = ZoneDelta(
            zone_id=zid,
            temp_expected=round(expected_temp, 2),
            temp_actual=sim_zone.temperature,
            temp_error=round(
                sim_zone.temperature - expected_temp, 3
            ),
            co2_expected=round(expected_co2, 1),
            co2_actual=sim_zone.co2_ppm,
            setpoint_tracking_error=round(tracking_err, 3),
            comfort_score_delta=round(comfort_delta, 1),
        )

    # Energy savings: pre-action power vs. post-action power.
    pre_power = sum(
        e.current_power_kw for e in state.equipment.values()
    ) or baseline_power_kw
    post_power = sim_result.total_power_kw
    savings_pct = (
        (pre_power - post_power) / pre_power * 100.0
        if pre_power > 0
        else 0.0
    )

    summary_parts = [
        f"Cycle {cycle_index}:",
        f"power {pre_power:.1f}->{post_power:.1f}kW "
        f"({savings_pct:+.1f}%)",
    ]
    comfort_changes = [
        f"{zid}:{d.comfort_score_delta:+.1f}"
        for zid, d in zone_deltas.items()
    ]
    if comfort_changes:
        summary_parts.append(
            "comfort " + " ".join(comfort_changes)
        )

    return FeedbackReport(
        cycle_index=cycle_index,
        timestamp=sim_result.timestamp,
        zone_deltas=zone_deltas,
        total_power_expected_kw=round(pre_power, 2),
        total_power_actual_kw=round(post_power, 2),
        energy_savings_pct=round(savings_pct, 2),
        policy_rules_active=list(plan.active_policy_rules),
        conflict_resolutions=list(plan.conflict_resolutions),
        overall_summary=" | ".join(summary_parts),
    )


def advance_state(
    state: BuildingState,
    sim_result: SimStepResult,
) -> BuildingState:
    """Produce the next BuildingState from a SimStepResult.

    Folds observed zone temperatures, CO₂, and PMV back into a
    new BuildingState snapshot for the next pipeline cycle.
    Equipment state is left unchanged (no actuator feedback yet).

    Args:
        state: The pre-action building state.
        sim_result: The simulator's observed post-action state.

    Returns:
        A new BuildingState for the next pipeline cycle.
    """
    new_zones: dict[str, ZoneState] = {}

    for zid, zone in state.zones.items():
        sim_zone = sim_result.zone_results.get(zid)
        if sim_zone is None:
            new_zones[zid] = zone
            continue

        new_zones[zid] = ZoneState(
            zone_id=zid,
            temperature=sim_zone.temperature,
            relative_humidity=zone.relative_humidity,
            pmv=sim_zone.pmv,
            co2_ppm=sim_zone.co2_ppm,
            occupancy_count=zone.occupancy_count,
            cooling_setpoint=sim_zone.cooling_setpoint_achieved,
            heating_setpoint=zone.heating_setpoint,
        )

    return BuildingState(
        timestamp=sim_result.timestamp,
        weather=state.weather,
        zones=new_zones,
        equipment=state.equipment,
        energy_price=state.energy_price,
        carbon_intensity=state.carbon_intensity,
    )
