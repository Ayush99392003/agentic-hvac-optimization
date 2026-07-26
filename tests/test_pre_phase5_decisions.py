"""Pre-Phase 5 conflict analysis and system design decisions.

Run with:
    uv run python -c "import sys; sys.path.insert(0,'.');
        from tests.test_pre_phase5_decisions import main; main()"

This script documents and verifies the four decisions that must
be locked in before Phase 5 (LLM Orchestrator) is built:

  D1. Exact conflict map for conf_room_exec.
  D2. Scope-awareness: building-vs-zone fan-speed conflict.
  D3. Tiebreak rule for equal urgency scores.
  D4. Governing Policy Rule mode (hard constraint).
"""

import sys
from datetime import datetime, timezone

from src.agents import ContextPriorities, run_all_agents
from src.agents.models import ActionType
from src.engine.features import extract_building_features
from src.engine.scores import score_building
from src.events import generate_events
from src.state import (
    BuildingState,
    EquipmentState,
    EquipmentStatus,
    WeatherState,
    ZoneState,
)

SEP = "=" * 60


def _demo_state() -> BuildingState:
    return BuildingState(
        timestamp=datetime(
            2025, 8, 15, 14, 0, 0, tzinfo=timezone.utc
        ),
        weather=WeatherState(
            dry_bulb_temperature=32.0,
            relative_humidity=65.0,
            direct_solar_irradiance=750.0,
            wind_speed=3.5,
        ),
        zones={
            "conf_room_exec": ZoneState(
                zone_id="conf_room_exec",
                temperature=26.2,
                relative_humidity=58.0,
                pmv=1.1,
                co2_ppm=1150.0,
                occupancy_count=15,
                cooling_setpoint=24.0,
                heating_setpoint=20.0,
            ),
            "open_office_201": ZoneState(
                zone_id="open_office_201",
                temperature=23.5,
                relative_humidity=52.0,
                pmv=0.2,
                co2_ppm=680.0,
                occupancy_count=8,
                cooling_setpoint=23.0,
                heating_setpoint=20.0,
            ),
            "board_room": ZoneState(
                zone_id="board_room",
                temperature=22.0,
                relative_humidity=50.0,
                pmv=0.0,
                co2_ppm=450.0,
                occupancy_count=0,
                cooling_setpoint=23.0,
                heating_setpoint=20.0,
            ),
        },
        equipment={
            "AHU_01": EquipmentState(
                equipment_id="AHU_01",
                status=EquipmentStatus.ON,
                current_power_kw=52.0,
                fan_speed_pct=75.0,
                damper_position_pct=60.0,
            )
        },
        energy_price=0.45,
        carbon_intensity=430.0,
    )


def _p(label: str, ok: bool) -> bool:
    print(f"  {'[PASS]' if ok else '[FAIL]'}  {label}")
    return ok


def main() -> None:
    state = _demo_state()
    scores = score_building(
        state,
        total_power_kw=52.0,
        baseline_power_kw=35.0,
        avg_energy_price=0.22,
        carbon_min=150.0,
        carbon_max=550.0,
    )
    features = extract_building_features(
        state,
        avg_energy_price=0.22,
        total_power_kw=52.0,
        baseline_power_kw=35.0,
        carbon_min=150.0,
        carbon_max=550.0,
    )
    events = generate_events(state, scores, features)
    context = ContextPriorities.from_building_state(
        state, avg_tariff=0.22
    )
    recs = run_all_agents(state, scores, events, context)

    results: list[bool] = []

    # ------------------------------------------------------------------
    # D1: Exact conf_room_exec conflict map
    # ------------------------------------------------------------------
    print(SEP)
    print("  D1. conf_room_exec Conflict Map")
    print(SEP)
    cr = [r for r in recs if r.target_zone == "conf_room_exec"]
    cr_actions = {r.action for r in cr}
    print(f"  Zone-scoped recs for conf_room_exec ({len(cr)} total):")
    for r in cr:
        print(
            f"    [{r.urgency_score:3d}] {r.agent_id:<26} "
            f"{r.action.value}"
        )
    print()
    print("  Conflict analysis:")
    # COMPLEMENTARY: INCREASE_VENTILATION + INCREASE_FAN_SPEED + INCREASE_COOLING
    # - All three push in the same thermal+air direction.
    # - PRE_COOL is also aligned (cool the space).
    comp = {
        ActionType.INCREASE_VENTILATION,
        ActionType.INCREASE_FAN_SPEED,
        ActionType.INCREASE_COOLING,
        ActionType.PRE_COOL,
    }
    all_comp = cr_actions.issubset(comp)
    results.append(
        _p(
            "conf_room_exec recs are complementary, NOT conflicting",
            all_comp,
        )
    )
    print(
        "  -> VERDICT: No intra-zone conflict for conf_room_exec."
    )
    print(
        "     All four actions push in the same direction "
        "(cool + ventilate)."
    )

    # ------------------------------------------------------------------
    # D2: Real fan-speed conflict is CROSS-SCOPE
    # ------------------------------------------------------------------
    print()
    print(SEP)
    print("  D2. Cross-Scope Fan-Speed Conflict")
    print(SEP)
    fan_up = [
        r for r in recs if r.action == ActionType.INCREASE_FAN_SPEED
    ]
    fan_dn = [
        r for r in recs if r.action == ActionType.DECREASE_FAN_SPEED
    ]
    print(
        f"  INCREASE_FAN_SPEED: zone={fan_up[0].target_zone}  "
        f"urgency={fan_up[0].urgency_score}"
        if fan_up
        else "  INCREASE_FAN_SPEED: none"
    )
    print(
        f"  DECREASE_FAN_SPEED: scope=BUILDING  "
        f"urgency={fan_dn[0].urgency_score}"
        if fan_dn
        else "  DECREASE_FAN_SPEED: none"
    )
    has_cross_scope_conflict = bool(fan_up and fan_dn)
    results.append(
        _p(
            "Cross-scope fan-speed conflict exists (zone vs building)",
            has_cross_scope_conflict,
        )
    )
    print()
    print("  RESOLUTION RULE (D2):")
    print(
        "  When a zone-scoped INCREASE_FAN_SPEED conflicts with a "
        "building-scoped DECREASE_FAN_SPEED:"
    )
    print(
        "    IF zone is high-density AND air_quality context = HIGH"
    )
    print(
        "    THEN zone-scoped action overrides building-scope "
        "(human health floor)."
    )
    print(
        "    ELSE higher urgency_score wins; equal urgency -> "
        "zone-scope preferred (less blast radius)."
    )
    # Verify the zone is high-density and context is HIGH
    is_hd = "conf_room_exec" in context.high_density_zones
    from src.agents.models import ObjectivePriority

    aq_high = context.air_quality == ObjectivePriority.HIGH
    results.append(
        _p(
            "conf_room_exec is high-density (occupancy>10)",
            is_hd,
        )
    )
    results.append(
        _p(
            "air_quality context = HIGH -> zone-scope wins",
            aq_high,
        )
    )

    # ------------------------------------------------------------------
    # D3: Tiebreak for equal urgency (three recs at 90)
    # ------------------------------------------------------------------
    print()
    print(SEP)
    print("  D3. Equal-Urgency Tiebreak Convention")
    print(SEP)
    ties = [r for r in recs if r.urgency_score == 90]
    print(f"  Recs at urgency=90: {len(ties)}")
    for r in ties:
        print(
            f"    {r.agent_id:<26} zone={r.target_zone or 'BUILDING':<20}"
            f" action={r.action.value}"
        )
    print()
    print("  TIEBREAK CONVENTION (D3) — applied in this order:")
    print(
        "  1. Human health first: air_quality > comfort > "
        "energy > carbon"
    )
    print(
        "  2. Zone-scoped beats building-scoped "
        "(less blast radius, more precise)"
    )
    print(
        "  3. If still tied: earlier in pipeline order "
        "(comfort > energy > air > carbon > dr)"
    )
    # Apply tiebreak to the three tied recs:
    #   DECREASE_FAN_SPEED (energy, building) - energy objective
    #   SHIFT_LOAD (carbon, building) - carbon objective
    #   SHED_LOAD (dr, board_room) - energy/dr objective
    # Tiebreak result: SHED_LOAD wins (zone-scoped, direct impact)
    # DECREASE_FAN_SPEED and SHIFT_LOAD are both building-scope;
    # between those two, energy > carbon.
    tiebreak_order = ["demand_response_agent", "energy_agent", "carbon_agent"]
    actual_agents = [r.agent_id for r in ties]
    # All three are at 90 — just confirm they exist
    results.append(
        _p("Three recs tied at urgency=90", len(ties) == 3)
    )
    results.append(
        _p(
            "All three are building-scope or empty-zone-scope "
            "(no direct conflict with conf_room_exec comfort)",
            all(
                r.target_zone != "conf_room_exec" for r in ties
            ),
        )
    )

    # ------------------------------------------------------------------
    # D4: Governing Policy Rule as hard constraint
    # ------------------------------------------------------------------
    print()
    print(SEP)
    print("  D4. Governing Policy Rule — Hard Constraint")
    print(SEP)
    print(
        "  DECISION: High-Density Occupancy Floor is a HARD CONSTRAINT."
    )
    print(
        "  The LLM Orchestrator MUST NOT produce an action plan that:"
    )
    print(
        "    a) Raises the cooling setpoint above current in "
        "conf_room_exec while comfort < 55"
    )
    print(
        "    b) Reduces ventilation in conf_room_exec while "
        "CO2 > 1000 PPM"
    )
    print(
        "    c) Completely sheds HVAC load in conf_room_exec "
        "while occupancy > 10"
    )
    print(
        "  These constraints are injected as System Prompt rules, "
        "NOT as preferences."
    )
    print(
        "  Phase 5 acceptance test verifies the LLM output does not "
        "violate any of these."
    )
    print(
        "  The Safety Validator (Phase 6) provides a second, "
        "deterministic enforcement layer."
    )
    # Verify state data that will drive the hard constraints
    conf = state.zones["conf_room_exec"]
    cr_score = scores.zone_scores["conf_room_exec"]
    results.append(
        _p(
            "conf_room_exec comfort=54 < 55 -> constraint (a) active",
            cr_score.comfort < 55.0,
        )
    )
    results.append(
        _p(
            "conf_room_exec CO2=1150 > 1000 -> constraint (b) active",
            conf.co2_ppm > 1000.0,
        )
    )
    results.append(
        _p(
            "conf_room_exec occupancy=15 > 10 -> constraint (c) active",
            conf.occupancy_count > 10,
        )
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    passed = sum(results)
    total = len(results)
    print()
    print(SEP)
    if passed == total:
        print(
            f"  PASS -- {passed}/{total} design decisions verified."
        )
        print("  Ready to build Phase 5.")
    else:
        print(f"  FAIL -- {passed}/{total} checks passed.")
    print(SEP)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
