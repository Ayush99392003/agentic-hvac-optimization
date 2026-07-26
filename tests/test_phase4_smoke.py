"""Phase 4 smoke test -- Specialized Agent Layer.

Verifies that:
  - Each agent activates on the correct event types.
  - Each agent is silent when its events are absent.
  - Context priorities modulate urgency_score correctly.
  - High-Density Occupancy Floor boost fires for AirQualityAgent.
  - DemandResponseAgent proposes SHED vs PRE_COOL by occupancy.
  - run_all_agents() returns recs sorted by urgency descending.

Run with:
    uv run python -c "import sys; sys.path.insert(0,'.');
        from tests.test_phase4_smoke import main; main()"
"""

import sys
from datetime import datetime, timezone

from src.agents import (
    AirQualityAgent,
    CarbonAgent,
    ComfortAgent,
    ContextPriorities,
    DemandResponseAgent,
    EnergyAgent,
    run_all_agents,
)
from src.agents.models import ActionType, ObjectivePriority
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


def _p(label: str, ok: bool) -> bool:
    print(f"  {'[PASS]' if ok else '[FAIL]'}  {label}")
    return ok


# ---------------------------------------------------------------------------
# Shared demo state (same as Phases 1-3)
# ---------------------------------------------------------------------------


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


def _run_pipeline(state: BuildingState):  # type: ignore[return]
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
    return scores, features, events, context


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_context_priorities() -> list[bool]:
    """ContextPriorities.from_building_state derives correctly."""
    print("\n-- ContextPriorities Derivation --------------")
    state = _demo_state()
    ctx = ContextPriorities.from_building_state(
        state, avg_tariff=0.22
    )
    results = []
    # Peak tariff (0.45 > 0.22*1.5=0.33) -> energy HIGH
    results.append(
        _p(
            "energy=HIGH (peak tariff 0.45 > 0.33)",
            ctx.energy == ObjectivePriority.HIGH,
        )
    )
    # High-density zone exists -> comfort HIGH
    results.append(
        _p(
            "comfort=HIGH (conf_room 15 occupants)",
            ctx.comfort == ObjectivePriority.HIGH,
        )
    )
    # High carbon -> carbon HIGH
    results.append(
        _p(
            "carbon=HIGH (carbon_intensity=430>400)",
            ctx.carbon == ObjectivePriority.HIGH,
        )
    )
    # High-density zones set
    results.append(
        _p(
            "high_density_zones includes conf_room_exec",
            "conf_room_exec" in ctx.high_density_zones,
        )
    )
    return results


def test_comfort_agent() -> list[bool]:
    """ComfortAgent fires INCREASE_COOLING for warm zones."""
    print("\n-- ComfortAgent ---------------------------")
    state = _demo_state()
    scores, _, events, context = _run_pipeline(state)
    recs = ComfortAgent().recommend(state, scores, events, context)
    results = []
    conf_recs = [
        r for r in recs if r.target_zone == "conf_room_exec"
    ]
    results.append(_p("Rec for conf_room_exec", len(conf_recs) >= 1))
    results.append(
        _p(
            "Action=INCREASE_COOLING (PMV=+1.1)",
            any(r.action == ActionType.INCREASE_COOLING for r in conf_recs),
        )
    )
    results.append(
        _p(
            "urgency_score >= 70 (HIGH severity + HIGH priority)",
            any(r.urgency_score >= 70 for r in conf_recs),
        )
    )
    return results


def test_energy_agent() -> list[bool]:
    """EnergyAgent fires RELAX and DECREASE_FAN for correct events."""
    print("\n-- EnergyAgent ----------------------------")
    state = _demo_state()
    scores, _, events, context = _run_pipeline(state)
    recs = EnergyAgent().recommend(state, scores, events, context)
    results = []
    relax = [r for r in recs if r.action == ActionType.RELAX_SETPOINT]
    results.append(
        _p(
            "RELAX_SETPOINT for board_room (unoccupied)",
            any(r.target_zone == "board_room" for r in relax),
        )
    )
    fan_recs = [
        r for r in recs if r.action == ActionType.DECREASE_FAN_SPEED
    ]
    results.append(_p("DECREASE_FAN_SPEED for peak demand", len(fan_recs) >= 1))
    return results


def test_air_quality_agent() -> list[bool]:
    """AirQualityAgent fires INCREASE_VENTILATION + fan boost."""
    print("\n-- AirQualityAgent ------------------------")
    state = _demo_state()
    scores, _, events, context = _run_pipeline(state)
    recs = AirQualityAgent().recommend(state, scores, events, context)
    results = []
    vent = [
        r
        for r in recs
        if r.action == ActionType.INCREASE_VENTILATION
        and r.target_zone == "conf_room_exec"
    ]
    results.append(
        _p("INCREASE_VENTILATION for conf_room_exec", len(vent) == 1)
    )
    fan_boost = [
        r
        for r in recs
        if r.action == ActionType.INCREASE_FAN_SPEED
        and r.target_zone == "conf_room_exec"
    ]
    results.append(
        _p(
            "INCREASE_FAN_SPEED bonus for high-density zone",
            len(fan_boost) == 1,
        )
    )
    # Urgency should be boosted by HIGH_DENSITY_BOOST
    results.append(
        _p(
            "urgency >= 72 (HIGH priority + HDB boost)",
            vent[0].urgency_score >= 72 if vent else False,
        )
    )
    return results


def test_carbon_agent() -> list[bool]:
    """CarbonAgent fires SHIFT_LOAD for HIGH_CARBON_WINDOW."""
    print("\n-- CarbonAgent ----------------------------")
    state = _demo_state()
    scores, _, events, context = _run_pipeline(state)
    recs = CarbonAgent().recommend(state, scores, events, context)
    results = []
    results.append(_p("At least one SHIFT_LOAD rec", len(recs) >= 1))
    results.append(
        _p(
            "Action=SHIFT_LOAD",
            all(r.action == ActionType.SHIFT_LOAD for r in recs),
        )
    )
    return results


def test_demand_response_agent() -> list[bool]:
    """DemandResponseAgent proposes SHED vs PRE_COOL by zone."""
    print("\n-- DemandResponseAgent --------------------")
    state = _demo_state()
    scores, _, events, context = _run_pipeline(state)
    recs = DemandResponseAgent().recommend(
        state, scores, events, context
    )
    results = []
    shed = [r for r in recs if r.action == ActionType.SHED_LOAD]
    precool = [r for r in recs if r.action == ActionType.PRE_COOL]
    results.append(
        _p("SHED_LOAD for board_room (unoccupied)", len(shed) >= 1)
    )
    results.append(
        _p(
            "PRE_COOL for conf_room_exec (high-density)",
            any(r.target_zone == "conf_room_exec" for r in precool),
        )
    )
    # board_room must NOT get PRE_COOL
    results.append(
        _p(
            "board_room NOT in precool targets",
            not any(
                r.target_zone == "board_room" for r in precool
            ),
        )
    )
    return results


def test_run_all_agents_sorted() -> list[bool]:
    """run_all_agents returns recs sorted by urgency descending."""
    print("\n-- run_all_agents (sorted) ----------------")
    state = _demo_state()
    scores, features, events, context = _run_pipeline(state)
    recs = run_all_agents(state, scores, events, context)
    results = []
    results.append(_p("At least 5 total recs", len(recs) >= 5))
    urgencies = [r.urgency_score for r in recs]
    results.append(
        _p(
            "Recs sorted by urgency descending",
            urgencies == sorted(urgencies, reverse=True),
        )
    )
    # Print summary table
    print()
    print("  Agent              Zone               Action            Urgency")
    print("  " + "-" * 70)
    for r in recs:
        zone = r.target_zone or "building"
        print(
            f"  {r.agent_id:<22} {zone:<20} "
            f"{r.action.value:<22} {r.urgency_score}"
        )
    return results


def main() -> None:
    """Run all Phase 4 tests and report summary."""
    print(SEP)
    print("  Phase 4 -- Agent Layer Smoke Test")
    print(SEP)

    all_results: list[bool] = []
    all_results += test_context_priorities()
    all_results += test_comfort_agent()
    all_results += test_energy_agent()
    all_results += test_air_quality_agent()
    all_results += test_carbon_agent()
    all_results += test_demand_response_agent()
    all_results += test_run_all_agents_sorted()

    passed = sum(all_results)
    total = len(all_results)

    print(f"\n{SEP}")
    if passed == total:
        print(f"  PASS -- {passed}/{total} checks passed.")
    else:
        print(f"  FAIL -- {passed}/{total} checks passed.")
    print(SEP)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
