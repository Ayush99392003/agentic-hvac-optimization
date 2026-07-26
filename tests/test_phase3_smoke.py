"""Phase 3 smoke test -- Event Generator.

Verifies that:
  - All six detection rules fire correctly for the demo scenario.
  - Severity mapping from scores to EventSeverity is correct.
  - EventList query helpers (by_type, by_zone, has_critical) work.
  - No false positives fire when all metrics are nominal.

Run with:
    uv run python -c "import sys; sys.path.insert(0,'.');
        from tests.test_phase3_smoke import main; main()"
"""

import sys
from datetime import datetime, timezone

from src.engine.features import extract_building_features
from src.engine.scores import score_building
from src.events import EventList, EventSeverity, EventType, generate_events
from src.state import (
    BuildingState,
    EquipmentState,
    EquipmentStatus,
    WeatherState,
    ZoneState,
)

SEP = "=" * 60
PASS = "[PASS]"
FAIL = "[FAIL]"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _check_bool(label: str, got: bool, expected: bool) -> bool:
    ok = got == expected
    print(f"  {'[PASS]' if ok else '[FAIL]'}  {label}")
    if not ok:
        print(f"        got={got}  expected={expected}")
    return ok


def _check_contains(
    label: str, event_list: EventList, event_type: EventType
) -> bool:
    ok = event_type in event_list.active_event_types
    print(f"  {'[PASS]' if ok else '[FAIL]'}  {label}")
    return ok


def _check_not_contains(
    label: str, event_list: EventList, event_type: EventType
) -> bool:
    ok = event_type not in event_list.active_event_types
    print(f"  {'[PASS]' if ok else '[FAIL]'}  {label}")
    return ok


# ---------------------------------------------------------------------------
# Demo scenario fixture
# ---------------------------------------------------------------------------


def _build_demo_state() -> BuildingState:
    """Return the architecture section 6.3 demo BuildingState."""
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
                occupancy_count=0,   # Unoccupied
                cooling_setpoint=23.0,
                heating_setpoint=20.0,
            ),
        },
        equipment={
            "AHU_01": EquipmentState(
                equipment_id="AHU_01",
                status=EquipmentStatus.ON,
                current_power_kw=52.0,  # elevated vs 35 baseline
                fan_speed_pct=75.0,
                damper_position_pct=60.0,
            )
        },
        energy_price=0.45,       # peak tariff
        carbon_intensity=430.0,  # high carbon window
    )


def _run_full_pipeline(
    state: BuildingState,
) -> tuple[EventList, object, object]:
    """Run Phases 2+3 on a BuildingState and return all outputs."""
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
    return events, scores, features


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_demo_scenario_events() -> list[bool]:
    """All expected events fire for the demo scenario."""
    print("\n-- Demo Scenario Event Detection -------------")
    state = _build_demo_state()
    events, scores, _ = _run_full_pipeline(state)

    results = []
    results.append(
        _check_contains(
            "ZONE_OVERHEATING fires (conf_room PMV=1.1)",
            events,
            EventType.ZONE_OVERHEATING,
        )
    )
    results.append(
        _check_contains(
            "POOR_AIR_QUALITY fires (conf_room CO2=1150)",
            events,
            EventType.POOR_AIR_QUALITY,
        )
    )
    results.append(
        _check_contains(
            "ZONE_UNDERUTILIZED fires (board_room unoccupied)",
            events,
            EventType.ZONE_UNDERUTILIZED,
        )
    )
    results.append(
        _check_contains(
            "PEAK_DEMAND_RISK fires (energy_score<50)",
            events,
            EventType.PEAK_DEMAND_RISK,
        )
    )
    results.append(
        _check_contains(
            "HIGH_CARBON_WINDOW fires (carbon_score=30<35)",
            events,
            EventType.HIGH_CARBON_WINDOW,
        )
    )
    results.append(
        _check_contains(
            "COOLING_INEFFICIENT fires (comfort+energy both low)",
            events,
            EventType.COOLING_INEFFICIENT,
        )
    )
    results.append(
        _check_bool(
            "has_critical=True (conf_room comfort=54 -> CRITICAL)",
            events.has_critical,
            True,
        )
    )
    return results


def test_zone_filter_helpers() -> list[bool]:
    """EventList query helpers return correct subsets."""
    print("\n-- EventList Query Helpers -------------------")
    state = _build_demo_state()
    events, _, _ = _run_full_pipeline(state)

    results = []

    conf_events = events.by_zone("conf_room_exec")
    ok1 = len(conf_events) >= 2
    print(
        f"  {'[PASS]' if ok1 else '[FAIL]'}  "
        f"conf_room has >=2 events (got {len(conf_events)})"
    )
    results.append(ok1)

    critical_events = events.by_severity(EventSeverity.CRITICAL)
    ok2 = len(critical_events) >= 1
    print(
        f"  {'[PASS]' if ok2 else '[FAIL]'}  "
        f">= 1 CRITICAL event (got {len(critical_events)})"
    )
    results.append(ok2)

    aq_events = events.by_type(EventType.POOR_AIR_QUALITY)
    ok3 = any(e.zone_id == "conf_room_exec" for e in aq_events)
    print(
        f"  {'[PASS]' if ok3 else '[FAIL]'}  "
        f"POOR_AIR_QUALITY linked to conf_room_exec"
    )
    results.append(ok3)

    return results


def test_no_false_positives_nominal() -> list[bool]:
    """No events fire when all metrics are nominal."""
    print("\n-- Nominal State: No False Positives ---------")
    nominal_state = BuildingState(
        timestamp=datetime(
            2025, 8, 15, 9, 0, 0, tzinfo=timezone.utc
        ),
        weather=WeatherState(
            dry_bulb_temperature=22.0,
            relative_humidity=50.0,
            direct_solar_irradiance=200.0,
            wind_speed=2.0,
        ),
        zones={
            "zone_a": ZoneState(
                zone_id="zone_a",
                temperature=22.5,
                relative_humidity=48.0,
                pmv=0.1,
                co2_ppm=550.0,
                occupancy_count=3,
                cooling_setpoint=24.0,
                heating_setpoint=20.0,
            )
        },
        equipment={},
        energy_price=0.18,    # below average
        carbon_intensity=180.0,  # clean window
    )

    scores = score_building(
        nominal_state,
        total_power_kw=30.0,
        baseline_power_kw=35.0,   # under baseline -> no penalty
        avg_energy_price=0.22,
        carbon_min=150.0,
        carbon_max=550.0,
    )
    features = extract_building_features(
        nominal_state,
        avg_energy_price=0.22,
        total_power_kw=30.0,
        baseline_power_kw=35.0,
        carbon_min=150.0,
        carbon_max=550.0,
    )
    events = generate_events(nominal_state, scores, features)

    results = []
    results.append(
        _check_not_contains(
            "No ZONE_OVERHEATING (PMV=0.1)",
            events,
            EventType.ZONE_OVERHEATING,
        )
    )
    results.append(
        _check_not_contains(
            "No POOR_AIR_QUALITY (CO2=550)",
            events,
            EventType.POOR_AIR_QUALITY,
        )
    )
    results.append(
        _check_not_contains(
            "No PEAK_DEMAND_RISK (low tariff+power)",
            events,
            EventType.PEAK_DEMAND_RISK,
        )
    )
    results.append(
        _check_not_contains(
            "No HIGH_CARBON_WINDOW (carbon score clean)",
            events,
            EventType.HIGH_CARBON_WINDOW,
        )
    )
    ok_count = len(events.events) == 0
    print(
        f"  {'[PASS]' if ok_count else '[FAIL]'}  "
        f"Total events = 0 (got {len(events.events)})"
    )
    results.append(ok_count)
    return results


def test_severity_overheating() -> list[bool]:
    """ZONE_OVERHEATING severity scales correctly with score."""
    print("\n-- Severity Mapping --------------------------")
    results = []

    # PMV=1.1 -> comfort=54 -> CRITICAL (<=30 is critical, 30<54<=55 is HIGH)
    state = _build_demo_state()
    events, _, _ = _run_full_pipeline(state)
    oh_events = events.by_type(EventType.ZONE_OVERHEATING)
    conf_oh = [e for e in oh_events if e.zone_id == "conf_room_exec"]
    ok1 = (
        len(conf_oh) == 1
        and conf_oh[0].severity == EventSeverity.HIGH
    )
    print(
        f"  {'[PASS]' if ok1 else '[FAIL]'}  "
        f"conf_room OVERHEATING severity=HIGH "
        f"(comfort=54, threshold HIGH<=55)"
    )
    results.append(ok1)

    # Verify POOR_AIR_QUALITY also lands as CRITICAL (air=45<=30?)
    # air=45 -> 30<45<=55 -> HIGH
    aq_events = events.by_type(EventType.POOR_AIR_QUALITY)
    conf_aq = [e for e in aq_events if e.zone_id == "conf_room_exec"]
    ok2 = (
        len(conf_aq) == 1
        and conf_aq[0].severity == EventSeverity.HIGH
    )
    print(
        f"  {'[PASS]' if ok2 else '[FAIL]'}  "
        f"conf_room AIR_QUALITY severity=HIGH "
        f"(air_quality=45, threshold HIGH<=55)"
    )
    results.append(ok2)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all Phase 3 tests and report summary."""
    print(SEP)
    print("  Phase 3 -- Event Generator Smoke Test")
    print(SEP)

    all_results: list[bool] = []
    all_results += test_demo_scenario_events()
    all_results += test_zone_filter_helpers()
    all_results += test_no_false_positives_nominal()
    all_results += test_severity_overheating()

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
