"""Phase 6 smoke test -- Action Engine & Safety Validator.

Verifies:
  1. Delta-to-absolute conversion for zone and building commands.
  2. Safety Validator clamps out-of-bound setpoints.
  3. Safety Validator rejects deadband violations.
  4. CO2 emergency override forces damper to 100%.
  5. Full pipeline: State -> Scores -> Events -> Agents ->
     Orchestrator (fallback) -> ActionSet.

Run with:
    uv run python -c "import sys; sys.path.insert(0,'.');
        from tests.test_phase6_smoke import main; main()"
"""

import sys
from datetime import datetime, timezone

from src.action import (
    ActionSet,
    CommandStatus,
    build_action_set,
)
from src.action.models import (
    CO2_EMERGENCY_PPM,
    COOLING_SETPOINT_MAX,
    COOLING_SETPOINT_MIN,
    DAMPER_MAX,
    FAN_SPEED_MIN,
)
from src.agents import ContextPriorities, run_all_agents
from src.engine.features import extract_building_features
from src.engine.scores import score_building
from src.events import generate_events
from src.orchestrator.orchestrator import _rule_based_fallback
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
# Shared demo state
# ---------------------------------------------------------------------------


def _demo_state() -> BuildingState:
    return BuildingState(
        timestamp=datetime(2025, 8, 15, 14, 0, 0, tzinfo=timezone.utc),
        weather=WeatherState(
            dry_bulb_temperature=32.0, relative_humidity=65.0,
            direct_solar_irradiance=750.0, wind_speed=3.5,
        ),
        zones={
            "conf_room_exec": ZoneState(
                zone_id="conf_room_exec", temperature=26.2,
                relative_humidity=58.0, pmv=1.1, co2_ppm=1150.0,
                occupancy_count=15, cooling_setpoint=24.0,
                heating_setpoint=20.0,
            ),
            "open_office_201": ZoneState(
                zone_id="open_office_201", temperature=23.5,
                relative_humidity=52.0, pmv=0.2, co2_ppm=680.0,
                occupancy_count=8, cooling_setpoint=23.0,
                heating_setpoint=20.0,
            ),
            "board_room": ZoneState(
                zone_id="board_room", temperature=22.0,
                relative_humidity=50.0, pmv=0.0, co2_ppm=450.0,
                occupancy_count=0, cooling_setpoint=23.0,
                heating_setpoint=20.0,
            ),
        },
        equipment={
            "AHU_01": EquipmentState(
                equipment_id="AHU_01", status=EquipmentStatus.ON,
                current_power_kw=52.0, fan_speed_pct=75.0,
                damper_position_pct=60.0,
            )
        },
        energy_price=0.45,
        carbon_intensity=430.0,
    )


def _run_to_plan(state: BuildingState):  # type: ignore[return]
    scores = score_building(
        state, total_power_kw=52.0, baseline_power_kw=35.0,
        avg_energy_price=0.22, carbon_min=150.0, carbon_max=550.0,
    )
    features = extract_building_features(
        state, avg_energy_price=0.22, total_power_kw=52.0,
        baseline_power_kw=35.0, carbon_min=150.0, carbon_max=550.0,
    )
    events = generate_events(state, scores, features)
    context = ContextPriorities.from_building_state(
        state, avg_tariff=0.22
    )
    recs = run_all_agents(state, scores, events, context)
    plan = _rule_based_fallback(
        state, scores, events, context, recs,
        reason="test fallback"
    )
    return scores, events, context, recs, plan


# ---------------------------------------------------------------------------
# T1: Delta-to-absolute conversion
# ---------------------------------------------------------------------------


def test_delta_to_absolute() -> list[bool]:
    """Action Engine converts deltas to absolute values correctly."""
    print("\n-- T1: Delta-to-Absolute Conversion ----------")
    state = _demo_state()
    _, _, _, _, plan = _run_to_plan(state)
    action_set = build_action_set(plan, state)

    results: list[bool] = []

    cr_cmd = action_set.zone_commands.get("conf_room_exec")
    results.append(_p("conf_room_exec command present", cr_cmd is not None))

    if cr_cmd:
        # Plan: cool_delta=-1.0, current=24.0 -> absolute=23.0
        cs = cr_cmd.cooling_setpoint
        results.append(
            _p(
                f"Cooling setpoint: 24.0 + (-1.0) = 23.0 (got {cs})",
                cs is not None and abs(cs - 23.0) < 0.01,
            )
        )
        # Plan: damper_delta=+20.0, current=60.0 -> absolute=80.0
        dmp = cr_cmd.damper_position_pct
        results.append(
            _p(
                f"Damper: 60.0 + 20.0 = 80.0 (got {dmp})",
                dmp is not None and abs(dmp - 80.0) < 0.01,
            )
        )
        # Plan: fan_delta=+15.0, current=75.0 -> absolute=90.0
        fan = cr_cmd.fan_speed_pct
        results.append(
            _p(
                f"Fan speed: 75.0 + 15.0 = 90.0 (got {fan})",
                fan is not None and abs(fan - 90.0) < 0.01,
            )
        )

    br_cmd = action_set.zone_commands.get("board_room")
    results.append(_p("board_room command present", br_cmd is not None))
    if br_cmd:
        # Plan: setpoint_delta=+2.0, current=23.0 -> 25.0
        cs = br_cmd.cooling_setpoint
        results.append(
            _p(
                f"board_room setpoint: 23.0 + 2.0 = 25.0 (got {cs})",
                cs is not None and abs(cs - 25.0) < 0.01,
            )
        )

    return results


# ---------------------------------------------------------------------------
# T2: Safety Validator clamps out-of-bound commands
# ---------------------------------------------------------------------------


def test_safety_clamp() -> list[bool]:
    """Safety Validator clamps values to physical bounds."""
    print("\n-- T2: Safety Validator Clamps ---------------")
    from src.action.engine import _validate_zone_command
    from src.action.models import ZoneSetpointCommand

    state = _demo_state()
    results: list[bool] = []

    # Cooling setpoint above max (26°C) -> should be clamped
    cmd = ZoneSetpointCommand(
        zone_id="conf_room_exec",
        cooling_setpoint=30.0,  # > COOLING_SETPOINT_MAX=26
    )
    _validate_zone_command(cmd, "conf_room_exec", state)
    results.append(
        _p(
            f"Cooling 30.0 clamped to {COOLING_SETPOINT_MAX} "
            f"(got {cmd.cooling_setpoint})",
            cmd.cooling_setpoint is not None
            and abs(cmd.cooling_setpoint - COOLING_SETPOINT_MAX) < 0.01,
        )
    )
    results.append(
        _p("Status=CLAMPED", cmd.status == CommandStatus.CLAMPED)
    )

    # Fan speed below min (10%) -> should be clamped
    cmd2 = ZoneSetpointCommand(
        zone_id="open_office_201", fan_speed_pct=5.0
    )
    _validate_zone_command(cmd2, "open_office_201", state)
    results.append(
        _p(
            f"Fan 5.0 clamped to {FAN_SPEED_MIN} "
            f"(got {cmd2.fan_speed_pct})",
            cmd2.fan_speed_pct is not None
            and abs(cmd2.fan_speed_pct - FAN_SPEED_MIN) < 0.01,
        )
    )
    results.append(
        _p("Status=CLAMPED", cmd2.status == CommandStatus.CLAMPED)
    )

    return results


# ---------------------------------------------------------------------------
# T3: Safety Validator rejects deadband violations
# ---------------------------------------------------------------------------


def test_safety_deadband_rejection() -> list[bool]:
    """Safety Validator rejects cooling setpoints that violate deadband."""
    print("\n-- T3: Deadband Violation Rejection ----------")
    from src.action.engine import _validate_zone_command
    from src.action.models import ZoneSetpointCommand

    # Use a state where current cooling=21.5 so delta=-1.0 is
    # within the 2.0°C rate-limit, but absolute 20.5°C is only
    # 0.5°C above heating=20.0, violating the 1°C deadband.
    tight_state = BuildingState(
        timestamp=datetime(2025, 8, 15, 14, 0, tzinfo=timezone.utc),
        weather=WeatherState(
            dry_bulb_temperature=22.0, relative_humidity=50.0,
            direct_solar_irradiance=200.0, wind_speed=2.0,
        ),
        zones={
            "tight_zone": ZoneState(
                zone_id="tight_zone", temperature=22.0,
                relative_humidity=50.0, pmv=0.1, co2_ppm=500.0,
                occupancy_count=2,
                cooling_setpoint=21.5,  # current — delta -1.0 = 20.5
                heating_setpoint=20.0,
            )
        },
        equipment={},
        energy_price=0.22,
        carbon_intensity=250.0,
    )

    results: list[bool] = []

    # Setpoint 20.5 = delta -1.0 (within rate limit).
    # Gap to heating: 20.5 - 20.0 = 0.5 < DEADBAND_MIN=1.0 -> REJECT.
    cmd = ZoneSetpointCommand(
        zone_id="tight_zone", cooling_setpoint=20.5
    )
    _validate_zone_command(cmd, "tight_zone", tight_state)
    results.append(
        _p(
            "Deadband violation rejected (cool=20.5, heat=20.0, gap=0.5)",
            cmd.status == CommandStatus.REJECTED,
        )
    )
    results.append(
        _p(
            "Cooling setpoint nullified on rejection",
            cmd.cooling_setpoint is None,
        )
    )
    has_note = any("REJECTED" in n for n in cmd.validator_notes)
    results.append(_p("Validator notes mention REJECTED", has_note))

    # Valid setpoint: 22.5°C (gap = 2.5 > 1.0 deadband, delta=+1.0 within rate-limit)
    cmd_ok = ZoneSetpointCommand(
        zone_id="tight_zone", cooling_setpoint=22.5
    )
    _validate_zone_command(cmd_ok, "tight_zone", tight_state)
    results.append(
        _p("Valid setpoint 22.5 APPROVED", cmd_ok.status == CommandStatus.APPROVED)
    )

    return results


# ---------------------------------------------------------------------------
# T4: CO2 emergency override
# ---------------------------------------------------------------------------


def test_co2_emergency_override() -> list[bool]:
    """CO2 >= 1200 PPM forces damper to 100% regardless of plan."""
    print("\n-- T4: CO2 Emergency Override ----------------")
    from src.action.engine import _validate_zone_command
    from src.action.models import ZoneSetpointCommand

    # Build a state with CO2 at emergency level
    emergency_state = BuildingState(
        timestamp=datetime(2025, 8, 15, 14, 0, tzinfo=timezone.utc),
        weather=WeatherState(
            dry_bulb_temperature=32.0, relative_humidity=65.0,
            direct_solar_irradiance=750.0, wind_speed=3.5,
        ),
        zones={
            "danger_zone": ZoneState(
                zone_id="danger_zone", temperature=25.0,
                relative_humidity=55.0, pmv=0.5,
                co2_ppm=CO2_EMERGENCY_PPM + 50.0,  # 1250 PPM
                occupancy_count=12, cooling_setpoint=24.0,
                heating_setpoint=20.0,
            )
        },
        equipment={},
        energy_price=0.22,
        carbon_intensity=300.0,
    )

    results: list[bool] = []

    # Plan specifies damper=50% but CO2 emergency should force 100%
    cmd = ZoneSetpointCommand(
        zone_id="danger_zone", damper_position_pct=50.0
    )
    _validate_zone_command(cmd, "danger_zone", emergency_state)

    results.append(
        _p(
            f"Damper forced to {DAMPER_MAX}% (CO2 emergency override)",
            cmd.damper_position_pct is not None
            and abs(cmd.damper_position_pct - DAMPER_MAX) < 0.01,
        )
    )
    results.append(
        _p("Status=CLAMPED (forced override)", cmd.status == CommandStatus.CLAMPED)
    )
    has_emergency_note = any(
        "emergency" in n.lower() for n in cmd.validator_notes
    )
    results.append(
        _p("Emergency note in validator_notes", has_emergency_note)
    )

    return results


# ---------------------------------------------------------------------------
# T5: Full pipeline end-to-end
# ---------------------------------------------------------------------------


def test_full_pipeline() -> list[bool]:
    """Full pipeline State->Scores->Events->Agents->Plan->ActionSet."""
    print("\n-- T5: Full Pipeline End-to-End --------------")
    state = _demo_state()
    _, _, _, _, plan = _run_to_plan(state)
    action_set = build_action_set(plan, state)

    results: list[bool] = []
    results.append(
        _p("ActionSet produced", isinstance(action_set, ActionSet))
    )
    results.append(
        _p(
            "No REJECTED commands in demo scenario",
            not action_set.has_rejections,
        )
    )

    print(f"\n  has_rejections : {action_set.has_rejections}")
    print(f"  has_clamps     : {action_set.has_clamps}")
    print()
    print("  Zone commands:")
    for zid, cmd in action_set.zone_commands.items():
        print(
            f"    {zid:<20} "
            f"cool={cmd.cooling_setpoint}  "
            f"damper={cmd.damper_position_pct}  "
            f"fan={cmd.fan_speed_pct}  "
            f"[{cmd.status.value}]"
        )
        for note in cmd.validator_notes:
            print(f"      -> {note}")
    if action_set.building_command:
        bc = action_set.building_command
        print(
            f"  Building cmd  : fan={bc.fan_speed_pct}  "
            f"[{bc.status.value}]"
        )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all Phase 6 tests and report summary."""
    print(SEP)
    print("  Phase 6 -- Action Engine & Safety Validator")
    print(SEP)

    all_results: list[bool] = []
    all_results += test_delta_to_absolute()
    all_results += test_safety_clamp()
    all_results += test_safety_deadband_rejection()
    all_results += test_co2_emergency_override()
    all_results += test_full_pipeline()

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
