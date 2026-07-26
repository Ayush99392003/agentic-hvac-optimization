"""Phase 7 smoke test -- Simulation & Feedback Loop.

Verifies:
  T1: ReplayedSimulator advances state with first-order dynamics.
  T2: CO2 decreases when ventilation is active (vent > damper min).
  T3: CO2 rises when ventilation is off (damper below threshold).
  T4: compute_feedback() produces a valid FeedbackReport.
  T5: advance_state() folds sim result into a new BuildingState.
  T6: Multi-cycle loop: 3 consecutive cycles converge toward targets.
  T7: get_simulator() returns ReplayedSimulator by default.
  T8: EnergyPlusSimulator raises NotImplementedError (stub guard).

Run with:
    uv run python -c "import sys; sys.path.insert(0,'.');
        from tests.test_phase7_smoke import main; main()"
"""

import os
import sys
from datetime import datetime, timezone

from src.action import build_action_set
from src.agents import ContextPriorities, run_all_agents
from src.engine.features import extract_building_features
from src.engine.scores import score_building
from src.events import generate_events
from src.orchestrator.orchestrator import _rule_based_fallback
from src.simulation import (
    EnergyPlusSimulator,
    FeedbackReport,
    ReplayedSimulator,
    SimStepResult,
    advance_state,
    compute_feedback,
    get_simulator,
)
from src.state import (
    BuildingState,
    EquipmentState,
    EquipmentStatus,
    WeatherState,
    ZoneState,
)

SEP = "=" * 60
PIPELINE_KWARGS = dict(
    avg_energy_price=0.22,
    total_power_kw=52.0,
    baseline_power_kw=35.0,
    carbon_min=150.0,
    carbon_max=550.0,
)


def _p(label: str, ok: bool) -> bool:
    print(f"  {'[PASS]' if ok else '[FAIL]'}  {label}")
    return ok


def _demo_state() -> BuildingState:
    return BuildingState(
        timestamp=datetime(2025, 8, 15, 14, 0, tzinfo=timezone.utc),
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


def _run_to_action_set(state: BuildingState):  # type: ignore[return]
    scores = score_building(state, **PIPELINE_KWARGS)
    features = extract_building_features(state, **PIPELINE_KWARGS)
    events = generate_events(state, scores, features)
    context = ContextPriorities.from_building_state(
        state, avg_tariff=0.22
    )
    recs = run_all_agents(state, scores, events, context)
    plan = _rule_based_fallback(
        state, scores, events, context, recs, reason="test"
    )
    action_set = build_action_set(plan, state, context)
    return scores, events, context, recs, plan, action_set


# ---------------------------------------------------------------------------
# T1: ReplayedSimulator advances state
# ---------------------------------------------------------------------------


def test_replayed_sim_advances_state() -> list[bool]:
    """ReplayedSimulator reduces temperature toward setpoint."""
    print("\n-- T1: ReplayedSimulator Advances State ------")
    state = _demo_state()
    _, _, _, _, _, action_set = _run_to_action_set(state)
    sim = ReplayedSimulator()
    result = sim.step(state, action_set)

    results: list[bool] = []
    results.append(
        _p("SimStepResult returned", isinstance(result, SimStepResult))
    )
    results.append(
        _p(
            "simulator_id=replayed",
            result.simulator_id == "replayed",
        )
    )
    # conf_room was 26.2°C with setpoint moving toward 23.0°C
    cr = result.zone_results.get("conf_room_exec")
    results.append(
        _p("conf_room_exec zone result present", cr is not None)
    )
    if cr:
        results.append(
            _p(
                f"Temperature decreased toward setpoint "
                f"({state.zones['conf_room_exec'].temperature:.1f} "
                f"-> {cr.temperature:.2f})",
                cr.temperature
                < state.zones["conf_room_exec"].temperature,
            )
        )
    results.append(
        _p(
            f"total_power_kw > 0 (got {result.total_power_kw})",
            result.total_power_kw > 0,
        )
    )
    return results


# ---------------------------------------------------------------------------
# T2: CO2 decreases with active ventilation
# ---------------------------------------------------------------------------


def test_co2_decreases_with_vent() -> list[bool]:
    """CO2 falls when damper >= 30% and fan >= 40%."""
    print("\n-- T2: CO2 Decreases With Ventilation --------")
    state = _demo_state()
    _, _, _, _, _, action_set = _run_to_action_set(state)
    sim = ReplayedSimulator()
    result = sim.step(state, action_set)

    results: list[bool] = []
    cr = result.zone_results.get("conf_room_exec")
    if cr:
        pre = state.zones["conf_room_exec"].co2_ppm
        post = cr.co2_ppm
        # With damper=80%, fan=90% -> vent active -> CO2 should drop
        results.append(
            _p(
                f"conf_room CO2 decreased with vent active "
                f"({pre:.0f} -> {post:.0f} PPM)",
                post < pre,
            )
        )
    else:
        results.append(_p("conf_room result present", False))
    return results


# ---------------------------------------------------------------------------
# T3: CO2 rises without ventilation
# ---------------------------------------------------------------------------


def test_co2_rises_without_vent() -> list[bool]:
    """CO2 rises when damper is below effective threshold."""
    print("\n-- T3: CO2 Rises Without Ventilation ---------")
    from src.action.models import ActionSet

    state = _demo_state()
    # Provide empty action set (no damper change, low damper state)
    closed_state = BuildingState(
        timestamp=state.timestamp,
        weather=state.weather,
        zones=state.zones,
        equipment={
            "AHU_01": EquipmentState(
                equipment_id="AHU_01",
                status=EquipmentStatus.ON,
                current_power_kw=20.0,
                fan_speed_pct=20.0,   # below MIN_EFFECTIVE_FAN=40
                damper_position_pct=10.0,  # below MIN_EFFECTIVE_DAMPER=30
            )
        },
        energy_price=state.energy_price,
        carbon_intensity=state.carbon_intensity,
    )
    sim = ReplayedSimulator()
    result = sim.step(closed_state, ActionSet())

    results: list[bool] = []
    cr = result.zone_results.get("conf_room_exec")
    if cr:
        pre = closed_state.zones["conf_room_exec"].co2_ppm
        post = cr.co2_ppm
        results.append(
            _p(
                f"conf_room CO2 rises without vent "
                f"({pre:.0f} -> {post:.0f} PPM)",
                post > pre,
            )
        )
    else:
        results.append(_p("conf_room result present", False))
    return results


# ---------------------------------------------------------------------------
# T4: FeedbackReport computation
# ---------------------------------------------------------------------------


def test_feedback_report() -> list[bool]:
    """compute_feedback() produces a valid FeedbackReport."""
    print("\n-- T4: FeedbackReport Computation ------------")
    state = _demo_state()
    scores, events, context, recs, plan, action_set = (
        _run_to_action_set(state)
    )
    sim = ReplayedSimulator()
    sim_result = sim.step(state, action_set)

    report = compute_feedback(
        cycle_index=0,
        state=state,
        plan=plan,
        sim_result=sim_result,
        avg_energy_price=0.22,
        baseline_power_kw=35.0,
        carbon_min=150.0,
        carbon_max=550.0,
    )

    results: list[bool] = []
    results.append(
        _p("FeedbackReport returned", isinstance(report, FeedbackReport))
    )
    results.append(
        _p("cycle_index=0", report.cycle_index == 0)
    )
    results.append(
        _p(
            "zone_deltas contains conf_room_exec",
            "conf_room_exec" in report.zone_deltas,
        )
    )
    cr_d = report.zone_deltas.get("conf_room_exec")
    if cr_d:
        results.append(
            _p(
                f"conf_room comfort improved "
                f"(delta={cr_d.comfort_score_delta:+.1f})",
                cr_d.comfort_score_delta > 0,
            )
        )
        results.append(
            _p(
                "temp_error is small (|err| < 2°C)",
                abs(cr_d.temp_error) < 2.0,
            )
        )
    results.append(
        _p(
            "overall_summary non-empty",
            len(report.overall_summary) > 0,
        )
    )
    print(f"\n  Summary : {report.overall_summary}")
    return results


# ---------------------------------------------------------------------------
# T5: advance_state() closes the loop
# ---------------------------------------------------------------------------


def test_advance_state() -> list[bool]:
    """advance_state() produces a new BuildingState from sim result."""
    print("\n-- T5: advance_state() Closes the Loop ------")
    state = _demo_state()
    _, _, _, _, _, action_set = _run_to_action_set(state)
    sim = ReplayedSimulator()
    sim_result = sim.step(state, action_set)

    next_state = advance_state(state, sim_result)

    results: list[bool] = []
    results.append(
        _p(
            "next_state timestamp advanced 15 min",
            next_state.timestamp > state.timestamp,
        )
    )
    results.append(
        _p(
            "conf_room temperature updated in next_state",
            next_state.zones["conf_room_exec"].temperature
            != state.zones["conf_room_exec"].temperature,
        )
    )
    results.append(
        _p(
            "conf_room cooling_setpoint updated to achieved value",
            next_state.zones["conf_room_exec"].cooling_setpoint
            == sim_result.zone_results["conf_room_exec"].cooling_setpoint_achieved,
        )
    )
    results.append(
        _p(
            "zone count preserved",
            len(next_state.zones) == len(state.zones),
        )
    )
    return results


# ---------------------------------------------------------------------------
# T6: Multi-cycle convergence
# ---------------------------------------------------------------------------


def test_multi_cycle_convergence() -> list[bool]:
    """3 cycles: conf_room temperature converges toward 23°C target."""
    print("\n-- T6: Multi-Cycle Convergence ---------------")
    state = _demo_state()
    sim = ReplayedSimulator()
    temps: list[float] = [
        state.zones["conf_room_exec"].temperature
    ]
    co2s: list[float] = [state.zones["conf_room_exec"].co2_ppm]

    for cycle in range(3):
        scores, events, context, recs, plan, action_set = (
            _run_to_action_set(state)
        )
        sim_result = sim.step(state, action_set)
        report = compute_feedback(
            cycle_index=cycle,
            state=state,
            plan=plan,
            sim_result=sim_result,
            avg_energy_price=0.22,
            baseline_power_kw=35.0,
            carbon_min=150.0,
            carbon_max=550.0,
        )
        state = advance_state(state, sim_result)
        temps.append(state.zones["conf_room_exec"].temperature)
        co2s.append(state.zones["conf_room_exec"].co2_ppm)
        print(
            f"  Cycle {cycle}: "
            f"T={temps[-1]:.2f}°C  "
            f"CO2={co2s[-1]:.0f}PPM  "
            f"power={sim_result.total_power_kw:.1f}kW  "
            f"savings={report.energy_savings_pct:+.1f}%"
        )

    results: list[bool] = []
    results.append(
        _p(
            "Temperature decreasing across 3 cycles (converging)",
            temps[-1] < temps[0],
        )
    )
    results.append(
        _p(
            "CO2 decreasing across 3 cycles (vent active)",
            co2s[-1] < co2s[0],
        )
    )
    return results


# ---------------------------------------------------------------------------
# T7: get_simulator() returns ReplayedSimulator by default
# ---------------------------------------------------------------------------


def test_get_simulator_default() -> list[bool]:
    """get_simulator() returns ReplayedSimulator with no env var."""
    print("\n-- T7: get_simulator() Default ---------------")
    os.environ.pop("ENERGYPLUS_IDF_PATH", None)
    sim = get_simulator()
    results: list[bool] = []
    results.append(
        _p(
            "Default backend is ReplayedSimulator",
            isinstance(sim, ReplayedSimulator),
        )
    )
    return results


# ---------------------------------------------------------------------------
# T8: EnergyPlusSimulator stub guard
# ---------------------------------------------------------------------------


def test_energyplus_stub() -> list[bool]:
    """EnergyPlusSimulator.step() raises NotImplementedError."""
    print("\n-- T8: EnergyPlusSimulator Stub Guard --------")
    state = _demo_state()
    _, _, _, _, _, action_set = _run_to_action_set(state)
    sim = EnergyPlusSimulator()

    results: list[bool] = []
    try:
        sim.step(state, action_set)
        results.append(_p("NotImplementedError raised", False))
    except NotImplementedError:
        results.append(_p("NotImplementedError raised correctly", True))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all Phase 7 tests and report summary."""
    print(SEP)
    print("  Phase 7 -- Simulation & Feedback Loop")
    print(SEP)

    all_results: list[bool] = []
    all_results += test_replayed_sim_advances_state()
    all_results += test_co2_decreases_with_vent()
    all_results += test_co2_rises_without_vent()
    all_results += test_feedback_report()
    all_results += test_advance_state()
    all_results += test_multi_cycle_convergence()
    all_results += test_get_simulator_default()
    all_results += test_energyplus_stub()

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
