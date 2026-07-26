"""Phase 5 smoke test -- LLM Orchestrator.

Tests the orchestrator in three modes:
  1. Rule-based fallback (no API key, most important path for demo).
  2. D4 hard-constraint acceptance test (LLM plan that violates
     constraints is rejected and substituted with fallback).
  3. D2 cross-scope fan-speed conflict resolution.
  4. Prompt builder correctness (hard constraints in system prompt).

No real LLM call is made — the acceptance test stubs out the LLM
response at the orchestrator boundary.

Run with:
    uv run python -c "import sys; sys.path.insert(0,'.');
        from tests.test_phase5_smoke import main; main()"
"""

import json
import sys
from datetime import datetime, timezone
from unittest.mock import patch

from src.agents import ContextPriorities, run_all_agents
from src.agents.models import ActionType
from src.engine.features import extract_building_features
from src.engine.scores import score_building
from src.events import generate_events
from src.orchestrator.models import (
    BuildingActionPlan,
    OrchestratorPlan,
    ZoneActionPlan,
)
from src.orchestrator.orchestrator import (
    _rule_based_fallback,
    _verify_hard_constraints,
    orchestrate,
)
from src.orchestrator.prompts import (
    build_system_prompt,
    build_user_prompt,
)
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
    if not ok:
        print(f"        FAILED: {label}")
    return ok


# ---------------------------------------------------------------------------
# Shared demo state
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
    recs = run_all_agents(state, scores, events, context)
    return scores, events, context, recs


# ---------------------------------------------------------------------------
# T1: Rule-based fallback (no API key)
# ---------------------------------------------------------------------------


def test_rule_based_fallback() -> list[bool]:
    """Fallback produces a valid plan respecting D2 and D4."""
    print("\n-- T1: Rule-Based Fallback -------------------")
    state = _demo_state()
    scores, events, context, recs = _run_pipeline(state)

    plan = _rule_based_fallback(
        state, scores, events, context, recs,
        reason="test: no API key",
    )

    results: list[bool] = []

    results.append(
        _p("resolved_by=rule_based_fallback",
           plan.resolved_by == "rule_based_fallback")
    )

    # conf_room_exec must have a plan
    cr_plan = plan.zone_plans.get("conf_room_exec")
    results.append(_p("conf_room_exec has a zone plan", cr_plan is not None))

    if cr_plan:
        # D4-a: cooling_setpoint_delta must be <= 0 (cool or no change)
        cs = cr_plan.cooling_setpoint_delta
        results.append(
            _p(
                "D4-a: conf_room cooling delta <= 0 "
                f"(got {cs})",
                cs is None or cs <= 0,
            )
        )
        # D4-b: damper must be >= 0 (open or no change)
        dmp = cr_plan.damper_delta_pct
        results.append(
            _p(
                "D4-b: conf_room damper delta >= 0 "
                f"(got {dmp})",
                dmp is None or dmp >= 0,
            )
        )
        # Fan speed should be increased for high-density zone (D2)
        fan = cr_plan.fan_speed_delta_pct
        results.append(
            _p(
                "D2: conf_room fan_speed_delta > 0 "
                f"(zone-scope override, got {fan})",
                fan is not None and fan > 0,
            )
        )

    # board_room must have RELAX_SETPOINT effect (cooling delta > 0)
    br_plan = plan.zone_plans.get("board_room")
    results.append(_p("board_room has a zone plan", br_plan is not None))
    if br_plan:
        cs = br_plan.cooling_setpoint_delta
        results.append(
            _p(
                f"board_room setpoint relaxed (delta>0, got {cs})",
                cs is not None and cs > 0,
            )
        )

    # Building-wide fan: D2 override should suppress DECREASE
    bplan = plan.building_plan
    results.append(_p("building_plan present", bplan is not None))
    if bplan:
        results.append(
            _p(
                "D2: building fan_speed_delta suppressed "
                f"(got {bplan.fan_speed_delta_pct})",
                bplan.fan_speed_delta_pct is None,
            )
        )

    # D2 conflict resolution recorded
    has_d2_note = any(
        "D2" in cr for cr in plan.conflict_resolutions
    )
    results.append(
        _p("D2 conflict resolution recorded in audit trace", has_d2_note)
    )

    return results


# ---------------------------------------------------------------------------
# T2: D4 Hard-constraint acceptance test
# ---------------------------------------------------------------------------


def test_d4_constraint_enforcement() -> list[bool]:
    """LLM plan violating D4 is rejected and fallback substituted."""
    print("\n-- T2: D4 Hard-Constraint Enforcement --------")
    state = _demo_state()
    scores, events, context, recs = _run_pipeline(state)

    results: list[bool] = []

    # Build a deliberately violating LLM response:
    # Constraint D4-a: raises cooling setpoint in conf_room_exec
    # Constraint D4-b: reduces ventilation in conf_room_exec
    bad_llm_json = json.dumps({
        "zone_plans": {
            "conf_room_exec": {
                "cooling_setpoint_delta": 2.0,   # VIOLATION: raises setpoint
                "damper_delta_pct": -15.0,        # VIOLATION: reduces vent
                "fan_speed_delta_pct": None,
                "rationale": "LLM bad plan for test."
            },
            "board_room": {
                "cooling_setpoint_delta": 2.0,
                "damper_delta_pct": None,
                "fan_speed_delta_pct": None,
                "rationale": "Empty zone setpoint relaxed."
            }
        },
        "building_plan": {
            "fan_speed_delta_pct": -10.0,
            "rationale": "Reduce building fan for peak demand."
        },
        "active_policy_rules": [],
        "conflict_resolutions": [],
        "overall_rationale": "Deliberately bad plan for test."
    })

    # Mock call_llm to return the bad JSON
    with patch(
        "src.orchestrator.orchestrator.call_llm",
        return_value=bad_llm_json,
    ):
        plan = orchestrate(
            state, scores, events, context, recs
        )

    results.append(
        _p(
            "Bad LLM plan rejected -> resolved_by=rule_based_fallback",
            plan.resolved_by == "rule_based_fallback",
        )
    )

    # Verify the substituted fallback plan is D4-compliant
    cr_plan = plan.zone_plans.get("conf_room_exec")
    if cr_plan:
        cs = cr_plan.cooling_setpoint_delta
        results.append(
            _p(
                f"Fallback: conf_room cooling delta <= 0 (got {cs})",
                cs is None or cs <= 0,
            )
        )
        dmp = cr_plan.damper_delta_pct
        results.append(
            _p(
                f"Fallback: conf_room damper delta >= 0 (got {dmp})",
                dmp is None or dmp >= 0,
            )
        )
    else:
        results.append(_p("conf_room_exec plan present after fallback", False))
        results.append(_p("damper delta check skipped", False))

    return results


# ---------------------------------------------------------------------------
# T3: D4 constraint verifier identifies violations correctly
# ---------------------------------------------------------------------------


def test_d4_verifier() -> list[bool]:
    """_verify_hard_constraints catches all three violation types."""
    print("\n-- T3: Constraint Verifier -------------------")
    state = _demo_state()
    scores, events, context, recs = _run_pipeline(state)

    results: list[bool] = []

    # Plan with D4-a + D4-b violations
    bad_plan = OrchestratorPlan(
        zone_plans={
            "conf_room_exec": ZoneActionPlan(
                zone_id="conf_room_exec",
                cooling_setpoint_delta=1.5,   # D4-a violation
                damper_delta_pct=-10.0,        # D4-b violation
                fan_speed_delta_pct=None,
                rationale="test",
            )
        },
        building_plan=None,
        overall_rationale="test",
    )

    violations = _verify_hard_constraints(
        bad_plan, state, scores, context
    )
    results.append(
        _p("D4-a violation detected (setpoint raised)", any(
            "D4-A" in v for v in violations
        ))
    )
    results.append(
        _p("D4-b violation detected (vent reduced)", any(
            "D4-B" in v for v in violations
        ))
    )

    # Clean plan — no violations
    good_plan = OrchestratorPlan(
        zone_plans={
            "conf_room_exec": ZoneActionPlan(
                zone_id="conf_room_exec",
                cooling_setpoint_delta=-1.0,   # correct: cool more
                damper_delta_pct=20.0,          # correct: open damper
                fan_speed_delta_pct=15.0,
                rationale="test",
            )
        },
        building_plan=None,
        overall_rationale="test",
    )
    clean_violations = _verify_hard_constraints(
        good_plan, state, scores, context
    )
    results.append(
        _p(
            "No violations for compliant plan "
            f"(got {len(clean_violations)})",
            len(clean_violations) == 0,
        )
    )

    return results


# ---------------------------------------------------------------------------
# T4: Prompt builder encodes hard constraints
# ---------------------------------------------------------------------------


def test_prompt_builder() -> list[bool]:
    """System prompt contains hard constraint blocks (D4)."""
    print("\n-- T4: Prompt Builder ------------------------")
    state = _demo_state()
    scores, events, context, recs = _run_pipeline(state)

    sys_prompt = build_system_prompt(state, scores, context)
    usr_prompt = build_user_prompt(
        state, scores, events, context, recs
    )

    results: list[bool] = []
    results.append(
        _p("HARD CONSTRAINTS header in system prompt",
           "HARD CONSTRAINTS" in sys_prompt)
    )
    results.append(
        _p("CONSTRAINT-A in system prompt (no setpoint raise)",
           "CONSTRAINT-A" in sys_prompt)
    )
    results.append(
        _p("CONSTRAINT-B in system prompt (no vent reduce)",
           "CONSTRAINT-B" in sys_prompt)
    )
    results.append(
        _p("CONSTRAINT-C in system prompt (no full shed)",
           "CONSTRAINT-C" in sys_prompt)
    )
    results.append(
        _p("D2 rule in system prompt",
           "D2" in sys_prompt)
    )
    results.append(
        _p("D3 tiebreak in system prompt",
           "D3" in sys_prompt)
    )
    results.append(
        _p("User prompt contains zone scores",
           "conf_room_exec" in usr_prompt)
    )
    results.append(
        _p("User prompt contains recommendations",
           "AGENT RECOMMENDATIONS" in usr_prompt)
    )
    results.append(
        _p("User prompt contains events",
           "DETECTED EVENTS" in usr_prompt)
    )
    results.append(
        _p("User prompt contains context priorities",
           "CONTEXT PRIORITIES" in usr_prompt)
    )

    # Print prompt lengths for judge demo reference
    print(
        f"\n  System prompt : {len(sys_prompt):,} chars"
    )
    print(f"  User prompt   : {len(usr_prompt):,} chars")

    return results


# ---------------------------------------------------------------------------
# T5: Full pipeline (fallback) end-to-end
# ---------------------------------------------------------------------------


def test_full_pipeline_fallback() -> list[bool]:
    """Full pipeline State->Scores->Events->Agents->Orchestrator."""
    print("\n-- T5: End-to-End (Fallback Path) ------------")
    state = _demo_state()
    scores, events, context, recs = _run_pipeline(state)

    # orchestrate() with no API key -> fallback
    plan = orchestrate(state, scores, events, context, recs)

    results: list[bool] = []
    results.append(
        _p("Plan produced (any resolver)",
           isinstance(plan, OrchestratorPlan))
    )
    results.append(
        _p("overall_rationale non-empty",
           len(plan.overall_rationale) > 0)
    )
    # All zone plans reference valid zones
    for zid in plan.zone_plans:
        results.append(
            _p(
                f"zone_plan key '{zid}' is a known zone",
                zid in state.zones,
            )
        )

    print(f"\n  Resolved by   : {plan.resolved_by}")
    print(f"  Policy rules  : {len(plan.active_policy_rules)}")
    print(f"  Conflicts     : {len(plan.conflict_resolutions)}")
    print(f"\n  Zone plans:")
    for zid, zp in plan.zone_plans.items():
        print(
            f"    {zid:<20} "
            f"cool_delta={zp.cooling_setpoint_delta}  "
            f"damper={zp.damper_delta_pct}  "
            f"fan={zp.fan_speed_delta_pct}"
        )
    if plan.building_plan:
        print(
            f"  Building fan  : {plan.building_plan.fan_speed_delta_pct}"
        )
    print(f"\n  Conflict resolutions:")
    for cr in plan.conflict_resolutions:
        print(f"    - {cr}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all Phase 5 tests and report summary."""
    print(SEP)
    print("  Phase 5 -- LLM Orchestrator Smoke Test")
    print(SEP)

    all_results: list[bool] = []
    all_results += test_rule_based_fallback()
    all_results += test_d4_constraint_enforcement()
    all_results += test_d4_verifier()
    all_results += test_prompt_builder()
    all_results += test_full_pipeline_fallback()

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
