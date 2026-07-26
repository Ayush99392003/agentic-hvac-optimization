"""Phase 2 smoke test — Feature Extractor & Score Engine.

Validates every formula against the worked examples from the
architecture document (Section 7) and the demo scenario
(Section 6.3).

Run with:
    uv run python -c "import sys; sys.path.insert(0,'.');
        from tests.test_phase2_smoke import main; main()"
"""

import sys
from datetime import datetime, timezone

from src.engine import (
    BuildingScoreReport,
    ZoneFeatureVector,
    extract_zone_features,
    score_building,
    score_zone,
)
from src.engine.features import (
    AirQualityFeature,
    ComfortFeature,
    EnergyFeature,
    extract_building_features,
)
from src.engine.scores import (
    _air_quality_score,
    _carbon_score,
    _comfort_score,
    _energy_score,
)
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


def check(
    label: str, got: float, expected: float, tol: float = 0.01
) -> bool:
    """Assert a numeric result is within tolerance and print result.

    Args:
        label: Human-readable test label.
        got: Computed value.
        expected: Expected value from architecture doc.
        tol: Absolute tolerance (default 0.01).

    Returns:
        True if within tolerance, False otherwise.
    """
    ok = abs(got - expected) <= tol
    tag = PASS if ok else FAIL
    print(f"  {tag}  {label}")
    print(f"        got={got:.4f}  expected={expected:.4f}")
    return ok


def test_comfort_score_formulas() -> list[bool]:
    """Verify Comfort Score formula (architecture §7.1).

    Returns:
        List of boolean pass/fail results.
    """
    print("\n-- Comfort Score (arch §7.1) -----------------")
    results = []

    # PMV = 0.0 -> 100 (perfect comfort)
    results.append(check("PMV=0.0  -> 100", _comfort_score(0.0), 100.0))
    # PMV = 0.3 -> 94
    results.append(check("PMV=+0.3 ->  94", _comfort_score(0.3), 94.0))
    # Boundary |PMV|=0.5: both branches must give 90
    results.append(
        check("PMV=+0.5 ->  90 (branch-1)", _comfort_score(0.5), 90.0)
    )
    results.append(
        check("PMV=-0.5 ->  90 (branch-1)", _comfort_score(-0.5), 90.0)
    )
    # PMV = +1.1 -> 54  (demo scenario conf_room_exec)
    results.append(
        check("PMV=+1.1 ->  54 (demo)", _comfort_score(1.1), 54.0)
    )
    # PMV = +3.0 -> max(0, 90-60*2.5)= max(0,-60) = 0
    results.append(
        check("PMV=+3.0 ->   0 (max penalty)", _comfort_score(3.0), 0.0)
    )
    return results


def test_air_quality_score_formulas() -> list[bool]:
    """Verify Air Quality Score formula (architecture §7.2).

    Returns:
        List of boolean pass/fail results.
    """
    print("\n-- Air Quality Score (arch §7.2) -------------")
    results = []

    # CO2 <= 600 -> 100
    results.append(
        check("CO2=500   -> 100", _air_quality_score(500.0), 100.0)
    )
    results.append(
        check("CO2=600   -> 100", _air_quality_score(600.0), 100.0)
    )
    # CO2 = 1150 PPM -> 100 - 55 = 45 (demo scenario)
    results.append(
        check(
            "CO2=1150  ->  45 (demo)", _air_quality_score(1150.0), 45.0
        )
    )
    # CO2 = 1600 -> 100-100 = 0 (floor at 0)
    results.append(
        check("CO2=1600  ->   0 (floor)", _air_quality_score(1600.0), 0.0)
    )
    return results


def test_energy_score_formulas() -> list[bool]:
    """Verify Energy Score formula (architecture §7.3).

    Returns:
        List of boolean pass/fail results.
    """
    print("\n-- Energy Score (arch §7.3) ------------------")
    results = []

    # Baseline case: ratio=1.0, price=1.0 -> 100
    results.append(
        check(
            "ratio=1.0, price=1.0 -> 100 (baseline)",
            _energy_score(1.0, 1.0),
            100.0,
        )
    )
    # Below baseline (under-consumption) -> 100 (no penalty)
    results.append(
        check(
            "ratio=0.8, price=1.5 -> 100 (no penalty below base)",
            _energy_score(0.8, 1.5),
            100.0,
        )
    )
    # ratio=1.35, price=2.0 -> 100 - 100*(0.35*2.0) = 30
    results.append(
        check(
            "ratio=1.35, price=2.0 -> 30 (demo peak tariff)",
            _energy_score(1.35, 2.0),
            30.0,
        )
    )
    # ratio=1.5, price=1.0 -> 100-50 = 50
    results.append(
        check(
            "ratio=1.5, price=1.0 -> 50",
            _energy_score(1.5, 1.0),
            50.0,
        )
    )
    return results


def test_carbon_score_formulas() -> list[bool]:
    """Verify Carbon Score formula (architecture §7.4).

    Returns:
        List of boolean pass/fail results.
    """
    print("\n-- Carbon Score (arch §7.4) ------------------")
    results = []

    # At minimum -> 100
    results.append(
        check(
            "C_curr=C_min -> 100 (cleanest)",
            _carbon_score(120.0, 120.0, 520.0),
            100.0,
        )
    )
    # At maximum -> 0
    results.append(
        check(
            "C_curr=C_max -> 0 (dirtiest)",
            _carbon_score(520.0, 120.0, 520.0),
            0.0,
        )
    )
    # C_curr=450, window=[150,550] -> 100-100*(300/400) = 25
    results.append(
        check(
            "C=450, [150,550] -> 25 (arch example)",
            _carbon_score(450.0, 150.0, 550.0),
            25.0,
        )
    )
    # Demo scenario: C=430, estimate window [150,550]
    results.append(
        check(
            "C=430, [150,550] -> 30 (demo)",
            _carbon_score(430.0, 150.0, 550.0),
            30.0,
        )
    )
    return results


def test_feature_extraction() -> list[bool]:
    """Verify feature labels for the demo zone scenario.

    Returns:
        List of boolean pass/fail results.
    """
    print("\n-- Feature Extraction (demo scenario) --------")
    results = []

    conf_room = ZoneState(
        zone_id="conf_room_exec",
        temperature=26.2,
        relative_humidity=58.0,
        pmv=1.1,
        co2_ppm=1150.0,
        occupancy_count=15,
        cooling_setpoint=24.0,
        heating_setpoint=20.0,
    )

    fv: ZoneFeatureVector = extract_zone_features(conf_room)

    ok1 = fv.comfort == ComfortFeature.HOT
    print(f"  {'[PASS]' if ok1 else '[FAIL]'}  comfort=HOT  (PMV=1.1)")
    results.append(ok1)

    ok2 = fv.air_quality == AirQualityFeature.DEGRADED
    print(
        f"  {'[PASS]' if ok2 else '[FAIL]'}  "
        f"air_quality=DEGRADED  (CO2=1150)"
    )
    results.append(ok2)

    ok3 = fv.is_high_density
    print(f"  {'[PASS]' if ok3 else '[FAIL]'}  is_high_density=True (15 occ)")
    results.append(ok3)

    return results


def test_full_pipeline() -> list[bool]:
    """Run the full Phase 2 pipeline on the demo BuildingState.

    Returns:
        List of boolean pass/fail results.
    """
    print("\n-- Full Pipeline (BuildingState -> ScoreReport) -")
    results = []

    weather = WeatherState(
        dry_bulb_temperature=32.0,
        relative_humidity=65.0,
        direct_solar_irradiance=750.0,
        wind_speed=3.5,
    )
    conf_room = ZoneState(
        zone_id="conf_room_exec",
        temperature=26.2,
        relative_humidity=58.0,
        pmv=1.1,
        co2_ppm=1150.0,
        occupancy_count=15,
        cooling_setpoint=24.0,
        heating_setpoint=20.0,
    )
    open_office = ZoneState(
        zone_id="open_office_201",
        temperature=23.5,
        relative_humidity=52.0,
        pmv=0.2,
        co2_ppm=680.0,
        occupancy_count=8,
        cooling_setpoint=23.0,
        heating_setpoint=20.0,
    )
    ahu = EquipmentState(
        equipment_id="AHU_01",
        status=EquipmentStatus.ON,
        current_power_kw=42.5,
        fan_speed_pct=75.0,
        damper_position_pct=60.0,
    )
    state = BuildingState(
        timestamp=datetime(2025, 8, 15, 14, 0, 0, tzinfo=timezone.utc),
        weather=weather,
        zones={
            "conf_room_exec": conf_room,
            "open_office_201": open_office,
        },
        equipment={"AHU_01": ahu},
        energy_price=0.45,
        carbon_intensity=430.0,
    )

    report: BuildingScoreReport = score_building(
        state,
        total_power_kw=42.5,
        baseline_power_kw=35.0,
        avg_energy_price=0.22,
        carbon_min=150.0,
        carbon_max=550.0,
    )

    # conf_room comfort=54
    results.append(
        check(
            "conf_room_exec comfort -> 54",
            report.zone_scores["conf_room_exec"].comfort,
            54.0,
        )
    )
    # conf_room air_quality=45
    results.append(
        check(
            "conf_room_exec air_quality -> 45",
            report.zone_scores["conf_room_exec"].air_quality,
            45.0,
        )
    )
    # open_office comfort: PMV=0.2 -> 96
    results.append(
        check(
            "open_office_201 comfort -> 96",
            report.zone_scores["open_office_201"].comfort,
            96.0,
        )
    )
    # worst comfort zone should be conf_room_exec
    worst = report.worst_comfort_zone()
    ok_worst = worst == "conf_room_exec"
    print(
        f"  {'[PASS]' if ok_worst else '[FAIL]'}  "
        f"worst_comfort_zone=conf_room_exec"
    )
    results.append(ok_worst)

    print(f"\n  Energy Score : {report.energy:.2f}")
    print(f"  Carbon Score : {report.carbon:.2f}")
    return results


def main() -> None:
    """Run all Phase 2 tests and report summary."""
    print(SEP)
    print("  Phase 2 -- Feature & Score Engine Smoke Test")
    print(SEP)

    all_results: list[bool] = []
    all_results += test_comfort_score_formulas()
    all_results += test_air_quality_score_formulas()
    all_results += test_energy_score_formulas()
    all_results += test_carbon_score_formulas()
    all_results += test_feature_extraction()
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
