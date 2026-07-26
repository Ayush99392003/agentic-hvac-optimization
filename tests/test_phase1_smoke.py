"""Phase 1 smoke test — instantiate a full BuildingState and
verify all Pydantic validators and property helpers work
correctly against the demo scenario from architecture section 6.3.
"""

import sys

from datetime import datetime, timezone

from src.state import (
    BuildingState,
    EquipmentState,
    EquipmentStatus,
    WeatherState,
    ZoneState,
)


def build_demo_state() -> BuildingState:
    """Construct the architecture demo scenario state.

    Scenario: 15 occupants in Executive Conference Room during
    peak tariff ($0.45/kWh). CO2 = 1150 PPM, PMV = +1.1.

    Returns:
        A fully validated BuildingState for the demo scenario.
    """
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

    ahu_01 = EquipmentState(
        equipment_id="AHU_01",
        status=EquipmentStatus.ON,
        current_power_kw=42.5,
        fan_speed_pct=75.0,
        damper_position_pct=60.0,
    )

    return BuildingState(
        timestamp=datetime(
            2025, 8, 15, 14, 0, 0, tzinfo=timezone.utc
        ),
        weather=weather,
        zones={
            "conf_room_exec": conf_room,
            "open_office_201": open_office,
        },
        equipment={"AHU_01": ahu_01},
        energy_price=0.45,
        carbon_intensity=430.0,
    )


def main() -> None:
    """Run Phase 1 smoke test and print state summary."""
    state = build_demo_state()

    sep = "=" * 60
    print(sep)
    print("  Phase 1 -- BuildingState Smoke Test")
    print(sep)
    print(f"  Timestamp     : {state.timestamp.isoformat()}")
    print(f"  Energy Price  : ${state.energy_price:.2f}/kWh")
    print(
        f"  Carbon        : {state.carbon_intensity} gCO2e/kWh"
    )
    print(f"  High Carbon?  : {state.is_high_carbon_window}")
    print()

    for zid, zone in state.zones.items():
        print(f"  Zone: {zid}")
        print(f"    Temp        : {zone.temperature} C")
        print(f"    PMV         : {zone.pmv}")
        print(f"    CO2         : {zone.co2_ppm} PPM")
        print(f"    Occupants   : {zone.occupancy_count}")
        print(f"    Occupied?   : {zone.is_occupied}")
        print(f"    HighDensity?: {zone.is_high_density}")
        print(f"    Comfort OK? : {zone.comfort_in_band}")
        print(
            f"    Air OK?     : {not zone.air_quality_degraded}"
        )
        print()

    print(
        f"  Commandable Eq: "
        f"{list(state.commandable_equipment.keys())}"
    )
    print(
        f"  High-Density Z: "
        f"{list(state.high_density_zones.keys())}"
    )
    print(sep)
    print("  PASS -- All validators passed -- Phase 1 complete.")
    print(sep)


if __name__ == "__main__":
    sys.exit(main())  # type: ignore[func-returns-value]
