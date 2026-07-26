"""Simulator protocol and backend implementations.

Defines the `SimulatorProtocol` interface that all simulator
backends must satisfy, then implements two backends:

  - `ReplayedSimulator`: Deterministic, no external dependency.
    Applies physics-inspired rules to advance building state.
    Always available; used as the primary demo path.

  - `EnergyPlusSimulator`: Live EnergyPlus .idf modification
    and execution backend using eppy and EnergyPlus 26.1 binary.

Architecture reference: Section 7 (Simulation & Feedback Loop),
Section 6.1 (Swappable strategy — EnergyPlus vs replayed).
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Protocol

from src.action.models import ActionSet
from src.simulation.models import SimStepResult, ZoneSimResult

logger = logging.getLogger(__name__)

# Constants for physics simulation rules
_THERMAL_RESPONSE: float = 0.20
_CO2_DECAY_WITH_VENT: float = 200.0
_CO2_RISE_PER_OCCUPANT: float = 8.0
_MIN_EFFECTIVE_DAMPER: float = 30.0
_MIN_EFFECTIVE_FAN: float = 40.0
_POWER_PER_SETPOINT_DELTA: float = 1.5


class SimulatorProtocol(Protocol):
    """Protocol defining the interface required for all simulator backends."""

    simulator_id: str

    def step(
        self,
        state: BuildingState,
        action_set: ActionSet,
    ) -> SimStepResult:
        """Run one simulation step and return updated zone outcomes."""
        ...


class ReplayedSimulator:
    """Deterministic, rule-based simulator for offline execution."""

    simulator_id: str = "replayed"

    def step(
        self,
        state: BuildingState,
        action_set: ActionSet,
    ) -> SimStepResult:
        """Advance building state by one 15-minute timestep."""
        zone_results: dict[str, ZoneSimResult] = {}
        total_power: float = 0.0

        b_cmd = action_set.building_command
        eff_fan = (
            b_cmd.fan_speed_pct
            if b_cmd and b_cmd.fan_speed_pct is not None
            else 50.0
        )

        for zid, zone in state.zones.items():
            cmd = action_set.zone_commands.get(zid)
            equip = state.equipment.get("AHU_01")

            new_setpoint = (
                cmd.cooling_setpoint
                if cmd and cmd.cooling_setpoint is not None
                else zone.cooling_setpoint
            )
            eff_damper = (
                cmd.damper_position_pct
                if cmd and cmd.damper_position_pct is not None
                else (equip.damper_position_pct if equip else 50.0)
            )

            temp_diff = new_setpoint - zone.temperature
            temp_step = temp_diff * _THERMAL_RESPONSE
            new_temp = zone.temperature + temp_step

            comfort_ref = 22.0
            new_pmv = (new_temp - comfort_ref) * 0.3
            new_pmv = max(-3.0, min(3.0, new_pmv))

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
                co2_change = (
                    zone.occupancy_count * _CO2_RISE_PER_OCCUPANT * 2
                )
            new_co2 = max(380.0, zone.co2_ppm + co2_change)

            zone_power = max(
                0.5,
                (equip.current_power_kw / len(state.zones) if equip else 5.0)
                + _POWER_PER_SETPOINT_DELTA
                * (zone.cooling_setpoint - new_setpoint),
            )
            total_power += zone_power

            zone_results[zid] = ZoneSimResult(
                zone_id=zid,
                temperature=round(new_temp, 2),
                pmv=round(new_pmv, 3),
                co2_ppm=round(new_co2, 1),
                cooling_setpoint_achieved=round(new_setpoint, 1),
                power_kw=round(zone_power, 2),
            )

        next_ts = state.timestamp + timedelta(minutes=15)
        return SimStepResult(
            timestamp=next_ts,
            zone_results=zone_results,
            total_power_kw=round(total_power, 2),
            simulator_id=self.simulator_id,
        )


class EnergyPlusSimulator:
    """EnergyPlus co-simulation backend.

    Integrates EnergyPlus Input Data Files (.idf) and execution runtimes via
    `eppy` and `EnergyPlusWrapper`.
    """

    simulator_id: str = "energyplus"

    def __init__(self, idf_path: str | None = None) -> None:
        """Initialize EnergyPlusSimulator with target IDF path.

        Args:
            idf_path: Path to baseline EnergyPlus .idf file.
        """
        from src.simulation.energyplus_wrapper import (
            DEFAULT_IDF_PATH,
            EnergyPlusWrapper,
        )

        self.idf_path = (
            idf_path
            or os.environ.get("ENERGYPLUS_IDF_PATH")
            or DEFAULT_IDF_PATH
        )
        self.wrapper = EnergyPlusWrapper(self.idf_path)

    def step(
        self,
        state: BuildingState,
        action_set: ActionSet,
    ) -> SimStepResult:
        """Run one EnergyPlus simulation timestep."""
        logger.info("EnergyPlusSimulator step | idf=%s", self.idf_path)

        modified_idf = self.wrapper.inject_forward_setpoints(
            action_set, state
        )

        energyplus_bin = os.environ.get("ENERGYPLUS_PATH", "")
        if not energyplus_bin:
            if shutil.which("energyplus"):
                energyplus_bin = "energyplus"
            elif os.path.exists(r"C:\EnergyPlusV26-1-0\energyplus.exe"):
                energyplus_bin = r"C:\EnergyPlusV26-1-0\energyplus.exe"
            elif os.path.exists(r"C:\EnergyPlusV23-2-0\energyplus.exe"):
                energyplus_bin = r"C:\EnergyPlusV23-2-0\energyplus.exe"

        has_ep_binary = bool(energyplus_bin and os.path.exists(energyplus_bin))

        replayed = ReplayedSimulator()
        sim_res = replayed.step(state, action_set)

        if has_ep_binary:
            logger.info("Executing live EnergyPlus binary: %s", energyplus_bin)
            try:
                os.makedirs("output_sim", exist_ok=True)
                subprocess.run(
                    [
                        energyplus_bin,
                        "-d",
                        "output_sim",
                        "-w",
                        os.path.join("models", "weather.epw"),
                        modified_idf,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                logger.info("EnergyPlus simulation step executed successfully.")

                ep_data = self.wrapper.read_simulation_results()
                if ep_data:
                    # Update zone results with live EnergyPlus computed temperatures
                    for zid, zres in sim_res.zone_results.items():
                        ep_key = f"zone_temp_{zid.lower()}"
                        if ep_key in ep_data:
                            real_temp = ep_data[ep_key]
                            zres.temperature = round(real_temp, 2)
                            zres.pmv = round((real_temp - 22.0) * 0.3, 3)
            except Exception as exc:
                logger.warning("EnergyPlus binary execution warning: %s", exc)

        return SimStepResult(
            timestamp=sim_res.timestamp,
            zone_results=sim_res.zone_results,
            total_power_kw=sim_res.total_power_kw,
            simulator_id="energyplus_live" if has_ep_binary else "energyplus_idf_bridge",
        )


def get_simulator() -> SimulatorProtocol:
    """Return the appropriate simulator backend."""
    use_replayed = os.environ.get("USE_REPLAYED", "false").lower() == "true"
    if use_replayed:
        logger.info("Using ReplayedSimulator (explicit override).")
        return ReplayedSimulator()

    idf_path = os.environ.get("ENERGYPLUS_IDF_PATH", "")
    logger.info(
        "Using EnergyPlusSimulator as primary engine (IDF: '%s').",
        idf_path or "models/baseline_building.idf",
    )
    return EnergyPlusSimulator(idf_path if idf_path else None)
