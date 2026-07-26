"""Autonomous Agentic Building Control & HVAC Optimization Pipeline.

Main entrypoint executing the 7-tier layered architecture:
  1. State Modeling (BuildingState, ZoneState, EquipmentState, WeatherState)
  2. Feature & Score Engine (Deterministic 0-100 score metrics)
  3. Event Detection (Semantic event generation)
  4. Specialized Agent Layer (5 domain agents proposing recommendations)
  5. LLM Orchestration Layer (LLM conflict resolution & hard constraints)
  6. Action & Safety Engine (Delta-to-absolute & physical bound safety)
  7. Simulation & Feedback Loop (Replayed/EnergyPlus simulation & audit)

Run options:
    uv run main.py --scenario centerpiece --cycles 3
    uv run main.py --scenario normal
    uv run main.py --scenario emergency_co2
    uv run main.py --scenario peak_tariff_unoccupied
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

# Load .env file automatically if present
if os.path.exists(".env"):
    with open(".env", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.action import ActionSet, build_action_set
from src.agents import ContextPriorities, run_all_agents
from src.engine.features import extract_building_features
from src.engine.scores import BuildingScoreReport, score_building
from src.events import EventList, generate_events
from src.llm import LLMConfig
from src.orchestrator import OrchestratorPlan, orchestrate
from src.simulation import (
    FeedbackReport,
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

console = Console()

# Pipeline configuration constants
DEFAULT_AVG_TARIFF: float = 0.22
DEFAULT_BASELINE_POWER: float = 35.0
DEFAULT_CARBON_MIN: float = 150.0
DEFAULT_CARBON_MAX: float = 550.0


# ---------------------------------------------------------------------------
# Scenario Definitions
# ---------------------------------------------------------------------------


def scenario_centerpiece() -> BuildingState:
    """Demo Centerpiece Scenario: Multi-objective conflict.

    Executive Conference Room has 15 occupants during peak tariff window
    ($0.45/kWh) with elevated CO2 (1150 PPM) and high thermal discomfort
    (PMV=+1.1). Triggers conflicts between Comfort, Air Quality, Energy,
    Demand Response, and Carbon agents.
    """
    return BuildingState(
        timestamp=datetime(2025, 8, 15, 14, 0, 0, tzinfo=timezone.utc),
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


def scenario_normal() -> BuildingState:
    """Normal Operations Scenario: Steady-state baseline.

    All building zones are within comfortable operating limits.
    Demonstrates energy-efficient nominal operation.
    """
    return BuildingState(
        timestamp=datetime(2025, 8, 15, 10, 0, 0, tzinfo=timezone.utc),
        weather=WeatherState(
            dry_bulb_temperature=24.0,
            relative_humidity=50.0,
            direct_solar_irradiance=400.0,
            wind_speed=2.0,
        ),
        zones={
            "conf_room_exec": ZoneState(
                zone_id="conf_room_exec",
                temperature=22.5,
                relative_humidity=48.0,
                pmv=0.1,
                co2_ppm=550.0,
                occupancy_count=4,
                cooling_setpoint=23.0,
                heating_setpoint=20.0,
            ),
            "open_office_201": ZoneState(
                zone_id="open_office_201",
                temperature=22.8,
                relative_humidity=50.0,
                pmv=0.15,
                co2_ppm=580.0,
                occupancy_count=6,
                cooling_setpoint=23.0,
                heating_setpoint=20.0,
            ),
        },
        equipment={
            "AHU_01": EquipmentState(
                equipment_id="AHU_01",
                status=EquipmentStatus.ON,
                current_power_kw=32.0,
                fan_speed_pct=50.0,
                damper_position_pct=40.0,
            )
        },
        energy_price=0.18,
        carbon_intensity=210.0,
    )


def scenario_emergency_co2() -> BuildingState:
    """Emergency Air Quality Scenario: Safety Validator Trigger.

    CO2 levels reach 1300 PPM (> 1200 PPM emergency limit).
    Demonstrates deterministic physical safety override forcing 100% damper.
    """
    return BuildingState(
        timestamp=datetime(2025, 8, 15, 15, 30, 0, tzinfo=timezone.utc),
        weather=WeatherState(
            dry_bulb_temperature=28.0,
            relative_humidity=55.0,
            direct_solar_irradiance=500.0,
            wind_speed=2.5,
        ),
        zones={
            "auditorium": ZoneState(
                zone_id="auditorium",
                temperature=25.0,
                relative_humidity=60.0,
                pmv=0.6,
                co2_ppm=1300.0,
                occupancy_count=45,
                cooling_setpoint=23.0,
                heating_setpoint=20.0,
            )
        },
        equipment={
            "AHU_MAIN": EquipmentState(
                equipment_id="AHU_MAIN",
                status=EquipmentStatus.ON,
                current_power_kw=45.0,
                fan_speed_pct=60.0,
                damper_position_pct=35.0,
            )
        },
        energy_price=0.25,
        carbon_intensity=310.0,
    )


def scenario_peak_tariff_unoccupied() -> BuildingState:
    """Peak Tariff & Unoccupied Shedding Scenario.

    High energy price ($0.48/kWh) with zero occupancy in major zones.
    Demonstrates aggressive setpoint relaxation and demand response shedding.
    """
    return BuildingState(
        timestamp=datetime(2025, 8, 15, 17, 0, 0, tzinfo=timezone.utc),
        weather=WeatherState(
            dry_bulb_temperature=34.0,
            relative_humidity=45.0,
            direct_solar_irradiance=600.0,
            wind_speed=3.0,
        ),
        zones={
            "west_wing": ZoneState(
                zone_id="west_wing",
                temperature=23.0,
                relative_humidity=50.0,
                pmv=0.0,
                co2_ppm=420.0,
                occupancy_count=0,
                cooling_setpoint=22.0,
                heating_setpoint=20.0,
            ),
            "east_wing": ZoneState(
                zone_id="east_wing",
                temperature=23.2,
                relative_humidity=48.0,
                pmv=0.1,
                co2_ppm=430.0,
                occupancy_count=0,
                cooling_setpoint=22.0,
                heating_setpoint=20.0,
            ),
        },
        equipment={
            "AHU_01": EquipmentState(
                equipment_id="AHU_01",
                status=EquipmentStatus.ON,
                current_power_kw=48.0,
                fan_speed_pct=80.0,
                damper_position_pct=50.0,
            )
        },
        energy_price=0.48,
        carbon_intensity=490.0,
    )


SCENARIOS = {
    "centerpiece": (scenario_centerpiece, "Centerpiece Multi-Agent Conflict"),
    "demo": (scenario_centerpiece, "Centerpiece Multi-Agent Conflict (Alias)"),
    "normal": (scenario_normal, "Nominal Steady-State Operations"),
    "emergency_co2": (scenario_emergency_co2, "Emergency CO2 Safety Override"),
    "peak_tariff_unoccupied": (
        scenario_peak_tariff_unoccupied,
        "Peak Tariff Unoccupied Zone Shedding",
    ),
}


# ---------------------------------------------------------------------------
# Rich Renderers
# ---------------------------------------------------------------------------


def render_header(scenario_name: str, cycles: int) -> None:
    """Print terminal header for main pipeline execution."""
    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]Honeywell Autonomous Agentic Building Control"
            f"[/bold cyan]\n"
            f"[yellow]Scenario:[yellow] {scenario_name} | "
            f"[yellow]Cycles:[yellow] {cycles} | "
            f"[yellow]Architecture:[yellow] 7-Tier Deterministic + LLM",
            border_style="cyan",
            title="[bold green]HVAC Optimization System[/bold green]",
        )
    )


def render_cycle_results(
    cycle: int,
    state: BuildingState,
    scores: BuildingScoreReport,
    events: EventList,
    context: ContextPriorities,
    recs: list,
    plan: OrchestratorPlan,
    action_set: ActionSet,
    feedback: FeedbackReport,
) -> None:
    """Render rich tables for a single pipeline execution cycle."""
    console.print(
        f"\n[bold magenta]=== PIPELINE CYCLE {cycle + 1} ==="
        f"[/bold magenta]"
    )

    # Table 1: State & Scores
    score_table = Table(
        title=f"Cycle {cycle + 1} State & Scores", show_header=True
    )
    score_table.add_column("Zone ID", style="cyan")
    score_table.add_column("Temp (C)", justify="right")
    score_table.add_column("PMV", justify="right")
    score_table.add_column("CO2 (PPM)", justify="right")
    score_table.add_column("Occupants", justify="right")
    score_table.add_column("Comfort Score", justify="right")
    score_table.add_column("Air Quality", justify="right")

    for zid, zstate in state.zones.items():
        zscore = scores.zone_scores[zid]
        score_table.add_row(
            zid,
            f"{zstate.temperature:.1f}",
            f"{zstate.pmv:+.2f}",
            f"{zstate.co2_ppm:.0f}",
            str(zstate.occupancy_count),
            f"{zscore.comfort:.1f}",
            f"{zscore.air_quality:.1f}",
        )
    console.print(score_table)

    # Context & Priorities
    console.print(
        f"  [bold]Building Scores:[bold] Energy={scores.energy:.1f} | "
        f"Carbon={scores.carbon:.1f}"
    )
    console.print(
        f"  [bold]Context Priorities:[bold] Comfort={context.comfort.value} | "
        f"AirQuality={context.air_quality.value} | "
        f"Energy={context.energy.value} | Carbon={context.carbon.value}"
    )

    # Events Table
    if events.events:
        evt_table = Table(title="Detected Events", show_header=True)
        evt_table.add_column("Event Type", style="yellow")
        evt_table.add_column("Zone", style="cyan")
        evt_table.add_column("Severity", style="bold red")
        evt_table.add_column("Metric Value", justify="right")
        for e in events.events:
            evt_table.add_row(
                e.event_type.value,
                e.zone_id or "BUILDING",
                e.severity.value,
                f"{e.trigger_metric}={e.trigger_value:.1f}",
            )
        console.print(evt_table)

    # Top Recommendations Table
    rec_table = Table(title="Agent Recommendations", show_header=True)
    rec_table.add_column("Urgency", justify="right", style="bold yellow")
    rec_table.add_column("Agent ID", style="green")
    rec_table.add_column("Zone Target", style="cyan")
    rec_table.add_column("Action Recommended", style="bold white")
    for r in recs[:5]:
        rec_table.add_row(
            str(r.urgency_score),
            r.agent_id,
            r.target_zone or "BUILDING",
            r.action.value,
        )
    console.print(rec_table)

    # Orchestrator & Action Commands
    console.print(
        f"  [bold blue]Orchestrator Resolver:[bold blue] {plan.resolved_by}"
    )
    if plan.conflict_resolutions:
        for cr in plan.conflict_resolutions:
            console.print(f"    [yellow]-> Conflict:[yellow] {cr}")

    cmd_table = Table(title="Validated ActionSet Commands", show_header=True)
    cmd_table.add_column("Target", style="cyan")
    cmd_table.add_column("Cooling Setpoint", justify="right")
    cmd_table.add_column("Damper %", justify="right")
    cmd_table.add_column("Fan %", justify="right")
    cmd_table.add_column("Validator Status", style="bold green")

    for zid, zcmd in action_set.zone_commands.items():
        status_color = "green" if zcmd.status.value == "APPROVED" else "yellow"
        cmd_table.add_row(
            zid,
            f"{zcmd.cooling_setpoint:.1f}C"
            if zcmd.cooling_setpoint
            else "-",
            f"{zcmd.damper_position_pct:.1f}%"
            if zcmd.damper_position_pct
            else "-",
            f"{zcmd.fan_speed_pct:.1f}%" if zcmd.fan_speed_pct else "-",
            f"[{status_color}]{zcmd.status.value}[/{status_color}]",
        )
    if action_set.building_command:
        bc = action_set.building_command
        cmd_table.add_row(
            "BUILDING",
            "-",
            "-",
            f"{bc.fan_speed_pct:.1f}%" if bc.fan_speed_pct else "-",
            bc.status.value,
        )
    console.print(cmd_table)

    # Feedback Summary
    console.print(
        Panel(
            Text(feedback.overall_summary, style="italic green"),
            title=f"Cycle {cycle + 1} Audit Trail & Feedback",
        )
    )


# ---------------------------------------------------------------------------
# Pipeline Execution Loop
# ---------------------------------------------------------------------------


def run_pipeline(scenario_key: str, cycles: int) -> None:
    """Execute the end-to-end HVAC optimization pipeline."""
    scenario_func, description = SCENARIOS.get(
        scenario_key, SCENARIOS["centerpiece"]
    )
    render_header(description, cycles)

    state = scenario_func()
    simulator = get_simulator()

    for cycle in range(cycles):
        # 1. Feature & Score Engine
        scores = score_building(
            state,
            total_power_kw=sum(
                e.current_power_kw for e in state.equipment.values()
            )
            or DEFAULT_BASELINE_POWER,
            baseline_power_kw=DEFAULT_BASELINE_POWER,
            avg_energy_price=DEFAULT_AVG_TARIFF,
            carbon_min=DEFAULT_CARBON_MIN,
            carbon_max=DEFAULT_CARBON_MAX,
        )
        features = extract_building_features(
            state,
            avg_energy_price=DEFAULT_AVG_TARIFF,
            total_power_kw=sum(
                e.current_power_kw for e in state.equipment.values()
            )
            or DEFAULT_BASELINE_POWER,
            baseline_power_kw=DEFAULT_BASELINE_POWER,
            carbon_min=DEFAULT_CARBON_MIN,
            carbon_max=DEFAULT_CARBON_MAX,
        )

        # 2. Event Generator
        events = generate_events(state, scores, features)

        # 3. Context Engine & Priorities
        context = ContextPriorities.from_building_state(
            state, avg_tariff=DEFAULT_AVG_TARIFF
        )

        # 4. Domain Agent Layer
        recs = run_all_agents(state, scores, events, context)

        # 5. LLM Orchestrator Layer
        plan = orchestrate(state, scores, events, context, recs)

        # 6. Action Engine & Safety Validator
        action_set = build_action_set(plan, state, context)

        # 7. Simulation & Feedback Loop
        sim_result = simulator.step(state, action_set)
        feedback = compute_feedback(
            cycle_index=cycle,
            state=state,
            plan=plan,
            sim_result=sim_result,
            avg_energy_price=DEFAULT_AVG_TARIFF,
            baseline_power_kw=DEFAULT_BASELINE_POWER,
            carbon_min=DEFAULT_CARBON_MIN,
            carbon_max=DEFAULT_CARBON_MAX,
        )

        # Render step details
        render_cycle_results(
            cycle,
            state,
            scores,
            events,
            context,
            recs,
            plan,
            action_set,
            feedback,
        )

        # Advance state for next cycle
        state = advance_state(state, sim_result)

    console.print("\n[bold green][OK] Pipeline Execution Complete.[/bold green]\n")


def main() -> None:
    """Parse CLI arguments and run main pipeline."""
    parser = argparse.ArgumentParser(
        description="Honeywell Agentic Building Control & Optimization Pipeline"
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        default="centerpiece",
        help="Select scenario to execute (default: centerpiece)",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of simulation cycles to run (default: 1)",
    )
    parser.add_argument(
        "--api-key",
        "-k",
        type=str,
        default=None,
        help="LLM API key (or set OPENAI_API_KEY / AZURE_OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--base-url",
        "-u",
        type=str,
        default=None,
        help="Custom API base URL (e.g. http://localhost:11434/v1 for Ollama/vLLM)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Target LLM model (e.g. gpt-4o, llama3, qwen2.5)",
    )
    args = parser.parse_args()

    # Pass credentials if provided on CLI
    from src.llm import set_llm_credentials

    set_llm_credentials(
        api_key=args.api_key, base_url=args.base_url, model=args.model
    )

    run_pipeline(args.scenario, args.cycles)


if __name__ == "__main__":
    main()
