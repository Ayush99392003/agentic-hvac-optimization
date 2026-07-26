"""Action Engine — translates OrchestratorPlan into validated
setpoint commands.

Two sub-components:
  1. Action Engine: converts ZoneActionPlan deltas + current state
     into absolute ZoneSetpointCommand values.
  2. Safety Validator: clamps or rejects commands that breach
     physical safety bounds (architecture §2 Action Validation
     Safety Net — must hard-reject regardless of LLM output).

Architecture reference: Section 6 (Action & Safety Engine node),
Section 2 (Error Handling — Safety Validator must hard-reject
setpoints exceeding safe physical limits).
"""

from __future__ import annotations

import logging

from src.action.models import (
    CO2_EMERGENCY_PPM,
    COOLING_SETPOINT_MAX,
    COOLING_SETPOINT_MIN,
    DAMPER_MAX,
    DAMPER_MIN,
    DEADBAND_MIN,
    FAN_SPEED_MAX,
    FAN_SPEED_MIN,
    MAX_SETPOINT_DELTA_PER_STEP,
    ActionSet,
    BuildingCommand,
    CommandStatus,
    ZoneSetpointCommand,
)
from src.orchestrator.models import OrchestratorPlan
from src.state.building import BuildingState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action Engine — delta to absolute conversion
# ---------------------------------------------------------------------------


def _translate_zone_plan(
    zone_id: str,
    state: BuildingState,
    plan: "OrchestratorPlan",
) -> ZoneSetpointCommand | None:
    """Convert a ZoneActionPlan delta into an absolute command.

    Args:
        zone_id: Zone to translate.
        state: Current building state (provides current values).
        plan: Orchestrator plan with delta values.

    Returns:
        A ZoneSetpointCommand with absolute setpoints, or None
        if this zone has no plan.
    """
    zone_plan = plan.zone_plans.get(zone_id)
    if zone_plan is None:
        return None

    zone = state.zones[zone_id]
    equip = next(
        (e for e in state.equipment.values()), None
    )

    # Cooling setpoint: current + delta
    new_cooling: float | None = None
    if zone_plan.cooling_setpoint_delta is not None:
        new_cooling = (
            zone.cooling_setpoint + zone_plan.cooling_setpoint_delta
        )

    # Damper position: current + delta (use equipment if available)
    new_damper: float | None = None
    if zone_plan.damper_delta_pct is not None:
        current_damper = (
            equip.damper_position_pct if equip else 50.0
        )
        new_damper = current_damper + zone_plan.damper_delta_pct

    # Fan speed: current + delta
    new_fan: float | None = None
    if zone_plan.fan_speed_delta_pct is not None:
        current_fan = equip.fan_speed_pct if equip else 60.0
        new_fan = current_fan + zone_plan.fan_speed_delta_pct

    return ZoneSetpointCommand(
        zone_id=zone_id,
        cooling_setpoint=new_cooling,
        damper_position_pct=new_damper,
        fan_speed_pct=new_fan,
    )


def _translate_building_plan(
    state: BuildingState,
    plan: "OrchestratorPlan",
) -> BuildingCommand | None:
    """Convert building-level delta into an absolute command.

    Args:
        state: Current building state.
        plan: Orchestrator plan.

    Returns:
        A BuildingCommand, or None if no building plan.
    """
    if plan.building_plan is None:
        return None

    bplan = plan.building_plan
    equip = next(
        (e for e in state.equipment.values()), None
    )

    new_fan: float | None = None
    if bplan.fan_speed_delta_pct is not None:
        current_fan = equip.fan_speed_pct if equip else 60.0
        new_fan = current_fan + bplan.fan_speed_delta_pct

    return BuildingCommand(fan_speed_pct=new_fan)


# ---------------------------------------------------------------------------
# Safety Validator — physical bound enforcement
# ---------------------------------------------------------------------------


def _validate_zone_command(
    cmd: ZoneSetpointCommand,
    zone_id: str,
    state: BuildingState,
) -> None:
    """Apply physical safety bounds to a zone command in-place.

    Args:
        cmd: The command to validate and possibly clamp.
        zone_id: Zone identifier (for logging).
        state: Current building state (deadband check).
    """
    zone = state.zones[zone_id]
    notes: list[str] = []
    was_clamped = False
    is_rejected = False

    # --- CO2 emergency override ---
    if (
        zone.co2_ppm >= CO2_EMERGENCY_PPM
        and cmd.damper_position_pct is not None
        and cmd.damper_position_pct < DAMPER_MAX
    ):
        cmd.damper_position_pct = DAMPER_MAX
        was_clamped = True
        notes.append(
            f"CO2 emergency ({zone.co2_ppm:.0f}PPM >= "
            f"{CO2_EMERGENCY_PPM:.0f}): damper forced to 100%."
        )

    # --- Cooling setpoint bounds ---
    if cmd.cooling_setpoint is not None:
        raw = cmd.cooling_setpoint
        current_sp = state.zones[zone_id].cooling_setpoint

        # Rate-limit: max delta per timestep (inclusive bound).
        delta = raw - current_sp
        if abs(delta) > MAX_SETPOINT_DELTA_PER_STEP:
            limited = current_sp + (
                MAX_SETPOINT_DELTA_PER_STEP
                if delta > 0
                else -MAX_SETPOINT_DELTA_PER_STEP
            )
            was_clamped = True
            notes.append(
                f"Setpoint rate-limited: delta "
                f"{delta:+.1f}C exceeds "
                f"+/-{MAX_SETPOINT_DELTA_PER_STEP}C/step; "
                f"clamped {raw:.1f} -> {limited:.1f}C."
            )
            cmd.cooling_setpoint = limited
            raw = limited

        clamped = max(
            COOLING_SETPOINT_MIN,
            min(COOLING_SETPOINT_MAX, raw),
        )
        if abs(clamped - raw) > 0.001:
            was_clamped = True
            notes.append(
                f"Cooling setpoint clamped "
                f"{raw:.1f} -> {clamped:.1f}C "
                f"(bounds [{COOLING_SETPOINT_MIN}, "
                f"{COOLING_SETPOINT_MAX}])."
            )
            cmd.cooling_setpoint = clamped

        # Deadband check: heating_setpoint must stay < cooling - 1C
        if (
            cmd.cooling_setpoint - zone.heating_setpoint
            < DEADBAND_MIN
        ):
            # Reject the setpoint change — would cause short-cycling
            is_rejected = True
            notes.append(
                f"REJECTED: cooling setpoint "
                f"{cmd.cooling_setpoint:.1f}C would violate "
                f"deadband (heating={zone.heating_setpoint:.1f}C, "
                f"min gap={DEADBAND_MIN}C)."
            )
            cmd.cooling_setpoint = None  # nullify the command

    # --- Damper bounds ---
    if cmd.damper_position_pct is not None:
        raw = cmd.damper_position_pct
        clamped = max(DAMPER_MIN, min(DAMPER_MAX, raw))
        if abs(clamped - raw) > 0.001:
            was_clamped = True
            notes.append(
                f"Damper clamped {raw:.1f} -> {clamped:.1f}%."
            )
            cmd.damper_position_pct = clamped

    # --- Fan speed bounds ---
    if cmd.fan_speed_pct is not None:
        raw = cmd.fan_speed_pct
        clamped = max(FAN_SPEED_MIN, min(FAN_SPEED_MAX, raw))
        if abs(clamped - raw) > 0.001:
            was_clamped = True
            notes.append(
                f"Fan speed clamped {raw:.1f} -> {clamped:.1f}%."
            )
            cmd.fan_speed_pct = clamped

    cmd.validator_notes = notes
    if is_rejected:
        cmd.status = CommandStatus.REJECTED
    elif was_clamped:
        cmd.status = CommandStatus.CLAMPED
    else:
        cmd.status = CommandStatus.APPROVED

    if notes:
        logger.info(
            "Safety Validator zone '%s': %s",
            zone_id,
            "; ".join(notes),
        )


def _validate_building_command(cmd: BuildingCommand) -> None:
    """Apply physical safety bounds to a building command.

    Args:
        cmd: The command to validate and possibly clamp.
    """
    if cmd.fan_speed_pct is None:
        cmd.status = CommandStatus.APPROVED
        return

    raw = cmd.fan_speed_pct
    clamped = max(FAN_SPEED_MIN, min(FAN_SPEED_MAX, raw))
    if abs(clamped - raw) > 0.001:
        cmd.fan_speed_pct = clamped
        cmd.status = CommandStatus.CLAMPED
        cmd.validator_notes = [
            f"Building fan clamped {raw:.1f} -> {clamped:.1f}%."
        ]
        logger.info(
            "Safety Validator building: fan clamped %s -> %s%%",
            raw,
            clamped,
        )
    else:
        cmd.status = CommandStatus.APPROVED


def _enforce_d2_fan_scope(
    zone_commands: dict[str, ZoneSetpointCommand],
    building_cmd: BuildingCommand | None,
    state: BuildingState,
    context: object,  # ContextPriorities, avoid circular import
) -> None:
    """Deterministic D2 post-check: zone fan increase beats building
    fan decrease when the zone is high-density and air_quality=HIGH.

    This runs independently of the LLM prompt so D2 is enforced even
    if the LLM ignores the prompt instruction.

    Args:
        zone_commands: Translated zone commands (mutated in-place).
        building_cmd: Building command (mutated in-place).
        state: Current building state.
        context: ContextPriorities instance.
    """
    from src.agents.models import ObjectivePriority

    if building_cmd is None or building_cmd.fan_speed_pct is None:
        return

    # Check if building fan direction is DOWN vs current.
    equip = next((e for e in state.equipment.values()), None)
    current_fan = equip.fan_speed_pct if equip else 60.0
    building_going_down = building_cmd.fan_speed_pct < current_fan

    if not building_going_down:
        return

    # Find any zone-scoped fan that is going UP, in a HD zone,
    # with air_quality = HIGH.
    ctx = context  # type: ignore[assignment]
    if ctx.air_quality != ObjectivePriority.HIGH:
        return

    for zid, zcmd in zone_commands.items():
        if (
            zid not in ctx.high_density_zones
            or zcmd.fan_speed_pct is None
        ):
            continue
        zone_fan_going_up = zcmd.fan_speed_pct > current_fan
        if zone_fan_going_up:
            old_val = building_cmd.fan_speed_pct
            building_cmd.fan_speed_pct = None
            building_cmd.status = CommandStatus.CLAMPED
            building_cmd.validator_notes.append(
                f"D2 post-check: building fan suppressed "
                f"({old_val:.1f}% -> None) because zone '{zid}' "
                f"(high-density, air_quality=HIGH) requires fan "
                f"increase ({zcmd.fan_speed_pct:.1f}%)."
            )
            logger.info(
                "D2 enforced: building fan suppressed for HD zone '%s'.",
                zid,
            )
            return  # Only need one HD zone conflict to suppress.


def build_action_set(
    plan: OrchestratorPlan,
    state: BuildingState,
    context: object = None,  # ContextPriorities, optional for back-compat
) -> ActionSet:
    """Translate an OrchestratorPlan into a validated ActionSet.

    Steps:
      1. Action Engine: convert all delta values to absolute.
      2. Safety Validator: clamp / reject out-of-bound commands.
      3. D2 post-check: deterministically enforce fan-scope rule.

    Args:
        plan: The resolved OrchestratorPlan from Phase 5.
        state: Current building state (provides current values
            and heating setpoint for deadband check).
        context: Optional ContextPriorities for D2 post-check.
            When provided, D2 is enforced deterministically.

    Returns:
        A validated ActionSet ready for the EnergyPlus interface.
    """
    zone_commands: dict[str, ZoneSetpointCommand] = {}

    # Translate + validate each zone plan
    for zid in state.zones:
        cmd = _translate_zone_plan(zid, state, plan)
        if cmd is None:
            continue
        _validate_zone_command(cmd, zid, state)
        zone_commands[zid] = cmd

    # Translate + validate building plan
    building_cmd: BuildingCommand | None = (
        _translate_building_plan(state, plan)
    )
    if building_cmd is not None:
        _validate_building_command(building_cmd)

    # D2 deterministic post-check (runs regardless of LLM path)
    if context is not None:
        _enforce_d2_fan_scope(
            zone_commands, building_cmd, state, context
        )

    action_set = ActionSet(
        zone_commands=zone_commands,
        building_command=building_cmd,
    )

    if action_set.has_rejections:
        logger.error(
            "ActionSet contains REJECTED commands. "
            "Check Safety Validator notes."
        )
    if action_set.has_clamps:
        logger.warning(
            "ActionSet has clamped commands. "
            "Review physical bounds."
        )

    return action_set
