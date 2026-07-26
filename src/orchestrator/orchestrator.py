"""LLM Orchestrator — Phase 5 of the HVAC optimization pipeline.

Resolves trade-offs between competing agent recommendations using
an LLM. When the LLM is unavailable, a deterministic rule-based
fallback is engaged (architecture §2 Error Handling).

Architecture reference: Section 5 (LLM Orchestrator node),
Section 4.2 (Interaction Rule — urgency_score + context priority).

D2 Resolution Rule (encoded in both LLM prompt and fallback):
  Zone-scoped INCREASE_FAN_SPEED for a high-density zone overrides
  building-scoped DECREASE_FAN_SPEED when air_quality = HIGH.

D4 Policy Rule (hard constraint, not preference):
  High-Density Occupancy Floor constraints are injected into the
  system prompt and verified in the acceptance test.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.agents.models import (
    ActionType,
    ContextPriorities,
    ObjectivePriority,
    Recommendation,
)
from src.engine.scores import BuildingScoreReport
from src.events.models import EventList
from src.llm.client import LLMConfig, call_llm
from src.orchestrator.models import (
    BuildingActionPlan,
    OrchestratorPlan,
    ZoneActionPlan,
)
from src.orchestrator.prompts import build_system_prompt, build_user_prompt
from src.state.building import BuildingState

logger = logging.getLogger(__name__)

# Action delta magnitudes applied by the rule-based fallback.
_SETPOINT_DELTA_COOL: float = -1.0   # °C per step
_SETPOINT_DELTA_HEAT: float = 1.0    # °C per step
_DAMPER_DELTA_OPEN: float = 20.0     # percentage points
_FAN_DELTA_UP: float = 15.0          # percentage points
_FAN_DELTA_DOWN: float = -10.0       # percentage points


# ---------------------------------------------------------------------------
# LLM response parser
# ---------------------------------------------------------------------------


def _parse_llm_response(
    raw: str,
    state: BuildingState,
) -> OrchestratorPlan:
    """Parse the LLM's JSON response into an OrchestratorPlan.

    Args:
        raw: Raw LLM response string (expected to be pure JSON).
        state: Building state for zone validation.

    Returns:
        A parsed OrchestratorPlan.

    Raises:
        ValueError: If the response is not valid JSON or is
            missing required fields.
    """
    # Strip markdown fences if the LLM included them.
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            ln
            for ln in lines
            if not ln.startswith("```")
        ).strip()

    try:
        data: dict[str, Any] = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM response is not valid JSON: {exc}\n"
            f"Raw response (first 300 chars): {raw[:300]}"
        ) from exc

    zone_plans: dict[str, ZoneActionPlan] = {}
    for zid, zdata in data.get("zone_plans", {}).items():
        if zid not in state.zones:
            logger.warning(
                "LLM referenced unknown zone '%s' — skipping.", zid
            )
            continue
        zone_plans[zid] = ZoneActionPlan(
            zone_id=zid,
            cooling_setpoint_delta=zdata.get(
                "cooling_setpoint_delta"
            ),
            damper_delta_pct=zdata.get("damper_delta_pct"),
            fan_speed_delta_pct=zdata.get("fan_speed_delta_pct"),
            rationale=zdata.get("rationale", "LLM-provided plan."),
        )

    building_data = data.get("building_plan")
    building_plan = None
    if building_data:
        building_plan = BuildingActionPlan(
            fan_speed_delta_pct=building_data.get(
                "fan_speed_delta_pct"
            ),
            rationale=building_data.get(
                "rationale", "LLM-provided building directive."
            ),
        )

    return OrchestratorPlan(
        zone_plans=zone_plans,
        building_plan=building_plan,
        active_policy_rules=data.get("active_policy_rules", []),
        conflict_resolutions=data.get("conflict_resolutions", []),
        overall_rationale=data.get(
            "overall_rationale",
            "Plan resolved by LLM Orchestrator.",
        ),
        resolved_by="llm",
    )


# ---------------------------------------------------------------------------
# Rule-based fallback (AGENTS.md §2: LLM Orchestration Fallback)
# ---------------------------------------------------------------------------


def _rule_based_fallback(
    state: BuildingState,
    scores: BuildingScoreReport,
    events: EventList,
    context: ContextPriorities,
    recs: list[Recommendation],
    reason: str,
) -> OrchestratorPlan:
    """Rule-based recommendation merger used when LLM fails.

    Applies a deterministic priority hierarchy:
      1. Hard constraints (High-Density Floor).
      2. Human health: air quality > comfort.
      3. Energy / demand response.
      4. Carbon (lowest priority).

    Cross-scope D2 conflict resolved explicitly: if high-density
    zone needs INCREASE_FAN_SPEED and energy wants
    DECREASE_FAN_SPEED, zone takes precedence.

    Args:
        state: Building state.
        scores: Score report.
        events: Event list.
        context: Context priorities.
        recs: All agent recommendations sorted by urgency desc.
        reason: Human-readable reason why fallback was triggered.

    Returns:
        A deterministic OrchestratorPlan.
    """
    logger.warning(
        "Engaging rule-based fallback. Reason: %s", reason
    )

    zone_plans: dict[str, ZoneActionPlan] = {}
    policy_rules: list[str] = []
    conflict_resolutions: list[str] = []

    # Group recs by zone
    zone_recs: dict[str | None, list[Recommendation]] = {}
    for r in recs:
        zone_recs.setdefault(r.target_zone, []).append(r)

    # Identify high-density zones needing fan increase (D2)
    hd_fan_up_zones: set[str] = set()
    for r in recs:
        if (
            r.action == ActionType.INCREASE_FAN_SPEED
            and r.target_zone in context.high_density_zones
            and context.air_quality == ObjectivePriority.HIGH
        ):
            hd_fan_up_zones.add(r.target_zone)

    # --- Per-zone plans ---
    for zid, zone in state.zones.items():
        zs = scores.zone_scores[zid]
        zrecs = zone_recs.get(zid, [])

        cs_delta: float | None = None
        damper_delta: float | None = None
        fan_delta: float | None = None
        rationale_parts: list[str] = []

        # Air quality (highest priority)
        if any(
            r.action == ActionType.INCREASE_VENTILATION
            for r in zrecs
        ):
            damper_delta = _DAMPER_DELTA_OPEN
            rationale_parts.append(
                f"Ventilation increased "
                f"(CO2={zone.co2_ppm:.0f}PPM)."
            )
            if zid in context.high_density_zones:
                policy_rules.append(
                    f"High-Density Floor active for '{zid}': "
                    f"ventilation is non-negotiable."
                )

        if zid in hd_fan_up_zones:
            fan_delta = _FAN_DELTA_UP
            rationale_parts.append(
                f"Fan speed increased for high-density zone "
                f"'{zid}' (D2 override)."
            )

        # Comfort
        if any(
            r.action == ActionType.INCREASE_COOLING
            for r in zrecs
        ) and cs_delta is None:
            cs_delta = _SETPOINT_DELTA_COOL
            rationale_parts.append(
                f"Cooling increased (PMV={zone.pmv:+.2f}, "
                f"comfort={zs.comfort:.1f})."
            )

        # Pre-cool
        if any(
            r.action == ActionType.PRE_COOL for r in zrecs
        ) and cs_delta is None:
            cs_delta = _SETPOINT_DELTA_COOL
            rationale_parts.append("Pre-cooling applied.")

        # Relax setpoint for unoccupied
        if (
            any(
                r.action == ActionType.RELAX_SETPOINT
                for r in zrecs
            )
            and not zone.is_occupied
        ):
            cs_delta = 2.0  # raise setpoint by 2°C
            rationale_parts.append(
                f"Zone '{zid}' unoccupied — setpoint relaxed."
            )

        if rationale_parts:
            zone_plans[zid] = ZoneActionPlan(
                zone_id=zid,
                cooling_setpoint_delta=cs_delta,
                damper_delta_pct=damper_delta,
                fan_speed_delta_pct=fan_delta,
                rationale=" ".join(rationale_parts),
            )

    # --- Building-wide plan ---
    # D2: If any high-density zone needs fan increase,
    # suppress the building DECREASE_FAN_SPEED.
    bld_recs = zone_recs.get(None, [])
    bld_fan_delta: float | None = None

    has_decrease = any(
        r.action == ActionType.DECREASE_FAN_SPEED
        for r in bld_recs
    )
    if has_decrease and hd_fan_up_zones:
        conflict_resolutions.append(
            "D2 fan-speed conflict: building-scope DECREASE_FAN_SPEED "
            "suppressed because high-density zones "
            f"{sorted(hd_fan_up_zones)} require INCREASE_FAN_SPEED "
            "with air_quality=HIGH context."
        )
        bld_fan_delta = None
    elif has_decrease:
        bld_fan_delta = _FAN_DELTA_DOWN

    bld_rationale = (
        "Rule-based building directives. "
        + (
            "Fan speed suppressed (D2 override)."
            if has_decrease and hd_fan_up_zones
            else (
                "Fan speed reduced (peak demand)."
                if bld_fan_delta is not None
                else "No building-wide fan change warranted."
            )
        )
    )

    building_plan = BuildingActionPlan(
        fan_speed_delta_pct=bld_fan_delta,
        rationale=bld_rationale,
    )

    return OrchestratorPlan(
        zone_plans=zone_plans,
        building_plan=building_plan,
        active_policy_rules=policy_rules,
        conflict_resolutions=conflict_resolutions,
        overall_rationale=(
            f"Rule-based fallback engaged ({reason}). "
            f"Priority hierarchy applied: air quality > comfort "
            f"> energy > carbon. D2 scope conflict resolved."
        ),
        resolved_by="rule_based_fallback",
    )


# ---------------------------------------------------------------------------
# Hard-constraint acceptance test (D4)
# ---------------------------------------------------------------------------


def _verify_hard_constraints(
    plan: OrchestratorPlan,
    state: BuildingState,
    scores: BuildingScoreReport,
    context: ContextPriorities,
) -> list[str]:
    """Verify the plan does not violate Hard-Density Floor constraints.

    Args:
        plan: The LLM-produced OrchestratorPlan.
        state: Building state.
        scores: Score report.
        context: Context priorities.

    Returns:
        List of violation descriptions. Empty list = no violations.
    """
    violations: list[str] = []

    for zid in context.high_density_zones:
        zone = state.zones.get(zid)
        zs = scores.zone_scores.get(zid)
        zplan = plan.zone_plans.get(zid)
        if zone is None or zs is None or zplan is None:
            continue

        # D4-a: Must not raise cooling setpoint when comfort < 55
        if (
            zs.comfort < 55.0
            and zplan.cooling_setpoint_delta is not None
            and zplan.cooling_setpoint_delta > 0
        ):
            violations.append(
                f"[VIOLATION D4-A] Zone '{zid}': plan raises "
                f"cooling setpoint (+{zplan.cooling_setpoint_delta}C)"
                f" while comfort={zs.comfort:.1f} < 55."
            )

        # D4-b: Must not reduce ventilation when CO2 > 1000
        if (
            zone.co2_ppm > 1000.0
            and zplan.damper_delta_pct is not None
            and zplan.damper_delta_pct < 0
        ):
            violations.append(
                f"[VIOLATION D4-B] Zone '{zid}': plan reduces "
                f"ventilation ({zplan.damper_delta_pct:.1f} pct) "
                f"while CO2={zone.co2_ppm:.0f}PPM > 1000."
            )

    return violations


# ---------------------------------------------------------------------------
# Public orchestration entry point
# ---------------------------------------------------------------------------


def orchestrate(
    state: BuildingState,
    scores: BuildingScoreReport,
    events: EventList,
    context: ContextPriorities,
    recs: list[Recommendation],
    config: LLMConfig | None = None,
) -> OrchestratorPlan:
    """Run the LLM Orchestrator to produce a resolved action plan.

    Attempts an LLM call; falls back to rule-based resolution on
    any failure. Verifies the resulting plan against hard
    constraints (D4) regardless of resolution path.

    Args:
        state: Current building state.
        scores: Score report from the Score Engine.
        events: Event list from the Event Generator.
        context: Governing context priorities.
        recs: Agent recommendations sorted by urgency descending.
        config: LLM execution config. Defaults to LLMConfig().

    Returns:
        A verified OrchestratorPlan. If the LLM plan violates
        hard constraints, rule-based fallback is substituted.
    """
    system_prompt = build_system_prompt(state, scores, context)
    user_prompt = build_user_prompt(
        state, scores, events, context, recs
    )

    plan: OrchestratorPlan | None = None
    fallback_reason: str = ""

    # --- Attempt LLM resolution ---
    try:
        raw = call_llm(system_prompt, user_prompt, config)
        plan = _parse_llm_response(raw, state)
        logger.info("LLM plan resolved successfully.")
    except RuntimeError as exc:
        fallback_reason = str(exc)
        logger.warning("LLM unavailable: %s", exc)
    except ValueError as exc:
        fallback_reason = f"LLM parse error: {exc}"
        logger.warning("LLM parse error: %s", exc)

    # --- Verify hard constraints; substitute fallback if violated ---
    if plan is not None:
        violations = _verify_hard_constraints(
            plan, state, scores, context
        )
        if violations:
            fallback_reason = (
                "LLM plan violated hard constraints: "
                + "; ".join(violations)
            )
            logger.error(
                "LLM plan rejected — constraint violations: %s",
                violations,
            )
            plan = None

    # --- Rule-based fallback ---
    if plan is None:
        plan = _rule_based_fallback(
            state,
            scores,
            events,
            context,
            recs,
            fallback_reason,
        )

    return plan
