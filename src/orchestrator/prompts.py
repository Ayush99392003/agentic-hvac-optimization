"""Prompt builder for the LLM Orchestrator.

Constructs the system and user prompts sent to the LLM.
The system prompt encodes hard constraints (D4 decision).
The user prompt encodes all context, events, scores, and
ranked recommendations.

Architecture reference: Section 5 (LLM Orchestrator),
Section 6.3 (Demo Scenario — Governing Policy Rule).
"""

from __future__ import annotations

import json

from src.agents.models import ContextPriorities, Recommendation
from src.engine.scores import BuildingScoreReport
from src.events.models import EventList
from src.state.building import BuildingState

# ---------------------------------------------------------------------------
# Hard constraint templates (D4 decision)
# ---------------------------------------------------------------------------

_HARD_CONSTRAINT_TEMPLATE = """\
HARD CONSTRAINTS — You MUST NOT violate these regardless of \
agent recommendations:
{constraints}

These constraints are non-negotiable. If any recommended action \
would violate a constraint, REJECT that action and explain why \
in your rationale.
"""

_OUTPUT_FORMAT = """\
Respond with ONLY a valid JSON object in this exact format:
{
  "zone_plans": {
    "<zone_id>": {
      "cooling_setpoint_delta": <float or null>,
      "damper_delta_pct": <float or null>,
      "fan_speed_delta_pct": <float or null>,
      "rationale": "<string>"
    }
  },
  "building_plan": {
    "fan_speed_delta_pct": <float or null>,
    "rationale": "<string>"
  },
  "active_policy_rules": ["<string>", ...],
  "conflict_resolutions": ["<string>", ...],
  "overall_rationale": "<string>"
}

Do not include markdown fences, explanation text, or any content \
outside the JSON object.
"""


def _build_hard_constraints(
    state: BuildingState,
    scores: BuildingScoreReport,
    context: ContextPriorities,
) -> list[str]:
    """Derive active hard constraints from current state (D4).

    Args:
        state: Current building state.
        scores: Score report for this timestep.
        context: Governing context priorities.

    Returns:
        List of human-readable constraint strings to inject
        into the system prompt.
    """
    constraints: list[str] = []

    for zid in context.high_density_zones:
        zone = state.zones.get(zid)
        zs = scores.zone_scores.get(zid)
        if zone is None or zs is None:
            continue

        if zs.comfort < 55.0:
            constraints.append(
                f"[CONSTRAINT-A] Zone '{zid}' has {zone.occupancy_count}"
                f" occupants and comfort score {zs.comfort:.1f}. "
                f"You MUST NOT raise its cooling setpoint."
            )
        if zone.co2_ppm > 1000.0:
            constraints.append(
                f"[CONSTRAINT-B] Zone '{zid}' CO2 is "
                f"{zone.co2_ppm:.0f} PPM. "
                f"You MUST NOT reduce its ventilation "
                f"(damper_delta_pct must be >= 0)."
            )
        if zone.occupancy_count > 10:
            constraints.append(
                f"[CONSTRAINT-C] Zone '{zid}' has "
                f"{zone.occupancy_count} occupants "
                f"(High-Density Floor active). "
                f"You MUST NOT completely shed its HVAC load."
            )

    return constraints


def build_system_prompt(
    state: BuildingState,
    scores: BuildingScoreReport,
    context: ContextPriorities,
) -> str:
    """Build the system-role prompt with hard constraints.

    Args:
        state: Current building state.
        scores: Score report.
        context: Context priorities.

    Returns:
        System prompt string.
    """
    constraints = _build_hard_constraints(state, scores, context)

    constraint_block = ""
    if constraints:
        numbered = "\n".join(
            f"  {i + 1}. {c}" for i, c in enumerate(constraints)
        )
        constraint_block = _HARD_CONSTRAINT_TEMPLATE.format(
            constraints=numbered
        )

    return f"""\
You are the HVAC Orchestrator for an intelligent building control \
system. Your role is to:
  1. Read the detected events and agent recommendations.
  2. Resolve conflicts between competing recommendations.
  3. Apply the governing context priorities as weighting signals.
  4. Produce a single, coherent action plan for this timestep.

{constraint_block}
SCOPE CONFLICT RESOLUTION RULE (D2):
When a zone-scoped fan increase conflicts with a building-scoped \
fan decrease, and the zone is high-density with air_quality \
priority = HIGH, the zone-scoped action takes precedence.

TIEBREAK CONVENTION (D3) for equal urgency scores:
  1. Human health (air_quality) > comfort > energy > carbon.
  2. Zone-scoped beats building-scoped.
  3. Pipeline order: comfort > energy > air > carbon > dr.

{_OUTPUT_FORMAT}"""


def _format_recommendations(
    recs: list[Recommendation],
) -> str:
    """Serialize recommendations as a numbered list for the prompt.

    Args:
        recs: Recommendations sorted by urgency descending.

    Returns:
        Formatted multi-line string.
    """
    lines = []
    for i, r in enumerate(recs, 1):
        scope = r.target_zone or "BUILDING"
        lines.append(
            f"  {i}. [{r.urgency_score:3d}] "
            f"{r.agent_id} | zone={scope} | "
            f"action={r.action.value}"
        )
        lines.append(f"       Rationale: {r.rationale}")
    return "\n".join(lines)


def _format_events(events: EventList) -> str:
    """Serialize the event list for the prompt.

    Args:
        events: Event list from the Event Generator.

    Returns:
        Formatted multi-line string.
    """
    if not events.events:
        return "  (none)"
    lines = []
    for e in events.events:
        zone = e.zone_id or "BUILDING"
        lines.append(
            f"  - [{e.severity.value}] {e.event_type.value} | "
            f"zone={zone} | "
            f"{e.trigger_metric}={e.trigger_value:.1f} | "
            f"score={e.trigger_score:.1f}"
        )
    return "\n".join(lines)


def _format_scores(
    state: BuildingState,
    scores: BuildingScoreReport,
) -> str:
    """Serialize scores and key zone metrics for the prompt.

    Args:
        state: Building state (for raw metrics).
        scores: Score report.

    Returns:
        Formatted multi-line string.
    """
    lines = [
        f"  Building: energy={scores.energy:.1f}  "
        f"carbon={scores.carbon:.1f}  "
        f"tariff=${state.energy_price:.3f}/kWh  "
        f"carbon_intensity={state.carbon_intensity:.0f}gCO2e/kWh"
    ]
    for zid, zs in scores.zone_scores.items():
        z = state.zones[zid]
        lines.append(
            f"  Zone '{zid}': comfort={zs.comfort:.1f}  "
            f"air_quality={zs.air_quality:.1f}  "
            f"PMV={z.pmv:+.2f}  CO2={z.co2_ppm:.0f}PPM  "
            f"occupants={z.occupancy_count}"
        )
    return "\n".join(lines)


def _format_context(context: ContextPriorities) -> str:
    """Serialize context priorities for the prompt.

    Args:
        context: Context priorities from the Context Engine.

    Returns:
        Formatted string.
    """
    hd = (
        ", ".join(sorted(context.high_density_zones))
        if context.high_density_zones
        else "none"
    )
    return (
        f"  comfort={context.comfort.value}  "
        f"air_quality={context.air_quality.value}  "
        f"energy={context.energy.value}  "
        f"carbon={context.carbon.value}  "
        f"high_density_zones=[{hd}]"
    )


def build_user_prompt(
    state: BuildingState,
    scores: BuildingScoreReport,
    events: EventList,
    context: ContextPriorities,
    recs: list[Recommendation],
) -> str:
    """Build the user-role prompt with all scenario data.

    Args:
        state: Current building state.
        scores: Score report.
        events: Detected events.
        context: Governing context priorities.
        recs: All agent recommendations sorted by urgency desc.

    Returns:
        User prompt string.
    """
    return f"""\
=== CURRENT BUILDING STATE ===
{_format_scores(state, scores)}

=== CONTEXT PRIORITIES ===
{_format_context(context)}

=== DETECTED EVENTS ({len(events.events)} total) ===
{_format_events(events)}

=== AGENT RECOMMENDATIONS ({len(recs)} total, sorted by urgency) ===
{_format_recommendations(recs)}

Now produce the JSON action plan that resolves all conflicts and \
respects all hard constraints. Ensure your conflict_resolutions \
field explicitly addresses the fan-speed scope conflict (D2) and \
any other trade-off you resolve."""
