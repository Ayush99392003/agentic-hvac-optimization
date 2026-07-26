"""Orchestrator output models — the LLM Orchestrator's resolved
action plan consumed by the Action Engine (Phase 6).

Architecture reference: Section 2.2 (Outputs — Action Plan JSON),
Section 6.3 (Demo Centerpiece Scenario — single-step JSON).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ZoneActionPlan(BaseModel):
    """Resolved action plan for a single zone.

    This is the final per-zone output from the LLM Orchestrator.
    Values here are passed directly to the Action Engine for
    setpoint command generation and Safety Validator checking.

    Attributes:
        zone_id: Zone this plan targets.
        cooling_setpoint_delta: Change in cooling setpoint (°C).
            Negative = more cooling. None = no change.
        damper_delta_pct: Change in outdoor-air damper position
            (percentage points). Positive = more fresh air.
            None = no change.
        fan_speed_delta_pct: Change in fan speed (percentage
            points). Positive = faster. None = no change.
        rationale: Human-readable explanation of why this plan
            was chosen over alternatives.
    """

    zone_id: str = Field(..., description="Zone identifier.")
    cooling_setpoint_delta: Optional[float] = Field(
        None,
        description=(
            "Cooling setpoint change (°C). Negative = cool more."
        ),
    )
    damper_delta_pct: Optional[float] = Field(
        None,
        description="Damper position change (pct points).",
    )
    fan_speed_delta_pct: Optional[float] = Field(
        None,
        description="Fan speed change (pct points).",
    )
    rationale: str = Field(
        ...,
        description="Why this plan was selected.",
        min_length=1,
    )

    model_config = {"frozen": True}


class BuildingActionPlan(BaseModel):
    """Building-wide action directives from the LLM Orchestrator.

    Applied after zone-level plans — zone-level overrides take
    precedence for high-density zones (D2 resolution rule).

    Attributes:
        fan_speed_delta_pct: Building-wide fan speed change.
            Overridden for high-density zones by ZoneActionPlan.
            None = no change.
        rationale: Justification for building-wide directives.
    """

    fan_speed_delta_pct: Optional[float] = Field(
        None, description="Building-wide fan speed change."
    )
    rationale: str = Field(
        ..., description="Justification.", min_length=1
    )

    model_config = {"frozen": True}


class OrchestratorPlan(BaseModel):
    """Full resolved action plan from the LLM Orchestrator.

    Contains per-zone plans plus building-wide directives,
    active policy rules, and resolution metadata for the audit
    trace.

    Attributes:
        zone_plans: Per-zone action plans keyed by zone_id.
        building_plan: Building-wide directives (may be None
            if no building-level changes are warranted).
        active_policy_rules: Human-readable policy rules that
            were enforced as hard constraints (e.g. High-Density
            Occupancy Floor).
        conflict_resolutions: Descriptions of trade-offs resolved
            (one per conflict pair), for the audit trace.
        overall_rationale: Top-level rationale from the LLM
            summarising the full plan in one paragraph.
        resolved_by: Identifies whether plan was resolved by the
            LLM or the rule-based fallback.
    """

    zone_plans: dict[str, ZoneActionPlan] = Field(
        default_factory=dict,
        description="Zone ID -> ZoneActionPlan mapping.",
    )
    building_plan: Optional[BuildingActionPlan] = Field(
        None, description="Building-wide directives."
    )
    active_policy_rules: list[str] = Field(
        default_factory=list,
        description="Hard constraints applied this step.",
    )
    conflict_resolutions: list[str] = Field(
        default_factory=list,
        description="Audit trace entries for resolved conflicts.",
    )
    overall_rationale: str = Field(
        ...,
        description="LLM top-level rationale.",
        min_length=1,
    )
    resolved_by: str = Field(
        default="llm",
        description="'llm' or 'rule_based_fallback'.",
    )

    model_config = {"frozen": True}
