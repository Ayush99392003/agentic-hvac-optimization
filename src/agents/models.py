"""Recommendation and related models for the Agent Layer.

Each domain agent (Phase 4) outputs a list of Recommendation
objects. These are consumed by the LLM Orchestrator (Phase 5)
which resolves conflicts and synthesises the final action plan.

Architecture reference: Section 4.2 (Inter-Agent Communication
Protocol), Section 1.2 (Actions ontology).
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Typed vocabulary of actions an agent may propose.

    Agents only propose actions that fall within their domain
    (see architecture Section 1.2 Ontological Relationships
    table). The Action Engine (Phase 6) translates these into
    concrete setpoint commands.

    Attributes:
        INCREASE_COOLING: Lower the zone cooling setpoint.
        DECREASE_COOLING: Raise the zone cooling setpoint
            (reduce cooling intensity).
        INCREASE_HEATING: Raise the zone heating setpoint.
        DECREASE_HEATING: Lower the zone heating setpoint.
        INCREASE_VENTILATION: Open outdoor-air damper further /
            increase fresh air fraction.
        DECREASE_VENTILATION: Reduce outdoor-air damper position.
        INCREASE_FAN_SPEED: Raise AHU fan speed percentage.
        DECREASE_FAN_SPEED: Lower AHU fan speed percentage.
        RELAX_SETPOINT: Widen the deadband / raise the cooling
            setpoint in unoccupied or low-priority zones.
        PRE_COOL: Pre-cool a zone ahead of occupancy or peak
            tariff window.
        SHIFT_LOAD: Defer HVAC energy draw to a lower-carbon
            or lower-tariff window.
        SHED_LOAD: Curtail non-critical HVAC loads immediately.
    """

    INCREASE_COOLING = "INCREASE_COOLING"
    DECREASE_COOLING = "DECREASE_COOLING"
    INCREASE_HEATING = "INCREASE_HEATING"
    DECREASE_HEATING = "DECREASE_HEATING"
    INCREASE_VENTILATION = "INCREASE_VENTILATION"
    DECREASE_VENTILATION = "DECREASE_VENTILATION"
    INCREASE_FAN_SPEED = "INCREASE_FAN_SPEED"
    DECREASE_FAN_SPEED = "DECREASE_FAN_SPEED"
    RELAX_SETPOINT = "RELAX_SETPOINT"
    PRE_COOL = "PRE_COOL"
    SHIFT_LOAD = "SHIFT_LOAD"
    SHED_LOAD = "SHED_LOAD"


class ObjectivePriority(str, Enum):
    """Context Engine objective priority label.

    Set by the Context Engine (between Event Generator and LLM
    Orchestrator) based on occupancy, tariff window, and
    safety rules. Agents receive these labels alongside events
    to inform their urgency_score calculation.

    Attributes:
        HIGH: Objective is non-negotiable at this timestep.
        MEDIUM: Objective is important but can be traded off.
        LOW: Objective is a background preference.
    """

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ContextPriorities(BaseModel):
    """Governing objective priorities set by the Context Engine.

    The LLM Orchestrator uses these alongside agent urgency
    scores to resolve trade-offs: a HIGH-priority objective
    takes precedence over lower-priority ones even if the
    individual agent's urgency_score is not the highest.

    Attributes:
        comfort: Priority level for thermal comfort objective.
        air_quality: Priority level for indoor air quality.
        energy: Priority level for energy efficiency.
        carbon: Priority level for carbon footprint reduction.
        high_density_zones: Set of zone IDs currently under
            the High-Density Occupancy Floor policy
            (occupancy > 10 -> comfort + air_quality = HIGH).
    """

    comfort: ObjectivePriority = Field(
        ObjectivePriority.MEDIUM,
        description="Thermal comfort objective priority.",
    )
    air_quality: ObjectivePriority = Field(
        ObjectivePriority.MEDIUM,
        description="Indoor air quality objective priority.",
    )
    energy: ObjectivePriority = Field(
        ObjectivePriority.MEDIUM,
        description="Energy efficiency objective priority.",
    )
    carbon: ObjectivePriority = Field(
        ObjectivePriority.LOW,
        description="Carbon footprint objective priority.",
    )
    high_density_zones: frozenset[str] = Field(
        default_factory=frozenset,
        description=(
            "Zone IDs with High-Density Floor policy active."
        ),
    )

    model_config = {"frozen": True}

    @classmethod
    def from_building_state(
        cls,
        state: object,  # BuildingState — avoid circular import
        *,
        avg_tariff: float,
    ) -> "ContextPriorities":
        """Derive context priorities from building state.

        Applies the High-Density Occupancy Floor policy and
        peak tariff detection deterministically.

        Args:
            state: A BuildingState instance.
            avg_tariff: Historical average tariff ($/kWh).

        Returns:
            A ContextPriorities instance.
        """
        from src.state.building import BuildingState

        assert isinstance(state, BuildingState)

        high_density = frozenset(
            zid
            for zid, z in state.zones.items()
            if z.is_high_density
        )

        # High-Density Floor: comfort + air_quality -> HIGH
        comfort_p = (
            ObjectivePriority.HIGH
            if high_density
            else ObjectivePriority.MEDIUM
        )
        air_p = (
            ObjectivePriority.HIGH
            if high_density
            else ObjectivePriority.MEDIUM
        )

        # Peak tariff: energy shedding priority -> HIGH
        is_peak = state.energy_price > avg_tariff * 1.5
        energy_p = (
            ObjectivePriority.HIGH
            if is_peak
            else ObjectivePriority.MEDIUM
        )

        # High carbon window: carbon -> HIGH
        carbon_p = (
            ObjectivePriority.HIGH
            if state.is_high_carbon_window
            else ObjectivePriority.LOW
        )

        return cls(
            comfort=comfort_p,
            air_quality=air_p,
            energy=energy_p,
            carbon=carbon_p,
            high_density_zones=high_density,
        )


class Recommendation(BaseModel):
    """A single action recommendation from a domain agent.

    The LLM Orchestrator collects all recommendations from
    activated agents, evaluates urgency_score against context
    priorities, and resolves conflicts to produce a final
    action plan.

    Attributes:
        recommendation_id: Unique UUID for this recommendation.
        agent_id: Identifier of the proposing agent.
        target_zone: Zone the action targets, or None for
            building-wide recommendations.
        target_equipment: Equipment ID targeted, or None.
        action: The proposed ActionType.
        urgency_score: Integer 1-100 expressing how urgently
            this action is needed. Derived deterministically
            from the triggering event's severity and the
            relevant objective's context priority.
        rationale: Human-readable explanation linking the
            triggering metric to the proposed action.
        supporting_event_types: Event types that motivated
            this recommendation (for audit trace).
    """

    recommendation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique recommendation identifier.",
    )
    agent_id: str = Field(
        ..., description="Proposing agent identifier."
    )
    target_zone: Optional[str] = Field(
        None, description="Target zone ID."
    )
    target_equipment: Optional[str] = Field(
        None, description="Target equipment ID."
    )
    action: ActionType = Field(
        ..., description="Proposed action."
    )
    urgency_score: int = Field(
        ...,
        description="Urgency score (1-100).",
        ge=1,
        le=100,
    )
    rationale: str = Field(
        ...,
        description="Human-readable justification.",
        min_length=1,
    )
    supporting_event_types: list[str] = Field(
        default_factory=list,
        description="Event types that motivated this action.",
    )

    model_config = {"frozen": True}


def _urgency_from_severity_and_priority(
    severity_value: str,
    priority: ObjectivePriority,
    base_ranges: dict[str, tuple[int, int]] | None = None,
) -> int:
    """Compute urgency_score from event severity and priority.

    Severity sets the base range; context priority adjusts
    within that range.

    Base ranges (severity -> [low, high]):
        CRITICAL -> [75, 90]
        HIGH     -> [55, 74]
        MEDIUM   -> [35, 54]
        LOW      -> [10, 34]

    Priority adjustment:
        HIGH   -> +8
        MEDIUM -> +0
        LOW    -> -8

    Args:
        severity_value: EventSeverity string value.
        priority: Context objective priority.
        base_ranges: Override for severity base midpoints (tests).

    Returns:
        Urgency score clamped to [1, 100].
    """
    midpoints: dict[str, int] = {
        "CRITICAL": 82,
        "HIGH": 64,
        "MEDIUM": 44,
        "LOW": 22,
    }
    priority_adj: dict[str, int] = {
        "HIGH": 8,
        "MEDIUM": 0,
        "LOW": -8,
    }
    base = midpoints.get(severity_value, 44)
    adj = priority_adj.get(priority.value, 0)
    return max(1, min(100, base + adj))
