"""Event models for the HVAC optimization pipeline.

Each Event represents a high-level semantic condition detected
from the Score Engine output (Phase 2). Events drive agent
activation in Phase 4 — agents reason about events, not raw
numbers.

Architecture reference: Section 3 (Event Generator node),
Section 1.2 (Ontology Events table).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """High-level semantic events detectable in a building.

    Each event maps to one or more domain agents that are
    activated to propose recommendations (see architecture
    Section 1.2, Ontological Relationships table).

    Attributes:
        ZONE_OVERHEATING: A zone's thermal comfort score has
            dropped below the critical threshold while occupied.
            Activates: Comfort Agent.
        ZONE_UNDERUTILIZED: An unoccupied zone is receiving
            significant HVAC conditioning, wasting energy.
            Activates: Energy Agent.
        POOR_AIR_QUALITY: Indoor CO₂ has degraded beyond the
            acceptable threshold.
            Activates: Air Quality Agent.
        PEAK_DEMAND_RISK: Building-wide energy score indicates
            power draw approaching or exceeding peak demand
            limits during a high-tariff window.
            Activates: Energy Agent, Demand Response Agent.
        HIGH_CARBON_WINDOW: The grid is currently running on
            high-carbon sources; deferring or shedding load
            reduces the carbon footprint.
            Activates: Carbon Agent.
        COOLING_INEFFICIENT: A zone is both thermally
            uncomfortable and consuming excess energy — cooling
            is running hard but not achieving comfort.
            Activates: Comfort Agent, Energy Agent.
    """

    ZONE_OVERHEATING = "ZONE_OVERHEATING"
    ZONE_UNDERUTILIZED = "ZONE_UNDERUTILIZED"
    POOR_AIR_QUALITY = "POOR_AIR_QUALITY"
    PEAK_DEMAND_RISK = "PEAK_DEMAND_RISK"
    HIGH_CARBON_WINDOW = "HIGH_CARBON_WINDOW"
    COOLING_INEFFICIENT = "COOLING_INEFFICIENT"


class EventSeverity(str, Enum):
    """Severity classification of a detected event.

    Severity determines how urgently an agent should respond and
    is reflected in the ``urgency_score`` it assigns to its
    resulting recommendation (architecture Section 4.2).

    Attributes:
        LOW: Informational — monitor but no immediate action.
        MEDIUM: Action recommended in the near term.
        HIGH: Prompt corrective action required.
        CRITICAL: Immediate action required; human health or
            equipment safety may be at risk.
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# Severity thresholds used by the Event Generator.
# Scores below CRITICAL_SCORE trigger CRITICAL severity,
# scores below HIGH_SCORE trigger HIGH severity, etc.
CRITICAL_SCORE: float = 30.0
HIGH_SCORE: float = 55.0
MEDIUM_SCORE: float = 75.0


def _severity_from_score(score: float) -> EventSeverity:
    """Derive EventSeverity from a 0-100 score.

    Lower score = more severe.

    Args:
        score: A 0-100 score from the Score Engine.

    Returns:
        The corresponding EventSeverity enum member.
    """
    if score <= CRITICAL_SCORE:
        return EventSeverity.CRITICAL
    if score <= HIGH_SCORE:
        return EventSeverity.HIGH
    if score <= MEDIUM_SCORE:
        return EventSeverity.MEDIUM
    return EventSeverity.LOW


class Event(BaseModel):
    """A single detected building condition event.

    Events are produced by the Event Generator and consumed by
    the LLM Orchestrator, which uses them to activate the
    relevant domain agents. Events appear verbatim in the
    decision audit trace.

    Attributes:
        event_id: Unique UUID string for this event instance.
        event_type: Semantic category of the condition.
        severity: Urgency level of the event.
        zone_id: The affected zone identifier, or None for
            building-wide events (e.g. PEAK_DEMAND_RISK).
        trigger_score: The 0-100 score that triggered the event,
            providing a numeric anchor for the audit trace.
        trigger_metric: Human-readable name of the metric that
            crossed the threshold (e.g. ``"comfort_score"``).
        trigger_value: The raw metric value at detection time
            (e.g. PMV=1.1, CO₂=1150 PPM).
        description: Human-readable one-sentence summary of
            what was detected and why it matters.
        timestamp: UTC time when the event was detected.
    """

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique event instance identifier.",
    )
    event_type: EventType = Field(
        ..., description="Semantic event category."
    )
    severity: EventSeverity = Field(
        ..., description="Urgency level."
    )
    zone_id: Optional[str] = Field(
        None,
        description=(
            "Affected zone ID, or None for building-wide events."
        ),
    )
    trigger_score: float = Field(
        ...,
        description="Score that crossed the detection threshold.",
        ge=0.0,
        le=100.0,
    )
    trigger_metric: str = Field(
        ...,
        description="Name of the score / metric that triggered.",
    )
    trigger_value: float = Field(
        ...,
        description="Raw sensor or computed metric value.",
    )
    description: str = Field(
        ...,
        description="Human-readable event summary.",
        min_length=1,
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="UTC timestamp of event detection.",
    )

    model_config = {"frozen": True}


class EventList(BaseModel):
    """Ordered collection of events detected for one timestep.

    Passed alongside BuildingState and BuildingScoreReport to the
    LLM Orchestrator, which reads events to select and activate
    the relevant domain agents.

    Attributes:
        events: All events detected at this simulation step.
        timestamp: UTC timestamp of the detection pass.
    """

    events: list[Event] = Field(
        default_factory=list,
        description="Detected events for this timestep.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="UTC timestamp of the detection pass.",
    )

    model_config = {"frozen": True}

    def by_type(self, event_type: EventType) -> list[Event]:
        """Filter events by type.

        Args:
            event_type: The EventType to filter by.

        Returns:
            List of matching Event instances.
        """
        return [e for e in self.events if e.event_type == event_type]

    def by_severity(self, severity: EventSeverity) -> list[Event]:
        """Filter events by severity.

        Args:
            severity: The EventSeverity to filter by.

        Returns:
            List of matching Event instances.
        """
        return [e for e in self.events if e.severity == severity]

    def by_zone(self, zone_id: str) -> list[Event]:
        """Return all events affecting a specific zone.

        Args:
            zone_id: Zone identifier to filter by.

        Returns:
            List of matching Event instances.
        """
        return [e for e in self.events if e.zone_id == zone_id]

    @property
    def has_critical(self) -> bool:
        """Return True if any CRITICAL severity event is present.

        Returns:
            True if at least one event has CRITICAL severity.
        """
        return any(
            e.severity == EventSeverity.CRITICAL
            for e in self.events
        )

    @property
    def active_event_types(self) -> set[EventType]:
        """Return the set of EventType values present.

        Returns:
            Set of all distinct EventType values in this list.
        """
        return {e.event_type for e in self.events}
