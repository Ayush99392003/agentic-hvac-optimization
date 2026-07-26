"""Events package for the HVAC optimization pipeline.

Exports the Event type system and the public generate_events()
entry point consumed by the LLM Orchestrator (Phase 5).
"""

from src.events.generator import generate_events
from src.events.models import (
    CRITICAL_SCORE,
    HIGH_SCORE,
    MEDIUM_SCORE,
    Event,
    EventList,
    EventSeverity,
    EventType,
)

__all__ = [
    "Event",
    "EventList",
    "EventType",
    "EventSeverity",
    "CRITICAL_SCORE",
    "HIGH_SCORE",
    "MEDIUM_SCORE",
    "generate_events",
]
