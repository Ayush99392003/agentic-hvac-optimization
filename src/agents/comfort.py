"""Comfort Agent — advocates for occupant thermal comfort.

Activates on ZONE_OVERHEATING events. For each uncomfortable
occupied zone, proposes INCREASE_COOLING (warm) or
INCREASE_HEATING (cool). Urgency scales with severity and the
governing comfort context priority.

Architecture reference: Section 4.1 (Agent 1 — Comfort Agent),
Section 1.2 (Overheating -> Comfort Agent).
"""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.agents.models import (
    ActionType,
    ContextPriorities,
    Recommendation,
    _urgency_from_severity_and_priority,
)
from src.engine.scores import BuildingScoreReport
from src.events.models import EventList, EventType
from src.state.building import BuildingState


class ComfortAgent(BaseAgent):
    """Domain agent responsible for occupant thermal comfort.

    Monitors ZONE_OVERHEATING events and proposes setpoint
    adjustments to return PMV toward the ASHRAE-55 acceptable
    band (±0.5). Does not propose energy-reducing actions —
    those belong to the Energy Agent.
    """

    agent_id: str = "comfort_agent"

    def recommend(
        self,
        state: BuildingState,
        scores: BuildingScoreReport,
        events: EventList,
        context: ContextPriorities,
    ) -> list[Recommendation]:
        """Produce cooling or heating recommendations for each
        ZONE_OVERHEATING event.

        Args:
            state: Current building state snapshot.
            scores: Score report from the Score Engine.
            events: Event list from the Event Generator.
            context: Governing objective priorities.

        Returns:
            One Recommendation per ZONE_OVERHEATING event.
        """
        recommendations: list[Recommendation] = []

        for event in events.by_type(EventType.ZONE_OVERHEATING):
            if event.zone_id is None:
                continue
            zone = state.zones.get(event.zone_id)
            if zone is None:
                continue

            # Warm zone -> cool it; cool zone -> heat it
            if zone.pmv > 0:
                action = ActionType.INCREASE_COOLING
                direction = "cooling"
            else:
                action = ActionType.INCREASE_HEATING
                direction = "heating"

            urgency = _urgency_from_severity_and_priority(
                event.severity.value, context.comfort
            )

            recommendations.append(
                Recommendation(
                    agent_id=self.agent_id,
                    target_zone=event.zone_id,
                    action=action,
                    urgency_score=urgency,
                    rationale=(
                        f"Zone '{event.zone_id}' PMV={zone.pmv:+.2f} "
                        f"(comfort score: "
                        f"{scores.zone_scores[event.zone_id].comfort:.1f}). "
                        f"Increase {direction} to return PMV toward "
                        f"ASHRAE-55 band. "
                        f"Comfort priority: {context.comfort.value}."
                    ),
                    supporting_event_types=[event.event_type.value],
                )
            )

        return recommendations
