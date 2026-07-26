"""Air Quality Agent — advocates for indoor air quality.

Activates on POOR_AIR_QUALITY events. Proposes ventilation
and fan speed increases to dilute elevated CO₂ concentrations.
In high-density zones the recommendation carries maximum
urgency regardless of tariff conditions.

Architecture reference: Section 4.1 (Agent 3 — Air Quality
Agent), Section 1.2 (Poor Air Quality -> Air Quality Agent).
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


class AirQualityAgent(BaseAgent):
    """Domain agent responsible for indoor air quality.

    Proposes INCREASE_VENTILATION for zones with POOR_AIR_QUALITY
    events. In high-density zones (occupancy > 10) the urgency
    score is boosted to ensure the High-Density Occupancy Floor
    policy is represented strongly to the LLM Orchestrator.
    """

    agent_id: str = "air_quality_agent"

    # Urgency boost applied when High-Density Floor is active.
    HIGH_DENSITY_BOOST: int = 10

    def recommend(
        self,
        state: BuildingState,
        scores: BuildingScoreReport,
        events: EventList,
        context: ContextPriorities,
    ) -> list[Recommendation]:
        """Produce ventilation recommendations for each
        POOR_AIR_QUALITY event.

        Args:
            state: Current building state snapshot.
            scores: Score report from the Score Engine.
            events: Event list from the Event Generator.
            context: Governing objective priorities.

        Returns:
            One or two Recommendations per POOR_AIR_QUALITY event
            (INCREASE_VENTILATION + INCREASE_FAN_SPEED for
            high-density zones).
        """
        recommendations: list[Recommendation] = []

        for event in events.by_type(EventType.POOR_AIR_QUALITY):
            if event.zone_id is None:
                continue
            zone = state.zones.get(event.zone_id)
            if zone is None:
                continue

            aq_score = scores.zone_scores[event.zone_id].air_quality
            base_urgency = _urgency_from_severity_and_priority(
                event.severity.value, context.air_quality
            )

            is_hd = event.zone_id in context.high_density_zones
            urgency = min(
                100,
                base_urgency
                + (self.HIGH_DENSITY_BOOST if is_hd else 0),
            )
            hd_note = (
                " High-Density Floor policy active "
                "(occupancy > 10)."
                if is_hd
                else ""
            )

            recommendations.append(
                Recommendation(
                    agent_id=self.agent_id,
                    target_zone=event.zone_id,
                    action=ActionType.INCREASE_VENTILATION,
                    urgency_score=urgency,
                    rationale=(
                        f"Zone '{event.zone_id}' CO2 is "
                        f"{zone.co2_ppm:.0f} PPM "
                        f"(air quality score: {aq_score:.1f}). "
                        f"Increase outdoor-air damper to dilute "
                        f"CO2.{hd_note} "
                        f"Air quality priority: "
                        f"{context.air_quality.value}."
                    ),
                    supporting_event_types=[event.event_type.value],
                )
            )

            # For high-density zones, also boost fan speed to
            # ensure diluted air circulates quickly.
            if is_hd:
                recommendations.append(
                    Recommendation(
                        agent_id=self.agent_id,
                        target_zone=event.zone_id,
                        action=ActionType.INCREASE_FAN_SPEED,
                        urgency_score=urgency,
                        rationale=(
                            f"High-density zone '{event.zone_id}' "
                            f"({zone.occupancy_count} occupants): "
                            f"boost fan speed to accelerate CO2 "
                            f"dilution alongside damper increase."
                        ),
                        supporting_event_types=[
                            event.event_type.value
                        ],
                    )
                )

        return recommendations
