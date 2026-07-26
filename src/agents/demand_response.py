"""Demand Response Agent — advocates for peak demand shedding.

Activates on PEAK_DEMAND_RISK events. Proposes SHED_LOAD for
non-critical zones and PRE_COOL for comfort-critical zones
before the peak window deepens.

Architecture reference: Section 4.1 (Agent 5 — Demand Response
Agent), Section 1.2 (Peak Demand Risk -> DR Agent).
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


class DemandResponseAgent(BaseAgent):
    """Domain agent responsible for peak demand management.

    Proposes SHED_LOAD for unoccupied zones and PRE_COOL for
    high-density zones to reduce demand before peak prices
    intensify. Conflict with comfort is intentional — the LLM
    Orchestrator arbitrates using context priorities.
    """

    agent_id: str = "demand_response_agent"

    def recommend(
        self,
        state: BuildingState,
        scores: BuildingScoreReport,
        events: EventList,
        context: ContextPriorities,
    ) -> list[Recommendation]:
        """Produce demand-shedding and pre-cooling recommendations.

        For each PEAK_DEMAND_RISK event:
          - SHED_LOAD for unoccupied zones.
          - PRE_COOL for high-density zones (deferred cooling
            before peak deepens, rather than during).

        Args:
            state: Current building state snapshot.
            scores: Score report from the Score Engine.
            events: Event list from the Event Generator.
            context: Governing objective priorities.

        Returns:
            List of demand-response Recommendation objects.
        """
        recommendations: list[Recommendation] = []

        peak_events = events.by_type(EventType.PEAK_DEMAND_RISK)
        if not peak_events:
            return recommendations

        # Use the first (and typically only) peak event.
        peak_event = peak_events[0]
        base_urgency = _urgency_from_severity_and_priority(
            peak_event.severity.value, context.energy
        )

        for zid, zone in state.zones.items():
            if not zone.is_occupied:
                # Unoccupied zone: full demand shed
                recommendations.append(
                    Recommendation(
                        agent_id=self.agent_id,
                        target_zone=zid,
                        action=ActionType.SHED_LOAD,
                        urgency_score=base_urgency,
                        rationale=(
                            f"Zone '{zid}' is unoccupied during "
                            f"peak demand window "
                            f"(${state.energy_price:.3f}/kWh). "
                            f"Shed HVAC load immediately with "
                            f"zero occupant impact."
                        ),
                        supporting_event_types=[
                            peak_event.event_type.value
                        ],
                    )
                )
            elif zone.is_high_density:
                # High-density zone: propose pre-cool instead of
                # shed — do not override comfort floor.
                recommendations.append(
                    Recommendation(
                        agent_id=self.agent_id,
                        target_zone=zid,
                        action=ActionType.PRE_COOL,
                        urgency_score=max(1, base_urgency - 15),
                        rationale=(
                            f"High-density zone '{zid}' "
                            f"({zone.occupancy_count} occupants) "
                            f"during peak demand window. "
                            f"Propose pre-cooling rather than load "
                            f"shed to preserve occupant comfort. "
                            f"LLM Orchestrator resolves final "
                            f"trade-off."
                        ),
                        supporting_event_types=[
                            peak_event.event_type.value
                        ],
                    )
                )

        return recommendations
