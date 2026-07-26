"""Energy Agent — advocates for energy efficiency.

Activates on ZONE_UNDERUTILIZED and PEAK_DEMAND_RISK events.
Proposes setpoint relaxation for empty zones and demand-shedding
fan speed reductions during peak tariff windows.

Architecture reference: Section 4.1 (Agent 2 — Energy Agent).
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


class EnergyAgent(BaseAgent):
    """Domain agent responsible for energy efficiency.

    Proposes setpoint relaxation for unoccupied zones and
    fan-speed reductions during peak demand windows. Does not
    override High-Density Occupancy Floor decisions — the LLM
    Orchestrator handles that trade-off.
    """

    agent_id: str = "energy_agent"

    def recommend(
        self,
        state: BuildingState,
        scores: BuildingScoreReport,
        events: EventList,
        context: ContextPriorities,
    ) -> list[Recommendation]:
        """Produce energy-shedding recommendations.

        One RELAX_SETPOINT per ZONE_UNDERUTILIZED event and one
        DECREASE_FAN_SPEED per PEAK_DEMAND_RISK event.

        Args:
            state: Current building state snapshot.
            scores: Score report from the Score Engine.
            events: Event list from the Event Generator.
            context: Governing objective priorities.

        Returns:
            List of energy-saving Recommendation objects.
        """
        recommendations: list[Recommendation] = []

        # --- Underutilized zones: relax setpoints ---
        for event in events.by_type(EventType.ZONE_UNDERUTILIZED):
            if event.zone_id is None:
                continue
            zone = state.zones.get(event.zone_id)
            if zone is None or zone.is_occupied:
                continue

            urgency = _urgency_from_severity_and_priority(
                event.severity.value, context.energy
            )
            recommendations.append(
                Recommendation(
                    agent_id=self.agent_id,
                    target_zone=event.zone_id,
                    action=ActionType.RELAX_SETPOINT,
                    urgency_score=urgency,
                    rationale=(
                        f"Zone '{event.zone_id}' is unoccupied. "
                        f"Relaxing HVAC setpoint saves energy "
                        f"with zero occupant impact. "
                        f"Energy priority: {context.energy.value}."
                    ),
                    supporting_event_types=[event.event_type.value],
                )
            )

        # --- Peak demand: reduce fan speed building-wide ---
        for event in events.by_type(EventType.PEAK_DEMAND_RISK):
            urgency = _urgency_from_severity_and_priority(
                event.severity.value, context.energy
            )
            recommendations.append(
                Recommendation(
                    agent_id=self.agent_id,
                    target_zone=None,
                    action=ActionType.DECREASE_FAN_SPEED,
                    urgency_score=urgency,
                    rationale=(
                        f"Building energy score is "
                        f"{scores.energy:.1f}. "
                        f"Tariff: ${state.energy_price:.3f}/kWh. "
                        f"Reducing fan speed curtails peak demand. "
                        f"Energy priority: {context.energy.value}."
                    ),
                    supporting_event_types=[event.event_type.value],
                )
            )

        return recommendations
