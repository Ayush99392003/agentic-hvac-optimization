"""Carbon Agent — advocates for carbon footprint reduction.

Activates on HIGH_CARBON_WINDOW events. Proposes SHIFT_LOAD
to defer non-essential HVAC energy consumption to cleaner grid
windows. Does not propose shedding of comfort-critical loads —
that conflict is resolved by the LLM Orchestrator.

Architecture reference: Section 4.1 (Agent 4 — Carbon Agent),
Section 1.2 (High Carbon -> Carbon Agent).
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


class CarbonAgent(BaseAgent):
    """Domain agent responsible for grid carbon footprint.

    Proposes SHIFT_LOAD during high-carbon grid windows to
    defer non-critical HVAC energy draw. Does not interfere
    with comfort-critical zones — the orchestrator decides.
    """

    agent_id: str = "carbon_agent"

    def recommend(
        self,
        state: BuildingState,
        scores: BuildingScoreReport,
        events: EventList,
        context: ContextPriorities,
    ) -> list[Recommendation]:
        """Produce load-shifting recommendations for each
        HIGH_CARBON_WINDOW event.

        Args:
            state: Current building state snapshot.
            scores: Score report from the Score Engine.
            events: Event list from the Event Generator.
            context: Governing objective priorities.

        Returns:
            One SHIFT_LOAD Recommendation per HIGH_CARBON_WINDOW
            event.
        """
        recommendations: list[Recommendation] = []

        for event in events.by_type(EventType.HIGH_CARBON_WINDOW):
            urgency = _urgency_from_severity_and_priority(
                event.severity.value, context.carbon
            )
            recommendations.append(
                Recommendation(
                    agent_id=self.agent_id,
                    target_zone=None,
                    action=ActionType.SHIFT_LOAD,
                    urgency_score=urgency,
                    rationale=(
                        f"Grid carbon intensity is "
                        f"{state.carbon_intensity:.0f} gCO2e/kWh "
                        f"(carbon score: {scores.carbon:.1f}). "
                        f"Shift non-essential HVAC loads to a "
                        f"cleaner grid window. "
                        f"Carbon priority: {context.carbon.value}."
                    ),
                    supporting_event_types=[event.event_type.value],
                )
            )

        return recommendations
