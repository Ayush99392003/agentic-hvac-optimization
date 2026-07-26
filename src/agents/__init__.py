"""Agents package — all five domain agents.

Exports the Recommendation / ContextPriorities types, all
agent classes, and a convenience function to run all agents
and collect their recommendations in one call.
"""

from __future__ import annotations

from src.agents.air_quality import AirQualityAgent
from src.agents.base import BaseAgent
from src.agents.carbon import CarbonAgent
from src.agents.comfort import ComfortAgent
from src.agents.demand_response import DemandResponseAgent
from src.agents.energy import EnergyAgent
from src.agents.models import (
    ActionType,
    ContextPriorities,
    ObjectivePriority,
    Recommendation,
    _urgency_from_severity_and_priority,
)
from src.engine.scores import BuildingScoreReport
from src.events.models import EventList
from src.state.building import BuildingState

# Canonical set of all five domain agents.
ALL_AGENTS: list[BaseAgent] = [
    ComfortAgent(),
    EnergyAgent(),
    AirQualityAgent(),
    CarbonAgent(),
    DemandResponseAgent(),
]


def run_all_agents(
    state: BuildingState,
    scores: BuildingScoreReport,
    events: EventList,
    context: ContextPriorities,
) -> list[Recommendation]:
    """Run all domain agents and collect their recommendations.

    Each agent is run independently. Agents with no relevant
    events return an empty list — zero overhead for inactive
    agents.

    Args:
        state: Current building state snapshot.
        scores: Score report from the Score Engine.
        events: Event list from the Event Generator.
        context: Governing objective priorities.

    Returns:
        Combined list of all Recommendation objects produced by
        all agents, sorted by urgency_score descending.
    """
    all_recs: list[Recommendation] = []
    for agent in ALL_AGENTS:
        recs = agent.recommend(state, scores, events, context)
        all_recs.extend(recs)

    # Sort highest urgency first so the LLM Orchestrator reads
    # the most critical recommendations at the top of its prompt.
    all_recs.sort(key=lambda r: r.urgency_score, reverse=True)
    return all_recs


__all__ = [
    "BaseAgent",
    "ComfortAgent",
    "EnergyAgent",
    "AirQualityAgent",
    "CarbonAgent",
    "DemandResponseAgent",
    "ALL_AGENTS",
    "run_all_agents",
    "Recommendation",
    "ActionType",
    "ContextPriorities",
    "ObjectivePriority",
]
