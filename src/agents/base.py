"""Abstract base class for all domain agents.

Every specialized agent (Comfort, Energy, Air Quality, Carbon,
Demand Response) must subclass BaseAgent and implement the
`recommend` method. The LLM Orchestrator calls each activated
agent through this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.agents.models import ContextPriorities, Recommendation
from src.engine.scores import BuildingScoreReport
from src.events.models import EventList
from src.state.building import BuildingState


class BaseAgent(ABC):
    """Abstract domain agent that produces action recommendations.

    Subclasses must define ``agent_id`` and implement
    ``recommend()``. No agent may interact directly with
    EnergyPlus or issue hardware commands.

    Attributes:
        agent_id: Unique string identifier for the agent,
            embedded in every Recommendation it produces.
    """

    agent_id: str

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Enforce that subclasses declare ``agent_id``."""
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "agent_id") or not isinstance(
            getattr(cls, "agent_id"), str
        ):
            raise TypeError(
                f"{cls.__name__} must declare a string "
                f"class attribute 'agent_id'."
            )

    @abstractmethod
    def recommend(
        self,
        state: BuildingState,
        scores: BuildingScoreReport,
        events: EventList,
        context: ContextPriorities,
    ) -> list[Recommendation]:
        """Produce a list of action recommendations.

        Args:
            state: Current building state snapshot.
            scores: Score report from the Score Engine.
            events: Event list from the Event Generator.
            context: Governing objective priorities from the
                Context Engine.

        Returns:
            A (possibly empty) list of Recommendation objects.
            Agents return an empty list when they have no
            relevant events to act on.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(agent_id={self.agent_id!r})"
