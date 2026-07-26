"""Simulation interface models — Phase 7 Feedback Loop.

Defines the data types flowing between the Action Engine and the
Simulation Layer: the stepped simulation result (what actually
happened after commands were applied) and the confidence/feedback
state that the pipeline uses to update agent weights.

Architecture reference: Section 7 (Simulation & Feedback Loop),
Section 2.2 (Audit Trail — expected vs. actual comparison).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ZoneSimResult(BaseModel):
    """Observed zone state after one simulation step.

    Produced by whichever simulator backend is active. Used by
    the FeedbackEngine to compare against the expected state the
    Orchestrator intended.

    Attributes:
        zone_id: Zone identifier.
        temperature: Observed zone temperature (°C).
        pmv: Observed PMV.
        co2_ppm: Observed CO₂ concentration (PPM).
        cooling_setpoint_achieved: Setpoint actually reached (°C).
            May differ from commanded value due to thermal lag.
        power_kw: Observed zone-level power draw (kW).
    """

    zone_id: str = Field(..., description="Zone identifier.")
    temperature: float = Field(..., description="Observed temp (°C).")
    pmv: float = Field(..., description="Observed PMV.")
    co2_ppm: float = Field(..., description="Observed CO2 (PPM).")
    cooling_setpoint_achieved: float = Field(
        ..., description="Setpoint reached (°C)."
    )
    power_kw: float = Field(..., description="Observed power (kW).")

    model_config = {"frozen": True}


class SimStepResult(BaseModel):
    """Full simulation result for one timestep.

    Produced by `SimulatorProtocol.step()` and consumed by the
    FeedbackEngine.

    Attributes:
        timestamp: Simulation clock time for this result.
        zone_results: Per-zone observed state.
        total_power_kw: Total building power draw (kW).
        simulator_id: ID of the backend that produced this result
            ('replayed' or 'energyplus').
    """

    timestamp: datetime = Field(
        ..., description="Simulation clock time."
    )
    zone_results: dict[str, ZoneSimResult] = Field(
        default_factory=dict,
        description="Zone ID -> ZoneSimResult.",
    )
    total_power_kw: float = Field(
        ..., description="Total building power draw (kW)."
    )
    simulator_id: str = Field(
        default="replayed",
        description="Backend that produced this result.",
    )

    model_config = {"frozen": True}


class ZoneDelta(BaseModel):
    """Expected-vs-actual comparison for one zone.

    Attributes:
        zone_id: Zone identifier.
        temp_expected: Temperature the pipeline intended to achieve.
        temp_actual: Temperature the simulation observed.
        temp_error: Signed error (actual - expected).
        co2_expected: CO₂ the pipeline expected.
        co2_actual: CO₂ the simulation observed.
        setpoint_tracking_error: Difference between commanded
            setpoint and achieved setpoint (°C).
        comfort_score_delta: Comfort score change this step.
    """

    zone_id: str
    temp_expected: float
    temp_actual: float
    temp_error: float
    co2_expected: float
    co2_actual: float
    setpoint_tracking_error: float
    comfort_score_delta: float

    model_config = {"frozen": True}


class FeedbackReport(BaseModel):
    """Audit trail entry for one complete pipeline cycle.

    Produced by the FeedbackEngine and stored in the session
    history. Contains the full expected-vs-actual comparison
    required by §2.2 of the architecture doc.

    Attributes:
        cycle_index: Sequential pipeline cycle number.
        timestamp: Wall-clock time of this cycle.
        zone_deltas: Per-zone expected-vs-actual comparisons.
        total_power_expected_kw: Building power the pipeline
            intended to draw (from the pre-action state).
        total_power_actual_kw: Building power observed after step.
        energy_savings_pct: Percentage power reduction achieved.
        policy_rules_active: Hard constraints that were active
            this cycle (from the OrchestratorPlan).
        conflict_resolutions: Conflict resolution audit entries.
        overall_summary: Human-readable one-line cycle summary.
    """

    cycle_index: int = Field(..., description="Cycle sequence number.")
    timestamp: datetime = Field(
        ..., description="Wall-clock time."
    )
    zone_deltas: dict[str, ZoneDelta] = Field(
        default_factory=dict,
        description="Zone ID -> ZoneDelta.",
    )
    total_power_expected_kw: float
    total_power_actual_kw: float
    energy_savings_pct: float = Field(
        ..., description="(expected - actual) / expected * 100."
    )
    policy_rules_active: list[str] = Field(
        default_factory=list
    )
    conflict_resolutions: list[str] = Field(
        default_factory=list
    )
    overall_summary: str = Field(
        ..., description="One-line cycle summary."
    )

    model_config = {"frozen": True}
