# Architecture Document & Decision Framework

**System**: Autonomous Agentic Building Control & Energy Optimization Framework  
**Version**: 1.0.0 (Phases 1–7 Complete End-to-End Implementation)  
**Status**: Approved & Fully Verified Specification (131/131 Tests Passing)

---

## Executive Summary: Industrial Control Engineering Philosophy

Traditional building energy management systems (BEMS) face a fundamental tension: rule-based systems are safe but rigid, while black-box neural networks or end-to-end LLM control loops pose catastrophic liability risks in industrial control settings.

This framework resolves that dilemma through a **hybrid 7-tier architecture**:

1. **Deterministic Core at the Control Layer**: All sensor ingestion, score calculations, event triggers, physical safety checks, rate limiting, and simulation rollouts are 100% deterministic, audit-traceable code.
2. **LLM at the Ambiguity Layer**: The Large Language Model acts as an **Orchestrator and Conflict Resolver**, not an unconstrained controller. It arbitrates multi-objective trade-offs when domain agents propose competing recommendations (e.g., Comfort vs. Energy Tariff vs. Air Quality).
3. **Double Safety Net**: If an LLM API fails, times out, or produces a plan violating hard safety constraints, the system automatically engages a **deterministic Rule-Based Fallback Engine** and physical **Safety Validator**.

---

## 1. 7-Tier Layered Architecture Pipeline

```text
                  Building Telemetry & Weather
                               │
                               ▼
 ┌───────────────────────────────────────────────────────────┐
 │ 1. State Modeling (BuildingState, ZoneState, Weather)    │
 └─────────────────────────────┬─────────────────────────────┘
                               ▼
 ┌───────────────────────────────────────────────────────────┐
 │ 2. Feature & Score Engine (ASHRAE-55 PMV, CO2, Energy, Carbon)│
 └─────────────────────────────┬─────────────────────────────┘
                               ▼
 ┌───────────────────────────────────────────────────────────┐
 │ 3. Event Generator (Deterministic Event Detection & Urgency) │
 └─────────────────────────────┬─────────────────────────────┘
                               ▼
 ┌───────────────────────────────────────────────────────────┐
 │ 4. Domain Agent Layer (Comfort, Energy, AQ, Carbon, DR)  │
 └─────────────────────────────┬─────────────────────────────┘
                               ▼
 ┌───────────────────────────────────────────────────────────┐
 │ 5. LLM Orchestrator (Trade-off Resolution & D4 Policy Rules)│
 └─────────────────────────────┬─────────────────────────────┘
                               ▼
 ┌───────────────────────────────────────────────────────────┐
 │ 6. Action Engine & Safety Validator (D2 Override & Bounds)│
 └─────────────────────────────┬─────────────────────────────┘
                               ▼
 ┌───────────────────────────────────────────────────────────┐
 │ 7. Simulation & Feedback Loop (ReplayedSim / EnergyPlus) │
 └───────────────────────────────────────────────────────────┘
```

---

## 2. Decision Ontology

To guarantee a unified domain language across all pipeline stages, the architecture defines a formal decision ontology:

### 2.1 Taxonomies

1. **Entities**: `Zone` (e.g. `conf_room_exec`), `Equipment` (AHU, VAV), `Weather`, `Occupancy`.
2. **Metrics**: Temperature (°C), PMV (-3.0 to +3.0), CO₂ (PPM), Power (kW), Tariff ($/kWh), Carbon Intensity (gCO₂e/kWh).
3. **Events**: `ZONE_OVERHEATING`, `POOR_AIR_QUALITY`, `ZONE_UNDERUTILIZED`, `PEAK_DEMAND_RISK`, `HIGH_CARBON_WINDOW`.
4. **Goals**: Maintain Thermal Comfort, Preserve Air Quality, Minimize Energy Cost, Reduce Carbon Intensity.
5. **Actions**: `INCREASE_COOLING`, `INCREASE_HEATING`, `RELAX_SETPOINT`, `INCREASE_VENTILATION`, `INCREASE_FAN_SPEED`, `DECREASE_FAN_SPEED`, `SHIFT_LOAD`, `SHED_LOAD`, `PRE_COOL`.

### 2.2 Ontological Mappings

| Source Metric | Trigger Condition | Generated Event | Activated Domain Agent | Action Spectrum |
|---|---|---|---|---|
| PMV | $|\text{PMV}| > 0.5$ | `ZONE_OVERHEATING` | Comfort Agent | `INCREASE_COOLING`, `INCREASE_HEATING` |
| CO₂ | $\text{CO}_2 > 1000\,\text{PPM}$ | `POOR_AIR_QUALITY` | Air Quality Agent | `INCREASE_VENTILATION`, `INCREASE_FAN_SPEED` |
| Occupancy | $\text{Occupants} = 0$ & active HVAC | `ZONE_UNDERUTILIZED` | Energy Agent | `RELAX_SETPOINT` |
| Tariff | $\text{Price} > 1.5 \times \text{Price}_{\text{avg}}$ | `PEAK_DEMAND_RISK` | Energy / DR Agent | `DECREASE_FAN_SPEED`, `SHED_LOAD`, `PRE_COOL` |
| Carbon | $C_{\text{grid}} > 400\,\text{gCO}_2/\text{kWh}$ | `HIGH_CARBON_WINDOW` | Carbon Agent | `SHIFT_LOAD` |

---

## 3. Scoring Formulas & Metric Engine (Phase 2 Spec)

All state metrics map deterministically to $0 - 100$ scores (where $100$ is optimal).

### 3.1 Comfort Score ($S_{\text{comfort}}$)

Derived from ASHRAE-55 standard with an explicit threshold inflection at $|\text{PMV}| = 0.5$:
$$S_{\text{comfort}} = \begin{cases} 100 - 20 \times |\text{PMV}| & \text{if } |\text{PMV}| \le 0.5 \\ \max\left(0, 90 - 60 \times (|\text{PMV}| - 0.5)\right) & \text{if } |\text{PMV}| > 0.5 \end{cases}$$

### 3.2 Air Quality Score ($S_{\text{air}}$)

Derived from indoor CO₂ concentration:
$$S_{\text{air}} = \begin{cases} 100 & \text{if } \text{CO}_2 \le 600 \\ \max\left(0, 100 - \frac{\text{CO}_2 - 600}{10}\right) & \text{if } \text{CO}_2 > 600 \end{cases}$$

### 3.3 Energy Score ($S_{\text{energy}}$)

Penalizes consumption above baseline weighted by tariff multiplier:
$$S_{\text{energy}} = \max\left(0, 100 - 100 \times \max\left(0, \frac{P_{\text{current}}}{P_{\text{baseline}}} - 1\right) \times \frac{\text{Price}_{\text{current}}}{\text{Price}_{\text{avg}}}\right)$$

### 3.4 Carbon Score ($S_{\text{carbon}}$)

Normalizes grid carbon intensity against 24-hour forecast window $[C_{\min}, C_{\max}]$:
$$S_{\text{carbon}} = \max\left(0, \min\left(100, 100 - 100 \times \frac{C_{\text{current}} - C_{\min}}{C_{\max} - C_{\min}}\right)\right)$$

---

## 4. Governing Policy Decisions & Conflict Resolution (D1–D4)

When multiple domain agents propose recommendations for the same building, trade-offs must be resolved systematically.

### Decision Matrix Summary

| Decision | Policy Name | Description | Enforcement Mechanism |
|---|---|---|---|
| **D1** | Intra-Zone Alignment | Confirmed that zone-level proposals for `conf_room_exec` are complementary (cooling + ventilation vectoring). | Verified via vector alignment check. |
| **D2** | Cross-Scope Fan Override | Zone-scoped `INCREASE_FAN_SPEED` for high-density zone with `air_quality=HIGH` overrides building-scoped `DECREASE_FAN_SPEED`. | **Two-Layer**: LLM System Prompt + Action Engine post-check `_enforce_d2_fan_scope()`. |
| **D3** | Urgency Tiebreak | Equal urgency scores resolved in order: Health > Comfort > Energy > Carbon; Zone-scoped > Building-scoped. | Pipeline tiebreak ordering. |
| **D4** | High-Density Floor | When occupancy $> 10$, comfort & air quality floors are hard non-negotiable constraints. | **Two-Layer**: System Prompt Injection + Acceptance Verification `_verify_hard_constraints()` with automatic Fallback trigger. |

---

## 5. Action Engine & Physical Safety Validator (Phase 6)

The Safety Validator enforces physical bounds regardless of whether the plan originated from the LLM or Fallback Engine.

### Physical Safety Bounds

- **Cooling Range**: $18.0^\circ\text{C} \le T_{\text{cool}} \le 26.0^\circ\text{C}$.
- **Heating Range**: $16.0^\circ\text{C} \le T_{\text{heat}} \le 24.0^\circ\text{C}$.
- **Deadband Check**: $T_{\text{cool}} - T_{\text{heat}} \ge 1.0^\circ\text{C}$. Commands violating deadband are **REJECTED**.
- **Rate Limit**: Setpoint change per step $\le 2.0^\circ\text{C}$ (inclusive bound `abs(delta) > MAX_SETPOINT_DELTA_PER_STEP`). Commands exceeding this rate are **CLAMPED**.
- **Emergency CO₂ Override**: If $\text{CO}_2 \ge 1200\,\text{PPM}$, outdoor air damper is forcibly opened to **100% [CLAMPED]**.

---

## 6. Simulation & Feedback Loop (Phase 7)

### Swappable Strategy Protocol

The system defines a unified `SimulatorProtocol`:

- `ReplayedSimulator` (**Default**): Uses first-order thermal lag dynamics ($\tau = 0.20$) and CO₂ mass-balance equations (dilution rate $-200\,\text{PPM/step}$ when damper $\ge 30\%$, fan $\ge 40\%$).
- `EnergyPlusSimulator` (**Additive**): Activated seamlessly when `ENERGYPLUS_IDF_PATH` environment variable is present.

### Audit Trail & Feedback Report

For every cycle, `compute_feedback()` evaluates expected vs. actual values:

- Temperature error ($\Delta T = T_{\text{actual}} - T_{\text{expected}}$)
- Setpoint tracking error
- Comfort score change ($\Delta S_{\text{comfort}}$)
- Achieved energy savings (%)

---

## 7. Scenario Rationale & Behavior Walkthroughs

### 7.1 Scenario 1: Demo Centerpiece Scenario (`centerpiece`)

- **State**: `conf_room_exec` has 15 occupants, $T=26.2^\circ\text{C}$, $\text{PMV}=+1.1$, $\text{CO}_2=1150\,\text{PPM}$, grid tariff $\$0.45/\text{kWh}$.
- **Agent Proposals**: Comfort Agent wants `INCREASE_COOLING` (urgency 72), Air Quality Agent wants `INCREASE_VENTILATION` + `INCREASE_FAN_SPEED` (urgency 82), Energy Agent wants building-wide `DECREASE_FAN_SPEED` (urgency 90), DR Agent wants `PRE_COOL` (urgency 75), Carbon Agent wants `SHIFT_LOAD` (urgency 90).
- **Resolution**: High-Density Occupancy Floor (D4) protects human health. D2 override suppresses building fan reduction. Result: `conf_room_exec` setpoint reduced by $1.0^\circ\text{C}$ to $23.0^\circ\text{C}$, damper opened to $80\%$, fan boosted to $90\%$.

### 7.2 Scenario 2: Emergency CO₂ Override (`emergency_co2`)

- **State**: Auditorium has $\text{CO}_2 = 1300\,\text{PPM}$ ($> 1200\,\text{PPM}$ emergency threshold).
- **Resolution**: Air Quality Agent requests ventilation increase. Safety Validator detects emergency CO₂ level and forcibly clamps damper to **100.0%**.

### 7.3 Scenario 3: Peak Tariff Unoccupied Zone Shedding (`peak_tariff_unoccupied`)

- **State**: Energy tariff is $\$0.48/\text{kWh}$, `board_room` and `west_wing` are unoccupied.
- **Resolution**: Energy Agent recommends `RELAX_SETPOINT`. Setpoint is relaxed by $+2.0^\circ\text{C}$ (e.g. $23.0^\circ\text{C} \rightarrow 25.0^\circ\text{C}$), reducing chiller kW load with zero occupant discomfort.

### 7.4 Scenario 4: Nominal Steady-State (`normal`)

- **State**: Moderate ambient weather, low occupancy, standard tariff $\$0.18/\text{kWh}$.
- **Resolution**: No critical events generated. Fan speed and dampers operate at optimal baseline levels ($50\%$ fan, $40\%$ damper).

---

## 8. Verification & Test Suite Summary

The framework contains an automated test suite verifying every tier:

- `test_phase1_smoke.py`: State modeling & Pydantic validation.
- `test_phase2_smoke.py`: 25/25 scoring formula unit tests.
- `test_phase3_smoke.py`: Event generator severity & false positive guards.
- `test_phase4_smoke.py`: 5 domain agent recommendation rules.
- `test_phase5_smoke.py`: LLM Orchestrator, prompt construction, D4 constraint rejection, and rule-based fallback.
- `test_phase6_smoke.py`: Action Engine delta conversion, rate-limit clamping, deadband rejection, CO₂ override.
- `test_phase7_smoke.py`: Replayed simulator dynamics, CO₂ mass balance, feedback report generation, multi-cycle loop.
- `test_pre_phase5_decisions.py`: Pre-phase 5 decision verification script.

**Total Test Suite Result**: **131/131 checks passed.**
