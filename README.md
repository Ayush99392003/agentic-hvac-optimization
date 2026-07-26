# Autonomous Agentic Building Control & Energy Optimization Framework

> **Honeywell HVAC Optimization System** — Hybrid Deterministic + LLM 7-Tier Architecture for Intelligent Building Control.

---

## Executive Overview

In industrial control systems (ICS) and building energy management (BEMS), relying solely on an unconstrained Large Language Model (LLM) to write hardware setpoints introduces unacceptable operational and liability risks. Conversely, traditional fixed rule-based systems lack the flexibility to resolve complex multi-objective trade-offs dynamically.

This framework implements a **hybrid 7-tier architecture**:

- **Deterministic Control Core**: State modeling, feature extraction, scoring formulas, event detection, physical safety validation, rate limiting, and simulation rollouts are 100% deterministic code.
- **LLM Ambiguity Layer**: The LLM acts strictly as an **Orchestrator and Trade-Off Resolver** when specialized domain agents propose competing recommendations (e.g., Comfort vs. Peak Tariff vs. Air Quality).
- **Dual-Layer Safety Net**: If the LLM provider fails, times out, or produces a plan violating hard safety rules (e.g. High-Density Occupancy Floor), the system automatically engages a **deterministic Rule-Based Fallback Engine** and physical **Safety Validator**.

---

## 7-Tier System Architecture

```text
                  Building Sensors & Weather
                              │
                              ▼
 ┌───────────────────────────────────────────────────────────┐
 │ 1. State Modeling (BuildingState, ZoneState, Weather)    │
 └─────────────────────────────┬─────────────────────────────┘
                               ▼
 ┌───────────────────────────────────────────────────────────┐
 │ 2. Feature & Score Engine (ASHRAE-55 PMV, CO2, Energy)    │
 └─────────────────────────────┬─────────────────────────────┘
                               ▼
 ┌───────────────────────────────────────────────────────────┐
 │ 3. Event Generator (Deterministic Event & Severity Mapping)│
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

## Quick Start Guide

### Prerequisites

- Python $\ge 3.11$
- `uv` package manager (`pip install uv` or `curl -sSf https://astral.sh/uv/install.sh`)

### Installation

Clone the repository and synchronize the virtual environment:

```bash
uv sync
```

### Execution Commands

Run the main pipeline in interactive terminal mode:

```bash
# 1. Run Demo Centerpiece Scenario (Multi-agent conflict resolution)
uv run main.py --scenario centerpiece --cycles 3

# 2. Run Emergency CO2 Safety Override Scenario
uv run main.py --scenario emergency_co2

# 3. Run Peak Tariff Unoccupied Shedding Scenario
uv run main.py --scenario peak_tariff_unoccupied

# 4. Run Normal Steady-State Operations Scenario
uv run main.py --scenario normal
```

---

## Governing Policy Matrix (Decisions D1–D4)

The pipeline incorporates four core policy decisions:

| Decision | Policy Name | Implementation & Enforcement |
|---|---|---|
| **D1** | Intra-Zone Vector Alignment | Confirmed that zone-level proposals for `conf_room_exec` are complementary (cooling + ventilation vectoring). |
| **D2** | Cross-Scope Fan Override | Zone-scoped `INCREASE_FAN_SPEED` for high-density zone with `air_quality=HIGH` overrides building-scoped `DECREASE_FAN_SPEED`. **Enforced via System Prompt AND deterministic Action Engine post-check `_enforce_d2_fan_scope()`**. |
| **D3** | Urgency Tiebreak | Equal urgency scores resolved in order: Health > Comfort > Energy > Carbon; Zone-scoped > Building-scoped. |
| **D4** | High-Density Floor | When occupancy $> 10$, comfort & air quality floors are hard non-negotiable constraints. **Enforced via System Prompt + `_verify_hard_constraints()` with automatic Fallback trigger**. |

---

## Physical Safety & Action Validation Rules

The **Safety Validator** (Phase 6) enforces strict physical limits before any setpoint command reaches the HVAC hardware or simulator:

- **Cooling Range**: $18.0^\circ\text{C} \le T_{\text{cool}} \le 26.0^\circ\text{C}$.
- **Deadband Check**: Cooling setpoint must maintain $\ge 1.0^\circ\text{C}$ gap above heating setpoint. Violations are **REJECTED**.
- **Rate Limit**: Maximum setpoint change per step $\le 2.0^\circ\text{C}$ (inclusive bound `abs(delta) > MAX_SETPOINT_DELTA_PER_STEP`). Excess deltas are **CLAMPED**.
- **Emergency CO₂ Override**: When indoor $\text{CO}_2 \ge 1200\,\text{PPM}$, outdoor air damper is forcibly opened to **100.0% [CLAMPED]**.

---

## Simulation & Swappable Strategy Protocol

Phase 7 implements a `SimulatorProtocol` with two backends:

1. **`ReplayedSimulator` (Default)**: Physics-inspired first-order thermal dynamics ($\tau = 0.20$) and CO₂ mass balance (dilution rate $-200\,\text{PPM/step}$ when damper $\ge 30\%$, fan $\ge 40\%$).
2. **`EnergyPlusSimulator` (Additive)**: Activated seamlessly when `ENERGYPLUS_IDF_PATH` environment variable is set.

---

## Automated Verification & Test Suite

The system includes a comprehensive automated test suite spanning all 7 pipeline tiers (**131/131 checks passing**):

```bash
# Run complete test suite across all 7 phases
uv run python -c "import sys, io, contextlib; sys.path.insert(0, '.'); from tests.test_phase1_smoke import main as p1; from tests.test_phase2_smoke import main as p2; from tests.test_phase3_smoke import main as p3; from tests.test_phase4_smoke import main as p4; from tests.test_phase5_smoke import main as p5; from tests.test_phase6_smoke import main as p6; from tests.test_phase7_smoke import main as p7; [fn() for fn in (p1,p2,p3,p4,p5,p6,p7)]"
```

Individual phase smoke tests:

```bash
uv run python -c "import sys; sys.path.insert(0,'.'); from tests.test_phase1_smoke import main; main()"
uv run python -c "import sys; sys.path.insert(0,'.'); from tests.test_phase2_smoke import main; main()"
uv run python -c "import sys; sys.path.insert(0,'.'); from tests.test_phase3_smoke import main; main()"
uv run python -c "import sys; sys.path.insert(0,'.'); from tests.test_phase4_smoke import main; main()"
uv run python -c "import sys; sys.path.insert(0,'.'); from tests.test_phase5_smoke import main; main()"
uv run python -c "import sys; sys.path.insert(0,'.'); from tests.test_phase6_smoke import main; main()"
uv run python -c "import sys; sys.path.insert(0,'.'); from tests.test_phase7_smoke import main; main()"
uv run python -c "import sys; sys.path.insert(0,'.'); from tests.test_pre_phase5_decisions import main; main()"
```

---

## Environment Variables & Optional LLM Integration

The application operates seamlessly in **offline fallback mode** when no API key is set. To activate live LLM orchestration via OpenAI:

```bash
# Optional: Set OpenAI API key for live gpt-4o orchestration
export OPENAI_API_KEY="your-api-key-here"

# Optional: Set EnergyPlus IDF path for live co-simulation
export ENERGYPLUS_IDF_PATH="/path/to/building.idf"

# Optional: Set Arize Phoenix collector endpoint for OpenTelemetry tracing
export PHOENIX_COLLECTOR_ENDPOINT="http://localhost:6006/v1/traces"
```

---

## Directory Structure

```text
├── docs/
│   └── architecture.md            # Comprehensive system & decision specification
├── main.py                        # Executable pipeline entrypoint with rich UI
├── pyproject.toml                 # Project tool & dependency configuration
├── src/
│   ├── state/                     # Tier 1: BuildingState & telemetry models
│   ├── engine/                    # Tier 2: Feature Extraction & Score Engine
│   ├── events/                    # Tier 3: Event Generator & Severity mapping
│   ├── agents/                    # Tier 4: Domain Agents (Comfort, Energy, AQ, Carbon, DR)
│   ├── llm/                       # Centralized OpenAI Responses API client
│   ├── orchestrator/              # Tier 5: LLM Orchestrator & Prompt Builder
│   ├── action/                    # Tier 6: Action Engine & Safety Validator
│   └── simulation/                # Tier 7: SimulatorProtocol, ReplayedSim & Feedback
└── tests/                         # Phase 1-7 Smoke Test Suites (131/131 passing)
```
