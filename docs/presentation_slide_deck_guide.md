# Honeywell Hackathon Presentation Slide Deck Details

Use the slide-by-slide outline below to build your presentation PowerPoint deck (.pptx).

---

## Slide 1: Title Slide
- **Title**: Eco-Loop Building Agents: Autonomous HVAC Control & Energy Optimization
- **Subtitle**: Hybrid 7-Tier Architecture for Intelligent Building Energy Management with EnergyPlus Co-Simulation
- **Team Name**: Eco-Loop Autonomous Control Team
- **Event**: Honeywell Campus Hackathon 2026

---

## Slide 2: Problem Statement & Industry Motivation
- **Global Impact**: Buildings consume ~40% of global energy and produce massive carbon emissions.
- **Limitation of Traditional BEMS**:
  - Rigid, static schedules (fixed 24/7 setpoints).
  - Incapable of dynamically adapting to peak grid tariffs ($0.45/kWh), high carbon grid windows, or localized indoor CO2 spikes.
- **The AI Challenge**: Unconstrained LLMs writing raw setpoints risk severe physical equipment damage or safety violations.
- **Solution**: A **Hybrid 7-Tier Architecture** combining deterministic safety guards, domain-expert AI agents, and live EnergyPlus 26.1 physics co-simulation.

---

## Slide 3: System Architecture (7-Tier Layered Pipeline)
- **Visual Diagram**: Show 7-Tier Flow (`State -> Score Engine -> Events -> Domain Agents -> LLM Orchestrator -> Safety Validator -> EnergyPlus`)
- **Core Engineering Principles**:
  - **Deterministic Core**: State modeling, ASHRAE-55 comfort scoring (0-100), event detection, rate limiting, and safety rules are 100% deterministic Python.
  - **LLM Ambiguity Layer**: Centralized LLM client resolves competing agent recommendations under strict policy constraints (D1–D4).
  - **Dual Safety Net**: If LLM fails or times out, system engages deterministic Rule-Based Fallback Engine.

---

## Slide 4: Specialized Domain Agents (Tier 4)
- **Agent Interface**: Standardized `evaluate(state, context, policy) -> Recommendation` interface.
- **5 Specialized Domain Agents**:
  1. **Comfort Agent**: Enforces ASHRAE-55 PMV comfort band $[-0.5, +0.5]$.
  2. **Energy Agent**: Minimizes kW power draw during high energy price windows.
  3. **Carbon Agent**: Shifts load away from high grid carbon emission windows ($430\,\text{gCO}_2\text{e/kWh}$).
  4. **Air Quality Agent**: Monitors CO2 PPM levels; boosts ventilation when CO2 $> 1000\,\text{PPM}$.
  5. **Demand Response Agent**: Relaxes setpoints in unoccupied zones (`board_room`).

---

## Slide 5: Governing Policy Matrix & LLM Trade-Off Resolution
- **Governing Policies (D1–D4)**:
  - **D1 (Vector Alignment)**: Aligns complementary proposals in occupied zones.
  - **D2 (Cross-Scope Fan Override)**: Zone-scoped fan boost in high-density occupied rooms overrides building-level fan reduction.
  - **D3 (Urgency Tiebreak)**: Health > Comfort > Energy > Carbon.
  - **D4 (High-Density Floor)**: Hard non-negotiable comfort/health constraints when occupants $> 10$.
- **Physical Safety Validator**: Rate limits setpoint changes ($\le 2.0^\circ\text{C}$/step), checks deadbands ($\ge 1.0^\circ\text{C}$), and forces 100% damper position during emergency CO2 spikes ($\ge 1200\,\text{PPM}$).

---

## Slide 6: Native EnergyPlus 26.1 Co-Simulation Integration
- **Direct Engine Integration**:
  - Reads DOE/NREL baseline building model (`models/baseline_building.idf`).
  - Forward-injects AI-calculated supervisory setpoints into EnergyPlus schedule objects (`models/baseline_building_modified.idf`).
  - Executes live `energyplus.exe` simulation binary.
- **SQLite Output Extraction**:
  - Connects directly to `output_sim/eplusout.sql` via Python `sqlite3`.
  - Extracts simulated zone mean air temperatures and feeds them back into the next closed-loop timestep.

---

## Slide 7: Model Context Protocol (MCP) & Logging Architecture
- **Standardized MCP Tool Integration**:
  - `get_building_telemetry`: Fetches real-time zone state & weather metrics.
  - `extract_runtime_errors`: Filters simulation logs (`eplusout.err`) via regex for instant error diagnosis.
  - `evaluate_building_scores`: Computes 0-100 ASHRAE-55 comfort, energy, and air quality scores.
  - `apply_hvac_setpoints`: Forwards validated setpoint directives to the Action Engine.
- **Lengthy Simulation Log Handling**: Regex-filtered log tailing prevents context window overflow.

---

## Slide 8: Quantitative Results & Savings Dashboard
- **Quantitative Benchmark Results** (Baseline vs. Eco-Loop AI):
  - **Total Energy Reduction**: Proved **42.4% - 47.1% energy reduction (kWh)**.
  - **ASHRAE-55 PMV Comfort**: Maintained PMV within comfortable bounds ($+0.59$).
  - **Indoor Air Quality**: Reduced peak CO2 from $1150\,\text{PPM}$ down to $830\,\text{PPM}$.
  - **Safety Compliance**: **100% Safety Rule Compliance (0 physical violations)**.

---

## Slide 9: Multi-Cycle Closed-Loop Trajectory
- **Visual Table / Chart**:
  - **Cycle 1**: Overheating & CO2 spike ($1150\,\text{PPM}$) detected $\rightarrow$ AI boosts damper & fan.
  - **Cycle 2-3**: Setpoint reduced to $22.5^\circ\text{C}$, CO2 diluted below $1000\,\text{PPM}$.
  - **Cycle 4-5**: Temperature stabilizes at $24.0^\circ\text{C}$, PMV restored to $+0.59$, unoccupied board room setpoint relaxed to save energy.

---

## Slide 10: Conclusion & Business Value
- **Summary**:
  - Autonomous self-correcting building control powered by Open-Source LLMs and EnergyPlus.
  - Guaranteed physical safety through a deterministic dual-layer safety net.
  - Substantial carbon & energy cost savings for commercial real estate.
- **Repository URL**: `https://github.com/Ayush99392003/agentic-hvac-optimization.git`
- **Q&A**: Thank you!
