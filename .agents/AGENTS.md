# Workspace Agent & Engineering Guidelines

This document specifies the software engineering pipelines, error handling strategies, directory structure, coding rules, documentation, logging, tracing, and git hygiene for this agentic building energy/HVAC optimization system.

---

## 1. Code Basic Software Engineering Pipelines

- **Layered Architecture**: The system must adhere strictly to a 7-tier pipeline:
  1. **State Modeling** (`BuildingState`, `ZoneState`, `EquipmentState`, `WeatherState`)
  2. **Feature & Score Engine** (Deterministic conversion of state metrics to features and numerical scores)
  3. **Event Detection** (Deterministic transformation of metrics/scores into high-level semantics/events)
  4. **Specialized Agent Layer** (Domain agents producing candidate recommendations)
  5. **LLM Orchestration Layer** (Resolving agent recommendation trade-offs and synthesizing final plans)
  6. **Action & Safety Engine** (Validating setpoints and generating executable HVAC commands)
  7. **Simulation & Feedback Loop** (Interfacing with EnergyPlus and updating state/confidence memory)
- **Modular Design & Separation of Concerns**: Each component must be decoupled and independently testable without requiring an active LLM or EnergyPlus engine.
- **Explicit Interfaces & Type Safety**: All inter-module communication must use strictly typed Pydantic models or dataclasses.

---

## 2. Error Handling & Fallbacks

- **Specific Exception Catching**: Catch specific exceptions only (`ValueError`, `KeyError`, API-specific errors). Never use bare `except:` blocks.
- **Actionable Error Messages**: Include exact context (e.g. failing zone ID, unexpected parameter range, upstream service name) in error messages.
- **Fallback Mechanisms**:
  - **Telemetry Degradation**: If sensor data is missing or corrupted, default to historical zone state or safe nominal setpoints and issue an `Event`.
  - **LLM Orchestration Fallback**: If the LLM provider fails, times out, or produces unparseable output, fallback to a rule-based recommendation merger prioritizing human safety and zone comfort bounds.
  - **Action Validation Safety Net**: The Action Engine/Safety Validator MUST hard-reject setpoints exceeding safe physical limits regardless of LLM output.

---

## 3. Structure Files & Code Rules

- **Directory Layout**:
  ```text
  ├── .agents/
  │   └── AGENTS.md
  ├── docs/
  │   └── architecture.md
  ├── src/
  │   ├── state/
  │   ├── engine/
  │   ├── events/
  │   ├── agents/
  │   ├── orchestrator/
  │   ├── action/
  │   ├── simulation/
  │   └── llm/
  ├── tests/
  └── pyproject.toml
  ```
- **PEP 8 Compliance**: Strictly enforce PEP 8 formatting, naming conventions, imports, and whitespace.
- **Line Length**: Maximum line length of **79 characters** (no exceptions).
- **LLM Integration**:
  - Centralize all LLM calls in a single dedicated module (`src/llm/client.py`).
  - All other pipeline components send requests alongside execution config parameters to this single LLM module.
- **Dependency & Environment Management**:
  - Use `uv` exclusively for dependency and environment management (`uv run`, `uv add`, `uv sync`).
  - Virtual environment resides in `.venv` (never commit).

---

## 4. Documentation Update Rules

- **Google-Style Docstrings**: All public modules, classes, and functions must have complete Google-style docstrings.
- **Architecture Synchronization**: When data schemas, agent interfaces, or pipeline event structures change, update `docs/architecture.md` immediately.
- **README & User Guidance**: Keep installation steps (`uv sync`), execution commands (`uv run main.py`), and environment configuration instructions accurate.

---

## 5. Log & Tracing Update Rules

- **Rich Output**: All console output must use the `rich` library (`rich.logging`, tables, panels, progress bars). Avoid bare `print()` statements outside of quick local debugging.
- **Structured Log Tracing**: Include clear execution logs with consistent contextual details (e.g., session identifiers, module name, and execution duration) via standard logging utilities.

---

## 6. Git Update Rules

- **Clean Commits**: Commit code in modular, logical units.
- **User Approval before Meta Changes**: Ask user permission before mutating `.gitignore` or `README.md`.
- **Ignore Rules**: `.gitignore` must cover `__pycache__/`, `.venv/`, `.env`, all test artifacts, `*.log`, output directories, and generated telemetry caches.
