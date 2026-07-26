# Honeywell Hackathon Submission Package Checklist

Follow this checklist to prepare your final `.zip` file for upload to the hackathon portal.

---

## 📦 What to Include in the `.zip` Submission File

Zip the entire root project directory. Ensure the zip contains:

1. **Source Code (Deliverable 1)**:
   - `main.py`
   - `src/` directory (`state/`, `engine/`, `events/`, `agents/`, `llm/`, `orchestrator/`, `action/`, `simulation/`, `mcp/`, `dashboard/`)
   - `tests/` directory (131/131 passing smoke test suite)
   - `pyproject.toml` and `uv.lock`

2. **Building Models (.idf) (Deliverable 2)**:
   - `models/baseline_building.idf` (DOE/NREL Baseline Building Model)
   - `models/baseline_building_modified.idf` (AI-Modified Runtime Building Model)
   - `models/weather.epw` (Weather Data File)

3. **Quantitative Savings Dashboard (Deliverable 3)**:
   - `savings_report.json` (Exported metrics comparing Baseline vs. AI strategy)

4. **Architectural & Technical Report (Deliverable 4)**:
   - `docs/architecture.md` (System architecture report covering MCP tool-calling, prompt engineering, latency management, simulation log handling)

5. **Presentation & Video Guidance (Deliverable 5)**:
   - `docs/presentation_slide_deck_guide.md` (Slide-by-slide outline for PowerPoint)
   - `docs/video_demo_script.md` (3-minute video recording script)
   - Link to GitHub Repository: `https://github.com/Ayush99392003/agentic-hvac-optimization.git`

---

## 🔒 Files Excluded from `.zip` (Handled automatically by `.gitignore`)

- `.env` (Your API credentials remain local and secure)
- `.venv/` (Virtual environment folder)
- `__pycache__/`
- `output_sim/*.sql`, `output_sim/*.eso` (Large simulation caches)
