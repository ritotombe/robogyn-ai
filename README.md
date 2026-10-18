# RoboGyn-AI v0.1 — Clinical Video Assessment Prototype

Locally running application for robotic suturing assessment in gynecologic surgery, combining:
1. **Automated Video-Motion Analysis** (frame-to-frame dense optical flow, spatial motion heatmap, and automatic task duration)
2. **Structured GEARS Scoring** (*Global Evaluative Assessment of Robotic Skills* — Surgeon Self-Rating and Independent Expert Rating across all 6 clinical domains)
3. **Rubric-Based Feedback & Self–Expert Calibration** (domain-level comparison, practice priorities, and local CSV/JSON data export)

## Methodological & Clinical Privacy Principles
- **AI-Assisted Adjunct, Not an Unvalidated Autonomous Judge:** GEARS measures surgical technical skill, whereas computer vision measures observable video features. Version 0.1 pairs objective optical-flow measurements alongside surgeon and expert ratings rather than generating an unvalidated autonomous GEARS score.
- **100% Local Processing:** All video analysis and scoring run locally (`127.0.0.1`) with zero external cloud or API calls. Use de-identified operative recordings in a research/training environment.

## Quick Start

### Windows (PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py --server.address 127.0.0.1
```

### macOS / Linux (Terminal)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py --server.address 127.0.0.1
```

Then open **http://127.0.0.1:8501** in your browser.
