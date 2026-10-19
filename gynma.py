"""GyNMA (Gynecologic Needle Motion Assessment) — Novelty Rubric Module."""
from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

GYNMA_DOMAINS: List[Dict[str, str]] = [
    {
        "id": "needle_grasp",
        "name": "Needle Grasp Geometry",
        "definition": "Grasp location and needle orientation relative to driver jaws",
        "anchors": "0: Inadequate / deviation  |  1: Partially satisfactory  |  2: Satisfactory grasp",
        "feedback": "Review stable grasp and planned needle orientation.",
    },
    {
        "id": "entry_exit",
        "name": "Entry and Exit Placement",
        "definition": "Accuracy of planned entry and exit points and tissue bite",
        "anchors": "0: Inadequate bite / target  |  1: Inconsistent placement  |  2: Accurate entry & exit",
        "feedback": "Review planned bite geometry and entry/exit targets.",
    },
    {
        "id": "arc",
        "name": "Needle Arc Adherence",
        "definition": "Needle follows its curvature during tissue passage",
        "anchors": "0: Substantial translation  |  1: Minor arc deviation  |  2: Follows needle curvature",
        "feedback": "Review curved-needle trajectory and unnecessary translation.",
    },
    {
        "id": "rotation",
        "name": "Rotational Continuity",
        "definition": "Controlled visible needle/driver rotation with limited reversals",
        "anchors": "0: Frequent reversals  |  1: Occasional interruption  |  2: Smooth continuous rotation",
        "feedback": "Review interruptions and unplanned rotation reversals.",
    },
    {
        "id": "transition",
        "name": "Rotation Transition Control",
        "definition": "Timing and control of visible rotation changes during drive/withdrawal",
        "anchors": "0: Abrupt / uncontrolled  |  1: Moderate transition control  |  2: Smooth drive-to-withdrawal",
        "feedback": "Review transitions between driving, withdrawal and repositioning.",
    },
    {
        "id": "regrasp",
        "name": "Needle Regrasp Efficiency",
        "definition": "Purposeful needle release/regrasp without unnecessary attempts",
        "anchors": "0: Repeated failed regrasps  |  1: Extra regrasp attempts  |  2: Economical regrasp",
        "feedback": "Review regrasp timing and economy of manipulation.",
    },
]

ANCHORS: Dict[Any, str] = {
    "N/A": "Not assessable from this view",
    0: "Inadequate / substantial deviation",
    1: "Partially satisfactory / inconsistent",
    2: "Satisfactory for defined task",
}

PHASES: List[str] = ["Positioning", "Entry", "Driving", "Exit", "Withdrawal", "Regrasp"]


def render_gynma(
    assessment_id: str = "RG-001",
    procedure: str = "Robotic hysterectomy",
    task: str = "Vaginal cuff suturing",
    segment_start: Optional[float] = None,
    segment_end: Optional[float] = None,
    trimmed_video_path: Optional[str] = None,
    ai_gynma_scores: Optional[Dict[str, int]] = None,
    ai_gynma_reasons: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    st.markdown("#### GyNMA — Gynecologic Needle Motion Assessment (Novelty Rubric)")
    st.caption(
        "6-domain needle kinematics rubric scored 0–2 per domain (Maximum Total = 12). "
        "Supports Surgeon Self-Rating, Expert Rating, and AI Motion Benchmark."
    )

    if trimmed_video_path:
        with st.container(border=True):
            v_col, meta_col = st.columns([1.25, 1.25])
            with v_col:
                st.video(trimmed_video_path)
            with meta_col:
                st.markdown("##### Stitch & Suturing Phase Metadata")
                mc1, mc2 = st.columns(2)
                with mc1:
                    stitch_id = st.text_input("Stitch / Needle-Pass ID", value="ST-001", key="gynma_stitch")
                    event_time = st.number_input(
                        "Event Timestamp (s)",
                        min_value=0.0,
                        value=float(segment_start or 0.0),
                        step=0.5,
                        key="gynma_time",
                    )
                with mc2:
                    phase = st.selectbox("Suturing Phase Under Review", PHASES, key="gynma_phase")
                    visibility = st.selectbox(
                        "View Adequacy",
                        ["Adequate", "Partially obscured", "Not assessable"],
                        key="gynma_visibility",
                    )
    else:
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                stitch_id = st.text_input("Stitch / Needle-Pass ID", value="ST-001", key="gynma_stitch")
            with c2:
                phase = st.selectbox("Suturing Phase Under Review", PHASES, key="gynma_phase")
            with c3:
                event_time = st.number_input(
                    "Event Timestamp (s)",
                    min_value=0.0,
                    value=float(segment_start or 0.0),
                    step=0.5,
                    key="gynma_time",
                )
            with c4:
                visibility = st.selectbox(
                    "View Adequacy",
                    ["Adequate", "Partially obscured", "Not assessable"],
                    key="gynma_visibility",
                )

    rows: List[Dict[str, Any]] = []
    grid_cols = st.columns(2)
    options_list: List[Any] = [0, 1, 2, "N/A"]

    for idx, d in enumerate(GYNMA_DOMAINS):
        code = d["id"]
        label = d["name"]
        definition = d["definition"]
        feedback = d["feedback"]
        ai_val = (ai_gynma_scores or {}).get(code, 1)
        ai_reason = (ai_gynma_reasons or {}).get(code, "")

        default_exp = int(st.session_state.get("gynma_expert_scores", {}).get(code, 1))
        default_exp_idx = options_list.index(default_exp) if default_exp in options_list else 1

        with grid_cols[idx % 2]:
            with st.container(border=True):
                h_l, h_r = st.columns([2.6, 1.4])
                h_l.markdown(f"**{label}**")
                h_r.markdown(f"AI Score: **{ai_val} / 2**")
                st.caption(f"{definition}  •  {d['anchors']}")
                if ai_reason:
                    st.caption(f"AI Basis: {ai_reason}")

                sc_col1, sc_col2 = st.columns(2)
                with sc_col1:
                    s = st.selectbox(
                        "Surgeon Self-Rating (0–2)",
                        options_list,
                        index=1,
                        format_func=lambda v: f"{v} — {ANCHORS[v]}",
                        key=f"gynma_self_{code}",
                    )
                with sc_col2:
                    e = st.selectbox(
                        "Expert Rating (0–2)",
                        options_list,
                        index=default_exp_idx,
                        format_func=lambda v: f"{v} — {ANCHORS[v]}",
                        key=f"gynma_expert_{code}",
                    )
                    if isinstance(e, int) and "gynma_expert_scores" in st.session_state:
                        st.session_state["gynma_expert_scores"][code] = int(e)

                rows.append(
                    {
                        "domain_id": code,
                        "GyNMA Domain": label,
                        "AI Score (0–2)": ai_val,
                        "Self Rating (0–2)": None if s == "N/A" else int(s),
                        "Expert Rating (0–2)": None if e == "N/A" else int(e),
                        "Practice Prompt": feedback,
                    }
                )

    with st.container(border=True):
        note = st.text_area(
            "Timestamped Observations / Clinical Notes",
            key="gynma_notes",
            placeholder="Describe visible needle handling behavior (without patient identifiers)...",
            height=80,
        )

        df = pd.DataFrame(rows)
        sc = df["Self Rating (0–2)"].dropna()
        ex = df["Expert Rating (0–2)"].dropna()
        ai_sum = int(df["AI Score (0–2)"].sum())

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("AI GyNMA Total", f"{ai_sum} / 12")
        k2.metric("Self Subtotal", f"{int(sc.sum())} / 12" if len(sc) == 6 else f"{len(sc)} / 6 rated")
        k3.metric("Expert Subtotal", f"{int(ex.sum())} / 12" if len(ex) == 6 else f"{len(ex)} / 6 rated")
        matched = df.dropna(subset=["Self Rating (0–2)", "Expert Rating (0–2)"])
        k4.metric(
            "Mean Self–Expert Gap",
            f"{(matched['Self Rating (0–2)'] - matched['Expert Rating (0–2)']).mean():+.2f}" if len(matched) else "N/A",
        )

        st.dataframe(
            df[["GyNMA Domain", "AI Score (0–2)", "Self Rating (0–2)", "Expert Rating (0–2)", "Practice Prompt"]],
            hide_index=True,
            use_container_width=True,
        )

        if len(ex):
            st.markdown("**Suggested Review Priorities (Lowest Expert-Rated Domains)**")
            for _, r in df.dropna(subset=["Expert Rating (0–2)"]).sort_values("Expert Rating (0–2)").head(2).iterrows():
                st.markdown(f"- **{r['GyNMA Domain']} ({int(r['Expert Rating (0–2)'])} / 2):** {r['Practice Prompt']}")

        record: Dict[str, Any] = {
            "instrument": "GyNMA",
            "instrument_version": "0.3 novelty",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "assessment_id": assessment_id,
            "procedure": procedure,
            "task": task,
            "stitch_id": stitch_id,
            "phase": phase,
            "event_time_s": event_time,
            "segment_start_s": segment_start,
            "segment_end_s": segment_end,
            "visibility": visibility,
            "ai_total_12": ai_sum,
            "self_total_12": int(sc.sum()) if len(sc) == 6 else None,
            "expert_total_12": int(ex.sum()) if len(ex) == 6 else None,
            "notes": note,
            "domains": rows,
        }

        xlsx_buf = io.BytesIO()
        with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
            df.drop(columns=["domain_id"]).to_excel(writer, sheet_name="GyNMA Assessment", index=False)
            for sheet_name in writer.sheets:
                ws = writer.sheets[sheet_name]
                for col_cells in ws.columns:
                    max_len = max(len(str(c.value or "")) for c in col_cells)
                    ws.column_dimensions[col_cells[0].column_letter].width = max(max_len + 4, 14)

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "Export GyNMA Report (.xlsx)",
                data=xlsx_buf.getvalue(),
                file_name=f"{stitch_id}_gynma_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                "Export GyNMA JSON",
                data=json.dumps(record, indent=2),
                file_name=f"{stitch_id}_gynma_report.json",
                mime="application/json",
                use_container_width=True,
            )

    return record
