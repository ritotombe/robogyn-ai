from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

from gynma import GYNMA_DOMAINS, render_gynma

GEARS_DOMAINS: List[Dict[str, Any]] = [
    {
        "id": "depth_perception",
        "name": "Depth Perception",
        "anchors": "1: Overshoots target  |  3: Quick correction  |  5: Accurate plane",
    },
    {
        "id": "bimanual_dexterity",
        "name": "Bimanual Dexterity",
        "anchors": "1: One-hand dominant  |  3: Sub-optimal pairing  |  5: Synergistic traction",
    },
    {
        "id": "efficiency",
        "name": "Efficiency",
        "anchors": "1: Tentative / stalls  |  3: Organized pace  |  5: Fluid progression",
    },
    {
        "id": "force_sensitivity",
        "name": "Force Sensitivity",
        "anchors": "1: Rough tissue pull  |  3: Minor trauma  |  5: Optimal tension",
    },
    {
        "id": "autonomy",
        "name": "Autonomy",
        "anchors": "1: Frequent stalls  |  3: Moderate flow  |  5: Continuous flow",
    },
    {
        "id": "robotic_control",
        "name": "Robotic Control",
        "anchors": "1: Poor view / collisions  |  3: Occasional clutch  |  5: Optimal control",
    },
]


@dataclass
class VideoMotionMetrics:
    video_filename: str
    fps: float
    segment_start_sec: float
    segment_end_sec: float
    task_duration_sec: float
    frames_analyzed: int
    mean_optical_flow: float
    peak_optical_flow: float
    motion_cv: float
    active_motion_pct: float
    idle_time_pct: float
    bimanual_symmetry_ratio: float
    burst_count: int
    timestamps_sec: List[float]
    flow_magnitudes: List[float]


# ---------------------------------------------------------------------------
# Clean Clinical CSS (No Raw HTML Templates)
# ---------------------------------------------------------------------------


def apply_clean_theme() -> None:
    st.markdown(
        "<style>"
        "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');"
        "html, body, [class*='css'], .stApp { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; background-color: #f8fafc !important; color: #0f172a !important; }"
        ".block-container { padding-top: 2rem !important; padding-bottom: 2.5rem !important; max-width: 1200px !important; }"
        "div[data-testid='stVerticalBlockBorderWrapper'] { background-color: #ffffff !important; border: 1px solid #e2e8f0 !important; border-radius: 12px !important; padding: 1rem 1.15rem !important; box-shadow: 0 1px 4px rgba(15, 23, 42, 0.03); }"
        "div[data-testid='stMetric'] { background-color: #ffffff !important; border: 1px solid #e2e8f0; border-radius: 10px; padding: 0.75rem 1rem; }"
        "div[data-testid='stMetricLabel'] { font-size: 0.78rem !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.03em; color: #64748b !important; }"
        "div[data-testid='stMetricValue'] { font-size: 1.45rem !important; font-weight: 700 !important; color: #0f172a !important; }"
        "div[data-testid='stTabs'] button[data-baseweb='tab'] { font-weight: 600 !important; font-size: 0.92rem !important; padding: 0.6rem 1.1rem !important; }"
        "div[data-baseweb='input'] > div { border-radius: 8px !important; border-color: #cbd5e1 !important; }"
        ".stButton > button, .stDownloadButton > button { border-radius: 8px !important; font-weight: 600 !important; }"
        ".stButton > button[kind='primary'], .stDownloadButton > button[kind='primary'] { background-color: #4f46e5 !important; border-color: #4f46e5 !important; color: #ffffff !important; }"
        "</style>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Browser-Playable Video Transcoding & Frame Preview Helpers
# ---------------------------------------------------------------------------


def ensure_browser_playable_mp4(video_path: str) -> str:
    if not video_path or not os.path.exists(video_path) or imageio_ffmpeg is None:
        return video_path

    try:
        stat = os.stat(video_path)
        key = f"{video_path}_{stat.st_size}_{stat.st_mtime}"
        file_hash = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
        out_path = os.path.join(tempfile.gettempdir(), f"robogyn_h264_{file_hash}.mp4")

        if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
            return out_path

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i",
            video_path,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "26",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            out_path,
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=60)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
            return out_path
    except Exception:
        pass

    return video_path


def ensure_trimmed_segment_mp4(video_path: str, start_sec: float, end_sec: float) -> str:
    if not video_path or not os.path.exists(video_path) or imageio_ffmpeg is None:
        return video_path

    try:
        stat = os.stat(video_path)
        key = f"{video_path}_{stat.st_size}_{stat.st_mtime}_{start_sec:.2f}_{end_sec:.2f}"
        file_hash = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
        out_path = os.path.join(tempfile.gettempdir(), f"robogyn_seg_{file_hash}.mp4")

        if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
            return out_path

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe,
            "-y",
            "-ss",
            f"{max(0.0, start_sec):.2f}",
            "-to",
            f"{max(0.1, end_sec):.2f}",
            "-i",
            video_path,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "25",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            out_path,
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=60)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
            return out_path
    except Exception:
        pass

    return ensure_browser_playable_mp4(video_path)


def extract_frame_rgb(video_path: str, timestamp_sec: float) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_frame = min(max(0, int(round(timestamp_sec * fps))), max(0, total_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Video & AI Scoring Engine
# ---------------------------------------------------------------------------


def generate_demo_video(duration_sec: int = 30, fps: int = 15) -> str:
    width, height = 480, 360
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_path = tmp.name
    tmp.close()

    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for idx in range(duration_sec * fps):
        t = idx / float(fps)
        frame = np.full((height, width, 3), (65, 55, 135), dtype=np.uint8)
        cuff_y = int(height * 0.52 + 4 * np.sin(t * 0.8))
        cv2.line(frame, (80, cuff_y), (width - 80, cuff_y), (50, 40, 105), 6)

        lx = int(width * 0.35 + 18 * np.sin(t * 1.3))
        ly = int(cuff_y - 15 + 10 * np.cos(t * 1.3))
        cv2.line(frame, (0, height), (lx, ly), (180, 180, 190), 8)

        phase = t % 6.0
        rx = int(width * 0.62 + (45 * np.sin(phase * 2.2) if phase < 4.2 else 3 * np.sin(t * 0.5)))
        ry = int(cuff_y + (35 * np.cos(phase * 2.2) if phase < 4.2 else 20))
        cv2.line(frame, (width, height), (rx, ry), (165, 175, 185), 9)
        cv2.ellipse(frame, (rx - 10, ry - 6), (14, 10), 30, 0, 210, (245, 245, 255), 2)
        writer.write(frame)

    writer.release()
    return ensure_browser_playable_mp4(tmp_path)


def get_video_info(path: str) -> Tuple[float, float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file.")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, (frames / fps if frames > 0 else 0.0)


def _clamp_score(val: float) -> int:
    return int(max(1, min(5, round(val))))


def analyze_video_and_predict_ai_gears(
    video_path: str,
    video_name: str,
    start_sec: float,
    end_sec: float,
    sample_hz: float = 8.0,
) -> Tuple[VideoMotionMetrics, Dict[str, int], Dict[str, str], Optional[np.ndarray]]:
    fps, total_dur = get_video_info(video_path)
    start_sec = max(0.0, float(start_sec))
    end_sec = min(float(total_dur), float(end_sec))
    start_f = int(round(start_sec * fps))
    end_f = int(round(end_sec * fps))
    stride = max(1, int(round(fps / sample_hz)))

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    prev_gray = None
    accum_map = None
    rep_bgr = None
    timestamps: List[float] = []
    mags: List[float] = []
    left_mags: List[float] = []
    right_mags: List[float] = []

    curr = start_f
    while curr <= end_f:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if (curr - start_f) % stride == 0:
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (320, max(1, int(round(h * 320.0 / w)))))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if rep_bgr is None:
                rep_bgr = small.copy()
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                mid_x = mag.shape[1] // 2
                timestamps.append(round(curr / fps, 2))
                mags.append(round(float(np.mean(mag)), 4))
                left_mags.append(float(np.mean(mag[:, :mid_x])))
                right_mags.append(float(np.mean(mag[:, mid_x:])))
                accum_map = mag.astype(np.float64) if accum_map is None else accum_map + mag
            prev_gray = gray
        curr += 1
    cap.release()

    if not mags:
        mags, timestamps = [0.0], [round(start_sec, 2)]
        left_mags, right_mags = [0.1], [0.1]

    arr = np.array(mags, dtype=np.float64)
    mean_m = float(np.mean(arr))
    peak_m = float(np.max(arr))
    std_m = float(np.std(arr))
    cv_m = float(std_m / mean_m) if mean_m > 1e-6 else 0.0

    active_pct = float(np.mean(arr >= max(0.25, 0.30 * mean_m)) * 100.0)
    idle_pct = max(0.0, 100.0 - active_pct)

    above = arr > (mean_m + 1.5 * std_m)
    bursts = int(np.sum(np.diff(np.concatenate(([0], above.astype(int)))) == 1))

    l_mean = float(np.mean(left_mags))
    r_mean = float(np.mean(right_mags))
    bimanual_sym = float(min(l_mean, r_mean) / max(l_mean, r_mean, 1e-6))

    heatmap_rgb = None
    if accum_map is not None and rep_bgr is not None:
        norm = cv2.normalize(accum_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_rgb = cv2.cvtColor(
            cv2.addWeighted(rep_bgr, 0.58, cv2.applyColorMap(norm, cv2.COLORMAP_JET), 0.42, 0),
            cv2.COLOR_BGR2RGB,
        )

    dur = max(1.0, end_sec - start_sec)
    bursts_per_min = (bursts / dur) * 60.0
    peak_ratio = peak_m / max(mean_m, 1e-4)

    ai_scores: Dict[str, int] = {
        "depth_perception": _clamp_score(5.0 - max(0.0, (bursts_per_min - 3.0) * 0.28)),
        "bimanual_dexterity": _clamp_score(1.5 + 3.8 * bimanual_sym),
        "efficiency": _clamp_score(1.2 + (active_pct / 100.0) * 4.0),
        "force_sensitivity": _clamp_score(5.2 - max(0.0, (peak_ratio - 1.8) * 0.85)),
        "autonomy": _clamp_score(5.0 - max(0.0, (idle_pct - 12.0) * 0.075)),
        "robotic_control": _clamp_score(5.0 - max(0.0, (cv_m - 0.35) * 3.0)),
    }

    ai_reasons: Dict[str, str] = {
        "depth_perception": f"{bursts} motion spikes ({dur:.1f}s segment)",
        "bimanual_dexterity": f"{bimanual_sym * 100:.0f}% L-R motion symmetry",
        "efficiency": f"{active_pct:.1f}% active / {idle_pct:.1f}% idle",
        "force_sensitivity": f"{peak_ratio:.2f}x peak-to-mean velocity",
        "autonomy": f"{100.0 - idle_pct:.1f}% steady progression",
        "robotic_control": f"Smoothness CV = {cv_m:.2f}",
    }

    metrics = VideoMotionMetrics(
        video_filename=video_name,
        fps=round(fps, 1),
        segment_start_sec=round(start_sec, 1),
        segment_end_sec=round(end_sec, 1),
        task_duration_sec=round(dur, 1),
        frames_analyzed=len(mags),
        mean_optical_flow=round(mean_m, 3),
        peak_optical_flow=round(peak_m, 3),
        motion_cv=round(cv_m, 3),
        active_motion_pct=round(active_pct, 1),
        idle_time_pct=round(idle_pct, 1),
        bimanual_symmetry_ratio=round(bimanual_sym, 2),
        burst_count=bursts,
        timestamps_sec=timestamps,
        flow_magnitudes=mags,
    )
    return metrics, ai_scores, ai_reasons, heatmap_rgb


def predict_ai_gynma_scores(m: VideoMotionMetrics) -> Tuple[Dict[str, int], Dict[str, str]]:
    dur = max(1.0, m.task_duration_sec)
    bursts_per_min = (m.burst_count / dur) * 60.0
    peak_ratio = m.peak_optical_flow / max(m.mean_optical_flow, 1e-4)

    def _clamp2(val: float) -> int:
        return int(max(0, min(2, round(val))))

    gynma_scores: Dict[str, int] = {
        "needle_grasp": _clamp2(0.4 + 1.8 * m.bimanual_symmetry_ratio),
        "entry_exit": _clamp2(2.0 - max(0.0, (bursts_per_min - 3.0) * 0.14)),
        "arc": _clamp2(2.0 - max(0.0, (m.motion_cv - 0.38) * 1.45)),
        "rotation": _clamp2(2.0 - max(0.0, (m.idle_time_pct - 12.0) * 0.038)),
        "transition": _clamp2(2.1 - max(0.0, (peak_ratio - 1.8) * 0.42)),
        "regrasp": _clamp2(0.3 + (m.active_motion_pct / 100.0) * 1.9),
    }

    gynma_reasons: Dict[str, str] = {
        "needle_grasp": f"{m.bimanual_symmetry_ratio * 100:.0f}% L-R driver stability",
        "entry_exit": f"{m.burst_count} corrective spikes ({dur:.1f}s)",
        "arc": f"Arc smoothness CV = {m.motion_cv:.2f}",
        "rotation": f"{100.0 - m.idle_time_pct:.1f}% continuous rotation flow",
        "transition": f"{peak_ratio:.2f}x drive-to-withdrawal transition ratio",
        "regrasp": f"{m.active_motion_pct:.1f}% active manipulation economy",
    }
    return gynma_scores, gynma_reasons


# ---------------------------------------------------------------------------
# Main Application (Clean, Emoji-Free, Pure Streamlit Components)
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="RoboGyn-AI", layout="wide", initial_sidebar_state="collapsed")
    apply_clean_theme()

    if "video_path" not in st.session_state:
        st.session_state["video_path"] = None
        st.session_state["browser_video_path"] = None
        st.session_state["video_name"] = ""
        st.session_state["last_uploaded_sig"] = ""
    if "metrics" not in st.session_state:
        st.session_state["metrics"] = None
        st.session_state["heatmap"] = None
    if "ai_scores" not in st.session_state:
        st.session_state["ai_scores"] = {d["id"]: 3 for d in GEARS_DOMAINS}
    if "ai_reasons" not in st.session_state:
        st.session_state["ai_reasons"] = {d["id"]: "Run AI analysis in Step 1" for d in GEARS_DOMAINS}
    if "expert_scores" not in st.session_state:
        st.session_state["expert_scores"] = {d["id"]: 3 for d in GEARS_DOMAINS}
    if "gynma_ai_scores" not in st.session_state:
        st.session_state["gynma_ai_scores"] = {d["id"]: 1 for d in GYNMA_DOMAINS}
    if "gynma_ai_reasons" not in st.session_state:
        st.session_state["gynma_ai_reasons"] = {d["id"]: "Run AI analysis in Step 1" for d in GYNMA_DOMAINS}
    if "gynma_expert_scores" not in st.session_state:
        st.session_state["gynma_expert_scores"] = {d["id"]: 1 for d in GYNMA_DOMAINS}
    if "active_tab_idx" not in st.session_state:
        st.session_state["active_tab_idx"] = 0

    st.markdown("## RoboGyn-AI")
    st.caption(
        "Robotic Suturing Video Assessment  |  Standard GEARS (1–5) & Novelty GyNMA Needle Motion Assessment (0–2)"
    )

    step_labels = [
        "1. Video Input",
        "2. Scoring AI",
        "3. Scoring Expert",
        "4. AI vs. Expert Comparison",
        "5. GyNMA Module (Novelty)",
    ]
    active_idx = int(st.session_state.get("active_tab_idx", 0))
    nav_cols = st.columns(5)
    for idx, label in enumerate(step_labels):
        with nav_cols[idx]:
            if st.button(
                label,
                key=f"step_nav_btn_{idx}",
                type="primary" if active_idx == idx else "secondary",
                use_container_width=True,
            ):
                st.session_state["active_tab_idx"] = idx
                st.rerun()

    # =======================================================================
    # STEP 1: Video Input
    # =======================================================================
    if active_idx == 0:
        with st.container(border=True):
            up = st.file_uploader(
                "Upload Surgical Video (MP4, MOV, AVI)",
                type=["mp4", "mov", "avi"],
            )
            if up is not None:
                sig = f"{up.name}_{up.size}"
                if st.session_state.get("last_uploaded_sig") != sig:
                    ext = os.path.splitext(up.name)[1] or ".mp4"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                        tmp.write(up.getbuffer())
                        raw_path = tmp.name
                    with st.spinner("Preparing video preview..."):
                        browser_path = ensure_browser_playable_mp4(raw_path)
                    st.session_state["video_path"] = raw_path
                    st.session_state["browser_video_path"] = browser_path
                    st.session_state["video_name"] = up.name
                    st.session_state["last_uploaded_sig"] = sig
                    st.session_state["seg_start"] = 0.0
                    st.session_state["seg_end"] = 30.0

        vpath = st.session_state["video_path"]
        bpath = st.session_state.get("browser_video_path") or vpath
        if vpath and os.path.exists(vpath):
            fps, dur = get_video_info(vpath)
            v_col, s_col = st.columns([1.2, 1.3])

            with s_col:
                with st.container(border=True):
                    st.markdown("#### Segment Selection")
                    st.caption(f"File: {st.session_state['video_name']}  |  {fps:.1f} FPS  |  Total {dur:.1f}s")
                    max_d = max(1.0, round(dur, 1))
                    default_s = min(float(st.session_state.get("seg_start", 0.0)), max(0.0, max_d - 0.5))
                    default_e = min(float(st.session_state.get("seg_end", min(max_d, 30.0))), max_d)
                    if default_e <= default_s:
                        default_e = max_d
                    start_s, end_s = st.slider(
                        "Select Start and End Time (seconds)",
                        min_value=0.0,
                        max_value=max_d,
                        value=(default_s, default_e),
                        step=0.5,
                    )
                    st.session_state["seg_start"] = start_s
                    st.session_state["seg_end"] = end_s
                    mc1, mc2 = st.columns(2)
                    mc1.metric("Segment Window", f"{start_s:.1f}s - {end_s:.1f}s")
                    mc2.metric("Task Duration", f"{end_s - start_s:.1f} s")

                    f_start = extract_frame_rgb(vpath, start_s)
                    f_end = extract_frame_rgb(vpath, end_s)
                    if f_start is not None and f_end is not None:
                        th1, th2 = st.columns(2)
                        th1.image(f_start, caption=f"Start Frame ({start_s:.1f}s)", use_container_width=True)
                        th2.image(f_end, caption=f"End Frame ({end_s:.1f}s)", use_container_width=True)

                    if st.button("Run AI Video Analysis & Scoring", type="primary", use_container_width=True):
                        with st.spinner("Analyzing video motion and generating AI GEARS & GyNMA scores..."):
                            m, ai_sc, ai_rs, hmap = analyze_video_and_predict_ai_gears(
                                vpath, st.session_state["video_name"], start_s, end_s
                            )
                            g_sc, g_rs = predict_ai_gynma_scores(m)
                            st.session_state["metrics"] = m
                            st.session_state["ai_scores"] = ai_sc
                            st.session_state["ai_reasons"] = ai_rs
                            st.session_state["gynma_ai_scores"] = g_sc
                            st.session_state["gynma_ai_reasons"] = g_rs
                            st.session_state["heatmap"] = hmap
                            st.session_state["trimmed_video_path"] = ensure_trimmed_segment_mp4(vpath, start_s, end_s)
                            st.session_state["active_tab_idx"] = 1
                        st.rerun()

            with v_col:
                with st.container(border=True):
                    st.video(bpath, start_time=int(start_s))

    # =======================================================================
    # STEP 2: Scoring AI (GEARS 1–5 + GyNMA Novelty 0–2)
    # =======================================================================
    elif active_idx == 1:
        m: Optional[VideoMotionMetrics] = st.session_state["metrics"]
        if m is None:
            st.info("Upload a video in Step 1 and click 'Run AI Video Analysis & Scoring'.")
        else:
            ai_total = sum(st.session_state["ai_scores"].values())
            gynma_ai_total = sum(st.session_state["gynma_ai_scores"].values())
            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric("AI GEARS Total", f"{ai_total} / 30")
            k2.metric("AI GyNMA Total", f"{gynma_ai_total} / 12")
            k3.metric("Task Duration", f"{m.task_duration_sec:.1f} s")
            k4.metric("Active Motion", f"{m.active_motion_pct:.1f}%")
            k5.metric("Bimanual Balance", f"{m.bimanual_symmetry_ratio * 100:.0f}%")
            k6.metric("Motion Spikes", f"{m.burst_count}")

            p_col, h_col = st.columns([1.65, 1.0])
            with p_col:
                with st.container(border=True):
                    fig = go.Figure(
                        go.Scatter(
                            x=m.timestamps_sec,
                            y=m.flow_magnitudes,
                            mode="lines",
                            fill="tozeroy",
                            line=dict(color="#4f46e5", width=2),
                            fillcolor="rgba(79, 70, 229, 0.12)",
                        )
                    )
                    fig.update_layout(
                        title=dict(text="Optical Flow Motion Profile", font=dict(size=14)),
                        xaxis_title="Time (seconds)",
                        yaxis_title="Optical Flow (px/step)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=255,
                        margin=dict(l=15, r=15, t=35, b=15),
                        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
                        yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
                    )
                    st.plotly_chart(fig, use_container_width=True)
            with h_col:
                with st.container(border=True):
                    st.markdown("**Spatial Motion Heatmap**")
                    if st.session_state["heatmap"] is not None:
                        st.image(st.session_state["heatmap"], use_container_width=True)

            st.markdown("#### A. AI Standard GEARS Domain Scores (Scale 1–5, Max 30)")
            grid_cols = st.columns(2)
            for idx, d in enumerate(GEARS_DOMAINS):
                did = d["id"]
                sc = st.session_state["ai_scores"][did]
                reason = st.session_state["ai_reasons"][did]
                with grid_cols[idx % 2]:
                    with st.container(border=True):
                        h_left, h_right = st.columns([3, 1])
                        h_left.markdown(f"**{d['name']}**")
                        h_right.markdown(f"**{sc} / 5**")
                        st.progress(sc / 5.0)
                        st.caption(reason)

            st.markdown("#### B. AI Novelty GyNMA — Gynecologic Needle Motion Scores (Scale 0–2, Max 12)")
            g_cols = st.columns(2)
            for idx, d in enumerate(GYNMA_DOMAINS):
                did = d["id"]
                sc = st.session_state["gynma_ai_scores"][did]
                reason = st.session_state["gynma_ai_reasons"][did]
                with g_cols[idx % 2]:
                    with st.container(border=True):
                        h_left, h_right = st.columns([3, 1])
                        h_left.markdown(f"**{d['name']}**")
                        h_right.markdown(f"**{sc} / 2**")
                        st.progress(sc / 2.0)
                        st.caption(f"{d['definition']}  •  {reason}")

            if st.button("Continue to 3. Scoring Expert", type="primary", use_container_width=True):
                st.session_state["active_tab_idx"] = 2
                st.rerun()

    # =======================================================================
    # STEP 3: Scoring Expert (Selected Segment Video + GEARS + GyNMA Novelty)
    # =======================================================================
    elif active_idx == 2:
        for d in GEARS_DOMAINS:
            did = d["id"]
            if f"exp_slider_{did}" in st.session_state:
                st.session_state["expert_scores"][did] = int(st.session_state[f"exp_slider_{did}"])
        for d in GYNMA_DOMAINS:
            did = d["id"]
            if f"exp_gynma_slider_{did}" in st.session_state:
                st.session_state["gynma_expert_scores"][did] = int(st.session_state[f"exp_gynma_slider_{did}"])

        exp_tot_now = sum(st.session_state["expert_scores"].values())
        exp_gynma_tot = sum(st.session_state["gynma_expert_scores"].values())

        vpath_exp = st.session_state.get("video_path")
        if vpath_exp and os.path.exists(vpath_exp):
            seg_s = float(st.session_state.get("seg_start", 0.0))
            seg_e = float(st.session_state.get("seg_end", 30.0))
            trimmed_p = ensure_trimmed_segment_mp4(vpath_exp, seg_s, seg_e)
            with st.container(border=True):
                ev_col1, ev_col2 = st.columns([1.25, 1.35])
                with ev_col1:
                    st.video(trimmed_p)
                with ev_col2:
                    st.markdown("#### Selected Video Segment for Expert Review")
                    st.caption(f"File: {st.session_state.get('video_name', '')}")
                    m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                    m_c1.metric("Start - End", f"{seg_s:.1f}s - {seg_e:.1f}s")
                    m_c2.metric("Duration", f"{seg_e - seg_s:.1f} s")
                    m_c3.metric("Expert GEARS", f"{exp_tot_now} / 30")
                    m_c4.metric("Expert GyNMA", f"{exp_gynma_tot} / 12")
        else:
            top_c1, top_c2, top_c3 = st.columns([2.4, 1, 1])
            with top_c1:
                st.markdown("#### Expert Clinical Evaluation (GEARS & GyNMA Novelty)")
                st.caption("Rate the 6 standard GEARS domains (1–5) and 6 Novelty GyNMA needle domains (0–2).")
            with top_c2:
                st.metric("Expert GEARS Total", f"{exp_tot_now} / 30")
            with top_c3:
                st.metric("Expert GyNMA Total", f"{exp_gynma_tot} / 12")

        st.markdown("#### A. Expert Standard GEARS Scoring (Scale 1–5, Max 30)")
        exp_cols = st.columns(2)
        for idx, d in enumerate(GEARS_DOMAINS):
            did = d["id"]
            cur_val = int(st.session_state["expert_scores"][did])
            with exp_cols[idx % 2]:
                with st.container(border=True):
                    hdr_l, hdr_r = st.columns([2.5, 1.5])
                    hdr_l.markdown(f"**{d['name']}**")
                    hdr_r.markdown(f"Score: **{cur_val} / 5**")

                    st.caption(d["anchors"])
                    st.session_state["expert_scores"][did] = st.slider(
                        d["name"],
                        min_value=1,
                        max_value=5,
                        value=cur_val,
                        step=1,
                        key=f"exp_slider_{did}",
                        label_visibility="collapsed",
                    )

        st.markdown("#### B. Expert Novelty GyNMA — Gynecologic Needle Motion Scoring (Scale 0–2, Max 12)")
        g_exp_cols = st.columns(2)
        for idx, d in enumerate(GYNMA_DOMAINS):
            did = d["id"]
            cur_gval = int(st.session_state["gynma_expert_scores"][did])
            with g_exp_cols[idx % 2]:
                with st.container(border=True):
                    hdr_l, hdr_r = st.columns([2.5, 1.5])
                    hdr_l.markdown(f"**{d['name']}**")
                    hdr_r.markdown(f"Score: **{cur_gval} / 2**")

                    st.caption(f"{d['definition']}  •  {d['anchors']}")
                    st.session_state["gynma_expert_scores"][did] = st.slider(
                        d["name"],
                        min_value=0,
                        max_value=2,
                        value=cur_gval,
                        step=1,
                        key=f"exp_gynma_slider_{did}",
                        label_visibility="collapsed",
                    )

        if st.button("Continue to 4. AI vs. Expert Comparison", type="primary", use_container_width=True):
            st.session_state["active_tab_idx"] = 3
            st.rerun()

    # =======================================================================
    # STEP 4: AI vs. Expert Comparison (GEARS + GyNMA Novelty)
    # =======================================================================
    elif active_idx == 3:
        m = st.session_state.get("metrics")
        ai_scores = st.session_state["ai_scores"]
        exp_scores = st.session_state["expert_scores"]
        ai_tot = sum(ai_scores.values())
        exp_tot = sum(exp_scores.values())

        diffs = [ai_scores[d["id"]] - exp_scores[d["id"]] for d in GEARS_DOMAINS]
        abs_diffs = [abs(x) for x in diffs]
        mae = float(np.mean(abs_diffs))
        agreement_pct = max(0.0, (1.0 - mae / 4.0) * 100.0)

        g_ai_scores = st.session_state["gynma_ai_scores"]
        g_exp_scores = st.session_state["gynma_expert_scores"]
        g_ai_tot = sum(g_ai_scores.values())
        g_exp_tot = sum(g_exp_scores.values())
        g_diffs = [g_ai_scores[d["id"]] - g_exp_scores[d["id"]] for d in GYNMA_DOMAINS]
        g_mae = float(np.mean([abs(x) for x in g_diffs]))
        g_agreement_pct = max(0.0, (1.0 - g_mae / 2.0) * 100.0)

        st.markdown("#### A. Standard GEARS Comparison (Scale 1–5, Total / 30)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AI GEARS Total", f"{ai_tot} / 30")
        c2.metric("Expert GEARS Total", f"{exp_tot} / 30")
        c3.metric("GEARS Difference", f"{ai_tot - exp_tot:+d}")
        c4.metric("GEARS Agreement", f"{agreement_pct:.1f}% (MAE {mae:.2f})")

        names = [d["name"] for d in GEARS_DOMAINS]
        ai_list = [ai_scores[d["id"]] for d in GEARS_DOMAINS]
        exp_list = [exp_scores[d["id"]] for d in GEARS_DOMAINS]

        ch1, ch2 = st.columns([1.35, 1.0])
        with ch1:
            with st.container(border=True):
                fig_bar = go.Figure(
                    data=[
                        go.Bar(name="AI Score", x=names, y=ai_list, marker=dict(color="#4f46e5", cornerradius=6)),
                        go.Bar(name="Expert Score", x=names, y=exp_list, marker=dict(color="#10b981", cornerradius=6)),
                    ]
                )
                fig_bar.update_layout(
                    barmode="group",
                    title=dict(text="GEARS: AI Score vs. Expert Score (1–5)", font=dict(size=14)),
                    yaxis=dict(range=[0, 5.5], dtick=1, title="Score (1–5)", gridcolor="#f1f5f9"),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=295,
                    margin=dict(l=15, r=15, t=45, b=15),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        with ch2:
            with st.container(border=True):
                fig_rad = go.Figure()
                fig_rad.add_trace(
                    go.Scatterpolar(
                        r=ai_list + [ai_list[0]],
                        theta=names + [names[0]],
                        fill="toself",
                        name="AI Score",
                        line=dict(color="#4f46e5", width=2),
                        fillcolor="rgba(79, 70, 229, 0.18)",
                    )
                )
                fig_rad.add_trace(
                    go.Scatterpolar(
                        r=exp_list + [exp_list[0]],
                        theta=names + [names[0]],
                        fill="toself",
                        name="Expert Score",
                        line=dict(color="#10b981", width=2),
                        fillcolor="rgba(16, 185, 129, 0.18)",
                    )
                )
                fig_rad.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 5], dtick=1, gridcolor="#e2e8f0")),
                    title=dict(text="GEARS Radar Overlay", font=dict(size=14)),
                    paper_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=295,
                    margin=dict(l=30, r=30, t=45, b=15),
                )
                st.plotly_chart(fig_rad, use_container_width=True)

        comp_rows = []
        for d in GEARS_DOMAINS:
            did = d["id"]
            av, ev = ai_scores[did], exp_scores[did]
            diff = av - ev
            comp_rows.append(
                {
                    "GEARS Domain": d["name"],
                    "AI (1–5)": av,
                    "Expert (1–5)": ev,
                    "Diff (AI - Exp)": f"{diff:+d}" if diff != 0 else "0 (Match)",
                    "Abs Error": abs(diff),
                    "AI Motion Basis": st.session_state["ai_reasons"][did],
                }
            )

        with st.container(border=True):
            st.markdown("##### GEARS Domain Comparison Table")
            df_comp = pd.DataFrame(comp_rows)
            st.dataframe(
                df_comp,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "GEARS Domain": st.column_config.TextColumn("GEARS Domain", width="medium"),
                    "AI (1–5)": st.column_config.NumberColumn("AI (1–5)", width="small"),
                    "Expert (1–5)": st.column_config.NumberColumn("Expert (1–5)", width="small"),
                    "Diff (AI - Exp)": st.column_config.TextColumn("Diff (AI - Exp)", width="small"),
                    "Abs Error": st.column_config.NumberColumn("Abs Error", width="small"),
                    "AI Motion Basis": st.column_config.TextColumn("AI Motion Basis", width="large"),
                },
            )

        st.markdown("#### B. Novelty GyNMA Comparison — Gynecologic Needle Motion Assessment (Scale 0–2, Total / 12)")
        gc1, gc2, gc3, gc4 = st.columns(4)
        gc1.metric("AI GyNMA Total", f"{g_ai_tot} / 12")
        gc2.metric("Expert GyNMA Total", f"{g_exp_tot} / 12")
        gc3.metric("GyNMA Difference", f"{g_ai_tot - g_exp_tot:+d}")
        gc4.metric("GyNMA Agreement", f"{g_agreement_pct:.1f}% (MAE {g_mae:.2f})")

        g_names = [d["name"] for d in GYNMA_DOMAINS]
        g_ai_list = [g_ai_scores[d["id"]] for d in GYNMA_DOMAINS]
        g_exp_list = [g_exp_scores[d["id"]] for d in GYNMA_DOMAINS]

        gch1, gch2 = st.columns([1.35, 1.0])
        with gch1:
            with st.container(border=True):
                fig_gbar = go.Figure(
                    data=[
                        go.Bar(name="AI GyNMA", x=g_names, y=g_ai_list, marker=dict(color="#4f46e5", cornerradius=6)),
                        go.Bar(name="Expert GyNMA", x=g_names, y=g_exp_list, marker=dict(color="#10b981", cornerradius=6)),
                    ]
                )
                fig_gbar.update_layout(
                    barmode="group",
                    title=dict(text="GyNMA Novelty: AI Score vs. Expert Score (0–2)", font=dict(size=14)),
                    yaxis=dict(range=[0, 2.4], dtick=1, title="Score (0–2)", gridcolor="#f1f5f9"),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=295,
                    margin=dict(l=15, r=15, t=45, b=15),
                )
                st.plotly_chart(fig_gbar, use_container_width=True)

        with gch2:
            with st.container(border=True):
                fig_grad = go.Figure()
                fig_grad.add_trace(
                    go.Scatterpolar(
                        r=g_ai_list + [g_ai_list[0]],
                        theta=g_names + [g_names[0]],
                        fill="toself",
                        name="AI GyNMA",
                        line=dict(color="#4f46e5", width=2),
                        fillcolor="rgba(79, 70, 229, 0.18)",
                    )
                )
                fig_grad.add_trace(
                    go.Scatterpolar(
                        r=g_exp_list + [g_exp_list[0]],
                        theta=g_names + [g_names[0]],
                        fill="toself",
                        name="Expert GyNMA",
                        line=dict(color="#10b981", width=2),
                        fillcolor="rgba(16, 185, 129, 0.18)",
                    )
                )
                fig_grad.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 2], dtick=1, gridcolor="#e2e8f0")),
                    title=dict(text="GyNMA Needle Kinematics Radar", font=dict(size=14)),
                    paper_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=295,
                    margin=dict(l=30, r=30, t=45, b=15),
                )
                st.plotly_chart(fig_grad, use_container_width=True)

        g_comp_rows = []
        for d in GYNMA_DOMAINS:
            did = d["id"]
            av, ev = g_ai_scores[did], g_exp_scores[did]
            diff = av - ev
            g_comp_rows.append(
                {
                    "GyNMA Domain": d["name"],
                    "AI (0–2)": av,
                    "Expert (0–2)": ev,
                    "Diff (AI - Exp)": f"{diff:+d}" if diff != 0 else "0 (Match)",
                    "Abs Error": abs(diff),
                    "AI Motion Basis": st.session_state["gynma_ai_reasons"][did],
                }
            )

        with st.container(border=True):
            st.markdown("##### GyNMA Novelty Domain Comparison Table")
            df_gcomp = pd.DataFrame(g_comp_rows)
            st.dataframe(
                df_gcomp,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "GyNMA Domain": st.column_config.TextColumn("GyNMA Domain", width="medium"),
                    "AI (0–2)": st.column_config.NumberColumn("AI (0–2)", width="small"),
                    "Expert (0–2)": st.column_config.NumberColumn("Expert (0–2)", width="small"),
                    "Diff (AI - Exp)": st.column_config.TextColumn("Diff (AI - Exp)", width="small"),
                    "Abs Error": st.column_config.NumberColumn("Abs Error", width="small"),
                    "AI Motion Basis": st.column_config.TextColumn("AI Motion Basis", width="large"),
                },
            )

            summary_rows = [
                {"Metric": "Video Filename", "Value": st.session_state.get("video_name", "")},
                {"Metric": "Exported At", "Value": datetime.now().isoformat(timespec="seconds")},
                {"Metric": "Segment Start (s)", "Value": round(float(st.session_state.get("seg_start", 0.0)), 1)},
                {"Metric": "Segment End (s)", "Value": round(float(st.session_state.get("seg_end", 30.0)), 1)},
                {"Metric": "AI GEARS Total Score (/ 30)", "Value": ai_tot},
                {"Metric": "Expert GEARS Total Score (/ 30)", "Value": exp_tot},
                {"Metric": "GEARS Difference (AI - Expert)", "Value": ai_tot - exp_tot},
                {"Metric": "GEARS MAE", "Value": round(mae, 3)},
                {"Metric": "GEARS Agreement Rate (%)", "Value": round(agreement_pct, 1)},
                {"Metric": "AI GyNMA Total Score (/ 12)", "Value": g_ai_tot},
                {"Metric": "Expert GyNMA Total Score (/ 12)", "Value": g_exp_tot},
                {"Metric": "GyNMA Difference (AI - Expert)", "Value": g_ai_tot - g_exp_tot},
                {"Metric": "GyNMA MAE", "Value": round(g_mae, 3)},
                {"Metric": "GyNMA Agreement Rate (%)", "Value": round(g_agreement_pct, 1)},
            ]
            if m is not None:
                summary_rows.extend(
                    [
                        {"Metric": "Task Duration (s)", "Value": m.task_duration_sec},
                        {"Metric": "Active Motion (%)", "Value": m.active_motion_pct},
                        {"Metric": "Idle Time (%)", "Value": m.idle_time_pct},
                        {"Metric": "Bimanual Symmetry (%)", "Value": round(m.bimanual_symmetry_ratio * 100.0, 1)},
                        {"Metric": "Motion Spikes", "Value": m.burst_count},
                        {"Metric": "Motion Smoothness CV", "Value": m.motion_cv},
                    ]
                )
            df_summary = pd.DataFrame(summary_rows)

            xlsx_buf = io.BytesIO()
            with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
                df_comp.to_excel(writer, sheet_name="GEARS Comparison", index=False)
                df_gcomp.to_excel(writer, sheet_name="GyNMA Novelty Comparison", index=False)
                df_summary.to_excel(writer, sheet_name="Summary & Metrics", index=False)
                for sheet_name in writer.sheets:
                    ws = writer.sheets[sheet_name]
                    for col_cells in ws.columns:
                        max_len = max(len(str(c.value or "")) for c in col_cells)
                        col_letter = col_cells[0].column_letter
                        ws.column_dimensions[col_letter].width = max(max_len + 4, 14)

            st.download_button(
                "Export Full GEARS & GyNMA Comparison Report (.xlsx)",
                data=xlsx_buf.getvalue(),
                file_name="robogyn_gears_and_gynma_comparison.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True,
            )

    # =======================================================================
    # STEP 5: GyNMA Module (Full Stitch-Level Novelty Assessment)
    # =======================================================================
    elif active_idx == 4:
        vpath_g = st.session_state.get("video_path")
        seg_s = float(st.session_state.get("seg_start", 0.0))
        seg_e = float(st.session_state.get("seg_end", 30.0))
        trimmed_p = ensure_trimmed_segment_mp4(vpath_g, seg_s, seg_e) if (vpath_g and os.path.exists(vpath_g)) else None
        render_gynma(
            assessment_id="RG-001",
            procedure="Robotic hysterectomy",
            task="Vaginal cuff suturing",
            segment_start=seg_s,
            segment_end=seg_e,
            trimmed_video_path=trimmed_p,
            ai_gynma_scores=st.session_state.get("gynma_ai_scores"),
            ai_gynma_reasons=st.session_state.get("gynma_ai_reasons"),
        )


if __name__ == "__main__":
    main()

