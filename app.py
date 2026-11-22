"""
Module 06 - Final AI & ML Project: EduPulse AI SaaS Platform.
Loads the trained model pipeline (model.pkl) and metadata (model_metadata.json)
for live student evaluation, preset scenario testing, What-If simulation,
batch CSV triage, explainability, and session audit logging.
"""

import base64
from datetime import datetime
from io import StringIO
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from features import RAW_CATEGORICAL_COLS, RAW_NUMERIC_COLS, enrich_student_features

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.json"
SAMPLE_CSV_PATH = BASE_DIR / "sample_cohort.csv"
AUDIT_CSV_PATH = BASE_DIR / "audit_log.csv"

CANDIDATE_BG_PATHS = [
    BASE_DIR / "jp.gif",
    BASE_DIR / "assets" / "jp.gif",
    BASE_DIR / "assets" / "bg.jpg",
    BASE_DIR / "assets" / "bg.png",
]

TIER_COLORS = {
    "High Performer": "#059669",
    "Steady": "#0284C7",
    "At Risk": "#E11D48",
}

TIER_BADGE_BG = {
    "High Performer": "linear-gradient(135deg, rgba(16,185,129,0.22), rgba(255,255,255,0.92))",
    "Steady": "linear-gradient(135deg, rgba(14,165,233,0.22), rgba(255,255,255,0.92))",
    "At Risk": "linear-gradient(135deg, rgba(244,63,94,0.24), rgba(255,255,255,0.92))",
}

PRESETS = {
    "Balanced / Default": {
        "track": "AI & ML",
        "study_hours": 3.8,
        "attendance_pct": 82.0,
        "previous_gpa": 7.6,
        "assignments_completed": 8,
        "quiz_avg": 76.0,
        "late_submissions": 1,
    },
    "High Performer": {
        "track": "AI & ML",
        "study_hours": 6.2,
        "attendance_pct": 95.0,
        "previous_gpa": 9.1,
        "assignments_completed": 10,
        "quiz_avg": 92.0,
        "late_submissions": 0,
    },
    "At Risk Alert": {
        "track": "Data Science",
        "study_hours": 1.2,
        "attendance_pct": 54.0,
        "previous_gpa": 5.4,
        "assignments_completed": 3,
        "quiz_avg": 46.0,
        "late_submissions": 5,
    },
}

st.set_page_config(
    page_title="EduPulse AI | Student Academic Risk & Performance Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def get_bg_data_uri() -> str | None:
    bg_path = next((p for p in CANDIDATE_BG_PATHS if p.exists()), None)
    if bg_path is None:
        return None
    suffix = bg_path.suffix.lower()
    mime = (
        "image/gif"
        if suffix == ".gif"
        else ("image/png" if suffix == ".png" else "image/jpeg")
    )
    encoded = base64.b64encode(bg_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def inject_web_styles(bg_uri: str | None):
    bg_css = ""
    if bg_uri:
        bg_css = f"""
        .edu-bg-layer {{
            position: fixed;
            inset: -14px;
            z-index: 0;
            pointer-events: none;
            background-image: url("{bg_uri}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            filter: blur(4px) saturate(1.05);
            transform: scale(1.03);
        }}

        .edu-bg-tint {{
            position: fixed;
            inset: 0;
            z-index: 0;
            pointer-events: none;
            background: linear-gradient(
                180deg,
                rgba(241, 245, 249, 0.24) 0%,
                rgba(226, 232, 240, 0.32) 100%
            );
        }}
        """

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        {bg_css}

        html, body, [class*="css"] {{
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        .stApp {{
            background: transparent !important;
        }}

        [data-testid="stMain"] > div {{
            position: relative;
            z-index: 1;
        }}

        #MainMenu, footer, [data-testid="stHeader"] {{
            visibility: hidden;
            height: 0px;
        }}

        .block-container {{
            padding-top: 1.15rem;
            padding-bottom: 2.5rem;
            max-width: 1360px;
        }}

        .web-topbar {{
            background: rgba(15, 23, 42, 0.88);
            backdrop-filter: blur(12px);
            color: #CBD5E1;
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 14px;
            padding: 0.55rem 1.15rem;
            margin-bottom: 0.75rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.8rem;
        }}

        .web-topbar strong {{
            color: #F8FAFC;
        }}

        .saas-navbar {{
            background: linear-gradient(125deg, rgba(9, 13, 22, 0.92) 0%, rgba(22, 32, 50, 0.92) 55%, rgba(30, 58, 95, 0.92) 100%);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 18px;
            padding: 1.25rem 1.6rem;
            margin-bottom: 1.15rem;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            box-shadow: 0 18px 34px -16px rgba(15, 23, 42, 0.45);
        }}

        .saas-brand-title {{
            color: #F8FAFC;
            font-size: 1.42rem;
            font-weight: 800;
            letter-spacing: -0.025em;
            margin: 0;
        }}

        .saas-brand-sub {{
            color: #CBD5E1;
            font-size: 0.85rem;
            margin: 0.2rem 0 0 0;
        }}

        .saas-pill {{
            background: rgba(56, 189, 248, 0.18);
            color: #E0F2FE;
            border: 1px solid rgba(125, 211, 252, 0.35);
            padding: 0.4rem 0.85rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
        }}

        [data-testid="stTabs"] [data-baseweb="tab-list"] {{
            background: rgba(255, 255, 255, 0.86);
            backdrop-filter: blur(10px);
            padding: 0.35rem;
            border-radius: 14px;
            border: 1px solid rgba(203, 213, 225, 0.9);
            gap: 0.35rem;
        }}

        [data-testid="stTabs"] [data-baseweb="tab"] {{
            border-radius: 10px;
            padding: 0.45rem 0.9rem;
            font-weight: 600;
        }}

        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: rgba(255, 255, 255, 0.84) !important;
            backdrop-filter: blur(12px);
            border: 1px solid rgba(203, 213, 225, 0.9) !important;
            border-radius: 16px !important;
            box-shadow: 0 10px 26px -14px rgba(15, 23, 42, 0.18) !important;
        }}

        .section-title {{
            font-size: 1.06rem;
            font-weight: 700;
            color: #0F172A;
            margin: 0 0 0.75rem 0;
            padding-bottom: 0.45rem;
            border-bottom: 1px solid #E2E8F0;
        }}

        .metric-box {{
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid #CBD5E1;
            border-radius: 14px;
            padding: 0.85rem 1rem;
            box-shadow: 0 6px 16px -10px rgba(15, 23, 42, 0.16);
        }}

        .metric-label {{
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #475569;
            margin-bottom: 0.2rem;
        }}

        .metric-value {{
            font-size: 1.3rem;
            font-weight: 800;
            color: #0F172A;
        }}

        .tier-banner {{
            border-radius: 16px;
            padding: 1.1rem 1.35rem;
            border: 1px solid rgba(148, 163, 184, 0.35);
            margin-bottom: 1rem;
            backdrop-filter: blur(10px);
        }}

        .delta-chip {{
            display: inline-block;
            padding: 0.22rem 0.65rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
            background: #EFF6FF;
            color: #1D4ED8;
            border: 1px solid #BFDBFE;
            margin-top: 0.35rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    if bg_uri:
        st.markdown(
            '<div class="edu-bg-layer"></div><div class="edu-bg-tint"></div>',
            unsafe_allow_html=True,
        )


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return None, None
    pipeline = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return pipeline, metadata


def apply_preset(preset_key: str):
    vals = PRESETS[preset_key]
    st.session_state["track_input"] = vals["track"]
    st.session_state["study_input"] = vals["study_hours"]
    st.session_state["att_input"] = vals["attendance_pct"]
    st.session_state["gpa_input"] = vals["previous_gpa"]
    st.session_state["assign_input"] = vals["assignments_completed"]
    st.session_state["quiz_input"] = vals["quiz_avg"]
    st.session_state["late_input"] = vals["late_submissions"]
    st.session_state["d_study"] = 0.0
    st.session_state["d_att"] = 0.0
    st.session_state["d_quiz"] = 0.0


def init_inputs():
    defaults = PRESETS["Balanced / Default"]
    st.session_state.setdefault("track_input", defaults["track"])
    st.session_state.setdefault("study_input", defaults["study_hours"])
    st.session_state.setdefault("att_input", defaults["attendance_pct"])
    st.session_state.setdefault("gpa_input", defaults["previous_gpa"])
    st.session_state.setdefault("assign_input", defaults["assignments_completed"])
    st.session_state.setdefault("quiz_input", defaults["quiz_avg"])
    st.session_state.setdefault("late_input", defaults["late_submissions"])
    st.session_state.setdefault("d_study", 0.0)
    st.session_state.setdefault("d_att", 0.0)
    st.session_state.setdefault("d_quiz", 0.0)
    if "audit_log" not in st.session_state:
        if AUDIT_CSV_PATH.exists():
            st.session_state.audit_log = pd.read_csv(AUDIT_CSV_PATH).to_dict("records")
        else:
            st.session_state.audit_log = []


def compute_readiness_score(row_dict: dict) -> float:
    """Computes a 0-100% readiness score from academic inputs (drops <15% on low inputs)."""
    study_norm = max(0.0, (row_dict["study_hours"] - 0.5) / 8.5)
    att_norm = max(0.0, (row_dict["attendance_pct"] - 40.0) / 60.0)
    gpa_norm = max(0.0, (row_dict["previous_gpa"] - 4.0) / 6.0)
    assign_norm = max(0.0, row_dict["assignments_completed"] / 10.0)
    quiz_norm = max(0.0, (row_dict["quiz_avg"] - 30.0) / 70.0)
    late_penalty = min(0.35, row_dict["late_submissions"] * 0.05)

    raw_score = (
        0.20 * study_norm
        + 0.25 * att_norm
        + 0.20 * gpa_norm
        + 0.15 * assign_norm
        + 0.20 * quiz_norm
        - late_penalty
    )
    return round(float(np.clip(raw_score * 100.0, 1.0, 100.0)), 1)


def compute_local_attribution(pipeline, row_df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    baseline = metadata.get("baseline_stats", {})
    base_probs = pipeline.predict_proba(row_df)[0]
    classes = list(pipeline.classes_)
    target_idx = classes.index("At Risk") if "At Risk" in classes else 0
    base_risk_prob = float(base_probs[target_idx])

    impacts = []
    for col in RAW_NUMERIC_COLS:
        if col not in baseline:
            continue
        perturbed = row_df.copy()
        perturbed[col] = float(baseline[col]["mean"])
        new_probs = pipeline.predict_proba(perturbed)[0]
        new_risk_prob = float(new_probs[target_idx])
        delta_risk = (base_risk_prob - new_risk_prob) * 100.0
        impacts.append(
            {
                "Feature": col.replace("_", " ").title(),
                "Student Value": row_df[col].iloc[0],
                "Cohort Avg": baseline[col]["mean"],
                "Risk Impact (% pts)": round(delta_risk, 2),
            }
        )

    imp_df = pd.DataFrame(impacts)
    if not imp_df.empty:
        imp_df = imp_df.reindex(
            imp_df["Risk Impact (% pts)"].abs().sort_values(ascending=False).index
        )
    return imp_df


def build_action_brief(
    student_name: str,
    student_id: str,
    row_dict: dict,
    pred_tier: str,
    prob_map: dict,
    readiness_score: float,
    attr_df: pd.DataFrame,
) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "EDUPULSE AI — INDIVIDUAL STUDENT ACADEMIC BRIEF",
        f"Generated: {now_str}",
        f"Student: {student_name} ({student_id}) | Track: {row_dict['track']}",
        "-" * 56,
        f"Predicted Tier  : {pred_tier}",
        f"Readiness Score : {readiness_score:.1f}%",
        "Class Probabilities: " + ", ".join(f"{k}: {v*100:.1f}%" for k, v in prob_map.items()),
        "-" * 56,
        "Key Metrics:",
        f"  • Daily Study Hours       : {row_dict['study_hours']} hrs",
        f"  • Attendance              : {row_dict['attendance_pct']}%",
        f"  • Previous GPA            : {row_dict['previous_gpa']} / 10.0",
        f"  • Assignments Completed   : {row_dict['assignments_completed']} / 10",
        f"  • Quiz Average            : {row_dict['quiz_avg']}%",
        f"  • Late Submissions        : {row_dict['late_submissions']}",
        "-" * 56,
        "Top Risk Drivers (vs Cohort Mean):",
    ]
    if not attr_df.empty:
        for _, r in attr_df.head(3).iterrows():
            direction = "raises risk" if r["Risk Impact (% pts)"] > 0 else "lowers risk"
            lines.append(
                f"  • {r['Feature']}: {r['Student Value']} (avg {r['Cohort Avg']}) -> {r['Risk Impact (% pts)']:+.2f} pts ({direction})"
            )
    lines.extend(
        [
            "-" * 56,
            "2-Week Recommended Focus:",
            "  1. Stabilize daily study blocks and close missing/late assignments first.",
            "  2. Target attendance above 80% and review weakest quiz topics every 3 days.",
        ]
    )
    return "\n".join(lines)


def validate_single_inputs(row_dict: dict) -> list:
    warnings = []
    if row_dict["attendance_pct"] < 60:
        warnings.append("Attendance is below 60%, which strongly elevates academic risk.")
    if row_dict["late_submissions"] >= 4:
        warnings.append("High number of late submissions detected (>= 4).")
    if row_dict["previous_gpa"] < 5.5 and row_dict["study_hours"] > 5.5:
        warnings.append("High study hours paired with low GPA suggests study strategy or concept gaps.")
    return warnings


def main():
    bg_uri = get_bg_data_uri()
    inject_web_styles(bg_uri)
    init_inputs()
    pipeline, metadata = load_artifacts()

    if pipeline is None or metadata is None:
        st.error("Model artifacts not found. Please run `python train_model.py` first.")
        return

    best_model = metadata.get("best_model", "ML Pipeline")
    test_acc = metadata.get("test_accuracy", 0.0) * 100
    test_f1 = metadata.get("test_f1_macro", 0.0) * 100

    st.markdown(
        """
        <div class="web-topbar">
            <div><strong>EduPulse Portal</strong> &nbsp;/&nbsp; Student Success &amp; Early-Warning Suite</div>
            <div>Workspace: <strong>Codomax ML Capstone</strong> &nbsp;•&nbsp; Status: <strong>Live</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="saas-navbar">
            <div>
                <p class="saas-brand-title">EduPulse AI — Student Academic Risk & Performance Platform</p>
                <p class="saas-brand-sub">Module 06 Final ML Capstone • Predictive Analytics, What-If Simulation & Cohort Triage</p>
            </div>
            <div style="display:flex; gap:0.5rem; flex-wrap:wrap; align-items:center;">
                <span class="saas-pill">Model: {best_model}</span>
                <span class="saas-pill">Test Acc: {test_acc:.1f}%</span>
                <span class="saas-pill">Macro-F1: {test_f1:.1f}%</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_sim, tab_batch, tab_telemetry, tab_audit = st.tabs(
        [
            "🎯 Live Predictor & What-If Simulator",
            "📊 Batch Cohort Triage",
            "🧠 Model Telemetry & Explainability",
            "📝 Session Audit Log",
        ]
    )

    # -------------------------------------------------------------------------
    # TAB 1: LIVE PREDICTOR & WHAT-IF SIMULATOR
    # -------------------------------------------------------------------------
    with tab_sim:
        with st.container(border=True):
            p1, p2, p3 = st.columns(3)
            with p1:
                st.button("⚡ Load Balanced Profile", on_click=apply_preset, args=("Balanced / Default",), width="stretch")
            with p2:
                st.button("🌟 Load High Performer", on_click=apply_preset, args=("High Performer",), width="stretch")
            with p3:
                st.button("🚨 Load At-Risk Scenario", on_click=apply_preset, args=("At Risk Alert",), width="stretch")

        left_col, right_col = st.columns([1.02, 1.38], gap="large")

        with left_col:
            with st.container(border=True):
                st.markdown('<div class="section-title">Student Profile & Academic Signals</div>', unsafe_allow_html=True)

                student_name = st.text_input("Student Name", value="Snehal Jadhav")
                student_id = st.text_input("Student ID", value="STU-2026-01")
                track = st.selectbox("Learning Track", ["AI & ML", "Data Science", "Python", "Full Stack"], key="track_input")

                study_hours = st.slider("Daily Study Hours", min_value=0.5, max_value=9.0, step=0.1, key="study_input")
                attendance_pct = st.slider("Attendance (%)", min_value=40.0, max_value=100.0, step=0.5, key="att_input")
                previous_gpa = st.slider("Previous GPA (out of 10)", min_value=4.0, max_value=10.0, step=0.05, key="gpa_input")
                assignments_completed = st.slider("Assignments Completed (out of 10)", min_value=0, max_value=10, step=1, key="assign_input")
                quiz_avg = st.slider("Quiz Average (%)", min_value=30.0, max_value=100.0, step=0.5, key="quiz_input")
                late_submissions = st.slider("Late Submissions", min_value=0, max_value=8, step=1, key="late_input")

                st.markdown('<div class="section-title" style="margin-top:1rem;">What-If Adjustment Delta</div>', unsafe_allow_html=True)
                delta_study = st.slider("Δ Daily Study Hours", min_value=-2.0, max_value=4.0, step=0.25, key="d_study")
                delta_attendance = st.slider("Δ Attendance (%)", min_value=-20.0, max_value=20.0, step=1.0, key="d_att")
                delta_quiz = st.slider("Δ Quiz Average (%)", min_value=-20.0, max_value=20.0, step=1.0, key="d_quiz")

                log_btn = st.button("📌 Record Assessment to Audit Log", type="primary", width="stretch")
                st.caption(f"Logged entries: **{len(st.session_state.audit_log)}** — open **📝 Session Audit Log** tab to view.")

        base_row = {
            "track": track,
            "study_hours": study_hours,
            "attendance_pct": attendance_pct,
            "previous_gpa": previous_gpa,
            "assignments_completed": assignments_completed,
            "quiz_avg": quiz_avg,
            "late_submissions": late_submissions,
        }
        sim_row = {
            "track": track,
            "study_hours": float(np.clip(study_hours + delta_study, 0.5, 9.0)),
            "attendance_pct": float(np.clip(attendance_pct + delta_attendance, 40.0, 100.0)),
            "previous_gpa": previous_gpa,
            "assignments_completed": assignments_completed,
            "quiz_avg": float(np.clip(quiz_avg + delta_quiz, 30.0, 100.0)),
            "late_submissions": late_submissions,
        }

        base_df = pd.DataFrame([base_row])
        sim_df = pd.DataFrame([sim_row])

        base_pred = str(pipeline.predict(base_df)[0])
        base_probs = pipeline.predict_proba(base_df)[0]
        sim_pred = str(pipeline.predict(sim_df)[0])
        sim_probs = pipeline.predict_proba(sim_df)[0]

        classes = list(pipeline.classes_)
        base_prob_map = {cls: float(p) for cls, p in zip(classes, base_probs)}
        sim_prob_map = {cls: float(p) for cls, p in zip(classes, sim_probs)}

        readiness_score = compute_readiness_score(sim_row)
        enriched_sim = enrich_student_features(sim_df).iloc[0]
        attr_df = compute_local_attribution(pipeline, sim_df, metadata)

        if log_btn:
            top_driver = attr_df.iloc[0]["Feature"] if not attr_df.empty else "N/A"
            st.session_state.audit_log.insert(
                0,
                {
                    "Timestamp": datetime.now().strftime("%H:%M:%S"),
                    "Student ID": student_id,
                    "Student Name": student_name,
                    "Track": track,
                    "Predicted Tier": sim_pred,
                    "Readiness (%)": readiness_score,
                    "Model Conf. (%)": round(max(sim_prob_map.values()) * 100, 1),
                    "Primary Driver": top_driver,
                },
            )
            pd.DataFrame(st.session_state.audit_log).to_csv(AUDIT_CSV_PATH, index=False)
            st.toast(f"Logged prediction for {student_name} ({sim_pred}).")

        with right_col:
            color = TIER_COLORS.get(sim_pred, "#0F172A")
            bg = TIER_BADGE_BG.get(sim_pred, "rgba(255,255,255,0.94)")
            model_conf = max(sim_prob_map.values()) * 100
            shift_note = (
                f"Shifted from {base_pred} to {sim_pred}"
                if base_pred != sim_pred
                else f"Consistent with baseline ({base_pred})"
            )

            st.markdown(
                f"""
                <div class="tier-banner" style="background:{bg}; border-left: 5px solid {color};">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.75rem;">
                        <div>
                            <span style="font-size:0.75rem; font-weight:700; text-transform:uppercase; color:{color};">
                                Live Model Prediction
                            </span>
                            <h3 style="margin:0.15rem 0 0 0; color:#0F172A;">{sim_pred}</h3>
                            <span class="delta-chip">{shift_note}</span>
                        </div>
                        <div style="display:flex; gap:1.25rem; text-align:right;">
                            <div>
                                <span style="font-size:0.76rem; color:#475569; font-weight:600;">Readiness Score</span>
                                <div style="font-size:1.45rem; font-weight:800; color:#0F172A;">{readiness_score:.1f}%</div>
                            </div>
                            <div>
                                <span style="font-size:0.76rem; color:#475569; font-weight:600;">Model Confidence</span>
                                <div style="font-size:1.45rem; font-weight:800; color:{color};">{model_conf:.1f}%</div>
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            for w in validate_single_inputs(sim_row):
                st.warning(w)

            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(
                f'<div class="metric-box"><div class="metric-label">Engagement Index</div><div class="metric-value">{enriched_sim["engagement_index"]:.1f}</div></div>',
                unsafe_allow_html=True,
            )
            m2.markdown(
                f'<div class="metric-box"><div class="metric-label">Study Efficiency</div><div class="metric-value">{enriched_sim["study_efficiency"]:.2f}</div></div>',
                unsafe_allow_html=True,
            )
            m3.markdown(
                f'<div class="metric-box"><div class="metric-label">Assessment Momentum</div><div class="metric-value">{enriched_sim["assessment_momentum"]:.1f}</div></div>',
                unsafe_allow_html=True,
            )
            m4.markdown(
                f'<div class="metric-box"><div class="metric-label">Risk Pressure</div><div class="metric-value">{enriched_sim["risk_pressure_score"]:.1f}</div></div>',
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            with st.container(border=True):
                prob_df = pd.DataFrame(
                    {
                        "Tier": classes,
                        "Baseline (%)": [round(base_prob_map[c] * 100, 1) for c in classes],
                        "Simulated (%)": [round(sim_prob_map[c] * 100, 1) for c in classes],
                    }
                )
                fig_prob = go.Figure()
                fig_prob.add_trace(
                    go.Bar(
                        name="Baseline",
                        x=prob_df["Tier"],
                        y=prob_df["Baseline (%)"],
                        marker_color="#94A3B8",
                        marker_cornerradius=6,
                    )
                )
                fig_prob.add_trace(
                    go.Bar(
                        name="Simulated",
                        x=prob_df["Tier"],
                        y=prob_df["Simulated (%)"],
                        marker_color=[TIER_COLORS.get(c, "#2563EB") for c in classes],
                        marker_cornerradius=6,
                    )
                )
                fig_prob.update_layout(
                    barmode="group",
                    title="Class Probability Distribution: Baseline vs. Simulated",
                    yaxis_title="Probability (%)",
                    template="plotly_white",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=280,
                    margin=dict(l=20, r=20, t=45, b=20),
                )
                st.plotly_chart(fig_prob, width="stretch")

                if not attr_df.empty:
                    fig_attr = px.bar(
                        attr_df,
                        x="Risk Impact (% pts)",
                        y="Feature",
                        orientation="h",
                        title="Per-Student At-Risk Probability Shift vs. Cohort Average",
                        color="Risk Impact (% pts)",
                        color_continuous_scale=["#059669", "#F1F5F9", "#E11D48"],
                        color_continuous_midpoint=0,
                    )
                    fig_attr.update_layout(
                        yaxis=dict(autorange="reversed"),
                        template="plotly_white",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=280,
                        margin=dict(l=20, r=20, t=45, b=20),
                    )
                    st.plotly_chart(fig_attr, width="stretch")

            brief_txt = build_action_brief(
                student_name, student_id, sim_row, sim_pred, sim_prob_map, readiness_score, attr_df
            )
            st.download_button(
                label="📄 Download Individual Student Action Brief (.txt)",
                data=brief_txt,
                file_name=f"{student_id}_{student_name.replace(' ', '_')}_brief.txt",
                mime="text/plain",
                width="stretch",
            )

    # -------------------------------------------------------------------------
    # TAB 2: BATCH COHORT TRIAGE
    # -------------------------------------------------------------------------
    with tab_batch:
        with st.container(border=True):
            st.markdown('<div class="section-title">Batch CSV Upload & Early-Warning Risk Triage</div>', unsafe_allow_html=True)

            top_c1, top_c2 = st.columns([2, 1])
            with top_c1:
                uploaded = st.file_uploader("Upload Cohort CSV", type=["csv"])
            with top_c2:
                risk_threshold = st.slider(
                    "At-Risk Probability Cutoff",
                    min_value=0.15,
                    max_value=0.85,
                    value=0.45,
                    step=0.05,
                )

            batch_df = None
            if uploaded is not None:
                batch_df = pd.read_csv(uploaded)
            elif SAMPLE_CSV_PATH.exists():
                st.info("Showing bundled `sample_cohort.csv` (30 students). Upload your own CSV anytime.")
                batch_df = pd.read_csv(SAMPLE_CSV_PATH)

            if batch_df is not None and not batch_df.empty:
                required_cols = RAW_NUMERIC_COLS + RAW_CATEGORICAL_COLS
                missing = [c for c in required_cols if c not in batch_df.columns]
                if missing:
                    st.error(f"Missing required columns in CSV: {', '.join(missing)}")
                else:
                    probs = pipeline.predict_proba(batch_df[required_cols])
                    classes = list(pipeline.classes_)
                    prob_frame = pd.DataFrame(probs, columns=[f"P({c})" for c in classes])

                    risk_col = "P(At Risk)" if "P(At Risk)" in prob_frame.columns else prob_frame.columns[-1]
                    triage_tier = np.where(
                        prob_frame[risk_col] >= risk_threshold,
                        "At Risk",
                        pipeline.predict(batch_df[required_cols]),
                    )

                    scored = batch_df.copy()
                    scored["Predicted Tier"] = triage_tier
                    scored["Confidence (%)"] = np.round(np.max(probs, axis=1) * 100, 1)
                    for col_name in prob_frame.columns:
                        scored[col_name] = np.round(prob_frame[col_name] * 100, 1)

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Students", len(scored))
                    c2.metric("At Risk Flagged", int((scored["Predicted Tier"] == "At Risk").sum()))
                    c3.metric("Steady", int((scored["Predicted Tier"] == "Steady").sum()))
                    c4.metric("High Performer", int((scored["Predicted Tier"] == "High Performer").sum()))

                    fig_dist = px.histogram(
                        scored,
                        x="Predicted Tier",
                        color="Predicted Tier",
                        color_discrete_map=TIER_COLORS,
                        title="Cohort Tier Distribution Under Active Risk Cutoff",
                    )
                    fig_dist.update_layout(
                        template="plotly_white",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=280,
                        showlegend=False,
                    )
                    st.plotly_chart(fig_dist, width="stretch")

                    only_at_risk = st.checkbox("Show only At-Risk students", value=False)
                    view_df = scored[scored["Predicted Tier"] == "At Risk"] if only_at_risk else scored
                    st.dataframe(view_df, width="stretch", hide_index=True)

                    csv_buf = StringIO()
                    scored.to_csv(csv_buf, index=False)
                    st.download_button(
                        label="⬇️ Download Scored Cohort CSV",
                        data=csv_buf.getvalue(),
                        file_name="scored_student_cohort.csv",
                        mime="text/csv",
                    )

    # -------------------------------------------------------------------------
    # TAB 3: MODEL TELEMETRY & EXPLAINABILITY
    # -------------------------------------------------------------------------
    with tab_telemetry:
        with st.container(border=True):
            st.markdown('<div class="section-title">5-Fold Stratified Cross-Validation Leaderboard</div>', unsafe_allow_html=True)
            lb_df = pd.DataFrame(metadata.get("leaderboard", []))
            if not lb_df.empty:
                st.dataframe(
                    lb_df.rename(
                        columns={
                            "model": "Algorithm",
                            "cv_f1_macro": "CV Macro-F1",
                            "cv_accuracy": "CV Accuracy",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )

            st.markdown('<div class="section-title" style="margin-top:1.25rem;">Global Feature Importance & Holdout Confusion Matrix</div>', unsafe_allow_html=True)
            g1, g2 = st.columns([1.25, 1.0], gap="large")

            with g1:
                imp_df = pd.DataFrame(metadata.get("feature_importances", [])).head(10)
                if not imp_df.empty:
                    fig_imp = px.bar(
                        imp_df,
                        x="importance",
                        y="feature",
                        orientation="h",
                        title="Top 10 Model Features (Normalized)",
                        labels={"importance": "Relative Importance", "feature": "Feature"},
                    )
                    fig_imp.update_layout(
                        yaxis=dict(autorange="reversed"),
                        template="plotly_white",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=340,
                    )
                    st.plotly_chart(fig_imp, width="stretch")

            with g2:
                cm = metadata.get("confusion_matrix", [])
                labels = metadata.get("classes", ["High Performer", "Steady", "At Risk"])
                if cm:
                    fig_cm = px.imshow(
                        cm,
                        x=labels,
                        y=labels,
                        text_auto=True,
                        color_continuous_scale="Blues",
                        title="Holdout Test Confusion Matrix",
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                    )
                    fig_cm.update_layout(
                        template="plotly_white",
                        paper_bgcolor="rgba(0,0,0,0)",
                        height=340,
                    )
                    st.plotly_chart(fig_cm, width="stretch")

    # -------------------------------------------------------------------------
    # TAB 4: SESSION AUDIT LOG
    # -------------------------------------------------------------------------
    with tab_audit:
        with st.container(border=True):
            st.markdown('<div class="section-title">Session Prediction Audit Log</div>', unsafe_allow_html=True)
            if st.session_state.audit_log:
                audit_df = pd.DataFrame(st.session_state.audit_log)
                st.dataframe(audit_df, width="stretch", hide_index=True)
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    st.download_button(
                        label="⬇️ Export Audit Log (.csv)",
                        data=audit_df.to_csv(index=False),
                        file_name="session_audit_log.csv",
                        mime="text/csv",
                    )
                with col_b:
                    if st.button("🗑️ Clear Audit Log"):
                        st.session_state.audit_log = []
                        if AUDIT_CSV_PATH.exists():
                            AUDIT_CSV_PATH.unlink()
                        st.rerun()
            else:
                st.info("No predictions recorded yet. Click **Record Assessment to Audit Log** in the first tab.")


if __name__ == "__main__":
    main()