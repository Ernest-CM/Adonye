"""
Parkinson's Disease Detection — Streamlit Demo
Run: streamlit run app.py
"""

import os
import sys
import tempfile
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="PD Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
SPEECH_FEATURES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
]

DEMO_PD = [
    119.992, 157.302, 74.997, 0.00784, 0.00007, 0.0037, 0.00554, 0.01109,
    0.04374, 0.426, 0.02182, 0.0313, 0.02971, 0.06545, 0.02211, 21.033,
    0.414783, 0.815285, -4.813031, 0.266482, 2.301442, 0.284654,
]
DEMO_HC = [
    197.076, 206.896, 192.055, 0.00289, 0.00001, 0.00166, 0.00168, 0.00498,
    0.01098, 0.097, 0.00563, 0.0068, 0.00802, 0.01689, 0.00339, 26.775,
    0.422229, 0.741367, -7.3483, 0.177551, 1.743867, 0.085569,
]

STRATEGIES = {
    "hybrid":      {"label": "Hybrid Fusion ★", "needs": ["speech", "handwriting", "gait"]},
    "late":        {"label": "Late Fusion",      "needs": ["speech", "handwriting", "gait"]},
    "early":       {"label": "Early Fusion",     "needs": ["speech", "handwriting", "gait"]},
    "speech":      {"label": "Speech Only",      "needs": ["speech"]},
    "handwriting": {"label": "Handwriting Only", "needs": ["handwriting"]},
    "gait":        {"label": "Gait Only",        "needs": ["gait"]},
}

MODEL_AUC = {
    "hybrid": 0.9814, "late": 0.9565, "early": 0.8199,
    "speech": 0.9814, "handwriting": 0.9470, "gait": 0.7000,
}

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark clinical theme */
.stApp {
    background: #080d14;
    color: #e8edf4;
}

/* Main header */
.pd-header {
    display: flex;
    align-items: baseline;
    gap: 16px;
    padding: 32px 0 8px;
    border-bottom: 1px solid #1e2d42;
    margin-bottom: 32px;
}
.pd-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #e8edf4;
    margin: 0;
}
.pd-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #4a6080;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* Section card */
.modality-card {
    background: #0d1520;
    border: 1px solid #1e2d42;
    border-radius: 10px;
    padding: 20px 22px 16px;
    margin-bottom: 16px;
    position: relative;
}
.modality-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #3a8fd6;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.modality-label::before {
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #3a8fd6;
}

/* Result panel */
.result-panel {
    background: #0d1520;
    border: 1px solid #1e2d42;
    border-radius: 12px;
    padding: 28px 32px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-panel.pd-positive {
    border-color: #c0392b;
    background: linear-gradient(135deg, #1a0d0d 0%, #0d1520 60%);
}
.result-panel.hc-positive {
    border-color: #1a7a4a;
    background: linear-gradient(135deg, #0a1a12 0%, #0d1520 60%);
}

/* Probability ring (SVG-based) */
.prob-ring-wrap {
    display: flex;
    justify-content: center;
    margin: 8px 0 16px;
}

/* Risk badge */
.risk-badge {
    display: inline-block;
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    padding: 10px 28px;
    border-radius: 6px;
    margin: 8px 0;
}
.risk-badge.pd { background: #3d1212; color: #e05252; border: 1px solid #6b2020; }
.risk-badge.hc { background: #0d2b1a; color: #3dba77; border: 1px solid #1a5c36; }

.prob-value {
    font-family: 'DM Mono', monospace;
    font-size: 3.2rem;
    font-weight: 500;
    line-height: 1;
    margin: 4px 0;
}
.prob-value.pd { color: #e05252; }
.prob-value.hc { color: #3dba77; }

.prob-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4a6080;
    margin-bottom: 12px;
}

/* Latency chip */
.latency-chip {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #4a6080;
    background: #111c2a;
    border: 1px solid #1e2d42;
    border-radius: 20px;
    padding: 4px 14px;
    margin-top: 8px;
}

/* AUC badge */
.auc-chip {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #3a8fd6;
    letter-spacing: 0.1em;
}

/* Stat rows */
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #111c2a;
    font-size: 0.82rem;
}
.stat-row:last-child { border-bottom: none; }
.stat-key { color: #4a6080; font-family: 'DM Mono', monospace; font-size: 0.72rem; }
.stat-val { color: #a8c0d6; font-weight: 500; }

/* Sidebar overrides */
[data-testid="stSidebar"] {
    background: #0a0f18 !important;
    border-right: 1px solid #1e2d42;
}
[data-testid="stSidebar"] * { color: #c0ccd8; }

/* Button overrides */
.stButton > button {
    background: #111c2a;
    border: 1px solid #1e2d42;
    color: #a8c0d6;
    border-radius: 6px;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.06em;
    padding: 6px 16px;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    border-color: #3a8fd6;
    color: #3a8fd6;
    background: #0d1a2a;
}

/* Primary run button */
div[data-testid="stButton"] button[kind="primary"] {
    background: #1a3a5c;
    border: 1px solid #3a8fd6;
    color: #7ac4f0;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 0.04em;
    padding: 12px 0;
    width: 100%;
    border-radius: 8px;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    background: #1e4a72;
    border-color: #5ab4f8;
    color: #a0d8ff;
}

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
    background: #080d14 !important;
    border: 1px dashed #1e2d42 !important;
    border-radius: 8px !important;
}

/* Progress bar */
.stProgress > div > div { background: #1e2d42; border-radius: 4px; }
.stProgress > div > div > div { border-radius: 4px; }

/* Selectbox */
[data-testid="stSelectbox"] select,
[data-baseweb="select"] {
    background: #0d1520;
    border-color: #1e2d42;
    color: #c0ccd8;
}

/* Slider */
[data-testid="stSlider"] { color: #4a6080; }

/* Input */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background: #080d14;
    border-color: #1e2d42;
    color: #c0ccd8;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
}

/* Divider */
hr { border-color: #1e2d42 !important; }

/* Warning / info boxes */
[data-testid="stAlert"] {
    background: #0d1520;
    border-color: #1e2d42;
    color: #8aabcc;
}
</style>
""", unsafe_allow_html=True)


# ── Pipeline loader ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    from src.inference.pipeline import PDInferencePipeline
    return PDInferencePipeline()


# ── Helpers ───────────────────────────────────────────────────────────────────
def svg_gauge(prob: float, is_pd: bool) -> str:
    """Return an SVG arc gauge for the probability value."""
    r = 70
    cx = cy = 90
    stroke_bg = "#1e2d42"
    stroke_fg = "#e05252" if is_pd else "#3dba77"
    circumference = 2 * 3.14159 * r
    arc = circumference * 0.75          # 270-degree arc
    offset = arc * (1 - prob)           # unfilled portion
    # rotate so arc starts at bottom-left
    rotate = "rotate(135, 90, 90)"
    return f"""
    <svg width="180" height="180" viewBox="0 0 180 180">
      <circle cx="{cx}" cy="{cy}" r="{r}"
              fill="none" stroke="{stroke_bg}" stroke-width="10"
              stroke-dasharray="{arc:.1f} {circumference:.1f}"
              transform="{rotate}" stroke-linecap="round"/>
      <circle cx="{cx}" cy="{cy}" r="{r}"
              fill="none" stroke="{stroke_fg}" stroke-width="10"
              stroke-dasharray="{arc - offset:.1f} {circumference:.1f}"
              transform="{rotate}" stroke-linecap="round"
              style="transition: stroke-dasharray 0.5s ease;"/>
      <text x="{cx}" y="{cy - 6}" text-anchor="middle"
            font-family="DM Mono, monospace" font-size="22" font-weight="500"
            fill="{stroke_fg}">{prob*100:.1f}%</text>
      <text x="{cx}" y="{cy + 16}" text-anchor="middle"
            font-family="DM Sans, sans-serif" font-size="9" letter-spacing="2"
            fill="#4a6080" text-transform="uppercase">PD RISK</text>
    </svg>"""


def save_upload(uploaded_file) -> str:
    """Save an uploaded file to a temp path and return the path."""
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 24px">
      <div style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;
                  letter-spacing:-0.02em;color:#e8edf4">PD Detection</div>
      <div style="font-family:DM Mono,monospace;font-size:0.65rem;
                  letter-spacing:0.14em;color:#3a8fd6;text-transform:uppercase;
                  margin-top:2px">Multimodal System v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Fusion Strategy**")
    strategy_key = st.selectbox(
        "Strategy",
        options=list(STRATEGIES.keys()),
        format_func=lambda k: STRATEGIES[k]["label"],
        label_visibility="collapsed",
    )
    needs = STRATEGIES[strategy_key]["needs"]

    st.markdown(
        f'<div class="auc-chip">Model AUC: {MODEL_AUC[strategy_key]:.4f}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("**Decision Threshold**")
    threshold = st.slider(
        "Threshold", min_value=0.1, max_value=0.9,
        value=0.5, step=0.05, label_visibility="collapsed",
    )
    st.caption(f"Classify as PD if probability ≥ {threshold:.2f}")

    st.markdown("---")
    st.caption(
        "Models trained on UCI Parkinson's (speech), "
        "PhysioNet GaitPDB (gait), and spiral-drawing images (handwriting)."
    )


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="pd-header">
  <h1 class="pd-title">Parkinson's Disease Detection</h1>
  <span class="pd-subtitle">Multimodal Clinical AI · Research Prototype</span>
</div>
""", unsafe_allow_html=True)


# ── Input columns ─────────────────────────────────────────────────────────────
input_col, result_col = st.columns([3, 2], gap="large")

speech_features = None
hw_path = None
gait_path = None

with input_col:

    # ── Speech ────────────────────────────────────────────────────────────────
    if "speech" in needs:
        st.markdown('<div class="modality-card">', unsafe_allow_html=True)
        st.markdown('<div class="modality-label">Speech Biomarkers</div>', unsafe_allow_html=True)

        demo_col1, demo_col2, spacer = st.columns([1, 1, 3])
        with demo_col1:
            if st.button("▶ Demo PD"):
                st.session_state["speech_vals"] = DEMO_PD[:]
        with demo_col2:
            if st.button("▶ Demo HC"):
                st.session_state["speech_vals"] = DEMO_HC[:]

        uploaded_csv = st.file_uploader(
            "Upload CSV (one row, 22 features)",
            type=["csv"], key="speech_csv",
            label_visibility="collapsed",
            help="CSV must contain 22 acoustic features in the correct column order.",
        )

        if uploaded_csv:
            import pandas as pd
            try:
                df = pd.read_csv(uploaded_csv)
                if df.shape[1] >= 22:
                    st.session_state["speech_vals"] = df.iloc[0, :22].tolist()
                    st.success(f"Loaded {df.shape[0]} row(s) — using first row.")
                else:
                    st.error(f"Expected ≥22 columns, got {df.shape[1]}.")
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")

        vals = st.session_state.get("speech_vals", [0.0] * 22)

        with st.expander("Feature values", expanded=False):
            cols_a, cols_b = st.columns(2)
            updated = []
            for i, feat in enumerate(SPEECH_FEATURES):
                col = cols_a if i % 2 == 0 else cols_b
                updated.append(
                    col.number_input(
                        feat, value=float(vals[i]),
                        format="%.6f", key=f"sf_{i}",
                        label_visibility="visible",
                    )
                )
            speech_features = np.array(updated, dtype=np.float32)
        else:
            speech_features = np.array(vals, dtype=np.float32)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Handwriting ───────────────────────────────────────────────────────────
    if "handwriting" in needs:
        st.markdown('<div class="modality-card">', unsafe_allow_html=True)
        st.markdown('<div class="modality-label">Handwriting / Spiral Drawing</div>', unsafe_allow_html=True)

        hw_file = st.file_uploader(
            "Upload spiral drawing image",
            type=["png", "jpg", "jpeg"],
            key="hw_file",
            label_visibility="collapsed",
        )
        if hw_file:
            preview_col, _ = st.columns([1, 2])
            with preview_col:
                st.image(hw_file, caption="Uploaded drawing", use_container_width=True)
            hw_path = save_upload(hw_file)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Gait ──────────────────────────────────────────────────────────────────
    if "gait" in needs:
        st.markdown('<div class="modality-card">', unsafe_allow_html=True)
        st.markdown('<div class="modality-label">Gait Recording (.ts)</div>', unsafe_allow_html=True)

        gait_file = st.file_uploader(
            "Upload PhysioNet .ts gait file",
            type=["ts", "txt"],
            key="gait_file",
            label_visibility="collapsed",
        )
        if gait_file:
            st.caption(f"✓  {gait_file.name}  ({gait_file.size / 1024:.1f} KB)")
            gait_path = save_upload(gait_file)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Run button ────────────────────────────────────────────────────────────
    st.markdown("<div style='height:8px'/>", unsafe_allow_html=True)

    missing = []
    if "speech"      in needs and speech_features is None: missing.append("speech")
    if "handwriting" in needs and hw_path is None:          missing.append("handwriting image")
    if "gait"        in needs and gait_path is None:        missing.append("gait recording")

    run_disabled = bool(missing)
    run_label = "Run Analysis" if not missing else f"Waiting for: {', '.join(missing)}"

    run_clicked = st.button(run_label, type="primary", disabled=run_disabled, use_container_width=True)


# ── Results panel ─────────────────────────────────────────────────────────────
with result_col:

    if "last_result" not in st.session_state:
        # Placeholder
        st.markdown("""
        <div class="modality-card" style="text-align:center;padding:48px 24px">
          <div style="font-family:DM Mono,monospace;font-size:0.72rem;
                      letter-spacing:0.12em;color:#2a3d52;text-transform:uppercase">
            Awaiting Input
          </div>
          <div style="font-size:2.8rem;margin:16px 0;opacity:0.15">🧠</div>
          <div style="font-family:DM Sans,sans-serif;font-size:0.85rem;
                      color:#2a3d52;max-width:200px;margin:0 auto;line-height:1.6">
            Load patient data and click Run Analysis
          </div>
        </div>
        """, unsafe_allow_html=True)

    if run_clicked and not missing:
        with st.spinner("Running inference…"):
            try:
                pipeline = load_pipeline()
                result = pipeline.predict(
                    speech_features=speech_features if "speech" in needs else None,
                    handwriting_image_path=hw_path if "handwriting" in needs else None,
                    gait_file_path=gait_path if "gait" in needs else None,
                    fusion_strategy=strategy_key,
                    threshold=threshold,
                )
                st.session_state["last_result"] = result
            except Exception as e:
                st.error(f"Inference failed: {e}")

    if "last_result" in st.session_state:
        r = st.session_state["last_result"]
        prob = r["probability"]
        is_pd = r["prediction"] == 1
        cls = "pd" if is_pd else "hc"
        panel_cls = "pd-positive" if is_pd else "hc-positive"

        st.markdown(
            f'<div class="result-panel {panel_cls}">',
            unsafe_allow_html=True,
        )

        # Gauge SVG
        st.markdown(
            f'<div class="prob-ring-wrap">{svg_gauge(prob, is_pd)}</div>',
            unsafe_allow_html=True,
        )

        # Badge
        label_text = "Parkinson's Disease" if is_pd else "Healthy Control"
        st.markdown(
            f'<div style="text-align:center">'
            f'<span class="risk-badge {cls}">{label_text}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Probability bar
        bar_color = "#e05252" if is_pd else "#3dba77"
        st.markdown(f"""
        <div style="margin:20px 0 12px">
          <div style="display:flex;justify-content:space-between;
                      font-family:DM Mono,monospace;font-size:0.68rem;
                      color:#4a6080;margin-bottom:6px">
            <span>0%</span><span>PD Probability</span><span>100%</span>
          </div>
          <div style="background:#111c2a;border-radius:4px;height:8px;overflow:hidden">
            <div style="background:{bar_color};width:{prob*100:.1f}%;height:100%;
                        border-radius:4px;transition:width 0.5s ease"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Stats table
        st.markdown(f"""
        <div style="margin-top:16px">
          <div class="stat-row">
            <span class="stat-key">Strategy</span>
            <span class="stat-val">{r['model_used'].title()} Fusion</span>
          </div>
          <div class="stat-row">
            <span class="stat-key">Probability</span>
            <span class="stat-val" style="color:{bar_color}">{prob:.4f}</span>
          </div>
          <div class="stat-row">
            <span class="stat-key">Threshold</span>
            <span class="stat-val">{r['threshold']:.2f}</span>
          </div>
          <div class="stat-row">
            <span class="stat-key">Latency</span>
            <span class="stat-val">{r['latency_ms']:.1f} ms
              {"✓" if r.get("within_target") else "⚠"}</span>
          </div>
          <div class="stat-row">
            <span class="stat-key">Model AUC</span>
            <span class="stat-val">{MODEL_AUC[r['model_used']]:.4f}</span>
          </div>
          <div class="stat-row">
            <span class="stat-key">Timestamp</span>
            <span class="stat-val" style="font-family:DM Mono,monospace;
              font-size:0.7rem">{r['timestamp'][:19]}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Clear result", key="clear_btn"):
            del st.session_state["last_result"]
            st.rerun()


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:center;
            padding:4px 0 16px">
  <span style="font-family:DM Mono,monospace;font-size:0.68rem;color:#2a3d52;
               letter-spacing:0.1em">RESEARCH PROTOTYPE · NOT FOR CLINICAL USE</span>
  <span style="font-family:DM Mono,monospace;font-size:0.68rem;color:#2a3d52">
    Hybrid Fusion · AUC 0.9814 · p95 &lt; 315 ms
  </span>
</div>
""", unsafe_allow_html=True)
