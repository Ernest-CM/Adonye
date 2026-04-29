"""
Parkinson's Disease Detection — Streamlit Demo
Run: python -m streamlit run app.py
"""

import os
import sys
import tempfile
import warnings
import numpy as np
import streamlit as st

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

# Real patient data from UCI Parkinson's dataset
DEMO_PD = [119.992, 157.302, 74.997, 0.00784, 0.00007, 0.0037, 0.00554, 0.01109,
           0.04374, 0.426, 0.02182, 0.0313, 0.02971, 0.06545, 0.02211, 21.033,
           0.414783, 0.815285, -4.813031, 0.266482, 2.301442, 0.284654]
DEMO_HC = [197.076, 206.896, 192.055, 0.00289, 0.00001, 0.00166, 0.00168, 0.00498,
           0.01098, 0.097, 0.00563, 0.0068, 0.00802, 0.01689, 0.00339, 26.775,
           0.422229, 0.741367, -7.3483, 0.177551, 1.743867, 0.085569]

STRATEGIES = {
    "hybrid":      {"label": "Hybrid Fusion ★",  "needs": ["speech", "handwriting", "gait"], "auc": 0.9814},
    "late":        {"label": "Late Fusion",       "needs": ["speech", "handwriting", "gait"], "auc": 0.9565},
    "early":       {"label": "Early Fusion",      "needs": ["speech", "handwriting", "gait"], "auc": 0.8199},
    "speech":      {"label": "Speech Only",       "needs": ["speech"],                        "auc": 0.9814},
    "handwriting": {"label": "Handwriting Only",  "needs": ["handwriting"],                   "auc": 0.9470},
    "gait":        {"label": "Gait Only",         "needs": ["gait"],                          "auc": 0.7000},
}

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Serif:wght@600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #f7f8fa; color: #1a1d23; }

/* ── Layout ── */
.block-container { padding-top: 2rem !important; }

/* ── Step card ── */
.step-card {
    background: #fff;
    border: 1px solid #e2e5eb;
    border-radius: 12px;
    padding: 20px 22px 18px;
    margin-bottom: 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,.04);
}
.step-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
}
.step-num {
    width: 28px; height: 28px;
    background: #1a56db;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem; font-weight: 500;
    color: #fff; flex-shrink: 0;
}
.step-title {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.92rem; font-weight: 600; color: #1a1d23;
}
.step-desc {
    font-size: 0.8rem; color: #6b7280; margin-top: 2px;
}

/* ── Sample buttons ── */
.sample-row { display: flex; gap: 8px; margin-bottom: 10px; }

/* ── Result card ── */
.result-card {
    background: #fff;
    border-radius: 14px;
    padding: 28px 24px 24px;
    border: 2px solid #e2e5eb;
    box-shadow: 0 2px 8px rgba(0,0,0,.06);
    text-align: center;
}
.result-card.pd { border-color: #ef4444; background: #fff9f9; }
.result-card.hc { border-color: #10b981; background: #f0fdf7; }

.result-icon { font-size: 3rem; line-height: 1; margin-bottom: 8px; }
.result-label {
    font-family: 'IBM Plex Serif', serif;
    font-size: 1.5rem; font-weight: 600; margin-bottom: 4px;
}
.result-label.pd { color: #dc2626; }
.result-label.hc { color: #059669; }

.result-prob {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.6rem; font-weight: 500; line-height: 1.1;
    margin: 8px 0 4px;
}
.result-prob.pd { color: #dc2626; }
.result-prob.hc { color: #059669; }

.result-sub {
    font-size: 0.75rem; color: #9ca3af;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.08em; text-transform: uppercase;
}

/* ── Stat grid ── */
.stat-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 8px; margin-top: 18px;
}
.stat-box {
    background: #f7f8fa; border-radius: 8px;
    padding: 10px 12px;
    text-align: left;
}
.stat-box-label {
    font-size: 0.68rem; color: #9ca3af;
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase; letter-spacing: 0.06em;
    margin-bottom: 2px;
}
.stat-box-val {
    font-size: 0.92rem; font-weight: 600; color: #1a1d23;
    font-family: 'IBM Plex Mono', monospace;
}

/* ── Pending card ── */
.pending-card {
    background: #fff; border: 2px dashed #d1d5db;
    border-radius: 14px; padding: 40px 24px;
    text-align: center; color: #9ca3af;
}
.pending-icon { font-size: 2.5rem; margin-bottom: 12px; opacity: 0.4; }
.pending-text { font-size: 0.85rem; line-height: 1.6; }

/* ── Info callout ── */
.info-callout {
    background: #eff6ff; border: 1px solid #bfdbfe;
    border-radius: 8px; padding: 10px 14px;
    font-size: 0.8rem; color: #1d4ed8; margin-top: 8px;
}
.warn-callout {
    background: #fefce8; border: 1px solid #fde047;
    border-radius: 8px; padding: 10px 14px;
    font-size: 0.8rem; color: #92400e; margin-top: 8px;
}

/* ── Strategy badge ── */
.strategy-badge {
    display: inline-block;
    background: #eff6ff; color: #1d4ed8;
    border: 1px solid #bfdbfe;
    border-radius: 20px; padding: 3px 12px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem; font-weight: 500;
    margin-top: 6px;
}

/* ── Button overrides ── */
.stButton > button {
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    transition: all 0.15s !important;
}
div[data-testid="stButton"] button[kind="primary"] {
    background: #1a56db !important;
    border: none !important;
    color: #fff !important;
    padding: 12px 0 !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    width: 100% !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    background: #1648c0 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #fff !important; border-right: 1px solid #e2e5eb; }
[data-testid="stSidebar"] .block-container { padding-top: 1rem !important; }

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {
    background: #f9fafb !important;
    border: 1.5px dashed #d1d5db !important;
    border-radius: 8px !important;
}

/* ── Progress bar ── */
.stProgress > div > div { background: #e5e7eb; border-radius: 4px; }

/* ── Expander ── */
[data-testid="stExpander"] { border: 1px solid #e2e5eb !important; border-radius: 8px !important; }

/* ── Divider ── */
hr { border-color: #e2e5eb !important; }
</style>
""", unsafe_allow_html=True)


# ── Pipeline (cached — loads once, reused across all runs) ────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    from src.inference.pipeline import PDInferencePipeline
    return PDInferencePipeline()


# ── Helpers ───────────────────────────────────────────────────────────────────
def save_upload(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[-1] or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded_file.read())
        return f.name


def prob_bar(prob: float, is_pd: bool) -> str:
    color = "#ef4444" if is_pd else "#10b981"
    pct = prob * 100
    return f"""
    <div style="margin:14px 0 6px">
      <div style="display:flex;justify-content:space-between;
                  font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
                  color:#9ca3af;margin-bottom:5px">
        <span>0%</span><span style="color:{color};font-weight:500">{pct:.1f}%  PD probability</span><span>100%</span>
      </div>
      <div style="background:#e5e7eb;border-radius:6px;height:10px;overflow:hidden">
        <div style="background:{color};width:{pct:.1f}%;height:100%;
                    border-radius:6px"></div>
      </div>
    </div>"""


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:4px 0 20px">
      <div style="font-family:'IBM Plex Serif',serif;font-size:1.15rem;
                  font-weight:600;color:#1a1d23">PD Detection System</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
                  color:#9ca3af;margin-top:3px">Research Prototype · v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### Detection Strategy")
    strategy_key = st.selectbox(
        "Strategy", options=list(STRATEGIES.keys()),
        format_func=lambda k: STRATEGIES[k]["label"],
        label_visibility="collapsed",
    )
    info = STRATEGIES[strategy_key]

    st.markdown(f"""
    <div style="font-size:0.78rem;color:#6b7280;margin:6px 0 4px;line-height:1.5">
    {"Uses <b>all three</b> modalities (speech + handwriting + gait)" if len(info["needs"]) == 3
     else f"Uses <b>{info['needs'][0]}</b> only"}
    </div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;
                color:#1a56db">Validation AUC: {info['auc']:.4f}</div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### Decision Threshold")
    threshold = st.slider("Threshold", 0.1, 0.9, 0.5, 0.05, label_visibility="collapsed")
    st.markdown(f"""
    <div style="font-size:0.76rem;color:#6b7280;line-height:1.5">
    Classify as PD if probability ≥ <b>{threshold:.2f}</b><br>
    Lower = more sensitive (catches more PD, more false alarms).<br>
    Higher = more specific (fewer false alarms, may miss PD).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### About")
    st.markdown("""
    <div style="font-size:0.76rem;color:#6b7280;line-height:1.6">
    Multimodal machine learning system for early Parkinson's Disease screening.
    Trained on voice recordings, spiral drawings, and gait sensor data.
    <br><br>
    <b>⚠ Not for clinical use.</b>
    </div>
    """, unsafe_allow_html=True)


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:24px">
  <h1 style="font-family:'IBM Plex Serif',serif;font-size:1.9rem;font-weight:600;
             color:#1a1d23;margin:0 0 4px">Parkinson's Disease Detection</h1>
  <p style="color:#6b7280;font-size:0.9rem;margin:0">
    Upload patient data below, then click <b>Run Analysis</b> to get a prediction.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Two-column layout ─────────────────────────────────────────────────────────
left_col, right_col = st.columns([3, 2], gap="large")

needs      = info["needs"]
speech_features = None
hw_path    = None
gait_path  = None

with left_col:

    # ── STEP 1: Strategy info ─────────────────────────────────────────────────
    modality_labels = {"speech": "Voice Recording", "handwriting": "Handwriting Image", "gait": "Gait Recording"}
    needed_str = " + ".join(modality_labels[m] for m in needs)
    st.markdown(f"""
    <div class="step-card" style="background:#f0f7ff;border-color:#bfdbfe">
      <div style="font-size:0.82rem;color:#1d4ed8">
        <b>Selected strategy:</b> {info['label']} &nbsp;·&nbsp;
        <b>Requires:</b> {needed_str} &nbsp;·&nbsp;
        <b>Validation AUC:</b> {info['auc']:.4f}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── STEP: Speech ──────────────────────────────────────────────────────────
    if "speech" in needs:
        st.markdown("""
        <div class="step-card">
          <div class="step-header">
            <div class="step-num">1</div>
            <div>
              <div class="step-title">Voice / Speech Features</div>
              <div class="step-desc">22 acoustic biomarkers extracted from a sustained vowel recording</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="font-size:0.8rem;color:#374151;margin:-8px 0 10px;font-weight:500">
          Quick demo — load pre-computed features from a real patient:
        </div>
        """, unsafe_allow_html=True)

        col_pd, col_hc, col_space = st.columns([1, 1, 2])
        with col_pd:
            if st.button("🔴  Parkinson's sample", use_container_width=True):
                st.session_state["speech_vals"] = DEMO_PD[:]
                st.session_state["speech_source"] = "pd_demo"
        with col_hc:
            if st.button("🟢  Healthy sample", use_container_width=True):
                st.session_state["speech_vals"] = DEMO_HC[:]
                st.session_state["speech_source"] = "hc_demo"

        source = st.session_state.get("speech_source", "none")
        if source == "pd_demo":
            st.markdown('<div class="info-callout">✓ Loaded: Parkinson\'s patient voice features (UCI dataset)</div>', unsafe_allow_html=True)
        elif source == "hc_demo":
            st.markdown('<div class="info-callout">✓ Loaded: Healthy control voice features (UCI dataset)</div>', unsafe_allow_html=True)

        with st.expander("Or upload a CSV / view/edit feature values"):
            uploaded_csv = st.file_uploader(
                "Upload CSV (one row, columns = feature names or in order)",
                type=["csv"], key="speech_csv",
            )
            if uploaded_csv:
                import pandas as pd
                try:
                    df = pd.read_csv(uploaded_csv)
                    vals_from_csv = df.iloc[0, :22].tolist()
                    st.session_state["speech_vals"] = vals_from_csv
                    st.session_state["speech_source"] = "csv"
                    st.success(f"Loaded from CSV ({df.shape[1]} columns found)")
                except Exception as e:
                    st.error(f"Could not parse CSV: {e}")

            vals = st.session_state.get("speech_vals", DEMO_PD[:])
            c1, c2 = st.columns(2)
            updated = []
            for i, feat in enumerate(SPEECH_FEATURES):
                col = c1 if i % 2 == 0 else c2
                updated.append(col.number_input(feat, value=float(vals[i]),
                                                format="%.6f", key=f"sf_{i}"))
            speech_features = np.array(updated, dtype=np.float32)

        if speech_features is None:
            vals = st.session_state.get("speech_vals", DEMO_PD[:])
            speech_features = np.array(vals, dtype=np.float32)

    # ── STEP: Handwriting ─────────────────────────────────────────────────────
    step_n = 2 if "speech" in needs else 1
    if "handwriting" in needs:
        st.markdown(f"""
        <div class="step-card">
          <div class="step-header">
            <div class="step-num">{step_n}</div>
            <div>
              <div class="step-title">Handwriting / Spiral Drawing</div>
              <div class="step-desc">Upload a JPG/PNG photo of the patient's hand-drawn spiral</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        step_n += 1

        hw_file = st.file_uploader("Upload spiral drawing image (.jpg / .png)",
                                   type=["jpg", "jpeg", "png"], key="hw_upload")
        if hw_file:
            c_img, c_info = st.columns([1, 2])
            with c_img:
                st.image(hw_file, caption="Uploaded spiral", use_container_width=True)
            with c_info:
                st.markdown(f"""
                <div style="font-size:0.8rem;color:#374151;line-height:1.7;padding-top:4px">
                  <b>File:</b> {hw_file.name}<br>
                  <b>Size:</b> {hw_file.size/1024:.1f} KB<br><br>
                  <span style="color:#059669">✓ Ready for analysis</span>
                </div>
                """, unsafe_allow_html=True)
            hw_path = save_upload(hw_file)
        else:
            st.markdown("""
            <div class="warn-callout">
              📁 Sample files: <code>data/raw/handwriting/SpiralPatients/</code> (PD)
              &nbsp;or&nbsp; <code>data/raw/handwriting/SpiralControl/</code> (HC)
            </div>
            """, unsafe_allow_html=True)

    # ── STEP: Gait ────────────────────────────────────────────────────────────
    if "gait" in needs:
        st.markdown(f"""
        <div class="step-card">
          <div class="step-header">
            <div class="step-num">{step_n}</div>
            <div>
              <div class="step-title">Gait Recording</div>
              <div class="step-desc">PhysioNet GaitPDB ground-reaction-force file (.txt or .ts)</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        gait_file = st.file_uploader("Upload gait file (.txt / .ts)",
                                     type=["txt", "ts"], key="gait_upload")
        if gait_file:
            st.markdown(f"""
            <div style="font-size:0.8rem;color:#059669;margin-top:4px">
              ✓ {gait_file.name} ({gait_file.size/1024:.1f} KB) — ready
            </div>
            """, unsafe_allow_html=True)
            gait_path = save_upload(gait_file)
        else:
            st.markdown("""
            <div class="warn-callout">
              📁 Sample files: <code>data/raw/gait/GaPt03_01.txt</code> (PD)
              &nbsp;or&nbsp; <code>data/raw/gait/GaCo01_01.txt</code> (HC)
            </div>
            """, unsafe_allow_html=True)

    # ── Run button ────────────────────────────────────────────────────────────
    st.markdown("<div style='height:6px'/>", unsafe_allow_html=True)

    missing = []
    if "handwriting" in needs and hw_path is None:   missing.append("handwriting image")
    if "gait"        in needs and gait_path is None:  missing.append("gait recording")

    if missing:
        st.markdown(f"""
        <div class="warn-callout" style="margin-bottom:10px">
          ⏳ Still needed: <b>{", ".join(missing)}</b>
        </div>
        """, unsafe_allow_html=True)

    run_label = "Run Analysis" if not missing else f"Upload {', '.join(missing)} to continue"
    run_clicked = st.button(run_label, type="primary", disabled=bool(missing), use_container_width=True)


# ── Results panel ─────────────────────────────────────────────────────────────
with right_col:
    st.markdown("<div style='height:4px'/>", unsafe_allow_html=True)

    if run_clicked and not missing:
        with st.spinner("Loading models and running inference…"):
            try:
                pipeline = load_pipeline()
                result = pipeline.predict(
                    speech_features=speech_features if "speech" in needs else None,
                    handwriting_image_path=hw_path  if "handwriting" in needs else None,
                    gait_file_path=gait_path        if "gait" in needs else None,
                    fusion_strategy=strategy_key,
                    threshold=threshold,
                )
                st.session_state["last_result"] = result
            except Exception as e:
                st.error(f"Inference failed: {e}")

    if "last_result" not in st.session_state:
        st.markdown("""
        <div class="pending-card">
          <div class="pending-icon">🧠</div>
          <div class="pending-text">
            <b>Results will appear here</b><br>
            Fill in the required inputs on the left,<br>then click <b>Run Analysis</b>.
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        r      = st.session_state["last_result"]
        prob   = r["probability"]
        is_pd  = r["prediction"] == 1
        cls    = "pd" if is_pd else "hc"
        icon   = "⚠️" if is_pd else "✅"
        label  = "Parkinson's Disease Detected" if is_pd else "Healthy — No PD Detected"
        conf   = "High" if abs(prob - 0.5) > 0.3 else ("Moderate" if abs(prob - 0.5) > 0.15 else "Low")

        st.markdown(f"""
        <div class="result-card {cls}">
          <div class="result-icon">{icon}</div>
          <div class="result-label {cls}">{label}</div>
          <div class="result-prob {cls}">{prob*100:.1f}%</div>
          <div class="result-sub">PD probability</div>
          {prob_bar(prob, is_pd)}
          <div class="stat-grid">
            <div class="stat-box">
              <div class="stat-box-label">Strategy</div>
              <div class="stat-box-val">{r['model_used'].title()}</div>
            </div>
            <div class="stat-box">
              <div class="stat-box-label">Confidence</div>
              <div class="stat-box-val">{conf}</div>
            </div>
            <div class="stat-box">
              <div class="stat-box-label">Threshold</div>
              <div class="stat-box-val">{r['threshold']:.2f}</div>
            </div>
            <div class="stat-box">
              <div class="stat-box-label">Latency</div>
              <div class="stat-box-val">{r['latency_ms']:.0f} ms</div>
            </div>
            <div class="stat-box">
              <div class="stat-box-label">Model AUC</div>
              <div class="stat-box-val">{STRATEGIES[r['model_used']]['auc']:.4f}</div>
            </div>
            <div class="stat-box">
              <div class="stat-box-label">Time</div>
              <div class="stat-box-val" style="font-size:0.75rem">{r['timestamp'][11:19]}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:10px'/>", unsafe_allow_html=True)
        if st.button("↩ Clear and run again", use_container_width=True):
            del st.session_state["last_result"]
            st.rerun()

    # ── How it works ──────────────────────────────────────────────────────────
    st.markdown("<div style='height:16px'/>", unsafe_allow_html=True)
    with st.expander("ℹ️  How to use this app"):
        st.markdown("""
        **Quickest demo (30 seconds):**
        1. Sidebar → select **Speech Only**
        2. Click **🔴 Parkinson's sample**
        3. Click **Run Analysis**

        **Full multimodal demo:**
        1. Sidebar → select **Hybrid Fusion ★** (best model)
        2. Click **🔴 Parkinson's sample** for speech
        3. Upload `data/raw/handwriting/SpiralPatients/0002-1.jpg`
        4. Upload `data/raw/gait/GaPt03_01.txt`
        5. Click **Run Analysis**

        **Healthy control demo:**
        - Use **🟢 Healthy sample** for speech
        - Upload from `SpiralControl/` and `GaCo01_01.txt`

        **The threshold slider** controls sensitivity.
        Lower = catches more PD cases. Higher = fewer false alarms.
        """)

    with st.expander("🎤  Can I use a live voice recording?"):
        st.markdown("""
        **No** — and here's why:

        The speech features this model uses (jitter, shimmer, RPDE, etc.) are
        **pre-extracted acoustic biomarkers**, not raw audio. They are computed
        by specialist software (like Praat) from a **sustained "ahhh"** phonation,
        typically 3–6 seconds of a held vowel.

        A live microphone gives raw audio waveform. To use it you would need to:
        1. Record a sustained vowel
        2. Run Praat or `praat-parselmouth` to extract the 22 MDVP features
        3. Feed *those values* to this model

        That audio → features pipeline is not built into this demo, but it is
        technically feasible as a future extension.
        """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:center;
            padding:2px 0 12px;font-family:'IBM Plex Mono',monospace;
            font-size:0.68rem;color:#9ca3af">
  <span>⚠ Research prototype · Not for clinical use</span>
  <span>Hybrid Fusion · AUC 0.9814 · p95 &lt; 315 ms</span>
</div>
""", unsafe_allow_html=True)
