"""
Parkinson's Disease Detection — Streamlit Demo
Run: python -m streamlit run app.py
"""

import os, sys, tempfile, warnings
import numpy as np
import streamlit as st

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="PD Detection", page_icon="🧠", layout="centered")

# ── Constants ─────────────────────────────────────────────────────────────────
SPEECH_FEATURES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
]

DEMO_PD = [119.992, 157.302, 74.997, 0.00784, 0.00007, 0.0037, 0.00554, 0.01109,
           0.04374, 0.426, 0.02182, 0.0313, 0.02971, 0.06545, 0.02211, 21.033,
           0.414783, 0.815285, -4.813031, 0.266482, 2.301442, 0.284654]

DEMO_HC = [197.076, 206.896, 192.055, 0.00289, 0.00001, 0.00166, 0.00168, 0.00498,
           0.01098, 0.097, 0.00563, 0.0068, 0.00802, 0.01689, 0.00339, 26.775,
           0.422229, 0.741367, -7.3483, 0.177551, 1.743867, 0.085569]

STRATEGIES = {
    "hybrid":      {"label": "Hybrid Fusion (Best — AUC 0.9814)", "needs": ["speech", "handwriting", "gait"], "auc": 0.9814},
    "late":        {"label": "Late Fusion (AUC 0.9565)",           "needs": ["speech", "handwriting", "gait"], "auc": 0.9565},
    "early":       {"label": "Early Fusion (AUC 0.8199)",          "needs": ["speech", "handwriting", "gait"], "auc": 0.8199},
    "speech":      {"label": "Speech Only (AUC 0.9814)",           "needs": ["speech"],                        "auc": 0.9814},
    "handwriting": {"label": "Handwriting Only (AUC 0.9470)",      "needs": ["handwriting"],                   "auc": 0.9470},
    "gait":        {"label": "Gait Only (AUC 0.7000)",             "needs": ["gait"],                          "auc": 0.7000},
}


@st.cache_resource(show_spinner="Loading models…")
def load_pipeline():
    from src.inference.pipeline import PDInferencePipeline
    return PDInferencePipeline()


def save_upload(f) -> str:
    suffix = os.path.splitext(f.name)[-1] or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(f.read())
        return tmp.name


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧠 Parkinson's Disease Detection")
st.caption("Multimodal ML system · Undergraduate Research Project · Not for clinical use")
st.divider()

# ── Strategy ──────────────────────────────────────────────────────────────────
st.subheader("1. Select detection strategy")
strategy_key = st.selectbox(
    "Strategy",
    options=list(STRATEGIES.keys()),
    format_func=lambda k: STRATEGIES[k]["label"],
    label_visibility="collapsed",
)
needs = STRATEGIES[strategy_key]["needs"]
st.caption(f"Requires: **{' + '.join(needs)}**")
st.divider()

# ── Inputs ────────────────────────────────────────────────────────────────────
st.subheader("2. Upload patient data")

speech_features = None
hw_path = None
gait_path = None

# Speech
if "speech" in needs:
    st.markdown("**Voice / Speech Features**")
    st.caption("22 acoustic biomarkers from a sustained vowel recording (e.g. 'ahhh')")

    c1, c2 = st.columns(2)
    if c1.button("Load Parkinson's sample data", use_container_width=True):
        st.session_state["speech_vals"] = DEMO_PD[:]
        st.session_state["speech_label"] = "pd"
    if c2.button("Load Healthy control sample data", use_container_width=True):
        st.session_state["speech_vals"] = DEMO_HC[:]
        st.session_state["speech_label"] = "hc"

    label = st.session_state.get("speech_label")
    if label == "pd":
        st.success("Loaded: Parkinson's patient voice features (UCI Parkinson's dataset)")
    elif label == "hc":
        st.success("Loaded: Healthy control voice features (UCI Parkinson's dataset)")

    with st.expander("Or upload a CSV / manually edit feature values"):
        csv_file = st.file_uploader("Upload CSV (one row, 22 columns in feature order)",
                                    type=["csv"], key="csv_up")
        if csv_file:
            import pandas as pd
            try:
                df = pd.read_csv(csv_file)
                st.session_state["speech_vals"] = df.iloc[0, :22].tolist()
                st.session_state["speech_label"] = "csv"
                st.success("CSV loaded.")
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")

        vals = st.session_state.get("speech_vals", DEMO_PD[:])
        c_a, c_b = st.columns(2)
        updated = []
        for i, feat in enumerate(SPEECH_FEATURES):
            col = c_a if i % 2 == 0 else c_b
            updated.append(col.number_input(feat, value=float(vals[i]),
                                            format="%.6f", key=f"sf_{i}"))
        speech_features = np.array(updated, dtype=np.float32)

    if speech_features is None:
        speech_features = np.array(
            st.session_state.get("speech_vals", DEMO_PD[:]), dtype=np.float32
        )

# Handwriting
if "handwriting" in needs:
    st.markdown("**Handwriting / Spiral Drawing**")
    st.caption("Upload a JPG/PNG photo of the patient's hand-drawn spiral. "
               "Sample files: `data/raw/handwriting/SpiralPatients/` (PD) or `SpiralControl/` (HC)")
    hw_file = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"], key="hw_up")
    if hw_file:
        st.image(hw_file, width=200, caption=hw_file.name)
        hw_path = save_upload(hw_file)

# Gait
if "gait" in needs:
    st.markdown("**Gait Recording**")
    st.caption("Upload a PhysioNet GaitPDB ground-reaction-force file (.txt or .ts). "
               "Sample files: `data/raw/gait/GaPt03_01.txt` (PD) or `GaCo01_01.txt` (HC)")
    gait_file = st.file_uploader("Choose gait file", type=["txt", "ts"], key="gait_up")
    if gait_file:
        st.success(f"Uploaded: {gait_file.name} ({gait_file.size / 1024:.1f} KB)")
        gait_path = save_upload(gait_file)

st.divider()

# ── Threshold ─────────────────────────────────────────────────────────────────
st.subheader("3. Set decision threshold")
threshold = st.slider(
    "Classify as Parkinson's if probability ≥ threshold",
    min_value=0.1, max_value=0.9, value=0.5, step=0.05,
)
st.caption("Lower = more sensitive (catches more PD). Higher = more specific (fewer false alarms).")
st.divider()

# ── Run ───────────────────────────────────────────────────────────────────────
st.subheader("4. Run analysis")

missing = []
if "handwriting" in needs and hw_path is None:  missing.append("handwriting image")
if "gait"        in needs and gait_path is None: missing.append("gait recording")

if missing:
    st.warning(f"Still needed before running: **{', '.join(missing)}**")

run = st.button("▶  Run Analysis", type="primary",
                disabled=bool(missing), use_container_width=True)

# ── Results ───────────────────────────────────────────────────────────────────
if run and not missing:
    with st.spinner("Running inference…"):
        try:
            pipeline = load_pipeline()
            result = pipeline.predict(
                speech_features=speech_features if "speech" in needs else None,
                handwriting_image_path=hw_path  if "handwriting" in needs else None,
                gait_file_path=gait_path        if "gait" in needs else None,
                fusion_strategy=strategy_key,
                threshold=threshold,
            )
            st.session_state["result"] = result
        except Exception as e:
            st.error(f"Inference failed: {e}")

if "result" in st.session_state:
    r = st.session_state["result"]
    prob = r["probability"]
    is_pd = r["prediction"] == 1

    st.divider()
    st.subheader("Result")

    if is_pd:
        st.error(f"### ⚠️ Parkinson's Disease Detected\nPD probability: **{prob*100:.1f}%**")
    else:
        st.success(f"### ✅ Healthy — No PD Detected\nPD probability: **{prob*100:.1f}%**")

    st.progress(prob)

    col1, col2, col3, col4 = st.columns(4)
    latency = r["latency_ms"]
    latency_note = " (incl. load)" if latency > 2000 else ""
    col1.metric("Probability", f"{prob:.4f}")
    col2.metric("Threshold",   f"{threshold:.2f}")
    col3.metric("Latency",     f"{latency:.0f} ms{latency_note}")
    col4.metric("Model AUC",   f"{STRATEGIES[strategy_key]['auc']:.4f}")

    if st.button("Clear result"):
        del st.session_state["result"]
        st.rerun()

# ── FAQ ───────────────────────────────────────────────────────────────────────
st.divider()
with st.expander("Can I use a live voice recording?"):
    st.markdown("""
    **No** — the 22 speech features this model uses (jitter, shimmer, RPDE, etc.)
    are **pre-extracted acoustic biomarkers**, not raw audio.
    They are computed by specialist software (e.g. Praat) from a sustained "ahhh"
    vowel phonation.

    A live microphone gives raw audio. To use it you would first need to run it
    through a vocal analysis tool to extract the 22 MDVP features, then pass those
    values to this model. That pipeline is not included in this demo.
    """)
