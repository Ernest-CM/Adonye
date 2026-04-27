"""
Evaluate all six trained models on their respective held-out test sets.
Produces metrics_table.csv, confusion matrix PNGs, and ROC curve PNGs.

Usage:
    python -m src.evaluation.evaluate_all
"""

import os
import json
import numpy as np
import tensorflow as tf

from src.config import (
    RANDOM_SEED,
    SPEECH_RAW_DIR, SPEECH_PROCESSED_DIR, SPEECH_SPLITS_PATH,
    HANDWRITING_RAW_DIR, HANDWRITING_PROCESSED_DIR, HANDWRITING_SPLITS_PATH,
    GAIT_RAW_DIR, GAIT_PROCESSED_DIR, GAIT_SPLITS_PATH,
    SAVED_MODELS_DIR, RESULTS_DIR,
)
from src.data.speech_loader import load_speech_data
from src.data.handwriting_loader import load_handwriting_data
from src.data.gait_loader import load_gait_data
from src.models.speech_model import get_speech_embedding_model
from src.models.handwriting_model import get_handwriting_embedding_model
from src.models.gait_model import get_gait_embedding_model
from src.models.late_fusion import LateFusionPredictor
from src.models.hybrid_fusion import load_hybrid_fusion_model, predict_hybrid
from src.evaluation.metrics import (
    compute_all_metrics, plot_confusion_matrix, plot_roc_curve, build_metrics_table,
)


def evaluate_all() -> None:
    np.random.seed(RANDOM_SEED)
    os.makedirs(os.path.join(RESULTS_DIR, "confusion_matrices"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "roc_curves"), exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    _, _, _, _, X_s_te, y_s_te = load_speech_data(
        SPEECH_RAW_DIR, SPEECH_PROCESSED_DIR, SPEECH_SPLITS_PATH
    )
    _, _, _, _, X_h_te, y_h_te = load_handwriting_data(
        HANDWRITING_RAW_DIR, HANDWRITING_PROCESSED_DIR, HANDWRITING_SPLITS_PATH
    )
    _, _, _, _, X_g_te, y_g_te = load_gait_data(
        GAIT_RAW_DIR, GAIT_PROCESSED_DIR, GAIT_SPLITS_PATH
    )

    n_test = min(len(y_s_te), len(y_h_te), len(y_g_te))

    # ── Load trained models ───────────────────────────────────────────────────
    def _load_keras(subdir):
        path = os.path.join(SAVED_MODELS_DIR, subdir, "best_model.keras")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Trained model not found: {path}")
        return tf.keras.models.load_model(path)

    speech_model = _load_keras("speech_baseline")
    hw_model     = _load_keras("handwriting_baseline")
    gait_model   = _load_keras("gait_baseline")
    ef_model     = _load_keras("early_fusion")

    # Late fusion weights
    lf_weights_path = os.path.join(SAVED_MODELS_DIR, "late_fusion", "weights.json")
    if os.path.exists(lf_weights_path):
        with open(lf_weights_path) as f:
            lf_weights = json.load(f)
    else:
        print("Warning: Late fusion weights not found. Using equal weights.")
        lf_weights = {"speech": 1.0, "handwriting": 1.0, "gait": 1.0}
    late_predictor = LateFusionPredictor(weights=lf_weights)

    # Hybrid fusion
    hf_checkpoint = os.path.join(SAVED_MODELS_DIR, "hybrid_fusion", "best_model.pt")
    hybrid_model = load_hybrid_fusion_model(hf_checkpoint)

    speech_emb_model = get_speech_embedding_model(speech_model)
    hw_emb_model     = get_handwriting_embedding_model(hw_model)
    gait_emb_model   = get_gait_embedding_model(gait_model)

    # ── Evaluate each model ───────────────────────────────────────────────────
    all_results = {}

    def _eval_and_record(name, y_true, y_prob):
        metrics = compute_all_metrics(y_true, y_prob)
        all_results[name] = metrics
        y_pred = (y_prob.flatten() >= 0.5).astype(int)
        plot_confusion_matrix(
            y_true, y_pred, name,
            os.path.join(RESULTS_DIR, "confusion_matrices", f"{name.replace(' ', '_')}.png"),
        )
        plot_roc_curve(
            y_true, y_prob, name,
            os.path.join(RESULTS_DIR, "roc_curves", f"{name.replace(' ', '_')}.png"),
        )
        print(f"\n[{name}] {metrics}")

    # 1. Speech Baseline
    p_s = speech_model.predict(X_s_te, verbose=0)
    _eval_and_record("Speech Baseline", y_s_te, p_s)

    # 2. Handwriting Baseline
    p_h = hw_model.predict(X_h_te, verbose=0)
    _eval_and_record("Handwriting Baseline", y_h_te, p_h)

    # 3. Gait Baseline
    p_g = gait_model.predict(X_g_te, verbose=0)
    _eval_and_record("Gait Baseline", y_g_te, p_g)

    # 4. Early Fusion (same-size combined test set)
    ef_test = [X_s_te[:n_test], X_h_te[:n_test], X_g_te[:n_test]]
    p_ef = ef_model.predict(ef_test, verbose=0)
    _eval_and_record("Early Fusion", y_s_te[:n_test], p_ef)

    # 5. Late Fusion
    _, p_lf = late_predictor.predict(
        p_s[:n_test], p_h[:n_test], p_g[:n_test]
    )
    _eval_and_record("Late Fusion", y_s_te[:n_test], p_lf)

    # 6. Hybrid Fusion
    s_embs = speech_emb_model.predict(X_s_te[:n_test], verbose=0)
    h_embs = hw_emb_model.predict(X_h_te[:n_test], verbose=0)
    g_embs = gait_emb_model.predict(X_g_te[:n_test], verbose=0)
    _, p_hf = predict_hybrid(hybrid_model, s_embs, h_embs, g_embs)
    _eval_and_record("Hybrid Fusion", y_s_te[:n_test], p_hf)

    # ── Save combined table ───────────────────────────────────────────────────
    build_metrics_table(all_results, os.path.join(RESULTS_DIR, "metrics_table.csv"))
    print("\n=== Evaluation complete ===")


if __name__ == "__main__":
    evaluate_all()
