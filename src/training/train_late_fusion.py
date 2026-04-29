"""
Optimise Late Fusion weights via grid search (no gradient training).

Usage:
    python -m src.training.train_late_fusion
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
    SAVED_MODELS_DIR,
)
from src.data.speech_loader import load_speech_data
from src.data.handwriting_loader import load_handwriting_data
from src.data.gait_loader import load_gait_data
from src import safe_load_model
from src.models.late_fusion import optimise_late_fusion_weights, LateFusionPredictor


def train() -> LateFusionPredictor:
    np.random.seed(RANDOM_SEED)

    print("\n=== Optimising Late Fusion Weights ===")

    speech_path = os.path.join(SAVED_MODELS_DIR, "speech_baseline", "best_model.keras")
    hw_path     = os.path.join(SAVED_MODELS_DIR, "handwriting_baseline", "best_model.keras")
    gait_path   = os.path.join(SAVED_MODELS_DIR, "gait_baseline", "best_model.keras")

    for p in [speech_path, hw_path, gait_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing trained model: {p}")

    speech_model = safe_load_model(speech_path)
    hw_model     = safe_load_model(hw_path)
    gait_model   = safe_load_model(gait_path)

    _, _, X_s_v, y_s_v, _, _ = load_speech_data(
        SPEECH_RAW_DIR, SPEECH_PROCESSED_DIR, SPEECH_SPLITS_PATH
    )
    _, _, X_h_v, y_h_v, _, _ = load_handwriting_data(
        HANDWRITING_RAW_DIR, HANDWRITING_PROCESSED_DIR, HANDWRITING_SPLITS_PATH
    )
    _, _, X_g_v, y_g_v, _, _ = load_gait_data(
        GAIT_RAW_DIR, GAIT_PROCESSED_DIR, GAIT_SPLITS_PATH
    )

    n_val = min(len(y_s_v), len(y_h_v), len(y_g_v))
    val_y = y_s_v[:n_val]

    best_weights = optimise_late_fusion_weights(
        speech_model, hw_model, gait_model,
        X_s_v[:n_val], X_h_v[:n_val], X_g_v[:n_val],
        val_y,
    )

    save_dir = os.path.join(SAVED_MODELS_DIR, "late_fusion")
    os.makedirs(save_dir, exist_ok=True)
    weights_path = os.path.join(save_dir, "weights.json")
    with open(weights_path, "w") as f:
        json.dump(best_weights, f, indent=2)
    print(f"Weights saved to {weights_path}")

    return LateFusionPredictor(weights=best_weights)


if __name__ == "__main__":
    train()
