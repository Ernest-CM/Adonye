"""
Train the Hybrid Fusion model (PyTorch cross-attention on Keras embeddings).

Usage:
    python -m src.training.train_hybrid_fusion
    python -m src.training.train_hybrid_fusion --smoke-test
"""

import os
import argparse
import numpy as np
import tensorflow as tf

from src.config import (
    RANDOM_SEED, EPOCHS,
    SPEECH_RAW_DIR, SPEECH_PROCESSED_DIR, SPEECH_SPLITS_PATH,
    HANDWRITING_RAW_DIR, HANDWRITING_PROCESSED_DIR, HANDWRITING_SPLITS_PATH,
    GAIT_RAW_DIR, GAIT_PROCESSED_DIR, GAIT_SPLITS_PATH,
    SAVED_MODELS_DIR,
)
from src.data.speech_loader import load_speech_data
from src.data.handwriting_loader import load_handwriting_data
from src.data.gait_loader import load_gait_data
from src.models.speech_model import get_speech_embedding_model
from src.models.handwriting_model import get_handwriting_embedding_model
from src.models.gait_model import get_gait_embedding_model
from src.models.hybrid_fusion import train_hybrid_fusion


def train(smoke_test: bool = False):
    np.random.seed(RANDOM_SEED)

    print("\n=== Training Hybrid Fusion ===")

    speech_path = os.path.join(SAVED_MODELS_DIR, "speech_baseline", "best_model.keras")
    hw_path     = os.path.join(SAVED_MODELS_DIR, "handwriting_baseline", "best_model.keras")
    gait_path   = os.path.join(SAVED_MODELS_DIR, "gait_baseline", "best_model.keras")

    for p in [speech_path, hw_path, gait_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing trained model: {p}\n"
                "Train all three unimodal baselines first."
            )

    speech_model = tf.keras.models.load_model(speech_path)
    hw_model     = tf.keras.models.load_model(hw_path)
    gait_model   = tf.keras.models.load_model(gait_path)

    speech_emb_model = get_speech_embedding_model(speech_model)
    hw_emb_model     = get_handwriting_embedding_model(hw_model)
    gait_emb_model   = get_gait_embedding_model(gait_model)

    X_s_tr, y_s_tr, X_s_v, y_s_v, _, _ = load_speech_data(
        SPEECH_RAW_DIR, SPEECH_PROCESSED_DIR, SPEECH_SPLITS_PATH
    )
    X_h_tr, y_h_tr, X_h_v, y_h_v, _, _ = load_handwriting_data(
        HANDWRITING_RAW_DIR, HANDWRITING_PROCESSED_DIR, HANDWRITING_SPLITS_PATH
    )
    X_g_tr, y_g_tr, X_g_v, y_g_v, _, _ = load_gait_data(
        GAIT_RAW_DIR, GAIT_PROCESSED_DIR, GAIT_SPLITS_PATH
    )

    n_train = min(len(y_s_tr), len(y_h_tr), len(y_g_tr))
    n_val   = min(len(y_s_v),  len(y_h_v),  len(y_g_v))

    rng = np.random.default_rng(RANDOM_SEED)
    tr_s_idx = rng.choice(len(y_s_tr), n_train, replace=False)
    tr_h_idx = rng.choice(len(y_h_tr), n_train, replace=False)
    tr_g_idx = rng.choice(len(y_g_tr), n_train, replace=False)

    train_data = (
        X_s_tr[tr_s_idx], X_h_tr[tr_h_idx], X_g_tr[tr_g_idx],
        y_s_tr[tr_s_idx],
    )
    val_data = (
        X_s_v[:n_val], X_h_v[:n_val], X_g_v[:n_val],
        y_s_v[:n_val],
    )

    save_dir = os.path.join(SAVED_MODELS_DIR, "hybrid_fusion")

    model = train_hybrid_fusion(
        speech_emb_model, hw_emb_model, gait_emb_model,
        train_data, val_data,
        save_dir=save_dir,
        epochs=EPOCHS,
        smoke_test=smoke_test,
    )
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    train(smoke_test=args.smoke_test)
