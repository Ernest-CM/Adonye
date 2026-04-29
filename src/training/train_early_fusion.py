"""
Train the Early Fusion model (frozen unimodal embeddings + shared head).

Usage:
    python -m src.training.train_early_fusion
    python -m src.training.train_early_fusion --smoke-test
"""

import os
import argparse
import numpy as np
import tensorflow as tf

from src.config import (
    RANDOM_SEED, EPOCHS, BATCH_SIZE, EARLY_STOPPING_PATIENCE,
    SPEECH_RAW_DIR, SPEECH_PROCESSED_DIR, SPEECH_SPLITS_PATH,
    HANDWRITING_RAW_DIR, HANDWRITING_PROCESSED_DIR, HANDWRITING_SPLITS_PATH,
    GAIT_RAW_DIR, GAIT_PROCESSED_DIR, GAIT_SPLITS_PATH,
    SAVED_MODELS_DIR, LOGS_DIR,
)
from src.data.speech_loader import load_speech_data
from src.data.handwriting_loader import load_handwriting_data
from src.data.gait_loader import load_gait_data
from src import safe_load_model
from src.models.early_fusion import build_early_fusion_model


def _load_unimodal_models():
    speech_path = os.path.join(SAVED_MODELS_DIR, "speech_baseline", "best_model.keras")
    hw_path     = os.path.join(SAVED_MODELS_DIR, "handwriting_baseline", "best_model.keras")
    gait_path   = os.path.join(SAVED_MODELS_DIR, "gait_baseline", "best_model.keras")

    for p in [speech_path, hw_path, gait_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing trained model: {p}\n"
                "Train all three unimodal baselines first."
            )

    return (
        safe_load_model(speech_path),
        safe_load_model(hw_path),
        safe_load_model(gait_path),
    )


def train(smoke_test: bool = False) -> tf.keras.Model:
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    print("\n=== Training Early Fusion ===")
    speech_model, hw_model, gait_model = _load_unimodal_models()

    X_s_tr, y_s_tr, X_s_v, y_s_v, X_s_te, y_s_te = load_speech_data(
        SPEECH_RAW_DIR, SPEECH_PROCESSED_DIR, SPEECH_SPLITS_PATH
    )
    X_h_tr, y_h_tr, X_h_v, y_h_v, X_h_te, y_h_te = load_handwriting_data(
        HANDWRITING_RAW_DIR, HANDWRITING_PROCESSED_DIR, HANDWRITING_SPLITS_PATH
    )
    X_g_tr, y_g_tr, X_g_v, y_g_v, X_g_te, y_g_te = load_gait_data(
        GAIT_RAW_DIR, GAIT_PROCESSED_DIR, GAIT_SPLITS_PATH
    )

    # Build combined datasets by resampling to the smallest split size
    n_train = min(len(y_s_tr), len(y_h_tr), len(y_g_tr))
    n_val   = min(len(y_s_v),  len(y_h_v),  len(y_g_v))
    n_test  = min(len(y_s_te), len(y_h_te), len(y_g_te))

    rng = np.random.default_rng(RANDOM_SEED)
    tr_s_idx = rng.choice(len(y_s_tr), n_train, replace=False)
    tr_h_idx = rng.choice(len(y_h_tr), n_train, replace=False)
    tr_g_idx = rng.choice(len(y_g_tr), n_train, replace=False)
    # Use speech labels as the combined label (all modalities should agree on PD/HC)
    y_train = y_s_tr[tr_s_idx]
    y_val   = y_s_v[:n_val]
    y_test  = y_s_te[:n_test]

    train_inputs = [X_s_tr[tr_s_idx], X_h_tr[tr_h_idx], X_g_tr[tr_g_idx]]
    val_inputs   = [X_s_v[:n_val],    X_h_v[:n_val],     X_g_v[:n_val]]
    test_inputs  = [X_s_te[:n_test],  X_h_te[:n_test],   X_g_te[:n_test]]

    model = build_early_fusion_model(speech_model, hw_model, gait_model)
    model.summary()

    save_dir = os.path.join(SAVED_MODELS_DIR, "early_fusion")
    os.makedirs(save_dir, exist_ok=True)

    epochs = 3 if smoke_test else EPOCHS

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_roc_auc", patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True, mode="max",
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_dir, "best_model.keras"),
            monitor="val_roc_auc", save_best_only=True, mode="max", verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_roc_auc", factor=0.5, patience=7, mode="max", verbose=1,
        ),
    ]

    model.fit(
        train_inputs, y_train,
        validation_data=(val_inputs, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    test_results = model.evaluate(test_inputs, y_test, verbose=0)
    print("\n[Early Fusion] Test results:")
    for name, val in zip(model.metrics_names, test_results):
        print(f"  {name}: {val:.4f}")

    model.save(os.path.join(save_dir, "final_model.keras"))
    print(f"\nModel saved to {save_dir}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    train(smoke_test=args.smoke_test)
