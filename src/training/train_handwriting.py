"""
Train the handwriting unimodal baseline model (DenseNet121, 2-phase training).

Usage:
    python -m src.training.train_handwriting
    python -m src.training.train_handwriting --smoke-test
"""

import os
import argparse
import numpy as np
import tensorflow as tf

from src.config import (
    RANDOM_SEED, EPOCHS, BATCH_SIZE, EARLY_STOPPING_PATIENCE,
    HANDWRITING_PHASE1_EPOCHS, HANDWRITING_PHASE2_LR,
    HANDWRITING_RAW_DIR, HANDWRITING_PROCESSED_DIR, HANDWRITING_SPLITS_PATH,
    SAVED_MODELS_DIR, LOGS_DIR,
)
from src.data.handwriting_loader import load_handwriting_data, get_handwriting_class_weights
from src.models.handwriting_model import build_handwriting_model, unfreeze_for_phase2


def train(smoke_test: bool = False) -> tf.keras.Model:
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    print("\n=== Training Handwriting Baseline ===")
    X_train, y_train, X_val, y_val, X_test, y_test = load_handwriting_data(
        HANDWRITING_RAW_DIR, HANDWRITING_PROCESSED_DIR, HANDWRITING_SPLITS_PATH
    )

    class_weights = get_handwriting_class_weights(y_train)
    print(f"Class weights: {class_weights}")

    save_dir = os.path.join(SAVED_MODELS_DIR, "handwriting_baseline")
    os.makedirs(save_dir, exist_ok=True)
    log_dir = os.path.join(LOGS_DIR, "handwriting")
    os.makedirs(log_dir, exist_ok=True)

    phase1_epochs = 2 if smoke_test else HANDWRITING_PHASE1_EPOCHS
    phase2_epochs = 3 if smoke_test else (EPOCHS - HANDWRITING_PHASE1_EPOCHS)

    # ── Phase 1: head-only training ──────────────────────────────────────────
    print("\n--- Phase 1: training head only ---")
    model = build_handwriting_model(freeze_base=True)
    model.summary(line_length=100)

    checkpoint_path = os.path.join(save_dir, "best_model.keras")

    phase1_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_roc_auc", save_best_only=True, mode="max", verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(log_dir, "phase1")),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=phase1_epochs,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=phase1_callbacks,
        verbose=1,
    )

    # ── Phase 2: fine-tune top DenseNet layers ───────────────────────────────
    print("\n--- Phase 2: fine-tuning top DenseNet layers ---")
    model = unfreeze_for_phase2(model)

    phase2_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_roc_auc", patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True, mode="max",
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_roc_auc", save_best_only=True, mode="max", verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_roc_auc", factor=0.5, patience=7, mode="max", verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(log_dir, "phase2")),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        initial_epoch=phase1_epochs,
        epochs=phase1_epochs + phase2_epochs,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=phase2_callbacks,
        verbose=1,
    )

    # Load best checkpoint and evaluate on test
    model = tf.keras.models.load_model(checkpoint_path)
    test_results = model.evaluate(X_test, y_test, verbose=0)
    print("\n[Handwriting] Test results:")
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
