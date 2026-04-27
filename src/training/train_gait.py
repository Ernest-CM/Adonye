"""
Train the gait unimodal baseline model (Conv1D-LSTM).

Usage:
    python -m src.training.train_gait
    python -m src.training.train_gait --smoke-test
"""

import os
import argparse
import numpy as np
import tensorflow as tf

from src.config import (
    RANDOM_SEED, EPOCHS, BATCH_SIZE, EARLY_STOPPING_PATIENCE,
    GAIT_RAW_DIR, GAIT_PROCESSED_DIR, GAIT_SPLITS_PATH,
    SAVED_MODELS_DIR, LOGS_DIR,
)
from src.data.gait_loader import load_gait_data, get_gait_class_weights
from src.models.gait_model import build_gait_model


def train(smoke_test: bool = False) -> tf.keras.Model:
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    print("\n=== Training Gait Baseline ===")
    X_train, y_train, X_val, y_val, X_test, y_test = load_gait_data(
        GAIT_RAW_DIR, GAIT_PROCESSED_DIR, GAIT_SPLITS_PATH
    )

    class_weights = get_gait_class_weights(y_train)
    print(f"Class weights: {class_weights}")

    model = build_gait_model()
    model.summary()

    save_dir = os.path.join(SAVED_MODELS_DIR, "gait_baseline")
    os.makedirs(save_dir, exist_ok=True)
    log_dir = os.path.join(LOGS_DIR, "gait")
    os.makedirs(log_dir, exist_ok=True)

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
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    test_results = model.evaluate(X_test, y_test, verbose=0)
    print("\n[Gait] Test results:")
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
