import tensorflow as tf
from tensorflow.keras import layers, regularizers
from src.config import (
    GAIT_SEQUENCE_LEN, GAIT_N_FEATURES,
    GAIT_EMBEDDING_DIM, DROPOUT_RATE_GAIT, L2_LAMBDA,
)


def build_gait_model(
    seq_len: int = GAIT_SEQUENCE_LEN,
    n_features: int = GAIT_N_FEATURES,
    embedding_dim: int = GAIT_EMBEDDING_DIM,
    dropout_rate: float = DROPOUT_RATE_GAIT,
    l2_lambda: float = L2_LAMBDA,
) -> tf.keras.Model:
    """
    Lightweight Conv1D model for stride sequence classification.

    Architecture:
        Input(300, 8) → Conv1D(32, k=5) → BN → MaxPool(2)
                      → Conv1D(64, k=3) → BN → GlobalAvgPool
                      → Dropout → Dense(32) [embedding] → Dense(1, sigmoid)

    GlobalAveragePooling replaces LSTM — more stable on small datasets.
    """
    reg = regularizers.l2(l2_lambda)
    inputs = tf.keras.Input(shape=(seq_len, n_features), name="gait_input")

    x = layers.Conv1D(32, kernel_size=5, activation="relu", padding="same",
                      kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(64, kernel_size=3, activation="relu", padding="same",
                      kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dropout(dropout_rate)(x)
    embedding = layers.Dense(embedding_dim, activation="relu",
                             kernel_regularizer=reg, name="gait_embedding")(x)
    output = layers.Dense(1, activation="sigmoid", name="gait_output")(embedding)

    model = tf.keras.Model(inputs=inputs, outputs=output, name="gait_model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="roc_auc")],
    )
    return model


def get_gait_embedding_model(trained_model: tf.keras.Model) -> tf.keras.Model:
    """Returns a model that outputs the 32-dim gait embedding."""
    return tf.keras.Model(
        inputs=trained_model.input,
        outputs=trained_model.get_layer("gait_embedding").output,
        name="gait_embedding_model",
    )
