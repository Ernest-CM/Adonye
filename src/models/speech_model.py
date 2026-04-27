import tensorflow as tf
from tensorflow.keras import layers, regularizers
from src.config import SPEECH_FEATURE_DIM, SPEECH_EMBEDDING_DIM, DROPOUT_RATE_SPEECH, L2_LAMBDA


def build_speech_model(
    input_dim: int = SPEECH_FEATURE_DIM,
    embedding_dim: int = SPEECH_EMBEDDING_DIM,
    dropout_rate: float = DROPOUT_RATE_SPEECH,
    l2_lambda: float = L2_LAMBDA,
) -> tf.keras.Model:
    """
    Dense feed-forward classifier on 22 pre-extracted acoustic features.

    Architecture:
        Input(22) → Dense(128) → BN → Dropout → Dense(64) → BN → Dropout
                  → Dense(32) [embedding] → Dense(1, sigmoid)
    """
    reg = regularizers.l2(l2_lambda)

    inputs = tf.keras.Input(shape=(input_dim,), name="speech_input")
    x = layers.Dense(128, activation="relu", kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    embedding = layers.Dense(embedding_dim, activation="relu", name="speech_embedding")(x)
    output = layers.Dense(1, activation="sigmoid", name="speech_output")(embedding)

    model = tf.keras.Model(inputs=inputs, outputs=output, name="speech_model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="roc_auc")],
    )
    return model


def get_speech_embedding_model(trained_model: tf.keras.Model) -> tf.keras.Model:
    """Returns a model that outputs the 32-dim embedding (strips the final sigmoid layer)."""
    return tf.keras.Model(
        inputs=trained_model.input,
        outputs=trained_model.get_layer("speech_embedding").output,
        name="speech_embedding_model",
    )
