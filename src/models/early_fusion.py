import tensorflow as tf
from tensorflow.keras import layers
from src.config import (
    SPEECH_FEATURE_DIM, HANDWRITING_IMG_SIZE, GAIT_SEQUENCE_LEN, GAIT_N_FEATURES,
    SPEECH_EMBEDDING_DIM, HANDWRITING_EMBEDDING_DIM, GAIT_EMBEDDING_DIM,
    DROPOUT_RATE_FUSION,
)
from src.models.speech_model import get_speech_embedding_model
from src.models.handwriting_model import get_handwriting_embedding_model
from src.models.gait_model import get_gait_embedding_model


def build_early_fusion_model(
    speech_model: tf.keras.Model,
    handwriting_model: tf.keras.Model,
    gait_model: tf.keras.Model,
    dropout_rate: float = DROPOUT_RATE_FUSION,
) -> tf.keras.Model:
    """
    Early Fusion: concatenate frozen unimodal embeddings, train a shared head.

    Embedding dimensions: 32 (speech) + 256 (handwriting) + 32 (gait) = 320
    Classification head: Dense(256) → BN → Dropout → Dense(128) → Dropout → Dense(1)

    Unimodal weights are frozen — only the fusion head is trained.
    """
    speech_emb_model = get_speech_embedding_model(speech_model)
    hw_emb_model     = get_handwriting_embedding_model(handwriting_model)
    gait_emb_model   = get_gait_embedding_model(gait_model)

    speech_emb_model.trainable = False
    hw_emb_model.trainable     = False
    gait_emb_model.trainable   = False

    speech_input = tf.keras.Input(shape=(SPEECH_FEATURE_DIM,),       name="ef_speech_input")
    hw_input     = tf.keras.Input(shape=HANDWRITING_IMG_SIZE,         name="ef_hw_input")
    gait_input   = tf.keras.Input(shape=(GAIT_SEQUENCE_LEN, GAIT_N_FEATURES), name="ef_gait_input")

    speech_emb = speech_emb_model(speech_input)
    hw_emb     = hw_emb_model(hw_input)
    gait_emb   = gait_emb_model(gait_input)

    fused = layers.Concatenate(name="fused_embedding")([speech_emb, hw_emb, gait_emb])

    x = layers.Dense(256, activation="relu")(fused)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate * 0.75)(x)
    output = layers.Dense(1, activation="sigmoid", name="ef_output")(x)

    model = tf.keras.Model(
        inputs=[speech_input, hw_input, gait_input],
        outputs=output,
        name="early_fusion_model",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="roc_auc")],
    )
    return model
