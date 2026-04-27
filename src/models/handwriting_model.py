import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from src.config import (
    HANDWRITING_IMG_SIZE, HANDWRITING_EMBEDDING_DIM,
    DROPOUT_RATE_HANDWRITING, DENSENET_FREEZE_LAYERS, DENSENET_UNFREEZE_TOP,
)


def build_handwriting_model(
    img_size: tuple = HANDWRITING_IMG_SIZE,
    embedding_dim: int = HANDWRITING_EMBEDDING_DIM,
    dropout_rate: float = DROPOUT_RATE_HANDWRITING,
    freeze_base: bool = True,
) -> tf.keras.Model:
    """
    DenseNet121 pretrained on ImageNet with a custom classification head.

    Phase 1: call with freeze_base=True  (train head only, lr=1e-3)
    Phase 2: unfreeze top layers and recompile with lr=1e-5

    Architecture (head):
        DenseNet121 (frozen/partial) → GlobalAveragePooling2D
        → Dense(256, relu) [embedding] → Dropout → Dense(1, sigmoid)
    """
    base = DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=img_size,
    )

    if freeze_base:
        base.trainable = False
    else:
        base.trainable = True
        for layer in base.layers[:-DENSENET_UNFREEZE_TOP]:
            layer.trainable = False

    inputs = tf.keras.Input(shape=img_size, name="handwriting_input")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    embedding = layers.Dense(embedding_dim, activation="relu", name="hw_embedding")(x)
    x = layers.Dropout(dropout_rate)(embedding)
    output = layers.Dense(1, activation="sigmoid", name="hw_output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=output, name="handwriting_model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="roc_auc")],
    )
    return model


def unfreeze_for_phase2(model: tf.keras.Model) -> tf.keras.Model:
    """
    Unfreeze the top DENSENET_UNFREEZE_TOP layers of the DenseNet121 base
    and recompile with a lower learning rate for fine-tuning.
    """
    base = model.get_layer("densenet121")
    base.trainable = True
    for layer in base.layers[:-DENSENET_UNFREEZE_TOP]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="roc_auc")],
    )
    return model


def get_handwriting_embedding_model(trained_model: tf.keras.Model) -> tf.keras.Model:
    """Returns a model that outputs the 256-dim embedding."""
    return tf.keras.Model(
        inputs=trained_model.input,
        outputs=trained_model.get_layer("hw_embedding").output,
        name="hw_embedding_model",
    )
