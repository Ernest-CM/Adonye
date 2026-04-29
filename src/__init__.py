import sys
import tensorflow as tf

# TF 2.16+ removed `from tensorflow.keras import X` style imports.
# Patch sys.modules using TF's own bundled Keras so model serialization stays consistent.
_keras = tf.keras
sys.modules.setdefault("tensorflow.keras", _keras)
sys.modules.setdefault("tensorflow.keras.layers", _keras.layers)
sys.modules.setdefault("tensorflow.keras.applications", _keras.applications)
sys.modules.setdefault("tensorflow.keras.optimizers", _keras.optimizers)
sys.modules.setdefault("tensorflow.keras.metrics", _keras.metrics)
sys.modules.setdefault("tensorflow.keras.callbacks", _keras.callbacks)
sys.modules.setdefault("tensorflow.keras.models", _keras.models)
sys.modules.setdefault("tensorflow.keras.regularizers", _keras.regularizers)


def safe_load_model(path: str) -> "tf.keras.Model":
    """Load a .keras model, tolerating quantization_config saved by newer Keras versions."""
    try:
        return tf.keras.models.load_model(path)
    except TypeError:
        class _CompatDense(tf.keras.layers.Dense):
            def __init__(self, *args, **kwargs):
                kwargs.pop("quantization_config", None)
                super().__init__(*args, **kwargs)

        return tf.keras.models.load_model(
            path, custom_objects={"Dense": _CompatDense}
        )
