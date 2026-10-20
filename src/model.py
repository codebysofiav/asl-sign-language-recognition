"""Keras models for the static ASL image classification experiments."""

from __future__ import annotations

import tensorflow as tf


def build_cnn(
    input_shape: tuple[int, int, int] = (200, 200, 3),
    num_classes: int = 29,
    dropout_rate: float = 0.4,
) -> tf.keras.Model:
    """Build a compact conventional CNN for the first ASL baseline.

    The model expects RGB float images in [0, 1], as produced by
    ``src.data.build_tf_dataset``. Its final layer returns logits, so compile
    it with a sparse categorical cross-entropy loss using ``from_logits=True``.
    """
    if len(input_shape) != 3 or input_shape[-1] != 3:
        raise ValueError("input_shape must be (height, width, 3) for RGB images.")
    if any(dimension < 16 for dimension in input_shape[:2]):
        raise ValueError("Image height and width must be at least 16 pixels.")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")
    if not 0 <= dropout_rate < 1:
        raise ValueError("dropout_rate must be in [0, 1).")

    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = inputs

    # Four two-convolution blocks progressively learn richer visual features.
    for block_index, filters in enumerate((32, 64, 128, 256), start=1):
        for conv_index in (1, 2):
            x = tf.keras.layers.Conv2D(
                filters,
                kernel_size=3,
                padding="same",
                use_bias=False,
                name=f"block{block_index}_conv{conv_index}",
            )(x)
            x = tf.keras.layers.BatchNormalization(name=f"block{block_index}_bn{conv_index}")(x)
            x = tf.keras.layers.Activation("relu", name=f"block{block_index}_relu{conv_index}")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2, name=f"block{block_index}_pool")(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="classifier_features")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="classifier_dropout")(x)
    logits = tf.keras.layers.Dense(num_classes, name="class_logits")(x)

    return tf.keras.Model(inputs=inputs, outputs=logits, name="asl_cnn_baseline")


def compile_classifier(
    model: tf.keras.Model,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Compile a logits-output classifier for integer encoded class labels."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def build_resnet50v2(
    input_shape: tuple[int, int, int] = (200, 200, 3),
    num_classes: int = 29,
    dropout_rate: float = 0.3,
    weights: str | None = "imagenet",
    train_base: bool = False,
) -> tf.keras.Model:
    """Build a ResNet50V2 transfer-learning classifier.

    Input images use the project's [0, 1] range. The included rescaling layer
    maps them to [-1, 1], as expected by the Keras ResNetV2 preprocessing.
    The pretrained backbone is frozen by default for feature extraction.
    """
    _validate_transfer_model_args(input_shape, num_classes, dropout_rate)
    backbone = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        pooling="avg",
    )
    backbone.trainable = train_base

    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0, name="resnetv2_preprocess")(inputs)
    x = backbone(x, training=False)
    x = tf.keras.layers.Dropout(dropout_rate, name="classifier_dropout")(x)
    logits = tf.keras.layers.Dense(num_classes, name="class_logits")(x)
    return tf.keras.Model(inputs, logits, name="asl_resnet50v2")


def build_efficientnetb0(
    input_shape: tuple[int, int, int] = (200, 200, 3),
    num_classes: int = 29,
    dropout_rate: float = 0.3,
    weights: str | None = "imagenet",
    train_base: bool = False,
) -> tf.keras.Model:
    """Build an EfficientNetB0 transfer-learning classifier.

    The input rescaling layer restores [0, 255] values for Keras EfficientNet,
    which includes its own input normalization. The pretrained backbone is
    frozen by default for feature extraction.
    """
    _validate_transfer_model_args(input_shape, num_classes, dropout_rate)
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        pooling="avg",
    )
    backbone.trainable = train_base

    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = tf.keras.layers.Rescaling(scale=255.0, name="efficientnet_input_range")(inputs)
    x = backbone(x, training=False)
    x = tf.keras.layers.Dropout(dropout_rate, name="classifier_dropout")(x)
    logits = tf.keras.layers.Dense(num_classes, name="class_logits")(x)
    return tf.keras.Model(inputs, logits, name="asl_efficientnetb0")


def build_keras_model(
    model_name: str,
    *,
    input_shape: tuple[int, int, int] = (200, 200, 3),
    num_classes: int = 29,
    dropout_rate: float = 0.3,
    weights: str | None = "imagenet",
    train_base: bool = False,
) -> tf.keras.Model:
    """Build one supported Keras architecture by its configuration name."""
    name = model_name.strip().lower()
    if name in {"cnn", "custom_cnn", "asl_cnn_baseline"}:
        return build_cnn(input_shape, num_classes, dropout_rate)
    if name in {"resnet50v2", "resnet50_v2"}:
        return build_resnet50v2(input_shape, num_classes, dropout_rate, weights, train_base)
    if name in {"efficientnetb0", "efficientnet_b0"}:
        return build_efficientnetb0(input_shape, num_classes, dropout_rate, weights, train_base)
    raise ValueError(
        f"Unsupported Keras model {model_name!r}. Choose cnn, resnet50v2, or efficientnetb0."
    )


def _validate_transfer_model_args(
    input_shape: tuple[int, int, int], num_classes: int, dropout_rate: float
) -> None:
    if len(input_shape) != 3 or input_shape[-1] != 3:
        raise ValueError("input_shape must be (height, width, 3) for RGB images.")
    if any(dimension < 32 for dimension in input_shape[:2]):
        raise ValueError("Pretrained Keras application inputs must be at least 32 pixels per side.")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")
    if not 0 <= dropout_rate < 1:
        raise ValueError("dropout_rate must be in [0, 1).")
