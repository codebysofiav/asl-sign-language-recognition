"""Modelos del proyecto."""
from tensorflow import keras
from tensorflow.keras import layers


def build_baseline(num_classes, img_size, cfg):
    """CNN baseline: aumento de datos + bloques Conv-BN-ReLU + GAP + clasificador."""
    m = cfg["model"]
    inputs = keras.Input((*img_size, 3))
    x = layers.Rescaling(1 / 255)(inputs)
    # Aumento (solo activo en entrenamiento). Sin flip: cambiaría la mano/seña.
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomTranslation(0.05, 0.05)(x)

    for f in m["filters"]:
        for _ in range(2):
            x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(m["dropout"])(x)
    x = layers.Dense(m["dense_units"], activation="relu")(x)
    x = layers.Dropout(m["dropout"])(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)  # float32 con mixed precision
    return keras.Model(inputs, outputs, name="baseline_cnn")


def compile_model(model, learning_rate):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
