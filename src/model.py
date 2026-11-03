"""Modelos del proyecto: baseline CNN y transfer learning (MobileNetV2, EfficientNetB0)."""
from tensorflow import keras
from tensorflow.keras import layers


def _augment(x):
    """Aumento de datos (solo activo en entrenamiento). Sin flip: cambiaría la mano/seña."""
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.1)(x)
    return layers.RandomTranslation(0.05, 0.05)(x)


def build_baseline(num_classes, img_size, cfg):
    """CNN baseline: aumento de datos + bloques Conv-BN-ReLU + GAP + clasificador."""
    m = cfg["model"]
    inputs = keras.Input((*img_size, 3))
    x = layers.Rescaling(1 / 255)(inputs)
    x = _augment(x)

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


# nombre -> (constructor, preprocesamiento de entrada)
#   "tf":   MobileNetV2 espera entradas en [-1, 1]
#   "none": EfficientNet ya incluye su normalización interna y espera 0-255
TRANSFER_MODELS = {
    "mobilenetv2": (keras.applications.MobileNetV2, "tf"),
    "efficientnetb0": (keras.applications.EfficientNetB0, "none"),
}


def build_transfer(name, num_classes, img_size, cfg):
    """Base preentrenada (congelada) + aumento + cabeza de clasificación."""
    m = cfg["model"]
    constructor, pre = TRANSFER_MODELS[name]
    base = constructor(include_top=False, weights=m.get("weights", "imagenet"), input_shape=(*img_size, 3))
    base.trainable = False

    inputs = keras.Input((*img_size, 3))
    x = _augment(inputs)
    if pre == "tf":
        x = layers.Rescaling(1 / 127.5, offset=-1)(x)
    x = base(x, training=False)  # BatchNorm de la base siempre en modo inferencia
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(m["dropout"])(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    return keras.Model(inputs, outputs, name=name)


def build_model(name, num_classes, img_size, cfg):
    if name == "baseline":
        return build_baseline(num_classes, img_size, cfg)
    if name in TRANSFER_MODELS:
        return build_transfer(name, num_classes, img_size, cfg)
    raise ValueError(f"Modelo desconocido: {name}. Opciones: baseline, {', '.join(TRANSFER_MODELS)}")


def set_finetune_layers(model, n_layers):
    """Fase 2: descongela las últimas `n_layers` capas de la base (BatchNorm sigue congelada)."""
    base = next(l for l in model.layers if isinstance(l, keras.Model))
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    return base


def compile_model(model, learning_rate):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
