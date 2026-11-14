"""Modelos del proyecto: baseline CNN y transfer learning (MobileNetV2, EfficientNetB0)."""
from tensorflow import keras
from tensorflow.keras import layers


def _legacy_augment(x):
    """Aumento básico original (rotación ±18°, zoom ±10 %, traslación ±5 %). Sin volteo."""
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.1)(x)
    return layers.RandomTranslation(0.05, 0.05)(x)


def build_augmentation(a, value_range, x):
    """Aumento configurable desde el YAML (sección `augment`). Solo activo al entrenar.

    rotation     fracción de vuelta (0.03 ≈ ±11°)        zoom         ±z (0.3 ≈ 0,77x a 1,43x)
    translation  fracción del tamaño                      brightness / contrast / saturation  factores ±
    hue          cambio de tono (mejor 0: altera el color de piel)
    blur         intensidad máxima del desenfoque (0-1)   erasing      probabilidad de borrar un parche
    Sin volteo horizontal: cambiaría la mano y la seña.
    """
    f32 = {"dtype": "float32"}  # las capas de aumento van en float32 (el desenfoque no admite float16)
    if a.get("rotation"):
        x = layers.RandomRotation(a["rotation"], **f32)(x)
    if a.get("zoom"):
        x = layers.RandomZoom((-a["zoom"], a["zoom"]), **f32)(x)
    if a.get("translation"):
        x = layers.RandomTranslation(a["translation"], a["translation"], **f32)(x)
    jitter = {
        "brightness_factor": a.get("brightness") or None,
        "contrast_factor": a.get("contrast") or None,
        "saturation_factor": a.get("saturation") or None,
        "hue_factor": a.get("hue") or None,
    }
    if any(v for v in jitter.values()):
        x = layers.RandomColorJitter(value_range=value_range, **jitter, **f32)(x)
    if a.get("blur"):
        x = layers.RandomGaussianBlur(factor=a["blur"], value_range=value_range, **f32)(x)
    if a.get("erasing"):
        x = layers.RandomErasing(factor=a["erasing"], scale=(0.02, 0.15), value_range=value_range, **f32)(x)
    return x


def _augment(x, cfg, value_range):
    """Aumento según la config; si no hay sección `augment`, el básico original."""
    if "augment" in cfg:
        return build_augmentation(cfg["augment"], value_range, x)
    return _legacy_augment(x)


def build_baseline(num_classes, img_size, cfg):
    """CNN baseline: aumento de datos + bloques Conv-BN-ReLU + GAP + clasificador."""
    m = cfg["model"]
    inputs = keras.Input((*img_size, 3))
    x = layers.Rescaling(1 / 255)(inputs)
    x = _augment(x, cfg, (0, 1))

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
    x = _augment(inputs, cfg, (0, 255))
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


def compile_model(model, learning_rate, label_smoothing=0.0):
    """Con label_smoothing > 0 la pérdida es categórica y las etiquetas deben ir en one-hot."""
    if label_smoothing > 0:
        loss = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    else:
        loss = "sparse_categorical_crossentropy"
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=loss, metrics=["accuracy"])
    return model
