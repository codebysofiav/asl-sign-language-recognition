"""Herramientas para mejorar y diagnosticar los modelos sin reentrenar:
barrido de zoom, TTA, ensamble, métricas con fotos propias, Grad-CAM y vista previa del aumento."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


# ---------- zoom, TTA y ensamble ----------
def zoom_array(img, factor, pad_mode="reflect"):
    """img: HxWx3 (0-255). factor > 1 acerca (recorta el centro); factor < 1 aleja y rellena los bordes.
    pad_mode al alejar: "reflect" (refleja la imagen; puede duplicar partes de la mano), "edge" (repite
    el píxel del borde) o "mean" (color medio de la imagen). Válido para factor >= 0.5."""
    if factor == 1:
        return img
    h, w = img.shape[:2]
    if factor > 1:
        ch, cw = round(h / factor), round(w / factor)
        top, left = (h - ch) // 2, (w - cw) // 2
        out = img[top:top + ch, left:left + cw]
    else:
        ph, pw = round(h * (1 / factor - 1) / 2), round(w * (1 / factor - 1) / 2)
        if pad_mode == "mean":
            out = np.empty((h + 2 * ph, w + 2 * pw, 3), dtype=img.dtype)
            out[:] = img.reshape(-1, 3).mean(0)
            out[ph:ph + h, pw:pw + w] = img
        elif pad_mode in ("reflect", "edge"):
            out = np.pad(img, ((ph, ph), (pw, pw), (0, 0)), mode=pad_mode)
        else:
            raise ValueError("pad_mode debe ser 'reflect', 'edge' o 'mean'")
    return tf.image.resize(out, (h, w)).numpy()


def predict_scales(model, images, scales, batch_size=64, pad_mode="reflect"):
    """{escala: probabilidades} con las imágenes ampliadas/reducidas por cada factor."""
    out = {}
    for s in scales:
        batch = images if s == 1 else np.stack([zoom_array(i, s, pad_mode) for i in images])
        out[s] = model.predict(batch, batch_size=batch_size, verbose=0)
    return out


def tta_probs(probs_by_scale, scales):
    """Promedio de probabilidades de varias escalas (TTA multiescala)."""
    return np.mean([probs_by_scale[s] for s in scales], axis=0)


def accuracy(probs, true_ids):
    return float((np.asarray(probs).argmax(1) == np.asarray(true_ids)).mean())


# ---------- métricas con fotos propias ----------
def own_metrics(res):
    """Resumen de un resultado de predict_own_photos: accuracy, sobreconfianza, 'clase refugio'
    y accuracy por sesión (si res tiene la columna `sesion`)."""
    top = res.pred.value_counts()
    out = {
        "accuracy": round(res.correct.mean(), 4),
        "confianza media en errores": round(res.loc[~res.correct, "conf"].mean(), 3) if (~res.correct).any() else np.nan,
        "clase refugio": top.index[0],
        "% predicciones a la refugio": round(top.iloc[0] / len(res), 3),
    }
    if "sesion" in res:
        for ses, g in res.groupby("sesion"):
            out[f"acc {ses}"] = round(g.correct.mean(), 3)
    return out


# ---------- Grad-CAM ----------
def _chain(model):
    """(capas hasta las características, capas posteriores). Transfer: la base anidada;
    baseline: el último ReLU convolucional. Asume una cadena lineal de capas."""
    nested = [l for l in model.layers if isinstance(l, keras.Model)]
    feat = nested[0] if nested else [l for l in model.layers if isinstance(l, keras.layers.ReLU)][-1]
    i = model.layers.index(feat)
    return model.layers[1:i + 1], model.layers[i + 1:]


def gradcam(model, images, class_ids=None):
    """Mapa de calor Grad-CAM (N, h, w) en 0-1 y probabilidades (N, clases).
    class_ids=None usa la clase predicha; si no, la clase indicada para cada imagen.
    El gradiente se calcula sobre la puntuación ANTES del softmax: con predicciones muy seguras el
    gradiente de la probabilidad se anula (y en float16 desaparece del todo), dejando el mapa vacío."""
    head, tail = _chain(model)
    *mid, last = tail
    if not isinstance(last, keras.layers.Dense):
        raise ValueError("Se esperaba una capa Dense al final del modelo")
    x = tf.convert_to_tensor(images, tf.float32)
    with tf.GradientTape() as tape:
        f = x
        for l in head:
            f = l(f, training=False)
        f = tf.cast(f, tf.float32)
        tape.watch(f)
        h = f
        for l in mid:
            h = l(h, training=False)
        logits = tf.matmul(tf.cast(h, tf.float32), last.kernel) + last.bias
        idx = tf.argmax(logits, axis=1, output_type=tf.int32) if class_ids is None else tf.constant(class_ids, tf.int32)
        score = tf.gather(logits, idx, batch_dims=1)
    grads = tape.gradient(score, f)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(f * weights[:, None, None, :], axis=-1))
    cam = tf.math.divide_no_nan(cam, tf.reduce_max(cam, axis=(1, 2), keepdims=True))
    return cam.numpy(), tf.nn.softmax(logits).numpy()


def plot_gradcam(model, images, true_ids, class_names, indices, n_cols=4, alpha=0.45):
    """Cada foto con su Grad-CAM de la clase predicha. Título: predicción (confianza) y clase real."""
    indices = list(indices)
    cam, probs = gradcam(model, images[indices])
    n_rows = int(np.ceil(len(indices) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 3.4 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, k, c, p in zip(axes.flat, indices, cam, probs):
        h, w = images[k].shape[:2]
        heat = tf.image.resize(c[..., None], (h, w)).numpy()[..., 0]
        pred, real = int(p.argmax()), int(true_ids[k])
        ax.imshow(images[k].astype("uint8"))
        ax.imshow(heat, cmap="jet", alpha=alpha)
        ax.set_title(f"{class_names[pred]} {p.max():.0%} (real {class_names[real]})", fontsize=9,
                     color="green" if pred == real else "red")
    plt.tight_layout()
    plt.show()


# ---------- vista previa del aumento de datos ----------
def preview_augmentation(cfg, image, n=7):
    """Muestra una imagen (HxWx3, 0-255) y `n` versiones con el aumento de la config (modo entrenamiento)."""
    from src.model import _augment

    h, w = image.shape[:2]
    inp = keras.Input((h, w, 3))
    aug = keras.Model(inp, _augment(inp, cfg, (0, 255)))
    batch = np.repeat(image[None].astype("float32"), n, axis=0)
    out = aug(batch, training=True).numpy()
    fig, axes = plt.subplots(1, n + 1, figsize=(2.3 * (n + 1), 2.6))
    axes[0].imshow(image.astype("uint8"))
    axes[0].set_title("original")
    for ax, o in zip(axes[1:], out):
        ax.imshow(np.clip(o, 0, 255).astype("uint8"))
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
