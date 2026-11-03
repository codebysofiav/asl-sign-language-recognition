"""Visualización de predicciones: una imagen, una cuadrícula, o fotos propias."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageOps

IMG_EXT = {".jpg", ".jpeg", ".png"}


def load_images(paths, img_size, pre_size=200):
    """Lee imágenes del disco y las deja como en el entrenamiento.

    Corrige la rotación EXIF de los celulares, recorta al cuadrado central y, si es
    más grande que `pre_size` (el dataset original es 200x200), reduce con suavizado
    antes del reescalado final. Para imágenes del dataset no cambia nada.
    Devuelve float32 (N, H, W, 3) en 0-255; el modelo ya incluye el reescalado.
    """
    imgs = []
    for p in paths:
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
        s = min(w, h)
        left, top = (w - s) // 2, (h - s) // 2
        im = im.crop((left, top, left + s, top + s))
        if s > pre_size:
            im = im.resize((pre_size, pre_size), Image.LANCZOS)
        imgs.append(tf.image.resize(np.asarray(im, dtype="float32"), img_size))
    return tf.stack(imgs).numpy()


def load_test_samples(test_df, raw_dir, class_names, img_size, indices):
    """Carga las filas `indices` del conjunto de test. Devuelve (imágenes, etiquetas reales)."""
    rows = test_df.iloc[list(indices)]
    images = load_images([Path(raw_dir) / p for p in rows.path], img_size)
    idx = {c: i for i, c in enumerate(class_names)}
    return images, rows.label.map(idx).to_numpy()


def _draw_image(ax, image, probs, class_names, true_label=None):
    pred = int(np.argmax(probs))
    ax.imshow(image.astype("uint8"))
    ax.set_xticks([])
    ax.set_yticks([])
    text = f"{class_names[pred]} {100 * probs[pred]:.0f}%"
    ok = true_label is None or pred == true_label
    if true_label is not None:
        text += f" ({class_names[true_label]})"
    ax.set_xlabel(text, color="blue" if ok else "red")


def _draw_bars(ax, probs, class_names, true_label=None, show_names=True):
    pred = int(np.argmax(probs))
    bars = ax.bar(range(len(probs)), probs, color="#bbbbbb")
    bars[pred].set_color("red")
    if true_label is not None:
        bars[true_label].set_color("blue")  # si acierta, queda azul
    ax.set_ylim(0, 1)
    if show_names:
        ax.set_xticks(range(len(probs)), class_names, rotation=90, fontsize=8)
    else:
        ax.set_xticks([])
    ax.set_yticks([0, 0.5, 1])


def plot_single(image, probs, class_names, true_label=None):
    """Una imagen con su predicción y las probabilidades de las 29 clases."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), gridspec_kw={"width_ratios": [1, 3]})
    _draw_image(axes[0], image, probs, class_names, true_label)
    _draw_bars(axes[1], probs, class_names, true_label)
    fig.tight_layout()
    plt.show()


def plot_grid(images, probs, class_names, true_labels=None, n_cols=3):
    """Cuadrícula: cada muestra = imagen + barras de probabilidad.

    azul = acierto / clase real, rojo = predicción equivocada.
    `true_labels=None` sirve para fotos propias sin etiqueta.
    """
    n = len(images)
    n_rows = int(np.ceil(n / n_cols))
    fig = plt.figure(figsize=(4.2 * n_cols, 2.2 * n_rows))
    for i in range(n):
        true = None if true_labels is None else int(true_labels[i])
        ax_img = fig.add_subplot(n_rows, 2 * n_cols, 2 * i + 1)
        _draw_image(ax_img, images[i], probs[i], class_names, true)
        ax_bar = fig.add_subplot(n_rows, 2 * n_cols, 2 * i + 2)
        _draw_bars(ax_bar, probs[i], class_names, true, show_names=False)
    fig.tight_layout()
    plt.show()


def list_own_photos(root, class_names):
    """Fotos propias organizadas como root/<clase>/<archivo>. Devuelve DataFrame(path, label)."""
    root = Path(root)
    lookup = {c.lower(): c for c in class_names}  # 'a' y 'A' se aceptan por igual
    rows, bad_dirs, bad_files = [], [], []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        cls = lookup.get(d.name.lower())
        if cls is None:
            bad_dirs.append(d.name)
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in IMG_EXT:
                rows.append((f, cls))
            else:
                bad_files.append(f.name)
    if bad_dirs:
        print("Carpetas ignoradas (no son clases válidas):", bad_dirs)
    if bad_files:
        print("Archivos ignorados (formato no soportado, usa JPG/PNG):", bad_files)
    if not rows:
        raise FileNotFoundError(f"No hay fotos válidas en {root}/<clase>/")
    return pd.DataFrame(rows, columns=["path", "label"])


def predict_own_photos(model, df, class_names, img_size):
    """Predice sobre las fotos de `df`. Devuelve (resultados, imágenes, probabilidades)."""
    images = load_images(df.path, img_size)
    probs = model.predict(images, verbose=0)
    idx = {c: i for i, c in enumerate(class_names)}
    res = df.copy()
    res["true_id"] = res.label.map(idx).to_numpy()
    res["pred"] = [class_names[i] for i in probs.argmax(1)]
    res["conf"] = probs.max(1)
    res["correct"] = res.pred == res.label
    return res, images, probs
