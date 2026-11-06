"""Detecta la mano con MediaPipe (API Tasks) y recorta un cuadrado alrededor de ella.

Se ejecuta en el entorno .venv-mp (no importa TensorFlow):
    .venv-mp/bin/python -m src.mediapipe_crop --input data/mis_fotos \
        --output data/mis_fotos_recortes --csv data/mis_fotos_recortes/deteccion.csv

Entrada: carpeta con subcarpetas por clase (<clase>/<imagen>).
Salida:  recortes cuadrados (si se pasa --output) y un CSV con el resultado por imagen.
Si no se detecta mano, el recorte es el cuadrado central de la imagen y detected=False.
"""
import argparse
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# El kernel de Jupyter hereda MPLBACKEND=module://matplotlib_inline..., que este entorno no tiene.
# MediaPipe importa matplotlib, así que se fuerza un backend sin ventanas antes de importarlo.
os.environ["MPLBACKEND"] = "Agg"

IMG_EXT = {".jpg", ".jpeg", ".png"}
DETECT_MAX_SIDE = 1024  # se reduce solo para detectar; el recorte sale de la imagen original


def square_crop_box(xmin, ymin, xmax, ymax, w, h, margin=0.25):
    """Cuadrado alrededor del rectángulo de la mano (+margen), siempre dentro de la imagen."""
    side = max(xmax - xmin, ymax - ymin) * (1 + 2 * margin)
    side = min(side, w, h)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    left = min(max(cx - side / 2, 0), w - side)
    top = min(max(cy - side / 2, 0), h - side)
    return int(round(left)), int(round(top)), int(round(left + side)), int(round(top + side))


def center_box(w, h):
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return left, top, left + s, top + s


def make_mediapipe_detector(model_path, min_conf=0.5):
    """Devuelve detect(rgb) -> (landmarks Nx2 normalizados, score, mano) o None."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=min_conf,
        min_hand_presence_confidence=min_conf,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(rgb):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = landmarker.detect(image)
        if not result.hand_landmarks:
            return None
        pts = np.array([[p.x, p.y] for p in result.hand_landmarks[0]])
        cat = result.handedness[0][0]
        return pts, float(cat.score), cat.category_name

    return detect


def list_images(input_dir, max_per_class=None, seed=0):
    """[(clase, ruta)] de input_dir/<clase>/*; opcionalmente una muestra por clase."""
    rng = random.Random(seed)
    items = []
    for d in sorted(p for p in Path(input_dir).iterdir() if p.is_dir()):
        files = sorted(f for f in d.iterdir() if f.suffix.lower() in IMG_EXT)
        if max_per_class and len(files) > max_per_class:
            files = sorted(rng.sample(files, max_per_class))
        items += [(d.name, f) for f in files]
    return items


def process_folder(input_dir, detect, output_dir=None, csv_path=None, size=200, margin=0.25,
                   max_per_class=None, seed=0):
    """Procesa todas las imágenes. `detect` se inyecta para poder probar sin MediaPipe."""
    input_dir = Path(input_dir)
    items = list_images(input_dir, max_per_class, seed)
    if not items:
        raise FileNotFoundError(f"No hay imágenes en {input_dir}/<clase>/")
    rows, t0 = [], time.time()
    for k, (label, f) in enumerate(items, 1):
        with Image.open(f) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")
        w, h = img.size
        scale = DETECT_MAX_SIDE / max(w, h)
        small = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS) if scale < 1 else img
        found = detect(np.asarray(small))
        if found is None:
            box, score, hand = center_box(w, h), 0.0, ""
        else:
            pts, score, hand = found
            xs, ys = pts[:, 0] * w, pts[:, 1] * h
            box = square_crop_box(xs.min(), ys.min(), xs.max(), ys.max(), w, h, margin)
        rel = f"{label}/{f.name}"
        rows.append({"path": rel, "label": label, "detected": found is not None,
                     "score": round(score, 3), "hand": hand,
                     "left": box[0], "top": box[1], "right": box[2], "bottom": box[3], "w": w, "h": h})
        if output_dir:
            out = Path(output_dir) / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            img.crop(box).resize((size, size), Image.LANCZOS).save(out, quality=95)
        if k % 200 == 0:
            print(f"  {k}/{len(items)} imágenes ({k / (time.time() - t0):.1f} img/s)", flush=True)

    if csv_path:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    n_det = sum(r["detected"] for r in rows)
    print(f"{len(rows)} imágenes | mano detectada en {n_det} ({n_det / len(rows):.1%}) | "
          f"{time.time() - t0:.0f} s ({len(rows) / (time.time() - t0):.1f} img/s)")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", help="carpeta de recortes (opcional; sin ella solo se mide la detección)")
    ap.add_argument("--csv", help="CSV con el resultado por imagen")
    ap.add_argument("--model", default="assets/hand_landmarker.task")
    ap.add_argument("--margin", type=float, default=0.25, help="margen alrededor de la mano (fracción)")
    ap.add_argument("--size", type=int, default=200, help="lado del recorte guardado (px)")
    ap.add_argument("--min-conf", type=float, default=0.5, help="confianza mínima de detección")
    ap.add_argument("--max-per-class", type=int, help="muestra de N imágenes por clase")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    process_folder(a.input, make_mediapipe_detector(a.model, a.min_conf), a.output, a.csv,
                   a.size, a.margin, a.max_per_class, a.seed)
