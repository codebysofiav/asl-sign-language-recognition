"""Prepare MediaPipe hand crops in a class-folder dataset, preserving paths."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hand_crop import HandCropper  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Carpeta con una subcarpeta por clase.")
    parser.add_argument("--output", type=Path, required=True, help="Destino de las imágenes procesadas.")
    parser.add_argument("--manifest", type=Path, required=True, help="CSV de rutas y detecciones.")
    parser.add_argument("--model", type=Path, default=ROOT / "data/mediapipe/hand_landmarker.task")
    parser.add_argument("--margin", type=float, default=0.30, help="Margen proporcional alrededor de la mano.")
    parser.add_argument("--max-per-class", type=int, default=None, help="Muestras aleatorias por clase para un piloto.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla del piloto.")
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()
    input_dir = args.input if args.input.is_absolute() else ROOT / args.input
    output_dir = args.output if args.output.is_absolute() else ROOT / args.output
    manifest = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    model_path = args.model if args.model.is_absolute() else ROOT / args.model
    if not input_dir.is_dir():
        parser.error(f"No existe la carpeta de entrada: {input_dir}")
    if args.progress_every < 1:
        parser.error("--progress-every debe ser mayor que cero")
    if args.max_per_class is not None and args.max_per_class < 1:
        parser.error("--max-per-class debe ser mayor que cero")

    images = [path for path in sorted(input_dir.rglob("*")) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    if args.max_per_class is not None:
        rng = np.random.default_rng(args.seed)
        sampled = []
        for label in sorted({path.parent.name for path in images}):
            class_images = [path for path in images if path.parent.name == label]
            if len(class_images) > args.max_per_class:
                indices = np.sort(rng.choice(len(class_images), size=args.max_per_class, replace=False))
                class_images = [class_images[index] for index in indices]
            sampled.extend(class_images)
        images = sorted(sampled)
    rows = [(path, path.relative_to(input_dir).as_posix(), path.parent.name) for path in images]
    if not rows:
        parser.error(f"No hay imágenes en {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    failures = 0
    with HandCropper(model_path, margin=args.margin) as cropper, manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("path", "label", "detected"))
        writer.writeheader()
        for index, (source, relative, label) in enumerate(rows, 1):
            destination = output_dir / Path(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                crop, detected = cropper.crop(source)
                image_format = "PNG" if destination.suffix.lower() == ".png" else "JPEG"
                crop.save(destination, format=image_format)
            except Exception as error:
                # Keep the sample and record the miss; evaluation will treat it as `nothing`.
                detected = False
                failures += 1
                with Image.open(source) as original:
                    image_format = "PNG" if destination.suffix.lower() == ".png" else "JPEG"
                    original.convert("RGB").save(destination, format=image_format)
                print(f"Error en {source}: {error}", file=sys.stderr)
            writer.writerow({"path": relative, "label": label, "detected": int(detected)})
            counts[f"{label}:total"] += 1
            counts[f"{label}:detected"] += int(detected)
            if index % args.progress_every == 0 or index == len(rows):
                f.flush()
                print(f"Procesadas {index}/{len(rows)} imágenes")

    detected_total = sum(counts[f"{label}:detected"] for label in {row[2] for row in rows})
    print(f"Recortes: {output_dir}\nManifiesto: {manifest}")
    print(f"Manos detectadas: {detected_total}/{len(rows)} ({detected_total / len(rows):.1%}); errores: {failures}")
    for label in sorted({row[2] for row in rows}):
        total, detected = counts[f"{label}:total"], counts[f"{label}:detected"]
        print(f"  {label}: {detected}/{total} detectadas")


if __name__ == "__main__":
    main()
