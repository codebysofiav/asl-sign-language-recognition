"""Extrae puntos de manos desde carpetas organizadas por clase."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.landmarks import FEATURES_PER_HAND, HandLandmarkExtractor  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def collect_images(
    input_dir: Path,
    max_per_class: int | None,
    seed: int,
) -> list[tuple[Path, str]]:
    """Lee imágenes de input_dir/<clase>/<archivo> en orden reproducible."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de entrada: {input_dir}")

    rows: list[tuple[Path, str]] = []
    rng = np.random.default_rng(seed)
    for class_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        class_images = sorted(
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if max_per_class is not None and len(class_images) > max_per_class:
            selected = np.sort(rng.choice(len(class_images), size=max_per_class, replace=False))
            class_images = [class_images[index] for index in selected]
        rows.extend((path, class_dir.name) for path in class_images)
    if not rows:
        raise FileNotFoundError(f"No se encontraron imágenes en {input_dir}/<clase>/")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Carpeta con una subcarpeta por clase.")
    parser.add_argument("--output", type=Path, required=True, help="Archivo .npz que se va a generar.")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "data" / "mediapipe" / "hand_landmarker.task",
        help="Modelo .task de MediaPipe.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Límite por clase para una ejecución piloto; omitir para procesar todo.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla para seleccionar muestras del piloto.")
    parser.add_argument("--progress-every", type=int, default=100, help="Frecuencia de mensajes de avance.")
    args = parser.parse_args()

    input_dir = args.input if args.input.is_absolute() else ROOT / args.input
    output = args.output if args.output.is_absolute() else ROOT / args.output
    model_path = args.model if args.model.is_absolute() else ROOT / args.model
    if args.max_per_class is not None and args.max_per_class < 1:
        parser.error("--max-per-class debe ser mayor que cero.")
    if args.progress_every < 1:
        parser.error("--progress-every debe ser mayor que cero.")

    rows = collect_images(input_dir, args.max_per_class, args.seed)
    features = np.full((len(rows), FEATURES_PER_HAND), np.nan, dtype=np.float32)
    labels = np.asarray([label for _, label in rows], dtype=str)
    paths = np.asarray([str(path.relative_to(input_dir)) for path, _ in rows], dtype=str)
    detected = np.zeros(len(rows), dtype=bool)
    failures: Counter[str] = Counter()

    with HandLandmarkExtractor(model_path) as extractor:
        for index, (path, label) in enumerate(rows):
            try:
                features[index], detected[index] = extractor.extract(path)
            except Exception as error:
                failures[label] += 1
                print(f"Error en {path}: {error}", file=sys.stderr)

            if (index + 1) % args.progress_every == 0 or index + 1 == len(rows):
                print(f"Procesadas {index + 1}/{len(rows)} imágenes")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=features,
        labels=labels,
        paths=paths,
        detected=detected,
    )

    print(f"Archivo de características: {output}")
    print(f"Manos detectadas: {detected.sum()}/{len(rows)} ({detected.mean():.1%})")
    for label in sorted(set(labels)):
        mask = labels == label
        print(f"  {label}: {detected[mask].sum()}/{mask.sum()} detectadas")
    if failures:
        print(f"Imágenes con error de lectura/proceso: {sum(failures.values())}")


if __name__ == "__main__":
    main()
