"""Evaluate a trained hand-crop experiment on labeled external photos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import evaluate_hand_pipeline  # noqa: E402
from src.visualize import list_own_photos  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Directorio experiments/<fecha>_<nombre>.")
    parser.add_argument("--photos", type=Path, default=Path("data/mis_fotos"))
    parser.add_argument("--cropped-dir", type=Path, default=Path("data/processed/hand_crops/mis_fotos"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/hand_crops/mis_fotos_manifest.csv"))
    args = parser.parse_args()
    run = args.run if args.run.is_absolute() else ROOT / args.run
    photos_dir = args.photos if args.photos.is_absolute() else ROOT / args.photos
    cropped_dir = args.cropped_dir if args.cropped_dir.is_absolute() else ROOT / args.cropped_dir
    manifest = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    classes = json.loads((run / "final/classes.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    photos = list_own_photos(photos_dir, classes)
    photos["path"] = photos.path.map(lambda path: Path(path).relative_to(photos_dir).as_posix())
    metrics = evaluate_hand_pipeline(
        keras.models.load_model(run / "final/model.keras"),
        photos,
        cropped_dir,
        manifest,
        classes,
        config["data"]["img_size"],
        config["data"]["batch_size"],
        run / "external_photos",
    )
    print(
        f"Fotos externas: accuracy={metrics['accuracy']:.4f}; "
        f"detección en señas={metrics['hand_detection_rate_on_signs']:.1%}; "
        f"fallos de detección en señas={metrics['missed_hand_count']}"
    )
    print(f"Detalle guardado en {run / 'external_photos'}")


if __name__ == "__main__":
    main()
