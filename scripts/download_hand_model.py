"""Descarga el modelo oficial Hand Landmarker de MediaPipe."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "mediapipe" / "hand_landmarker.task",
        help="Ruta de destino (por defecto: data/mediapipe/hand_landmarker.task).",
    )
    parser.add_argument("--force", action="store_true", help="Descarga aunque el archivo ya exista.")
    args = parser.parse_args()

    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and not args.force:
        print(f"El modelo ya existe: {output}")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    request = Request(MODEL_URL, headers={"User-Agent": "asl-sign-language-recognition/1.0"})
    try:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as file:
            while chunk := response.read(1024 * 1024):
                file.write(chunk)
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    print(f"Modelo descargado en: {output}")


if __name__ == "__main__":
    main()
