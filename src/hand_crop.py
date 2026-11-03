"""Crop a hand region with MediaPipe for RGB image classifiers."""

from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image, ImageOps


class HandCropper:
    """Detect one hand and return a padded square RGB crop."""

    def __init__(self, model_path: str | Path, margin: float = 0.30):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"No existe {model_path}. Ejecuta `python scripts/download_hand_model.py`."
            )
        if margin < 0:
            raise ValueError("margin debe ser mayor o igual a cero")
        self.margin = margin
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
        )
        self._detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def crop(self, image_path: str | Path) -> tuple[Image.Image, bool]:
        """Return (crop, detected); on misses, return the full oriented image."""
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        rgb = np.asarray(image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)
        if not result.hand_landmarks:
            return image, False

        points = np.asarray([(p.x * image.width, p.y * image.height) for p in result.hand_landmarks[0]])
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
        side = max(x1 - x0, y1 - y0)
        if side < 2:
            return image, False
        side *= 1 + 2 * self.margin
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        left, top = int(round(cx - side / 2)), int(round(cy - side / 2))
        right, bottom = left + int(round(side)), top + int(round(side))

        # Pad at image boundaries instead of shrinking the hand crop.
        crop = Image.new("RGB", (right - left, bottom - top), (0, 0, 0))
        source_box = (max(0, left), max(0, top), min(image.width, right), min(image.height, bottom))
        crop.paste(image.crop(source_box), (source_box[0] - left, source_box[1] - top))
        return crop, True

    def close(self) -> None:
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
