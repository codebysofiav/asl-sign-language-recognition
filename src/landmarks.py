"""Extracción y normalización de puntos de referencia de manos con MediaPipe."""

from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image, ImageOps


NUM_LANDMARKS = 21
COORDINATES_PER_LANDMARK = 3
FEATURES_PER_HAND = NUM_LANDMARKS * COORDINATES_PER_LANDMARK


def normalize_hand_landmarks(points: np.ndarray) -> np.ndarray | None:
    """Centra en la muñeca y escala por la longitud muñeca-base del dedo medio.

    Conserva la orientación de la mano. Devuelve 63 características o None si
    los puntos no son válidos o no permiten calcular una escala estable.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (NUM_LANDMARKS, COORDINATES_PER_LANDMARK):
        raise ValueError(f"Se esperaba forma (21, 3); se recibió {points.shape}.")
    if not np.isfinite(points).all():
        return None

    relative = points - points[0]  # muñeca como origen
    scale = float(np.linalg.norm(relative[9]))  # base del dedo medio
    if scale < 1e-6:
        return None
    return (relative / scale).reshape(FEATURES_PER_HAND).astype(np.float32)


class HandLandmarkExtractor:
    """Detector MediaPipe en modo imagen, reutilizable para muchos archivos."""

    def __init__(self, model_path: str | Path, num_hands: int = 1):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"No existe el modelo de MediaPipe: {model_path}. "
                "Ejecuta `python scripts/download_hand_model.py` primero."
            )

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def extract(self, image_path: str | Path) -> tuple[np.ndarray, bool]:
        """Devuelve (63 características, detectada); puntos ausentes son NaN."""
        with Image.open(image_path) as image:
            rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return np.full(FEATURES_PER_HAND, np.nan, dtype=np.float32), False

        hand = result.hand_landmarks[0]
        points = np.asarray([(point.x, point.y, point.z) for point in hand], dtype=np.float32)
        features = normalize_hand_landmarks(points)
        if features is None:
            return np.full(FEATURES_PER_HAND, np.nan, dtype=np.float32), False
        return features, True

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandLandmarkExtractor":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
