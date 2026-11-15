"""Real-time inference helpers for the static-map and J/Z flow models."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf


IMAGE_SIZE = 224
SEQUENCE_FRAMES = 17
FLOW_SIZE = 128
FLOW_CLIP_PIXELS = 20.0
TEMPORAL_SAMPLE_EVERY = 3  # ~1.6 s at 30 FPS for 17 sampled frames

FINGER_COLORS = [(255, 90, 90), (255, 210, 40), (80, 235, 100), (50, 190, 255), (220, 90, 255)]
FINGER_EDGES = [
    [(0, 1), (1, 2), (2, 3), (3, 4)],
    [(0, 5), (5, 6), (6, 7), (7, 8)],
    [(0, 9), (9, 10), (10, 11), (11, 12)],
    [(0, 13), (13, 14), (14, 15), (15, 16)],
    [(0, 17), (17, 18), (18, 19), (19, 20)],
]
PALM_EDGES = [(5, 9), (9, 13), (13, 17)]
HAND_EDGES = [edge for finger in FINGER_EDGES for edge in finger] + PALM_EDGES
POINT_COLORS = {
    **{i: FINGER_COLORS[0] for i in range(1, 5)},
    **{i: FINGER_COLORS[1] for i in range(5, 9)},
    **{i: FINGER_COLORS[2] for i in range(9, 13)},
    **{i: FINGER_COLORS[3] for i in range(13, 17)},
    **{i: FINGER_COLORS[4] for i in range(17, 21)},
    0: (255, 255, 255),
}


def landmarks_to_map(points: np.ndarray) -> np.ndarray | None:
    """Render normalized MediaPipe XY landmarks exactly like notebook 09."""
    xy = np.asarray(points, dtype=np.float32)
    if xy.shape != (21, 2) or not np.isfinite(xy).all():
        return None
    xy = xy.copy()
    xy -= xy[0]
    palm_scale = float(np.linalg.norm(xy[9]))
    if palm_scale < 1e-6:
        return None
    xy /= palm_scale
    extent = float(np.max(np.abs(xy)))
    if extent < 1e-6:
        return None
    xy = xy / extent * 0.40 * (IMAGE_SIZE - 1) + (IMAGE_SIZE - 1) / 2
    coords = [(int(round(x)), int(round(y))) for x, y in xy]

    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    width, radius = max(2, IMAGE_SIZE // 70), max(3, IMAGE_SIZE // 48)
    for edges, color in zip(FINGER_EDGES, FINGER_COLORS):
        for start, end in edges:
            draw.line((coords[start], coords[end]), fill=color, width=width)
    for start, end in PALM_EDGES:
        draw.line((coords[start], coords[end]), fill=(235, 235, 235), width=width)
    for index, (x, y) in enumerate(coords):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                     fill=POINT_COLORS[index], outline=(255, 255, 255))
    return np.asarray(canvas, dtype=np.uint8)


class LiveSignPipeline:
    """Load saved Keras models and process webcam frames from both branches."""

    def __init__(
        self,
        static_model_path: str | Path,
        classes_path: str | Path,
        hand_model_path: str | Path,
        temporal_model_path: str | Path | None = None,
        dynamic_threshold: float = 0.80,
        dynamic_stride: int = 5,
    ):
        self.static_model_path = Path(static_model_path)
        self.classes_path = Path(classes_path)
        self.hand_model_path = Path(hand_model_path)
        self.temporal_model_path = Path(temporal_model_path) if temporal_model_path else None
        for path in (self.static_model_path, self.classes_path, self.hand_model_path):
            if not path.is_file():
                raise FileNotFoundError(f"No encuentro el artefacto requerido: {path}")

        try:
            for device in tf.config.list_physical_devices("GPU"):
                tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError:
            pass

        self.static_model = tf.keras.models.load_model(self.static_model_path, compile=False)
        import json
        self.classes = json.loads(self.classes_path.read_text(encoding="utf-8"))
        if self.static_model.output_shape[-1] != len(self.classes):
            raise ValueError("classes.json no coincide con la cantidad de salidas del modelo estático.")

        self.temporal_model = None
        if self.temporal_model_path and self.temporal_model_path.is_file():
            self.temporal_model = tf.keras.models.load_model(self.temporal_model_path, compile=False)
            expected = (None, SEQUENCE_FRAMES - 1, FLOW_SIZE, FLOW_SIZE, 2)
            if tuple(self.temporal_model.input_shape) != expected:
                raise ValueError(
                    f"El modelo temporal espera {self.temporal_model.input_shape}; "
                    f"la app genera {expected}."
                )

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.hand_model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.frames = deque(maxlen=SEQUENCE_FRAMES)
        self.dynamic_threshold = dynamic_threshold
        self.dynamic_stride = max(1, int(dynamic_stride))
        self.frame_index = 0
        self.latest_dynamic = (None, 0.0)

    def _static_prediction(self, landmark_points: np.ndarray | None):
        if landmark_points is None:
            if "nothing" in self.classes:
                return "nothing", 1.0
            return None, 0.0
        image = landmarks_to_map(landmark_points)
        if image is None:
            return None, 0.0
        probabilities = self.static_model(np.expand_dims(image.astype(np.float32), 0), training=False).numpy()[0]
        index = int(np.argmax(probabilities))
        return self.classes[index], float(probabilities[index])

    @staticmethod
    def _fixed_crops(frames: list[np.ndarray], points: list[np.ndarray]) -> list[np.ndarray]:
        all_points = np.concatenate(points, axis=0)
        x_min, y_min = all_points.min(axis=0)
        x_max, y_max = all_points.max(axis=0)
        center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
        side = max(float(max(x_max - x_min, y_max - y_min) * 1.7), 32.0)
        x0, y0 = int(np.floor(center_x - side / 2)), int(np.floor(center_y - side / 2))
        side = int(np.ceil(side))
        crops = []
        for frame in frames:
            height, width = frame.shape[:2]
            x1, y1 = x0 + side, y0 + side
            left, top = max(0, -x0), max(0, -y0)
            right, bottom = max(0, x1 - width), max(0, y1 - height)
            padded = cv2.copyMakeBorder(frame, top, bottom, left, right,
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))
            crop = padded[y0 + top:y1 + top, x0 + left:x1 + left]
            crops.append(cv2.resize(crop, (FLOW_SIZE, FLOW_SIZE), interpolation=cv2.INTER_AREA))
        return crops

    def _dynamic_prediction(self):
        if self.temporal_model is None or len(self.frames) != SEQUENCE_FRAMES:
            return self.latest_dynamic
        window = list(self.frames)
        if any(points is None for _, points in window):
            self.latest_dynamic = ("other", 0.0)
            return self.latest_dynamic
        crops = self._fixed_crops([frame for frame, _ in window], [points for _, points in window])
        grays = [cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) for crop in crops]
        flows = []
        for previous, current in zip(grays[:-1], grays[1:]):
            flow = cv2.calcOpticalFlowFarneback(
                previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            flows.append(np.clip(flow / FLOW_CLIP_PIXELS, -1.0, 1.0).astype(np.float32))
        tensor = np.expand_dims(np.stack(flows), axis=0)
        probabilities = self.temporal_model(tensor, training=False).numpy()[0]
        labels = ["J", "Z", "other"]
        index = int(np.argmax(probabilities))
        self.latest_dynamic = (labels[index], float(probabilities[index]))
        return self.latest_dynamic

    def process(self, frame_bgr: np.ndarray):
        """Return annotated frame and (label, confidence, source, hand_detected)."""
        self.frame_index += 1
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        timestamp_ms = int(round(self.frame_index * 1000 / 30))
        result = self.detector.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
        )
        points = None
        annotated = frame_bgr.copy()
        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            points = np.asarray([(point.x * frame_bgr.shape[1], point.y * frame_bgr.shape[0])
                                 for point in hand], dtype=np.float32)
            for start, end in HAND_EDGES:
                cv2.line(annotated, tuple(points[start].astype(int)), tuple(points[end].astype(int)),
                         (80, 220, 120), 2)
            for point in points:
                cv2.circle(annotated, tuple(point.astype(int)), 3, (40, 180, 255), -1)

        static_label, static_confidence = self._static_prediction(points)
        if self.frame_index % TEMPORAL_SAMPLE_EVERY == 0:
            self.frames.append((frame_bgr.copy(), points))
        if self.temporal_model is not None and len(self.frames) == SEQUENCE_FRAMES:
            if self.frame_index % self.dynamic_stride == 0:
                self._dynamic_prediction()
        dynamic_label, dynamic_confidence = self.latest_dynamic

        if dynamic_label in {"J", "Z"} and dynamic_confidence >= self.dynamic_threshold:
            prediction = (dynamic_label, dynamic_confidence, "dinámico", points is not None)
        else:
            prediction = (static_label, static_confidence, "estático", points is not None)
        return annotated, prediction

    def close(self):
        self.detector.close()


class SpellingBuffer:
    """Accumulate accepted letters only while recording is active."""

    def __init__(self, stable_frames: int = 5):
        self.active = False
        self.text = ""
        self.stable_frames = stable_frames
        self._candidate = None
        self._count = 0
        self._last_accepted = None

    def observe(self, label: str | None, confidence: float, threshold: float = 0.80):
        if label is None or confidence < threshold:
            self._candidate, self._count = None, 0
            return None
        if label == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = label, 1
        if self._count < self.stable_frames:
            return None
        if label in {"nothing", "other"}:
            self._last_accepted = None
            return None
        if label == self._last_accepted:
            return None
        if self.active:
            if label == "space":
                if self.text and not self.text.endswith(" "):
                    self.text += " "
                accepted = " "
            elif len(label) == 1 and label.isalpha():
                self.text += label.upper()
                accepted = label.upper()
            else:
                accepted = None
            self._last_accepted = label
            return accepted
        return None

    def toggle(self):
        self.active = not self.active
        self._candidate, self._count, self._last_accepted = None, 0, None
        if not self.active:
            return self.text
        return None

    def clear(self):
        self.text = ""
        self._candidate, self._count, self._last_accepted = None, 0, None
