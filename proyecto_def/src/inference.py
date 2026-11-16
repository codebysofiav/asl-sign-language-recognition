"""Inference pipeline matching the Colab training notebooks in this folder."""

from __future__ import annotations

import json
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
FRAME_SAMPLE_EVERY = 3
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


def render_landmark_map(points_xy: np.ndarray) -> np.ndarray | None:
    """Build the same normalized RGB skeleton map as the map training notebook."""
    xy = np.asarray(points_xy, dtype=np.float32)
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
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=POINT_COLORS[index], outline=(255, 255, 255))
    return np.asarray(canvas, dtype=np.uint8)


def crop_hand(frame_bgr: np.ndarray, points_xy: np.ndarray, margin: float = 0.30) -> np.ndarray | None:
    """Square padded crop matching the MediaPipe crop training notebook."""
    height, width = frame_bgr.shape[:2]
    x0, y0 = points_xy.min(axis=0)
    x1, y1 = points_xy.max(axis=0)
    side = max(float(max(x1 - x0, y1 - y0)), 2.0) * (1 + 2 * margin)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left, top = int(round(cx - side / 2)), int(round(cy - side / 2))
    side = max(int(round(side)), 2)
    right, bottom = left + side, top + side
    pad_left, pad_top = max(0, -left), max(0, -top)
    pad_right, pad_bottom = max(0, right - width), max(0, bottom - height)
    padded = cv2.copyMakeBorder(frame_bgr, pad_top, pad_bottom, pad_left, pad_right,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
    crop = padded[top + pad_top:bottom + pad_top, left + pad_left:right + pad_left]
    if crop.size == 0:
        return None
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return np.asarray(Image.fromarray(crop_rgb).resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
    ), dtype=np.uint8)


class RecognitionEngine:
    """Run one static branch (maps or crop) plus an optional J/Z video model."""

    def __init__(self, artifact_root: str | Path, static_mode: str = "maps", confidence: float = 0.80):
        self.artifact_root = Path(artifact_root)
        self.static_mode = static_mode
        static_dir = self.artifact_root / static_mode
        required = [static_dir / "model.keras", static_dir / "classes.json", self.artifact_root / "hand_landmarker.task"]
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(f"Falta el artefacto requerido: {path}")
        self.static_model = tf.keras.models.load_model(required[0], compile=False)
        self.classes = json.loads(required[1].read_text(encoding="utf-8"))
        self.dynamic_path = self.artifact_root / "video" / "jz_flow_cnn.keras"
        self.dynamic_model = (
            tf.keras.models.load_model(self.dynamic_path, compile=False)
            if self.dynamic_path.is_file() else None
        )
        dynamic_classes_path = self.artifact_root / "video" / "classes.json"
        self.dynamic_classes = (
            json.loads(dynamic_classes_path.read_text(encoding="utf-8"))
            if dynamic_classes_path.is_file() else ["J", "Z", "other"]
        )
        if self.dynamic_model is not None and self.dynamic_model.output_shape[-1] != len(self.dynamic_classes):
            raise ValueError("video/classes.json no coincide con las salidas del modelo temporal.")
        self.dynamic_confidence_threshold = confidence
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.artifact_root / "hand_landmarker.task")),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.frame_index = 0
        self.frames = deque(maxlen=SEQUENCE_FRAMES)
        self.latest_dynamic = ("other", 0.0, -1)

    @property
    def dynamic_available(self) -> bool:
        return self.dynamic_model is not None

    def _static_predict(self, frame_bgr, points):
        if points is None:
            return ("nothing", 1.0) if "nothing" in self.classes else (None, 0.0)
        if self.static_mode == "maps":
            points_normalized = points / np.array([frame_bgr.shape[1], frame_bgr.shape[0]], dtype=np.float32)
            image = render_landmark_map(points_normalized)
        else:
            image = crop_hand(frame_bgr, points)
        if image is None:
            return None, 0.0
        probs = self.static_model(np.expand_dims(image.astype(np.float32), 0), training=False).numpy()[0]
        idx = int(np.argmax(probs))
        return self.classes[idx], float(probs[idx])

    @staticmethod
    def _fixed_crops(frames, points_list):
        all_points = np.concatenate(points_list, axis=0)
        x_min, y_min = all_points.min(axis=0)
        x_max, y_max = all_points.max(axis=0)
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        side = max(float(max(x_max - x_min, y_max - y_min) * 1.7), 32.0)
        x0, y0 = int(np.floor(cx - side / 2)), int(np.floor(cy - side / 2))
        side = int(np.ceil(side))
        crops = []
        for frame in frames:
            h, w = frame.shape[:2]
            x1, y1 = x0 + side, y0 + side
            left, top = max(0, -x0), max(0, -y0)
            right, bottom = max(0, x1 - w), max(0, y1 - h)
            padded = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            crop = padded[y0 + top:y1 + top, x0 + left:x1 + left]
            crops.append(cv2.resize(crop, (FLOW_SIZE, FLOW_SIZE), interpolation=cv2.INTER_AREA))
        return crops

    def _predict_dynamic(self):
        if not self.dynamic_available or len(self.frames) < SEQUENCE_FRAMES:
            return
        window = list(self.frames)
        frames = [item[0] for item in window]
        points = [item[1] for item in window]
        if any(item is None for item in points):
            self.latest_dynamic = ("other", 0.0, self.frame_index)
            return
        gray = [cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                for crop in self._fixed_crops(frames, points)]
        flow = []
        for previous, current in zip(gray[:-1], gray[1:]):
            field = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow.append(np.clip(field / FLOW_CLIP_PIXELS, -1.0, 1.0).astype(np.float32))
        probs = self.dynamic_model(np.expand_dims(np.stack(flow), 0), training=False).numpy()[0]
        idx = int(np.argmax(probs))
        self.latest_dynamic = (self.dynamic_classes[idx], float(probs[idx]), self.frame_index)

    def predict(self, frame_bgr):
        self.frame_index += 1
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        timestamp_ms = int(self.frame_index * 1000 / 30)
        result = self.detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
        points = None
        display = frame_bgr.copy()
        if result.hand_landmarks:
            points = np.asarray([(p.x * frame_bgr.shape[1], p.y * frame_bgr.shape[0])
                                 for p in result.hand_landmarks[0]], dtype=np.float32)
            for start, end in HAND_EDGES:
                cv2.line(display, tuple(points[start].astype(int)), tuple(points[end].astype(int)), (40, 220, 110), 2)
            for point in points:
                cv2.circle(display, tuple(point.astype(int)), 3, (20, 170, 255), -1)
        static_label, static_score = self._static_predict(frame_bgr, points)
        if self.frame_index % FRAME_SAMPLE_EVERY == 0:
            self.frames.append((frame_bgr.copy(), points))
        if self.dynamic_available and len(self.frames) == SEQUENCE_FRAMES and self.frame_index % 5 == 0:
            self._predict_dynamic()
        dynamic_label, dynamic_score, dynamic_frame = self.latest_dynamic
        if self.frame_index - dynamic_frame > 20:
            dynamic_label, dynamic_score = "other", 0.0
        if dynamic_label in {"J", "Z"} and dynamic_score >= self.dynamic_confidence_threshold:
            return display, dynamic_label, dynamic_score, "video", points is not None
        return display, static_label, static_score, self.static_mode, points is not None

    def close(self):
        self.detector.close()
