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
VIDEO_MAP_SIZE = 96
# Initial-pose candidate and event-gating heuristics. Landmark distances are
# normalized by palm size; tune with camera recordings before treating as final.
POSE_CANDIDATE_STABLE_SAMPLES = 2
CANDIDATE_TIP_MOTION_THRESHOLD = 0.035
MOTION_ONSET_SAMPLES = 2
MOTION_RELEASE_SAMPLES = 2
POSE_CANDIDATE_TIMEOUT_SAMPLES = 12
MOTION_PREDICTION_MAX_AGE = 10
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
VIDEO_MAP_POINT_COLORS = [FINGER_COLORS[0]] * 4 + [FINGER_COLORS[1]] * 4 + [FINGER_COLORS[2]] * 4 + [FINGER_COLORS[3]] * 4 + [FINGER_COLORS[4]] * 4


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


def render_video_map_sequence(points_sequence: np.ndarray) -> np.ndarray:
    """Render 17 clip-normalized 96x96 maps exactly as Notebook 14 does."""
    seq = np.asarray(points_sequence, dtype=np.float32)
    if seq.shape != (SEQUENCE_FRAMES, 21, 3) or not np.isfinite(seq).all():
        raise ValueError(f"Secuencia de landmarks inválida: {seq.shape}")
    center = seq[:, 0, :2].mean(axis=0)
    palm = np.linalg.norm(seq[:, 9, :2] - seq[:, 0, :2], axis=1)
    scale = max(float(np.median(palm)), 1e-5) * 3.0
    xy = np.clip((seq[:, :, :2] - center[None, None, :]) / scale + 0.5, 0.0, 1.0)
    xy = np.rint(xy * (VIDEO_MAP_SIZE - 1)).astype(np.int32)
    maps = []
    for frame_points in xy:
        image = np.zeros((VIDEO_MAP_SIZE, VIDEO_MAP_SIZE, 3), dtype=np.uint8)
        for edges, color in zip(FINGER_EDGES, FINGER_COLORS):
            for start, end in edges:
                cv2.line(image, tuple(frame_points[start]), tuple(frame_points[end]), color, 2, cv2.LINE_AA)
        for start, end in PALM_EDGES:
            cv2.line(image, tuple(frame_points[start]), tuple(frame_points[end]), (235, 235, 235), 2, cv2.LINE_AA)
        for idx, (x, y) in enumerate(frame_points):
            color = (255, 255, 255) if idx == 0 else VIDEO_MAP_POINT_COLORS[idx - 1]
            cv2.circle(image, (int(x), int(y)), 2, color, -1, cv2.LINE_AA)
        maps.append(image)
    return np.stack(maps).astype(np.uint8)


class RecognitionEngine:
    """Run static recognition and route dynamic gestures by mode or initial pose."""

    def __init__(self, artifact_root: str | Path, static_mode: str = "maps", routing_mode: str = "auto", confidence: float | None = None):
        self.artifact_root = Path(artifact_root)
        self.static_mode = static_mode
        if routing_mode not in {"auto", "static", "dynamic"}:
            raise ValueError("routing_mode debe ser auto, static o dynamic")
        self.routing_mode = routing_mode
        static_dir = self.artifact_root / static_mode
        required = [static_dir / "model.keras", static_dir / "classes.json", self.artifact_root / "hand_landmarker.task"]
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(f"Falta el artefacto requerido: {path}")
        self.static_model = tf.keras.models.load_model(required[0], compile=False)
        self.classes = json.loads(required[1].read_text(encoding="utf-8"))
        self.dynamic_map_path = self.artifact_root / "video_map" / "jz_landmark_map_gru.keras"
        self.dynamic_flow_path = self.artifact_root / "video" / "jz_flow_cnn.keras"
        self.dynamic_type = "maps" if self.dynamic_map_path.is_file() else "flow"
        self.dynamic_path = self.dynamic_map_path if self.dynamic_type == "maps" else self.dynamic_flow_path
        self.dynamic_model = (
            tf.keras.models.load_model(self.dynamic_path, compile=False)
            if self.dynamic_path.is_file() else None
        )
        dynamic_classes_path = (self.artifact_path("video_map", "classes.json") if self.dynamic_type == "maps"
                                else self.artifact_path("video", "classes.json"))
        self.dynamic_classes = (
            json.loads(dynamic_classes_path.read_text(encoding="utf-8"))
            if dynamic_classes_path.is_file() else ["J", "Z", "other"]
        )
        if self.dynamic_model is not None and self.dynamic_model.output_shape[-1] != len(self.dynamic_classes):
            raise ValueError(f"Las clases en video{ '_map' if self.dynamic_type == 'maps' else ''}/classes.json no coinciden con el modelo temporal.")
        self.dynamic_confidence_threshold = confidence if confidence is not None else (0.50 if self.dynamic_type == "maps" else 0.80)
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
        self.previous_motion_points = None
        self.motion_active = False
        self.motion_high_count = 0
        self.motion_low_count = 0
        self.motion_start_frame = -1
        self.pose_candidate = None
        self.pose_candidate_count = 0
        self.pending_candidate = None
        self.pending_candidate_samples = 0
        self.pending_tip_index = None
        self.previous_candidate_tip = None

    def artifact_path(self, folder, name):
        return self.artifact_root / folder / name

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
        if self.dynamic_type == "maps":
            valid_indices = [i for i, hand_points in enumerate(points) if hand_points is not None]
            if len(valid_indices) < 12:
                self.latest_dynamic = ("other", 0.0, self.frame_index)
                return
            valid_points = []
            for i in valid_indices:
                frame, hand_points = frames[i], points[i]
                h, w = frame.shape[:2]
                denom = float(max(h, w))
                valid_points.append(np.column_stack((hand_points[:, 0] / denom, hand_points[:, 1] / denom,
                                                     np.zeros(21, dtype=np.float32))))
            valid_points = np.stack(valid_points)
            # Match Notebook 14: interpolate missing detections independently for
            # every landmark coordinate across the 17 sampled frames.
            points_sequence = np.empty((SEQUENCE_FRAMES, 21, 3), dtype=np.float32)
            sample_positions = np.asarray(valid_indices, dtype=np.float32)
            for point_idx in range(21):
                for axis in range(3):
                    points_sequence[:, point_idx, axis] = np.interp(
                        np.arange(SEQUENCE_FRAMES), sample_positions, valid_points[:, point_idx, axis]
                    )
            maps = render_video_map_sequence(points_sequence)
            model_input = maps[None, ...]
        else:
            if any(item is None for item in points):
                self.latest_dynamic = ("other", 0.0, self.frame_index)
                return
            gray = [cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    for crop in self._fixed_crops(frames, points)]
            flow = []
            for previous, current in zip(gray[:-1], gray[1:]):
                field = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow.append(np.clip(field / FLOW_CLIP_PIXELS, -1.0, 1.0).astype(np.float32))
            model_input = np.expand_dims(np.stack(flow), 0)
        probs = self.dynamic_model(model_input, training=False).numpy()[0]
        idx = int(np.argmax(probs))
        self.latest_dynamic = (self.dynamic_classes[idx], float(probs[idx]), self.frame_index)

    @staticmethod
    def _initial_pose_candidate(points):
        """Return a J-like pinky or Z-like index initial handshape candidate."""
        wrist = points[0]

        def extension_ratio(tip_index, joint_index):
            joint_distance = float(np.linalg.norm(points[joint_index] - wrist))
            tip_distance = float(np.linalg.norm(points[tip_index] - wrist))
            return tip_distance / max(joint_distance, 1e-6)

        index = extension_ratio(8, 6)
        middle = extension_ratio(12, 10)
        ring = extension_ratio(16, 14)
        pinky = extension_ratio(20, 18)
        thumb = extension_ratio(4, 2)
        other_fingers = [middle, ring]

        # The J starting shape resembles I (pinky extended); the Z starting
        # shape uses the index finger. A thumb-out Y should not arm J.
        if pinky > 1.12 and pinky > max(index, *other_fingers) + 0.06 and thumb < 1.18:
            return "J"
        if index > 1.12 and index > max(pinky, *other_fingers) + 0.06 and thumb < 1.18:
            return "Z"
        return None

    def _clear_pending_candidate(self):
        self.pending_candidate = None
        self.pending_candidate_samples = 0
        self.pending_tip_index = None
        self.previous_candidate_tip = None
        self.motion_high_count = 0
        self.motion_low_count = 0

    def _update_motion_gate(self, points):
        """Arm on a J/Z-like initial pose; run video after its finger moves."""
        if points is None:
            self.previous_motion_points = None
            if self.motion_active:
                self.motion_low_count += 1
                if self.motion_low_count >= MOTION_RELEASE_SAMPLES:
                    self.motion_active = False
                    self._clear_pending_candidate()
            else:
                self.pose_candidate = None
                self.pose_candidate_count = 0
                self._clear_pending_candidate()
            return

        previous = self.previous_motion_points
        self.previous_motion_points = points.copy()
        candidate = self._initial_pose_candidate(points)
        if candidate == self.pose_candidate:
            self.pose_candidate_count += 1
        else:
            self.pose_candidate = candidate
            self.pose_candidate_count = 1 if candidate else 0

        if not self.motion_active and self.pending_candidate is None:
            if self.pose_candidate_count >= POSE_CANDIDATE_STABLE_SAMPLES:
                self.pending_candidate = candidate
                self.pending_candidate_samples = 0
                self.pending_tip_index = 20 if candidate == "J" else 8
                self.previous_candidate_tip = points[self.pending_tip_index].copy()
            return

        if self.pending_candidate is None:
            return

        self.pending_candidate_samples += 1
        if not self.motion_active and self.pending_candidate_samples > POSE_CANDIDATE_TIMEOUT_SAMPLES:
            # Static I/D-like poses time out to the static recognizer.
            self._clear_pending_candidate()
            return

        if previous is None:
            return
        palm_scale = float(np.linalg.norm(points[9] - points[0]))
        if palm_scale < 1e-6:
            return
        tip = points[self.pending_tip_index]
        tip_motion = float(np.linalg.norm(tip - self.previous_candidate_tip) / palm_scale)
        self.previous_candidate_tip = tip.copy()

        if tip_motion >= CANDIDATE_TIP_MOTION_THRESHOLD:
            self.motion_high_count += 1
            self.motion_low_count = 0
            if not self.motion_active and self.motion_high_count >= MOTION_ONSET_SAMPLES:
                self.motion_active = True
                self.motion_start_frame = self.frame_index
        else:
            self.motion_high_count = 0
            if self.motion_active:
                self.motion_low_count += 1
                if self.motion_low_count >= MOTION_RELEASE_SAMPLES:
                    self.motion_active = False
                    self._clear_pending_candidate()
            elif self.pending_candidate_samples > POSE_CANDIDATE_TIMEOUT_SAMPLES:
                self._clear_pending_candidate()

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
        static_label, static_score = (None, 0.0)
        if self.routing_mode != "dynamic":
            static_label, static_score = self._static_predict(frame_bgr, points)
        if self.frame_index % FRAME_SAMPLE_EVERY == 0:
            self.frames.append((frame_bgr.copy(), points))
            if self.routing_mode == "auto":
                self._update_motion_gate(points)
        should_evaluate_dynamic = self.routing_mode == "dynamic" or self.motion_active
        if (self.dynamic_available and should_evaluate_dynamic and len(self.frames) == SEQUENCE_FRAMES
                and self.frame_index % 5 == 0):
            self._predict_dynamic()
        dynamic_label, dynamic_score, dynamic_frame = self.latest_dynamic
        if self.frame_index - dynamic_frame > MOTION_PREDICTION_MAX_AGE:
            dynamic_label, dynamic_score = "other", 0.0
        recent_dynamic_is_current_event = self.routing_mode == "dynamic" or dynamic_frame >= self.motion_start_frame
        dynamic_is_active = self.routing_mode == "dynamic" or self.motion_active
        if (dynamic_is_active and dynamic_label in {"J", "Z"}
                and dynamic_score >= self.dynamic_confidence_threshold
                and recent_dynamic_is_current_event):
            return display, dynamic_label, dynamic_score, f"video-{self.dynamic_type}", points is not None
        if self.routing_mode == "dynamic" and self.dynamic_available:
            # In explicit temporal mode, low-confidence output is a reject response.
            return display, "other", dynamic_score, f"video-{self.dynamic_type}", points is not None
        if self.dynamic_available and self.motion_active:
            # Do not let the static model add an accidental letter while a
            # moving gesture is being buffered/classified.
            return display, "other", dynamic_score, f"video-{self.dynamic_type}", points is not None
        # Outside a motion event, always return to the static classifier. Do not
        # keep showing the previous J/Z result after the hand has settled.
        return display, static_label, static_score, self.static_mode, points is not None

    def close(self):
        self.detector.close()
