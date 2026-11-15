"""Run the trained static and temporal ASL models on a local webcam."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.live_pipeline import LiveSignPipeline, SpellingBuffer  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Índice de la cámara local (default: 0).")
    parser.add_argument(
        "--static-model", type=Path,
        default=ROOT / "data/processed/pipeline_mapas_25k/model/model.keras",
    )
    parser.add_argument(
        "--classes", type=Path,
        default=ROOT / "data/processed/pipeline_mapas_25k/model/classes.json",
    )
    parser.add_argument(
        "--hand-model", type=Path,
        default=ROOT / "data/processed/pipeline_mapas_25k/hand_landmarker.task",
    )
    parser.add_argument(
        "--temporal-model", type=Path,
        default=ROOT / "data/processed/video_flow/jz_flow_cnn.keras",
        help="Si el archivo aún no existe, se ejecuta solo el modelo estático.",
    )
    parser.add_argument("--confidence", type=float, default=0.80)
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = LiveSignPipeline(
        static_model_path=args.static_model,
        classes_path=args.classes,
        hand_model_path=args.hand_model,
        temporal_model_path=args.temporal_model,
        dynamic_threshold=args.confidence,
    )
    buffer = SpellingBuffer()
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        pipeline.close()
        raise RuntimeError(f"No pude abrir la cámara {args.camera}.")

    print("Teclas: S iniciar/detener captura | C borrar texto | Q salir")
    if pipeline.temporal_model is None:
        print("No se encontró modelo temporal: se inicia en modo solo estático.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("No se pudo leer un fotograma de la cámara.")
                break
            frame = cv2.flip(frame, 1)
            annotated, (label, confidence, source, hand_detected) = pipeline.process(frame)
            accepted = buffer.observe(label, confidence)

            mode = "GRABANDO" if buffer.active else "VISTA PREVIA (sin guardar)"
            color = (50, 220, 100) if buffer.active else (0, 190, 255)
            cv2.putText(annotated, mode, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            shown_label = label if label is not None else "sin predicción"
            cv2.putText(annotated, f"Predicción: {shown_label} ({confidence:.2f}) [{source}]",
                        (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
            cv2.putText(annotated, f"Texto: {buffer.text or '—'}", (15, annotated.shape[0] - 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if accepted:
                cv2.putText(annotated, f"Aceptada: {accepted!r}", (15, 94),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 150), 2)
            if not hand_detected:
                cv2.putText(annotated, "Mano no detectada", (15, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 190, 255), 2)
            cv2.imshow("ASL - prediccion en vivo", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                final_text = buffer.toggle()
                print(f"Captura detenida. Texto: {final_text}") if final_text is not None else print("Captura iniciada.")
            if key == ord("c"):
                buffer.clear()
                print("Texto borrado.")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        pipeline.close()


if __name__ == "__main__":
    main()
