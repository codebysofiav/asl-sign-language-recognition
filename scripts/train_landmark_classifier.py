"""Entrena y evalúa un MLP pequeño con los puntos extraídos de MediaPipe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import make_splits  # noqa: E402

NO_HAND_CLASS = "nothing"


def load_archive(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return (
            archive["features"].astype(np.float32),
            archive["labels"].astype(str),
            archive["paths"].astype(str),
            archive["detected"].astype(bool),
        )


def predict_with_no_hand_rule(
    model: object,
    features: np.ndarray,
    labels: np.ndarray,
    detected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sin puntos detectados predice `nothing`; si hay puntos usa el MLP."""
    valid = detected & np.isfinite(features).all(axis=1)
    predictions = np.full(len(labels), NO_HAND_CLASS, dtype=str)
    confidence = np.full(len(labels), np.nan, dtype=np.float32)
    if valid.any():
        predictions[valid] = model.predict(features[valid])
        confidence[valid] = model.predict_proba(features[valid]).max(axis=1)
    return predictions, valid, confidence


def score_split(
    model: object,
    features: np.ndarray,
    labels: np.ndarray,
    detected: np.ndarray,
    class_names: list[str],
) -> dict[str, object]:
    predictions, valid, _ = predict_with_no_hand_rule(model, features, labels, detected)
    report = classification_report(
        labels,
        predictions,
        labels=class_names,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics: dict[str, object] = {
        "images": int(len(labels)),
        "detected": int(valid.sum()),
        "detection_rate": float(valid.mean()) if len(labels) else 0.0,
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(
            labels, predictions, labels=class_names, average="macro", zero_division=0
        )),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=class_names).tolist(),
    }
    if valid.any():
        detected_labels = model.classes_
        metrics["accuracy_when_hand_detected"] = float(
            accuracy_score(labels[valid], predictions[valid])
        )
        metrics["macro_f1_when_hand_detected"] = float(
            f1_score(
                labels[valid],
                predictions[valid],
                labels=detected_labels,
                average="macro",
                zero_division=0,
            )
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "data" / "processed" / "train_landmarks.npz",
        help="NPZ creado por extract_hand_landmarks.py sobre el dataset de entrenamiento.",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "baseline.yaml")
    parser.add_argument(
        "--external-features",
        type=Path,
        default=None,
        help="NPZ opcional de data/mis_fotos; se evalúa, nunca se usa para entrenar.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=ROOT / "experiments" / "landmark_mlp.joblib",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=ROOT / "results" / "metrics" / "landmark_classifier.json",
    )
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    raw_dir = Path(data_cfg["raw_dir"])
    if not raw_dir.is_absolute():
        raw_dir = ROOT / raw_dir

    features_path = args.features if args.features.is_absolute() else ROOT / args.features
    x_all, y_all, paths, detected = load_archive(features_path)
    if len(x_all) == 0:
        raise ValueError("El archivo de características está vacío.")

    split_df = make_splits(
        raw_dir,
        data_cfg["split"],
        data_cfg["seed"],
        cache_dir=ROOT / "data" / "processed",
    )
    split_by_path = split_df.set_index("path")["split"].to_dict()
    labels_by_path = split_df.set_index("path")["label"].to_dict()
    split_names = np.asarray([split_by_path.get(path, "missing") for path in paths])
    expected_labels = np.asarray([labels_by_path.get(path, "") for path in paths])
    if np.any(split_names == "missing"):
        missing = int(np.sum(split_names == "missing"))
        raise ValueError(
            f"{missing} rutas del NPZ no están en la partición del dataset. "
            "Extrae las características usando la carpeta raw_dir de baseline.yaml."
        )
    if np.any(y_all != expected_labels):
        raise ValueError("Las etiquetas del NPZ no coinciden con las clases del dataset.")

    class_names = sorted(split_df.label.unique().tolist())
    if NO_HAND_CLASS not in class_names:
        raise ValueError(f"El dataset debe contener la clase especial `{NO_HAND_CLASS}`.")
    valid = detected & np.isfinite(x_all).all(axis=1)
    split_counts: dict[str, dict[str, int]] = {}
    for split_name in ("train", "val", "test"):
        in_split = split_names == split_name
        split_counts[split_name] = {
            "images": int(in_split.sum()),
            "detected": int((in_split & valid).sum()),
        }

    train_mask = valid & (split_names == "train")
    val_mask = split_names == "val"
    test_mask = split_names == "test"
    if len(np.unique(y_all[train_mask])) < 2:
        raise ValueError("El conjunto de entrenamiento válido necesita al menos dos clases.")
    if not val_mask.any() or not test_mask.any():
        raise ValueError(
            "Faltan imágenes de validación o test. "
            "Extrae todo el dataset sin --max-per-class antes de entrenar."
        )

    classifier = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=100,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=int(data_cfg["seed"]),
            verbose=True,
        ),
    )
    classifier.fit(x_all[train_mask], y_all[train_mask])

    metrics: dict[str, object] = {
        "model": "StandardScaler + MLPClassifier(128, 64)",
        "features_file": str(features_path),
        "seed": int(data_cfg["seed"]),
        "feature_dim": int(x_all.shape[1]),
        "detection_counts": split_counts,
        "validation": score_split(
            classifier,
            x_all[val_mask],
            y_all[val_mask],
            valid[val_mask],
            class_names,
        ),
        "test": score_split(
            classifier,
            x_all[test_mask],
            y_all[test_mask],
            valid[test_mask],
            class_names,
        ),
    }

    external_path = args.external_features
    if external_path is not None:
        if not external_path.is_absolute():
            external_path = ROOT / external_path
        x_ext, y_ext, paths_ext, detected_ext = load_archive(external_path)
        external_valid = detected_ext & np.isfinite(x_ext).all(axis=1)
        unknown_labels = sorted(set(y_ext) - set(class_names))
        if unknown_labels:
            raise ValueError(f"Clases externas que no existen en entrenamiento: {unknown_labels}")

        external_metrics = score_split(
            classifier, x_ext, y_ext, external_valid, class_names
        )
        predicted, external_valid, confidence = predict_with_no_hand_rule(
            classifier, x_ext, y_ext, detected_ext
        )
        details = pd.DataFrame(
            {
                "path": paths_ext,
                "label": y_ext,
                "prediction": predicted,
                "method": np.where(external_valid, "landmark_mlp", "no_hand_as_nothing"),
                "confidence": confidence,
            }
        )
        metrics_output_arg = (
            args.metrics_output
            if args.metrics_output.is_absolute()
            else ROOT / args.metrics_output
        )
        prediction_path = metrics_output_arg.with_name("landmark_external_predictions.csv")
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        details.to_csv(prediction_path, index=False)
        external_metrics["predictions_file"] = str(prediction_path)
        metrics["external_photos"] = external_metrics

    model_output = args.model_output if args.model_output.is_absolute() else ROOT / args.model_output
    metrics_output = (
        args.metrics_output if args.metrics_output.is_absolute() else ROOT / args.metrics_output
    )
    model_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "classifier": classifier,
            "no_hand_class": NO_HAND_CLASS,
            "feature_dim": int(x_all.shape[1]),
        },
        model_output,
    )
    metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Modelo guardado en: {model_output}")
    print(f"Métricas guardadas en: {metrics_output}")
    print(f"Accuracy test interno (fin a fin): {metrics['test']['accuracy']:.4f}")
    if "external_photos" in metrics:
        external = metrics["external_photos"]
        print(
            "Accuracy fotos propias (fin a fin): "
            f"{external['accuracy']:.4f}; "
            f"detección de mano: {external['detection_rate']:.1%} "
            f"({external['detected']}/{external['images']})"
        )


if __name__ == "__main__":
    main()
