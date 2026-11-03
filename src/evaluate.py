"""Evaluación y gráficas."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, ds, class_names, out_dir):
    """Evalúa en ds; guarda metrics.json y confusion_matrix.png en out_dir."""
    out_dir = Path(out_dir)
    y_true, y_pred = [], []
    for x, y in ds:
        y_pred.append(np.argmax(model.predict_on_batch(x), axis=1))
        y_true.append(y.numpy())
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)

    labels = list(range(len(class_names)))
    metrics = {
        "accuracy": float((y_true == y_pred).mean()),
        "report": classification_report(
            y_true, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0
        ),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(labels, class_names, rotation=90)
    ax.set_yticks(labels, class_names)
    ax.set_xlabel("Predicha")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión (test, normalizada)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)
    return metrics


def evaluate_hand_pipeline(model, rows, image_dir, manifest_path, class_names, img_size, batch_size, out_dir):
    """Evaluate crop + classifier end to end, counting missed signs as `nothing`."""
    from src.data import make_dataset

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    detection = dict(zip(manifest.path.astype(str), manifest.detected.astype(int).astype(bool)))
    rows = rows.copy().reset_index(drop=True)
    missing = sorted(set(rows.path.astype(str)) - set(detection))
    if missing:
        raise ValueError(f"El manifiesto no contiene {len(missing)} rutas; ejemplo: {missing[0]}")
    if "nothing" not in {name.lower() for name in class_names}:
        raise ValueError("La evaluación del flujo requiere una clase `nothing` para representar ausencia de mano.")
    no_hand_index = next(i for i, name in enumerate(class_names) if name.lower() == "nothing")
    indices = {name: i for i, name in enumerate(class_names)}
    ds = make_dataset(rows, image_dir, class_names, tuple(img_size), batch_size)
    probabilities = np.concatenate([model.predict_on_batch(x) for x, _ in ds])
    y_true = rows.label.map(indices).to_numpy()
    detected = np.asarray([detection[str(path)] for path in rows.path], dtype=bool)
    y_pred = probabilities.argmax(axis=1)
    y_pred[~detected] = no_hand_index
    confidence = probabilities.max(axis=1)
    confidence[~detected] = 1.0  # regla determinista de ausencia de mano
    metrics = {
        "accuracy": float((y_true == y_pred).mean()),
        "hand_detection_rate_on_signs": float(detected[rows.label.str.lower() != "nothing"].mean()),
        "missed_hand_count": int((~detected & (rows.label.str.lower() != "nothing").to_numpy()).sum()),
        "report": classification_report(
            y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names,
            output_dict=True, zero_division=0,
        ),
    }
    output = rows.copy()
    output["detected"] = detected
    output["pred"] = [class_names[index] for index in y_pred]
    output["confidence"] = confidence
    output["correct"] = y_true == y_pred
    output.to_csv(out_dir / "pipeline_predictions.csv", index=False)
    (out_dir / "pipeline_metrics.json").write_text(json.dumps(metrics, indent=2))

    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(labels, class_names, rotation=90)
    ax.set_yticks(labels, class_names)
    ax.set_xlabel("Predicha")
    ax.set_ylabel("Real")
    ax.set_title("Flujo completo: recorte MediaPipe + CNN")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_dir / "pipeline_confusion_matrix.png", dpi=120)
    plt.close(fig)
    return metrics


def plot_curves(history_csv, out_png):
    """Curvas de pérdida y accuracy desde history.csv."""
    h = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in zip(axes, ("loss", "accuracy"), ("Pérdida", "Accuracy")):
        ax.plot(h.epoch, h[key], label="train")
        ax.plot(h.epoch, h[f"val_{key}"], label="val")
        ax.set_title(title)
        ax.set_xlabel("Época")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
