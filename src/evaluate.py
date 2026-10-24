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
