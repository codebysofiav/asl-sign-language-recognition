"""Evaluation helpers for saved image classifiers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support


def evaluate_classifier(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    class_names: list[str],
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    """Evaluate a classifier and return summary, per-class, and confusion metrics.

    The dataset must be the held-out test partition and must not be shuffled.
    The model is called with ``training=False`` so dropout and normalization
    layers use inference behavior.
    """
    if not class_names:
        raise ValueError("class_names cannot be empty.")

    true_batches: list[np.ndarray] = []
    predicted_batches: list[np.ndarray] = []
    for images, labels in dataset:
        logits = model(images, training=False)
        true_batches.append(labels.numpy().astype("int64"))
        predicted_batches.append(tf.argmax(logits, axis=1).numpy().astype("int64"))

    if not true_batches:
        raise ValueError("Cannot evaluate on an empty dataset.")

    y_true = np.concatenate(true_batches)
    y_pred = np.concatenate(predicted_batches)
    label_ids = list(range(len(class_names)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_ids,
        average="macro",
        zero_division=0,
    )
    per_class = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    per_class_frame = pd.DataFrame(per_class).T.rename_axis("class")
    matrix = confusion_matrix(y_true, y_pred, labels=label_ids)
    summary = {
        "accuracy": float(np.mean(y_true == y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "samples": int(len(y_true)),
    }
    return summary, per_class_frame, matrix
