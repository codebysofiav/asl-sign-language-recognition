"""Reproducible Keras training utilities for the ASL baseline."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import tensorflow as tf


def set_random_seeds(seed: int = 42, *, deterministic_ops: bool = True) -> None:
    """Set Python, NumPy, and TensorFlow seeds for repeatable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    if deterministic_ops:
        tf.config.experimental.enable_op_determinism()


def train_model(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    *,
    epochs: int = 20,
    patience: int = 5,
    checkpoint_path: str | Path = "models/baseline_best.keras",
    history_path: str | Path = "results/metrics/baseline_history.csv",
    monitor: str = "val_loss",
    verbose: int = 1,
) -> tf.keras.callbacks.History:
    """Train on train/validation only, save the best model, and log each epoch.

    The test partition is deliberately not accepted by this function to keep
    it out of model fitting and checkpoint selection.
    """
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if patience < 0:
        raise ValueError("patience cannot be negative.")
    if not monitor.startswith("val_"):
        raise ValueError("monitor must be a validation metric, such as 'val_loss'.")

    checkpoint = Path(checkpoint_path)
    history_file = Path(history_path)
    if checkpoint.suffix != ".keras":
        raise ValueError("checkpoint_path must use the .keras model format.")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history_file.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint),
            monitor=monitor,
            mode="min" if monitor.endswith("loss") else "max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="min" if monitor.endswith("loss") else "max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(history_file), append=False),
    ]

    return model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
    )
