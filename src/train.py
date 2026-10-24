"""Entrenamiento con checkpoints y reanudación automática.

Uso:
    python -m src.train --config configs/baseline.yaml --name baseline

Cada corrida vive en experiments/AAAAMMDD-HHMM_<name>/ con:
    last.keras   último checkpoint (se sobrescribe cada época; permite reanudar)
    best.keras   mejor modelo según `monitor` (se sobrescribe al mejorar)
    history.csv  métricas por época
    state.json   época completada, mejor valor y estado (running/done)
    config.yaml  configuración usada
    curves.png, metrics.json, confusion_matrix.png, final/model.keras  (al terminar)

Si se interrumpe, volver a lanzar el mismo comando reanuda la corrida inconclusa
más reciente con ese nombre. Usa --new para forzar una corrida nueva.
"""
import argparse
import csv
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import yaml
from tensorflow import keras

from src.data import build_datasets
from src.evaluate import evaluate_model, plot_curves
from src.model import build_baseline, compile_model

EXPERIMENTS_DIR = Path("experiments")


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _save_model(model, path):
    """Guarda en archivo temporal y reemplaza: un corte no deja checkpoints corruptos."""
    tmp = path.with_name("tmp_" + path.name)
    model.save(tmp)
    os.replace(tmp, path)


def _setup_gpu():
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)  # comparte VRAM con Windows
    print(f"GPUs detectadas: {gpus or 'ninguna (se usará CPU)'}")


def find_run(name, resume=True):
    """Devuelve (run_dir, reanudando). Reanuda la última corrida inconclusa con ese nombre."""
    pattern = re.compile(rf"^\d{{8}}-\d{{4}}_{re.escape(name)}$")
    if resume and EXPERIMENTS_DIR.is_dir():
        runs = sorted(p for p in EXPERIMENTS_DIR.iterdir() if p.is_dir() and pattern.match(p.name))
        if runs:
            last = runs[-1]
            state_file = last / "state.json"
            if state_file.exists() and json.loads(state_file.read_text())["status"] != "done":
                return last, True
    run_dir = EXPERIMENTS_DIR / f"{datetime.now():%Y%m%d-%H%M}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, False


class CheckpointManager(keras.callbacks.Callback):
    """Al final de cada época: last.keras, best.keras, history.csv y state.json."""

    def __init__(self, run_dir, monitor, state):
        super().__init__()
        self.run_dir, self.monitor, self.state = run_dir, monitor, state
        self.mode_max = "loss" not in monitor

    def on_epoch_end(self, epoch, logs=None):
        logs = {k: float(v) for k, v in (logs or {}).items()}
        lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
        _save_model(self.model, self.run_dir / "last.keras")

        cur, best = logs.get(self.monitor), self.state["best"]
        if cur is not None and (best is None or (cur > best if self.mode_max else cur < best)):
            self.state["best"] = cur
            _save_model(self.model, self.run_dir / "best.keras")

        self.state["epoch"] = epoch + 1
        _write_json(self.run_dir / "state.json", self.state)

        row = {"epoch": epoch + 1, **logs, "lr": lr}
        hist = self.run_dir / "history.csv"
        new = not hist.exists()
        with open(hist, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)


def train(config_path, name, resume=True):
    cfg = load_config(config_path)
    t = cfg["train"]
    keras.utils.set_random_seed(cfg["data"]["seed"])
    _setup_gpu()
    if t.get("mixed_precision"):
        keras.mixed_precision.set_global_policy("mixed_float16")

    datasets, class_names, _ = build_datasets(cfg)
    run_dir, resuming = find_run(name, resume)
    shutil.copy(config_path, run_dir / "config.yaml")
    print(f"Corrida: {run_dir} ({'reanudando' if resuming else 'nueva'})")

    if resuming:
        model = keras.models.load_model(run_dir / "last.keras")
        state = json.loads((run_dir / "state.json").read_text())
    else:
        img_size = tuple(cfg["data"]["img_size"])
        model = compile_model(build_baseline(len(class_names), img_size, cfg), t["learning_rate"])
        state = {"epoch": 0, "best": None, "status": "running"}
        _write_json(run_dir / "state.json", state)

    callbacks = [
        CheckpointManager(run_dir, t["monitor"], state),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=t["reduce_lr_patience"], min_lr=1e-6),
        keras.callbacks.EarlyStopping(monitor=t["monitor"], patience=t["early_stopping_patience"]),
    ]
    model.fit(
        datasets["train"],
        validation_data=datasets["val"],
        epochs=t["epochs"],
        initial_epoch=state["epoch"],
        callbacks=callbacks,
        shuffle=False,  # el barajado ya lo hace tf.data
        verbose=2,
    )

    # Cierre: modelo final = mejor checkpoint; evaluación en test; curvas.
    best = keras.models.load_model(run_dir / "best.keras")
    (run_dir / "final").mkdir(exist_ok=True)
    best.save(run_dir / "final" / "model.keras")
    (run_dir / "final" / "classes.json").write_text(json.dumps(class_names))
    metrics = evaluate_model(best, datasets["test"], class_names, run_dir)
    plot_curves(run_dir / "history.csv", run_dir / "curves.png")
    state["status"] = "done"
    _write_json(run_dir / "state.json", state)
    print(f"Test accuracy: {metrics['accuracy']:.4f} | Resultados en {run_dir}")
    return run_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True, help="nombre del experimento")
    ap.add_argument("--new", action="store_true", help="no reanudar; crear corrida nueva")
    args = ap.parse_args()
    train(args.config, args.name, resume=not args.new)
