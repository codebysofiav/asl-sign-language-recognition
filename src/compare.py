"""Comparación de corridas guardadas en experiments/."""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def latest_run(name, experiments_dir="experiments"):
    """Última corrida terminada (con final/model.keras) llamada `name`."""
    pat = re.compile(rf"^\d{{8}}-\d{{4}}_{re.escape(name)}$")
    runs = [
        p for p in sorted(Path(experiments_dir).iterdir())
        if pat.match(p.name) and (p / "final" / "model.keras").exists()
    ]
    if not runs:
        raise FileNotFoundError(f"No hay una corrida terminada llamada '{name}' en {experiments_dir}/")
    return runs[-1]


def summarize_run(run_dir):
    """Una fila de resumen: parámetros, épocas, mejores métricas y tiempo por época."""
    run_dir = Path(run_dir)
    hist = pd.read_csv(run_dir / "history.csv")
    metrics = json.loads((run_dir / "metrics.json").read_text())
    info_file = run_dir / "model_info.json"
    info = json.loads(info_file.read_text()) if info_file.exists() else {}
    if "params" not in info:  # corridas anteriores a model_info.json
        from tensorflow import keras

        info["params"] = keras.models.load_model(run_dir / "final" / "model.keras").count_params()
    return {
        "corrida": run_dir.name,
        "parámetros (M)": round(info["params"] / 1e6, 2),
        "épocas": len(hist),
        "mejor val_accuracy": round(hist.val_accuracy.max(), 4),
        "mínimo val_loss": round(hist.val_loss.min(), 4),
        "test accuracy": round(metrics["accuracy"], 4),
        "s/época (mediana)": round(hist.time_s.median(), 1) if "time_s" in hist else None,
    }


def summary_table(runs):
    """`runs`: dict {etiqueta: run_dir}. Devuelve DataFrame con una fila por modelo."""
    return pd.DataFrame.from_dict({label: summarize_run(rd) for label, rd in runs.items()}, orient="index")


def plot_comparison(runs):
    """val_accuracy y val_loss por época; línea punteada = inicio del fine-tuning."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, (label, rd) in enumerate(runs.items()):
        h = pd.read_csv(Path(rd) / "history.csv")
        color = f"C{i}"
        axes[0].plot(h.epoch, h.val_accuracy, label=label, color=color)
        axes[1].plot(h.epoch, h.val_loss, label=label, color=color)
        cfg = yaml.safe_load((Path(rd) / "config.yaml").read_text())
        if "epochs_head" in cfg["train"]:
            for ax in axes:
                ax.axvline(cfg["train"]["epochs_head"] + 0.5, color=color, ls=":", alpha=0.7)
    axes[0].set_title("val_accuracy")
    axes[1].set_title("val_loss")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("Época")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    plt.show()


def evaluate_on_photos(runs, photos_df):
    """Accuracy de cada modelo sobre fotos propias. Devuelve (tabla, {etiqueta: resultados})."""
    from tensorflow import keras

    from src.visualize import predict_own_photos

    rows, details = {}, {}
    for label, rd in runs.items():
        rd = Path(rd)
        cfg = yaml.safe_load((rd / "config.yaml").read_text())
        class_names = json.loads((rd / "final" / "classes.json").read_text())
        model = keras.models.load_model(rd / "final" / "model.keras")
        res, _, _ = predict_own_photos(model, photos_df, class_names, tuple(cfg["data"]["img_size"]))
        rows[label] = {
            "fotos": len(res),
            "aciertos": int(res.correct.sum()),
            "accuracy fotos propias": round(res.correct.mean(), 4),
        }
        details[label] = res
    return pd.DataFrame.from_dict(rows, orient="index"), details
