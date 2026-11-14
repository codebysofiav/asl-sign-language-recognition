"""Fine-tuning corto de un modelo con parte de las fotos propias, evaluado por sesión retenida.

Idea: se entrena con las fotos de todas las sesiones menos una (repetidas varias veces y mezcladas
con imágenes del dataset para que no olvide) y se evalúa en la sesión que quedó fuera. Se repite
dejando fuera cada sesión (leave-one-session-out). No se elige nada con las fotos retenidas:
número de épocas y learning rate se fijan de antemano.
"""
from pathlib import Path

import pandas as pd
from tensorflow import keras

from src.data import make_dataset, make_splits, subsample
from src.model import compile_model
from src.visualize import predict_own_photos


def replay_sample(cfg, per_class=100):
    """Muestra del conjunto de entrenamiento del dataset (rutas absolutas) para no olvidar."""
    d = cfg["data"]
    splits = make_splits(d["raw_dir"], d["split"], d["seed"])
    df = subsample(splits[splits.split == "train"], per_class, d["seed"]).copy()
    df["path"] = [str(Path(d["raw_dir"]) / p) for p in df.path]
    return df[["path", "label"]]


def finetune_fold(model_path, train_df, replay_df, class_names, img_size, epochs=6, lr=3e-5,
                  oversample=5, batch_size=32, seed=0):
    """Carga el modelo, lo ajusta con train_df (x oversample) + replay_df y lo devuelve."""
    keras.utils.set_random_seed(seed)
    model = keras.models.load_model(model_path, compile=False)  # el optimizador se crea de nuevo
    compile_model(model, lr)
    own = pd.concat([train_df[["path", "label"]]] * oversample, ignore_index=True)
    mix = pd.concat([own, replay_df[["path", "label"]]], ignore_index=True)
    ds = make_dataset(mix, ".", class_names, img_size, batch_size, training=True, seed=seed)
    model.fit(ds, epochs=epochs, shuffle=False, verbose=0)
    return model


def leave_one_session_out(model_path, photos, replay_df, class_names, img_size, **kw):
    """Tabla por sesión retenida: accuracy antes y después del fine-tuning, y la media ponderada."""
    rows = []
    for ses in sorted(photos.sesion.unique()):
        test = photos[photos.sesion == ses]
        train = photos[photos.sesion != ses]
        base = keras.models.load_model(model_path, compile=False)
        antes = predict_own_photos(base, test, class_names, img_size)[0].correct.mean()
        model = finetune_fold(model_path, train, replay_df, class_names, img_size, **kw)
        despues = predict_own_photos(model, test, class_names, img_size)[0].correct.mean()
        rows.append({"sesión retenida": ses, "fotos retenidas": len(test), "fotos de entrenamiento": len(train),
                     "antes": round(antes, 4), "después": round(despues, 4)})
        print(f"  {ses}: {antes:.1%} → {despues:.1%} ({len(test)} fotos retenidas)", flush=True)
        keras.backend.clear_session()
    tabla = pd.DataFrame(rows)
    n = tabla["fotos retenidas"]
    media = {"sesión retenida": "media ponderada", "fotos retenidas": int(n.sum()), "fotos de entrenamiento": "",
             "antes": round((tabla.antes * n).sum() / n.sum(), 4), "después": round((tabla["después"] * n).sum() / n.sum(), 4)}
    return pd.concat([tabla, pd.DataFrame([media])], ignore_index=True)
