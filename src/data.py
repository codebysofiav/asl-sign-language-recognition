"""Carga, partición estratificada y pipeline tf.data del dataset ASL Alphabet."""
from pathlib import Path

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

IMG_EXT = {".jpg", ".jpeg", ".png"}


def list_classes(raw_dir):
    """Clases = subcarpetas de raw_dir, en orden alfabético."""
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"No existe {raw_dir}. Descarga el dataset de Kaggle, ponlo en data/ y ejecuta ./setup.sh"
        )
    return sorted(p.name for p in raw_dir.iterdir() if p.is_dir())


def list_images(raw_dir):
    """DataFrame con columnas path (relativa a raw_dir) y label."""
    raw_dir = Path(raw_dir)
    rows = [
        (f"{cls}/{f.name}", cls)
        for cls in list_classes(raw_dir)
        for f in sorted((raw_dir / cls).iterdir())
        if f.suffix.lower() in IMG_EXT
    ]
    if not rows:
        raise FileNotFoundError(f"No se encontraron imágenes en {raw_dir}")
    return pd.DataFrame(rows, columns=["path", "label"])


def make_splits(raw_dir, ratios, seed, cache_dir="data/processed"):
    """Partición estratificada train/val/test. Se guarda en CSV para reproducibilidad."""
    train, val, test = ratios["train"], ratios["val"], ratios["test"]
    if abs(train + val + test - 1) > 1e-6:
        raise ValueError("Las proporciones deben sumar 1")
    name = f"splits_s{seed}_{round(train*100)}-{round(val*100)}-{round(test*100)}.csv"
    cache = Path(cache_dir) / name
    if cache.exists():
        return pd.read_csv(cache)

    df = list_images(raw_dir)
    tr, rest = train_test_split(df, test_size=val + test, stratify=df.label, random_state=seed)
    va, te = train_test_split(rest, test_size=test / (val + test), stratify=rest.label, random_state=seed)
    out = pd.concat(
        [tr.assign(split="train"), va.assign(split="val"), te.assign(split="test")],
        ignore_index=True,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def make_dataset(df, raw_dir, class_names, img_size, batch_size, training=False, seed=42):
    """tf.data a partir de rutas; las imágenes se leen del disco bajo demanda."""
    idx = {c: i for i, c in enumerate(class_names)}
    paths = [str(Path(raw_dir) / p) for p in df.path]
    labels = df.label.map(idx).to_numpy("int32")
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        return tf.image.resize(img, img_size), label

    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_datasets(cfg):
    """Devuelve ({'train','val','test'}: tf.data, class_names, splits_df)."""
    d = cfg["data"]
    class_names = list_classes(d["raw_dir"])
    splits = make_splits(d["raw_dir"], d["split"], d["seed"])
    size = tuple(d["img_size"])
    datasets = {
        s: make_dataset(
            splits[splits.split == s], d["raw_dir"], class_names, size,
            d["batch_size"], training=(s == "train"), seed=d["seed"],
        )
        for s in ("train", "val", "test")
    }
    return datasets, class_names, splits
