"""Utilities for creating reproducible ASL image data splits and pipelines."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class DatasetSplits:
    """Stratified file lists plus class metadata for one dataset."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    class_names: list[str]
    duplicates_removed: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_stratified_splits(
    class_root: str | Path,
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    image_extensions: Iterable[str] = DEFAULT_IMAGE_EXTENSIONS,
) -> DatasetSplits:
    """Create reproducible stratified splits from class-named subfolders.

    Exact duplicate files are collapsed before splitting so copies cannot land
    in separate splits. If identical bytes appear under different labels, an
    error is raised because the label assignment is ambiguous. The caller
    should pass the training class root, not a separately supplied test folder.
    """
    root = Path(class_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Class directory does not exist: {root}")
    if not (0 < validation_fraction < 1 and 0 < test_fraction < 1):
        raise ValueError("validation_fraction and test_fraction must be between 0 and 1.")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("The validation and test fractions must sum to less than 1.")

    extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in image_extensions}
    class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class subdirectories found in {root}")

    # Hash all candidate images. Sorting makes the retained copy deterministic.
    seen_hashes: dict[str, tuple[str, str]] = {}
    rows: list[dict[str, object]] = []
    duplicates_removed = 0
    for class_dir in class_dirs:
        class_files = sorted(
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )
        if not class_files:
            raise ValueError(f"Class folder contains no images: {class_dir}")
        for path in class_files:
            digest = _sha256(path)
            previous = seen_hashes.get(digest)
            if previous:
                previous_label, previous_path = previous
                if previous_label != class_dir.name:
                    raise ValueError(
                        "Identical image bytes have conflicting labels: "
                        f"{previous_path} ({previous_label}) and {path} ({class_dir.name})."
                    )
                duplicates_removed += 1
                continue
            seen_hashes[digest] = (class_dir.name, str(path))
            rows.append({
                "path": str(path),
                "label": class_dir.name,
                "sha256": digest,
            })

    frame = pd.DataFrame(rows)
    class_names = [path.name for path in class_dirs]
    label_to_id = {name: index for index, name in enumerate(class_names)}
    frame["label_id"] = frame["label"].map(label_to_id).astype("int64")

    # First reserve the final test fraction, then take validation from the rest.
    train_validation, test = train_test_split(
        frame,
        test_size=test_fraction,
        random_state=seed,
        stratify=frame["label_id"],
    )
    relative_validation_fraction = validation_fraction / (1 - test_fraction)
    train, validation = train_test_split(
        train_validation,
        test_size=relative_validation_fraction,
        random_state=seed,
        stratify=train_validation["label_id"],
    )

    return DatasetSplits(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
        class_names=class_names,
        duplicates_removed=duplicates_removed,
    )


def build_tf_dataset(
    frame: pd.DataFrame,
    *,
    image_size: tuple[int, int] = (200, 200),
    batch_size: int = 32,
    shuffle: bool = False,
    seed: int = 42,
    shuffle_buffer: int = 10_000,
):
    """Build a batched ``tf.data.Dataset`` yielding float images and labels.

    JPEG/PNG pixels are decoded as RGB and converted to float32 in [0, 1].
    Resizing defaults to the dimensions observed in the exploration notebook.
    TensorFlow is imported here so split inspection does not require importing
    the deep-learning runtime.
    """
    if frame.empty:
        raise ValueError("Cannot create a TensorFlow dataset from an empty split.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if len(image_size) != 2 or any(size < 1 for size in image_size):
        raise ValueError("image_size must be a pair of positive integers.")

    import tensorflow as tf

    paths = frame["path"].astype(str).to_numpy()
    labels = frame["label_id"].astype("int32").to_numpy()
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(
            min(len(frame), shuffle_buffer), seed=seed, reshuffle_each_iteration=True
        )

    def decode_and_preprocess(path, label):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, image_size, antialias=True)
        image = tf.ensure_shape(image, (image_size[0], image_size[1], 3))
        return image, label

    return (
        dataset
        .map(decode_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
