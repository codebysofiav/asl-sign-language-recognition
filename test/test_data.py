from src.data import list_classes, make_splits


def _fake_dataset(root, n_classes=3, n_per_class=20):
    for c in range(n_classes):
        d = root / f"C{c}"
        d.mkdir(parents=True)
        for i in range(n_per_class):
            (d / f"{i}.jpg").write_bytes(b"x")


def test_split_proporciones_y_sin_solapamiento(tmp_path):
    raw = tmp_path / "raw"
    _fake_dataset(raw)
    ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
    df = make_splits(raw, ratios, seed=1, cache_dir=tmp_path / "cache")

    assert len(df) == 60
    assert df.path.is_unique  # ninguna imagen en dos particiones
    counts = df.split.value_counts()
    assert counts["train"] == 42 and counts["val"] == 9 and counts["test"] == 9
    # estratificado: cada clase aparece en las 3 particiones
    assert (df.groupby("label").split.nunique() == 3).all()


def test_clases_ordenadas(tmp_path):
    _fake_dataset(tmp_path / "raw")
    assert list_classes(tmp_path / "raw") == ["C0", "C1", "C2"]
