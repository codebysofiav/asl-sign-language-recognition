"""Preparación de las fotos propias: normalización y asignación de sesiones.

Las fotos propias no traen información de sesión (persona, fondo), así que se proponen
dos agrupaciones aproximadas y se confirman a ojo con una hoja de contacto:
  - "numero": rangos del número de archivo dentro de cada letra (A_01...A_05 = grupo 1, ...)
  - "fondo":  agrupación automática (k-means) por el color de los bordes de la foto
Ninguna separa personas con certeza: con 3 personas y fondos repetidos, una "sesión" es
solo una aproximación.
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.visualize import list_own_photos


def normalize_own_photos(src_dir, dst_dir, class_names, size=200, overwrite=False):
    """Copia las fotos a dst_dir como las ve el modelo: corrige la rotación EXIF, recorta al
    cuadrado central y reduce a size x size (como el dataset original).

    Así evaluar y entrenar usan exactamente la misma imagen. Devuelve DataFrame(path, label).
    """
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    photos = list_own_photos(src_dir, class_names)
    hechas = 0
    for p in photos.path:
        out = dst_dir / Path(p).relative_to(src_dir)
        if out.exists() and not overwrite:
            continue
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
        s = min(w, h)
        left, top = (w - s) // 2, (h - s) // 2
        im = im.crop((left, top, left + s, top + s)).resize((size, size), Image.LANCZOS)
        out.parent.mkdir(parents=True, exist_ok=True)
        im.save(out, quality=95)
        hechas += 1
    print(f"{len(photos)} fotos | {hechas} normalizadas ahora | destino: {dst_dir}")
    return list_own_photos(dst_dir, class_names)


def clave(path, base_dir):
    """Ruta relativa 'clase/archivo', identifica cada foto aunque cambie la carpeta base."""
    return str(Path(path).relative_to(base_dir))


def _numero(path):
    nums = re.findall(r"\d+", Path(path).stem)
    return int(nums[-1]) if nums else 0


def sessions_by_number(df, bin_size=5, n_groups=3):
    """Grupo según el número de archivo: 1-5 -> 1, 6-10 -> 2, 11+ -> 3 (con bin_size=5, n_groups=3)."""
    k = ((df.path.map(_numero) - 1) // bin_size).clip(lower=0, upper=n_groups - 1) + 1
    return k.map(lambda i: f"num-{i}")


def _border_features(path, border=10):
    a = np.asarray(Image.open(path).convert("RGB").resize((100, 100)), dtype="float64")
    mask = np.ones(a.shape[:2], bool)
    mask[border:-border, border:-border] = False
    px = a[mask]
    return np.concatenate([px.mean(0), px.std(0), np.median(px, axis=0)])


def sessions_by_background(df, k=6, seed=0):
    """Grupos por color/luz de los bordes de la foto (k-means). Ajusta k mirando la hoja de contacto."""
    feats = np.stack([_border_features(p) for p in df.path])
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(StandardScaler().fit_transform(feats))
    orden = pd.Series(labels).value_counts().index.tolist()  # grupo 1 = el más grande
    nombre = {c: f"fondo-{i + 1}" for i, c in enumerate(orden)}
    return pd.Series([nombre[c] for c in labels], index=df.index)


def assign_sessions(df, method="numero", **kw):
    """Devuelve una copia de df con la columna `sesion`. method: 'numero' o 'fondo'."""
    out = df.copy()
    if method == "numero":
        out["sesion"] = sessions_by_number(out, **kw)
    elif method == "fondo":
        out["sesion"] = sessions_by_background(out, **kw)
    else:
        raise ValueError("method debe ser 'numero' o 'fondo'")
    return out


def save_sessions(df, base_dir, csv_path):
    out = pd.DataFrame({"clave": df.path.map(lambda p: clave(p, base_dir)), "label": df.label, "sesion": df.sesion})
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_path, index=False)
    return out


def load_sessions(df, base_dir, csv_path):
    """Añade la columna `sesion` a df leyendo el CSV; fotos nuevas quedan como 'sin_asignar'."""
    tabla = pd.read_csv(csv_path).set_index("clave").sesion
    out = df.copy()
    out["sesion"] = out.path.map(lambda p: tabla.get(clave(p, base_dir), "sin_asignar"))
    n = int((out.sesion == "sin_asignar").sum())
    if n:
        print(f"Aviso: {n} fotos no están en {csv_path} (sesion = 'sin_asignar'). Vuelve a asignar sesiones.")
    return out


def contact_sheet(df, per_group=10, seed=0, title=""):
    """Una fila por grupo con fotos al azar, para comprobar a ojo si el grupo es coherente."""
    grupos = sorted(df.sesion.unique())
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(grupos), per_group, figsize=(1.9 * per_group, 2.1 * len(grupos)), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, g in enumerate(grupos):
        sub = df[df.sesion == g]
        pick = sub.iloc[rng.permutation(len(sub))[:per_group]]
        for j, r in enumerate(pick.itertuples()):
            axes[i, j].imshow(Image.open(r.path))
            axes[i, j].set_title(r.label, fontsize=8)
        axes[i, 0].text(-0.08, 0.5, f"{g}\n(n={len(sub)})", transform=axes[i, 0].transAxes,
                        ha="right", va="center", fontsize=10)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
