"""Generate the standalone, Colab-ready training notebooks in notebooks/."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
NOTEBOOKS = HERE / "notebooks"
NOTEBOOKS.mkdir(parents=True, exist_ok=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None,
        "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True),
    }


def write_notebook(filename: str, cells: list[dict]) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {"name": filename, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (NOTEBOOKS / filename).write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


STATIC_SETUP = r'''# Colab: instala las dependencias y conecta Drive.
%pip install -q mediapipe==0.10.35 scikit-learn
from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
import csv, hashlib, json, shutil, urllib.request, warnings
import mediapipe as mp
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
SAMPLES_PER_CLASS = 1724  # None para usar todos los archivos disponibles por clase
HEAD_EPOCHS = 8
FINETUNE_EPOCHS = 12
AUTOTUNE = tf.data.AUTOTUNE
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

DRIVE_ROOT = Path("/content/drive/MyDrive/proyecto-ia2")
# Dataset variado principal. Para comparar contra ASL raw, cambia solamente DATA_ROOT
# a la carpeta interna que contiene las carpetas de clases.
DATA_ROOT = DRIVE_ROOT / "dataset_asl/final_dataset"
MODE = "@@MODE@@"
RUN_NAME = "integrated_varied_1724"
WORK_DIR = DRIVE_ROOT / f"data/processed/proyecto_def/{MODE}/{RUN_NAME}"
EXPORT_DIR = DRIVE_ROOT / f"exports/proyecto_def/{MODE}/{RUN_NAME}"
MAPS_DIR = WORK_DIR / "maps"
CROPS_DIR = WORK_DIR / "crops"
MANIFEST = WORK_DIR / "manifest.csv"
SPLIT_FILE = WORK_DIR / "split.csv"
MODEL_ASSET = DRIVE_ROOT / "exports/proyecto_def/hand_landmarker.task"
WORK_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_ROOT.is_dir():
    raise FileNotFoundError(f"No encuentro el dataset integrado: {DATA_ROOT}")
if not MODEL_ASSET.exists():
    MODEL_ASSET.parent.mkdir(parents=True, exist_ok=True)
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_ASSET)

LETTERS = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in {"J", "Z"}]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def normalize_label(name):
    value = name.strip().lower()
    aliases = {"delete": "del", "backspace": "del", "nothing": "nothing", "space": "space"}
    return value.upper() if len(value) == 1 else aliases.get(value, value)

folder_by_label = {}
for folder in DATA_ROOT.iterdir():
    if folder.is_dir():
        label = normalize_label(folder.name)
        if label in folder_by_label:
            raise ValueError(f"Carpetas duplicadas para clase {label}: {folder_by_label[label]} y {folder}")
        folder_by_label[label] = folder
missing = sorted(set(LETTERS) - set(folder_by_label))
if missing:
    raise ValueError(f"Faltan carpetas de letras estáticas: {missing}; revisa DATA_ROOT")
SPECIALS = [label for label in ("space", "nothing", "del") if label in folder_by_label]
CLASSES = LETTERS + SPECIALS
CLASS_TO_ID = {label: index for index, label in enumerate(CLASSES)}
print("Raíz:", DATA_ROOT)
print("Clases usadas:", CLASSES)
print("Clases J/Z excluidas; las reconoce la rama temporal.")
print("Salida y checkpoints en:", EXPORT_DIR)
'''

STATIC_PREPROCESS = r'''# Construcción del conjunto balanceado y preprocesamiento reanudable.
def choose_sources():
    rows = []
    for label in CLASSES:
        files = sorted(p for p in folder_by_label[label].rglob("*")
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if SAMPLES_PER_CLASS is not None and len(files) > SAMPLES_PER_CLASS:
            rng = np.random.default_rng(SEED + sum(map(ord, label)))
            idx = np.sort(rng.choice(len(files), size=SAMPLES_PER_CLASS, replace=False))
            files = [files[i] for i in idx]
        rows.extend({"source": p, "source_rel": p.relative_to(DATA_ROOT).as_posix(), "label": label}
                    for p in files)
    counts = pd.Series([r["label"] for r in rows]).value_counts().reindex(CLASSES, fill_value=0)
    display(counts.rename("seleccionadas").to_frame())
    return rows

FINGER_COLORS = [(255,90,90),(255,210,40),(80,235,100),(50,190,255),(220,90,255)]
FINGER_EDGES = [[(0,1),(1,2),(2,3),(3,4)],[(0,5),(5,6),(6,7),(7,8)],
                [(0,9),(9,10),(10,11),(11,12)],[(0,13),(13,14),(14,15),(15,16)],
                [(0,17),(17,18),(18,19),(19,20)]]
PALM_EDGES = [(5,9),(9,13),(13,17)]
POINT_COLORS = {**{i:FINGER_COLORS[0] for i in range(1,5)},
                **{i:FINGER_COLORS[1] for i in range(5,9)},
                **{i:FINGER_COLORS[2] for i in range(9,13)},
                **{i:FINGER_COLORS[3] for i in range(13,17)},
                **{i:FINGER_COLORS[4] for i in range(17,21)},0:(255,255,255)}

def landmarks_to_map(hand):
    xy = np.asarray([(p.x,p.y) for p in hand], dtype=np.float32)
    if xy.shape != (21,2) or not np.isfinite(xy).all(): return None
    xy -= xy[0]
    scale = float(np.linalg.norm(xy[9]))
    if scale < 1e-6: return None
    xy /= scale
    extent = float(np.max(np.abs(xy)))
    if extent < 1e-6: return None
    xy = xy / extent * 0.40 * (IMAGE_SIZE-1) + (IMAGE_SIZE-1)/2
    pts = [(int(round(x)),int(round(y))) for x,y in xy]
    canvas = Image.new("RGB",(IMAGE_SIZE,IMAGE_SIZE),(0,0,0)); draw = ImageDraw.Draw(canvas)
    width, radius = max(2,IMAGE_SIZE//70), max(3,IMAGE_SIZE//48)
    for edges,color in zip(FINGER_EDGES,FINGER_COLORS):
        for a,b in edges: draw.line((pts[a],pts[b]),fill=color,width=width)
    for a,b in PALM_EDGES: draw.line((pts[a],pts[b]),fill=(235,235,235),width=width)
    for i,(x,y) in enumerate(pts):
        draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=POINT_COLORS[i],outline=(255,255,255))
    return canvas

def crop_from_image(image, hand, margin=0.30):
    points = np.asarray([(p.x*image.width,p.y*image.height) for p in hand],dtype=np.float32)
    x0,y0 = points.min(axis=0); x1,y1 = points.max(axis=0)
    side = max(float(max(x1-x0,y1-y0)),2.0)*(1+2*margin)
    cx,cy=(x0+x1)/2,(y0+y1)/2
    left,top=int(round(cx-side/2)),int(round(cy-side/2)); side=max(int(round(side)),2)
    right,bottom=left+side,top+side
    result=Image.new("RGB",(side,side),(0,0,0))
    box=(max(0,left),max(0,top),min(image.width,right),min(image.height,bottom))
    if box[2] <= box[0] or box[3] <= box[1]: return None
    result.paste(image.crop(box),(box[0]-left,box[1]-top))
    return result.resize((IMAGE_SIZE,IMAGE_SIZE),Image.Resampling.BILINEAR)

def make_detector():
    options=mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_ASSET)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,num_hands=1,
        min_hand_detection_confidence=0.5,min_hand_presence_confidence=0.5)
    return mp.tasks.vision.HandLandmarker.create_from_options(options)

def prepare_images():
    output_root = MAPS_DIR if MODE == "maps" else CROPS_DIR
    output_root.mkdir(parents=True,exist_ok=True)
    previous={}
    if MANIFEST.exists():
        old=pd.read_csv(MANIFEST).drop_duplicates("source_rel",keep="last")
        previous={str(r.source_rel):r._asdict() for r in old.itertuples(index=False)}
    append_header=not MANIFEST.exists() or MANIFEST.stat().st_size==0
    detector=make_detector(); rows=[]
    try:
        with MANIFEST.open("a",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=["source_rel","label","output_rel","detected","error"])
            if append_header: writer.writeheader()
            sources=choose_sources()
            for i,row in enumerate(sources,1):
                rel=row["source_rel"]
                name=hashlib.sha1(rel.encode("utf-8")).hexdigest()+(".png" if MODE=="maps" else ".jpg")
                output_rel=f"{row['label']}/{name}"; destination=output_root/output_rel
                old=previous.get(rel)
                if old and destination.is_file():
                    rows.append(old); continue
                destination.parent.mkdir(parents=True,exist_ok=True)
                detected=False; error=""
                try:
                    with Image.open(row["source"]) as raw:
                        image=ImageOps.exif_transpose(raw).convert("RGB")
                    if row["label"]=="nothing":
                        result=Image.new("RGB",(IMAGE_SIZE,IMAGE_SIZE),(0,0,0))
                    else:
                        rgb=np.asarray(image)
                        result_mp=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
                        if result_mp.hand_landmarks:
                            hand=result_mp.hand_landmarks[0]
                            if MODE=="maps": result=landmarks_to_map(hand)
                            else: result=crop_from_image(image,hand)
                            detected=result is not None
                        else: result=None
                    if result is None: result=Image.new("RGB",(IMAGE_SIZE,IMAGE_SIZE),(0,0,0))
                    result.save(destination,format="PNG" if MODE=="maps" else "JPEG",quality=95)
                except Exception as exc:
                    error=f"{type(exc).__name__}: {exc}"
                    result=Image.new("RGB",(IMAGE_SIZE,IMAGE_SIZE),(0,0,0))
                    result.save(destination,format="PNG" if MODE=="maps" else "JPEG",quality=95)
                record={"source_rel":rel,"label":row["label"],"output_rel":output_rel,
                        "detected":int(detected),"error":error}
                writer.writerow(record); rows.append(record)
                if i%25==0 or i==len(sources):
                    stream.flush(); print(f"{i}/{len(sources)} procesadas | última: {rel}")
    finally: detector.close()
    return pd.DataFrame(rows).drop_duplicates("source_rel",keep="last")

manifest_df=prepare_images()
display(manifest_df.groupby(["label","detected"]).size().unstack(fill_value=0))
'''

STATIC_TRAIN = r'''# Filtra fallos de detección y fija las particiones para poder reanudar.
usable=manifest_df[(manifest_df.detected.astype(int)==1) | (manifest_df.label=="nothing")].copy()
counts=usable.label.value_counts()
too_small=counts[counts<10]
if not too_small.empty: raise ValueError(f"Clases con menos de 10 ejemplos utilizables: {too_small.to_dict()}")

if SPLIT_FILE.exists():
    split_df=pd.read_csv(SPLIT_FILE)
    if set(split_df.source_rel)!=set(usable.source_rel):
        raise ValueError("Cambió el conjunto procesado desde la partición anterior. Renombra/elimina split.csv para crear otra partición.")
else:
    train_val,test=train_test_split(usable,test_size=0.15,random_state=SEED,stratify=usable.label)
    train,val=train_test_split(train_val,test_size=(0.15/0.85),random_state=SEED,stratify=train_val.label)
    split_df=pd.concat([train.assign(split="train"),val.assign(split="validation"),test.assign(split="test")],ignore_index=True)
    split_df.to_csv(SPLIT_FILE,index=False)
display(split_df.groupby(["split","label"]).size().unstack(fill_value=0))
train_df=split_df[split_df.split=="train"].copy()
val_df=split_df[split_df.split=="validation"].copy()
test_df=split_df[split_df.split=="test"].copy()
output_root=MAPS_DIR if MODE=="maps" else CROPS_DIR

def make_dataset(frame,training=False):
    paths=[str(output_root/p) for p in frame.output_rel.astype(str)]
    labels=frame.label.map(CLASS_TO_ID).to_numpy(dtype=np.int32)
    ds=tf.data.Dataset.from_tensor_slices((paths,labels))
    if training: ds=ds.shuffle(len(frame),seed=SEED,reshuffle_each_iteration=True)
    def load(path,label):
        image=tf.io.decode_image(tf.io.read_file(path),channels=3,expand_animations=False)
        image=tf.image.resize(image,(IMAGE_SIZE,IMAGE_SIZE),antialias=True)
        image=tf.cast(image,tf.float32)
        if training and MODE=="crop":
            image=tf.image.random_contrast(image,0.85,1.15)
            image=tf.image.random_brightness(image,max_delta=15.0)
            image=tf.clip_by_value(image,0.0,255.0)
        return image,label
    return ds.map(load,num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

train_ds=make_dataset(train_df,True); val_ds=make_dataset(val_df); test_ds=make_dataset(test_df)

def build_model():
    inputs=tf.keras.Input((IMAGE_SIZE,IMAGE_SIZE,3),name="static_image")
    x=tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    backbone=tf.keras.applications.MobileNetV2(include_top=False,weights="imagenet",
        input_shape=(IMAGE_SIZE,IMAGE_SIZE,3),name="mobilenetv2_backbone")
    backbone.trainable=False
    x=backbone(x,training=False)
    x=tf.keras.layers.GlobalAveragePooling2D()(x)
    x=tf.keras.layers.Dropout(0.30)(x)
    outputs=tf.keras.layers.Dense(len(CLASSES),activation="softmax",dtype="float32",name="class_probs")(x)
    model=tf.keras.Model(inputs,outputs,name=f"mobilenetv2_{MODE}")
    return model

def callbacks_for(phase):
    checkpoint_dir=EXPORT_DIR/"checkpoints"/phase
    checkpoint_dir.mkdir(parents=True,exist_ok=True)
    return [
        tf.keras.callbacks.BackupAndRestore(backup_dir=str(EXPORT_DIR/"backup"/phase),save_freq="epoch"),
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint_dir/"best.keras"),monitor="val_loss",
            mode="min",save_best_only=True,verbose=1),
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint_dir/"epoch-{epoch:02d}.weights.h5"),
            save_weights_only=True,save_best_only=False,save_freq="epoch"),
        tf.keras.callbacks.CSVLogger(str(EXPORT_DIR/f"history_{phase}.csv"),append=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=1,min_lr=1e-6),
    ]

def train_or_resume():
    final_path=EXPORT_DIR/"model.keras"
    if final_path.exists() and (EXPORT_DIR/"training_complete.json").exists():
        print("Entrenamiento completo ya existe; cargo el modelo guardado.")
        return tf.keras.models.load_model(final_path,compile=False)
    head_best=EXPORT_DIR/"checkpoints/head/best.keras"
    head_done=EXPORT_DIR/"head_complete.json"
    if head_done.exists() and head_best.exists():
        model=tf.keras.models.load_model(head_best,compile=False)
    else:
        model=build_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss="sparse_categorical_crossentropy",metrics=["accuracy"])
        model.fit(train_ds,validation_data=val_ds,epochs=HEAD_EPOCHS,callbacks=callbacks_for("head"),verbose=2)
        if not head_best.exists(): raise RuntimeError("No se guardó el checkpoint de la cabeza.")
        head_done.write_text(json.dumps({"complete":True}),encoding="utf-8")
        model=tf.keras.models.load_model(head_best,compile=False)
    backbone=model.get_layer("mobilenetv2_backbone")
    backbone.trainable=True
    for layer in backbone.layers[:-30]: layer.trainable=False
    for layer in backbone.layers:
        if isinstance(layer,tf.keras.layers.BatchNormalization): layer.trainable=False
    fine_best=EXPORT_DIR/"checkpoints/finetune/best.keras"
    fine_done=EXPORT_DIR/"finetune_complete.json"
    if fine_done.exists() and fine_best.exists():
        model=tf.keras.models.load_model(fine_best,compile=False)
    else:
        model.compile(optimizer=tf.keras.optimizers.Adam(5e-5),
                      loss="sparse_categorical_crossentropy",metrics=["accuracy"])
        model.fit(train_ds,validation_data=val_ds,epochs=FINETUNE_EPOCHS,
                  callbacks=callbacks_for("finetune"),verbose=2)
        if not fine_best.exists(): raise RuntimeError("No se guardó el checkpoint de fine-tuning.")
        fine_done.write_text(json.dumps({"complete":True}),encoding="utf-8")
        model=tf.keras.models.load_model(fine_best,compile=False)
    model.save(final_path)
    (EXPORT_DIR/"classes.json").write_text(json.dumps(CLASSES,ensure_ascii=False,indent=2),encoding="utf-8")
    (EXPORT_DIR/"metadata.json").write_text(json.dumps({"mode":MODE,"image_size":IMAGE_SIZE,
        "classes":CLASSES,"source":str(DATA_ROOT),"run_name":RUN_NAME,
        "preprocessing":"MediaPipe map" if MODE=="maps" else "MediaPipe square crop",
        "crop_margin":0.30 if MODE=="crop" else None,"seed":SEED},ensure_ascii=False,indent=2),encoding="utf-8")
    shutil.copy2(MODEL_ASSET,EXPORT_DIR/"hand_landmarker.task")
    (EXPORT_DIR/"training_complete.json").write_text(json.dumps({"complete":True}),encoding="utf-8")
    return model

model=train_or_resume()
'''

STATIC_EVAL = r'''# Evaluación final en la partición de prueba (no se usa para ajustar hiperparámetros).
y_true=test_df.label.map(CLASS_TO_ID).to_numpy(dtype=np.int32)
y_prob=model.predict(test_ds,verbose=0)
y_pred=y_prob.argmax(axis=1)
metrics={"n_test":int(len(test_df)),"accuracy":float(accuracy_score(y_true,y_pred)),
         "macro_f1":float(f1_score(y_true,y_pred,average="macro",zero_division=0)),
         "classification_report":classification_report(y_true,y_pred,labels=list(range(len(CLASSES))),
             target_names=CLASSES,output_dict=True,zero_division=0)}
(EXPORT_DIR/"metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8")
pd.DataFrame({"source_rel":test_df.source_rel,"true":[CLASSES[i] for i in y_true],
              "prediction":[CLASSES[i] for i in y_pred]}).to_csv(EXPORT_DIR/"test_predictions.csv",index=False)
cm=confusion_matrix(y_true,y_pred,labels=list(range(len(CLASSES))),normalize="true")
fig,ax=plt.subplots(figsize=(12,10))
ConfusionMatrixDisplay(cm,display_labels=CLASSES).plot(ax=ax,xticks_rotation=90,cmap="Blues",colorbar=False)
fig.tight_layout(); fig.savefig(EXPORT_DIR/"confusion_matrix.png",dpi=140); plt.show()
print({k:v for k,v in metrics.items() if k!="classification_report"})
print("Modelo, clases, configuración y checkpoints:",EXPORT_DIR)
'''


def make_static_notebook(mode: str):
    label = "mapas de puntos" if mode == "maps" else "recortes de mano"
    notebook_name = "10_entrenamiento_mapas_colab.ipynb" if mode == "maps" else "11_entrenamiento_recortes_colab.ipynb"
    readme = f'''# Entrenamiento estático — {label}\n\nEste notebook es autocontenido y está pensado para ejecutarse en Colab con GPU. Usa como fuente principal `proyecto-ia2/dataset_asl/final_dataset`, procesa las imágenes con MediaPipe, entrena MobileNetV2 y guarda el modelo/checkpoints en Drive.\n\n**Clases:** letras estáticas A–I y K–Y; J/Z se reservan para el modelo temporal. Se incluyen `space`, `nothing` y `del` si existen en el dataset. `del` funciona como retroceso en la app. No se aplica rotación; la variante de recortes usa cambios moderados de brillo/contraste.\n\n**Reanudación:** el preprocesamiento escribe un manifiesto incremental en Drive. El entrenamiento guarda checkpoints por época y `BackupAndRestore` en Drive; al reconectar, vuelve a ejecutar las celdas de configuración, preprocesamiento y entrenamiento con la misma ruta y split. No borres las carpetas `backup/` ni `checkpoints/`.\n\n**Datos:** el conjunto integrado variado es el principal. El ASL raw de 87.000 imágenes no se concatena automáticamente; primero revisa si contiene imágenes duplicadas con el integrado. Para usar raw como comparación, cambia `DATA_ROOT` **y** asigna un `RUN_NAME` distinto en la celda inicial, para no mezclar manifiestos, splits ni checkpoints.\n'''
    setup = STATIC_SETUP.replace("@@MODE@@", mode)
    cells = [markdown(readme), code(setup), markdown("## Preprocesar con recuperación\n\nPuedes ejecutar esta celda otra vez si Colab se desconecta. Los archivos procesados y el manifiesto persistente permiten continuar sin repetir los que ya quedaron completos."), code(STATIC_PREPROCESS), markdown("## Split y entrenamiento\n\nEl split es estratificado por clase a nivel de imagen (70/15/15). Si hay copias idénticas entre particiones, las métricas pueden ser optimistas; conserva `source_rel` y revisa duplicados/procedencia."), code(STATIC_TRAIN), markdown("## Evaluación y exportación\n\nEl conjunto test se reserva antes de entrenar y solo se usa en esta celda."), code(STATIC_EVAL)]
    write_notebook(notebook_name,cells)


VIDEO_SETUP = r'''%pip install -q mediapipe==0.10.35 scikit-learn opencv-python-headless
from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import csv, hashlib, json, shutil, urllib.request, warnings
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

SEED=42; SEQUENCE_FRAMES=17; MIN_DETECTED_FRAMES=12; FLOW_SIZE=128; FLOW_CLIP_PIXELS=20.0
BATCH_SIZE=8; EPOCHS=35; AUTOTUNE=tf.data.AUTOTUNE
CLASSES=["J","Z","other"]; CLASS_TO_ID={name:i for i,name in enumerate(CLASSES)}
tf.keras.utils.set_random_seed(SEED); np.random.seed(SEED)
DRIVE_ROOT=Path("/content/drive/MyDrive/proyecto-ia2")
VIDEO_ROOT=DRIVE_ROOT/"data/videos_jz"
WORK_DIR=DRIVE_ROOT/"data/processed/proyecto_def/video"
EXPORT_DIR=DRIVE_ROOT/"exports/proyecto_def/video_balanced_v2"
MANIFEST=WORK_DIR/"video_manifest.csv"; SPLIT_FILE=WORK_DIR/"split_balanced_v2.csv"
MODEL_ASSET=DRIVE_ROOT/"exports/proyecto_def/hand_landmarker.task"
TEMPORAL_MODEL=EXPORT_DIR/"jz_flow_cnn.keras"
WORK_DIR.mkdir(parents=True,exist_ok=True); EXPORT_DIR.mkdir(parents=True,exist_ok=True)
if not VIDEO_ROOT.is_dir(): raise FileNotFoundError(f"No encuentro {VIDEO_ROOT}")
if not MODEL_ASSET.exists():
    MODEL_ASSET.parent.mkdir(parents=True,exist_ok=True)
    url="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    urllib.request.urlretrieve(url,MODEL_ASSET)
print("Videos:",VIDEO_ROOT,"\nSe esperan J/<signer>/clip, Z/<signer>/clip, other/<signer>/clip")
print("Los IDs de participante deben ser globales y repetirse en J y Z para la misma persona.")
print("Procesamiento y checkpoints:",EXPORT_DIR)
'''

VIDEO_PREPROCESS = r'''VIDEO_EXTENSIONS={".mp4",".avi",".mov",".mkv"}
def canonical_label(name):
    value=name.strip().lower()
    return {"j":"J","z":"Z","other":"other","none":"other","background":"other"}.get(value)

def sample_frames(path,count=SEQUENCE_FRAMES):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): cap.release(); return None
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total>=2:
        indices=np.linspace(0,total-1,count).round().astype(int); frames=[]
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx)); ok,frame=cap.read()
            if not ok: frames=[]; break
            frames.append(frame)
        cap.release()
        if len(frames)==count: return frames
    else:
        cap.release()

    # Fallback for playable files (often AVI) whose frame-count metadata is zero
    # or whose codec cannot seek reliably by frame index. Count with sequential
    # reads, reopen, then sample evenly without retaining the whole video in RAM.
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): cap.release(); return None
    actual_total=0
    while True:
        ok,_=cap.read()
        if not ok: break
        actual_total+=1
    cap.release()
    if actual_total<2: return None
    indices=np.linspace(0,actual_total-1,count).round().astype(int)
    wanted=set(indices.tolist()); selected={}
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): cap.release(); return None
    frame_index=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        if frame_index in wanted: selected[frame_index]=frame.copy()
        frame_index+=1
    cap.release()
    if not wanted.issubset(selected): return None
    return [selected[int(idx)] for idx in indices]

def get_points(frame,detector):
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
    if not result.hand_landmarks: return None
    h,w=frame.shape[:2]
    return np.asarray([(p.x*w,p.y*h) for p in result.hand_landmarks[0]],dtype=np.float32)

def fixed_crops(frames,points_by_frame):
    all_points=np.concatenate(points_by_frame,axis=0)
    x0,y0=all_points.min(axis=0); x1,y1=all_points.max(axis=0)
    cx,cy=(x0+x1)/2,(y0+y1)/2; side=max(float(max(x1-x0,y1-y0)*1.7),32.0)
    left,top=int(np.floor(cx-side/2)),int(np.floor(cy-side/2)); side=int(np.ceil(side))
    crops=[]
    for frame in frames:
        h,w=frame.shape[:2]; right,bottom=left+side,top+side
        pl,pt=max(0,-left),max(0,-top); pr,pb=max(0,right-w),max(0,bottom-h)
        padded=cv2.copyMakeBorder(frame,pt,pb,pl,pr,cv2.BORDER_CONSTANT,value=(0,0,0))
        crop=padded[top+pt:bottom+pt,left+pl:right+pl]
        crops.append(cv2.resize(crop,(FLOW_SIZE,FLOW_SIZE),interpolation=cv2.INTER_AREA))
    return crops

def video_to_flow(path,detector):
    frames=sample_frames(path)
    if frames is None: return None,"video ilegible o con menos de dos fotogramas"
    points=[get_points(frame,detector) for frame in frames]
    detected_points=[p for p in points if p is not None]
    if len(detected_points)<MIN_DETECTED_FRAMES:
        return None,f"MediaPipe detectó {len(detected_points)}/{SEQUENCE_FRAMES} fotogramas; mínimo {MIN_DETECTED_FRAMES}"
    # El recorte común usa todas las detecciones disponibles. Los fotogramas donde
    # falló MediaPipe se conservan para el flujo óptico; no invalidan el clip solos.
    crops=fixed_crops(frames,detected_points); gray=[cv2.cvtColor(c,cv2.COLOR_BGR2GRAY) for c in crops]
    fields=[]
    for previous,current in zip(gray[:-1],gray[1:]):
        flow=cv2.calcOpticalFlowFarneback(previous,current,None,0.5,3,15,3,5,1.2,0)
        fields.append(np.clip(flow/FLOW_CLIP_PIXELS,-1.0,1.0).astype(np.float32))
    return np.stack(fields),""

def prepare_video_flows():
    old={}
    if MANIFEST.exists():
        old_df=pd.read_csv(MANIFEST).drop_duplicates("video_rel",keep="last")
        old={str(r.video_rel):r._asdict() for r in old_df.itertuples(index=False)}
    video_files=sorted(p for p in VIDEO_ROOT.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    if not video_files: raise ValueError("No hay videos compatibles en VIDEO_ROOT")
    header=not MANIFEST.exists() or MANIFEST.stat().st_size==0
    options=mp.tasks.vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_ASSET)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,num_hands=1,min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5)
    detector=mp.tasks.vision.HandLandmarker.create_from_options(options); rows=[]
    try:
        with MANIFEST.open("a",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=["video_rel","label","signer","flow_rel","ok","error"])
            if header: writer.writeheader()
            for i,path in enumerate(video_files,1):
                rel=path.relative_to(VIDEO_ROOT); parts=rel.parts
                label=canonical_label(parts[0]) if len(parts)>=3 else None
                if label is None: warnings.warn(f"Ignoro ruta; debe ser clase/signante/video: {rel}"); continue
                signer=parts[1]
                if signer.lower() in {"unknown","desconocido","na"}:
                    warnings.warn(f"El ID de signante no es fiable: {rel}")
                out_name=hashlib.sha1(rel.as_posix().encode()).hexdigest()+".npy"
                out_rel=f"flows/{label}/{signer}/{out_name}"; output=WORK_DIR/out_rel
                prev=old.get(rel.as_posix())
                if prev and output.is_file() and int(prev["ok"])==1:
                    rows.append(prev); continue
                output.parent.mkdir(parents=True,exist_ok=True)
                flow,error=video_to_flow(path,detector); ok=flow is not None
                if ok: np.save(output,flow)
                record={"video_rel":rel.as_posix(),"label":label,"signer":signer,
                        "flow_rel":out_rel,"ok":int(ok),"error":error}
                writer.writerow(record); rows.append(record); stream.flush()
                if i%10==0 or i==len(video_files): print(f"{i}/{len(video_files)} | {rel} | {'OK' if ok else error}")
    finally: detector.close()
    return pd.DataFrame(rows).drop_duplicates("video_rel",keep="last")

manifest_df=prepare_video_flows()
print("Conteos procesados:")
display(manifest_df[manifest_df.ok.astype(int)==1].groupby(["label","signer"]).size().unstack(fill_value=0))
'''

VIDEO_TRAIN = r'''usable=manifest_df[manifest_df.ok.astype(int)==1].copy()
if not set(CLASSES).issubset(set(usable.label)):
    raise ValueError(f"Se requieren clips procesables para {CLASSES}; conteos: {usable.label.value_counts().to_dict()}")
if usable.signer.nunique()<3:
    raise ValueError("Se requieren al menos tres IDs de signante; crea carpetas reales por persona.")

if SPLIT_FILE.exists():
    split_df=pd.read_csv(SPLIT_FILE)
    if set(split_df.video_rel)!=set(usable.video_rel):
        raise ValueError("Cambió el conjunto de videos desde el split guardado. Usa split_balanced_v2.csv nuevo o revisa la versión existente.")
else:
    # Asigna signantes completos y busca un reparto cercano a 70/15/15 por clase.
    # Validation y test deben tener al menos cinco clips de cada etiqueta.
    signer_names=sorted(usable.signer.unique())
    group_counts=np.zeros((len(signer_names),len(CLASSES)),dtype=np.int32)
    for gi,signer in enumerate(signer_names):
        counts=usable[usable.signer==signer].label.value_counts()
        for ci,label in enumerate(CLASSES): group_counts[gi,ci]=int(counts.get(label,0))
    fractions=np.asarray([0.70,0.15,0.15],dtype=np.float64)
    totals=group_counts.sum(axis=0).astype(np.float64)
    targets=fractions[:,None]*totals[None,:]
    rng=np.random.default_rng(SEED)
    best_assignment=None; best_score=np.inf
    for _ in range(100000):
        assignment=rng.choice(3,size=len(signer_names),p=fractions)
        counts=np.stack([group_counts[assignment==split].sum(axis=0) for split in range(3)])
        if np.any(counts[0]<1) or np.any(counts[1:]<5): continue
        relative=(counts-targets)/np.maximum(targets,1.0)
        score=float(np.square(relative).sum())
        if score<best_score:
            best_score=score; best_assignment=assignment.copy()
    if best_assignment is None:
        summary={name:{label:int(group_counts[i,j]) for j,label in enumerate(CLASSES) if group_counts[i,j]}
                 for i,name in enumerate(signer_names)}
        raise ValueError("No encontré split por signante con las tres clases y al menos 5 clips "
                         f"por clase en validación/prueba. Signantes y conteos: {summary}")
    split_names=np.asarray(["train","validation","test"])
    signer_to_split={name:split_names[best_assignment[i]] for i,name in enumerate(signer_names)}
    split_df=usable.copy(); split_df["split"]=split_df.signer.map(signer_to_split)
    split_df.to_csv(SPLIT_FILE,index=False)
    print("Error relativo del balance por clase:",round(best_score,4))
display(split_df.groupby(["split","label"]).size().unstack(fill_value=0))
display(split_df.groupby("split").signer.unique().rename("signantes"))

def stack_split(frame):
    X=np.stack([np.load(WORK_DIR/p) for p in frame.flow_rel])
    y=frame.label.map(CLASS_TO_ID).to_numpy(dtype=np.int32)
    return X,y
train_df=split_df[split_df.split=="train"]; val_df=split_df[split_df.split=="validation"]
test_df=split_df[split_df.split=="test"]
X_train,y_train=stack_split(train_df); X_val,y_val=stack_split(val_df); X_test,y_test=stack_split(test_df)
class_weights=compute_class_weight(class_weight="balanced",classes=np.arange(len(CLASSES)),y=y_train)
CLASS_WEIGHTS={i:float(w) for i,w in enumerate(class_weights)}
print("Class weights:",CLASS_WEIGHTS)

def build_model():
    inputs=tf.keras.Input((SEQUENCE_FRAMES-1,FLOW_SIZE,FLOW_SIZE,2),name="optical_flow_sequence")
    x=tf.keras.layers.Conv3D(16,(3,5,5),strides=(1,2,2),padding="same",activation="relu")(inputs)
    x=tf.keras.layers.MaxPool3D((1,2,2))(x)
    x=tf.keras.layers.Conv3D(32,(3,3,3),padding="same",activation="relu")(x)
    x=tf.keras.layers.MaxPool3D((2,2,2))(x)
    x=tf.keras.layers.Conv3D(64,(3,3,3),padding="same",activation="relu")(x)
    x=tf.keras.layers.GlobalAveragePooling3D()(x); x=tf.keras.layers.Dropout(0.35)(x)
    outputs=tf.keras.layers.Dense(3,activation="softmax",name="class_probs")(x)
    model=tf.keras.Model(inputs,outputs,name="jz_optical_flow_cnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    return model

complete=EXPORT_DIR/"training_complete.json"
if TEMPORAL_MODEL.exists() and complete.exists():
    flow_model=tf.keras.models.load_model(TEMPORAL_MODEL,compile=False)
else:
    flow_model=build_model()
    ckpt_dir=EXPORT_DIR/"checkpoints"; ckpt_dir.mkdir(parents=True,exist_ok=True)
    callbacks=[
        tf.keras.callbacks.BackupAndRestore(backup_dir=str(EXPORT_DIR/"backup"),save_freq="epoch"),
        tf.keras.callbacks.ModelCheckpoint(str(ckpt_dir/"best.keras"),monitor="val_loss",mode="min",save_best_only=True,verbose=1),
        tf.keras.callbacks.ModelCheckpoint(str(ckpt_dir/"epoch-{epoch:02d}.weights.h5"),save_weights_only=True,save_best_only=False,save_freq="epoch"),
        tf.keras.callbacks.CSVLogger(str(EXPORT_DIR/"history.csv"),append=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=6,restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=2,min_lr=1e-6),
    ]
    flow_model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=EPOCHS,batch_size=BATCH_SIZE,
                   class_weight=CLASS_WEIGHTS,callbacks=callbacks,verbose=2)
    best=ckpt_dir/"best.keras"
    if best.exists(): flow_model=tf.keras.models.load_model(best,compile=False)
    flow_model.save(TEMPORAL_MODEL)
    (EXPORT_DIR/"classes.json").write_text(json.dumps(CLASSES,indent=2),encoding="utf-8")
    (EXPORT_DIR/"metadata.json").write_text(json.dumps({"classes":CLASSES,"sequence_frames":SEQUENCE_FRAMES,
        "flow_size":FLOW_SIZE,"flow_clip_pixels":FLOW_CLIP_PIXELS,"source":str(VIDEO_ROOT)},indent=2),encoding="utf-8")
    shutil.copy2(MODEL_ASSET,EXPORT_DIR/"hand_landmarker.task")
    complete.write_text(json.dumps({"complete":True}),encoding="utf-8")
'''

VIDEO_EVAL = r'''probs=flow_model.predict(X_test,batch_size=BATCH_SIZE,verbose=0); pred=probs.argmax(axis=1)
metrics={"n_test":int(len(y_test)),"accuracy":float(accuracy_score(y_test,pred)),
         "macro_f1":float(f1_score(y_test,pred,average="macro",zero_division=0)),
         "classification_report":classification_report(y_test,pred,labels=[0,1,2],target_names=CLASSES,
             output_dict=True,zero_division=0)}
(EXPORT_DIR/"metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8")
pd.DataFrame({"video_rel":test_df.video_rel,"signer":test_df.signer,"true":[CLASSES[i] for i in y_test],
              "prediction":[CLASSES[i] for i in pred]}).to_csv(EXPORT_DIR/"test_predictions.csv",index=False)
cm=confusion_matrix(y_test,pred,labels=[0,1,2],normalize="true")
fig,ax=plt.subplots(figsize=(6,5)); ConfusionMatrixDisplay(cm,display_labels=CLASSES).plot(ax=ax,cmap="Blues",colorbar=False,values_format=".2f")
fig.tight_layout(); fig.savefig(EXPORT_DIR/"confusion_matrix.png",dpi=150); plt.show()
print({k:v for k,v in metrics.items() if k!="classification_report"})
print("Modelo y checkpoints:",EXPORT_DIR)
'''


VIDEO_BINARY_TRAIN = r'''BINARY_CLASSES=["J","Z"]
BINARY_TO_ID={name:i for i,name in enumerate(BINARY_CLASSES)}
train_df=split_df[split_df.split=="train"].copy()
val_df=split_df[split_df.split=="validation"].copy()
test_df=split_df[split_df.split=="test"].copy()
train_bin_df=train_df[train_df.label.isin(BINARY_CLASSES)].copy()
val_bin_df=val_df[val_df.label.isin(BINARY_CLASSES)].copy()
if set(BINARY_CLASSES)-set(train_bin_df.label):
    raise ValueError(f"Faltan clases J/Z en train: {train_bin_df.label.value_counts().to_dict()}")

def stack_binary(frame):
    X=np.stack([np.load(WORK_DIR/p) for p in frame.flow_rel])
    y=frame.label.map(BINARY_TO_ID).to_numpy(dtype=np.int32)
    return X,y

X_train,y_train=stack_binary(train_bin_df)
X_val,y_val=stack_binary(val_bin_df)
class_weights=compute_class_weight(class_weight="balanced",classes=np.arange(2),y=y_train)
CLASS_WEIGHTS={i:float(w) for i,w in enumerate(class_weights)}
print("Clips usados para entrenar (solo J/Z):",train_bin_df.label.value_counts().to_dict())
print("Clips J/Z usados para validar el clasificador:",val_bin_df.label.value_counts().to_dict())
print("Pesos de clase:",CLASS_WEIGHTS)

def build_binary_model():
    inputs=tf.keras.Input((SEQUENCE_FRAMES-1,FLOW_SIZE,FLOW_SIZE,2),name="optical_flow_sequence")
    x=tf.keras.layers.Conv3D(16,(3,5,5),strides=(1,2,2),padding="same",activation="relu")(inputs)
    x=tf.keras.layers.MaxPool3D((1,2,2))(x)
    x=tf.keras.layers.Conv3D(32,(3,3,3),padding="same",activation="relu")(x)
    x=tf.keras.layers.MaxPool3D((2,2,2))(x)
    x=tf.keras.layers.Conv3D(64,(3,3,3),padding="same",activation="relu")(x)
    x=tf.keras.layers.GlobalAveragePooling3D()(x); x=tf.keras.layers.Dropout(0.35)(x)
    outputs=tf.keras.layers.Dense(2,activation="softmax",name="jz_probs")(x)
    model=tf.keras.Model(inputs,outputs,name="jz_binary_optical_flow_cnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    return model

complete=EXPORT_DIR/"training_complete.json"
if TEMPORAL_MODEL.exists() and complete.exists():
    flow_model=tf.keras.models.load_model(TEMPORAL_MODEL,compile=False)
else:
    flow_model=build_binary_model()
    ckpt_dir=EXPORT_DIR/"checkpoints"; ckpt_dir.mkdir(parents=True,exist_ok=True)
    callbacks=[
        tf.keras.callbacks.BackupAndRestore(backup_dir=str(EXPORT_DIR/"backup"),save_freq="epoch"),
        tf.keras.callbacks.ModelCheckpoint(str(ckpt_dir/"best.keras"),monitor="val_loss",mode="min",save_best_only=True,verbose=1),
        tf.keras.callbacks.ModelCheckpoint(str(ckpt_dir/"epoch-{epoch:02d}.weights.h5"),save_weights_only=True,save_best_only=False,save_freq="epoch"),
        tf.keras.callbacks.CSVLogger(str(EXPORT_DIR/"history.csv"),append=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=6,restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=2,min_lr=1e-6),
    ]
    flow_model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=EPOCHS,batch_size=BATCH_SIZE,
                   class_weight=CLASS_WEIGHTS,callbacks=callbacks,verbose=2)
    best=ckpt_dir/"best.keras"
    if best.exists(): flow_model=tf.keras.models.load_model(best,compile=False)
    flow_model.save(TEMPORAL_MODEL)
    (EXPORT_DIR/"classes.json").write_text(json.dumps(BINARY_CLASSES,indent=2),encoding="utf-8")
    (EXPORT_DIR/"metadata.json").write_text(json.dumps({"classes":BINARY_CLASSES,"sequence_frames":SEQUENCE_FRAMES,
        "flow_size":FLOW_SIZE,"flow_clip_pixels":FLOW_CLIP_PIXELS,"training_labels":BINARY_CLASSES,
        "rejection":"max binary softmax probability below calibrated threshold"},indent=2),encoding="utf-8")
    shutil.copy2(MODEL_ASSET,EXPORT_DIR/"hand_landmarker.task")
    complete.write_text(json.dumps({"complete":True}),encoding="utf-8")
'''

VIDEO_BINARY_CALIBRATE = r'''# Umbral seleccionado SOLO con validation. `other` no entrena el clasificador;
# sirve para calibrar el rechazo de secuencias que no son J ni Z.
def binary_predictions(frame):
    X=np.stack([np.load(WORK_DIR/p) for p in frame.flow_rel])
    probs=flow_model.predict(X,batch_size=BATCH_SIZE,verbose=0)
    return probs

val_probs=binary_predictions(val_df)
val_conf=val_probs.max(axis=1); val_choice=val_probs.argmax(axis=1)
val_true=val_df.label.to_numpy()
threshold_rows=[]
for threshold in np.arange(0.50,0.991,0.01):
    pred=np.where(val_conf>=threshold,np.asarray(BINARY_CLASSES)[val_choice],"other")
    threshold_rows.append({"threshold":float(threshold),
        "macro_f1":float(f1_score(val_true,pred,labels=["J","Z","other"],average="macro",zero_division=0)),
        "accuracy":float(accuracy_score(val_true,pred)),
        "other_false_accept_rate":float(np.mean(pred[val_true=="other"]!="other")) if np.any(val_true=="other") else np.nan,
        "jz_coverage":float(np.mean(pred[val_true!="other"]!="other")) if np.any(val_true!="other") else np.nan})
threshold_table=pd.DataFrame(threshold_rows)
chosen=threshold_table.sort_values(["macro_f1","threshold"],ascending=[False,False]).iloc[0]
REJECT_THRESHOLD=float(chosen.threshold)
print("Umbral elegido en validation:",REJECT_THRESHOLD)
display(chosen.to_frame("validation").T)
display(threshold_table.sort_values("threshold"))
(EXPORT_DIR/"reject_threshold.json").write_text(json.dumps({"threshold":REJECT_THRESHOLD,
    "selection_metric":"validation macro F1 over J, Z, other",
    "validation_metrics":chosen.to_dict()},ensure_ascii=False,indent=2),encoding="utf-8")

test_probs=binary_predictions(test_df)
test_conf=test_probs.max(axis=1); test_choice=test_probs.argmax(axis=1)
test_pred=np.where(test_conf>=REJECT_THRESHOLD,np.asarray(BINARY_CLASSES)[test_choice],"other")
test_true=test_df.label.to_numpy()
metrics={"n_test":int(len(test_df)),"reject_threshold":REJECT_THRESHOLD,
    "accuracy":float(accuracy_score(test_true,test_pred)),
    "macro_f1":float(f1_score(test_true,test_pred,labels=["J","Z","other"],average="macro",zero_division=0)),
    "classification_report":classification_report(test_true,test_pred,labels=["J","Z","other"],
        target_names=["J","Z","other"],output_dict=True,zero_division=0)}
(EXPORT_DIR/"metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8")
pd.DataFrame({"video_rel":test_df.video_rel,"signer":test_df.signer,"true":test_true,
    "prediction":test_pred,"max_binary_confidence":test_conf,
    "binary_choice":np.asarray(BINARY_CLASSES)[test_choice]}).to_csv(EXPORT_DIR/"test_predictions.csv",index=False)
cm=confusion_matrix(test_true,test_pred,labels=["J","Z","other"],normalize="true")
fig,ax=plt.subplots(figsize=(6,5)); ConfusionMatrixDisplay(cm,display_labels=["J","Z","other"]).plot(
    ax=ax,cmap="Blues",colorbar=False,values_format=".2f")
fig.tight_layout(); fig.savefig(EXPORT_DIR/"confusion_matrix.png",dpi=150); plt.show()
print({k:v for k,v in metrics.items() if k!="classification_report"})
print("Modelo binario, umbral calibrado, checkpoints y métricas:",EXPORT_DIR)
'''


VIDEO_MAP_PREPROCESS = r'''VIDEO_EXTENSIONS={".mp4",".avi",".mov",".mkv"}
MAP_SIZE=96; MIN_DETECTED_FRAMES=12
MAP_FINGERS=[[(0,1),(1,2),(2,3),(3,4)],[(0,5),(5,6),(6,7),(7,8)],
             [(0,9),(9,10),(10,11),(11,12)],[(0,13),(13,14),(14,15),(15,16)],
             [(0,17),(17,18),(18,19),(19,20)]]
MAP_COLORS=[(255,90,90),(255,210,40),(80,235,100),(50,190,255),(220,90,255)]
MAP_PALM_EDGES=[(5,9),(9,13),(13,17)]
MAP_POINT_COLORS=[MAP_COLORS[0]]*4+[MAP_COLORS[1]]*4+[MAP_COLORS[2]]*4+[MAP_COLORS[3]]*4+[MAP_COLORS[4]]*4

def canonical_label(name):
    value=name.strip().lower()
    return {"j":"J","z":"Z","other":"other","none":"other","background":"other"}.get(value)

def sample_frames(path,count=SEQUENCE_FRAMES):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): cap.release(); return None
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total>=2:
        indices=np.linspace(0,total-1,count).round().astype(int); frames=[]
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx)); ok,frame=cap.read()
            if not ok: frames=[]; break
            frames.append(frame)
        cap.release()
        if len(frames)==count: return frames
    else: cap.release()
    # Sequential fallback for videos with missing or inaccurate frame metadata.
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): cap.release(); return None
    total=0
    while True:
        ok,_=cap.read()
        if not ok: break
        total+=1
    cap.release()
    if total<2: return None
    wanted=set(np.linspace(0,total-1,count).round().astype(int).tolist())
    selected={}; cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): cap.release(); return None
    i=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        if i in wanted: selected[i]=frame.copy()
        i+=1
    cap.release()
    if not wanted.issubset(selected): return None
    return [selected[int(i)] for i in np.linspace(0,total-1,count).round().astype(int)]

def detect_points(frames,detector):
    seq=[]
    for frame in frames:
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        result=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
        if result.hand_landmarks:
            h,w=frame.shape[:2]
            # Use a square-normalized image coordinate system to preserve aspect ratio.
            denom=float(max(h,w))
            seq.append(np.asarray([(p.x*w/denom,p.y*h/denom,p.z) for p in result.hand_landmarks[0]],dtype=np.float32))
        else: seq.append(None)
    detected=sum(p is not None for p in seq)
    if detected<MIN_DETECTED_FRAMES: return None,f"MediaPipe detectó {detected}/{SEQUENCE_FRAMES}; mínimo {MIN_DETECTED_FRAMES}"
    valid=np.asarray([i for i,p in enumerate(seq) if p is not None],dtype=np.float32)
    stacked=np.stack([p for p in seq if p is not None])
    full=np.empty((SEQUENCE_FRAMES,21,3),dtype=np.float32)
    for point_idx in range(21):
        for axis in range(3):
            full[:,point_idx,axis]=np.interp(np.arange(SEQUENCE_FRAMES),valid,stacked[:,point_idx,axis])
    return full,""

def render_map_sequence(points_seq):
    # One clip-level origin and scale preserve fingertip/wrist trajectories across frames.
    wrist=points_seq[:,0,:2]
    center=wrist.mean(axis=0)
    palm=np.linalg.norm(points_seq[:,9,:2]-points_seq[:,0,:2],axis=1)
    scale=max(float(np.median(palm)),1e-5)*3.0
    xy=(points_seq[:,:,:2]-center[None,None,:])/scale+0.5
    xy=np.clip(xy,0.0,1.0)*(MAP_SIZE-1)
    maps=[]
    for frame_points in xy:
        image=np.zeros((MAP_SIZE,MAP_SIZE,3),dtype=np.uint8)
        coords=np.rint(frame_points).astype(np.int32)
        for edges,color in zip(MAP_FINGERS,MAP_COLORS):
            for start,end in edges:
                cv2.line(image,tuple(coords[start]),tuple(coords[end]),color,2,cv2.LINE_AA)
        for start,end in MAP_PALM_EDGES:
            cv2.line(image,tuple(coords[start]),tuple(coords[end]),(235,235,235),2,cv2.LINE_AA)
        for idx,(x,y) in enumerate(coords):
            color=(255,255,255) if idx==0 else MAP_POINT_COLORS[idx-1]
            cv2.circle(image,(int(x),int(y)),2,color,-1,cv2.LINE_AA)
        maps.append(image)
    return np.stack(maps).astype(np.uint8)

def prepare_landmark_maps():
    old={}
    if MANIFEST.exists():
        old_df=pd.read_csv(MANIFEST).drop_duplicates("video_rel",keep="last")
        old={str(r.video_rel):r._asdict() for r in old_df.itertuples(index=False)}
    files=sorted(p for p in VIDEO_ROOT.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    if not files: raise ValueError("No hay videos compatibles en VIDEO_ROOT")
    header=not MANIFEST.exists() or MANIFEST.stat().st_size==0
    options=mp.tasks.vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_ASSET)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,num_hands=1,min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5)
    detector=mp.tasks.vision.HandLandmarker.create_from_options(options); rows=[]
    try:
        with MANIFEST.open("a",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=["video_rel","label","signer","map_rel","ok","error"])
            if header: writer.writeheader()
            for i,path in enumerate(files,1):
                rel=path.relative_to(VIDEO_ROOT); parts=rel.parts
                label=canonical_label(parts[0]) if len(parts)>=3 else None
                if label is None: warnings.warn(f"Ignoro ruta; debe ser clase/signante/video: {rel}"); continue
                signer=parts[1]
                map_rel=f"maps/{label}/{signer}/{hashlib.sha1(rel.as_posix().encode()).hexdigest()}.npy"
                output=WORK_DIR/map_rel; previous=old.get(rel.as_posix())
                if previous and output.is_file() and int(previous["ok"])==1:
                    rows.append(previous); continue
                output.parent.mkdir(parents=True,exist_ok=True)
                frames=sample_frames(path)
                if frames is None: sequence,error=None,"video ilegible o con menos de dos fotogramas"
                else:
                    points,error=detect_points(frames,detector)
                    sequence=render_map_sequence(points) if points is not None else None
                ok=sequence is not None
                if ok: np.save(output,sequence)
                record={"video_rel":rel.as_posix(),"label":label,"signer":signer,
                    "map_rel":map_rel,"ok":int(ok),"error":error}
                writer.writerow(record); rows.append(record); stream.flush()
                if i%10==0 or i==len(files): print(f"{i}/{len(files)} | {rel} | {'OK' if ok else error}")
    finally: detector.close()
    return pd.DataFrame(rows).drop_duplicates("video_rel",keep="last")

manifest_df=prepare_landmark_maps()
print("Conteos procesados:")
display(manifest_df[manifest_df.ok.astype(int)==1].groupby(["label","signer"]).size().unstack(fill_value=0))
'''


VIDEO_MAP_TRAIN = r'''train_df=split_df[split_df.split=="train"].copy()
val_df=split_df[split_df.split=="validation"].copy()
test_df=split_df[split_df.split=="test"].copy()

def stack_maps(frame):
    X=np.stack([np.load(WORK_DIR/p) for p in frame.map_rel])
    y=frame.label.map(CLASS_TO_ID).to_numpy(dtype=np.int32)
    return X,y

X_train,y_train=stack_maps(train_df); X_val,y_val=stack_maps(val_df)
class_weights=compute_class_weight(class_weight="balanced",classes=np.arange(len(CLASSES)),y=y_train)
CLASS_WEIGHTS={i:float(w) for i,w in enumerate(class_weights)}
print("Clips:",{"train":len(train_df),"validation":len(val_df)})
print("Pesos de clase:",CLASS_WEIGHTS)

def build_map_sequence_model():
    inputs=tf.keras.Input((SEQUENCE_FRAMES,MAP_SIZE,MAP_SIZE,3),name="hand_landmark_map_sequence",dtype=tf.uint8)
    x=tf.keras.layers.TimeDistributed(tf.keras.layers.Rescaling(1./255))(inputs)
    for filters in (16,32,64):
        x=tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters,3,strides=2,padding="same",use_bias=False))(x)
        x=tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
        x=tf.keras.layers.TimeDistributed(tf.keras.layers.Activation("relu"))(x)
    x=tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x)
    x=tf.keras.layers.GRU(48,dropout=0.25,return_sequences=False,name="temporal_gru")(x)
    x=tf.keras.layers.Dropout(0.35)(x)
    outputs=tf.keras.layers.Dense(len(CLASSES),activation="softmax",name="class_probs")(x)
    model=tf.keras.Model(inputs,outputs,name="jz_landmark_map_cnn_gru")
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    return model

complete=EXPORT_DIR/"training_complete.json"
if TEMPORAL_MODEL.exists() and complete.exists():
    map_model=tf.keras.models.load_model(TEMPORAL_MODEL,compile=False)
else:
    map_model=build_map_sequence_model()
    ckpt_dir=EXPORT_DIR/"checkpoints"; ckpt_dir.mkdir(parents=True,exist_ok=True)
    callbacks=[
        tf.keras.callbacks.BackupAndRestore(backup_dir=str(EXPORT_DIR/"backup"),save_freq="epoch"),
        tf.keras.callbacks.ModelCheckpoint(str(ckpt_dir/"best.keras"),monitor="val_loss",mode="min",save_best_only=True,verbose=1),
        tf.keras.callbacks.ModelCheckpoint(str(ckpt_dir/"epoch-{epoch:02d}.weights.h5"),save_weights_only=True,save_best_only=False,save_freq="epoch"),
        tf.keras.callbacks.CSVLogger(str(EXPORT_DIR/"history.csv"),append=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=7,restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=3,min_lr=1e-6),
    ]
    history=map_model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=45,batch_size=8,
        class_weight=CLASS_WEIGHTS,callbacks=callbacks,verbose=2)
    best=ckpt_dir/"best.keras"
    if best.exists(): map_model=tf.keras.models.load_model(best,compile=False)
    map_model.save(TEMPORAL_MODEL)
    (EXPORT_DIR/"classes.json").write_text(json.dumps(CLASSES,indent=2),encoding="utf-8")
    (EXPORT_DIR/"metadata.json").write_text(json.dumps({"classes":CLASSES,"sequence_frames":SEQUENCE_FRAMES,
        "map_size":MAP_SIZE,"representation":"clip-normalized landmark map sequence",
        "model":"TimeDistributed CNN + GRU","preserves_motion":True},indent=2),encoding="utf-8")
    shutil.copy2(MODEL_ASSET,EXPORT_DIR/"hand_landmarker.task")
    complete.write_text(json.dumps({"complete":True}),encoding="utf-8")
'''


VIDEO_MAP_EVAL = r'''X_test,y_test=stack_maps(test_df)
probs=map_model.predict(X_test,batch_size=8,verbose=0); pred=probs.argmax(axis=1)
metrics={"n_test":int(len(y_test)),"accuracy":float(accuracy_score(y_test,pred)),
    "macro_f1":float(f1_score(y_test,pred,average="macro",zero_division=0)),
    "classification_report":classification_report(y_test,pred,labels=[0,1,2],target_names=CLASSES,
        output_dict=True,zero_division=0)}
(EXPORT_DIR/"metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8")
pd.DataFrame({"video_rel":test_df.video_rel,"signer":test_df.signer,
    "true":[CLASSES[i] for i in y_test],"prediction":[CLASSES[i] for i in pred]}).to_csv(
        EXPORT_DIR/"test_predictions.csv",index=False)
cm=confusion_matrix(y_test,pred,labels=[0,1,2],normalize="true")
fig,ax=plt.subplots(figsize=(6,5)); ConfusionMatrixDisplay(cm,display_labels=CLASSES).plot(
    ax=ax,cmap="Blues",colorbar=False,values_format=".2f")
fig.tight_layout(); fig.savefig(EXPORT_DIR/"confusion_matrix.png",dpi=150); plt.show()
print({k:v for k,v in metrics.items() if k!="classification_report"})
print("Modelo y checkpoints:",EXPORT_DIR)
'''


def main():
    make_static_notebook("maps")
    make_static_notebook("crop")
    video_cells=[
        markdown("# Entrenamiento temporal — J / Z / other\n\nVideo completo entra como secuencia; MediaPipe localiza la mano, se genera un recorte fijo del clip, se calcula flujo óptico Farneback y se clasifica con CNN 3D. El dataset debe estar en `proyecto-ia2/data/videos_jz/{J,Z,other}/<signer_id>/<clip>`. Se conserva el signante para separar train/validation/test por persona.\n\n**Reanudación:** extracción de flujo incremental con manifiesto en Drive; `BackupAndRestore`, checkpoint por época y mejor modelo. Si se interrumpe, vuelve a ejecutar configuración, preprocesamiento y entrenamiento. No borres `backup/`, `checkpoints/` ni `split_balanced_v2.csv`. Este notebook busca un reparto por signante cercano a 70/15/15 para cada clase y exige al menos cinco clips por clase en validation y test. La carpeta de exportación v2 evita reutilizar por accidente el modelo anterior."),
        code(VIDEO_SETUP), markdown("## Procesar videos a flujos ópticos\n\nLee `.mp4`, `.avi`, `.mov` y `.mkv`. Cada video se reduce a 17 fotogramas uniformes y produce 16 mapas de flujo de 128×128×2. Si los metadatos de fotogramas son incorrectos o no se puede buscar por índice, hace dos pasadas secuenciales para muestrear el clip. Se conserva un clip si MediaPipe detecta la mano en al menos 12 de los 17 fotogramas; los fotogramas con detección fallida no descartan por sí solos toda la secuencia."),
        code(VIDEO_PREPROCESS), markdown("## Split por signante y entrenamiento\n\nSe intenta construir un split por signante con las tres etiquetas presentes en train, validation y test. Un clip se conserva si MediaPipe detecta mano en al menos 12 de los 17 fotogramas; los fotogramas con detección fallida se mantienen en el cálculo de flujo. La CNN 3D usa pesos de clase calculados sobre train. Si el split no es posible, agrega variedad real por clase/persona; no asignes un ID inventado para simular signantes."),
        code(VIDEO_TRAIN), markdown("## Evaluación final\n\nLas métricas se calculan en signantes apartados. Debido al tamaño reducido y al origen heterogéneo de `other`, interpreta estos resultados como evaluación preliminar."), code(VIDEO_EVAL)
    ]
    write_notebook("12_entrenamiento_video_flujo_colab.ipynb",video_cells)

    binary_setup=VIDEO_SETUP.replace(
        'EXPORT_DIR=DRIVE_ROOT/"exports/proyecto_def/video_balanced_v2"',
        'EXPORT_DIR=DRIVE_ROOT/"exports/proyecto_def/video_jz_reject_v1"')
    binary_setup=binary_setup.replace(
        'SPLIT_FILE=WORK_DIR/"split_balanced_v2.csv"',
        'SPLIT_FILE=WORK_DIR/"split_jz_reject_v1.csv"')
    binary_setup=binary_setup.replace(
        'TEMPORAL_MODEL=EXPORT_DIR/"jz_flow_cnn.keras"',
        'TEMPORAL_MODEL=EXPORT_DIR/"jz_flow_binary.keras"')
    split_logic=VIDEO_TRAIN.split("def stack_split(frame):")[0]
    binary_train=split_logic+VIDEO_BINARY_TRAIN
    binary_cells=[
        markdown("# Experimento temporal — J/Z con rechazo\n\nEste notebook compara un clasificador binario entrenado exclusivamente con secuencias J y Z. Los clips `other` no forman parte del entrenamiento: se reservan para calibrar el umbral de rechazo y para evaluar cuántos movimientos ajenos se aceptan por error. MediaPipe localiza la mano, se calcula Farneback y una CNN 3D clasifica la secuencia.\n\n**Importante:** el softmax binario siempre favorece J o Z; la salida `other` se crea mediante un umbral calibrado con validation. Esto no equivale a reconocer todas las señas. Mantén signantes separados entre particiones.\n\n**Rutas nuevas:** usa `split_jz_reject_v1.csv` y `exports/proyecto_def/video_jz_reject_v1` para no reutilizar el split ni checkpoints del modelo de tres clases. El preprocesamiento reutiliza los flujos existentes si los archivos conservan las mismas rutas y contenido. Si reemplazaste un video usando exactamente el mismo nombre/ruta, cambia la versión de `WORK_DIR` o fuerza su reprocesamiento para no tomar el flujo anterior."),
        code(binary_setup),
        markdown("## Preprocesamiento incremental\n\nProcesa videos nuevos y reutiliza los flujos guardados de clips válidos. Revisa el conteo por signante y etiqueta antes de continuar."),
        code(VIDEO_PREPROCESS),
        markdown("## Split por signante y entrenamiento binario\n\nSe conserva un split de tres etiquetas para que validation y test tengan negativos `other` de signantes no vistos. El ajuste de pesos se hace únicamente con J/Z de train; `other` no participa en el gradiente. El entrenamiento guarda checkpoints por época y respaldo para recuperación en Drive."),
        code(binary_train),
        markdown("## Calibrar rechazo y evaluación final\n\nEl umbral se escoge con validation buscando el mejor macro F1 de J, Z y `other`. Después se calcula la evaluación en test una sola vez. Revisa también `other_false_accept_rate`, la cobertura J/Z y la matriz de confusión: un rechazo excesivo puede mejorar el F1 de `other` a costa de dejar sin reconocer muchas J/Z."),
        code(VIDEO_BINARY_CALIBRATE),
    ]
    write_notebook("13_entrenamiento_video_jz_rechazo_colab.ipynb",binary_cells)

    map_setup=VIDEO_SETUP.replace(
        'WORK_DIR=DRIVE_ROOT/"data/processed/proyecto_def/video"',
        'WORK_DIR=DRIVE_ROOT/"data/processed/proyecto_def/video_map_sequence_v1"')
    map_setup=map_setup.replace(
        'EXPORT_DIR=DRIVE_ROOT/"exports/proyecto_def/video_balanced_v2"',
        'EXPORT_DIR=DRIVE_ROOT/"exports/proyecto_def/video_map_sequence_v1"')
    map_setup=map_setup.replace(
        'SPLIT_FILE=WORK_DIR/"split_balanced_v2.csv"',
        'SPLIT_FILE=WORK_DIR/"split_map_sequence_v1.csv"')
    map_setup=map_setup.replace(
        'TEMPORAL_MODEL=EXPORT_DIR/"jz_flow_cnn.keras"',
        'TEMPORAL_MODEL=EXPORT_DIR/"jz_landmark_map_gru.keras"')
    map_split_logic=VIDEO_TRAIN.split("def stack_split(frame):")[0]
    map_train=map_split_logic+VIDEO_MAP_TRAIN
    map_cells=[
        markdown("# Video J/Z/other con mapas de landmarks\n\nEste experimento sustituye el flujo óptico por una secuencia de mapas RGB dibujados desde los 21 landmarks de MediaPipe. Cada clip genera 17 mapas de 96×96; la normalización usa un origen y escala comunes para todo el clip, preservando la trayectoria entre fotogramas. Un CNN pequeño extrae la postura por fotograma y una GRU aprende el movimiento temporal. Se conserva el split por signante y se guarda en rutas nuevas, sin sobrescribir los otros experimentos.\n\n**Limitación:** la detección de landmarks sigue siendo necesaria. Si falla en más de 5 de los 17 fotogramas, el clip se excluye; los fallos restantes se interpolan. `other` continúa significando los negativos presentes en el dataset, no todas las señas posibles."),
        code(map_setup),
        markdown("## Preprocesar videos a secuencias de mapas\n\nEl manifiesto y los mapas se guardan incrementalmente en Drive. Si agregas clips nuevos, vuelve a ejecutar la celda. Si reemplazas un video con el mismo nombre y ruta, cambia la versión de WORK_DIR o elimina su mapa/manifiesto asociado para forzar el reprocesamiento."),
        code(VIDEO_MAP_PREPROCESS),
        markdown("## Split por signante y entrenamiento\n\nEl split busca mantener signantes separados y las tres clases presentes en las particiones. El modelo usa pesos de clase, checkpoints por época y `BackupAndRestore` en Drive para reanudar después de una desconexión."),
        code(map_train),
        markdown("## Evaluación final\n\nEl test se evalúa una sola vez con signantes apartados. Compara accuracy, macro F1, reporte por clase y matriz de confusión con los resultados de flujo óptico; las métricas siguen siendo preliminares por el tamaño y la distribución de signantes."),
        code(VIDEO_MAP_EVAL),
    ]
    write_notebook("14_entrenamiento_video_mapas_colab.ipynb",map_cells)


if __name__ == "__main__":
    main()
