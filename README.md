# Clasificación de señas del alfabeto ASL

Proyecto académico de IA2 para clasificar imágenes de señas del alfabeto de la Lengua de Señas Americana (ASL) mediante redes neuronales convolucionales. La primera meta es construir y evaluar una línea base reproducible con imágenes estáticas. El reconocimiento dinámico y en tiempo real queda como trabajo posterior, sujeto al tiempo y a los recursos disponibles.

## Estado

El proyecto ya cuenta con exploración y preprocesamiento, una CNN baseline y experimentos de transferencia de aprendizaje en `notebooks/03_entrenamiento_baseline.ipynb` y `notebooks/04_transfer_learning_comparacion.ipynb`. El siguiente experimento controla el preprocesamiento: localizar y recortar la mano con MediaPipe antes de pasar la imagen RGB a una CNN.

## Dataset

Dataset inicial: [ASL Alphabet en Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).

1. Descarga el dataset desde Kaggle y extrae los archivos en `data/raw/`.
2. No subas imágenes del dataset a GitHub. La carpeta `data/raw/` está excluida por `.gitignore`.
3. Abre `notebooks/01_exploracion_datos.ipynb` y configura allí la ruta local si la estructura extraída lo requiere.
4. Registra en el notebook la estructura, conteos y observaciones reales; no des por hecho nombres de carpetas o cantidad de imágenes.

Cada integrante necesita obtener su propia copia local del dataset o coordinar un almacenamiento compartido aprobado. El código debe trabajar con rutas configurables y relativas al repositorio, no con rutas absolutas de una computadora.

## Entorno local

Se recomienda Python 3.10 o 3.11 de 64 bits y un entorno virtual por integrante. Desde la raíz del repositorio:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

En VS Code, selecciona el intérprete de `.venv` como kernel del notebook. El proyecto no requiere Docker para comenzar; el equipo puede compartir el mismo `requirements.txt` y las instrucciones anteriores.

> TensorFlow puede ejecutarse en Windows nativo con CPU. Para usar GPU NVIDIA con TensorFlow en Windows se requiere WSL2; las versiones posteriores a TensorFlow 2.10 no ofrecen soporte GPU nativo para Windows. El entrenamiento también puede trasladarse después a Colab o al clúster si se confirma el acceso.

## Estructura

```text
configs/       Configuración de experimentos
data/raw/      Dataset original local (ignorado por Git)
data/processed/ Datos derivados locales (ignorados por Git)
models/        Pesos y modelos entrenados (ignorados por Git)
notebooks/     Exploración y experimentos documentados
results/       Métricas y figuras de experimentos (revisar antes de subir)
src/           Código reutilizable del proyecto
test/          Pruebas del proyecto
```

## Experimento: recorte de mano + CNN

El recorte se prepara fuera del entrenamiento para poder reutilizar los mismos archivos con varias arquitecturas. Las configuraciones `baseline_handcrop.yaml`, `mobilenetv2_handcrop.yaml` y `efficientnetb0_handcrop.yaml` separan cada corrida; el notebook `05_mediapipe_cnn.ipynb` documenta el orden de comparación.

```bash
python scripts/download_hand_model.py

# Preprocesar el dataset original una vez (puede tardar en una CPU)
python scripts/prepare_hand_crops.py \
  --input data/raw/asl_alphabet_train/asl_alphabet_train \
  --output data/processed/hand_crops/asl_alphabet_train \
  --manifest data/processed/hand_crops/asl_alphabet_train_manifest.csv

# Entrenar el mismo baseline sobre las imágenes recortadas
python -m src.train --config configs/baseline_handcrop.yaml --name baseline_handcrop

# Las CNN preentrenadas usan los mismos recortes
python -m src.train --config configs/mobilenetv2_handcrop.yaml --name mobilenetv2_handcrop
python -m src.train --config configs/efficientnetb0_handcrop.yaml --name efficientnetb0_handcrop

# Preprocesar fotos externas y evaluar el flujo completo
python scripts/prepare_hand_crops.py --input data/mis_fotos \
  --output data/processed/hand_crops/mis_fotos \
  --manifest data/processed/hand_crops/mis_fotos_manifest.csv
python scripts/evaluate_hand_crop.py --run experiments/AAAAMMDD-HHMM_baseline_handcrop
```

El manifiesto registra qué imágenes tuvieron mano detectada. Las muestras sin detección se excluyen del ajuste y la validación del clasificador (salvo `nothing`); en la métrica end-to-end, cualquier seña sin detección se predice como `nothing` y cuenta como error. Por eso cada corrida guarda tanto las métricas condicionales del clasificador como `pipeline_metrics.json`, la matriz end-to-end y predicciones por imagen. El CSV de fotos externas queda dentro de la carpeta de la corrida.

## Plan de trabajo

1. Comparar CNN baseline de imagen completa contra la misma CNN con recorte MediaPipe.
2. Evaluar MobileNetV2 y EfficientNetB0 sobre los mismos recortes.
3. Comparar las métricas internas y las fotos externas, incluida la tasa de detección.
4. Después de estabilizar la comparación estática, explorar mapas de puntos y reconocimiento temporal.

No se deben subir al repositorio datasets completos, credenciales, entornos virtuales ni archivos grandes de entrenamiento. Las métricas y figuras que se decida conservar para documentar experimentos deben revisarse antes de agregarlas a Git.

## Prototipo con puntos de referencia de MediaPipe

Este experimento extrae 21 puntos de la mano por imagen, los centra en la muñeca y los escala por el tamaño de la palma. Luego entrena un MLP con las características del dataset original y puede evaluar fotos propias sin usarlas para entrenar.

En este dataset la clase `nothing` está representada por imágenes sin mano. La regla del experimento asigna esa clase cuando MediaPipe no detecta una mano; si no detecta una mano en otra clase, esa predicción cuenta como error en la evaluación.

```bash
python scripts/download_hand_model.py

# Prueba inicial: 5 imágenes escogidas al azar por clase
python scripts/extract_hand_landmarks.py \
  --input data/raw/asl_alphabet_train/asl_alphabet_train \
  --output data/processed/landmarks_pilot.npz \
  --max-per-class 5

# Lote completo del dataset de entrenamiento
python scripts/extract_hand_landmarks.py \
  --input data/raw/asl_alphabet_train/asl_alphabet_train \
  --output data/processed/train_landmarks.npz

# Después de copiar las fotos externas a data/mis_fotos/<clase>/
python scripts/extract_hand_landmarks.py \
  --input data/mis_fotos \
  --output data/processed/own_photos_landmarks.npz

# Entrenar y evaluar; las fotos externas solo se usan para evaluación
python scripts/train_landmark_classifier.py \
  --features data/processed/train_landmarks.npz \
  --external-features data/processed/own_photos_landmarks.npz
```

Las imágenes, el archivo `.task`, las características `.npz` y el modelo quedan bajo carpetas excluidas de Git. El script guarda las métricas en `results/metrics/landmark_classifier.json`; revísalas antes de decidir si agregarlas al repositorio. Las pruebas de reconocimiento con fotos propias deben reportarse aparte del test interno del dataset.
