# Proyecto definitivo — ASL fingerspelling

El entrenamiento se ejecuta en Colab y guarda modelos, checkpoints, manifiestos y métricas en Google Drive. La aplicación local abre la cámara y carga los modelos exportados. El alcance es deletreo de letras ASL y sugerencia ortográfica en inglés; no es traducción completa de ASL.

## Notebooks de Colab

Sube estos archivos de `notebooks/` a Colab. Los tres notebooks son independientes:

1. `10_entrenamiento_mapas_colab.ipynb`: modelo estático principal. MediaPipe detecta 21 landmarks, los convierte en un mapa RGB y MobileNetV2 los clasifica.
2. `12_entrenamiento_video_flujo_colab.ipynb`: rama temporal J/Z/other. MediaPipe recorta una región común de la mano, Farneback calcula el flujo óptico y una CNN 3D clasifica la secuencia.
3. `13_entrenamiento_video_jz_rechazo_colab.ipynb`: experimento binario J/Z con rechazo; entrena solo con J/Z y calibra el umbral de “ninguna” con `other` en validation.
4. `14_entrenamiento_video_mapas_colab.ipynb`: experimento temporal con 17 mapas de landmarks por video y CNN + GRU. Usa una transformación común por secuencia para preservar la trayectoria.
5. `11_entrenamiento_recortes_colab.ipynb`: alternativa/experimento de comparación para la rama estática. MediaPipe recorta la mano y MobileNetV2 clasifica el recorte. No se ejecuta al mismo tiempo que mapas en la app; se selecciona una rama estática.

Para obtener el primer prototipo integrado, ejecuta 10 y 12. Ejecuta 13 como comparación experimental del clasificador temporal con rechazo; no reemplaza automáticamente el modelo que usa la app. Ejecuta 14 para comparar la representación temporal basada en landmarks. Ejecuta 11 para comparar la representación de recorte con mapas. Se puede entrenar en sesiones distintas o en paralelo si Colab tiene recursos disponibles.

## Rutas esperadas en Drive

Los notebooks esperan una carpeta `proyecto-ia2` en `Mi unidad`:

```text
proyecto-ia2/
├── dataset_asl/final_dataset/<clase>/<imagen>
├── data/videos_jz/
│   ├── J/<signer_id>/<video>
│   ├── Z/<signer_id>/<video>
│   └── other/<signer_id>/<video>
├── data/raw/                       # dataset ASL de 87k, opcional como comparación
└── exports/proyecto_def/            # modelos y checkpoints generados
```

El dataset integrado variado es la fuente estática predeterminada y limita cada clase a 1.724 imágenes. No se concatena automáticamente con las 87k imágenes raw: primero compara su procedencia y duplicados. J/Z se excluyen de clasificación estática y se dejan a la rama temporal. Se incluyen `space`, `nothing` y `del` si existen como carpetas; `del` funciona como retroceso.

En `data/videos_jz`, la carpeta de segundo nivel debe identificar consistentemente a una persona. Por ejemplo, `J/participant_03`, `Z/participant_03` y `other/participant_03` se consideran un mismo signante. Para WLASL, usa el identificador de signante de sus metadatos si está disponible. El notebook 12 busca un split por signante cercano a 70/15/15 por clase y requiere al menos cinco videos por clase en validación y prueba. Si los signantes disponibles no permiten cumplirlo, se detiene e informa los grupos.

## Checkpoints y reanudación

El preprocesamiento estático y temporal usa manifiestos incrementales en Drive y puede continuar después de volver a ejecutar la celda. El entrenamiento guarda:

- `BackupAndRestore` por fase (`backup/`) para recuperar el estado al interrumpirse Colab.
- Pesos de cada época y el mejor modelo en `checkpoints/`.
- Historial, split y predicciones para reproducir el experimento.

No borres `backup/`, `checkpoints/`, `split.csv` ni los datos procesados si necesitas reanudar. Tras una desconexión, vuelve a montar Drive y ejecuta las celdas de configuración, preprocesamiento y entrenamiento. Los checkpoints viven en Drive, no en el disco efímero de Colab.

## Archivos para descargar a la aplicación local

Después de entrenar, copia estos archivos desde Drive a las rutas indicadas:

```text
proyecto_def/artifacts/
├── hand_landmarker.task
├── maps/
│   ├── model.keras
│   └── classes.json
├── crop/
│   ├── model.keras
│   └── classes.json
└── video/
    └── jz_flow_cnn.keras
```

Rutas de origen en Drive:

- Estático mapas: `exports/proyecto_def/maps/integrated_varied_1724/model.keras` y `classes.json`.
- Estático recortes: `exports/proyecto_def/crop/integrated_varied_1724/model.keras` y `classes.json`.
- Temporal: `exports/proyecto_def/video_balanced_v3/jz_flow_cnn.keras` (experimento de tres clases con split actualizado; si tu Drive conserva otra versión, usa la carpeta del entrenamiento que vas a evaluar).
- Temporal binario con rechazo: `exports/proyecto_def/video_jz_reject_v1/jz_flow_binary.keras`, junto con `classes.json` y `reject_threshold.json`. Es salida experimental; todavía requiere integrarse en la app local.
- Temporal mapas: `exports/proyecto_def/video_map_sequence_v1/jz_landmark_map_gru.keras`, junto con `classes.json`, `metadata.json` y `hand_landmarker.task`. Es salida experimental; la inferencia local requiere usar exactamente el mismo renderizado temporal.
- MediaPipe: `exports/proyecto_def/hand_landmarker.task` (incluido también en las carpetas de exportación después de completar cada entrenamiento).

La app necesita los dos archivos de la rama estática que vayas a seleccionar. Para el pipeline integrado también necesita el modelo temporal. El archivo `classes.json` conserva el orden de salida de la capa softmax. Los datos originales, los mapas/crops derivados, los flujos ópticos y los checkpoints no se necesitan para inferencia local.

## Ejecutar la app local

Desde esta carpeta, con Python 3.10/3.11 recomendado:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-local.txt
python app.py
```

Antes de descargar los modelos, puedes verificar solo la interfaz y la cámara con `python app.py --preview-only`. Para instalar únicamente lo que necesita esa vista previa, usa `python -m pip install -r requirements-preview.txt`. Este modo no carga TensorFlow/MediaPipe, no clasifica señas ni guarda texto.

La ventana muestra la vista previa y la predicción en vivo. La rama temporal se arma al estabilizar una postura inicial candidata de J (meñique extendido) o Z (índice extendido), y se activa cuando empieza a moverse ese dedo; mientras clasifica el movimiento, la predicción estática se suprime. Una postura estática candidata expira y vuelve a la clasificación estática. Estos umbrales geométricos son heurísticos iniciales y deben ajustarse probando la cámara. El menú permite elegir mapas o recortes. **Iniciar captura** comienza a acumular letras estables; **Detener y revisar texto** pausa la captura y ofrece sugerencias ortográficas en inglés con un diccionario local. `space` separa palabras, `del` borra el último carácter y `nothing`/`other` no añaden letras. El corrector es una sugerencia ortográfica por palabra, no un modelo de gramática ni un traductor de lengua de señas.

Si `video/jz_flow_cnn.keras` no está instalado, la app funciona solo con la rama estática. La ventana temporal y los umbrales son iniciales; deben ajustarse después de evaluar clips reales. En Windows la inferencia puede ejecutarse en CPU, aunque la velocidad dependerá del equipo.

## Evaluación y limitaciones

Los splits estáticos son estratificados por imagen, no por persona/fuente; duplicados entre particiones pueden inflar las métricas. La rama de video intenta separar por signante, pero el resultado depende de que los IDs de las carpetas sean correctos y haya diversidad suficiente de `other`. Reporta accuracy y macro F1, revisa matriz de confusión y no interpretes el prototipo como traducción completa de ASL.
