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
├── video_map/                 # recomendado: salida del Notebook 14
    ├── jz_landmark_map_gru.keras
    ├── classes.json
    └── metadata.json
└── video/                     # opcional: modelo anterior de flujo óptico
    ├── jz_flow_cnn.keras
    └── classes.json
```

Rutas de origen en Drive:

- Estático mapas: `exports/proyecto_def/maps/integrated_varied_1724/model.keras` y `classes.json`.
- Estático recortes: `exports/proyecto_def/crop/integrated_varied_1724/model.keras` y `classes.json`.
- Temporal preferido (Notebook 14): `exports/proyecto_def/video_map_sequence_v1/jz_landmark_map_gru.keras`, `classes.json`, `metadata.json` y `hand_landmarker.task`.
- Temporal anterior de flujo óptico (fallback): `exports/proyecto_def/video_balanced_v3/jz_flow_cnn.keras` y `classes.json`.
- Temporal binario con rechazo: `exports/proyecto_def/video_jz_reject_v1/jz_flow_binary.keras`, junto con `classes.json` y `reject_threshold.json`. Es salida experimental; todavía requiere integrarse en la app local.
- El modelo temporal de mapas se integra en la app y tiene prioridad si ambos modelos temporales están instalados. Copia sus cuatro archivos según la estructura anterior. La inferencia local reproduce el muestreo de 17 landmarks, normalización de secuencia y renderizado de Notebook 14. El modo automático usa la postura inicial y movimiento como compuerta heurística; el modo temporal permite probar la rama sin esa compuerta.
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

La ventana muestra la vista previa y la predicción en vivo. El menú de representación elige mapas o recortes para letras estáticas; el menú de enrutamiento elige `auto`, `static` o `dynamic`. `auto` conserva la compuerta heurística: estabiliza una postura inicial candidata de J (meñique extendido) o Z (índice extendido) y activa la rama temporal al detectar movimiento del dedo; durante la secuencia suprime letras estáticas. Una postura candidata sin movimiento expira y vuelve a la clasificación estática. `dynamic` ejecuta la rama temporal continuamente con ventanas móviles y sirve para evaluarla aparte; `static` desactiva la temporal. Los umbrales son iniciales y se deben ajustar con cámara real. **Iniciar captura** acumula letras estables; **Detener y corregir texto** pausa la captura y ofrece una sugerencia de escritura en inglés que se puede aceptar o rechazar. `space` separa palabras, `del` borra el último carácter y `nothing`/`other` no añaden letras.

### Prueba opcional de corrección contextual con T5

En la rama `codex/t5-correction-experiment`, el selector **Corrector** permite usar el diccionario actual (`dictionary`), T5-small (`t5-small`) o comparar ambos (`compare`). En modo compare la app muestra las dos sugerencias y permite elegir una o conservar el original. T5 recibe la frase completa al detener la captura. Instala las dependencias opcionales con `python -m pip install -r requirements-corrector-t5.txt`. La primera selección de T5 descarga el modelo `selfnify/proximity` y sus archivos a la caché local de Hugging Face; después puede cargarse desde la caché. El modelo tiene licencia Apache 2.0 y está orientado a corrección gramatical en inglés; sigue siendo experimental y puede cambiar texto correcto. Una palabra válida aislada como `boughs` puede permanecer igual porque no hay contexto suficiente para inferir que se quería escribir `bought`.

Si no está instalado ninguno de los modelos temporales, la app funciona solo con la rama estática. En el modelo de mapas se usa un umbral inicial de confianza de 0.50; en el de flujo óptico se conserva 0.80. Esos valores son provisionales, no calibrados para una aplicación final. La inferencia local se ejecuta en CPU, aunque la velocidad dependerá del equipo.

## Evaluación y limitaciones

Los splits estáticos son estratificados por imagen, no por persona/fuente; duplicados entre particiones pueden inflar las métricas. La rama de video intenta separar por signante, pero el resultado depende de que los IDs de las carpetas sean correctos y haya diversidad suficiente de `other`. Reporta accuracy y macro F1, revisa matriz de confusión y no interpretes el prototipo como traducción completa de ASL.
