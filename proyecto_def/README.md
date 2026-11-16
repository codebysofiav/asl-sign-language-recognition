# Proyecto definitivo — ASL fingerspelling

El entrenamiento se ejecuta en Colab y guarda modelos, checkpoints, manifiestos y métricas en Google Drive. La aplicación local abre la cámara y carga los modelos exportados. El alcance es deletreo de letras ASL y sugerencia ortográfica en inglés; no es traducción completa de ASL.

## Notebooks de Colab

Sube estos archivos de `notebooks/` a Colab. Los tres notebooks son independientes:

1. `10_entrenamiento_mapas_colab.ipynb`: modelo estático principal. MediaPipe detecta 21 landmarks, los convierte en un mapa RGB y MobileNetV2 los clasifica.
2. `12_entrenamiento_video_flujo_colab.ipynb`: rama temporal J/Z/other. MediaPipe recorta una región común de la mano, Farneback calcula el flujo óptico y una CNN 3D clasifica la secuencia.
3. `11_entrenamiento_recortes_colab.ipynb`: alternativa/experimento de comparación para la rama estática. MediaPipe recorta la mano y MobileNetV2 clasifica el recorte. No se ejecuta al mismo tiempo que mapas en la app; se selecciona una rama estática.

Para obtener el primer prototipo integrado, ejecuta 10 y 12. Ejecuta 11 para comparar la representación de recorte con mapas. Se puede entrenar en sesiones distintas o en paralelo si Colab tiene recursos disponibles.

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

En `data/videos_jz`, la carpeta de segundo nivel debe identificar consistentemente a una persona. Por ejemplo, `J/participant_03`, `Z/participant_03` y `other/participant_03` se consideran un mismo signante. Para WLASL, usa el identificador de signante de sus metadatos si está disponible. El notebook de video verifica que todas las particiones tengan las tres clases; si no puede hacerlo sin mezclar signantes, se detiene e informa los grupos.

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
- Temporal: `exports/proyecto_def/video/jz_flow_cnn.keras`.
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

Antes de descargar los modelos, puedes verificar solo la interfaz y la cámara con `python app.py --preview-only`. Este modo no clasifica señas ni guarda texto.

La ventana muestra la vista previa y la predicción en vivo. El menú permite elegir mapas o recortes. **Iniciar captura** comienza a acumular letras estables; **Detener y revisar texto** pausa la captura y ofrece sugerencias ortográficas en inglés con un diccionario local. `space` separa palabras, `del` borra el último carácter y `nothing`/`other` no añaden letras. El corrector es una sugerencia ortográfica por palabra, no un modelo de gramática ni un traductor de lengua de señas.

Si `video/jz_flow_cnn.keras` no está instalado, la app funciona solo con la rama estática. La ventana temporal y los umbrales son iniciales; deben ajustarse después de evaluar clips reales. En Windows la inferencia puede ejecutarse en CPU, aunque la velocidad dependerá del equipo.

## Evaluación y limitaciones

Los splits estáticos son estratificados por imagen, no por persona/fuente; duplicados entre particiones pueden inflar las métricas. La rama de video intenta separar por signante, pero el resultado depende de que los IDs de las carpetas sean correctos y haya diversidad suficiente de `other`. Reporta accuracy y macro F1, revisa matriz de confusión y no interpretes el prototipo como traducción completa de ASL.
