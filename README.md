# ASL Fingerspelling Recognition — IA2

Proyecto académico para reconocer letras del alfabeto ASL desde cámara. El prototipo combina clasificación estática de letras con reconocimiento temporal de J y Z. La app deletrea texto en inglés; no traduce ASL completo ni reconoce frases signadas.

## Estado y arquitectura

- **Estático (principal):** MediaPipe detecta la mano y dibuja mapas RGB de 21 puntos. MobileNetV2 clasifica las letras estáticas.
- **Estático (comparación):** MediaPipe recorta la mano y MobileNetV2 clasifica la imagen recortada.
- **Dinámico (principal):** para cada video se toman 17 fotogramas, se extraen los landmarks, se normaliza la secuencia y se crea un mapa por fotograma. Una CNN por fotograma más una GRU clasifica `J`, `Z` u `other`.
- **App local:** abre la cámara y ofrece enrutamiento `auto`, `static` o `dynamic`. En `auto`, una compuerta heurística de postura y movimiento activa temporalmente la rama J/Z. La interfaz añade letras estables al texto entre **Iniciar captura de letras** y **Detener y corregir texto**.
- **Corrección:** la versión estable propone correcciones ortográficas en inglés con `pyspellchecker`. La rama `codex/t5-correction-experiment` contiene una comparación experimental con T5-small; no es necesaria para ejecutar la versión estable.

La captura de J/Z y la detección automática siguen siendo experimentales. Las métricas de los videos y las condiciones de cámara limitan la precisión esperada.

## Notebooks

Los notebooks se conservan para exploración, comparación y entrenamiento. Los notebooks 10–14 son los del pipeline definitivo en Colab:

| Notebook | Propósito | Uso en el prototipo |
| --- | --- | --- |
| `proyecto_def/notebooks/10_entrenamiento_mapas_colab.ipynb` | Entrena la rama estática con mapas de landmarks. | Principal |
| `proyecto_def/notebooks/11_entrenamiento_recortes_colab.ipynb` | Entrena la alternativa de recortes de mano. | Comparación |
| `proyecto_def/notebooks/12_entrenamiento_video_flujo_colab.ipynb` | Experimento temporal con flujo óptico. | Fallback/comparación |
| `proyecto_def/notebooks/13_entrenamiento_video_jz_rechazo_colab.ipynb` | Experimento J/Z binario con rechazo. | Comparación; no lo carga actualmente la app |
| `proyecto_def/notebooks/14_entrenamiento_video_mapas_colab.ipynb` | Entrena la rama temporal CNN + GRU con mapas de 21 puntos. | Principal para J/Z |

Los notebooks `notebooks/01`–`04` son exploración, preprocesamiento y primeras comparaciones Keras. Permanecen junto con `src/` y `configs/` porque esos notebooks los usan; son antecedentes del pipeline definitivo.

## Datos y entrenamiento en Colab

Los datos no se guardan en GitHub. En Google Drive, los notebooks definitivos esperan:

```text
proyecto-ia2/
├── dataset_asl/final_dataset/<clase>/<imagen>    # conjunto estático variado
├── data/videos_jz/
│   ├── J/<signante>/<video>
│   ├── Z/<signante>/<video>
│   └── other/<signante>/<video>
├── data/raw/                                     # ASL Alphabet original, opcional
└── exports/proyecto_def/                         # modelos, métricas y checkpoints
```

Ejecuta el notebook 10 para el modelo estático principal y el 14 para el temporal basado en puntos. El 11 permite comparar recortes. Los notebooks 12 y 13 son experimentos alternativos. Los checkpoints, manifiestos y resultados se guardan en Drive para poder reanudar después de una desconexión de Colab. No subas datasets, pesos ni checkpoints a GitHub.

## Archivos de modelos para la app

Después del entrenamiento, descarga desde `exports/proyecto_def/` únicamente los artefactos que necesita la inferencia y colócalos así:

```text
proyecto_def/artifacts/
├── hand_landmarker.task
├── maps/
│   ├── model.keras
│   └── classes.json
├── crop/                         # opcional, para comparar la rama de recortes
│   ├── model.keras
│   └── classes.json
├── video_map/                    # modelo temporal recomendado (Notebook 14)
│   ├── jz_landmark_map_gru.keras
│   ├── classes.json
│   └── metadata.json
└── video/                        # opcional, fallback de flujo óptico
    ├── jz_flow_cnn.keras
    └── classes.json
```

Orígenes en Drive:

- Mapas estáticos: `exports/proyecto_def/maps/integrated_varied_1724/`.
- Recortes: `exports/proyecto_def/crop/integrated_varied_1724/`.
- Mapas temporales: `exports/proyecto_def/video_map_sequence_v1/`.
- Flujo óptico alternativo: la carpeta de exportación del entrenamiento de Notebook 12, por ejemplo `exports/proyecto_def/video_balanced_v3/`.
- MediaPipe: `exports/proyecto_def/hand_landmarker.task`.

La app carga primero `video_map/` si está presente; si falta, intenta cargar el modelo anterior de flujo óptico en `video/`. Para la rama estática seleccionada se requieren su modelo y `classes.json`. No se necesitan los datasets, mapas preprocesados, archivos de flujo óptico ni checkpoints para ejecutar la app.

## Instalación y ejecución local

Usa Python 3.10 u 3.11. Desde la raíz del repositorio, crea y activa un entorno virtual; luego instala las dependencias de la app:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r proyecto_def/requirements-local.txt
cd proyecto_def
python app.py
```

`python app.py --preview-only` abre la cámara sin cargar los modelos. `requirements.txt` en la raíz se conserva para los notebooks 01–04; `proyecto_def/requirements-local.txt` instala la app.

En la app, elige `maps` o `crop` para el clasificador estático y `auto`, `static` o `dynamic` para el enrutamiento; pulsa **Aplicar** después de cambiar la selección. **Iniciar captura de letras** permite acumular texto. **Detener y corregir texto** finaliza la captura y muestra una sugerencia ortográfica en inglés si la encuentra. `space` agrega un espacio, `del` elimina el último carácter, y `nothing`/`other` no escriben letras.

### Experimento T5 opcional

En la rama `codex/t5-correction-experiment`, instala o actualiza las dependencias opcionales desde `proyecto_def` con `python -m pip install --upgrade -r requirements-corrector-t5.txt`. Esta prueba usa Transformers 5 para ser compatible con la configuración de tokenizer publicada. El selector **Corrector** permite elegir el diccionario actual, T5-small o comparar ambos. El modelo se descarga a la caché local de Hugging Face la primera vez y las sugerencias se confirman antes de reemplazar el texto. T5-small tiene licencia Apache 2.0; su ficha describe un modelo de 60.5 millones de parámetros entrenado para corrección gramatical, pero sigue siendo experimental y puede cambiar texto correcto. Una palabra válida aislada como `boughs` puede permanecer igual sin contexto suficiente para inferir `bought` ([ficha del modelo](https://huggingface.co/selfnify/proximity)).

## Métricas y limitaciones

Reporta accuracy y macro F1, revisa matrices de confusión y resultados por signante. Los splits de imágenes pueden contener dependencia de fuente/persona; los de video dependen de que los identificadores de signante estén bien organizados. Los resultados no deben presentarse como una traducción completa de ASL ni como rendimiento garantizado en cualquier cámara o fondo.

## Estructura

```text
notebooks/                 # exploración y comparaciones iniciales
src/                       # código de soporte de notebooks 01–04
configs/                   # configuraciones de esas comparaciones
results/                   # métricas que se conservan
proyecto_def/               # app local, notebooks finales y código de inferencia
```
