# Clasificación de señas del alfabeto ASL

Proyecto académico de IA2 para clasificar imágenes de señas del alfabeto de la Lengua de Señas Americana (ASL) mediante redes neuronales convolucionales. La primera meta es construir y evaluar una línea base reproducible con imágenes estáticas. El reconocimiento dinámico y en tiempo real queda como trabajo posterior, sujeto al tiempo y a los recursos disponibles.

## Estado

El dataset ya fue explorado y el preprocesamiento está verificado. La copia local contiene 29 clases con 3.000 imágenes por clase, JPEG RGB de 200 × 200. El test incluido en Kaggle contiene 28 imágenes que son copias exactas de ejemplos de entrenamiento, por lo que no se usa como evaluación independiente. Las particiones de desarrollo se crean desde entrenamiento con una semilla fija.

## Dataset

Dataset inicial: [ASL Alphabet en Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).

1. Descarga el dataset desde Kaggle y extrae los archivos en `data/raw/`.
2. No subas imágenes del dataset a GitHub. La carpeta `data/raw/` está excluida por `.gitignore`.
3. Revisa `notebooks/01_exploracion_datos.ipynb` para comprobar estructura, conteos y calidad de tu copia.
4. Usa `notebooks/02_preprocesamiento.ipynb` para crear particiones estratificadas y verificar que no haya duplicados exactos entre ellas.

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

## Plan de trabajo

1. Exploración y preprocesamiento: completados y documentados en los notebooks 01 y 02.
2. Línea base y comparación Keras: `notebooks/03_entrenamiento_baseline.ipynb` y `notebooks/04_comparacion_modelos_keras.ipynb` permiten entrenar CNN, ResNet-50V2 y EfficientNetB0.
3. `configs/baseline.json` y `configs/comparison_keras.json` guardan la configuración inicial. En el notebook 04 el entrenamiento y la evaluación final tienen interruptores separados; la evaluación usa la partición de prueba reservada.
4. El notebook 04 guarda checkpoints bajo `models/` y, bajo `results/metrics/`, historiales por época, resúmenes de entrenamiento, métricas por clase, matrices de confusión y `keras_model_comparison.csv`.
5. La transferencia de aprendizaje se ejecuta inicialmente con las bases preentrenadas congeladas. El ajuste fino, PyTorch/ResNet-18 y el reconocimiento en tiempo real quedan para pasos posteriores.

Para ejecutar el notebook 04 en un clúster, prepara allí el mismo entorno Python, instala `requirements.txt`, coloca el dataset en la estructura `data/raw/` y selecciona el kernel del entorno. TensorFlow debe detectar la GPU del clúster; los pesos ImageNet de Keras pueden requerir una descarga en el primer uso. No subas modelos grandes a GitHub.

No se deben subir al repositorio datasets completos, credenciales, entornos virtuales ni archivos grandes de entrenamiento. Las métricas y figuras que se decida conservar para documentar experimentos deben revisarse antes de agregarlas a Git.
