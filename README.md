# Clasificación de señas del alfabeto ASL

Proyecto académico de IA2 para clasificar imágenes de señas del alfabeto de la Lengua de Señas Americana (ASL) mediante redes neuronales convolucionales. La primera meta es construir y evaluar una línea base reproducible con imágenes estáticas. El reconocimiento dinámico y en tiempo real queda como trabajo posterior, sujeto al tiempo y a los recursos disponibles.

## Estado

El repositorio está en etapa inicial. El análisis de datos comienza en `notebooks/01_exploracion_datos.ipynb`; todavía se debe descargar el dataset y verificar su estructura y contenido antes de definir rutas o cantidades.

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

## Plan de trabajo

1. Explorar y verificar el dataset.
2. Diseñar el preprocesamiento y las particiones sin fuga de datos.
3. Entrenar una CNN base reproducible.
4. Evaluar y documentar resultados.
5. Después de estabilizar la línea base, explorar transferencia de aprendizaje y, si es viable, reconocimiento dinámico o en tiempo real.

No se deben subir al repositorio datasets completos, credenciales, entornos virtuales ni archivos grandes de entrenamiento. Las métricas y figuras que se decida conservar para documentar experimentos deben revisarse antes de agregarlas a Git.
