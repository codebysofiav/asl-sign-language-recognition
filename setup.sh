#!/usr/bin/env bash
# Configura todo el entorno local (WSL2 + GPU NVIDIA). Es idempotente: se puede repetir.
# Uso: ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"

PY_VERSION="3.12"            # TensorFlow 2.21 soporta Python 3.10-3.13 (no 3.14)
GIT_NAME="Juan-David-Ascanio"
GIT_EMAIL="jascanio412@gmail.com"
KERNEL_NAME="asl-sign-language"

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

log "1/8 Verificando GPU"
nvidia-smi -L || { echo "nvidia-smi no responde: revisa el driver NVIDIA en Windows."; exit 1; }

log "2/8 Instalando uv (si falta)"
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

log "3/8 Creando entorno virtual (Python $PY_VERSION)"
[ -d .venv ] || uv venv --python "$PY_VERSION" .venv

log "4/8 Instalando dependencias"
uv pip install --python .venv/bin/python -r requirements.txt

log "5/8 Configurando rutas de GPU y verificando TensorFlow"
# En WSL2 la librería CUDA del driver (libcuda.so) vive en /usr/lib/wsl/lib y no siempre
# está en el path de librerías. Se fija aquí, sin tocar ~/.bashrc, y con rutas calculadas.
SITE=$(.venv/bin/python -c "import site; print(site.getsitepackages()[0])")
GPU_LD="/usr/lib/wsl/lib:$SITE/tensorflow"
export LD_LIBRARY_PATH="$GPU_LD:${LD_LIBRARY_PATH:-}"

# Terminal: se exporta cada vez que se activa el entorno (VS Code lo activa solo).
MARK="# >>> asl-gpu"
if ! grep -q "$MARK" .venv/bin/activate; then
  cat >> .venv/bin/activate <<ACT

$MARK
export LD_LIBRARY_PATH="$GPU_LD:\${LD_LIBRARY_PATH:-}"
# <<< asl-gpu
ACT
fi

TF_CPP_MIN_LOG_LEVEL=3 .venv/bin/python -c "
import tensorflow as tf
g = tf.config.list_physical_devices('GPU')
print('TensorFlow', tf.__version__, '| GPUs:', g)
raise SystemExit(0 if g else 1)
" || echo "⚠ TensorFlow no detecta la GPU; revisa la instalación antes de entrenar."

log "6/8 Registrando kernel de Jupyter"
# Los notebooks no activan el entorno: la variable se guarda en el propio kernel.
.venv/bin/python -m ipykernel install --user --name "$KERNEL_NAME" \
  --display-name "Python $PY_VERSION (ASL)" --env LD_LIBRARY_PATH "$GPU_LD"

log "7/8 Configurando Git"
git config --local user.name "$GIT_NAME"
git config --local user.email "$GIT_EMAIL"
git config --global credential.helper store   # guarda el token tras el primer push

log "8/8 Dataset"
mkdir -p data/raw experiments
if [ -d data/raw/asl_alphabet_train/asl_alphabet_train ]; then
  echo "Dataset ya extraído."
else
  ZIP=$(ls -1 data/*.zip 2>/dev/null | head -n1 || true)
  if [ -n "$ZIP" ]; then
    echo "Extrayendo $ZIP ..."
    .venv/bin/python - "$ZIP" <<'PY'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
members = [m for m in z.namelist() if m.startswith("asl_alphabet_train/")]
z.extractall("data/raw", members)
print(f"{len(members)} archivos extraídos en data/raw")
PY
  else
    echo "Falta el dataset. Descarga el .zip desde:"
    echo "  https://www.kaggle.com/datasets/grassknoted/asl-alphabet"
    echo "colócalo en data/ y vuelve a ejecutar ./setup.sh"
  fi
fi

log "Listo. Activa el entorno con: source .venv/bin/activate"
