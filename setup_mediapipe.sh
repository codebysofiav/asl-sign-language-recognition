#!/usr/bin/env bash
# Prepara el entorno aparte de MediaPipe (.venv-mp) y descarga el modelo de manos.
# Es independiente de .venv, así no se toca el entorno de TensorFlow. Idempotente.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

log "1/3 Entorno .venv-mp (Python 3.12)"
[ -d .venv-mp ] || uv venv --python 3.12 .venv-mp
uv pip install --python .venv-mp/bin/python mediapipe pillow numpy

log "2/3 Modelo hand_landmarker.task"
mkdir -p assets
if [ ! -s assets/hand_landmarker.task ]; then
  BASE=https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16
  curl -fL "$BASE/latest/hand_landmarker.task" -o assets/hand_landmarker.task \
    || curl -fL "$BASE/1/hand_landmarker.task" -o assets/hand_landmarker.task
fi
ls -lh assets/hand_landmarker.task

log "3/3 Verificación"
.venv-mp/bin/python -c "
import mediapipe as mp
from mediapipe.tasks.python import vision
print('MediaPipe', mp.__version__, '| API Tasks disponible:', hasattr(vision, 'HandLandmarker'))"

for entry in ".venv-mp/" "assets/"; do
  grep -qxF "$entry" .gitignore 2>/dev/null || echo "$entry" >> .gitignore
done
log "Listo. Ahora puedes ejecutar el notebook 05."
