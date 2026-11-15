"""Tkinter camera app for live ASL fingerspelling (English spelling suggestions)."""

from __future__ import annotations

import re
import argparse
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk
from spellchecker import SpellChecker

from src.inference import RecognitionEngine


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


class EnglishSpeller:
    """Offer offline English spelling suggestions after a capture is stopped."""

    def __init__(self):
        self.dictionary = SpellChecker(language="en")

    def suggest(self, text: str) -> str:
        parts = re.findall(r"[A-Za-z']+|[^A-Za-z']+", text)
        corrected = []
        for token in parts:
            if token.isalpha() and len(token) > 1:
                candidate = self.dictionary.correction(token.lower()) or token.lower()
                if token.isupper():
                    candidate = candidate.upper()
                elif token[:1].isupper():
                    candidate = candidate.capitalize()
                corrected.append(candidate)
            else:
                corrected.append(token)
        return "".join(corrected)


class TextAccumulator:
    def __init__(self, stable_frames: int = 6, confidence_threshold: float = 0.78):
        self.active = False
        self.text = ""
        self.stable_frames = stable_frames
        self.confidence_threshold = confidence_threshold
        self.candidate = None
        self.count = 0
        self.last_accepted = None

    def reset_gate(self):
        self.candidate, self.count, self.last_accepted = None, 0, None

    def observe(self, label, confidence):
        if label is None or confidence < self.confidence_threshold:
            self.candidate, self.count = None, 0
            return None
        if label == self.candidate:
            self.count += 1
        else:
            self.candidate, self.count = label, 1
        if self.count < self.stable_frames:
            return None
        key = str(label).strip().lower()
        if key in {"nothing", "other"}:
            self.last_accepted = None
            return None
        if key == self.last_accepted:
            return None
        if not self.active:
            return None
        if key == "space":
            if self.text and not self.text.endswith(" "):
                self.text += " "
                output = " "
            else:
                output = None
        elif key in {"del", "delete", "backspace"}:
            self.text = self.text[:-1]
            output = "<"
        elif len(key) == 1 and key.isalpha():
            output = key.upper()
            self.text += output
        else:
            output = None
        self.last_accepted = key
        return output

    def start(self):
        self.active = True
        self.reset_gate()

    def stop(self):
        self.active = False
        self.reset_gate()
        return self.text

    def clear(self):
        self.text = ""
        self.reset_gate()


class LiveApp:
    def __init__(self, root: tk.Tk, preview_only: bool = False):
        self.root = root
        self.preview_only = preview_only
        root.title("ASL Fingerspelling — Local Camera")
        root.geometry("1080x760")
        root.minsize(820, 640)

        self.mode = tk.StringVar(value="maps")
        self.status = tk.StringVar(value="Cargando modelo y cámara…")
        self.prediction = tk.StringVar(value="Predicción: —")
        self.session = TextAccumulator()
        self.corrector = EnglishSpeller()
        self.engine = None
        self.capture = None
        self.photo = None

        top = ttk.Frame(root, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="Clasificador estático:").pack(side="left")
        ttk.Combobox(top, textvariable=self.mode, values=("maps", "crop"),
                     state="readonly", width=12).pack(side="left", padx=6)
        self.reload_button = ttk.Button(top, text="Cargar rama", command=self.reload_engine)
        self.reload_button.pack(side="left", padx=5)
        ttk.Label(top, textvariable=self.status).pack(side="right")

        self.video_view = ttk.Label(root, anchor="center")
        self.video_view.pack(fill="both", expand=True, padx=10, pady=5)
        ttk.Label(root, textvariable=self.prediction, font=("Segoe UI", 13, "bold")).pack(pady=4)

        controls = ttk.Frame(root, padding=8)
        controls.pack(fill="x")
        self.start_button = ttk.Button(controls, text="Iniciar captura", command=self.start_capture)
        self.start_button.pack(side="left", padx=4)
        self.stop_button = ttk.Button(controls, text="Detener y revisar texto", command=self.stop_capture, state="disabled")
        self.stop_button.pack(side="left", padx=4)
        ttk.Button(controls, text="Borrar texto", command=self.clear_text).pack(side="left", padx=4)

        ttk.Label(root, text="Texto deletreado (puedes editarlo antes de aceptar sugerencias):").pack(anchor="w", padx=12)
        self.text_widget = tk.Text(root, height=4, wrap="word", font=("Segoe UI", 14))
        self.text_widget.pack(fill="x", padx=10, pady=(2, 10))

        try:
            if self.preview_only:
                self.reload_button.configure(state="disabled")
                self.start_button.configure(state="disabled")
                self.stop_button.configure(state="disabled")
                self.status.set("Vista previa: no se cargan modelos ni se guardan predicciones.")
            else:
                self.reload_engine(show_message=False)
            self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
            if not self.capture.isOpened():
                raise RuntimeError("No se pudo abrir la cámara. Revisa que no esté ocupada por otra aplicación.")
            if not self.preview_only:
                self.status.set("Vista previa activa; el texto solo se guarda al iniciar captura.")
            self.update_frame()
        except Exception as exc:
            messagebox.showerror("No se pudo iniciar", str(exc))
            self.status.set("Revisa los modelos, MediaPipe y el índice de cámara.")

        root.protocol("WM_DELETE_WINDOW", self.close)

    def reload_engine(self, show_message=True):
        try:
            new_engine = RecognitionEngine(ARTIFACTS, static_mode=self.mode.get())
            old_engine = self.engine
            self.engine = new_engine
            if old_engine:
                old_engine.close()
            message = f"Modelo {self.mode.get()} cargado"
            if not self.engine.dynamic_available:
                message += "; modelo temporal aún no instalado (solo estático)"
            self.status.set(message)
            if show_message:
                messagebox.showinfo("Modelos cargados", message)
        except Exception as exc:
            self.status.set("No se pudo cargar la rama seleccionada.")
            if show_message:
                messagebox.showerror("Error al cargar", str(exc))
            elif self.engine is None:
                raise

    def start_capture(self):
        if not self.engine:
            return
        self.session.start()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status.set("CAPTURANDO: letras estables se añaden al texto.")

    def stop_capture(self):
        self.session.text = self.text_widget.get("1.0", "end-1c")
        text = self.session.stop()
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.status.set("Captura detenida.")
        self._sync_text()
        if not text.strip():
            return
        suggestion = self.corrector.suggest(text)
        if suggestion != text:
            accept = messagebox.askyesno(
                "English spelling suggestion",
                f"Recognized text:\n{text}\n\nSuggested spelling:\n{suggestion}\n\nApply the suggestion?",
            )
            if accept:
                self.session.text = suggestion
                self._sync_text()
        else:
            messagebox.showinfo("English spelling check", "No spelling changes suggested.")

    def clear_text(self):
        self.session.clear()
        self.text_widget.delete("1.0", "end")

    def _sync_text(self):
        self.text_widget.delete("1.0", "end")
        self.text_widget.insert("1.0", self.session.text)

    def update_frame(self):
        if not self.capture or not self.capture.isOpened():
            return
        ok, frame = self.capture.read()
        if ok:
            if self.preview_only:
                display = cv2.flip(frame, 1)
                cv2.putText(display, "CAMERA PREVIEW ONLY - NO MODEL", (12, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 190, 255), 2)
                self.prediction.set("Vista previa solamente: falta descargar los modelos entrenados.")
            else:
                if not self.engine:
                    return
                display, label, confidence, source, detected = self.engine.predict(frame)
                display = cv2.flip(display, 1)  # espejo solo para la vista; el modelo ve el fotograma original
                accepted = self.session.observe(label, confidence)
                if accepted:
                    if accepted == "<":
                        self.text_widget.delete("1.0", "end")
                        self.text_widget.insert("1.0", self.session.text)
                    else:
                        self.text_widget.insert("end", accepted)
                mode_text = "CAPTURANDO" if self.session.active else "VISTA PREVIA — texto pausado"
                cv2.putText(display, mode_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (50, 210, 90) if self.session.active else (0, 180, 255), 2)
                shown = label if label is not None else "uncertain"
                cv2.putText(display, f"{shown}  {confidence:.2f}  [{source}]", (12, 56),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
                if not detected:
                    cv2.putText(display, "Hand not detected", (12, 84), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (120, 200, 255), 2)
                self.prediction.set(f"Predicción: {shown} | confianza {confidence:.2f} | rama {source}")
            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            image.thumbnail((1040, 520))
            self.photo = ImageTk.PhotoImage(image)
            self.video_view.configure(image=self.photo)
        self.root.after(15, self.update_frame)

    def close(self):
        if self.capture:
            self.capture.release()
        if self.engine:
            self.engine.close()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-only", action="store_true",
                        help="Abre la interfaz y cámara sin modelos para verificar el dispositivo.")
    args = parser.parse_args()
    root = tk.Tk()
    LiveApp(root, preview_only=args.preview_only)
    root.mainloop()


if __name__ == "__main__":
    main()
