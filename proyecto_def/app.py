"""Tkinter camera app for live ASL fingerspelling (English spelling suggestions)."""

from __future__ import annotations

import re
import argparse
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageDraw, ImageTk


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

COLORS = {
    "bg": "#F6F3FA", "card": "#FFFFFF", "card_tint": "#FAF8FE",
    "border": "#E5DDF2", "text": "#2D2A3E", "muted": "#6E6A86",
    "lavender": "#9D8DF1", "lavender_hover": "#8774E8", "lavender_soft": "#EFEAFF",
    "mint": "#6BCB9F", "mint_hover": "#52B788", "mint_soft": "#E6F8F0",
    "peach": "#F4978E", "peach_hover": "#E57A70", "sky_soft": "#E8F4FE",
    "neutral": "#EDE8F5", "neutral_hover": "#DFD7EC",
    "disabled_bg": "#ECE9F2", "disabled_fg": "#A8A3B8",
}


class RoundedButton(tk.Canvas):
    """Small canvas button used by the visual redesign."""

    def __init__(self, parent, text, command=None, bg_color=None, hover_color=None,
                 fg_color="#FFFFFF", radius=13, padx=15, pady=8,
                 font=("Segoe UI", 10, "bold"), state="normal"):
        super().__init__(parent, highlightthickness=0, bd=0, bg=parent.cget("bg"),
                         cursor="hand2" if state == "normal" else "arrow")
        self.command, self._text = command, text
        self._bg_color = bg_color or COLORS["lavender"]
        self._hover_color = hover_color or COLORS["lavender_hover"]
        self._fg_color, self._radius = fg_color, radius
        self._padx, self._pady, self._font = padx, pady, font
        self._state, self._hovered = state, False
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonRelease-1>", self._on_click)
        self._redraw()

    def _redraw(self):
        self.delete("all")
        text_id = self.create_text(0, 0, text=self._text, font=self._font)
        box = self.bbox(text_id)
        self.delete(text_id)
        text_w, text_h = ((box[2] - box[0], box[3] - box[1]) if box else (80, 20))
        width, height = text_w + 2 * self._padx, text_h + 2 * self._pady
        self.configure(width=width, height=height)
        fill = COLORS["disabled_bg"] if self._state == "disabled" else (
            self._hover_color if self._hovered else self._bg_color)
        fg = COLORS["disabled_fg"] if self._state == "disabled" else self._fg_color
        r = min(self._radius, width // 2, height // 2)
        pts = [r, 1, width-r, 1, width-1, r, width-1, height-r,
               width-r, height-1, r, height-1, 1, height-r, 1, r]
        self.create_polygon(pts, smooth=True, fill=fill, outline="")
        self.create_text(width / 2, height / 2, text=self._text, fill=fg, font=self._font)

    def configure(self, cnf=None, **kwargs):
        if "state" in kwargs:
            self._state = kwargs.pop("state")
            super().configure(cursor="hand2" if self._state == "normal" else "arrow")
            self._redraw()
        if "text" in kwargs:
            self._text = kwargs.pop("text")
            self._redraw()
        if kwargs or cnf:
            super().configure(cnf, **kwargs)

    config = configure

    def _on_enter(self, _event):
        if self._state == "normal":
            self._hovered = True
            self._redraw()

    def _on_leave(self, _event):
        self._hovered = False
        self._redraw()

    def _on_click(self, _event):
        if self._state == "normal" and self.command:
            self.command()


class RoundedCard(tk.Frame):
    def __init__(self, parent, radius=18, padding=12, bg_color=None):
        super().__init__(parent, bg=parent.cget("bg"))
        self.radius, self.padding = radius, padding
        self.bg_color = bg_color or COLORS["card"]
        self.canvas = tk.Canvas(self, bg=parent.cget("bg"), highlightthickness=0, bd=0)
        self.canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.inner = tk.Frame(self, bg=self.bg_color)
        self.inner.pack(fill="both", expand=True, padx=padding, pady=padding)
        self.bind("<Configure>", self._draw)

    def _draw(self, _event=None):
        w, h = self.winfo_width(), self.winfo_height()
        if w < 4 or h < 4:
            return
        self.canvas.delete("all")
        r = min(self.radius, w // 2, h // 2)
        pts = [r, 1, w-r, 1, w-1, r, w-1, h-r, w-r, h-1,
               r, h-1, 1, h-r, 1, r]
        self.canvas.create_polygon(pts, smooth=True, fill=self.bg_color,
                                   outline=COLORS["border"], width=1.3)


class PastelProgressBar(tk.Canvas):
    def __init__(self, parent, height=11, fill_color=None):
        super().__init__(parent, height=height, bg=parent.cget("bg"), highlightthickness=0)
        self.value, self.fill_color = 0.0, fill_color or COLORS["lavender"]
        self.bind("<Configure>", lambda _event: self.set_value(self.value))

    def set_value(self, value, fill_color=None):
        self.value = max(0.0, min(1.0, float(value)))
        if fill_color:
            self.fill_color = fill_color
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 4 or h < 4:
            return
        r = h // 2
        self.create_oval(0, 0, h, h, fill=COLORS["neutral"], outline="")
        self.create_oval(w-h, 0, w, h, fill=COLORS["neutral"], outline="")
        self.create_rectangle(r, 0, w-r, h, fill=COLORS["neutral"], outline="")
        fill_w = int(w * self.value)
        if fill_w > 0:
            x2 = max(fill_w, h)
            self.create_oval(0, 0, h, h, fill=self.fill_color, outline="")
            if x2 > h:
                self.create_oval(x2-h, 0, x2, h, fill=self.fill_color, outline="")
                self.create_rectangle(r, 0, x2-r, h, fill=self.fill_color, outline="")


def resize_to_fit(image, target_w, target_h):
    if target_w < 10 or target_h < 10:
        return image
    scale = min(target_w / image.width, target_h / image.height)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image if size == image.size else image.resize(size, Image.Resampling.BILINEAR)


def rounded_image(image, radius=16):
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, image.width - 1, image.height - 1),
                                          radius=radius, fill=255)
    result = Image.new("RGB", image.size, COLORS["card"])
    result.paste(image, mask=mask)
    return result


class EnglishSpeller:
    """Offer offline English spelling suggestions after a capture is stopped."""

    def __init__(self):
        from spellchecker import SpellChecker
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


class T5GrammarCorrector:
    """Optional local, context-aware English correction using a small T5 model."""

    model_name = "selfnify/proximity"

    def __init__(self):
        import torch
        import transformers
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from packaging.version import Version

        if Version(transformers.__version__) < Version("5.0"):
            raise RuntimeError(
                f"T5-small requiere transformers>=5.0; tienes {transformers.__version__}. "
                "Actualiza con: python -m pip install --upgrade -r requirements-corrector-t5.txt"
            )

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def suggest(self, text: str) -> str:
        prompt = f"grammar: {text}"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=128, num_beams=4,
                                         early_stopping=True)
        suggestion = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return suggestion or text


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
        self.routing = tk.StringVar(value="auto")
        self.status = tk.StringVar(value="Cargando modelo y cámara…")
        self.prediction = tk.StringVar(value="Predicción: —")
        self.session = TextAccumulator()
        self.corrector = None
        self.engine = None
        self.capture = None
        self.photo = None

        top = ttk.Frame(root, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="Clasificador estático:").pack(side="left")
        ttk.Combobox(top, textvariable=self.mode, values=("maps", "crop"),
                     state="readonly", width=12).pack(side="left", padx=6)
        ttk.Label(top, text="Enrutamiento:").pack(side="left", padx=(10, 0))
        ttk.Combobox(top, textvariable=self.routing, values=("auto", "static", "dynamic"),
                     state="readonly", width=10).pack(side="left", padx=6)
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
            from src.inference import RecognitionEngine
            new_engine = RecognitionEngine(ARTIFACTS, static_mode=self.mode.get(), routing_mode=self.routing.get())
            old_engine = self.engine
            self.engine = new_engine
            if old_engine:
                old_engine.close()
            message = f"Estático {self.mode.get()} | enrutamiento {self.routing.get()}"
            if not self.engine.dynamic_available:
                message += "; modelo temporal aún no instalado (solo estático)"
            elif self.engine.dynamic_type == "maps":
                message += "; temporal Notebook 14 (mapas de landmarks)"
            else:
                message += "; temporal de flujo óptico"
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
        try:
            if hasattr(self, "_get_corrector") and self.corrector_mode.get() == "compare":
                suggestions = {
                    "Diccionario": self._get_corrector("dictionary").suggest(text),
                    "T5-small": self._get_corrector("t5-small").suggest(text),
                }
                self.status.set("Comparación de correctores lista")
                self._show_correction_comparison(text, suggestions)
                return
            corrector = (self._get_corrector(self.corrector_mode.get())
                         if hasattr(self, "_get_corrector") else EnglishSpeller())
            suggestion = corrector.suggest(text)
            self.status.set("Captura detenida · sugerencia lista")
        except Exception as exc:
            messagebox.showerror(
                "Corrector unavailable",
                "No se pudo cargar o ejecutar el corrector seleccionado. Actualiza las dependencias "
                "opcionales de T5 y permite la descarga del modelo en el primer uso.\n\n"
                f"Detalle: {type(exc).__name__}: {exc}",
            )
            return
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
                cv2.putText(display, f"{shown}  {confidence:.2f}  [{source}/{self.routing.get()}]", (12, 56),
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


class StyledLiveApp(LiveApp):
    """Pastel visual layout while retaining the existing capture/inference rules."""

    def __init__(self, root: tk.Tk, preview_only: bool = False):
        self.root, self.preview_only = root, preview_only
        root.title("ASL Fingerspelling — Live Camera")
        root.geometry("1180x800")
        root.minsize(850, 620)
        root.configure(bg=COLORS["bg"])
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Pastel.TCombobox", fieldbackground=COLORS["lavender_soft"],
                        background=COLORS["neutral"], foreground=COLORS["text"],
                        bordercolor=COLORS["border"], padding=5)

        self.mode = tk.StringVar(value="maps")
        self.routing = tk.StringVar(value="auto")
        self.corrector_mode = tk.StringVar(value="dictionary")
        self.status = tk.StringVar(value="Preparando cámara…")
        self.prediction = tk.StringVar(value="Predicción: —")
        self.big_letter = tk.StringVar(value="—")
        self.sign_meta = tk.StringVar(value="Esperando detección")
        self.fps_text = tk.StringVar(value="— FPS")
        self.stats_text = tk.StringVar(value="0 caracteres • 0 palabras")
        self.session = TextAccumulator()  # conserva umbral y estabilidad actuales
        self.corrector = None
        self.correctors = {}
        self.engine = None
        self.capture = None
        self.photo = None
        self.recent_tokens = deque(maxlen=8)
        self.last_frame_time = time.monotonic()
        self.fps_smooth = 0.0

        self.main = tk.Frame(root, bg=COLORS["bg"])
        self.main.pack(fill="both", expand=True, padx=14, pady=12)
        self.main.columnconfigure(0, weight=3)
        self.main.columnconfigure(1, weight=1, minsize=265)
        self.main.rowconfigure(1, weight=1)

        # Header and model selectors
        self.header = RoundedCard(self.main, padding=10)
        self.header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        header_inner = self.header.inner
        tk.Label(header_inner, text="✦ ASL Fingerspelling", font=("Segoe UI", 15, "bold"),
                 bg=COLORS["card"], fg=COLORS["text"]).pack(side="left", padx=(5, 10))
        tk.Label(header_inner, text="Reconocimiento en vivo", font=("Segoe UI", 10),
                 bg=COLORS["card"], fg=COLORS["muted"]).pack(side="left", padx=(0, 10))
        tk.Label(header_inner, text="Estático", font=("Segoe UI", 9, "bold"),
                 bg=COLORS["card"], fg=COLORS["muted"]).pack(side="left", padx=(0, 4))
        self.mode_combo = ttk.Combobox(header_inner, textvariable=self.mode, values=("maps", "crop"),
                                       state="readonly", width=7, style="Pastel.TCombobox")
        self.mode_combo.pack(side="left", padx=(0, 8))
        tk.Label(header_inner, text="Enrutamiento", font=("Segoe UI", 9, "bold"),
                 bg=COLORS["card"], fg=COLORS["muted"]).pack(side="left", padx=(0, 4))
        self.routing_combo = ttk.Combobox(header_inner, textvariable=self.routing,
                                          values=("auto", "static", "dynamic"), state="readonly",
                                          width=9, style="Pastel.TCombobox")
        self.routing_combo.pack(side="left", padx=(0, 8))
        self.reload_button = RoundedButton(header_inner, "Aplicar", self.reload_engine,
                                           bg_color=COLORS["neutral"], hover_color=COLORS["neutral_hover"],
                                           fg_color=COLORS["text"], padx=12, pady=6)
        self.reload_button.pack(side="left")
        tk.Label(header_inner, textvariable=self.fps_text, font=("Segoe UI", 9, "bold"),
                 bg=COLORS["sky_soft"], fg="#2B6CB0", padx=9, pady=5).pack(side="right", padx=(6, 2))
        self.status_badge = tk.Label(header_inner, textvariable=self.status,
                                     font=("Segoe UI", 9, "bold"), bg=COLORS["lavender_soft"],
                                     fg=COLORS["lavender_hover"], padx=10, pady=5)
        self.status_badge.pack(side="right", padx=4)

        # Camera card and its unchanged start/stop controls
        self.video_card = RoundedCard(self.main, radius=20, padding=10)
        self.video_card.grid(row=1, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        self.video_card.inner.columnconfigure(0, weight=1)
        self.video_card.inner.rowconfigure(0, weight=1)
        self.video_card.inner.rowconfigure(1, weight=0)
        self.video_holder = tk.Frame(self.video_card.inner, bg=COLORS["card"])
        self.video_holder.grid(row=0, column=0, sticky="nsew")
        self.video_view = tk.Label(self.video_holder, bg=COLORS["card"], anchor="center")
        self.video_view.pack(fill="both", expand=True)
        controls = tk.Frame(self.video_card.inner, bg=COLORS["card"])
        controls.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self.start_button = RoundedButton(controls, "▶  Iniciar captura de letras", self.start_capture,
                                          bg_color=COLORS["mint"], hover_color=COLORS["mint_hover"])
        self.start_button.pack(side="left", padx=(0, 8))
        self.stop_button = RoundedButton(controls, "■  Detener y corregir texto", self.stop_capture,
                                         bg_color=COLORS["peach"], hover_color=COLORS["peach_hover"],
                                         state="disabled")
        self.stop_button.pack(side="left")

        # Prediction and confidence telemetry
        self.side_card = RoundedCard(self.main, radius=20, padding=12)
        self.side_card.grid(row=1, column=1, sticky="nsew", pady=(0, 10))
        prediction_box = tk.Frame(self.side_card.inner, bg=COLORS["lavender_soft"], padx=12, pady=10)
        prediction_box.pack(fill="x", pady=(0, 10))
        tk.Label(prediction_box, text="SEÑA DETECTADA", font=("Segoe UI", 9, "bold"),
                 bg=COLORS["lavender_soft"], fg=COLORS["lavender_hover"]).pack(anchor="w")
        row = tk.Frame(prediction_box, bg=COLORS["lavender_soft"])
        row.pack(fill="x", pady=(3, 5))
        tk.Label(row, textvariable=self.big_letter, font=("Segoe UI", 38, "bold"), width=3,
                 bg=COLORS["lavender_soft"], fg=COLORS["text"]).pack(side="left")
        tk.Label(row, textvariable=self.sign_meta, font=("Segoe UI", 10, "bold"), justify="left",
                 anchor="w", bg=COLORS["lavender_soft"], fg=COLORS["text"]).pack(side="left", fill="x", expand=True)
        self.conf_bar = PastelProgressBar(prediction_box, fill_color=COLORS["lavender"])
        self.conf_bar.pack(fill="x")
        tk.Label(self.side_card.inner, text="Historial reciente", font=("Segoe UI", 10, "bold"),
                 bg=COLORS["card"], fg=COLORS["muted"]).pack(anchor="w", pady=(3, 5))
        self.history = tk.Frame(self.side_card.inner, bg=COLORS["card"])
        self.history.pack(fill="x")
        self._render_history()
        tk.Label(self.side_card.inner, textvariable=self.prediction, font=("Segoe UI", 9),
                 bg=COLORS["card"], fg=COLORS["muted"], wraplength=250, justify="left").pack(anchor="w", pady=(12, 0))

        # Editable transcript area
        self.text_card = RoundedCard(self.main, padding=12)
        self.text_card.grid(row=2, column=0, columnspan=2, sticky="ew")
        text_header = tk.Frame(self.text_card.inner, bg=COLORS["card"])
        text_header.pack(fill="x", pady=(0, 6))
        tk.Label(text_header, text="Texto deletreado", font=("Segoe UI", 11, "bold"),
                 bg=COLORS["card"], fg=COLORS["text"]).pack(side="left")
        tk.Label(text_header, text="Corrector:", font=("Segoe UI", 9, "bold"),
                 bg=COLORS["card"], fg=COLORS["muted"]).pack(side="left", padx=(16, 5))
        self.corrector_combo = ttk.Combobox(text_header, textvariable=self.corrector_mode,
                                            values=("dictionary", "t5-small", "compare"), state="readonly",
                                            width=11, style="Pastel.TCombobox")
        self.corrector_combo.pack(side="left")
        tk.Label(text_header, textvariable=self.stats_text, font=("Segoe UI", 9, "bold"),
                 bg=COLORS["sky_soft"], fg="#2B6CB0", padx=8, pady=3).pack(side="right")
        self.clear_button = RoundedButton(text_header, "Limpiar", self.clear_text,
                                          bg_color=COLORS["neutral"], hover_color=COLORS["neutral_hover"],
                                          fg_color=COLORS["text"], padx=12, pady=5)
        self.clear_button.pack(side="right", padx=8)
        self.text_widget = tk.Text(self.text_card.inner, height=3, wrap="word", font=("Segoe UI", 16, "bold"),
                                   bg=COLORS["card_tint"], fg=COLORS["text"], bd=0,
                                   highlightthickness=1, highlightbackground=COLORS["border"],
                                   padx=12, pady=8)
        self.text_widget.pack(fill="x")
        self.text_widget.bind("<KeyRelease>", lambda _event: self._update_stats())

        try:
            if self.preview_only:
                self.reload_button.configure(state="disabled")
                self.start_button.configure(state="disabled")
                self.stop_button.configure(state="disabled")
                self.status.set("Vista previa sin modelo")
            else:
                self.reload_engine(show_message=False)
            self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
            if not self.capture.isOpened():
                raise RuntimeError("No se pudo abrir la cámara. Revisa que no esté ocupada.")
            if not self.preview_only:
                self.status.set("Listo · texto pausado")
            self.update_frame()
        except Exception as exc:
            messagebox.showerror("No se pudo iniciar", str(exc))
            self.status.set("Revisa cámara y modelos")
        root.protocol("WM_DELETE_WINDOW", self.close)

    def reload_engine(self, show_message=True):
        try:
            from src.inference import RecognitionEngine
            new_engine = RecognitionEngine(ARTIFACTS, static_mode=self.mode.get(),
                                           routing_mode=self.routing.get())
            old_engine, self.engine = self.engine, new_engine
            if old_engine:
                old_engine.close()
            message = f"{self.mode.get()} · {self.routing.get()}"
            if self.engine.dynamic_available:
                message += f" · temporal {self.engine.dynamic_type}"
            else:
                message += " · solo estático"
            self.status.set(message)
            if show_message:
                messagebox.showinfo("Modelos cargados", message)
        except Exception as exc:
            self.status.set("No se pudo cargar la rama")
            if show_message:
                messagebox.showerror("Error al cargar", str(exc))
            elif self.engine is None:
                raise

    def clear_text(self):
        super().clear_text()
        self.recent_tokens.clear()
        self._render_history()
        self._update_stats()

    def _render_history(self):
        for child in self.history.winfo_children():
            child.destroy()
        if not self.recent_tokens:
            tk.Label(self.history, text="Aún no se han añadido letras", font=("Segoe UI", 9, "italic"),
                     bg=COLORS["card"], fg=COLORS["muted"]).pack(anchor="w")
            return
        shades = [COLORS["lavender_soft"], COLORS["mint_soft"], COLORS["sky_soft"]]
        for index, token in enumerate(self.recent_tokens):
            tk.Label(self.history, text=token, font=("Segoe UI", 10, "bold"),
                     bg=shades[index % len(shades)], fg=COLORS["text"], padx=8, pady=3).pack(side="left", padx=(0, 5))

    def _update_stats(self):
        text = self.text_widget.get("1.0", "end-1c")
        self.stats_text.set(f"{len(text)} caracteres · {len(text.split()) if text.strip() else 0} palabras")

    def _sync_text(self):
        super()._sync_text()
        self._update_stats()

    def _get_corrector(self, mode=None):
        mode = mode or self.corrector_mode.get()
        if mode not in self.correctors:
            if mode == "t5-small":
                self.status.set("Cargando T5-small… (primera vez puede descargar el modelo)")
                self.root.update_idletasks()
                self.correctors[mode] = T5GrammarCorrector()
            elif mode == "dictionary":
                self.correctors[mode] = EnglishSpeller()
            else:
                raise ValueError(f"Corrector no reconocido: {mode}")
        return self.correctors[mode]

    def _show_correction_comparison(self, original, suggestions):
        dialog = tk.Toplevel(self.root)
        dialog.title("Comparar sugerencias de corrección")
        dialog.configure(bg=COLORS["bg"])
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)
        tk.Label(dialog, text="Texto original", font=("Segoe UI", 10, "bold"),
                 bg=COLORS["bg"], fg=COLORS["text"]).pack(anchor="w", padx=16, pady=(14, 3))
        tk.Label(dialog, text=original, font=("Segoe UI", 11), wraplength=520, justify="left",
                 bg=COLORS["card"], fg=COLORS["text"], padx=10, pady=8).pack(fill="x", padx=16)
        result = {"text": None}
        for name, suggestion in suggestions.items():
            tk.Label(dialog, text=f"Sugerencia {name}", font=("Segoe UI", 10, "bold"),
                     bg=COLORS["bg"], fg=COLORS["text"]).pack(anchor="w", padx=16, pady=(10, 3))
            tk.Label(dialog, text=suggestion, font=("Segoe UI", 11), wraplength=520, justify="left",
                     bg=COLORS["card"], fg=COLORS["text"], padx=10, pady=8).pack(fill="x", padx=16)
            RoundedButton(dialog, f"Usar sugerencia {name}",
                          command=lambda value=suggestion: choose(value),
                          bg_color=COLORS["mint"], hover_color=COLORS["mint_hover"],
                          padx=12, pady=5).pack(anchor="e", padx=16, pady=(5, 0))

        def close_dialog():
            dialog.destroy()

        def choose(value):
            result["text"] = value
            close_dialog()

        RoundedButton(dialog, "Conservar original", command=close_dialog,
                      bg_color=COLORS["neutral"], hover_color=COLORS["neutral_hover"],
                      fg_color=COLORS["text"], padx=12, pady=5).pack(anchor="e", padx=16, pady=12)
        dialog.wait_window()
        if result["text"] is not None:
            self.session.text = result["text"]
            self._sync_text()

    def update_frame(self):
        if not self.capture or not self.capture.isOpened():
            return
        ok, frame = self.capture.read()
        if ok:
            now = time.monotonic()
            fps = 1.0 / max(now - self.last_frame_time, 1e-4)
            self.last_frame_time = now
            self.fps_smooth = fps if self.fps_smooth == 0 else 0.85 * self.fps_smooth + 0.15 * fps
            self.fps_text.set(f"{self.fps_smooth:.0f} FPS")
            if self.preview_only:
                display = cv2.flip(frame, 1)
                cv2.putText(display, "VISTA PREVIA · SIN MODELO", (16, 34), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (80, 145, 220), 2, cv2.LINE_AA)
                self.prediction.set("Vista previa de cámara; la clasificación está desactivada.")
            else:
                if not self.engine:
                    return
                display, label, confidence, source, detected = self.engine.predict(frame)
                # Espejar el video anotado antes de dibujar texto para que los
                # rótulos de predicción permanezcan legibles en pantalla.
                display = cv2.flip(display, 1)
                accepted = self.session.observe(label, confidence)
                if accepted:
                    if accepted == "<":
                        self.text_widget.delete("1.0", "end")
                        self.text_widget.insert("1.0", self.session.text)
                        self.recent_tokens.append("⌫")
                    else:
                        self.text_widget.insert("end", accepted)
                        self.text_widget.see("end")
                        self.recent_tokens.append("␣" if accepted == " " else accepted)
                    self._render_history()
                    self._update_stats()
                shown = label.upper() if label is not None else "—"
                self.big_letter.set(shown if detected else "—")
                self.sign_meta.set((f"Confianza: {confidence:.0%}\nRama: {source}" if detected
                                    else "Mano no detectada"))
                self.conf_bar.set_value(confidence if detected else 0.0,
                                        COLORS["mint"] if confidence >= self.session.confidence_threshold else COLORS["lavender"])
                state = "CAPTURANDO" if self.session.active else "VISTA PREVIA · texto pausado"
                overlay = display.copy()
                cv2.rectangle(overlay, (0, 0), (display.shape[1], 43), (43, 39, 59), -1)
                cv2.addWeighted(overlay, 0.62, display, 0.38, 0, display)
                cv2.putText(display, f"{state}   {shown}   {confidence:.2f}   [{source}/{self.routing.get()}]",
                            (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (248, 244, 255), 2, cv2.LINE_AA)
                self.prediction.set(f"Predicción: {shown} · confianza {confidence:.2f} · rama {source}")
                if not detected:
                    cv2.putText(display, "Muestra tu mano frente a la cámara", (14, 72),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 200, 255), 2, cv2.LINE_AA)
            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            image = resize_to_fit(image, self.video_holder.winfo_width(), self.video_holder.winfo_height())
            image = rounded_image(image)
            self.photo = ImageTk.PhotoImage(image)
            self.video_view.configure(image=self.photo)
        self.root.after(15, self.update_frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-only", action="store_true",
                        help="Abre la interfaz y cámara sin modelos para verificar el dispositivo.")
    args = parser.parse_args()
    root = tk.Tk()
    StyledLiveApp(root, preview_only=args.preview_only)
    root.mainloop()


if __name__ == "__main__":
    main()
