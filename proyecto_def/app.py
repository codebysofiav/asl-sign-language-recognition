"""Tkinter camera app for live ASL fingerspelling (Pastel UI + Live Settings & Telemetry)."""

from __future__ import annotations

import argparse
import re
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageDraw, ImageTk
from spellchecker import SpellChecker

from src.inference import RecognitionEngine


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

# Paleta Pastel Suave de Alto Contraste
COLORS = {
    "bg": "#F6F3FA",              # Lavanda muy pálido (fondo principal)
    "card": "#FFFFFF",            # Blanco puro para tarjetas
    "card_tint": "#FAF8FE",       # Lavanda casi blanco para sub-paneles
    "border": "#E5DDF2",          # Borde lila pastel
    "text_primary": "#2D2A3E",    # Ciruela oscuro (texto principal)
    "text_secondary": "#6E6A86",  # Gris malva (texto secundario)

    # Acentos pastel
    "lavender": "#9D8DF1",
    "lavender_hover": "#8774E8",
    "lavender_soft": "#EFEAFF",

    "mint": "#6BCB9F",
    "mint_hover": "#52B788",
    "mint_soft": "#E6F8F0",

    "peach": "#F4978E",
    "peach_hover": "#E57A70",
    "peach_soft": "#FDECEB",

    "sky": "#7CC6FE",
    "sky_soft": "#E8F4FE",

    "butter": "#F6C26B",
    "butter_soft": "#FEF6E6",

    "neutral_btn": "#EDE8F5",
    "neutral_hover": "#DFD7EC",
    "disabled_bg": "#ECE9F2",
    "disabled_fg": "#A8A3B8",
}


class RoundedButton(tk.Canvas):
    """Botón personalizado con bordes redondeados y estados hover/disabled."""

    def __init__(
        self,
        parent,
        text: str,
        command=None,
        bg_color: str = COLORS["lavender"],
        hover_color: str = COLORS["lavender_hover"],
        fg_color: str = "#FFFFFF",
        radius: int = 14,
        padx: int = 16,
        pady: int = 8,
        font=("Segoe UI", 10, "bold"),
        state: str = "normal",
    ):
        super().__init__(
            parent,
            highlightthickness=0,
            bd=0,
            bg=parent.cget("bg"),
            cursor="hand2" if state == "normal" else "arrow",
        )
        self.command = command
        self._text = text
        self._bg_color = bg_color
        self._hover_color = hover_color
        self._fg_color = fg_color
        self._radius = radius
        self._padx = padx
        self._pady = pady
        self._font = font
        self._state = state
        self._hovered = False

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonRelease-1>", self._on_click)
        self._redraw()

    def _round_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = [
            x1 + r, y1,
            x2 - r, y1,
            x2, y1,
            x2, y1 + r,
            x2, y2 - r,
            x2, y2,
            x2 - r, y2,
            x1 + r, y2,
            x1, y2,
            x1, y2 - r,
            x1, y1 + r,
            x1, y1,
        ]
        return self.create_polygon(points, smooth=True, **kwargs)

    def _redraw(self):
        self.delete("all")
        temp_id = self.create_text(0, 0, text=self._text, font=self._font)
        bbox = self.bbox(temp_id)
        self.delete(temp_id)

        text_w = (bbox[2] - bbox[0]) if bbox else 76
        text_h = (bbox[3] - bbox[1]) if bbox else 20
        width = text_w + self._padx * 2
        height = text_h + self._pady * 2
        self.configure(width=width, height=height)

        if self._state == "disabled":
            fill = COLORS["disabled_bg"]
            fg = COLORS["disabled_fg"]
        else:
            fill = self._hover_color if self._hovered else self._bg_color
            fg = self._fg_color

        r = min(self._radius, width // 2, height // 2)
        self._round_rect(1, 1, width - 1, height - 1, r, fill=fill, outline="")
        self.create_text(width / 2, height / 2, text=self._text, fill=fg, font=self._font)

    def set_metrics(self, font=None, padx: int | None = None, pady: int | None = None):
        if font is not None:
            self._font = font
        if padx is not None:
            self._padx = padx
        if pady is not None:
            self._pady = pady
        self._redraw()

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
    """Contenedor con fondo redondeado dibujado en Canvas."""

    def __init__(
        self,
        parent,
        radius: int = 18,
        padding: int = 12,
        bg_color: str = COLORS["card"],
        border_color: str = COLORS["border"],
    ):
        super().__init__(parent, bg=parent.cget("bg"))
        self.radius = radius
        self.bg_color = bg_color
        self.border_color = border_color
        self._padding = padding

        self.canvas = tk.Canvas(self, bg=parent.cget("bg"), highlightthickness=0, bd=0)
        self.canvas.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.inner = tk.Frame(self, bg=bg_color)
        self.inner.pack(fill="both", expand=True, padx=padding, pady=padding)

        self.bind("<Configure>", self._draw_bg)

    def set_padding(self, padding: int):
        if padding != self._padding:
            self._padding = padding
            self.inner.pack_configure(padx=padding, pady=padding)

    def _draw_bg(self, _event=None):
        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 2 or h <= 2:
            return
        self.canvas.delete("bg_shape")
        r = min(self.radius, w // 2, h // 2)
        points = [
            1 + r, 1,
            w - 1 - r, 1,
            w - 1, 1,
            w - 1, 1 + r,
            w - 1, h - 1 - r,
            w - 1, h - 1,
            w - 1 - r, h - 1,
            1 + r, h - 1,
            1, h - 1,
            1, h - 1 - r,
            1, 1 + r,
            1, 1,
        ]
        self.canvas.create_polygon(
            points,
            smooth=True,
            fill=self.bg_color,
            outline=self.border_color,
            width=1.5,
            tags="bg_shape",
        )


class PastelProgressBar(tk.Canvas):
    """Barra de progreso redondeada pastel para confianza y temporizador."""

    def __init__(self, parent, height: int = 12, track_color: str = "#EDE8F5", fill_color: str = COLORS["mint"]):
        super().__init__(parent, height=height, bg=parent.cget("bg"), highlightthickness=0, bd=0)
        self.track_color = track_color
        self.fill_color = fill_color
        self._value = 0.0
        self.bind("<Configure>", lambda _e: self._draw())

    def set_value(self, value: float, fill_color: str | None = None):
        self._value = max(0.0, min(1.0, float(value)))
        if fill_color:
            self.fill_color = fill_color
        self._draw()

    def _draw(self):
        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 4 or h <= 2:
            return
        self.delete("all")
        r = h // 2
        self._pill(0, 0, w, h, r, self.track_color)
        fill_w = int(w * self._value)
        if fill_w > h:
            self._pill(0, 0, fill_w, h, r, self.fill_color)
        elif fill_w > 0:
            self._pill(0, 0, h, h, r, self.fill_color)

    def _pill(self, x1, y1, x2, y2, r, color):
        self.create_oval(x1, y1, x1 + 2 * r, y2, fill=color, outline="")
        self.create_oval(x2 - 2 * r, y1, x2, y2, fill=color, outline="")
        self.create_rectangle(x1 + r, y1, x2 - r, y2, fill=color, outline="")


def apply_rounded_corners(pil_img: Image.Image, radius: int = 18) -> Image.Image:
    """Recorta una imagen PIL con esquinas redondeadas sobre fondo blanco."""
    w, h = pil_img.size
    if w <= 4 or h <= 4:
        return pil_img
    r = min(radius, w // 2, h // 2)
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, w, h), radius=r, fill=255)
    bg = Image.new("RGB", (w, h), COLORS["card"])
    bg.paste(pil_img, mask=mask)
    return bg


def resize_to_fit(pil_img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Escala la imagen para llenar el contenedor disponible conservando proporción."""
    src_w, src_h = pil_img.size
    if src_w <= 0 or src_h <= 0 or target_w <= 10 or target_h <= 10:
        return pil_img
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    if (new_w, new_h) == (src_w, src_h):
        return pil_img
    return pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)


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
    """Acumula letras tras mantener una seña estable durante `hold_seconds`."""

    def __init__(self, hold_seconds: float = 1.0, confidence_threshold: float = 0.78):
        self.active = False
        self.text = ""
        self.hold_seconds = hold_seconds
        self.confidence_threshold = confidence_threshold
        self.candidate: str | None = None
        self.candidate_since: float | None = None
        self.last_accepted: str | None = None

    def reset_gate(self):
        self.candidate = None
        self.candidate_since = None
        self.last_accepted = None

    def get_countdown_info(self) -> tuple[str | None, float, float]:
        """Devuelve (etiqueta_candidata, segundos_restantes, progreso_0_a_1)."""
        if not self.active or not self.candidate or self.candidate_since is None:
            return None, 0.0, 0.0
        elapsed = time.monotonic() - self.candidate_since
        remaining = max(0.0, self.hold_seconds - elapsed)
        progress = min(1.0, elapsed / max(0.05, self.hold_seconds))
        return self.candidate.upper(), remaining, progress

    def observe(self, label, confidence):
        if label is None or confidence < self.confidence_threshold:
            self.candidate = None
            self.candidate_since = None
            return None

        key = str(label).strip().lower()
        if key in {"nothing", "other"}:
            self.candidate = None
            self.candidate_since = None
            self.last_accepted = None
            return None

        if not self.active:
            self.candidate = None
            self.candidate_since = None
            return None

        if key == self.last_accepted:
            self.candidate = None
            self.candidate_since = None
            return None

        now = time.monotonic()
        if key != self.candidate:
            self.candidate = key
            self.candidate_since = now
            return None

        elapsed = now - (self.candidate_since or now)
        if elapsed < self.hold_seconds:
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
        self.candidate = None
        self.candidate_since = None
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
        root.title("ASL Fingerspelling — Estudio Pastel")
        root.geometry("1180x800")
        root.minsize(820, 600)
        root.configure(bg=COLORS["bg"])

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Pastel.TCombobox",
            fieldbackground=COLORS["lavender_soft"],
            background=COLORS["neutral_btn"],
            foreground=COLORS["text_primary"],
            bordercolor=COLORS["border"],
            lightcolor=COLORS["border"],
            darkcolor=COLORS["border"],
            padding=5,
        )
        style.configure(
            "Pastel.Horizontal.TScale",
            background=COLORS["card_tint"],
            troughcolor=COLORS["lavender_soft"],
            bordercolor=COLORS["border"],
            lightcolor=COLORS["lavender"],
            darkcolor=COLORS["lavender"],
        )

        # Variables reactivas de configuración y estado
        self.hold_var = tk.DoubleVar(value=1.0)
        self.conf_var = tk.DoubleVar(value=0.78)
        self.mirror_var = tk.BooleanVar(value=True)
        self.auto_spell_var = tk.BooleanVar(value=True)
        self.camera_idx_var = tk.IntVar(value=0)

        self.mode = tk.StringVar(value="crop")
        self.status = tk.StringVar(value="Iniciando estudio…")
        self.big_letter_var = tk.StringVar(value="—")
        self.sign_meta_var = tk.StringVar(value="Esperando detección")
        self.timer_var = tk.StringVar(value="Listo (1.0 s)")
        self.fps_var = tk.StringVar(value="FPS: —")
        self.stats_var = tk.StringVar(value="0 letras • 0 palabras")
        self.hold_label_var = tk.StringVar(value="1.0 s")
        self.conf_label_var = tk.StringVar(value="78 %")

        self.session = TextAccumulator(
            hold_seconds=self.hold_var.get(),
            confidence_threshold=self.conf_var.get(),
        )
        self.corrector = EnglishSpeller()
        self.engine = None
        self.capture = None
        self.photo = None
        self._last_ui_bucket = None
        self._last_frame_ts = time.monotonic()
        self._fps_smooth = 0.0
        self.recent_tokens: deque[str] = deque(maxlen=8)

        # Contenedor principal responsivo
        self.main_container = tk.Frame(root, bg=COLORS["bg"])
        self.main_container.pack(fill="both", expand=True, padx=14, pady=12)
        self.main_container.columnconfigure(0, weight=3)  # Columna izquierda: cámara + texto
        self.main_container.columnconfigure(1, weight=1, minsize=290)  # Columna derecha: telemetría + ajustes
        self.main_container.rowconfigure(0, weight=0)  # Header
        self.main_container.rowconfigure(1, weight=1)  # Cuerpo principal
        self.main_container.rowconfigure(2, weight=0)  # Texto deletreado

        # ================= 1. BARRA SUPERIOR =================
        self.top_card = RoundedCard(self.main_container, radius=18, padding=10)
        self.top_card.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        title_box = tk.Frame(self.top_card.inner, bg=COLORS["card"])
        title_box.pack(side="left", padx=(4, 14))

        self.title_label = tk.Label(
            title_box,
            text="✦ ASL Fingerspelling",
            font=("Segoe UI", 15, "bold"),
            bg=COLORS["card"],
            fg=COLORS["text_primary"],
        )
        self.title_label.pack(side="left")

        tk.Label(
            title_box,
            text="  |  Reconocimiento en vivo",
            font=("Segoe UI", 11),
            bg=COLORS["card"],
            fg=COLORS["text_secondary"],
        ).pack(side="left")

        # Selector de rama
        branch_box = tk.Frame(self.top_card.inner, bg=COLORS["card"])
        branch_box.pack(side="left", padx=8)

        tk.Label(
            branch_box,
            text="Modelo:",
            font=("Segoe UI", 10, "bold"),
            bg=COLORS["card"],
            fg=COLORS["text_secondary"],
        ).pack(side="left", padx=(0, 6))

        self.mode_combo = ttk.Combobox(
            branch_box,
            textvariable=self.mode,
            values=("crop", "maps"),
            state="readonly",
            width=8,
            font=("Segoe UI", 10, "bold"),
            style="Pastel.TCombobox",
        )
        self.mode_combo.pack(side="left", padx=(0, 6))

        self.reload_button = RoundedButton(
            branch_box,
            text="Aplicar rama",
            command=self.reload_engine,
            bg_color=COLORS["neutral_btn"],
            hover_color=COLORS["neutral_hover"],
            fg_color=COLORS["text_primary"],
            radius=11,
            padx=12,
            pady=5,
            font=("Segoe UI", 9, "bold"),
        )
        self.reload_button.pack(side="left")

        # Badges de estado y FPS a la derecha
        self.fps_badge = tk.Label(
            self.top_card.inner,
            textvariable=self.fps_var,
            font=("Segoe UI", 9, "bold"),
            bg=COLORS["sky_soft"],
            fg="#2B6CB0",
            padx=10,
            pady=5,
        )
        self.fps_badge.pack(side="right", padx=(6, 2))

        self.status_badge = tk.Label(
            self.top_card.inner,
            textvariable=self.status,
            font=("Segoe UI", 10, "bold"),
            bg=COLORS["lavender_soft"],
            fg=COLORS["lavender_hover"],
            padx=12,
            pady=5,
        )
        self.status_badge.pack(side="right", padx=4)

        # ================= 2A. TARJETA DE VIDEO (IZQUIERDA) =================
        self.video_card = RoundedCard(self.main_container, radius=20, padding=10)
        self.video_card.grid(row=1, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        self.video_card.inner.columnconfigure(0, weight=1)
        self.video_card.inner.rowconfigure(0, weight=1)
        self.video_card.inner.rowconfigure(1, weight=0)

        self.video_holder = tk.Frame(self.video_card.inner, bg=COLORS["card"])
        self.video_holder.grid(row=0, column=0, sticky="nsew")
        self.video_holder.pack_propagate(False)

        self.video_view = tk.Label(self.video_holder, bg=COLORS["card"], anchor="center")
        self.video_view.pack(fill="both", expand=True)

        # Barra de botones debajo del video
        controls = tk.Frame(self.video_card.inner, bg=COLORS["card"])
        controls.grid(row=1, column=0, sticky="ew", pady=(10, 0))

        self.start_button = RoundedButton(
            controls,
            text="▶  Iniciar captura",
            command=self.start_capture,
            bg_color=COLORS["mint"],
            hover_color=COLORS["mint_hover"],
            radius=14,
            padx=18,
            pady=8,
            font=("Segoe UI", 11, "bold"),
        )
        self.start_button.pack(side="left", padx=(0, 8))

        self.stop_button = RoundedButton(
            controls,
            text="■  Detener y revisar",
            command=self.stop_capture,
            bg_color=COLORS["peach"],
            hover_color=COLORS["peach_hover"],
            radius=14,
            padx=18,
            pady=8,
            font=("Segoe UI", 11, "bold"),
            state="disabled",
        )
        self.stop_button.pack(side="left", padx=(0, 8))

        self.copy_button = RoundedButton(
            controls,
            text="Copiar",
            command=self.copy_text,
            bg_color=COLORS["sky_soft"],
            hover_color=COLORS["sky"],
            fg_color=COLORS["text_primary"],
            radius=14,
            padx=14,
            pady=8,
            font=("Segoe UI", 10, "bold"),
        )
        self.copy_button.pack(side="right", padx=(6, 0))

        self.clear_button = RoundedButton(
            controls,
            text="Limpiar",
            command=self.clear_text,
            bg_color=COLORS["neutral_btn"],
            hover_color=COLORS["neutral_hover"],
            fg_color=COLORS["text_primary"],
            radius=14,
            padx=14,
            pady=8,
            font=("Segoe UI", 10, "bold"),
        )
        self.clear_button.pack(side="right")

        # ================= 2B. PANEL LATERAL: TELEMETRÍA + AJUSTES (DERECHA) =================
        self.side_card = RoundedCard(self.main_container, radius=20, padding=12)
        self.side_card.grid(row=1, column=1, sticky="nsew", pady=(0, 10))
        self.side_card.inner.columnconfigure(0, weight=1)

        # Bloque: Predicción en vivo
        pred_box = tk.Frame(
            self.side_card.inner,
            bg=COLORS["lavender_soft"],
            padx=12,
            pady=10,
            highlightthickness=1,
            highlightbackground="#DFD5F8",
        )
        pred_box.pack(fill="x", pady=(0, 10))

        tk.Label(
            pred_box,
            text="SEÑA DETECTADA",
            font=("Segoe UI", 9, "bold"),
            bg=COLORS["lavender_soft"],
            fg=COLORS["lavender_hover"],
        ).pack(anchor="w")

        letter_row = tk.Frame(pred_box, bg=COLORS["lavender_soft"])
        letter_row.pack(fill="x", pady=(2, 4))

        self.big_letter_label = tk.Label(
            letter_row,
            textvariable=self.big_letter_var,
            font=("Segoe UI", 34, "bold"),
            bg=COLORS["lavender_soft"],
            fg=COLORS["text_primary"],
            width=3,
        )
        self.big_letter_label.pack(side="left")

        meta_col = tk.Frame(letter_row, bg=COLORS["lavender_soft"])
        meta_col.pack(side="left", fill="both", expand=True, padx=(6, 0))

        tk.Label(
            meta_col,
            textvariable=self.sign_meta_var,
            font=("Segoe UI", 10, "bold"),
            bg=COLORS["lavender_soft"],
            fg=COLORS["text_primary"],
            anchor="w",
            justify="left",
        ).pack(fill="x", pady=(4, 4))

        self.conf_bar = PastelProgressBar(meta_col, height=10, track_color="#DDD4F3", fill_color=COLORS["lavender"])
        self.conf_bar.pack(fill="x")

        # Bloque: Progreso de escritura
        timer_box = tk.Frame(
            self.side_card.inner,
            bg=COLORS["mint_soft"],
            padx=12,
            pady=8,
            highlightthickness=1,
            highlightbackground="#C9EEDB",
        )
        timer_box.pack(fill="x", pady=(0, 10))

        timer_hdr = tk.Frame(timer_box, bg=COLORS["mint_soft"])
        timer_hdr.pack(fill="x", pady=(0, 4))

        tk.Label(
            timer_hdr,
            text="PROGRESO DE CAPTURA",
            font=("Segoe UI", 9, "bold"),
            bg=COLORS["mint_soft"],
            fg=COLORS["mint_hover"],
        ).pack(side="left")

        tk.Label(
            timer_hdr,
            textvariable=self.timer_var,
            font=("Segoe UI", 9, "bold"),
            bg=COLORS["mint_soft"],
            fg=COLORS["text_primary"],
        ).pack(side="right")

        self.hold_bar = PastelProgressBar(timer_box, height=10, track_color="#CBEAD9", fill_color=COLORS["mint"])
        self.hold_bar.pack(fill="x")

        # Bloque: Historial reciente de letras aceptadas
        hist_box = tk.Frame(self.side_card.inner, bg=COLORS["card"])
        hist_box.pack(fill="x", pady=(0, 10))

        tk.Label(
            hist_box,
            text="Historial reciente",
            font=("Segoe UI", 10, "bold"),
            bg=COLORS["card"],
            fg=COLORS["text_secondary"],
        ).pack(anchor="w", pady=(0, 4))

        self.chips_frame = tk.Frame(hist_box, bg=COLORS["card"])
        self.chips_frame.pack(fill="x")
        self._render_history_chips()

        # Bloque: Ajustes dentro de la interfaz
        settings_box = tk.Frame(
            self.side_card.inner,
            bg=COLORS["card_tint"],
            padx=12,
            pady=10,
            highlightthickness=1,
            highlightbackground=COLORS["border"],
        )
        settings_box.pack(fill="both", expand=True)

        tk.Label(
            settings_box,
            text="⚙  Ajustes en vivo",
            font=("Segoe UI", 11, "bold"),
            bg=COLORS["card_tint"],
            fg=COLORS["text_primary"],
        ).pack(anchor="w", pady=(0, 8))

        # Slider 1: Tiempo por seña
        row_hold = tk.Frame(settings_box, bg=COLORS["card_tint"])
        row_hold.pack(fill="x")
        tk.Label(
            row_hold,
            text="Tiempo por seña:",
            font=("Segoe UI", 9, "bold"),
            bg=COLORS["card_tint"],
            fg=COLORS["text_secondary"],
        ).pack(side="left")
        tk.Label(
            row_hold,
            textvariable=self.hold_label_var,
            font=("Segoe UI", 9, "bold"),
            bg=COLORS["butter_soft"],
            fg="#B7791F",
            padx=6,
            pady=1,
        ).pack(side="right")

        hold_scale = ttk.Scale(
            settings_box,
            from_=0.2,
            to=3.0,
            variable=self.hold_var,
            command=self._on_settings_changed,
            style="Pastel.Horizontal.TScale",
        )
        hold_scale.pack(fill="x", pady=(2, 8))

        # Slider 2: Umbral mínimo de confianza
        row_conf = tk.Frame(settings_box, bg=COLORS["card_tint"])
        row_conf.pack(fill="x")
        tk.Label(
            row_conf,
            text="Confianza mínima:",
            font=("Segoe UI", 9, "bold"),
            bg=COLORS["card_tint"],
            fg=COLORS["text_secondary"],
        ).pack(side="left")
        tk.Label(
            row_conf,
            textvariable=self.conf_label_var,
            font=("Segoe UI", 9, "bold"),
            bg=COLORS["lavender_soft"],
            fg=COLORS["lavender_hover"],
            padx=6,
            pady=1,
        ).pack(side="right")

        conf_scale = ttk.Scale(
            settings_box,
            from_=0.40,
            to=0.95,
            variable=self.conf_var,
            command=self._on_settings_changed,
            style="Pastel.Horizontal.TScale",
        )
        conf_scale.pack(fill="x", pady=(2, 8))

        # Toggles de cámara y ortografía
        toggles_row = tk.Frame(settings_box, bg=COLORS["card_tint"])
        toggles_row.pack(fill="x", pady=(2, 6))

        tk.Checkbutton(
            toggles_row,
            text="Vista espejo",
            variable=self.mirror_var,
            bg=COLORS["card_tint"],
            activebackground=COLORS["card_tint"],
            fg=COLORS["text_primary"],
            selectcolor=COLORS["lavender_soft"],
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left")

        tk.Checkbutton(
            toggles_row,
            text="Revisar ortografía",
            variable=self.auto_spell_var,
            bg=COLORS["card_tint"],
            activebackground=COLORS["card_tint"],
            fg=COLORS["text_primary"],
            selectcolor=COLORS["mint_soft"],
            font=("Segoe UI", 9, "bold"),
        ).pack(side="right")

        # Selector de cámara física (0, 1, 2)
        cam_row = tk.Frame(settings_box, bg=COLORS["card_tint"])
        cam_row.pack(fill="x", pady=(4, 0))

        tk.Label(
            cam_row,
            text="Índice cámara:",
            font=("Segoe UI", 9, "bold"),
            bg=COLORS["card_tint"],
            fg=COLORS["text_secondary"],
        ).pack(side="left")

        cam_spin = tk.Spinbox(
            cam_row,
            from_=0,
            to=4,
            width=4,
            textvariable=self.camera_idx_var,
            font=("Segoe UI", 9, "bold"),
            bg=COLORS["lavender_soft"],
            fg=COLORS["text_primary"],
            bd=0,
            justify="center",
        )
        cam_spin.pack(side="left", padx=6)

        RoundedButton(
            cam_row,
            text="Reconectar",
            command=self.reconnect_camera,
            bg_color=COLORS["neutral_btn"],
            hover_color=COLORS["neutral_hover"],
            fg_color=COLORS["text_primary"],
            radius=10,
            padx=10,
            pady=3,
            font=("Segoe UI", 8, "bold"),
        ).pack(side="right")

        # ================= 3. TARJETA DE TEXTO DELETREADO (ABAJO) =================
        self.text_card = RoundedCard(self.main_container, radius=18, padding=10)
        self.text_card.grid(row=2, column=0, columnspan=2, sticky="ew")

        text_hdr = tk.Frame(self.text_card.inner, bg=COLORS["card"])
        text_hdr.pack(fill="x", pady=(0, 4))

        tk.Label(
            text_hdr,
            text="Texto deletreado (editable antes de corregir):",
            font=("Segoe UI", 10, "bold"),
            bg=COLORS["card"],
            fg=COLORS["text_secondary"],
        ).pack(side="left", padx=2)

        tk.Label(
            text_hdr,
            textvariable=self.stats_var,
            font=("Segoe UI", 9, "bold"),
            bg=COLORS["butter_soft"],
            fg="#B7791F",
            padx=8,
            pady=2,
        ).pack(side="right", padx=2)

        self.text_widget = tk.Text(
            self.text_card.inner,
            height=2,
            wrap="word",
            font=("Segoe UI", 18, "bold"),
            bg=COLORS["card_tint"],
            fg=COLORS["text_primary"],
            bd=0,
            highlightthickness=1,
            highlightbackground=COLORS["border"],
            highlightcolor=COLORS["lavender"],
            padx=12,
            pady=6,
        )
        self.text_widget.pack(fill="x")
        self.text_widget.bind("<KeyRelease>", lambda _e: self._update_text_stats())

        root.bind("<Configure>", self._on_window_resize)

        # ================= INICIALIZACIÓN =================
        try:
            self._open_camera(self.camera_idx_var.get())
            if self.preview_only:
                self.reload_button.configure(state="disabled")
                self.start_button.configure(state="disabled")
                self.stop_button.configure(state="disabled")
                self.status.set("Vista previa (sin modelos)")
            else:
                self.reload_engine(show_message=False)
                self.status.set("Listo • Pulsa Iniciar captura")
            self.update_frame()
        except Exception as exc:
            messagebox.showerror("No se pudo iniciar", str(exc))
            self.status.set("Error al iniciar cámara o modelos")

        root.protocol("WM_DELETE_WINDOW", self.close)

    # ---------------- AJUSTES Y UTILIDADES ----------------
    def _on_settings_changed(self, _val=None):
        hold = round(float(self.hold_var.get()), 1)
        conf = round(float(self.conf_var.get()), 2)
        self.session.hold_seconds = hold
        self.session.confidence_threshold = conf
        self.hold_label_var.set(f"{hold:.1f} s")
        self.conf_label_var.set(f"{int(conf * 100)} %")
        if not self.session.active:
            self.timer_var.set(f"Listo ({hold:.1f} s)")

    def _open_camera(self, index: int):
        if self.capture:
            self.capture.release()
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara en el índice {index}.")
        self.capture = cap

    def reconnect_camera(self):
        try:
            idx = int(self.camera_idx_var.get())
            self._open_camera(idx)
            self.status.set(f"Cámara {idx} conectada")
        except Exception as exc:
            messagebox.showerror("Error de cámara", str(exc))

    def _render_history_chips(self):
        for child in self.chips_frame.winfo_children():
            child.destroy()
        if not self.recent_tokens:
            tk.Label(
                self.chips_frame,
                text="Aún no se han escrito letras",
                font=("Segoe UI", 9, "italic"),
                bg=COLORS["card"],
                fg=COLORS["text_secondary"],
            ).pack(anchor="w")
            return
        chip_colors = [COLORS["lavender_soft"], COLORS["mint_soft"], COLORS["sky_soft"], COLORS["butter_soft"]]
        for i, tok in enumerate(self.recent_tokens):
            bg_c = chip_colors[i % len(chip_colors)]
            tk.Label(
                self.chips_frame,
                text=tok,
                font=("Segoe UI", 10, "bold"),
                bg=bg_c,
                fg=COLORS["text_primary"],
                padx=8,
                pady=3,
            ).pack(side="left", padx=(0, 5))

    def _update_text_stats(self):
        content = self.text_widget.get("1.0", "end-1c")
        chars = len(content)
        words = len(content.split()) if content.strip() else 0
        self.stats_var.set(f"{chars} caracteres • {words} palabras")

    def copy_text(self):
        content = self.text_widget.get("1.0", "end-1c")
        if not content:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.status.set("Texto copiado al portapapeles")

    def _on_window_resize(self, event):
        if event.widget is not self.root:
            return
        w, h = event.width, event.height
        if w < 940 or h < 660:
            bucket = "compact"
        elif w > 1400 or h > 920:
            bucket = "large"
        else:
            bucket = "normal"

        if bucket == self._last_ui_bucket:
            return
        self._last_ui_bucket = bucket

        if bucket == "compact":
            self.main_container.pack_configure(padx=8, pady=8)
            self.main_container.columnconfigure(1, minsize=250)
            for card in (self.top_card, self.video_card, self.side_card, self.text_card):
                card.set_padding(8)
            self.title_label.configure(font=("Segoe UI", 13, "bold"))
            self.big_letter_label.configure(font=("Segoe UI", 26, "bold"))
            self.text_widget.configure(font=("Segoe UI", 15, "bold"), height=2)
            self.start_button.set_metrics(font=("Segoe UI", 10, "bold"), padx=12, pady=6)
            self.stop_button.set_metrics(font=("Segoe UI", 10, "bold"), padx=12, pady=6)
        elif bucket == "large":
            self.main_container.pack_configure(padx=20, pady=16)
            self.main_container.columnconfigure(1, minsize=330)
            for card in (self.top_card, self.video_card, self.side_card, self.text_card):
                card.set_padding(14)
            self.title_label.configure(font=("Segoe UI", 17, "bold"))
            self.big_letter_label.configure(font=("Segoe UI", 40, "bold"))
            self.text_widget.configure(font=("Segoe UI", 21, "bold"), height=2)
            self.start_button.set_metrics(font=("Segoe UI", 12, "bold"), padx=22, pady=10)
            self.stop_button.set_metrics(font=("Segoe UI", 12, "bold"), padx=22, pady=10)
        else:
            self.main_container.pack_configure(padx=14, pady=12)
            self.main_container.columnconfigure(1, minsize=290)
            for card in (self.top_card, self.video_card, self.side_card, self.text_card):
                card.set_padding(10)
            self.title_label.configure(font=("Segoe UI", 15, "bold"))
            self.big_letter_label.configure(font=("Segoe UI", 34, "bold"))
            self.text_widget.configure(font=("Segoe UI", 18, "bold"), height=2)
            self.start_button.set_metrics(font=("Segoe UI", 11, "bold"), padx=18, pady=8)
            self.stop_button.set_metrics(font=("Segoe UI", 11, "bold"), padx=18, pady=8)

    # ---------------- MOTOR Y CAPTURA ----------------
    def reload_engine(self, show_message=True):
        try:
            new_engine = RecognitionEngine(ARTIFACTS, static_mode=self.mode.get())
            old_engine = self.engine
            self.engine = new_engine
            if old_engine:
                old_engine.close()
            branch_tag = "Estático + Video J/Z" if self.engine.dynamic_available else "Solo estático"
            message = f"Rama {self.mode.get().upper()} • {branch_tag}"
            self.status.set(message)
            self.status_badge.configure(bg=COLORS["lavender_soft"], fg=COLORS["lavender_hover"])
            if show_message:
                messagebox.showinfo("Modelos cargados", message)
        except Exception as exc:
            self.status.set("Error cargando rama")
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
        self.status.set("● CAPTURANDO EN VIVO")
        self.status_badge.configure(bg=COLORS["mint_soft"], fg=COLORS["mint_hover"])

    def stop_capture(self):
        self.session.text = self.text_widget.get("1.0", "end-1c")
        text = self.session.stop()
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.status.set("Captura pausada")
        self.status_badge.configure(bg=COLORS["lavender_soft"], fg=COLORS["lavender_hover"])
        self.timer_var.set("Pausado")
        self.hold_bar.set_value(0.0)
        self._sync_text()
        if not text.strip() or not self.auto_spell_var.get():
            return
        suggestion = self.corrector.suggest(text)
        if suggestion != text:
            accept = messagebox.askyesno(
                "Sugerencia ortográfica (Inglés)",
                f"Texto reconocido:\n{text}\n\nSugerencia:\n{suggestion}\n\n¿Aplicar sugerencia?",
            )
            if accept:
                self.session.text = suggestion
                self._sync_text()
        else:
            messagebox.showinfo("Revisión ortográfica", "No se sugieren cambios ortográficos.")

    def clear_text(self):
        self.session.clear()
        self.recent_tokens.clear()
        self._render_history_chips()
        self.text_widget.delete("1.0", "end")
        self._update_text_stats()

    def _sync_text(self):
        self.text_widget.delete("1.0", "end")
        self.text_widget.insert("1.0", self.session.text)
        self._update_text_stats()

    def update_frame(self):
        if not self.capture or not self.capture.isOpened():
            return
        ok, frame = self.capture.read()
        if ok:
            now = time.monotonic()
            dt = max(1e-4, now - self._last_frame_ts)
            self._last_frame_ts = now
            inst_fps = 1.0 / dt
            self._fps_smooth = inst_fps if self._fps_smooth == 0 else (0.85 * self._fps_smooth + 0.15 * inst_fps)
            self.fps_var.set(f"{self._fps_smooth:.0f} FPS")

            if self.preview_only:
                display = cv2.flip(frame, 1) if self.mirror_var.get() else frame.copy()
                cv2.putText(
                    display,
                    "VISTA PREVIA - SIN MODELO",
                    (18, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (245, 158, 11),
                    2,
                )
            else:
                if not self.engine:
                    return
                display, label, confidence, source, detected = self.engine.predict(frame)
                if self.mirror_var.get():
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
                    self._render_history_chips()
                    self._update_text_stats()

                cand, remaining, progress = self.session.get_countdown_info()
                h, w, _ = display.shape

                # Pill superior translúcido dentro del video
                overlay = display.copy()
                cv2.rectangle(overlay, (0, 0), (w, 48), (38, 34, 54), -1)
                cv2.addWeighted(overlay, 0.58, display, 0.42, 0, display)

                shown = label.upper() if label is not None else "—"
                if not detected:
                    header_txt = "Muestra tu mano frente a la camara"
                    header_color = (205, 198, 224)
                    self.big_letter_var.set("—")
                    self.sign_meta_var.set("Mano no detectada")
                    self.conf_bar.set_value(0.0)
                else:
                    self.big_letter_var.set(shown)
                    self.sign_meta_var.set(f"Confianza: {confidence * 100:.0f}%\nRama: {source}")
                    bar_color = COLORS["mint"] if confidence >= self.session.confidence_threshold else COLORS["butter"]
                    self.conf_bar.set_value(confidence, fill_color=bar_color)

                    if self.session.active and cand:
                        header_txt = f"Manteniendo '{cand}'... {remaining:.1f} s"
                        header_color = (160, 238, 200)
                    elif self.session.active:
                        header_txt = f"Capturando • Seña actual: {shown}"
                        header_color = (195, 225, 255)
                    else:
                        header_txt = f"Vista previa • Seña: {shown}"
                        header_color = (238, 232, 250)

                cv2.putText(
                    display,
                    header_txt,
                    (16, 31),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    header_color,
                    2,
                    cv2.LINE_AA,
                )

                # Barra inferior pastel en video + barra lateral de progreso
                if self.session.active and cand:
                    bar_h = 10
                    cv2.rectangle(display, (0, h - bar_h), (w, h), (45, 42, 62), -1)
                    fill_w = int(w * progress)
                    cv2.rectangle(display, (0, h - bar_h), (fill_w, h), (159, 203, 107), -1)
                    self.timer_var.set(f"{remaining:.1f} s")
                    self.hold_bar.set_value(progress, fill_color=COLORS["mint"])
                elif self.session.active:
                    self.timer_var.set(f"Esperando ({self.session.hold_seconds:.1f} s)")
                    self.hold_bar.set_value(0.0)
                else:
                    self.timer_var.set(f"Listo ({self.session.hold_seconds:.1f} s)")
                    self.hold_bar.set_value(0.0)

            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)

            avail_w = self.video_holder.winfo_width()
            avail_h = self.video_holder.winfo_height()
            if avail_w > 20 and avail_h > 20:
                image = resize_to_fit(image, avail_w, avail_h)
            else:
                image.thumbnail((840, 460))

            image = apply_rounded_corners(image, radius=18)
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
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Abre la interfaz y cámara sin modelos para verificar el dispositivo.",
    )
    args = parser.parse_args()
    root = tk.Tk()
    LiveApp(root, preview_only=args.preview_only)
    root.mainloop()


if __name__ == "__main__":
    main()