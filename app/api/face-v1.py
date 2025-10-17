# face_recognition_system_dashboard_cards_fixed.py
# - Light gray UI, top menu only
# - Model tab is second (after Training)
# - Real-time dashboard shows detections, emotions (top 3), and card counts
# - Card rectangles & counting use your YOLO logic
# - Frame Skip controls playback pacing
# - Pillow >=10 compatible (_avatar_for_person uses ImageFont + textbbox)

import os
import cv2
import pickle
import numpy as np
import threading
from collections import defaultdict, Counter

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk, ImageDraw, ImageFont

# ML
from mtcnn import MTCNN
from keras_facenet import FaceNet
from deepface import DeepFace

import torch

# --- UI Compatibility Layer (Tkinter + ttkbootstrap) ---
TTBOOTSTRAP_AVAILABLE = False
try:
    import ttkbootstrap as tb
    from ttkbootstrap.constants import *
    TTBOOTSTRAP_AVAILABLE = True
    print("✅ ttkbootstrap successfully loaded.")
except ImportError:
    print("❌ ttkbootstrap not installed. Falling back to classic Tkinter style.")


# ---------- Helpers ----------
def color_for(label: str):
    l = (label or "").lower()
    if "red" in l:
        return (0, 0, 255)
    if "yellow" in l:
        return (0, 255, 255)
    if "green" in l:
        return (0, 255, 0)
    # also support abbreviations (for rectangle color only fallback)
    if l in {"rc", "r", "redcard", "red-card"}:
        return (0, 0, 255)
    if l in {"yc", "y", "yellowcard", "yellow-card"}:
        return (0, 255, 255)
    if l in {"gc", "g", "greencard", "green-card"}:
        return (0, 255, 0)
    return (255, 255, 255)


def classify_card_label(label: str, cls_id=None):
    """
    Normalize a model label to 'red'/'yellow'/'green'/None.
    Tries many variants and falls back to class indices (0/1/2) if helpful.
    """
    if label is None:
        label = ""
    l = str(label).strip().lower()

    # Common direct matches / variants
    if any(tok in l for tok in ["red", "redcard", "red-card", "red_card"]):
        return "red"
    if any(tok in l for tok in ["yellow", "yellowcard", "yellow-card", "yellow_card"]):
        return "yellow"
    if any(tok in l for tok in ["green", "greencard", "green-card", "green_card"]):
        return "green"

    # Abbreviations some models use
    if l in {"rc", "r"}:
        return "red"
    if l in {"yc", "y"}:
        return "yellow"
    if l in {"gc", "g"}:
        return "green"

    # Sometimes names are like 'class_0','0','1','2' or ints
    if cls_id is not None:
        try:
            cid = int(cls_id)
            if cid in (0,):
                # adjust here if your dataset maps differently
                return "red"
            if cid in (1,):
                return "yellow"
            if cid in (2,):
                return "green"
        except Exception:
            pass

    # Try last-chunk heuristics: "card_red" -> red
    parts = [p for p in l.replace("-", "_").split("_") if p]
    if parts:
        if parts[-1] in {"red", "yellow", "green"}:
            return parts[-1]

    return None


def to_xyxy_from_xywh(x, y, w, h):
    return (x, y, x + w, y + h)


def rect_center_xyxy(x1, y1, x2, y2):
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def center_dist(c1, c2):
    return ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2) ** 0.5


class FaceRecognitionSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System Pro")
        self.root.geometry("1200x820")
        self.root.minsize(1000, 720)

        # ---- Light theme palette (as requested) ----
        BASE_BG = "#c5ccd6"   # slightly darker light gray
        self.root.configure(bg=BASE_BG)

        # Models and data
        self.detector = MTCNN()
        self.embedder = FaceNet()
        self.threshold = 0.9
        self.known_faces = {}
        self.face_samples = {}
        self.embeddings_path = "known_embeddings.pkl"
        self.samples_path = "face_samples.pkl"
        self.video_running = False
        self.status_var = tk.StringVar()

        # YOLO (card model)
        self.card_model = None
        self.card_names = {}
        self.card_imgsz = 960
        self.yolo_model_path = r"C:\yassine\20251010\app\best.pt"

        # Dashboard state
        self.dashboard_img_cache = []
        self.best_face_preview = {}   # (video, person) -> PIL.Image
        self.dashboard_rows = {}      # (video, person) -> tree iid

        self.load_model_data()
        self.setup_ui_style(BASE_BG)
        self.create_main_interface()

        threading.Thread(target=self.load_yolo_model, args=(self.yolo_model_path,), daemon=True).start()

    # ---------------- UI style ----------------
    def setup_ui_style(self, BASE_BG):
        global TTBOOTSTRAP_AVAILABLE
        if TTBOOTSTRAP_AVAILABLE:
            try:
                self.tb_style = tb.Style(theme="flatly")
                return
            except Exception:
                TTBOOTSTRAP_AVAILABLE = False

        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        accent = "#0d6efd"
        bg = BASE_BG

        self.style.configure('TFrame', background=bg)
        self.style.configure('TLabel', background=bg, font=('Segoe UI', 11))
        self.style.configure('Header.TLabel', font=('Segoe UI', 13, 'bold'), background=bg)
        self.style.configure('TButton', padding=8, relief="flat", background=accent,
                             foreground="white", font=('Segoe UI', 11, 'bold'))
        self.style.map('TButton', background=[('active', '#0b5ed7')], foreground=[('active', 'white')])
        self.style.configure('TEntry', fieldbackground='white', bordercolor='#d0d0d0')
        self.style.configure('Horizontal.TProgressbar', thickness=14)
        self.style.configure("Treeview", font=('Segoe UI', 11), rowheight=56,
                             background="white", fieldbackground="white", foreground="#1f2937")
        self.style.configure("Treeview.Heading", font=('Segoe UI', 11, 'bold'))

    # ---------------- Layout ----------------
    def create_main_interface(self):
        # Top bar only
        top_frame = ttk.Frame(self.root, padding=(10, 8))
        top_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top_frame, text="Face Recognition System Pro", style='Header.TLabel').pack(side=tk.LEFT)

        actions = ttk.Frame(top_frame)
        actions.pack(side=tk.RIGHT)
        ttk.Button(actions, text="⚙️ Settings", command=self._dummy_action).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(actions, text="ℹ️ About", command=self.show_about).pack(side=tk.RIGHT, padx=(6, 8))

        # Tabs in your requested order: Training, Model, Image, Video, Dashboard
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=(6, 12))

        self.create_training_tab()
        self.create_model_viewer_tab()
        self.create_image_recognition_tab()
        self.create_video_recognition_tab()
        self.create_dashboard_tab()

        # Status bar
        status_frame = ttk.Frame(self.root, padding=(4, 2))
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))
        self.small_progress = ttk.Progressbar(status_frame, style='Horizontal.TProgressbar',
                                              mode='determinate', length=220)
        self.small_progress.pack(side=tk.RIGHT, padx=8)
        self.update_status("Ready")

    def _dummy_action(self):
        messagebox.showinfo("Info", "This is a UI placeholder.")

    def show_about(self):
        messagebox.showinfo("About", "FaceNet + MTCNN + DeepFace + YOLO Card Detection\nReal-time Video Dashboard.")

    # ---------------- Model data I/O ----------------
    def load_model_data(self):
        if os.path.exists(self.embeddings_path):
            try:
                with open(self.embeddings_path, "rb") as f:
                    self.known_faces = pickle.load(f)
            except Exception:
                self.known_faces = {}
        if os.path.exists(self.samples_path):
            try:
                with open(self.samples_path, "rb") as f:
                    self.face_samples = pickle.load(f)
            except Exception:
                self.face_samples = {}

    def save_model_data(self):
        with open(self.embeddings_path, 'wb') as f:
            pickle.dump(self.known_faces, f)
        with open(self.samples_path, 'wb') as f:
            pickle.dump(self.face_samples, f)

    # ---------------- Status helpers ----------------
    def update_status(self, message, progress=None):
        self.status_var.set(message)
        try:
            if progress is not None:
                self.small_progress['value'] = progress
            self.root.update_idletasks()
        except Exception:
            pass

    def log_message(self, message):
        if hasattr(self, 'training_log'):
            self.training_log.config(state=tk.NORMAL)
            self.training_log.insert(tk.END, message + "\n")
            self.training_log.config(state=tk.DISABLED)
            self.training_log.see(tk.END)
        self.update_status(message)

    # ---------------- YOLO load ----------------
    def load_yolo_model(self, model_path=None):
        if model_path is None:
            model_path = self.yolo_model_path
        try:
            from ultralytics import YOLO
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.card_model = YOLO(model_path)
            self.card_model.to(device)
            try:
                dummy = np.zeros((320, 320, 3), dtype=np.uint8)
                _ = self.card_model.predict(dummy, imgsz=320, conf=0.25, verbose=False)
            except Exception:
                pass
            names = self.card_model.names
            if isinstance(names, list):
                names = {i: n for i, n in enumerate(names)}
            self.card_names = names
            self.update_status(f"YOLO loaded on {device}.")
        except Exception as e:
            self.card_model = None
            self.card_names = {}
            self.update_status(f"Failed to load YOLO: {str(e)}")

    # ---------------- Tabs ----------------
    def create_training_tab(self):
        self.training_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.training_tab, text="🧠  Model Training")

        ttk.Label(self.training_tab, text="Train Face Recognition Model", style='Header.TLabel').pack(pady=(8, 12), anchor='w')

        folder_frame = ttk.Frame(self.training_tab)
        folder_frame.pack(fill=tk.X, padx=4, pady=4)
        ttk.Label(folder_frame, text="Training Images Folder:").pack(side=tk.LEFT)
        self.training_folder_entry = ttk.Entry(folder_frame, width=58)
        self.training_folder_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=8)
        ttk.Button(folder_frame, text="📁 Browse", command=self.select_training_folder).pack(side=tk.LEFT)

        options_frame = ttk.Frame(self.training_tab)
        options_frame.pack(fill=tk.X, padx=4, pady=8)
        self.append_var = tk.IntVar(value=0)
        ttk.Checkbutton(options_frame, text="Append to existing model", variable=self.append_var).pack(side=tk.LEFT)

        ttk.Button(self.training_tab, text="🚀 Train Model", command=self.train_model).pack(pady=10, anchor='w', padx=4)

        self.training_progress = ttk.Progressbar(self.training_tab, style='Horizontal.TProgressbar', mode='determinate')
        self.training_progress.pack(fill=tk.X, padx=4, pady=(6, 12))

        log_frame = ttk.Frame(self.training_tab)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.training_log = tk.Text(log_frame, wrap=tk.WORD, state=tk.DISABLED, bg='white', font=('Segoe UI', 11))
        scrollbar = ttk.Scrollbar(log_frame, command=self.training_log.yview)
        self.training_log.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.training_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def create_model_viewer_tab(self):
        self.model_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.model_tab, text="📁  Model Viewer")

        ttk.Label(self.model_tab, text="View Trained Face Models", style='Header.TLabel').pack(pady=(8, 12), anchor='w')
        ttk.Button(self.model_tab, text="🔽 Load YOLO Model",
                   command=lambda: threading.Thread(target=self._load_yolo_thread, daemon=True).start()).pack(pady=6, anchor='w', padx=4)

        self.model_canvas = tk.Canvas(self.model_tab, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.model_tab, orient="vertical", command=self.model_canvas.yview)
        self.model_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.model_canvas.pack(side="left", fill="both", expand=True)

        self.model_inner_frame = ttk.Frame(self.model_canvas)
        self.model_canvas.create_window((0, 0), window=self.model_inner_frame, anchor="nw")
        self.model_inner_frame.bind("<Configure>", lambda e: self.model_canvas.configure(scrollregion=self.model_canvas.bbox("all")))

        self.display_model_faces()

    def create_image_recognition_tab(self):
        self.image_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.image_tab, text="📷  Image Recognition")

        ttk.Label(self.image_tab, text="Recognize Faces in Images", style='Header.TLabel').pack(pady=(8, 12), anchor='w')

        folder_frame = ttk.Frame(self.image_tab)
        folder_frame.pack(fill=tk.X, padx=4, pady=4)
        ttk.Label(folder_frame, text="Test Images Folder:").pack(side=tk.LEFT)
        self.image_folder_entry = ttk.Entry(folder_frame, width=58)
        self.image_folder_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=8)
        ttk.Button(folder_frame, text="📁 Browse", command=self.select_image_folder).pack(side=tk.LEFT)

        threshold_frame = ttk.Frame(self.image_tab)
        threshold_frame.pack(fill=tk.X, padx=4, pady=8)
        ttk.Label(threshold_frame, text="Recognition Threshold:").pack(side=tk.LEFT)
        self.threshold_slider = ttk.Scale(threshold_frame, from_=0.5, to=1.5, value=0.9, orient=tk.HORIZONTAL)
        self.threshold_slider.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=8)
        self.threshold_value = ttk.Label(threshold_frame, text="0.90")
        self.threshold_value.pack(side=tk.LEFT, padx=(6, 0))
        self.threshold_slider.config(command=lambda v: self.threshold_value.config(text=f"{float(v):.2f}"))

        # Card toggles (image)
        card_row = ttk.Frame(self.image_tab)
        card_row.pack(fill=tk.X, padx=4, pady=4)
        self.card_detect_image_var = tk.IntVar(value=0)
        ttk.Checkbutton(card_row, text="Enable Card Detection (YOLOv8)", variable=self.card_detect_image_var).pack(side=tk.LEFT)
        ttk.Label(card_row, text="Card Confidence:").pack(side=tk.LEFT, padx=(12, 4))
        self.card_conf_slider_img = ttk.Scale(card_row, from_=0.10, to=0.90, value=0.50, orient=tk.HORIZONTAL)
        self.card_conf_slider_img.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.card_conf_value_img = ttk.Label(card_row, text="0.50")
        self.card_conf_value_img.pack(side=tk.LEFT)
        self.card_conf_slider_img.config(command=lambda v: self.card_conf_value_img.config(text=f"{float(v):.2f}"))

        ttk.Button(self.image_tab, text="🔍 Process Images", command=self.process_images).pack(pady=8, anchor='w', padx=4)

        self.image_results_notebook = ttk.Notebook(self.image_tab)
        self.image_results_notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=8)
        self.recognized_tab = ttk.Frame(self.image_results_notebook)
        self.image_results_notebook.add(self.recognized_tab, text="✅ Recognized Faces")
        self.unrecognized_tab = ttk.Frame(self.image_results_notebook)
        self.image_results_notebook.add(self.unrecognized_tab, text="❓ Unrecognized Faces")

        self.recognized_canvas = self.setup_results_canvas(self.recognized_tab)
        self.unrecognized_canvas = self.setup_results_canvas(self.unrecognized_tab)

    def create_video_recognition_tab(self):
        self.video_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.video_tab, text="🎥  Video Recognition")

        ttk.Label(self.video_tab, text="Recognize Faces in Video", style='Header.TLabel').pack(pady=(8, 12), anchor='w')

        file_frame = ttk.Frame(self.video_tab)
        file_frame.pack(fill=tk.X, padx=4, pady=4)
        ttk.Label(file_frame, text="Video File:").pack(side=tk.LEFT)
        self.video_entry = ttk.Entry(file_frame, width=50)
        self.video_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=8)
        ttk.Button(file_frame, text="📁 Browse", command=self.select_video_file).pack(side=tk.LEFT)

        settings_frame = ttk.Frame(self.video_tab)
        settings_frame.pack(fill=tk.X, padx=4, pady=8)

        threshold_frame = ttk.Frame(settings_frame)
        threshold_frame.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)
        ttk.Label(threshold_frame, text="Recognition Threshold:").pack(side=tk.LEFT)
        self.video_threshold_slider = ttk.Scale(threshold_frame, from_=0.5, to=1.5, value=0.9, orient=tk.HORIZONTAL)
        self.video_threshold_slider.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=6)
        self.video_threshold_value = ttk.Label(threshold_frame, text="0.90")
        self.video_threshold_value.pack(side=tk.LEFT)
        self.video_threshold_slider.config(command=lambda v: self.video_threshold_value.config(text=f"{float(v):.2f}"))

        skip_frame = ttk.Frame(settings_frame)
        skip_frame.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)
        ttk.Label(skip_frame, text="Frame Skip:").pack(side=tk.LEFT)
        self.frame_skip_slider = ttk.Scale(skip_frame, from_=1, to=60, value=5, orient=tk.HORIZONTAL)
        self.frame_skip_slider.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=6)
        self.frame_skip_value = ttk.Label(skip_frame, text="5")
        self.frame_skip_value.pack(side=tk.LEFT)
        self.frame_skip_slider.config(command=lambda v: self.frame_skip_value.config(text=str(int(float(v)))))

        # Card controls for video
        card_row_v = ttk.Frame(self.video_tab)
        card_row_v.pack(fill=tk.X, padx=4, pady=(0, 8))
        self.card_detect_video_var = tk.IntVar(value=1)
        ttk.Checkbutton(card_row_v, text="Enable Card Detection (YOLOv8)", variable=self.card_detect_video_var).pack(side=tk.LEFT)
        ttk.Label(card_row_v, text="Card Confidence:").pack(side=tk.LEFT, padx=(12, 4))
        self.card_conf_slider_vid = ttk.Scale(card_row_v, from_=0.05, to=0.90, value=0.50, orient=tk.HORIZONTAL)
        self.card_conf_slider_vid.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.card_conf_value_vid = ttk.Label(card_row_v, text="0.50")
        self.card_conf_value_vid.pack(side=tk.LEFT)
        self.card_conf_slider_vid.config(command=lambda v: self.card_conf_value_vid.config(text=f"{float(v):.2f}"))

        # Controls
        control_frame = ttk.Frame(self.video_tab)
        control_frame.pack(fill=tk.X, padx=4, pady=8)
        self.process_video_btn = ttk.Button(control_frame, text="▶️ Process Video", command=self.process_video)
        self.process_video_btn.pack(side=tk.LEFT)
        self.stop_video_btn = ttk.Button(control_frame, text="⏹ Stop", command=self.stop_video_processing, state=tk.DISABLED)
        self.stop_video_btn.pack(side=tk.LEFT, padx=8)

        # Preview (light gray background)
        video_frame = ttk.Frame(self.video_tab)
        video_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=8)
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        try:
            self.video_label.configure(background="#d0d6df")
        except Exception:
            pass

        # Results below
        self.video_results_notebook = ttk.Notebook(self.video_tab)
        self.video_results_notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=8)
        self.video_recognized_tab = ttk.Frame(self.video_results_notebook)
        self.video_results_notebook.add(self.video_recognized_tab, text="✅ Recognized Faces")
        self.video_unrecognized_tab = ttk.Frame(self.video_results_notebook)
        self.video_results_notebook.add(self.video_unrecognized_tab, text="❓ Unrecognized Faces")

        self.video_recognized_canvas = self.setup_results_canvas(self.video_recognized_tab)
        self.video_unrecognized_canvas = self.setup_results_canvas(self.video_unrecognized_tab)

    def create_dashboard_tab(self):
        self.dashboard_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.dashboard_tab, text="🎞  Video Dashboard")

        header_row = ttk.Frame(self.dashboard_tab)
        header_row.pack(fill=tk.X)
        ttk.Label(header_row, text="Per-Video Player Summary (Detections • Emotions • Cards)", style='Header.TLabel').pack(side=tk.LEFT, pady=(4, 8))
        ttk.Button(header_row, text="🧹 Clear Dashboard", command=self._clear_dashboard).pack(side=tk.RIGHT, padx=4, pady=(4, 8))

        table_wrap = ttk.Frame(self.dashboard_tab)
        table_wrap.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        columns = ("video", "detections", "best_score", "emotions", "red", "yellow", "green")
        self.dashboard_tree = ttk.Treeview(table_wrap, columns=columns, show="tree headings")
        self.dashboard_tree.heading("#0", text="Player")
        self.dashboard_tree.column("#0", width=260, minwidth=220, anchor="w")

        self.dashboard_tree.heading("video", text="Video")
        self.dashboard_tree.column("video", width=300, minwidth=200, anchor="w")

        self.dashboard_tree.heading("detections", text="Detections")
        self.dashboard_tree.column("detections", width=110, anchor="center")

        self.dashboard_tree.heading("best_score", text="Best Score")
        self.dashboard_tree.column("best_score", width=110, anchor="center")

        self.dashboard_tree.heading("emotions", text="Emotions (Top 3)")
        self.dashboard_tree.column("emotions", width=270, anchor="w")

        self.dashboard_tree.heading("red", text="Red")
        self.dashboard_tree.column("red", width=60, anchor="center")

        self.dashboard_tree.heading("yellow", text="Yellow")
        self.dashboard_tree.column("yellow", width=70, anchor="center")

        self.dashboard_tree.heading("green", text="Green")
        self.dashboard_tree.column("green", width=70, anchor="center")

        style = ttk.Style(self.root)
        style.configure("Treeview", rowheight=56)

        self.dashboard_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        yscroll = ttk.Scrollbar(table_wrap, orient="vertical", command=self.dashboard_tree.yview)
        self.dashboard_tree.configure(yscrollcommand=yscroll.set)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)

    # ---------------- shared canvas helper ----------------
    def setup_results_canvas(self, parent):
        canvas = tk.Canvas(parent, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner_frame, anchor="nw")
        inner_frame._canvas = canvas
        inner_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        return inner_frame

    # ---------------- Selectors & training ----------------
    def select_training_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.training_folder_entry.delete(0, tk.END)
            self.training_folder_entry.insert(0, folder)

    def select_image_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.image_folder_entry.delete(0, tk.END)
            self.image_folder_entry.insert(0, folder)

    def select_video_file(self):
        file = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if file:
            self.video_entry.delete(0, tk.END)
            self.video_entry.insert(0, file)

    def train_model(self):
        folder = self.training_folder_entry.get()
        if not folder:
            messagebox.showerror("Error", "Please select a training folder")
            return
        append = self.append_var.get() == 1
        threading.Thread(target=self._train_model_thread, args=(folder, append), daemon=True).start()

    def _train_model_thread(self, folder, append):
        try:
            self.log_message("Starting model training...")
            self.training_progress["value"] = 0
            embeddings_dict = {} if not append else self.known_faces.copy()
            samples_dict = {} if not append else self.face_samples.copy()
            person_folders = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
            if not person_folders:
                self.log_message("Error: No sub-folders found in the training folder")
                self.log_message("Each person should have their own sub-folder with images")
                return
            total_persons = len(person_folders)
            processed_persons = 0
            for person in person_folders:
                person_folder = os.path.join(folder, person)
                self.log_message(f"\nProcessing: {person}")
                person_embeddings = []
                sample_images = []
                image_count = 0
                for img_file in os.listdir(person_folder):
                    img_path = os.path.join(person_folder, img_file)
                    if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                    try:
                        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                        faces = self.detector.detect_faces(img)
                        if faces:
                            x, y, w, h = faces[0]['box']
                            x, y = max(0, x), max(0, y)
                            face = img[y:y+h, x:x+w]
                            face = cv2.resize(face, (160, 160))
                            embedding = self.embedder.embeddings([face])[0]
                            person_embeddings.append(embedding)
                            if len(sample_images) < 3:
                                sample_img = Image.fromarray(img)
                                sample_img.thumbnail((200, 200))
                                sample_images.append(sample_img)
                            image_count += 1
                            self.log_message(f"  Processed: {img_file}")
                        else:
                            self.log_message(f"  No face found in: {img_file}")
                    except Exception as e:
                        self.log_message(f"  Error processing {img_file}: {str(e)}")
                if person_embeddings:
                    avg_embedding = np.mean(person_embeddings, axis=0)
                    embeddings_dict[person] = avg_embedding
                    samples_dict[person] = sample_images
                    self.log_message(f"Successfully processed {image_count} images for {person}")
                else:
                    self.log_message(f"Warning: No valid faces found for {person}")
                processed_persons += 1
                progress = (processed_persons / total_persons) * 100
                self.training_progress["value"] = progress
                self.update_status(f"Training... ({processed_persons}/{total_persons})", progress=progress)
                self.root.update()
            self.known_faces = embeddings_dict
            self.face_samples = samples_dict
            self.save_model_data()
            self.log_message("\nTraining completed successfully!")
            self.log_message(f"Model saved to: {self.embeddings_path}")
            self.display_model_faces()
        except Exception as e:
            self.log_message(f"\nError during training: {str(e)}")

    # ---------------- Image processing ----------------
    def process_images(self):
        folder = self.image_folder_entry.get()
        if not folder:
            messagebox.showerror("Error", "Please select an images folder")
            return
        self.threshold = float(self.threshold_slider.get())
        for widget in self.recognized_canvas.winfo_children():
            widget.destroy()
        for widget in self.unrecognized_canvas.winfo_children():
            widget.destroy()
        threading.Thread(target=self._process_images_thread, args=(folder,), daemon=True).start()

    def _process_images_thread(self, folder):
        try:
            recognized = []
            unrecognized = []
            conf = float(self.card_conf_slider_img.get())
            for img_file in os.listdir(folder):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                img_path = os.path.join(folder, img_file)
                frame = cv2.imread(img_path)
                if frame is None:
                    continue
                rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # ---- Cards ----
                if self.card_detect_image_var.get() == 1 and self.card_model is not None:
                    results = self.card_model.predict(frame, imgsz=self.card_imgsz, conf=conf, verbose=False)
                    names = self.card_names if self.card_names else self.card_model.names
                    if isinstance(names, list):
                        names = {i: n for i, n in enumerate(names)}
                    for r in results:
                        boxes = r.boxes.xyxy.cpu().numpy()
                        classes = r.boxes.cls.cpu().numpy()
                        confs = r.boxes.conf.cpu().numpy()
                        for (x1, y1, x2, y2, cls, cscore) in zip(
                            boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], classes, confs
                        ):
                            cls = int(cls)
                            raw = names.get(cls, f"{cls}")
                            card_kind = classify_card_label(raw, cls)
                            label = (raw if raw is not None else f"Card {cls}") + f" ({cscore:.2f})"
                            color = color_for(raw)
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            cv2.putText(frame, label, (int(x1), max(0, int(y1) - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # ---- Faces + emotion ----
                faces = self.detector.detect_faces(rgb_img)
                for face in faces:
                    x, y, w, h = face['box']
                    x = max(0, x); y = max(0, y)
                    x2 = min(x + max(0, w), rgb_img.shape[1]-1)
                    y2 = min(y + max(0, h), rgb_img.shape[0]-1)
                    face_crop = rgb_img[y:y2, x:x2]
                    try:
                        face_resized = cv2.resize(face_crop, (160, 160))
                        embedding = self.embedder.embeddings([face_resized])[0]
                        name = "Unknown"; min_dist = float("inf")
                        for known_name, known_emb in self.known_faces.items():
                            dist = np.linalg.norm(embedding - known_emb)
                            if dist < min_dist and dist < self.threshold:
                                min_dist = dist; name = known_name
                        try:
                            analysis = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
                            emotion = analysis[0]['dominant_emotion'] if isinstance(analysis, list) and analysis else analysis.get('dominant_emotion', 'N/A')
                        except Exception:
                            emotion = "N/A"
                        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{name} | {emotion}", (x, max(0, y - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                        img_display = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        img_display.thumbnail((500, 500))
                        result = {'image': img_display, 'name': name, 'score': min_dist,
                                  'emotion': emotion, 'filename': img_file}
                        (recognized if name != "Unknown" else unrecognized).append(result)
                    except Exception as e:
                        print(f"Error processing face in {img_file}: {e}")

            self.root.after(0, lambda: self.display_image_results(recognized, unrecognized))
        except Exception as e:
            self.root.after(0, lambda: self.update_status(f"Error: {str(e)}"))

    def display_image_results(self, recognized, unrecognized):
        self.display_results_grid(self.recognized_canvas, recognized, "Recognized Faces")
        self.display_results_grid(self.unrecognized_canvas, unrecognized, "Unrecognized Faces")
        self.update_status(f"Images done. Recognized: {len(recognized)}, Unrecognized: {len(unrecognized)}")

    def display_results_grid(self, canvas, results, title):
        for widget in canvas.winfo_children():
            widget.destroy()
        if not results:
            ttk.Label(canvas, text=f"No {title.lower()} found").pack(pady=20)
        else:
            row, col = 0, 0
            max_cols = 3
            for result in results:
                frame = ttk.Frame(canvas, borderwidth=1, relief="solid", padding=8)
                frame.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")
                photo = ImageTk.PhotoImage(result['image'])
                label = ttk.Label(frame, image=photo)
                label.image = photo
                label.pack()
                info = f"File: {result.get('filename', '')}\n"
                if result.get('name', "Unknown") != "Unknown":
                    info += f"Identity: {result.get('name')}\nScore: {result.get('score', 0):.2f}\n"
                info += f"Emotion: {result.get('emotion', 'N/A')}"
                ttk.Label(frame, text=info, justify=tk.LEFT).pack(pady=(6, 0))
                col += 1
                if col >= max_cols:
                    col = 0
                    row += 1
        if hasattr(canvas, "_canvas"):
            canvas._canvas.update_idletasks()
            canvas._canvas.configure(scrollregion=canvas._canvas.bbox("all"))

    # ---------------- Video processing + dashboard ----------------
    def process_video(self):
        video_path = self.video_entry.get()
        if not video_path:
            messagebox.showerror("Error", "Please select a video file")
            return
        self.threshold = float(self.video_threshold_slider.get())
        self.frame_skip = int(self.frame_skip_slider.get())
        for widget in self.video_recognized_canvas.winfo_children():
            widget.destroy()
        for widget in self.video_unrecognized_canvas.winfo_children():
            widget.destroy()
        self.process_video_btn.config(state=tk.DISABLED)
        self.stop_video_btn.config(state=tk.NORMAL)
        self.video_running = True
        threading.Thread(target=self._process_video_thread, args=(video_path,), daemon=True).start()

    def stop_video_processing(self):
        self.video_running = False
        self.process_video_btn.config(state=tk.NORMAL)
        self.stop_video_btn.config(state=tk.DISABLED)

    def _process_video_thread(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.root.after(0, lambda: self.update_status("Error: cannot open video"))
                self.root.after(0, self.stop_video_processing)
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0 or (isinstance(fps, float) and np.isnan(fps)):
                fps = 25.0
            delay = max(0.0, self.frame_skip / float(fps))

            frame_count = 0
            recognized_faces = defaultdict(list)
            unrecognized_faces = []

            # per-person stats (for current video)
            video_stats = defaultdict(lambda: {
                "detections": 0, "best_score": None, "emotions": Counter(),
                "red": 0, "yellow": 0, "green": 0
            })
            video_base = os.path.basename(video_path)
            conf = float(self.card_conf_slider_vid.get())

            while cap.isOpened() and self.video_running:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % self.frame_skip != 0:
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # ---- Card detection ----
                card_dets = []
                reds = yellows = greens = 0
                if self.card_detect_video_var.get() == 1 and self.card_model is not None:
                    results = self.card_model.predict(frame, imgsz=self.card_imgsz, conf=conf, verbose=False)
                    names = self.card_names if self.card_names else self.card_model.names
                    if isinstance(names, list):
                        names = {i: n for i, n in enumerate(names)}
                    for r in results:
                        boxes = r.boxes.xyxy.cpu().numpy()
                        classes = r.boxes.cls.cpu().numpy()
                        confs = r.boxes.conf.cpu().numpy()
                        for (x1, y1, x2, y2, cls, cscore) in zip(
                            boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], classes, confs
                        ):
                            cls = int(cls)
                            raw = names.get(cls, f"{cls}")
                            kind = classify_card_label(raw, cls)
                            card_dets.append((int(x1), int(y1), int(x2), int(y2), raw, float(cscore), kind))
                            color = color_for(raw)
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            cv2.putText(frame, f"{raw} ({cscore:.2f})", (int(x1), max(0, int(y1) - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            if kind == "red":    reds += 1
                            if kind == "yellow": yellows += 1
                            if kind == "green":  greens += 1

                # ---- Face recognition + emotion ----
                faces = self.detector.detect_faces(rgb_frame)
                recognized_now = []
                updated_people = set()

                for face in faces:
                    x, y, w, h = face['box']
                    x = max(0, x); y = max(0, y)
                    x2 = min(x + max(0, w), rgb_frame.shape[1]-1)
                    y2 = min(y + max(0, h), rgb_frame.shape[0]-1)
                    face_img = rgb_frame[y:y2, x:x2]
                    try:
                        face_resized = cv2.resize(face_img, (160, 160))
                        embedding = self.embedder.embeddings([face_resized])[0]
                        name = "Unknown"; min_dist = float("inf")
                        for known_name, known_emb in self.known_faces.items():
                            dist = np.linalg.norm(embedding - known_emb)
                            if dist < min_dist and dist < self.threshold:
                                min_dist = dist; name = known_name
                        try:
                            analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
                            emotion = analysis[0]['dominant_emotion'] if isinstance(analysis, list) and analysis else analysis.get('dominant_emotion', 'N/A')
                        except Exception:
                            emotion = "N/A"

                        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{name} | {emotion}", (x, max(0, y - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                        frame_display = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        frame_display.thumbnail((1000, 1000))
                        result = {'frame': frame_count, 'name': name, 'score': min_dist,
                                  'emotion': emotion, 'image': frame_display.copy()}
                        if name == "Unknown":
                            unrecognized_faces.append(result)
                        else:
                            recognized_faces[name].append(result)
                            vs = video_stats[name]
                            vs["detections"] += 1
                            if np.isfinite(min_dist):
                                if vs["best_score"] is None or min_dist < vs["best_score"]:
                                    vs["best_score"] = min_dist
                            if emotion and emotion != "N/A":
                                vs["emotions"][emotion] += 1
                            updated_people.add(name)
                            prev = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            prev.thumbnail((128, 128))
                            self.best_face_preview[(video_base, name)] = prev

                        recognized_now.append((name, x, y, x2 - x, y2 - y))
                    except Exception as e:
                        print(f"Face error @frame {frame_count}: {e}")

                # ---- Associate each detected card to nearest recognized face (else Unknown) ----
                for (x1, y1, x2, y2, raw_name, _score, kind) in card_dets:
                    if kind not in {"red", "yellow", "green"}:
                        continue
                    card_center = rect_center_xyxy(x1, y1, x2, y2)
                    best_name = None
                    best_d = None
                    for (nm, fx, fy, fw, fh) in recognized_now:
                        cx1, cy1, cx2, cy2 = to_xyxy_from_xywh(fx, fy, fw, fh)
                        f_center = rect_center_xyxy(cx1, cy1, cx2, cy2)
                        d = center_dist(card_center, f_center)
                        if best_d is None or d < best_d:
                            best_d = d
                            best_name = nm
                    key_name = best_name if best_name else "Unknown"
                    vs = video_stats[key_name]
                    vs[kind] += 1
                    updated_people.add(key_name)

                # If there were cards but no faces, still reflect in dashboard under Unknown
                if card_dets and not recognized_now:
                    updated_people.add("Unknown")

                # ---- Update preview and status ----
                frame_display2 = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_display2.thumbnail((1100, 1100))
                self.root.after(0, lambda f=frame_display2, n=len(card_dets), r=reds, y=yellows, g=greens:
                                self.update_video_display_with_count(f, n, r, y, g))

                # Periodic side grids
                if frame_count % max(1, (self.frame_skip * 2)) == 0:
                    self.root.after(0, lambda: self.update_video_results(recognized_faces, unrecognized_faces))

                # LIVE dashboard update
                if updated_people:
                    names_to_update = list(updated_people)
                    self.root.after(0, lambda nb=video_base, vs=video_stats, ppl=names_to_update:
                                    self._update_dashboard_live(nb, vs, ppl))

                # Playback pacing (normal feel; adjustable by Frame Skip)
                if delay > 0:
                    threading.Event().wait(delay)

            # final updates
            self.root.after(0, lambda: self.update_video_results(recognized_faces, unrecognized_faces))
            self.root.after(0, lambda nb=video_base, vs=video_stats: self._update_dashboard_live(nb, vs, list(vs.keys())))
            self.root.after(0, self.stop_video_processing)
            cap.release()
        except Exception as e:
            self.root.after(0, lambda: self.update_status(f"Error: {str(e)}"))
            self.root.after(0, self.stop_video_processing)

    def update_video_display_with_count(self, frame, card_count, r, y, g):
        photo = ImageTk.PhotoImage(frame)
        self.video_label.config(image=photo)
        self.video_label.image = photo
        self.update_status(f"Cards this frame → total:{card_count}  R:{r}  Y:{y}  G:{g}")

    def update_video_results(self, recognized, unrecognized):
        self.display_video_results_grid(self.video_recognized_canvas, recognized, "Recognized Faces")
        self.display_video_results_grid(self.video_unrecognized_canvas, unrecognized, "Unrecognized Faces")

    def display_video_results_grid(self, canvas, results, title):
        for widget in canvas.winfo_children():
            widget.destroy()
        if not results:
            ttk.Label(canvas, text=f"No {title.lower()} found").pack(pady=20)
        else:
            row, col = 0, 0
            max_cols = 3
            if isinstance(results, dict):
                for name, faces in results.items():
                    if not faces:
                        continue
                    best_face = min(faces, key=lambda x: x['score'])
                    frame = ttk.Frame(canvas, borderwidth=1, relief="solid", padding=8)
                    frame.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")
                    photo = ImageTk.PhotoImage(best_face['image'])
                    label = ttk.Label(frame, image=photo)
                    label.image = photo
                    label.pack()
                    info = f"Name: {name}\nDetections: {len(faces)}\nBest Score: {best_face['score']:.2f}\nEmotion: {best_face['emotion']}"
                    ttk.Label(frame, text=info, justify=tk.LEFT).pack(pady=(6, 0))
                    col += 1
                    if col >= max_cols:
                        col = 0
                        row += 1
            else:
                for face in results[:20]:
                    frame = ttk.Frame(canvas, borderwidth=1, relief="solid", padding=8)
                    frame.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")
                    photo = ImageTk.PhotoImage(face['image'])
                    label = ttk.Label(frame, image=photo)
                    label.image = photo
                    label.pack()
                    info = f"Frame: {face['frame']}\nEmotion: {face['emotion']}"
                    ttk.Label(frame, text=info, justify=tk.LEFT).pack(pady=(6, 0))
                    col += 1
                    if col >= max_cols:
                        col = 0
                        row += 1
        if hasattr(canvas, "_canvas"):
            canvas._canvas.update_idletasks()
            canvas._canvas.configure(scrollregion=canvas._canvas.bbox("all"))

    # ---------------- Model viewer ----------------
    def _load_yolo_thread(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch model", "*.pt"), ("All files","*.*")])
        if not path:
            return
        self.load_yolo_model(path)

    def display_model_faces(self):
        for widget in self.model_inner_frame.winfo_children():
            widget.destroy()
        if not self.known_faces:
            ttk.Label(self.model_inner_frame, text="No faces in the model yet. Train a model first.", style='Header.TLabel').pack(pady=20)
            return
        row, col = 0, 0
        max_cols = 3
        for name in sorted(self.known_faces.keys()):
            frame = ttk.Frame(self.model_inner_frame, borderwidth=1, relief="solid", padding=10)
            frame.grid(row=row, column=col, padx=12, pady=12, sticky="nsew")
            ttk.Label(frame, text=name, style='Header.TLabel').pack()
            if name in self.face_samples and self.face_samples[name]:
                sample_frame = ttk.Frame(frame)
                sample_frame.pack(pady=(6,6))
                for sample_img in self.face_samples[name]:
                    photo = ImageTk.PhotoImage(sample_img)
                    label = ttk.Label(sample_frame, image=photo)
                    label.image = photo
                    label.pack(side=tk.LEFT, padx=6)
            else:
                ttk.Label(frame, text="No sample images available").pack(pady=(6,0))
            ttk.Label(frame, text=f"Embedding size: {len(self.known_faces[name])}", font=('Segoe UI', 9)).pack(pady=(6,0))
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
        self.model_canvas.update_idletasks()
        self.model_canvas.configure(scrollregion=self.model_canvas.bbox("all"))

    # ---------------- Dashboard helpers ----------------
    def _clear_dashboard(self):
        for iid in self.dashboard_tree.get_children():
            self.dashboard_tree.delete(iid)
        self.dashboard_img_cache.clear()
        self.best_face_preview.clear()
        self.dashboard_rows.clear()
        self.update_status("Dashboard cleared.")
        self.dashboard_tree.update_idletasks()

    def _avatar_for_person(self, video_base, person):
        if not person:
            person = "Unknown"

        img = self.best_face_preview.get((video_base, person))
        if img is None:
            if person in self.face_samples and self.face_samples[person]:
                img = self.face_samples[person][0]
            else:
                # Fallback: generate a simple circle avatar with the first letter
                size = 64
                img = Image.new("RGB", (size, size), (232, 235, 241))
                d = ImageDraw.Draw(img)
                txt = (person[:1] or "?").upper()

                # Pillow >=10: use textbbox; load a default font
                try:
                    font = ImageFont.load_default()
                    bbox = d.textbbox((0, 0), txt, font=font)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                    d.text(((size - tw) // 2, (size - th) // 2), txt, fill=(30, 41, 59), font=font)
                except Exception:
                    # very defensive fallback
                    d.text((size // 2 - 6, size // 2 - 6), txt, fill=(30, 41, 59))

        # round crop to a circle and return PhotoImage (cached)
        target = 48
        avatar = img.convert("RGB").resize((target, target), Image.LANCZOS)

        mask = Image.new("L", (target, target), 0)
        md = ImageDraw.Draw(mask)
        md.ellipse((0, 0, target - 1, target - 1), fill=255)

        out = Image.new("RGBA", (target, target), (0, 0, 0, 0))
        out.paste(avatar, (0, 0), mask)

        pimg = ImageTk.PhotoImage(out)
        # keep a reference so Tk doesn't garbage-collect it
        self.dashboard_img_cache.append(pimg)
        return pimg

    def _update_dashboard_live(self, video_base, per_person_stats, people_to_update):
        """Update/insert rows for the given people; show live card counts."""
        for person in people_to_update:
            if not person:
                person = "Unknown"
            stats = per_person_stats.get(person, {
                "detections": 0, "best_score": None, "emotions": Counter(),
                "red": 0, "yellow": 0, "green": 0
            })
            em_counter = stats.get("emotions", Counter())
            em_str = " | ".join([f"{k}×{v}" for k, v in em_counter.most_common(3)]) if em_counter else "N/A"
            best_score = stats.get("best_score", None)
            best_score_str = f"{best_score:.2f}" if best_score is not None else "N/A"
            key = (video_base, person)
            iid = self.dashboard_rows.get(key)

            # Coerce to strings for Treeview robustness
            values = (
                str(video_base),
                str(stats.get("detections", 0)),
                str(best_score_str),
                str(em_str),
                str(stats.get("red", 0)),
                str(stats.get("yellow", 0)),
                str(stats.get("green", 0)),
            )

            if iid is None:
                pimg = self._avatar_for_person(video_base, person)
                iid = self.dashboard_tree.insert("", "end", text=person, image=pimg, values=values)
                self.dashboard_rows[key] = iid
            else:
                self.dashboard_tree.item(iid, values=values)

        self.dashboard_tree.update_idletasks()
        self.update_status(f"Dashboard live update — {video_base}")


# ---------------- Run app ----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionSystem(root)
    root.mainloop()
