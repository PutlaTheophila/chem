"""Interactive tangent labeling tool with timeline and prediction overlay.

Open via:   python label_tool.py

Workflow:
  1. Click "Open Video..." to load any .mp4 / .mov / .avi / .mkv.
  2. First time per video: click two points on the frame to set the horizontal
     axis (stored at dataset/axis/<video>.json).
  3. Click two points to draw the yellow tangent; angle vs axis is computed
     and saved (dataset/labels/<video>.csv).
  4. If models/tangent_best.pt exists, predictions appear as a dashed yellow
     line on each new frame. SPACE accepts, click 2 pts to override.
  5. "Predict All" runs the model over every frame and plots the angle
     timeline. Labels overlay as green dots.

Keys: Left/Right (prev/next), J/L (jump -/+ stride), Home/End (first/last),
Space (accept prediction), C (clear current), R (reload model), A (re-set axis).
"""
import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from tkinter import (
    Tk, Canvas, StringVar, BOTH, X, Y, LEFT, RIGHT, BOTTOM, TOP, HORIZONTAL,
    filedialog, messagebox,
)
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

ROOT = Path(__file__).parent
DATASET = ROOT / "dataset"
MODELS = ROOT / "models"
IMG_DIR = DATASET / "images"
LABEL_DIR = DATASET / "labels"
AXIS_DIR = DATASET / "axis"

LABEL_COLS = ["frame", "tx1", "ty1", "tx2", "ty2", "angle_deg", "source"]
SOURCE_COLOR = {"manual": "#2ecc71", "corrected": "#27ae60", "model_accepted": "#16a085"}


def angle_between(ax, tan) -> float:
    (a1, a2), (t1, t2) = ax, tan
    vh = np.array([a2[0] - a1[0], a2[1] - a1[1]], dtype=float)
    vt = np.array([t2[0] - t1[0], t2[1] - t1[1]], dtype=float)
    nh, nt = np.linalg.norm(vh), np.linalg.norm(vt)
    if nh < 1e-6 or nt < 1e-6:
        return float("nan")
    return math.degrees(math.acos(float(np.clip(np.dot(vh, vt) / (nh * nt), -1.0, 1.0))))


def video_stem(video_path: Path) -> str:
    return video_path.stem.replace(" ", "_")


@dataclass
class FrameLabel:
    frame: int
    t1: tuple
    t2: tuple
    angle: float
    source: str


class Predictor:
    def __init__(self, ckpt_path: Path):
        self.ckpt_path = ckpt_path
        self.model = None
        self.device = None
        self.tfm = None
        self.mtime = 0.0
        self._init()

    def _init(self):
        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms
        except ImportError:
            return
        self.torch = torch
        self.nn = nn
        self.models_mod = models
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.load()

    def load(self) -> bool:
        if not self.ckpt_path.exists() or self.device is None:
            return False
        mtime = self.ckpt_path.stat().st_mtime
        if self.model is not None and mtime == self.mtime:
            return True
        m = self.models_mod.resnet18(weights=None)
        m.fc = self.nn.Linear(m.fc.in_features, 4)
        ckpt = self.torch.load(self.ckpt_path, map_location=self.device)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        m.load_state_dict(state)
        m.eval().to(self.device)
        self.model = m
        self.mtime = mtime
        return True

    def predict(self, frame_bgr: np.ndarray):
        if self.model is None:
            return None
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        with self.torch.no_grad():
            x = self.tfm(pil).unsqueeze(0).to(self.device)
            out = self.model(x).squeeze(0).cpu().numpy()
        out = np.clip(out, 0.0, 1.0)
        return ((out[0] * w, out[1] * h), (out[2] * w, out[3] * h))


class App:
    def __init__(self, root: Tk, stride: int):
        self.root = root
        self.stride = stride
        self.root.title("Tangent Labeler")
        self.root.geometry("1400x900")
        self._setup_style()

        self.video_path: Path | None = None
        self.cap: cv2.VideoCapture | None = None
        self.n_frames = 0
        self.fps = 30.0
        self.cur_frame = 0
        self.cur_bgr: np.ndarray | None = None
        self.cur_w = 0
        self.cur_h = 0
        self.disp_scale = 1.0

        self.axis: tuple | None = None
        self.axis_mode = False
        self.pending_axis: list = []
        self.pending_tan: list = []

        self.labels: dict[int, FrameLabel] = {}
        self.preds_all: dict[int, tuple] = {}  # frame -> ((t1,t2), angle)
        self.pred_current: tuple | None = None

        self.photo = None
        self.predict_job = None  # generator for incremental predict-all

        self.predictor = Predictor(MODELS / "tangent_best.pt")

        self._build_ui()
        self._bind_keys()
        self._refresh_stats()
        self._refresh_plot()

    def _setup_style(self):
        s = ttk.Style()
        try:
            s.theme_use("clam")
        except Exception:
            pass
        s.configure("TButton", padding=6)
        s.configure("Toolbar.TButton", padding=(10, 6))
        s.configure("Card.TLabelframe", padding=8)
        s.configure("Card.TLabelframe.Label", font=("Helvetica", 11, "bold"))
        s.configure("Stat.TLabel", font=("Helvetica", 10))
        s.configure("StatBold.TLabel", font=("Helvetica", 10, "bold"))
        s.configure("Status.TLabel", padding=4, background="#222", foreground="#eee")

    def _build_ui(self):
        # toolbar
        bar = ttk.Frame(self.root, padding=(8, 6))
        bar.pack(side=TOP, fill=X)
        ttk.Button(bar, text="Open Video…", style="Toolbar.TButton",
                   command=self.open_dialog).pack(side=LEFT, padx=(0, 4))
        ttk.Separator(bar, orient="vertical").pack(side=LEFT, fill=Y, padx=6)
        ttk.Button(bar, text="Set Axis (A)", style="Toolbar.TButton",
                   command=self.enter_axis_mode).pack(side=LEFT, padx=2)
        ttk.Button(bar, text="Clear Label (C)", style="Toolbar.TButton",
                   command=self.clear_current).pack(side=LEFT, padx=2)
        ttk.Separator(bar, orient="vertical").pack(side=LEFT, fill=Y, padx=6)
        ttk.Button(bar, text="Reload Model (R)", style="Toolbar.TButton",
                   command=self.reload_model).pack(side=LEFT, padx=2)
        self.predict_btn = ttk.Button(bar, text="Predict All Frames",
                                      style="Toolbar.TButton",
                                      command=self.toggle_predict_all)
        self.predict_btn.pack(side=LEFT, padx=2)
        ttk.Separator(bar, orient="vertical").pack(side=LEFT, fill=Y, padx=6)
        ttk.Label(bar, text="Goto:").pack(side=LEFT)
        self.goto_var = StringVar()
        e = ttk.Entry(bar, textvariable=self.goto_var, width=8)
        e.pack(side=LEFT, padx=(2, 2))
        e.bind("<Return>", lambda ev: self._goto_from_entry())
        ttk.Button(bar, text="Go", command=self._goto_from_entry).pack(side=LEFT)

        # main split: video (left) | side panel (right)
        body = ttk.PanedWindow(self.root, orient="horizontal")
        body.pack(fill=BOTH, expand=True, padx=8, pady=(0, 4))

        # left = video + slider
        left = ttk.Frame(body)
        body.add(left, weight=4)
        self.canvas = Canvas(left, bg="#111", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Configure>", lambda e: self.redraw())

        slider_row = ttk.Frame(left)
        slider_row.pack(fill=X, pady=(4, 0))
        self.slider_var = StringVar(value="0")
        self.slider = ttk.Scale(slider_row, from_=0, to=1, orient=HORIZONTAL,
                                command=self._on_slider)
        self.slider.pack(side=LEFT, fill=X, expand=True, padx=(0, 8))
        self.frame_label = ttk.Label(slider_row, text="0 / 0", width=14)
        self.frame_label.pack(side=RIGHT)

        # right = stats + label list
        right = ttk.Frame(body, padding=(4, 0))
        body.add(right, weight=1)

        stats_card = ttk.LabelFrame(right, text="Session", style="Card.TLabelframe")
        stats_card.pack(fill=X, pady=(0, 8))
        self.stat_vars = {k: StringVar(value="—") for k in
                          ["video", "frames", "fps", "axis", "model", "labels", "preds"]}
        rows = [
            ("Video:", "video"), ("Frames:", "frames"), ("FPS:", "fps"),
            ("Axis:", "axis"), ("Model:", "model"),
            ("Labels:", "labels"), ("Preds cached:", "preds"),
        ]
        for i, (k, key) in enumerate(rows):
            ttk.Label(stats_card, text=k, style="StatBold.TLabel").grid(row=i, column=0, sticky="w")
            ttk.Label(stats_card, textvariable=self.stat_vars[key], style="Stat.TLabel"
                      ).grid(row=i, column=1, sticky="w", padx=(8, 0))
        stats_card.columnconfigure(1, weight=1)

        list_card = ttk.LabelFrame(right, text="Labels", style="Card.TLabelframe")
        list_card.pack(fill=BOTH, expand=True)
        cols = ("frame", "angle", "src")
        self.tree = ttk.Treeview(list_card, columns=cols, show="headings", height=12)
        self.tree.heading("frame", text="Frame")
        self.tree.heading("angle", text="Angle°")
        self.tree.heading("src", text="Source")
        self.tree.column("frame", width=70, anchor="e")
        self.tree.column("angle", width=70, anchor="e")
        self.tree.column("src", width=110, anchor="w")
        vsb = ttk.Scrollbar(list_card, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=LEFT, fill=BOTH, expand=True)
        vsb.pack(side=RIGHT, fill=Y)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # bottom: timeline plot
        plot_frame = ttk.LabelFrame(self.root, text="Angle Timeline",
                                    style="Card.TLabelframe", padding=4)
        plot_frame.pack(fill=X, padx=8, pady=(0, 4))
        self.fig = Figure(figsize=(10, 1.8), dpi=100, facecolor="#fafafa")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#fafafa")
        self.fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.22)
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.plot_canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        self.plot_canvas.mpl_connect("button_press_event", self._on_plot_click)

        # status bar
        self.status = ttk.Label(self.root, text="Open a video to begin.",
                                style="Status.TLabel", anchor="w")
        self.status.pack(side=BOTTOM, fill=X)

    def _bind_keys(self):
        self.root.bind("<Left>", lambda e: self.goto(self.cur_frame - 1))
        self.root.bind("<Right>", lambda e: self.goto(self.cur_frame + 1))
        self.root.bind("j", lambda e: self.goto(self.cur_frame - self.stride))
        self.root.bind("l", lambda e: self.goto(self.cur_frame + self.stride))
        self.root.bind("<Home>", lambda e: self.goto(0))
        self.root.bind("<End>", lambda e: self.goto(self.n_frames - 1))
        self.root.bind("<space>", lambda e: self.accept_prediction())
        self.root.bind("c", lambda e: self.clear_current())
        self.root.bind("r", lambda e: self.reload_model())
        self.root.bind("a", lambda e: self.enter_axis_mode())

    # ---- file IO ----
    def open_dialog(self):
        path = filedialog.askopenfilename(
            title="Select a video",
            filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")])
        if path:
            self.open_video(Path(path))

    def open_video(self, path: Path):
        if self.cap is not None:
            self.cap.release()
        self.video_path = path
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            messagebox.showerror("Open Video", f"Failed to open\n{path}")
            return
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.cur_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cur_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.axis = self._load_axis()
        self.labels = self._load_labels()
        self.preds_all = {}
        self.axis_mode = self.axis is None
        self.pending_axis.clear()
        self.pending_tan.clear()
        if self.predict_job is not None:
            self.predict_job = None
            self.predict_btn.configure(text="Predict All Frames")
        self.slider.configure(from_=0, to=max(self.n_frames - 1, 1))
        self.root.title(f"Tangent Labeler — {path.name}")
        self.goto(0)
        self._refresh_tree()
        self._refresh_plot()
        self._refresh_stats()
        if self.axis_mode:
            self.status.configure(text="Click two points to set the horizontal axis.")

    def _axis_path(self) -> Path:
        return AXIS_DIR / f"{video_stem(self.video_path)}.json"

    def _labels_path(self) -> Path:
        return LABEL_DIR / f"{video_stem(self.video_path)}.csv"

    def _frame_image_path(self, frame: int) -> Path:
        return IMG_DIR / video_stem(self.video_path) / f"frame_{frame:06d}.png"

    def _load_axis(self):
        p = self._axis_path()
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        return (tuple(d["ax1"]), tuple(d["ax2"]))

    def _save_axis(self):
        AXIS_DIR.mkdir(parents=True, exist_ok=True)
        self._axis_path().write_text(json.dumps(
            {"ax1": list(self.axis[0]), "ax2": list(self.axis[1])}))

    def _load_labels(self) -> dict[int, FrameLabel]:
        out: dict[int, FrameLabel] = {}
        p = self._labels_path()
        if not p.exists():
            return out
        with p.open() as f:
            for row in csv.DictReader(f):
                fr = int(row["frame"])
                out[fr] = FrameLabel(
                    frame=fr,
                    t1=(float(row["tx1"]), float(row["ty1"])),
                    t2=(float(row["tx2"]), float(row["ty2"])),
                    angle=float(row["angle_deg"]),
                    source=row.get("source", "manual"),
                )
        return out

    def _save_labels(self):
        LABEL_DIR.mkdir(parents=True, exist_ok=True)
        with self._labels_path().open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(LABEL_COLS)
            for fr in sorted(self.labels):
                lb = self.labels[fr]
                w.writerow([lb.frame, lb.t1[0], lb.t1[1], lb.t2[0], lb.t2[1],
                            f"{lb.angle:.4f}", lb.source])

    def _save_frame_image(self, frame: int):
        out = self._frame_image_path(frame)
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            cv2.imwrite(str(out), self.cur_bgr)

    # ---- navigation ----
    def _goto_from_entry(self):
        try:
            self.goto(int(self.goto_var.get() or 0))
        except ValueError:
            pass

    def goto(self, frame: int):
        if self.cap is None or self.n_frames == 0:
            return
        frame = max(0, min(self.n_frames - 1, frame))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, bgr = self.cap.read()
        if not ok:
            return
        self.cur_frame = frame
        self.cur_bgr = bgr
        self.pending_tan.clear()
        self.pred_current = None
        if (frame not in self.labels and not self.axis_mode and
                self.predictor.load()):
            self.pred_current = self.predictor.predict(bgr)
        # update slider without re-triggering goto
        self.slider.set(frame)
        self.frame_label.configure(text=f"{frame} / {self.n_frames - 1}")
        self.redraw()
        self._refresh_plot_marker()
        self._refresh_stats()

    def _on_slider(self, value):
        try:
            f = int(float(value))
        except ValueError:
            return
        if f != self.cur_frame:
            self.goto(f)

    # ---- click handlers ----
    def _disp_to_image(self, x: float, y: float):
        return (x / self.disp_scale, y / self.disp_scale)

    def _image_to_disp(self, x: float, y: float):
        return (x * self.disp_scale, y * self.disp_scale)

    def on_click(self, ev):
        if self.cur_bgr is None:
            return
        ix, iy = self._disp_to_image(ev.x, ev.y)
        if self.axis_mode:
            self.pending_axis.append((ix, iy))
            if len(self.pending_axis) == 2:
                self.axis = (self.pending_axis[0], self.pending_axis[1])
                self._save_axis()
                self.axis_mode = False
                self.pending_axis.clear()
                self.status.configure(text="Axis set. Now click two points to draw the tangent.")
                # refresh predictions now that axis exists
                if self.predictor.load():
                    self.pred_current = self.predictor.predict(self.cur_bgr)
            self.redraw()
            self._refresh_stats()
            return
        if self.axis is None:
            self.status.configure(text="Set the axis first (Set Axis button or press A).")
            return
        self.pending_tan.append((ix, iy))
        if len(self.pending_tan) == 2:
            t1, t2 = self.pending_tan
            self.pending_tan.clear()
            src = "corrected" if self.pred_current is not None else "manual"
            self._commit_tangent(t1, t2, source=src)
        else:
            self.redraw()

    def _commit_tangent(self, t1, t2, source: str):
        ang = angle_between(self.axis, (t1, t2))
        self.labels[self.cur_frame] = FrameLabel(self.cur_frame, t1, t2, ang, source)
        self._save_frame_image(self.cur_frame)
        self._save_labels()
        self.pred_current = None
        self.redraw()
        self._refresh_tree()
        self._refresh_plot()
        self._refresh_stats()

    def enter_axis_mode(self):
        self.axis_mode = True
        self.pending_axis.clear()
        self.status.configure(text="Click two points to set the horizontal axis.")
        self.redraw()

    def clear_current(self):
        if self.cur_frame in self.labels:
            del self.labels[self.cur_frame]
            self._save_labels()
            self._refresh_tree()
            self._refresh_plot()
            self._refresh_stats()
        self.pending_tan.clear()
        self.redraw()

    def reload_model(self):
        ok = self.predictor.load()
        self.status.configure(
            text="Model reloaded from models/tangent_best.pt." if ok else
                 "No model found at models/tangent_best.pt — run train.py first.")
        self.preds_all.clear()
        self._refresh_plot()
        self._refresh_stats()
        if self.cur_bgr is not None:
            self.goto(self.cur_frame)

    def accept_prediction(self):
        if self.pred_current is None:
            return
        self._commit_tangent(self.pred_current[0], self.pred_current[1],
                             source="model_accepted")

    # ---- predict-all (incremental, non-blocking) ----
    def toggle_predict_all(self):
        if self.predict_job is not None:
            self.predict_job = None
            self.predict_btn.configure(text="Predict All Frames")
            self.status.configure(text="Prediction stopped.")
            return
        if self.cap is None:
            return
        if not self.predictor.load():
            messagebox.showinfo("Predict All", "No trained model at models/tangent_best.pt.")
            return
        if self.axis is None:
            messagebox.showinfo("Predict All", "Set the axis first.")
            return
        self.preds_all = {}
        self.predict_btn.configure(text="Stop Predict")
        # use a separate VideoCapture so we don't disrupt navigation
        self.predict_job = {"cap": cv2.VideoCapture(str(self.video_path)), "i": 0}
        self.root.after(1, self._predict_chunk)

    def _predict_chunk(self):
        if self.predict_job is None:
            return
        cap = self.predict_job["cap"]
        i = self.predict_job["i"]
        CHUNK = 12
        for _ in range(CHUNK):
            ok, bgr = cap.read()
            if not ok:
                cap.release()
                self.predict_job = None
                self.predict_btn.configure(text="Predict All Frames")
                self.status.configure(
                    text=f"Done. Predicted {len(self.preds_all)} frames.")
                self._refresh_plot()
                self._refresh_stats()
                return
            tan = self.predictor.predict(bgr)
            if tan is not None:
                ang = angle_between(self.axis, tan)
                self.preds_all[i] = (tan, ang)
            i += 1
        self.predict_job["i"] = i
        if i % 60 == 0:
            self.status.configure(
                text=f"Predicting… {i}/{self.n_frames} ({100*i/max(self.n_frames,1):.0f}%)")
            self._refresh_plot()
            self._refresh_stats()
        self.root.after(1, self._predict_chunk)

    # ---- rendering ----
    def redraw(self):
        self.canvas.delete("all")
        if self.cur_bgr is None:
            self.canvas.create_text(
                self.canvas.winfo_width() // 2 or 200,
                self.canvas.winfo_height() // 2 or 100,
                text="Click  Open Video…  to begin",
                fill="#888", font=("Helvetica", 18))
            return
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        self.disp_scale = max(min(cw / self.cur_w, ch / self.cur_h), 1e-3)
        dw = max(int(self.cur_w * self.disp_scale), 1)
        dh = max(int(self.cur_h * self.disp_scale), 1)
        rgb = cv2.cvtColor(self.cur_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((dw, dh), Image.BILINEAR)
        self.photo = ImageTk.PhotoImage(pil)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

        if self.axis is not None:
            p1 = self._image_to_disp(*self.axis[0])
            p2 = self._image_to_disp(*self.axis[1])
            self.canvas.create_line(p1[0], p1[1], p2[0], p2[1],
                                    fill="white", width=2)

        for pt in self.pending_axis:
            x, y = self._image_to_disp(*pt)
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5,
                                    outline="#e74c3c", width=2)

        lb = self.labels.get(self.cur_frame)
        if lb is not None:
            p1 = self._image_to_disp(*lb.t1)
            p2 = self._image_to_disp(*lb.t2)
            self.canvas.create_line(p1[0], p1[1], p2[0], p2[1],
                                    fill="#f1c40f", width=3)
        elif self.pred_current is not None:
            p1 = self._image_to_disp(*self.pred_current[0])
            p2 = self._image_to_disp(*self.pred_current[1])
            self.canvas.create_line(p1[0], p1[1], p2[0], p2[1],
                                    fill="#f1c40f", width=2, dash=(6, 4))

        for pt in self.pending_tan:
            x, y = self._image_to_disp(*pt)
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5,
                                    outline="#f1c40f", width=2)

        # HUD
        ang_str = ""
        if lb is not None:
            ang_str = f"{lb.angle:.2f}° (labeled)"
        elif self.pred_current is not None and self.axis is not None:
            ang_str = f"{angle_between(self.axis, self.pred_current):.2f}° (pred)"
        t = self.cur_frame / self.fps if self.fps else 0.0
        hud = f"Frame {self.cur_frame}  t={t:.2f}s   {ang_str}"
        self.canvas.create_rectangle(10, 10, 10 + 10 * len(hud), 38,
                                     fill="#000", outline="", stipple="gray50")
        self.canvas.create_text(16, 24, anchor="w", text=hud,
                                fill="white", font=("Menlo", 13, "bold"))

    def _refresh_stats(self):
        sv = self.stat_vars
        sv["video"].set(self.video_path.name if self.video_path else "—")
        sv["frames"].set(str(self.n_frames) if self.n_frames else "—")
        sv["fps"].set(f"{self.fps:.2f}" if self.fps else "—")
        if self.axis_mode:
            sv["axis"].set("setting…")
        else:
            sv["axis"].set("set ✓" if self.axis else "not set")
        sv["model"].set("loaded ✓" if (self.predictor.model is not None) else "none")
        sv["labels"].set(str(len(self.labels)))
        sv["preds"].set(str(len(self.preds_all)))

    def _refresh_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for fr in sorted(self.labels):
            lb = self.labels[fr]
            self.tree.insert("", "end", iid=str(fr),
                             values=(fr, f"{lb.angle:.2f}", lb.source))

    def _on_tree_select(self, _ev):
        sel = self.tree.selection()
        if not sel:
            return
        try:
            self.goto(int(sel[0]))
        except ValueError:
            pass

    def _refresh_plot(self):
        self.ax.clear()
        self.ax.set_xlim(0, max(self.n_frames - 1, 1))
        self.ax.set_ylim(0, 180)
        self.ax.set_xlabel("frame", fontsize=8)
        self.ax.set_ylabel("angle°", fontsize=8)
        self.ax.tick_params(labelsize=8)
        self.ax.grid(True, alpha=0.25)
        if self.preds_all:
            xs = sorted(self.preds_all.keys())
            ys = [self.preds_all[x][1] for x in xs]
            self.ax.plot(xs, ys, color="#f1c40f", lw=1.2, label="prediction")
        if self.labels:
            xs = sorted(self.labels.keys())
            ys = [self.labels[x].angle for x in xs]
            cs = [SOURCE_COLOR.get(self.labels[x].source, "#2ecc71") for x in xs]
            self.ax.scatter(xs, ys, c=cs, s=22, edgecolor="#1e7e3a",
                            linewidth=0.6, zorder=3, label="labels")
        self._cur_line = self.ax.axvline(self.cur_frame, color="#e74c3c",
                                         lw=1.2, alpha=0.8)
        if self.labels or self.preds_all:
            self.ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
        self.fig.tight_layout(pad=0.5)
        self.plot_canvas.draw_idle()

    def _refresh_plot_marker(self):
        try:
            self._cur_line.set_xdata([self.cur_frame, self.cur_frame])
            self.plot_canvas.draw_idle()
        except Exception:
            self._refresh_plot()

    def _on_plot_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        self.goto(int(round(event.xdata)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stride", type=int, default=10)
    args = p.parse_args()
    root = Tk()
    App(root, args.stride)
    root.mainloop()


if __name__ == "__main__":
    main()
