"""Manual angle measurer.

Workflow:
  1. Open a video (or image).
  2. Scrub to the frame you want to measure.
  3. Click two points to define the axis (axis A, axis B).
  4. Click a single point on the needle. The tool fits a local tangent
     through the needle at that click (binary mask + skeleton + PCA in a
     window around the click), extends both lines to their intersection,
     and reports the angle between them.

Run:
    python manual_angle.py
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from skimage.morphology import skeletonize


WINDOW_HALF_DEFAULT = 40       # local patch half-size around click (px, video coords)
TANGENT_RADIUS_DEFAULT = 18    # PCA neighbourhood radius around click (px)


def local_needle_tangent(
    gray: np.ndarray,
    click_xy: np.ndarray,
    window_half: int,
    pca_radius: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit a unit-tangent vector to the needle near ``click_xy``.

    Strategy: crop a window around the click, Otsu-threshold both
    polarities, skeletonize each, pick the candidate whose skeleton has
    the most pixels inside the PCA neighbourhood, then PCA on those
    pixels to get the principal direction.
    """
    h, w = gray.shape
    cx, cy = float(click_xy[0]), float(click_xy[1])
    x0 = int(max(0, round(cx) - window_half))
    y0 = int(max(0, round(cy) - window_half))
    x1 = int(min(w, round(cx) + window_half + 1))
    y1 = int(min(h, round(cy) + window_half + 1))
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    patch = gray[y0:y1, x0:x1]
    blur = cv2.GaussianBlur(patch, (3, 3), 0)

    best_pts: np.ndarray | None = None
    best_count = 0
    for invert in (False, True):
        src = 255 - blur if invert else blur
        _, bw = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        if bw.sum() == 0:
            continue
        # Keep only the connected component nearest the click.
        n, labels = cv2.connectedComponents(bw)
        if n <= 1:
            continue
        click_local = np.array([cx - x0, cy - y0])
        best_label = 0
        best_d = float("inf")
        for L in range(1, n):
            ys, xs = np.where(labels == L)
            if len(xs) == 0:
                continue
            d = float(np.min((xs - click_local[0]) ** 2 + (ys - click_local[1]) ** 2))
            if d < best_d:
                best_d = d
                best_label = L
        if best_label == 0:
            continue
        comp = (labels == best_label).astype(np.uint8)
        sk = skeletonize(comp > 0).astype(np.uint8)
        ys, xs = np.where(sk > 0)
        if len(xs) < 6:
            continue
        pts = np.column_stack([xs + x0, ys + y0]).astype(np.float64)
        inside = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) <= pca_radius
        sub = pts[inside]
        if len(sub) >= 6 and len(sub) > best_count:
            best_count = len(sub)
            best_pts = sub

    if best_pts is None or len(best_pts) < 6:
        return None
    centroid = best_pts.mean(axis=0)
    centred = best_pts - centroid
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    direction = vh[0]
    n = float(np.linalg.norm(direction))
    if n < 1e-9:
        return None
    return centroid, direction / n


def line_line_intersection(
    p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray,
) -> np.ndarray | None:
    """Intersection of two parametric lines p + t·d. Returns None if parallel."""
    A = np.column_stack([d1, -d2])
    rhs = p2 - p1
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-9:
        return None
    t = (rhs[0] * A[1, 1] - rhs[1] * A[0, 1]) / det
    return p1 + t * d1


def acute_angle_deg(d1: np.ndarray, d2: np.ndarray) -> float:
    c = float(np.dot(d1, d2)) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-12)
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(abs(c))))


def signed_angle_deg(d_from: np.ndarray, d_to: np.ndarray) -> float:
    a = np.arctan2(d_from[1], d_from[0])
    b = np.arctan2(d_to[1], d_to[0])
    diff = np.degrees(b - a)
    return ((diff + 180.0) % 360.0) - 180.0


class ManualAngleGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Manual Angle Measurer")
        root.geometry("1180x780")

        self.cap: cv2.VideoCapture | None = None
        self.video_path: Path | None = None
        self.n_frames = 0
        self.frame_w = 0
        self.frame_h = 0
        self.current_idx = 0
        self.current_frame_bgr: np.ndarray | None = None
        self.disp_scale = 1.0
        self.disp_offset = (0, 0)
        self._photo: ImageTk.PhotoImage | None = None

        self.points: list[tuple[float, float]] = []
        self.tangent: tuple[np.ndarray, np.ndarray] | None = None

        self._build_ui()

    # ----------------------------------------------------------- ui ---
    def _build_ui(self) -> None:
        tb = ttk.Frame(self.root, padding=4)
        tb.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(tb, text="Open Video/Image…", command=self.open_media).pack(side=tk.LEFT)
        ttk.Button(tb, text="Reset Points", command=self.reset_points).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(tb, text="Window half:").pack(side=tk.LEFT, padx=(16, 2))
        self.win_var = tk.IntVar(value=WINDOW_HALF_DEFAULT)
        ttk.Spinbox(tb, from_=10, to=200, textvariable=self.win_var, width=4,
                    command=self._refit_tangent).pack(side=tk.LEFT)
        ttk.Label(tb, text="PCA radius:").pack(side=tk.LEFT, padx=(10, 2))
        self.pca_var = tk.IntVar(value=TANGENT_RADIUS_DEFAULT)
        ttk.Spinbox(tb, from_=4, to=80, textvariable=self.pca_var, width=4,
                    command=self._refit_tangent).pack(side=tk.LEFT)

        body = ttk.Frame(self.root)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(body, bg="#202020", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Configure>", lambda e: self._refresh_canvas())

        right = ttk.Frame(body, padding=8)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Label(right, text="Picks", font=("", 11, "bold")).pack(anchor="w")
        self.pick_lbls: list[ttk.Label] = []
        for name, color in (("axis A", "#9af09a"),
                             ("axis B", "#9af09a"),
                             ("needle click", "#5ec8ff")):
            lbl = ttk.Label(right, text=f"{name}: —", foreground=color)
            lbl.pack(anchor="w")
            self.pick_lbls.append(lbl)
        ttk.Separator(right).pack(fill=tk.X, pady=6)
        ttk.Label(right, text="Result", font=("", 11, "bold")).pack(anchor="w")
        self.result_lbl = ttk.Label(right, text="—", justify="left", font=("", 11))
        self.result_lbl.pack(anchor="w")
        ttk.Separator(right).pack(fill=tk.X, pady=6)
        ttk.Label(right, text=(
            "1) click axis A\n"
            "2) click axis B\n"
            "3) click a point on the needle\n"
            "   → tangent is auto-fit there"
        ), foreground="#cfcfcf", justify="left").pack(anchor="w")

        bottom = ttk.Frame(self.root, padding=(6, 2))
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        self.scrub = ttk.Scale(bottom, from_=0, to=0, orient=tk.HORIZONTAL,
                               command=self._on_scrub)
        self.scrub.pack(side=tk.TOP, fill=tk.X)
        self.scrub.state(["disabled"])
        self.status_var = tk.StringVar(value="Open a video or image to begin.")
        ttk.Label(bottom, textvariable=self.status_var).pack(side=tk.LEFT)

    # ------------------------------------------------------ actions ---
    def open_media(self) -> None:
        path = filedialog.askopenfilename(
            title="Open video or image",
            filetypes=[("Video/Image",
                        "*.mp4 *.mov *.avi *.mkv *.webm "
                        "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                       ("All", "*.*")],
        )
        if not path:
            return
        p = Path(path)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            frame = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if frame is None:
                messagebox.showerror("Open", f"Cannot read image {p}")
                return
            self.video_path = p
            self.current_frame_bgr = frame
            self.frame_h, self.frame_w = frame.shape[:2]
            self.n_frames = 1
            self.current_idx = 0
            self.scrub.configure(from_=0, to=0)
            self.scrub.state(["disabled"])
        else:
            cap = cv2.VideoCapture(str(p))
            if not cap.isOpened():
                messagebox.showerror("Open", f"Cannot open {p}")
                return
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ok, frame = cap.read()
            if not ok:
                cap.release()
                messagebox.showerror("Open", "Cannot read first frame.")
                return
            self.cap = cap
            self.video_path = p
            self.current_frame_bgr = frame
            self.frame_h, self.frame_w = frame.shape[:2]
            self.n_frames = max(n, 1)
            self.current_idx = 0
            self.scrub.configure(from_=0, to=max(self.n_frames - 1, 0))
            self.scrub.set(0)
            if self.n_frames > 1:
                self.scrub.state(["!disabled"])
            else:
                self.scrub.state(["disabled"])
        self.reset_points()
        self.status_var.set(
            f"Loaded {p.name}  ({self.frame_w}x{self.frame_h}, {self.n_frames} frames). "
            f"Click axis A."
        )
        self._refresh_canvas()

    def reset_points(self) -> None:
        self.points = []
        self.tangent = None
        for i, lbl in enumerate(self.pick_lbls):
            name = ("axis A", "axis B", "needle click")[i]
            lbl.config(text=f"{name}: —")
        self.result_lbl.config(text="—")
        if self.current_frame_bgr is not None:
            self.status_var.set("Click axis A.")
        self._refresh_canvas()

    def on_canvas_click(self, event) -> None:
        if self.current_frame_bgr is None:
            return
        if len(self.points) >= 3:
            return
        vx, vy = self._canvas_to_video(event.x, event.y)
        if vx < 0 or vx >= self.frame_w or vy < 0 or vy >= self.frame_h:
            return
        self.points.append((vx, vy))
        i = len(self.points) - 1
        name = ("axis A", "axis B", "needle click")[i]
        self.pick_lbls[i].config(text=f"{name}: ({vx:.0f}, {vy:.0f})")
        if i == 2:
            self._refit_tangent()
            if self.tangent is None:
                self.status_var.set(
                    "Could not detect needle direction at that click. "
                    "Try a different point or enlarge the window."
                )
            else:
                self.status_var.set("Done. Reset to measure again.")
        else:
            nxt = ("axis A", "axis B", "needle click")[i + 1]
            self.status_var.set(f"Click {nxt}.")
        self._update_result()
        self._refresh_canvas()

    def _refit_tangent(self) -> None:
        if len(self.points) < 3 or self.current_frame_bgr is None:
            return
        gray = cv2.cvtColor(self.current_frame_bgr, cv2.COLOR_BGR2GRAY)
        click = np.array(self.points[2], dtype=np.float64)
        self.tangent = local_needle_tangent(
            gray, click,
            window_half=int(self.win_var.get()),
            pca_radius=int(self.pca_var.get()),
        )
        self._update_result()
        self._refresh_canvas()

    # ------------------------------------------------ frame scrubbing ---
    def _on_scrub(self, value) -> None:
        if self.cap is None:
            return
        try:
            idx = int(float(value))
        except (TypeError, ValueError):
            return
        if idx == self.current_idx:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            return
        self.current_idx = idx
        self.current_frame_bgr = frame
        if len(self.points) == 3:
            self._refit_tangent()
        else:
            self._refresh_canvas()
        self.status_var.set(f"frame {idx}/{self.n_frames - 1}")

    # --------------------------------------------------- computation ---
    def _axis_dir(self) -> tuple[np.ndarray, np.ndarray] | None:
        if len(self.points) < 2:
            return None
        a = np.array(self.points[0], dtype=np.float64)
        b = np.array(self.points[1], dtype=np.float64)
        d = b - a
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            return None
        return a, d / n

    def _update_result(self) -> None:
        axis = self._axis_dir()
        if axis is None or self.tangent is None:
            self.result_lbl.config(text="—")
            return
        a_pt, a_dir = axis
        t_pt, t_dir = self.tangent
        inter = line_line_intersection(a_pt, a_dir, t_pt, t_dir)
        acute = acute_angle_deg(a_dir, t_dir)
        signed = signed_angle_deg(a_dir, t_dir)
        if inter is None:
            inter_txt = "parallel (no intersection)"
        else:
            inter_txt = f"({inter[0]:.1f}, {inter[1]:.1f})"
        self.result_lbl.config(
            text=(f"acute angle    {acute:6.2f}°\n"
                  f"signed (axis→needle)\n"
                  f"               {signed:+7.2f}°\n"
                  f"intersection   {inter_txt}")
        )

    # ------------------------------------------------------ display ---
    def _refresh_canvas(self) -> None:
        if self.current_frame_bgr is None:
            self.canvas.delete("all")
            return
        self.canvas.delete("all")
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        scale = min(cw / self.frame_w, ch / self.frame_h)
        if scale <= 0:
            return
        new_w = max(1, int(self.frame_w * scale))
        new_h = max(1, int(self.frame_h * scale))
        ox = (cw - new_w) // 2
        oy = (ch - new_h) // 2
        self.disp_scale = scale
        self.disp_offset = (ox, oy)
        rgb = cv2.cvtColor(self.current_frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((new_w, new_h), Image.BILINEAR)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(ox, oy, anchor=tk.NW, image=self._photo)

        def to_canvas(vx, vy):
            return ox + vx * scale, oy + vy * scale

        diag = float(np.hypot(self.frame_w, self.frame_h))

        axis = self._axis_dir()
        if axis is not None:
            a_pt, a_dir = axis
            p1 = a_pt - a_dir * diag
            p2 = a_pt + a_dir * diag
            self.canvas.create_line(*to_canvas(*p1), *to_canvas(*p2),
                                    fill="#9af09a", dash=(6, 4), width=2)

        if self.tangent is not None:
            t_pt, t_dir = self.tangent
            p1 = t_pt - t_dir * diag
            p2 = t_pt + t_dir * diag
            self.canvas.create_line(*to_canvas(*p1), *to_canvas(*p2),
                                    fill="#5ec8ff", dash=(6, 4), width=2)

        if axis is not None and self.tangent is not None:
            a_pt, a_dir = axis
            t_pt, t_dir = self.tangent
            inter = line_line_intersection(a_pt, a_dir, t_pt, t_dir)
            if inter is not None:
                cxp, cyp = to_canvas(*inter)
                r = 7
                self.canvas.create_oval(cxp - r, cyp - r, cxp + r, cyp + r,
                                        outline="#ffd400", width=2)
                arc_r = 36
                start_deg = -np.degrees(np.arctan2(a_dir[1], a_dir[0]))
                end_deg = -np.degrees(np.arctan2(t_dir[1], t_dir[0]))
                extent = ((end_deg - start_deg + 180) % 360) - 180
                self.canvas.create_arc(cxp - arc_r, cyp - arc_r,
                                       cxp + arc_r, cyp + arc_r,
                                       start=start_deg, extent=extent,
                                       style=tk.ARC, outline="#ffd400", width=2)
                acute = acute_angle_deg(a_dir, t_dir)
                self.canvas.create_text(cxp + arc_r + 8, cyp - arc_r - 8,
                                        anchor=tk.W,
                                        text=f"{acute:.2f}°",
                                        fill="#ffd400",
                                        font=("", 12, "bold"))

        colors = ("#9af09a", "#9af09a", "#5ec8ff")
        names = ("A", "B", "N")
        for i, (vx, vy) in enumerate(self.points):
            cxp, cyp = to_canvas(vx, vy)
            color = colors[i]
            r = 6
            self.canvas.create_oval(cxp - r, cyp - r, cxp + r, cyp + r,
                                    outline=color, width=2)
            self.canvas.create_line(cxp - 10, cyp, cxp + 10, cyp,
                                    fill=color, width=2)
            self.canvas.create_line(cxp, cyp - 10, cxp, cyp + 10,
                                    fill=color, width=2)
            self.canvas.create_text(cxp + 10, cyp - 12, anchor=tk.W,
                                    text=names[i], fill=color,
                                    font=("", 10, "bold"))

        if len(self.points) == 3:
            wx, wy = self.points[2]
            wh = int(self.win_var.get())
            r1 = to_canvas(wx - wh, wy - wh)
            r2 = to_canvas(wx + wh, wy + wh)
            self.canvas.create_rectangle(*r1, *r2,
                                         outline="#5ec8ff", dash=(2, 3))
            pr = int(self.pca_var.get()) * scale
            cxp, cyp = to_canvas(wx, wy)
            self.canvas.create_oval(cxp - pr, cyp - pr, cxp + pr, cyp + pr,
                                    outline="#5ec8ff", dash=(2, 3))

    def _canvas_to_video(self, cx: int, cy: int) -> tuple[float, float]:
        ox, oy = self.disp_offset
        s = self.disp_scale
        return (cx - ox) / s, (cy - oy) / s


def main() -> None:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("aqua")
    except tk.TclError:
        pass
    ManualAngleGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
