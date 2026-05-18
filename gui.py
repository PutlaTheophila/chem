"""Tkinter GUI for the wire-bend analyzer.

Workflow:
  1. Open a video.
  2. Click three points on the displayed first frame:
       FP  -> fixed reference point
       axis -> a second point along the reference axis
       track -> the point on the moving structure to track
  3. Press "Run Analysis". The tracker walks every frame, writes
     bend.csv and bend_annotated.mp4 next to the video, and shows
     the angle-vs-time plot. Use the slider to scrub through the
     annotated frames.
"""

from __future__ import annotations

import csv
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

import analyzer


POINT_LABELS = ("axis A", "axis B", "FP (on axis)", "track")
POINT_COLORS = ("#9af09a", "#9af09a", "#ffd400", "#5ec8ff")
N_POINTS = len(POINT_LABELS)


class WireBendGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Wire Bend Analyzer")
        root.geometry("1180x740")

        self.video_path: Path | None = None
        self.first_frame_bgr: np.ndarray | None = None
        self.frame_w = 0
        self.frame_h = 0
        self.disp_scale = 1.0  # video px -> canvas px
        self.disp_offset = (0, 0)
        self.points_video: list[tuple[float, float]] = []  # in video coords
        self.angles: np.ndarray | None = None
        self.times: np.ndarray | None = None
        self.annot_cap: cv2.VideoCapture | None = None
        self.annot_n = 0
        self.csv_path: Path | None = None
        self.video_out_path: Path | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._worker: threading.Thread | None = None

        self._build_ui()

    # ----------------------------------------------------------- ui ---
    def _build_ui(self) -> None:
        tb = ttk.Frame(self.root, padding=4)
        tb.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(tb, text="Open Video…", command=self.open_video).pack(side=tk.LEFT)
        ttk.Button(tb, text="Reset Points", command=self.reset_points).pack(side=tk.LEFT, padx=(6, 0))
        self.run_btn = ttk.Button(tb, text="Run Analysis", command=self.run_analysis, state=tk.DISABLED)
        self.run_btn.pack(side=tk.LEFT, padx=(6, 0))
        self.plot_btn = ttk.Button(tb, text="Show Plot", command=self.show_plot, state=tk.DISABLED)
        self.plot_btn.pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(tb, text="Template half:").pack(side=tk.LEFT, padx=(16, 2))
        self.tmpl_var = tk.IntVar(value=30)
        ttk.Spinbox(tb, from_=8, to=80, textvariable=self.tmpl_var, width=4).pack(side=tk.LEFT)
        ttk.Label(tb, text="Search half:").pack(side=tk.LEFT, padx=(10, 2))
        self.search_var = tk.IntVar(value=120)
        ttk.Spinbox(tb, from_=20, to=400, textvariable=self.search_var, width=4).pack(side=tk.LEFT)
        # OFF by default: the user is picking a specific point (often the
        # wire tip just outside the ball), and Hough-circle tracking would
        # otherwise drag the result back to the ball centre every frame.
        self.snap_ball_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tb, text="Snap track→ball", variable=self.snap_ball_var).pack(
            side=tk.LEFT, padx=(10, 0)
        )

        body = ttk.Frame(self.root)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(body, bg="#202020", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Configure>", lambda e: self._refresh_canvas())

        right = ttk.Frame(body, padding=8)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Label(right, text="Points", font=("", 11, "bold")).pack(anchor="w")
        self.points_lbls = []
        for name, color in zip(POINT_LABELS, POINT_COLORS):
            lbl = ttk.Label(right, text=f"{name}: —", foreground=color)
            lbl.pack(anchor="w")
            self.points_lbls.append(lbl)
        ttk.Separator(right).pack(fill=tk.X, pady=6)
        ttk.Label(right, text="Results", font=("", 11, "bold")).pack(anchor="w")
        self.result_lbl = ttk.Label(right, text="—", justify="left")
        self.result_lbl.pack(anchor="w")

        bottom = ttk.Frame(self.root, padding=(6, 2))
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        self.scrub = ttk.Scale(bottom, from_=0, to=0, orient=tk.HORIZONTAL,
                               command=self._on_scrub)
        self.scrub.pack(side=tk.TOP, fill=tk.X)
        self.scrub.state(["disabled"])
        self.status_var = tk.StringVar(value="Open a video to begin.")
        ttk.Label(bottom, textvariable=self.status_var).pack(side=tk.LEFT)
        self.progress = ttk.Progressbar(bottom, length=240, mode="determinate")
        self.progress.pack(side=tk.RIGHT)

    # ------------------------------------------------------- actions ---
    def open_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Open video",
            filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv *.webm"),
                       ("All", "*.*")],
        )
        if not path:
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Open video", f"Cannot open {path}")
            return
        ok, frame = cap.read()
        cap.release()
        if not ok:
            messagebox.showerror("Open video", "Cannot read first frame.")
            return
        self.video_path = Path(path)
        self.first_frame_bgr = frame
        self.frame_h, self.frame_w = frame.shape[:2]
        self.reset_points()
        self._close_annot()
        self.scrub.configure(to=0)
        self.scrub.state(["disabled"])
        self.plot_btn.state(["disabled"])
        self.result_lbl.config(text="—")
        self.status_var.set(f"Loaded {self.video_path.name}  ({self.frame_w}x{self.frame_h}). "
                            f"Click FP, then axis end, then the point to track.")
        self._refresh_canvas()

    def reset_points(self) -> None:
        # Drop any playback so the canvas reverts to the raw first frame
        # and the user can re-pick from scratch.
        self._close_annot()
        self.scrub.configure(to=0)
        self.scrub.state(["disabled"])
        self.angles = None
        self.times = None
        self.points_video = []
        for i, lbl in enumerate(self.points_lbls):
            lbl.config(text=f"{POINT_LABELS[i]}: —")
        self.run_btn.state(["disabled"])
        self.plot_btn.state(["disabled"])
        self.result_lbl.config(text="—")
        if self.first_frame_bgr is not None:
            self.status_var.set("Click axis A, then axis B, then FP, then the point to track.")
        self._refresh_canvas()

    def on_canvas_click(self, event) -> None:
        if self.first_frame_bgr is None or self._worker is not None:
            return
        if len(self.points_video) >= N_POINTS:
            return
        vx, vy = self._canvas_to_video(event.x, event.y)
        if vx < 0 or vx >= self.frame_w or vy < 0 or vy >= self.frame_h:
            return
        # FP (3rd click) is snapped onto the axis line A-B so it sits
        # exactly on the user-chosen axis.
        if len(self.points_video) == 2:
            a = np.array(self.points_video[0], dtype=np.float64)
            b = np.array(self.points_video[1], dtype=np.float64)
            d = b - a
            nrm2 = float(np.dot(d, d))
            if nrm2 < 1e-6:
                messagebox.showerror("Axis", "Axis A and B coincide. Reset and re-pick.")
                return
            t = float(np.dot(np.array([vx, vy]) - a, d)) / nrm2
            snap = a + t * d
            vx, vy = float(snap[0]), float(snap[1])
        # Track (4th click): optionally snap to the nearest detected circle
        # (the ball). Template tracking then locks onto the ball, not an
        # arbitrary wire edge near the click.
        if len(self.points_video) == 3 and self.snap_ball_var.get():
            snapped = self._snap_to_ball(np.array([vx, vy], dtype=np.float64))
            if snapped is not None:
                vx, vy = float(snapped[0]), float(snapped[1])
        self.points_video.append((vx, vy))
        i = len(self.points_video) - 1
        self.points_lbls[i].config(text=f"{POINT_LABELS[i]}: ({vx:.0f}, {vy:.0f})")
        if len(self.points_video) == N_POINTS:
            self.run_btn.state(["!disabled"])
            self.status_var.set("Ready. Press Run Analysis.")
        else:
            nxt = POINT_LABELS[len(self.points_video)]
            self.status_var.set(f"Click {nxt}.")
        self._refresh_canvas()

    def run_analysis(self) -> None:
        if self.video_path is None or len(self.points_video) != N_POINTS:
            return
        if self._worker is not None:
            return
        a = np.array(self.points_video[0], dtype=np.float64)
        b = np.array(self.points_video[1], dtype=np.float64)
        fp = np.array(self.points_video[2], dtype=np.float64)
        tr = np.array(self.points_video[3], dtype=np.float64)
        # Base direction = from FP toward the LEFT (smaller-x) axis endpoint.
        # This is the 0° reference; the bend angle is measured from here
        # rotating to the FP→track vector.
        left_end = a if a[0] < b[0] else b
        v = left_end - fp
        nrm = float(np.linalg.norm(v))
        if nrm < 1.0:
            messagebox.showerror(
                "Axis", "FP sits at the left end of the axis — extend the axis further.",
            )
            return
        axis_dir = v / nrm
        out_dir = self.video_path.parent
        self.csv_path = out_dir / "bend.csv"
        self.video_out_path = out_dir / "bend_annotated.mp4"

        self.run_btn.state(["disabled"])
        self.progress.configure(value=0, maximum=100)
        self.status_var.set("Analyzing…")

        def work():
            try:
                self._run_in_worker(fp, axis_dir, tr)
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror("Analysis", str(exc)))
            finally:
                self.root.after(0, self._on_analysis_done)

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()

    def _run_in_worker(self, fp, axis_dir, tracked_pt_init) -> None:
        in_path = self.video_path
        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {in_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.video_out_path), fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"cannot open writer for {self.video_out_path}")

        template_half = int(self.tmpl_var.get())
        search_half = int(self.search_var.get())
        use_hough = bool(self.snap_ball_var.get())
        template_patch = None
        last_xy = tracked_pt_init.copy()
        rows: list[tuple[int, float, float | None, float | None, float | None]] = []
        last_log = time.time()
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t_s = idx / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if template_patch is None:
                ex = analyzer.template_extract(gray, tracked_pt_init, template_half)
                if ex is not None:
                    template_patch = ex[0]
            # Track the exact user-chosen point via NCC template matching
            # with adaptive refresh. Only fall back to Hough-circle ball
            # detection when the user explicitly asked to snap to the ball
            # (otherwise the tracker would always drag to the ball centre
            # even when the user picked the wire tip just outside the ball).
            tip = None
            if use_hough:
                res = analyzer.track_step_robust(
                    gray, template_patch, last_xy, search_half=search_half,
                )
                if res is not None:
                    tip, _, score = res
                    last_xy = tip.copy()
                    ex = analyzer.template_extract(gray, tip, template_half)
                    if ex is not None:
                        template_patch = ex[0]
            elif template_patch is not None:
                step = analyzer.template_track_step(
                    gray, template_patch, last_xy, search_half,
                )
                if step is not None:
                    tip, score = step
                    last_xy = tip.copy()
                    # Refresh template on high-confidence matches so the
                    # tracker adapts to slow appearance changes (rotation,
                    # lighting) as the wire bends.
                    if score > 0.85:
                        ex = analyzer.template_extract(gray, tip, template_half)
                        if ex is not None:
                            template_patch = ex[0]
            angle = None
            if tip is not None:
                angle = analyzer.polar_angle_from_fixed(fp, tip, axis_dir)
            rows.append((idx, t_s,
                         None if tip is None else float(tip[0]),
                         None if tip is None else float(tip[1]),
                         angle))
            measurement = None
            if tip is not None:
                measurement = {
                    "angle_deg": angle, "fixed_pt": fp, "axis_dir": axis_dir,
                    "tip_pt": tip, "tip_pts": np.empty((0, 2), dtype=np.float64),
                }
            writer.write(analyzer.annotate_frame(
                frame, None, measurement, t_s, idx, fp, axis_dir,
            ))
            idx += 1
            if time.time() - last_log > 0.1:
                pct = 100.0 * idx / max(n, 1)
                self.root.after(0, lambda p=pct, i=idx, total=n:
                                self._update_progress(p, i, total))
                last_log = time.time()
        cap.release()
        writer.release()

        with open(self.csv_path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["frame", "time_s", "angle_deg", "tip_x", "tip_y",
                         "base_x", "base_y", "base_dir_deg", "tip_dir_deg",
                         "n_skeleton", "mid_x", "mid_y", "needle_len"])
            for (fi, t_s, tx, ty, ang) in rows:
                wr.writerow([
                    fi, f"{t_s:.4f}",
                    "" if ang is None else f"{ang:.3f}",
                    "" if tx is None else f"{tx:.2f}",
                    "" if ty is None else f"{ty:.2f}",
                    f"{fp[0]:.2f}", f"{fp[1]:.2f}",
                    "", "", 0, "", "", "",
                ])

        self._results_rows = rows
        self._results_fps = fps

    def _update_progress(self, pct: float, done: int, total: int) -> None:
        self.progress.configure(value=pct)
        self.status_var.set(f"Analyzing… {done}/{total}  ({pct:.1f}%)")

    def _on_analysis_done(self) -> None:
        self._worker = None
        if not hasattr(self, "_results_rows"):
            self.status_var.set("Analysis failed.")
            self.run_btn.state(["!disabled"])
            return
        rows = self._results_rows
        ang = [r[4] for r in rows if r[4] is not None]
        ts = np.array([r[1] for r in rows], dtype=np.float64)
        angles = np.array([np.nan if r[4] is None else r[4] for r in rows], dtype=np.float64)
        self.times = ts
        self.angles = angles
        n_with = len(ang)
        n_total = len(rows)
        max_a = max(ang) if ang else float("nan")
        mean_a = float(np.nanmean(angles)) if n_with else float("nan")
        self.result_lbl.config(
            text=(f"frames     {n_total}\n"
                  f"detections {n_with} ({100*n_with/max(n_total,1):.1f}%)\n"
                  f"max angle  {max_a:+.2f}°\n"
                  f"mean       {mean_a:+.2f}°\n"
                  f"CSV  {self.csv_path.name}\n"
                  f"MP4  {self.video_out_path.name}"))
        self.status_var.set("Done. Scrub the annotated video below, or open the plot.")
        self.progress.configure(value=100)
        self._open_annot()
        self.run_btn.state(["!disabled"])
        self.plot_btn.state(["!disabled"])

    # ------------------------------------------------------ display ---
    def _open_annot(self) -> None:
        self._close_annot()
        if self.video_out_path is None or not self.video_out_path.exists():
            return
        cap = cv2.VideoCapture(str(self.video_out_path))
        if not cap.isOpened():
            return
        self.annot_cap = cap
        self.annot_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.scrub.configure(to=max(self.annot_n - 1, 0))
        self.scrub.set(0)
        self.scrub.state(["!disabled"])
        self._show_annot_frame(0)

    def _close_annot(self) -> None:
        if self.annot_cap is not None:
            self.annot_cap.release()
            self.annot_cap = None

    def _on_scrub(self, value) -> None:
        if self.annot_cap is None:
            return
        try:
            idx = int(float(value))
        except (TypeError, ValueError):
            return
        self._show_annot_frame(idx)

    def _show_annot_frame(self, idx: int) -> None:
        if self.annot_cap is None:
            return
        self.annot_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.annot_cap.read()
        if not ok:
            return
        self._draw_frame(frame)
        ang_txt = ""
        if self.angles is not None and 0 <= idx < len(self.angles):
            a = self.angles[idx]
            t = self.times[idx]
            ang_txt = f"   t={t:.2f}s   angle={a:+.2f}°" if not np.isnan(a) else f"   t={t:.2f}s   no detection"
        self.status_var.set(f"frame {idx}/{self.annot_n-1}{ang_txt}")

    def _refresh_canvas(self) -> None:
        if self.annot_cap is not None:
            return  # post-analysis: managed by scrubber
        if self.first_frame_bgr is None:
            self.canvas.delete("all")
            return
        self._draw_frame(self.first_frame_bgr, draw_pickers=True)

    def _draw_frame(self, frame_bgr: np.ndarray, draw_pickers: bool = False) -> None:
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
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((new_w, new_h), Image.BILINEAR)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(ox, oy, anchor=tk.NW, image=self._photo)

        if draw_pickers:
            def to_canvas(vx, vy):
                return ox + vx * scale, oy + vy * scale

            # Axis: faint dashed line through A and B (entire frame width),
            # then the BASE half (FP → left endpoint) thicker to show which
            # side is the 0° reference.
            if len(self.points_video) >= 2:
                a = np.array(self.points_video[0], dtype=np.float64)
                b = np.array(self.points_video[1], dtype=np.float64)
                d = b - a
                nrm = float(np.linalg.norm(d))
                if nrm > 1e-6:
                    u = d / nrm
                    diag = float(np.hypot(self.frame_w, self.frame_h))
                    p1 = a - u * diag
                    p2 = a + u * diag
                    c1 = to_canvas(*p1)
                    c2 = to_canvas(*p2)
                    self.canvas.create_line(*c1, *c2,
                                            fill=POINT_COLORS[0], dash=(4, 3))
            if len(self.points_video) >= 3:
                a = np.array(self.points_video[0], dtype=np.float64)
                b = np.array(self.points_video[1], dtype=np.float64)
                fp = np.array(self.points_video[2], dtype=np.float64)
                left_end = a if a[0] < b[0] else b
                c1 = to_canvas(*fp)
                c2 = to_canvas(*left_end)
                self.canvas.create_line(*c1, *c2,
                                        fill="#ffffff", width=3)
                self.canvas.create_text(*to_canvas((fp[0] + left_end[0]) / 2,
                                                   (fp[1] + left_end[1]) / 2 - 12),
                                        text="base (0°)", fill="#ffffff",
                                        font=("", 9, "bold"))
            if len(self.points_video) >= 4:
                fp = np.array(self.points_video[2], dtype=np.float64)
                tr = np.array(self.points_video[3], dtype=np.float64)
                c1 = to_canvas(*fp)
                c2 = to_canvas(*tr)
                self.canvas.create_line(*c1, *c2,
                                        fill=POINT_COLORS[3], dash=(6, 4), width=2)

            # Point markers, drawn after lines so they sit on top.
            for i, (vx, vy) in enumerate(self.points_video):
                cx, cy = to_canvas(vx, vy)
                color = POINT_COLORS[i]
                r = 6
                self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                        outline=color, width=2)
                self.canvas.create_line(cx - 10, cy, cx + 10, cy,
                                        fill=color, width=2)
                self.canvas.create_line(cx, cy - 10, cx, cy + 10,
                                        fill=color, width=2)
                self.canvas.create_text(cx + 10, cy - 12, anchor=tk.W,
                                        text=POINT_LABELS[i], fill=color,
                                        font=("", 10, "bold"))

    def _snap_to_ball(self, click_xy: np.ndarray, max_dist: float = 40.0,
                      min_r: int = 12, max_r: int = 60) -> np.ndarray | None:
        """Find Hough circles in the first frame and return the centre of
        the one nearest ``click_xy`` (within ``max_dist`` px), or ``None``."""
        if self.first_frame_bgr is None:
            return None
        gray = cv2.cvtColor(self.first_frame_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 1.5)
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
            param1=80, param2=20, minRadius=min_r, maxRadius=max_r,
        )
        if circles is None:
            return None
        cands = circles[0]
        d = np.hypot(cands[:, 0] - click_xy[0], cands[:, 1] - click_xy[1])
        i = int(np.argmin(d))
        if d[i] > max_dist:
            return None
        return np.array([float(cands[i, 0]), float(cands[i, 1])],
                        dtype=np.float64)

    def _canvas_to_video(self, cx: int, cy: int) -> tuple[float, float]:
        ox, oy = self.disp_offset
        s = self.disp_scale
        return (cx - ox) / s, (cy - oy) / s

    def show_plot(self) -> None:
        if self.angles is None:
            return
        win = tk.Toplevel(self.root)
        win.title("Bend angle over time")
        win.geometry("760x520")
        fig = Figure(figsize=(7.2, 4.8), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(self.times, self.angles, color="C0", lw=1.0)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Bend angle (deg)")
        ax.grid(alpha=0.3)
        ax.set_title(f"{self.video_path.name}")
        fcv = FigureCanvasTkAgg(fig, master=win)
        fcv.draw()
        fcv.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def main() -> None:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("aqua")
    except tk.TclError:
        pass
    WireBendGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
