"""Interactive Tk viewer for wire-bend videos.

Either load an already-analyzed pair (annotated MP4 + CSV) or upload a raw
video — the GUI will run the analyzer on it and load the result.

Usage:
    python gui_viewer.py                       # opens with "Open video..."
    python gui_viewer.py --video raw.mp4       # uploads a raw video → analyze
    python gui_viewer.py --analyzed a.mp4 --csv a.csv   # load existing output
"""

from __future__ import annotations

import argparse
import csv
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

import analyzer

MAX_VIDEO_WIDTH = 900


class BendViewer:
    """Tk GUI to scrub a wire-bend video alongside its bend-angle plot."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Wire Bend Viewer")
        self.root.configure(bg="#1e1e1e")
        plt.style.use("dark_background")

        # State (filled when a video is loaded)
        self.cap: Optional[cv2.VideoCapture] = None
        self.total_frames: int = 0
        self.fps: float = 30.0
        self.frame_w: int = 0
        self.frame_h: int = 0
        self.display_scale: float = 1.0
        self.display_w: int = 640
        self.display_h: int = 360
        self.rows: list[dict[str, str]] = []
        self.times: np.ndarray = np.zeros(0)
        self.angles: np.ndarray = np.zeros(0)

        # Playback state
        self.current_frame: int = 0
        self.is_playing: bool = False
        self.speed: float = 1.0
        self._after_id: Optional[str] = None
        self._photo: Optional[ImageTk.PhotoImage] = None

        # Build UI shell
        self._build_top_bar()
        self._build_plot()
        self._build_video_panel()
        self._build_controls()

        self._set_loaded(False)

    # ---------------------------------------------------------------- UI ---
    def _build_top_bar(self) -> None:
        bar = tk.Frame(self.root, bg="#252525")
        bar.pack(side=tk.TOP, fill=tk.X)
        btn_kw = dict(bg="#3a3a3a", fg="#f0f0f0", activebackground="#4a4a4a",
                      activeforeground="#ffffff", bd=0, padx=14, pady=6,
                      font=("Helvetica", 11))
        tk.Button(bar, text="Open video... (analyze)",
                  command=self._on_open_raw, **btn_kw).pack(side=tk.LEFT, padx=6, pady=6)
        tk.Button(bar, text="Open analyzed pair...",
                  command=self._on_open_pair, **btn_kw).pack(side=tk.LEFT, padx=6, pady=6)
        self.status_var = tk.StringVar(value="no video loaded")
        tk.Label(bar, textvariable=self.status_var, fg="#cfcfcf", bg="#252525",
                 font=("Helvetica", 10)).pack(side=tk.LEFT, padx=14)

    def _build_plot(self) -> None:
        self.figure: Figure = Figure(figsize=(9, 2.6), dpi=100, facecolor="#1e1e1e")
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor("#121212")
        self.ax.set_xlabel("time (s)")
        self.ax.set_ylabel("bend angle (deg)")
        self.ax.set_title("Bend angle vs time")
        (self._curve,) = self.ax.plot([], [], color="#7fb3ff", alpha=0.5, linewidth=1.2)
        self._cursor_line = self.ax.axvline(0.0, color="#ff9f43", linewidth=1.4, alpha=0.9)
        (self._cursor_point,) = self.ax.plot([], [], marker="o",
                                             color="#ff9f43", markersize=7, linestyle="None")
        self.ax.grid(True, alpha=0.2)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.X, padx=8, pady=(8, 4))

    def _build_video_panel(self) -> None:
        panel = tk.Frame(self.root, bg="#1e1e1e")
        panel.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)
        self.title_var = tk.StringVar(value="(no video)")
        tk.Label(panel, textvariable=self.title_var, font=("Helvetica", 12, "bold"),
                 fg="#f0f0f0", bg="#1e1e1e").pack(side=tk.TOP, anchor=tk.W, pady=(0, 4))
        self.video_label = tk.Label(panel, bg="#000000", bd=0,
                                    width=80, height=20)
        self.video_label.pack(side=tk.TOP)

    def _build_controls(self) -> None:
        controls = tk.Frame(self.root, bg="#1e1e1e")
        controls.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(4, 8))

        self.scrub_var = tk.IntVar(value=0)
        self.scrub_scale = tk.Scale(controls, from_=0, to=0, orient=tk.HORIZONTAL,
                                    variable=self.scrub_var, command=self._on_scrub,
                                    bg="#1e1e1e", fg="#f0f0f0", troughcolor="#333",
                                    highlightthickness=0, label="frame")
        self.scrub_scale.pack(side=tk.TOP, fill=tk.X)

        btn_row = tk.Frame(controls, bg="#1e1e1e")
        btn_row.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))
        btn_kw = dict(bg="#2d2d2d", fg="#f0f0f0", activebackground="#3d3d3d",
                      activeforeground="#ffffff", bd=0, padx=12, pady=4,
                      font=("Helvetica", 12))
        self.play_btn = tk.Button(btn_row, text="\u25B6 Play", command=self.play, **btn_kw)
        self.play_btn.pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="\u23F8 Pause", command=self.pause, **btn_kw
                  ).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="\u23EE Reset", command=self.reset, **btn_kw
                  ).pack(side=tk.LEFT, padx=2)

        self.speed_var = tk.DoubleVar(value=1.0)
        self.speed_label_var = tk.StringVar(value="speed: 1.00x")
        tk.Label(btn_row, textvariable=self.speed_label_var, fg="#f0f0f0",
                 bg="#1e1e1e", font=("Helvetica", 11)
                 ).pack(side=tk.LEFT, padx=(20, 6))
        tk.Scale(btn_row, from_=0.25, to=4.0, resolution=0.05, orient=tk.HORIZONTAL,
                 variable=self.speed_var, command=self._on_speed_change, length=240,
                 bg="#1e1e1e", fg="#f0f0f0", troughcolor="#333",
                 highlightthickness=0, showvalue=False).pack(side=tk.LEFT)

    # ------------------------------------------------------ file pickers --
    def _on_open_raw(self) -> None:
        path = filedialog.askopenfilename(
            title="Select a video to analyze",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v"),
                       ("All files", "*.*")],
        )
        if not path:
            return
        self._run_analysis(Path(path))

    def _on_open_pair(self) -> None:
        video = filedialog.askopenfilename(
            title="Select annotated video",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v"),
                       ("All files", "*.*")],
        )
        if not video:
            return
        csv_path = filedialog.askopenfilename(
            title="Select matching CSV",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not csv_path:
            return
        try:
            self._load_pair(Path(video), Path(csv_path))
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))

    # ----------------------------------------------------- analysis flow --
    def _run_analysis(self, raw_video: Path) -> None:
        """Run analyzer.process_video on a raw upload in a background thread.

        A modal progress dialog polls a queue for status; when done we load
        the produced (annotated mp4, csv) pair into the viewer."""
        out_dir = raw_video.parent
        stem = raw_video.stem
        out_csv = out_dir / f"{stem}_bend.csv"
        out_video = out_dir / f"{stem}_annotated.mp4"

        msg_q: queue.Queue = queue.Queue()

        def worker() -> None:
            try:
                analyzer.process_video(
                    in_path=raw_video, out_csv=out_csv, out_video=out_video,
                    show_progress=False,
                )
                msg_q.put(("done", str(out_video), str(out_csv)))
            except Exception as exc:
                msg_q.put(("err", str(exc)))

        # progress dialog
        dlg = tk.Toplevel(self.root)
        dlg.title("Analyzing video...")
        dlg.configure(bg="#1e1e1e")
        dlg.transient(self.root)
        dlg.grab_set()
        tk.Label(dlg, text=f"Analyzing:\n{raw_video.name}",
                 fg="#f0f0f0", bg="#1e1e1e", font=("Helvetica", 11),
                 padx=20, pady=10).pack()
        prog_var = tk.StringVar(value="Running detector on every frame... please wait.")
        tk.Label(dlg, textvariable=prog_var, fg="#cfcfcf", bg="#1e1e1e",
                 font=("Helvetica", 10), padx=20, pady=4).pack()
        spin_var = tk.StringVar(value="")
        tk.Label(dlg, textvariable=spin_var, fg="#7fb3ff", bg="#1e1e1e",
                 font=("Helvetica", 14), padx=20).pack(pady=(0, 14))

        spin_chars = "|/-\\"
        spin_idx = [0]

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        def poll() -> None:
            spin_idx[0] = (spin_idx[0] + 1) % len(spin_chars)
            spin_var.set(spin_chars[spin_idx[0]])
            try:
                msg = msg_q.get_nowait()
            except queue.Empty:
                dlg.after(100, poll)
                return
            dlg.grab_release()
            dlg.destroy()
            if msg[0] == "done":
                try:
                    self._load_pair(Path(msg[1]), Path(msg[2]))
                except Exception as exc:
                    messagebox.showerror("Load failed", str(exc))
            else:
                messagebox.showerror("Analysis failed", msg[1])

        dlg.after(100, poll)

    # ----------------------------------------------------------- loading --
    def _load_pair(self, video_path: Path, csv_path: Path) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        self.cap = cap
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.frame_w > MAX_VIDEO_WIDTH:
            self.display_scale = MAX_VIDEO_WIDTH / float(self.frame_w)
        else:
            self.display_scale = 1.0
        self.display_w = int(round(self.frame_w * self.display_scale))
        self.display_h = int(round(self.frame_h * self.display_scale))

        self.rows = self._load_csv(csv_path)
        self.times, self.angles = self._extract_series(self.rows)

        # refresh plot curve
        finite = np.isfinite(self.times) & np.isfinite(self.angles)
        self._curve.set_data(self.times[finite], self.angles[finite])
        self.ax.relim(); self.ax.autoscale_view()
        self.canvas.draw_idle()

        self.scrub_scale.configure(to=max(self.total_frames - 1, 0),
                                   length=self.display_w)
        self.status_var.set(
            f"loaded {video_path.name}  |  {csv_path.name}  |  "
            f"{self.total_frames} frames @ {self.fps:.2f} fps"
        )
        self._set_loaded(True)
        self._goto_frame(0)

    @staticmethod
    def _load_csv(csv_path: Path) -> list[dict[str, str]]:
        with csv_path.open("r", newline="") as fp:
            return list(csv.DictReader(fp))

    @staticmethod
    def _extract_series(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        times: list[float] = []
        angles: list[float] = []
        for r in rows:
            try:
                t = float(r.get("time_s", "") or "nan")
            except ValueError:
                t = float("nan")
            a_str = r.get("angle_deg", "")
            try:
                a = float(a_str) if a_str not in ("", None) else float("nan")
            except ValueError:
                a = float("nan")
            times.append(t); angles.append(a)
        return np.asarray(times, dtype=float), np.asarray(angles, dtype=float)

    def _set_loaded(self, loaded: bool) -> None:
        state = tk.NORMAL if loaded else tk.DISABLED
        for w in (self.play_btn, self.scrub_scale):
            w.configure(state=state)
        if not loaded:
            self.title_var.set("(no video — click Open video... to begin)")

    # ----------------------------------------------------------- control --
    def _on_scrub(self, _value: str) -> None:
        if self.cap is None:
            return
        idx = int(self.scrub_var.get())
        if idx != self.current_frame:
            self._goto_frame(idx)

    def _on_speed_change(self, _value: str) -> None:
        self.speed = float(self.speed_var.get())
        self.speed_label_var.set(f"speed: {self.speed:.2f}x")

    def play(self) -> None:
        if self.cap is None or self.is_playing:
            return
        if self.current_frame >= self.total_frames - 1:
            self._goto_frame(0)
        self.is_playing = True
        self._schedule_tick()

    def pause(self) -> None:
        self.is_playing = False
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def reset(self) -> None:
        self.pause()
        if self.cap is not None:
            self._goto_frame(0)

    def _schedule_tick(self) -> None:
        interval_ms = max(int(round(1000.0 / max(self.fps * self.speed, 0.1))), 1)
        self._after_id = self.root.after(interval_ms, self._tick)

    def _tick(self) -> None:
        if not self.is_playing or self.cap is None:
            return
        next_idx = self.current_frame + 1
        if next_idx >= self.total_frames:
            self._goto_frame(self.total_frames - 1)
            self.pause()
            return
        self._goto_frame(next_idx)
        self._schedule_tick()

    # ---------------------------------------------------------- rendering --
    def _goto_frame(self, idx: int) -> None:
        if self.cap is None:
            return
        idx = max(0, min(int(idx), max(self.total_frames - 1, 0)))
        self.current_frame = idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if ok and frame is not None:
            self._render_frame(frame)
        if int(self.scrub_var.get()) != idx:
            self.scrub_var.set(idx)
        self._update_plot_cursor(idx)
        self._update_title(idx)

    def _render_frame(self, frame_bgr: np.ndarray) -> None:
        if self.display_scale != 1.0:
            frame_bgr = cv2.resize(frame_bgr, (self.display_w, self.display_h),
                                   interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.video_label.configure(image=self._photo, width=self.display_w,
                                   height=self.display_h)

    def _update_plot_cursor(self, idx: int) -> None:
        if idx >= len(self.times):
            return
        t = float(self.times[idx]) if np.isfinite(self.times[idx]) else float(idx) / self.fps
        a = float(self.angles[idx]) if idx < len(self.angles) and np.isfinite(self.angles[idx]) else None
        self._cursor_line.set_xdata([t, t])
        if a is None:
            self._cursor_point.set_data([], [])
        else:
            self._cursor_point.set_data([t], [a])
        self.canvas.draw_idle()

    def _update_title(self, idx: int) -> None:
        if idx >= len(self.rows):
            self.title_var.set(f"t = ?    frame = {idx}    angle = n/a"); return
        row = self.rows[idx]
        try:
            t = float(row.get("time_s", "") or "nan")
        except ValueError:
            t = float("nan")
        angle_str = row.get("angle_deg", "")
        if angle_str in ("", None):
            angle_display = "n/a"
        else:
            try:
                angle_display = f"{float(angle_str):.2f}\u00B0"
            except ValueError:
                angle_display = "n/a"
        t_display = f"{t:.2f} s" if np.isfinite(t) else "?"
        self.title_var.set(f"t = {t_display}    frame = {idx}    angle = {angle_display}")

    # ----------------------------------------------------------- shutdown --
    def close(self) -> None:
        self.pause()
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wire-bend video viewer.")
    parser.add_argument("--video", type=Path,
                        help="Raw video to analyze on launch.")
    parser.add_argument("--analyzed", type=Path,
                        help="Pre-analyzed annotated video (skip analysis).")
    parser.add_argument("--csv", type=Path,
                        help="CSV matching --analyzed.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    root = tk.Tk()
    viewer = BendViewer(root)

    if args.analyzed and args.csv:
        try:
            viewer._load_pair(args.analyzed, args.csv)
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
    elif args.video:
        root.after(200, lambda: viewer._run_analysis(args.video))

    def _on_close() -> None:
        viewer.close()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
