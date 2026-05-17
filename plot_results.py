"""Publication-quality multi-panel report from analyzer.py CSV output."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


COLUMNS = [
    "frame", "time_s", "angle_deg",
    "tip_x", "tip_y", "base_x", "base_y",
    "base_dir_deg", "tip_dir_deg", "n_skeleton",
    "mid_x", "mid_y", "needle_len",
]


def load_csv(path: Path) -> dict[str, np.ndarray]:
    """Load analyzer CSV as a dict of numpy arrays. Empty cells become NaN
    (except for the integer columns frame/n_skeleton, which fall back to 0)."""
    cols: dict[str, list] = {c: [] for c in COLUMNS}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for c in COLUMNS:
                v = row.get(c, "")
                if v == "" or v is None:
                    cols[c].append(np.nan)
                else:
                    cols[c].append(float(v))
    out: dict[str, np.ndarray] = {}
    for c in COLUMNS:
        arr = np.asarray(cols[c], dtype=np.float64)
        out[c] = arr
    return out


def clamp_window(window: int, n: int) -> int | None:
    """Coerce window to an odd int <= n and >= 5. Returns None if not viable."""
    if n < 5:
        return None
    w = int(window)
    if w > n:
        w = n
    if w % 2 == 0:
        w -= 1
    if w < 5:
        return None
    return w


def smooth_angle(t: np.ndarray, a: np.ndarray, window: int) -> np.ndarray | None:
    w = clamp_window(window, len(a))
    if w is None or w <= 3:
        return None
    return savgol_filter(a, window_length=w, polyorder=3)


def bend_rate(t: np.ndarray, a_smooth: np.ndarray) -> np.ndarray:
    return np.gradient(a_smooth, t)


def make_report(
    data: dict[str, np.ndarray],
    csv_name: str,
    out_png: Path,
    smooth_window: int,
    fps_fallback: float,
) -> dict:
    frame = data["frame"]
    time_s = data["time_s"]
    angle = data["angle_deg"]
    tip_x = data["tip_x"]
    tip_y = data["tip_y"]

    # If time is all NaN, reconstruct from frame index using fps fallback.
    if np.all(np.isnan(time_s)):
        time_s = frame / fps_fallback

    mask = ~np.isnan(angle)
    t_det = time_s[mask]
    a_det = angle[mask]
    f_det = frame[mask]

    n_frames = len(frame)
    n_with = int(mask.sum())
    det_rate = 100.0 * n_with / max(n_frames, 1)

    a_smooth = smooth_angle(t_det, a_det, smooth_window) if n_with >= 5 else None

    if n_with > 0:
        i_max = int(np.argmax(a_det))
        max_ang = float(a_det[i_max])
        max_t = float(t_det[i_max])
        max_frame = int(f_det[i_max])
        mean_ang = float(np.mean(a_det))
        median_ang = float(np.median(a_det))
    else:
        max_ang = max_t = mean_ang = median_ang = float("nan")
        max_frame = -1

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_tl, ax_tr = axes[0]
    ax_bl, ax_br = axes[1]

    # Top-left: raw + smoothed angle
    ax_tl.plot(t_det, a_det, color="0.7", lw=1.0, label="raw")
    if a_smooth is not None:
        ax_tl.plot(t_det, a_smooth, color="C0", lw=2.0, label="smoothed")
    ax_tl.set_xlabel("Time (s)")
    ax_tl.set_ylabel("Bend angle (degrees)")
    ax_tl.set_title("Bend angle over time")
    ax_tl.grid(alpha=0.3)
    ax_tl.legend(loc="best")

    # Top-right: bend rate
    if a_smooth is not None and len(t_det) >= 2:
        rate = bend_rate(t_det, a_smooth)
        ax_tr.plot(t_det, rate, color="C3", lw=1.5)
    ax_tr.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax_tr.set_xlabel("Time (s)")
    ax_tr.set_ylabel("Bend rate (deg/s)")
    ax_tr.set_title("Bend rate (d angle / d t)")
    ax_tr.grid(alpha=0.3)

    # Bottom-left: needle-midpoint trajectory colored by time (falls back to
    # the tip column if mid_x/mid_y are missing — older CSVs).
    mx = data.get("mid_x", np.full_like(tip_x, np.nan))
    my = data.get("mid_y", np.full_like(tip_y, np.nan))
    if np.all(np.isnan(mx)):
        mx, my, label_xy = tip_x, tip_y, ("tip_x (px)", "tip_y (px)")
        title = "Tip trajectory"
    else:
        label_xy = ("mid_x (px)", "mid_y (px)")
        title = "Needle midpoint trajectory"
    mid_mask = ~(np.isnan(mx) | np.isnan(my))
    if mid_mask.any():
        sc = ax_bl.scatter(
            mx[mid_mask], my[mid_mask],
            c=time_s[mid_mask], cmap="viridis", s=12,
        )
        cbar = fig.colorbar(sc, ax=ax_bl)
        cbar.set_label("Time (s)")
    ax_bl.set_xlabel(label_xy[0])
    ax_bl.set_ylabel(label_xy[1])
    ax_bl.set_title(title)
    ax_bl.invert_yaxis()
    ax_bl.set_aspect("equal", adjustable="datalim")
    ax_bl.grid(alpha=0.3)

    # Bottom-right: angle histogram
    if n_with > 0:
        ax_br.hist(a_det, bins=40, color="C0", alpha=0.7, edgecolor="white")
        ax_br.axvline(mean_ang, color="C1", lw=2, label=f"mean = {mean_ang:.2f}")
        ax_br.axvline(median_ang, color="C2", lw=2, ls="--",
                      label=f"median = {median_ang:.2f}")
        ax_br.axvline(max_ang, color="C3", lw=2, ls=":",
                      label=f"max = {max_ang:.2f}")
        ax_br.legend(loc="best", fontsize=9)
        ax_br.text(
            0.98, 0.98,
            f"max {max_ang:.2f} deg\nat t = {max_t:.2f} s (frame {max_frame})",
            transform=ax_br.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.8),
        )
    ax_br.set_xlabel("Bend angle (degrees)")
    ax_br.set_ylabel("Count")
    ax_br.set_title("Angle distribution")
    ax_br.grid(alpha=0.3)

    fig.suptitle(csv_name, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    return {
        "n_frames": n_frames,
        "n_with": n_with,
        "det_rate": det_rate,
        "max_ang": max_ang,
        "max_t": max_t,
        "max_frame": max_frame,
        "mean_ang": mean_ang,
        "median_ang": median_ang,
    }


def print_summary(s: dict) -> None:
    print(f"N frames: {s['n_frames']}")
    print(f"N with angle: {s['n_with']}")
    print(f"Detection rate: {s['det_rate']:.1f}%")
    if s["n_with"] > 0:
        print(
            f"Max bend: {s['max_ang']:.3f} deg at t={s['max_t']:.3f}s "
            f"(frame {s['max_frame']})"
        )
        print(f"Mean bend: {s['mean_ang']:.3f}")
        print(f"Median bend: {s['median_ang']:.3f}")
    else:
        print("Max bend: n/a")
        print("Mean bend: n/a")
        print("Median bend: n/a")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot analyzer.py bend CSV.")
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out-png", type=Path, default=Path("report.png"))
    ap.add_argument("--smooth-window", type=int, default=15)
    ap.add_argument("--fps-fallback", type=float, default=30.0)
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"error: csv not found: {args.csv}", file=sys.stderr)
        raise SystemExit(2)

    data = load_csv(args.csv)
    stats = make_report(
        data=data,
        csv_name=args.csv.name,
        out_png=args.out_png,
        smooth_window=args.smooth_window,
        fps_fallback=args.fps_fallback,
    )
    print_summary(stats)


if __name__ == "__main__":
    main()
