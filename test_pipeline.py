"""Sanity tests for the wire-bend pipeline.

Run with:
    source venv/bin/activate
    python -m pytest test_pipeline.py -v
    # or, without pytest:
    python test_pipeline.py
"""

from __future__ import annotations

import csv
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

import analyzer
import plot_results


HERE = Path(__file__).parent


# ----------------------------------------------------- synthetic helpers ---
def make_straight_wire_frame(w: int = 400, h: int = 200) -> np.ndarray:
    """A black image with a horizontal bright wire entering from the right."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    y = h // 2
    cv2.line(img, (w - 1, y), (50, y), (255, 255, 255), thickness=3)
    return img


def make_bent_wire_frame(angle_deg: float, w: int = 400, h: int = 300) -> np.ndarray:
    """Bright wire entering from the right, then bending upward by `angle_deg`
    at a fixed pivot. Base is horizontal; tip rotates CCW by angle_deg.

    angle_deg = 0   →  straight horizontal
    angle_deg = 90  →  tip pointing straight up
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    pivot = (w // 2, h // 2)
    # base segment: from right edge to pivot
    cv2.line(img, (w - 1, pivot[1]), pivot, (255, 255, 255), thickness=3)
    # tip segment: from pivot, going in the rotated direction
    length = 80
    theta = math.radians(180.0 - angle_deg)  # 0° → leftward; 90° → upward
    tip = (int(round(pivot[0] + length * math.cos(theta))),
           int(round(pivot[1] - length * math.sin(theta))))
    cv2.line(img, pivot, tip, (255, 255, 255), thickness=3)
    return img


# --------------------------------------------------------------- tests ---
def test_extract_wire_mask_straight():
    frame = make_straight_wire_frame()
    mask = analyzer.extract_wire_mask(frame, threshold=170)
    assert mask is not None
    # mask should touch the right edge
    assert mask[:, -1].any()


def test_measure_bend_straight_is_near_zero():
    frame = make_straight_wire_frame()
    mask = analyzer.extract_wire_mask(frame, threshold=170)
    pts = analyzer.skeleton_points(mask, thin_radius=5.0)
    ordered = analyzer.order_along_wire(pts)
    m = analyzer.measure_bend(ordered, base_frac=0.3, tip_frac=0.15)
    assert m is not None
    assert m["angle_deg"] < 5.0, f"expected near-0, got {m['angle_deg']:.2f}"


def test_measure_bend_synthetic_angles():
    """Synthetic bends at 30/60/90° should recover within a few degrees."""
    for true_angle in (30.0, 60.0, 90.0):
        frame = make_bent_wire_frame(true_angle)
        mask = analyzer.extract_wire_mask(frame, threshold=170)
        pts = analyzer.skeleton_points(mask, thin_radius=5.0)
        ordered = analyzer.order_along_wire(pts)
        m = analyzer.measure_bend(ordered, base_frac=0.3, tip_frac=0.2)
        assert m is not None, f"no measurement at {true_angle}°"
        err = abs(m["angle_deg"] - true_angle)
        assert err < 6.0, (
            f"true {true_angle}° vs measured {m['angle_deg']:.2f}° (err {err:.2f}°)"
        )


def make_local_kink_frame(angle_deg: float, kink_frac: float = 0.88,
                          w: int = 600, h: int = 300) -> np.ndarray:
    """Straight wire from the right edge with a sharp kink in only the last
    `1 - kink_frac` fraction of its length. Models the user's video where
    only the tip bends while the rest of the wire stays straight."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    y = h // 2
    x_right = w - 1
    x_left = 40
    total_len = x_right - x_left
    kink_x = int(x_right - kink_frac * total_len)
    cv2.line(img, (x_right, y), (kink_x, y), (255, 255, 255), thickness=3)
    tip_len = kink_x - x_left
    theta = math.radians(180.0 - angle_deg)
    tip_end = (int(round(kink_x + tip_len * math.cos(theta))),
               int(round(y - tip_len * math.sin(theta))))
    cv2.line(img, (kink_x, y), tip_end, (255, 255, 255), thickness=3)
    return img


def test_measure_bend_local_tip_kink():
    """Straight base + sharp local kink at the tip — the failure mode the
    user observed. The new curvature-based detector should still recover the
    true angle within a few degrees."""
    for true_angle in (30.0, 60.0, 90.0):
        frame = make_local_kink_frame(true_angle)
        mask = analyzer.extract_wire_mask(frame, threshold=170)
        pts = analyzer.skeleton_points(mask, thin_radius=5.0)
        ordered = analyzer.order_along_wire(pts)
        m = analyzer.measure_bend(ordered)
        assert m is not None, f"no measurement at {true_angle}° local kink"
        err = abs(m["angle_deg"] - true_angle)
        assert err < 8.0, (
            f"local kink {true_angle}° vs measured {m['angle_deg']:.2f}° "
            f"(err {err:.2f}°)"
        )


def test_direction_angle_deg_signs():
    # +x → 0°
    assert abs(analyzer.direction_angle_deg(np.array([1.0, 0.0])) - 0.0) < 1e-6
    # in image coords, "up on screen" is -y; direction (0, -1) should be +90°
    assert abs(analyzer.direction_angle_deg(np.array([0.0, -1.0])) - 90.0) < 1e-6


def test_process_video_end_to_end_tmpdir():
    """Synthesize a 30-frame video that bends from 0° → 90° linearly and
    verify the analyzer produces a CSV with a monotonically increasing
    centroid-deflection angle. The metric reported by `process_video` is
    the angle of (needle-centroid - base-entry) from horizontal — for a
    wire that bends at its midpoint, the centroid moves by ~L/4 vertically
    while the horizontal arm length is ~3W/8, so 90° tip bend gives an
    end angle of order 7–10°, not 90°."""
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        vid_path = tmp / "synth.mp4"
        w, h, fps, n = 400, 300, 30.0, 30
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        wr = cv2.VideoWriter(str(vid_path), fourcc, fps, (w, h))
        assert wr.isOpened()
        for i in range(n):
            ang = 90.0 * i / (n - 1)
            wr.write(make_bent_wire_frame(ang, w=w, h=h))
        wr.release()

        out_csv = tmp / "bend.csv"
        out_video = tmp / "annot.mp4"
        analyzer.process_video(vid_path, out_csv, out_video, show_progress=False)

        rows = list(csv.DictReader(out_csv.open()))
        angles = [float(r["angle_deg"]) for r in rows if r["angle_deg"]]
        assert len(angles) >= n - 2, f"only {len(angles)} angles measured"
        assert angles[0] < 3.0, f"start angle {angles[0]:.2f}"
        assert angles[-1] > 4.0, f"end angle {angles[-1]:.2f}"
        # angle should overall increase (allow small jitter — count monotone steps)
        ups = sum(1 for a, b in zip(angles[:-1], angles[1:]) if b >= a - 1.0)
        assert ups > 0.8 * len(angles), "angle curve not monotone enough"


def test_real_csv_has_expected_shape():
    """The WhatsApp video has been analyzed already; check the artifacts."""
    csv_path = HERE / "bend.csv"
    if not csv_path.exists():
        # skip silently if user hasn't run the analyzer yet
        return
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 3009
    angles = [float(r["angle_deg"]) for r in rows if r["angle_deg"]]
    assert len(angles) == 3009
    assert max(angles) > 80.0, f"max angle only {max(angles):.2f}"
    assert min(angles) < 10.0, f"min angle {min(angles):.2f}"


def test_plot_results_smoke():
    """plot_results.py should produce a PNG without error for the real CSV."""
    csv_path = HERE / "bend.csv"
    if not csv_path.exists():
        return
    with tempfile.TemporaryDirectory() as tmpd:
        out = Path(tmpd) / "report.png"
        proc = subprocess.run(
            [sys.executable, str(HERE / "plot_results.py"),
             str(csv_path), "--out-png", str(out)],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert out.exists() and out.stat().st_size > 5000


def test_plot_results_clamp_window():
    assert plot_results.clamp_window(15, 100) == 15
    assert plot_results.clamp_window(16, 100) == 15  # forced odd
    assert plot_results.clamp_window(1000, 100) == 99
    assert plot_results.clamp_window(3, 100) is None  # too small


# ---------------------------------------------------------------- runner ---
def _run_all() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_all())
