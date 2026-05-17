"""Wire-bend analyzer.

Reads a video of a thin wire/cantilever being bent, computes the bend angle in
each frame, and writes:
  - A CSV with per-frame measurements.
  - An annotated MP4 visualising the detection.

Usage:
    python analyzer.py INPUT.mp4 [--out-csv out.csv] [--out-video out.mp4]
                       [--threshold 170] [--thin-radius 5]
                       [--base-frac 0.3] [--tip-frac 0.15]
                       [--show-progress]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize


@dataclass
class FrameMeasurement:
    frame: int
    time_s: float
    angle_deg: float | None
    tip_x: float | None
    tip_y: float | None
    base_x: float | None
    base_y: float | None
    base_dir_deg: float | None
    tip_dir_deg: float | None
    n_skeleton: int
    mid_x: float | None = None
    mid_y: float | None = None
    needle_len: float | None = None


def extract_wire_mask(frame_bgr: np.ndarray, threshold: int) -> np.ndarray | None:
    """Binary mask of the wire: the bright connected component touching the
    right image edge (the wire enters the frame from the right)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    right_labels = [L for L in np.unique(labels[:, -1]) if L != 0]
    if not right_labels:
        return None
    biggest = max(right_labels, key=lambda L: stats[L, cv2.CC_STAT_AREA])
    return (labels == biggest).astype(np.uint8) * 255


def _find_ball_and_tip(
    mask: np.ndarray,
    min_radius: float = 11.0,
):
    """Combined ball + wire-tip extraction in one skeleton walk.

    Returns a dict with keys 'ball' (np.array [x, y, r] or None) and
    'tip_pt' (np.array [x, y] — leftmost skeleton endpoint, the wire's
    free tip) and 'path' (the Nx2 walk from tip back to right entry,
    in skeleton pixel order). Returns None if the skeleton is too small
    to analyse."""
    if mask is None:
        return None
    h, w = mask.shape
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    sk = skeletonize(mask > 0).astype(np.uint8)
    sk = _prune_skeleton_spurs(sk, 12)
    ys_sk, xs_sk = np.where(sk > 0)
    if len(xs_sk) < 30:
        return None
    i_start = int(np.argmax(xs_sk))
    start = (int(ys_sk[i_start]), int(xs_sk[i_start]))
    dist_map, py, px = _bfs_dist(sk, start)
    from scipy import ndimage
    end_kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
    end_conv = ndimage.convolve(sk, end_kernel, mode="constant")
    endpoints = (end_conv == 11) & (dist_map > 0)
    ey, ex = np.where(endpoints)
    if len(ex) == 0:
        flat = dist_map.flatten()
        if flat.max() <= 0:
            return None
        fi = int(np.argmax(flat))
        fy, fx = int(fi // dist_map.shape[1]), int(fi % dist_map.shape[1])
    else:
        i_far = int(np.argmin(ex))
        fy, fx = int(ey[i_far]), int(ex[i_far])
    path: list[tuple[int, int]] = []
    cy, cx = fy, fx
    while cy >= 0 and cx >= 0:
        path.append((cx, cy))
        ny, nx = int(py[cy, cx]), int(px[cy, cx])
        if ny < 0:
            break
        cy, cx = ny, nx
    if len(path) < 40:
        return None
    path_arr = np.array(path, dtype=np.int32)
    thickness = dist[path_arr[:, 1], path_arr[:, 0]]

    from scipy.signal import find_peaks
    peaks, _ = find_peaks(thickness, distance=8, prominence=1.5)
    ball = None
    tip_zone = min(120, max(40, int(0.3 * len(path_arr))))
    for pi in sorted(peaks.tolist()):
        if pi > tip_zone:
            break
        r = float(thickness[pi])
        if r < min_radius:
            continue
        x_, y_ = int(path_arr[pi, 0]), int(path_arr[pi, 1])
        med = min(x_, w - x_, y_, h - y_)
        if med < 2.5 * r:
            continue
        s = 6
        win = dist[max(0, y_ - s): y_ + s + 1, max(0, x_ - s): x_ + s + 1]
        if win.size:
            rel = np.unravel_index(int(np.argmax(win)), win.shape)
            yy = max(0, y_ - s) + int(rel[0])
            xx = max(0, x_ - s) + int(rel[1])
            r = float(dist[yy, xx])
        else:
            xx, yy = x_, y_
        ball = np.array([float(xx), float(yy), r], dtype=np.float64)
        break

    tip_pt = np.array([float(fx), float(fy)], dtype=np.float64)
    return {"ball": ball, "tip_pt": tip_pt, "path": path_arr}


def find_ball_center(
    mask: np.ndarray,
    min_radius: float = 11.0,
    edge_margin: int | None = None,
) -> np.ndarray | None:
    """Ball centre = the first thickness peak along the wire skeleton walked
    from the leftmost endpoint (wire's free tip) back toward the right-edge
    entry. Returns (x, y, radius) or None.

    See `_find_ball_and_tip` for the underlying skeleton-walk implementation.
    """
    r = _find_ball_and_tip(mask, min_radius=min_radius)
    if r is None:
        return None
    return r["ball"]


def needle_direction(
    gray: np.ndarray,
    ball: np.ndarray,
    base_dir: np.ndarray,
    bright_thr: int = 130,
    search_r: int = 80,
    n_angles: int = 720,
    min_run: int = 6,
) -> tuple[np.ndarray, int, np.ndarray] | None:
    """Find the needle protruding from the ball by ray-casting.

    The needle is the thin, dim protrusion that emerges from the ball on a
    different side from the entry wire. Visually it sits at a gray level
    around 130-160, well below the bright wire body (>200), so a low
    threshold is used and we ignore rays whose direction points back
    toward where the wire entered (within ~60° of -base_dir).

    Returns (unit needle direction, run length in px, far point), or None
    if no protrusion of length `min_run` is found."""
    h, w = gray.shape
    bx, by, br = float(ball[0]), float(ball[1]), float(ball[2])
    excl = -base_dir
    best_len = -1
    best_theta = None
    best_far = None
    for k in range(n_angles):
        theta = -np.pi + 2 * np.pi * k / n_angles
        ct, st = float(np.cos(theta)), float(np.sin(theta))
        # exclude rays within 60° of where the wire came from
        if ct * excl[0] + st * excl[1] > 0.5:
            continue
        run = 0
        max_run = 0
        gaps = 0
        far_d = 0
        for d in range(int(br), int(br) + search_r):
            x = int(bx + d * ct)
            y = int(by + d * st)
            if not (0 <= x < w and 0 <= y < h):
                break
            if gray[y, x] >= bright_thr:
                run += 1
                gaps = 0
                if run > max_run:
                    max_run = run
                    far_d = d
            else:
                gaps += 1
                if gaps > 2:
                    run = 0
        if max_run > best_len:
            best_len = max_run
            best_theta = theta
            best_far = np.array([bx + far_d * ct, by + far_d * st], dtype=np.float64)
    if best_theta is None or best_len < min_run:
        return None
    return (
        np.array([np.cos(best_theta), np.sin(best_theta)], dtype=np.float64),
        int(best_len),
        best_far,
    )


def _bfs_dist(sk: np.ndarray, start_yx: tuple[int, int]):
    """BFS over an 8-connected skeleton. Returns (dist, parent_y, parent_x)
    arrays."""
    h, w = sk.shape
    dist = -np.ones((h, w), dtype=np.int32)
    py = np.full((h, w), -1, dtype=np.int32)
    px = np.full((h, w), -1, dtype=np.int32)
    sy, sx = start_yx
    if sk[sy, sx] == 0:
        ys, xs = np.where(sk > 0)
        i = int(np.argmin((xs - sx) ** 2 + (ys - sy) ** 2))
        sy, sx = int(ys[i]), int(xs[i])
    dist[sy, sx] = 0
    frontier = [(sy, sx)]
    while frontier:
        nxt = []
        for y, x in frontier:
            d = dist[y, x] + 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and sk[ny, nx] > 0 and dist[ny, nx] < 0:
                        dist[ny, nx] = d
                        py[ny, nx] = y
                        px[ny, nx] = x
                        nxt.append((ny, nx))
        frontier = nxt
    return dist, py, px


def _prune_skeleton_spurs(sk: np.ndarray, min_len: int = 15) -> np.ndarray:
    """Iteratively remove short branches from a skeleton. A branch is the
    chain of pixels starting at a degree-1 pixel and walking until reaching
    a junction (degree >= 3) or another endpoint. Branches shorter than
    `min_len` are deleted, then the process repeats until no more spurs
    are pruned. This collapses thinning artefacts (small whiskers around
    the ball) so only the main wire spine survives."""
    from scipy import ndimage
    sk = sk.copy().astype(np.uint8)
    h, w = sk.shape
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
    for _ in range(8):
        conv = ndimage.convolve(sk, kernel, mode="constant")
        ends = np.argwhere(conv == 11)  # degree-1
        if len(ends) == 0:
            break
        any_pruned = False
        for sy, sx in ends:
            if sk[sy, sx] == 0:
                continue
            chain = [(sy, sx)]
            prev = (-1, -1)
            while True:
                y, x = chain[-1]
                neighbours = []
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and sk[ny, nx] > 0:
                            if (ny, nx) != prev:
                                neighbours.append((ny, nx))
                if len(neighbours) != 1:
                    break
                if len(chain) >= min_len:
                    break
                prev = (y, x)
                chain.append(neighbours[0])
            if len(chain) < min_len:
                for cy, cx in chain[:-1]:
                    sk[cy, cx] = 0
                any_pruned = True
        if not any_pruned:
            break
    return sk


def _needle_from_mask(
    mask: np.ndarray,
    ball: np.ndarray,
    base_dir: np.ndarray | None = None,
    max_ball_gap: float = 12.0,
    min_protrusion: float = 4.0,
) -> np.ndarray | None:
    """Isolate the needle from a wire mask by erasing the ball region and
    picking the connected component that (a) is not connected to the right
    image edge (i.e. not the base wire), (b) sits near the ball, and (c) if
    `base_dir` is given, has its centroid on the far side of the ball from
    the wire entry — the needle protrudes opposite to where the wire comes
    in, not on the same side as the wire-ball junction. The directional
    guard rejects bright reflections / mask noise hugging the wire-side of
    the ball that would otherwise be picked as a fake needle.

    Returns the needle pixel coordinates as Nx2 float, or None."""
    h, w = mask.shape
    bx, by, br = float(ball[0]), float(ball[1]), float(ball[2])
    yy, xx = np.indices((h, w))
    near_ball = ((xx - bx) ** 2 + (yy - by) ** 2) <= (br + 3.0) ** 2
    m = mask.copy()
    m[near_ball] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    right_labels = set(int(L) for L in np.unique(labels[:, -1]) if L != 0)
    best_pts = None
    best_score = -np.inf
    for L in range(1, n):
        if L in right_labels:
            continue
        area = int(stats[L, cv2.CC_STAT_AREA])
        if area < 8:
            continue
        ys, xs = np.where(labels == L)
        d = np.sqrt((xs - bx) ** 2 + (ys - by) ** 2)
        d_min = float(d.min())
        d_max = float(d.max())
        if d_min > br + max_ball_gap:
            continue
        if d_max - br < min_protrusion:
            continue
        if base_dir is not None:
            cx_ = float(xs.mean()) - bx
            cy_ = float(ys.mean()) - by
            # base_dir points into the frame (away from the right-edge entry).
            # The needle must lie on that same side of the ball, not back
            # toward the entry — require a positive projection on base_dir.
            if cx_ * base_dir[0] + cy_ * base_dir[1] <= 0:
                continue
        score = (d_max - br) + 0.05 * area - d_min
        if score > best_score:
            best_score = score
            best_pts = np.column_stack([xs, ys]).astype(np.float64)
    return best_pts


def measure_tip_deflection(
    gray: np.ndarray,
    mask: np.ndarray,
    base_window: int = 80,
    tip_window: int = 40,
    spur_min_len: int = 15,
):
    """Needle angle and midpoint.

    The needle is the short section of wire that the sphere sits on. We
    isolate it as the skeleton segment from the wire's free tip back to
    the ball centre (or, if no ball is detected, the last ~40 pixels of
    the skeleton near the tip). A PCA line is fit through those pixels;
    the angle is the angle of that line vs the horizontal entry direction
    (-x), and the tracked point is the midpoint of the fitted needle
    (geometric centre of the two endpoints projected onto the PCA axis).

    Using a PCA line fit on the needle pixels (not a centroid of the
    whole wire mask) makes the angle robust: the long horizontal base
    wire no longer dominates the measurement, so the reported angle is
    the actual orientation of the short rod the sphere is mounted on.

    Returns a dict compatible with `annotate_frame` and the per-frame CSV
    schema, or None if the geometry can't be determined."""
    if mask is None or gray is None:
        return None
    h, w = mask.shape

    # base_pt: where the wire meets the right edge of the frame.
    right_col_ys = np.where(mask[:, -1] > 0)[0]
    if len(right_col_ys) == 0:
        return None
    base_pt = np.array([float(w - 1), float(np.mean(right_col_ys))],
                       dtype=np.float64)

    # Skeleton walk: path is ordered tip → right-edge entry, and (if the
    # ball is detectable) we know its index along that path.
    bt = _find_ball_and_tip(mask)
    if bt is None:
        return None
    ball = bt["ball"]
    path = bt["path"].astype(np.float64)  # ordered tip → entry, Nx2 (x, y)

    if ball is not None:
        bx, by = float(ball[0]), float(ball[1])
        d_to_ball = np.linalg.norm(path - np.array([bx, by]), axis=1)
        ball_idx = int(np.argmin(d_to_ball))
        # Needle = path from the free tip up to the ball-junction index.
        needle_pts = path[: ball_idx + 1]
    else:
        # Fall back: last ~40 skeleton points near the tip.
        n_take = min(40, max(8, len(path) // 4))
        needle_pts = path[:n_take]

    if len(needle_pts) < 6:
        return None

    # PCA fit through the needle skeleton.
    c = needle_pts.mean(axis=0)
    _, _, vt = np.linalg.svd(needle_pts - c, full_matrices=False)
    needle_dir = vt[0].astype(np.float64)
    # Orient needle_dir to point from the wire's base toward the free tip.
    base_to_tip = path[0] - path[-1]
    if float(needle_dir @ base_to_tip) < 0:
        needle_dir = -needle_dir

    # Project needle pixels onto the line to find the two endpoints, then
    # take their midpoint as the tracked feature.
    proj = (needle_pts - c) @ needle_dir
    p_lo, p_hi = float(proj.min()), float(proj.max())
    needle_base_end = c + p_lo * needle_dir   # ball-side end
    needle_tip_end = c + p_hi * needle_dir    # free-tip end
    needle_midpoint = 0.5 * (needle_base_end + needle_tip_end)
    needle_length = float(p_hi - p_lo)
    if needle_length < 6.0:
        return None

    # Angle between the needle direction and the horizontal entry axis (-x).
    # 0° = needle horizontal (along base wire), 90° = vertical, 180° = folded.
    base_dir_ref = np.array([-1.0, 0.0], dtype=np.float64)
    cos = float(np.clip(needle_dir @ base_dir_ref, -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cos)))

    base_pts = np.array(
        [[float(w - 1), float(y)] for y in right_col_ys], dtype=np.float64,
    )
    return {
        "angle_deg": angle_deg,
        "base_pts": base_pts,
        "tip_pts": needle_pts,
        "base_centroid": base_pt,
        "tip_centroid": needle_midpoint,
        "base_dir": base_dir_ref,
        "tip_dir": needle_dir,
        "tip_pt": needle_midpoint,
        "entry_pt": base_pt,
        "kink_pt": needle_midpoint,
        "ball_center": None if ball is None else ball[:2],
        "needle_base_end": needle_base_end,
        "needle_tip_end": needle_tip_end,
        "needle_midpoint": needle_midpoint,
        "needle_length": needle_length,
    }


def base_direction(mask: np.ndarray, sample_x_window: int = 80) -> np.ndarray | None:
    """PCA direction of the wire skeleton near the right-edge entry.

    Orients the unit vector toward decreasing x (into the frame), matching
    the geometry of a wire that enters from the right and bends to the
    left. Returns None if the skeleton has fewer than 8 pixels near the
    right edge."""
    if mask is None:
        return None
    sk = skeletonize(mask > 0).astype(np.uint8)
    ys, xs = np.where(sk > 0)
    if len(xs) < 10:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float64)
    x_max = pts[:, 0].max()
    sample = pts[pts[:, 0] >= x_max - sample_x_window]
    if len(sample) < 8:
        sample = pts
    c = sample.mean(axis=0)
    _, _, vt = np.linalg.svd(sample - c, full_matrices=False)
    v = vt[0]
    if v[0] > 0:
        v = -v
    return v


def skeleton_points(mask: np.ndarray, thin_radius: float) -> np.ndarray | None:
    """Skeletonise the thin parts of the mask. Distance-transform values above
    `thin_radius` are treated as blob (e.g. the ball) and excluded so the
    centerline stays on the wire."""
    if mask is None:
        return None
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    thin = (dist > 0) & (dist <= thin_radius)
    if thin.sum() < 30:
        return None
    sk = skeletonize(thin).astype(np.uint8)
    ys, xs = np.where(sk > 0)
    if len(xs) < 20:
        return None
    return np.column_stack([xs, ys]).astype(np.float64)


def _bfs_path(sk: np.ndarray, start_yx: tuple[int, int]) -> np.ndarray | None:
    """BFS-walk the skeleton from `start_yx` on 8-connectivity. Returns the
    sequence of (x, y) points along the shortest path from start to the
    farthest-reachable skeleton pixel — a true entry→tip walk with no
    there-and-back artefacts, regardless of skeleton branching."""
    h, w = sk.shape
    dist = -np.ones((h, w), dtype=np.int32)
    parent_y = np.full((h, w), -1, dtype=np.int32)
    parent_x = np.full((h, w), -1, dtype=np.int32)
    sy, sx = start_yx
    if sk[sy, sx] == 0:
        # snap start to nearest skeleton pixel
        ys, xs = np.where(sk > 0)
        if len(xs) == 0:
            return None
        i = int(np.argmin((xs - sx) ** 2 + (ys - sy) ** 2))
        sy, sx = int(ys[i]), int(xs[i])
    dist[sy, sx] = 0
    frontier = [(sy, sx)]
    far_y, far_x, far_d = sy, sx, 0
    while frontier:
        nxt = []
        for y, x in frontier:
            d = dist[y, x] + 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and sk[ny, nx] > 0 and dist[ny, nx] < 0:
                        dist[ny, nx] = d
                        parent_y[ny, nx] = y
                        parent_x[ny, nx] = x
                        nxt.append((ny, nx))
                        if d > far_d:
                            far_d = d
                            far_y, far_x = ny, nx
        frontier = nxt
    if far_d == 0:
        return None
    out = []
    y, x = far_y, far_x
    while y >= 0:
        out.append((x, y))
        py, px = int(parent_y[y, x]), int(parent_x[y, x])
        if py < 0:
            break
        y, x = py, px
    return np.array(out, dtype=np.float64)[::-1]


def base_wire_skeleton(
    mask: np.ndarray,
    thin_radius: float,
    ball_center: np.ndarray | None,
) -> np.ndarray | None:
    """Ordered (x, y) path along the base wire from the right-edge entry to
    the wire's terminus at the ball.

    The ball mask is dilated and removed from the wire before skeletonisation
    so the skeleton actually terminates at the ball boundary. A BFS from the
    entry point then returns a clean entry→tip path with no there-and-back
    artefacts (no looping along parallel skeleton pixels of a thick wire)."""
    if mask is None:
        return None
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    thin = ((dist > 0) & (dist <= thin_radius)).astype(np.uint8)
    if thin.sum() < 30:
        return None
    sk = skeletonize(thin).astype(np.uint8)
    if sk.sum() < 20:
        return None
    if ball_center is not None:
        # Remove skeleton pixels close to the ball centre to sever the
        # bridging skeleton around the ball without damaging the wire body.
        h, w = sk.shape
        yy, xx = np.indices((h, w))
        cut = ((xx - ball_center[0]) ** 2 + (yy - ball_center[1]) ** 2
               <= (thin_radius * 2 + 6) ** 2)
        sk = sk.copy()
        sk[cut] = 0
    n_sk, sk_labels, sk_stats, _ = cv2.connectedComponentsWithStats(sk, connectivity=8)
    if n_sk <= 1:
        return None
    # Base CC = the component containing the rightmost skeleton pixel
    # (the skeleton may stop one or two pixels short of the image edge,
    # so test by actual max-x of each component rather than column -1).
    best_label = 0
    best_max_x = -1
    for L in range(1, n_sk):
        x_max_L = int(sk_stats[L, cv2.CC_STAT_LEFT] + sk_stats[L, cv2.CC_STAT_WIDTH] - 1)
        if x_max_L > best_max_x:
            best_max_x = x_max_L
            best_label = L
    if best_label == 0:
        return None
    base_only = ((sk_labels == best_label).astype(np.uint8)) * sk
    ys, xs = np.where(base_only > 0)
    if len(xs) < 20:
        return None
    i = int(np.argmax(xs))
    start = (int(ys[i]), int(xs[i]))
    path = _bfs_path(base_only, start)
    if path is None or len(path) < 20:
        return None
    return path


def find_needle_pts(
    mask: np.ndarray,
    thin_radius: float,
    ball_center: np.ndarray | None,
) -> np.ndarray | None:
    """Skeleton pixels of the needle: the thin protrusion that emerges from
    the ball on a different side than the entry wire.

    Strategy: dilate the ball mask and subtract it from the wire mask, which
    severs the bridging skeleton at the ball. The resulting mask has the
    base wire (right-edge-connected) and the needle as separate components.
    The needle is then the non-base skeleton component nearest to the ball."""
    if mask is None or ball_center is None:
        return None
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    ball_mask = ((dist > thin_radius - 0.5).astype(np.uint8)) * 255
    if ball_mask.sum() < 25 * 255:
        return None
    kernel = np.ones((7, 7), np.uint8)
    ball_dilated = cv2.dilate(ball_mask, kernel, iterations=1)
    sep = mask.copy()
    sep[ball_dilated > 0] = 0
    sep_dist = cv2.distanceTransform(sep, cv2.DIST_L2, 5)
    thin = ((sep_dist > 0) & (sep_dist <= thin_radius)).astype(np.uint8)
    if thin.sum() < 6:
        return None
    sk = skeletonize(thin).astype(np.uint8)
    if sk.sum() < 4:
        return None
    n_sk, sk_labels, sk_stats, _ = cv2.connectedComponentsWithStats(sk, connectivity=8)
    if n_sk <= 1:
        return None
    right_col_labels = np.unique(sk_labels[:, -1])
    right_col_labels = right_col_labels[right_col_labels != 0]
    base_label = None
    if len(right_col_labels) > 0:
        base_label = int(max(
            right_col_labels.tolist(),
            key=lambda L: sk_stats[int(L), cv2.CC_STAT_AREA],
        ))
    best_pts = None
    best_score = -np.inf
    for L in range(1, n_sk):
        if L == base_label:
            continue
        area = int(sk_stats[L, cv2.CC_STAT_AREA])
        if area < 2:
            continue
        ys, xs = np.where(sk_labels == L)
        pts = np.column_stack([xs, ys]).astype(np.float64)
        d_min = float(np.linalg.norm(pts - ball_center, axis=1).min())
        if d_min > 35.0:
            continue
        score = area - d_min
        if score > best_score:
            best_score = score
            best_pts = pts
    return best_pts


def order_along_wire(pts: np.ndarray) -> np.ndarray:
    """Order skeleton pixels from the right-edge entry to the tip by greedy
    nearest-neighbour walking. Falls back to descending-x sort if the walk
    breaks down."""
    if len(pts) < 5:
        return pts[np.argsort(-pts[:, 0])]
    start_idx = int(np.argmax(pts[:, 0]))
    visited = np.zeros(len(pts), dtype=bool)
    order_idx = [start_idx]
    visited[start_idx] = True
    current = pts[start_idx]
    while True:
        diff = pts - current
        d2 = (diff * diff).sum(axis=1)
        d2[visited] = np.inf
        nxt = int(np.argmin(d2))
        if not np.isfinite(d2[nxt]) or d2[nxt] > 25.0:
            break
        visited[nxt] = True
        order_idx.append(nxt)
        current = pts[nxt]
    if visited.sum() < 0.5 * len(pts):
        return pts[np.argsort(-pts[:, 0])]
    return pts[order_idx]


def fit_direction(P: np.ndarray) -> np.ndarray:
    """Unit vector along the principal axis of the points (PCA)."""
    P = np.asarray(P, dtype=np.float64)
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    return vt[0]


def _smooth_path(pts: np.ndarray, window: int) -> np.ndarray:
    """Moving-average smooth the ordered point sequence (edge-padded)."""
    k = max(3, window)
    if k % 2 == 0:
        k += 1
    n = len(pts)
    if k >= n:
        k = max(3, n - (1 - n % 2))
    pad = k // 2
    padded = np.pad(pts, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(k) / k
    sx = np.convolve(padded[:, 0], kernel, mode="valid")
    sy = np.convolve(padded[:, 1], kernel, mode="valid")
    return np.column_stack([sx, sy])


def _tangents(path: np.ndarray, half_window: int) -> np.ndarray:
    """Unit tangent at each point via local windowed PCA, oriented in the
    direction of increasing index (entry → tip)."""
    n = len(path)
    tans = np.zeros_like(path)
    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n, i + half_window + 1)
        seg = path[lo:hi]
        if len(seg) < 3:
            tans[i] = (1.0, 0.0)
            continue
        c = seg.mean(axis=0)
        _, _, vt = np.linalg.svd(seg - c, full_matrices=False)
        v = vt[0]
        j_lo = max(0, i - 3)
        j_hi = min(n - 1, i + 3)
        ref = path[j_hi] - path[j_lo]
        if np.dot(v, ref) < 0:
            v = -v
        tans[i] = v
    return tans


def measure_bend(
    pts: np.ndarray,
    ball_center: np.ndarray | None = None,
    needle_pts: np.ndarray | None = None,
    base_frac: float = 0.3,
    tip_frac: float = 0.15,
):
    """Measure the bend angle.

    Priority for the tip direction:
    1. If `needle_pts` is supplied, the tip direction is the vector from the
       ball centre to the farthest needle skeleton point — the actual needle
       angle the user wants to measure.
    2. Else if `ball_center` is supplied, the tip direction is the vector from
       the wire-ball junction to the ball centre.
    3. Else (no ball, e.g. synthetic tests): curvature-based PCA on the
       post-kink skeleton segment.

    `base_frac` and `tip_frac` are accepted for backward compatibility.
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 30:
        return None
    xs = pts[:, 0]
    if xs.max() - xs.min() < 20:
        return None

    smooth_win = max(7, n // 25 | 1)
    half_w = max(8, n // 20)
    path = _smooth_path(pts, smooth_win)
    entry_pt = path[0]

    if ball_center is not None:
        d_to_ball = np.linalg.norm(path - ball_center, axis=1)
        junc_i = int(np.argmin(d_to_ball))
        if junc_i < 30:
            return None
        junction = path[junc_i]
        # Base segment: short anchor near the right-edge entry (the wire's
        # rigid, undeformed direction). Tip segment: short stretch of wire
        # just before it reaches the ball (the deformed tangent at the bend).
        # The angle between these two PCA directions is the bend angle in
        # the user's sense: 0° when straight, 90° when the wire ends
        # perpendicular to its entry, 180° when folded back.
        base_K = int(np.clip(junc_i * 0.25, 20, 100))
        tip_K = int(np.clip(junc_i * 0.25, 20, 100))
        if junc_i - base_K - tip_K < 20:
            base_K = max(15, junc_i // 4)
            tip_K = max(15, junc_i // 4)
        base_seg = path[:base_K]
        tip_seg = path[max(base_K, junc_i - tip_K): junc_i + 1]
        if len(base_seg) < 8 or len(tip_seg) < 8:
            return None
        db = fit_direction(base_seg)
        dt = fit_direction(tip_seg)
        ref = ball_center - path[0]
        if np.dot(db, ref) < 0:
            db = -db
        if np.dot(dt, ref) < 0:
            dt = -dt
        kink_pt = junction
        tip_centroid = tip_seg.mean(axis=0)
        tip_far_pt = junction
    else:
        tans = _tangents(path, half_w)
        step = max(half_w // 2, 5)
        ang_jump = np.zeros(n)
        for i in range(step, n - step):
            cos = float(np.clip(tans[i - step] @ tans[i + step], -1.0, 1.0))
            ang_jump[i] = np.degrees(np.arccos(cos))
        margin = max(half_w + 2, 10)
        if n - 2 * margin < 6:
            return None
        interior = np.arange(margin, n - margin)
        kink_i = int(interior[np.argmax(ang_jump[interior])])
        gap = max(half_w // 2, 4)
        base_seg = path[: max(kink_i - gap, 5)]
        tip_seg = path[min(kink_i + gap, n - 5):]
        if len(base_seg) < 6 or len(tip_seg) < 6:
            return None
        db = fit_direction(base_seg)
        dt = fit_direction(tip_seg)
        kink_pt = path[kink_i]
        tip_centroid = tip_seg.mean(axis=0)
        tip_far_pt = path[-1]

    tip_ref = tip_far_pt if ball_center is not None else path[-1]
    ref = tip_ref - entry_pt
    if np.dot(db, ref) < 0:
        db = -db
    if ball_center is None and np.dot(dt, ref) < 0:
        dt = -dt

    cos = float(np.clip(db @ dt, -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cos)))

    return {
        "angle_deg": angle_deg,
        "base_pts": base_seg,
        "tip_pts": tip_seg,
        "base_centroid": base_seg.mean(axis=0),
        "tip_centroid": tip_centroid,
        "base_dir": db,
        "tip_dir": dt,
        "tip_pt": tip_ref,
        "entry_pt": entry_pt,
        "kink_pt": kink_pt,
        "ball_center": ball_center,
    }


def direction_angle_deg(v: np.ndarray) -> float:
    """Angle of a 2D vector in degrees, CCW from +x. y is negated so the
    angle matches what a viewer sees on screen (image y grows downward)."""
    return float(np.degrees(np.arctan2(-v[1], v[0])))


def annotate_frame(frame_bgr, pts, measurement, t_s, frame_idx):
    out = frame_bgr.copy()
    if pts is not None and len(pts):
        for p in pts.astype(int):
            cv2.circle(out, tuple(p), 1, (0, 200, 255), -1)
    if measurement is not None:
        for p in measurement["base_pts"].astype(int):
            cv2.circle(out, tuple(p), 2, (0, 255, 0), -1)
        for p in measurement["tip_pts"].astype(int):
            cv2.circle(out, tuple(p), 2, (0, 0, 255), -1)
        bc = measurement["base_centroid"]
        tc = measurement["tip_centroid"]
        db = measurement["base_dir"]
        dt = measurement["tip_dir"]
        L = 80
        cv2.arrowedLine(
            out,
            tuple((bc - db * L * 0.3).astype(int)),
            tuple((bc + db * L).astype(int)),
            (0, 255, 0), 2, tipLength=0.25,
        )
        cv2.arrowedLine(
            out,
            tuple((tc - dt * L * 0.3).astype(int)),
            tuple((tc + dt * L).astype(int)),
            (0, 0, 255), 2, tipLength=0.25,
        )
        nb = measurement.get("needle_base_end")
        nt = measurement.get("needle_tip_end")
        nm = measurement.get("needle_midpoint")
        if nb is not None and nt is not None:
            cv2.line(
                out,
                tuple(np.asarray(nb).astype(int)),
                tuple(np.asarray(nt).astype(int)),
                (0, 255, 255), 2,
            )
        if nm is not None:
            mp = tuple(np.asarray(nm).astype(int))
            cv2.circle(out, mp, 8, (0, 255, 255), 2)
            cv2.drawMarker(out, mp, (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
        if measurement.get("ball_center") is not None:
            bc_pt = tuple(np.asarray(measurement["ball_center"]).astype(int))
            cv2.drawMarker(out, bc_pt, (255, 0, 0), cv2.MARKER_CROSS, 14, 2)
        ang = measurement["angle_deg"]
        cv2.rectangle(out, (5, 5), (280, 70), (0, 0, 0), -1)
        cv2.putText(out, f"bend = {ang:6.2f} deg", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(out, f"t = {t_s:6.2f} s  f={frame_idx}", (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    else:
        cv2.rectangle(out, (5, 5), (280, 40), (0, 0, 0), -1)
        cv2.putText(out, "no detection", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return out


def process_video(
    in_path: Path,
    out_csv: Path,
    out_video: Path,
    threshold: int = 130,
    thin_radius: float = 5.0,
    base_frac: float = 0.3,
    tip_frac: float = 0.15,
    show_progress: bool = False,
):
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {in_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"cannot open writer for {out_video}")

    rows: list[FrameMeasurement] = []
    t0 = time.time()
    last_log = t0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_s = idx / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Use a lower threshold for the wire mask so the dim needle protrusion
        # at the wire's tip is included alongside the bright wire body.
        mask = extract_wire_mask(frame, threshold)
        measurement = None
        m = FrameMeasurement(
            frame=idx, time_s=t_s,
            angle_deg=None, tip_x=None, tip_y=None,
            base_x=None, base_y=None,
            base_dir_deg=None, tip_dir_deg=None,
            n_skeleton=0,
        )
        ordered = None
        if mask is not None:
            sk = skeletonize(mask > 0).astype(np.uint8)
            ys, xs = np.where(sk > 0)
            ordered = np.column_stack([xs, ys]).astype(np.float64) if len(xs) else None
            m.n_skeleton = 0 if ordered is None else len(ordered)
            measurement = measure_tip_deflection(gray, mask)
        if measurement is None and mask is not None:
            # Fallback chain: if needle-from-ball detection fails, try
            # ball-based curvature (the wire's tangent just before the ball
            # gives the tip direction even when the needle is too small to
            # segment). If that also fails, do pure curvature on the skeleton.
            pts = skeleton_points(mask, thin_radius)
            curv_path = order_along_wire(pts) if pts is not None else None
            if curv_path is not None and len(curv_path) >= 10:
                ball = find_ball_center(mask)
                if ball is not None:
                    measurement = measure_bend(
                        curv_path, ball_center=ball[:2], needle_pts=None,
                        base_frac=base_frac, tip_frac=tip_frac,
                    )
                if measurement is None:
                    measurement = measure_bend(
                        curv_path, ball_center=None, needle_pts=None,
                        base_frac=base_frac, tip_frac=tip_frac,
                    )
                if measurement is not None:
                    ordered = curv_path
                    m.n_skeleton = len(curv_path)
        if measurement is not None:
            m.angle_deg = measurement["angle_deg"]
            m.tip_x = float(measurement["tip_pt"][0])
            m.tip_y = float(measurement["tip_pt"][1])
            m.base_x = float(measurement["base_centroid"][0])
            m.base_y = float(measurement["base_centroid"][1])
            m.base_dir_deg = direction_angle_deg(measurement["base_dir"])
            m.tip_dir_deg = direction_angle_deg(measurement["tip_dir"])
            nm = measurement.get("needle_midpoint")
            if nm is not None:
                m.mid_x = float(nm[0])
                m.mid_y = float(nm[1])
            nl = measurement.get("needle_length")
            if nl is not None:
                m.needle_len = float(nl)
        rows.append(m)
        writer.write(annotate_frame(frame, ordered, measurement, t_s, idx))

        if show_progress and time.time() - last_log > 1.0:
            done = idx + 1
            pct = 100 * done / max(n, 1)
            elapsed = time.time() - t0
            eta = elapsed / max(done, 1) * (n - done)
            print(f"  [{done}/{n}] {pct:5.1f}%  elapsed {elapsed:5.1f}s  eta {eta:5.1f}s",
                  file=sys.stderr)
            last_log = time.time()
        idx += 1

    cap.release()
    writer.release()

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "frame", "time_s", "angle_deg",
            "tip_x", "tip_y", "base_x", "base_y",
            "base_dir_deg", "tip_dir_deg", "n_skeleton",
            "mid_x", "mid_y", "needle_len",
        ])
        for m in rows:
            w.writerow([
                m.frame, f"{m.time_s:.4f}",
                "" if m.angle_deg is None else f"{m.angle_deg:.3f}",
                "" if m.tip_x is None else f"{m.tip_x:.2f}",
                "" if m.tip_y is None else f"{m.tip_y:.2f}",
                "" if m.base_x is None else f"{m.base_x:.2f}",
                "" if m.base_y is None else f"{m.base_y:.2f}",
                "" if m.base_dir_deg is None else f"{m.base_dir_deg:.3f}",
                "" if m.tip_dir_deg is None else f"{m.tip_dir_deg:.3f}",
                m.n_skeleton,
                "" if m.mid_x is None else f"{m.mid_x:.2f}",
                "" if m.mid_y is None else f"{m.mid_y:.2f}",
                "" if m.needle_len is None else f"{m.needle_len:.2f}",
            ])

    n_det = sum(1 for m in rows if m.angle_deg is not None)
    if show_progress:
        print(
            f"done: {len(rows)} frames, {n_det} with angle "
            f"({100 * n_det / max(len(rows), 1):.1f}%)",
            file=sys.stderr,
        )
    return rows, fps


def main():
    ap = argparse.ArgumentParser(description="Wire bend-angle analyzer")
    ap.add_argument("video", type=Path)
    ap.add_argument("--out-csv", type=Path, default=Path("bend.csv"))
    ap.add_argument("--out-video", type=Path, default=Path("bend_annotated.mp4"))
    ap.add_argument("--threshold", type=int, default=130)
    ap.add_argument("--thin-radius", type=float, default=5.0)
    ap.add_argument("--base-frac", type=float, default=0.3)
    ap.add_argument("--tip-frac", type=float, default=0.15)
    ap.add_argument("--show-progress", action="store_true")
    args = ap.parse_args()
    process_video(
        in_path=args.video,
        out_csv=args.out_csv,
        out_video=args.out_video,
        threshold=args.threshold,
        thin_radius=args.thin_radius,
        base_frac=args.base_frac,
        tip_frac=args.tip_frac,
        show_progress=args.show_progress,
    )


if __name__ == "__main__":
    main()
