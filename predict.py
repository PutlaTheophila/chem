"""Run the trained tangent model over a video and emit an annotated MP4 + CSV.

The horizontal axis is read from dataset/axis/<video>.json (set in
label_tool.py). For each frame we predict the two tangent endpoints, compute
the angle vs the axis, and overlay both lines plus the angle and timestamp.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

ROOT = Path(__file__).parent
DATASET = ROOT / "dataset"
AXIS_DIR = DATASET / "axis"
MODELS = ROOT / "models"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def video_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")


def load_axis(video: Path) -> tuple | None:
    p = AXIS_DIR / f"{video_stem(video)}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return (tuple(d["ax1"]), tuple(d["ax2"]))


def angle_between(ax, tan) -> float:
    (a1, a2), (t1, t2) = ax, tan
    vh = np.array([a2[0] - a1[0], a2[1] - a1[1]], dtype=float)
    vt = np.array([t2[0] - t1[0], t2[1] - t1[1]], dtype=float)
    nh, nt = np.linalg.norm(vh), np.linalg.norm(vt)
    if nh < 1e-6 or nt < 1e-6:
        return float("nan")
    return math.degrees(math.acos(float(np.clip(np.dot(vh, vt) / (nh * nt), -1.0, 1.0))))


def build_model(ckpt_path: Path, device):
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 4)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    m.load_state_dict(state)
    m.eval().to(device)
    return m


def make_tfm():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def draw_label(img, text, org, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--checkpoint", default=str(MODELS / "tangent_best.pt"))
    p.add_argument("--out-video", default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--axis", default=None,
                   help="Override axis as 'x1,y1,x2,y2'. Otherwise read from dataset/axis/.")
    args = p.parse_args()

    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"video not found: {video}")
    if args.axis:
        a = [float(v) for v in args.axis.split(",")]
        axis = ((a[0], a[1]), (a[2], a[3]))
    else:
        axis = load_axis(video)
        if axis is None:
            raise SystemExit(f"no axis for {video.name}; set one in label_tool.py or pass --axis")

    out_video = Path(args.out_video) if args.out_video else video.with_name(f"{video.stem}_pred.mp4")
    out_csv = Path(args.out_csv) if args.out_csv else video.with_name(f"{video.stem}_pred.csv")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    model = build_model(Path(args.checkpoint), device)
    tfm = make_tfm()

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    ax1, ax2 = axis
    ax1i = (int(round(ax1[0])), int(round(ax1[1])))
    ax2i = (int(round(ax2[0])), int(round(ax2[1])))

    csv_f = out_csv.open("w", newline="")
    cw = csv.writer(csv_f)
    cw.writerow(["frame", "time_s", "angle_deg", "tx1", "ty1", "tx2", "ty2"])

    frame_i = 0
    with torch.no_grad():
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            x = tfm(pil).unsqueeze(0).to(device)
            out = model(x).squeeze(0).cpu().numpy()
            out = np.clip(out, 0.0, 1.0)
            t1 = (out[0] * w, out[1] * h)
            t2 = (out[2] * w, out[3] * h)
            ang = angle_between(axis, (t1, t2))

            cv2.line(bgr, ax1i, ax2i, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(bgr, (int(t1[0]), int(t1[1])), (int(t2[0]), int(t2[1])), (0, 255, 255), 3, cv2.LINE_AA)
            t = frame_i / fps
            draw_label(bgr, f"angle: {ang:.2f} deg", (20, 50))
            draw_label(bgr, f"frame: {frame_i}  t={t:.2f}s", (20, 90))
            writer.write(bgr)
            cw.writerow([frame_i, f"{t:.4f}", f"{ang:.4f}",
                         f"{t1[0]:.2f}", f"{t1[1]:.2f}", f"{t2[0]:.2f}", f"{t2[1]:.2f}"])

            frame_i += 1
            if frame_i % 200 == 0:
                print(f"  {frame_i}/{n}")

    cap.release()
    writer.release()
    csv_f.close()
    print(f"Wrote {out_video}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
