import csv
import argparse
import math
import os
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
OUT_DIR = BASE_DIR / "output"
CSV_PATH = OUT_DIR / "pose_compare.csv"
MP_MODEL_PATH = ROOT_DIR / "00_models" / "pose_landmarker.task"
MP_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/1/"
    "pose_landmarker_heavy.task"
)


def calc_angle(p1, p2):
    dy = p1[1] - p2[1]
    dx = abs(p2[0] - p1[0])
    return math.degrees(math.atan2(dy, dx))


def build_mediapipe(path=MP_MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urlretrieve(MP_URL, path)

    base = python.BaseOptions(model_asset_path=str(path))
    opts = vision.PoseLandmarkerOptions(
        base_options=base,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(opts)


def build_mmpose(device):
    from mmpose.apis import MMPoseInferencer

    return MMPoseInferencer("human", device=device)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["all", "mediapipe", "mmpose"], default="all")
    return p.parse_args()


def pick_side(img_name, left_score, right_score):
    name = img_name.lower()
    if "_10_" in name or "_20_" in name or "_40_" in name:
        return "left"
    if "_60_" in name or "_61_" in name or "_81_" in name:
        return "right"
    return "left" if left_score >= right_score else "right"


def pick_instance(items):
    if not items:
        return None

    def score(row):
        vals = row.get("keypoint_scores") or []
        if not vals:
            return 0.0
        return float(np.mean(vals))

    return max(items, key=score)


def mp_points(path, det):
    img = cv2.imread(str(path))
    if img is None:
        return None

    h, w = img.shape[:2]
    mp_img = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
    )
    res = det.detect(mp_img)
    if not res.pose_landmarks:
        return None

    lm = res.pose_landmarks[0]
    vals = {
        "left_ear": (lm[7].x * w, lm[7].y * h, lm[7].visibility),
        "right_ear": (lm[8].x * w, lm[8].y * h, lm[8].visibility),
        "left_shoulder": (lm[11].x * w, lm[11].y * h, lm[11].visibility),
        "right_shoulder": (lm[12].x * w, lm[12].y * h, lm[12].visibility),
    }
    return vals


def mm_points(path, infer):
    gen = infer(str(path), return_vis=False, show=False)
    out = next(gen)
    preds = out.get("predictions") or []
    if not preds:
        return None

    items = preds[0] if isinstance(preds[0], list) else preds
    row = pick_instance(items)
    if row is None:
        return None

    pts = row.get("keypoints") or []
    scores = row.get("keypoint_scores") or []
    if len(pts) < 7 or len(scores) < 7:
        return None

    def get(idx):
        x, y = pts[idx][:2]
        s = scores[idx]
        return float(x), float(y), float(s)

    return {
        "left_ear": get(3),
        "right_ear": get(4),
        "left_shoulder": get(5),
        "right_shoulder": get(6),
    }


def make_row(backend, img_path, pts, vis_path):
    le = pts["left_ear"]
    re = pts["right_ear"]
    ls = pts["left_shoulder"]
    rs = pts["right_shoulder"]

    side = pick_side(img_path.name, le[2], re[2])
    ear = le if side == "left" else re
    c7 = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
    cva = calc_angle(c7, ear[:2])

    return {
        "backend": backend,
        "image": img_path.name,
        "status": "ok",
        "ear_side": side,
        "cva_angle": round(cva, 4),
        "ear_x": round(ear[0], 2),
        "ear_y": round(ear[1], 2),
        "ear_score": round(float(ear[2]), 4),
        "c7_x": round(c7[0], 2),
        "c7_y": round(c7[1], 2),
        "left_ear_x": round(le[0], 2),
        "left_ear_y": round(le[1], 2),
        "left_ear_score": round(float(le[2]), 4),
        "right_ear_x": round(re[0], 2),
        "right_ear_y": round(re[1], 2),
        "right_ear_score": round(float(re[2]), 4),
        "left_shoulder_x": round(ls[0], 2),
        "left_shoulder_y": round(ls[1], 2),
        "left_shoulder_score": round(float(ls[2]), 4),
        "right_shoulder_x": round(rs[0], 2),
        "right_shoulder_y": round(rs[1], 2),
        "right_shoulder_score": round(float(rs[2]), 4),
        "vis_path": str(vis_path),
    }


def draw(path, row, out_path):
    img = cv2.imread(str(path))
    if img is None:
        return

    left_ear = (int(row["left_ear_x"]), int(row["left_ear_y"]))
    right_ear = (int(row["right_ear_x"]), int(row["right_ear_y"]))
    left_sh = (int(row["left_shoulder_x"]), int(row["left_shoulder_y"]))
    right_sh = (int(row["right_shoulder_x"]), int(row["right_shoulder_y"]))
    ear = (int(row["ear_x"]), int(row["ear_y"]))
    c7 = (int(row["c7_x"]), int(row["c7_y"]))

    cv2.circle(img, left_ear, 6, (0, 255, 255), -1)
    cv2.circle(img, right_ear, 6, (0, 200, 200), -1)
    cv2.circle(img, left_sh, 6, (0, 255, 0), -1)
    cv2.circle(img, right_sh, 6, (255, 0, 0), -1)
    cv2.circle(img, ear, 8, (0, 0, 255), -1)
    cv2.circle(img, c7, 8, (255, 0, 255), -1)
    cv2.line(img, c7, ear, (255, 255, 255), 2)

    cv2.putText(img, f"{row['backend']}  CVA={row['cva_angle']:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f"ear={row['ear_side']}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, "C7 approx", (c7[0] + 10, c7[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.putText(img, "ear", (ear[0] + 10, ear[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def fail_row(backend, img_path, vis_path, status):
    return {
        "backend": backend,
        "image": img_path.name,
        "status": status,
        "ear_side": "",
        "cva_angle": "",
        "ear_x": "",
        "ear_y": "",
        "ear_score": "",
        "c7_x": "",
        "c7_y": "",
        "left_ear_x": "",
        "left_ear_y": "",
        "left_ear_score": "",
        "right_ear_x": "",
        "right_ear_y": "",
        "right_ear_score": "",
        "left_shoulder_x": "",
        "left_shoulder_y": "",
        "left_shoulder_score": "",
        "right_shoulder_x": "",
        "right_shoulder_y": "",
        "right_shoulder_score": "",
        "vis_path": str(vis_path),
    }


def save_csv(path, rows):
    cols = [
        "backend",
        "image",
        "status",
        "ear_side",
        "cva_angle",
        "ear_x",
        "ear_y",
        "ear_score",
        "c7_x",
        "c7_y",
        "left_ear_x",
        "left_ear_y",
        "left_ear_score",
        "right_ear_x",
        "right_ear_y",
        "right_ear_score",
        "left_shoulder_x",
        "left_shoulder_y",
        "left_shoulder_score",
        "right_shoulder_x",
        "right_shoulder_y",
        "right_shoulder_score",
        "vis_path",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def run_backend(name, paths, fn, out_dir):
    rows = []
    for img_path in paths:
        vis_path = out_dir / f"{img_path.stem}_{name}.jpg"
        pts = fn(img_path)
        if pts is None:
            rows.append(fail_row(name, img_path, vis_path, "no_pose"))
            print(f"[{name}] fail: {img_path.name}")
            continue

        row = make_row(name, img_path, pts, vis_path)
        draw(img_path, row, vis_path)
        rows.append(row)
        print(f"[{name}] {img_path.name} -> {row['cva_angle']:.2f}")

    return rows


if __name__ == "__main__":
    args = parse()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(DATA_DIR.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"no images in {DATA_DIR}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    rows = []

    if args.backend in ("all", "mediapipe"):
        mp_det = build_mediapipe()
        rows += run_backend(
            "mediapipe_heavy",
            paths,
            lambda path: mp_points(path, mp_det),
            OUT_DIR / "mediapipe_heavy",
        )

    if args.backend in ("all", "mmpose"):
        try:
            mm_det = build_mmpose(device)
        except Exception as e:
            print(f"mmpose skipped: {e}")
        else:
            rows += run_backend(
                "mmpose",
                paths,
                lambda path: mm_points(path, mm_det),
                OUT_DIR / "mmpose",
            )

    save_csv(CSV_PATH, rows)
    print(f"saved: {CSV_PATH}")
