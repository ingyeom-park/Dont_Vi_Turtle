import glob
import json
import math
import os
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "00_models" / "pose_landmarker.task"
DATA_DIR = "00_dataset"
OUT_PATH = "gt_annotations.json"
REF_CVA = 55.0
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/1/"
    "pose_landmarker_heavy.task"
)


def build(path=MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(path):
        urlretrieve(MODEL_URL, path)

    base = python.BaseOptions(model_asset_path=path)
    opts = vision.PoseLandmarkerOptions(
        base_options=base,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(opts)


def calc_angle(p1, p2):
    dy = p1[1] - p2[1]
    dx = abs(p2[0] - p1[0])
    return math.degrees(math.atan2(dy, dx))


def to_pitch(cva):
    deg = REF_CVA - cva
    rad = deg * math.pi / 180.0
    return rad, deg


def get_r(pitch, yaw, roll):
    rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)],
    ])
    ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)],
    ])
    rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1],
    ])
    return rz.dot(ry.dot(rx))


def get_cva(path, det):
    img = cv2.imread(path)
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
    left = (lm[11].x * w, lm[11].y * h)
    right = (lm[12].x * w, lm[12].y * h)
    ear = (lm[7].x * w, lm[7].y * h)
    c7 = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
    return calc_angle(c7, ear)


def get_grade(cva):
    if cva > 50:
        return "정상"
    if cva > 30:
        return "경미한 거북목"
    return "심한 거북목"


def run(path, out_path, det):
    side_paths = sorted(glob.glob(os.path.join(path, "**/*side*.jpg"), recursive=True))
    print(f"측면 사진 {len(side_paths)}장\n")

    rows = []
    miss = 0

    for i, side in enumerate(side_paths, 1):
        front = os.path.join(os.path.dirname(side), os.path.basename(side).replace("_side_", "_front_"))
        cva = get_cva(side, det)

        if cva is None:
            print(f"[{i}] 감지 실패: {os.path.basename(side)}")
            miss += 1
            continue

        pitch_rad, pitch_deg = to_pitch(cva)
        has_front = os.path.exists(front)
        grade = get_grade(cva)

        rows.append({
            "front_image": front if has_front else "정면사진없음",
            "side_image": side,
            "cva_angle": round(cva, 2),
            "pitch_deg": round(pitch_deg, 2),
            "pitch_rad": round(pitch_rad, 4),
            "yaw_deg": 0.0,
            "roll_deg": 0.0,
            "grade": grade,
            "rotation_matrix": get_r(pitch_rad, 0.0, 0.0).tolist(),
            "has_front": has_front,
        })

        mark = "정면O" if has_front else "정면X"
        print(f"[{i}] {os.path.basename(side)}: CVA={cva:.1f} pitch={pitch_deg:.1f} [{grade}] {mark}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    grades = [row["grade"] for row in rows]
    cvas = [row["cva_angle"] for row in rows]

    print(f"\n처리 성공: {len(rows)}장 / 실패: {miss}장")
    print(f"정면 paired: {sum(1 for row in rows if row['has_front'])}장")
    print(f"정상: {grades.count('정상')} / 경미: {grades.count('경미한 거북목')} / 심함: {grades.count('심한 거북목')}")
    if cvas:
        print(f"CVA 범위: {min(cvas):.1f} ~ {max(cvas):.1f} (평균 {np.mean(cvas):.1f})")
    print(f"저장: {out_path}")

    return rows


if __name__ == "__main__":
    det = build(MODEL_PATH)
    run(DATA_DIR, OUT_PATH, det)
