import glob
import os
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "00_models" / "pose_landmarker.task"
DATA_DIR = "00_dataset"
MAX_SIZE = 800
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


def run(path, det, show=True, max_size=MAX_SIZE):
    img = cv2.imread(path)
    if img is None:
        print(f"파일 없음: {path}")
        return None

    h, w = img.shape[:2]
    if max(h, w) > max_size:
        rate = max_size / max(h, w)
        img = cv2.resize(img, (int(w * rate), int(h * rate)))

    h, w = img.shape[:2]
    mp_img = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
    )
    res = det.detect(mp_img)
    if not res.pose_landmarks:
        print("포즈 감지 실패")
        return None

    lm = res.pose_landmarks[0]
    nose = (int(lm[0].x * w), int(lm[0].y * h))
    left = (int(lm[11].x * w), int(lm[11].y * h))
    right = (int(lm[12].x * w), int(lm[12].y * h))

    print(f"코:       X={nose[0]}, Y={nose[1]}")
    print(f"왼쪽 어깨: X={left[0]}, Y={left[1]}")
    print(f"오른쪽 어깨: X={right[0]}, Y={right[1]}")

    cv2.circle(img, nose, 6, (0, 0, 255), -1)
    cv2.circle(img, left, 6, (0, 255, 0), -1)
    cv2.circle(img, right, 6, (255, 0, 0), -1)
    cv2.putText(img, f"Face:{nose}", (nose[0] + 15, nose[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(img, f"L:{left}", (left[0] + 15, left[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img, f"R:{right}", (right[0] + 15, right[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if show:
        cv2.imshow("Pose Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        "nose": nose,
        "left_shoulder": left,
        "right_shoulder": right,
    }


def run_dir(path, det, show=True):
    paths = sorted(glob.glob(os.path.join(path, "**/*.jpg"), recursive=True))
    print(f"총 {len(paths)}장\n")

    for i, img_path in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] {os.path.basename(img_path)}")
        run(img_path, det, show=show)


if __name__ == "__main__":
    det = build(MODEL_PATH)
    run_dir(DATA_DIR, det, show=True)
