import os
import cv2
import glob
import json
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH    = 'pose_landmarker.task'
PHOTO_FOLDER  = 'dataset'
OUTPUT_JSON   = 'gt_annotations.json'
REFERENCE_CVA = 55.0


def setup_pose_detector(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        os.system(
            f'wget -q -O {model_path} '
            'https://storage.googleapis.com/mediapipe-models/'
            'pose_landmarker/pose_landmarker_heavy/float16/1/'
            'pose_landmarker_heavy.task'
        )
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(options)


def calculate_angle_2d(p1, p2):
    dy = p1[1] - p2[1]
    dx = abs(p2[0] - p1[0])
    return math.degrees(math.atan2(dy, dx))


def cva_to_pitch(cva_angle):
    pitch_deg = REFERENCE_CVA - cva_angle
    pitch_rad = pitch_deg * math.pi / 180.0
    return pitch_rad, pitch_deg


def get_R(pitch, yaw, roll):
    Rx = np.array([
        [1,             0,              0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch),  np.cos(pitch)],
    ])
    Ry = np.array([
        [ np.cos(yaw), 0, np.sin(yaw)],
        [           0, 1,           0],
        [-np.sin(yaw), 0, np.cos(yaw)],
    ])
    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll),  np.cos(roll), 0],
        [           0,             0, 1],
    ])
    return Rz.dot(Ry.dot(Rx))


def extract_cva(img_path, pose_detector):
    img = cv2.imread(img_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
    )
    results = pose_detector.detect(mp_image)
    if not results.pose_landmarks:
        return None

    lm  = results.pose_landmarks[0]
    ls  = (lm[11].x * w, lm[11].y * h)
    rs  = (lm[12].x * w, lm[12].y * h)
    ear = (lm[7].x  * w, lm[7].y  * h)
    c7  = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)

    return calculate_angle_2d(c7, ear)


def classify_grade(cva):
    if cva > 50:
        return '정상'
    elif cva > 30:
        return '경미한 거북목'
    else:
        return '심한 거북목'


def generate_gt_data(photo_folder, output_json, pose_detector):
    side_images = sorted(glob.glob(os.path.join(photo_folder, '**/*side*.jpg'), recursive=True))
    print(f"측면 사진 {len(side_images)}장\n")

    gt_data    = []
    skip_count = 0

    for i, side_path in enumerate(side_images):
        filename   = os.path.basename(side_path).replace('_side_', '_front_')
        front_path = os.path.join(os.path.dirname(side_path), filename)
        has_front  = os.path.exists(front_path)

        cva = extract_cva(side_path, pose_detector)
        if cva is None:
            print(f"  [{i+1}] 감지 실패: {os.path.basename(side_path)}")
            skip_count += 1
            continue

        pitch_rad, pitch_deg = cva_to_pitch(cva)
        R     = get_R(pitch_rad, 0.0, 0.0)
        grade = classify_grade(cva)

        gt_data.append({
            'front_image':     front_path if has_front else '정면사진없음',
            'side_image':      side_path,
            'cva_angle':       round(cva, 2),
            'pitch_deg':       round(pitch_deg, 2),
            'pitch_rad':       round(pitch_rad, 4),
            'yaw_deg':         0.0,
            'roll_deg':        0.0,
            'grade':           grade,
            'rotation_matrix': R.tolist(),
            'has_front':       has_front,
        })

        print(f"  [{i+1}] {os.path.basename(side_path)}: CVA={cva:.1f} pitch={pitch_deg:.1f} [{grade}] {'정면O' if has_front else '정면X'}")

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(gt_data, f, indent=2, ensure_ascii=False)

    grades = [d['grade'] for d in gt_data]
    cvas   = [d['cva_angle'] for d in gt_data]

    print(f"\n처리 성공: {len(gt_data)}장 / 실패: {skip_count}장")
    print(f"정면 paired: {sum(1 for d in gt_data if d['has_front'])}장")
    print(f"정상: {grades.count('정상')} / 경미: {grades.count('경미한 거북목')} / 심함: {grades.count('심한 거북목')}")
    if cvas:
        print(f"CVA 범위: {min(cvas):.1f} ~ {max(cvas):.1f} (평균 {np.mean(cvas):.1f})")
    print(f"저장: {output_json}")

    return gt_data


if __name__ == '__main__':
    detector = setup_pose_detector(MODEL_PATH)
    generate_gt_data(PHOTO_FOLDER, OUTPUT_JSON, detector)
