import os
import cv2
import glob
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH   = 'pose_landmarker.task'
PHOTO_FOLDER = 'dataset'
MAX_IMG_SIZE = 800


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


def process_pose_image(img_path, pose_detector, max_size=MAX_IMG_SIZE, show=True):
    img = cv2.imread(img_path)
    if img is None:
        print(f"파일 없음: {img_path}")
        return None

    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    h, w = img.shape[:2]
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
    )
    results = pose_detector.detect(mp_image)

    if not results.pose_landmarks:
        print("포즈 감지 실패")
        return None

    landmarks = results.pose_landmarks[0]

    nx,  ny  = int(landmarks[0].x  * w), int(landmarks[0].y  * h)
    lsx, lsy = int(landmarks[11].x * w), int(landmarks[11].y * h)
    rsx, rsy = int(landmarks[12].x * w), int(landmarks[12].y * h)

    print(f"코:       X={nx},  Y={ny}")
    print(f"왼쪽 어깨: X={lsx}, Y={lsy}")
    print(f"오른쪽 어깨: X={rsx}, Y={rsy}")

    cv2.circle(img, (nx,  ny),  6, (0,   0,   255), -1)
    cv2.circle(img, (lsx, lsy), 6, (0,   255,   0), -1)
    cv2.circle(img, (rsx, rsy), 6, (255,   0,   0), -1)
    cv2.putText(img, f"Face:({nx},{ny})",         (nx  + 15, ny),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,   0,   255), 2)
    cv2.putText(img, f"L_Shoulder:({lsx},{lsy})", (lsx + 15, lsy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,   255,   0), 2)
    cv2.putText(img, f"R_Shoulder:({rsx},{rsy})", (rsx + 15, rsy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,   0,   0), 2)

    if show:
        cv2.imshow('Pose Detection', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        'nose':           (nx,  ny),
        'left_shoulder':  (lsx, lsy),
        'right_shoulder': (rsx, rsy),
    }


def process_folder(folder_path, pose_detector, show=True):
    all_images = sorted(glob.glob(os.path.join(folder_path, '**/*.jpg'), recursive=True))
    print(f"총 {len(all_images)}장\n")

    for i, img_path in enumerate(all_images):
        print(f"\n[{i+1}/{len(all_images)}] {os.path.basename(img_path)}")
        process_pose_image(img_path, pose_detector, show=show)


if __name__ == '__main__':
    detector = setup_pose_detector(MODEL_PATH)
    process_folder(PHOTO_FOLDER, detector, show=True)
