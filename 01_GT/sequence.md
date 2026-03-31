1. `file_manager.py`

사진 이름 바꾸거나, 폴더 비교하고, 빠진 파일 채울 때 쓴다. 기본 폴더는 `00_dataset`이다.

2. `pose_detector.py`

MediaPipe로 랜드마크가 제대로 잡히는지 눈으로 확인할 때 쓴다.

3. `gt_generator.py`

측면 사진에서 CVA 뽑고 `gt_annotations.json` 만든다.

4. `data_split.py`

GT json을 train/test나 train/val/test로 나눈다.

5. `dataset.py`

GT json이 학습 코드에서 바로 먹히는지 확인하는 용도다.
