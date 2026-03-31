> 1. file_manager.py  

데이터셋 파일 정리 유틸리티. 파일 이름 변경, 폴더 간 비교/동기화, 중복 파일 탐색 등 dataset/ 폴더를 정리할 때 사용.

> 2. pose_detector.py  

MediaPipe로 이미지에서 포즈(코, 어깨)를 감지하고 시각화. 데이터셋 사진에 랜드마크가 제대로 잡히는지 확인하는 테스트용.

> 3. gt_generator.py

1. 측면사진(side)에서 CVA 각도 계산  
2. pitch로 변환
3. 회전행렬 생성
4. gt_annotations.json으로 저장

> 4. data_split.py    

gt_annotations.json을 8:2로 나눠 GT_train.json / GT_test.json 생성.

> 5. dataset.py

GT_train.json / GT_test.json을 읽어 PyTorch Dataset 형태로 제공. 모델 학습 코드에서 import해서 사용.