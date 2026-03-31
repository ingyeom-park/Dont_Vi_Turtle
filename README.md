# Dont_Vi_Turtle

측면 이미지에서 CVA 각도를 뽑아서 GT를 만들고, 정면 이미지를 넣어 CVA를 예측하는 쪽으로 정리한 프로젝트입니다.

크게 보면 흐름은 이렇습니다.

- `01_GT`: GT 생성
- `02_Train`: 학습
- `03_Test`: 검증 / 테스트
- `99_Test`: 포즈 백엔드 비교 실험용

## 폴더 구조

```text
00_dataset
00_models
01_GT
02_Train
03_Test
99_Test
