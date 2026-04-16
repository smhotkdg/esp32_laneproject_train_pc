# ESP32-S3용 초경량 차선 인식 프로젝트 - TuSimple 전용판

이 버전은 **TuSimple 데이터셋만 사용**하도록 정리한 프로젝트입니다.
이전의 BDD100K / CULane 다단계 학습 흐름은 제거했고, 아래 순서만 따라가면 됩니다.

1. TuSimple 준비
2. manifest + lane mask 생성
3. 학습
4. ONNX export
5. ESP-DL `.espdl` 양자화
6. PC 웹캠 테스트
7. ESP32-S3 배포

---

## 핵심 설계

- **입력은 기본 grayscale 1채널**
  - ESP32-S3에서 전처리와 메모리 부담을 줄이기 쉽습니다.
- **모델은 tiny segmentation**
  - ESP-DL에 맞추기 쉬운 Conv / Depthwise Conv / Add / Concat / Resize / ReLU 중심 구조입니다.
- **하단 ROI만 사용**
  - 상단 배경을 버리고 차선이 있는 하단 영역에 연산을 집중합니다.
- **TuSimple only**
  - 고속도로/정차선 상황에서 빠르게 시작하기 쉽습니다.

---

## 필요한 TuSimple 데이터만

이 프로젝트는 **TuSimple의 `train_set`만 있어도 동작**합니다.
로컬 검증용 `val`은 `prepare_datasets.py`가 **학습 데이터에서 자동 분리**합니다.

보통 아래 구조 둘 중 하나면 됩니다.

### 방법 A: TuSimple 루트 전체를 지정
```text
TuSimple/
 ├── train_set/
 │    ├── clips/
 │    ├── label_data_0313.json
 │    ├── label_data_0531.json
 │    └── label_data_0601.json
 └── test_set/          # 없어도 됨
```

### 방법 B: `train_set` 폴더만 지정
```text
train_set/
 ├── clips/
 ├── label_data_0313.json
 ├── label_data_0531.json
 └── label_data_0601.json
```

둘 다 지원합니다.

---

## 폴더 설명

- `prepare_datasets.py`
  - TuSimple JSON 라벨을 읽어서 lane mask PNG와 JSONL manifest를 생성합니다.
- `train_lanenet.py`
  - PyTorch 학습
- `export_onnx.py`
  - 학습된 체크포인트를 ONNX로 export
- `quantize_espdl.py`
  - ONNX를 `.espdl`로 양자화
- `webcam_test.py`
  - PC 웹캠 / 동영상 테스트
- `esp32_s3/`
  - ESP-IDF + ESP-DL + esp32-camera 예제

---

## Python 환경 준비

### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Linux / macOS
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 1. TuSimple manifest 생성

### TuSimple 루트를 지정하는 경우
```bash
python prepare_datasets.py \
  --tusimple-root D:/datasets/TuSimple \
  --out ./prepared_data
```

### `train_set`만 지정하는 경우
```bash
python prepare_datasets.py \
  --tusimple-root D:/datasets/TuSimple/train_set \
  --out ./prepared_data
```

생성 결과:
```text
prepared_data/
 ├── manifests/
 │    ├── tusimple_train.jsonl
 │    └── tusimple_val.jsonl
 ├── masks/
 │    └── tusimple/
 └── tusimple_stats.json
```

`tuSimple_stats.json`에는 train/val 개수와 누락 이미지 수가 기록됩니다.

---

## 2. 학습

TuSimple only 흐름에서는 기본적으로 **drivable head를 쓰지 않습니다.**
즉, 아래 명령에서 `--aux-drivable`는 넣지 않습니다.

### 기본 권장 학습
```bash
python train_lanenet.py \
  --train-manifest prepared_data/manifests/tusimple_train.jsonl \
  --val-manifest prepared_data/manifests/tusimple_val.jsonl \
  --save-dir runs/tusimple_only \
  --epochs 80 \
  --batch-size 16 \
  --input-channels 1 \
  --input-width 160 \
  --input-height 96 \
  --width-mult 1.0
```

주요 결과물:
```text
runs/tusimple_only/
 ├── best.pt
 ├── last.pt
 ├── history.json
 └── meta.json
```

### 더 빠르게 돌리고 싶을 때
```bash
python train_lanenet.py \
  --train-manifest prepared_data/manifests/tusimple_train.jsonl \
  --val-manifest prepared_data/manifests/tusimple_val.jsonl \
  --save-dir runs/tusimple_fast \
  --epochs 60 \
  --batch-size 16 \
  --input-channels 1 \
  --input-width 128 \
  --input-height 80 \
  --width-mult 0.75
```

### 정확도를 더 보고 싶을 때
아래 순서로 올려보면 됩니다.

1. `epochs 80 -> 100`
2. `160x96 -> 192x112`
3. `input_channels 1 -> 3`
4. `width_mult 1.0 -> 1.25`

단, ESP32-S3 속도 목표가 빡빡하면 먼저 `160x96 / grayscale / width_mult 1.0`부터 확인하는 것이 안전합니다.

---

## 3. PC 웹캠 테스트

### PyTorch 체크포인트로 테스트
```bash
python webcam_test.py --weights runs/tusimple_only/best.pt
```

### ONNX로 테스트
```bash
python webcam_test.py --weights runs/tusimple_only/lane_s3_tusimple_160x96.onnx
```

종료:
- `q`
- `ESC`

표시 내용:
- lane mask overlay
- 좌/우 차선 피팅 결과
- lane center offset
- FPS

---

## 4. ONNX export

```bash
python export_onnx.py \
  --checkpoint runs/tusimple_only/best.pt \
  --output runs/tusimple_only/lane_s3_tusimple_160x96.onnx \
  --input-channels 1 \
  --input-width 160 \
  --input-height 96
```

---

## 5. ESP-DL `.espdl` 양자화

```bash
python quantize_espdl.py \
  --onnx runs/tusimple_only/lane_s3_tusimple_160x96.onnx \
  --calib-manifest prepared_data/manifests/tusimple_val.jsonl \
  --output runs/tusimple_only/lane_s3_tusimple_160x96.espdl \
  --input-channels 1 \
  --input-width 160 \
  --input-height 96
```

---

## 6. ESP32-S3 배포

생성된 파일을 아래로 복사합니다.

```text
esp32_s3/main/models/model.espdl
```

그다음:
1. `esp32_s3/main/camera_pins.h`를 보드 핀맵에 맞게 수정
2. ESP-IDF로 빌드

```bash
cd esp32_s3
idf.py set-target esp32s3
idf.py build
idf.py -p COM3 flash monitor
```

---

## 7. 한 번에 돌리는 스크립트

`run_pipeline.sh`도 TuSimple 전용으로 수정되어 있습니다.
경로만 바꾼 뒤 실행하면 됩니다.

```bash
bash run_pipeline.sh
```

---

## 8. 추천 시작점

가장 무난한 시작 설정:

- dataset: **TuSimple only**
- input: **160x96**
- channels: **1**
- width multiplier: **1.0**
- epochs: **80**

이 설정으로 먼저 `best.pt`를 만든 뒤,
PC 웹캠에서 확인하고,
그다음 ONNX / `.espdl` 변환으로 가는 흐름을 권장합니다.

---

## 9. 주의할 점

- TuSimple only 학습은 **고속도로형 차선**에는 잘 맞지만, 도심/교차로/강한 가림에는 일반화가 약할 수 있습니다.
- 속도가 부족하면 해상도부터 줄이고,
- 정확도가 부족하면 epochs / 해상도 / RGB 입력 순으로 늘려보는 편이 안전합니다.
- `prepare_datasets.py`는 **TuSimple 루트와 `train_set` 둘 다 처리**하도록 수정되어 있습니다.

---

## 10. 가장 빠른 실행 순서

```bash
python prepare_datasets.py --tusimple-root D:/datasets/TuSimple --out ./prepared_data
python train_lanenet.py --train-manifest prepared_data/manifests/tusimple_train.jsonl --val-manifest prepared_data/manifests/tusimple_val.jsonl --save-dir runs/tusimple_only --epochs 80 --batch-size 16 --input-channels 1 --input-width 160 --input-height 96 --width-mult 1.0
python export_onnx.py --checkpoint runs/tusimple_only/best.pt --output runs/tusimple_only/lane_s3_tusimple_160x96.onnx --input-channels 1 --input-width 160 --input-height 96
python quantize_espdl.py --onnx runs/tusimple_only/lane_s3_tusimple_160x96.onnx --calib-manifest prepared_data/manifests/tusimple_val.jsonl --output runs/tusimple_only/lane_s3_tusimple_160x96.espdl --input-channels 1 --input-width 160 --input-height 96
python webcam_test.py --weights runs/tusimple_only/best.pt
```
