# ESP32-S3 배포 메모 (TuSimple 전용 모델 기준)

## 1. 모델 파일
양자화 결과물 `.espdl` 파일 이름을 `model.espdl`로 바꾸고 아래 위치에 넣습니다.
예: `lane_s3_tusimple_160x96.espdl -> model.espdl`

```text
esp32_s3/main/models/model.espdl
```

## 2. 카메라 핀 수정
`esp32_s3/main/camera_pins.h`를 현재 보드의 카메라 핀맵으로 바꿉니다.

이 프로젝트는 보드를 특정하지 않았기 때문에, **핀맵만 맞추면 나머지 추론 코드는 그대로 재사용**할 수 있게 만들었습니다.

## 3. 빌드
```bash
cd esp32_s3
idf.py set-target esp32s3
idf.py build
idf.py -p <PORT> flash monitor
```

## 4. 로그에서 볼 것
시리얼 로그에 아래 값이 반복 출력됩니다.

- `cap`: 카메라 캡처 시간
- `prep`: ROI crop + resize + float tensor 작성 시간
- `infer`: ESP-DL 추론 시간
- `post`: lane center 계산 시간
- `total`: 전체 프레임 처리 시간
- `offset`: 화면 중심 대비 차선 중심 오프셋

예상 사용법은 다음과 같습니다.

- `offset < 0`: 차량이 차선 중심보다 오른쪽에 있음 → 왼쪽 보정 필요
- `offset > 0`: 차량이 차선 중심보다 왼쪽에 있음 → 오른쪽 보정 필요

## 5. 먼저 맞춰야 하는 것
속도 목표가 빡빡하면 아래를 우선적으로 확인하세요.

1. PSRAM 활성화 여부
2. 240MHz 동작 여부
3. Flash QIO 여부
4. 카메라가 정말 `GRAYSCALE + QVGA`로 들어오는지
5. 모델 입력 크기가 `160x96`인지

## 6. 더 빨리 돌리고 싶을 때
- `160x96` → `128x80`
- `width_mult 1.0` → `0.75`
- 로그를 보고 `prep`가 크면 resize 최적화
- `infer`가 크면 모델 폭 줄이기
