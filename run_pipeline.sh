#!/usr/bin/env bash
set -euo pipefail

# TuSimple-only end-to-end pipeline.
# Update TUSIMPLE_ROOT first.

TUSIMPLE_ROOT=/data/TuSimple
PREP=./prepared_data
RUNS=./runs
RUN_NAME=tusimple_only

python prepare_datasets.py \
  --tusimple-root "${TUSIMPLE_ROOT}" \
  --out "${PREP}"

python train_lanenet.py \
  --train-manifest "${PREP}/manifests/tusimple_train.jsonl" \
  --val-manifest "${PREP}/manifests/tusimple_val.jsonl" \
  --save-dir "${RUNS}/${RUN_NAME}" \
  --epochs 80 \
  --batch-size 16 \
  --input-channels 1 \
  --input-width 160 \
  --input-height 96 \
  --width-mult 1.0

python export_onnx.py \
  --checkpoint "${RUNS}/${RUN_NAME}/best.pt" \
  --output "${RUNS}/${RUN_NAME}/lane_s3_tusimple_160x96.onnx" \
  --input-channels 1 \
  --input-width 160 \
  --input-height 96

python quantize_espdl.py \
  --onnx "${RUNS}/${RUN_NAME}/lane_s3_tusimple_160x96.onnx" \
  --calib-manifest "${PREP}/manifests/tusimple_val.jsonl" \
  --output "${RUNS}/${RUN_NAME}/lane_s3_tusimple_160x96.espdl" \
  --input-channels 1 \
  --input-width 160 \
  --input-height 96

python webcam_test.py --weights "${RUNS}/${RUN_NAME}/best.pt"
