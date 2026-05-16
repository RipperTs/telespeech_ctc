#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04"
MODEL_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${MODEL_NAME}.tar.bz2"
TARGET_DIR="${1:-models/telespeech}"
TMP_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

mkdir -p "${TARGET_DIR}"

if [[ -f "${TARGET_DIR}/model.int8.onnx" && -f "${TARGET_DIR}/tokens.txt" ]]; then
  echo "Model already exists in ${TARGET_DIR}"
  exit 0
fi

echo "Downloading ${MODEL_NAME}..."
curl -L "${MODEL_URL}" -o "${TMP_DIR}/${MODEL_NAME}.tar.bz2"

echo "Extracting model..."
tar -xjf "${TMP_DIR}/${MODEL_NAME}.tar.bz2" -C "${TMP_DIR}"

cp "${TMP_DIR}/${MODEL_NAME}/model.int8.onnx" "${TARGET_DIR}/model.int8.onnx"
cp "${TMP_DIR}/${MODEL_NAME}/tokens.txt" "${TARGET_DIR}/tokens.txt"

echo "Model is ready in ${TARGET_DIR}"
