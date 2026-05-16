#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-float32}"
TARGET_DIR="${2:-models/telespeech}"

case "${VARIANT}" in
  float32)
    MODEL_NAME="sherpa-onnx-telespeech-ctc-zh-2024-06-04"
    MODEL_FILE="model.onnx"
    ;;
  int8)
    MODEL_NAME="sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04"
    MODEL_FILE="model.int8.onnx"
    ;;
  *)
    echo "Usage: $0 [float32|int8] [target_dir]" >&2
    exit 1
    ;;
esac

MODEL_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${MODEL_NAME}.tar.bz2"
TMP_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

mkdir -p "${TARGET_DIR}"

if [[ -f "${TARGET_DIR}/${MODEL_FILE}" && -f "${TARGET_DIR}/tokens.txt" ]]; then
  echo "${VARIANT} model already exists in ${TARGET_DIR}"
  exit 0
fi

echo "Downloading ${MODEL_NAME}..."
curl -L "${MODEL_URL}" -o "${TMP_DIR}/${MODEL_NAME}.tar.bz2"

echo "Extracting model..."
tar -xjf "${TMP_DIR}/${MODEL_NAME}.tar.bz2" -C "${TMP_DIR}"

cp "${TMP_DIR}/${MODEL_NAME}/${MODEL_FILE}" "${TARGET_DIR}/${MODEL_FILE}"
cp "${TMP_DIR}/${MODEL_NAME}/tokens.txt" "${TARGET_DIR}/tokens.txt"

echo "${VARIANT} model is ready in ${TARGET_DIR}"
