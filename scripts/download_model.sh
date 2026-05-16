#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-float32}"
TARGET_DIR="${2:-models/telespeech}"
PUNCTUATION_DIR="${3:-models/punctuation}"

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

PUNCTUATION_NAME="sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12"
PUNCTUATION_FILE="model.onnx"

if [[ "${VARIANT}" = "int8" ]]; then
  PUNCTUATION_NAME="${PUNCTUATION_NAME}-int8"
  PUNCTUATION_FILE="model.int8.onnx"
fi

PUNCTUATION_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/${PUNCTUATION_NAME}.tar.bz2"
mkdir -p "${PUNCTUATION_DIR}"

if [[ -f "${PUNCTUATION_DIR}/${PUNCTUATION_FILE}" ]]; then
  echo "${VARIANT} punctuation model already exists in ${PUNCTUATION_DIR}"
  exit 0
fi

echo "Downloading ${PUNCTUATION_NAME}..."
curl -L "${PUNCTUATION_URL}" -o "${TMP_DIR}/${PUNCTUATION_NAME}.tar.bz2"

echo "Extracting punctuation model..."
tar -xjf "${TMP_DIR}/${PUNCTUATION_NAME}.tar.bz2" -C "${TMP_DIR}"

cp "${TMP_DIR}/${PUNCTUATION_NAME}/${PUNCTUATION_FILE}" "${PUNCTUATION_DIR}/${PUNCTUATION_FILE}"

echo "${VARIANT} punctuation model is ready in ${PUNCTUATION_DIR}"

VAD_DIR="${4:-models/vad}"
mkdir -p "${VAD_DIR}"

if [[ -f "${VAD_DIR}/silero_vad.onnx" ]]; then
  echo "VAD model already exists in ${VAD_DIR}"
  exit 0
fi

echo "Downloading silero_vad.onnx..."
curl -L "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx" \
  -o "${VAD_DIR}/silero_vad.onnx"

echo "VAD model is ready in ${VAD_DIR}"
