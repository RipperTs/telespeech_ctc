ARG RUNTIME=cpu
ARG BASE_IMAGE=python:3.11-slim

FROM ${BASE_IMAGE}

ARG RUNTIME=cpu
ARG MODEL_VARIANT=float32

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:$PATH \
    MODEL_DIR=/models/telespeech \
    SAMPLE_RATE=16000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      ffmpeg ca-certificates curl bzip2 python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python3 -m venv /opt/venv \
    && pip install --upgrade pip \
    && if [ "${RUNTIME}" = "gpu" ]; then \
        sed '/^sherpa-onnx==/d' requirements.txt > /tmp/requirements.txt \
        && pip install -r /tmp/requirements.txt \
        && pip install sherpa-onnx==1.13.2+cuda12.cudnn9 \
          -f https://k2-fsa.github.io/sherpa/onnx/cuda.html; \
      else \
        pip install -r requirements.txt; \
      fi

RUN set -eux; \
    if [ "${MODEL_VARIANT}" = "int8" ]; then \
      model_name="sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04"; \
      model_file="model.int8.onnx"; \
    elif [ "${MODEL_VARIANT}" = "float32" ]; then \
      model_name="sherpa-onnx-telespeech-ctc-zh-2024-06-04"; \
      model_file="model.onnx"; \
    else \
      echo "Unsupported MODEL_VARIANT=${MODEL_VARIANT}" >&2; \
      exit 1; \
    fi; \
    mkdir -p /models/telespeech /tmp/model; \
    curl -L "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${model_name}.tar.bz2" \
      -o /tmp/model/model.tar.bz2; \
    tar -xjf /tmp/model/model.tar.bz2 -C /tmp/model; \
    cp "/tmp/model/${model_name}/${model_file}" "/models/telespeech/${model_file}"; \
    cp "/tmp/model/${model_name}/tokens.txt" /models/telespeech/tokens.txt; \
    rm -rf /tmp/model

RUN set -eux; \
    if [ "${RUNTIME}" = "gpu" ]; then \
      punct_name="sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12"; \
      punct_file="model.onnx"; \
    else \
      punct_name="sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8"; \
      punct_file="model.int8.onnx"; \
    fi; \
    mkdir -p /models/punctuation /tmp/punctuation; \
    curl -L "https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/${punct_name}.tar.bz2" \
      -o /tmp/punctuation/model.tar.bz2; \
    tar -xjf /tmp/punctuation/model.tar.bz2 -C /tmp/punctuation; \
    cp "/tmp/punctuation/${punct_name}/${punct_file}" "/models/punctuation/${punct_file}"; \
    rm -rf /tmp/punctuation

RUN set -eux; \
    mkdir -p /models/vad; \
    curl -L "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx" \
      -o /models/vad/silero_vad.onnx

COPY app ./app

RUN mkdir -p /app \
    && if [ "${MODEL_VARIANT}" = "int8" ]; then \
      echo "ENCODER_FILE=model.int8.onnx" >> /app/image.env; \
    else \
      echo "ENCODER_FILE=model.onnx" >> /app/image.env; \
    fi; \
    if [ "${RUNTIME}" = "gpu" ]; then \
      echo "MODEL_PROVIDER=cuda" >> /app/image.env; \
      echo "PUNCTUATION_MODEL_FILE=model.onnx" >> /app/image.env; \
    else \
      echo "MODEL_PROVIDER=cpu" >> /app/image.env; \
      echo "PUNCTUATION_MODEL_FILE=model.int8.onnx" >> /app/image.env; \
    fi

ENV ENCODER_FILE=model.onnx \
    MODEL_PROVIDER=cpu \
    PUNCTUATION_MODEL_FILE=model.int8.onnx \
    ENABLE_PUNCTUATION=true \
    VAD_MODEL_FILE=silero_vad.onnx \
    ENABLE_VAD=true

EXPOSE 8000

CMD ["/bin/sh", "-c", "set -a && . /app/image.env && set +a && exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"]
