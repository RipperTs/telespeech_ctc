# TeleSpeech CTC OpenAI-Compatible ASR API

基于 `sherpa-onnx` 的 TeleSpeech CTC 模型，提供兼容 OpenAI/SiliconFlow 风格的语音转写接口。

## 镜像版本

镜像仓库：

```text
registry.cn-hangzhou.aliyuncs.com/ripper/telespeech-ctc
```

| Tag | 模型 | 运行环境 | 说明 |
| --- | --- | --- | --- |
| `latest` | float32 | GPU | 默认版本，等同 `float32-gpu` |
| `float32-gpu` | float32 | GPU, CUDA 12.8 + CUDNN9 | ASR float32，标点 float32 |
| `int8-gpu` | int8 | GPU, CUDA 12.8 + CUDNN9 | ASR int8，标点 float32 |
| `int8-cpu` | int8 | CPU | ASR int8，标点 int8 |

ASR 模型在构建镜像时已经打包到 `/models/telespeech`，标点模型打包到 `/models/punctuation`，VAD 模型打包到 `/models/vad`，运行时不需要再次下载。

## 接口

```bash
curl --request POST \
  --url http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer your-api-key" \
  -F "file=@path/to/your/audio.mp3" \
  -F "model=TeleAI/TeleSpeechASR"
```

返回：

```json
{
  "text": "识别结果"
}
```

未配置 `ASR_API_KEY` 时，不校验 `Authorization`。

## Docker 运行

GPU 默认版本：

```bash
docker run --rm \
  --gpus all \
  -p 8000:8000 \
  -e ASR_API_KEY=your-api-key \
  registry.cn-hangzhou.aliyuncs.com/ripper/telespeech-ctc:latest
```

CPU int8 版本：

```bash
docker run --rm \
  -p 8000:8000 \
  -e ASR_API_KEY=your-api-key \
  registry.cn-hangzhou.aliyuncs.com/ripper/telespeech-ctc:int8-cpu
```

使用 compose：

```bash
ASR_API_KEY=your-api-key docker compose up -d
```

## 本地开发

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_model.sh int8
ASR_API_KEY=your-api-key \
MODEL_DIR=models/telespeech \
ENCODER_FILE=model.int8.onnx \
MODEL_PROVIDER=cpu \
uvicorn app.main:app --reload
```

下载 float32 模型：

```bash
bash scripts/download_model.sh float32
```

## 本地构建镜像

CPU int8：

```bash
docker build \
  --build-arg RUNTIME=cpu \
  --build-arg BASE_IMAGE=python:3.11-slim \
  --build-arg MODEL_VARIANT=int8 \
  -t registry.cn-hangzhou.aliyuncs.com/ripper/telespeech-ctc:int8-cpu .
```

GPU float32：

```bash
docker build \
  --build-arg RUNTIME=gpu \
  --build-arg BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 \
  --build-arg MODEL_VARIANT=float32 \
  -t registry.cn-hangzhou.aliyuncs.com/ripper/telespeech-ctc:float32-gpu .
```

## GitHub Actions 发布

推送版本 tag 后自动构建并推送到阿里云镜像仓库：

```bash
git tag v1.0.0
git push origin v1.0.0
```

需要在 GitHub 仓库配置 Secrets：

| Secret | 说明 |
| --- | --- |
| `ALIYUN_REGISTRY_USERNAME` | 阿里云镜像仓库用户名 |
| `ALIYUN_REGISTRY_PASSWORD` | 阿里云镜像仓库密码 |

发布 tag：

- `latest`
- `float32-gpu`
- `int8-gpu`
- `int8-cpu`
- `v1.0.0-float32-gpu`
- `v1.0.0-int8-gpu`
- `v1.0.0-int8-cpu`

## 并发配置

默认配置偏稳妥：

- `RECOGNIZER_INSTANCES=1`：模型实例数，增加会提高并发但占更多内存
- `RECOGNIZER_THREADS=2`：单个模型实例的 ONNX 推理线程数
- `INFERENCE_WORKERS=2`：推理线程池大小
- `MAX_PENDING_REQUESTS=10`：允许等待的请求数，超过后返回 429
- `CHUNK_SECONDS=30`：长音频内部切片秒数，避免整段推理内存过高
- `ENABLE_VAD=true`：启用 Silero VAD 按静音切段，失败时回退固定切片
- `ENABLE_PUNCTUATION=true`：启用标点恢复

建议从默认值开始压测。GPU 版本优先增加 `RECOGNIZER_INSTANCES` 前，需要关注显存占用。

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ASR_API_KEY` | 空 | 配置后启用 Bearer Token 校验 |
| `MODEL_DIR` | `/models/telespeech` | 模型目录 |
| `ENCODER_FILE` | 镜像内置 | `model.onnx` 或 `model.int8.onnx` |
| `MODEL_PROVIDER` | 镜像内置 | `cuda` 或 `cpu` |
| `ENABLE_PUNCTUATION` | `true` | 是否启用标点恢复 |
| `PUNCTUATION_MODEL_DIR` | `/models/punctuation` | 标点模型目录 |
| `PUNCTUATION_MODEL_FILE` | 镜像内置 | GPU 为 `model.onnx`，CPU 为 `model.int8.onnx` |
| `ENABLE_VAD` | `true` | 是否启用 VAD 切段 |
| `VAD_MODEL_DIR` | `/models/vad` | VAD 模型目录 |
| `VAD_MODEL_FILE` | `silero_vad.onnx` | VAD 模型文件 |
| `VAD_THRESHOLD` | `0.2` | VAD 语音检测阈值 |
| `VAD_MIN_SILENCE_SECONDS` | `0.25` | 静音切段阈值 |
| `VAD_MIN_SPEECH_SECONDS` | `0.25` | 最短语音段阈值 |
| `MIN_SEGMENT_SECONDS` | `1.0` | 小于该时长的片段不会送入 ASR |
| `CHUNK_SECONDS` | `30` | 长音频内部切片秒数 |
| `MAX_UPLOAD_MB` | `200` | 单个音频最大上传大小 |
| `REQUEST_TIMEOUT_SECONDS` | `300` | 单次识别超时时间 |
| `MAX_PENDING_REQUESTS` | `10` | 最多允许等待的转写请求数 |
| `ALLOWED_MODELS` | `TeleAI/TeleSpeechASR,FunAudioLLM/SenseVoiceSmall,telespeech-ctc,whisper-1` | 接口允许的 model 参数 |

## 说明

TeleSpeech CTC 是中文/方言 ASR 模型。当前接口只实现转写能力，`language`、`prompt`、`response_format`、`temperature` 等 OpenAI 常见参数会被兼容接收，但返回始终保持：

```json
{"text":"..."}
```
