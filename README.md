# TeleSpeech CTC OpenAI-Compatible ASR API

基于 `sherpa-onnx` 的 TeleSpeech CTC 模型，提供兼容 OpenAI/SiliconFlow 风格的语音转写接口。

## 接口

```bash
curl --request POST \
  --url http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer your-api-key" \
  -F "file=@path/to/your/audio.mp3" \
  -F "model=FunAudioLLM/SenseVoiceSmall"
```

返回：

```json
{
  "text": "识别结果"
}
```

## 本地运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_model.sh
ASR_API_KEY=your-api-key MODEL_DIR=models/telespeech uvicorn app.main:app --reload
```

未配置 `ASR_API_KEY` 时，不校验 `Authorization`。

## Docker 部署

```bash
bash scripts/download_model.sh
ASR_API_KEY=your-api-key docker compose up --build -d
```

模型默认挂载到容器内 `/models/telespeech`，需要包含：

- `model.int8.onnx`
- `tokens.txt`

## 并发配置

默认配置偏稳妥：

- `RECOGNIZER_INSTANCES=1`：模型实例数，增加会提高并发但占更多内存
- `RECOGNIZER_THREADS=2`：单个模型实例的 ONNX 推理线程数
- `INFERENCE_WORKERS=2`：推理线程池大小

建议从默认值开始压测。CPU 核数充足时，可优先增加 `RECOGNIZER_INSTANCES`，同时注意内存占用。

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ASR_API_KEY` | 空 | 配置后启用 Bearer Token 校验 |
| `MODEL_DIR` | `/models/telespeech` | 模型目录 |
| `MAX_UPLOAD_MB` | `200` | 单个音频最大上传大小 |
| `REQUEST_TIMEOUT_SECONDS` | `300` | 单次识别超时时间 |
| `ALLOWED_MODELS` | `FunAudioLLM/SenseVoiceSmall,telespeech-ctc,whisper-1` | 接口允许的 model 参数 |

## 说明

TeleSpeech CTC 是中文/方言 ASR 模型。当前接口只实现转写能力，`language`、`prompt`、`response_format`、`temperature` 等 OpenAI 常见参数会被兼容接收，但返回始终保持：

```json
{"text":"..."}
```
