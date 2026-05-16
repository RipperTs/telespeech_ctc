from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    app_name: str = "TeleSpeech CTC API"
    api_key: str | None = Field(default=None, validation_alias="ASR_API_KEY")

    model_dir: Path = Field(default=Path("/models/telespeech"), validation_alias="MODEL_DIR")
    encoder_file: str = Field(default="model.onnx", validation_alias="ENCODER_FILE")
    tokens_file: str = Field(default="tokens.txt", validation_alias="TOKENS_FILE")
    model_type: str = Field(default="telespeech_ctc", validation_alias="MODEL_TYPE")
    model_provider: str = Field(default="cpu", validation_alias="MODEL_PROVIDER")
    punctuation_model_dir: Path = Field(
        default=Path("/models/punctuation"),
        validation_alias="PUNCTUATION_MODEL_DIR",
    )
    punctuation_model_file: str = Field(
        default="model.int8.onnx",
        validation_alias="PUNCTUATION_MODEL_FILE",
    )
    enable_punctuation: bool = Field(default=True, validation_alias="ENABLE_PUNCTUATION")
    vad_model_dir: Path = Field(default=Path("/models/vad"), validation_alias="VAD_MODEL_DIR")
    vad_model_file: str = Field(default="silero_vad.onnx", validation_alias="VAD_MODEL_FILE")
    enable_vad: bool = Field(default=True, validation_alias="ENABLE_VAD")
    vad_threshold: float = Field(default=0.2, validation_alias="VAD_THRESHOLD")
    vad_min_silence_seconds: float = Field(
        default=0.25,
        validation_alias="VAD_MIN_SILENCE_SECONDS",
    )
    vad_min_speech_seconds: float = Field(
        default=0.25,
        validation_alias="VAD_MIN_SPEECH_SECONDS",
    )
    min_segment_seconds: float = Field(default=1.0, validation_alias="MIN_SEGMENT_SECONDS")

    sample_rate: int = Field(default=16000, validation_alias="SAMPLE_RATE")
    chunk_seconds: int = Field(default=30, validation_alias="CHUNK_SECONDS")
    max_upload_mb: int = Field(default=200, validation_alias="MAX_UPLOAD_MB")
    request_timeout_seconds: int = Field(default=300, validation_alias="REQUEST_TIMEOUT_SECONDS")

    recognizer_instances: int = Field(default=1, validation_alias="RECOGNIZER_INSTANCES")
    recognizer_threads: int = Field(default=2, validation_alias="RECOGNIZER_THREADS")
    inference_workers: int = Field(default=2, validation_alias="INFERENCE_WORKERS")
    max_pending_requests: int = Field(default=10, validation_alias="MAX_PENDING_REQUESTS")

    allowed_models: str = Field(
        default="TeleAI/TeleSpeechASR,FunAudioLLM/SenseVoiceSmall,telespeech-ctc,whisper-1",
        validation_alias="ALLOWED_MODELS",
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.encoder_file

    @property
    def tokens_path(self) -> Path:
        return self.model_dir / self.tokens_file

    @property
    def punctuation_model_path(self) -> Path:
        return self.punctuation_model_dir / self.punctuation_model_file

    @property
    def vad_model_path(self) -> Path:
        return self.vad_model_dir / self.vad_model_file

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def allowed_model_names(self) -> set[str]:
        return {item.strip() for item in self.allowed_models.split(",") if item.strip()}

    @property
    def max_active_requests(self) -> int:
        return self.recognizer_instances + self.max_pending_requests


@lru_cache
def get_settings() -> Settings:
    return Settings()
