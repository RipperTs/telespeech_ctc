from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    app_name: str = "TeleSpeech CTC API"
    api_key: str | None = Field(default=None, validation_alias="ASR_API_KEY")

    model_dir: Path = Field(default=Path("/models/telespeech"), validation_alias="MODEL_DIR")
    encoder_file: str = Field(default="model.int8.onnx", validation_alias="ENCODER_FILE")
    tokens_file: str = Field(default="tokens.txt", validation_alias="TOKENS_FILE")
    model_type: str = Field(default="telespeech_ctc", validation_alias="MODEL_TYPE")

    sample_rate: int = Field(default=16000, validation_alias="SAMPLE_RATE")
    max_upload_mb: int = Field(default=200, validation_alias="MAX_UPLOAD_MB")
    request_timeout_seconds: int = Field(default=300, validation_alias="REQUEST_TIMEOUT_SECONDS")

    recognizer_instances: int = Field(default=1, validation_alias="RECOGNIZER_INSTANCES")
    recognizer_threads: int = Field(default=2, validation_alias="RECOGNIZER_THREADS")
    inference_workers: int = Field(default=2, validation_alias="INFERENCE_WORKERS")

    allowed_models: str = Field(
        default="FunAudioLLM/SenseVoiceSmall,telespeech-ctc,whisper-1",
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
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def allowed_model_names(self) -> set[str]:
        return {item.strip() for item in self.allowed_models.split(",") if item.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
