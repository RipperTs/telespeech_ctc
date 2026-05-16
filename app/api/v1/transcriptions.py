import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.dependencies import AsrServiceDep, RequestLimiterDep
from app.core.config import Settings, get_settings
from app.core.security import verify_api_key
from app.schemas.transcription import TranscriptionResponse
from app.services.asr import TranscriptionResult
from app.services.limiter import RequestLimitExceeded
from app.services.audio import (
    AudioProcessingError,
    AudioTooLargeError,
    convert_to_wav,
    read_wav_samples,
    save_upload_file,
)

router = APIRouter(tags=["audio"])


@router.post(
    "/audio/transcriptions",
    response_model=TranscriptionResponse,
    dependencies=[Depends(verify_api_key)],
)
async def create_transcription(
    asr_service: AsrServiceDep,
    request_limiter: RequestLimiterDep,
    file: UploadFile = File(...),
    model: str = Form(...),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: str | None = Form(default=None),
    temperature: float | None = Form(default=None),
    settings: Settings = Depends(get_settings),
) -> TranscriptionResponse:
    """OpenAI-compatible transcription endpoint with SiliconFlow-style response."""

    if model not in settings.allowed_model_names:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported model: {model}",
        )

    _ = language, prompt, response_format, temperature

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file is required",
        )

    try:
        await request_limiter.acquire()
        try:
            result = await asyncio.wait_for(
                _transcribe_upload(file, asr_service, settings),
                timeout=settings.request_timeout_seconds,
            )
        finally:
            request_limiter.release()
    except RequestLimitExceeded as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many pending transcription requests",
        ) from exc
    except TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Transcription timed out",
        ) from exc
    except AudioTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(exc),
        ) from exc
    except AudioProcessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return TranscriptionResponse(text=result.text)


async def _transcribe_upload(
    upload_file: UploadFile,
    asr_service,
    settings: Settings,
) -> TranscriptionResult:
    suffix = Path(upload_file.filename or "audio").suffix
    with TemporaryDirectory(prefix="telespeech-") as temp_dir:
        input_path = Path(temp_dir) / f"input{suffix}"
        wav_path = Path(temp_dir) / "audio.wav"

        await save_upload_file(upload_file, input_path, settings.max_upload_bytes)
        await convert_to_wav(input_path, wav_path, settings.sample_rate)

        samples, sample_rate = read_wav_samples(wav_path)
        return await asr_service.transcribe(samples, sample_rate)
