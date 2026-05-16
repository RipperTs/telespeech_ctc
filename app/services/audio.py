import asyncio
import wave
from pathlib import Path

import numpy as np
from fastapi import UploadFile


class AudioProcessingError(Exception):
    """Raised when uploaded audio cannot be prepared for ASR."""


class AudioTooLargeError(AudioProcessingError):
    """Raised when uploaded audio exceeds configured size limit."""


async def save_upload_file(upload_file: UploadFile, target_path: Path, max_bytes: int) -> None:
    """Persist an uploaded file while enforcing a hard size limit."""

    total_size = 0
    with target_path.open("wb") as output:
        while chunk := await upload_file.read(1024 * 1024):
            total_size += len(chunk)
            if total_size > max_bytes:
                raise AudioTooLargeError("Audio file is too large")
            output.write(chunk)


async def convert_to_wav(input_path: Path, output_path: Path, sample_rate: int) -> None:
    """Convert arbitrary audio input to mono 16-bit PCM WAV for sherpa-onnx."""

    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-acodec",
        "pcm_s16le",
        str(output_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await process.communicate()
    except asyncio.CancelledError:
        process.kill()
        await process.wait()
        raise

    if process.returncode != 0:
        message = stderr.decode("utf-8", errors="ignore").strip()
        raise AudioProcessingError(message or "Failed to convert audio")


def read_wav_samples(wav_path: Path) -> tuple[np.ndarray, int]:
    """Read mono int16 WAV and normalize samples to float32 in [-1, 1]."""

    with wave.open(str(wav_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()

        if channels != 1 or sample_width != 2:
            raise AudioProcessingError("WAV must be mono 16-bit PCM")

        frames = wav_file.readframes(wav_file.getnframes())

    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    return samples / 32768.0, sample_rate
