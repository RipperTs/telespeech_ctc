import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Queue

from app.core.config import Settings


@dataclass(frozen=True)
class TranscriptionResult:
    text: str


class RecognizerPool:
    """Small blocking pool that serializes access to each recognizer instance."""

    def __init__(self, recognizers: list[object]) -> None:
        self._queue: Queue[object] = Queue()
        for recognizer in recognizers:
            self._queue.put(recognizer)

    def transcribe(self, samples, sample_rate: int) -> TranscriptionResult:
        recognizer = self._queue.get()
        try:
            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples)
            recognizer.decode_streams([stream])
            return TranscriptionResult(text=stream.result.text.strip())
        finally:
            self._queue.put(recognizer)


class AsrService:
    """Thread-friendly wrapper around sherpa-onnx offline recognizers."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pool: RecognizerPool | None = None
        self._executor: ThreadPoolExecutor | None = None

    def load(self) -> None:
        self._ensure_model_files()

        import sherpa_onnx

        recognizers = [
            sherpa_onnx.OfflineRecognizer.from_telespeech_ctc(
                model=str(self._settings.model_path),
                tokens=str(self._settings.tokens_path),
                num_threads=self._settings.recognizer_threads,
                sample_rate=self._settings.sample_rate,
                decoding_method="greedy_search",
                provider="cpu",
            )
            for _ in range(self._settings.recognizer_instances)
        ]

        self._pool = RecognizerPool(recognizers)
        self._executor = ThreadPoolExecutor(max_workers=self._settings.inference_workers)

    async def transcribe(self, samples, sample_rate: int) -> TranscriptionResult:
        if self._pool is None or self._executor is None:
            raise RuntimeError("ASR service has not been loaded")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._pool.transcribe,
            samples,
            sample_rate,
        )

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)

    def _ensure_model_files(self) -> None:
        missing_paths: list[Path] = [
            path
            for path in (self._settings.model_path, self._settings.tokens_path)
            if not path.is_file()
        ]
        if missing_paths:
            missing = ", ".join(str(path) for path in missing_paths)
            raise FileNotFoundError(f"Missing model file(s): {missing}")
