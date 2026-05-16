from dataclasses import dataclass

import numpy as np

from app.core.config import Settings


@dataclass(frozen=True)
class SpeechSegment:
    samples: np.ndarray


class VadSegmenter:
    """Split long audio by speech activity and fall back to fixed chunks."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def split(self, samples: np.ndarray, sample_rate: int) -> list[SpeechSegment]:
        if not self._settings.enable_vad:
            return self._split_fixed(samples, sample_rate)

        if not self._settings.vad_model_path.is_file():
            return self._split_fixed(samples, sample_rate)

        try:
            segments = self._split_with_vad(samples, sample_rate)
        except RuntimeError:
            return self._split_fixed(samples, sample_rate)

        if not segments:
            return self._split_fixed(samples, sample_rate)

        return self._filter_short_segments(segments, sample_rate)

    def _split_with_vad(self, samples: np.ndarray, sample_rate: int) -> list[SpeechSegment]:
        import sherpa_onnx

        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(self._settings.vad_model_path)
        config.silero_vad.threshold = self._settings.vad_threshold
        config.silero_vad.min_silence_duration = self._settings.vad_min_silence_seconds
        config.silero_vad.min_speech_duration = self._settings.vad_min_speech_seconds
        config.silero_vad.max_speech_duration = self._settings.chunk_seconds
        config.sample_rate = sample_rate
        config.num_threads = self._settings.recognizer_threads
        config.provider = "cpu"

        vad = sherpa_onnx.VoiceActivityDetector(
            config,
            buffer_size_in_seconds=max(60, self._settings.chunk_seconds * 2),
        )

        segments: list[SpeechSegment] = []
        window_size = config.silero_vad.window_size
        buffer = samples

        for start in range(0, len(buffer), window_size):
            vad.accept_waveform(buffer[start : start + window_size])
            segments.extend(self._pop_segments(vad, sample_rate))

        vad.flush()
        segments.extend(self._pop_segments(vad, sample_rate))

        return segments

    def _pop_segments(self, vad, sample_rate: int) -> list[SpeechSegment]:
        segments: list[SpeechSegment] = []
        while not vad.empty():
            speech = np.asarray(vad.front.samples, dtype=np.float32)
            segments.extend(self._split_fixed(speech, sample_rate))
            vad.pop()
        return segments

    def _split_fixed(self, samples: np.ndarray, sample_rate: int) -> list[SpeechSegment]:
        chunk_size = self._settings.chunk_seconds * sample_rate
        min_samples = self._min_segment_samples(sample_rate)

        if len(samples) < min_samples:
            return []

        if chunk_size <= 0 or len(samples) <= chunk_size:
            return [SpeechSegment(samples=samples)] if len(samples) else []

        return [
            SpeechSegment(samples=samples[start : start + chunk_size])
            for start in range(0, len(samples), chunk_size)
            if len(samples[start : start + chunk_size]) >= min_samples
        ]

    def _filter_short_segments(
        self,
        segments: list[SpeechSegment],
        sample_rate: int,
    ) -> list[SpeechSegment]:
        min_samples = self._min_segment_samples(sample_rate)
        return [segment for segment in segments if len(segment.samples) >= min_samples]

    def _min_segment_samples(self, sample_rate: int) -> int:
        return max(1, int(self._settings.min_segment_seconds * sample_rate))
