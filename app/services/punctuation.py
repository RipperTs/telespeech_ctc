from queue import Queue

from app.core.config import Settings


class PunctuationService:
    """Thread-friendly wrapper around sherpa-onnx punctuation models."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._queue: Queue[object] = Queue()

    def load(self) -> None:
        if not self._settings.enable_punctuation:
            return

        if not self._settings.punctuation_model_path.is_file():
            raise FileNotFoundError(
                f"Missing punctuation model: {self._settings.punctuation_model_path}"
            )

        import sherpa_onnx

        config = sherpa_onnx.OfflinePunctuationConfig(
            model=sherpa_onnx.OfflinePunctuationModelConfig(
                ct_transformer=str(self._settings.punctuation_model_path),
                num_threads=self._settings.recognizer_threads,
                provider=self._settings.model_provider,
            )
        )
        self._queue.put(sherpa_onnx.OfflinePunctuation(config))

    def add_punctuation(self, text: str) -> str:
        if not self._settings.enable_punctuation or not text:
            return text

        punctuator = self._queue.get()
        try:
            return punctuator.add_punctuation(text)
        except RuntimeError:
            return text
        finally:
            self._queue.put(punctuator)
