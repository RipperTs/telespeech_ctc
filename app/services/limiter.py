import asyncio


class RequestLimitExceeded(Exception):
    """Raised when the service cannot accept more pending requests."""


class RequestLimiter:
    """A small non-blocking limiter for active and pending transcription requests."""

    def __init__(self, capacity: int) -> None:
        self._semaphore = asyncio.Semaphore(capacity)

    async def acquire(self) -> None:
        if self._semaphore.locked():
            raise RequestLimitExceeded
        await self._semaphore.acquire()

    def release(self) -> None:
        self._semaphore.release()
