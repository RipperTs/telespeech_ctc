from secrets import compare_digest

from fastapi import Depends, Header, HTTPException, status

from app.core.config import Settings, get_settings


async def verify_api_key(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    """Validate bearer token when ASR_API_KEY is configured."""

    if not settings.api_key:
        return

    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not compare_digest(token, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
