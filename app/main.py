from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.services.asr import AsrService


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    asr_service = AsrService(settings)
    asr_service.load()
    app.state.asr_service = asr_service

    try:
        yield
    finally:
        asr_service.close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.include_router(api_router, prefix="/v1")

    @app.get("/health")
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
