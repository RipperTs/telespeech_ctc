from fastapi import APIRouter

from app.api.v1.transcriptions import router as transcriptions_router

api_router = APIRouter()
api_router.include_router(transcriptions_router)
