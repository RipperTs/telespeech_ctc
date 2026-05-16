from typing import Annotated

from fastapi import Depends, Request

from app.services.asr import AsrService
from app.services.limiter import RequestLimiter


def get_asr_service(request: Request) -> AsrService:
    return request.app.state.asr_service


def get_request_limiter(request: Request) -> RequestLimiter:
    return request.app.state.request_limiter


AsrServiceDep = Annotated[AsrService, Depends(get_asr_service)]
RequestLimiterDep = Annotated[RequestLimiter, Depends(get_request_limiter)]
