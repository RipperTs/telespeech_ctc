from typing import Annotated

from fastapi import Depends, Request

from app.services.asr import AsrService


def get_asr_service(request: Request) -> AsrService:
    return request.app.state.asr_service


AsrServiceDep = Annotated[AsrService, Depends(get_asr_service)]
