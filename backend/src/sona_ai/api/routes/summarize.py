from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool

from sona_ai.api.routes._errors import route_error_handler
from sona_ai.api.schemas.summarize import SummarizeRequest

router = APIRouter()

@router.post("/summarize")
async def summarize(request: Request, body: SummarizeRequest):
    with route_error_handler("Error summarizing text: %s", byok_errors=True):
        result = await run_in_threadpool(
            request.app.state.summarization_service.summarize_adaptive,
            body.text,
            body.prompt,
            max_length=body.max_length,
            model=body.model,
            device=body.device,
            mode=body.mode,
            byok=body.byok.model_dump() if body.byok else None,
        )
        return result
