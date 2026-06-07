import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from sona_ai.core import setup_logging
from sona_ai.api.schemas.summarize import SummarizeRequest

logger = setup_logging()
router = APIRouter()

@router.post("/summarize")
async def summarize(request: Request, body: SummarizeRequest):
    try:
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
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status == 401:
            raise HTTPException(
                status_code=400,
                detail="BYOK authentication failed. Check provider, API key, and model.",
            ) from exc
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as e:
        logger.exception("Error summarizing text: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
