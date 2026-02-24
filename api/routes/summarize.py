from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from utils.utils import setup_logging
from api.schemas.summarize import SummarizeRequest

logger = setup_logging()
router = APIRouter()

@router.post("/summarize")
async def summarize(request: Request, body: SummarizeRequest):
    try:
        result = await run_in_threadpool(
            request.app.state.summarizer.generate,
            body.text,
            body.prompt,
            max_length=body.max_length
        )
        return {"summary": result}
    except Exception as e:
        logger.error(f"Error summarizing text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))