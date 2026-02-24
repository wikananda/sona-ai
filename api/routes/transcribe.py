from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from typing import Optional
import shutil
import uuid
import os

from utils.utils import setup_logging

logger = setup_logging()
router = APIRouter()

@router.post("/transcribe")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    language: Optional[str]=None,
    min_speakers: Optional[int]=None,
    max_speakers: Optional[int]=None
):
    filename = file.filename
    extension = os.path.splitext(filename)[1] # get the format
    temp_filename = f"/tmp/{uuid.uuid4()}{extension}"

    # Stream the file content, so not loading it as whole. Reduce the amount of RAM usage
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = await run_in_threadpool(
            request.app.state.asr.transcribe,
            temp_filename,
            language=language,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        return result
    except Exception as e:
        logger.error(f"Error transcribing file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)