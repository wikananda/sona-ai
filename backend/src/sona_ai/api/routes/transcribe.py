from fastapi import APIRouter, UploadFile, File, Request
from fastapi.concurrency import run_in_threadpool
from typing import Optional
import shutil
import uuid
import os

from sona_ai.api.routes._errors import route_error_handler

router = APIRouter()

@router.post("/transcribe")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    language: Optional[str]=None,
    model: Optional[str]=None,
    device: Optional[str]=None,
    min_speakers: Optional[int]=None,
    max_speakers: Optional[int]=None,
    extract_speakers: bool=True,
):
    # Legacy single-shot route. New UI should use project-scoped recording routes.
    filename = file.filename
    extension = os.path.splitext(filename)[1] # get the format
    temp_filename = f"/tmp/{uuid.uuid4()}{extension}"

    # Stream the file content, so not loading it as whole. Reduce the amount of RAM usage
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        with route_error_handler("Error transcribing file: %s", log_traceback=False):
            result = await run_in_threadpool(
                request.app.state.transcription_service.transcribe,
                temp_filename,
                language=language,
                model=model,
                device=device,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                extract_speakers=extract_speakers,
            )
            return result
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
