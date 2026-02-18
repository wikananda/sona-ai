from fastapi import HTTPException
from fastapi import FastAPI, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from summarization.FlanT5Inferencer import FlanT5Inferencer
from transcription.WhisperXEngine import WhisperXEngine
from utils.utils import load_config, setup_logging

from typing import Optional
import shutil
import uuid
import os

logger = setup_logging()
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    logger.info("Setting up environment...")
    
    # Load config and setup environment
    config = load_config('whisperx')
    WhisperXEngine.setup_environment(config=config)
    
    logger.info("Loading models...")
    
    # Load WhisperX transcription engine
    app.state.asr = WhisperXEngine(config)
    app.state.asr.load_models()
    
    # Load FlanT5 summarization model
    app.state.summarizer = FlanT5Inferencer(
        config_name="flan-t5",
        use_pretrained=True,
        device="auto",
        max_new_tokens=256,
        num_beams=4,
    )
    
    logger.info("All models loaded successfully!")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down...")
    logger.info("Cleaning up models...")
    app.state.asr.cleanup_models()
    app.state.summarizer.cleanup_models()
    logger.info("Cleanup complete!")

@app.post("/transcribe")
async def transcribe(
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
            app.state.asr.transcribe,
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