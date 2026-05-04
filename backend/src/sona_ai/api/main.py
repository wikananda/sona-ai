from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sona_ai.core import load_config, setup_logging
from sona_ai.db import init_db
from sona_ai.db.engine import SessionLocal
from sona_ai.db.models import Recording, RecordingStatus
from sona_ai.pipelines import build_speech_pipeline
from sona_ai.services import SummarizationService, TranscriptionService

from sona_ai.api.routes.projects import router as projects_router
from sona_ai.api.routes.transcribe import router as transcribe_router
from sona_ai.api.routes.summarize import router as summarize_router

logger = setup_logging()
app = FastAPI(title="Sona AI API")

# Allow frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # To be replaced later in production with our URL.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(transcribe_router)
app.include_router(summarize_router)
app.include_router(projects_router)

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing database...")
    init_db()
    _mark_interrupted_recordings_failed()

    logger.info("Setting up environment...")
    speech_config = load_config("speech")
    
    logger.info("Loading models...")
    
    speech_pipeline = build_speech_pipeline(speech_config)
    speech_pipeline.load_models()
    app.state.transcription_service = TranscriptionService(
        speech_pipeline,
        speech_config=speech_config,
        default_model=speech_config.get("transcription", {}).get("engine", "parakeet"),
    )
    
    app.state.summarization_service = SummarizationService(
        config=speech_config.get("summarization", {}).get("config", "llama"),
        use_pretrained=True,
        device="auto",
        max_new_tokens=256,
        num_beams=4,
    )
    
    logger.info("Speech models loaded. Summarization model will load on first use.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down...")
    logger.info("Cleaning up models...")
    app.state.transcription_service.close()
    app.state.summarization_service.close()
    logger.info("Cleanup complete!")


def _mark_interrupted_recordings_failed():
    db = SessionLocal()
    try:
        recordings = (
            db.query(Recording)
            .filter(Recording.status == RecordingStatus.PROCESSING)
            .all()
        )
        for recording in recordings:
            recording.status = RecordingStatus.FAILED
            recording.error = "Interrupted by server restart"
        db.commit()
    finally:
        db.close()
