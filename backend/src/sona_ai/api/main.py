from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sona_ai.core import load_config, setup_logging
from sona_ai.db import init_db
from sona_ai.db.engine import SessionLocal
from sona_ai.db.models import Recording, RecordingStatus
from sona_ai.pipelines import build_speech_pipeline
from sona_ai.services import SummarizationService, TranscriptionService
from sona_ai.services.parakeet_live_gateway import ParakeetLiveGateway
from sona_ai.services.nemotron_live_gateway import NemotronLiveGateway
from sona_ai.services.recording_job_manager import RecordingJobManager
from sona_ai.services.whisper_live_gateway import WhisperLiveGateway

from sona_ai.api.routes.projects import router as projects_router
from sona_ai.api.routes.live_transcription import router as live_transcription_router
from sona_ai.api.routes.runtime import router as runtime_router
from sona_ai.api.routes.transcribe import router as transcribe_router
from sona_ai.api.routes.summarize import router as summarize_router
from sona_ai.api.routes.chat import router as chat_router

import os

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
app.include_router(live_transcription_router)
app.include_router(runtime_router)
app.include_router(chat_router)


@app.on_event("startup")
async def startup_event():
    logger.info("Initializing database...")
    init_db()
    _mark_interrupted_recordings_failed()

    logger.info("Setting up environment...")
    speech_config = load_config(os.getenv("SONA_SPEECH_CONFIG", "speech"))
    
    logger.info("Loading models...")
    
    speech_pipeline = build_speech_pipeline(speech_config)
    speech_pipeline.load_models()
    app.state.transcription_service = TranscriptionService(
        speech_pipeline,
        speech_config=speech_config,
        default_model=speech_config.get("transcription", {}).get("engine", "parakeet"),
        default_device=speech_config.get("transcription", {}).get("device", "auto"),
    )
    app.state.recording_job_manager = RecordingJobManager()
    app.state.whisper_live_gateway = WhisperLiveGateway()
    app.state.nemotron_live_gateway = NemotronLiveGateway()
    app.state.parakeet_live_gateway = ParakeetLiveGateway(
        app.state.transcription_service,
    )
    
    app.state.summarization_service = SummarizationService(
        config=speech_config.get("summarization", {}).get("config", "llama"),
        use_pretrained=True,
        device="auto",
    )
    
    logger.info("Speech models loaded. Summarization model will load on first use.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down...")
    app.state.recording_job_manager.shutdown(wait=True)
    logger.info("Cleaning up models...")
    await app.state.whisper_live_gateway.close()
    await app.state.nemotron_live_gateway.close()
    await app.state.parakeet_live_gateway.close()
    app.state.transcription_service.close()
    app.state.summarization_service.close()
    logger.info("Cleanup complete!")


def _mark_interrupted_recordings_failed():
    db = SessionLocal()
    try:
        recordings = (
            db.query(Recording)
            .filter(Recording.status.in_([
                RecordingStatus.PENDING,
                RecordingStatus.PROCESSING,
            ]))
            .all()
        )
        for recording in recordings:
            recording.status = RecordingStatus.FAILED
            recording.processing_stage = "failed"
            recording.processing_job_id = None
            recording.error = "Interrupted by server restart"
        db.commit()
    finally:
        db.close()
