from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sona_ai.core import load_config, setup_logging
from sona_ai.diarization import PyannoteDiarizer
from sona_ai.pipelines import SpeechPipeline, WhisperXSpeakerAssigner
from sona_ai.services import SummarizationService, TranscriptionService
from sona_ai.summarization import LocalLLMSummarizer
from sona_ai.transcription import WhisperXTranscriber

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

@app.on_event("startup")
async def startup_event():
    logger.info("Setting up environment...")
    
    # Load config and setup environment
    config = load_config('whisperx')
    SpeechPipeline.setup_environment(config=config)
    
    logger.info("Loading models...")
    
    speech_pipeline = SpeechPipeline(
        transcriber=WhisperXTranscriber(config),
        diarizer=PyannoteDiarizer(config),
        speaker_assigner=WhisperXSpeakerAssigner(),
    )
    speech_pipeline.load_models()
    app.state.transcription_service = TranscriptionService(speech_pipeline)
    
    summarizer = LocalLLMSummarizer(
        config="llama",
        use_pretrained=True,
        device="auto",
        max_new_tokens=256,
        num_beams=4,
    )
    app.state.summarization_service = SummarizationService(summarizer)
    
    logger.info("All models loaded successfully!")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down...")
    logger.info("Cleaning up models...")
    app.state.transcription_service.close()
    app.state.summarization_service.close()
    logger.info("Cleanup complete!")
