from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from summarization.SummarizationInferencer import SummarizationInferencer
from transcription.WhisperXEngine import WhisperXEngine
from utils.utils import load_config, setup_logging

from api.routes.transcribe import router as transcribe_router
from api.routes.summarize import router as summarize_router

logger = setup_logging()
app = FastAPI()

# Allow frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # To be replaced later in production with our URL
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
    WhisperXEngine.setup_environment(config=config)
    
    logger.info("Loading models...")
    
    # Load WhisperX transcription engine
    app.state.asr = WhisperXEngine(config)
    app.state.asr.load_models()
    
    # Load FlanT5 summarization model
    app.state.summarizer = SummarizationInferencer(
        config="llama",
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