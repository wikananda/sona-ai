from fastapi import FastAPI
from summarization.FlanT5Inferencer import FlanT5Inferencer
from transcription.WhisperXEngine import WhisperXEngine
from utils.utils import load_config, setup_logging

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