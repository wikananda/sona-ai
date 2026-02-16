import os
from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import whisperx
import torch
import gc
from whisperx.diarize import DiarizationPipeline

import time
import math
import json
import numpy as np
import yaml
from utils.Timer import Timer
from utils.utils import load_config
import argparse
import warnings
import logging
from utils.utils import write_json, setup_logging, sanitize_for_json

logger = setup_logging()

config = load_config('whisperx')

# ----------- ENVIRONMENT HELPER -----------
def setup_environment(quiet=False):
    """
    Setting up pytorch and hugging face environment
    """
    if quiet:
        # Silence all standard Python warnings
        warnings.filterwarnings("ignore")
        # Silence specific noisy loggers from the libraries
        logging.getLogger("whisperx").setLevel(logging.ERROR)
        logging.getLogger("faster_whisper").setLevel(logging.ERROR)

    # loading model with no weights (legacy loading) for PyTorch 2.6.0+
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    # set hugging face directory to local relative directory
    os.environ["HF_HOME"] = config['cp_dir']['hf_cache']

def cleanup_models(*models):
    for m in models:
        del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

#  ----------- WHISPERX FUNCTIONS -----------
def load_models(config: dict):
    """
    Loading whisper and alignment model

    config: configuration dictionary (.yaml files)
    """
    # Whisper model
    logger.info("Loading necessary models...")
    assert os.getenv("HF_TOKEN") is not None, "Hugging Face token is not set"

    model = whisperx.load_model(
        config['model']['whisper_model'],
        language=config['model']['language'],
        device=config['model']['device'],
        compute_type=config['model']['compute_type'],
    )
    # Alignment model (wav2vec)
    align_model, align_metadata = whisperx.load_align_model(
        language_code=config['model']['language'],
        device=config['model']['device'],
        model_name=config['model']['align_model']
    )
    # Diarize model (pyannote)
    diarize_model = DiarizationPipeline(
        use_auth_token=os.getenv("HF_TOKEN"),
        device=config['model']['device'],
    )
    return model, align_model, align_metadata, diarize_model

def run_transcription(model, audio, config: dict):
    """
    Running transcription

    model: whisper model
    audio: audio file
    config: configuration dictionary (.yaml files)
    """
    logger.info("Running transcription...")
    with Timer("Transcription"):
        result = model.transcribe(
            audio,
            batch_size=config['model']['batch_size']
        )
    return result

def run_alignment(result, audio, align_model, align_metadata, config: dict):
    """
    Running force-alignment with wav2vec.
    Search the wav2vec fine-tuned model in huggingface

    result: transcription result
    audio: audio file
    align_model: alignment model
    align_metadata: alignment metadata
    """
    logger.info("Running alignment...")
    with Timer("Alignment"):
        result = whisperx.align(
            result['segments'],
            align_model,
            align_metadata,
            audio,
            device=config['model']['device'],
            return_char_alignments=False
    )
    return result

def run_diarization(audio, diarize_model, result, config: dict):
    """
    Running diarization with pyannote.

    audio: audio file
    diarize_model: diarization model
    result: transcription result
    """
    logger.info("Running diarization...")
    assert config['input']['min_speakers'] <= config['input']['max_speakers'], "min_speakers must be less than max_speakers"
    assert config['input']['min_speakers'] >= 1, "min_speakers must be greater than or equal to 1"
    assert config['input']['max_speakers'] >= 1, "max_speakers must be greater than or equal to 1"

    with Timer("Diarization"):
        diarize_result = diarize_model(
            audio,
            min_speakers=config['input']['min_speakers'],
            max_speakers=config['input']['max_speakers'],
        )

    # Assign diarization to transcription
    with Timer("Assign diarization to transcription"):
        result_final = whisperx.assign_word_speakers(diarize_result, result)
    return result_final, diarize_result


# ----------- HELPER BUILDING FUNCTIONS -----------
def build_conversations(result_segments):
    """
    Build final conversation JSON that is ready to show in FE
    
    result: final result from whisperx after sanitize_for_json()
    """
    conversations = []
    current_speaker = 'Unknown'
    previous_speaker = 'Unknown'
    for convo in result_segments:
        current_speaker = convo.get('speaker', previous_speaker)
        conversations.append({
            'speaker': current_speaker,
            'text': convo['text'],
            'start': convo['start'],
            'end': convo['end']
        })
        previous_speaker = current_speaker
    return conversations


# ----------- MAIN FUNCTIONS -----------
def run_whisperx_pipeline(config: dict):
    """
    Run the whole transcription pipeline

    config: configuration dictionary (.yaml files)
    """

    # Inferencing
    assert Path(config['input']['audio_file']).exists(), f"Audio file {config['input']['audio_file']} does not exist"
    audio = whisperx.load_audio(config['input']['audio_file'])
    model, align_model, align_metadata, diarize_model = load_models(config)
    result = run_transcription(model, audio, config)
    result = run_alignment(result, audio, align_model, align_metadata, config)
    result_final, diarize_result = run_diarization(audio, diarize_model, result, config)
    
    # Post-processing and saving
    logger.info("Saving result...")
    segments = sanitize_for_json(result_final['segments'])
    word_segments = sanitize_for_json(result_final['word_segments'])
    write_json(config['outputs']['segments'], segments)
    write_json(config['outputs']['word_segments'], word_segments)
    transcription = build_conversations(segments)
    write_json(config['outputs']['result'], transcription)

    # Cleanup
    cleanup_models(model, align_model, diarize_model)

    logger.info("Transcription process completed!")
    return transcription


if __name__ == '__main__':
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Sona AI - WhisperX Transcription Engine")
    parser.add_argument(
        "--quiet", "-q", 
        action="store_true", 
        help="Turn off all warnings and set logging to INFO and ERROR level only"
    )
    args = parser.parse_args()

    setup_environment(quiet=args.quiet)
    _ = run_whisperx_pipeline(config)
    # Force exit to prevent hanging due to lingering threads or process
    # Usually due to pyannote or whisperx (non-daemon threads)
    os._exit(0) 