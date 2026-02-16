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

class WhisperXEngine:
    def __init__(
        self,
        config: dict,
        min_speakers: int = None,
        max_speakers: int = None
    ):
        self.config = config
        self.model = None
        self.align_model = None
        self.align_metadata = None
        self.diarize_model = None
        self.audio = None

        if min_speakers is not None:
            self.config['input']['min_speakers'] = min_speakers
        if max_speakers is not None:
            self.config['input']['max_speakers'] = max_speakers

    def load_models(self):
        """
        Loading whisper and alignment model based on config files
        """
        logger.info("Loading necessary models...")
        assert os.getenv("HF_TOKEN") is not None, "Hugging Face token is not set"

        self.model = whisperx.load_model(
            self.config['model']['whisper_model'],
            language=self.config['model']['language'],
            device=self.config['model']['device'],
            compute_type=self.config['model']['compute_type'],
        )
        # Alignment model (wav2vec)
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code=self.config['model']['language'],
            device=self.config['model']['device'],
            model_name=self.config['model']['align_model']
        )
        # Diarize model (pyannote)
        self.diarize_model = DiarizationPipeline(
            use_auth_token=os.getenv("HF_TOKEN"),
            device=self.config['model']['device'],
        )

    def run_transcription(self, audio):
        """
        Running transcription

        model: whisper model
        audio: audio file
        config: configuration dictionary (.yaml files)
        """
        logger.info("Running transcription...")
        with Timer("Transcription"):
            result = self.model.transcribe(
                audio,
                batch_size=self.config['model']['batch_size']
            )
        return result

    def run_alignment(self, result, audio):
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
                self.align_model,
                self.align_metadata,
                audio,
                device=self.config['model']['device'],
                return_char_alignments=False
        )
        return result

    def run_diarization(self, audio, result):
        """
        Running diarization with pyannote.

        audio: audio file
        diarize_model: diarization model
        result: transcription result
        """
        logger.info("Running diarization...")
        assert self.config['input']['min_speakers'] <= self.config['input']['max_speakers'], "min_speakers must be less than max_speakers"
        assert self.config['input']['min_speakers'] >= 1, "min_speakers must be greater than or equal to 1"
        assert self.config['input']['max_speakers'] >= 1, "max_speakers must be greater than or equal to 1"

        with Timer("Diarization"):
            diarize_result = self.diarize_model(
                audio,
                min_speakers=self.config['input']['min_speakers'],
                max_speakers=self.config['input']['max_speakers'],
            )

        # Assign diarization to transcription
        with Timer("Assign diarization to transcription"):
            result_final = whisperx.assign_word_speakers(diarize_result, result)
        return result_final, diarize_result

    def build_conversations(self, result_segments):
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

    def cleanup_models(self):
        for m in [self.model, self.align_model, self.diarize_model]:
            if m is not None:
                del m
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def transcribe(self, audio_path: str):
        if self.model is None or self.align_model is None or self.diarize_model is None:
            raise ReferenceError("WhisperX model not yet initialized.")

        self.audio = whisperx.load_audio(audio_path)
        result = self.run_transcription(self.audio)
        result = self.run_alignment(result, self.audio)
        result_final, diarize_result = self.run_diarization(self.audio, result)

        segments = sanitize_for_json(result_final['segments'])
        conversations = self.build_conversations(segments)

        return {
            'conversations': conversations,
            'diarize_result': diarize_result,
            'result': result_final
        }

    @staticmethod
    def setup_environment(config: dict = None, quiet=False):
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
        
        # set hugging face directory to local relative directory if config provided
        if config and 'cp_dir' in config and 'hf_cache' in config['cp_dir']:
            os.environ["HF_HOME"] = config['cp_dir']['hf_cache']