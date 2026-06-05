import logging
import os
import warnings
from typing import Optional

from sona_ai.alignment.base import Aligner
from sona_ai.core import PROJECT_ROOT, setup_logging, write_json
from sona_ai.diarization.base import Diarizer
from sona_ai.pipelines.speaker_assignment import SpeakerAssigner
from sona_ai.transcription.base import Transcriber
from sona_ai.transcription.schemas import TranscriptionResult


logger = setup_logging()
output_dir = PROJECT_ROOT / "outputs" / "transcription"


class SpeechPipeline:
    def __init__(
        self,
        transcriber: Transcriber,
        aligner: Optional[Aligner] = None,
        diarizer: Optional[Diarizer] = None,
        speaker_assigner: Optional[SpeakerAssigner] = None,
        write_outputs: bool = True,
    ):
        self.transcriber = transcriber
        self.aligner = aligner
        self.diarizer = diarizer
        self.speaker_assigner = speaker_assigner or SpeakerAssigner()
        self.write_outputs = write_outputs

    def load_models(self):
        self.transcriber.load_models()
        if self.aligner is not None:
            self.aligner.load_models()
        if self.diarizer is not None:
            self.diarizer.load_models()

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        logger.info("Speech pipeline stage started: transcription")
        transcription = self.transcriber.transcribe(audio_path, language=language)
        logger.info(
            "Speech pipeline stage finished: transcription (%d segments)",
            len(transcription.segments),
        )
        if self.aligner is not None:
            logger.info("Speech pipeline stage started: alignment")
            transcription = self.aligner.align(transcription, audio_path)
            logger.info(
                "Speech pipeline stage finished: alignment (%d segments)",
                len(transcription.segments),
            )

        if self.diarizer is None:
            segments = transcription.to_segment_dicts()
            conversations = self._build_conversations(segments)
            result = {
                "transcript": conversations,
                "diarize_result": [],
                "result_raw": segments,
            }
            self._write_result(result)
            return result

        return self.extract_speakers(
            audio_path,
            transcription,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

    def extract_speakers(
        self,
        audio_path: str,
        transcription: TranscriptionResult,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        if self.diarizer is None:
            raise ValueError("Speaker extraction is not available because diarization is disabled")

        logger.info("Speech pipeline stage started: diarization")
        diarization = self.diarizer.diarize(
            audio_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        logger.info(
            "Speech pipeline stage finished: diarization (%d turns)",
            len(diarization.turns),
        )
        logger.info("Speech pipeline stage started: speaker assignment")
        segments = self.speaker_assigner.assign(transcription, diarization)
        speakers = sorted({
            segment.get("speaker")
            for segment in segments
            if segment.get("speaker")
        })
        logger.info(
            "Final transcript has %d speakers across %d segments: %s",
            len(speakers),
            len(segments),
            speakers,
        )
        logger.info("Speech pipeline stage finished: speaker assignment")
        conversations = self._build_conversations(segments)

        result = {
            "transcript": conversations,
            "diarize_result": diarization.to_dict(),
            "result_raw": segments,
        }
        self._write_result(result)
        return result

    def _build_conversations(self, result_segments):
        conversations = []

        for segment in result_segments:
            conversation = {
                "text": segment["text"],
                "start": segment["start"],
                "end": segment["end"],
            }
            if segment.get("speaker"):
                conversation["speaker"] = segment["speaker"]
            conversations.append(conversation)

        return conversations

    def _write_result(self, result):
        if not self.write_outputs:
            return

        write_json(output_dir / "conversations.json", result["transcript"])
        write_json(output_dir / "result_raw.json", result["result_raw"])

    def cleanup_models(self):
        self.transcriber.cleanup_models()
        if self.aligner is not None:
            self.aligner.cleanup_models()
        if self.diarizer is not None:
            self.diarizer.cleanup_models()

    @staticmethod
    def setup_environment(config: dict = None, quiet=False):
        if quiet:
            warnings.filterwarnings("ignore")
            logging.getLogger("faster_whisper").setLevel(logging.ERROR)

        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

        if config and "cp_dir" in config and "hf_cache" in config["cp_dir"]:
            cache_dir = PROJECT_ROOT / config["cp_dir"]["hf_cache"]
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"] = str(cache_dir)
            os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir / "hub")
            os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
            os.environ["TORCH_HOME"] = str(cache_dir / "torch")
            os.environ["PYANNOTE_CACHE"] = str(cache_dir / "pyannote")
            os.environ["NEMO_HOME"] = str(cache_dir / "nemo")
            os.environ["NEMO_CACHE_DIR"] = str(cache_dir / "nemo")
            os.environ["XDG_CACHE_HOME"] = str(cache_dir / "xdg")
            os.environ["MPLCONFIGDIR"] = str(cache_dir / "matplotlib")

    def close(self):
        self.cleanup_models()
