import logging
import os
import warnings
from typing import Optional

from sona_ai.core import PROJECT_ROOT, setup_logging, write_json
from sona_ai.diarization.base import Diarizer
from sona_ai.pipelines.speaker_assignment import WhisperXSpeakerAssigner
from sona_ai.transcription.base import Transcriber


logger = setup_logging()
output_dir = PROJECT_ROOT / "outputs" / "transcription"


class SpeechPipeline:
    def __init__(
        self,
        transcriber: Transcriber,
        diarizer: Optional[Diarizer] = None,
        speaker_assigner: Optional[WhisperXSpeakerAssigner] = None,
        write_outputs: bool = True,
    ):
        self.transcriber = transcriber
        self.diarizer = diarizer
        self.speaker_assigner = speaker_assigner or WhisperXSpeakerAssigner()
        self.write_outputs = write_outputs

    def load_models(self):
        self.transcriber.load_models()
        if self.diarizer is not None:
            self.diarizer.load_models()

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        transcription = self.transcriber.transcribe(audio_path, language=language)

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

        diarization = self.diarizer.diarize(
            audio_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        segments = self.speaker_assigner.assign(transcription, diarization)
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
        previous_speaker = "Unknown"

        for segment in result_segments:
            current_speaker = segment.get("speaker", previous_speaker)
            conversations.append({
                "speaker": current_speaker,
                "text": segment["text"],
                "start": segment["start"],
                "end": segment["end"],
            })
            previous_speaker = current_speaker

        return conversations

    def _write_result(self, result):
        if not self.write_outputs:
            return

        write_json(output_dir / "conversations.json", result["transcript"])
        write_json(output_dir / "result_raw.json", result["result_raw"])

    def cleanup_models(self):
        self.transcriber.cleanup_models()
        if self.diarizer is not None:
            self.diarizer.cleanup_models()

    @staticmethod
    def setup_environment(config: dict = None, quiet=False):
        if quiet:
            warnings.filterwarnings("ignore")
            logging.getLogger("whisperx").setLevel(logging.ERROR)
            logging.getLogger("faster_whisper").setLevel(logging.ERROR)

        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

        if config and "cp_dir" in config and "hf_cache" in config["cp_dir"]:
            os.environ["HF_HOME"] = str(PROJECT_ROOT / config["cp_dir"]["hf_cache"])

    def close(self):
        self.cleanup_models()
