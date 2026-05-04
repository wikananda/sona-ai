from sona_ai.alignment import Wav2Vec2Aligner
from sona_ai.diarization import PyannoteDiarizer
from sona_ai.pipelines import SpeechPipeline, WhisperXSpeakerAssigner
from sona_ai.transcription.whisperx_transcriber import WhisperXTranscriber


class WhisperXEngine(SpeechPipeline):
    """
    Backward-compatible wrapper for the old all-in-one WhisperXEngine.

    New code should compose SpeechPipeline with explicit transcriber and diarizer
    adapters instead of depending on this facade.
    """

    def __init__(self, config: dict):
        super().__init__(
            transcriber=WhisperXTranscriber(config),
            aligner=Wav2Vec2Aligner(config),
            diarizer=PyannoteDiarizer(config),
            speaker_assigner=WhisperXSpeakerAssigner(),
        )
