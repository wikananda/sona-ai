from .speaker_assignment import WhisperXSpeakerAssigner
from .speech_pipeline import SpeechPipeline

__all__ = ["SpeechPipeline", "WhisperXSpeakerAssigner", "build_speech_pipeline"]


def __getattr__(name):
    if name == "build_speech_pipeline":
        from .factory import build_speech_pipeline

        return build_speech_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
