__all__ = ["FasterWhisperTranscriber", "ParakeetTranscriber"]


def __getattr__(name):
    if name == "ParakeetTranscriber":
        from .parakeet_transcriber import ParakeetTranscriber

        return ParakeetTranscriber
    if name == "FasterWhisperTranscriber":
        from .faster_whisper_transcriber import FasterWhisperTranscriber

        return FasterWhisperTranscriber
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
