__all__ = ["ParakeetTranscriber", "WhisperXTranscriber"]


def __getattr__(name):
    if name == "WhisperXTranscriber":
        from .whisperx_transcriber import WhisperXTranscriber

        return WhisperXTranscriber
    if name == "ParakeetTranscriber":
        from .parakeet_transcriber import ParakeetTranscriber

        return ParakeetTranscriber
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
