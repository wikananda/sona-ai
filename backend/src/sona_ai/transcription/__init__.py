__all__ = ["WhisperXEngine", "WhisperXTranscriber"]


def __getattr__(name):
    if name == "WhisperXEngine":
        from .whisperx_engine import WhisperXEngine

        return WhisperXEngine
    if name == "WhisperXTranscriber":
        from .whisperx_transcriber import WhisperXTranscriber

        return WhisperXTranscriber
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
