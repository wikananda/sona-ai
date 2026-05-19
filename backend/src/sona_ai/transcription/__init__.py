__all__ = ["ParakeetTranscriber"]


def __getattr__(name):
    if name == "ParakeetTranscriber":
        from .parakeet_transcriber import ParakeetTranscriber

        return ParakeetTranscriber
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
