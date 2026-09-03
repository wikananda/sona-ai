__all__ = [
    "FasterWhisperTranscriber",
    "NemotronTranscriber",
    "ParakeetTranscriber",
    "WhisperMpsTranscriber",
]


def __getattr__(name):
    if name == "ParakeetTranscriber":
        from .parakeet_transcriber import ParakeetTranscriber

        return ParakeetTranscriber
    if name == "FasterWhisperTranscriber":
        from .faster_whisper_transcriber import FasterWhisperTranscriber

        return FasterWhisperTranscriber
    if name == "NemotronTranscriber":
        from .nemotron_transcriber import NemotronTranscriber

        return NemotronTranscriber
    if name == "WhisperMpsTranscriber":
        from .whisper_mps_transcriber import WhisperMpsTranscriber

        return WhisperMpsTranscriber
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
