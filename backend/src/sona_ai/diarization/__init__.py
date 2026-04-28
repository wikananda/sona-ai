from .schemas import DiarizationResult, SpeakerTurn

__all__ = ["DiarizationResult", "PyannoteDiarizer", "SpeakerTurn"]


def __getattr__(name):
    if name == "PyannoteDiarizer":
        from .pyannote_diarizer import PyannoteDiarizer

        return PyannoteDiarizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
