from .schemas import DiarizationResult, SpeakerTurn

__all__ = ["DiarizationResult", "PyannoteDiarizer", "SpeakerTurn"]


def __getattr__(name):
    if name == "PyannoteDiarizer":
        from .pyannote_diarizer import PyannoteDiarizer

        return PyannoteDiarizer

    if name == "ExternalCommunityDiarizer":
        from .external_community_diarizer import ExternalCommunityDiarizer

        return ExternalCommunityDiarizer
        
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
