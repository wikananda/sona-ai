__all__ = ["NoOpAligner", "Wav2Vec2Aligner"]


def __getattr__(name):
    if name == "NoOpAligner":
        from .noop_aligner import NoOpAligner

        return NoOpAligner

    if name == "Wav2Vec2Aligner":
        from .wav2vec2_aligner import Wav2Vec2Aligner

        return Wav2Vec2Aligner
        
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
