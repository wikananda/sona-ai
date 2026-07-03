__all__ = ["ExternalWav2Vec2Aligner", "NoOpAligner"]


def __getattr__(name):
    if name == "NoOpAligner":
        from .noop_aligner import NoOpAligner

        return NoOpAligner

    if name == "ExternalWav2Vec2Aligner":
        from .external_wav2vec2_aligner import ExternalWav2Vec2Aligner

        return ExternalWav2Vec2Aligner

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
