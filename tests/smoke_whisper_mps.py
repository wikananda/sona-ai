"""Hardware smoke test for Sona's Apple MPS Whisper backend.

Usage:
    PYTHONPATH=backend/src conda run -n sona-ai python \
        tests/smoke_whisper_mps.py data/raw/audio/audio.mp3
"""

import argparse
import statistics
import subprocess
import time
from copy import deepcopy

import numpy as np

from sona_ai.core import load_config
from sona_ai.transcription.whisper_mps_transcriber import WhisperMpsTranscriber


def decode_pcm(path: str, seconds: float) -> np.ndarray:
    process = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            path,
            "-t",
            str(seconds),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    return np.frombuffer(process.stdout, dtype="<f4").copy()


def assert_monotonic(result) -> None:
    words = [word for segment in result.segments for word in segment.words]
    previous_start = -1.0
    for word in words:
        if word.start is None or word.end is None:
            raise AssertionError("Whisper returned a word without timestamps")
        if word.start < previous_start or word.end < word.start:
            raise AssertionError("Whisper returned non-monotonic word timestamps")
        previous_start = word.start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--language", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-id", default="whisper-mps-turbo")
    args = parser.parse_args()
    if args.seconds < 4:
        parser.error("--seconds must be at least 4 for the rolling-window check")

    samples = decode_pcm(args.audio_path, args.seconds)
    if not samples.size:
        raise RuntimeError("The smoke-test audio decoded to zero samples")

    config = deepcopy(load_config("whisper-mps-turbo"))
    config["_sona_managed_model_id"] = args.cache_id
    if args.model_name:
        config["model"]["model_name"] = args.model_name
    if args.revision:
        config["model"]["revision"] = args.revision
    config["model"]["warmup_seconds"] = 1.0
    transcriber = WhisperMpsTranscriber(config)
    try:
        load_started = time.perf_counter()
        transcriber.load_models()
        load_seconds = time.perf_counter() - load_started
        pipeline_device = getattr(transcriber.pipeline, "device", None)
        if getattr(pipeline_device, "type", None) != "mps":
            raise AssertionError(f"Whisper pipeline is not on MPS: {pipeline_device}")

        whole_started = time.perf_counter()
        whole = transcriber.transcribe_samples(samples, language=args.language)
        whole_seconds = time.perf_counter() - whole_started
        assert_monotonic(whole)
        if not any(segment.text.strip() for segment in whole.segments):
            raise AssertionError("Whisper returned an empty transcript")

        window_latencies = []
        maximum_window = min(args.seconds, 16.0)
        for end_seconds in np.arange(4.0, maximum_window + 0.1, 2.0):
            window = samples[: round(end_seconds * 16000)]
            started = time.perf_counter()
            result = transcriber.transcribe_samples(window, language=args.language)
            window_latencies.append(time.perf_counter() - started)
            assert_monotonic(result)

        p95_index = max(0, round(0.95 * len(window_latencies)) - 1)
        p95 = sorted(window_latencies)[p95_index]
        print(f"device=mps model={transcriber.model_name}")
        print(f"load_seconds={load_seconds:.2f}")
        print(
            f"audio_seconds={len(samples) / 16000:.2f} "
            f"whole_seconds={whole_seconds:.2f} "
            f"realtime_factor={(len(samples) / 16000) / whole_seconds:.2f}x"
        )
        print(
            f"rolling_mean_seconds={statistics.mean(window_latencies):.2f} "
            f"rolling_p95_seconds={p95:.2f}"
        )
        print(f"text={whole.raw['text'][:240]}")
    finally:
        transcriber.cleanup_models()


if __name__ == "__main__":
    main()
