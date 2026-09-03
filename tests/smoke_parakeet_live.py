"""Opt-in real-model smoke test for Sona's realtime Parakeet gateway.

Example:
    PYTHONPATH=backend/src python tests/smoke_parakeet_live.py /tmp/jfk.pcm \
        --nemo-model /path/to/parakeet-tdt-0.6b-v3.nemo --expect country

The input must be headerless, mono, 16 kHz signed 16-bit little-endian PCM.
This script isn't named ``test_*.py`` so normal unit-test discovery won't load
NeMo or download a model.
"""

import argparse
import asyncio
import json
from copy import deepcopy
from pathlib import Path

from sona_ai.core import load_config
from sona_ai.services.parakeet_live_gateway import ParakeetLiveGateway
from sona_ai.transcription.parakeet_transcriber import ParakeetTranscriber


class PcmBrowser:
    """Small Starlette-WebSocket stand-in that streams PCM at capture speed."""

    def __init__(self, pcm: bytes, frame_milliseconds: int):
        self._delay = frame_milliseconds / 1000
        frame_bytes = 16000 * 2 * frame_milliseconds // 1000
        self._frames = [
            pcm[offset : offset + frame_bytes]
            for offset in range(0, len(pcm), frame_bytes)
        ]
        self.events: list[dict] = []

    async def receive(self) -> dict:
        if self._frames:
            await asyncio.sleep(self._delay)
            return {"type": "websocket.receive", "bytes": self._frames.pop(0)}
        return {
            "type": "websocket.receive",
            "text": json.dumps({"type": "stop"}),
        }

    async def send_json(self, event: dict) -> None:
        self.events.append(event)
        event_type = event.get("type")
        if event_type == "ready":
            print(f"READY: {event.get('engine')} ({event.get('model')})")
        elif event_type == "transcript":
            committed = " ".join(
                segment.get("text", "") for segment in event.get("committed", [])
            ).strip()
            provisional = (event.get("provisional") or {}).get("text", "").strip()
            print(
                f"REVISION {event.get('revision')}: "
                f"committed={committed!r} provisional={provisional!r}"
            )
        elif event_type == "final":
            text = " ".join(
                segment.get("text", "") for segment in event.get("segments", [])
            ).strip()
            print(f"FINAL: {text}")


class DirectTranscriptionService:
    """Expose a loaded transcriber through the production gateway contract."""

    def __init__(self, transcriber: ParakeetTranscriber):
        self.transcriber = transcriber

    def prepare_live_transcription(self, *, model: str, device: str) -> None:
        if model != "parakeet" or device != self.transcriber.device:
            raise ValueError("Smoke-test model selection does not match the loaded model")

    def transcribe_live_samples(self, samples, *, language, model: str, device: str):
        self.prepare_live_transcription(model=model, device=device)
        return self.transcriber.transcribe_samples(samples, language=language)


def load_transcriber(args: argparse.Namespace) -> ParakeetTranscriber:
    config = deepcopy(load_config("parakeet"))
    config["model"]["device"] = args.device
    transcriber = ParakeetTranscriber(config)

    if args.nemo_model is None:
        transcriber.load_models()
        return transcriber

    if not args.nemo_model.is_file():
        raise FileNotFoundError(args.nemo_model)
    transcriber._setup_cache_environment()
    transcriber._patch_numpy_sctypes()
    from nemo.collections.asr.models import EncDecRNNTBPEModel

    transcriber.model = EncDecRNNTBPEModel.restore_from(
        str(args.nemo_model),
        map_location=args.device,
    )
    transcriber.model = transcriber.model.to(args.device)
    transcriber.model.eval()
    return transcriber


async def run(args: argparse.Namespace) -> None:
    pcm = args.pcm.read_bytes()
    if not pcm or len(pcm) % 2:
        raise ValueError("PCM input must be non-empty and contain complete int16 samples")

    transcriber = load_transcriber(args)
    try:
        browser = PcmBrowser(pcm, args.frame_milliseconds)
        gateway = ParakeetLiveGateway(DirectTranscriptionService(transcriber))
        await gateway.relay(
            browser,
            model="parakeet",
            device=args.device,
            language=args.language,
        )

        final_events = [
            event for event in browser.events if event.get("type") == "final"
        ]
        if len(final_events) != 1:
            raise AssertionError(
                f"Expected one final event, received {len(final_events)}"
            )
        transcript = " ".join(
            segment.get("text", "")
            for segment in final_events[0].get("segments", [])
        ).strip()
        if not transcript:
            raise AssertionError("Parakeet returned an empty final transcript")
        if args.expect and args.expect.casefold() not in transcript.casefold():
            raise AssertionError(
                f"Expected {args.expect!r} in final transcript: {transcript!r}"
            )
        _validate_final_words(
            final_events[0].get("segments", []),
            audio_duration=len(pcm) / (16000 * 2),
        )
        print(f"PASS: {transcript}")
    finally:
        transcriber.cleanup_models()


def _validate_final_words(segments: list[dict], *, audio_duration: float) -> None:
    words = [word for segment in segments for word in segment.get("words", [])]
    previous_start = -1.0
    previous_word = None
    for word in words:
        start = float(word["start"])
        end = float(word["end"])
        if start < previous_start - 0.01:
            raise AssertionError("Final word timestamps are not monotonic")
        if start < 0 or end < start or end > audio_duration + 0.01:
            raise AssertionError(f"Invalid final word bounds: {word!r}")
        if previous_word is not None:
            same_token = _normalized_word(previous_word["word"]) == _normalized_word(
                word["word"]
            )
            same_time = (
                abs(float(previous_word["start"]) - start) <= 0.05
                and abs(float(previous_word["end"]) - end) <= 0.05
            )
            if same_token and same_time:
                raise AssertionError(f"Duplicated boundary word: {word!r}")
        previous_start = start
        previous_word = word


def _normalized_word(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("pcm", type=Path)
    parser.add_argument("--nemo-model", type=Path)
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frame-milliseconds", type=int, default=200)
    parser.add_argument("--expect")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
