"""Opt-in batch + realtime smoke test for a running Nemotron sidecar.

Example:
    PYTHONPATH=backend/src python tests/smoke_nemotron.py /tmp/english.pcm \
        --expect country

The input must be headerless mono, 16 kHz signed 16-bit little-endian PCM.
This script is intentionally outside normal ``test_*.py`` discovery.
"""

import argparse
import asyncio
import json
import tempfile
import wave
from copy import deepcopy
from pathlib import Path

from sona_ai.core import load_config
from sona_ai.services.nemotron_live_gateway import (
    NemotronLiveConfig,
    NemotronLiveGateway,
)
from sona_ai.transcription.nemotron_transcriber import NemotronTranscriber


class PcmBrowser:
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
            provisional = (event.get("provisional") or {}).get("text", "").strip()
            print(f"REVISION {event.get('revision')}: provisional={provisional!r}")
        elif event_type == "final":
            print(f"FINAL: {_event_text(event)}")


async def run(args: argparse.Namespace) -> None:
    pcm = args.pcm.read_bytes()
    if not pcm or len(pcm) % 2:
        raise ValueError("PCM input must be non-empty and contain complete int16 samples")

    with tempfile.TemporaryDirectory(prefix="sona-nemotron-smoke-") as temp_dir:
        wav_path = Path(temp_dir) / "audio.wav"
        _write_wav(wav_path, pcm)
        batch_text = _run_batch(args, wav_path)
        print(f"BATCH: {batch_text}")

    browser = PcmBrowser(pcm, args.frame_milliseconds)
    gateway = NemotronLiveGateway(NemotronLiveConfig(
        url=args.websocket_url,
        api_key=args.api_key,
        ready_timeout_seconds=120,
        stop_timeout_seconds=120,
    ))
    try:
        await gateway.relay(
            browser,
            model="nemotron-3.5",
            language=args.language,
        )
    finally:
        await gateway.close()

    finals = [event for event in browser.events if event.get("type") == "final"]
    if len(finals) != 1:
        raise AssertionError(f"Expected one final event, received {len(finals)}")
    live_text = _event_text(finals[0])
    _assert_transcript(live_text, args.expect, "realtime")
    _validate_words(finals[0], len(pcm) / (16000 * 2))
    print(f"PASS: batch={batch_text!r} realtime={live_text!r}")


def _run_batch(args: argparse.Namespace, wav_path: Path) -> str:
    config = deepcopy(load_config("nemotron-3.5"))
    config["server"]["url"] = args.http_url
    config["server"]["api_key"] = args.api_key
    transcriber = NemotronTranscriber(config)
    try:
        transcriber.load_models()
        result = transcriber.transcribe(str(wav_path), language=args.language)
    finally:
        transcriber.cleanup_models()
    text = " ".join(segment.text for segment in result.segments).strip()
    _assert_transcript(text, args.expect, "batch")
    return text


def _write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(pcm)


def _event_text(event: dict) -> str:
    return " ".join(
        segment.get("text", "") for segment in event.get("segments", [])
    ).strip()


def _assert_transcript(text: str, expected: str | None, mode: str) -> None:
    if not text:
        raise AssertionError(f"Nemotron returned an empty {mode} transcript")
    if expected and expected.casefold() not in text.casefold():
        raise AssertionError(f"Expected {expected!r} in {mode} transcript: {text!r}")


def _validate_words(final: dict, audio_duration: float) -> None:
    words = [
        word
        for segment in final.get("segments", [])
        for word in segment.get("words", [])
    ]
    previous_start = -1.0
    for word in words:
        start = float(word["start"])
        end = float(word["end"])
        if start < previous_start or start < 0 or end < start:
            raise AssertionError(f"Invalid final word ordering: {word!r}")
        if end > audio_duration + 0.1:
            raise AssertionError(f"Final word exceeds captured audio: {word!r}")
        previous_start = start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("pcm", type=Path)
    parser.add_argument("--http-url", default="http://127.0.0.1:8080")
    parser.add_argument("--websocket-url", default="ws://127.0.0.1:8080/v1/realtime")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--language", default="en")
    parser.add_argument("--frame-milliseconds", type=int, default=200)
    parser.add_argument("--expect")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
