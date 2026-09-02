"""Opt-in end-to-end smoke test for a running WhisperLive sidecar.

Example:
    PYTHONPATH=backend/src python tests/smoke_whisper_live.py /tmp/jfk.pcm

The input must be headerless, mono, 16 kHz signed 16-bit little-endian PCM.
This script isn't named ``test_*.py`` so normal unit-test discovery won't contact
the sidecar or download a model.
"""

import argparse
import asyncio
import json
from pathlib import Path

from sona_ai.services.whisper_live_gateway import (
    WHISPER_LIVE_MODELS,
    WhisperLiveConfig,
    WhisperLiveGateway,
)


class PcmBrowser:
    """Small Starlette-WebSocket stand-in that streams PCM in real time."""

    def __init__(self, pcm: bytes, frame_milliseconds: int = 200):
        frame_bytes = 16000 * 2 * frame_milliseconds // 1000
        self._frames = [
            pcm[offset : offset + frame_bytes]
            for offset in range(0, len(pcm), frame_bytes)
        ]
        self.events: list[dict] = []

    async def receive(self) -> dict:
        if self._frames:
            await asyncio.sleep(0.2)
            return {"type": "websocket.receive", "bytes": self._frames.pop(0)}
        return {
            "type": "websocket.receive",
            "text": json.dumps({"type": "stop"}),
        }

    async def send_json(self, event: dict) -> None:
        self.events.append(event)
        if event.get("type") in {"ready", "transcript", "final"}:
            print(json.dumps(event, ensure_ascii=False))


async def run(args: argparse.Namespace) -> None:
    pcm = args.pcm.read_bytes()
    if not pcm or len(pcm) % 2:
        raise ValueError("PCM input must be non-empty and contain complete int16 samples")

    # A tiny upstream override keeps local smoke tests quick while exercising
    # the same Sona model selection and relay path used by the application.
    WHISPER_LIVE_MODELS[args.sona_model] = args.upstream_model
    browser = PcmBrowser(pcm)
    gateway = WhisperLiveGateway(
        WhisperLiveConfig(
            url=args.url,
            ready_timeout_seconds=300,
            stop_timeout_seconds=40,
        )
    )

    await gateway.relay(browser, model=args.sona_model, language=args.language)

    final_events = [event for event in browser.events if event.get("type") == "final"]
    if len(final_events) != 1:
        raise AssertionError(f"Expected one final event, received {len(final_events)}")
    transcript = " ".join(
        segment["text"] for segment in final_events[0].get("segments", [])
    ).strip()
    if not transcript:
        raise AssertionError("WhisperLive returned an empty final transcript")
    print(f"PASS: {transcript}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("pcm", type=Path)
    parser.add_argument("--url", default="ws://127.0.0.1:9090")
    parser.add_argument("--language", default="en")
    parser.add_argument("--sona-model", default="faster-whisper-turbo")
    parser.add_argument("--upstream-model", default="tiny.en")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
