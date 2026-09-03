import asyncio
import json
import unittest

from sona_ai.services.whisper_live_gateway import WhisperLiveInputError
from sona_ai.services.whisper_mps_live_gateway import (
    WhisperMpsLiveConfig,
    WhisperMpsLiveGateway,
)
from sona_ai.transcription.schemas import TranscriptSegment, TranscriptionResult, WordSegment


class FakeService:
    def __init__(self):
        self.prepared = []
        self.calls = []

    def prepare_live_transcription(self, **settings):
        self.prepared.append(settings)

    def transcribe_live_samples(self, samples, **settings):
        self.calls.append((len(samples), settings))
        duration = len(samples) / 16000
        words = [WordSegment("hello", 0.005, min(0.05, duration))]
        return TranscriptionResult(
            segments=[TranscriptSegment(
                text="hello",
                start=words[0].start or 0.0,
                end=words[0].end or 0.0,
                words=words,
            )],
            language=settings.get("language"),
        )


class FakeBrowser:
    def __init__(self, incoming):
        self.incoming = asyncio.Queue()
        for message in incoming:
            self.incoming.put_nowait(message)
        self.sent = []

    async def receive(self):
        return await self.incoming.get()

    async def send_json(self, event):
        self.sent.append(event)


def audio_frame(seconds=0.02):
    return b"\x00\x00" * round(16000 * seconds)


def stop_message():
    return {
        "type": "websocket.receive",
        "text": json.dumps({"type": "stop"}),
    }


class WhisperMpsLiveGatewayTest(unittest.IsolatedAsyncioTestCase):
    def make_gateway(self):
        service = FakeService()
        gateway = WhisperMpsLiveGateway(
            service,
            WhisperMpsLiveConfig(
                chunk_seconds=0.02,
                left_context_seconds=0.1,
                right_context_seconds=0.02,
                max_session_seconds=1,
            ),
        )
        return gateway, service

    async def test_streams_on_mps_and_emits_authoritative_final(self):
        gateway, service = self.make_gateway()
        browser = FakeBrowser([
            {"type": "websocket.receive", "bytes": audio_frame()},
            {"type": "websocket.receive", "bytes": audio_frame()},
            {"type": "websocket.receive", "bytes": audio_frame()},
            stop_message(),
        ])

        await gateway.relay(
            browser,
            model="faster-whisper-turbo",
            device="mps",
            language="id",
        )

        self.assertEqual(service.prepared, [{
            "model": "faster-whisper-turbo",
            "device": "mps",
        }])
        self.assertGreaterEqual(len(service.calls), 1)
        self.assertEqual(service.calls[-1][1]["language"], "id")
        self.assertEqual(browser.sent[0]["type"], "ready")
        self.assertEqual(browser.sent[0]["engine"], "whisper-mps-live")
        self.assertEqual(browser.sent[-1]["type"], "final")
        self.assertEqual(browser.sent[-1]["segments"][0]["text"], "hello")

    async def test_accepts_large_v3_and_rejects_non_whisper_models(self):
        gateway, _ = self.make_gateway()
        browser = FakeBrowser([stop_message()])

        await gateway.relay(
            browser,
            model="faster-whisper-large-v3",
            device="mps",
            language=None,
        )
        self.assertEqual(browser.sent[0]["model"], "faster-whisper-large-v3")

        with self.assertRaises(WhisperLiveInputError):
            await gateway.relay(
                FakeBrowser([]),
                model="parakeet",
                device="mps",
                language="en",
            )


if __name__ == "__main__":
    unittest.main()
