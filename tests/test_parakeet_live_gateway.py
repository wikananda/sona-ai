import asyncio
import json
import unittest

from sona_ai.services.parakeet_live_gateway import (
    ParakeetLiveConfig,
    ParakeetLiveGateway,
    _ParakeetRelay,
)
from sona_ai.services.whisper_live_gateway import (
    WhisperLiveCapacityError,
    WhisperLiveInputError,
)
from sona_ai.transcription.schemas import (
    TranscriptSegment,
    TranscriptionResult,
    WordSegment,
)


class FakeService:
    def __init__(self):
        self.prepared = []
        self.sample_lengths = []

    def prepare_live_transcription(self, **settings):
        self.prepared.append(settings)

    def transcribe_live_samples(self, samples, **settings):
        self.sample_lengths.append(len(samples))
        duration = max(0.0, len(samples) / 16000 - 0.02)
        words = []
        if duration >= 0.015:
            words.append(WordSegment("hello", 0.005, 0.015))
        if duration >= 0.05:
            words.append(WordSegment("world", 0.025, 0.05))
        return TranscriptionResult(
            segments=[TranscriptSegment(
                text=" ".join(word.word for word in words),
                start=words[0].start if words else 0,
                end=words[-1].end if words else 0,
                words=words,
            )] if words else [],
            language=settings.get("language"),
        )


class FakeBrowser:
    def __init__(self, incoming, delay=0.005):
        self.incoming = asyncio.Queue()
        for message in incoming:
            self.incoming.put_nowait(message)
        self.delay = delay
        self.sent = []

    async def receive(self):
        if self.delay:
            await asyncio.sleep(self.delay)
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


class ParakeetLiveGatewayTest(unittest.IsolatedAsyncioTestCase):
    def make_gateway(self, service=None):
        service = service or FakeService()
        return ParakeetLiveGateway(
            service,
            ParakeetLiveConfig(
                chunk_seconds=0.02,
                left_context_seconds=0.1,
                right_context_seconds=0.02,
                max_session_seconds=1,
            ),
        ), service

    async def test_streams_overlapping_snapshots_and_emits_authoritative_final(self):
        gateway, service = self.make_gateway()
        browser = FakeBrowser([
            {"type": "websocket.receive", "bytes": audio_frame()},
            {"type": "websocket.receive", "bytes": audio_frame()},
            {"type": "websocket.receive", "bytes": audio_frame()},
            stop_message(),
        ])

        await gateway.relay(
            browser,
            model="parakeet",
            device="cpu",
            language="en",
        )

        self.assertEqual(service.prepared, [{"model": "parakeet", "device": "cpu"}])
        self.assertGreaterEqual(len(service.sample_lengths), 2)
        self.assertEqual(browser.sent[0]["type"], "ready")
        self.assertEqual(browser.sent[0]["engine"], "parakeet-live")
        final = browser.sent[-1]
        self.assertEqual(final["type"], "final")
        self.assertEqual(
            [word["word"] for item in final["segments"] for word in item["words"]],
            ["hello", "world"],
        )
        self.assertEqual(service.sample_lengths[-1], round(0.08 * 16000))

    async def test_stop_before_first_window_runs_one_padded_final_decode(self):
        gateway, service = self.make_gateway()
        browser = FakeBrowser([
            {"type": "websocket.receive", "bytes": audio_frame(0.01)},
            stop_message(),
        ], delay=0)

        await gateway.relay(
            browser,
            model="parakeet",
            device="cpu",
            language=None,
        )

        self.assertEqual(service.sample_lengths, [round(0.03 * 16000)])
        self.assertEqual(browser.sent[-1]["type"], "final")

    async def test_empty_session_has_empty_final_without_model_call(self):
        gateway, service = self.make_gateway()
        browser = FakeBrowser([stop_message()], delay=0)

        await gateway.relay(
            browser,
            model="parakeet",
            device="cpu",
            language=None,
        )

        self.assertEqual(service.sample_lengths, [])
        self.assertEqual(browser.sent[-1]["segments"], [])

    async def test_rejects_odd_pcm_frame(self):
        gateway, _ = self.make_gateway()
        browser = FakeBrowser([
            {"type": "websocket.receive", "bytes": b"\x00"},
        ], delay=0)

        with self.assertRaises(WhisperLiveInputError):
            await gateway.relay(
                browser,
                model="parakeet",
                device="cpu",
                language=None,
            )

    async def test_allows_only_one_live_session(self):
        gateway, _ = self.make_gateway()
        first_browser = FakeBrowser([], delay=0)
        first = asyncio.create_task(gateway.relay(
            first_browser,
            model="parakeet",
            device="cpu",
            language=None,
        ))
        while not first_browser.sent:
            await asyncio.sleep(0)

        with self.assertRaises(WhisperLiveCapacityError):
            await gateway.relay(
                FakeBrowser([], delay=0),
                model="parakeet",
                device="cpu",
                language=None,
            )

        first.cancel()
        await asyncio.gather(first, return_exceptions=True)

    def test_rolling_buffer_uses_integer_sample_positions(self):
        gateway, service = self.make_gateway()
        relay = _ParakeetRelay(
            browser=FakeBrowser([]),
            transcription_service=service,
            config=gateway.config,
            session_id="session",
            model="parakeet",
            device="cpu",
            language=None,
        )

        relay._append_audio(audio_frame(0.2))
        samples, window_start = relay._snapshot(final=False)

        self.assertEqual(len(samples), round(0.14 * 16000))
        self.assertAlmostEqual(window_start, 0.06)


if __name__ == "__main__":
    unittest.main()
