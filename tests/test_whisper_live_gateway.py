import asyncio
import json
import unittest

import numpy as np

from sona_ai.services.whisper_live_gateway import (
    WhisperLiveCapacityError,
    WhisperLiveConfig,
    WhisperLiveGateway,
    WhisperLiveInputError,
    WhisperLiveUnavailableError,
    _pcm16_to_float32,
    _upstream_options,
)


class FakeUpstream:
    def __init__(self, messages):
        self.messages = asyncio.Queue()
        for message in messages:
            self.messages.put_nowait(
                message if message is StopAsyncIteration else json.dumps(message)
            )
        self.sent = []

    async def send(self, data):
        self.sent.append(data)
        if data == b"END_OF_AUDIO":
            await self.messages.put(StopAsyncIteration)

    async def recv(self):
        message = await self.messages.get()
        if message is StopAsyncIteration:
            raise StopAsyncIteration
        return message

    def __aiter__(self):
        return self

    async def __anext__(self):
        message = await self.messages.get()
        if message is StopAsyncIteration:
            raise StopAsyncIteration
        return message


class FakeConnection:
    def __init__(self, upstream):
        self.upstream = upstream

    async def __aenter__(self):
        return self.upstream

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class FakeBrowser:
    def __init__(self, incoming):
        self.incoming = asyncio.Queue()
        for message in incoming:
            self.incoming.put_nowait(message)
        self.sent = []

    async def receive(self):
        return await self.incoming.get()

    async def send_json(self, data):
        self.sent.append(data)


class WhisperLiveGatewayTest(unittest.IsolatedAsyncioTestCase):
    def make_gateway(self, upstream):
        config = WhisperLiveConfig(
            ready_timeout_seconds=0.5,
            stop_timeout_seconds=0.5,
            finalization_grace_seconds=0.001,
            finalization_silence_seconds=0.01,
        )
        return WhisperLiveGateway(
            config=config,
            connect=lambda *args, **kwargs: FakeConnection(upstream),
        )

    async def test_relays_audio_and_emits_ready_transcript_and_final(self):
        upstream = FakeUpstream([
            {"uid": "ignored-until-known", "message": "SERVER_READY", "backend": "faster_whisper"},
            {
                "segments": [
                    {"start": "0", "end": "1", "text": "Hello", "completed": True},
                    {"start": "1", "end": "2", "text": "world", "completed": False},
                ]
            },
        ])
        browser = FakeBrowser([
            {"type": "websocket.receive", "bytes": b"\x00\x00\xff\x7f"},
            {"type": "websocket.receive", "text": json.dumps({"type": "stop"})},
        ])
        gateway = self.make_gateway(upstream)

        await gateway.relay(
            browser,
            model="faster-whisper-turbo",
            language="en",
        )

        self.assertEqual(browser.sent[0]["type"], "ready")
        self.assertEqual(browser.sent[0]["model"], "turbo")
        self.assertEqual(browser.sent[1]["type"], "transcript")
        self.assertEqual(browser.sent[-1]["type"], "final")
        self.assertEqual([item["text"] for item in browser.sent[-1]["segments"]], ["Hello", "world"])
        self.assertEqual(json.loads(upstream.sent[0])["model"], "turbo")
        self.assertEqual(upstream.sent[-1], b"END_OF_AUDIO")
        relayed_samples = np.frombuffer(upstream.sent[1], dtype=np.float32)
        self.assertAlmostEqual(float(relayed_samples[1]), 32767 / 32768)
        silence = np.frombuffer(upstream.sent[-2], dtype=np.float32)
        self.assertEqual(len(silence), 160)
        self.assertFalse(silence.any())

    async def test_rejects_odd_sized_pcm_frames(self):
        upstream = FakeUpstream([
            {"message": "SERVER_READY", "backend": "faster_whisper"},
        ])
        browser = FakeBrowser([
            {"type": "websocket.receive", "bytes": b"\x00"},
        ])

        with self.assertRaises(WhisperLiveInputError):
            await self.make_gateway(upstream).relay(
                browser,
                model="faster-whisper-large-v3",
                language=None,
            )

    async def test_reports_upstream_capacity_during_handshake(self):
        upstream = FakeUpstream([
            {"status": "WAIT", "message": 1.2},
        ])
        browser = FakeBrowser([])

        with self.assertRaises(WhisperLiveCapacityError):
            await self.make_gateway(upstream).relay(
                browser,
                model="faster-whisper-turbo",
                language=None,
            )

    async def test_reports_upstream_disconnect_before_browser_stop(self):
        upstream = FakeUpstream([
            {"message": "SERVER_READY", "backend": "faster_whisper"},
            StopAsyncIteration,
        ])
        browser = FakeBrowser([])

        with self.assertRaises(WhisperLiveUnavailableError):
            await self.make_gateway(upstream).relay(
                browser,
                model="faster-whisper-turbo",
                language=None,
            )

    def test_pcm_conversion_uses_full_int16_range(self):
        raw = np.asarray([-32768, 0, 32767], dtype="<i2").tobytes()
        converted = np.frombuffer(_pcm16_to_float32(raw), dtype=np.float32)

        self.assertEqual(float(converted[0]), -1.0)
        self.assertEqual(float(converted[1]), 0.0)
        self.assertAlmostEqual(float(converted[2]), 32767 / 32768)

    def test_options_are_compatible_with_float32_whisper_live_protocol(self):
        options = _upstream_options(
            session_id="session",
            model="large-v3",
            language=None,
        )

        self.assertNotIn("audio_format", options)
        self.assertTrue(options["word_timestamps"])
        self.assertEqual(options["task"], "transcribe")


if __name__ == "__main__":
    unittest.main()
