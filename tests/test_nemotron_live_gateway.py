import asyncio
import json
import unittest

from sona_ai.services.nemotron_live_gateway import (
    NemotronLiveConfig,
    NemotronLiveGateway,
)
from sona_ai.services.whisper_live_gateway import (
    WhisperLiveCapacityError,
    WhisperLiveInputError,
)


class FakeUpstream:
    def __init__(self, *, error=None):
        self.messages = asyncio.Queue()
        self.messages.put_nowait(json.dumps(
            error or {"type": "session.created", "session": {"id": "upstream"}}
        ))
        self.sent = []

    async def send(self, data):
        self.sent.append(data)
        if isinstance(data, str):
            message = json.loads(data)
            if message.get("type") == "session.update":
                await self.messages.put(json.dumps({
                    "type": "session.updated",
                    "session": message["session"],
                }))
            elif message.get("type") == "input_audio_buffer.commit":
                await self.messages.put(json.dumps({
                    "type": "input_audio_buffer.committed",
                    "item_id": "item-1",
                }))
                await self.messages.put(json.dumps({
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": "item-1",
                    "transcript": "Hello world.",
                    "language": "en-US",
                    "words": [
                        {
                            "word": "Hello",
                            "start": 0.0,
                            "end": 0.1,
                            "confidence": 0.95,
                        },
                        {
                            "word": "world",
                            "start": 0.1,
                            "end": 0.2,
                            "confidence": 0.9,
                        },
                    ],
                }))
        elif isinstance(data, bytes):
            await self.messages.put(json.dumps({
                "type": "conversation.item.input_audio_transcription.delta",
                "item_id": "item-1",
                "delta": "Hello ",
            }))

    async def recv(self):
        return await self.messages.get()

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.messages.get()


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

    async def send_json(self, event):
        self.sent.append(event)


def stop_message():
    return {
        "type": "websocket.receive",
        "text": json.dumps({"type": "stop"}),
    }


class NemotronLiveGatewayTest(unittest.IsolatedAsyncioTestCase):
    def make_gateway(self, upstream, *, api_key=""):
        connected = []

        def connect(url, **kwargs):
            connected.append((url, kwargs))
            return FakeConnection(upstream)

        gateway = NemotronLiveGateway(
            NemotronLiveConfig(
                url="ws://nemotron.test:8080/v1/realtime?client=sona",
                api_key=api_key,
                ready_timeout_seconds=0.5,
                stop_timeout_seconds=0.5,
                max_session_seconds=1,
            ),
            connect=connect,
        )
        return gateway, connected

    async def test_relays_raw_pcm_and_emits_ready_partial_and_word_timed_final(self):
        upstream = FakeUpstream()
        gateway, connected = self.make_gateway(upstream)
        pcm = b"\x00\x00" * 3200
        browser = FakeBrowser([
            {"type": "websocket.receive", "bytes": pcm},
            stop_message(),
        ])

        await gateway.relay(
            browser,
            model="nemotron-3.5",
            language="en",
        )

        self.assertEqual(connected[0][0], "ws://nemotron.test:8080/v1/realtime?client=sona")
        session_update = json.loads(upstream.sent[0])
        self.assertEqual(session_update["session"]["language"], "en-US")
        self.assertTrue(session_update["session"]["word_timestamps"])
        self.assertEqual(upstream.sent[1], pcm)
        self.assertEqual(json.loads(upstream.sent[-1])["type"], "input_audio_buffer.commit")
        self.assertEqual(browser.sent[0]["engine"], "nemotron-live")
        self.assertTrue(any(event["type"] == "transcript" for event in browser.sent))
        final = browser.sent[-1]
        self.assertEqual(final["type"], "final")
        self.assertEqual(final["segments"][0]["text"], "Hello world.")
        self.assertEqual(final["segments"][0]["words"][0]["score"], 0.95)

    async def test_api_key_is_added_without_dropping_existing_query(self):
        upstream = FakeUpstream()
        gateway, connected = self.make_gateway(upstream, api_key="a secret")
        browser = FakeBrowser([stop_message()])

        await gateway.relay(browser, model="nemotron-3.5", language=None)

        self.assertIn("client=sona", connected[0][0])
        self.assertIn("api_key=a+secret", connected[0][0])

    async def test_rejects_odd_pcm_frame(self):
        gateway, _ = self.make_gateway(FakeUpstream())
        browser = FakeBrowser([
            {"type": "websocket.receive", "bytes": b"\x00"},
        ])

        with self.assertRaises(WhisperLiveInputError):
            await gateway.relay(browser, model="nemotron-3.5", language="en")

    async def test_rejects_unsupported_language_before_connecting(self):
        gateway, connected = self.make_gateway(FakeUpstream())

        with self.assertRaisesRegex(WhisperLiveInputError, "does not support"):
            await gateway.relay(FakeBrowser([]), model="nemotron-3.5", language="id")
        self.assertEqual(connected, [])

    async def test_maps_upstream_capacity_error(self):
        upstream = FakeUpstream(error={
            "type": "error",
            "error": {"code": "session_limit_reached", "message": "busy"},
        })
        gateway, _ = self.make_gateway(upstream)

        with self.assertRaises(WhisperLiveCapacityError):
            await gateway.relay(FakeBrowser([]), model="nemotron-3.5", language="en")

    async def test_rejects_wrong_model(self):
        gateway, _ = self.make_gateway(FakeUpstream())

        with self.assertRaises(WhisperLiveInputError):
            await gateway.relay(FakeBrowser([]), model="parakeet", language="en")


if __name__ == "__main__":
    unittest.main()
