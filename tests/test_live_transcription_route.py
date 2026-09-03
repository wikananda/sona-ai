import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sona_ai.api.routes import live_transcription as live_route
from sona_ai.services.whisper_live_gateway import WhisperLiveInputError


class FakeSession:
    def __init__(self, project_exists=True):
        self.project_exists = project_exists
        self.closed = False

    def get(self, model, project_id):
        return object() if self.project_exists else None

    def close(self):
        self.closed = True


class FakeGateway:
    def __init__(self):
        self.calls = []

    async def relay(self, websocket, **settings):
        self.calls.append(settings)


class FakeWebSocket:
    def __init__(
        self,
        start,
        whisper_gateway,
        parakeet_gateway=None,
        nemotron_gateway=None,
        whisper_mps_gateway=None,
    ):
        self.messages = [{
            "type": "websocket.receive",
            "text": json.dumps(start),
        }]
        self.sent = []
        self.accepted = False
        self.closed_with = None
        self.app = SimpleNamespace(
            state=SimpleNamespace(
                whisper_live_gateway=whisper_gateway,
                whisper_mps_live_gateway=whisper_mps_gateway,
                parakeet_live_gateway=parakeet_gateway,
                nemotron_live_gateway=nemotron_gateway,
            ),
        )

    async def accept(self):
        self.accepted = True

    async def receive(self):
        return self.messages.pop(0)

    async def send_json(self, data):
        self.sent.append(data)

    async def close(self, code=1000):
        self.closed_with = code


def valid_start(**overrides):
    start = {
        "type": "start",
        "version": 1,
        "model": "faster-whisper-turbo",
        "device": "auto",
        "language": "auto",
        "audio": {
            "encoding": "pcm_s16le",
            "sample_rate": 16000,
            "channels": 1,
        },
    }
    start.update(overrides)
    return start


class LiveTranscriptionRouteTest(unittest.IsolatedAsyncioTestCase):
    async def test_validates_project_and_relays_normalized_start(self):
        gateway = FakeGateway()
        websocket = FakeWebSocket(valid_start(), gateway)
        session = FakeSession()

        with (
            patch.object(live_route, "SessionLocal", return_value=session),
            patch.object(live_route, "resolve_device", return_value="cpu"),
        ):
            await live_route.live_transcription_socket(websocket, "project-1")

        self.assertTrue(websocket.accepted)
        self.assertTrue(session.closed)
        self.assertEqual(websocket.closed_with, 1000)
        self.assertEqual(gateway.calls, [{
            "model": "faster-whisper-turbo",
            "language": None,
        }])

    async def test_dispatches_whisper_to_local_mps_gateway(self):
        whisper_gateway = FakeGateway()
        mps_gateway = FakeGateway()
        websocket = FakeWebSocket(
            valid_start(device="auto", language="id"),
            whisper_gateway,
            whisper_mps_gateway=mps_gateway,
        )
        session = FakeSession()

        with (
            patch.object(live_route, "SessionLocal", return_value=session),
            patch.object(live_route, "resolve_device", return_value="mps"),
        ):
            await live_route.live_transcription_socket(websocket, "project-1")

        self.assertEqual(whisper_gateway.calls, [])
        self.assertEqual(mps_gateway.calls, [{
            "model": "faster-whisper-turbo",
            "device": "auto",
            "language": "id",
        }])

    async def test_dispatches_parakeet_with_device_and_language(self):
        whisper_gateway = FakeGateway()
        parakeet_gateway = FakeGateway()
        websocket = FakeWebSocket(
            valid_start(model="parakeet", device="cpu", language="en"),
            whisper_gateway,
            parakeet_gateway,
        )
        session = FakeSession()

        with patch.object(live_route, "SessionLocal", return_value=session):
            await live_route.live_transcription_socket(websocket, "project-1")

        self.assertEqual(whisper_gateway.calls, [])
        self.assertEqual(parakeet_gateway.calls, [{
            "model": "parakeet",
            "device": "cpu",
            "language": "en",
        }])
        self.assertEqual(websocket.closed_with, 1000)

    async def test_dispatches_nemotron_without_python_device(self):
        whisper_gateway = FakeGateway()
        parakeet_gateway = FakeGateway()
        nemotron_gateway = FakeGateway()
        websocket = FakeWebSocket(
            valid_start(model="nemotron-3.5", device="cpu", language="fr"),
            whisper_gateway,
            parakeet_gateway,
            nemotron_gateway,
        )
        session = FakeSession()

        with patch.object(live_route, "SessionLocal", return_value=session):
            await live_route.live_transcription_socket(websocket, "project-1")

        self.assertEqual(whisper_gateway.calls, [])
        self.assertEqual(parakeet_gateway.calls, [])
        self.assertEqual(nemotron_gateway.calls, [{
            "model": "nemotron-3.5",
            "language": "fr",
        }])
        self.assertEqual(websocket.closed_with, 1000)

    async def test_rejects_unknown_model_before_gateway(self):
        gateway = FakeGateway()
        websocket = FakeWebSocket(valid_start(model="unknown"), gateway)

        await live_route.live_transcription_socket(websocket, "project-1")

        self.assertEqual(gateway.calls, [])
        self.assertEqual(websocket.closed_with, 1008)
        self.assertEqual(websocket.sent[0]["code"], "INVALID_LIVE_AUDIO")
        self.assertFalse(websocket.sent[0]["recoverable"])

    async def test_missing_project_is_reported_without_holding_db_session(self):
        gateway = FakeGateway()
        websocket = FakeWebSocket(valid_start(), gateway)
        session = FakeSession(project_exists=False)

        with patch.object(live_route, "SessionLocal", return_value=session):
            await live_route.live_transcription_socket(websocket, "missing")

        self.assertTrue(session.closed)
        self.assertEqual(gateway.calls, [])
        self.assertEqual(websocket.closed_with, 1008)

    def test_rejects_wrong_audio_contract(self):
        for audio in (
            None,
            {"encoding": "float32", "sample_rate": 16000, "channels": 1},
            {"encoding": "pcm_s16le", "sample_rate": 48000, "channels": 1},
            {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 2},
        ):
            with self.subTest(audio=audio):
                with self.assertRaises(WhisperLiveInputError):
                    live_route._validate_start(valid_start(audio=audio))

    def test_normalizes_language_and_model(self):
        settings = live_route._validate_start(valid_start(
            model=" FASTER-WHISPER-LARGE-V3 ",
            language=" id ",
        ))

        self.assertEqual(settings["model"], "faster-whisper-large-v3")
        self.assertEqual(settings["language"], "id")


if __name__ == "__main__":
    unittest.main()
