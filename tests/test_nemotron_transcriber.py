import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import httpx

from sona_ai.transcription.nemotron_transcriber import NemotronTranscriber


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "http://nemotron.test")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("failed", request=request, response=response)


class _Client:
    def __init__(self, *, ready=None, transcript=None, **kwargs):
        self.ready = ready or {"ready": True}
        self.transcript = transcript or {"text": "", "duration": 0.0, "words": []}
        self.kwargs = kwargs
        self.posts = []
        self.closed = False

    def get(self, path):
        self.get_path = path
        return _Response(self.ready)

    def post(self, path, *, data, files):
        upload = files["file"]
        self.posts.append({
            "path": path,
            "data": data,
            "filename": upload[0],
            "mime": upload[2],
            "audio": upload[1].read(),
            "source_path": upload[1].name,
        })
        return _Response(self.transcript)

    def close(self):
        self.closed = True


def _write_wav(path: Path, *, sample_rate=16000, channels=1, sample_width=2):
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(
            b"\0" * (sample_rate * channels * sample_width // 10)
        )


class NemotronTranscriberTest(unittest.TestCase):
    def _config(self):
        return {
            "model": {
                "model_name": "nvidia/nemotron-3.5-asr-streaming-0.6b",
                "runtime_model": "default",
                "language": "auto",
                "device": "cpu",
            },
            "server": {
                "url": "http://nemotron.test:8080",
                "connect_timeout_seconds": 2,
                "request_timeout_seconds": 30,
            },
        }

    def test_load_and_transcribe_pcm_wav_with_word_metadata(self):
        client = _Client(transcript={
            "text": "hello world",
            "language": "en-US",
            "duration": 1.2,
            "words": [
                {"word": "hello", "start": 0.1, "end": 0.5, "confidence": 0.9},
                {
                    "word": "world",
                    "start": 0.6,
                    "end": 1.1,
                    "confidence": 1.4,
                    "speaker": 0,
                },
            ],
        })
        transcriber = NemotronTranscriber(
            self._config(),
            client_factory=lambda **kwargs: _configure_client(client, kwargs),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            _write_wav(audio_path)
            transcriber.load_models()
            result = transcriber.transcribe(str(audio_path), language="en")

        self.assertEqual(client.get_path, "/ready")
        self.assertEqual(client.kwargs["base_url"], "http://nemotron.test:8080")
        self.assertEqual(client.posts[0]["path"], "/v1/audio/transcriptions")
        self.assertEqual(client.posts[0]["data"]["language"], "en-US")
        self.assertEqual(client.posts[0]["data"]["response_format"], "verbose_json")
        self.assertEqual(client.posts[0]["mime"], "audio/wav")
        self.assertEqual(result.language, "en-US")
        self.assertEqual(result.segments[0].text, "hello world")
        self.assertEqual(result.segments[0].end, 1.2)
        self.assertEqual(result.segments[0].words[1].score, 1.0)
        self.assertEqual(result.segments[0].words[1].speaker, "0")

        transcriber.cleanup_models()
        self.assertTrue(client.closed)

    def test_non_pcm_input_is_converted_and_temporary_wav_is_removed(self):
        client = _Client(transcript={"text": "converted", "duration": 0.2})
        transcriber = NemotronTranscriber(
            self._config(),
            client_factory=lambda **kwargs: _configure_client(client, kwargs),
        )

        def fake_ffmpeg(command, **kwargs):
            self.assertEqual(command[0], "ffmpeg")
            _write_wav(Path(command[-1]))

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "recording.webm"
            source.write_bytes(b"not a wav")
            transcriber.load_models()
            with patch("sona_ai.transcription.nemotron_transcriber.subprocess.run", fake_ffmpeg):
                result = transcriber.transcribe(str(source), language="fr")
            temporary_path = Path(client.posts[0]["source_path"])

        self.assertEqual(result.segments[0].text, "converted")
        self.assertEqual(client.posts[0]["data"]["language"], "fr-FR")
        self.assertFalse(temporary_path.exists())

    def test_unsupported_language_fails_before_upload(self):
        client = _Client()
        transcriber = NemotronTranscriber(
            self._config(),
            client_factory=lambda **kwargs: _configure_client(client, kwargs),
        )
        transcriber.load_models()

        with self.assertRaisesRegex(ValueError, "does not support"):
            transcriber.transcribe("unused.wav", language="id")
        self.assertEqual(client.posts, [])

    def test_unready_server_closes_client_and_reports_actionable_error(self):
        client = _Client(ready={"ready": False})
        transcriber = NemotronTranscriber(
            self._config(),
            client_factory=lambda **kwargs: _configure_client(client, kwargs),
        )

        with self.assertRaisesRegex(RuntimeError, "Start the local"):
            transcriber.load_models()
        self.assertTrue(client.closed)

    def test_api_key_uses_bearer_header(self):
        client = _Client()
        with patch.dict("os.environ", {"SONA_NEMOTRON_API_KEY": "secret"}):
            transcriber = NemotronTranscriber(
                self._config(),
                client_factory=lambda **kwargs: _configure_client(client, kwargs),
            )
            transcriber.load_models()

        self.assertEqual(client.kwargs["headers"], {"Authorization": "Bearer secret"})


def _configure_client(client, kwargs):
    client.kwargs = kwargs
    return client


if __name__ == "__main__":
    unittest.main()
