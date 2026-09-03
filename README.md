# Sona AI

Sona AI is a privacy-first, local-first speech workspace for working with personal and team audio recordings. It lets you create projects, upload or record audio, run transcription, optionally extract speakers, summarize transcripts, chat with a recording, and export results while keeping the core workflow under your control.

The app is designed around replaceable speech components instead of one locked pipeline. Transcription, alignment, diarization, and summarization are configured separately so models can be swapped without rewriting the whole app.

The default product direction is not interview-specific. Sona is meant for private audio workflows such as meetings, notes, calls, research sessions, lectures, voice memos, and any recording where the user wants control over where audio, transcripts, summaries, and API keys go.

## Features

- Project-based workspace with multiple recordings per project.
- Upload audio, record in the browser, or run basic live transcription.
- Browser recording can capture microphone audio, system audio, or both.
- Local storage by default for audio, transcripts, summaries, and chat data.
- Transcription engines include Nemotron 3.5 ASR, Parakeet, and Faster-Whisper large-v3/turbo.
- Optional Wav2Vec2 alignment through an external WhisperX-based aligner environment.
- Optional speaker diarization through an external pyannote Community-1 environment.
- Speaker extraction can run during transcription or later.
- Audio playback with clickable transcript timestamps.
- Transcript text editing and speaker rename/edit support.
- Adaptive markdown summaries with local LLM or BYOK OpenAI-compatible providers.
- Per-recording chat using OpenAI-compatible BYOK settings.
- Frontend-only PDF export for transcript or summary.
- Original audio is preserved for playback while a normalized ASR copy is generated for models.

## Privacy Model

Sona is built around local ownership first:

- Audio files are stored locally under `data/projects/`.
- The local database stores project, recording, transcript, summary, and chat records.
- Local ASR, alignment, diarization, and LLM paths can run without sending recording content to a hosted API.
- BYOK features are explicit opt-in. When used, only the requested summary or chat payload is sent to the selected OpenAI-compatible provider.
- Browser-stored API keys are intended for prototype use. Prefer short-lived or limited-scope keys when testing BYOK.

This does not make the app automatically production-secure. For shared deployment, add proper authentication, encrypted storage, secrets management, and a clear data retention policy.

## Architecture

```text
frontend/                 Next.js app
backend/src/sona_ai/      FastAPI backend package
configs/                  Runtime model and pipeline config
tools/                    Standalone helper scripts
tests/                    Backend tests and smoke scripts
data/projects/            Local project audio, transcripts, summaries, chat data
.models/                  Local model cache target
```

Important backend modules:

```text
backend/src/sona_ai/api/              FastAPI routes
backend/src/sona_ai/services/         App orchestration services
backend/src/sona_ai/pipelines/        Speech pipeline composition
backend/src/sona_ai/transcription/    ASR engines
backend/src/sona_ai/alignment/        Alignment engines
backend/src/sona_ai/diarization/      Diarization engines
backend/src/sona_ai/summarization/    Local and BYOK LLM summarizers
backend/src/sona_ai/storage/          Audio and file storage helpers
backend/src/sona_ai/db/               SQLite/SQLAlchemy persistence
```

## Environments

The main app runs in `sona-ai`. Two optional environments keep dependency-heavy speech tools isolated:

- `sona-ai`: backend API, frontend integration, ASR, summarization, storage.
- `sona-aligner`: external WhisperX/Wav2Vec2 alignment.
- `sona-diarization`: external pyannote Community-1 diarization.

This split avoids forcing WhisperX and pyannote dependency constraints into the main backend environment.

## Prerequisites

- Conda or another Python environment manager.
- Python 3.12 for the main environment.
- Node.js and npm for the frontend.
- `ffmpeg` available on your PATH.
- Hugging Face token if you use gated models such as pyannote.

On macOS, install ffmpeg with:

```bash
brew install ffmpeg
```

## Backend Setup

Create and install the main backend environment:

```bash
conda create -n sona-ai python=3.12
conda activate sona-ai
pip install -r backend/requirements.txt
pip install -e backend
```

Create a `.env` file if you need Hugging Face access:

```bash
cp .env.example .env
```

Example `.env` values:

```bash
HF_TOKEN=your_huggingface_token
SONA_HF_CACHE=.models
HF_HOME=.models
HUGGINGFACE_HUB_CACHE=.models/hub
TRANSFORMERS_CACHE=.models/transformers
```

Start the backend:

```bash
conda activate sona-ai
uvicorn sona_ai.api.main:app --app-dir backend/src --host 0.0.0.0 --port 8000 --reload
```

The API will run at:

```text
http://localhost:8000
```

## Realtime Whisper with WhisperLive

Whisper live transcription uses an isolated WhisperLive sidecar. Keeping it outside the main Python environment avoids its speech dependency pins conflicting with Sona's batch transcription stack. The browser connects only to Sona; Sona validates the session and proxies small PCM frames to WhisperLive.

Start the pinned CPU sidecar before the Sona backend:

```bash
docker compose -f compose.whisper-live.yml up --build -d
```

The compose file builds WhisperLive from the reviewed commit `c06c33982d2640b7aa6323acad45d5d8e63ad953`, binds it only to localhost, allows one client, and raises the meeting limit to six hours. The first session downloads and warms the selected Whisper model, so it can take longer to become ready.

The backend defaults to this sidecar URL:

```bash
SONA_WHISPER_LIVE_URL=ws://127.0.0.1:9090
SONA_WHISPER_LIVE_MAX_SESSIONS=1
```

If the backend runs in another container, set `SONA_WHISPER_LIVE_URL` to the sidecar's reachable service URL instead. For NVIDIA deployment, change the compose Dockerfile to `docker/Dockerfile.gpu` and configure Docker GPU access.

Whisper uses low-latency WebSocket streaming when the sidecar is available. If it is not available or disconnects, the browser keeps the original MediaRecorder audio and falls back to compatibility transcription. If the preview is incomplete, Sona submits that same original file through the normal whole-file background transcription flow.

Whole-file uploads do not use WhisperLive and are unchanged: they are saved first, then processed by Sona's normal transcription worker.

### Whisper on Apple GPU

On Apple Silicon, selecting `Auto device` or `MPS` routes both uploaded files
and realtime Whisper sessions through PyTorch's Metal Performance Shaders
backend. Sona verifies that the loaded pipeline reports `mps` before it sends
the realtime ready event. Selecting `CPU` keeps the existing Faster-Whisper
and WhisperLive paths, so Apple GPU support does not remove the stable CPU
fallback.

The model manager lists separate Apple GPU weights because Hugging Face
Transformers and Faster-Whisper use incompatible model formats. Turbo is the
recommended realtime model. Sona uses batch size 1 and a bounded overlapping
window for live inference; uploaded files use batch size 4 for throughput.
The first preview normally appears after four seconds of audio, followed by a
revision every two seconds. The defaults can be tuned with:

```bash
SONA_WHISPER_MPS_LIVE_MAX_SESSIONS=1
SONA_WHISPER_MPS_LIVE_CHUNK_SECONDS=2
SONA_WHISPER_MPS_LIVE_LEFT_CONTEXT_SECONDS=12
SONA_WHISPER_MPS_LIVE_RIGHT_CONTEXT_SECONDS=2
```

Do not raise the session limit on a 24 GB Mac without measuring unified-memory
pressure. If MPS loading, inference, or live pacing fails, Sona preserves the
original MediaRecorder audio and uses the existing compatibility flow.

Run the hardware smoke test on real English or Indonesian speech before
changing the timing defaults:

```bash
PYTHONPATH=backend/src conda run -n sona-ai python \
  tests/smoke_whisper_mps.py data/raw/audio/audio.mp3 --language id
```

## Nemotron 3.5 ASR (batch and realtime)

Nemotron uses NVIDIA's official NeMo-Speech.cpp runtime as an isolated local sidecar. Sona uses the same loaded model for whole-file HTTP transcription and native cache-aware realtime WebSocket transcription, so the existing Python NeMo and Transformers versions do not need to change.

Install the pinned NeMo-Speech.cpp `v0.1.0` release on Linux or macOS. Its installer verifies the selected native archive against NVIDIA's published SHA-256 file and selects Metal on Apple Silicon, CUDA when available, or CPU otherwise:

```bash
curl -fsSL https://raw.githubusercontent.com/NVIDIA/NeMo-Speech.cpp/v0.1.0/scripts/install.sh | \
  sh -s -- --version 0.1.0 --binary-only
export PATH="$HOME/.local/bin:$PATH"
nemo-speech --version
```

Download **Nemotron 3.5 ASR 0.6B** from Sona's model manager. Sona stores the revision-pinned q8 GGUF at `.models/nemotron-3.5/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf` and verifies its expected size. Then start the loopback-only sidecar:

```bash
./tools/nemotron/run_server.sh
```

The backend defaults to:

```bash
SONA_NEMOTRON_URL=http://127.0.0.1:8080
SONA_NEMOTRON_LIVE_URL=ws://127.0.0.1:8080/v1/realtime
SONA_NEMOTRON_LIVE_MAX_SESSIONS=1
```

To protect a non-loopback deployment, export the same `SONA_NEMOTRON_API_KEY` before starting both the sidecar and Sona. Keep the default loopback binding for a local installation.

The published checkpoint supports 32 transcription-ready locales covering Arabic, Bulgarian, Chinese, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Hindi, Hungarian, Italian, Japanese, Korean, Norwegian Bokmal, Polish, Portuguese, Romanian, Russian, Slovak, Spanish, Swedish, Turkish, Ukrainian, and Vietnamese. Specifying the language is more reliable than auto-detection. Indonesian is not supported; choose Whisper for Indonesian audio.

Both live and whole-file flows preserve the original recording. If the sidecar is unavailable during live capture, Sona keeps recording and moves to compatibility mode; the compatibility upload still requires the same Nemotron sidecar when Nemotron remains selected.

An opt-in end-to-end smoke test exercises both APIs against a running sidecar:

```bash
PYTHONPATH=backend/src conda run -n sona-ai python tests/smoke_nemotron.py /tmp/english.pcm
```

## Realtime Parakeet

Parakeet live transcription is built into the Sona backend and does not need a sidecar. The browser sends the same mono 16 kHz PCM stream used by WhisperLive, and the backend sends samples directly to the already-loaded NeMo model without creating temporary audio files or running ffmpeg.

Parakeet TDT 0.6B v3 is a full-context model rather than a cache-aware streaming checkpoint. Sona therefore uses bounded buffered inference based on NeMo's chunked-transcription layout: 10 seconds of left context, a 2-second decoding chunk, and 2 seconds of right context. The first preview normally appears after about 4 seconds of captured audio, followed by revisions every 2 seconds. Words appear provisionally right away and are committed after agreeing in two consecutive stable windows, which prevents a context-sensitive guess from becoming immutable too early.

The defaults can be tuned in `.env`:

```bash
SONA_PARAKEET_LIVE_MAX_SESSIONS=1
SONA_PARAKEET_LIVE_CHUNK_SECONDS=2
SONA_PARAKEET_LIVE_LEFT_CONTEXT_SECONDS=10
SONA_PARAKEET_LIVE_RIGHT_CONTEXT_SECONDS=2
```

Keep the session limit at one unless the host has enough compute for concurrent NeMo inference. If realtime decoding is unavailable or cannot keep up, the UI preserves the original MediaRecorder recording and offers the existing compatibility transcription path. Normal whole-file Parakeet uploads are unchanged.

Sona currently enables English for Parakeet. Choose a Whisper model for Indonesian recordings.

## Optional Aligner Environment

Use this when `configs/speech.yaml` enables `wav2vec2_external`.

```bash
conda create -n sona-aligner python=3.12
conda activate sona-aligner
pip install -r backend/requirements-aligner.txt
```

The backend calls the aligner as an external tool, so the main `sona-ai` environment does not need WhisperX installed.

## Optional Diarization Environment

Use this when `configs/speech.yaml` enables `community_external`.

```bash
conda create -n sona-diarization python=3.12
conda activate sona-diarization
pip install -r backend/requirements-diarization.txt
```

The diarization tool writes speaker turns that the backend later assigns to transcript segments.

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend will run at:

```text
http://localhost:3000
```

If the frontend is not running on the same machine as the backend, set:

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

For Vercel or a tunneled backend, set `NEXT_PUBLIC_API_BASE_URL` to the public backend URL without a trailing slash.

## Configuration

The main speech pipeline is controlled by:

```text
configs/speech.yaml
```

Current default shape:

```yaml
transcription:
  engine: "parakeet"
  config: "parakeet"
  device: "auto"

alignment:
  enabled: true
  engine: "wav2vec2_external"
  config: "wav2vec2"
  device: "cpu"

diarization:
  enabled: true
  engine: "community_external"
  config: "diarization-community"
  device: "cpu"

summarization:
  config: "qwen"
```

Dedicated config files live beside it:

- `configs/parakeet.yaml`
- `configs/faster-whisper-large-v3.yaml`
- `configs/faster-whisper-turbo.yaml`
- `configs/wav2vec2.yaml`
- `configs/diarization-community.yaml`
- `configs/speech.hf-full.yaml`
- `configs/qwen.yaml`
- `configs/llama.yaml`
- `configs/gemma.yaml`

The UI can choose language, model, device, and speaker options for a recording. The backend maps those choices onto the configured engines.

## Typical Workflow

1. Start the backend and frontend.
2. Create a project.
3. Click `+` in the recording sidebar.
4. Choose upload, browser recording, or live transcription.
5. Select language, model, device, and whether to extract speakers.
6. Review the transcript and use the audio player to verify timestamps.
7. Edit transcript text or speaker names if needed.
8. Use the Summary tab for local or BYOK summarization.
9. Use the Chat tab to ask questions about the recording.
10. Export transcript or summary as PDF.

## Audio Storage

Uploaded and recorded audio is preserved in its original format for playback:

```text
data/projects/<project_id>/<recording_id>.<original_extension>
```

For transcription, the backend creates a normalized ASR copy:

```text
data/projects/<project_id>/<recording_id>.asr.wav
```

This is intentional. Playback uses the original file, while ASR/alignment/diarization use the normalized 16 kHz mono WAV copy.

## BYOK Providers

BYOK currently targets OpenAI-compatible chat completion APIs. This includes providers such as:

- OpenAI
- Groq
- OpenRouter
- Other providers that expose an OpenAI-compatible `/chat/completions` API

API keys entered in the frontend are handled client-side and sent only when making the requested summarization or chat call. Avoid using production keys in a shared or untrusted browser.

## Checks

Backend targeted test:

```bash
PYTHONPATH=backend/src conda run -n sona-ai python -m unittest discover -s tests -p 'test_audio_storage.py'
```

Frontend checks:

```bash
cd frontend
npm test
npx tsc --noEmit
npm run lint
npm run build
```

## Deployment Notes

The frontend can be deployed to Vercel.

The backend is model-heavy and should usually run on your laptop, a VM, or a GPU/CPU machine with enough memory. For a prototype, you can expose the local backend through a tunnel and set the deployed frontend's `NEXT_PUBLIC_API_BASE_URL` to that tunnel URL.

### Hugging Face Backend Docker Space

This repo includes a backend-only Docker path for a Hugging Face Docker Space. It is meant for demo use, not durable production hosting.

The Docker image:

- Runs FastAPI on port `7860`.
- Uses `configs/speech.hf-full.yaml`.
- Creates `sona-ai`, `sona-aligner`, and `sona-diarization` conda environments inside the container.
- Supports Parakeet, Faster-Whisper, Wav2Vec2 alignment, and pyannote Community-1 diarization on CPU.
- Expects BYOK for summary/chat; local LLM dependencies are omitted from the image.
- Stores uploaded recordings, SQLite data, and model cache inside the Space filesystem.

Create a Hugging Face Space with SDK `Docker`, push this repo, and set these Space variables:

```bash
SONA_SPEECH_CONFIG=speech.hf-full
SONA_HF_CACHE=.models
HF_HOME=.models
HUGGINGFACE_HUB_CACHE=.models/hub
TRANSFORMERS_CACHE=.models/transformers
```

Set this as a Space secret only if needed:

```bash
HF_TOKEN=your_huggingface_token
```

Local Docker smoke test:

```bash
docker build -t sona-ai-backend-hf .
docker run --rm -p 7860:7860 -e SONA_SPEECH_CONFIG=speech.hf-full sona-ai-backend-hf
```

Then open:

```text
http://localhost:7860/docs
```

For a deployed frontend, set:

```bash
NEXT_PUBLIC_API_BASE_URL=https://<space-owner>-<space-name>.hf.space
```

Without persistent Hugging Face storage, Space restarts can remove:

- `backend/data/sona.db`
- `data/projects/`
- `.models/`

That is acceptable for a short demo, but it means projects, recordings, and downloaded models may disappear or redownload after restart.

For wider deployment, use persistent storage for:

- SQLite or another database.
- `data/projects/`.
- `.models/`.

## Status

This is an active prototype. The main architecture goal is to keep the app independent from any single speech stack:

- ASR can change without replacing diarization.
- Diarization can run later.
- Alignment can be isolated from the main backend.
- Local LLM and BYOK LLM paths can coexist.
