# Sona AI

Sona AI is a privacy-first, local-first speech workspace for working with personal and team audio recordings. It lets you create projects, upload or record audio, run transcription, optionally extract speakers, summarize transcripts, chat with a recording, and export results while keeping the core workflow under your control.

The app is designed around replaceable speech components instead of one locked pipeline. Transcription, alignment, diarization, and summarization are configured separately so models can be swapped without rewriting the whole app.

The default product direction is not interview-specific. Sona is meant for private audio workflows such as meetings, notes, calls, research sessions, lectures, voice memos, and any recording where the user wants control over where audio, transcripts, summaries, and API keys go.

## Features

- Project-based workspace with multiple recordings per project.
- Upload audio, record in the browser, or run basic live transcription.
- Browser recording can capture microphone audio, system audio, or both.
- Local storage by default for audio, transcripts, summaries, and chat data.
- Transcription engines include Parakeet and Faster-Whisper large-v3/turbo.
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
npx tsc --noEmit
npm run lint
```

## Deployment Notes

The frontend can be deployed to Vercel.

The backend is model-heavy and should usually run on your laptop, a VM, or a GPU/CPU machine with enough memory. For a prototype, you can expose the local backend through a tunnel and set the deployed frontend's `NEXT_PUBLIC_API_BASE_URL` to that tunnel URL.

### Hugging Face Backend Docker Space

This repo includes a minimal backend-only Docker path for a Hugging Face Docker Space. It is meant for demo use, not durable production hosting.

The Docker image:

- Runs FastAPI on port `7860`.
- Uses `configs/speech.hf.yaml`.
- Runs Parakeet transcription on CPU.
- Disables alignment and diarization to avoid extra external environments inside the demo container.
- Expects BYOK for summary/chat; local LLM dependencies are omitted from the minimal image.
- Stores uploaded recordings, SQLite data, and model cache inside the Space filesystem.

Create a Hugging Face Space with SDK `Docker`, push this repo, and set these Space variables:

```bash
SONA_SPEECH_CONFIG=speech.hf
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
docker run --rm -p 7860:7860 -e SONA_SPEECH_CONFIG=speech.hf sona-ai-backend-hf
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
