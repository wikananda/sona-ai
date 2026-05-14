# pyannote Community-1 Diarization Tool

This tutorial explains how to add `pyannote/speaker-diarization-community-1` to Sona AI without breaking the current backend environment.

The important architectural choice is this:

```text
Do not install pyannote.audio 4.x directly into the main sona-ai environment yet.
```

The current backend environment contains WhisperX and Nemo/Parakeet. Those packages have dependency constraints that conflict with pyannote 4.x:

- `whisperx==3.7.4` requires `pyannote-audio<4.0.0`
- Nemo commonly requires older `protobuf`
- WhisperX and numba require narrower `numpy` versions than pyannote 4 may install

So the safer approach is to run Community-1 as a standalone diarization tool in a separate conda environment and have the backend call that tool later.

## Target Architecture

The goal is to make diarization independent from ASR:

```text
Main backend env: sona-ai
  - FastAPI
  - Parakeet/Nemo
  - WhisperX
  - pyannote 3.x if still needed
  - speaker assignment

Separate diarization env: sona-diarization
  - pyannote.audio 4.x
  - pyannote Community-1

Tool boundary:
  audio file -> Community-1 tool -> diarization JSON -> backend SpeakerAssigner
```

The standalone tool should output a neutral JSON format:

```json
[
  {
    "start": 0.42,
    "end": 3.91,
    "speaker": "SPEAKER_00"
  },
  {
    "start": 4.12,
    "end": 7.33,
    "speaker": "SPEAKER_01"
  }
]
```

This keeps Community-1 dependency problems isolated while still allowing Sona AI to use its diarization result.

## References

- https://huggingface.co/pyannote/speaker-diarization-community-1
- https://pypi.org/project/pyannote-audio/
- https://github.com/pyannote/pyannote-audio/releases

## 1. Repair the Main Environment if Needed

If you already installed pyannote 4 into `sona-ai` and saw dependency conflict messages, repair the main environment first.

Activate the main backend env:

```bash
conda activate sona-ai
```

Reinstall versions compatible with the current stack:

```bash
python -m pip install \
  "numpy>=2.0.2,<2.1.0" \
  "protobuf==3.20.3" \
  "pyannote.audio>=3.3.2,<4.0.0"
```

Verify:

```bash
python -c "import numpy, google.protobuf, pyannote.audio; print(numpy.__version__); print(google.protobuf.__version__); print(pyannote.audio.__version__)"
```

Expected shape:

```text
2.0.x
3.20.3
3.x
```

Do not install Community-1 dependencies into this environment for now.

## 2. Accept Community-1 Access

Open the model page while logged into Hugging Face:

```text
https://huggingface.co/pyannote/speaker-diarization-community-1
```

Accept the user conditions.

Make sure the repo root `.env` contains:

```bash
HF_TOKEN=hf_...
```

The standalone tool will load this token from `.env`.

## 3. Create the Separate Diarization Environment

Create a new conda environment:

```bash
conda create -n sona-diarization python=3.12 -y
conda activate sona-diarization
```

Install Community-1 dependencies:

```bash
python -m pip install "pyannote.audio>=4.0,<5" python-dotenv
```

Install `ffmpeg` if it is not already installed:

```bash
brew install ffmpeg
```

Verify pyannote:

```bash
python -c "import pyannote.audio; print(pyannote.audio.__version__)"
```

Expected shape:

```text
4.x
```

## 4. Add a Dedicated Requirements File

Create:

```text
backend/requirements-diarization-community.txt
```

Suggested content:

```text
pyannote.audio>=4.0,<5
python-dotenv
```

Do not merge this into `backend/requirements.txt` yet. Keep it separate so the main backend environment stays compatible with WhisperX and Nemo.

## 5. Create the Standalone Community-1 Tool

Create this folder:

```text
tools/diarization
```

Create this file:

```text
tools/diarization/diarize_community.py
```

Use this implementation:

```python
import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline


MODEL_NAME = "pyannote/speaker-diarization-community-1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pyannote Community-1 diarization.")
    parser.add_argument("audio_path", help="Path to the input audio file.")
    parser.add_argument("output_path", help="Path to write diarization JSON.")
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--cache-dir", default="cp/hf_cache/pyannote-community")
    parser.add_argument("--regular", action="store_true", help="Use regular diarization instead of exclusive diarization.")
    return parser.parse_args()


def normalize_speaker(label: object) -> str:
    label_text = str(label)
    if label_text.startswith("SPEAKER_"):
        return label_text
    if label_text.isdigit():
        return f"SPEAKER_{int(label_text):02d}"
    return label_text


def annotation_to_turns(annotation) -> list[dict]:
    turns = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        turns.append(
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "speaker": normalize_speaker(speaker),
            }
        )
    return turns


def normalize_audio(input_path: Path) -> Path:
    output_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output_path = Path(output_file.name)
    output_file.close()

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to normalize audio before diarization.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed to normalize audio: {input_path}") from exc

    return output_path


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    load_dotenv(repo_root / ".env")

    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN is not set. Add it to the repo root .env file.")

    audio_path = Path(args.audio_path)
    output_path = Path(args.output_path)
    cache_dir = Path(args.cache_dir)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline.from_pretrained(
        MODEL_NAME,
        token=token,
        cache_dir=cache_dir,
    )
    pipeline.to(torch.device(args.device))

    diarization_kwargs = {}
    if args.num_speakers is not None:
        diarization_kwargs["num_speakers"] = args.num_speakers
    else:
        if args.min_speakers is not None:
            diarization_kwargs["min_speakers"] = args.min_speakers
        if args.max_speakers is not None:
            diarization_kwargs["max_speakers"] = args.max_speakers

    normalized_audio_path = normalize_audio(audio_path)
    try:
        output = pipeline(str(normalized_audio_path), **diarization_kwargs)
    finally:
        normalized_audio_path.unlink(missing_ok=True)

    if args.regular:
        annotation = getattr(output, "speaker_diarization", output)
    else:
        annotation = getattr(output, "exclusive_speaker_diarization", output)

    turns = annotation_to_turns(annotation)

    with output_path.open("w") as f:
        json.dump(turns, f, indent=2)

    speakers = sorted({turn["speaker"] for turn in turns})
    print(f"Wrote {len(turns)} turns with {len(speakers)} speakers: {speakers}")


if __name__ == "__main__":
    main()
```

Why this script uses `exclusive_speaker_diarization` by default:

- Community-1 returns both regular and exclusive diarization.
- Exclusive diarization assigns at most one speaker at each instant.
- That is easier to reconcile with ASR timestamps.
- It should work better with Sona's existing speaker assignment logic.

Why this script normalizes audio before diarization:

- MP3 and other compressed files can decode to slightly unexpected sample counts.
- pyannote validates chunk sizes strictly.
- A decoded MP3 chunk can fail with an error like `resulted in 478895 samples instead of the expected 480000 samples`.
- Converting to temporary `16 kHz`, `mono`, `PCM WAV` before diarization avoids that mismatch.
- The temporary WAV keeps the same timeline, so the output timestamps still line up with the original audio.

## 6. Test the Tool Directly

From the repo root:

```bash
conda run -n sona-diarization python tools/diarization/diarize_community.py \
  data/raw/audio/audio_indo.mp3 \
  outputs/diarization/community_audio_indo.json \
  --min-speakers 2 \
  --max-speakers 5 \
  --device cpu
```

Inspect the JSON:

```bash
python -m json.tool outputs/diarization/community_audio_indo.json | head -80
```

Expected output shape:

```json
[
  {
    "start": 0.4,
    "end": 3.9,
    "speaker": "SPEAKER_00"
  }
]
```

If the script prints multiple speakers and writes JSON, Community-1 is working independently of the main backend.

## 6.1. Troubleshooting MP3 Sample Count Errors

If you see an error like this:

```text
ValueError: requested chunk [ 00:00:00.000 -->  00:00:10.000] from audio_indo file resulted in 478895 samples instead of the expected 480000 samples
```

the model is not failing because of speaker count. pyannote is failing while cropping decoded audio chunks.

This commonly happens with compressed audio such as MP3. The fix is to normalize audio to WAV before sending it to pyannote:

```bash
ffmpeg -y -i data/raw/audio/audio_indo.mp3 -ac 1 -ar 16000 -c:a pcm_s16le data/raw/audio/audio_indo_16k.wav
```

Then run:

```bash
conda run -n sona-diarization python tools/diarization/diarize_community.py \
  data/raw/audio/audio_indo_16k.wav \
  outputs/diarization/community_audio_indo.json \
  --min-speakers 2 \
  --max-speakers 5 \
  --device cpu
```

The recommended script above does this automatically by creating a temporary normalized WAV with `ffmpeg`.

## 7. Add a Backend External Diarizer Wrapper

Once the standalone tool works, add a backend wrapper that conforms to the existing diarizer contract.

The current contract is:

```text
backend/src/sona_ai/diarization/base.py
```

It expects:

```python
load_models()
diarize(audio_path, min_speakers=None, max_speakers=None) -> DiarizationResult
cleanup_models()
```

Create:

```text
backend/src/sona_ai/diarization/external_community_diarizer.py
```

Suggested implementation:

```python
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from sona_ai.core import PROJECT_ROOT, setup_logging
from sona_ai.diarization.schemas import DiarizationResult, SpeakerTurn


logger = setup_logging()


class ExternalCommunityDiarizer:
    def __init__(self, config: dict):
        self.config = config
        diarization_config = config.get("diarization", {})
        self.conda_env = diarization_config.get("conda_env", "sona-diarization")
        self.tool_path = PROJECT_ROOT / diarization_config.get(
            "tool_path",
            "tools/diarization/diarize_community.py",
        )
        self.device = config.get("model", {}).get("device", "cpu")
        self.cache_dir = PROJECT_ROOT / config.get("cp_dir", {}).get(
            "hf_cache",
            "cp/hf_cache",
        ) / "pyannote-community"

    def load_models(self) -> None:
        if not self.tool_path.is_file():
            raise FileNotFoundError(f"Community diarization tool not found: {self.tool_path}")

    def diarize(
        self,
        audio_path: str,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> DiarizationResult:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as output_file:
            output_path = Path(output_file.name)

        cmd = [
            "conda",
            "run",
            "-n",
            self.conda_env,
            "python",
            str(self.tool_path),
            audio_path,
            str(output_path),
            "--device",
            self.device,
            "--cache-dir",
            str(self.cache_dir),
        ]

        if min_speakers is not None:
            cmd.extend(["--min-speakers", str(min_speakers)])
        if max_speakers is not None:
            cmd.extend(["--max-speakers", str(max_speakers)])

        logger.info("Running external Community-1 diarizer...")
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

        with output_path.open("r") as f:
            rows = json.load(f)

        turns = [
            SpeakerTurn(
                start=float(row["start"]),
                end=float(row["end"]),
                speaker=str(row["speaker"]),
            )
            for row in rows
        ]

        speakers = sorted({turn.speaker for turn in turns})
        logger.info(
            "Community-1 diarization detected %d speakers across %d turns: %s",
            len(speakers),
            len(turns),
            speakers,
        )

        return DiarizationResult(turns=turns, raw=rows)

    def cleanup_models(self) -> None:
        return None
```

This wrapper does not import `pyannote.audio` in the main backend process. That is the whole point. The pyannote 4 dependency stays inside `sona-diarization`.

## 8. Export the New Diarizer

Open:

```text
backend/src/sona_ai/diarization/__init__.py
```

Add lazy import support for `ExternalCommunityDiarizer`.

Example shape:

```python
def __getattr__(name):
    if name == "PyannoteDiarizer":
        from .pyannote_diarizer import PyannoteDiarizer

        return PyannoteDiarizer

    if name == "ExternalCommunityDiarizer":
        from .external_community_diarizer import ExternalCommunityDiarizer

        return ExternalCommunityDiarizer

    raise AttributeError(name)
```

Keep the existing exports that are already in that file.

## 9. Add Config for the External Diarizer

Option A: add the config to `configs/whisperx.yaml`.

```yaml
diarization:
    engine: "community_external"
    conda_env: "sona-diarization"
    tool_path: "tools/diarization/diarize_community.py"
```

Option B: create a dedicated config file:

```text
configs/diarization-community.yaml
```

Example:

```yaml
model:
    device: "cpu"

input:
    min_speakers: 1
    max_speakers: 5

diarization:
    engine: "community_external"
    conda_env: "sona-diarization"
    tool_path: "tools/diarization/diarize_community.py"

cp_dir:
    hf_cache: "cp/hf_cache"
```

If you use a dedicated config file, update `configs/speech.yaml`:

```yaml
diarization:
    enabled: true
    config: "diarization-community"
```

This is cleaner because diarization config is no longer hidden inside `whisperx.yaml`.

## 10. Update the Pipeline Factory

Open:

```text
backend/src/sona_ai/pipelines/factory.py
```

Find:

```python
from sona_ai.diarization import PyannoteDiarizer

diarization_engine_config = deepcopy(load_config(config_name))
device = diarization_config.get("device")
if device is not None:
    diarization_engine_config.setdefault("model", {})["device"] = device
return PyannoteDiarizer(diarization_engine_config)
```

Change the factory so it can choose between the internal pyannote 3 diarizer and the external Community-1 diarizer:

```python
diarization_engine_config = deepcopy(load_config(config_name))
device = diarization_config.get("device")
if device is not None:
    diarization_engine_config.setdefault("model", {})["device"] = device

engine = (
    diarization_config.get("engine")
    or diarization_engine_config.get("diarization", {}).get("engine")
    or "pyannote"
)

if engine == "community_external":
    from sona_ai.diarization import ExternalCommunityDiarizer

    return ExternalCommunityDiarizer(diarization_engine_config)

if engine == "pyannote":
    from sona_ai.diarization import PyannoteDiarizer

    return PyannoteDiarizer(diarization_engine_config)

raise ValueError(f"Unsupported diarization engine: {engine}")
```

Then `configs/speech.yaml` can select the external engine:

```yaml
diarization:
    enabled: true
    config: "diarization-community"
    engine: "community_external"
```

## 11. Test the Full Pipeline

First compile the backend:

```bash
conda run -n sona-ai python -m compileall backend/src
```

Then run your transcript test:

```bash
PYTHONPATH=backend/src python tests/transcript_test.py
```

If you want to be explicit about the backend environment:

```bash
conda run -n sona-ai env PYTHONPATH=backend/src python tests/transcript_test.py
```

Expected logs:

```text
Running external Community-1 diarizer...
Community-1 diarization detected 2 speakers across ...
Assigning speakers by word/diarization overlap...
Final transcript has 2 speakers across ...
```

If Community-1 detects multiple speakers but the final transcript only has one, the diarizer is working and the issue is in:

```text
backend/src/sona_ai/pipelines/speaker_assignment.py
```

## 12. Backend Runtime Notes

When running the backend normally:

```bash
conda activate sona-ai
PYTHONPATH=backend/src uvicorn sona_ai.api.main:app --reload --host 0.0.0.0 --port 8000
```

The backend will remain in `sona-ai`, but each Community-1 diarization call will spawn:

```bash
conda run -n sona-diarization python tools/diarization/diarize_community.py ...
```

That means:

- first run will be slower because Community-1 downloads models
- model cache should go under `cp/hf_cache/pyannote-community`
- dependency conflicts stay isolated
- later you can replace subprocess execution with a small local HTTP service if startup overhead becomes annoying

## 13. Recommended Implementation Order

Use this order to avoid debugging too many things at once:

1. Repair `sona-ai` if pyannote 4 polluted it.
2. Create `sona-diarization`.
3. Accept Community-1 model access on Hugging Face.
4. Create `backend/requirements-diarization-community.txt`.
5. Create `tools/diarization/diarize_community.py`.
6. Test the standalone script on `data/raw/audio/audio_indo.mp3`.
7. Add `ExternalCommunityDiarizer`.
8. Export it from `sona_ai.diarization`.
9. Add `configs/diarization-community.yaml`.
10. Update `_build_diarizer()` in `factory.py`.
11. Run `tests/transcript_test.py`.
12. Restart the backend and test re-transcribe from the UI.

## 14. Why This Is Better Than Forcing Community-1 Into WhisperX

WhisperX originally gave Sona AI a convenient bundle:

```text
ASR -> alignment -> diarization -> speaker assignment
```

That was useful early on, but now Sona AI needs model flexibility:

```text
ASR:
  - Parakeet
  - WhisperX
  - future model

Alignment:
  - WhisperX wav2vec alignment
  - optional no-op aligner
  - future forced aligner

Diarization:
  - pyannote 3.1
  - pyannote Community-1
  - Sortformer or other future diarizer

Speaker assignment:
  - Sona-owned overlap/midpoint assignment logic
```

Owning the tool boundary lets Sona change ASR, alignment, and diarization independently.

The key rule:

```text
Do not build the ML models yourself.
Build stable wrappers around proven models.
```
