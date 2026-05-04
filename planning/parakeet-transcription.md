# Parakeet ASR Integration Plan

## Goal

Add NVIDIA Parakeet as a swappable ASR backend alongside WhisperX, so the `Transcriber` Protocol slot in `SpeechPipeline` can be filled by either engine via config.

## Verdict

Feasible. The current `Transcriber` Protocol gives the seam we need. The main refactor is splitting transcription and alignment into independent steps so that a non-WhisperX transcriber can either skip alignment (using its own word timings) or reuse a wav2vec2 aligner.

## Design questions to settle

### 1. Alignment: native Parakeet timestamps vs. wav2vec2 forced alignment

Parakeet TDT models (e.g. `parakeet-tdt-0.6b-v2`) emit word-level timestamps natively, so an external aligner isn't strictly required. But the current `WhisperXTranscriber` bundles transcribe + align in one class, and `WhisperXSpeakerAssigner` expects a `transcription.raw` shaped like `whisperx.align`'s output (with `word_segments`).

**Cleanest path:**

- Extract a new `Aligner` Protocol, e.g.
  ```python
  class Aligner(Protocol):
      def load_models(self) -> None: ...
      def align(self, transcription: TranscriptionResult, audio_path: str) -> TranscriptionResult: ...
      def cleanup_models(self) -> None: ...
  ```
- Split `WhisperXTranscriber` into:
  - `WhisperXTranscriber` — transcribe only
  - `Wav2Vec2Aligner` — align only (wraps `whisperx.load_align_model` + `whisperx.align`)
- Make `SpeechPipeline` take an optional `aligner` and skip the step when the transcriber already provides word-level timing.
- `ParakeetTranscriber.transcribe()` returns a `TranscriptionResult` with native word timings — no aligner required.
- Optionally still wire Parakeet through `Wav2Vec2Aligner` for a fair comparison vs. WhisperX.

### 2. Dependency + scope tradeoffs

- Parakeet is distributed through NVIDIA NeMo (`nemo_toolkit[asr]`). Heavy: pulls torch + lightning + hydra. Pin carefully against existing `torch==2.8.0` / `transformers==4.48.1`.
- `parakeet-tdt-0.6b-v2` is **English-only**. Current config exposes `language` (`en` / `id` / auto). Either gate the engine on language, or pick a multilingual Parakeet variant later.
- Audio: NeMo expects 16 kHz mono, which `whisperx.load_audio` already produces — the loader can be reused.
- Adapter shape: NeMo returns `Hypothesis` objects. Add a `from_parakeet_hypotheses` (or similar) classmethod on `transcription/schemas.py:TranscriptionResult`, mirroring the existing `from_whisperx_result`.

## Suggested order of work

1. **Refactor (no behavior change):** extract `Aligner` Protocol; split `WhisperXTranscriber` into `WhisperXTranscriber` + `Wav2Vec2Aligner`; thread an optional `aligner` through `SpeechPipeline`. Existing smoke tests should still pass.
2. **New adapter:** add `configs/parakeet.yaml` and `transcription/parakeet_transcriber.py` using native timestamps. Add `from_parakeet_hypotheses` to `TranscriptionResult`.
3. **Wire it up:** select the engine in `api/main.py` startup based on a config switch (e.g. `engine: whisperx | parakeet`) so swapping is one config change.
4. **Speaker assignment:** decide whether `WhisperXSpeakerAssigner`'s WhisperX-native fast path needs a Parakeet-flavored equivalent, or whether non-WhisperX transcribers should always fall back to `assign_by_overlap`.

## Open follow-ups

- Benchmark: WER and timestamp quality of Parakeet (native) vs. WhisperX + wav2vec2 alignment on the same audio.
- Decide whether the engine switch lives in `whisperx.yaml`/`parakeet.yaml` (one config per engine) or in a top-level `app.yaml` that points at an engine config.
- If Parakeet stays English-only, decide UX: hide the language selector when Parakeet is selected, or auto-switch engines based on the requested language.
