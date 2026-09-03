# Configs

This directory intentionally stays flat because the backend loads configs by
name, for example `load_config("speech")` resolves to `configs/speech.yaml`.

## Pipeline

- `speech.yaml`: top-level speech pipeline selection.
- `speech.hf-full.yaml`: Hugging Face Docker Space profile with CPU ASR, alignment, and diarization enabled.
- `parakeet.yaml`: NVIDIA Parakeet ASR.
- `faster-whisper-large-v3.yaml`: Faster-Whisper Large-v3 ASR.
- `faster-whisper-turbo.yaml`: Faster-Whisper Turbo ASR.
- `whisper-mps-large-v3.yaml`: PyTorch Whisper Large-v3 on Apple MPS.
- `whisper-mps-turbo.yaml`: PyTorch Whisper Turbo on Apple MPS.
- `wav2vec2.yaml`: external WhisperX-backed Wav2Vec2 alignment.
- `diarization-community.yaml`: external pyannote Community-1 diarization.

Speech runtime device defaults live in `speech.yaml`. Component configs keep
model names, model-specific options, and external tool paths. Model downloads
share the global cache root configured by `SONA_HF_CACHE`, defaulting to
`.models/`.

## Summarization

- `qwen.yaml`: GGUF Qwen summarizer.
- `llama.yaml`: Hugging Face Llama summarizer.
- `gemma.yaml`: Hugging Face Gemma summarizer.

Each LLM config owns its own `limits.max_input_length` and
`limits.max_output_tokens` defaults.
