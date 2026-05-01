from sona_ai.core import load_config
from sona_ai.pipelines import SpeechPipeline, WhisperXSpeakerAssigner
from sona_ai.transcription import WhisperXTranscriber
from sona_ai.diarization import PyannoteDiarizer

audio_path = "data/raw/audio/audio_indo.mp3"

config = load_config("whisperx")
SpeechPipeline.setup_environment(config)

pipeline = SpeechPipeline(
    transcriber=WhisperXTranscriber(config),
    diarizer=PyannoteDiarizer(config),
    speaker_assigner=WhisperXSpeakerAssigner(),
)

pipeline.load_models()

try:
    result = pipeline.transcribe(
        audio_path,
        language="id",
        min_speakers=2,
        max_speakers=2,
    )

    print("\nTranscript preview:\n")
    for segment in result["transcript"][:20]:
        print(
            f"[{segment['start']:.2f} - {segment['end']:.2f}] "
            f"{segment['speaker']}: {segment['text']}"
        )

    print("\nSaved outputs:")
    print("outputs/transcription/conversations.json")
    print("outputs/transcription/result_raw.json")

finally:
    print("\nCleaning up...")
    pipeline.cleanup_models()