from sona_ai.core import load_config
from sona_ai.diarization import PyannoteDiarizer
from sona_ai.pipelines import SpeechPipeline, WhisperXSpeakerAssigner
from sona_ai.transcription import ParakeetTranscriber


audio_path = "data/raw/audio/audio.mp3"

parakeet_config = load_config("parakeet")
diarization_config = load_config("whisperx")
SpeechPipeline.setup_environment(parakeet_config)

pipeline = SpeechPipeline(
    transcriber=ParakeetTranscriber(parakeet_config),
    aligner=None,
    diarizer=PyannoteDiarizer(diarization_config),
    speaker_assigner=WhisperXSpeakerAssigner(),
)

pipeline.load_models()

try:
    result = pipeline.transcribe(
        audio_path,
        language="en",
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
