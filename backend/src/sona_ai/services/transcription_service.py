import threading
from collections.abc import Callable
from typing import Optional

from sona_ai.core import load_config, validate_device_available, setup_logging
from sona_ai.pipelines import SpeechPipeline, build_speech_pipeline
from sona_ai.transcription.live_timestamps import segment_live_timestamps
from sona_ai.transcription.schemas import TranscriptionResult
from sona_ai.services.pipeline_profile import (
    PipelineProfile,
    TRANSCRIPTION_MODEL_PROFILES,
    resolve_pipeline_profile,
    speech_config_for_profile,
)
from sona_ai.services.model_download_service import model_download_service


SUPPORTED_TRANSCRIPTION_MODELS = set(TRANSCRIPTION_MODEL_PROFILES)

logger = setup_logging()
ProgressCallback = Callable[[str, bool], None]


class TranscriptionService:
    def __init__(
        self,
        pipeline: SpeechPipeline,
        speech_config: Optional[dict] = None,
        default_model: str = "parakeet",
        default_device: str = "cpu",
    ):
        self.default_model = self._normalize_model(default_model)
        self.default_device = validate_device_available(default_device)
        self.speech_config = speech_config or load_config("speech")

        default_profile = resolve_pipeline_profile(
            self.speech_config,
            model=self.default_model,
            device=self.default_device,
        )
        self._pipelines = {
            default_profile.cache_key(): pipeline
        }
        model_download_service.mark_profile_installed(default_profile)

        self._pipeline_lock = threading.Lock()
        self._transcription_lock = threading.Lock()

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        model: Optional[str] = None,
        device: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        extract_speakers: bool = True,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        pipeline = self._get_pipeline(model, device, extract_speakers=extract_speakers)
        logger.info("Waiting for transcription lock...")
        with self._transcription_lock:
            logger.info("Transcription lock acquired.")
            return pipeline.transcribe(
                audio_path,
                language=language,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                progress_callback=progress_callback,
            )

    def extract_speakers(
        self,
        audio_path: str,
        transcription: TranscriptionResult,
        model: Optional[str] = None,
        device: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        pipeline = self._get_pipeline(model, device, extract_speakers=True)
        logger.info("Waiting for transcription lock...")
        with self._transcription_lock:
            logger.info("Transcription lock acquired for speaker extraction.")
            return pipeline.extract_speakers(
                audio_path,
                transcription,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                progress_callback=progress_callback,
            )

    def transcribe_live_chunk(
        self,
        audio_path: str,
        language: Optional[str] = None,
        model: Optional[str] = None,
        device: Optional[str] = None,
    ) -> TranscriptionResult:
        pipeline = self._get_pipeline(
            model,
            device,
            alignment_enabled=False,
            extract_speakers=False,
        )
        logger.info("Waiting for transcription lock...")
        with self._transcription_lock:
            logger.info("Transcription lock acquired for live chunk.")
            transcription = pipeline.transcriber.transcribe(audio_path, language=language)
        return segment_live_timestamps(transcription, audio_path)

    def align_live_segments(
        self,
        audio_path: str,
        language: Optional[str],
        segments: list[dict],
        model: Optional[str] = None,
        device: Optional[str] = None,
    ) -> tuple[list[dict], bool]:
        """Align rough live-transcription segments against the saved audio.

        Returns the aligned segment dicts and whether alignment produced a new
        transcript (as opposed to passing the rough transcript through).
        """
        rough_transcription = TranscriptionResult.from_aligned_result(
            {
                "language": language,
                "segments": segments,
            }
        )
        aligned_transcription = self.align_transcript(
            audio_path,
            rough_transcription,
            model=model,
            device=device,
        )
        aligned_transcription = segment_live_timestamps(aligned_transcription, audio_path)
        return (
            aligned_transcription.to_segment_dicts(),
            aligned_transcription is not rough_transcription,
        )

    def align_transcript(
        self,
        audio_path: str,
        transcription: TranscriptionResult,
        model: Optional[str] = None,
        device: Optional[str] = None,
    ) -> TranscriptionResult:
        pipeline = self._get_pipeline(
            model,
            device,
            alignment_enabled=True,
            extract_speakers=False,
        )
        if pipeline.aligner is None:
            return transcription

        logger.info("Waiting for transcription lock...")
        with self._transcription_lock:
            logger.info("Transcription lock acquired for transcript alignment.")
            return pipeline.aligner.align(transcription, audio_path)

    def close(self):
        for pipeline in self._pipelines.values():
            pipeline.cleanup_models()

    def _get_pipeline(
        self,
        model: Optional[str],
        device: Optional[str],
        alignment_enabled: Optional[bool] = None,
        extract_speakers: bool = True,
    ) -> SpeechPipeline:
        profile = self.resolve_profile(
            model,
            device,
            alignment_enabled=alignment_enabled,
            extract_speakers=extract_speakers,
        )
        key = profile.cache_key()
        logger.info(
            "Using speech pipeline: transcription=%s/%s alignment=%s/%s/%s "
            "diarization=%s/%s/%s device=%s",
            profile.transcription_engine,
            profile.transcription_config,
            profile.alignment_enabled,
            profile.alignment_engine,
            profile.alignment_config,
            profile.diarization_enabled,
            profile.diarization_engine,
            profile.diarization_config,
            profile.device,
        )
        if key in self._pipelines:
            model_download_service.mark_profile_installed(profile)
            return self._pipelines[key]

        with self._pipeline_lock:
            if key in self._pipelines:
                model_download_service.mark_profile_installed(profile)
                return self._pipelines[key]

            profile_config = speech_config_for_profile(self.speech_config, profile)
            pipeline = build_speech_pipeline(
                profile_config,
                device=profile.device,
            )
            pipeline.load_models()
            model_download_service.mark_profile_installed(profile)
            self._pipelines[key] = pipeline
            return pipeline

    def _normalize_model(self, model: str) -> str:
        model_name = model.lower().strip()
        if model_name not in SUPPORTED_TRANSCRIPTION_MODELS:
            allowed = ", ".join(sorted(SUPPORTED_TRANSCRIPTION_MODELS))
            raise ValueError(f"Unsupported transcription model: {model}. Use one of: {allowed}")
        return model_name

    def resolve_profile(
        self,
        model: Optional[str] = None,
        device: Optional[str] = None,
        alignment_enabled: Optional[bool] = None,
        extract_speakers: bool = True,
    ) -> PipelineProfile:
        model_name = self._normalize_model(model or self.default_model)
        device_name = validate_device_available(device or self.default_device)
        return resolve_pipeline_profile(
            self.speech_config,
            model=model_name,
            device=device_name,
            alignment_enabled=alignment_enabled,
            diarization_enabled=extract_speakers,
        )
