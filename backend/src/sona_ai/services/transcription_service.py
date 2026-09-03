import threading
from collections.abc import Callable
from typing import Optional

from sona_ai.core import (
    load_config,
    resolve_device,
    setup_logging,
    validate_device_available,
)
from sona_ai.pipelines import SpeechPipeline, build_speech_pipeline
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
        self._transcribers = {
            self._transcriber_key(
                default_profile,
                actual_device=getattr(pipeline.transcriber, "device", None),
            ): pipeline.transcriber,
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
            return pipeline.transcriber.transcribe(audio_path, language=language)

    def prepare_live_transcription(
        self,
        *,
        model: str,
        device: str,
    ) -> None:
        self._get_live_transcriber(model=model, device=device)

    def transcribe_live_samples(
        self,
        samples,
        *,
        language: Optional[str],
        model: str,
        device: str,
    ) -> TranscriptionResult:
        transcriber = self._get_live_transcriber(model=model, device=device)
        transcribe_samples = getattr(transcriber, "transcribe_samples", None)
        if transcribe_samples is None:
            raise ValueError(f"{model} does not support direct live PCM transcription.")

        logger.info("Waiting for transcription lock for live PCM...")
        with self._transcription_lock:
            logger.info("Transcription lock acquired for live PCM.")
            return transcribe_samples(samples, language=language)

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
        # Loading happens under the pipeline lock and inference happens under
        # the transcription lock. Taking both in that order makes shutdown wait
        # for either operation before models are cleared.
        with self._pipeline_lock:
            with self._transcription_lock:
                pipelines = list(self._pipelines.values())
                self._pipelines.clear()
                self._transcribers.clear()
                for pipeline in pipelines:
                    pipeline.cleanup_models()

    def _get_live_transcriber(self, *, model: str, device: str):
        profile = self.resolve_profile(
            model,
            device,
            alignment_enabled=False,
            extract_speakers=False,
        )
        key = self._transcriber_key(profile)
        transcriber = self._transcribers.get(key)
        if transcriber is not None:
            return transcriber

        pipeline = self._get_pipeline(
            model,
            device,
            alignment_enabled=False,
            extract_speakers=False,
        )
        with self._pipeline_lock:
            return self._transcribers.setdefault(key, pipeline.transcriber)

    @staticmethod
    def _transcriber_key(
        profile: PipelineProfile,
        *,
        actual_device: Optional[str] = None,
    ) -> tuple[str, str, str]:
        return (
            profile.transcription_engine,
            profile.transcription_config,
            resolve_device(actual_device or profile.device),
        )

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
            self._transcribers.setdefault(
                self._transcriber_key(
                    profile,
                    actual_device=getattr(pipeline.transcriber, "device", None),
                ),
                pipeline.transcriber,
            )
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
