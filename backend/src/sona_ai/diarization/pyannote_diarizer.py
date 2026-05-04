import gc
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from dotenv import load_dotenv

from sona_ai.core import PROJECT_ROOT, Timer, setup_logging
from sona_ai.diarization.schemas import DiarizationResult, SpeakerTurn


load_dotenv(PROJECT_ROOT / ".env")
logger = setup_logging()


class PyannoteDiarizer:
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.model_name = (
            self.config.get("diarization", {}).get("model_name")
            or "pyannote/speaker-diarization-3.1"
        )
        self.cache_dir = self._cache_dir()

    def load_models(self):
        logger.info("Loading pyannote diarization model...")
        if os.getenv("HF_TOKEN") is None:
            raise EnvironmentError("Hugging Face token is not set")

        from pyannote.audio import Pipeline

        self._patch_huggingface_hub_auth_kwarg()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        pipeline = Pipeline.from_pretrained(
            self.model_name,
            use_auth_token=os.getenv("HF_TOKEN"),
            cache_dir=self.cache_dir,
        )
        self.model = pipeline.to(torch.device(self.config["model"]["device"]))

    def diarize(
        self,
        audio_path: str,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> DiarizationResult:
        if self.model is None:
            raise ReferenceError("Diarization model is not initialized.")

        min_s, max_s = self._resolve_speaker_bounds(min_speakers, max_speakers)
        import whisperx

        audio = whisperx.load_audio(audio_path)
        audio_data = {
            "waveform": torch.from_numpy(audio[None, :]),
            "sample_rate": 16000,
        }

        logger.info("Running diarization...")
        with Timer("Diarization"):
            diarization = self.model(
                audio_data,
                min_speakers=min_s,
                max_speakers=max_s,
            )
        diarization_df = self._to_dataframe(diarization)
        turns = self._build_turns(diarization_df)
        speakers = sorted({turn.speaker for turn in turns})
        logger.info(
            "Diarization detected %d speakers across %d turns: %s",
            len(speakers),
            len(turns),
            speakers,
        )

        return DiarizationResult(
            turns=turns,
            raw=diarization_df,
        )

    def _cache_dir(self) -> Path:
        cache_dir = self.config.get("cp_dir", {}).get("hf_cache", "cp/hf_cache")
        return PROJECT_ROOT / cache_dir / "pyannote"

    def _patch_huggingface_hub_auth_kwarg(self) -> None:
        from huggingface_hub import hf_hub_download

        if getattr(hf_hub_download, "_sona_auth_compat", False):
            compat_download = hf_hub_download
        else:
            def compat_download(*args, **kwargs):
                if "use_auth_token" in kwargs and "token" not in kwargs:
                    kwargs["token"] = kwargs.pop("use_auth_token")
                else:
                    kwargs.pop("use_auth_token", None)
                return hf_hub_download(*args, **kwargs)

            compat_download._sona_auth_compat = True

            import huggingface_hub

            huggingface_hub.hf_hub_download = compat_download

        for module in list(sys.modules.values()):
            if module is None:
                continue
            module_name = getattr(module, "__name__", "")
            if not module_name.startswith("pyannote.audio"):
                continue
            if hasattr(module, "hf_hub_download"):
                setattr(module, "hf_hub_download", compat_download)

    def _resolve_speaker_bounds(
        self,
        min_speakers: Optional[int],
        max_speakers: Optional[int],
    ) -> tuple[int, int]:
        min_s = min_speakers if min_speakers is not None else self.config["input"]["min_speakers"]
        max_s = max_speakers if max_speakers is not None else self.config["input"]["max_speakers"]

        if min_s > max_s:
            raise ValueError("min_speakers must be less than or equal to max_speakers")
        if min_s < 1:
            raise ValueError("min_speakers must be greater than or equal to 1")
        if max_s < 1:
            raise ValueError("max_speakers must be greater than or equal to 1")

        return min_s, max_s

    def _to_dataframe(self, diarization) -> pd.DataFrame:
        if isinstance(diarization, pd.DataFrame):
            return diarization

        rows = []
        for segment, label, speaker in diarization.itertracks(yield_label=True):
            rows.append({
                "segment": segment,
                "label": label,
                "speaker": speaker,
                "start": segment.start,
                "end": segment.end,
            })

        return pd.DataFrame(rows)

    def _build_turns(self, diarization) -> list[SpeakerTurn]:
        turns = []
        if hasattr(diarization, "itertracks"):
            diarization = self._to_dataframe(diarization)

        for _, row in diarization.iterrows():
            segment = row.get("segment")
            start = row.get("start", getattr(segment, "start", None))
            end = row.get("end", getattr(segment, "end", None))
            speaker = row.get("speaker", row.get("label", "Unknown"))

            if start is None or end is None:
                continue

            turns.append(
                SpeakerTurn(
                    start=float(start),
                    end=float(end),
                    speaker=str(speaker),
                )
            )
        return turns

    def cleanup_models(self):
        if self.model is not None:
            del self.model
            self.model = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
