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
        cache_root = PROJECT_ROOT / config.get("cp_dir", {}).get("hf_cache", "cp/hf_cache")
        self.cache_dir = cache_root / "pyannote-community"

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
            len(speakers), len(turns), speakers,
        )

        return DiarizationResult(turns=turns, raw=rows)

    def cleanup_models(self) -> None:
        return None
