import os
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(os.getenv("SONA_AI_ROOT", BACKEND_ROOT.parent)).resolve()

