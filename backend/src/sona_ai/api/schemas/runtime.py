from typing import Optional

from pydantic import BaseModel


class TranscriptionPreflightRequest(BaseModel):
    language: Optional[str] = None
    model: str
    device: str = "auto"
    extract_speakers: bool = True
    alignment_enabled: Optional[bool] = None
