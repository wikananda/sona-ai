from typing import Optional

from pydantic import BaseModel


class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None


class RecordingRetranscribe(BaseModel):
    language: Optional[str] = None
    model: Optional[str] = None
    device: Optional[str] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None


class TranscriptSpeakerRename(BaseModel):
    speakers: dict[str, str]
