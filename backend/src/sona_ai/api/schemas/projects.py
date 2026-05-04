from typing import Optional

from pydantic import BaseModel


class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None


class TranscriptSpeakerRename(BaseModel):
    speakers: dict[str, str]
