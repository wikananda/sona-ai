from typing import Literal
from pydantic import BaseModel

from sona_ai.api.schemas.byok import BYOKSettings

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class RecordingChatRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []
    byok_settings: BYOKSettings
