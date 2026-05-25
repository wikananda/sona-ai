from typing import Literal, Optional
from pydantic import BaseModel

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatBYOKSettings(BaseModel):
    provider: Literal["openai", "groq", "openrouter", "custom"]
    api_key: str
    model: str
    base_url: Optional[str] = None

class RecordingChatRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []
    byok_settings: ChatBYOKSettings