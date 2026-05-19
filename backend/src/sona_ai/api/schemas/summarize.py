from pydantic import BaseModel
from typing import Literal, Optional

class BYOKSummarizeSettings(BaseModel):
    provider: Literal["openai", "groq", "openrouter", "custom"]
    api_key: str
    model: str
    base_url: Optional[str] = None

class SummarizeRequest(BaseModel):
    text: str
    prompt: Optional[str] = None
    max_length: Optional[int] = None
    model: Optional[str] = "qwen"
    device: Optional[str] = "auto"
    mode: Literal["local", "byok"] = "local"
    byok: Optional[BYOKSummarizeSettings] = None


class RecordingSummaryRequest(BaseModel):
    prompt: Optional[str] = None
    max_length: Optional[int] = None
    model: Optional[str] = "qwen"
    device: Optional[str] = "auto"
    mode: Literal["local", "byok"] = "local"
    byok: Optional[BYOKSummarizeSettings] = None
