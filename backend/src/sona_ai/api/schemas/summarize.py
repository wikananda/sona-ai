from pydantic import BaseModel
from typing import Literal, Optional

from sona_ai.api.schemas.byok import BYOKSettings

class SummarizeRequest(BaseModel):
    text: str
    prompt: Optional[str] = None
    max_length: Optional[int] = None
    model: Optional[str] = "qwen"
    device: Optional[str] = "auto"
    mode: Literal["local", "byok"] = "local"
    byok: Optional[BYOKSettings] = None


class RecordingSummaryRequest(BaseModel):
    prompt: Optional[str] = None
    max_length: Optional[int] = None
    model: Optional[str] = "qwen"
    device: Optional[str] = "auto"
    mode: Literal["local", "byok"] = "local"
    byok: Optional[BYOKSettings] = None
