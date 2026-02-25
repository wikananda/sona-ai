from pydantic import BaseModel
from typing import Optional

class SummarizeRequest(BaseModel):
    text: str
    prompt: Optional[str] = None
    max_length: Optional[int] = 2048