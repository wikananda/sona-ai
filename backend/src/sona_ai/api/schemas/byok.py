from pydantic import BaseModel
from typing import Literal, Optional

class BYOKSettings(BaseModel):
    provider: Literal["openai", "groq", "openrouter", "custom"]
    api_key: str
    model: str
    base_url: Optional[str] = None
