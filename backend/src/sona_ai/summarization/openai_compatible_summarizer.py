from typing import Optional
import httpx
from sona_ai.summarization.prompts import build_prompt
from sona_ai.core import PROVIDER_BASE_URLS

class OpenAICompatibleSummarizer:
    def generate(
        self,
        text: str,
        api_key: str,
        model: str,
        provider: str,
        base_url: Optional[str] = None,
        prompt: Optional[str] = None,
        max_length: int = 2048,
        max_tokens: int = 256,
    ) -> str:
        resolved_base_url = self._resolve_base_url(provider, base_url)
        formatted_prompt = build_prompt(text, prompt)

        response = httpx.post(
            f"{resolved_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You summarize transcripts accurately and concisely."
                        ),
                    },
                    {
                        "role": "user",
                        "content": formatted_prompt[:max_length * 4],
                    },
                ],
                "temperature": 0.2,
                "max_tokens": max_tokens,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"].strip()

    def _resolve_base_url(self, provider: str, base_url: Optional[str]) -> str:
        if provider == "custom":
            if not base_url:
                raise ValueError("Custom BYOK provider requires base_url")
            return base_url.rstrip("/")

        if provider not in PROVIDER_BASE_URLS:
            raise ValueError(f"Unsupported BYOK provider: {provider}")
        return PROVIDER_BASE_URLS[provider]