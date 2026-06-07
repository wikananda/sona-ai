import httpx
from typing import Optional
from sona_ai.core import PROVIDER_BASE_URLS
from typing import List

class OpenAICompatibleChat:
    def answer(
        self,
        transcript: str,
        question: str,
        history:list[dict],
        api_key: str,
        model: str,
        provider: str,
        base_url: str | None = None,
    ) -> str:
        resolved_base_url = self._resolve_base_url(provider, base_url)
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You answer questions about one audio recording transcript. "
                    "Use only the transcript as evidence. "
                    "If the answer is not in the transcript, say you cannot find it "
                    "in the recording. Be concise and factual. "
                    "Format answers in clean Markdown. Use short paragraphs by default. "
                    "Use headings when the user asks for sections or structure. "
                    "Use bullet lists only when the user asks for a list, steps, "
                    "key points, comparisons, or when bullets clearly improve scanning. "
                    "Avoid generic preambles like \"Here's a summary\" unless the user "
                    "explicitly asks for an introduction."
                ),
            },
            {
                "role": "user",
                "content": f"Transcript:\n{transcript}",
            },
        ]

        messages.extend(history[-10:])
        messages.append({"role": "user", "content": question})

        response = httpx.post(
            f"{resolved_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.2,
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


def _chat_context_from_segments(segments: list) -> str:
    lines = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text") or "").strip()
        if not text:
            continue

        speaker = str(segment.get("speaker") or "Speaker").strip()
        start = segment.get("start")
        end = segment.get("end")
        
        if start is not None and end is not None:
            lines.append(f"[{float(start):.2f}-{float(end):.2f}] {speaker}: {text}")
        else:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)
