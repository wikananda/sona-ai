from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .gemma4_summarizer import Gemma4Summarizer
    from .gguf_summarizer import GGUFLLMSummarizer
    from .hf_summarizer import LocalLLMSummarizer
    from .openai_compatible_summarizer import OpenAICompatibleSummarizer

__all__ = [
    "Gemma4Summarizer",
    "GGUFLLMSummarizer",
    "LocalLLMSummarizer",
    "OpenAICompatibleSummarizer",
]


def __getattr__(name: str):
    if name == "Gemma4Summarizer":
        from .gemma4_summarizer import Gemma4Summarizer

        return Gemma4Summarizer
    if name == "GGUFLLMSummarizer":
        from .gguf_summarizer import GGUFLLMSummarizer

        return GGUFLLMSummarizer
    if name == "LocalLLMSummarizer":
        from .hf_summarizer import LocalLLMSummarizer

        return LocalLLMSummarizer
    if name == "OpenAICompatibleSummarizer":
        from .openai_compatible_summarizer import OpenAICompatibleSummarizer

        return OpenAICompatibleSummarizer
    raise AttributeError(name)
