from typing import Optional


# Locales supported out of the box by nvidia/nemotron-3.5-asr-streaming-0.6b.
# The model card also lists adaptation-only locales; those are intentionally not
# accepted here because the published checkpoint cannot transcribe them reliably.
NEMOTRON_LOCALES = frozenset({
    "ar-AR",
    "bg-BG",
    "cs-CZ",
    "da-DK",
    "de-DE",
    "en-GB",
    "en-US",
    "es-ES",
    "es-US",
    "et-EE",
    "fi-FI",
    "fr-CA",
    "fr-FR",
    "hi-IN",
    "hr-HR",
    "hu-HU",
    "it-IT",
    "ja-JP",
    "ko-KR",
    "nb-NO",
    "nl-NL",
    "pl-PL",
    "pt-BR",
    "pt-PT",
    "ro-RO",
    "ru-RU",
    "sk-SK",
    "sv-SE",
    "tr-TR",
    "uk-UA",
    "vi-VN",
    "zh-CN",
})

_DEFAULT_LOCALES = {
    "ar": "ar-AR",
    "bg": "bg-BG",
    "cs": "cs-CZ",
    "da": "da-DK",
    "de": "de-DE",
    "en": "en-US",
    "es": "es-ES",
    "et": "et-EE",
    "fi": "fi-FI",
    "fr": "fr-FR",
    "hi": "hi-IN",
    "hr": "hr-HR",
    "hu": "hu-HU",
    "it": "it-IT",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "nb": "nb-NO",
    "nl": "nl-NL",
    "no": "nb-NO",
    "pl": "pl-PL",
    "pt": "pt-BR",
    "ro": "ro-RO",
    "ru": "ru-RU",
    "sk": "sk-SK",
    "sv": "sv-SE",
    "tr": "tr-TR",
    "uk": "uk-UA",
    "vi": "vi-VN",
    "zh": "zh-CN",
}
_CANONICAL_LOCALES = {locale.casefold(): locale for locale in NEMOTRON_LOCALES}


def resolve_nemotron_language(language: Optional[str]) -> str:
    """Return the locale expected by Nemotron, or ``auto`` for detection."""

    value = str(language or "").strip().replace("_", "-")
    if not value or value.casefold() in {"auto", "none"}:
        return "auto"

    normalized = value.casefold()
    locale = _CANONICAL_LOCALES.get(normalized) or _DEFAULT_LOCALES.get(normalized)
    if locale is not None:
        return locale

    raise ValueError(
        f"Nemotron 3.5 does not support language '{value}'. "
        "Choose Auto or one of its supported non-Indonesian languages."
    )
