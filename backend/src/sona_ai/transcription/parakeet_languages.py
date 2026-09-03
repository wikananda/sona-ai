from typing import Optional


# Languages published for nvidia/parakeet-tdt-0.6b-v3. The checkpoint performs
# automatic language detection, so ``None`` represents the auto-detect choice.
PARAKEET_LANGUAGES = frozenset({
    "bg",
    "hr",
    "cs",
    "da",
    "nl",
    "en",
    "et",
    "fi",
    "fr",
    "de",
    "el",
    "hu",
    "it",
    "lv",
    "lt",
    "mt",
    "pl",
    "pt",
    "ro",
    "sk",
    "sl",
    "es",
    "sv",
    "ru",
    "uk",
})


def resolve_parakeet_language(language: Optional[str]) -> Optional[str]:
    """Return a supported base language code, or ``None`` for auto detection."""

    value = str(language or "").strip().replace("_", "-").casefold()
    if not value or value in {"auto", "none"}:
        return None

    normalized = value.split("-", maxsplit=1)[0]
    if normalized in PARAKEET_LANGUAGES:
        return normalized

    raise ValueError(
        f"Parakeet does not support language '{language}'. "
        "Choose Auto or one of its 25 supported European languages."
    )
