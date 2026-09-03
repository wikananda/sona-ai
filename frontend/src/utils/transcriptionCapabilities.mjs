// Language capabilities are intentionally kept in a framework-free module so
// the same catalogue can be exercised by Node's built-in test runner.

export const WHISPER_LANGUAGE_CODES = Object.freeze([
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
    "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
    "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
    "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
    "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
    "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
    "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
    "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
    "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue",
]);

export const PARAKEET_LANGUAGE_CODES = Object.freeze([
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de",
    "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk",
    "sl", "es", "sv", "ru", "uk",
]);

// These are the 32 out-of-the-box locales collapsed to 28 language choices.
// Nemotron's eight adaptation-ready locales are not enabled because they need
// fine-tuning before they provide full transcription.
export const NEMOTRON_LANGUAGE_CODES = Object.freeze([
    "ar", "bg", "cs", "da", "de", "en", "es", "et", "fi", "fr",
    "hi", "hr", "hu", "it", "ja", "ko", "nb", "nl", "pl", "pt",
    "ro", "ru", "sk", "sv", "tr", "uk", "vi", "zh",
]);

export const TRANSCRIPTION_LANGUAGE_DATA = Object.freeze([
    { value: "auto", label: "Auto detect", aliases: ["automatic", "detect"] },
    { value: "af", label: "Afrikaans" },
    { value: "sq", label: "Albanian" },
    { value: "am", label: "Amharic" },
    { value: "ar", label: "Arabic" },
    { value: "hy", label: "Armenian" },
    { value: "as", label: "Assamese" },
    { value: "az", label: "Azerbaijani" },
    { value: "ba", label: "Bashkir" },
    { value: "eu", label: "Basque" },
    { value: "be", label: "Belarusian" },
    { value: "bn", label: "Bengali" },
    { value: "bs", label: "Bosnian" },
    { value: "br", label: "Breton" },
    { value: "bg", label: "Bulgarian" },
    { value: "yue", label: "Cantonese", aliases: ["Yue"] },
    { value: "ca", label: "Catalan", aliases: ["Valencian"] },
    { value: "zh", label: "Chinese (Mandarin)", aliases: ["Chinese", "Mandarin", "Simplified Chinese"] },
    { value: "hr", label: "Croatian" },
    { value: "cs", label: "Czech" },
    { value: "da", label: "Danish" },
    { value: "nl", label: "Dutch", aliases: ["Flemish"] },
    { value: "en", label: "English" },
    { value: "et", label: "Estonian" },
    { value: "fo", label: "Faroese" },
    { value: "fi", label: "Finnish" },
    { value: "fr", label: "French" },
    { value: "gl", label: "Galician" },
    { value: "ka", label: "Georgian" },
    { value: "de", label: "German" },
    { value: "el", label: "Greek" },
    { value: "gu", label: "Gujarati" },
    { value: "ht", label: "Haitian Creole", aliases: ["Haitian"] },
    { value: "ha", label: "Hausa" },
    { value: "haw", label: "Hawaiian" },
    { value: "he", label: "Hebrew" },
    { value: "hi", label: "Hindi" },
    { value: "hu", label: "Hungarian" },
    { value: "is", label: "Icelandic" },
    { value: "id", label: "Bahasa Indonesia", englishName: "Indonesian", aliases: ["Bahasa"] },
    { value: "it", label: "Italian" },
    { value: "ja", label: "Japanese" },
    { value: "jw", label: "Javanese", aliases: ["Basa Jawa"] },
    { value: "kn", label: "Kannada" },
    { value: "kk", label: "Kazakh" },
    { value: "km", label: "Khmer", aliases: ["Cambodian"] },
    { value: "ko", label: "Korean" },
    { value: "lo", label: "Lao", aliases: ["Laotian"] },
    { value: "la", label: "Latin" },
    { value: "lv", label: "Latvian" },
    { value: "ln", label: "Lingala" },
    { value: "lt", label: "Lithuanian" },
    { value: "lb", label: "Luxembourgish", aliases: ["Letzeburgesch"] },
    { value: "mk", label: "Macedonian" },
    { value: "mg", label: "Malagasy" },
    { value: "ms", label: "Malay", aliases: ["Bahasa Melayu"] },
    { value: "ml", label: "Malayalam" },
    { value: "mt", label: "Maltese" },
    { value: "mi", label: "Maori", aliases: ["Māori"] },
    { value: "mr", label: "Marathi" },
    { value: "mn", label: "Mongolian" },
    { value: "my", label: "Myanmar (Burmese)", aliases: ["Burmese"] },
    { value: "ne", label: "Nepali" },
    { value: "no", label: "Norwegian" },
    { value: "nb", label: "Norwegian Bokmål", aliases: ["Norwegian Bokmal", "Bokmal"] },
    { value: "nn", label: "Norwegian Nynorsk", aliases: ["Nynorsk"] },
    { value: "oc", label: "Occitan" },
    { value: "ps", label: "Pashto", aliases: ["Pushto"] },
    { value: "fa", label: "Persian", aliases: ["Farsi"] },
    { value: "pl", label: "Polish" },
    { value: "pt", label: "Portuguese" },
    { value: "pa", label: "Punjabi", aliases: ["Panjabi"] },
    { value: "ro", label: "Romanian", aliases: ["Moldavian", "Moldovan"] },
    { value: "ru", label: "Russian" },
    { value: "sa", label: "Sanskrit" },
    { value: "sr", label: "Serbian" },
    { value: "sn", label: "Shona" },
    { value: "sd", label: "Sindhi" },
    { value: "si", label: "Sinhala", aliases: ["Sinhalese"] },
    { value: "sk", label: "Slovak" },
    { value: "sl", label: "Slovenian" },
    { value: "so", label: "Somali" },
    { value: "es", label: "Spanish", aliases: ["Castilian"] },
    { value: "su", label: "Sundanese", aliases: ["Basa Sunda"] },
    { value: "sw", label: "Swahili" },
    { value: "sv", label: "Swedish" },
    { value: "tl", label: "Tagalog", aliases: ["Filipino"] },
    { value: "tg", label: "Tajik" },
    { value: "ta", label: "Tamil" },
    { value: "tt", label: "Tatar" },
    { value: "te", label: "Telugu" },
    { value: "th", label: "Thai" },
    { value: "bo", label: "Tibetan" },
    { value: "tr", label: "Turkish" },
    { value: "tk", label: "Turkmen" },
    { value: "uk", label: "Ukrainian" },
    { value: "ur", label: "Urdu" },
    { value: "uz", label: "Uzbek" },
    { value: "vi", label: "Vietnamese" },
    { value: "cy", label: "Welsh" },
    { value: "yi", label: "Yiddish" },
    { value: "yo", label: "Yoruba" },
]);

const WHISPER_LANGUAGES = new Set(WHISPER_LANGUAGE_CODES);
const PARAKEET_LANGUAGES = new Set(PARAKEET_LANGUAGE_CODES);
const NEMOTRON_LANGUAGES = new Set(NEMOTRON_LANGUAGE_CODES);
const LANGUAGE_BY_CODE = new Map(
    TRANSCRIPTION_LANGUAGE_DATA.map((language) => [language.value, language]),
);

/** @param {string | null | undefined} language */
export function normalizeTranscriptionLanguage(language) {
    const normalized = String(language ?? "").trim().toLowerCase().replaceAll("_", "-");
    if (!normalized || normalized === "none" || normalized === "auto") return "auto";
    return normalized.split("-")[0];
}

/** @param {string} language */
export function transcriptionLanguageName(language) {
    const normalized = normalizeTranscriptionLanguage(language);
    return LANGUAGE_BY_CODE.get(normalized)?.label ?? language.toUpperCase();
}

/**
 * @param {string} model
 * @param {string} language
 */
export function modelSupportsLanguage(model, language) {
    const normalized = normalizeTranscriptionLanguage(language);
    if (normalized === "auto") return true;
    if (model === "parakeet") return PARAKEET_LANGUAGES.has(normalized);
    if (model === "nemotron-3.5") return NEMOTRON_LANGUAGES.has(normalized);
    if (model === "faster-whisper-large-v3" || model === "faster-whisper-turbo") {
        return WHISPER_LANGUAGES.has(normalized);
    }
    return false;
}

/**
 * @param {string} model
 * @param {string} language
 */
export function compatibleTranscriptionModel(model, language) {
    if (modelSupportsLanguage(model, language)) return model;
    return [
        "faster-whisper-turbo",
        "faster-whisper-large-v3",
        "nemotron-3.5",
        "parakeet",
    ].find((candidate) => modelSupportsLanguage(candidate, language)) ?? model;
}

/**
 * @param {string} model
 * @param {string} language
 */
export function unsupportedLanguageReason(model, language) {
    if (modelSupportsLanguage(model, language)) return "";
    return `Not supported for ${transcriptionLanguageName(language)}`;
}
