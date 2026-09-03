import { RuntimeDevice, TranscriptionModel } from "@/src/api/sonaApi";
import type { LiveTranscriptionEngine } from "@/src/api/liveTranscriptionSocket";

export interface TranscriptionSettings {
    language?: string;
    model: TranscriptionModel;
    device: RuntimeDevice;
    minSpeakers?: number | "";
    maxSpeakers?: number | "";
}

export const TRANSCRIPTION_LANGUAGES = [
    { label: "Auto detect", value: "auto" },
    { label: "Arabic", value: "ar" },
    { label: "Bulgarian", value: "bg" },
    { label: "Chinese (Simplified)", value: "zh" },
    { label: "Croatian", value: "hr" },
    { label: "Czech", value: "cs" },
    { label: "Danish", value: "da" },
    { label: "Dutch", value: "nl" },
    { label: "English", value: "en" },
    { label: "Estonian", value: "et" },
    { label: "Finnish", value: "fi" },
    { label: "French", value: "fr" },
    { label: "German", value: "de" },
    { label: "Hindi", value: "hi" },
    { label: "Hungarian", value: "hu" },
    { label: "Indonesian", value: "id" },
    { label: "Italian", value: "it" },
    { label: "Japanese", value: "ja" },
    { label: "Korean", value: "ko" },
    { label: "Norwegian Bokmal", value: "nb" },
    { label: "Polish", value: "pl" },
    { label: "Portuguese", value: "pt" },
    { label: "Romanian", value: "ro" },
    { label: "Russian", value: "ru" },
    { label: "Slovak", value: "sk" },
    { label: "Spanish", value: "es" },
    { label: "Swedish", value: "sv" },
    { label: "Turkish", value: "tr" },
    { label: "Ukrainian", value: "uk" },
    { label: "Vietnamese", value: "vi" },
];

const NEMOTRON_LANGUAGES = new Set([
    "auto",
    "ar",
    "bg",
    "cs",
    "da",
    "de",
    "en",
    "es",
    "et",
    "fi",
    "fr",
    "hi",
    "hr",
    "hu",
    "it",
    "ja",
    "ko",
    "nb",
    "nl",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sv",
    "tr",
    "uk",
    "vi",
    "zh",
]);

export const TRANSCRIPTION_MODELS: {
    label: string;
    value: TranscriptionModel;
}[] = [
    { label: "Parakeet", value: "parakeet" },
    { label: "Nemotron 3.5 ASR", value: "nemotron-3.5" },
    { label: "Whisper Large-v3", value: "faster-whisper-large-v3" },
    { label: "Whisper Large-v3 Turbo", value: "faster-whisper-turbo" },
];

export function isTranscriptionModel(value: string): value is TranscriptionModel {
    return TRANSCRIPTION_MODELS.some((item) => item.value === value);
}

export function isModelLanguageCompatible(
    model: TranscriptionModel,
    language: string,
): boolean {
    const normalizedLanguage = language.toLowerCase().trim();
    if (model === "parakeet") {
        return normalizedLanguage === "auto" || normalizedLanguage === "en";
    }
    if (model === "nemotron-3.5") {
        return NEMOTRON_LANGUAGES.has(normalizedLanguage);
    }
    return true;
}

export function compatibleModelForLanguage(
    model: TranscriptionModel,
    language: string,
): TranscriptionModel {
    return isModelLanguageCompatible(model, language)
        ? model
        : "faster-whisper-turbo";
}

export function transcriptionModelLanguageNote(
    model: TranscriptionModel,
    language: string,
): string {
    if (model === "nemotron-3.5") {
        return language === "auto"
            ? "Nemotron can auto-detect its supported locales, but an explicit language is more reliable."
            : "Nemotron uses the local NeMo-Speech.cpp sidecar for batch and realtime transcription.";
    }
    if (language === "id") {
        return "Indonesian uses Whisper; Nemotron 3.5 and Parakeet do not support it."
    }
    if (!["auto", "en"].includes(language)) {
        return "Parakeet is currently available for English only in Sona."
    }
    return "";
}

export function liveEngineLabel(engine: LiveTranscriptionEngine): string {
    if (engine === "nemotron-live") return "Nemotron";
    if (engine === "parakeet-live") return "Parakeet";
    return "Whisper";
}

export function numberOrEmpty(value: string): number | "" {
    if (!value) return "";
    return Math.max(1, Number.parseInt(value, 10));
}

export function deviceLabel(device: RuntimeDevice): string {
    if (device === "auto") return "Auto device";
    return device.toUpperCase();
}
