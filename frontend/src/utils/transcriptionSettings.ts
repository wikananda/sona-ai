import { RuntimeDevice, TranscriptionModel } from "@/src/api/sonaApi";
import type { LiveTranscriptionEngine } from "@/src/api/liveTranscriptionSocket";
import {
    compatibleTranscriptionModel,
    modelSupportsLanguage,
    transcriptionLanguageName as languageName,
    TRANSCRIPTION_LANGUAGE_DATA,
    unsupportedLanguageReason as incompatibilityReason,
} from "@/src/utils/transcriptionCapabilities.mjs";

export interface TranscriptionSettings {
    language?: string;
    model: TranscriptionModel;
    device: RuntimeDevice;
    minSpeakers?: number | "";
    maxSpeakers?: number | "";
}

export interface TranscriptionLanguage {
    value: string;
    label: string;
    englishName?: string;
    aliases?: readonly string[];
}

export const TRANSCRIPTION_LANGUAGES = TRANSCRIPTION_LANGUAGE_DATA as readonly TranscriptionLanguage[];

export const TRANSCRIPTION_MODELS: {
    label: string;
    value: TranscriptionModel;
    description: string;
}[] = [
    {
        label: "Parakeet",
        value: "parakeet",
        description: "Fast multilingual ASR · 25 languages",
    },
    {
        label: "Nemotron 3.5 ASR",
        value: "nemotron-3.5",
        description: "Native streaming ASR · 28 languages",
    },
    {
        label: "Whisper Large-v3",
        value: "faster-whisper-large-v3",
        description: "Highest Whisper accuracy · 100 languages",
    },
    {
        label: "Whisper Large-v3 Turbo",
        value: "faster-whisper-turbo",
        description: "Faster Whisper inference · 100 languages",
    },
];

export function isTranscriptionModel(value: string): value is TranscriptionModel {
    return TRANSCRIPTION_MODELS.some((item) => item.value === value);
}

export function isModelLanguageCompatible(
    model: TranscriptionModel,
    language: string,
): boolean {
    return modelSupportsLanguage(model, language);
}

export function compatibleModelForLanguage(
    model: TranscriptionModel,
    language: string,
): TranscriptionModel {
    return compatibleTranscriptionModel(model, language) as TranscriptionModel;
}

export function transcriptionLanguageName(language: string): string {
    return languageName(language);
}

export function unsupportedLanguageReason(
    model: TranscriptionModel,
    language: string,
): string {
    return incompatibilityReason(model, language);
}

export function transcriptionModelLanguageNote(
    model: TranscriptionModel,
    language: string,
): string {
    const selectedLanguage = transcriptionLanguageName(language);
    if (language === "auto") {
        return `${TRANSCRIPTION_MODELS.find((item) => item.value === model)?.label ?? "This model"} will detect the spoken language automatically.`;
    }
    return `${TRANSCRIPTION_MODELS.find((item) => item.value === model)?.label ?? "This model"} supports ${selectedLanguage}.`;
}

export function liveEngineLabel(engine: LiveTranscriptionEngine): string {
    if (engine === "whisper-mps-live") return "Whisper (Apple GPU)";
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
