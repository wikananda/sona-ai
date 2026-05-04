import { RuntimeDevice, TranscriptionModel } from "@/src/api/sonaApi";

export interface TranscriptionSettings {
    language?: string;
    model: TranscriptionModel;
    device: RuntimeDevice;
    minSpeakers?: number | "";
    maxSpeakers?: number | "";
}

export const TRANSCRIPTION_LANGUAGES = [
    { label: "Auto detect", value: "auto" },
    { label: "English", value: "en" },
    { label: "Indonesian", value: "id" },
];

export const TRANSCRIPTION_MODELS: {
    label: string;
    value: TranscriptionModel;
}[] = [
    { label: "Parakeet", value: "parakeet" },
    { label: "WhisperX", value: "whisperx" },
];

export function numberOrEmpty(value: string): number | "" {
    if (!value) return "";
    return Math.max(1, Number.parseInt(value, 10));
}

export function deviceLabel(device: RuntimeDevice): string {
    if (device === "auto") return "Auto device";
    return device.toUpperCase();
}
