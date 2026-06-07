"use client";

import { FormEvent, useState } from "react";
import { RuntimeDevice, RuntimeDevices, TranscriptionModel } from "@/src/api/sonaApi";
import AudioFileDropzone from "@/src/components/AudioFileDropzone";
import {
    deviceLabel,
    numberOrEmpty,
    TRANSCRIPTION_LANGUAGES,
    TRANSCRIPTION_MODELS,
} from "@/src/utils/transcriptionSettings";

interface Props {
    onUpload: (params: {
        files: File[];
        language?: string;
        model: TranscriptionModel;
        device: RuntimeDevice;
        minSpeakers?: number | "";
        maxSpeakers?: number | "";
        extractSpeakers?: boolean;
    }) => Promise<void>;
    isUploading: boolean;
    runtimeDevices: RuntimeDevices;
}

export default function RecordingUploader({ onUpload, isUploading, runtimeDevices }: Props) {
    const [file, setFile] = useState<File | null>(null);
    const [language, setLanguage] = useState("auto");
    const [model, setModel] = useState<TranscriptionModel>("parakeet");
    const [device, setDevice] = useState<RuntimeDevice>(runtimeDevices.default);
    const [minSpeakers, setMinSpeakers] = useState<number | "">("");
    const [maxSpeakers, setMaxSpeakers] = useState<number | "">("");
    const [extractSpeakers, setExtractSpeakers] = useState(true);
    const [speakerError, setSpeakerError] = useState("");
    const selectedDevice = runtimeDevices.available.includes(device)
        ? device
        : runtimeDevices.default;

    const handleSubmit = async (event: FormEvent) => {
        event.preventDefault();
        if (!file) return;
        if (extractSpeakers && (minSpeakers === "" || maxSpeakers === "")) {
            setSpeakerError("Min and max speakers are required when extracting speakers.");
            return;
        }

        setSpeakerError("");
        await onUpload({
            files: [file],
            language,
            model,
            device: selectedDevice,
            minSpeakers: extractSpeakers ? minSpeakers : "",
            maxSpeakers: extractSpeakers ? maxSpeakers : "",
            extractSpeakers,
        });
        setFile(null);
    };

    return (
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            <AudioFileDropzone
                file={file}
                onFileChange={(nextFile) => {
                    setFile(nextFile);
                    setSpeakerError("");
                }}
                disabled={isUploading}
                title="Drop one audio file here"
                helperText="Choose from computer or drag audio into this area."
            />

            {file && (
                <div className="flex flex-col gap-3">
                    <label className="flex flex-col gap-1">
                        <span className="text-xs font-medium text-zinc-500">Language</span>
                        <select
                            value={language}
                            onChange={(event) => setLanguage(event.target.value)}
                            disabled={isUploading}
                            className="min-h-10 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                            {TRANSCRIPTION_LANGUAGES.map((item) => (
                                <option key={item.value} value={item.value}>
                                    {item.label}
                                </option>
                            ))}
                        </select>
                    </label>

                    <label className="flex flex-col gap-1">
                        <span className="text-xs font-medium text-zinc-500">Model</span>
                        <select
                            value={model}
                            onChange={(event) => setModel(event.target.value as TranscriptionModel)}
                            disabled={isUploading}
                            className="min-h-10 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                            {TRANSCRIPTION_MODELS.map((item) => (
                                <option key={item.value} value={item.value}>
                                    {item.label}
                                </option>
                            ))}
                        </select>
                    </label>

                    <label className="flex flex-col gap-1">
                        <span className="text-xs font-medium text-zinc-500">Device</span>
                        <select
                            value={selectedDevice}
                            onChange={(event) => setDevice(event.target.value as RuntimeDevice)}
                            disabled={isUploading}
                            className="min-h-10 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                            {runtimeDevices.available.map((item) => (
                                <option key={item} value={item}>
                                    {deviceLabel(item)}
                                </option>
                            ))}
                        </select>
                    </label>

                    <label className="flex items-start gap-2 text-sm text-zinc-600">
                        <input
                            type="checkbox"
                            checked={extractSpeakers}
                            onChange={(event) => {
                                setExtractSpeakers(event.target.checked);
                                setSpeakerError("");
                            }}
                            disabled={isUploading}
                            className="mt-0.5 h-4 w-4 rounded border-zinc-300 text-zinc-950 focus:ring-zinc-950 disabled:cursor-not-allowed disabled:opacity-50"
                        />
                        <span>
                            Extract speakers
                            <span className="block text-xs text-zinc-500">
                                Uncheck to transcribe first and run diarization later.
                            </span>
                        </span>
                    </label>

                    {extractSpeakers && (
                        <div className="grid gap-3 sm:grid-cols-2">
                            <input
                                type="number"
                                min="1"
                                value={minSpeakers}
                                onChange={(event) => setMinSpeakers(numberOrEmpty(event.target.value))}
                                disabled={isUploading}
                                placeholder="Min speakers"
                                className="min-h-10 rounded-md border border-zinc-300 px-3 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                            />

                            <input
                                type="number"
                                min="1"
                                value={maxSpeakers}
                                onChange={(event) => setMaxSpeakers(numberOrEmpty(event.target.value))}
                                disabled={isUploading}
                                placeholder="Max speakers"
                                className="min-h-10 rounded-md border border-zinc-300 px-3 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                            />
                        </div>
                    )}

                    <div className="flex items-center gap-2">
                        <button
                            type="submit"
                            disabled={isUploading || !file}
                            className="min-h-10 flex-1 rounded-md bg-zinc-950 px-4 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-40"
                        >
                            {isUploading ? "Uploading" : "Upload"}
                        </button>
                        <button
                            type="button"
                            onClick={() => {
                                setFile(null);
                                setSpeakerError("");
                            }}
                            disabled={isUploading}
                            className="min-h-10 rounded-md border border-zinc-300 px-3 text-sm font-medium text-zinc-700 hover:border-zinc-400 hover:text-zinc-950 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                            Clear
                        </button>
                    </div>
                </div>
            )}

            {speakerError && (
                <p className="mt-3 text-sm text-red-700">{speakerError}</p>
            )}
        </form>
    );
}
