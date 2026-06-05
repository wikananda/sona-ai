"use client";

import { FormEvent, useState } from "react";
import { RuntimeDevice, RuntimeDevices, TranscriptionModel } from "@/src/api/sonaApi";
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
    const [files, setFiles] = useState<File[]>([]);
    const [language, setLanguage] = useState("auto");
    const [model, setModel] = useState<TranscriptionModel>("parakeet");
    const [device, setDevice] = useState<RuntimeDevice>(runtimeDevices.default);
    const [minSpeakers, setMinSpeakers] = useState<number | "">("");
    const [maxSpeakers, setMaxSpeakers] = useState<number | "">("");
    const [extractSpeakers, setExtractSpeakers] = useState(true);
    const selectedDevice = runtimeDevices.available.includes(device)
        ? device
        : runtimeDevices.default;

    const handleSubmit = async (event: FormEvent) => {
        event.preventDefault();
        if (!files.length) return;

        await onUpload({
            files,
            language,
            model,
            device: selectedDevice,
            minSpeakers,
            maxSpeakers,
            extractSpeakers,
        });
        setFiles([]);
    };

    return (
        <form onSubmit={handleSubmit} className="rounded-lg border border-zinc-200 bg-white p-4">
            <div className="grid gap-4 xl:grid-cols-[1.5fr_0.8fr_0.8fr_0.7fr_auto]">
                <label className="flex min-h-11 cursor-pointer items-center rounded-md border border-dashed border-zinc-300 px-3 text-sm text-zinc-600">
                    <input
                        type="file"
                        accept="audio/*"
                        multiple
                        className="sr-only"
                        onChange={(event) => {
                            setFiles(Array.from(event.target.files ?? []));
                        }}
                    />
                    {files.length ? `${files.length} file(s) selected` : "Choose audio files"}
                </label>

                <select
                    value={language}
                    onChange={(event) => setLanguage(event.target.value)}
                    className="min-h-11 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none focus:border-zinc-900"
                >
                    {TRANSCRIPTION_LANGUAGES.map((item) => (
                        <option key={item.value} value={item.value}>
                            {item.label}
                        </option>
                    ))}
                </select>

                <select
                    value={model}
                    onChange={(event) => setModel(event.target.value as TranscriptionModel)}
                    className="min-h-11 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none focus:border-zinc-900"
                >
                    {TRANSCRIPTION_MODELS.map((item) => (
                        <option key={item.value} value={item.value}>
                            {item.label}
                        </option>
                    ))}
                </select>

                <select
                    value={selectedDevice}
                    onChange={(event) => setDevice(event.target.value as RuntimeDevice)}
                    className="min-h-11 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none focus:border-zinc-900"
                >
                    {runtimeDevices.available.map((item) => (
                        <option key={item} value={item}>
                            {deviceLabel(item)}
                        </option>
                    ))}
                </select>

                <button
                    type="submit"
                    disabled={isUploading || !files.length}
                    className="min-h-11 rounded-md bg-zinc-950 px-4 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-40"
                >
                    {isUploading ? "Uploading" : "Upload"}
                </button>
            </div>

            <label className="mt-3 flex w-fit items-start gap-2 text-sm text-zinc-600">
                <input
                    type="checkbox"
                    checked={extractSpeakers}
                    onChange={(event) => setExtractSpeakers(event.target.checked)}
                    disabled={isUploading}
                    className="mt-0.5 h-4 w-4 rounded border-zinc-300 text-zinc-950 focus:ring-zinc-950 disabled:cursor-not-allowed disabled:opacity-50"
                />
                <span>
                    Extract speakers
                    <span className="block text-xs text-zinc-500">
                        Uncheck to transcribe first and run diarization from the transcript tab later.
                    </span>
                </span>
            </label>

            {extractSpeakers && (
                <div className="mt-3 grid gap-3 sm:grid-cols-2 xl:max-w-md">
                    <input
                        type="number"
                        min="1"
                        value={minSpeakers}
                        onChange={(event) => setMinSpeakers(numberOrEmpty(event.target.value))}
                        placeholder="Min speakers"
                        className="min-h-11 rounded-md border border-zinc-300 px-3 text-sm outline-none focus:border-zinc-900"
                    />

                    <input
                        type="number"
                        min="1"
                        value={maxSpeakers}
                        onChange={(event) => setMaxSpeakers(numberOrEmpty(event.target.value))}
                        placeholder="Max speakers"
                        className="min-h-11 rounded-md border border-zinc-300 px-3 text-sm outline-none focus:border-zinc-900"
                    />
                </div>
            )}

            {files.length > 0 && (
                <div className="mt-4 border-t border-zinc-200 pt-3">
                    <div className="mb-2 flex items-center justify-between gap-3">
                        <p className="text-xs font-semibold uppercase text-zinc-500">
                            Selected audio
                        </p>
                        <button
                            type="button"
                            onClick={() => setFiles([])}
                            disabled={isUploading}
                            className="text-xs font-medium text-zinc-500 hover:text-red-700 disabled:cursor-not-allowed disabled:opacity-40"
                        >
                            Clear
                        </button>
                    </div>
                    <ul className="flex max-h-36 flex-col gap-1 overflow-y-auto">
                        {files.map((file) => (
                            <li
                                key={`${file.name}-${file.lastModified}-${file.size}`}
                                className="grid grid-cols-[1fr_auto] gap-3 rounded-md bg-zinc-50 px-3 py-2 text-sm"
                            >
                                <span className="truncate font-medium text-zinc-800">
                                    {file.name}
                                </span>
                                <span className="text-xs text-zinc-500">
                                    {formatFileSize(file.size)}
                                </span>
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </form>
    );
}

function formatFileSize(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    const kb = bytes / 1024;
    if (kb < 1024) return `${kb.toFixed(1)} KB`;
    return `${(kb / 1024).toFixed(1)} MB`;
}
