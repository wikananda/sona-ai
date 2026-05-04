"use client";

import { FormEvent, useState } from "react";
import { TranscriptionModel } from "@/src/api/sonaApi";

interface Props {
    onUpload: (params: {
        files: File[];
        language?: string;
        model: TranscriptionModel;
        minSpeakers?: number | "";
        maxSpeakers?: number | "";
    }) => Promise<void>;
    isUploading: boolean;
}

const LANGUAGES = [
    { label: "Auto detect", value: "auto" },
    { label: "English", value: "en" },
    { label: "Indonesian", value: "id" },
];

const MODELS: { label: string; value: TranscriptionModel }[] = [
    { label: "Parakeet", value: "parakeet" },
    { label: "WhisperX", value: "whisperx" },
];

export default function RecordingUploader({ onUpload, isUploading }: Props) {
    const [files, setFiles] = useState<File[]>([]);
    const [language, setLanguage] = useState("auto");
    const [model, setModel] = useState<TranscriptionModel>("parakeet");
    const [minSpeakers, setMinSpeakers] = useState<number | "">("");
    const [maxSpeakers, setMaxSpeakers] = useState<number | "">("");

    const handleSubmit = async (event: FormEvent) => {
        event.preventDefault();
        if (!files.length) return;

        await onUpload({
            files,
            language,
            model,
            minSpeakers,
            maxSpeakers,
        });
        setFiles([]);
    };

    return (
        <form onSubmit={handleSubmit} className="rounded-lg border border-zinc-200 bg-white p-4">
            <div className="grid gap-4 xl:grid-cols-[1.5fr_0.8fr_0.8fr_0.7fr_0.7fr_auto]">
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
                    {LANGUAGES.map((item) => (
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
                    {MODELS.map((item) => (
                        <option key={item.value} value={item.value}>
                            {item.label}
                        </option>
                    ))}
                </select>

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

                <button
                    type="submit"
                    disabled={isUploading || !files.length}
                    className="min-h-11 rounded-md bg-zinc-950 px-4 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-40"
                >
                    {isUploading ? "Uploading" : "Upload"}
                </button>
            </div>
        </form>
    );
}

function numberOrEmpty(value: string): number | "" {
    if (!value) return "";
    return Math.max(1, Number.parseInt(value, 10));
}
