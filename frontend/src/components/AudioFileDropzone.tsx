"use client";

import { DragEvent, useRef, useState } from "react";

interface Props {
    file: File | null;
    onFileChange: (file: File | null) => void;
    disabled?: boolean;
    title?: string;
    helperText?: string;
    className?: string;
}

export default function AudioFileDropzone({
    file,
    onFileChange,
    disabled = false,
    title = "Drop audio here",
    helperText = "Supports common audio files such as MP3, WAV, and M4A.",
    className = "",
}: Props) {
    const inputRef = useRef<HTMLInputElement | null>(null);
    const [dragDepth, setDragDepth] = useState(0);
    const [fileError, setFileError] = useState("");
    const isDragging = dragDepth > 0;

    const handleFileSelection = (selectedFile: File | undefined) => {
        if (!selectedFile) return;

        if (!isAudioFile(selectedFile)) {
            setFileError("Please choose an audio file.");
            onFileChange(null);
            return;
        }

        setFileError("");
        onFileChange(selectedFile);
    };

    const handleDragEnter = (event: DragEvent<HTMLDivElement>) => {
        event.preventDefault();
        event.stopPropagation();
        if (disabled) return;

        setDragDepth((current) => current + 1);
    };

    const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
        event.preventDefault();
        event.stopPropagation();
        if (disabled) return;

        event.dataTransfer.dropEffect = "copy";
    };

    const handleDragLeave = (event: DragEvent<HTMLDivElement>) => {
        event.preventDefault();
        event.stopPropagation();
        if (disabled) return;

        setDragDepth((current) => Math.max(0, current - 1));
    };

    const handleDrop = (event: DragEvent<HTMLDivElement>) => {
        event.preventDefault();
        event.stopPropagation();
        setDragDepth(0);
        if (disabled) return;

        const droppedFiles = Array.from(event.dataTransfer.files);
        const audioFile = droppedFiles.find(isAudioFile);
        if (!audioFile) {
            setFileError("Drop an audio file to continue.");
            onFileChange(null);
            return;
        }

        handleFileSelection(audioFile);
    };

    return (
        <div className={`flex flex-col gap-2 ${className}`}>
            <div
                onDragEnter={handleDragEnter}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`flex min-h-32 flex-col items-center justify-center gap-2 rounded-lg border border-dashed px-4 py-5 text-center transition-colors ${isDragging
                    ? "border-zinc-900 bg-zinc-100"
                    : "border-zinc-300 bg-zinc-50 hover:border-zinc-500 hover:bg-zinc-100"
                    } ${disabled ? "cursor-not-allowed opacity-60" : ""}`}
            >
                <input
                    ref={inputRef}
                    type="file"
                    accept="audio/*"
                    disabled={disabled}
                    className="sr-only"
                    onChange={(event) => {
                        handleFileSelection(event.target.files?.[0]);
                        event.currentTarget.value = "";
                    }}
                />
                <span className="text-sm font-medium text-zinc-800">
                    {file ? file.name : title}
                </span>
                <span className="text-xs text-zinc-500">
                    {file ? formatFileSize(file.size) : helperText}
                </span>
                <button
                    type="button"
                    onClick={() => inputRef.current?.click()}
                    disabled={disabled}
                    className="mt-1 min-h-9 rounded-md border border-zinc-300 bg-white px-3 text-sm font-medium text-zinc-700 hover:cursor-pointer hover:border-zinc-400 hover:text-zinc-950 disabled:cursor-not-allowed disabled:opacity-50"
                >
                    Choose from computer
                </button>
            </div>
            {fileError && (
                <p className="text-sm text-red-700">{fileError}</p>
            )}
        </div>
    );
}

function isAudioFile(file: File): boolean {
    if (file.type.startsWith("audio/")) return true;

    return /\.(mp3|wav|m4a|aac|flac|ogg|opus|webm)$/i.test(file.name);
}

function formatFileSize(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    const kb = bytes / 1024;
    if (kb < 1024) return `${kb.toFixed(1)} KB`;
    return `${(kb / 1024).toFixed(1)} MB`;
}
