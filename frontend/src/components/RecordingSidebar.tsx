"use client";

import {
    Recording,
    RuntimeDevice,
    RuntimeDevices,
    TranscriptionModel,
} from "@/src/api/sonaApi";
import BrowserAudioRecorder from "@/src/components/BrowserAudioRecorder";
import LiveTranscriptionPanel from "@/src/components/LiveTranscriptionPanel";
import RecordingStatusBadge from "@/src/components/RecordingStatusBadge";
import RecordingUploader from "@/src/components/RecordingUploader";
import { useState } from "react";

type AddRecordingMode = "upload" | "record" | "live";

interface Props {
    projectId: string;
    recordings: Recording[];
    selectedId?: string;
    onSelect: (recordingId: string) => void;
    onDelete: (recordingId: string) => void;
    onRename: (recordingId: string, name: string) => Promise<void>;
    renamingId?: string;
    onUpload: (params: {
        files: File[];
        language?: string;
        model: TranscriptionModel;
        device: RuntimeDevice;
        minSpeakers?: number | "";
        maxSpeakers?: number | "";
        extractSpeakers?: boolean;
    }) => Promise<void>;
    onLiveSaved: (recording: Recording) => Promise<void>;
    isUploading: boolean;
    runtimeDevices: RuntimeDevices;
}

export default function RecordingSidebar({
    projectId,
    recordings,
    selectedId,
    onSelect,
    onDelete,
    onRename,
    renamingId,
    onUpload,
    onLiveSaved,
    isUploading,
    runtimeDevices,
}: Props) {
    const [editingId, setEditingId] = useState<string>();
    const [draftName, setDraftName] = useState("");
    const [isUploadOpen, setIsUploadOpen] = useState(false);
    const [addMode, setAddMode] = useState<AddRecordingMode>("upload");

    const startEditing = (recording: Recording) => {
        setEditingId(recording.id);
        setDraftName(recording.original_name);
    };

    const cancelEditing = () => {
        setEditingId(undefined);
        setDraftName("");
    };

    const submitRename = async (recording: Recording) => {
        const nextName = draftName.trim()
        if (!nextName || nextName === recording.original_name) {
            cancelEditing();
            return;
        }

        await onRename(recording.id, nextName);
        cancelEditing();
    }

    return (
        <aside className="min-h-[520px] border-r border-zinc-200 bg-zinc-50">
            <div className="border-b border-zinc-200 px-4 py-3">
                <div className="flex items-center justify-between gap-3">
                    <h2 className="text-sm font-semibold text-zinc-900">Recordings</h2>
                    <button
                        type="button"
                        onClick={() => setIsUploadOpen(true)}
                        disabled={isUploading}
                        aria-label="Add recording"
                        className="flex h-8 w-8 items-center justify-center rounded-md border border-zinc-300 bg-white text-lg leading-none text-zinc-700 hover:border-zinc-400 cursor-pointer hover:text-zinc-950 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                        +
                    </button>
                </div>
            </div>
            <div className="flex flex-col">
                {recordings.length === 0 && (
                    <div className="px-4 py-8 text-sm text-zinc-500">
                        No recordings yet.
                    </div>
                )}

                {recordings.map((recording) => (
                    <div
                        key={recording.id}
                        className={`border-b border-zinc-200 px-4 py-3 transition-colors ${selectedId === recording.id
                            ? "bg-white"
                            : "bg-zinc-50 hover:bg-white"
                            }`}
                    >
                        <button
                            type="button"
                            onClick={() => onSelect(recording.id)}
                            className="w-full text-left"
                        >
                            <div className="flex items-start justify-between gap-3">
                                <div className="min-w-0">
                                    {editingId === recording.id ? (
                                        <form
                                            onSubmit={async (event) => {
                                                event.preventDefault();
                                                await submitRename(recording);
                                            }}
                                            className="flex gap-2"
                                        >
                                            <input
                                                value={draftName}
                                                onChange={(event) => setDraftName(event.target.value)}
                                                className="min-w-0 flex-1 rounded-md border border-zinc-300 py-1 px-1 text-sm"
                                                autoFocus
                                            />
                                            <button
                                                type="submit"
                                                disabled={renamingId === recording.id}
                                                className="text-xs font-medium text-zinc-900 disabled:opacity-50"
                                            >
                                                Save
                                            </button>
                                            <button
                                                type="button"
                                                onClick={cancelEditing}
                                                disabled={renamingId === recording.id}
                                                className="text-xs font-medium text-zinc-500 hover:text-zinc-900"
                                            >
                                                Cancel
                                            </button>
                                        </form>
                                    ) : (
                                        <p className="truncate text-sm font-medium text-zinc-900">
                                            {recording.original_name}
                                        </p>
                                    )}
                                    <p className="mt-1 text-xs text-zinc-500">
                                        {formatDate(recording.created_at)}
                                    </p>
                                </div>
                                <RecordingStatusBadge status={recording.status} />
                            </div>
                            {recording.error && (
                                <p className="mt-2 line-clamp-2 text-xs text-red-700">
                                    {recording.error}
                                </p>
                            )}
                        </button>
                        <div className="flex gap-2">
                            <button
                                type="button"
                                onClick={(event) => {
                                    event.stopPropagation();
                                    onDelete(recording.id);
                                }}
                                className="mt-2 inline-block text-xs font-medium text-zinc-500 hover:text-red-700"
                            >
                                Delete
                            </button>
                            <button
                                type="button"
                                onClick={(event) => {
                                    event.stopPropagation();
                                    startEditing(recording);
                                }}
                                className="mt-2 inline-block text-xs font-medium text-zinc-500 hover:text-zinc-900"
                            >
                                Rename
                            </button>
                        </div>
                    </div>
                ))}
            </div>
            {isUploadOpen && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 px-4">
                    <div className="w-full max-w-4xl rounded-lg bg-white shadow-xl">
                        <div className="flex items-start justify-between gap-4 border-b border-zinc-200 px-5 py-4">
                            <div>
                                <h2 className="text-base font-semibold text-zinc-950">
                                    Add recording
                                </h2>
                                <p className="mt-1 text-sm text-zinc-500">
                                    Upload audio, record first, or transcribe live.
                                </p>
                            </div>
                            <button
                                type="button"
                                onClick={() => setIsUploadOpen(false)}
                                disabled={isUploading}
                                aria-label="Close add recording"
                                className="rounded-md px-2 py-1 text-xl leading-none text-zinc-500 hover:text-zinc-950 cursor-pointer disabled:cursor-not-allowed disabled:opacity-50"
                            >
                                x
                            </button>
                        </div>
                        <div className="p-5">
                            <div className="mb-4 grid grid-cols-3 gap-2 rounded-lg bg-zinc-100 p-1">
                                <button
                                    type="button"
                                    onClick={() => setAddMode("upload")}
                                    disabled={isUploading}
                                    className={`min-h-10 rounded-md text-sm font-medium transition-colors ${addMode === "upload"
                                        ? "bg-white text-zinc-950 shadow-sm"
                                        : "text-zinc-600 hover:text-zinc-950"
                                        } disabled:cursor-not-allowed disabled:opacity-50`}
                                >
                                    Upload audio
                                </button>
                                <button
                                    type="button"
                                    onClick={() => setAddMode("record")}
                                    disabled={isUploading}
                                    className={`min-h-10 rounded-md text-sm font-medium transition-colors ${addMode === "record"
                                        ? "bg-white text-zinc-950 shadow-sm"
                                        : "text-zinc-600 hover:text-zinc-950"
                                        } disabled:cursor-not-allowed disabled:opacity-50`}
                                >
                                    Record audio
                                </button>
                                <button
                                    type="button"
                                    onClick={() => setAddMode("live")}
                                    disabled={isUploading}
                                    className={`min-h-10 rounded-md text-sm font-medium transition-colors ${addMode === "live"
                                        ? "bg-white text-zinc-950 shadow-sm"
                                        : "text-zinc-600 hover:text-zinc-950"
                                        } disabled:cursor-not-allowed disabled:opacity-50`}
                                >
                                    Live transcription
                                </button>
                            </div>

                            {addMode === "upload" ? (
                                <RecordingUploader
                                    onUpload={async (params) => {
                                        await onUpload(params);
                                        setIsUploadOpen(false);
                                    }}
                                    isUploading={isUploading}
                                    runtimeDevices={runtimeDevices}
                                />
                            ) : addMode === "record" ? (
                                <BrowserAudioRecorder
                                    onUpload={async (params) => {
                                        await onUpload(params);
                                        setIsUploadOpen(false);
                                    }}
                                    isUploading={isUploading}
                                    runtimeDevices={runtimeDevices}
                                />
                            ) : (
                                <LiveTranscriptionPanel
                                    projectId={projectId}
                                    runtimeDevices={runtimeDevices}
                                    onSaved={async (recording) => {
                                        await onLiveSaved(recording);
                                        setIsUploadOpen(false);
                                    }}
                                />
                            )}
                        </div>
                    </div>
                </div>
            )}
        </aside>
    );
}

function formatDate(value: string): string {
    return new Intl.DateTimeFormat(undefined, {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
    }).format(new Date(value));
}
