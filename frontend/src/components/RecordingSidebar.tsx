"use client";

import {
    Recording,
} from "@/src/api/sonaApi";
import RecordingStatusBadge from "@/src/components/RecordingStatusBadge";
import { formatDateTime } from "@/src/utils/formatters";
import { memo, useState } from "react";

interface Props {
    recordings: Recording[];
    selectedId?: string;
    isDraftSelected?: boolean;
    hasDraft?: boolean;
    onSelect: (recordingId: string) => void;
    onSelectDraft: () => void;
    onStartDraft: () => void;
    onDelete: (recordingId: string) => void;
    onRename: (recordingId: string, name: string) => Promise<void>;
    renamingId?: string;
    isUploading: boolean;
}

function RecordingSidebar({
    recordings,
    selectedId,
    isDraftSelected = false,
    hasDraft = false,
    onSelect,
    onSelectDraft,
    onStartDraft,
    onDelete,
    onRename,
    renamingId,
    isUploading,
}: Props) {
    const [editingId, setEditingId] = useState<string>();
    const [draftName, setDraftName] = useState("");

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
                        onClick={onStartDraft}
                        disabled={isUploading}
                        aria-label="Add recording"
                        className="flex h-8 w-8 items-center justify-center rounded-md border border-zinc-300 bg-white text-lg leading-none text-zinc-700 hover:border-zinc-400 cursor-pointer hover:text-zinc-950 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                        +
                    </button>
                </div>
            </div>
            <div className="flex flex-col">
                {recordings.length === 0 && !hasDraft && (
                    <div className="px-4 py-8 text-sm text-zinc-500">
                        No recordings yet.
                    </div>
                )}

                {hasDraft && (
                    <button
                        type="button"
                        onClick={onSelectDraft}
                        className={`w-full border-b border-zinc-200 px-4 py-3 text-left transition-colors ${isDraftSelected
                            ? "bg-white"
                            : "bg-zinc-50 hover:bg-white"
                            }`}
                    >
                        <div className="flex items-start justify-between gap-3">
                            <div className="min-w-0">
                                <p className="truncate text-sm font-medium text-zinc-900">
                                    New recording
                                </p>
                                <p className="mt-1 text-xs text-zinc-500">
                                    Draft, not saved yet
                                </p>
                            </div>
                            <span className="rounded-full bg-zinc-100 px-2 py-1 text-xs font-medium text-zinc-600">
                                Draft
                            </span>
                        </div>
                    </button>
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
                                        {formatDateTime(recording.created_at)}
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
                                className="mt-2 inline-block text-xs font-medium text-zinc-500 hover:cursor-pointer hover:text-red-700"
                            >
                                Delete
                            </button>
                            <button
                                type="button"
                                onClick={(event) => {
                                    event.stopPropagation();
                                    startEditing(recording);
                                }}
                                className="mt-2 inline-block text-xs font-medium text-zinc-500 hover:cursor-pointer hover:text-zinc-900"
                            >
                                Rename
                            </button>
                        </div>
                    </div>
                ))}
            </div>
        </aside>
    );
}

export default memo(RecordingSidebar);
