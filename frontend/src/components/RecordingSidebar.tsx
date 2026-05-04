"use client";

import { Recording } from "@/src/api/sonaApi";
import RecordingStatusBadge from "@/src/components/RecordingStatusBadge";

interface Props {
    recordings: Recording[];
    selectedId?: string;
    onSelect: (recordingId: string) => void;
    onDelete: (recordingId: string) => void;
}

export default function RecordingSidebar({
    recordings,
    selectedId,
    onSelect,
    onDelete,
}: Props) {
    return (
        <aside className="min-h-[520px] border-r border-zinc-200 bg-zinc-50">
            <div className="border-b border-zinc-200 px-4 py-3">
                <h2 className="text-sm font-semibold text-zinc-900">Recordings</h2>
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
                        className={`border-b border-zinc-200 px-4 py-3 transition-colors ${
                            selectedId === recording.id
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
                                    <p className="truncate text-sm font-medium text-zinc-900">
                                        {recording.original_name}
                                    </p>
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
                    </div>
                ))}
            </div>
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
