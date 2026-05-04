"use client";

import { useState } from "react";
import { Recording, summarizeTranscript } from "@/src/api/sonaApi";
import RecordingStatusBadge from "@/src/components/RecordingStatusBadge";
import SummaryPanel from "@/src/components/SummaryPanel";
import TranscriptPanel from "@/src/components/TranscriptPanel";
import sanitizeTranscript from "@/src/utils/sanitizeTranscript";

type DetailTab = "transcript" | "summary";

interface Props {
    recording?: Recording | null;
    isLoading: boolean;
}

export default function RecordingDetail({ recording, isLoading }: Props) {
    const [activeTab, setActiveTab] = useState<DetailTab>("transcript");
    const [summary, setSummary] = useState("");
    const [isSummarizing, setIsSummarizing] = useState(false);

    if (isLoading && !recording) {
        return <div className="p-6 text-sm text-zinc-500">Loading recording...</div>;
    }

    if (!recording) {
        return <div className="p-6 text-sm text-zinc-500">Select a recording.</div>;
    }

    const segments = recording.transcript?.segments ?? [];

    const handleSummarize = async () => {
        if (!segments.length) return;

        setIsSummarizing(true);
        setSummary("");
        try {
            const nextSummary = await summarizeTranscript({
                text: sanitizeTranscript(segments),
            });
            setSummary(nextSummary);
        } finally {
            setIsSummarizing(false);
        }
    };

    return (
        <section className="flex min-h-[520px] flex-col bg-white">
            <div className="border-b border-zinc-200 px-6 py-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="min-w-0">
                        <h2 className="truncate text-lg font-semibold text-zinc-950">
                            {recording.original_name}
                        </h2>
                        <p className="mt-1 text-sm text-zinc-500">
                            {recording.model} / {recording.language_hint ?? "auto language"}
                        </p>
                    </div>
                    <RecordingStatusBadge status={recording.status} />
                </div>
            </div>

            <div className="flex-1 p-6">
                {recording.status === "pending" && (
                    <p className="text-sm text-zinc-500">Waiting to start transcription.</p>
                )}
                {recording.status === "processing" && (
                    <p className="text-sm text-zinc-500">Transcription is running.</p>
                )}
                {recording.status === "failed" && (
                    <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-800">
                        {recording.error ?? "Transcription failed."}
                    </div>
                )}
                {recording.status === "done" && (
                    <div className="flex flex-col gap-5">
                        <div className="flex border-b border-zinc-200">
                            <TabButton
                                label="Transcription"
                                isActive={activeTab === "transcript"}
                                onClick={() => setActiveTab("transcript")}
                            />
                            <TabButton
                                label="Summary"
                                isActive={activeTab === "summary"}
                                onClick={() => setActiveTab("summary")}
                            />
                        </div>

                        {activeTab === "transcript" && (
                            <TranscriptPanel segments={segments} />
                        )}

                        {activeTab === "summary" && (
                            <div className="flex flex-col gap-4">
                                <div className="flex items-center justify-between gap-4">
                                    <div>
                                        <h3 className="text-sm font-semibold text-zinc-900">
                                            Summary
                                        </h3>
                                        <p className="mt-1 text-sm text-zinc-500">
                                            Generate a concise summary from this recording transcript.
                                        </p>
                                    </div>
                                    <button
                                        type="button"
                                        onClick={handleSummarize}
                                        disabled={isSummarizing || !segments.length}
                                        className="min-h-10 shrink-0 rounded-md bg-zinc-950 px-4 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-40"
                                    >
                                        {isSummarizing
                                            ? "Summarizing"
                                            : summary
                                              ? "Re-summarize"
                                              : "Summarize"}
                                    </button>
                                </div>

                                {!summary && !isSummarizing && (
                                    <div className="rounded-md border border-zinc-200 bg-zinc-50 p-5 text-sm text-zinc-500">
                                        No summary generated yet.
                                    </div>
                                )}
                                <SummaryPanel summary={summary} isLoading={isSummarizing} />
                            </div>
                        )}
                    </div>
                )}
            </div>
        </section>
    );
}

function TabButton({
    label,
    isActive,
    onClick,
}: {
    label: string;
    isActive: boolean;
    onClick: () => void;
}) {
    return (
        <button
            type="button"
            onClick={onClick}
            className={`min-h-11 border-b-2 px-4 text-sm font-medium transition-colors ${
                isActive
                    ? "border-zinc-950 text-zinc-950"
                    : "border-transparent text-zinc-500 hover:text-zinc-950"
            }`}
        >
            {label}
        </button>
    );
}
