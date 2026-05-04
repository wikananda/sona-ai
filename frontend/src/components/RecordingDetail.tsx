"use client";

import { FormEvent, useState } from "react";
import {
    Recording,
    RetranscribeParams,
    RuntimeDevice,
    RuntimeDevices,
    summarizeTranscript,
    SummaryModel,
    TranscriptionModel,
} from "@/src/api/sonaApi";
import RecordingStatusBadge from "@/src/components/RecordingStatusBadge";
import SummaryPanel from "@/src/components/SummaryPanel";
import TranscriptPanel from "@/src/components/TranscriptPanel";
import sanitizeTranscript from "@/src/utils/sanitizeTranscript";
import {
    deviceLabel,
    numberOrEmpty,
    TRANSCRIPTION_LANGUAGES,
    TRANSCRIPTION_MODELS,
} from "@/src/utils/transcriptionSettings";

type DetailTab = "transcript" | "summary";

interface Props {
    recording?: Recording | null;
    isLoading: boolean;
    runtimeDevices: RuntimeDevices;
    isRetranscribing?: boolean;
    onRetranscribe?: (
        recordingId: string,
        settings: RetranscribeParams,
    ) => Promise<void>;
    isRenamingSpeakers?: boolean;
    onRenameSpeakers?: (
        recordingId: string,
        speakers: Record<string, string>,
    ) => Promise<void>;
}

export default function RecordingDetail({
    recording,
    isLoading,
    runtimeDevices,
    isRetranscribing = false,
    onRetranscribe,
    isRenamingSpeakers = false,
    onRenameSpeakers,
}: Props) {
    const [activeTab, setActiveTab] = useState<DetailTab>("transcript");
    const [isSpeakerEditorOpen, setIsSpeakerEditorOpen] = useState(false);
    const [isRetranscribeEditorOpen, setIsRetranscribeEditorOpen] = useState(false);
    const [retranscribeLanguage, setRetranscribeLanguage] = useState("auto");
    const [retranscribeModel, setRetranscribeModel] =
        useState<TranscriptionModel>("parakeet");
    const [retranscribeDevice, setRetranscribeDevice] =
        useState<RuntimeDevice>(runtimeDevices.default);
    const [retranscribeMinSpeakers, setRetranscribeMinSpeakers] =
        useState<number | "">("");
    const [retranscribeMaxSpeakers, setRetranscribeMaxSpeakers] =
        useState<number | "">("");
    const [summary, setSummary] = useState("");
    const [summaryModel, setSummaryModel] = useState<SummaryModel>("qwen");
    const [summaryDevice, setSummaryDevice] = useState<RuntimeDevice>(runtimeDevices.default);
    const [isSummarizing, setIsSummarizing] = useState(false);
    const selectedSummaryDevice = runtimeDevices.available.includes(summaryDevice)
        ? summaryDevice
        : runtimeDevices.default;

    if (isLoading && !recording) {
        return <div className="p-6 text-sm text-zinc-500">Loading recording...</div>;
    }

    if (!recording) {
        return <div className="p-6 text-sm text-zinc-500">Select a recording.</div>;
    }

    const segments = recording.transcript?.segments ?? [];
    const canRetranscribe =
        Boolean(onRetranscribe) &&
        (recording.status === "done" || recording.status === "failed");
    const canRenameSpeakers =
        Boolean(onRenameSpeakers) &&
        segments.some((segment) => Boolean(segment.speaker));

    const handleSummarize = async () => {
        if (!segments.length) return;

        setIsSummarizing(true);
        setSummary("");
        try {
            const nextSummary = await summarizeTranscript({
                text: sanitizeTranscript(segments),
                model: summaryModel,
                device: selectedSummaryDevice,
            });
            setSummary(nextSummary);
        } finally {
            setIsSummarizing(false);
        }
    };

    const handleRenameSpeakers = async (speakers: Record<string, string>) => {
        if (!onRenameSpeakers) return;

        await onRenameSpeakers(recording.id, speakers);
        setSummary("");
    };

    const openRetranscribeEditor = () => {
        const recordingDevice = runtimeDevices.available.includes(recording.device)
            ? recording.device
            : runtimeDevices.default;

        setRetranscribeLanguage(recording.language_hint ?? "auto");
        setRetranscribeModel(recording.model);
        setRetranscribeDevice(recordingDevice);
        setRetranscribeMinSpeakers(recording.min_speakers ?? "");
        setRetranscribeMaxSpeakers(recording.max_speakers ?? "");
        setIsRetranscribeEditorOpen(true);
    };

    const closeRetranscribeEditor = () => {
        if (isRetranscribing) return;
        setIsRetranscribeEditorOpen(false);
    };

    const handleRetranscribeSubmit = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        if (!onRetranscribe) return;

        const selectedDevice = runtimeDevices.available.includes(retranscribeDevice)
            ? retranscribeDevice
            : runtimeDevices.default;

        await onRetranscribe(recording.id, {
            language: retranscribeLanguage,
            model: retranscribeModel,
            device: selectedDevice,
            minSpeakers: retranscribeMinSpeakers,
            maxSpeakers: retranscribeMaxSpeakers,
        });
        setSummary("");
        setIsRetranscribeEditorOpen(false);
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
                    <div className="flex flex-col gap-4">
                        <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-800">
                            {recording.error ?? "Transcription failed."}
                        </div>
                        {canRetranscribe && (
                            <div>
                                <button
                                    type="button"
                                    onClick={openRetranscribeEditor}
                                    disabled={isRetranscribing}
                                    className="rounded-md border border-zinc-300 px-3 py-1.5 text-sm font-medium text-zinc-700 transition-colors hover:border-zinc-400 hover:text-zinc-950 disabled:cursor-not-allowed disabled:opacity-60"
                                >
                                    {isRetranscribing ? "Re-transcribing..." : "Re-transcribe"}
                                </button>
                            </div>
                        )}
                    </div>
                )}
                {recording.status === "done" && (
                    <div className="flex flex-col gap-5">
                        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-200">
                            <div className="flex">
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
                            {activeTab === "transcript" && (canRenameSpeakers || canRetranscribe) && (
                                <div className="mb-2 flex flex-wrap items-center gap-2">
                                    {canRenameSpeakers && (
                                        <button
                                            type="button"
                                            onClick={() => setIsSpeakerEditorOpen(true)}
                                            disabled={isRenamingSpeakers}
                                            className="rounded-md border border-zinc-300 px-3 py-1.5 text-sm font-medium text-zinc-700 transition-colors hover:border-zinc-400 hover:text-zinc-950 disabled:cursor-not-allowed disabled:opacity-60"
                                        >
                                            {isRenamingSpeakers ? "Saving speakers..." : "Edit speakers"}
                                        </button>
                                    )}
                                    {canRetranscribe && (
                                        <button
                                            type="button"
                                            onClick={openRetranscribeEditor}
                                            disabled={isRetranscribing}
                                            className="rounded-md border border-zinc-300 px-3 py-1.5 text-sm font-medium text-zinc-700 transition-colors hover:border-zinc-400 hover:text-zinc-950 disabled:cursor-not-allowed disabled:opacity-60"
                                        >
                                            {isRetranscribing ? "Re-transcribing..." : "Re-transcribe"}
                                        </button>
                                    )}
                                </div>
                            )}
                        </div>

                        {activeTab === "transcript" && (
                            <TranscriptPanel
                                segments={segments}
                                isSavingSpeakers={isRenamingSpeakers}
                                onRenameSpeakers={handleRenameSpeakers}
                                isSpeakerEditorOpen={isSpeakerEditorOpen}
                                onSpeakerEditorClose={() => setIsSpeakerEditorOpen(false)}
                            />
                        )}

                        {activeTab === "summary" && (
                            <SummaryPanel
                                summary={summary}
                                isLoading={isSummarizing}
                                selectedModel={summaryModel}
                                onModelChange={setSummaryModel}
                                selectedDevice={selectedSummaryDevice}
                                onDeviceChange={setSummaryDevice}
                                runtimeDevices={runtimeDevices}
                                onSummarize={handleSummarize}
                                canSummarize={segments.length > 0}
                            />
                        )}
                    </div>
                )}
            </div>

            {isRetranscribeEditorOpen && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/35 px-4">
                    <form
                        onSubmit={handleRetranscribeSubmit}
                        className="w-full max-w-2xl rounded-lg bg-white p-5 shadow-xl"
                    >
                        <div className="flex items-start justify-between gap-4">
                            <h3 className="text-base font-semibold text-zinc-950">
                                Re-transcribe settings
                            </h3>
                            <button
                                type="button"
                                onClick={closeRetranscribeEditor}
                                disabled={isRetranscribing}
                                className="text-sm font-medium text-zinc-500 hover:text-zinc-950 disabled:cursor-not-allowed disabled:opacity-50"
                            >
                                Close
                            </button>
                        </div>

                        <div className="mt-5 grid gap-4 md:grid-cols-2">
                            <label className="flex flex-col gap-1">
                                <span className="text-xs font-medium text-zinc-500">
                                    Language
                                </span>
                                <select
                                    value={retranscribeLanguage}
                                    onChange={(event) => setRetranscribeLanguage(event.target.value)}
                                    disabled={isRetranscribing}
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
                                <span className="text-xs font-medium text-zinc-500">
                                    Model
                                </span>
                                <select
                                    value={retranscribeModel}
                                    onChange={(event) => setRetranscribeModel(
                                        event.target.value as TranscriptionModel,
                                    )}
                                    disabled={isRetranscribing}
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
                                <span className="text-xs font-medium text-zinc-500">
                                    Device
                                </span>
                                <select
                                    value={
                                        runtimeDevices.available.includes(retranscribeDevice)
                                            ? retranscribeDevice
                                            : runtimeDevices.default
                                    }
                                    onChange={(event) => setRetranscribeDevice(
                                        event.target.value as RuntimeDevice,
                                    )}
                                    disabled={isRetranscribing}
                                    className="min-h-10 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                                >
                                    {runtimeDevices.available.map((device) => (
                                        <option key={device} value={device}>
                                            {deviceLabel(device)}
                                        </option>
                                    ))}
                                </select>
                            </label>

                            <div className="grid gap-4 sm:grid-cols-2">
                                <label className="flex flex-col gap-1">
                                    <span className="text-xs font-medium text-zinc-500">
                                        Min speakers
                                    </span>
                                    <input
                                        type="number"
                                        min="1"
                                        value={retranscribeMinSpeakers}
                                        onChange={(event) => setRetranscribeMinSpeakers(
                                            numberOrEmpty(event.target.value),
                                        )}
                                        disabled={isRetranscribing}
                                        placeholder="Auto"
                                        className="min-h-10 rounded-md border border-zinc-300 px-3 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                                    />
                                </label>

                                <label className="flex flex-col gap-1">
                                    <span className="text-xs font-medium text-zinc-500">
                                        Max speakers
                                    </span>
                                    <input
                                        type="number"
                                        min="1"
                                        value={retranscribeMaxSpeakers}
                                        onChange={(event) => setRetranscribeMaxSpeakers(
                                            numberOrEmpty(event.target.value),
                                        )}
                                        disabled={isRetranscribing}
                                        placeholder="Auto"
                                        className="min-h-10 rounded-md border border-zinc-300 px-3 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                                    />
                                </label>
                            </div>
                        </div>

                        <div className="mt-6 flex justify-end gap-3">
                            <button
                                type="button"
                                onClick={closeRetranscribeEditor}
                                disabled={isRetranscribing}
                                className="min-h-10 rounded-md border border-zinc-300 px-4 text-sm font-medium text-zinc-700 hover:border-zinc-400 hover:text-zinc-950 disabled:cursor-not-allowed disabled:opacity-50"
                            >
                                Cancel
                            </button>
                            <button
                                type="submit"
                                disabled={isRetranscribing}
                                className="min-h-10 rounded-md bg-zinc-950 px-4 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-50"
                            >
                                {isRetranscribing ? "Starting..." : "Start re-transcribe"}
                            </button>
                        </div>
                    </form>
                </div>
            )}
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
