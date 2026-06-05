"use client";

import { FormEvent, useRef, useState } from "react";
import {
    recordingAudioUrl,
    Recording,
    RecordingSummaryParams,
    RetranscribeParams,
    SpeakerExtractionParams,
    RuntimeDevice,
    RuntimeDevices,
    LocalLLMModel,
    TranscriptionModel,
    SummaryMode,
    BYOKProvider,
} from "@/src/api/sonaApi";
import RecordingStatusBadge from "@/src/components/RecordingStatusBadge";
import RecordingChatPanel from "@/src/components/RecordingChatPanel";
import SummaryPanel from "@/src/components/SummaryPanel";
import TranscriptPanel from "@/src/components/TranscriptPanel";
import {
    deviceLabel,
    isTranscriptionModel,
    numberOrEmpty,
    TRANSCRIPTION_LANGUAGES,
    TRANSCRIPTION_MODELS,
} from "@/src/utils/transcriptionSettings";

type DetailTab = "transcript" | "summary" | "chat";

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
    isExtractingSpeakers?: boolean;
    onExtractSpeakers?: (
        recordingId: string,
        settings: SpeakerExtractionParams,
    ) => Promise<void>;
    isSummarizing?: boolean;
    onSummarize?: (
        recordingId: string,
        settings: RecordingSummaryParams,
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
    isExtractingSpeakers = false,
    onExtractSpeakers,
    isSummarizing = false,
    onSummarize,
}: Props) {
    const audioRef = useRef<HTMLAudioElement | null>(null);
    const [activeTab, setActiveTab] = useState<DetailTab>("transcript");
    const [currentTime, setCurrentTime] = useState(0);
    const [isSpeakerEditorOpen, setIsSpeakerEditorOpen] = useState(false);
    const [isRetranscribeEditorOpen, setIsRetranscribeEditorOpen] = useState(false);
    const [isSpeakerExtractionEditorOpen, setIsSpeakerExtractionEditorOpen] = useState(false);
    const [retranscribeLanguage, setRetranscribeLanguage] = useState("auto");
    const [retranscribeModel, setRetranscribeModel] =
        useState<TranscriptionModel>("parakeet");
    const [retranscribeDevice, setRetranscribeDevice] =
        useState<RuntimeDevice>(runtimeDevices.default);
    const [retranscribeMinSpeakers, setRetranscribeMinSpeakers] =
        useState<number | "">("");
    const [retranscribeMaxSpeakers, setRetranscribeMaxSpeakers] =
        useState<number | "">("");
    const [retranscribeExtractSpeakers, setRetranscribeExtractSpeakers] = useState(true);
    const [extractMinSpeakers, setExtractMinSpeakers] = useState<number | "">("");
    const [extractMaxSpeakers, setExtractMaxSpeakers] = useState<number | "">("");
    const [retranscribeError, setRetranscribeError] = useState("");
    const [speakerExtractionError, setSpeakerExtractionError] = useState("");
    const [localLLMModel, setLocalLLMModel] = useState<LocalLLMModel>("qwen");
    const [summaryDevice, setSummaryDevice] = useState<RuntimeDevice>(runtimeDevices.default);
    const [summaryMode, setSummaryMode] = useState<SummaryMode>("local");
    const [byokProvider, setBYOKProvider] = useState<BYOKProvider>("openai");
    const [byokApiKey, setBYOKApiKey] = useState("");
    const [byokModel, setBYOKModel] = useState("gpt-4o-mini");
    const [byokBaseUrl, setBYOKBaseUrl] = useState("");
    const selectedSummaryDevice = runtimeDevices.available.includes(summaryDevice)
        ? summaryDevice
        : runtimeDevices.default;
    const [summaryInstruction, setSummaryInstruction] = useState("");

    if (isLoading && !recording) {
        return <div className="p-6 text-sm text-zinc-500">Loading recording...</div>;
    }

    if (!recording) {
        return <div className="p-6 text-sm text-zinc-500">Select a recording.</div>;
    }

    const segments = recording.transcript?.segments ?? [];
    const summary = recording.summary?.text ?? "";
    const hasTranscript = segments.length > 0;
    const hasDiarization = Boolean(recording.transcript?.diarization_engine);
    const isProcessingRecording =
        recording.status === "pending" || recording.status === "processing";
    const isSpeakerExtractionRunning = isProcessingRecording && hasTranscript && !hasDiarization;
    const activeSegmentIndex = segments.findIndex(
        (segment) => currentTime >= segment.start && currentTime < segment.end,
    );
    const canRetranscribe =
        Boolean(onRetranscribe) &&
        (recording.status === "done" || recording.status === "failed");
    const canRenameSpeakers =
        Boolean(onRenameSpeakers) &&
        !isProcessingRecording &&
        segments.some((segment) => Boolean(segment.speaker));
    const canExtractSpeakers =
        Boolean(onExtractSpeakers) &&
        hasTranscript &&
        !hasDiarization &&
        !isProcessingRecording;

    const handleSummarize = async () => {
        if (!onSummarize || !segments.length) return;

        await onSummarize(recording.id, {
            model: localLLMModel,
            device: selectedSummaryDevice,
            mode: summaryMode,
            prompt: summaryInstruction.trim() || undefined,
            byok: summaryMode === "byok" ? {
                provider: byokProvider,
                apiKey: byokApiKey,
                model: byokModel,
                baseUrl: byokBaseUrl,
            } : undefined,
        });
    };

    const handleRenameSpeakers = async (speakers: Record<string, string>) => {
        if (!onRenameSpeakers) return;

        await onRenameSpeakers(recording.id, speakers);
    };

    const handleSeekToSegment = async (start: number) => {
        const audio = audioRef.current;
        if (!audio) return;

        audio.currentTime = start;
        await audio.play().catch(() => undefined);
    };

    const openRetranscribeEditor = () => {
        const recordingDevice = runtimeDevices.available.includes(recording.device)
            ? recording.device
            : runtimeDevices.default;

        setRetranscribeLanguage(recording.language_hint ?? "auto");
        setRetranscribeModel(
            isTranscriptionModel(recording.model) ? recording.model : "parakeet",
        );
        setRetranscribeDevice(recordingDevice);
        setRetranscribeMinSpeakers(recording.min_speakers ?? "");
        setRetranscribeMaxSpeakers(recording.max_speakers ?? "");
        setRetranscribeExtractSpeakers(true);
        setRetranscribeError("");
        setIsRetranscribeEditorOpen(true);
    };

    const closeRetranscribeEditor = () => {
        if (isRetranscribing) return;
        setRetranscribeError("");
        setIsRetranscribeEditorOpen(false);
    };

    const openSpeakerExtractionEditor = () => {
        setExtractMinSpeakers(recording.min_speakers ?? "");
        setExtractMaxSpeakers(recording.max_speakers ?? "");
        setSpeakerExtractionError("");
        setIsSpeakerExtractionEditorOpen(true);
    };

    const closeSpeakerExtractionEditor = () => {
        if (isExtractingSpeakers) return;
        setSpeakerExtractionError("");
        setIsSpeakerExtractionEditorOpen(false);
    };

    const handleRetranscribeSubmit = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        if (!onRetranscribe) return;

        const selectedDevice = runtimeDevices.available.includes(retranscribeDevice)
            ? retranscribeDevice
            : runtimeDevices.default;
        if (
            retranscribeExtractSpeakers &&
            (retranscribeMinSpeakers === "" || retranscribeMaxSpeakers === "")
        ) {
            setRetranscribeError("Min and max speakers are required when extracting speakers.");
            return;
        }

        setRetranscribeError("");
        await onRetranscribe(recording.id, {
            language: retranscribeLanguage,
            model: retranscribeModel,
            device: selectedDevice,
            minSpeakers: retranscribeExtractSpeakers ? retranscribeMinSpeakers : "",
            maxSpeakers: retranscribeExtractSpeakers ? retranscribeMaxSpeakers : "",
            extractSpeakers: retranscribeExtractSpeakers,
        });
        setIsRetranscribeEditorOpen(false);
    };

    const handleSpeakerExtractionSubmit = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        if (!onExtractSpeakers) return;
        if (extractMinSpeakers === "" || extractMaxSpeakers === "") {
            setSpeakerExtractionError("Min and max speakers are required.");
            return;
        }

        setSpeakerExtractionError("");
        await onExtractSpeakers(recording.id, {
            minSpeakers: extractMinSpeakers,
            maxSpeakers: extractMaxSpeakers,
        });
        setIsSpeakerExtractionEditorOpen(false);
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
                {recording.status === "pending" && !hasTranscript && (
                    <p className="text-sm text-zinc-500">Waiting to start transcription.</p>
                )}
                {recording.status === "processing" && !hasTranscript && (
                    <p className="text-sm text-zinc-500">Transcription is running.</p>
                )}
                {recording.status === "failed" && !hasTranscript && (
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
                {(recording.status === "done" || hasTranscript) && (
                    <div className="flex flex-col gap-5">
                        {isSpeakerExtractionRunning && (
                            <div className="rounded-md border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
                                Speaker extraction is running. You can keep reviewing the transcript while diarization finishes.
                            </div>
                        )}
                        {recording.status === "failed" && hasTranscript && (
                            <div className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-800">
                                {recording.error ?? "Processing failed, but the transcript is still available."}
                            </div>
                        )}
                        <audio
                            key={recording.id}
                            ref={audioRef}
                            controls
                            src={recordingAudioUrl(recording.id)}
                            onTimeUpdate={(event) => {
                                setCurrentTime(event.currentTarget.currentTime);
                            }}
                            onLoadedMetadata={() => setCurrentTime(0)}
                            className="w-full"
                        />

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
                                <TabButton
                                    label="Chat"
                                    isActive={activeTab === "chat"}
                                    onClick={() => setActiveTab("chat")}
                                />
                            </div>
                            {activeTab === "transcript" && (
                                canRenameSpeakers ||
                                canRetranscribe ||
                                canExtractSpeakers ||
                                isSpeakerExtractionRunning
                            ) && (
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
                                    {canExtractSpeakers && (
                                        <button
                                            type="button"
                                            onClick={openSpeakerExtractionEditor}
                                            disabled={isExtractingSpeakers}
                                            className="rounded-md border border-zinc-300 px-3 py-1.5 text-sm font-medium text-zinc-700 transition-colors hover:border-zinc-400 hover:text-zinc-950 disabled:cursor-not-allowed disabled:opacity-60"
                                        >
                                            {isExtractingSpeakers ? "Extracting speakers..." : "Extract speakers"}
                                        </button>
                                    )}
                                    {isSpeakerExtractionRunning && (
                                        <span className="text-sm text-amber-700">
                                            Extracting speakers...
                                        </span>
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
                                activeSegmentIndex={activeSegmentIndex}
                                onSeekToSegment={handleSeekToSegment}
                            />
                        )}

                        {activeTab === "summary" && (
                            <SummaryPanel
                                summary={summary}
                                isLoading={isSummarizing}
                                selectedModel={localLLMModel}
                                onModelChange={setLocalLLMModel}
                                selectedDevice={selectedSummaryDevice}
                                onDeviceChange={setSummaryDevice}
                                runtimeDevices={runtimeDevices}
                                selectedMode={summaryMode}
                                onModeChange={setSummaryMode}
                                byokProvider={byokProvider}
                                onBYOKProviderChange={setBYOKProvider}
                                byokApiKey={byokApiKey}
                                onBYOKApiKeyChange={setBYOKApiKey}
                                byokModel={byokModel}
                                onBYOKModelChange={setBYOKModel}
                                byokBaseUrl={byokBaseUrl}
                                onBYOKBaseUrlChange={setBYOKBaseUrl}
                                customInstruction={summaryInstruction}
                                onCustomInstructionChange={setSummaryInstruction}
                                formatName={recording.summary?.format_name}
                                onSummarize={handleSummarize}
                                canSummarize={
                                    Boolean(onSummarize) &&
                                    segments.length > 0 &&
                                    !isProcessingRecording
                                }
                            />
                        )}
                        {activeTab === "chat" && (
                            <RecordingChatPanel
                                recordingId={recording.id}
                                canChat={segments.length > 0}
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

                            <label className="flex items-start gap-2 md:col-span-2">
                                <input
                                    type="checkbox"
                                    checked={retranscribeExtractSpeakers}
                                    onChange={(event) => {
                                        setRetranscribeExtractSpeakers(event.target.checked);
                                        setRetranscribeError("");
                                    }}
                                    disabled={isRetranscribing}
                                    className="mt-0.5 h-4 w-4 rounded border-zinc-300 text-zinc-950 focus:ring-zinc-950 disabled:cursor-not-allowed disabled:opacity-50"
                                />
                                <span className="text-sm text-zinc-600">
                                    Extract speakers during re-transcribe
                                    <span className="block text-xs text-zinc-500">
                                        Uncheck this to re-run ASR now and extract speakers later.
                                    </span>
                                </span>
                            </label>

                            {retranscribeExtractSpeakers && (
                                <div className="grid gap-4 sm:grid-cols-2 md:col-span-2">
                                    <label className="flex flex-col gap-1">
                                        <span className="text-xs font-medium text-zinc-500">
                                            Min speakers
                                        </span>
                                        <input
                                            type="number"
                                            min="1"
                                            value={retranscribeMinSpeakers}
                                            onChange={(event) => {
                                                setRetranscribeMinSpeakers(
                                                    numberOrEmpty(event.target.value),
                                                );
                                                setRetranscribeError("");
                                            }}
                                            disabled={isRetranscribing}
                                            placeholder="Required"
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
                                            onChange={(event) => {
                                                setRetranscribeMaxSpeakers(
                                                    numberOrEmpty(event.target.value),
                                                );
                                                setRetranscribeError("");
                                            }}
                                            disabled={isRetranscribing}
                                            placeholder="Required"
                                            className="min-h-10 rounded-md border border-zinc-300 px-3 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                                        />
                                    </label>
                                </div>
                            )}
                        </div>

                        {retranscribeError && (
                            <p className="mt-4 text-sm text-red-700">{retranscribeError}</p>
                        )}

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

            {isSpeakerExtractionEditorOpen && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/35 px-4">
                    <form
                        onSubmit={handleSpeakerExtractionSubmit}
                        className="w-full max-w-md rounded-lg bg-white p-5 shadow-xl"
                    >
                        <div className="flex items-start justify-between gap-4">
                            <div>
                                <h3 className="text-base font-semibold text-zinc-950">
                                    Extract speaker settings
                                </h3>
                                <p className="mt-1 text-sm text-zinc-500">
                                    Set speaker bounds for diarization.
                                </p>
                            </div>
                            <button
                                type="button"
                                onClick={closeSpeakerExtractionEditor}
                                disabled={isExtractingSpeakers}
                                className="text-sm font-medium text-zinc-500 hover:text-zinc-950 disabled:cursor-not-allowed disabled:opacity-50"
                            >
                                Close
                            </button>
                        </div>

                        <div className="mt-5 grid gap-4 sm:grid-cols-2">
                            <label className="flex flex-col gap-1">
                                <span className="text-xs font-medium text-zinc-500">
                                    Min speakers
                                </span>
                                <input
                                    type="number"
                                    min="1"
                                    value={extractMinSpeakers}
                                    onChange={(event) => {
                                        setExtractMinSpeakers(
                                            numberOrEmpty(event.target.value),
                                        );
                                        setSpeakerExtractionError("");
                                    }}
                                    disabled={isExtractingSpeakers}
                                    placeholder="Required"
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
                                    value={extractMaxSpeakers}
                                    onChange={(event) => {
                                        setExtractMaxSpeakers(
                                            numberOrEmpty(event.target.value),
                                        );
                                        setSpeakerExtractionError("");
                                    }}
                                    disabled={isExtractingSpeakers}
                                    placeholder="Required"
                                    className="min-h-10 rounded-md border border-zinc-300 px-3 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                                />
                            </label>
                        </div>

                        {speakerExtractionError && (
                            <p className="mt-4 text-sm text-red-700">{speakerExtractionError}</p>
                        )}

                        <div className="mt-6 flex justify-end gap-3">
                            <button
                                type="button"
                                onClick={closeSpeakerExtractionEditor}
                                disabled={isExtractingSpeakers}
                                className="min-h-10 rounded-md border border-zinc-300 px-4 text-sm font-medium text-zinc-700 hover:border-zinc-400 hover:text-zinc-950 disabled:cursor-not-allowed disabled:opacity-50"
                            >
                                Cancel
                            </button>
                            <button
                                type="submit"
                                disabled={isExtractingSpeakers}
                                className="min-h-10 rounded-md bg-zinc-950 px-4 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-50"
                            >
                                {isExtractingSpeakers ? "Starting..." : "Start extraction"}
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
            className={`min-h-11 border-b-2 px-4 text-sm font-medium transition-colors ${isActive
                ? "border-zinc-950 text-zinc-950"
                : "border-transparent text-zinc-500 hover:text-zinc-950"
                }`}
        >
            {label}
        </button>
    );
}
