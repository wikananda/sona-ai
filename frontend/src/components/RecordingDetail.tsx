"use client";

import { memo, useRef, useState } from "react";
import {
    recordingAudioUrl,
    Recording,
    RecordingSummaryParams,
    RecordingSummaryUpdateParams,
    RetranscribeParams,
    SpeakerExtractionParams,
    RuntimeDevice,
    RuntimeDevices,
    LocalLLMModel,
    SummaryMode,
    TranscriptSegmentUpdateParams,
} from "@/src/api/sonaApi";
import { BYOKResolvedModelPreset } from "@/src/hooks/useBYOKSettings";
import { selectBYOKPreset } from "@/src/hooks/useBYOKPreset";
import RecordingStatusBadge from "@/src/components/RecordingStatusBadge";
import RecordingChatPanel from "@/src/components/RecordingChatPanel";
import SummaryPanel from "@/src/components/SummaryPanel";
import TranscriptPanel from "@/src/components/TranscriptPanel";
import RetranscribeEditorModal from "@/src/components/recording-detail/RetranscribeEditorModal";
import SpeakerExtractionEditorModal from "@/src/components/recording-detail/SpeakerExtractionEditorModal";
import TabButton from "@/src/components/recording-detail/TabButton";
import RecordingProgressPanel from "@/src/components/recording-detail/RecordingProgressPanel";

type DetailTab = "transcript" | "summary" | "chat";

interface Props {
    recording?: Recording | null;
    isLoading: boolean;
    runtimeDevices: RuntimeDevices;
    isRetranscribing?: boolean;
    onRetranscribe?: (
        recordingId: string,
        settings: RetranscribeParams,
    ) => Promise<boolean>;
    isCanceling?: boolean;
    onCancel?: (recordingId: string) => Promise<void>;
    isRenamingSpeakers?: boolean;
    onRenameSpeakers?: (
        recordingId: string,
        speakers: Record<string, string>,
    ) => Promise<void>;
    isEditingTranscript?: boolean;
    onUpdateTranscriptSegment?: (
        recordingId: string,
        segmentIndex: number,
        params: TranscriptSegmentUpdateParams,
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
    isUpdatingSummary?: boolean;
    onUpdateSummary?: (
        recordingId: string,
        params: RecordingSummaryUpdateParams,
    ) => Promise<void>;
    byokModelPresets: BYOKResolvedModelPreset[];
    onOpenSettings: () => void;
}

function RecordingDetail({
    recording,
    isLoading,
    runtimeDevices,
    isRetranscribing = false,
    onRetranscribe,
    isCanceling = false,
    onCancel,
    isRenamingSpeakers = false,
    onRenameSpeakers,
    isEditingTranscript = false,
    onUpdateTranscriptSegment,
    isExtractingSpeakers = false,
    onExtractSpeakers,
    isSummarizing = false,
    onSummarize,
    isUpdatingSummary = false,
    onUpdateSummary,
    byokModelPresets,
    onOpenSettings,
}: Props) {
    const audioRef = useRef<HTMLAudioElement | null>(null);
    const [activeTab, setActiveTab] = useState<DetailTab>("transcript");
    const [currentTime, setCurrentTime] = useState(0);
    const [isSpeakerEditorOpen, setIsSpeakerEditorOpen] = useState(false);
    const [isRetranscribeEditorOpen, setIsRetranscribeEditorOpen] = useState(false);
    const [isSpeakerExtractionEditorOpen, setIsSpeakerExtractionEditorOpen] = useState(false);
    const [localLLMModel, setLocalLLMModel] = useState<LocalLLMModel>("qwen");
    const [summaryDevice, setSummaryDevice] = useState<RuntimeDevice>(runtimeDevices.default);
    const [summaryMode, setSummaryMode] = useState<SummaryMode>("local");
    const [summaryBYOKPresetId, setSummaryBYOKPresetId] = useState("");
    const [chatBYOKPresetId, setChatBYOKPresetId] = useState("");
    const {
        effectivePresetId: effectiveSummaryBYOKPresetId,
        selectedSettings: selectedSummaryBYOKSettings,
    } = selectBYOKPreset(byokModelPresets, summaryBYOKPresetId);
    const { effectivePresetId: effectiveChatBYOKPresetId } =
        selectBYOKPreset(byokModelPresets, chatBYOKPresetId);
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
    const isSpeakerExtractionRunning =
        isProcessingRecording &&
        ["diarizing", "assigning_speakers"].includes(recording.progress?.stage ?? "");
    const activeSegmentIndex = segments.findIndex(
        (segment) => currentTime >= segment.start && currentTime < segment.end,
    );
    const canRetranscribe =
        Boolean(onRetranscribe) &&
        (
            recording.status === "done" ||
            recording.status === "failed" ||
            recording.status === "canceled"
        );
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
            byok: summaryMode === "byok" ? selectedSummaryBYOKSettings : undefined,
        });
    };

    const handleUpdateSummary = async (text: string) => {
        if (!onUpdateSummary) return;

        await onUpdateSummary(recording.id, { text });
    };

    const handleRenameSpeakers = async (speakers: Record<string, string>) => {
        if (!onRenameSpeakers) return;

        await onRenameSpeakers(recording.id, speakers);
    };

    const handleUpdateTranscriptSegment = async (
        segmentIndex: number,
        params: TranscriptSegmentUpdateParams,
    ) => {
        if (!onUpdateTranscriptSegment) return;

        await onUpdateTranscriptSegment(recording.id, segmentIndex, params);
    };

    const handleSeekToSegment = async (start: number) => {
        const audio = audioRef.current;
        if (!audio) return;

        audio.currentTime = start;
        await audio.play().catch(() => undefined);
    };

    const handleRetranscribe = async (
        settings: RetranscribeParams,
    ): Promise<boolean> => {
        if (!onRetranscribe) return false;

        return onRetranscribe(recording.id, settings);
    };

    const handleExtractSpeakers = async (settings: SpeakerExtractionParams) => {
        if (!onExtractSpeakers) return;

        await onExtractSpeakers(recording.id, settings);
    };

    const handleCancelProcessing = async () => {
        if (!onCancel) return;
        await onCancel(recording.id);
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
                <div className="flex flex-col gap-5">
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

                    {recording.status === "pending" && !hasTranscript && (
                        <RecordingProgressPanel
                            recording={recording}
                            isCanceling={isCanceling}
                            onCancel={onCancel ? handleCancelProcessing : undefined}
                        />
                    )}
                    {recording.status === "processing" && !hasTranscript && (
                        <RecordingProgressPanel
                            recording={recording}
                            isCanceling={isCanceling}
                            onCancel={onCancel ? handleCancelProcessing : undefined}
                        />
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
                                        onClick={() => setIsRetranscribeEditorOpen(true)}
                                        disabled={isRetranscribing}
                                        className="rounded-md border border-zinc-300 px-3 py-1.5 text-sm font-medium text-zinc-700 transition-colors hover:cursor-pointer hover:border-zinc-400 hover:text-zinc-950 disabled:cursor-not-allowed disabled:opacity-60"
                                    >
                                        {isRetranscribing ? "Re-transcribing..." : "Re-transcribe"}
                                    </button>
                                </div>
                            )}
                        </div>
                    )}
                    {recording.status === "canceled" && !hasTranscript && (
                        <div className="flex flex-col gap-4">
                            <div className="rounded-md border border-zinc-200 bg-zinc-50 p-4 text-sm text-zinc-700">
                                Transcription was canceled.
                            </div>
                            {canRetranscribe && (
                                <div>
                                    <button
                                        type="button"
                                        onClick={() => setIsRetranscribeEditorOpen(true)}
                                        disabled={isRetranscribing}
                                        className="rounded-md border border-zinc-300 px-3 py-1.5 text-sm font-medium text-zinc-700 transition-colors hover:cursor-pointer hover:border-zinc-400 hover:text-zinc-950 disabled:cursor-not-allowed disabled:opacity-60"
                                    >
                                        {isRetranscribing ? "Re-transcribing..." : "Re-transcribe"}
                                    </button>
                                </div>
                            )}
                        </div>
                    )}
                    {(recording.status === "done" || hasTranscript) && (
                        <>
                            {isProcessingRecording && (
                                <RecordingProgressPanel
                                    recording={recording}
                                    helperText="You can keep reviewing the current transcript while processing finishes."
                                    isCanceling={isCanceling}
                                    onCancel={onCancel ? handleCancelProcessing : undefined}
                                />
                            )}
                            {recording.status === "failed" && hasTranscript && (
                                <div className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-800">
                                    {recording.error ?? "Processing failed, but the transcript is still available."}
                                </div>
                            )}
                            {recording.status === "canceled" && hasTranscript && (
                                <div className="rounded-md border border-zinc-200 bg-zinc-50 p-3 text-sm text-zinc-700">
                                    Processing was canceled, but the previous transcript is still available.
                                </div>
                            )}

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
                                                    className="rounded-md border border-zinc-300 px-3 py-1.5 text-sm font-medium text-zinc-700 transition-colors hover:border-zinc-400 hover:text-zinc-950 hover:cursor-pointer disabled:cursor-not-allowed disabled:opacity-60"
                                                >
                                                    {isRenamingSpeakers ? "Saving speakers..." : "Edit speakers"}
                                                </button>
                                            )}
                                            {canExtractSpeakers && (
                                                <button
                                                    type="button"
                                                    onClick={() => setIsSpeakerExtractionEditorOpen(true)}
                                                    disabled={isExtractingSpeakers}
                                                    className="rounded-md border border-zinc-300 px-3 py-1.5 text-sm font-medium text-zinc-700 transition-colors hover:cursor-pointer hover:border-zinc-400 hover:text-zinc-950 disabled:cursor-not-allowed disabled:opacity-60"
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
                                                    onClick={() => setIsRetranscribeEditorOpen(true)}
                                                    disabled={isRetranscribing}
                                                    className="rounded-md border border-zinc-300 px-3 py-1.5 text-sm font-medium text-zinc-700 transition-colors hover:cursor-pointer hover:border-zinc-400 hover:text-zinc-950 disabled:cursor-not-allowed disabled:opacity-60"
                                                >
                                                    {isRetranscribing ? "Re-transcribing..." : "Re-transcribe"}
                                                </button>
                                            )}
                                        </div>
                                    )}
                            </div>

                            {activeTab === "transcript" && (
                                <TranscriptPanel
                                    recordingName={recording.original_name}
                                    segments={segments}
                                    isSavingSpeakers={isRenamingSpeakers}
                                    onRenameSpeakers={handleRenameSpeakers}
                                    isSavingSegment={isEditingTranscript}
                                    onUpdateSegment={handleUpdateTranscriptSegment}
                                    isSpeakerEditorOpen={isSpeakerEditorOpen}
                                    onSpeakerEditorClose={() => setIsSpeakerEditorOpen(false)}
                                    activeSegmentIndex={activeSegmentIndex}
                                    onSeekToSegment={handleSeekToSegment}
                                />
                            )}

                            {activeTab === "summary" && (
                                <SummaryPanel
                                    recordingName={recording.original_name}
                                    summary={summary}
                                    isLoading={isSummarizing}
                                    selectedModel={localLLMModel}
                                    onModelChange={setLocalLLMModel}
                                    selectedDevice={selectedSummaryDevice}
                                    onDeviceChange={setSummaryDevice}
                                    runtimeDevices={runtimeDevices}
                                    selectedMode={summaryMode}
                                    onModeChange={setSummaryMode}
                                    byokModelPresets={byokModelPresets}
                                    selectedBYOKPresetId={effectiveSummaryBYOKPresetId}
                                    onBYOKPresetChange={setSummaryBYOKPresetId}
                                    onOpenSettings={onOpenSettings}
                                    customInstruction={summaryInstruction}
                                    onCustomInstructionChange={setSummaryInstruction}
                                    onSummarize={handleSummarize}
                                    isSavingSummary={isUpdatingSummary}
                                    onUpdateSummary={handleUpdateSummary}
                                    canSummarize={
                                        Boolean(onSummarize) &&
                                        segments.length > 0 &&
                                        !isProcessingRecording &&
                                        (summaryMode === "local" || Boolean(selectedSummaryBYOKSettings))
                                    }
                                />
                            )}
                            {activeTab === "chat" && (
                                <RecordingChatPanel
                                    recordingId={recording.id}
                                    canChat={segments.length > 0}
                                    byokModelPresets={byokModelPresets}
                                    selectedBYOKPresetId={effectiveChatBYOKPresetId}
                                    onBYOKPresetChange={setChatBYOKPresetId}
                                    onOpenSettings={onOpenSettings}
                                />
                            )}
                        </>
                    )}
                </div>
            </div>

            {isRetranscribeEditorOpen && (
                <RetranscribeEditorModal
                    recording={recording}
                    runtimeDevices={runtimeDevices}
                    isRetranscribing={isRetranscribing}
                    onClose={() => setIsRetranscribeEditorOpen(false)}
                    onSubmit={handleRetranscribe}
                />
            )}

            {isSpeakerExtractionEditorOpen && (
                <SpeakerExtractionEditorModal
                    recording={recording}
                    isExtractingSpeakers={isExtractingSpeakers}
                    onClose={() => setIsSpeakerExtractionEditorOpen(false)}
                    onSubmit={handleExtractSpeakers}
                />
            )}
        </section>
    );
}

export default memo(RecordingDetail);
