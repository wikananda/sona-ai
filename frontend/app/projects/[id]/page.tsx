"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
    cancelRecording,
    deleteRecording,
    extractRecordingSpeakers,
    getProject,
    getRecording,
    preflightTranscriptionModels,
    getRuntimeDevices,
    RuntimeModel,
    Project,
    Recording,
    renameRecording,
    RecordingSummaryParams,
    RecordingSummaryUpdateParams,
    RetranscribeParams,
    SpeakerExtractionParams,
    RuntimeDevice,
    RuntimeDevices,
    renameTranscriptSpeakers,
    retranscribeRecording,
    summarizeRecording,
    TranscriptionModel,
    TranscriptSegmentUpdateParams,
    uploadProjectRecording,
    updateTranscriptSegment,
    updateRecordingSummary,
} from "@/src/api/sonaApi";
import AddRecordingPanel, { AddRecordingMode } from "@/src/components/AddRecordingPanel";
import BYOKSettingsModal from "@/src/components/BYOKSettingsModal";
import MissingSpeechModelsModal from "@/src/components/MissingSpeechModelsModal";
import RecordingDetail from "@/src/components/RecordingDetail";
import RecordingSidebar from "@/src/components/RecordingSidebar";
import { useBYOKSettings } from "@/src/hooks/useBYOKSettings";
import { usePolling } from "@/src/hooks/usePolling";
import { getErrorMessage } from "@/src/utils/errorHandling";

const DRAFT_RECORDING_ID = "draft-recording";

interface PendingSpeechModelGate {
    models: RuntimeModel[];
    resolve: (ready: boolean) => void;
}

export default function ProjectDetailPage() {
    const params = useParams<{ id: string }>();
    const projectId = params.id;
    const byokSettings = useBYOKSettings();

    const [project, setProject] = useState<Project | null>(null);
    const [selectedRecordingId, setSelectedRecordingId] = useState<string>();
    const [selectedRecording, setSelectedRecording] = useState<Recording | null>(null);
    const [draftMode, setDraftMode] = useState<AddRecordingMode | null>(null);
    const [previousRecordingId, setPreviousRecordingId] = useState<string>();
    const [isLoadingProject, setIsLoadingProject] = useState(true);
    const [isLoadingRecording, setIsLoadingRecording] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [isSettingsOpen, setIsSettingsOpen] = useState(false);
    const [renamingRecordingId, setRenamingRecordingId] = useState<string>();
    const [retranscribingId, setRetranscribingId] = useState<string>();
    const [cancelingRecordingId, setCancelingRecordingId] = useState<string>();
    const [renamingSpeakerId, setRenamingSpeakerId] = useState<string>();
    const [editingTranscriptId, setEditingTranscriptId] = useState<string>();
    const [extractingSpeakerId, setExtractingSpeakerId] = useState<string>();
    const [summarizingId, setSummarizingId] = useState<string>();
    const [updatingSummaryId, setUpdatingSummaryId] = useState<string>();
    const [runtimeDevices, setRuntimeDevices] = useState<RuntimeDevices>({
        default: "auto",
        available: ["auto", "cpu"],
        torch: { cuda: false, mps: false },
    });
    const [pendingSpeechModelGate, setPendingSpeechModelGate] =
        useState<PendingSpeechModelGate | null>(null);
    const [error, setError] = useState("");

    const recordings = useMemo(() => project?.recordings ?? [], [project]);
    const hasDraft = draftMode !== null;
    const isDraftSelected = selectedRecordingId === DRAFT_RECORDING_ID;
    const hasActiveRecordings = recordings.some((recording) =>
        recording.status === "pending" || recording.status === "processing"
    );

    const refreshProject = useCallback(async () => {
        const data = await getProject(projectId);
        setProject(data);
        setSelectedRecordingId((current) => {
            if (current === DRAFT_RECORDING_ID) {
                return current;
            }
            if (current && data.recordings?.some((recording) => recording.id === current)) {
                return current;
            }
            return data.recordings?.[0]?.id;
        });
    }, [projectId]);

    const refreshSelectedRecording = useCallback(async () => {
        if (!selectedRecordingId || selectedRecordingId === DRAFT_RECORDING_ID) {
            setSelectedRecording(null);
            return;
        }

        setIsLoadingRecording(true);
        try {
            const data = await getRecording(selectedRecordingId);
            setSelectedRecording(data);
        } finally {
            setIsLoadingRecording(false);
        }
    }, [selectedRecordingId]);

    useEffect(() => {
        refreshProject()
            .catch((err) => setError(err.message))
            .finally(() => setIsLoadingProject(false));
    }, [refreshProject]);

    useEffect(() => {
        getRuntimeDevices()
            .then(setRuntimeDevices)
            .catch((err) => setError(err.message));
    }, []);

    useEffect(() => {
        refreshSelectedRecording().catch((err) => setError(err.message));
    }, [refreshSelectedRecording]);

    usePolling(() => {
        refreshProject().catch((err) => setError(err.message));
        refreshSelectedRecording().catch((err) => setError(err.message));
    }, 3000, hasActiveRecordings);

    const handleStartDraft = useCallback(() => {
        setDraftMode((current) => current ?? "upload");
        setPreviousRecordingId((current) => {
            if (draftMode) return current;
            return selectedRecordingId !== DRAFT_RECORDING_ID
                ? selectedRecordingId
                : current;
        });
        setSelectedRecording(null);
        setSelectedRecordingId(DRAFT_RECORDING_ID);
    }, [draftMode, selectedRecordingId]);

    const handleSelectDraft = useCallback(() => {
        if (!draftMode) return;

        setSelectedRecording(null);
        setSelectedRecordingId(DRAFT_RECORDING_ID);
    }, [draftMode]);

    const handleCancelDraft = useCallback(() => {
        const fallbackRecordingId = previousRecordingId &&
            recordings.some((recording) => recording.id === previousRecordingId)
            ? previousRecordingId
            : recordings[0]?.id;

        setDraftMode(null);
        setPreviousRecordingId(undefined);
        setSelectedRecordingId(fallbackRecordingId);
        if (!recordings.length) {
            setSelectedRecording(null);
        }
    }, [previousRecordingId, recordings]);

    const handleSelectRecording = useCallback((recordingId: string) => {
        setDraftMode(null);
        setPreviousRecordingId(undefined);
        setSelectedRecordingId(recordingId);
    }, []);

    const ensureSpeechModelsReady = useCallback(async (params: {
        model: TranscriptionModel;
        device: RuntimeDevice;
        language?: string;
        extractSpeakers?: boolean;
        alignmentEnabled?: boolean;
    }): Promise<boolean> => {
        const result = await preflightTranscriptionModels({
            model: params.model,
            device: params.device,
            language: params.language,
            extractSpeakers: params.extractSpeakers,
            alignmentEnabled: params.alignmentEnabled,
        });
        if (result.missing_model_ids.length === 0) {
            return true;
        }

        const missingModels = result.required_models.filter((model) =>
            result.missing_model_ids.includes(model.id),
        );

        return new Promise<boolean>((resolve) => {
            setPendingSpeechModelGate({
                models: missingModels,
                resolve,
            });
        });
    }, []);

    const handleUpload = useCallback(async (params: {
        files: File[];
        language?: string;
        model: TranscriptionModel;
        device: RuntimeDevice;
        minSpeakers?: number | "";
        maxSpeakers?: number | "";
        extractSpeakers?: boolean;
    }): Promise<boolean> => {
        let modelsReady: boolean;
        try {
            modelsReady = await ensureSpeechModelsReady({
                model: params.model,
                device: params.device,
                language: params.language,
                extractSpeakers: params.extractSpeakers,
            });
        } catch (err) {
            setError(getErrorMessage(err, "Failed to check model availability"));
            throw err;
        }
        if (!modelsReady) {
            return false;
        }

        setIsUploading(true);
        setError("");
        try {
            let firstRecording: Recording | undefined;
            for (const file of params.files) {
                const recording = await uploadProjectRecording({
                    projectId,
                    file,
                    language: params.language,
                    model: params.model,
                    device: params.device,
                    minSpeakers: params.minSpeakers,
                    maxSpeakers: params.maxSpeakers,
                    extractSpeakers: params.extractSpeakers,
                });
                firstRecording = firstRecording ?? recording;
            }
            await refreshProject();
            if (firstRecording) {
                setDraftMode(null);
                setPreviousRecordingId(undefined);
                setSelectedRecordingId(firstRecording.id);
            }
            return true;
        } catch (err) {
            setError(getErrorMessage(err, "Failed to upload recording"));
            throw err;
        } finally {
            setIsUploading(false);
        }
    }, [ensureSpeechModelsReady, projectId, refreshProject]);

    const handleLiveRecordingSaved = useCallback(async (recording: Recording) => {
        setDraftMode(null);
        setPreviousRecordingId(undefined);
        setSelectedRecordingId(recording.id);
        setSelectedRecording(recording);
        await refreshProject();
    }, [refreshProject]);

    const handleDeleteRecording = useCallback(async (recordingId: string) => {
        if (!window.confirm("Delete this recording and transcript?")) return;

        setError("");
        try {
            await deleteRecording(recordingId);
            if (selectedRecordingId === recordingId) {
                setSelectedRecording(null);
                setSelectedRecordingId(undefined);
            }
            await refreshProject();
        } catch (err) {
            setError(getErrorMessage(err, "Failed to delete recording"));
        }
    }, [selectedRecordingId, refreshProject]);

    const handleRenameRecording = useCallback(async (recordingId: string, name: string) => {
        setError("");
        setRenamingRecordingId(recordingId);
        try {
            const updated = await renameRecording(recordingId, name);
            setProject((current) => {
                if (!current?.recordings) return current;
                return {
                    ...current,
                    recordings: current.recordings.map((recording) =>
                        recording.id === recordingId ? { ...recording, ...updated } : recording,
                    ),
                };
            });

            setSelectedRecording((current) =>
                current?.id === recordingId ? { ...current, ...updated } : current,
            );
        } catch (err) {
            setError(getErrorMessage(err, "Failed to rename recording"));
            throw err;
        } finally {
            setRenamingRecordingId(undefined);
        }
    }, []);

    const handleRetranscribeRecording = useCallback(async (
        recordingId: string,
        settings: RetranscribeParams,
    ): Promise<boolean> => {
        let modelsReady: boolean;
        try {
            modelsReady = await ensureSpeechModelsReady({
                model: settings.model,
                device: settings.device,
                language: settings.language,
                extractSpeakers: settings.extractSpeakers,
            });
        } catch (err) {
            setError(getErrorMessage(err, "Failed to check model availability"));
            throw err;
        }
        if (!modelsReady) {
            return false;
        }

        setError("");
        setRetranscribingId(recordingId);
        try {
            const recording = await retranscribeRecording(recordingId, settings);
            setSelectedRecording(recording);
            await refreshProject();
            return true;
        } catch (err) {
            setError(getErrorMessage(err, "Failed to re-transcribe recording"));
            throw err;
        } finally {
            setRetranscribingId(undefined);
        }
    }, [ensureSpeechModelsReady, refreshProject]);

    const handleLiveStartRequest = useCallback(async (params: {
        model: TranscriptionModel;
        device: RuntimeDevice;
        language?: string;
    }): Promise<boolean> => {
        try {
            return await ensureSpeechModelsReady({
                model: params.model,
                device: params.device,
                language: params.language,
                extractSpeakers: false,
                alignmentEnabled: false,
            });
        } catch (err) {
            setError(getErrorMessage(err, "Failed to check model availability"));
            throw err;
        }
    }, [ensureSpeechModelsReady]);

    const handleCancelRecording = useCallback(async (recordingId: string) => {
        setError("");
        setCancelingRecordingId(recordingId);
        try {
            const recording = await cancelRecording(recordingId);
            setSelectedRecording((current) =>
                current?.id === recordingId ? recording : current,
            );
            await refreshProject();
        } catch (err) {
            setError(getErrorMessage(err, "Failed to cancel recording"));
            throw err;
        } finally {
            setCancelingRecordingId(undefined);
        }
    }, [refreshProject]);

    const handleRenameTranscriptSpeakers = useCallback(async (
        recordingId: string,
        speakers: Record<string, string>,
    ) => {
        setError("");
        setRenamingSpeakerId(recordingId);
        try {
            const recording = await renameTranscriptSpeakers(recordingId, speakers);
            setSelectedRecording(recording);
        } catch (err) {
            setError(getErrorMessage(err, "Failed to rename speakers"));
            throw err;
        } finally {
            setRenamingSpeakerId(undefined);
        }
    }, []);

    const handleUpdateTranscriptSegment = useCallback(async (
        recordingId: string,
        segmentIndex: number,
        params: TranscriptSegmentUpdateParams,
    ) => {
        setError("");
        setEditingTranscriptId(recordingId);
        try {
            const recording = await updateTranscriptSegment(recordingId, segmentIndex, params);
            setSelectedRecording(recording);
        } catch (err) {
            setError(getErrorMessage(err, "Failed to update transcript"));
            throw err;
        } finally {
            setEditingTranscriptId(undefined);
        }
    }, []);

    const handleExtractSpeakers = useCallback(async (
        recordingId: string,
        params: SpeakerExtractionParams,
    ) => {
        setError("");
        setExtractingSpeakerId(recordingId);
        try {
            const recording = await extractRecordingSpeakers(recordingId, params);
            setSelectedRecording(recording);
            await refreshProject();
        } catch (err) {
            setError(getErrorMessage(err, "Failed to extract speakers"));
            throw err;
        } finally {
            setExtractingSpeakerId(undefined);
        }
    }, [refreshProject]);

    const handleSummarizeRecording = useCallback(async (
        recordingId: string,
        params: RecordingSummaryParams,
    ) => {
        setError("");
        setSummarizingId(recordingId);
        try {
            const recording = await summarizeRecording(recordingId, params);
            setSelectedRecording((current) =>
                current?.id === recordingId ? recording : current,
            );
        } catch (err) {
            setError(getErrorMessage(err, "Failed to summarize recording"));
            throw err;
        } finally {
            setSummarizingId(undefined);
        }
    }, []);

    const handleUpdateRecordingSummary = useCallback(async (
        recordingId: string,
        params: RecordingSummaryUpdateParams,
    ) => {
        setError("");
        setUpdatingSummaryId(recordingId);
        try {
            const recording = await updateRecordingSummary(recordingId, params);
            setSelectedRecording((current) =>
                current?.id === recordingId ? recording : current,
            );
        } catch (err) {
            setError(getErrorMessage(err, "Failed to update summary"));
            throw err;
        } finally {
            setUpdatingSummaryId(undefined);
        }
    }, []);

    const handleOpenSettings = useCallback(() => {
        setIsSettingsOpen(true);
    }, []);

    if (isLoadingProject) {
        return (
            <main className="min-h-screen bg-zinc-100 p-6 text-sm text-zinc-500">
                Loading project...
            </main>
        );
    }

    if (!project) {
        return (
            <main className="min-h-screen bg-zinc-100 p-6 text-sm text-zinc-500">
                Project not found.
            </main>
        );
    }

    return (
        <main className="min-h-screen bg-zinc-100 text-zinc-950">
            <div className="mx-auto flex w-full max-w-7xl flex-col gap-5 px-5 py-6">
                <header className="flex flex-wrap items-start justify-between gap-4">
                    <div className="flex flex-col gap-3">
                        <Link href="/" className="text-sm font-medium text-zinc-600 hover:text-zinc-950 cursor-pointer">
                            Back to projects
                        </Link>
                        <div>
                            <h1 className="text-2xl font-semibold">{project.name}</h1>
                            {project.description && (
                                <p className="mt-1 text-sm text-zinc-600">{project.description}</p>
                            )}
                        </div>
                    </div>
                    <button
                        type="button"
                        onClick={() => setIsSettingsOpen(true)}
                        className="min-h-10 rounded-md border border-zinc-300 bg-white px-4 text-sm font-medium text-zinc-700 hover:border-zinc-400 hover:text-zinc-950 hover:cursor-pointer"
                    >
                        Settings
                    </button>
                </header>

                {error && (
                    <div className="rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
                        {error}
                    </div>
                )}

                <div className="grid overflow-hidden rounded-lg border border-zinc-200 bg-white lg:grid-cols-[360px_1fr]">
                    <RecordingSidebar
                        recordings={recordings}
                        selectedId={selectedRecordingId}
                        isDraftSelected={isDraftSelected}
                        hasDraft={hasDraft}
                        onSelect={handleSelectRecording}
                        onSelectDraft={handleSelectDraft}
                        onStartDraft={handleStartDraft}
                        onDelete={handleDeleteRecording}
                        onRename={handleRenameRecording}
                        renamingId={renamingRecordingId}
                        isUploading={isUploading}
                    />
                    {isDraftSelected && draftMode ? (
                        <AddRecordingPanel
                            key="draft-recording"
                            projectId={projectId}
                            mode={draftMode}
                            onModeChange={setDraftMode}
                            onCancel={handleCancelDraft}
                            onUpload={handleUpload}
                            onBeforeLiveStart={handleLiveStartRequest}
                            onLiveSaved={handleLiveRecordingSaved}
                            isUploading={isUploading}
                            runtimeDevices={runtimeDevices}
                        />
                    ) : (
                        <RecordingDetail
                            key={selectedRecording?.id ?? "empty-recording"}
                            recording={selectedRecording}
                            isLoading={isLoadingRecording}
                            runtimeDevices={runtimeDevices}
                            isRetranscribing={retranscribingId === selectedRecording?.id}
                            onRetranscribe={handleRetranscribeRecording}
                            isCanceling={cancelingRecordingId === selectedRecording?.id}
                            onCancel={handleCancelRecording}
                            isRenamingSpeakers={renamingSpeakerId === selectedRecording?.id}
                            onRenameSpeakers={handleRenameTranscriptSpeakers}
                            isEditingTranscript={editingTranscriptId === selectedRecording?.id}
                            onUpdateTranscriptSegment={handleUpdateTranscriptSegment}
                            isExtractingSpeakers={extractingSpeakerId === selectedRecording?.id}
                            onExtractSpeakers={handleExtractSpeakers}
                            isSummarizing={summarizingId === selectedRecording?.id}
                            onSummarize={handleSummarizeRecording}
                            isUpdatingSummary={updatingSummaryId === selectedRecording?.id}
                            onUpdateSummary={handleUpdateRecordingSummary}
                            byokModelPresets={byokSettings.validModelPresets}
                            onOpenSettings={handleOpenSettings}
                        />
                    )}
                </div>
            </div>
            {isSettingsOpen && (
                <BYOKSettingsModal
                    settings={byokSettings.settings}
                    onSave={(settings) => {
                        byokSettings.setSettings(settings);
                        setIsSettingsOpen(false);
                    }}
                    onClearSavedKeys={byokSettings.clearSavedKeys}
                    onClose={() => setIsSettingsOpen(false)}
                />
            )}
            {pendingSpeechModelGate && (
                <MissingSpeechModelsModal
                    models={pendingSpeechModelGate.models}
                    onInstalled={() => {
                        pendingSpeechModelGate.resolve(true);
                        setPendingSpeechModelGate(null);
                    }}
                    onCancel={() => {
                        pendingSpeechModelGate.resolve(false);
                        setPendingSpeechModelGate(null);
                    }}
                />
            )}
        </main>
    );
}
