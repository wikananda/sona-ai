"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
    deleteRecording,
    getProject,
    getRecording,
    getRuntimeDevices,
    Project,
    Recording,
    RuntimeDevice,
    RuntimeDevices,
    retranscribeRecording,
    TranscriptionModel,
    uploadProjectRecording,
} from "@/src/api/sonaApi";
import RecordingDetail from "@/src/components/RecordingDetail";
import RecordingSidebar from "@/src/components/RecordingSidebar";
import RecordingUploader from "@/src/components/RecordingUploader";

export default function ProjectDetailPage() {
    const params = useParams<{ id: string }>();
    const projectId = params.id;

    const [project, setProject] = useState<Project | null>(null);
    const [selectedRecordingId, setSelectedRecordingId] = useState<string>();
    const [selectedRecording, setSelectedRecording] = useState<Recording | null>(null);
    const [isLoadingProject, setIsLoadingProject] = useState(true);
    const [isLoadingRecording, setIsLoadingRecording] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [retranscribingId, setRetranscribingId] = useState<string>();
    const [runtimeDevices, setRuntimeDevices] = useState<RuntimeDevices>({
        default: "auto",
        available: ["auto", "cpu"],
        torch: { cuda: false, mps: false },
    });
    const [error, setError] = useState("");

    const recordings = useMemo(() => project?.recordings ?? [], [project]);
    const hasActiveRecordings = recordings.some((recording) =>
        recording.status === "pending" || recording.status === "processing"
    );

    const refreshProject = useCallback(async () => {
        const data = await getProject(projectId);
        setProject(data);
        setSelectedRecordingId((current) => {
            if (current && data.recordings?.some((recording) => recording.id === current)) {
                return current;
            }
            return data.recordings?.[0]?.id;
        });
    }, [projectId]);

    const refreshSelectedRecording = useCallback(async () => {
        if (!selectedRecordingId) {
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

    useEffect(() => {
        if (!hasActiveRecordings) return;

        const intervalId = window.setInterval(() => {
            refreshProject().catch((err) => setError(err.message));
            refreshSelectedRecording().catch((err) => setError(err.message));
        }, 3000);

        return () => window.clearInterval(intervalId);
    }, [hasActiveRecordings, refreshProject, refreshSelectedRecording]);

    const handleUpload = async (params: {
        files: File[];
        language?: string;
        model: TranscriptionModel;
        device: RuntimeDevice;
        minSpeakers?: number | "";
        maxSpeakers?: number | "";
    }) => {
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
                });
                firstRecording = firstRecording ?? recording;
            }
            await refreshProject();
            if (firstRecording) {
                setSelectedRecordingId(firstRecording.id);
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to upload recording");
        } finally {
            setIsUploading(false);
        }
    };

    const handleDeleteRecording = async (recordingId: string) => {
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
            setError(err instanceof Error ? err.message : "Failed to delete recording");
        }
    };

    const handleRetranscribeRecording = async (recordingId: string) => {
        if (!window.confirm("Re-transcribe this recording with its current settings?")) return;

        setError("");
        setRetranscribingId(recordingId);
        try {
            const recording = await retranscribeRecording(recordingId);
            setSelectedRecording(recording);
            await refreshProject();
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to re-transcribe recording");
        } finally {
            setRetranscribingId(undefined);
        }
    };

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
                <header className="flex flex-col gap-3">
                    <Link href="/" className="text-sm font-medium text-zinc-600 hover:text-zinc-950">
                        Back to projects
                    </Link>
                    <div>
                        <h1 className="text-2xl font-semibold">{project.name}</h1>
                        {project.description && (
                            <p className="mt-1 text-sm text-zinc-600">{project.description}</p>
                        )}
                    </div>
                </header>

                <RecordingUploader
                    onUpload={handleUpload}
                    isUploading={isUploading}
                    runtimeDevices={runtimeDevices}
                />

                {error && (
                    <div className="rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
                        {error}
                    </div>
                )}

                <div className="grid overflow-hidden rounded-lg border border-zinc-200 bg-white lg:grid-cols-[360px_1fr]">
                    <RecordingSidebar
                        recordings={recordings}
                        selectedId={selectedRecordingId}
                        onSelect={setSelectedRecordingId}
                        onDelete={handleDeleteRecording}
                    />
                    <RecordingDetail
                        recording={selectedRecording}
                        isLoading={isLoadingRecording}
                        runtimeDevices={runtimeDevices}
                        isRetranscribing={retranscribingId === selectedRecording?.id}
                        onRetranscribe={handleRetranscribeRecording}
                    />
                </div>
            </div>
        </main>
    );
}
