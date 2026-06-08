"use client";

import {
    Recording,
    RuntimeDevice,
    RuntimeDevices,
    TranscriptionModel,
} from "@/src/api/sonaApi";
import BrowserAudioRecorder from "@/src/components/BrowserAudioRecorder";
import LiveTranscriptionPanel from "@/src/components/LiveTranscriptionPanel";
import RecordingUploader from "@/src/components/RecordingUploader";

export type AddRecordingMode = "upload" | "record" | "live";

interface Props {
    projectId: string;
    mode: AddRecordingMode;
    onModeChange: (mode: AddRecordingMode) => void;
    onCancel: () => void;
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

const ADD_MODES: Array<{ value: AddRecordingMode; label: string }> = [
    { value: "upload", label: "Upload audio" },
    { value: "record", label: "Record audio" },
    { value: "live", label: "Live transcription" },
];

export default function AddRecordingPanel({
    projectId,
    mode,
    onModeChange,
    onCancel,
    onUpload,
    onLiveSaved,
    isUploading,
    runtimeDevices,
}: Props) {
    return (
        <section className="min-h-[520px] bg-white">
            <div className="border-b border-zinc-200 px-6 py-5">
                <div className="flex flex-wrap items-start justify-between gap-4">
                    <div>
                        <h2 className="text-xl font-semibold text-zinc-950">
                            New recording
                        </h2>
                        <p className="mt-1 text-sm text-zinc-500">
                            Upload an audio file, record first, or transcribe live.
                        </p>
                    </div>
                    <button
                        type="button"
                        onClick={onCancel}
                        disabled={isUploading}
                        className="min-h-10 rounded-md border border-zinc-300 px-4 text-sm font-medium text-zinc-700 hover:border-zinc-400 hover:text-zinc-950 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                        Cancel
                    </button>
                </div>
            </div>

            <div className="px-6 py-5">
                <div className="mb-5 grid grid-cols-3 gap-2 rounded-lg bg-zinc-100 p-1">
                    {ADD_MODES.map((item) => (
                        <button
                            key={item.value}
                            type="button"
                            onClick={() => onModeChange(item.value)}
                            disabled={isUploading}
                            className={`min-h-10 rounded-md text-sm font-medium transition-colors ${mode === item.value
                                ? "bg-white text-zinc-950 shadow-sm"
                                : "text-zinc-600 hover:text-zinc-950"
                                } disabled:cursor-not-allowed disabled:opacity-50`}
                        >
                            {item.label}
                        </button>
                    ))}
                </div>

                {mode === "upload" ? (
                    <RecordingUploader
                        onUpload={onUpload}
                        isUploading={isUploading}
                        runtimeDevices={runtimeDevices}
                    />
                ) : mode === "record" ? (
                    <BrowserAudioRecorder
                        onUpload={onUpload}
                        isUploading={isUploading}
                        runtimeDevices={runtimeDevices}
                    />
                ) : (
                    <LiveTranscriptionPanel
                        projectId={projectId}
                        runtimeDevices={runtimeDevices}
                        onSaved={onLiveSaved}
                    />
                )}
            </div>
        </section>
    );
}
