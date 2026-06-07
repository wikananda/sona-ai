"use client";

import { useEffect, useRef, useState } from "react";
import { RuntimeDevice, RuntimeDevices, TranscriptionModel } from "@/src/api/sonaApi";
import RecordingSettingsForm from "@/src/components/RecordingSettingsForm";

type RecordingSource = "microphone" | "screen";
type RecorderState = "idle" | "requesting" | "recording" | "stopping" | "preview";

interface Props {
    onUpload: (params: {
        files: File[];
        language?: string;
        model: TranscriptionModel;
        device: RuntimeDevice;
        minSpeakers?: number | "";
        maxSpeakers?: number | "";
        extractSpeakers?: boolean;
    }) => Promise<void>;
    isUploading: boolean;
    runtimeDevices: RuntimeDevices;
}

const MIME_TYPES = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
    "audio/ogg;codecs=opus",
];

export default function BrowserAudioRecorder({
    onUpload,
    isUploading,
    runtimeDevices,
}: Props) {
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const chunksRef = useRef<Blob[]>([]);
    const startedAtRef = useRef<number>(0);
    const [source, setSource] = useState<RecordingSource>("microphone");
    const [recorderState, setRecorderState] = useState<RecorderState>("idle");
    const [elapsedSeconds, setElapsedSeconds] = useState(0);
    const [recordedFile, setRecordedFile] = useState<File | null>(null);
    const [previewUrl, setPreviewUrl] = useState("");
    const [error, setError] = useState("");

    useEffect(() => {
        if (recorderState !== "recording") return;

        const intervalId = window.setInterval(() => {
            setElapsedSeconds(Math.floor((Date.now() - startedAtRef.current) / 1000));
        }, 500);

        return () => window.clearInterval(intervalId);
    }, [recorderState]);

    useEffect(() => {
        return () => {
            stopActiveStream();
            if (previewUrl) {
                URL.revokeObjectURL(previewUrl);
            }
        };
    }, [previewUrl]);

    const startRecording = async () => {
        if (!hasMediaRecorderSupport()) {
            setError("Audio recording is not supported in this browser.");
            return;
        }

        setError("");
        setRecordedFile(null);
        if (previewUrl) {
            URL.revokeObjectURL(previewUrl);
            setPreviewUrl("");
        }

        setRecorderState("requesting");
        try {
            const stream = await createCaptureStream(source);
            const mimeType = chooseMimeType();
            const recorder = mimeType
                ? new MediaRecorder(stream.recordingStream, { mimeType })
                : new MediaRecorder(stream.recordingStream);

            streamRef.current = stream.cleanupStream;
            chunksRef.current = [];
            mediaRecorderRef.current = recorder;
            startedAtRef.current = Date.now();
            setElapsedSeconds(0);

            recorder.addEventListener("dataavailable", (event) => {
                if (event.data.size > 0) {
                    chunksRef.current.push(event.data);
                }
            });

            recorder.addEventListener("stop", () => {
                const finalMimeType = recorder.mimeType || mimeType || "audio/webm";
                const blob = new Blob(chunksRef.current, { type: finalMimeType });
                const file = new File(
                    [blob],
                    buildRecordingFilename(source, finalMimeType),
                    { type: finalMimeType },
                );

                stopActiveStream();
                mediaRecorderRef.current = null;
                setRecordedFile(file);
                setPreviewUrl(URL.createObjectURL(blob));
                setElapsedSeconds(Math.floor((Date.now() - startedAtRef.current) / 1000));
                setRecorderState("preview");
            });

            recorder.start();
            setRecorderState("recording");
        } catch (err) {
            stopActiveStream();
            mediaRecorderRef.current = null;
            setRecorderState("idle");
            setError(errorMessage(err, source));
        }
    };

    const stopRecording = () => {
        const recorder = mediaRecorderRef.current;
        if (!recorder || recorder.state === "inactive") return;

        setRecorderState("stopping");
        recorder.stop();
    };

    const clearRecording = () => {
        if (previewUrl) {
            URL.revokeObjectURL(previewUrl);
            setPreviewUrl("");
        }
        setRecordedFile(null);
        setElapsedSeconds(0);
        setRecorderState("idle");
        setError("");
    };

    const canChangeSource = recorderState === "idle" || recorderState === "preview";
    const isBusy = isUploading || recorderState === "requesting" || recorderState === "stopping";

    return (
        <div className="flex flex-col gap-4">
            <div className="grid grid-cols-2 gap-2 rounded-lg bg-zinc-100 p-1">
                <button
                    type="button"
                    onClick={() => setSource("microphone")}
                    disabled={!canChangeSource || isUploading}
                    className={`min-h-10 rounded-md text-sm font-medium transition-colors ${source === "microphone"
                        ? "bg-white text-zinc-950 shadow-sm"
                        : "text-zinc-600 hover:text-zinc-950"
                        } disabled:cursor-not-allowed disabled:opacity-50`}
                >
                    Microphone
                </button>
                <button
                    type="button"
                    onClick={() => setSource("screen")}
                    disabled={!canChangeSource || isUploading}
                    className={`min-h-10 rounded-md text-sm font-medium transition-colors ${source === "screen"
                        ? "bg-white text-zinc-950 shadow-sm"
                        : "text-zinc-600 hover:text-zinc-950"
                        } disabled:cursor-not-allowed disabled:opacity-50`}
                >
                    Screen audio
                </button>
            </div>

            <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                        <p className="text-sm font-medium text-zinc-900">
                            {source === "microphone" ? "Record microphone" : "Record screen or meeting audio"}
                        </p>
                        <p className="mt-1 text-xs text-zinc-500">
                            {source === "microphone"
                                ? "Your browser will ask for microphone permission."
                                : "Choose a tab, window, or screen that includes audio when prompted."}
                        </p>
                    </div>
                    <span className="rounded-full bg-white px-3 py-1 text-sm font-medium text-zinc-700 ring-1 ring-zinc-200">
                        {formatDuration(elapsedSeconds)}
                    </span>
                </div>

                <div className="mt-4 flex gap-2">
                    {recorderState === "recording" ? (
                        <button
                            type="button"
                            onClick={stopRecording}
                            className="min-h-10 flex-1 rounded-md bg-red-700 px-4 text-sm font-medium text-white hover:bg-red-800"
                        >
                            Stop recording
                        </button>
                    ) : (
                        <button
                            type="button"
                            onClick={startRecording}
                            disabled={isBusy}
                            className="min-h-10 flex-1 rounded-md bg-zinc-950 px-4 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-40"
                        >
                            {recorderState === "requesting" ? "Requesting access" : "Start recording"}
                        </button>
                    )}

                    {recordedFile && (
                        <button
                            type="button"
                            onClick={clearRecording}
                            disabled={isUploading}
                            className="min-h-10 rounded-md border border-zinc-300 px-3 text-sm font-medium text-zinc-700 hover:border-zinc-400 hover:text-zinc-950 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                            Re-record
                        </button>
                    )}
                </div>

                {error && (
                    <p className="mt-3 text-sm text-red-700">{error}</p>
                )}
            </div>

            {recordedFile && previewUrl && (
                <div className="flex flex-col gap-3 rounded-lg border border-zinc-200 p-4">
                    <div>
                        <p className="truncate text-sm font-medium text-zinc-900">
                            {recordedFile.name}
                        </p>
                        <p className="mt-1 text-xs text-zinc-500">
                            Preview the recording before uploading it for transcription.
                        </p>
                    </div>
                    <audio controls src={previewUrl} className="w-full" />
                    <RecordingSettingsForm
                        file={recordedFile}
                        onUpload={onUpload}
                        isUploading={isUploading}
                        runtimeDevices={runtimeDevices}
                        onClear={clearRecording}
                        uploadLabel="Upload recording"
                    />
                </div>
            )}
        </div>
    );

    function stopActiveStream() {
        streamRef.current?.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
    }
}

async function createCaptureStream(source: RecordingSource): Promise<{
    recordingStream: MediaStream;
    cleanupStream: MediaStream;
}> {
    if (source === "microphone") {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        return {
            recordingStream: stream,
            cleanupStream: stream,
        };
    }

    const displayStream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: true,
    });
    const audioTracks = displayStream.getAudioTracks();
    if (audioTracks.length === 0) {
        displayStream.getTracks().forEach((track) => track.stop());
        throw new Error("NO_SCREEN_AUDIO");
    }

    return {
        recordingStream: new MediaStream(audioTracks),
        cleanupStream: displayStream,
    };
}

function chooseMimeType(): string {
    if (!hasMediaRecorderSupport()) return "";

    return MIME_TYPES.find((mimeType) => MediaRecorder.isTypeSupported(mimeType)) ?? "";
}

function hasMediaRecorderSupport(): boolean {
    return typeof MediaRecorder !== "undefined" && Boolean(navigator.mediaDevices);
}

function buildRecordingFilename(source: RecordingSource, mimeType: string): string {
    const timestamp = new Date()
        .toISOString()
        .replace(/[:.]/g, "-")
        .slice(0, 19);
    return `${source}-recording-${timestamp}.${extensionForMimeType(mimeType)}`;
}

function extensionForMimeType(mimeType: string): string {
    if (mimeType.includes("mp4")) return "m4a";
    if (mimeType.includes("ogg")) return "ogg";
    return "webm";
}

function formatDuration(seconds: number): string {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${String(remainingSeconds).padStart(2, "0")}`;
}

function errorMessage(err: unknown, source: RecordingSource): string {
    if (err instanceof Error && err.message === "NO_SCREEN_AUDIO") {
        return "No audio track was shared. Choose a tab/window with audio enabled, or use microphone mode.";
    }
    if (err instanceof DOMException && err.name === "NotAllowedError") {
        return "Recording permission was denied.";
    }
    if (err instanceof DOMException && err.name === "NotFoundError") {
        return source === "microphone"
            ? "No microphone was found."
            : "No screen or audio source was selected.";
    }
    return err instanceof Error ? err.message : "Failed to start recording.";
}
