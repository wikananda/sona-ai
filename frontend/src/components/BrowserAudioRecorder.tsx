"use client";

import { ChangeEvent, useEffect, useRef, useState } from "react";
import { RuntimeDevice, RuntimeDevices, TranscriptionModel } from "@/src/api/sonaApi";
import RecordingSettingsForm from "@/src/components/RecordingSettingsForm";

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
    const audioContextRef = useRef<AudioContext | null>(null);
    const chunksRef = useRef<Blob[]>([]);
    const startedAtRef = useRef<number>(0);
    const [includeMicrophone, setIncludeMicrophone] = useState(true);
    const [includeSystemAudio, setIncludeSystemAudio] = useState(false);
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
            if (!includeMicrophone && !includeSystemAudio) {
                throw new Error("Choose microphone, system audio, or both.");
            }
            const stream = await createCaptureStream({
                includeMicrophone,
                includeSystemAudio,
            });
            const mimeType = chooseMimeType();
            const recorder = mimeType
                ? new MediaRecorder(stream.recordingStream, { mimeType })
                : new MediaRecorder(stream.recordingStream);

            streamRef.current = stream.cleanupStream;
            audioContextRef.current = stream.audioContext;
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
                    buildRecordingFilename(
                        sourceLabelForFilename({ includeMicrophone, includeSystemAudio }),
                        finalMimeType,
                    ),
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
            setError(errorMessage(err, includeMicrophone));
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
            <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                        <p className="text-sm font-medium text-zinc-900">
                            Record audio
                        </p>
                        <p className="mt-1 text-xs text-zinc-500">
                            Choose microphone, system audio, or both before recording.
                        </p>
                    </div>
                    <span className="rounded-full bg-white px-3 py-1 text-sm font-medium text-zinc-700 ring-1 ring-zinc-200">
                        {formatDuration(elapsedSeconds)}
                    </span>
                </div>

                <div className="mt-4 flex flex-col gap-2 rounded-md border border-zinc-200 bg-white p-3">
                    <p className="text-xs font-medium text-zinc-500">Audio sources</p>
                    <SourceCheckbox
                        label="Microphone"
                        description="Capture your voice."
                        checked={includeMicrophone}
                        disabled={!canChangeSource || isUploading}
                        onChange={(event) => setIncludeMicrophone(event.target.checked)}
                    />
                    <SourceCheckbox
                        label="System audio"
                        description="Capture shared tab, window, or meeting audio."
                        checked={includeSystemAudio}
                        disabled={!canChangeSource || isUploading}
                        onChange={(event) => setIncludeSystemAudio(event.target.checked)}
                    />
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
        audioContextRef.current?.close().catch(() => undefined);
        audioContextRef.current = null;
    }
}

function SourceCheckbox({
    label,
    description,
    checked,
    disabled,
    onChange,
}: {
    label: string;
    description: string;
    checked: boolean;
    disabled: boolean;
    onChange: (event: ChangeEvent<HTMLInputElement>) => void;
}) {
    return (
        <label className="flex items-start gap-2 text-sm text-zinc-700">
            <input
                type="checkbox"
                checked={checked}
                disabled={disabled}
                onChange={onChange}
                className="mt-0.5 h-4 w-4 rounded border-zinc-300 text-zinc-950 focus:ring-zinc-950 disabled:cursor-not-allowed disabled:opacity-50"
            />
            <span>
                {label}
                <span className="block text-xs text-zinc-500">{description}</span>
            </span>
        </label>
    );
}

async function createCaptureStream({
    includeMicrophone,
    includeSystemAudio,
}: {
    includeMicrophone: boolean;
    includeSystemAudio: boolean;
}): Promise<{
    recordingStream: MediaStream;
    cleanupStream: MediaStream;
    audioContext: AudioContext | null;
}> {
    if (includeMicrophone && !includeSystemAudio) {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        return {
            recordingStream: stream,
            cleanupStream: stream,
            audioContext: null,
        };
    }

    if (!includeMicrophone && includeSystemAudio) {
        const displayStream = await captureSystemAudio();
        return {
            recordingStream: new MediaStream(displayStream.getAudioTracks()),
            cleanupStream: displayStream,
            audioContext: null,
        };
    }

    const [micStream, displayStream] = await Promise.all([
        navigator.mediaDevices.getUserMedia({ audio: true }),
        captureSystemAudio(),
    ]);
    const audioContext = new AudioContext();
    const destination = audioContext.createMediaStreamDestination();
    audioContext.createMediaStreamSource(micStream).connect(destination);
    audioContext.createMediaStreamSource(displayStream).connect(destination);
    const cleanupStream = new MediaStream([
        ...micStream.getTracks(),
        ...displayStream.getTracks(),
    ]);

    return {
        recordingStream: destination.stream,
        cleanupStream,
        audioContext,
    };
}

async function captureSystemAudio(): Promise<MediaStream> {
    const displayStream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: true,
    });
    if (displayStream.getAudioTracks().length === 0) {
        displayStream.getTracks().forEach((track) => track.stop());
        throw new Error("NO_SCREEN_AUDIO");
    }
    return displayStream;
}

function chooseMimeType(): string {
    if (!hasMediaRecorderSupport()) return "";

    return MIME_TYPES.find((mimeType) => MediaRecorder.isTypeSupported(mimeType)) ?? "";
}

function hasMediaRecorderSupport(): boolean {
    return typeof MediaRecorder !== "undefined" && Boolean(navigator.mediaDevices);
}

function buildRecordingFilename(source: string, mimeType: string): string {
    const timestamp = new Date()
        .toISOString()
        .replace(/[:.]/g, "-")
        .slice(0, 19);
    return `${source}-recording-${timestamp}.${extensionForMimeType(mimeType)}`;
}

function sourceLabelForFilename({
    includeMicrophone,
    includeSystemAudio,
}: {
    includeMicrophone: boolean;
    includeSystemAudio: boolean;
}): string {
    if (includeMicrophone && includeSystemAudio) return "mixed";
    if (includeSystemAudio) return "system";
    return "microphone";
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

function errorMessage(err: unknown, hasMicrophone: boolean): string {
    if (err instanceof Error && err.message === "NO_SCREEN_AUDIO") {
        return "No audio track was shared. Choose a tab/window with audio enabled, or use microphone mode.";
    }
    if (err instanceof DOMException && err.name === "NotAllowedError") {
        return "Recording permission was denied.";
    }
    if (err instanceof DOMException && err.name === "NotFoundError") {
        return hasMicrophone
            ? "No microphone was found."
            : "No screen or audio source was selected.";
    }
    return err instanceof Error ? err.message : "Failed to start recording.";
}
