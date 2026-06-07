"use client";

import { useEffect, useRef, useState } from "react";
import {
    Recording,
    RuntimeDevice,
    RuntimeDevices,
    SpeakerSegment,
    TranscriptionModel,
    saveLiveRecording,
    transcribeLiveChunk,
} from "@/src/api/sonaApi";
import {
    deviceLabel,
    TRANSCRIPTION_LANGUAGES,
    TRANSCRIPTION_MODELS,
} from "@/src/utils/transcriptionSettings";

type RecordingSource = "microphone" | "screen";
type LiveState = "idle" | "requesting" | "recording" | "stopping" | "saving";

interface Props {
    projectId: string;
    runtimeDevices: RuntimeDevices;
    onSaved: (recording: Recording) => void;
}

const CHUNK_SECONDS = 8;
const CHUNK_MS = CHUNK_SECONDS * 1000;
const MIME_TYPES = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
    "audio/ogg;codecs=opus",
];

export default function LiveTranscriptionPanel({
    projectId,
    runtimeDevices,
    onSaved,
}: Props) {
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const chunksRef = useRef<Blob[]>([]);
    const chunkIndexRef = useRef(0);
    const startedAtRef = useRef(0);
    const processingQueueRef = useRef<Promise<void>>(Promise.resolve());
    const finalMimeTypeRef = useRef("audio/webm");
    const segmentsRef = useRef<SpeakerSegment[]>([]);
    const chunkErrorRef = useRef("");

    const [source, setSource] = useState<RecordingSource>("microphone");
    const [state, setState] = useState<LiveState>("idle");
    const [elapsedSeconds, setElapsedSeconds] = useState(0);
    const [language, setLanguage] = useState("auto");
    const [model, setModel] = useState<TranscriptionModel>("parakeet");
    const [device, setDevice] = useState<RuntimeDevice>(runtimeDevices.default);
    const [segments, setSegments] = useState<SpeakerSegment[]>([]);
    const [error, setError] = useState("");

    const selectedDevice = runtimeDevices.available.includes(device)
        ? device
        : runtimeDevices.default;
    const isBusy = state === "requesting" || state === "stopping" || state === "saving";
    const isRecording = state === "recording";
    const hasNonEnglishLanguage = !["auto", "en"].includes(language);

    useEffect(() => {
        if (hasNonEnglishLanguage && model === "parakeet") {
            setModel("faster-whisper-turbo");
        }
    }, [hasNonEnglishLanguage, model]);

    useEffect(() => {
        if (!isRecording) return;

        const intervalId = window.setInterval(() => {
            setElapsedSeconds(Math.floor((Date.now() - startedAtRef.current) / 1000));
        }, 500);

        return () => window.clearInterval(intervalId);
    }, [isRecording]);

    useEffect(() => {
        return () => {
            stopActiveStream();
        };
    }, []);

    const startLiveTranscription = async () => {
        if (!hasMediaRecorderSupport()) {
            setError("Audio recording is not supported in this browser.");
            return;
        }

        setState("requesting");
        setError("");
        setSegments([]);
        segmentsRef.current = [];
        chunkErrorRef.current = "";
        chunksRef.current = [];
        chunkIndexRef.current = 0;
        processingQueueRef.current = Promise.resolve();

        try {
            const stream = await createCaptureStream(source);
            const mimeType = chooseMimeType();
            const recorder = mimeType
                ? new MediaRecorder(stream.recordingStream, { mimeType })
                : new MediaRecorder(stream.recordingStream);
            const startedAt = Date.now();

            streamRef.current = stream.cleanupStream;
            mediaRecorderRef.current = recorder;
            startedAtRef.current = startedAt;
            finalMimeTypeRef.current = recorder.mimeType || mimeType || "audio/webm";
            setElapsedSeconds(0);

            recorder.addEventListener("dataavailable", (event) => {
                if (event.data.size <= 0) return;
                chunksRef.current.push(event.data);
                enqueueChunk();
            });

            recorder.addEventListener("stop", () => {
                void finalizeLiveRecording();
            });

            recorder.start(CHUNK_MS);
            setState("recording");
        } catch (err) {
            stopActiveStream();
            mediaRecorderRef.current = null;
            setState("idle");
            setError(errorMessage(err, source));
        }
    };

    const stopLiveTranscription = () => {
        const recorder = mediaRecorderRef.current;
        if (!recorder || recorder.state === "inactive") return;

        setState("stopping");
        recorder.stop();
    };

    const discardLiveTranscription = () => {
        stopActiveStream();
        chunksRef.current = [];
        chunkIndexRef.current = 0;
        processingQueueRef.current = Promise.resolve();
        mediaRecorderRef.current = null;
        segmentsRef.current = [];
        chunkErrorRef.current = "";
        setSegments([]);
        setElapsedSeconds(0);
        setError("");
        setState("idle");
    };

    const enqueueChunk = () => {
        const chunkIndex = chunkIndexRef.current;
        const fullRecordingBlob = new Blob(chunksRef.current, {
            type: finalMimeTypeRef.current,
        });
        const file = new File(
            [fullRecordingBlob],
            `live-chunk-${chunkIndex}.${extensionForMimeType(finalMimeTypeRef.current)}`,
            { type: finalMimeTypeRef.current },
        );

        chunkIndexRef.current += 1;

        processingQueueRef.current = processingQueueRef.current
            .then(async () => {
                const result = await transcribeLiveChunk({
                    projectId,
                    file,
                    chunkIndex,
                    chunkStart: 0,
                    language,
                    model,
                    device: selectedDevice,
                });
                segmentsRef.current = result.segments;
                setSegments(segmentsRef.current);
            })
            .catch((err) => {
                const message = err instanceof Error
                    ? err.message
                    : "Failed to transcribe live chunk";
                chunkErrorRef.current = message;
                setError(message);
            });
    };

    const finalizeLiveRecording = async () => {
        stopActiveStream();
        mediaRecorderRef.current = null;
        setState("saving");

        try {
            await processingQueueRef.current;
            if (chunkErrorRef.current) {
                throw new Error(chunkErrorRef.current);
            }
            const finalSegments = segmentsRef.current;
            if (!finalSegments.length) {
                throw new Error("No live transcript was generated.");
            }
            const finalMimeType = finalMimeTypeRef.current;
            const finalFile = new File(
                chunksRef.current,
                `live-recording-${timestampForFilename()}.${extensionForMimeType(finalMimeType)}`,
                { type: finalMimeType },
            );
            const recording = await saveLiveRecording({
                projectId,
                file: finalFile,
                segments: finalSegments,
                language,
                model,
                device: selectedDevice,
            });
            onSaved(recording);
            discardLiveTranscription();
        } catch (err) {
            setState("idle");
            setError(err instanceof Error ? err.message : "Failed to save live recording");
        }
    };

    return (
        <section className="rounded-lg border border-zinc-200 bg-white">
            <div className="border-b border-zinc-200 px-4 py-3">
                <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                        <h2 className="text-sm font-semibold text-zinc-900">
                            Live transcription
                        </h2>
                        <p className="mt-1 text-sm text-zinc-500">
                            Record short chunks and save the final transcript into this project.
                        </p>
                    </div>
                    <span className="rounded-full bg-zinc-100 px-3 py-1 text-sm font-medium text-zinc-700">
                        {formatDuration(elapsedSeconds)}
                    </span>
                </div>
            </div>

            <div className="grid gap-4 p-4 lg:grid-cols-[280px_1fr]">
                <div className="flex flex-col gap-3">
                    <div className="grid grid-cols-2 gap-2 rounded-lg bg-zinc-100 p-1">
                        <button
                            type="button"
                            onClick={() => setSource("microphone")}
                            disabled={state !== "idle"}
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
                            disabled={state !== "idle"}
                            className={`min-h-10 rounded-md text-sm font-medium transition-colors ${source === "screen"
                                ? "bg-white text-zinc-950 shadow-sm"
                                : "text-zinc-600 hover:text-zinc-950"
                                } disabled:cursor-not-allowed disabled:opacity-50`}
                        >
                            Screen audio
                        </button>
                    </div>

                    <label className="flex flex-col gap-1">
                        <span className="text-xs font-medium text-zinc-500">Language</span>
                        <select
                            value={language}
                            onChange={(event) => setLanguage(event.target.value)}
                            disabled={state !== "idle"}
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
                        <span className="text-xs font-medium text-zinc-500">Model</span>
                        <select
                            value={model}
                            onChange={(event) => setModel(event.target.value as TranscriptionModel)}
                            disabled={state !== "idle"}
                            className="min-h-10 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                            {TRANSCRIPTION_MODELS.map((item) => (
                                <option
                                    key={item.value}
                                    value={item.value}
                                    disabled={hasNonEnglishLanguage && item.value === "parakeet"}
                                >
                                    {item.label}
                                </option>
                            ))}
                        </select>
                        {hasNonEnglishLanguage && (
                            <span className="text-xs text-zinc-500">
                                Parakeet currently supports English only.
                            </span>
                        )}
                    </label>

                    <label className="flex flex-col gap-1">
                        <span className="text-xs font-medium text-zinc-500">Device</span>
                        <select
                            value={selectedDevice}
                            onChange={(event) => setDevice(event.target.value as RuntimeDevice)}
                            disabled={state !== "idle"}
                            className="min-h-10 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                            {runtimeDevices.available.map((item) => (
                                <option key={item} value={item}>
                                    {deviceLabel(item)}
                                </option>
                            ))}
                        </select>
                    </label>

                    <div className="flex gap-2">
                        {isRecording ? (
                            <button
                                type="button"
                                onClick={stopLiveTranscription}
                                className="min-h-10 flex-1 rounded-md bg-red-700 px-4 text-sm font-medium text-white hover:bg-red-800"
                            >
                                Stop and save
                            </button>
                        ) : (
                            <button
                                type="button"
                                onClick={startLiveTranscription}
                                disabled={isBusy}
                                className="min-h-10 flex-1 rounded-md bg-zinc-950 px-4 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-40"
                            >
                                {state === "requesting"
                                    ? "Requesting access"
                                    : state === "saving"
                                        ? "Saving"
                                        : "Start live"}
                            </button>
                        )}
                        <button
                            type="button"
                            onClick={discardLiveTranscription}
                            disabled={isRecording || isBusy || (!segments.length && !elapsedSeconds)}
                            className="min-h-10 rounded-md border border-zinc-300 px-3 text-sm font-medium text-zinc-700 hover:border-zinc-400 hover:text-zinc-950 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                            Discard
                        </button>
                    </div>

                    {error && <p className="text-sm text-red-700">{error}</p>}
                </div>

                <div className="min-h-[260px] rounded-md border border-zinc-200 bg-zinc-50 p-4">
                    {segments.length === 0 ? (
                        <div className="flex h-full items-center text-sm text-zinc-500">
                            Live transcript chunks will appear here.
                        </div>
                    ) : (
                        <div className="flex flex-col gap-3">
                            {segments.map((segment, index) => (
                                <div key={`${segment.start}-${index}`} className="text-sm">
                                    <span className="mr-3 font-mono text-xs text-zinc-500">
                                        {formatTime(segment.start)}
                                    </span>
                                    <span className="text-zinc-900">{segment.text}</span>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </section>
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

function extensionForMimeType(mimeType: string): string {
    if (mimeType.includes("mp4")) return "m4a";
    if (mimeType.includes("ogg")) return "ogg";
    return "webm";
}

function timestampForFilename(): string {
    return new Date()
        .toISOString()
        .replace(/[:.]/g, "-")
        .slice(0, 19);
}

function formatDuration(seconds: number): string {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${String(remainingSeconds).padStart(2, "0")}`;
}

function formatTime(seconds: number): string {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
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
    return err instanceof Error ? err.message : "Failed to start live transcription.";
}
